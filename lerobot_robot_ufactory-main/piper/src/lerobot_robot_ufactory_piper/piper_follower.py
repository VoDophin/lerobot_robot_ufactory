import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cached_property
from typing import Any, Callable, TypeVar

from lerobot.cameras import Camera
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.errors import DeviceNotConnectedError

from .config import DualPiperFollowerConfig, PiperFollowerConfig
from .motors import PiperMotorsBus
from .motors.tables import CALIBRATION, MOTORS
from .pose import axis_angle_to_rpy_degrees, clamp, rpy_degrees_to_axis_angle, vector_step_towards

logger = logging.getLogger(__name__)
T = TypeVar("T")

JOINT_KEYS = tuple(f"joint{i}.pos" for i in range(1, 7)) + ("gripper.pos",)
POSE_KEYS = ("pose.x", "pose.y", "pose.z", "pose.rx", "pose.ry", "pose.rz")


class PiperFollower(Robot):
    config_class = PiperFollowerConfig
    name = "piper_follower"

    def __init__(self, config: PiperFollowerConfig, prefix: str = "") -> None:
        super().__init__(config)
        self.config = config
        self.prefix = f"{prefix}." if prefix else ""
        self.bus = PiperMotorsBus(
            id=config.id or prefix or "piper",
            port=config.port,
            motors=MOTORS.copy(),
            calibration=CALIBRATION.copy(),
        )
        self.cameras: dict[str, Camera] = make_cameras_from_configs(config.cameras)
        self._camera_executor: ThreadPoolExecutor | None = None
        if self.cameras:
            self._camera_executor = ThreadPoolExecutor(
                max_workers=len(self.cameras), thread_name_prefix=f"{prefix or 'piper'}-camera"
            )

    def _key(self, local_key: str) -> str:
        return f"{self.prefix}{local_key}"

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        state = {self._key(key): float for key in self._state_keys}
        images = {
            self._key(name): (camera.height, camera.width, 3)
            for name, camera in self.cameras.items()
        }
        return {**state, **images}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {self._key(key): float for key in self._state_keys}

    @property
    def _state_keys(self) -> tuple[str, ...]:
        gripper_key = ("gripper.pos",) if self.config.use_gripper else ()
        if self.config.control_space == "joint":
            return tuple(f"joint{i}.pos" for i in range(1, 7)) + gripper_key
        return POSE_KEYS + gripper_key

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(camera.is_connected for camera in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        try:
            if self.config.configure_role_on_connect:
                self.bus.set_follower()
                time.sleep(0.1)
            self.bus.enable_torque()
            # This integration uses the Piper's fixed factory ranges, so it is
            # normally already calibrated. Avoid moving the gripper merely
            # because LeRobot calls connect(calibrate=True) by default.
            if calibrate and not self.is_calibrated:
                self.calibrate()
            if self.config.park_on_connect:
                self.bus.parking()
            for camera in self.cameras.values():
                camera.connect()
            if self.cameras and self._camera_executor is None:
                self._camera_executor = ThreadPoolExecutor(
                    max_workers=len(self.cameras),
                    thread_name_prefix=f"{self.id or 'piper'}-camera",
                )
            # Piper SDK documents the first feedback frame as zero-valued.
            time.sleep(0.2)
        except BaseException:
            for camera in self.cameras.values():
                if camera.is_connected:
                    camera.disconnect()
            self.bus.disconnect(disable_torque=True, park=False)
            raise
        logger.info("Connected Piper follower %s on %s", self.id, self.config.port)

    def calibrate(self) -> None:
        self.bus.clear_gripper()

    def configure(self) -> None:
        # Role and motion mode are refreshed on connect/send_action respectively.
        return None

    def setup_motors(self) -> None:
        self.bus.connect()
        self.bus.set_follower()

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        if self.config.control_space == "joint":
            local = {
                f"{motor}.pos": value
                for motor, value in self.bus.get_joint_position().items()
                if f"{motor}.pos" in self._state_keys
            }
        else:
            x, y, z, roll, pitch, yaw = self.bus.get_end_pose()
            rx, ry, rz = rpy_degrees_to_axis_angle(roll, pitch, yaw)
            local = dict(zip(POSE_KEYS, (x, y, z, rx, ry, rz), strict=True))
            if self.config.use_gripper:
                local["gripper.pos"] = self.bus.get_joint_position()["gripper"] / 100.0

        observation = {self._key(key): value for key, value in local.items()}
        if self._camera_executor is not None:
            futures = {
                name: self._camera_executor.submit(camera.async_read)
                for name, camera in self.cameras.items()
            }
            for name, future in futures.items():
                observation[self._key(name)] = future.result()
        return observation

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        local = self._strip_and_validate(action)
        if self.config.control_space == "joint":
            sent = self._send_joint_action(local)
        else:
            sent = self._send_cartesian_action(local)
        return {self._key(key): value for key, value in sent.items()}

    def _strip_and_validate(self, action: dict[str, Any]) -> dict[str, float]:
        local: dict[str, float] = {}
        for key, value in action.items():
            if self.prefix:
                if not key.startswith(self.prefix):
                    continue
                key = key[len(self.prefix) :]
            local[key] = float(value)
        missing = set(self._state_keys) - set(local)
        if missing:
            raise KeyError(f"Action for {self.id} is missing keys: {sorted(missing)}")
        return {key: local[key] for key in self._state_keys}

    def _send_joint_action(self, local: dict[str, float]) -> dict[str, float]:
        goal = {key.removesuffix(".pos"): value for key, value in local.items()}
        if self.config.max_relative_target is not None:
            present = self.bus.get_joint_position()
            paired = {key: (value, present[key]) for key, value in goal.items()}
            goal = ensure_safe_goal_position(paired, self.config.max_relative_target)
        self.bus.set_joint_position(goal, speed_percent=self.config.move_speed_percent)
        return {f"{motor}.pos": value for motor, value in goal.items()}

    def _send_cartesian_action(self, local: dict[str, float]) -> dict[str, float]:
        x, y, z, roll, pitch, yaw = self.bus.get_end_pose()
        current_rx, current_ry, current_rz = rpy_degrees_to_axis_angle(roll, pitch, yaw)
        current_xyz = (x, y, z)
        current_rotation = (current_rx, current_ry, current_rz)
        target = tuple(local[key] for key in POSE_KEYS)
        # Clamp the target before limiting the step. Clamping the already-limited
        # command could cause a large jump when the current pose is outside bounds.
        bounded_xyz = (
            clamp(target[0], self.config.workspace_x),
            clamp(target[1], self.config.workspace_y),
            clamp(target[2], self.config.workspace_z),
        )
        limited_xyz = vector_step_towards(
            current_xyz, bounded_xyz, self.config.max_cartesian_step_mm
        )
        limited_rotation = vector_step_towards(
            current_rotation, target[3:6], self.config.max_rotation_step_rad
        )
        limited = [*limited_xyz, *limited_rotation]

        roll_deg, pitch_deg, yaw_deg = axis_angle_to_rpy_degrees(*limited[3:6])
        self.bus.set_end_pose(
            (*limited[:3], roll_deg, pitch_deg, yaw_deg),
            move_mode=self.config.move_mode,
            speed_percent=self.config.move_speed_percent,
        )
        sent = dict(zip(POSE_KEYS, limited, strict=True))
        if self.config.use_gripper:
            gripper_unit = min(1.0, max(0.0, local["gripper.pos"]))
            self.bus.set_gripper_percent(gripper_unit * 100.0, effort=self.config.gripper_effort)
            sent["gripper.pos"] = gripper_unit
        return sent

    def parking(self) -> None:
        self.bus.parking()

    def disconnect(self) -> None:
        for camera in self.cameras.values():
            if camera.is_connected:
                camera.disconnect()
        if self._camera_executor is not None:
            self._camera_executor.shutdown(wait=True)
            self._camera_executor = None
        self.bus.disconnect(
            disable_torque=self.config.disable_torque_on_disconnect,
            park=self.config.park_on_disconnect,
        )


class DualPiperFollower(Robot):
    config_class = DualPiperFollowerConfig
    name = "dual_piper_follower"

    def __init__(self, config: DualPiperFollowerConfig) -> None:
        super().__init__(config)
        self.config = config
        self.robots: dict[str, PiperFollower] = {}
        for side, robot_config in config.robots.items():
            if not isinstance(robot_config, PiperFollowerConfig):
                raise TypeError(f"{side} must use type uf::piper, got {robot_config.type}")
            self.robots[side] = PiperFollower(robot_config, prefix=side)
        self.cameras = {
            f"{side}.{name}": camera
            for side, robot in self.robots.items()
            for name, camera in robot.cameras.items()
        }
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dual-piper")

    @property
    def observation_features(self) -> dict:
        return self._merge(lambda robot: robot.observation_features)

    @property
    def action_features(self) -> dict:
        return self._merge(lambda robot: robot.action_features)

    @property
    def is_connected(self) -> bool:
        return all(robot.is_connected for robot in self.robots.values())

    @property
    def is_calibrated(self) -> bool:
        return all(robot.is_calibrated for robot in self.robots.values())

    def _merge(self, getter: Callable[[PiperFollower], dict]) -> dict:
        merged: dict = {}
        for robot in self.robots.values():
            merged.update(getter(robot))
        return merged

    def _parallel(self, fn: Callable[[PiperFollower], T]) -> list[T]:
        futures: list[Future[T]] = [self._executor.submit(fn, robot) for robot in self.robots.values()]
        return [future.result() for future in futures]

    def connect(self, calibrate: bool = True) -> None:
        try:
            if self.config.parallel_connect:
                self._parallel(lambda robot: robot.connect(calibrate=calibrate))
            else:
                for robot in self.robots.values():
                    robot.connect(calibrate=calibrate)
        except BaseException:
            for robot in self.robots.values():
                if robot.bus.is_connected:
                    try:
                        robot.disconnect()
                    except Exception:
                        logger.exception("Failed to clean up %s after a partial connect", robot.id)
            raise

    def calibrate(self) -> None:
        for robot in self.robots.values():
            robot.calibrate()

    def configure(self) -> None:
        for robot in self.robots.values():
            robot.configure()

    def get_observation(self) -> dict[str, Any]:
        observations = (
            self._parallel(lambda robot: robot.get_observation())
            if self.config.parallel_observation
            else [robot.get_observation() for robot in self.robots.values()]
        )
        merged: dict[str, Any] = {}
        for observation in observations:
            merged.update(observation)
        return merged

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        def send(robot: PiperFollower) -> dict[str, Any]:
            return robot.send_action(action)

        sent_actions = (
            self._parallel(send)
            if self.config.parallel_action
            else [send(robot) for robot in self.robots.values()]
        )
        merged: dict[str, Any] = {}
        for sent in sent_actions:
            merged.update(sent)
        return merged

    def disconnect(self) -> None:
        try:
            self._parallel(lambda robot: robot.disconnect())
        finally:
            self._executor.shutdown(wait=True)
