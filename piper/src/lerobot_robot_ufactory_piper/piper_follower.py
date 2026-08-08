import logging
import math
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
from .pose import axis_angle_to_rpy_degrees, clamp, rotation_distance, rpy_degrees_to_axis_angle, vector_step_towards

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
        self._last_pose_command_time_s = 0.0
        self._last_gripper_command_time_s = 0.0
        self._last_gripper_command: float | None = None
        self._last_gripper_debug_time_s = 0.0
        self._last_pose_command: tuple[float, float, float, float, float, float] | None = None
        self._locked_rotation: tuple[float, float, float] | None = None
        self._force_step_cartesian = False
        self._last_tracking_warning_time_s = 0.0
        self._last_following_warning_s = 0.0
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
        return {self._key(key): float for key in self._action_keys}

    @property
    def _state_keys(self) -> tuple[str, ...]:
        if self.config.control_space == "joint":
            return JOINT_KEYS
        return POSE_KEYS + ("gripper.pos",)

    @property
    def _action_keys(self) -> tuple[str, ...]:
        if self.config.send_gripper:
            return self._state_keys
        return tuple(key for key in self._state_keys if key != "gripper.pos")

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(camera.is_connected for camera in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        self._last_pose_command = None
        self._locked_rotation = None
        self._last_pose_command_time_s = 0.0
        self._last_gripper_command_time_s = 0.0
        self._last_gripper_command = None
        self.bus.connect(piper_init=self.config.piper_init_on_connect)
        try:
            if self.config.configure_role_on_connect:
                self.bus.set_follower()
                time.sleep(0.1)
            if self.config.enable_torque_on_connect:
                self.bus.enable_torque()
            self._wait_for_feedback()
            # This integration uses the Piper's fixed factory ranges, so it is
            # normally already calibrated. Avoid moving the gripper merely
            # because LeRobot calls connect(calibrate=True) by default.
            if calibrate and not self.is_calibrated:
                self.calibrate()
            if getattr(self.config, "startup_tcp_pose", None) is not None:
                self.move_to_tcp_pose(
                    tuple(self.config.startup_tcp_pose),
                    timeout_s=self.config.startup_move_timeout_s,
                )
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
            self.bus.disconnect(
                disable_torque=self.config.disable_torque_on_disconnect,
                park=False,
            )
            raise
        logger.info("Connected Piper follower %s on %s", self.id, self.config.port)

    def calibrate(self) -> None:
        self.bus.clear_gripper()

    def configure(self) -> None:
        # Role and motion mode are refreshed on connect/send_action respectively.
        return None


    def _wait_for_feedback(self) -> None:
        """Wait until the SDK returns a finite, non-zero end pose."""
        deadline = time.monotonic() + self.config.feedback_startup_timeout_s
        while time.monotonic() < deadline:
            try:
                x, y, z, roll, pitch, yaw = self.bus.get_end_pose()
                if (
                    all(math.isfinite(v) for v in (x, y, z, roll, pitch, yaw))
                    and (x != 0.0 or y != 0.0 or z != 0.0)
                ):
                    return
            except Exception:
                pass
            time.sleep(0.05)
        raise RuntimeError(
            f"Piper {self.id} did not return valid feedback within "
            f"{self.config.feedback_startup_timeout_s:.1f}s"
        )

    def move_to_tcp_pose(
        self,
        pose_mm_rpy_deg: tuple[float, ...] | list[float],
        *,
        timeout_s: float = 30.0,
        translation_tolerance_mm: float = 2.0,
        rotation_tolerance_rad: float = 0.02,
    ) -> None:
        if not self.bus.is_connected:
            raise DeviceNotConnectedError(f"{self} CAN bus is not connected")
        if self.config.control_space != "cartesian":
            raise RuntimeError("startup_tcp_pose requires cartesian control")
        target = tuple(float(v) for v in pose_mm_rpy_deg)
        if len(target) != 6 or not all(math.isfinite(v) for v in target):
            raise ValueError("startup_tcp_pose must contain six finite values")
        target_rotation = rpy_degrees_to_axis_angle(*target[3:6])
        action = dict(zip(POSE_KEYS, (*target[:3], *target_rotation), strict=True))
        action["gripper.pos"] = self.bus.get_joint_position()["gripper"] / 100.0
        deadline = time.monotonic() + timeout_s
        previous_force = self._force_step_cartesian
        self._force_step_cartesian = True
        try:
            while True:
                current = self.bus.get_end_pose()
                current_rotation = rpy_degrees_to_axis_angle(*current[3:6])
                translation_error = math.dist(current[:3], target[:3])
                angular_error = rotation_distance(current_rotation, target_rotation)
                if (
                    translation_error <= translation_tolerance_mm
                    and angular_error <= rotation_tolerance_rad
                ):
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Piper {self.id} did not reach startup pose in {timeout_s:.1f}s "
                        f"(translation error {translation_error:.2f} mm)"
                    )
                self.send_action(action)
                time.sleep(0.02)
        finally:
            self._force_step_cartesian = previous_force

    def setup_motors(self) -> None:
        self.bus.connect()
        self.bus.set_follower()

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        if self.config.control_space == "joint":
            local = {f"{motor}.pos": value for motor, value in self.bus.get_joint_position().items()}
        else:
            x, y, z, roll, pitch, yaw = self.bus.get_end_pose()
            rx, ry, rz = rpy_degrees_to_axis_angle(roll, pitch, yaw)
            gripper = self.bus.get_joint_position()["gripper"] / 100.0
            local = dict(zip(POSE_KEYS, (x, y, z, rx, ry, rz), strict=True))
            local["gripper.pos"] = gripper

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
        if not self.bus.is_connected:
            raise DeviceNotConnectedError(f"{self} CAN bus is not connected")

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
        missing = set(self._action_keys) - set(local)
        if missing:
            raise KeyError(f"Action for {self.id} is missing keys: {sorted(missing)}")
        return {key: local[key] for key in self._action_keys}

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

        is_cpv = self.config.move_mode == "move_cpv"
        if self.config.lock_orientation and self._locked_rotation is None:
            self._locked_rotation = current_rotation
        desired_rotation = (
            self._locked_rotation
            if self.config.lock_orientation and self._locked_rotation is not None
            else target[3:6]
        )

        # MOVE CPV is a streaming target mode. Advance from the last command,
        # not from lagging feedback; otherwise every cycle creates another tiny
        # endpoint relative to a different origin. Pause target advancement if
        # the physical arm falls too far behind the command stream.
        command_xyz = current_xyz
        command_rotation = current_rotation
        tracking_blocked = False
        if is_cpv and self._last_pose_command is not None:
            command_xyz = self._last_pose_command[:3]
            command_rotation = self._last_pose_command[3:6]
            if self.config.max_tracking_error_mm is not None:
                tracking_error_sq = sum(
                    (current_xyz[index] - command_xyz[index]) ** 2 for index in range(3)
                )
                tracking_blocked = (
                    tracking_error_sq > self.config.max_tracking_error_mm**2
                )
                if tracking_blocked:
                    now_s = time.monotonic()
                    if now_s - self._last_tracking_warning_time_s >= 1.0:
                        logger.warning(
                            "Piper CPV target paused: tracking error %.1f mm exceeds %.1f mm",
                            tracking_error_sq**0.5,
                            self.config.max_tracking_error_mm,
                        )
                        self._last_tracking_warning_time_s = now_s

        translation_error_sq = sum(
            (bounded_xyz[index] - command_xyz[index]) ** 2 for index in range(3)
        )
        translation_in_deadband = (
            translation_error_sq <= self.config.translation_deadband_mm**2
        )
        rotation_error_sq = sum(
            (desired_rotation[index] - command_rotation[index]) ** 2 for index in range(3)
        )
        rotation_in_deadband = rotation_error_sq <= self.config.rotation_deadband_rad**2

        direct_command = (
            getattr(self.config, "cartesian_command_mode", "step") == "direct"
            and not getattr(self, "_force_step_cartesian", False)
        )
        if direct_command and not (translation_in_deadband and rotation_in_deadband):
            translation_error = math.dist(current_xyz, bounded_xyz)
            angular_error = rotation_distance(current_rotation, desired_rotation)
            if (
                translation_error > self.config.max_cartesian_following_error_mm
                or angular_error > self.config.max_rotation_following_error_rad
            ):
                now_s = time.monotonic()
                if now_s - self._last_following_warning_s >= 1.0:
                    logger.warning(
                        "Piper %s direct target exceeds following limits "
                        "(translation %.1f/%.1f mm, rotation %.3f/%.3f rad); holding",
                        self.id,
                        translation_error,
                        self.config.max_cartesian_following_error_mm,
                        angular_error,
                        self.config.max_rotation_following_error_rad,
                    )
                    self._last_following_warning_s = now_s
                limited_xyz = current_xyz
                limited_rotation = current_rotation
            else:
                max_direct_step = getattr(self.config, "direct_max_step_mm", None)
                if max_direct_step is not None and max_direct_step > 0:
                    limited_xyz = vector_step_towards(
                        current_xyz, bounded_xyz, max_direct_step
                    )
                else:
                    limited_xyz = bounded_xyz
                max_direct_rot = getattr(self.config, "direct_max_step_rad", None)
                if max_direct_rot is not None and max_direct_rot > 0:
                    limited_rotation = vector_step_towards(
                        current_rotation, desired_rotation, max_direct_rot
                    )
                else:
                    limited_rotation = desired_rotation
        elif tracking_blocked:
            limited_xyz = command_xyz
            limited_rotation = command_rotation
        else:
            translation_target = command_xyz if translation_in_deadband else bounded_xyz
            rotation_target = command_rotation if rotation_in_deadband else desired_rotation
            limited_xyz = vector_step_towards(
                command_xyz, translation_target, self.config.max_cartesian_step_mm
            )
            limited_rotation = vector_step_towards(
                command_rotation, rotation_target, self.config.max_rotation_step_rad
            )
        limited = [*limited_xyz, *limited_rotation]

        roll_deg, pitch_deg, yaw_deg = axis_angle_to_rpy_degrees(*limited[3:6])
        gripper_unit = None
        if self.config.send_gripper:
            gripper_unit = min(1.0, max(0.0, local["gripper.pos"]))
        now_s = time.monotonic()
        pose_command_needed = (
            self.config.send_cartesian_pose
            and not tracking_blocked
            and not (translation_in_deadband and rotation_in_deadband)
        )
        pose_due = (
            now_s - self._last_pose_command_time_s
            >= self.config.min_command_interval_s
        )
        if pose_command_needed and pose_due:
            self.bus.set_end_pose(
                (*limited[:3], roll_deg, pitch_deg, yaw_deg),
                move_mode=self.config.move_mode,
                speed_percent=self.config.move_speed_percent,
            )
            self._last_pose_command = tuple(limited)
            self._last_pose_command_time_s = now_s

        if gripper_unit is not None:
            gripper_interval_s = (
                self.config.gripper_min_command_interval_s
                if self.config.gripper_min_command_interval_s is not None
                else self.config.min_command_interval_s
            )
            keepalive_s = (
                self.config.gripper_keepalive_s
                if self.config.gripper_keepalive_s is not None
                else max(gripper_interval_s, 0.001)
            )
            gripper_due = (
                now_s - self._last_gripper_command_time_s >= gripper_interval_s
            )
            gripper_changed = (
                self._last_gripper_command is None
                or abs(gripper_unit - self._last_gripper_command)
                >= self.config.gripper_command_deadband
            )
            keepalive_due = (
                now_s - self._last_gripper_command_time_s >= keepalive_s
            )
            if gripper_due and (gripper_changed or keepalive_due):
                self.bus.set_gripper_percent(
                    gripper_unit * 100.0,
                    effort=self.config.gripper_effort,
                    ctrl_code=self.config.gripper_ctrl_code,
                )
                if (
                    self.config.gripper_debug
                    and now_s - self._last_gripper_debug_time_s >= 0.5
                ):
                    logger.info(
                        "Piper %s gripper command: %.1f mm (normalized %.3f, ctrl 0x%02X)",
                        self.id,
                        gripper_unit * 68.0,
                        gripper_unit,
                        self.config.gripper_ctrl_code,
                    )
                    self._last_gripper_debug_time_s = now_s
                self._last_gripper_command = gripper_unit
                self._last_gripper_command_time_s = now_s
        sent = dict(zip(POSE_KEYS, limited, strict=True))
        if gripper_unit is not None:
            sent["gripper.pos"] = gripper_unit
        return sent

    def parking(self) -> None:
        self.bus.parking()

    def disconnect(self) -> None:
        if (
            self.bus.is_connected
            and self.config.control_space == "cartesian"
            and not self.config.disable_torque_on_disconnect
            and getattr(self.config, "hold_position_on_disconnect", False)
        ):
            pose = self.bus.get_end_pose()
            if all(math.isfinite(v) for v in pose):
                self.bus.set_end_pose(
                    pose,
                    move_mode=self.config.move_mode,
                    speed_percent=self.config.move_speed_percent,
                )
                time.sleep(0.05)
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
        self._last_pose_command = None
        self._locked_rotation = None
        self._last_pose_command_time_s = 0.0
        self._last_gripper_command_time_s = 0.0
        self._last_gripper_command = None
        self._last_gripper_debug_time_s = 0.0


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
