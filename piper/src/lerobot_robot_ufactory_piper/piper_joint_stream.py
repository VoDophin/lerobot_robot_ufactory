"""Piper joint-streaming follower (official-style motion layer).

Receives the same Cartesian Pika teleop actions as ``PiperFollower``, but
executes them by solving IK with the URDF kinematics (pure numpy) and
streaming joint targets through the SDK's joint control mode
(``MotionCtrl_2(0x01,0x01,speed,0xad)`` + ``JointCtrl``), with per-cycle
Cartesian and joint step limits. This is the motion layer the official
PikaAnyArm stack uses: it avoids the MOVE P re-planning that causes jitter
and fixed-speed chasing.

Kept as a separate robot type (``uf::piper_joint_stream``) so the MOVE P
path (V13/V14, ``uf::piper``) is completely untouched.
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from lerobot.cameras import Camera, CameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot, RobotConfig
from lerobot.utils.errors import DeviceNotConnectedError

from .motors import PiperMotorsBus
from .motors.tables import CALIBRATION, MOTORS
from .piper_kinematics import PiperKinematics
from .pose import (
    axis_angle_to_rpy_degrees,
    clamp,
    rpy_degrees_to_axis_angle,
    vector_step_towards,
)

logger = logging.getLogger(__name__)

POSE_KEYS = ("pose.x", "pose.y", "pose.z", "pose.rx", "pose.ry", "pose.rz")
_RAD_TO_MILLIDEG = 1000.0 * 180.0 / math.pi


@RobotConfig.register_subclass("uf::piper_joint_stream")
@dataclass(kw_only=True)
class PiperJointStreamConfig(RobotConfig):
    """One Piper arm controlled by joint-space streaming."""

    port: str
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    configure_role_on_connect: bool = True
    park_on_connect: bool = False
    park_on_disconnect: bool = False
    disable_torque_on_disconnect: bool = True

    # Cartesian target clamps (same safety semantics as PiperFollower).
    max_cartesian_step_mm: float = 8.0
    max_rotation_step_rad: float = 0.08
    translation_deadband_mm: float = 0.5
    rotation_deadband_rad: float = 0.004
    workspace_x: tuple[float, float] | None = None
    workspace_y: tuple[float, float] | None = None
    workspace_z: tuple[float, float] | None = None

    # Joint streaming.
    max_joint_step_deg: float = 3.0
    ik_max_iter: int = 10
    ik_damping: float = 1e-3
    ik_weight_ori: float = 1.0
    ik_residual_limit: float = 0.03
    move_speed_percent: int = 30
    # Base frame transform (from verify_ik_fk.py --calibrate; identity by
    # default). T_sdk = X_base @ chain(q) @ X_ee.
    base_xyz_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # End-effector frame that matches the SDK pose. Verify with
    # piper/tools/verify_ik_fk.py.
    ee_xyz_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ee_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)

    send_gripper: bool = True
    gripper_effort: int = 1000
    gripper_ctrl_code: int = 0x03
    gripper_command_deadband: float = 0.005
    gripper_min_command_interval_s: float | None = None
    gripper_keepalive_s: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.id = "piper_joint_stream_follower" if self.id is None else self.id
        if not 1 <= self.move_speed_percent <= 100:
            raise ValueError("move_speed_percent must be in [1, 100]")
        if self.max_cartesian_step_mm <= 0 or self.max_rotation_step_rad <= 0:
            raise ValueError("Cartesian step limits must be positive")
        if self.max_joint_step_deg <= 0:
            raise ValueError("max_joint_step_deg must be positive")
        if not 0 < self.ik_damping:
            raise ValueError("ik_damping must be positive")
        for name in ("workspace_x", "workspace_y", "workspace_z"):
            bounds = getattr(self, name)
            if bounds is not None and bounds[0] >= bounds[1]:
                raise ValueError(f"{name} must be ordered as (min, max)")


class PiperJointStreamFollower(Robot):
    config_class = PiperJointStreamConfig
    name = "piper_joint_stream_follower"

    def __init__(self, config: PiperJointStreamConfig, prefix: str = "") -> None:
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
        self._kin: PiperKinematics | None = None
        self._last_motion_ctrl: tuple[int, int, int, int] | None = None
        self._last_joint_cmd_rad: Any = None
        self._last_gripper_cmd: float | None = None
        self._last_gripper_time_s: float = 0.0
        self._last_ik_warning_s: float = 0.0
        if self.cameras:
            self._camera_executor = ThreadPoolExecutor(
                max_workers=len(self.cameras),
                thread_name_prefix=f"{prefix or 'piper'}-camera",
            )

    def _key(self, local_key: str) -> str:
        return f"{self.prefix}{local_key}"

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        state = {self._key(key): float for key in POSE_KEYS}
        if self.config.send_gripper:
            state[self._key("gripper.pos")] = float
        images = {
            self._key(name): (camera.height, camera.width, 3)
            for name, camera in self.cameras.items()
        }
        return {**state, **images}

    @cached_property
    def action_features(self) -> dict[str, type]:
        keys = POSE_KEYS
        if self.config.send_gripper:
            keys = POSE_KEYS + ("gripper.pos",)
        return {self._key(key): float for key in keys}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(
            camera.is_connected for camera in self.cameras.values()
        )

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def _kinematics(self) -> PiperKinematics:
        if self._kin is None:
            self._kin = PiperKinematics(
                base_xyzrpy_m=tuple(v / 1000.0 for v in self.config.base_xyz_mm),
                base_rpy_rad=tuple(math.radians(v) for v in self.config.base_rpy_deg),
                ee_xyzrpy_m=tuple(v / 1000.0 for v in self.config.ee_xyz_mm),
                ee_rpy_rad=tuple(math.radians(v) for v in self.config.ee_rpy_deg),
            )
        return self._kin

    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        try:
            if self.config.configure_role_on_connect:
                self.bus.set_follower()
                time.sleep(0.1)
            self.bus.enable_torque()
            for camera in self.cameras.values():
                camera.connect()
            if self.cameras and self._camera_executor is None:
                self._camera_executor = ThreadPoolExecutor(
                    max_workers=len(self.cameras),
                    thread_name_prefix=f"{self.id or 'piper'}-camera",
                )
            time.sleep(0.2)
        except BaseException:
            for camera in self.cameras.values():
                if camera.is_connected:
                    camera.disconnect()
            self.bus.disconnect(disable_torque=True, park=False)
            raise
        logger.info(
            "Connected joint-stream follower %s on %s",
            self.id,
            self.config.port,
        )

    def calibrate(self) -> None:
        self.bus.clear_gripper()

    def configure(self) -> None:
        return None

    def setup_motors(self) -> None:
        self.bus.connect()
        self.bus.set_follower()

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        x, y, z, roll, pitch, yaw = self.bus.get_end_pose()
        rx, ry, rz = rpy_degrees_to_axis_angle(roll, pitch, yaw)
        local = dict(zip(POSE_KEYS, (x, y, z, rx, ry, rz), strict=True))
        if self.config.send_gripper:
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
        sent = self._send_cartesian_joint_stream(local)
        return {self._key(key): value for key, value in sent.items()}

    def _strip_and_validate(self, action: dict[str, Any]) -> dict[str, float]:
        local: dict[str, float] = {}
        for key, value in action.items():
            if self.prefix:
                if not key.startswith(self.prefix):
                    continue
                key = key[len(self.prefix) :]
            local[key] = float(value)
        required = set(POSE_KEYS)
        if self.config.send_gripper:
            required.add("gripper.pos")
        missing = required - set(local)
        if missing:
            raise KeyError(f"Action for {self.id} is missing keys: {sorted(missing)}")
        return {key: local[key] for key in required}

    def _current_joints_rad(self) -> np.ndarray:
        js = self.bus.piper.GetArmJointMsgs().joint_state
        raw = np.array(
            [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6],
            dtype=float,
        )
        return np.radians(raw / 1000.0)

    def _send_cartesian_joint_stream(
        self, local: dict[str, float]
    ) -> dict[str, float]:
        x, y, z, roll, pitch, yaw = self.bus.get_end_pose()
        current_rx, current_ry, current_rz = rpy_degrees_to_axis_angle(
            roll, pitch, yaw
        )
        current_xyz = np.array([x, y, z], dtype=float)
        current_rot = np.array([current_rx, current_ry, current_rz], dtype=float)
        target = np.array([local[key] for key in POSE_KEYS], dtype=float)

        bounded_xyz = np.array(
            [
                clamp(target[0], self.config.workspace_x),
                clamp(target[1], self.config.workspace_y),
                clamp(target[2], self.config.workspace_z),
            ],
            dtype=float,
        )
        in_deadband = (
            np.linalg.norm(bounded_xyz - current_xyz)
            <= self.config.translation_deadband_mm
            and np.linalg.norm(target[3:6] - current_rot)
            <= self.config.rotation_deadband_rad
        )
        sent = dict(zip(POSE_KEYS, target, strict=True))
        if self.config.send_gripper:
            sent["gripper.pos"] = local["gripper.pos"]
        if in_deadband:
            return sent

        limited_xyz = vector_step_towards(
            tuple(current_xyz), tuple(bounded_xyz), self.config.max_cartesian_step_mm
        )
        limited_rot = vector_step_towards(
            tuple(current_rot),
            tuple(target[3:6]),
            self.config.max_rotation_step_rad,
        )
        roll_deg, pitch_deg, yaw_deg = axis_angle_to_rpy_degrees(*limited_rot)

        q_current = self._current_joints_rad()
        kin = self._kinematics()
        q_target, residual = kin.ik(
            limited_xyz,
            (roll_deg, pitch_deg, yaw_deg),
            q_current,
            max_iter=self.config.ik_max_iter,
            damping=self.config.ik_damping,
            weight_ori=self.config.ik_weight_ori,
        )
        if residual > self.config.ik_residual_limit:
            now_s = time.monotonic()
            if now_s - self._last_ik_warning_s >= 1.0:
                logger.warning(
                    "IK residual %.4f > %.4f; holding last joint target",
                    residual,
                    self.config.ik_residual_limit,
                )
                self._last_ik_warning_s = now_s
            q_target = (
                self._last_joint_cmd_rad
                if self._last_joint_cmd_rad is not None
                else q_current
            )
        else:
            self._last_joint_cmd_rad = q_target

        max_joint_step = math.radians(self.config.max_joint_step_deg)
        delta = np.clip(q_target - q_current, -max_joint_step, max_joint_step)
        q_send = q_current + delta
        millideg = tuple(int(round(v * _RAD_TO_MILLIDEG)) for v in q_send)
        self._send_joints(millideg)

        if self.config.send_gripper:
            self._send_gripper(local["gripper.pos"], sent)
        return sent

    def _send_joints(self, millideg: tuple[int, ...]) -> None:
        cmd = (0x01, 0x01, self.config.move_speed_percent, 0xAD)
        if self._last_motion_ctrl != cmd:
            self.bus.piper.MotionCtrl_2(*cmd)
            self._last_motion_ctrl = cmd
        self.bus.piper.JointCtrl(*millideg)

    def _send_gripper(self, gripper_unit: float, sent: dict[str, float]) -> None:
        gripper_unit = min(1.0, max(0.0, gripper_unit))
        interval_s = (
            self.config.gripper_min_command_interval_s
            if self.config.gripper_min_command_interval_s is not None
            else 0.03
        )
        keepalive_s = (
            self.config.gripper_keepalive_s
            if self.config.gripper_keepalive_s is not None
            else max(interval_s, 0.001)
        )
        now_s = time.monotonic()
        due = now_s - self._last_gripper_time_s >= interval_s
        changed = (
            self._last_gripper_cmd is None
            or abs(gripper_unit - self._last_gripper_cmd)
            >= self.config.gripper_command_deadband
        )
        keepalive_due = now_s - self._last_gripper_time_s >= keepalive_s
        if due and (changed or keepalive_due):
            self.bus.set_gripper_percent(
                gripper_unit * 100.0,
                effort=self.config.gripper_effort,
                ctrl_code=self.config.gripper_ctrl_code,
            )
            self._last_gripper_cmd = gripper_unit
            self._last_gripper_time_s = now_s
        sent["gripper.pos"] = gripper_unit

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
        self._last_joint_cmd_rad = None
