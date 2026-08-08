import math

from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig
from lerobot.teleoperators import TeleoperatorConfig

from lerobot_robot_ufactory.teleoperators.pika_teleop import PikaTeleopConfig


@RobotConfig.register_subclass("uf::piper")
@dataclass(kw_only=True)
class PiperFollowerConfig(RobotConfig):
    """One Piper follower arm.

    ``cartesian`` actions use millimetres for xyz and radians as an axis-angle
    vector for rx/ry/rz, matching the UFACTORY Pika teleoperator.
    """

    port: str
    control_space: str = "joint"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    configure_role_on_connect: bool = True
    # The SDK's PiperInit helper performs its own control writes. Keep the
    # historical default for existing single-arm profiles, while allowing
    # dual-arm profiles to use the same explicit, piper_init=False connection
    # path as the verified direct gripper test.
    piper_init_on_connect: bool = True
    # Existing arm profiles explicitly enable all joints on connect. A
    # gripper-only diagnostic can leave the already-running arm state untouched
    # to exactly match a direct SDK GripperCtrl test.
    enable_torque_on_connect: bool = True
    park_on_connect: bool = False
    park_on_disconnect: bool = False
    disable_torque_on_disconnect: bool = True

    # Joint mode values are normalized to [-100, 100] (gripper [0, 100]).
    max_relative_target: float | dict[str, float] | None = 5.0

    # Cartesian/Pika safety limits. Translation is mm, rotation is rad.
    max_cartesian_step_mm: float = 10.0
    max_rotation_step_rad: float = 0.10
    translation_deadband_mm: float = 0.0
    rotation_deadband_rad: float = 0.0
    lock_orientation: bool = False
    # Allows a true gripper-only profile: observations and gripper commands
    # remain active, but EndPoseCtrl/ModeCtrl are never sent by send_action.
    send_cartesian_pose: bool = True
    # Ported from the senior lerobot_real fork: 'step' keeps the original
    # per-cycle limiter; 'direct' sends the full target with following-error
    # guards (smoother MOVE P tracking).
    cartesian_command_mode: str = "step"
    max_cartesian_following_error_mm: float = 600.0
    max_rotation_following_error_rad: float = 3.2
    # Optional per-cycle caps in direct mode (None = unbounded, V16 behavior).
    # Set to ~25 mm / 0.35 rad to prevent large single-cycle target jumps
    # from stalling the firmware MOVE P replanner on fast gestures.
    direct_max_step_mm: float | None = None
    direct_max_step_rad: float | None = None
    startup_tcp_pose: tuple[float, ...] | None = None
    startup_move_timeout_s: float = 30.0
    hold_position_on_disconnect: bool = False
    feedback_startup_timeout_s: float = 5.0
    max_tracking_error_mm: float | None = None
    workspace_x: tuple[float, float] | None = None
    workspace_y: tuple[float, float] | None = None
    workspace_z: tuple[float, float] | None = None
    move_mode: str = "move_p"
    move_speed_percent: int = 20
    gripper_effort: int = 1000
    # Piper SDK gripper control code. 0x01 enables the gripper; the official
    # Piper LeRobot adapter uses 0x03 to enable it while clearing gripper errors.
    gripper_ctrl_code: int = 0x01
    send_gripper: bool = True
    min_command_interval_s: float = 0.0
    gripper_command_deadband: float = 0.0
    gripper_min_command_interval_s: float | None = None
    gripper_keepalive_s: float | None = None
    gripper_debug: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.id = "piper_follower" if self.id is None else self.id
        if self.control_space not in ("joint", "cartesian"):
            raise ValueError(f"Unsupported Piper control_space: {self.control_space}")
        if self.move_mode not in ("move_p", "move_l", "move_cpv"):
            raise ValueError(f"Unsupported Piper move_mode: {self.move_mode}")
        if not 1 <= self.move_speed_percent <= 100:
            raise ValueError("move_speed_percent must be in [1, 100]")
        if self.max_cartesian_step_mm <= 0 or self.max_rotation_step_rad <= 0:
            raise ValueError("Cartesian step limits must be positive")
        if self.translation_deadband_mm < 0 or self.rotation_deadband_rad < 0:
            raise ValueError("Cartesian deadbands must be non-negative")
        if self.max_tracking_error_mm is not None and self.max_tracking_error_mm <= 0:
            raise ValueError("max_tracking_error_mm must be positive when set")
        if self.min_command_interval_s < 0:
            raise ValueError("min_command_interval_s must be non-negative")
        if self.gripper_command_deadband < 0:
            raise ValueError("gripper_command_deadband must be non-negative")
        if self.gripper_ctrl_code not in (0x01, 0x03):
            raise ValueError("gripper_ctrl_code must be 0x01 or 0x03")
        if (
            self.gripper_min_command_interval_s is not None
            and self.gripper_min_command_interval_s < 0
        ):
            raise ValueError("gripper_min_command_interval_s must be non-negative")
        if self.gripper_keepalive_s is not None and self.gripper_keepalive_s <= 0:
            raise ValueError("gripper_keepalive_s must be positive when set")
        for name in ("workspace_x", "workspace_y", "workspace_z"):
            bounds = getattr(self, name)
            if bounds is not None and bounds[0] >= bounds[1]:
                raise ValueError(f"{name} must be ordered as (min, max)")
        if self.cartesian_command_mode not in ("step", "direct"):
            raise ValueError("cartesian_command_mode must be 'step' or 'direct'")
        if self.max_cartesian_following_error_mm <= 0 or self.max_rotation_following_error_rad <= 0:
            raise ValueError("following error limits must be positive")
        if self.startup_tcp_pose is not None and len(self.startup_tcp_pose) != 6:
            raise ValueError("startup_tcp_pose must contain six values")
        if self.startup_move_timeout_s <= 0:
            raise ValueError("startup_move_timeout_s must be positive")
        if self.direct_max_step_mm is not None and self.direct_max_step_mm <= 0:
            raise ValueError("direct_max_step_mm must be positive when set")
        if self.direct_max_step_rad is not None and self.direct_max_step_rad <= 0:
            raise ValueError("direct_max_step_rad must be positive when set")


@RobotConfig.register_subclass("uf::dual_piper")
@dataclass(kw_only=True)
class DualPiperFollowerConfig(RobotConfig):
    robots: dict[str, RobotConfig] = field(default_factory=dict)
    parallel_connect: bool = True
    parallel_observation: bool = True
    parallel_action: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        self.id = "dual_piper_follower" if self.id is None else self.id
        if len(self.robots) != 2:
            raise ValueError("uf::dual_piper requires exactly two follower arms")
        ports = [getattr(robot, "port", None) for robot in self.robots.values()]
        if any(port is None for port in ports):
            raise ValueError("each dual Piper follower must define a CAN port")
        if len(set(ports)) != len(ports):
            raise ValueError("dual Piper followers must use distinct CAN ports")


@TeleoperatorConfig.register_subclass("uf::piper_leader")
@dataclass(kw_only=True)
class PiperLeaderConfig(TeleoperatorConfig):
    port: str
    configure_role_on_connect: bool = True
    disable_torque_on_disconnect: bool = True

    def __post_init__(self) -> None:
        self.id = "piper_leader" if self.id is None else self.id


@TeleoperatorConfig.register_subclass("uf::dual_piper_leader")
@dataclass(kw_only=True)
class DualPiperLeaderConfig(TeleoperatorConfig):
    teleops: dict[str, TeleoperatorConfig] = field(default_factory=dict)
    parallel_connect: bool = True
    parallel_read: bool = True

    def __post_init__(self) -> None:
        self.id = "dual_piper_leader" if self.id is None else self.id
        if len(self.teleops) != 2:
            raise ValueError("uf::dual_piper_leader requires exactly two leader arms")


@TeleoperatorConfig.register_subclass("uf::dual_pika_teleop")
@dataclass(kw_only=True)
class DualPikaTeleopConfig(TeleoperatorConfig):
    teleops: dict[str, TeleoperatorConfig] = field(default_factory=dict)
    parallel_read: bool = True

    def __post_init__(self) -> None:
        self.id = "dual_pika" if self.id is None else self.id
        if len(self.teleops) != 2:
            raise ValueError("uf::dual_pika_teleop requires exactly two Pika devices")
        ports = [getattr(teleop, "port", None) for teleop in self.teleops.values()]
        if any(port is None for port in ports):
            raise ValueError("each dual Pika teleoperator must define a serial port")
        if len(set(ports)) != len(ports):
            raise ValueError("dual Pika teleoperators must use distinct serial ports")
        tracker_ids = [
            getattr(teleop, "tracker_device_id", None)
            for teleop in self.teleops.values()
        ]
        if any(tracker_ids):
            if not all(tracker_ids):
                raise ValueError(
                    "configure tracker_device_id for both Pikas or for neither"
                )
            if len(set(tracker_ids)) != len(tracker_ids):
                raise ValueError("dual Pikas must use distinct tracker_device_id values")


@TeleoperatorConfig.register_subclass("uf::piper_pika_teleop")
@dataclass
class PiperPikaTeleopConfig(PikaTeleopConfig):
    """Pika input tuned for Piper without changing the base Pika profile."""

    gripper_filter_window: int = 3
    gripper_filter_alpha: float = 0.35
    gripper_deadband: float = 0.01
    gripper_max_step: float = 0.08
    gripper_use_direct_distance: bool = False
    gripper_distance_min_mm: float = 0.0
    gripper_distance_max_mm: float = 100.0
    use_calibrated_rotation_mapping: bool = False
    apply_piper_tool_axis_correction: bool = False
    # "calibrated" = fixed-matrix body-frame mapping; "world_delta" = port of
    # the senior Lerobot-Real robot_base rotation only; "senior" = the full
    # senior robot_base control target (world-delta rotation + gripper-center
    # tool offset, translation and rotation coupled through tracker_to_robot_eef).
    rotation_style: str = "calibrated"
    # Rebuild the calibrated rotation mapping at teleop enable from the actual
    # end-effector orientation, so pitch/yaw/roll keep the correct axes at any
    # arm pose (not just the pose used when the matrix was derived).
    pose_adaptive_rotation: bool = False
    # Fixed rotation from the Vive/Pika tracking world to the robot base, in
    # degrees. Used by rotation_style=senior as Q (world -> base).
    tracker_world_to_robot_base_rpy: tuple[float, ...] = (0, 0, 0)
    rotation_dominant_axis: bool = False
    rotation_scale: float = 1.0
    rotation_filter_alpha: float = 1.0
    use_raw_translation_mapping: bool = False
    # Dual-arm: force which tracker this side uses (e.g. T20 / T21), and
    # optionally override the raw translation matrix per side (None = use the
    # measured module-level matrix).
    tracker_device_id: str | None = None
    raw_translation_matrix: tuple[tuple[float, float, float], ...] | None = None
    # Override the calibrated rotation mapping (raw tracker relative rotation
    # vector -> EEF-local rotation vector) when re-derived for a specific
    # pose. Columns are the mapped directions for raw X (roll), raw Y (pitch),
    # raw Z (yaw). None keeps the built-in calibrated matrices.
    rotation_mapping_matrix: tuple[tuple[float, float, float], ...] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.gripper_filter_window < 1 or self.gripper_filter_window % 2 == 0:
            raise ValueError("gripper_filter_window must be a positive odd number")
        if not 0 < self.gripper_filter_alpha <= 1:
            raise ValueError("gripper_filter_alpha must be in (0, 1]")
        if self.gripper_deadband < 0:
            raise ValueError("gripper_deadband must be non-negative")
        if not 0 < self.gripper_max_step <= 1:
            raise ValueError("gripper_max_step must be in (0, 1]")
        if self.gripper_distance_min_mm >= self.gripper_distance_max_mm:
            raise ValueError(
                "gripper_distance_min_mm must be smaller than "
                "gripper_distance_max_mm"
            )
        if self.rotation_scale == 0 or abs(self.rotation_scale) > 1:
            raise ValueError("rotation_scale must be in [-1, 0) or (0, 1]")
        if self.rotation_style not in ("calibrated", "world_delta", "senior"):
            raise ValueError(
                "rotation_style must be 'calibrated', 'world_delta' or 'senior'"
            )
        if len(self.tracker_world_to_robot_base_rpy) != 3 or not all(
            math.isfinite(value) for value in self.tracker_world_to_robot_base_rpy
        ):
            raise ValueError(
                "tracker_world_to_robot_base_rpy must contain three finite values"
            )
        if not 0 < self.rotation_filter_alpha <= 1:
            raise ValueError("rotation_filter_alpha must be in (0, 1]")
        if self.raw_translation_matrix is not None:
            if len(self.raw_translation_matrix) != 3 or any(
                len(row) != 3 for row in self.raw_translation_matrix
            ):
                raise ValueError("raw_translation_matrix must be a 3x3 matrix")
        if self.rotation_mapping_matrix is not None:
            if len(self.rotation_mapping_matrix) != 3 or any(
                len(row) != 3 for row in self.rotation_mapping_matrix
            ):
                raise ValueError("rotation_mapping_matrix must be a 3x3 matrix")
