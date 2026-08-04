from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig
from lerobot.teleoperators import TeleoperatorConfig


@RobotConfig.register_subclass("uf::piper")
@dataclass(kw_only=True)
class PiperFollowerConfig(RobotConfig):
    """One Piper follower arm.

    ``cartesian`` actions use millimetres for xyz and radians as an axis-angle
    vector for rx/ry/rz, matching the UFACTORY Pika teleoperator.
    """

    port: str
    control_space: str = "joint"
    use_gripper: bool = True
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    configure_role_on_connect: bool = True
    park_on_connect: bool = False
    park_on_disconnect: bool = False
    disable_torque_on_disconnect: bool = True

    # Joint mode values are normalized to [-100, 100] (gripper [0, 100]).
    max_relative_target: float | dict[str, float] | None = 5.0

    # Cartesian/Pika safety limits. Translation is mm, rotation is rad.
    max_cartesian_step_mm: float = 10.0
    max_rotation_step_rad: float = 0.10
    workspace_x: tuple[float, float] | None = None
    workspace_y: tuple[float, float] | None = None
    workspace_z: tuple[float, float] | None = None
    move_mode: str = "move_p"
    move_speed_percent: int = 20
    gripper_effort: int = 1000

    def __post_init__(self) -> None:
        super().__post_init__()
        self.id = "piper_follower" if self.id is None else self.id
        if self.control_space not in ("joint", "cartesian"):
            raise ValueError(f"Unsupported Piper control_space: {self.control_space}")
        if not 1 <= self.move_speed_percent <= 100:
            raise ValueError("move_speed_percent must be in [1, 100]")
        if self.max_cartesian_step_mm <= 0 or self.max_rotation_step_rad <= 0:
            raise ValueError("Cartesian step limits must be positive")
        for name in ("workspace_x", "workspace_y", "workspace_z"):
            bounds = getattr(self, name)
            if bounds is not None and bounds[0] >= bounds[1]:
                raise ValueError(f"{name} must be ordered as (min, max)")


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
