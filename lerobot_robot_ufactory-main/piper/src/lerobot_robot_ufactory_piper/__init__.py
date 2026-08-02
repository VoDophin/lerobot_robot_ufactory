"""Dual-arm Piper plugin for the UFACTORY LeRobot integration.

Importing this package registers all added RobotConfig and TeleoperatorConfig
choices. Existing files in ``lerobot_robot_ufactory`` are not patched here.
"""

# Import the parent plugin first so uf::pika_teleop and uf::multiple_mock_robot
# are registered before nested YAML configurations are decoded.
import lerobot_robot_ufactory  # noqa: F401

from .config import (
    DualPikaTeleopConfig,
    DualPiperFollowerConfig,
    DualPiperLeaderConfig,
    PiperFollowerConfig,
    PiperLeaderConfig,
)
from .piper_follower import DualPiperFollower, PiperFollower
from .piper_leader import DualPiperLeader, PiperLeader
from .pika_teleop import DualPikaTeleop

__all__ = [
    "DualPikaTeleop",
    "DualPikaTeleopConfig",
    "DualPiperFollower",
    "DualPiperFollowerConfig",
    "DualPiperLeader",
    "DualPiperLeaderConfig",
    "PiperFollower",
    "PiperFollowerConfig",
    "PiperLeader",
    "PiperLeaderConfig",
]

