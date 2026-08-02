from concurrent.futures import ThreadPoolExecutor
from typing import Any

from lerobot.teleoperators import Teleoperator
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot_robot_ufactory.teleoperators.base_teleop import UFBaseTeleop

from .config import DualPiperLeaderConfig, PiperLeaderConfig
from .motors import PiperMotorsBus
from .motors.tables import CALIBRATION, MOTORS


class PiperLeader(Teleoperator):
    config_class = PiperLeaderConfig
    name = "piper_leader"

    def __init__(self, config: PiperLeaderConfig, prefix: str = "") -> None:
        super().__init__(config)
        self.config = config
        self.prefix = f"{prefix}." if prefix else ""
        self.bus = PiperMotorsBus(
            id=config.id or prefix or "piper-leader",
            port=config.port,
            motors=MOTORS.copy(),
            calibration=CALIBRATION.copy(),
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{self.prefix}{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if self.config.configure_role_on_connect:
            self.bus.set_leader()
        self.bus.enable_torque()

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def setup_motors(self) -> None:
        self.bus.connect()
        self.bus.set_leader()

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        return {
            f"{self.prefix}{motor}.pos": value
            for motor, value in self.bus.get_leader_position().items()
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return None

    def disconnect(self) -> None:
        self.bus.disconnect(
            disable_torque=self.config.disable_torque_on_disconnect,
            park=False,
        )


class DualPiperLeader(UFBaseTeleop):
    config_class = DualPiperLeaderConfig
    name = "dual_piper_leader"

    def __init__(self, config: DualPiperLeaderConfig) -> None:
        super().__init__(config)
        self.config = config
        self.teleops: dict[str, PiperLeader] = {}
        for side, teleop_config in config.teleops.items():
            if not isinstance(teleop_config, PiperLeaderConfig):
                raise TypeError(f"{side} must use type uf::piper_leader, got {teleop_config.type}")
            self.teleops[side] = PiperLeader(teleop_config, prefix=side)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dual-piper-leader")

    @property
    def action_features(self) -> dict[str, type]:
        result: dict[str, type] = {}
        for teleop in self.teleops.values():
            result.update(teleop.action_features)
        return result

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return all(teleop.is_connected for teleop in self.teleops.values())

    @property
    def is_calibrated(self) -> bool:
        return all(teleop.is_calibrated for teleop in self.teleops.values())

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate)
        try:
            if self.config.parallel_connect:
                futures = [
                    self._executor.submit(teleop.connect, calibrate=calibrate)
                    for teleop in self.teleops.values()
                ]
                for future in futures:
                    future.result()
            else:
                for teleop in self.teleops.values():
                    teleop.connect(calibrate=calibrate)
        except BaseException:
            for teleop in self.teleops.values():
                if teleop.is_connected:
                    try:
                        teleop.disconnect()
                    except Exception:
                        pass
            super().disconnect()
            raise

    def calibrate(self) -> None:
        for teleop in self.teleops.values():
            teleop.calibrate()

    def configure(self) -> None:
        for teleop in self.teleops.values():
            teleop.configure()

    def set_teleop_enabled(self, enabled: bool, obs: dict | None = None) -> None:
        # A mechanical Piper leader is continuously readable; episode gating is
        # handled by the parent recorder rather than by the CAN device.
        return None

    def get_action(self) -> dict[str, Any]:
        if self.config.parallel_read:
            futures = [self._executor.submit(teleop.get_action) for teleop in self.teleops.values()]
            actions = [future.result() for future in futures]
        else:
            actions = [teleop.get_action() for teleop in self.teleops.values()]
        merged: dict[str, Any] = {}
        for action in actions:
            merged.update(action)
        return merged

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return None

    def disconnect(self) -> None:
        try:
            futures = [self._executor.submit(teleop.disconnect) for teleop in self.teleops.values()]
            for future in futures:
                future.result()
        finally:
            self._executor.shutdown(wait=True)
            super().disconnect()
