import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from lerobot_robot_ufactory.teleoperators.base_teleop import UFBaseTeleop
from lerobot_robot_ufactory.teleoperators.pika_teleop import (
    PikaTeleop as UFactoryPikaTeleop,
)
from lerobot_robot_ufactory.teleoperators.pika_teleop import PikaTeleopConfig

from .config import DualPikaTeleopConfig


class _PikaTeleop(UFactoryPikaTeleop):
    """Local bug-fix subclass; the parent Pika implementation is left untouched."""

    def run(self) -> None:
        self._is_connected = True
        initial_state = self.pika_sense.get_command_state()
        current_state = initial_state
        sleep_time = 1.0 / self.config.frequency
        while not self.stop_event.is_set():
            time.sleep(sleep_time)
            state = self.pika_sense.get_command_state()
            if state == current_state:
                continue
            current_state = state
            if not self._teleop_enabled and current_state != initial_state:
                self.set_teleop_enabled(True, self._last_action)
                time.sleep(1.0)
            elif self._teleop_enabled and current_state == initial_state:
                if self._last_action is not None:
                    self.set_teleop_enabled(False)
                else:
                    with self._data_lock:
                        self._teleop_enabled = False
                        self.begin_tracker_robot_matrix = None


class DualPikaTeleop(UFBaseTeleop):
    config_class = DualPikaTeleopConfig
    name = "dual_pika_teleop"

    def __init__(self, config: DualPikaTeleopConfig) -> None:
        super().__init__(config)
        self.config = config
        self.teleops: dict[str, _PikaTeleop] = {}
        for side, teleop_config in config.teleops.items():
            if not isinstance(teleop_config, PikaTeleopConfig):
                raise TypeError(f"{side} must use type uf::pika_teleop, got {teleop_config.type}")
            self.teleops[side] = _PikaTeleop(teleop_config, prefix=side)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dual-pika")

    @property
    def action_features(self) -> dict[str, type]:
        result: dict[str, type] = {}
        for side, teleop in self.teleops.items():
            names = teleop.action_features["names"]
            result.update({f"{side}.{name}": float for name in names})
        return result

    @property
    def feedback_features(self) -> dict[str, type]:
        return self.action_features

    @property
    def is_connected(self) -> bool:
        return all(teleop.is_connected for teleop in self.teleops.values())

    @property
    def is_calibrated(self) -> bool:
        return all(teleop.is_calibrated for teleop in self.teleops.values())

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate)
        try:
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
        for teleop in self.teleops.values():
            teleop.set_teleop_enabled(enabled, obs)

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
            for teleop in self.teleops.values():
                teleop.disconnect()
        finally:
            self._executor.shutdown(wait=True)
            super().disconnect()
