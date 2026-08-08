import logging
import time
from typing import Any

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from piper_sdk import C_PiperInterface_V2

from .tables import PARKING_POSITION

logger = logging.getLogger(__name__)


class PiperMotorsBus:
    """CAN bus adapter derived from the standalone lerobot_robot_piper project."""

    apply_drive_mode = False

    def __init__(
        self,
        id: str,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration],
    ) -> None:
        self.id = id
        self.port = port
        self.motors = motors
        self.calibration = calibration
        # The field setup uses a generic gs_usb/candleLight adapter rather than
        # the Piper SDK's auto-detected official adapter path.  Disabling the
        # judge flag keeps SDK activation checks from making assumptions about
        # the USB-CAN hardware while SocketCAN still owns bitrate/link setup.
        self.piper = C_PiperInterface_V2(port, False)
        self._is_connected = False
        self._last_motion_ctrl: tuple[int, int, int, int] | None = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, handshake: bool = True, *, piper_init: bool = True) -> None:
        if self._is_connected:
            return
        connect_port = getattr(self.piper, "ConnectPort", None)
        if callable(connect_port):
            result = connect_port(piper_init=piper_init)
            if result is False:
                raise ConnectionError(f"Failed to open CAN port {self.port!r} for {self.id}")
        self._is_connected = True
        self._last_motion_ctrl = None

    def disconnect(self, disable_torque: bool = True, park: bool = False) -> None:
        if not self._is_connected:
            return
        if park:
            self.parking()
        if disable_torque:
            self._disable_arm()
        for method_name in ("DisconnectPort", "ClosePort", "Disconnect"):
            close_port = getattr(self.piper, method_name, None)
            if callable(close_port):
                try:
                    close_port()
                except Exception:
                    logger.debug("Ignoring Piper %s cleanup failure", method_name, exc_info=True)
                break
        self._is_connected = False

    def read(self, data_name: str, motor: str) -> int | float:
        return self.get_joint_position().get(motor, 0.0)

    def write(self, data_name: str, motor: str, value: int | float) -> None:
        current = self.get_joint_position()
        current[motor] = value
        self.set_joint_position(current)

    def sync_read(
        self, data_name: str, motors: str | list[str] | None = None
    ) -> dict[str, int | float]:
        position = self.get_joint_position()
        if motors is None:
            return position
        selected = [motors] if isinstance(motors, str) else motors
        return {motor: position[motor] for motor in selected if motor in position}

    def sync_write(self, data_name: str, values: dict[str, int | float]) -> None:
        self.set_joint_position(values)

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        retries = num_retry if num_retry > 0 else 50
        while retries > 0:
            if self._enable_arm():
                return
            retries -= 1
            time.sleep(0.1)
        raise TimeoutError(f"Timed out enabling Piper arm {self.id} on {self.port}")

    def disable_torque(
        self, motors: str | list[str] | None = None, num_retry: int = 0
    ) -> None:
        self._disable_arm()

    def _enable_arm(self) -> bool:
        return self._call_arm_power_method(("EnablePiper", "EnableArm"))

    def _disable_arm(self) -> bool:
        return self._call_arm_power_method(("DisablePiper", "DisableArm"))

    def _call_arm_power_method(self, method_names: tuple[str, ...]) -> bool:
        """Call the arm enable/disable method across Piper SDK versions.

        Older examples use ``EnablePiper``/``DisablePiper`` while newer
        ``C_PiperInterface_V2`` builds expose ``EnableArm``/``DisableArm``.
        Some SDK builds take no argument and some take a motor index/group, so
        try the no-argument call first and then the all-motors id used by Piper.
        """

        type_errors: list[TypeError] = []
        for method_name in method_names:
            method = getattr(self.piper, method_name, None)
            if not callable(method):
                continue
            for args in ((), (7,)):
                try:
                    result = method(*args)
                except TypeError as exc:
                    type_errors.append(exc)
                    continue
                return True if result is None else bool(result)
        if type_errors:
            raise type_errors[-1]
        raise AttributeError(
            f"Piper SDK object has none of the expected methods: {', '.join(method_names)}"
        )

    def read_calibration(self) -> dict[str, MotorCalibration]:
        return self.calibration

    def write_calibration(
        self, calibration_dict: dict[str, MotorCalibration], cache: bool = True
    ) -> None:
        self.calibration = calibration_dict

    def clear_gripper(self) -> None:
        self.piper.GripperCtrl(0, 1000, 0x03, 0)

    def set_follower(self) -> None:
        self.piper.MasterSlaveConfig(0xFC, 0, 0, 0)

    def set_leader(self) -> None:
        self.piper.MasterSlaveConfig(0xFA, 0, 0, 0)

    def emergency_stop(self) -> None:
        """Latch the Piper software emergency stop without resetting power."""
        self.piper.EmergencyStop(0x01)

    def resume_from_emergency_stop(self) -> None:
        """Clear a latched software emergency stop; this does not send a pose target."""
        self.piper.EmergencyStop(0x02)

    def parking(self) -> None:
        self.set_joint_position(PARKING_POSITION)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            status = self.piper.GetArmStatus()
            if not status.arm_status.motion_status:
                return
            time.sleep(0.1)
        raise TimeoutError(f"Piper arm {self.id} did not finish parking in 10 seconds")

    def get_joint_position(self) -> dict[str, float]:
        joint = self.piper.GetArmJointMsgs().joint_state
        gripper = self.piper.GetArmGripperMsgs().gripper_state
        raw = {
            "joint1": float(joint.joint_1),
            "joint2": float(joint.joint_2),
            "joint3": float(joint.joint_3),
            "joint4": float(joint.joint_4),
            "joint5": float(joint.joint_5),
            "joint6": float(joint.joint_6),
            "gripper": float(gripper.grippers_angle),
        }
        return self._normalize(raw)

    def get_leader_position(self) -> dict[str, float]:
        joint = self.piper.GetArmJointCtrl().joint_ctrl
        gripper = self.piper.GetArmGripperCtrl().gripper_ctrl
        raw = {
            "joint1": float(joint.joint_1),
            "joint2": float(joint.joint_2),
            "joint3": float(joint.joint_3),
            "joint4": float(joint.joint_4),
            "joint5": float(joint.joint_5),
            "joint6": float(joint.joint_6),
            "gripper": float(gripper.grippers_angle),
        }
        return self._normalize(raw)

    def get_end_pose(self) -> tuple[float, float, float, float, float, float]:
        pose = self.piper.GetArmEndPoseMsgs().end_pose
        # SDK units are 0.001 mm and 0.001 degree.
        return (
            float(pose.X_axis) / 1000.0,
            float(pose.Y_axis) / 1000.0,
            float(pose.Z_axis) / 1000.0,
            float(pose.RX_axis) / 1000.0,
            float(pose.RY_axis) / 1000.0,
            float(pose.RZ_axis) / 1000.0,
        )

    def set_joint_position(self, action: dict[str, float], speed_percent: int = 30) -> None:
        missing = set(self.motors) - set(action)
        if missing:
            raise KeyError(f"Piper joint action is missing: {sorted(missing)}")
        raw = self._unnormalize(action)
        self._set_motion_ctrl(0x01, 0x01, speed_percent, 0x00)
        self.piper.JointCtrl(*(int(raw[f"joint{i}"]) for i in range(1, 7)))
        self.set_gripper_percent(action["gripper"])

    def set_end_pose(
        self,
        pose_mm_rpy_deg: tuple[float, float, float, float, float, float],
        *,
        move_mode: str,
        speed_percent: int,
    ) -> None:
        move_mode_codes = {
            "move_p": 0x00,
            "move_l": 0x02,
            "move_cpv": 0x05,
        }
        try:
            move_mode_code = move_mode_codes[move_mode]
        except KeyError as exc:
            raise ValueError(f"Unsupported Piper move mode: {move_mode}") from exc
        raw = [int(round(value * 1000.0)) for value in pose_mm_rpy_deg]
        self._set_motion_ctrl(0x01, move_mode_code, speed_percent, 0x00)
        self.piper.EndPoseCtrl(*raw)

    def set_gripper_percent(
        self,
        value: float,
        effort: int = 1000,
        ctrl_code: int = 0x01,
    ) -> None:
        if ctrl_code not in (0x01, 0x03):
            raise ValueError("Piper gripper ctrl_code must be 0x01 or 0x03")
        raw = self._unnormalize({"gripper": value})["gripper"]
        self.piper.GripperCtrl(abs(int(raw)), effort, ctrl_code, 0)

    def _set_motion_ctrl(
        self,
        ctrl_mode: int,
        move_mode: int,
        speed_percent: int,
        mit_mode: int,
    ) -> None:
        command = (ctrl_mode, move_mode, speed_percent, mit_mode)
        if self._last_motion_ctrl == command:
            return
        mode_ctrl = getattr(self.piper, "ModeCtrl", None)
        if callable(mode_ctrl):
            mode_ctrl(*command)
        else:
            self.piper.MotionCtrl_2(*command)
        self._last_motion_ctrl = command
        time.sleep(0.02)

    def _normalize(self, raw_values: dict[str, float]) -> dict[str, float]:
        result: dict[str, float] = {}
        for motor, value in raw_values.items():
            calibration = self.calibration[motor]
            minimum, maximum = calibration.range_min, calibration.range_max
            bounded = min(maximum, max(minimum, value))
            if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
                result[motor] = ((bounded - minimum) / (maximum - minimum)) * 200.0 - 100.0
            elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
                result[motor] = ((bounded - minimum) / (maximum - minimum)) * 100.0
            else:
                raise NotImplementedError(self.motors[motor].norm_mode)
        return result

    def _unnormalize(self, values: dict[str, float]) -> dict[str, int]:
        result: dict[str, int] = {}
        for motor, value in values.items():
            calibration = self.calibration[motor]
            minimum, maximum = calibration.range_min, calibration.range_max
            if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
                bounded = min(100.0, max(-100.0, float(value)))
                raw = ((bounded + 100.0) / 200.0) * (maximum - minimum) + minimum
            elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
                bounded = min(100.0, max(0.0, float(value)))
                raw = (bounded / 100.0) * (maximum - minimum) + minimum
            else:
                raise NotImplementedError(self.motors[motor].norm_mode)
            result[motor] = int(raw)
        return result
