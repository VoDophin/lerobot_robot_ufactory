import logging
import time
from typing import Any

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.motors_bus import MotorsBusBase
from piper_sdk import C_PiperInterface_V2
from wego_piper.port_handler import PortHandler

from .tables import PARKING_POSITION

logger = logging.getLogger(__name__)


class PiperMotorsBus(MotorsBusBase):
    """CAN bus adapter derived from the standalone lerobot_robot_piper project."""

    apply_drive_mode = False

    def __init__(
        self,
        id: str,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration],
    ) -> None:
        super().__init__(port, motors, calibration)
        self.id = id
        self.port_handler = PortHandler()
        self.piper = C_PiperInterface_V2(port)
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, handshake: bool = True) -> None:
        if self._is_connected:
            return
        self.port_handler.setupPort(self.piper)
        if not self.port_handler.openPort():
            raise ConnectionError(f"Failed to open CAN port {self.port!r} for {self.id}")
        self._is_connected = True

    def disconnect(self, disable_torque: bool = True, park: bool = False) -> None:
        if not self._is_connected:
            return
        if park:
            self.parking()
        if disable_torque:
            self.piper.DisablePiper()
        self.port_handler.closePort()
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
            if self.piper.EnablePiper():
                return
            retries -= 1
            time.sleep(0.1)
        raise TimeoutError(f"Timed out enabling Piper arm {self.id} on {self.port}")

    def disable_torque(
        self, motors: str | list[str] | None = None, num_retry: int = 0
    ) -> None:
        self.piper.DisablePiper()

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
        self.piper.ModeCtrl(0x01, 0x01, speed_percent, 0x00)
        self.piper.JointCtrl(*(int(raw[f"joint{i}"]) for i in range(1, 7)))
        self.set_gripper_percent(action["gripper"])

    def set_end_pose(
        self,
        pose_mm_rpy_deg: tuple[float, float, float, float, float, float],
        *,
        move_mode: str,
        speed_percent: int,
    ) -> None:
        move_mode_code = 0x00 if move_mode == "move_p" else 0x02
        raw = [int(round(value * 1000.0)) for value in pose_mm_rpy_deg]
        self.piper.ModeCtrl(0x01, move_mode_code, speed_percent, 0x00)
        self.piper.EndPoseCtrl(*raw)

    def set_gripper_percent(self, value: float, effort: int = 1000) -> None:
        raw = self._unnormalize({"gripper": value})["gripper"]
        self.piper.GripperCtrl(abs(int(raw)), effort, 0x03, 0)

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

