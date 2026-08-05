import math
import time
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Any, Tuple

import numpy as np
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.utils.errors import DeviceNotConnectedError

from lerobot_robot_ufactory.devices.pika import PikaDevice
from lerobot_robot_ufactory.devices.umi.vive_tracker.transformations import Transformations
from lerobot_robot_ufactory.teleoperators.base_teleop import UFBaseTeleop


@TeleoperatorConfig.register_subclass("uf::pika_split_teleop")
@dataclass
class SplitPikaTeleopConfig(TeleoperatorConfig):
    port: str = None
    frequency: int = 100
    use_gripper: bool = True
    scale_xyz: float = 0.4

    translation_map_rpy: Tuple[float, float, float] = (0, 0, 0)
    lock_orientation: bool = True

    rotation_map_rpy: Tuple[float, float, float] = (0, 0, 0)
    rotation_scale: float = 0.25

    # added by ljr
    rotation_axis_map: tuple[int, int, int] = (0, 1, 2)
    rotation_axis_sign: tuple[float, float, float] = (1.0, 1.0, 1.0)

    robot_base_pose: Tuple[float, ...] = (300, 0, 250, 180, 0, 0)

    def __post_init__(self):
        self.id = "pika_split_teleop" if self.id is None else self.id
        if self.frequency <= 0:
            raise ValueError("frequency must be positive")
        if self.scale_xyz <= 0:
            raise ValueError("scale_xyz must be positive")
        if self.rotation_scale <= 0:
            raise ValueError("rotation_scale must be positive")


class SplitPikaTeleop(UFBaseTeleop, Thread):
    config_class = SplitPikaTeleopConfig
    name = "Pika Split Teleop For Piper"

    def __init__(self, config: SplitPikaTeleopConfig, prefix: str = ""):
        UFBaseTeleop.__init__(self, config)
        Thread.__init__(self)
        self.stop_event = Event()
        self.config = config
        self.prefix = "" if not prefix else f"{prefix}."

        self._is_connected = False
        self._is_calibrated = True
        self._data_lock = Lock()
        self._teleop_enabled = False
        self._last_action: dict[str, float] | None = None

        self._pika_start_xyz: np.ndarray | None = None
        self._pika_start_rot: np.ndarray | None = None
        self._piper_start_xyz: np.ndarray | None = None
        self._piper_start_rot_vec: np.ndarray | None = None

        tx, ty, tz, rx, ry, rz = self.config.robot_base_pose
        self._fallback_pose = np.array(
            [tx, ty, tz, math.radians(rx), math.radians(ry), math.radians(rz)],
            dtype=float,
        )

        self.translation_map = self._rpy_deg_to_matrix(self.config.translation_map_rpy)
        self.rotation_map = self._rpy_deg_to_matrix(self.config.rotation_map_rpy)

        self.pika_device = PikaDevice(1, pika_sense_port=self.config.port)
        self.pika_sense = self.pika_device.pika_sense

    @staticmethod
    def _rpy_deg_to_matrix(rpy_deg: Tuple[float, float, float]) -> np.ndarray:
        roll, pitch, yaw = (math.radians(v) for v in rpy_deg)
        return Transformations.rpy_to_rotation_matrix(roll, pitch, yaw)

    @property
    def action_features(self) -> dict:
        names = {
            "pose.x": 0,
            "pose.y": 1,
            "pose.z": 2,
            "pose.rx": 3,
            "pose.ry": 4,
            "pose.rz": 5,
        }
        if self.config.use_gripper:
            names["gripper.pos"] = 6
        return {
            "dtype": "float32",
            "shape": (len(names),),
            "names": names,
        }

    @property
    def feedback_features(self) -> dict:
        return self.action_features

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = False) -> None:
        super().connect(calibrate)
        self.start()

    def disconnect(self):
        if not self._is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")
        self.stop_event.set()
        self._is_connected = False
        self.join()

    def _current_pika_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        pose = self.pika_sense.get_pose(self.pika_device.pika_tracker_device)
        if not pose:
            return None
        xyz = np.array(
            [
                pose.position[0] * 1000.0 * self.config.scale_xyz,
                pose.position[1] * 1000.0 * self.config.scale_xyz,
                pose.position[2] * 1000.0 * self.config.scale_xyz,
            ],
            dtype=float,
        )
        print("RAW_QUAT", pose.rotation)
        matrix = Transformations.xyzq_to_rotation_matrix(
            xyz[0], xyz[1], xyz[2], pose.rotation
        )
        rot = matrix[:3, :3]
        return xyz, rot

    def set_teleop_enabled(self, enabled: bool, obs: dict | None = None) -> None:
        with self._data_lock:
            if enabled:
                if obs is not None:
                    piper_pose = np.array(
                        [
                            obs[f"{self.prefix}pose.x"],
                            obs[f"{self.prefix}pose.y"],
                            obs[f"{self.prefix}pose.z"],
                            obs[f"{self.prefix}pose.rx"],
                            obs[f"{self.prefix}pose.ry"],
                            obs[f"{self.prefix}pose.rz"],
                        ],
                        dtype=float,
                    )
                elif self._last_action is not None:
                    piper_pose = np.array(
                        [
                            self._last_action[f"{self.prefix}pose.x"],
                            self._last_action[f"{self.prefix}pose.y"],
                            self._last_action[f"{self.prefix}pose.z"],
                            self._last_action[f"{self.prefix}pose.rx"],
                            self._last_action[f"{self.prefix}pose.ry"],
                            self._last_action[f"{self.prefix}pose.rz"],
                        ],
                        dtype=float,
                    )
                else:
                    piper_pose = self._fallback_pose.copy()

                current = self._current_pika_pose()
                if current is None:
                    self._teleop_enabled = False
                    print(f"[{self.prefix}PIKA_SPLIT] No tracker pose; teleop not started")
                    return

                pika_xyz, pika_rot = current
                self._pika_start_xyz = pika_xyz
                self._pika_start_rot = pika_rot
                self._piper_start_xyz = piper_pose[:3]
                self._piper_start_rot_vec = piper_pose[3:6]
                self._teleop_enabled = True
                self._last_action = self._pose_to_action(piper_pose)
                print(f"[{self.prefix}PIKA_SPLIT] Teleoperation is start")
            else:
                self._teleop_enabled = False
                self._pika_start_xyz = None
                self._pika_start_rot = None
                self._piper_start_xyz = None
                self._piper_start_rot_vec = None
                print(f"[{self.prefix}PIKA_SPLIT] Teleoperation has paused")

    def _pose_to_action(self, pose: np.ndarray) -> dict[str, float]:
        action = {
            f"{self.prefix}pose.x": float(pose[0]),
            f"{self.prefix}pose.y": float(pose[1]),
            f"{self.prefix}pose.z": float(pose[2]),
            f"{self.prefix}pose.rx": float(pose[3]),
            f"{self.prefix}pose.ry": float(pose[4]),
            f"{self.prefix}pose.rz": float(pose[5]),
        }
        if self.config.use_gripper:
            action[f"{self.prefix}gripper.pos"] = 0.0
        return action

    def run(self):
        self._is_connected = True
        init_state = self.pika_sense.get_command_state()
        curr_state = init_state
        sleep_time = 1.0 / self.config.frequency

        while not self.stop_event.is_set():
            time.sleep(sleep_time)
            state = self.pika_sense.get_command_state()
            if state == curr_state:
                continue
            curr_state = state
            if not self._teleop_enabled and curr_state != init_state:
                self.set_teleop_enabled(True, self._last_action)
                time.sleep(1.0)
            elif self._teleop_enabled and curr_state == init_state:
                self.set_teleop_enabled(False)

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "SplitPikaTeleop is not connected. You need to run connect() first."
            )

        with self._data_lock:
            if self._last_action is None:
                self._last_action = self._pose_to_action(self._fallback_pose)

            if not self._teleop_enabled:
                return dict(self._last_action)

            if (
                self._pika_start_xyz is None
                or self._pika_start_rot is None
                or self._piper_start_xyz is None
                or self._piper_start_rot_vec is None
            ):
                return dict(self._last_action)

        current = self._current_pika_pose()
        if current is None:
            return dict(self._last_action)

        pika_xyz, pika_rot = current

        with self._data_lock:
            pika_delta_xyz = pika_xyz - self._pika_start_xyz
            target_xyz = self._piper_start_xyz + self.translation_map @ pika_delta_xyz
#            target_xyz = self._piper_start_xyz 
            if self.config.lock_orientation:
                target_rot_vec = self._piper_start_rot_vec
            else:
                relative_rot = self._pika_start_rot.T @ pika_rot
                # relative_vec = Transformations.rotation_matrix_to_rxryrz(relative_rot)
                # mapped_vec = self.rotation_map @ relative_vec
                # mapped_vec = mapped_vec * self.config.rotation_scale
                relative_vec = np.asarray(
                    Transformations.rotation_matrix_to_rxryrz(relative_rot),
                    dtype=float,
                )

                mapped_vec = self.rotation_map @ relative_vec

                mapped_vec = np.array(
                   [
                        mapped_vec[self.config.rotation_axis_map[0]] * self.config.rotation_axis_sign[0],
                        mapped_vec[self.config.rotation_axis_map[1]] * self.config.rotation_axis_sign[1],
                        mapped_vec[self.config.rotation_axis_map[2]] * self.config.rotation_axis_sign[2],
                    ],
                    dtype=float,
                )

                mapped_vec = mapped_vec * self.config.rotation_scale

                piper_start_rot = Transformations.rxryrz_to_rotation_matrix(
                    *self._piper_start_rot_vec
                )
                target_rot = piper_start_rot @ Transformations.rxryrz_to_rotation_matrix(
                    *mapped_vec
                )
                target_rot_vec = Transformations.rotation_matrix_to_rxryrz(target_rot)
                print("SPLIT_ROT", target_rot_vec)

            target_pose = np.array(
                [
                    target_xyz[0],
                    target_xyz[1],
                    target_xyz[2],
                    target_rot_vec[0],
                    target_rot_vec[1],
                    target_rot_vec[2],
                ],
                dtype=float,
            )
            self._last_action.update(self._pose_to_action(target_pose))

            if self.config.use_gripper:
                distance = self.pika_sense.get_gripper_distance()
                if distance is not None:
                    gripper_pos = min(1.0, max(0.0, float(distance) / 100.0))
                    print("GRIPPER_DISTANCE", distance, "GRIPPER_POS", gripper_pos)
                    self._last_action[f"{self.prefix}gripper.pos"] = gripper_pos

            return dict(self._last_action)

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError
