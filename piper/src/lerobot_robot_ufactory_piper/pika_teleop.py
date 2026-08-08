import math
import time
from collections import deque

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from lerobot_robot_ufactory.teleoperators.base_teleop import UFBaseTeleop
from lerobot_robot_ufactory.devices.umi.vive_tracker.transformations import (
    Transformations,
)
from lerobot_robot_ufactory.teleoperators.pika_teleop import (
    PikaTeleop as UFactoryPikaTeleop,
)
from lerobot_robot_ufactory.teleoperators.pika_teleop import PikaTeleopConfig

from .config import DualPikaTeleopConfig, PiperPikaTeleopConfig


# Calibrated on 2026-08-03 using Pika right/forward/up translation tests.
# Piper base convention: +X forward, -Y right, +Z up.
_ORIGINAL_PIKA_GET_ACTION = UFactoryPikaTeleop.get_action
_PIKA_TO_PIPER_TRANSLATION = np.array(
    [
        [0.52718794, -0.81988978, -0.22327926],
        [0.59104387,  0.16501522,  0.78958035],
        [-0.61052438, -0.54822508, 0.57158486],
    ],
    dtype=float,
)

# Calibrated from three Pika-only gestures on 2026-08-03.
# Re-fit on the new server (star, 2026-08-07) with
# piper/tools/measure_pika_piper_mapping.py for the current lighthouse/world
# frame. Maps raw tracker deltas in the lighthouse/world frame directly to the
# Piper base frame: Pika forward -> Piper +X, Pika right -> Piper -Y,
# Pika up -> +Z. The fit is within ~4 degrees of the previous matrix, i.e. the
# world frame is essentially unchanged.
_RAW_TO_PIPER_TRANSLATION = np.array(
    [
        [0.884906701, -0.441374693, -0.148756551],
        [0.441254089, 0.896674458, -0.035633477],
        [0.149113915, -0.034107134, 0.988231625],
    ],
    dtype=float,
)
# The rows map the
# original local axis-angle vector to the Piper gesture convention:
#   Pika tip right -> -RX, Pika tip up -> +RY, clockwise roll -> +RZ.
# The matrix is orthonormal. Experimental profiles can additionally keep only
# the dominant mapped axis to reject natural hand-motion cross coupling.
_PIKA_TO_PIPER_ROTATION = np.array(
    [
        [0.99266317, 0.11349864, -0.04168796],
        [-0.11927281, 0.97574876, -0.18354388],
        [0.01984499, 0.18716949, 0.98212716],
    ],
    dtype=float,
)

# Piper MOVE P rotation components are expressed in the end-effector command
# frame.  Around the verified 2026-08-03 test pose, that frame is rotated by
# 90 degrees relative to the operator's visual up/right convention.  Keep this
# compensation opt-in so the V2 and first-version profiles remain unchanged.
_PIPER_TOOL_AXIS_CORRECTION = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)


class _PikaTeleop(UFactoryPikaTeleop):
    """Local bug-fix subclass; the parent Pika implementation is left untouched."""

    def __init__(self, config, prefix: str = "") -> None:
        super().__init__(config, prefix=prefix)
        tracker_id = getattr(self.config, "tracker_device_id", None)
        if tracker_id:
            self.pika_device.pika_tracker_device = tracker_id
        matrix = getattr(self.config, "raw_translation_matrix", None)
        self._raw_to_piper = (
            np.asarray(matrix, dtype=float)
            if matrix is not None
            else _RAW_TO_PIPER_TRANSLATION
        )
        self._raw_start_xyz: np.ndarray | None = None
        self._last_mapped_translation: np.ndarray | None = None

    def set_teleop_enabled(self, enabled: bool, obs: dict | None = None) -> None:
        super().set_teleop_enabled(enabled, obs)
        # Every enable event must establish a fresh relative-motion origin.
        # Reusing the previous run's tracker origin can command a large jump.
        self._raw_start_xyz = None
        self._last_mapped_translation = None

    def get_action(self) -> dict[str, Any]:
        # The parent returns its mutable internal cache.  Apply the Piper axis
        # mapping to a copy so a temporary tracker dropout cannot transform the
        # already-transformed cache again on the next control cycle.
        action = dict(_ORIGINAL_PIKA_GET_ACTION(self))

        if not self._teleop_enabled:
            return action

        keys = (
            f"{self.prefix}pose.x",
            f"{self.prefix}pose.y",
            f"{self.prefix}pose.z",
        )
        origin = np.asarray(self._last_robot_pose[:3], dtype=float)
        raw_target = np.asarray([action[key] for key in keys], dtype=float)
        corrected_target = self._translation_target(raw_target, origin)

        for key, value in zip(keys, corrected_target, strict=True):
            action[key] = float(value)

        return action

    def _translation_target(
        self, raw_target: np.ndarray, origin: np.ndarray
    ) -> np.ndarray:
        if not getattr(self.config, "use_raw_translation_mapping", False):
            return origin + _PIKA_TO_PIPER_TRANSLATION @ (raw_target - origin)
        pose = self.pika_sense.get_pose(self.pika_device.pika_tracker_device)
        if pose is None:
            if self._last_mapped_translation is not None:
                return self._last_mapped_translation.copy()
            return origin.copy()
        raw_m = np.asarray(pose.position, dtype=float)
        if raw_m.shape != (3,) or not np.all(np.isfinite(raw_m)):
            if self._last_mapped_translation is not None:
                return self._last_mapped_translation.copy()
            return origin.copy()
        if self._raw_start_xyz is None:
            self._raw_start_xyz = raw_m.copy()
        delta_mm = (raw_m - self._raw_start_xyz) * 1000.0 * self.config.scale_xyz
        mapped = origin + self._raw_to_piper @ delta_mm
        self._last_mapped_translation = mapped.copy()
        return mapped

    def run(self) -> None:
        self._is_connected = True
        initial_state = self.pika_sense.get_command_state()
        current_state = initial_state
        sleep_time = 1.0 / self.config.frequency
        while not self.stop_event.is_set():
            time.sleep(sleep_time)
            # A DualPikaTeleop is enabled and paused as one unit by the parent
            # control loop.  Letting each child's hardware button state toggle
            # itself can immediately undo the parent's Enter-to-start event and
            # leave one or both sides paused.  Single-Pika instances do not set
            # this flag and retain the original button behavior.
            if getattr(self, "_external_enable_control", False):
                continue
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


class PiperPikaTeleop(_PikaTeleop):
    """Single-Pika profile with stable, rate-limited Piper gripper input."""

    config_class = PiperPikaTeleopConfig
    name = "piper_pika_teleop"

    def __init__(self, config: PiperPikaTeleopConfig, prefix: str = "") -> None:
        super().__init__(config, prefix=prefix)
        if config.rotation_mapping_matrix is not None:
            self._rotation_map = np.asarray(
                config.rotation_mapping_matrix, dtype=float
            )
        elif config.apply_piper_tool_axis_correction:
            self._rotation_map = (
                _PIPER_TOOL_AXIS_CORRECTION @ _PIKA_TO_PIPER_ROTATION
            )
        else:
            self._rotation_map = _PIKA_TO_PIPER_ROTATION
        self._gripper_samples: deque[float] = deque(
            maxlen=config.gripper_filter_window
        )
        self._filtered_gripper: float | None = None
        self._last_direct_gripper: float | None = None
        self._filtered_rotation_delta: np.ndarray | None = None
        self._begin_tracker_rot: np.ndarray | None = None
        self._begin_tracker_senior: tuple[np.ndarray, np.ndarray] | None = None
        if config.rotation_style == "senior":
            world_to_base_rpy = tuple(
                math.radians(float(v))
                for v in config.tracker_world_to_robot_base_rpy
            )
            self._senior_q = Transformations.rpy_to_rotation_matrix(
                *world_to_base_rpy
            )
            self._senior_eef_to_tracker = np.linalg.inv(
                self.tracker_to_robot_matrix
            )

    def set_teleop_enabled(self, enabled: bool, obs: dict | None = None) -> None:
        super().set_teleop_enabled(enabled, obs)
        self._gripper_samples.clear()
        self._last_direct_gripper = None
        self._filtered_rotation_delta = None
        self._begin_tracker_rot = None
        self._begin_tracker_senior = None
        if (
            enabled
            and self.config.pose_adaptive_rotation
            and self.config.rotation_style == "calibrated"
        ):
            # Same columns/signs as the derived flipped matrix, but rebuilt for
            # the current EEF orientation so yaw stays vertical, pitch stays
            # horizontal and roll stays about the gripper approach axis.
            r_eef = Transformations.rxryrz_to_rotation_matrix(
                *self._last_robot_pose[3:6]
            )
            e2 = np.array([0.0, 1.0, 0.0])
            e3 = np.array([0.0, 0.0, 1.0])
            self._rotation_map = np.column_stack(
                [-e3, -(r_eef.T @ e2), -(r_eef.T @ e3)]
            )
        # Begin at the actual Piper gripper position supplied in ``obs`` and
        # approach the Pika value gradually. This avoids a jump on Enter.
        self._filtered_gripper = (
            float(self._last_gripper_pos) if self.config.use_gripper else None
        )

    def get_action(self) -> dict[str, Any]:
        action = super().get_action()

        if (
            self._teleop_enabled
            and self.config.use_calibrated_rotation_mapping
        ):
            rotation_keys = (
                f"{self.prefix}pose.rx",
                f"{self.prefix}pose.ry",
                f"{self.prefix}pose.rz",
            )
            origin_rotation = Transformations.rxryrz_to_rotation_matrix(
                *self._last_robot_pose[3:6]
            )
            if self.config.rotation_style == "senior":
                # Full senior Lerobot-Real robot_base control target:
                #   G (gripper center) = Q @ (R_now R_start^T) @ Q^T @ G_base
                #   p_G = p_G0 + Q @ (p_now - p_start);  S (J6) = G @ E^-1
                pose = self.pika_sense.get_pose(
                    self.pika_device.pika_tracker_device
                )
                if pose is None:
                    return action
                position = np.asarray(pose.position, dtype=float)
                rotation = np.asarray(pose.rotation, dtype=float)
                if (
                    position.shape != (3,)
                    or rotation.shape != (4,)
                    or not np.all(np.isfinite(position))
                    or not np.all(np.isfinite(rotation))
                ):
                    return action
                p_now = position * 1000.0 * self.config.scale_xyz
                r_now = Transformations.quaternion_to_rotation_matrix(rotation)
                if self._begin_tracker_senior is None:
                    self._begin_tracker_senior = (p_now.copy(), r_now.copy())
                p0, r0 = self._begin_tracker_senior
                delta_rot = r_now @ r0.T
                q = self._senior_q
                gripper_base = (
                    self.robot_base_matrix @ self._senior_eef_to_tracker
                )
                target_gripper = np.eye(4)
                target_gripper[:3, :3] = (
                    q @ delta_rot @ q.T @ gripper_base[:3, :3]
                )
                target_gripper[:3, 3] = (
                    gripper_base[:3, 3] + q @ (p_now - p0)
                )
                robot_target = target_gripper @ self.tracker_to_robot_matrix
                senior_vec = np.asarray(
                    Transformations.rotation_matrix_to_xyzrxryrz(robot_target),
                    dtype=float,
                )
                pose_keys = (
                    f"{self.prefix}pose.x",
                    f"{self.prefix}pose.y",
                    f"{self.prefix}pose.z",
                    f"{self.prefix}pose.rx",
                    f"{self.prefix}pose.ry",
                    f"{self.prefix}pose.rz",
                )
                for key, value in zip(pose_keys, senior_vec, strict=True):
                    action[key] = float(value)
            elif self.config.rotation_style == "world_delta":
                # Senior Lerobot-Real robot_base formula: track the tracker's
                # WORLD-frame rotation delta, conjugated into the robot base:
                #   target = Q @ (R_now @ R_start^T) @ Q^T @ R_base
                pose = self.pika_sense.get_pose(
                    self.pika_device.pika_tracker_device
                )
                if pose is None:
                    return action
                rotation = np.asarray(pose.rotation, dtype=float)
                if rotation.shape != (4,) or not np.all(np.isfinite(rotation)):
                    return action
                tracker_rotation = (
                    Transformations.quaternion_to_rotation_matrix(rotation)
                )
                if self._begin_tracker_rot is None:
                    self._begin_tracker_rot = tracker_rotation.copy()
                delta_world = tracker_rotation @ self._begin_tracker_rot.T
                delta_vector = np.asarray(
                    Transformations.rotation_matrix_to_rxryrz(delta_world),
                    dtype=float,
                )
                if self.config.rotation_dominant_axis:
                    dominant_index = int(np.argmax(np.abs(delta_vector)))
                    dominant_vector = np.zeros(3, dtype=float)
                    dominant_vector[dominant_index] = delta_vector[dominant_index]
                    delta_vector = dominant_vector
                if self.config.rotation_filter_alpha < 1.0:
                    if self._filtered_rotation_delta is None:
                        self._filtered_rotation_delta = delta_vector.copy()
                    else:
                        alpha = self.config.rotation_filter_alpha
                        self._filtered_rotation_delta = (
                            alpha * delta_vector
                            + (1.0 - alpha) * self._filtered_rotation_delta
                        )
                    delta_vector = self._filtered_rotation_delta
                delta_vector *= self.config.rotation_scale
                delta_scaled = Transformations.rxryrz_to_rotation_matrix(
                    *delta_vector
                )
                q = self._raw_to_piper  # world -> robot base rotation
                target_matrix = q @ delta_scaled @ q.T @ origin_rotation
                corrected_vector = np.asarray(
                    Transformations.rotation_matrix_to_rxryrz(target_matrix),
                    dtype=float,
                )
                for key, value in zip(
                    rotation_keys, corrected_vector, strict=True
                ):
                    action[key] = float(value)
            else:
                target_rotation = Transformations.rxryrz_to_rotation_matrix(
                    *(float(action[key]) for key in rotation_keys)
                )
                relative_rotation = origin_rotation.T @ target_rotation
                relative_vector = Transformations.rotation_matrix_to_rxryrz(
                    relative_rotation
                )
                mapped_vector = self._rotation_map @ relative_vector
                if self.config.rotation_dominant_axis:
                    dominant_index = int(np.argmax(np.abs(mapped_vector)))
                    dominant_vector = np.zeros(3, dtype=float)
                    dominant_vector[dominant_index] = mapped_vector[dominant_index]
                    mapped_vector = dominant_vector
                if self.config.rotation_filter_alpha < 1.0:
                    if self._filtered_rotation_delta is None:
                        self._filtered_rotation_delta = mapped_vector.copy()
                    else:
                        alpha = self.config.rotation_filter_alpha
                        self._filtered_rotation_delta = (
                            alpha * mapped_vector
                            + (1.0 - alpha) * self._filtered_rotation_delta
                        )
                    mapped_vector = self._filtered_rotation_delta
                mapped_vector *= self.config.rotation_scale
                corrected_rotation = (
                    origin_rotation
                    @ Transformations.rxryrz_to_rotation_matrix(*mapped_vector)
                )
                corrected_vector = Transformations.rotation_matrix_to_rxryrz(
                    corrected_rotation
                )
                for key, value in zip(
                    rotation_keys, corrected_vector, strict=True
                ):
                    action[key] = float(value)

        if not self.config.use_gripper or not self._teleop_enabled:
            return action

        key = f"{self.prefix}gripper.pos"
        if self.config.gripper_use_direct_distance:
            # Pika reports physical jaw opening: approximately 0 mm closed and
            # 100 mm open. Piper's normalized gripper command follows the same
            # direction (0 closed, 1 open). The base UFACTORY implementation
            # inverts this value for xArm, so read the cached SDK distance here
            # and keep the Piper mapping explicit.
            distance_mm = self.pika_sense.get_gripper_distance()
            if distance_mm is not None:
                span_mm = (
                    self.config.gripper_distance_max_mm
                    - self.config.gripper_distance_min_mm
                )
                raw = (
                    float(distance_mm) - self.config.gripper_distance_min_mm
                ) / span_mm
                raw = min(1.0, max(0.0, raw))
                self._last_direct_gripper = raw
            elif self._last_direct_gripper is not None:
                raw = self._last_direct_gripper
            else:
                # On a first-frame serial dropout, hold the actual Piper
                # opening captured in set_teleop_enabled instead of jumping.
                raw = min(1.0, max(0.0, float(self._last_gripper_pos)))
        else:
            raw = min(1.0, max(0.0, float(action[key])))
        self._gripper_samples.append(raw)
        target = float(np.median(np.asarray(self._gripper_samples, dtype=float)))

        if self._filtered_gripper is None:
            self._filtered_gripper = target
        error = target - self._filtered_gripper
        if abs(error) > self.config.gripper_deadband:
            delta = self.config.gripper_filter_alpha * error
            delta = min(
                self.config.gripper_max_step,
                max(-self.config.gripper_max_step, delta),
            )
            self._filtered_gripper = min(
                1.0, max(0.0, self._filtered_gripper + delta)
            )

        action[key] = float(self._filtered_gripper)
        return action


# Apply the local fixed state-monitor loop to single-Pika teleoperation.
UFactoryPikaTeleop.run = _PikaTeleop.run
UFactoryPikaTeleop.get_action = _PikaTeleop.get_action

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
            if isinstance(teleop_config, PiperPikaTeleopConfig):
                self.teleops[side] = PiperPikaTeleop(teleop_config, prefix=side)
            else:
                self.teleops[side] = _PikaTeleop(teleop_config, prefix=side)
            self.teleops[side]._external_enable_control = True
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
