"""Pose conversion helpers for UFACTORY/Pika axis-angle and Piper RPY."""

import math

import numpy as np

from lerobot_robot_ufactory.devices.umi.vive_tracker.transformations import Transformations


def axis_angle_to_rpy_degrees(rx: float, ry: float, rz: float) -> tuple[float, float, float]:
    rotation = Transformations.rxryrz_to_rotation_matrix(rx, ry, rz)
    roll, pitch, yaw = Transformations.rotation_matrix_to_rpy(rotation)
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def rpy_degrees_to_axis_angle(
    roll_deg: float, pitch_deg: float, yaw_deg: float
) -> tuple[float, float, float]:
    rotation = Transformations.rpy_to_rotation_matrix(
        math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    )
    result = Transformations.rotation_matrix_to_rxryrz(rotation)
    return tuple(float(value) for value in result)


def rotation_distance(
    rxryrz_a: tuple[float, float, float],
    rxryrz_b: tuple[float, float, float],
) -> float:
    """True angular distance (rad) between two axis-angle rotations."""
    ra = Transformations.rxryrz_to_rotation_matrix(*rxryrz_a)
    rb = Transformations.rxryrz_to_rotation_matrix(*rxryrz_b)
    d = ra.T @ rb
    cos_theta = min(1.0, max(-1.0, (float(np.trace(d)) - 1.0) / 2.0))
    return math.acos(cos_theta)


def clamp(value: float, bounds: tuple[float, float] | None) -> float:
    if bounds is None:
        return value
    return min(bounds[1], max(bounds[0], value))


def vector_step_towards(
    current: tuple[float, ...], target: tuple[float, ...], max_step: float
) -> tuple[float, ...]:
    delta = tuple(target_value - current_value for current_value, target_value in zip(current, target))
    norm = math.sqrt(sum(value * value for value in delta))
    if norm <= max_step or norm == 0.0:
        return target
    scale = max_step / norm
    return tuple(current_value + value * scale for current_value, value in zip(current, delta))
