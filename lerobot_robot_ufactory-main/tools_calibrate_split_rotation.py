#!/usr/bin/env python

import argparse
import time

import numpy as np

from lerobot_robot_ufactory.devices.pika import PikaDevice
from lerobot_robot_ufactory.devices.umi.vive_tracker.transformations import Transformations


AXIS_NAMES = ("rx", "ry", "rz")


def _normalize_quat(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        raise ValueError("zero quaternion")
    return q / norm


def _average_quaternion(quats):
    base = _normalize_quat(quats[0])
    aligned = []
    for q in quats:
        q = _normalize_quat(q)
        if np.dot(base, q) < 0:
            q = -q
        aligned.append(q)
    avg = np.mean(aligned, axis=0)
    return _normalize_quat(avg)


def _read_quaternion(pika_sense, tracker_device, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pose = pika_sense.get_pose(tracker_device)
        if pose is not None:
            return _normalize_quat(pose.rotation)
        time.sleep(0.02)
    raise RuntimeError("timeout waiting for Pika tracker pose")


def _sample_rotation(pika_sense, tracker_device, samples, interval_s, timeout_s):
    quats = []
    while len(quats) < samples:
        quats.append(_read_quaternion(pika_sense, tracker_device, timeout_s))
        time.sleep(interval_s)
    return Transformations.quaternion_to_rotation_matrix(_average_quaternion(quats))


def _format_vec(vec):
    return "[" + ", ".join(f"{v:+.4f}" for v in vec) + "]"


def _axis_config_from_vectors(vectors):
    axis_map = []
    axis_sign = []
    used = set()
    for vec in vectors:
        src = int(np.argmax(np.abs(vec)))
        sign = 1.0 if vec[src] >= 0 else -1.0
        axis_map.append(src)
        axis_sign.append(sign)
        used.add(src)
    return axis_map, axis_sign, len(used) == 3


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate rotation_axis_map/rotation_axis_sign for uf::pika_split_teleop."
    )
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Pika Sense serial port")
    parser.add_argument("--samples", type=int, default=40, help="samples to average for each pose")
    parser.add_argument("--interval", type=float, default=0.02, help="seconds between samples")
    parser.add_argument("--timeout", type=float, default=2.0, help="seconds to wait for each sample")
    args = parser.parse_args()

    pika_device = PikaDevice(1, pika_sense_port=args.port)
    pika_sense = pika_device.pika_sense
    tracker_device = pika_device.pika_tracker_device
    print(f"Tracker device: {tracker_device}")
    print()
    print("标定要求:")
    print("1. 每次只做旋转，尽量不要平移。")
    print("2. 每次从中立姿态开始，绕一个你希望 Piper 正方向旋转的轴转 30~60 度。")
    print("3. 三次动作分别对应 Piper 的 +rx、+ry、+rz 语义方向。")
    print()

    input("把 Pika 放到中立姿态，按 Enter 采样...")
    start_rot = _sample_rotation(pika_sense, tracker_device, args.samples, args.interval, args.timeout)

    vectors = []
    for axis_name in AXIS_NAMES:
        input(f"回到中立姿态，然后绕你希望 Piper +{axis_name} 的方向旋转 30~60 度，按 Enter 采样...")
        end_rot = _sample_rotation(pika_sense, tracker_device, args.samples, args.interval, args.timeout)
        relative_rot = start_rot.T @ end_rot
        vec = np.asarray(Transformations.rotation_matrix_to_rxryrz(relative_rot), dtype=float)
        vectors.append(vec)
        angle_deg = float(np.linalg.norm(vec) * 180.0 / np.pi)
        dominant = int(np.argmax(np.abs(vec)))
        sign = "+" if vec[dominant] >= 0 else "-"
        print(f"+{axis_name} raw axis-angle rad: {_format_vec(vec)}")
        print(f"+{axis_name} angle: {angle_deg:.1f} deg, dominant source: {sign}{AXIS_NAMES[dominant]}")
        print()

    axis_map, axis_sign, unique = _axis_config_from_vectors(vectors)
    sign_text = [int(v) for v in axis_sign]
    print("建议写入 config/pika/pika_piper_record_config.yaml:")
    print("  rotation_map_rpy: [0, 0, 0]")
    print(f"  rotation_axis_map: {axis_map}")
    print(f"  rotation_axis_sign: {sign_text}")
    print("  rotation_scale: 1.0")
    print()
    if not unique:
        print("警告: 三个动作里有至少两个被判定成同一个来源轴。")
        print("这通常说明某次旋转不够纯，建议重新测一次。")
    print("如果实机方向整体太猛或太慢，只改 rotation_scale，不要改 rotation_axis_map。")

    if hasattr(pika_sense, "disconnect"):
        pika_sense.disconnect()


if __name__ == "__main__":
    main()
