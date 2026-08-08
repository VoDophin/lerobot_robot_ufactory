#!/usr/bin/env python3
"""Measure the real Pika tracker axes and what the current piper mapping
would command, so the mapping matrices can be corrected with real data.

Read-only: Pika only, no Piper, no CAN. Run on the robot Linux PC:

    conda activate uf_lerobot
    cd ~/lerobot_robot_ufactory
    python piper/tools/measure_pika_piper_mapping.py --port /dev/ttyUSB50

For every gesture: put the Pika back at the neutral pose, press Enter, then
perform the motion and hold it, press Enter again. The script prints:

- raw tracker position delta (meters, lighthouse/world frame) and direction;
- what the current pipeline would command on the Piper: translation delta
  (mm, after scale + _PIKA_TO_PIPER_TRANSLATION) and rotation delta (rad,
  after _PIKA_TO_PIPER_ROTATION + tool-axis correction + rotation_scale).

Paste the full output back into the chat so the matrices can be re-derived.
"""

import argparse
import math
import time

import numpy as np

from lerobot_robot_ufactory.devices.pika import PikaDevice
from lerobot_robot_ufactory.devices.umi.vive_tracker.transformations import (
    Transformations,
)


# Keep in sync with piper/src/lerobot_robot_ufactory_piper/pika_teleop.py
_PIKA_TO_PIPER_TRANSLATION = np.array(
    [
        [0.52718794, -0.81988978, -0.22327926],
        [0.59104387, 0.16501522, 0.78958035],
        [-0.61052438, -0.54822508, 0.57158486],
    ],
    dtype=float,
)
_PIKA_TO_PIPER_ROTATION = np.array(
    [
        [0.99266317, 0.11349864, -0.04168796],
        [-0.11927281, 0.97574876, -0.18354388],
        [0.01984499, 0.18716949, 0.98212716],
    ],
    dtype=float,
)
_PIPER_TOOL_AXIS_CORRECTION = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)

_TRACKER_TO_ROBOT_EEF_DEG = (180.0, -90.0, 0.0)
_ROBOT_BASE_POSE = (300.0, 0.0, 250.0, math.radians(180.0), 0.0, 0.0)


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return None
    return v / n


def _fmt_vec(v) -> str:
    return "[" + ", ".join(f"{x:+.4f}" for x in v) + "]"


def _avg_quaternion(quats) -> np.ndarray:
    base = np.asarray(quats[0], dtype=float)
    base /= np.linalg.norm(base)
    aligned = []
    for q in quats:
        q = np.asarray(q, dtype=float)
        q /= np.linalg.norm(q)
        if np.dot(base, q) < 0:
            q = -q
        aligned.append(q)
    avg = np.mean(aligned, axis=0)
    return avg / np.linalg.norm(avg)


def _sample(sense, dev, n, interval_s, timeout_s) -> tuple[np.ndarray, np.ndarray]:
    positions: list[np.ndarray] = []
    quats: list[np.ndarray] = []
    deadline = time.monotonic() + n * interval_s + timeout_s
    while len(positions) < n and time.monotonic() < deadline:
        pose = sense.get_pose(dev)
        if pose is not None:
            positions.append(np.array(pose.position, dtype=float))
            quats.append(np.array(pose.rotation, dtype=float))
        time.sleep(interval_s)
    if not positions:
        raise RuntimeError("没有读到 tracker 位姿，检查 Pika 串口和定位")
    mean_pos = np.mean(positions, axis=0)
    mean_q = _avg_quaternion(quats)
    if mean_q.shape != (4,):
        raise RuntimeError(
            f"SDK 返回的旋转不是四元数 (长度 {mean_q.shape[0]}: {mean_q.tolist()})，"
            "请把这一行贴给 Codex"
        )
    return mean_pos, mean_q


def _parent_robot_target(
    start_pos, start_q, end_pos, end_q, scale_xyz
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate PikaTeleop.get_action (parent): absolute robot target."""
    t2r = Transformations.xyzrpy_to_rotation_matrix(
        0.0,
        0.0,
        0.0,
        *[math.radians(v) for v in _TRACKER_TO_ROBOT_EEF_DEG],
    )
    begin = (
        Transformations.xyzq_to_rotation_matrix(
            start_pos[0] * 1000.0 * scale_xyz,
            start_pos[1] * 1000.0 * scale_xyz,
            start_pos[2] * 1000.0 * scale_xyz,
            start_q,
        )
        @ t2r
    )
    end = (
        Transformations.xyzq_to_rotation_matrix(
            end_pos[0] * 1000.0 * scale_xyz,
            end_pos[1] * 1000.0 * scale_xyz,
            end_pos[2] * 1000.0 * scale_xyz,
            end_q,
        )
        @ t2r
    )
    delta = np.linalg.inv(begin) @ end
    robot_base = Transformations.xyzrpy_to_rotation_matrix(*_ROBOT_BASE_POSE)
    target = robot_base @ delta
    pose = Transformations.rotation_matrix_to_xyzrxryrz(target)
    return np.asarray(pose[:3], dtype=float), np.asarray(pose[3:6], dtype=float)


def _piper_plugin_command(
    parent_xyz, parent_rot_vec, scale_xyz, rotation_scale
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate the piper plugin: translation matrix + rotation remap."""
    origin_xyz = np.asarray(_ROBOT_BASE_POSE[:3], dtype=float)
    corrected_xyz = origin_xyz + _PIKA_TO_PIPER_TRANSLATION @ (
        parent_xyz - origin_xyz
    )
    origin_rot = Transformations.rxryrz_to_rotation_matrix(*_ROBOT_BASE_POSE[3:6])
    target_rot = Transformations.rxryrz_to_rotation_matrix(*parent_rot_vec)
    relative = origin_rot.T @ target_rot
    rel_vec = np.asarray(Transformations.rotation_matrix_to_rxryrz(relative), dtype=float)
    mapped = _PIPER_TOOL_AXIS_CORRECTION @ (_PIKA_TO_PIPER_ROTATION @ rel_vec)
    mapped = mapped * rotation_scale
    return corrected_xyz - origin_xyz, mapped


def _kabsch_matrix(src_dirs, dst_dirs) -> np.ndarray:
    """Best orthonormal rotation mapping src -> dst (Kabsch)."""
    H = np.zeros((3, 3), dtype=float)
    for src, dst in zip(src_dirs, dst_dirs):
        H += np.outer(dst, src)
    U, _, Vt = np.linalg.svd(H)
    M = U @ Vt
    if np.linalg.det(M) < 0:
        Vt[-1] *= -1
        M = U @ Vt
    return M


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyUSB50")
    parser.add_argument(
        "--tracker",
        default=None,
        help="explicit tracker id (e.g. T20) to skip the SDK device-list "
        "auto-detect, which can race and miss trackers",
    )
    parser.add_argument("--scale", type=float, default=0.8, help="scale_xyz (mm command)")
    parser.add_argument("--rotation-scale", type=float, default=0.75, dest="rotation_scale")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--interval", type=float, default=0.02)
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()

    pika = PikaDevice(
        1, pika_sense_port=args.port, pika_tracker_device=args.tracker
    )
    sense = pika.pika_sense
    dev = pika.pika_tracker_device
    print(f"Tracker device: {dev}")
    print(f"scale_xyz={args.scale}, rotation_scale={args.rotation_scale}")
    print("每个动作先回到中立位姿按 Enter 采样起点，再动作并保持按 Enter 采样终点。")
    print()

    gestures = [
        ("向前推约 10cm", "希望 Piper 向前 (+X)"),
        ("向右推约 10cm", "希望 Piper 向右 (-Y)"),
        ("向上抬约 10cm", "希望 Piper 向上 (+Z)"),
        ("只低头约 30 度", "希望 Piper 低头"),
        ("只向右扭腕约 30 度", "希望 Piper 向右扭腕"),
        ("绕手柄轴滚转约 30 度", "希望 Piper 滚转"),
    ]

    results = []
    for idx, (gesture, expected) in enumerate(gestures, start=1):
        input(f"{idx}. 回到中立位姿，按 Enter 采样起点...")
        start_pos, start_q = _sample(
            sense, dev, args.samples, args.interval, args.timeout
        )
        input(f"    然后 {gesture}，保持住，按 Enter 采样终点...")
        end_pos, end_q = _sample(
            sense, dev, args.samples, args.interval, args.timeout
        )

        raw_delta = end_pos - start_pos
        raw_dir = _unit(raw_delta)
        raw_len_m = float(np.linalg.norm(raw_delta))
        start_rot = Transformations.quaternion_to_rotation_matrix(start_q)
        end_rot = Transformations.quaternion_to_rotation_matrix(end_q)
        print(f"\n===== 动作 {idx}: {gesture} =====")
        print(f"预期: {expected}")
        print(f"原始 tracker 位移 (m): {_fmt_vec(raw_delta)}")
        print(f"原始位移幅度: {raw_len_m * 1000.0:.0f} mm")
        if raw_dir is not None:
            print(f"原始位移方向 (单位向量): {_fmt_vec(raw_dir)}")
        if idx <= 3 and raw_len_m < 0.05:
            print("警告: 位移太小 (<5cm)，该动作请推远一点重测")

        parent_xyz, parent_rot = _parent_robot_target(
            start_pos, start_q, end_pos, end_q, args.scale
        )
        cmd_xyz, cmd_rot = _piper_plugin_command(
            parent_xyz, parent_rot, args.scale, args.rotation_scale
        )
        xyz_dir = _unit(cmd_xyz)
        print(
            f"父类映射后目标位移 (mm): {_fmt_vec(parent_xyz - np.asarray(_ROBOT_BASE_POSE[:3], dtype=float))}"
        )
        print(f"插件最终命令位移 (mm): {_fmt_vec(cmd_xyz)}")
        if xyz_dir is not None:
            print(f"命令位移方向 (单位向量): {_fmt_vec(xyz_dir)}")

        if idx >= 4:
            raw_rel = np.asarray(
                Transformations.rotation_matrix_to_rxryrz(start_rot.T @ end_rot),
                dtype=float,
            )
            rot_deg = np.degrees(np.linalg.norm(cmd_rot))
            print(
                f"原始相对旋转 (rad): {_fmt_vec(raw_rel)}  "
                f"幅度 {np.degrees(np.linalg.norm(raw_rel)):.1f} deg"
            )
            print(f"插件命令旋转 delta (rad): {_fmt_vec(cmd_rot)}  幅度 {rot_deg:.2f} deg")
            if np.linalg.norm(raw_rel) < 0.15:
                print("警告: 原始旋转幅度太小 (<8.6 deg)，该动作请转到 20~40 度重测")
        print()
        is_translation = idx <= 3
        raw_rel = None
        if idx >= 4:
            raw_rel = np.asarray(
                Transformations.rotation_matrix_to_rxryrz(start_rot.T @ end_rot),
                dtype=float,
            )
        results.append(
            (
                gesture,
                expected,
                raw_delta,
                raw_dir,
                cmd_xyz,
                cmd_rot,
                raw_len_m,
                is_translation,
                raw_rel,
            )
        )

    print("========== 汇总（请完整复制） ==========")
    translation_dirs = []
    translation_dst = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    for (
        gesture,
        expected,
        raw_delta,
        raw_dir,
        cmd_xyz,
        cmd_rot,
        raw_len_m,
        is_translation,
        raw_rel,
    ) in results:
        print(f"\n{gesture} | {expected}")
        if raw_dir is not None:
            print(f"  原始方向: {_fmt_vec(raw_dir)}")
            if is_translation:
                translation_dirs.append(raw_dir)
        if is_translation:
            print(f"  原始位移幅度: {raw_len_m * 1000.0:.0f} mm")
            print(f"  命令平移 (mm): {_fmt_vec(cmd_xyz)}")
        else:
            if raw_rel is not None:
                print(
                    f"  原始旋转 (rad): {_fmt_vec(raw_rel)}  "
                    f"幅度 {np.degrees(np.linalg.norm(raw_rel)):.1f} deg"
                )
            print(f"  命令旋转 (rad): {_fmt_vec(cmd_rot)}")

    if len(translation_dirs) == 3:
        m_correct = _kabsch_matrix(translation_dirs, translation_dst)
        print("\n========== 修正平移映射矩阵（复制给 Codex） ==========")
        print("M_correct (raw tracker -> Piper base):")
        print("[")
        for row in m_correct:
            print("  [" + ", ".join(f"{v:+.9f}" for v in row) + "],")
        print("]")
        for name, src in zip(["向前", "向右", "向上"], translation_dirs):
            out = m_correct @ src
            print(f"  校验 {name}: M@raw = {_fmt_vec(out)}")


if __name__ == "__main__":
    main()
