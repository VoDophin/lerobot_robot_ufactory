#!/usr/bin/env python3
"""Calibrate the Pika tracker -> gripper-center offset (tracker_to_robot_eef).

Principle: hold the Pika gripper center against a fixed point and rotate the
Pika in at least two directions without moving that point. The tracker moves
on a sphere centered at the fixed point:

    p_i = c + R_i * r

where p_i is the tracker position, R_i its rotation matrix, c the fixed
point (world frame) and r the tracker offset (tracker frame). The 6 unknown
values (c, r) are solved by least squares; the residual tells you how well
the fixed point stayed still.

Run on the robot Linux PC (Pika only, no Piper):

    conda activate uf_lerobot
    cd ~/lerobot_robot_ufactory
    python piper/tools/calibrate_tracker_offset.py --port /dev/ttyUSB50

Repeat 2-3 times. Only adopt a result when the residual RMS is below ~5 mm
and the attitude coverage is above ~30 degrees, and when runs agree within a
few mm. Write the candidate into a NEW test profile (the rotation part
180/-90/0 stays as it is).
"""

import argparse
import math
import time

import numpy as np

from lerobot_robot_ufactory.devices.pika import PikaDevice
from lerobot_robot_ufactory.devices.umi.vive_tracker.transformations import (
    Transformations,
)


def _rotation_degrees(R: np.ndarray) -> float:
    """Angle of the rotation matrix R, in degrees."""
    cos_theta = min(1.0, max(-1.0, (np.trace(R) - 1.0) / 2.0))
    return math.degrees(math.acos(cos_theta))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyUSB50")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="scale_xyz of the profile that will use this offset (V13/V14 = 1.0)",
    )
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--interval", type=float, default=0.02)
    args = parser.parse_args()

    pika = PikaDevice(1, pika_sense_port=args.port)
    sense = pika.pika_sense
    dev = pika.pika_tracker_device
    print(f"Tracker device: {dev}")
    print("把 Pika 夹持中心轻轻抵住固定点。")
    print("倒计时结束后，保持中心不动，绕至少两个方向缓慢转动 Pika。")
    print()
    for left in (5, 4, 3, 2, 1):
        print(f"{left}秒后开始采样……")
        time.sleep(1.0)

    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    start = time.monotonic()
    print(f"采样 {args.duration:.0f} 秒……")
    while time.monotonic() - start < args.duration:
        pose = sense.get_pose(dev)
        if pose is not None:
            positions.append(np.array(pose.position, dtype=float))
            rotations.append(
                Transformations.quaternion_to_rotation_matrix(
                    np.asarray(pose.rotation, dtype=float)
                )
            )
        time.sleep(args.interval)

    if len(positions) < 30:
        raise RuntimeError(f"有效帧太少 ({len(positions)})，检查 tracker 是否可见")
    print(f"样本数： {len(positions)}")

    P = np.stack(positions)
    R = np.stack(rotations)

    # Least squares for [c (3), r (3)]:  p_i = c + R_i * r
    M = np.zeros((3 * len(P), 6), dtype=float)
    b = np.zeros((3 * len(P),), dtype=float)
    for i in range(len(P)):
        M[3 * i : 3 * i + 3, 0:3] = np.eye(3)
        M[3 * i : 3 * i + 3, 3:6] = R[i]
        b[3 * i : 3 * i + 3] = P[i]
    normal = M.T @ M
    rank = int(np.linalg.matrix_rank(normal, tol=1e-9))
    x = np.linalg.lstsq(M, b, rcond=None)[0]
    c = x[0:3]
    r = x[3:6]

    residuals = np.linalg.norm(P - c - R @ r, axis=1)
    rms_mm = float(np.sqrt(np.mean(residuals**2))) * 1000.0
    max_mm = float(np.max(residuals)) * 1000.0

    # Attitude coverage: max rotation angle between the mean and any sample.
    r_mean = np.mean(R, axis=0)
    u, _, vt = np.linalg.svd(r_mean)
    r_mean_orth = u @ vt
    coverage = max(_rotation_degrees(r_mean_orth.T @ Ri) for Ri in R)

    offset_mm = r * 1000.0
    scaled_mm = offset_mm * args.scale

    print()
    print("========== 标定结果 ==========")
    print(f"样本数： {len(positions)}")
    print(f"矩阵秩： {rank} （应为6）")
    print(f"最大姿态覆盖： {coverage:.2f}°")
    print(f"固定点残差RMS： {rms_mm:.3f} mm")
    print(f"固定点最大残差： {max_mm:.3f} mm")
    print()
    print(f"Pika定位器到夹持中心的物理偏移(mm)： {np.round(offset_mm, 3).tolist()}")
    print(
        f"考虑scale_xyz={args.scale}后的配置偏移(mm)： "
        f"{np.round(scaled_mm, 3).tolist()}"
    )
    print()
    print("候选YAML：")
    x, y, z = (round(v, 3) for v in scaled_mm)
    print(f"tracker_to_robot_eef: [{x}, {y}, {z}, 180, -90, 0]")
    print()
    if rms_mm > 5.0:
        print("警告：残差RMS > 5mm，本次标定质量差，请重测（中心要真的不动）")
    if coverage < 30.0:
        print("警告：姿态覆盖 < 30°，旋转不够，请转更大角度重测")
    print("建议：重复运行 2~3 次，只有数值一致（±5mm 内）才写进配置。")


if __name__ == "__main__":
    main()
