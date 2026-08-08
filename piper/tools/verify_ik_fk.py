#!/usr/bin/env python3
"""Pre-flight check / calibration for the joint-stream motion layer (V15).

Mode 1 (default): single-pose FK comparison.

    python piper/tools/verify_ik_fk.py --port can0

Mode 2 (calibration, the important one):

    python piper/tools/verify_ik_fk.py --port can0 --calibrate

The SDK end-pose frame differs from the raw URDF frame by a fixed transform:

    T_sdk = X_base @ chain(q) @ X_ee

Calibration samples 6+ distinct poses (move the arm in drag/teach mode, keep
the orientation varied), then solves X_base and X_ee with a hand-eye method
(Tsai-Lenz for rotation, least squares for translation). Write the printed
``base_xyz_mm`` / ``base_rpy_deg`` / ``ee_xyz_mm`` / ``ee_rpy_deg`` into the
V15 profile. The tool is read-only: no torque enable, no commanded motion.
"""

import argparse
import math
import time

import numpy as np

from lerobot_robot_ufactory_piper.motors import PiperMotorsBus
from lerobot_robot_ufactory_piper.motors.tables import CALIBRATION, MOTORS
from lerobot_robot_ufactory_piper.piper_kinematics import (
    PiperKinematics,
    _rpy_to_matrix,
    xyzrpy_to_matrix,
)


def rotation_error_deg(rpy_a_deg, rpy_b_deg) -> float:
    ra = _rpy_to_matrix(*(math.radians(v) for v in rpy_a_deg))
    rb = _rpy_to_matrix(*(math.radians(v) for v in rpy_b_deg))
    d = ra.T @ rb
    cos_theta = min(1.0, max(-1.0, (np.trace(d) - 1.0) / 2.0))
    return math.degrees(math.acos(cos_theta))


def _quat_from_rot(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _rot_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _q_left(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [w, -x, -y, -z],
            [x, w, -z, y],
            [y, z, w, -x],
            [z, -y, x, w],
        ]
    )


def _q_right(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [w, -x, -y, -z],
            [x, w, z, -y],
            [y, -z, w, x],
            [z, y, -x, w],
        ]
    )


def _average_quats(quats: list[np.ndarray]) -> np.ndarray:
    base = quats[0] / np.linalg.norm(quats[0])
    aligned = []
    for q in quats:
        q = q / np.linalg.norm(q)
        if np.dot(base, q) < 0:
            q = -q
        aligned.append(q)
    avg = np.mean(aligned, axis=0)
    return avg / np.linalg.norm(avg)


def _solve_hand_eye(
    U: list[np.ndarray], S: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Solve S = X @ U @ Y for X (base) and Y (ee)."""
    n = len(U)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            A = S[i][:3, :3] @ S[j][:3, :3].T
            B = U[i][:3, :3] @ U[j][:3, :3].T
            rows.append(_q_left(_quat_from_rot(A)) - _q_right(_quat_from_rot(B)))
    M = np.vstack(rows)
    _, _, Vt = np.linalg.svd(M)
    qx = Vt[-1]
    Rx = _rot_from_quat(qx)

    ry_quats = []
    for i in range(n):
        ry_quats.append(
            _quat_from_rot(U[i][:3, :3].T @ Rx.T @ S[i][:3, :3])
        )
    Ry = _rot_from_quat(_average_quats(ry_quats))

    A = []
    b = []
    for i in range(n):
        A.append(np.hstack([np.eye(3), Rx @ U[i][:3, :3]]))
        b.append(S[i][:3, 3] - Rx @ U[i][:3, 3])
    sol = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)[0]
    tx, ty = sol[:3], sol[3:6]

    X = np.eye(4)
    X[:3, :3] = Rx
    X[:3, 3] = tx
    Y = np.eye(4)
    Y[:3, :3] = Ry
    Y[:3, 3] = ty
    return X, Y


def _read_sample(bus) -> tuple[np.ndarray, np.ndarray]:
    time.sleep(1.0)
    x, y, z, roll, pitch, yaw = bus.get_end_pose()
    S = xyzrpy_to_matrix(
        np.array([x, y, z]) / 1000.0,
        np.radians(np.array([roll, pitch, yaw])),
    )
    js = bus.piper.GetArmJointMsgs().joint_state
    q_rad = np.radians(
        np.array(
            [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6],
            dtype=float,
        )
        / 1000.0
    )
    return q_rad, S


def calibrate(bus, n_poses: int) -> None:
    print()
    print("========== 手眼标定 ==========")
    print("接下来把机械臂切换到拖动/示教模式，依次摆出 N 个明显不同的姿态。")
    print("每次：摆好姿态 -> 保持住 -> 按 Enter 采样。姿态要变化大（俯仰/偏航/滚转都换）。")
    print()
    urdf_kin = PiperKinematics()
    U_list, S_list = [], []
    for i in range(1, n_poses + 1):
        input(f"第 {i}/{n_poses} 个姿态，摆好并保持，按 Enter 采样...")
        q_rad, S = _read_sample(bus)
        U = urdf_kin.chain_matrix(q_rad)
        U_list.append(U)
        S_list.append(S)
        print(f"  关节(rad): {np.round(q_rad, 4).tolist()}")
        print(f"  SDK位姿(mm/deg): {np.round(np.array([*S[:3,3]*1000, *np.degrees(_rpy(S))]), 2).tolist()}")

    X, Y = _solve_hand_eye(U_list, S_list)
    print()
    print("========== 标定结果 ==========")
    print("base_xyz_mm:", np.round(X[:3, 3] * 1000.0, 3).tolist())
    print("base_rpy_deg:", np.round(np.degrees(_rpy(X)), 4).tolist())
    print("ee_xyz_mm:", np.round(Y[:3, 3] * 1000.0, 3).tolist())
    print("ee_rpy_deg:", np.round(np.degrees(_rpy(Y)), 4).tolist())
    print()
    print("========== 校验（用标定帧重算 FK vs SDK） ==========")
    kin = PiperKinematics(
        base_xyzrpy_m=X[:3, 3],
        base_rpy_rad=_rpy(X),
        ee_xyzrpy_m=Y[:3, 3],
        ee_rpy_rad=_rpy(Y),
    )
    worst_pos = 0.0
    worst_rot = 0.0
    for i, (q_rad, S) in enumerate(zip(U_list, S_list), start=1):
        fk = kin.forward(q_rad)
        pos_err = float(np.linalg.norm(fk[:3, 3] - S[:3, 3])) * 1000.0
        rot_err = rotation_error_deg(
            np.degrees(_rpy(fk)), np.degrees(_rpy(S))
        )
        worst_pos = max(worst_pos, pos_err)
        worst_rot = max(worst_rot, rot_err)
        print(f"姿态 {i}: 位置误差 {pos_err:.2f} mm, 姿态误差 {rot_err:.2f} deg")
    print()
    print(f"最大位置误差: {worst_pos:.2f} mm, 最大姿态误差: {worst_rot:.2f} deg")
    if worst_pos < 5.0 and worst_rot < 3.0:
        print("结论: 标定可用，把上面的 base_*/ee_* 写进 V15 配置。")
    else:
        print("警告: 残差偏大，请增加姿态数量/变化幅度后重跑。")


def _rpy(T: np.ndarray):
    R = T[:3, :3]
    return np.array(
        [
            math.atan2(R[2, 1], R[2, 2]),
            math.asin(-R[2, 0]),
            math.atan2(R[1, 0], R[0, 0]),
        ]
    )


def single_pose_check(bus) -> None:
    time.sleep(1.5)
    x, y, z, roll, pitch, yaw = bus.get_end_pose()
    sdk_pose = np.array([x, y, z, roll, pitch, yaw], dtype=float)
    js = bus.piper.GetArmJointMsgs().joint_state
    q_rad = np.radians(
        np.array(
            [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6],
            dtype=float,
        )
        / 1000.0
    )
    print()
    print("当前关节角 (rad):", np.round(q_rad, 5).tolist())
    print("SDK 末端位姿 (mm / deg):", np.round(sdk_pose, 3).tolist())
    if np.allclose(q_rad, 0.0) or np.allclose(sdk_pose[:3], 0.0):
        print("警告: 还没收到有效反馈帧，请确认机械臂上电、CAN 正常。")
        return
    print()
    candidates = [
        ((190.0, 0.0, 0.0), (0.0, -90.0, 0.0), "官方默认 (X=190mm, pitch=-90)"),
        ((0.0, 0.0, 0.0), (0.0, -90.0, 0.0), "仅 pitch=-90"),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "joint6 原帧 (无偏移)"),
        ((190.0, 0.0, 0.0), (0.0, 0.0, 0.0), "仅 X=190mm"),
    ]
    print("========== FK 对比（我们的 FK vs SDK 位姿） ==========")
    for ee_xyz_mm, ee_rpy_deg, label in candidates:
        kin = PiperKinematics(
            ee_xyzrpy_m=tuple(v / 1000.0 for v in ee_xyz_mm),
            ee_rpy_rad=tuple(math.radians(v) for v in ee_rpy_deg),
        )
        fk = np.asarray(kin.forward_xyzrpy(q_rad), dtype=float)
        pos_err = float(np.linalg.norm(fk[:3] - sdk_pose[:3]))
        rot_err = rotation_error_deg(fk[3:6], sdk_pose[3:6])
        print(f"{label}\n  FK: {np.round(fk, 2).tolist()}\n"
              f"  位置误差: {pos_err:.2f} mm   姿态误差: {rot_err:.2f} deg")
    print()
    print("如果所有候选误差都大，请改用 --calibrate 做多姿态手眼标定。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="can0")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--poses", type=int, default=6)
    args = parser.parse_args()

    bus = PiperMotorsBus(
        id="verify",
        port=args.port,
        motors=MOTORS.copy(),
        calibration=CALIBRATION.copy(),
    )
    print(f"连接 CAN 端口 {args.port}（只读，不使能、不运动）...")
    bus.connect(piper_init=False)
    try:
        if args.calibrate:
            calibrate(bus, args.poses)
        else:
            single_pose_check(bus)
    finally:
        bus.disconnect(disable_torque=False, park=False)


if __name__ == "__main__":
    main()
