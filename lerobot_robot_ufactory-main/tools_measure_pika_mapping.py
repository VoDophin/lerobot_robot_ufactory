import math
import time
import numpy as np

from lerobot_robot_ufactory.devices.pika import PikaDevice


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-6:
        raise RuntimeError("移动距离太小，无法计算方向")
    return v / n


def rpy_from_matrix(R):
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.asin(-R[2, 0])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def read_pos(sense, dev, samples=30):
    pts = []
    for _ in range(samples):
        pose = sense.get_pose(dev)
        if pose:
            pts.append(np.array(pose.position, dtype=float) * 1000.0)
        time.sleep(0.02)
    if not pts:
        raise RuntimeError("没有读到 Tracker pose")
    return np.mean(pts, axis=0)


port = input("Pika port, e.g. /dev/pika_sense or /dev/ttyUSB0: ").strip()
pika = PikaDevice(1, pika_sense_port=port)
sense = pika.pika_sense
dev = pika.pika_tracker_device

print("Tracker device:", dev)
print()
print("接下来请保持 Pika 姿态尽量不变，只做平移。")
print("每一步移动 5~10cm，方向按你希望 Piper 运动的方向来。")
print()

input("1. 把 Pika 放在中立位置，按 Enter 采样...")
p0 = read_pos(sense, dev)
print("p0 =", p0)

input("2. 从中立位置移动到你希望 Piper +X 的方向，按 Enter 采样...")
px = read_pos(sense, dev)
vx = normalize(px - p0)
print("+X vector =", vx)

input("3. 回到中立位置，再移动到你希望 Piper +Y 的方向，按 Enter 采样...")
py = read_pos(sense, dev)
vy = normalize(py - p0)
print("+Y vector =", vy)

input("4. 回到中立位置，再移动到你希望 Piper +Z 的方向，按 Enter 采样...")
pz = read_pos(sense, dev)
vz = normalize(pz - p0)
print("+Z vector raw =", vz)

# 正交化，避免手动移动不完全垂直
vx = normalize(vx)
vy = normalize(vy - np.dot(vy, vx) * vx)
vz = normalize(np.cross(vx, vy))

R = np.column_stack([vx, vy, vz])

print()
print("3x3 rotation matrix:")
print(R)

roll, pitch, yaw = rpy_from_matrix(R)
print()
print("tracker_to_robot_eef candidate:")
print(f"[0, 0, 0, {roll:.3f}, {pitch:.3f}, {yaw:.3f}]")

print()
print("如果测试方向整体反了，可以再试逆矩阵版本:")
R_inv = R.T
roll2, pitch2, yaw2 = rpy_from_matrix(R_inv)
print(f"[0, 0, 0, {roll2:.3f}, {pitch2:.3f}, {yaw2:.3f}]")
