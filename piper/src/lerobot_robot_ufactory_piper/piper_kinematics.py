"""URDF-based FK/IK for the Piper arm: pure numpy, zero third-party deps.

The official PikaAnyArm stack uses pinocchio; here the same URDF
(piper_description.urdf) is parsed with the stdlib XML parser and the
kinematic chain is evaluated with numpy. The IK is a Levenberg-Marquardt
damped least-squares solver with a numerical Jacobian, seeded with the
current joint values so successive solutions stay continuous.

The SDK end-pose frame generally differs from the raw URDF frame by a fixed
base transform and a fixed end-effector transform:

    T_sdk = X_base @ chain(q) @ X_ee

Both transforms are configurable (``base_xyzrpy_m``/``base_rpy_rad`` and
``ee_xyzrpy_m``/``ee_rpy_rad``) and calibrated by
piper/tools/verify_ik_fk.py --calibrate.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np

_URDF = Path(__file__).resolve().parents[2] / "urdf" / "piper_description.urdf"
_ACTUATED = {"revolute", "continuous"}


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cp * cy, -cr * sy + sr * sp * cy, sr * sy + cr * sp * cy],
            [cp * sy, cr * cy + sr * sp * sy, -sr * cy + cr * sp * sy],
            [-sp, sr * cp, cr * cp],
        ]
    )


def xyzrpy_to_matrix(
    xyz_m: Iterable[float], rpy_rad: Iterable[float]
) -> np.ndarray:
    xyz = np.asarray(list(xyz_m), dtype=float)
    rpy = np.asarray(list(rpy_rad), dtype=float)
    T = np.eye(4)
    T[:3, :3] = _rpy_to_matrix(*rpy)
    T[:3, 3] = xyz
    return T


def matrix_to_xyzrpy(T: np.ndarray) -> tuple[float, float, float, float, float, float]:
    R = T[:3, :3]
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.asin(-R[2, 0])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return (T[0, 3], T[1, 3], T[2, 3], roll, pitch, yaw)


def _axis_angle_vector(R: np.ndarray) -> np.ndarray:
    cos_theta = min(1.0, max(-1.0, (np.trace(R) - 1.0) / 2.0))
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-9:
        diag = np.sqrt(np.maximum(np.diag(R) + 1.0, 0.0) / 2.0)
        axis = diag
        if R[0, 1] < 0:
            axis[1] *= -1
        if R[0, 2] < 0:
            axis[2] *= -1
        axis /= np.linalg.norm(axis)
    else:
        axis = axis / n
    return axis * theta


def _rotation_about(axis: np.ndarray, theta: float) -> np.ndarray:
    u = np.asarray(axis, dtype=float)
    n = np.linalg.norm(u)
    if n < 1e-12:
        return np.eye(3)
    u = u / n
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def _log_se3(T: np.ndarray) -> np.ndarray:
    return np.concatenate([T[:3, 3], _axis_angle_vector(T[:3, :3])])


def _parse_urdf(path: Path) -> list[dict]:
    root = ET.parse(str(path)).getroot()
    joints = []
    for node in root.iter("joint"):
        origin = node.find("origin")
        xyz = (0.0, 0.0, 0.0)
        rpy = (0.0, 0.0, 0.0)
        if origin is not None:
            if origin.get("xyz"):
                xyz = tuple(float(v) for v in origin.get("xyz").split())
            if origin.get("rpy"):
                rpy = tuple(float(v) for v in origin.get("rpy").split())
        axis = (0.0, 0.0, 1.0)
        axis_el = node.find("axis")
        if axis_el is not None and axis_el.get("xyz"):
            axis = tuple(float(v) for v in axis_el.get("xyz").split())
        lower, upper = -math.pi, math.pi
        limit = node.find("limit")
        if limit is not None:
            if limit.get("lower") is not None:
                lower = float(limit.get("lower"))
            if limit.get("upper") is not None:
                upper = float(limit.get("upper"))
        joints.append(
            {
                "name": node.get("name"),
                "type": node.get("type", "fixed"),
                "parent": node.find("parent").get("link"),
                "child": node.find("child").get("link"),
                "xyz": np.asarray(xyz, dtype=float),
                "rpy": np.asarray(rpy, dtype=float),
                "axis": np.asarray(axis, dtype=float),
                "lower": lower,
                "upper": upper,
            }
        )
    return joints


def _build_chain(joints: list[dict]) -> list[dict]:
    by_parent: dict[str, list[dict]] = {}
    for j in joints:
        by_parent.setdefault(j["parent"], []).append(j)
    frontier = [j for j in joints if j["parent"] == "base_link"]
    seen: set[str] = set()
    chain: list[dict] = []
    while frontier:
        j = frontier.pop(0)
        if j["name"] in seen:
            continue
        seen.add(j["name"])
        chain.append(j)
        if j["name"] == "joint6":
            return chain
        frontier.extend(by_parent.get(j["child"], []))
    raise RuntimeError("URDF 链条未找到 joint6")


class PiperKinematics:
    """FK/IK with configurable base and end-effector frame transforms.

    ``forward(q)`` reproduces the SDK frame when the transforms are
    calibrated (``T_sdk = X_base @ chain(q) @ X_ee``).
    """

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        base_xyzrpy_m: Iterable[float] = (0.0, 0.0, 0.0),
        base_rpy_rad: Iterable[float] = (0.0, 0.0, 0.0),
        ee_xyzrpy_m: Iterable[float] = (0.0, 0.0, 0.0),
        ee_rpy_rad: Iterable[float] = (0.0, 0.0, 0.0),
    ) -> None:
        path = Path(urdf_path) if urdf_path is not None else _URDF
        if not path.exists():
            raise FileNotFoundError(f"Piper URDF not found: {path}")
        self.chain = _build_chain(_parse_urdf(path))
        self.joint_indices = [
            i for i, j in enumerate(self.chain) if j["type"] in _ACTUATED
        ]
        if len(self.joint_indices) != 6:
            raise RuntimeError(
                f"URDF 链条应有 6 个驱动关节，实际 {len(self.joint_indices)}"
            )
        self.lower = np.array(
            [self.chain[i]["lower"] for i in self.joint_indices], dtype=float
        )
        self.upper = np.array(
            [self.chain[i]["upper"] for i in self.joint_indices], dtype=float
        )
        self.base_tf = xyzrpy_to_matrix(
            np.asarray(list(base_xyzrpy_m), dtype=float),
            np.asarray(list(base_rpy_rad), dtype=float),
        )
        self.ee_tf = xyzrpy_to_matrix(
            np.asarray(list(ee_xyzrpy_m), dtype=float),
            np.asarray(list(ee_rpy_rad), dtype=float),
        )
        self._origins = [
            xyzrpy_to_matrix(j["xyz"], j["rpy"]) for j in self.chain
        ]
        self._actuated_flags = [j["type"] in _ACTUATED for j in self.chain]

    def chain_matrix(self, q_rad: np.ndarray) -> np.ndarray:
        """Raw URDF chain transform (joint1..6), no base/ee."""
        T = np.eye(4)
        qi = 0
        for idx, origin in enumerate(self._origins):
            T = T @ origin
            if self._actuated_flags[idx]:
                R = _rotation_about(
                    self.chain[idx]["axis"], float(np.asarray(q_rad).reshape(-1)[qi])
                )
                T[:3, :3] = T[:3, :3] @ R
                qi += 1
        return T

    def _model_matrix(self, q_rad: np.ndarray) -> np.ndarray:
        """Chain @ ee: the frame the IK iterates on."""
        return self.chain_matrix(q_rad) @ self.ee_tf

    def forward(self, q_rad: np.ndarray) -> np.ndarray:
        """FK in the SDK frame: base @ chain @ ee (meters)."""
        return self.base_tf @ self._model_matrix(q_rad)

    def forward_xyzrpy(
        self, q_rad: np.ndarray
    ) -> tuple[float, float, float, float, float, float]:
        x, y, z, roll, pitch, yaw = matrix_to_xyzrpy(self.forward(q_rad))
        return (
            x * 1000.0,
            y * 1000.0,
            z * 1000.0,
            math.degrees(roll),
            math.degrees(pitch),
            math.degrees(yaw),
        )

    def _jacobian(self, q_rad: np.ndarray) -> np.ndarray:
        eps = 1e-6
        T_cur = self._model_matrix(q_rad)
        T_inv = np.linalg.inv(T_cur)
        J = np.zeros((6, 6))
        for i in range(6):
            qp = q_rad.copy()
            qp[i] += eps
            J[:, i] = _log_se3(T_inv @ self._model_matrix(qp)) / eps
        return J

    def ik(
        self,
        target_xyz_mm: Iterable[float],
        target_rpy_deg: Iterable[float],
        q_seed_rad: Iterable[float],
        max_iter: int = 10,
        tol: float = 1e-4,
        damping: float = 1e-3,
        weight_ori: float = 1.0,
        jac_reuse: int = 2,
    ) -> tuple[np.ndarray, float]:
        """LM IK. Target is in the SDK frame; base/ee handled internally."""
        target_sdk = xyzrpy_to_matrix(
            np.asarray(list(target_xyz_mm), dtype=float) / 1000.0,
            np.radians(np.asarray(list(target_rpy_deg), dtype=float)),
        )
        target_model = np.linalg.inv(self.base_tf) @ target_sdk
        q = np.clip(
            np.asarray(q_seed_rad, dtype=float).reshape(-1)[:6],
            self.lower,
            self.upper,
        )
        W2 = np.diag([1.0, 1.0, 1.0, weight_ori, weight_ori, weight_ori]) ** 2
        lam = float(damping)
        max_step = math.radians(0.5)
        residual = float("inf")
        J = None
        for it in range(max_iter):
            T_cur = self._model_matrix(q)
            err = _log_se3(np.linalg.inv(T_cur) @ target_model)
            residual = float(np.linalg.norm(err))
            if residual < tol:
                break
            if J is None or it % jac_reuse == 0:
                J = self._jacobian(q)
            A = J.T @ W2 @ J + lam * np.eye(6)
            dq = np.linalg.solve(A, J.T @ W2 @ err)
            step = float(np.max(np.abs(dq)))
            if step > max_step:
                dq = dq * (max_step / step)
            q_new = np.clip(q + dq, self.lower, self.upper)
            err_new = _log_se3(
                np.linalg.inv(self._model_matrix(q_new)) @ target_model
            )
            if np.linalg.norm(err_new) < residual:
                q = q_new
                lam = max(float(damping), lam * 0.7)
            else:
                lam = min(1.0, lam * 3.0)
        return q, residual
