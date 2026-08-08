"""Single shared pysurvive context for dual-Pika teleoperation.

Two pysurvive contexts cannot coexist (the second one fails with
LIBUSB_ERROR_BUSY), and every context sees BOTH trackers. This module keeps
ONE context plus one reader thread that caches each device's latest pose by
name and persistent hardware serial, then patches the pika SDK so each Pika
reads its own tracker.  The libsurvive names (for example T20 / T21) are
assigned during discovery and may swap between process starts; dual-Pika
profiles must therefore use the persistent LHR serial as ``tracker_device_id``.

If pysurvive is unavailable or the shared context cannot start, the original
SDK path is kept as a fallback (single-Pika setups keep working unchanged).
"""

from __future__ import annotations

import threading
import time

import numpy as np

try:
    import pysurvive
except Exception:  # pragma: no cover - dependency optional
    pysurvive = None


_MAX_POSE_AGE_S = 0.5


class _Pose:
    __slots__ = ("position", "rotation", "received_at")

    def __init__(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        received_at: float,
    ) -> None:
        self.position = position
        self.rotation = rotation
        self.received_at = received_at


class SharedViveTracker:
    _instance: "SharedViveTracker | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self.ctx = None
        self._lock = threading.Lock()
        self._poses: dict[str, _Pose] = {}
        self._serial_by_name: dict[str, str] = {}
        self._thread: threading.Thread | None = None
        self._stop = False

    @classmethod
    def instance(cls) -> "SharedViveTracker":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def ensure_started(self) -> bool:
        if pysurvive is None:
            return False
        if self.ctx is not None:
            return True
        with self._lock:
            if self.ctx is not None:
                return True
            try:
                self.ctx = pysurvive.SimpleContext(["pysurvive", "--v", "0"])
            except Exception:
                return False
            self._stop = False
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True

    def _run(self) -> None:
        while not self._stop:
            try:
                updated = self.ctx.NextUpdated()
            except Exception:
                time.sleep(0.005)
                continue
            if updated:
                try:
                    name = updated.Name().decode("utf-8")
                    serial = None
                    serial_number = getattr(
                        pysurvive, "simple_serial_number", None
                    )
                    ptr = getattr(updated, "ptr", None)
                    if serial_number is not None and ptr is not None:
                        value = serial_number(ptr)
                        if isinstance(value, bytes):
                            value = value.decode("utf-8", errors="replace")
                        value = str(value).strip()
                        if value:
                            serial = value
                    pose, _ts = updated.Pose()
                    position = np.array(
                        [pose.Pos[0], pose.Pos[1], pose.Pos[2]], dtype=float
                    )
                    rotation = np.array(
                        [
                            pose.Rot[0],
                            pose.Rot[1],
                            pose.Rot[2],
                            pose.Rot[3],
                        ],
                        dtype=float,
                    )
                    latest = _Pose(position, rotation, time.monotonic())
                    with self._lock:
                        self._poses[name] = latest
                        if serial:
                            self._poses[serial] = latest
                            self._serial_by_name[name] = serial
                except Exception:
                    pass

    def get_pose(
        self, device: str, max_age_s: float = _MAX_POSE_AGE_S
    ) -> _Pose | None:
        with self._lock:
            pose = self._poses.get(device)
            if pose is None:
                return None
            if time.monotonic() - pose.received_at > max_age_s:
                return None
            return pose

    def devices(self) -> list[str]:
        with self._lock:
            return list(self._poses.keys())

    def tracker_serials(self) -> list[str]:
        """Return persistent serials for non-lighthouse tracked objects."""
        with self._lock:
            return sorted(
                serial
                for name, serial in self._serial_by_name.items()
                if not name.startswith("LH")
            )

    def tracker_identities(self) -> dict[str, str]:
        """Return the current transient-name to persistent-serial mapping."""
        with self._lock:
            return {
                name: serial
                for name, serial in self._serial_by_name.items()
                if not name.startswith("LH")
            }

    def shutdown(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


def _install_shared_tracker_patch() -> None:
    if pysurvive is None:
        return

    try:
        from pika.tracker import vive_tracker as vt_mod
    except Exception:
        return
    tracker_cls = getattr(vt_mod, "ViveTracker", None)
    if tracker_cls is None or getattr(tracker_cls, "_uf_shared_patched", False):
        return

    orig_connect = tracker_cls.connect
    orig_get_pose = getattr(tracker_cls, "get_pose", None)
    orig_disconnect = getattr(tracker_cls, "disconnect", None)

    def connect(self) -> None:
        shared = SharedViveTracker.instance()
        if shared.ensure_started():
            self._uf_shared = shared
            self.context = shared.ctx
            return
        if orig_connect is not None:
            return orig_connect(self)

    def get_pose(self, device: str) -> _Pose | None:
        shared = getattr(self, "_uf_shared", None)
        if shared is not None:
            return shared.get_pose(device)
        if orig_get_pose is not None:
            return orig_get_pose(self, device)
        return None

    def disconnect(self) -> None:
        # Keep the shared context alive for the other side.
        if not hasattr(self, "_uf_shared") and orig_disconnect is not None:
            return orig_disconnect(self)

    tracker_cls.connect = connect
    tracker_cls.get_pose = get_pose
    tracker_cls.disconnect = disconnect
    tracker_cls._uf_shared_patched = True

    # Route Sense.get_pose by tracker id through the shared cache so each
    # Pika reads only its own tracker.
    try:
        from pika.sense import Sense as SenseCls
    except Exception:
        return
    if getattr(SenseCls, "_uf_shared_patched", False):
        return
    orig_sense_get_pose = getattr(SenseCls, "get_pose", None)

    def sense_get_pose(self, device: str):
        shared = SharedViveTracker.instance()
        # Once the shared dual-tracker context is active, never fall back to
        # the SDK's independent context or to an indefinitely cached pose.
        # Returning None makes the teleoperator hold its last safe target.
        if shared.ctx is not None:
            return shared.get_pose(device)
        if orig_sense_get_pose is not None:
            return orig_sense_get_pose(self, device)
        return None

    SenseCls.get_pose = sense_get_pose
    SenseCls._uf_shared_patched = True


_install_shared_tracker_patch()

__all__: list[str] = ["SharedViveTracker"]
