"""Identify which Vive tracker belongs to the left and right Pika.

This tool starts only the shared Vive tracking context. It never opens a CAN
interface, enables a motor, or sends a robot command.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from lerobot_robot_ufactory_piper.shared_vive_tracker import SharedViveTracker


def _tracker_names(shared: SharedViveTracker) -> list[str]:
    # T20/T21 are discovery-order aliases and can swap across process starts.
    # Only persistent LHR serials are safe for left/right configuration.
    return shared.tracker_serials()


def _wait_for_trackers(
    shared: SharedViveTracker, timeout_s: float = 20.0
) -> list[str]:
    deadline = time.monotonic() + timeout_s
    names: list[str] = []
    while time.monotonic() < deadline:
        names = _tracker_names(shared)
        fresh = [name for name in names if shared.get_pose(name) is not None]
        if len(fresh) == 2:
            return fresh
        time.sleep(0.1)
    raise RuntimeError(
        f"Expected two fresh Pika tracker serials, found {names}. "
        "Check both Pikas and lighthouse visibility."
    )


def _positions(
    shared: SharedViveTracker, names: list[str]
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in names:
        pose = shared.get_pose(name)
        if pose is None:
            raise RuntimeError(f"Tracker {name} lost or stale during identification")
        result[name] = np.asarray(pose.position, dtype=float).copy()
    return result


def _measure_motion(
    shared: SharedViveTracker,
    names: list[str],
    side: str,
    duration_s: float,
) -> dict[str, float]:
    input(
        f"\nKeep the other Pika still. Press Enter, then move only the {side} "
        f"Pika by 10-15 cm for {duration_s:.0f} seconds: "
    )
    baseline = _positions(shared, names)
    maximum_mm = {name: 0.0 for name in names}
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        current = _positions(shared, names)
        for name in names:
            distance_mm = float(
                np.linalg.norm(current[name] - baseline[name]) * 1000.0
            )
            maximum_mm[name] = max(maximum_mm[name], distance_mm)
        time.sleep(0.02)
    print(f"{side.capitalize()}-hand motion:")
    for name in names:
        print(f"  {name}: {maximum_mm[name]:.1f} mm")
    return maximum_mm


def _dominant_tracker(motion: dict[str, float], side: str) -> str:
    ordered = sorted(motion.items(), key=lambda item: item[1], reverse=True)
    winner, winner_mm = ordered[0]
    runner_up_mm = ordered[1][1]
    if winner_mm < 30.0:
        raise RuntimeError(
            f"{side} motion was only {winner_mm:.1f} mm; "
            "repeat with a clear 10-15 cm motion"
        )
    if winner_mm < max(3.0 * runner_up_mm, runner_up_mm + 20.0):
        raise RuntimeError(
            f"{side} result is ambiguous ({winner_mm:.1f} vs "
            f"{runner_up_mm:.1f} mm); keep the other Pika still and retry"
        )
    return winner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only left/right Vive tracker identification for dual Pika"
    )
    parser.add_argument("--duration-s", type=float, default=5.0)
    args = parser.parse_args()
    if args.duration_s < 3.0:
        parser.error("--duration-s must be at least 3 seconds")

    shared = SharedViveTracker.instance()
    if not shared.ensure_started():
        raise RuntimeError("Could not start the shared pysurvive tracker context")

    print("READ ONLY: no CAN interface or Piper robot will be opened.")
    print("Waiting for exactly two fresh persistent Pika tracker serials...")
    try:
        names = _wait_for_trackers(shared)
        print("Detected:", names)
        print("Current transient-name mapping:", shared.tracker_identities())
        left_motion = _measure_motion(shared, names, "left", args.duration_s)
        print("Return the left Pika to a comfortable position and keep it still.")
        time.sleep(2.0)
        right_motion = _measure_motion(shared, names, "right", args.duration_s)
        left = _dominant_tracker(left_motion, "left")
        right = _dominant_tracker(right_motion, "right")
        if left == right:
            raise RuntimeError(
                "Both phases selected the same tracker; retry while moving only one Pika"
            )
        print("\nIdentification passed.")
        print("Suggested YAML:")
        print(f"  left tracker_device_id: {left}")
        print(f"  right tracker_device_id: {right}")
        print("Use these LHR serials, never T20/T21, in dual-arm configs.")
    finally:
        shared.shutdown()


if __name__ == "__main__":
    main()
