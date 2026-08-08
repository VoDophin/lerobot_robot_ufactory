import argparse
from pathlib import Path
from typing import Any

import yaml


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return "REPLACE_" in value
    if isinstance(value, dict):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_placeholder(item) for item in value)
    return False


def check(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    errors: list[str] = []
    robot = data.get("robot", {})
    teleop = data.get("teleop", {})
    dataset = data.get("dataset", {})

    if not robot.get("type") or not teleop.get("type"):
        errors.append("robot.type and teleop.type are required")

    robots = robot.get("robots", {})
    teleops = teleop.get("teleops", {})
    if robots or teleops:
        # Dual-arm structure: robot.robots / teleop.teleops with left/right.
        if set(robots) != {"left", "right"}:
            errors.append("robot.robots must contain exactly left and right")
        if set(teleops) != {"left", "right"}:
            errors.append("teleop.teleops must contain exactly left and right")
        if set(robots) != set(teleops):
            errors.append("robot and teleop side names must match")
        port_configs = [*robots.values(), *teleops.values()]
    else:
        # Single-arm structure: top-level robot/teleop.
        if not robot.get("port"):
            errors.append("robot.port is required")
        if not teleop.get("port"):
            errors.append("teleop.port is required")
        port_configs = [robot, teleop]

    if not dataset.get("repo_id") or not dataset.get("single_task"):
        errors.append("dataset.repo_id and dataset.single_task are required")
    if _contains_placeholder(data):
        errors.append("replace every REPLACE_* placeholder before hardware use")

    ports = [
        config.get("port")
        for config in port_configs
        if isinstance(config, dict) and config.get("port")
    ]
    if len(ports) != len(set(ports)):
        errors.append("CAN/serial port names must not be reused in one configuration")
    if teleop.get("type") == "uf::dual_pika_teleop" and teleops:
        tracker_ids = [
            config.get("tracker_device_id")
            for config in teleops.values()
            if isinstance(config, dict)
        ]
        uses_piper_pika = any(
            config.get("type") == "uf::piper_pika_teleop"
            for config in teleops.values()
            if isinstance(config, dict)
        )
        if uses_piper_pika:
            if len(tracker_ids) != 2 or not all(tracker_ids):
                errors.append("both Piper Pikas must define tracker_device_id")
            elif len(set(tracker_ids)) != 2:
                errors.append(
                    "left and right Pika must use distinct tracker_device_id values"
                )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Static check for single/dual Piper YAML files")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    failed = False
    for path in args.paths:
        errors = check(path)
        if errors:
            failed = True
            print(f"[FAIL] {path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"[OK]   {path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
