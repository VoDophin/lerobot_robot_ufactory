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
    robots = robot.get("robots", {})
    teleops = teleop.get("teleops", {})

    if not robot.get("type") or not teleop.get("type"):
        errors.append("robot.type and teleop.type are required")
    if robot.get("type") == "uf::dual_piper" or teleop.get("type") == "uf::dual_pika_teleop":
        if set(robots) != {"left", "right"}:
            errors.append("robot.robots must contain exactly left and right")
        if set(teleops) != {"left", "right"}:
            errors.append("teleop.teleops must contain exactly left and right")
        if set(robots) != set(teleops):
            errors.append("robot and teleop side names must match")
    elif robot.get("type") != "uf::piper" or teleop.get("type") != "uf::pika_teleop":
        errors.append("single-arm configs must use robot.type=uf::piper and teleop.type=uf::pika_teleop")
    if not dataset.get("repo_id") or not dataset.get("single_task"):
        errors.append("dataset.repo_id and dataset.single_task are required")
    if _contains_placeholder(data):
        errors.append("replace every REPLACE_* placeholder before hardware use")

    configs = [robot, teleop]
    if isinstance(robots, dict):
        configs.extend(robots.values())
    if isinstance(teleops, dict):
        configs.extend(teleops.values())
    ports = [config.get("port") for config in configs if isinstance(config, dict) and config.get("port")]
    if len(ports) != len(set(ports)):
        errors.append("CAN/serial port names must not be reused in one configuration")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Static check for dual-arm Piper YAML files")
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
