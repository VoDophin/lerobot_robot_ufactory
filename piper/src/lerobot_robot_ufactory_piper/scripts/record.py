"""Piper record entry: adds safe dataset-root handling before the parent
recorder runs. A crashed run can leave an empty dataset directory behind;
LeRobot then refuses to create a new dataset there (FileExistsError). We move
such leftover directories aside automatically. Directories that actually
contain episode data (parquet/video) are left untouched.
"""

import os
import sys
import time
from pathlib import Path


def _config_path_from_argv() -> str | None:
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg.startswith("--config_path="):
            return arg.split("=", 1)[1]
        if arg == "--config_path" and i + 1 < len(args):
            return args[i + 1]
    return None


def _ensure_fresh_dataset_root(config_path: str) -> None:
    try:
        import yaml
    except Exception:
        return
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except Exception:
        return
    root = (raw or {}).get("dataset", {}).get("root")
    if not root:
        return
    path = Path(root).expanduser()
    if not path.exists() or not path.is_dir():
        return
    has_data = any(path.rglob("*.parquet")) or any(path.rglob("*.mp4"))
    if has_data:
        print(
            f"[piper-record] 数据集目录 {path} 已存在且有数据，未改动；"
            "如需续采请使用 -r，或手动改名后重跑。"
        )
        return
    backup = path.parent / f"{path.name}_backup_{time.strftime('%Y%m%d_%H%M%S')}"
    os.rename(str(path), str(backup))
    print(
        f"[piper-record] 数据集目录 {path} 为空/残留，已备份到 {backup}，"
        "本次从全新目录开始。"
    )


def main() -> None:
    # Importing our package (done before this module by Python) registers all
    # added config types. Reuse the parent project's recorder unchanged.
    config_path = _config_path_from_argv()
    if config_path:
        _ensure_fresh_dataset_root(config_path)
    from lerobot_robot_ufactory.scripts.uf_lerobot_record import main as parent_main

    parent_main()


if __name__ == "__main__":
    main()
