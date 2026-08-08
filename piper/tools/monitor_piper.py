#!/usr/bin/env python3
"""Real-time Piper joint + end-pose monitor (read-only, no motion).

Usage:
    python piper/tools/monitor_piper.py --port can0
"""

import argparse
import time

from piper_sdk import C_PiperInterface_V2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="can0")
    parser.add_argument("--interval", type=float, default=0.2)
    args = parser.parse_args()

    p = C_PiperInterface_V2(args.port, False)
    p.ConnectPort(piper_init=False)
    time.sleep(1.0)
    print("只读监视器，不会控制机械臂。")
    print("推荐安全位姿：X=150~250, |Y|<100, Z=250~350, |J5|=20~35")
    print("按 Ctrl+C 结束。")
    try:
        while True:
            e = p.GetArmEndPoseMsgs().end_pose
            j = p.GetArmJointMsgs().joint_state
            print(
                f"X={e.X_axis / 1000:7.1f} Y={e.Y_axis / 1000:7.1f} "
                f"Z={e.Z_axis / 1000:7.1f} mm  "
                f"J1={j.joint_1 / 1000:6.1f} J2={j.joint_2 / 1000:6.1f} "
                f"J3={j.joint_3 / 1000:6.1f} J4={j.joint_4 / 1000:6.1f} "
                f"J5={j.joint_5 / 1000:6.1f} J6={j.joint_6 / 1000:6.1f}",
                flush=True,
            )
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("监视结束")
    finally:
        p.DisconnectPort()


if __name__ == "__main__":
    main()
