# piper 项目说明（Pika / Piper 遥操作与数据采集）

本目录是 `lerobot_robot_ufactory` 的 Piper 子项目：复用父项目的 Pika、相机和录制流程，加入 Piper 单臂/双臂 follower、leader、Pika 遥操作与三种单臂执行模式。**不修改父项目文件**。

## 目录结构

```text
piper/
├── config/                # 运行配置（YAML，见下方版本说明）
├── src/lerobot_robot_ufactory_piper/
│   ├── piper_follower.py     # MOVE P 执行层（step / direct / bounded direct）
│   ├── piper_joint_stream.py # 关节空间流式执行层（纯 numpy IK）
│   ├── piper_kinematics.py   # 纯 numpy URDF 运动学（FK/IK）
│   └── pika_teleop.py        # Pika 遥操作（坐标映射、旋转、夹爪）
├── tools/                # 标定 / 测量 / 监控 / 校验工具
├── urdf/                 # Piper 运动学模型（无需 mesh）
└── README.md             # 本说明
```

## 配置版本说明（config/）

单臂遥操作配置按「控制方法 + 版本」命名：

| 配置 | 版本 | 说明 |
|---|---|---|
| `single_pika_piper_movep_step_v14.yaml` | V14 保留 | MOVE P 阶梯限幅，每周期小步，保守稳定，慢速精细任务 |
| `single_pika_piper_joint_stream_v15.yaml` | V15 保留 | 关节空间流式（纯 numpy IK），绕开固件 IK 抖动，进阶（先做手眼标定） |
| `single_pika_piper_setting.yaml` | **单臂终稿**（原 v1_final） | MOVE P 直通 + 单周期限幅（bounded direct，25 mm / 0.35 rad），顺滑跟手，日常采集推荐 |
| `single_pika_piper_setting_record.yaml` | 单臂录制版 | 同上 + Intel D435i RGB 相机（相机路径按机器修改） |
| `single_pika_piper.yaml` | 基础模板 | 占位符模板，替换 CAN/相机/数据集后使用 |

双臂配置（不参与单臂版本编号）：`dual_pika_direct.yaml`（只采集）、`dual_pika_piper.yaml`（Pika 控双臂）、`dual_piper_leader_follower.yaml`（Piper 主从）。

### 与旧版（V16/V17）的对比

| 参数 | V16 时代 | V17 时代（旧服务器效果好） | 当前 v1_final |
|---|---|---|---|
| 指令模式 | direct 满目标直发 | direct + 25 mm / 0.35 rad 限幅 | direct + 25 mm / 0.35 rad 限幅 |
| 防卡顿机制 | 无（快速甩动会卡） | bounded direct（已验证） | 同 V17 |
| 速度 | 30% | 30% | 70% |
| 控制频率 / 旋转滤波 | 50 Hz / 0.4 | 50 Hz / 0.4 | 100 Hz / 0.8（更低延迟） |
| 启动摆位 | 自动移到 startup_tcp_pose | 自动 | 不自动（手动摆好姿态，手腕俯仰余量更大） |
| 旋转映射 | 固定矩阵（旧服务器方向正确） | 同 | 本地坐标系校准矩阵（新服务器方向正确，翻转矩阵在机器上验证） |
| tracker 绑定 | SDK 自动检测（有竞态） | 同 | 显式 `tracker_device_id: T20` |

结论：当前 single_pika_piper_setting 保留了 V17 已验证的防卡顿限幅和跟手直通，同时修正了 tracker 检测竞态与旋转方向，并把启动摆位改为手动（手腕活动范围更好）。

## 安装

```bash
cd /path/to/lerobot_robot_ufactory
conda activate uf_lerobot
pip install -e .
pip install -e ./piper
```

CAN 口按 Piper 官方方式配置（1 Mbps）：

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 txqueuelen 1000
sudo ip link set can0 up
ip -details link show can0        # 确认 state UP、ERROR-ACTIVE
timeout 3 candump can0            # 应能收到 2A1~2A8 反馈帧
```

## 运行

```bash
# 单臂遥操作（推荐）
uf-piper-teleop --config_path=piper/config/single_pika_piper_setting.yaml

# 录制（带 D435i RGB；相机路径需匹配当前机器）
uf-piper-record --config_path=piper/config/single_pika_piper_setting_record.yaml
```

交互（headless / SSH）：回车开始；录制满一个 episode 时长后回车保存并进入下一个；Ctrl+C 退出。

## 常用命令速查

### 双臂（双 Pika 遥操作双 Piper）

```bash
uf-piper-teleop --config_path=piper/config/dual_pika_piper.yaml
```

首次使用前需要：

1. **固定两个 Pika 串口**（两个 CH340 无序列号、by-id 相同，必须按 USB 物理口固定）：
   ```bash
   udevadm info -a -n /dev/ttyUSB0 | grep KERNELS   # 分别对两个口执行，记下物理路径
   sudo tee /etc/udev/rules.d/98-pika-left-right.rules <<'EOF'
   SUBSYSTEM=="tty", ENV{ID_PATH}=="pci-0000:XX:00.0-usb-0:2.4:1.0", MODE:="0666", GROUP="dialout", SYMLINK+="pika_left"
   SUBSYSTEM=="tty", ENV{ID_PATH}=="pci-0000:XX:00.0-usb-0:1.4:1.0", MODE:="0666", GROUP="dialout", SYMLINK+="pika_right"
   EOF
   sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=tty
   ls -l /dev/pika_left /dev/pika_right
   ```
2. **确认两个 CAN 接口**（can0 / can1）对应哪只手臂，开机后 `ip link show | grep can` 确认实际名字，并按上文 CAN 配置拉起两个口。
3. **确认两个 tracker 都开机可见**（T20 / T21），用测量工具确认左右绑定，不对就在 `dual_pika_piper.yaml` 里交换 `tracker_device_id`。
4. 手动把左右臂摆到镜像安全位姿再开始（配置不自动摆位）。

双臂配置已对齐单臂终稿参数（calibrated 旋转 + 翻转矩阵、速度 70、滤波 0.8、25/0.35 限幅）。

### 环境与安装

```bash
conda activate uf_lerobot
cd ~/lerobot_robot_ufactory
python -m pip install -e . --no-deps
python -m pip install -e ./piper --no-deps
```

### 实时监控（只读）

```bash
python piper/tools/monitor_piper.py --port can0
```

安全参考位姿：`X=150~250, |Y|<100, Z=250~350, |J5|=20~35`。

### 标定 / 测量

```bash
# 基站（灯塔）标定：首次 / 移动基站后
uf-vive-calibrate --force-calibrate

# Pika 定位器到夹持中心偏移（手腕有假平移时）
python piper/tools/calibrate_tracker_offset.py --port /dev/ttyUSB0

# Pika 轴向测量（生成平移映射矩阵，改基站布局后跑）
python piper/tools/measure_pika_piper_mapping.py --port /dev/ttyUSB0 --tracker T20

# joint_stream 前的手眼标定（基座 + 末端帧）
python piper/tools/verify_ik_fk.py --port can0 --calibrate
```

### 排查：进程 / 串口 / 相机

```bash
# 确认没有别的控制器在跑（官方 piper_ros 会抢 can0）
ps aux | grep -E "piper_ctrl|roslaunch|roscore" | grep -v grep
pkill -f "uf-piper-teleop"
pkill -f "uf-piper-record"

# Pika 串口占用检查（无输出 = 干净）
fuser -v /dev/ttyUSB0

# 相机节点
ls -l /dev/v4l/by-id/ | grep -E "435i|405"
lerobot-find-cameras opencv
```

### Windows → Linux 同步

Windows（`D:\DLproject\lerobot_robot_ufactory`）：

```powershell
git add -A
git commit -m "说明"
git push origin main
```

Linux 拉取（GitHub 直连不稳时用镜像）：

```bash
git fetch https://gh-proxy.com/https://github.com/Amadeuszero0/lerobot_robot_ufactory.git main
git merge FETCH_HEAD
git log -1 --oneline
```

## 工具（tools/）

| 工具 | 用途 | 什么时候做 |
|---|---|---|
| `uf-vive-calibrate` | 基站位置标定 | 首次 / 移动基站后 |
| `tools/calibrate_tracker_offset.py` | tracker 到夹持中心偏移 | 换安装方式 / 手腕有假平移 |
| `tools/measure_pika_piper_mapping.py` | Pika 轴向测量，生成平移映射矩阵 | 改基站布局后 |
| `tools/monitor_piper.py` | 关节/末端位姿实时监控（只读） | 日常排查 |
| `tools/verify_ik_fk.py` | 手眼标定与 FK/IK 校验 | joint_stream 首次使用前 |

## 常见问题

1. **`No Pika tracker found`**：SDK 设备列表检测有竞态；v1_final 已用 `tracker_device_id: T20` 显式指定跳过，若旧配置遇到就重跑一次或补上该字段。
2. **`Message NOT sent` / `EnableArm send failed`**：CAN 没配好或机械臂没供电/急停；先跑上面的 CAN 配置并 `candump` 确认有帧。
3. **can0 反复 `Network is down`**：USB-CAN 适配器供电/接触问题，换主板直连 USB 口或换线。
4. **旋转方向反**：`rotation_mapping_matrix` 在配置里，某列取反即翻转对应轴方向；整体反可用 `rotation_scale: -1.0`。
5. **手腕俯仰没余量**：v1_final 启动不自动摆位，手动把夹爪摆到舒服姿态再开始。
6. **快速甩动卡顿**：确认 `direct_max_step_mm: 25.0` / `direct_max_step_rad: 0.35` 在；仍卡可降到 15 / 0.20。
7. **录制目录已存在**：录制入口会自动把残留目录改名备份；想续采加 `-r`。
