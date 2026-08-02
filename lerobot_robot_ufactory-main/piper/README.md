# 双臂 Pika / Piper 数据采集集成

这个目录是 `lerobot_robot_ufactory` 的**独立子项目**。它复用父项目的 Pika、相机和录制流程，加入 Piper 单臂/双臂 follower、Piper leader，以及双 Pika 组合器。父项目已有文件不需要修改。

## 支持的三种方案

| 方案 | 配置文件 | 动作空间 | 硬件 |
|---|---|---|---|
| Pika 直接采集 | `config/dual_pika_direct.yaml` | 双侧末端位姿 + 夹爪 | 2×Pika + 相机，不驱动机械臂 |
| Pika 遥操 Piper | `config/dual_pika_piper.yaml` | 双侧末端位姿 + 夹爪 | 2×Pika + 2×Piper follower |
| Piper 遥操 Piper | `config/dual_piper_leader_follower.yaml` | 双侧关节位置 + 夹爪 | 2×Piper leader + 2×Piper follower |

数据键使用 `left.` / `right.` 前缀。例如，Pika 路径包含 `left.pose.x`、`right.pose.rz`、`left.gripper.pos`；主从臂路径包含 `left.joint1.pos` 到 `right.joint6.pos`。

## 安装

建议在已安装 Pika、Piper SDK 的同一个 LeRobot 0.4.3 环境里执行：

```bash
cd /path/to/lerobot_robot_ufactory
pip install -e .
pip install -e ./piper
```

确认版本和入口：

```bash
python -c "import lerobot; print(lerobot.__version__)"
uf-piper-record --help
```

该子项目要求 `lerobot==0.4.3`，与父项目一致。`piper_sdk` 推荐使用 0.3.0 或更新版本。若 `wego_piper` 是本地源码安装，请确保当前环境能够 `import wego_piper`。

## 第一次运行前

1. 按 Piper 官方方式激活 CAN 口，波特率为 1 Mbps。四臂主从方案需要 `can_leader1`、`can_leader2`、`can_follower1`、`can_follower2` 四个不同接口。
2. 复制所需 YAML，替换所有 `REPLACE_*`：Pika 串口、相机 `/dev/v4l/by-id/...`、任务文本等。
3. 根据真实安装位置校准两侧 `tracker_to_robot_eef`。Pika→Piper 还必须确认 `workspace_x/y/z` 是各自 Piper **本体坐标系**下的安全范围。
4. 静态检查配置：

```bash
uf-piper-check-config config/dual_pika_piper.yaml
```

5. 上真机前，先急停可达、净空、低速试运行；第一轮建议保持示例中的 5 mm / 0.05 rad 单周期限幅和 15% 速度。

## 录制命令

在 `piper/` 目录执行：

```bash
# 方案一：双 Pika 直接采
uf-piper-record --config_path=config/dual_pika_direct.yaml

# 方案二：双 Pika -> 双 Piper
uf-piper-record --config_path=config/dual_pika_piper.yaml

# 方案三：双 Piper leader -> 双 Piper follower
uf-piper-record --config_path=config/dual_piper_leader_follower.yaml
```

需要续采或异步保存时，可沿用父项目参数：

```bash
uf-piper-record -r -a --config_path=config/dual_piper_leader_follower.yaml
```

录制界面快捷键沿用父项目：空格开始，左箭头重录，右箭头保存当前 episode，Esc 停止。

## 三种模式的实现说明

### 1. Pika 直接采集

每侧使用父项目的 `uf::mock_robot` 承载相机和 Pika 位姿，不会向机械臂发送命令。双 Pika 子遥操器的 `id` 必须与对应 mock robot 的 `teleop_id` 一致。

### 2. Pika 遥操 Piper

Pika 输出与父项目 XArm 路径相同的 `xyz(mm) + 轴角(rad)`。本插件在发送命令前转换为 Piper SDK `EndPoseCtrl` 所需的 `xyz(mm) + RPY(deg)`，由 Piper 固件的 MOVE P 模式完成末端位姿控制。每周期先按照当前位置做平移/旋转限幅，再做工作空间裁剪。

注意：固件内部逆解在奇异位姿或不可达点可能失败。真实安装方向、末端工具偏置和安全工作空间不能从代码自动推断，必须在低速下逐臂标定。Pika 的 `gripper.pos` 为 0–1；发送给 Piper 前转换为 0–100。

### 3. Piper 主从双臂

leader 与 follower 都使用归一化关节值：六个关节为 -100–100，夹爪为 0–100。每侧动作以相同前缀配对。`max_relative_target` 对单周期关节目标做限幅，避免 leader/follower 初始姿态差异导致大步跳变。

## 安全默认值

- `park_on_connect: false`：连接时不会自动回零，避免双臂在未知环境中突然运动。
- `park_on_disconnect: false`：退出时不自动走停车轨迹。
- `disable_torque_on_disconnect: true`：退出时关闭 follower 扭矩。
- Pika→Piper 默认 MOVE P 15% 速度、5 mm / 0.05 rad 单周期限幅。
- 主从臂默认归一化关节单周期限幅 3.0。

如果现场要求断开后保持姿态，需要在充分评估坠臂风险后调整 `disable_torque_on_disconnect`。不要在负载未支撑时直接关闭扭矩。

## 已知边界

- 当前环境没有连接真实 Pika、相机和 CAN 机械臂，因此只能完成静态、导入和配置级验证；首次硬件联调必须逐臂进行。
- Pika→Piper 依赖 Piper SDK 的末端位姿控制，不是 MoveIt/外部 IK。不可达和奇异位姿由固件能力决定。
- 示例相机路径和工作空间只是模板，不是现场标定值。
- 本目录聚焦 LeRobot 双臂采集；原 Piper 项目的 Tk GUI 没有重复拷贝，仍可从 `lerobot_robot_piper-master` 单独使用。

详细的文件来源和改动清单见 `MODIFICATION_RECORD.md`。

