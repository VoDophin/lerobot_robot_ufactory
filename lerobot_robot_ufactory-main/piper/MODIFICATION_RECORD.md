# Piper 集成修改记录

日期：2026-08-01

## 约束与结果

- 集成目标：在 `lerobot_robot_ufactory` 根目录下增加独立 `piper/` 子项目。
- 未修改父项目任何已有文件；未修改 `lerobot_robot_piper-master` 任何文件。
- 新子项目通过 LeRobot 0.4.3 的第三方插件发现和独立命令入口注册配置。

## 来源

Piper 的 follower、leader、CAN 总线归一化和标定表参考本机项目：

`D:/DLproject/lerobot_robot_piper-master/lerobot_robot_piper/`

Pika、双臂前缀、mock robot 和录制流程复用本机父项目：

`D:/DLproject/lerobot_robot_ufactory/src/lerobot_robot_ufactory/`

没有覆盖或移动上述来源文件。具体许可证说明见 `THIRD_PARTY_NOTICES.md`。

## 新增内容

| 文件/目录 | 用途 |
|---|---|
| `pyproject.toml` | 独立可编辑安装、依赖和三个命令入口 |
| `src/lerobot_robot_ufactory_piper/config.py` | 注册 `uf::piper`、`uf::dual_piper`、`uf::piper_leader`、`uf::dual_piper_leader`、`uf::dual_pika_teleop` |
| `src/.../motors/piper_motors_bus.py` | Piper CAN、关节/夹爪归一化、末端位姿读写、角色配置 |
| `src/.../piper_follower.py` | 单/双 Piper follower，关节和笛卡尔两种控制空间，限幅，相机 |
| `src/.../piper_leader.py` | 单/双 Piper leader，双侧前缀和并行读取 |
| `src/.../pika_teleop.py` | 双 Pika 组合器；在本地子类修复父项目 Pika 状态线程中的 `self.self` 拼写问题 |
| `src/.../pose.py` | Pika 轴角与 Piper RPY 之间的转换 |
| `src/.../scripts/` | 复用父项目录制/遥操入口，以及 YAML 静态检查器 |
| `config/dual_pika_direct.yaml` | 双 Pika 无机械臂直接采集模板 |
| `config/dual_pika_piper.yaml` | 双 Pika 遥操双 Piper 模板 |
| `config/dual_piper_leader_follower.yaml` | 双 Piper 主从臂模板 |
| `README.md` | 安装、配置、运行、安全说明 |

## 相对来源项目的行为调整

1. follower 的连接/断开不再默认停车；是否停车完全由配置控制。
2. `disable_torque_on_disconnect` 被实际使用。
3. `connect()` 可按配置自动写入 follower/leader 角色，不依赖先调用单独 setup 命令。
4. 双臂的连接、观测、动作和 leader 读取可并行，异常会回传主线程。
5. 相机和关节/位姿键统一加 `left.`、`right.` 前缀。
6. 新增 Piper 笛卡尔 follower：读取固件末端 RPY，转换为数据集轴角；动作反向转换后调用 `EndPoseCtrl`。
7. 新增笛卡尔单周期限幅、工作空间裁剪、主从关节单周期限幅。
8. Pika 夹爪 0–1 与 Piper 夹爪 0–100 在边界处显式换算。

## 验证记录

完成集成后执行：

- 全部 Python 文件 `compileall` 语法检查。
- 三份 YAML 的解析和结构检查。
- 搜索父项目已有文件的变更状态/新增范围，确认写入只发生在新 `piper/` 目录。
- 若本机当前 Python 环境包含 LeRobot 0.4.3、Pika、Piper SDK，则额外执行插件导入和配置注册检查。

真实硬件运动、相机读取和 CAN 时序没有在本次离线环境中执行；必须按 README 逐臂低速联调。

