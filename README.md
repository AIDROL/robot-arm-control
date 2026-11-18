# Robot Arm Control：光学实验机械臂仿真

本仓库是 AIDROL 光学机器人项目的机械臂控制与仿真部分，使用 xArm7 模型探索光学器件的视觉定位、抓取装配和精调工具（FAT）闭环调光。项目包含 Isaac Sim 和 MuJoCo 两套仿真，以及基于 LightPipes 的光学计算。

目前的验收结果来自仿真。机械臂、夹爪、相机、镜座及光学模型仍需与实物标定和验证；仿真误差不能直接视为实机精度。

## 最新演示：五件装配与 FAT 闭环

实验从空的实验区开始，机械臂依次放置五件器件，再抓取基于新 STEP 模型的 FAT。工具相机利用 ArUco 标记对准镜座，双轴接触调节后根据探测器图像反馈调光，最后将工具放回并重新测量。

https://github.com/user-attachments/assets/9f2d193b-f4ac-44a1-89d8-9c4fe5396701

视频约 53 秒，以 5 倍速展示全景、全局相机、末端 RGB-D、FAT 侧视和记录的探测器测量。[打开备用 MP4](docs/media/fat-cad-michelson-replay.mp4?raw=true)

### 本次仿真验收结果

| 验收项 | 结果 |
| --- | --- |
| 五件器件放置 | 5/5 完成，最大平面误差 1.345 mm |
| FAT 光学反馈 | 12 次调节，非预期接触 0 次 |
| FAT 放回误差 | 1.347 mm |
| 工具撤出后干涉可见度 | 99.9959%（512 网格），99.9958%（1024 网格） |

视频来自本机验收记录 `fat-cad-full-20261118-02`，对应 250.21 秒 PhysX 轨迹。FAT 调光使用 Isaac 内部控制器和标量光学模型；镜座为匹配新 FAT CAD 的仿真设计。这一记录保存在生成它的机器上，**不会随仓库克隆下载**。

## 环境准备

以下 Isaac Sim 流程已在 Ubuntu 22.04、Isaac Sim 5.1 和 Python 3.11 环境运行。先在仓库根目录创建 Python 环境并安装光学依赖：

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install -r requirements-optics.txt
```

Isaac Sim 使用自己的 `python.sh`。脚本默认查找 `~/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh`；如果安装位置不同，设置：

```bash
export ISAAC_SIM_PYTHON=/path/to/isaac-sim/python.sh
```

Isaac 启动和环境背景见 [Isaac Sim 交接文档](docs/isaac-sim-5.1-handoff.md)。其中的旧分支和迁移计划属于历史记录，请以本页的当前入口为准。

## 运行与回放最新流程

所有命令从仓库根目录执行。新实验需要使用**尚不存在的输出目录**，以保留已有证据：

```bash
# 从空实验区开始：五件装配、FAT 调光、工具放回及验收
# 如果 runs/fat-cad-full-new 已存在，请换一个目录名
env -u PYTHONPATH .venv/bin/python scripts/fat_closed_loop.py run --assembly \
  --folder runs/fat-cad-full-new
```

查看进度与验收结果：

```bash
# 查询运行阶段和日志位置
env -u PYTHONPATH .venv/bin/python scripts/fat_closed_loop.py status \
  --folder runs/fat-cad-full-new

# 完成后校验证据清单和验收结果
env -u PYTHONPATH .venv/bin/python scripts/fat_closed_loop.py verify \
  --folder runs/fat-cad-full-new
```

打开 Isaac 三维回放，默认包含 ArUco 识别与末端深度相机视图：

```bash
env -u PYTHONPATH .venv/bin/python scripts/fat_closed_loop.py replay \
  --folder runs/fat-cad-full-new --camera overview --speed 4
```

可将 `--camera overview` 改为 `global`、`wrist` 或 `fat`，也可在回放窗口切换视角。如果实验仍在运行，可给回放命令加 `--wait`，等待实验完成和证据封存后再打开。回放读取已保存的轨迹，不会重新运行物理仿真。

如果本机已有本次验收目录，也可以将上述命令中的目录替换成 `runs/fat-cad-full-20261118-02`。运行结果中的轨迹、图像、测量、审计及 `evidence-manifest.json` 共同构成验收证据；较大的运行目录不会随 Git 仓库分发。

## 其他实验入口

| 实验 | 内容与运行方式 |
| --- | --- |
| 历史 Isaac 全流程 | 五件装配、FAT 接触调节与光学验收。仓库包含历史验收轨迹，可按[验收文档](docs/isaac-michelson-validation-2026-11-03.md)查看。 |
| MuJoCo Michelson 全流程 | 视觉装配、接触调节和光学反馈：`./.venv/bin/python -m simulation.michelson_experiment --output runs/michelson-live --view`。详见[全流程说明](docs/michelson-full-process.md)。 |
| MuJoCo FAT 接触验证 | 预装配镜座下的独立调镜验证：`./.venv/bin/python -m simulation.fat_mechanics --michelson --output runs/michelson-demo --view`。详见[FAT 接触验证](docs/fat-contact-michelson.md)。 |
| 单件视觉抓放 | 全局相机定位和腕部 RGB-D 抓取修正：`./.venv/bin/python -m simulation.vision_pick_place --seed 7 --output runs/visual-grasp-demo --view`。详见[视觉抓放](docs/vision-pick-place.md)。 |
| 基础五件搬运 | 在线笛卡尔控制演示：`./.venv/bin/python simulation/line_assembly.py --seed 7`。详见[控制器说明](docs/task-space-control.md)。 |

这些入口对应不同场景和验证范围。例如，历史几何 FAT 演示不包含真实接触调节；独立 FAT 接触验证从预装配镜座开始。请结合各自文档解读结果。

## 仓库结构与文档

| 路径 | 内容 |
| --- | --- |
| `simulation/isaac/` | Isaac Sim 物理仿真、相机与回放 |
| `simulation/mujoco/` | MuJoCo 后端；共享光学和实验代码位于 `simulation/` |
| `scripts/` | 最新 FAT 全流程运行、状态查询、校验和回放入口 |
| `assets/`、`configs/` | 机械臂、FAT 和光学器件模型及场景配置 |
| `docs/` | 实验、验证、相机与接口说明 |
| `runs/` | 本机运行记录；仅部分历史验收资料纳入仓库 |

- [Isaac 相机与光学查看](docs/isaac-camera-optics-view.md)：相机画面、探测器测量和回放选项。
- [相机检测叠加](docs/camera-detection-overlay.md)：ArUco 检测框、角点、ID 与坐标轴。
- [历史 Isaac Michelson 验收](docs/isaac-michelson-validation-2026-11-03.md)：历史轨迹、结果与验收范围。
- [光学模型](docs/lightpipes-integration.md)与[光学探测器](docs/optical-detector.md)：LightPipes 传播计算和测量结果。
- [项目分工与协作](docs/project-management.md)：仓库职责与贡献流程。

机械臂模型基于 [MuJoCo Menagerie 的 UFACTORY xArm7](https://github.com/google-deepmind/mujoco_menagerie/tree/main/ufactory_xarm7)。真实夹爪、机器人参数和相机标定仍需在硬件到位后确认。
