# Ubuntu / Isaac Sim 5.1 迁移交接（2026-11-02）

## 当前基线与目标

当前源码位于 GitHub `AIDROL/robot-arm-control` 的 `vision-grasp-camera-player` 分支。现有仿真在 **MuJoCo 3.16 + Python 3.11** 上实现了视觉抓放、五器件 Michelson 装配、FAT 物理接触调节和 LightPipes 光学反馈。Ubuntu 上的第一阶段目标是重建**最小视觉闭环**：Isaac Sim 图像 → ROS 2 → OpenCV ArUco 位姿 → 机械臂接近 → 新图像复测。不要把已有 MuJoCo 代码当作可直接在 Isaac Sim 执行的控制脚本。

目标电脑由用户报告为 Ubuntu 22.04、ROS 2 Humble、Isaac Sim `5.1.0-rc.19+release.26219.9c81211b.gl`、RTX 5060 Laptop 8151 MiB、NVIDIA 驱动 570.211.01。Isaac Sim 5.1 文档列出的 Linux 驱动为 580.65.06，最低显存 16 GB。先用 Isaac Sim Compatibility Checker 和单路 720p 场景实测；驱动升级应使用适合该笔记本的厂商/Ubuntu 方案。8 GB 显存不宜作为双路 4K 加腕部相机的默认配置。

## 在 Ubuntu 获取和验证代码

```bash
git clone https://github.com/AIDROL/robot-arm-control.git
cd robot-arm-control
git switch vision-grasp-camera-player
git pull --ff-only
python3.11 -m venv .venv
./.venv/bin/python -m pip install -r requirements-optics.txt
./.venv/bin/python -m unittest discover -s tests
```

上面命令安装的是**独立的 MuJoCo 对照环境**。Isaac Sim 5.1 使用自己的 Python 3.11 环境；外部 Ubuntu 22.04/Humble 节点一般使用系统 Python 3.10。通过 ROS 2 标准消息交换数据，避免在启动 Isaac Sim 的终端 source 系统 ROS 2 的 Python 3.10 环境。若后续必须在 Isaac 进程内使用自定义 ROS 消息，应按 5.1 文档另建 Python 3.11 版本的 ROS 工作区。

本仓库的 `runs/` 被 Git 忽略。`runs/michelson-full-demo` 的 `.mjb`、`.npz`、测量图和视频**不会通过 git clone 获得**；可在 Ubuntu 重新运行，或单独复制 Mac 上该目录。读源码和运行单元测试不依赖演示录像。保存证据后不要把回放当成 Ubuntu 上的新物理闭环验证。

## 必须保留的接口语义

| 现有模块/文件 | 迁移中承担的职责 |
| --- | --- |
| `simulation/global_camera.py`, `simulation/wrist_camera.py`, `simulation/fat_vision.py` | 图像 ArUco 检测、内外参和位姿推算；迁移时让图像来自 Isaac/ROS，而非 MuJoCo renderer。 |
| `simulation/vision_overlay.py`, `simulation/camera_player.py` | 检测框、角点、ID、坐标轴显示；各路相机保持独立窗口。 |
| `simulation/vision_pick_place.py`, `simulation/task_space_control.py` | 图像定位后的抓放阶段和控制逻辑；Isaac 版必须接真实关节状态与命令，不可直接复用 `MjModel/MjData` 调用。 |
| `assets/optical_housing/highousing.stl`, `simulation/housing_profile.py` | 已选空心镜架的外形、尺寸、标记/抓持几何；导入 USD/碰撞体后逐一核对单位与朝向。 |
| `configs/optical_setups/michelson_full.json`, `simulation/michelson_experiment.py` | 五器件目标和完整顺序；在单件视觉闭环稳定后再迁移。 |
| `simulation/fat_mechanics.py`, `simulation/michelson_assembly_optics.py` | FAT 接触、螺杆/镜面机构和光学观测；物理约束、摩擦与接触验收必须在 Isaac 中重新验证。 |
| `simulation/optical_sensor.py`, `simulation/michelson_optics.py` | LightPipes 强度图、功率和可见度。光波传播继续用 LightPipes，Isaac 负责几何、运动与相机图像。 |

坐标约定：世界 Z 向上，长度米，角度弧度。镜架 STL 名义尺寸 72 × 64 × 108 mm，带立柱/底座后名义光轴高于桌面 100 mm。侧面 ArUco 位于平面，两侧凹槽供夹爪夹持；光轴应穿过镜架开口。详细配置见 `simulation/housing_profile.py`，不要仅凭显示外观推断局部轴或碰撞形状。相机光学坐标系与 ROS REP 103/TF 的变换需要显式保存和验证。

## 迁移次序和验收

1. **环境门槛**：Isaac 5.1 空场景稳定运行，Compatibility Checker 结果和实际驱动记录在实验日志。确认 ROS 2 Bridge 可发布 `/clock`。
2. **单件视觉**：桌面、一个现有 STL 镜架和顶/侧 ArUco、一台 1280×720 相机。Isaac 发布 `sensor_msgs/Image` 和 `CameraInfo`；外部 Humble/OpenCV 节点订阅并输出带时间戳的检测结果及位姿。移动、旋转镜架后，结果必须从**新图像**随之变化；仅允许读仿真真值做误差评估。
3. **机械臂闭环**：导入 xArm7/夹爪 URDF 或经核实的资产，核对关节轴、限位、TCP 和夹爪接触。由检测位姿规划接近，发布关节目标，读取 `JointState`，动作后重新拍图。初期不要加入多相机、4K 或 Michelson。
4. **抓放与工具**：单件物理抓放、腕部 RGB-D、FAT 侧相机，逐项与 MuJoCo 的图像、接触和放置误差日志对照。镜架 STL 的凹槽和开口需要匹配的碰撞体，避免用封闭凸包堵住光路/夹持区。
5. **完整实验**：五件从备料区到空实验区的装配，真实达到位姿的底座固定，FAT 双侧接触调镜，再把实际几何输入 LightPipes，最后验收归还后的新测量。参考 `docs/michelson-full-process.md`；MuJoCo 的理想磁座、六角轴和相位扫描参数均是暂定模型，不构成实机参数。

最小闭环的 ROS 消息先采用标准类型，不引入自定义消息：`Image`、`CameraInfo`、`JointState`、`geometry_msgs/PoseStamped`、`/tf`、`/clock`，机械臂命令可用标准关节轨迹接口。需要统一时间戳、frame_id、相机内参和视觉结果的来源；检测失败时禁止下降抓取。画面中分别开相机窗口显示角点与检测框，不把多个 sensor 画面合成在机器人 render window 内。

## 已验证结果与边界

Mac/MuJoCo 完整演示曾从空实验区连续运行至 263.796 s，五件抓放和 FAT 流程成功，归还后 LightPipes 两种网格的可见度约 98.0035%/98.0039%。该次器件粗放 XY 误差约 1.3–5.5 mm，FAT 归还误差 2.921 mm。见 `docs/michelson-full-process.md`。这些结果**尚未在 Ubuntu/Isaac Sim 重现**。机器人/镜架标定、真实 D435 外参、光学功率测量、磁座刚度、螺杆传动和真实夹爪仍需后续确认。

Isaac Sim 5.1 官方资料：[系统要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)、[ROS 2 安装与 Python 环境](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_ros.html)、[MJCF 导入](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/import_mjcf.html)。
