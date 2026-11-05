# Ubuntu / Isaac Sim 5.1 迁移实测（2026-11-03）

后续完整原生物理与光学迁移已通过，见 [Isaac Michelson 最终验收](isaac-michelson-validation-2026-11-03.md)。最终记录 `runs/isaac-michelson-final-20261103-006` 从空实验区完成五件装配、FAT 接触调镜、归还后双网格新测量和独立审计，88 项测试通过。下文保留最初的环境、ROS 2 图像桥接和导入基线记录。

## 本机环境

- Ubuntu 22.04.5, ROS 2 Humble, Isaac Sim 5.1.0 standalone。
- RTX 5060 Laptop，8151 MiB 显存，NVIDIA 570.211.01。
- Isaac Compatibility Checker：驱动/GPU/CPU/RAM/系统支持；显存 8.55 GB 小于检查器要求的 10 GB，整体结果 `FAILED`。Isaac 5.1 无窗口空场景和下面的单路 720p 场景仍实测可运行。这不保证加上机械臂、深度相机或完整实验后可运行。
- `vision-grasp-camera-player` 分支；独立 `.venv` 使用 Python 3.11，按 `requirements-optics.txt` 安装。`python -m unittest discover -s tests`：76 项通过。

## 已完成的 Isaac → ROS 2 → 图像位姿基线

`simulation/isaac/isaac_vision_scene.py` 在 Isaac 3.11 中从仓库原始 STL 构建单镜架场景，附 24 mm 顶面 ArUco ID 0、12 mm 侧面 ArUco ID 10、桌面及单台 1280×720 顶视相机。它发布 `/clock`、`/tf_static`、`/optics/camera/image_raw` 和 `/optics/camera/camera_info`。运行中途把镜架从 `(-0.08, 0, 0.114)` 平移到 `(0.08, 0.03, 0.114)`，并旋转 25°；Z 是镜架局部原点相对桌面的高度。此相机只验收顶标，侧标留待侧视相机验证。

`simulation/isaac/ros_aruco_pose_node.py` 用系统 `/usr/bin/python3` 和 ROS 2 Humble 订阅图像与内参，只由新图像解算 ID 0 的位姿，向 `/optics/marker_0/pose` 发布 `PoseStamped`。没有读取 Isaac 的物体真值来产生检测。`--view` 可开独立 OpenCV 画面；`--output` 保存逐帧 JSONL。

在两个终端分别运行（第二个终端不要 source 系统 ROS 2 到 Isaac 的 Python 3.11 环境）：

```bash
cd /home/terry/Desktop/workspace/robot-arm-control
/usr/bin/python3 simulation/isaac/ros_aruco_pose_node.py --view --output runs/isaac-vision/detections.jsonl
```

```bash
cd /home/terry/Desktop/workspace/robot-arm-control
env -u PYTHONPATH -u ROS_DISTRO -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  /home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  simulation/isaac/isaac_vision_scene.py --frames 240 --capture-dir runs/isaac-vision
```

必须清理 `PYTHONPATH`：当前登录环境把 `/opt/ros/humble/.../python3.10` 加到路径中，Isaac 3.11 若误加载它会报 `rclpy._rclpy_pybind11` 缺失。清理后 Isaac 桥接扩展加载其内置 Humble `rclpy`。

2026-11-03 运行 90 帧的证据在 `runs/isaac-vision-20261103/`（Git 忽略）。外部节点收到 28 帧且 28 帧均检出 ID 0，时间戳递增。镜架移动前相机光学坐标中位数约 `(0, 0.0823, 0.4473)` m，移动后约 `(-0.0311, -0.0826, 0.4463)` m，变化 `(-0.0309, -0.1648, -0.0010)` m；与该相机朝向下的预期 `(-0.03, -0.16, 0)` m 接近。重投影误差约 0.1–0.3 px。这个数值只能证明图像闭环的位移响应，不能作为绝对毫米精度标定。

另用系统 ROS 2 CLI 在场景运行时确认能发现 `/clock`、`/tf_static`、`/optics/camera/image_raw`、`/optics/camera/camera_info`；`ros2 topic echo --once /clock` 收到仿真时间约 15.067 s。

## xArm7 导入与单关节物理命令

`assets/xarm7_menagerie/` 保留当前 MuJoCo 使用的 UFACTORY Menagerie MJCF、网格和 BSD-3-Clause 许可证。`simulation/isaac/isaac_import_xarm7.py` 用 Isaac 5.1 导入器生成 `assets/xarm7_isaac/` USD 分层资产。导入的 USD 有 `joint1`–`joint7` 七个臂关节，另有夹爪/闭环关节；Isaac articulation 暴露 13 个自由度。

`simulation/isaac/isaac_arm_smoke.py` 独立加载导入的 USD，以 Isaac 物理步进给关节 1 下发位置目标。导入器给该关节的位置刚度为 0；烟雾测试显式设 `kp=1000, kd=50` 后，位置从约 `0.00530 rad` 到 `0.10531 rad`，目标 `0.10530 rad`。这是单关节可控性测试，不代表关节限位、TCP、夹爪接触或视觉引导接近已验收。

`isaac_vision_scene.py --with-arm` 在同一 720p 场景加载 xArm7，将其放在桌面边缘，并把镜架放在清晰视区。相机高度为 1.2 m，焦距相应调大；机械臂保持转向备料区的停放姿态。90 帧实测记录在 `runs/isaac-arm-vision-20261103/`：外部节点收到 28 帧，25 帧检出 ID 0；镜架移动前后相机坐标中位数差约 `(-0.0310, -0.1659, 0.0041)` m，预期平面变化为 `(-0.03, -0.16, 0)` m。这证明单路 720p 图像在已加载机械臂的场景中仍可用，但机械臂未受图像位姿驱动。

```bash
env -u PYTHONPATH -u ROS_DISTRO -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  /home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  simulation/isaac/isaac_arm_smoke.py
```

合并场景在上述第二个终端命令末尾添加 `--with-arm` 即可运行。

## 初始阶段的迁移缺口（历史记录）

最初此记录完成的是交接文档的环境门槛、单件顶视视觉，以及独立的机械臂导入/单关节命令烟雾测试。当时完整抓放、FAT 和光学链路尚未验收。后续完整后端已加入源几何碰撞分解、原生有限力驱动与状态核对、腕部 RGB-D、侧相机对准、物理抓放和光学闭环，并在这台 8 GB 显存机器实测通过。完整控制运行在 Isaac Python 进程；`JointState` / 轨迹 ROS 接口仍属于独立的后续工作。最初的 `isaac_vision_scene.py` 单镜架视觉基线继续保留，不承担完整抓放验收。
