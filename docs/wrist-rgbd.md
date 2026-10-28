# 末端 RGB-D 接管抓取

**模型边界更正：当前是理想 RGB-D 代理，尚不是 RealSense D435 模型。** 以下安装位和 65° 垂直视场角是仿真假设。D435 的 RGB/深度传感器有不同视场与光心，需要厂商标定数据进行对齐；现模型直接使用同一理想视点。已有亚毫米定位结果不能用来证明 D435 的实机性能。

## D435 参数与安装标定

官方资料给出机身约 90 × 25 × 25 mm，深度最大分辨率 1280 × 720；RGB 视场约 69.4° × 42.5°，深度视场名义约 87° × 58°，具体模式有差异。D435 最小深度与分辨率相关：1280 × 720 约 280 mm，848 × 480 约 195 mm，640 × 480 约 175 mm。来源：[D400 datasheet，表 4-11](https://www.realsenseai.com/download/21345/?tmstv=1770190765)、[官方调优说明](https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance/)、[Intel D435 规格](https://www.intel.com/content/www/us/en/products/sku/128255/intel-realsense-depth-camera-d435/specifications.html)。

之前观测目标中心的深度约 218 mm，低于 D435 默认 1280 × 720 模式的约 280 mm 下限；原程序 30 mm 的有效深度下限属于理想代理，不能直接用于该硬件模式。后续先确定实际深度模式，再调整安装位/预抓取高度，并验证目标位于 RGB 与深度共同视野及有效测距范围，不凭理想仿真成功判定实机可行。

当前光心相对夹爪基座为 `[65,0,40] mm`，TCP 相对同一基座为 `[0,0,140] mm`，故光心相对 TCP 为 `[65,0,-100] mm`，直线距离约 119 mm。此距离不是相机壳体与夹爪的最近间隙，也没有厂家通用数值；它取决于安装支架。当前没有相机实体碰撞体，因此也没验证真实安装干涉。

真实使用时需区分三层：相机所选流的内参/畸变，RGB 与深度传感器之间的外参，以及相机光学系与机械臂工具/TCP 之间的手眼外参。前两者先从 RealSense SDK 读取工厂标定并核验；手眼外参必须在相机固定安装后采集多个有充分旋转变化的机器人姿态与固定标定板图像求解，换支架或拆装后重新核验。还需单独确定实际 TCP。参考：[RealSense 投影/对齐](https://dev.realsenseai.com/docs/projection-texture-mapping-and-occlusion-with-intel-realsense-depth-cameras/)、[OpenCV 手眼标定](https://docs.opencv.org/5.0/main_modules/calib.html)。

在当前仿真中，安装变换由建模精确指定，可直接作为配置使用；已有坐标一致性测试不等同于模拟过一次真实手眼标定。日志现在明确标记 `ideal_rgbd_proxy_not_D435` 与外参来源。

默认抓放流程现在为双全局相机粗定位 → 预抓取 → 末端 RGB-D 拍摄 → 三维位置修正 → 下降/真实夹取 → 转运/放置/验证。精定位失败时终止，不回退到粗定位或器件真值。

## 运行

要弹出实时 MuJoCo 窗口，请添加 `--view`：

```bash
./.venv/bin/python -m simulation.vision_pick_place --seed 7 --output runs/wrist-live-demo --view
```

macOS 下入口会自动通过项目虚拟环境的 `mjpython` 启动（显式用 Python 调用启动脚本，兼容含空格的项目路径）。Linux 下直接启动。窗口显示机械臂运动和当前阶段。GUI 以约 30 Hz 同步，物理控制仍每 2 ms 执行。

`--view` 现在默认同时打开原生桌面窗口 **Robot Camera Player**，四个面板显示全局 A/B 的 RGB、末端 RGB，以及末端米制深度的伪彩预览。MuJoCo 左下角也同步显示当前末端 RGB，不再停留在单张抓取关键帧。原生播放器使用 OpenCV Cocoa GUI（Mac），独立进程运行，不占用 MuJoCo UI 主线程。

播放器按空格暂停/继续画面，Q/Esc 或关闭播放器窗口退出播放器；这不暂停机械臂仿真。关闭 MuJoCo 窗口后，关联播放器与监看服务一并退出。它是实时监看器，尚无时间轴或历史录像回放功能。也可添加 `--camera-display browser` 使用之前的浏览器界面。两种界面读取同一本机监看数据；原生播放器每次获取四路的同一缓存快照，避免逐路请求时跨帧。

已验证本机 Cocoa 窗口实际可见、连续显示 16 次更新；记录与最终画面为 `runs/native-player-check/camera-player-status.json`、`runs/native-player-check/camera-player.jpg`。原生播放器不依赖 RealSense 设备或其 SDK。RealSense Viewer 是实机/RealSense 文件工具，MuJoCo 理想 RGB-D 代理没有直接作为 RealSense 硬件接入它。

相机监看使用独立的 640 × 360 小帧缓冲，每仿真秒采样 5 次；实际墙钟帧率随机器负载而变，页面显示实测帧率、仿真时间、帧号与动作阶段。4K 全局定位和 1280 × 720 末端精定位仍使用原来的分辨率，预览不参与控制。四路同一时刻顺序渲染，只保留最新四帧 JPEG，不累计保存预览录像；HTTP 线程只读取缓存，不访问 MuJoCo，不跨线程复用渲染器。仅监听本机 `127.0.0.1` 的临时端口，不公开到局域网。

结束后播放器显示 SUCCESS 或失败码，画面停在最终状态；MuJoCo 窗口保留，关闭 MuJoCo 窗口后停止监看服务并退出。浏览器模式还显示中文任务结束提示。若选择浏览器但未自动打开，复制终端 `Live camera monitor:` 后的网址即可。重复运行需打开新打印的网址。可用 `--camera-fps 2` 降低预览负载，或用 `--no-camera-monitor` 禁用监看；禁用时左下角仍只展示一次精定位关键帧。

验证：`runs/live-camera-monitor/result.json` 中完整抓放通过；`runs/camera-monitor-check.json` 核对了机械臂位姿改变后四路图像均变化，HTTP 图像均为 640 × 360，状态帧号与仿真时间推进。原生 4K 模型帧缓冲配置在构建预览缓冲后恢复。

不加 `--view` 是离屏验证，不弹窗；`--record-gif` 只保存动画，不负责打开窗口。两者可同时使用。抓放命令的输出目录重名时自动增加 `-001`、`-002` 等编号，保留旧结果；启动时打印实际目录。

项目根目录执行；抓放输出目录重名时自动编号：

```bash
./.venv/bin/python -m simulation.vision_pick_place --seed 7 --output runs/wrist-demo --record-gif
```

`--record-gif` 需要 ffmpeg，可省略。`--coarse-only` 运行之前的全局粗定位基线。

```bash
# 可追溯的偏差注入：仅修改粗定位控制消息，原始检测仍保留
./.venv/bin/python -m simulation.vision_pick_place --seed 0 --coarse-offset-mm 5 -5 --output runs/wrist-offset-demo
# 负例：到预抓取位后将深度置为无效，预期退出码 1
./.venv/bin/python -m simulation.vision_pick_place --seed 7 --invalid-wrist-depth --output runs/wrist-invalid-demo
# 独立的已知距离平面深度测试
./.venv/bin/python -m simulation.wrist_camera --output runs/wrist-depth-demo
```

## 相机与数据来源

相机分辨率 1280 × 720、垂直视场角 65°，刚性安装在 `xarm_gripper_base_link`。安装位置（该坐标系下）为 `[65, 0, 40] mm`，朝向局部 `[0, 0, 250] mm`，从夹爪侧面斜视目标顶部。这是可调整的虚拟安装位，没有相机外壳碰撞体，也没有完成真实硬件安装设计。

`wrist_camera.py` 中的固定手眼变换 `E_FROM_D` 使用 OpenCV 相机光学坐标（右、下、前）。世界相机位姿由机械臂 FK 的工具位姿乘以固定手眼变换得到；单元测试核对了与 MuJoCo 实际渲染相机的轴向及位置一致。估计器 `refine` 仅接受 RGB、深度、内参、FK/手眼外参、粗定位姿态和标记 ID，不接收 MuJoCo model/data。

RGB 和深度来自同一个 renderer、同一仿真时刻。深度使用 MuJoCo Renderer 的米制光学 Z 输出，不将 OpenGL 缓冲原值当距离。由目标 ID 的四角构建内部 ROI，剔除无效和越界深度，要求至少 50 个有效像素且有效比例不低于 60%。对深度点拟合局部表面，要求至少 80% 的点在平面 0.5 mm 内，再将标记中心射线与测得的表面相交，以保留斜视几何。位置由深度重建，不使用预设台面高度代替深度；姿态保留全局 yaw。精/粗位置差超过 20 mm 则拒绝，之后继续使用原有的桌面工作区/直立物体验证。

`WristController` 在预抓取完成后暂停推进，精定位成功后替换下降、闭合和抬升目标；转运和目标放置坐标保持配置值。动作阶段使用不可变 `Phase` 替换，避免共享目标数组被重复叠加修正。未增加自主重试或连续视觉伺服。

新增证据：`wrist_rgb.png`、`wrist_detections.png`、`wrist_depth_m.npy`（米制原始深度）、`wrist_depth.png`（仅预览）以及 `fine_observation.json`。日志保留有效深度数量、ROI 平面内点数、中心深度、修正量、内参、手眼和 FK 变换。`result.json` 将粗定位误差、末端精定位误差和最终放置误差分开记录。误差诊断使用器件真值，但不会改变控制目标。

## 2026-10-28 验证

| 场景 | 粗定位位置误差 | 末端位置误差 | 最终放置 XY 误差 | 结果 |
| --- | --- | --- | --- | --- |
| seed 7 | 0.295 mm | 0.177 mm | 1.856 mm | 13 项抓放验收通过 |
| seed 0，粗定位额外注入 X+5/Y−5 mm | 6.987 mm | 0.083 mm | 0.779 mm | 13 项抓放验收通过 |
| seed 7，末端深度无效 | — | 拒绝 | — | `INVALID_DEPTH`，未进入下降/闭合 |

证据：`runs/wrist-rgbd-seed7-verified`、`runs/wrist-rgbd-offset5`、`runs/wrist-rgbd-invalid`。

独立平面测试在 0.2 m、0.4 m 处测得最大深度误差分别约 0.00027 mm、0.00173 mm，验证了单位和光学 Z 约定。Mac OpenGL 后端提示 `ARB_clip_control unavailable`、深度精度可能受限；保留该提示，不将理想平面测试精度解读为真实传感器精度。报告位于 `runs/wrist-depth-calibration/depth_check.json`。

十二个单元测试覆盖现有全局定位/规划消息验证、末端实际使用深度修改 Z、无效深度与错误标记拒绝、共享动作数组修正，以及 FK/手眼与渲染坐标约定。运行 `./.venv/bin/python -m unittest discover -s tests -v`。

## 精度边界

两类相机目前均为理想针孔渲染，未加镜头畸变、深度噪声、曝光模糊或标定误差。这里没有证明真实相机的亚毫米性能。

末端位置误差与最终放置误差是不同指标。seed 7 中定位约 0.177 mm，但最终放置仍约 1.856 mm，转运中的器件相对夹爪漂移约 2.118 mm；现有 TCP 阶段收敛阈值是 5 mm。因此下一阶段应分开分析运动控制与接触/夹持/放置误差，不能只通过增加相机分辨率推断最终放置精度会改善。

实现遵循 [MuJoCo Python 渲染接口](https://mujoco.readthedocs.io/en/stable/python.html)，单位通过上述独立平面测试核验。
