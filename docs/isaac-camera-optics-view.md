# Isaac 相机与调光画面

实际实现按后端放在 `simulation/isaac/` 和 `simulation/mujoco/`。`simulation/` 根目录保留旧命令的兼容入口，以及共用的 `michelson_assembly_optics.py` 和 `optical_sensor.py`。Isaac 的机械几何和独立运动学仍从原 MJCF 读取；动力学、接触、机器人物理状态和 RGB/深度帧来自 PhysX / Isaac。

## 立即查看已验收实验

在仓库目录运行：

```bash
cd /home/terry/Desktop/workspace/robot-arm-control
./.venv/bin/python simulation/isaac/camera_optics_monitor.py \
  --folder runs/isaac-michelson-final-20261103-006
```

窗口四格依次显示 Isaac 顶视全局 RGB、腕部 RGB 与公制深度小图、FAT 侧视 RGB，以及 LightPipes 八相位扫描中的第 0 帧探测器图像。底部是从每次实际探测器测量读出的可见度、波前相对倾角和可见度随时间变化曲线。探测器图像采用固定对数显示增益；定量结果仍由原始 16 位像素计算。每格标注采样时刻与当前显示时刻的差值，旧帧不会假装成当前图像。

要按仿真时间看相机与调光过程：

```bash
./.venv/bin/python simulation/isaac/camera_optics_monitor.py \
  --folder runs/isaac-michelson-final-20261103-006 \
  --replay --start-seconds 150 --speed 2
```

按 `1`、`2`、`3`、`4` 放大全局、腕部、侧视或探测器画面，按 `0` 回四格；重放时按空格暂停或继续，按 `q` 或 Esc 退出。Isaac 三维场景另开终端播放：

```bash
bash simulation/isaac/run_isaac_michelson_replay.sh \
  --folder runs/isaac-michelson-final-20261103-006 \
  --start-seconds 150 --speed 2
```

两个播放窗口读取同一条已保存的 PhysX 轨迹/观测；手动分别启动时墙钟时间不保证逐帧同步。重放不重新运行控制或验收。

## 新实验中的实时更新

先在终端一启动新实验，再在终端二关注同一输出目录：

```bash
bash simulation/isaac/run_isaac_michelson.sh \
  --output runs/isaac-live-001 --stage full --view
```

```bash
./.venv/bin/python simulation/isaac/camera_optics_monitor.py \
  --folder runs/isaac-live-001
```

`--view` 现在会启动独立显示进程：物理进程按原 2 ms 步长继续求解，显示进程以目标 30 FPS 展示最新实际刚体状态，并用 100 ms 缓冲做显示插值。插值只影响画面，控制、接触检查、验收均读取物理进程的实际状态。三路相机先在仿真时间 0 生成首帧，再开始机械臂运动；之后以约 5 Hz 持续生成 640×360 预览，腕部预览包括公制深度。监视器自动读取 `live/preview.json`，因此相机画面会跟随运动，而无需等控制步骤调用某台相机。

高分辨率控制采样仍保留：全局相机在装配前后采样，腕部相机在五次抓取前采样，FAT 侧视在工具插入期间多次采样。这些文件与持续显示预览分别保存。没有 `--view` 的运行只产生原来的控制采样。`--replay` 也继续按原始采样时间查看归档图像。

调光反馈约每 3 **仿真秒**做一次新八相位探测器测量，测量完成后窗口更新可见度与曲线。空台面尚未装好光路时，探测器区域显示“装配完成后可用”。显示的 FPS 与物理推进速度是两个指标；窗口同时显示物理时间和它相对墙钟的速度。保留原接触求解参数时，这台机器的物理计算仍可能慢于实时。

每次原始图像可在运行目录直接查看：`before-*.png`、`after-*.png`、`*-wrist.png`、`*-wrist-depth-m.npy`、`fat-side-*.png`，以及 `optics/<测量名>/phase-00-detector16.png` 到 `phase-07-detector16.png`。对应的 `*-vision.json`、`*-wrist.json`、`fat-side-*.json` 和 `optics/<测量名>/measurement.json` 记录时间与测量数值。`--snapshot <路径> --no-window` 可生成当前四格 PNG。

持续预览在 `live/global.jpg`、`live/wrist.jpg`、`live/side.jpg` 和 `live/wrist-depth-m.npy`，只保留最新帧；它们不作为物理验收原始数据。`live/state.npz` 是最新原生状态，`live-view.log` 是显示进程日志，`live/viewer-performance.json` 保存退出时的帧率和首帧时间。关闭三维 live 窗口会请求物理进程停止并保存当前运行。

## 2026-11-04 实测

`runs/isaac-live-display-check-002` 在真实 PhysX 单件夹持、抬升过程中同时运行独立显示与三路预览：2494 帧的中位帧率 29.85 FPS，95% 帧间隔不超过 50.8 ms；持续相机预览更新 418 次，三路首帧均在仿真时间 0。夹持测试通过，抬升 240.65 mm，搬运期间双侧接触 100%。启动阶段观测到 GPU 总显存约 3588 MiB / 8151 MiB。此项验证的是显示和单件夹持流程；完整装配、调镜与归还的既有验收记录仍是 `isaac-michelson-final-20261103-006`。

旧的同进程显示每 33 个物理步才刷新一次，在用户这次约 427 秒推进 96 秒的运行中，显示只有约 3–4 FPS。分离显示后，物理进程仍按原 500 Hz 仿真步长求解，实测计算时间仍慢于实时；显示帧率不会再被这个步进速度直接限制。

## 模糊与画面闪动修复

后续截图暴露出另一项显示问题：初始化相机渲染产品后，实际抗锯齿设置变成 `/rtx/post/aa/op=3`，且 DLSS 为 Performance 模式。原来的 SimulationApp `anti_aliasing=1` 并没有保证运行时继续使用该选择。冻结实际机械臂姿态后，画面仍有边缘闪动和模糊，确认存在显示问题。

现在通过 Replicator 的 `set_render_rtx_realtime(antialiasing="FXAA")` 明确启用非时域抗锯齿；同时关闭显示进程的运动模糊、DLSS 帧生成、随机反射和间接漫反射。视口使用原生 1440×810、分辨率比例 1；启动后读回并保存 `live/render-settings.json`，若 FXAA 被初始化流程覆盖则报告错误。上述设置只影响独立显示进程，控制用相机、机械参数和 LightPipes 光学测量保持原流程。

实测 `runs/isaac-render-quality-balanced-20261104`：原生物理场景门槛通过，显示中位数 29.85 FPS，95% 帧间隔不超过 54.6 ms。1080p 在本机与物理进程并行时约为 16 FPS，因此默认采用 1440×810。冻结同一实际姿态的渲染对照在该运行的 `frozen-render-baseline/` 与 `frozen-render-proof/`；将图像统一到 1280×720 后，静止轮廓的 95% 边缘漂移由约 2.32 像素降至约 0.96 像素。这是显示稳定性诊断，不能换算成机械定位精度。当前 90 项单元测试通过。
