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
  --replay --start-seconds 150 --speed 4
```

按 `1`、`2`、`3`、`4` 放大全局、腕部、侧视或探测器画面，按 `0` 回四格；重放时按空格暂停或继续，按 `q` 或 Esc 退出。Isaac 三维场景另开终端播放：

```bash
bash simulation/isaac/run_isaac_michelson_replay.sh \
  --folder runs/isaac-michelson-final-20261103-006 \
  --start-seconds 150 --speed 4
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

`--view` 会启动独立显示进程：物理进程按原 2 ms 步长继续求解，显示进程用 100 ms 缓冲插值展示最新实际刚体状态。插值只影响画面，控制、接触检查、验收均读取物理进程的实际状态。单视口默认目标 60 FPS，可用窗口中的 Overview / Global / Wrist / FAT 按钮切换全分辨率视角，也可自由调整观察角度。三路预览相机先在仿真时间 0 生成首帧，再开始机械臂运动。

打开 `camera_optics_monitor.py --folder <run>` 后，监视器自动请求连续相机预览：显示进程在整个监视期间持续渲染三路 480×270 RGB，腕部包括公制深度，以最高 15 Hz 发布预览；此时主视口目标限制为 30 FPS，为所有画面留出 GPU 预算。关闭监视器后自动恢复单视口 60 FPS。监视器异常退出时请求在两秒内过期，不会留下持续后台渲染。`--view-fps`、`--camera-fps` 和 `--preview-width` 可调整各项目标；预览设置不影响高分辨率控制相机。监视器持续读取 `live/preview.json`，无需等控制步骤调用某台相机。

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

## 2026-11-06 剩余频闪定位与修复

上面的 FXAA 修改只改善了最初的模糊，剩余频闪在冻结原生刚体姿态后仍能复现。本次直接连续捕获实际视口 30 帧，用 `check_display_stability.py` 检查静止轮廓和像素变化，不用平均 FPS 代替频闪检查。

两项因素共同造成问题：显示相机原来每 0.2 秒启停一次渲染产品，另外原场景的穹顶灯阴影采样仍存在轮廓闪动。固定姿态时跳过重复 USD 写入、移除显示副本的 articulation root 都不能消除问题。逐项对照结果如下（1440×810；像素变化指相邻帧灰度差大于 5 的比例）：

| 显示设置 | 明显变化像素平均比例 | 冻结轮廓的最大 95 分位偏移 |
| --- | ---: | ---: |
| 修改前 | 约 0.703% | 1 px |
| 仅保持相机连续渲染 | 约 0.043% | 1 px |
| 仅关闭显示阴影 | 约 0.100% | 1 px |
| 两项同时应用，正式 live 入口 | 约 0.0035% | 0 px |

最初以 FXAA 隔离频闪后，性能剖分确认原生像素渲染在多相机场景中仍会占满 GPU：状态读取、姿态写入均小于 1 ms，渲染 95 分位耗时约 57 ms。单纯保留 FXAA 时，请求 60 FPS 仍有长帧。最终 `display_rendering.py` 使用输出 1440×810 的 DLSS **Quality**（`aa/op=3`、`execMode=2`），关闭显示副本的阴影和先前已关闭的反射/间接光等效果，保留原漫反射照明。它与最初引发模糊的 DLSS Performance 配置不同，已用实际视口截图重新验证。

控制进程中的灯光、相机、物理求解参数和光学验收管线不使用该显示配置。相机预览按需启用，并在监视期间持续渲染；只限制 JPEG/深度发布频率，不在每个采样周期启停渲染产品。监视器取消原来 150 ms 的事件等待上限。刚体姿态采用一次 USD change block 更新，显示帧和预览发布分别使用持续的墙钟截止时间调度。

三维 replay 默认目标为 60 FPS，同样有 Overview / Global / Wrist / FAT 视角按钮。`--camera wrist` 或 `--camera fat` 可指定启动视角。`--fps 30` 可另设显示目标；该参数不改变原记录中的运动速度。回放时钟在视口预热后开始，避免初始化耗时跳过开头的运动。

可复跑的冻结检查（固定相机，不操作视口）：

```bash
bash simulation/isaac/run_isaac_michelson_replay.sh \
  --folder runs/isaac-michelson-final-20261103-006 \
  --start-seconds 16.89 --freeze --max-frames 240 \
  --render-probe-dir runs/replay-stability-check
./.venv/bin/python -m simulation.isaac.check_display_stability \
  runs/replay-stability-check
```

检查要求至少 20 张非空白视口截图，95 分位轮廓变化不超过 0.5 px，明显变化像素平均比例不超过 0.5%。最终 Quality 配置在三路预览启用时也通过：轮廓偏移 0 px，明显变化像素比例约 0.0008%。

最终真实运动测试 `runs/isaac-display-final-motion-20261106` 的物理场景门槛通过，三路首帧均在仿真时间 0；单视口平均约 58.3 FPS，95% 帧间隔约 16.7 ms。另一个真实抓取/抬升测试 `runs/isaac-display-fix-grip-20261106` 的夹持门槛通过，抬升 240.7 mm，搬运期间双侧接触 100%。它检验的是显示并发时原物理门槛，不替代完整 Michelson 验收。

通过实际启动、关闭监视器检验模式切换：单视口平均约 60 FPS，95% 帧间隔约 16.7 ms；三路预览启用时，主视口平均约 30 FPS，95% 帧间隔约 33.4 ms；监视器关闭后恢复 60 FPS。截图检查有额外开销，不用其平均 FPS 代替独立运动测试。物理计算仍按原精度推进，可能慢于墙钟时间，这与显示是否流畅分别记录。

最终三维 replay 在 8 倍速覆盖完整记录（0–240.8 s，装配、调镜和归还），2200 显示帧平均约 60.00 FPS，95% 帧间隔约 16.71 ms。最终 Quality 配置的 replay 冻结检查也通过；腕部和 FAT 入口均已实际切换并截图检查。完整测试套件 95 项通过，原已验收记录未改写。本机汇总报告为 `runs/isaac-display-fix-20261106/final-verification.json`，临时对照原型归档在同目录的 `debug-probes/`，不属于生产入口。

## 2026-11-09 展示质感与播放控制

11 月 6 日的配置为隔离频闪而关闭显示阴影，均匀穹顶光让白色机械臂与背景接近，缺少立体感。原导出的 STL 三角网格也没有曲面法线，定向照明会暴露棱面。当前显示副本改为零角度主光、正面冷色补光和定向阴影，保留 DLSS Quality；增加缎面表面材质，为机械臂曲面补充保留锐边的平滑法线，并关闭视口辅助网格。修改只应用于独立显示副本；物理几何、接触材质、控制采样和已验收记录保持原样。

动作慢与帧率分别处理：此前 2 倍速回放实测稳定 60 FPS，但 240.846 秒的记录仍需约两分钟播放。现在三维 replay 默认 4 倍速，完整流程约 60 秒；窗口可切换 1 / 2 / 4 / 8 倍速、暂停/继续、从头播放、前后跳转 10 秒。切换速度或恢复播放不会造成时间跳变。结束后保持最后场景，便于继续检查各相机视角；批量检查可用 `--exit-after 3` 自动关闭。回放包含归还后最终探测器测量时刻的原生姿态。live 仍遵循实际物理进度，回放倍速不会加速物理计算。

独立光学监视器仍使用自己的播放时钟，三维窗口的倍速、暂停与跳转不会同步控制它。原始光学结果未插值或重新计算。

固定原生姿态 16.89 秒的同角度对照中，机械臂固定区域的灰度 90–10 分位差从 7 增至超过 25；该值用于确认形状明暗恢复，不代表主观画质评分。主视角 30 张视口连续截图的 95 分位轮廓变化为 0 px，明显变化像素平均约 0.010%，通过原来的频闪检查；目标视口仍为 1440×810、60 FPS。

完整 4 倍速回放覆盖 0–240.846 秒，3674 帧平均 59.98 FPS，95% 帧间隔约 16.71 ms；含一秒终点停留和清理耗时的总播放计时为 61.28 秒。102 项测试通过。全局、腕部、FAT 相机均实际启动并截图检查，保持约 60 FPS；但这三个局部冻结检查仍测出 1 px 的细边变化，未通过严格的 0.5 px 轮廓门槛，不能宣称所有角度完全消除时域变化。改用 FXAA 降到约 46 FPS，仍未改善该指标；提高直接光采样和关闭环境遮蔽也未改善，因此未纳入生产配置。

新画质下的三路预览预算检查使用冻结原生姿态，不运行新物理求解：300 帧平均 30.00 FPS，95% 帧间隔 33.38 ms，预览更新 147 次；该显示进程也测出 1 px 细边变化。完整指标、通过及未通过的检查、修改前后截图保存在本机 `runs/isaac-display-quality-20261109/verification.json`，不替代物理验收。


## 2026-11-12 激光器与探测屏的可辨认外观

旧场景只用小圆片代表激光源和探测器，不能把它们称为已经建好的器件实体。当前增加激光器筒身、端环、红色出光口、支承，以及带边框的方形探测屏。探测面大小与配置的 `detector_window_m` 一致，出光口和感光面均对齐既有的器件局部光学平面。旧通用调镜架的遮挡外观隐藏，保留原来的碰撞形状、质量和惯量；这是兼容已验收夹持支架的示意模型，不是实物供应商 CAD，也不代表已重新验收整套实验。

新场景由 `simulation/optical_device_visuals.py` 生成这些纯视觉几何。旧 replay 在内存中的显示副本补充同一套模型，不改写原 `scene.usda`、轨迹、配置和光学记录。Isaac Stage 显示名标注各器件角色，原 ID 保留以兼容已录轨迹。

回放和 live 的视角按钮增加 Laser / Detector。近景相机随器件移动；例如直接检查装配后的探测屏：

```bash
bash simulation/isaac/run_isaac_michelson_replay.sh \
  --folder runs/isaac-michelson-final-20261103-006 \
  --start-seconds 150 --freeze --camera detector
```

将 `--camera detector` 改成 `--camera laser` 可看激光器。去掉 `--freeze` 和 `--start-seconds 150` 即可从头播放；运行中可以暂停并切换这些角度。

探测屏上的浅色空白面只用于辨认器件，显示材质提供中性补亮；没有把装饰纹理或 RTX 照明当成 LightPipes 探测结果。实测八相位探测器图像、可见度和调光曲线仍由独立光学监视器显示。

本次 105 项测试通过；两种近景实测约 60 FPS；模型与原验收源模型的质量、惯量和碰撞几何逐项保持一致，旧 replay 显示补丁的原生属性检查也通过。新生成的 Isaac 场景通过相机识别与 settle 门槛（3.234 仿真秒），不代替新全流程验收。截图和汇总保存在本机 `runs/isaac-device-visuals-20261112/`。


## 2026-11-12 恢复地面与视口网格

迁移的 MuJoCo `floor` 仍作为 `/World/Static/g0000` 下的物理地面存在，未删除。原地面材质的棋盘纹理没有被导入器转成 USD 纹理，只留下白色材质乘数；11 月 9 日的展示配置又关闭了视口网格，因此地面看起来是一大片白色。当前显示入口恢复网格，将既有地面显示为灰色，并在 Stage 中标为 Ground Plane。新生成场景也使用源地面颜色作为无纹理时的回退。

这是现有物理地面的显示恢复，没有额外叠加地面碰撞体，也没有加载另一个环境或改写旧验收 USD；并非重新导入 NVIDIA `default_environment.usd` 资产。重启 live/replay 后生效。
