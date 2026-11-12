# Michelson 可选同步视频导出

可选视频导出使用已通过独立审计的 `runs/isaac-michelson-final-20261103-006`。它包含从空实验区开始的五件装配、FAT 抓取与插入、接触调镜、退出归还、归还后两种网格的新光学测量。原始物理过程长 240.846 秒；默认导出为 2 倍速、30 FPS，结尾额外停留 3 秒，整段约 2 分 3 秒。需要在 Isaac 内自由调整视角时，使用本文末尾的三维 replay 入口；该入口的显示设置和相机切换说明见 [Isaac 相机与调光画面](isaac-camera-optics-view.md)。

## 一条命令播放

```bash
cd /home/terry/Desktop/workspace/robot-arm-control
bash simulation/isaac/run_michelson_presentation.sh
```

默认电影为 `runs/isaac-michelson-presentation-20261104/michelson-complete.mp4`。它也可以直接用 Ubuntu 视频播放器打开。播放只需项目 `.venv`，不启动 Isaac 或物理求解，因此播放帧率不受现场仿真计算速度影响。

成片已逐帧解码检查：3704 帧、1920×1080、30 FPS，长 123.47 秒；源文件哈希、首尾画面、相机标定和归还后最终光学测量均通过。最终播放器本机 6 秒检查显示 180 帧、零跳帧，约 30 FPS。电影旁的 `presentation-verification.json` 与 `playback-verification.json` 保存检查结果。

播放器按键：空格暂停/继续，左右方向键前后跳 5 秒，`q` 或 Esc 退出。`0` 显示完整多画面，`1` 放大主视角，`2` 全局相机，`3` 腕部 RGB-D，`4` FAT 侧相机，`5` 光学结果。选取单个面板时采用原有视频分辨率放大。

`A` 回装配开头，`F` 跳到 FAT 阶段，`O` 跳到光学反馈调节，`R` 跳到工具退出归还（使用小写按键即可）。Home/End 跳到视频开头/结尾；结尾按空格可以从头重新播放。章节位置由源实验事件和电影时间映射计算。

## 同一条时间轴上的内容

主画面按实际 PhysX 刚体轨迹连续插值播放。装配阶段是全景；调镜时平滑接近光学器件，归还时退回全景。全局、腕部 RGB-D、FAT 侧视从第 0 帧到结束都连续渲染，所有画面和文字使用同一个源时间，不再各自用墙钟计时。相机外参从源机械模型编译，并与已归档的 FAT 控制相机外参核对。

下方探测器显示该时刻最近一次已记录测量的第 0 相位原始 16 位图像，采用固定对数显示增益。可见度、像素波前倾角、测量时刻和已保持时间一并显示；曲线只显示已经发生的测量点，不提前使用未来数据，也不插值光学数值。旋钮读数来自同一时刻的实际关节记录。光路装配完成前，光学区域明确显示等待装配；工具归还后显示新测量及 512/1024 双网格检查。三维中的细线仅示意最近一次记录的主光线路径。

相机图像是从已记录实际姿态重新渲染的连续视图；它们与控制时按需保存的图片区分开来。光学图像和数值直接取自已验收运行，不是新跑出的实验结果。镜头遮挡和侧视相机在工具深插入后的暗部也保留，不替换成理想视角。

## 重新导出

导出使用项目虚拟环境中的独立 FFmpeg 编码器：

```bash
./.venv/bin/python -m pip install -r requirements-presentation.txt
bash simulation/isaac/export_michelson_presentation.sh \
  --folder runs/isaac-michelson-final-20261103-006 \
  --output runs/my-presentation/michelson.mp4 --speed 2 --fps 30
```

输出电影路径必须是新的。逐帧渲染速度会慢于播放速度；导出完成后电影以固定 30 FPS 播放。日志为 `runs/isaac-presentation-export.log`。电影旁的同名 JSON 保存源记录和电影的 SHA-256、起止时间、相机列表、实际渲染帧数和出现过的光学测量。`first-frame.png`、`last-frame.png`、`adjustment-preview.png` 方便检查画面。

需要自由转动三维视角时仍可使用原入口：

```bash
bash simulation/isaac/run_isaac_michelson_replay.sh \
  --folder runs/isaac-michelson-final-20261103-006 --speed 4
```

该 Isaac 入口用于交互三维展示，默认 4 倍速，完整流程约 60 秒；窗口支持 1 / 2 / 4 / 8 倍速、暂停/继续、从头播放、前后 10 秒，以及 Overview / Global / Wrist / FAT 相机按钮和自由视角。结束后场景保持打开，便于继续观察；需要自动关闭时可加 `--exit-after 3`。上面的电影播放器用于固定画面的可选导出展示，其默认速度仍为 2 倍。
