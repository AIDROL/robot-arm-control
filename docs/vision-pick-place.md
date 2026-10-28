# 单镜架视觉定位抓放

本阶段将双 4K 全局 RGB 定位接入现有笛卡尔抓放控制器，只操作一个镜架。其余四个镜架仍留在场景中作为障碍物。FAT 保留在桌面上，本阶段不取用。

后续更新：默认流程已加入[末端 RGB-D 精定位](wrist-rgbd.md)。本文以下描述的是粗定位基线；运行时添加 `--coarse-only` 可复现其控制流程。

## 运行与证据

从项目根目录运行，需要可用的 macOS 图形上下文：

```bash
./.venv/bin/python -m simulation.vision_pick_place --seed 7 --output runs/visual-grasp-demo
```

输出目录重名时自动增加 `-001`、`-002` 等编号，保留旧结果。默认目标为 `lens_1`，可用 `--object pinhole` 等选择一个目标，但当前只验收过 `lens_1`。附加 `--record-gif` 可录制演示，需要 ffmpeg。

输出包括：两台相机原始 4K 图像和角点叠图、`observations.json`（每路内外参、时间、检测与择优结果）、`result.json`（所用视觉位姿、计划、状态时序、验收指标与失败码）、`trace.json`（每 0.1 秒 TCP/目标/夹爪指令）、抬升后/释放前/撤离前和最终 4K 关键帧，以及可选 GIF。

`transfer.png` 对应完成抬升、开始转运的时刻；`release.png` 对应完成下降、开始松爪的时刻；`retreat.png` 对应完成松爪、开始撤离的时刻。

负例验证：

```bash
./.venv/bin/python -m simulation.vision_pick_place --seed 7 --hide-target-marker --output runs/visual-grasp-hidden
```

该负例会隐藏目标顶部标记，预期退出码为 1，`failure_code=MARKER_NOT_DETECTED`、`control_started=false`、`simulation_seconds=0`，无机械臂运动。

## 控制数据来源

1. 从两台原生 3840 × 2160 RGB 图像检测顶部 ArUco，采用已知高度平面/直立姿态约束，保留原始 PnP 诊断结果。
2. `select_objects` 仅消费相机观测，检查两机一致性，计算标记到物体中心的固定偏移。该函数不接收仿真 model/data。
3. `require_pose` 拒绝检测失败、双机冲突、错误坐标系、超过 0.1 秒的过期观测、非有限/非法变换以及违背桌面几何先验的结果。
4. 给 `CartesianController` 显式传入 `{object_name: T_world_object}`。这一分支规划时仅读取机械臂自己的 TCP 位姿，不读取器件 `xpos/xmat`。其余对象不会被加入抓放计划。
5. 执行预抓取、下降、真实夹爪接触、抬升、转运、下降、张开、撤离和回位。使用关节/TCP 反馈与雅可比控制，不瞬移器件、不使用 mocap 跟随或焊接约束抓取器件。
6. 目标放置位置来自配置中的目标行坐标。仿真物体真值只供独立的接触/抬升/稳定性监测与验收使用，不回写控制目标。意外接触使运行停止。

视觉只在抓取前观测一次；此版本不进行连续视觉伺服，没有末端 RGB-D 精定位，也没有恢复重试。物体须直立放在已知台面上；移动后需要重新拍摄，不能把初始视觉位姿当作实时反馈。

## 验收

同时检查控制流程完成、两侧夹爪接触并抬离台面、转运时双侧接触比例 >80%、相对夹爪位移漂移 <10 mm、转运台面净空 >50 mm、最终 XY 误差 <10 mm、高度误差 <5 mm、yaw/tilt <5°、夹爪实际关节已打开且不再接触器件、松爪后静止至少 0.5 秒，以及无意外接触。

静止要求线速度 <5 mm/s、角速度 <0.05 rad/s。桌面检查使用旋转后长方体的最低点，允许 MuJoCo 软接触压入小于 1 mm。第一轮旧接触参数导致 2.32 mm 压入，已保留失败证据。视觉场景的桌面/物体接触参数调整为 `solref=[0.005,1]` 后重新通过，验收阈值保持不变；历史 FAT/抓放场景使用默认参数。

## 2026-10-28 实测

| 场景 | 结果 | 最终 XY 误差 | 转运相对漂移 | 连续稳定 | 最大接触压入 |
| --- | --- | --- | --- | --- | --- |
| seed 7，lens_1 | 通过 | 1.886 mm | 2.108 mm | 1.002 s | 0.838 mm |
| seed 0，lens_1 | 通过 | 0.672 mm | 1.979 mm | 1.002 s | 0.841 mm |
| seed 7，隐藏目标标记 | 按预期拒绝启动 | — | — | — | — |

证据位于 `runs/vision-pick-place-seed7-contact`、`runs/vision-pick-place-seed0` 和 `runs/vision-pick-place-hidden`。seed 7 的 GIF 是缩略延时演示，定位输入与关键帧仍为原生 4K。以上为两次理想仿真验证，不是多器件成功率或真实硬件精度证明。

`./.venv/bin/python -m unittest discover -s tests -v` 的七个测试通过，覆盖图像定位、空白图拒绝、错误平面拒绝、无器件真值字段时仍能构造视觉抓放计划、双机冲突拒绝、检测失败拒绝和消息有效性检查。

下一阶段在预抓取位接入末端 RGB-D，让近距离观测修正最终抓取点。
