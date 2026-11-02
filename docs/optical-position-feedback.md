# 第二项：功率引导的物理位置调整

当前支持三器件空间滤波场景中，针孔的 Y 向位置闭环；不包含 FAT 机构转动或任意六自由度光学优化。

## 控制与执行

1. 用全局相机重新检测针孔，验证新鲜、有效的视觉位姿；相对首帧视觉位置超过 1 mm 时停止。
2. 从实际仿真状态计算 LightPipes 场，生成 16 位探测像素及功率观测；仿真状态属于被控对象，不暴露给搜索策略。
3. PowerSearch 只读观测有效性、时间戳和通过率。若通过率达到参数文件的 80% 门限则接受；否则从 +Y 开始探测。功率下降时反向、步长减半。
4. 初始请求位移 0.2 mm，每步不超过 0.2 mm，最多 8 次、累计请求路程不超过 1 mm；小于 6.25 µm 停止。无信号、过期、无效视觉、物理失败、碰撞等会停止，不盲目继续。
5. 机械臂重新视觉抓取针孔。放置目标来自刚拍摄的腕部 RGB-D 本体位姿加有界位移；保留台面高度。相机估计误差与释放误差仍存在，请求位移不等于实际精确位移。
6. 通过夹爪摩擦接触抬起、移动、下降，检查桌面支撑后释放。重新视觉复查、重新计算光学输出，再决定下一步。

没有使用物体实际位置误差选择方向，没有直接改写物体自由关节位姿，没有理想平移台或夹爪焊接。仿真真值仅用于光学被控对象和独立机械验收。FAT 仍保留原 mocap 对孔代理。

实现：`simulation/optical_adjustment.py`；物理相对重放入口为 `vision_pick_place.run(relative_shift_m=...)`。只有 `--adjust-optics` 才启用调整，原验收入口行为保留。

## 运行

从已有装配快照执行第二项，并继续完整链路：

```bash
./.venv/bin/python -m simulation.experiment --seed 7 \
  --layout configs/optical_setups/three_mounts.json \
  --optics-spec configs/optical_setups/three_mounts_optics.json \
  --adjust-optics \
  --resume-assembly runs/three-post-chain-002/assembly-state.npz \
  --output runs/power-feedback-demo --view
```

不加 resume 则从新场景装配开始。各相机与光学探测面继续独立窗口；探测窗口在调整前启动，随每次新光学观测刷新。

只运行第二项可用 `python -m simulation.optical_adjustment --checkpoint ... --output ...`（无 GUI）。

## 验证

`runs/optical-adjustment-integrated` 从旧装配快照恢复（仿真 101.79 s），执行一次物理重抓，并继续多网格验收和 FAT 对孔归还，206.87 s 完成，整体 success=true。

- 像素观测通过率：9.6514% → 99.4948%，未改变 80% 门限。
- 请求 +Y 0.2 mm，实际动作由视觉与接触控制执行，不声称精确实现请求值。
- 独立多网格通过率：1024 为 99.5004%、1536 为 99.4892%、2048 为 99.4953%；最后两档相对差约 0.0062%。光学任务与观测有效性均通过。
- 每次观测保存状态快照、实际光学位姿、视觉观测、原始 16 位探测图和决策；每次物理动作保存独立抓放结果。`adjustment/detector` 只保留最新显示帧，逐步证据保存在 observation-* 中。
- 调整前 assembly-state.npz 保留；调整后另存 post-adjustment-state.npz，不覆盖原始装配证据。
- 55 项测试通过，包括功率下降反向减步、门限接受、过期/无效观测停止、次数/路程上限和真值字段不影响决策。

首轮 `runs/optical-adjustment-001` 使用 0.1 mm 初始步长，曾改善到 76.22%，随后下降并在无有效光信号时停止。因该步长与视觉重放/释放误差相当，改用仍受上限约束的 0.2 mm 初始探测；旧失败记录保留。`runs/optical-adjustment-002` 为单独闭环验证，一次重抓达到约 99.49%。这不是多随机布局成功率统计。

## 边界

这是第二项的首个具体闭环：针孔 Y 向平移。未实现 Z 高度调整、自动寻找激光、任意元件/布局恢复或真实硬件标定。低于观测有效性门限时停止，而不是靠真值恢复光束。第三项现已在独立的预装 Michelson 台架实现接触抓取、柔顺接合、旋钮驱动及镜架倾角闭环；见 [FAT 接触与 Michelson 验证](fat-contact-michelson.md)。本三器件装配入口的 FAT 阶段仍使用原代理，尚未自动替换。
