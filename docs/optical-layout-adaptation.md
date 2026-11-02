# 三器件布局、立柱和底座适配

日期：2026-10-31。

2026-11-01：已接入可选 [LightPipes 薄透镜/针孔验证](lightpipes-integration.md)。下文未实现光学传播的描述属于此前阶段；当前设计通过、实际位姿低透射失败的结果分别保留。

## 论文与仓库

《[AI-driven robotics for optics](https://pmc.ncbi.nlm.nih.gov/articles/PMC13488923/)》描述放置后重新视觉测量，误差超过 0.5 mm 时重新摆放；明确指出灾难性故障自主恢复尚未实现。装配演示包含四件器件的 Michelson（两镜、分束器、相机）和三件器件的 4f 清束（两透镜、针孔）。

另一篇《[A Framework for Closed-Loop Robotic Assembly, Alignment and Self-Recovery of Precision Optical Systems](https://arxiv.org/html/2603.21496v1#S5.SS3)》V-C 才展示激光腔器件移位、旋钮失调后的恢复，不能混为同一篇的结果。

MIT 旧版 [AUTO-Arm README](https://github.com/AutoArm/AUTO-Arm/blob/main/README.md) 用干涉仪描述装配精度验证；[interferometer.py](https://github.com/AutoArm/AUTO-Arm/blob/main/interferometer.py) 有相机、两镜、分束器的抓放，也含固定坐标及注释关闭的调节步骤，不是可以直接迁移的通用恢复器。

当前采用受 4f 演示启发的三器件直线装配，未导入 15 器件案例及其理想六轴台。

## 几何与标定

原 STL 72 × 64 × 108 mm、标记、夹持凹槽和内部孔位保持不变，壳体下新增 60 mm 支撑：

| 项目 | 暂定尺寸/参数 |
| --- | --- |
| 圆形底座 | 直径 54 mm，高 12 mm，质量 120 g |
| 立柱 | 直径 12.6 mm，高 48 mm，质量 25 g |
| 壳体质量 | 原假设 120 g |
| 光轴高于桌面 | 100 mm |
| 壳体原点高于桌面 | 114 mm |

尺寸与质量是可替换的仿真假设，不是论文测量值。立柱底座为自由刚体上的碰撞几何，随外壳通过夹爪接触搬运，无夹爪 weld 或物体位姿瞬移。磁吸力和底座锁紧未实现。

新配置为 `selected_stl_post_v2`；历史无支撑配置 `selected_stl_v1` 保留为 `STL_FRAME`，可用构场参数 `housing_model='stl-frame'` 生成。相机顶部高度、本体原点、抓取/放置高度、FAT 对孔、包围盒底部和检查点版本同步更新。旧无支撑检查点会被拒绝。

搬运底部净空从 160 mm 增为 220 mm，防止新增底座撞到已放置器件的顶部。发现碰撞的首轮记录保留在 `runs/three-post-chain-001`，没有放宽碰撞门限。

## 布局入口

配置：`configs/optical_setups/three_mounts.json`；适配器：`simulation/optical_layout.py`。

第一版只支持明确的台面坐标（m）、已标定世界偏移、三个已知库存、直线连接、100 mm 光轴高度和 +90° 外壳朝向。检查单位、方向含义、库存、高度、桌面外廓、夹持间距及名义径向工作区；完整运动可行性仍由执行和碰撞检查判断，不将半径筛查视为完整 IK 认证。

输入是光学中心。适配器依据本体到光学中心的外参计算本体和抓取位置，不能直接把布局坐标当 TCP。光学目标 X 为 -180、0、180 mm；本体 X 对应 -176、4、184 mm；本体世界 Z 为 874 mm，光学中心 Z 为 860 mm，夹持中心 Z 为 912.1 mm。

布局 SHA-256 和几何版本绑定场景及检查点。控制目标来自布局及相机观测；实际状态用于独立评价。无有效相机观测时停止，不以真值补齐。

```bash
./.venv/bin/python -m simulation.experiment --seed 7 \
  --layout configs/optical_setups/three_mounts.json \
  --output runs/three-post-demo --view
```

流程：三件视觉抓放 → 孔径几何检查 → FAT 取用 → 三处侧标记视觉对孔 → 归还。全局 A/B、腕部 RGB/深度、FAT RGB 继续各自独立原生窗口。

不加 `--layout` 时仍是五件流程，但默认同样带立柱底座；单件视觉入口也使用新配置。独立 `housing_grasp` 演示保留原裸壳体，不代表新支撑整体的验证。

## 本轮验证

`runs/three-post-chain-002` 从新场景一次连续完成全流程；`success=true`，仿真时长 174.952 s。没有恢复已完成状态、调用理想精调台或实施扰动后恢复。

| 器件 | 放置 XY 误差 mm | FAT 双端最大中心误差 mm（含 1.5 mm 名义孔前停距） |
| --- | ---: | ---: |
| lens_1 | 0.083 | 1.652 |
| pinhole | 0.349 | 1.387 |
| lens_2 | 0.148 | 1.294 |

42 项测试通过，涵盖坐标转换、无效布局拒绝、三件场景/布局绑定、自由接触支撑、搬运净空和原有视觉控制。结果属于单次集成验证，不是精度统计或硬件保证。

内部仍为孔径/接口示意，没有透镜焦距、针孔尺寸及传播模型，因此没有“4f 已清束”的结论。FAT 仍为 mocap 随动代理，未实现物理插孔、旋钮驱动或光学反馈。自动重抓恢复尚未新增，机械失败会停止。

下一阶段先定义这三件的光学参数、激光输入和探测面，将实际光学位姿送入计算，取得光斑/通过率反馈，再接视觉复查和有界重抓，不必先搭复杂干涉仪。
