# LightPipes 三器件验证（2026-11-01）

## 接入方式

使用资料包相同的 LightPipes 2.1.5（`Forvard` 自由传播、`Lens` 薄透镜、孔径强度掩膜），为当前三器件实现独立后端；没有搬入资料包固定 15 元件的 Michelson 验证器或理想六轴精调台。接口和模型说明可参见 [LightPipes 官方文档](https://opticspy.github.io/lightpipes/command-reference.html)。

`three_mounts.json` 保持机械布局和原 SHA 不变；光学参数独立保存在 `configs/optical_setups/three_mounts_optics.json`，绑定布局 SHA。机械坐标与光学参数分别版本化，旧机械快照可用于此次计算。

链路：布局/参数 → MuJoCo 视觉装配 → 保存积分状态 → 重建模型并从实际 optical_axis sites 导出位置/方向 → LightPipes 设计/实际对照 → 分别记录机械和光学结果。仿真真值仅用于光学状态审核，未回灌视觉抓放控制。当前不是光斑反馈驱动旋钮调节。

## 首版参数及适用范围

- 波长 632.8 nm、输入功率 1 mW、源面高斯腰半径 0.8 mm。
- 两透镜焦距均为 180 mm，通光半径 10 mm；针孔直径 150 µm。
- 固定入射光沿世界 +X，横向坐标为世界 Y/Z；光轴高 100 mm。源面 X=-200 mm，探测面 X=360 mm，均在台面坐标中描述。
- 元件实际 X 决定传播距离，实际 Y/Z 决定薄透镜和孔径偏心；轴方向影响孔径投影。仅支持距 X 轴不超过 1° 的小角度近轴情况，超出直接拒绝。忽略倾斜面的纵向相位修正，不是完整三维波动模型。
- 参数均为待团队确认的仿真假设，非厂商/实验室标定值。没有加入像差、材质色散、偏振、表面缺陷或磁吸/旋钮机构。
- 暂定光学门限为输入功率通过率 ≥80%、输出质心偏移 ≤0.5 mm；在计算前写入参数文件，没有按实际结果放宽。
- 6 mm 计算窗口，1024/1536/2048 方形采样网格；最后两档功率相对差需 ≤2%。输出完整各档数据，不能将该有限收敛检查等同于高精度光学认证。

输入是理想高斯光。本阶段证明聚焦、针孔透射及装配偏差的影响；尚未使用畸变/高阶模式输入验证“清束改善”，不能只因通过率高便宣称清束效果已验证。

## 安装与运行

```bash
./.venv/bin/python -m pip install -r requirements-optics.txt
./.venv/bin/python -m simulation.lightpipes_validation \
  --checkpoint runs/three-post-chain-002/assembly-state.npz \
  --output runs/optics-review
```

这会检查原布局/几何版本、恢复快照并刷新 FK，不运行机械臂，不使用原设计位姿冒充实际位姿。输出 `report.json`、`actual-layout.json`、参数副本、强度数组和对比图。输出目录不得已存在。

在完整实验中启用：

```bash
./.venv/bin/python -m simulation.experiment --seed 7 \
  --layout configs/optical_setups/three_mounts.json \
  --optics-spec configs/optical_setups/three_mounts_optics.json \
  --output runs/three-optics-demo --view
```

也可加 `--resume-assembly runs/three-post-chain-002/assembly-state.npz`，避免重复抓放。光学计算在装配后执行一次，随后仍运行 FAT 对孔/归还来验证现有机械链路；这不构成自动光学修正。要求光学验证时，总 success 必须同时满足机械和光学门限。失败会记录 `OPTICAL_ACCEPTANCE_FAILED`，不会改写实际位姿以获取通过结果。

## 当前证据

`runs/lightpipes-three-verified` 使用先前完成装配的真实积分状态，未重复机械实验。三档计算结果：

| 状态 | N=1024 通过率 | N=1536 | N=2048 | 最后两档相对功率差 | 光学门限 |
| --- | ---: | ---: | ---: | ---: | --- |
| 设计位姿 | 99.5575% | 99.5704% | 99.5844% | 0.0141% | 通过 |
| 实际装配 | 9.7345% | 9.5599% | 9.6515% | 0.9490% | 未通过 |

设计与实际都通过数值功率收敛门限；实际失败来自低通过率。另将设计针孔横向移动 1 mm，负例被正确拒绝。独立 Gaussian q/ABCD 预测与针孔数值透射相符；全部 45 项测试通过。

当前光学失败是有效结果：原机械误差虽在亚毫米范围，仍可明显降低小针孔透射。下一步应基于针孔面光斑/功率建立可观察的精调反馈，并补齐对应机构自由度，而不是调用理想六轴台把光学中心直接追到目标。

完整入口接线验证：`runs/three-optics-integrated` 从已完成装配快照恢复，光学计算后继续完成 FAT 三处对孔和归还；`mechanical_success=true`，`success=false`，`failure_code=OPTICAL_ACCEPTANCE_FAILED`。该退出码 1 是保留光学不通过的预期结果，不是程序崩溃。未重复抓放。
