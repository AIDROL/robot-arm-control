# 空心外壳与内部光学镜架原型

**当前选型：使用原始 `highousing.stl`。** 用户已确认 ±X 两侧凹进去的区域用于夹持，ArUco 贴在平整的 -Y 面。资产和接口约定见 [已选外壳](../assets/optical_housing/README.md)。当前默认查看入口：

```bash
./.venv/bin/python -m simulation.stl_preview --view
```

该入口已添加顶部 ID 0 和平整面 ID 10 标记，保留原始 STL；下文的 `hollow_mount` 是此前的参数化方案原型，供结构参考，不是当前选定外壳。

独立预览入口：

```bash
./.venv/bin/python -m simulation.hollow_mount --view
```

默认装有一个镜面几何占位件；`--empty` 移除镜面以查看内部通光口。默认输出到 `runs/hollow-mount-preview/`，`--output` 可以另选目录。生成四个视角的 MuJoCo 图片、`housing.xml` 和 `geometry-check.json`。重复运行覆盖输出目录中的同名预览文件。macOS 窗口模式自动使用 mjpython。

## 结构依据与修正

参考 [CAD_Engraver](https://github.com/ShrishChou/CAD_Engraver) 的外壳结构，以及 [A Framework for Closed-Loop Robotic Assembly, Alignment and Self-Recovery of Precision Optical Systems，Fig. 1(c) 与 IV-A](https://arxiv.org/html/2603.21496v1#S4.SS1)。后者明确区分承载光学器件的定制外壳与内部 kinematic mount，FAT 接合的是标准镜架的调节旋钮。用户提供的模型图也说明了这一安装关系。

本地模型使用原创几何体和凸网格分块，没有复制上游 STL 或复现其精确 CAD。第一版误把调节接口做成外壳侧壁孔；当前版本已修正：侧面 ArUco 绕 Z 轴转 +90°，从 -X 面移到 -Y 开口上方；调节孔也面向 -Y，但安装在内部独立镜架的两个旋钮头上。外壳侧墙恢复完整，不在侧墙开 FAT 孔。

模型层级为 `mount`（外壳）→ `mount_kinematic_frame`（内部固定架及两只旋钮）→ `mount_optic_carrier`（镜片承载环和镜面）。目前这些部件固定连接，没有可动调节关节。

## 当前尺寸与坐标

单位 mm，原点在外壳中心，Z 向上，FAT 从 -Y 向 +Y 接近。

| 项目 | 当前值 |
| --- | --- |
| 外壳外尺寸 | 保留 50 × 50 × 80 |
| 壁厚 | 暂定 5 |
| 前后外壳开口 | 40 × 51；上边抬高以露出内部镜架和接口 |
| 顶部 ArUco | ID 0，黑边 24；中心 (0, 0, 40.8)，保留原位置 |
| 开口上方 ArUco | ID 10，黑边 12；中心 (0, -25.8, 24)，法向 (0, -1, 0) |
| 下方 FAT 接口 | (12, -18, -22) |
| 上方 FAT 接口 | (-12, -18, 2) |
| 接口间距和高度 | 保留原 24 × 24 的对角分布与 Z 高度；绕 Z 转 90° |
| 接口深度 | 改为距外壳前表面向内 7，不沿用旧侧墙外凸平面 |
| 旋钮头外半径 | 保留 4.5 |
| FAT 接合孔 | 暂定半径 2.3、深度 4，有底的盲孔 |
| 内部镜架 | 暂定 38 × 38 的方框，中央孔径 23 |
| 镜面占位件 | 暂定直径 22，不代表具体采购型号 |
| 光轴距外壳底面 | 暂定 30，不是已确认的实验光轴高度 |

开口上方横梁现在从局部 Z=16 开始；旧完整实验的 Z=+10 抓取点不能直接沿用到该面。后续接入任务时需重新确认夹爪接触区域、抓取高度、镜架到标记的外参和 FAT 接近方向。当前完整实验仍使用原代理，本次只修改独立原型。

## 验证与限制

外壳、内部镜架和接合孔采用凸块组合，空隙不会被整体凸包封住。检查区分外壳通光、内部镜架通光与镜面遮挡：49 条采样射线穿过外壳；不计镜面时中央轴线能穿过内部镜架；装上镜面后轴线会碰到镜面几何体。这里的射线检查仅验证几何，没有计算反射。

`tests/test_hollow_mount.py` 用自由体球形探针检查 FAT 可以从开口接近两个孔、偏离孔心时撞到孔壁、过度插入时撞到盲孔底部；同时检查标记法向旋转、接口归属于内部镜架、空镜架通光和完整侧墙遮挡。CLI 报告也检查 JSON 可序列化。

预览器会显示几何组 0–4；直接用其他工具加载导出的 XML 时，需打开 group 3（内部镜架）和 group 4（镜面），MuJoCo 默认可能隐藏它们。标记是实际渲染的字典编码，不是图片后期叠加。

尚未实现镜架调节运动、螺纹、FAT 驱动或光学传播。材料密度、所有新增机械尺寸与相机标定关系仍需按真实器件确认。

## 直接导入原始 STL 的对照预览

`simulation/stl_preview.py` 可将原始二进制 STL 直接交给 MuJoCo 导入，按生成脚本的毫米约定统一乘 0.001，仅做居中、落到台面的平移，保留所有网格顶点和开窗。原始 `highousing.stl` 已保存在 `assets/optical_housing/`；上游版本与校验见该目录 README。重新获取可使用 [原始模型链接](https://github.com/ShrishChou/CAD_Engraver/blob/main/highousing.stl)。

```bash
./.venv/bin/python -m simulation.stl_preview --view
# 或指定另一个本地原始 STL：
./.venv/bin/python -m simulation.stl_preview --stl /absolute/path/to/housing.stl --bare --view
```

默认输出 `runs/selected-housing-preview/`，包括四个视角、可重载的 `housing.xml`、原 STL 的逐字节副本和带 SHA-256 的导入记录。当前检查件大小 72 × 64 × 108 mm，454 个顶点、952 个三角面，导出场景已重新加载验证。默认图片展示原始外壳与新增的两个标记，没有添加内部镜架。使用 `--bare` 可查看不带标记的原网格。为避免把整体凸包误当成开孔碰撞结构，此预览关闭了外壳碰撞；不能直接当作抓放任务模型使用。
