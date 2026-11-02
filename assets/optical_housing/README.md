# 已选用的光学器件外壳

用户已选用 `highousing.stl` 作为当前外壳外观，原始网格不修改。原始尺寸约 72 × 64 × 108 mm；MuJoCo 按毫米转米（统一比例 0.001）导入。

- 来源：[ShrishChou/CAD_Engraver/highousing.stl](https://github.com/ShrishChou/CAD_Engraver/blob/adad25ca4a27f4480518f8484316d1ef3706c73f/highousing.stl)
- 文件对应提交：`adad25ca4a27f4480518f8484316d1ef3706c73f`
- SHA-256：`815981fe57a1e078a52855fc3ce756ff0e036d3fc6d6e695a603d51cf7c2ef96`
- 上游未声明许可证；本地检查资产不附加本项目许可证。

接口坐标使用未居中的 STL 原坐标，单位 mm：

| 接口 | 坐标 / 朝向 |
| --- | --- |
| 顶部 ArUco | ID 0，黑边 24；中心 (0, 0, 102.35)，法向 +Z |
| 平整面 ArUco | ID 10，黑边 12；中心 (0, -32.35, 88.1)，法向 -Y |
| 左夹持凹槽中心 | (-33, 0, 88.1)，法向 -X |
| 右夹持凹槽中心 | (33, 0, 88.1)，法向 +X |
| 夹持中心 | (0, 0, 88.1) |

±X 凹槽接触面比外侧 X=±36 内缩 3 mm，接触面间距 66 mm。前后 Y=±32 的上部面平整，侧面标记贴在其中的 -Y 面，不覆盖夹持凹槽。新增标签用独立非碰撞几何体表示，没有改写 STL。

`simulation.stl_preview` 默认加载本资产并添加标记及接口参考点；`--bare` 查看原网格。预览器只做外观和接口几何验证，未启用空心碰撞分块或验证夹爪抓取，预览器本身不执行抓放。

实际接触验证已另设入口 `./.venv/bin/python -m simulation.housing_grasp --view`，使用 `selected_housing.py` 的分块碰撞结构。单件夹住凹槽、抬起、搬运、松开并稳定支撑的任务已通过，见 [物理夹持验证](../../docs/housing-grasp.md)；视觉/FAT 完整实验现已接入该资产及其碰撞结构，见 [STL 集成记录](../../docs/stl-integration.md)。

完整实验最终将外壳绕世界 Z 转 +90°：局部通光轴 Y 沿世界 X 排列方向，局部夹持面 ±X 最终朝世界 ±Y，侧面标记/调节接口朝世界 +X。原 STL 本身和上述局部坐标不变。
