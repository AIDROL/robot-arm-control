# MIT 开源代码对照审查

日期：2026-11-01。本次只审查代码与已有运行证据，没有修改控制器或执行实机脚本。

## 对照范围

- [OptSetGen](https://github.com/sachin4594/OptSetGen)：布局变换、相似变换/朝向、元件间距、机械臂占地与工作区的几何检查，不是机械臂/FAT 控制库，也不是 LightPipes 波动验证。
- [evaluation-results](https://github.com/AI-Driven-Robotics-for-Optics/evaluation-results)：模型生成布局的评测结果与提示词，不代表这些布局全部完成实机装配。
- [AutoArm/AUTO-Arm](https://github.com/AutoArm/AUTO-Arm)：本次主要参考的实机相机、抓放、FAT 和光斑处理脚本。也查看其上游 ShrishChou/AUTO-Arm 根目录；两者文件集不同，不能当成同一个版本。

结论：我们的流程骨架相近，但不是其算法/硬件流程的完整复现。

## 逐项对照

| 环节 | 读取到的 MIT 代码 | 本项目当前实现 | 是否一致 |
| --- | --- | --- | --- |
| 全局标定 | `fulladjustment.py:401` 读取 stereo_calibration2.npz 内的 mtx1/dist1/mtx2/dist2/R/T，并作尺度补偿 | global_camera.py:129 从 MuJoCo 相机 FOV/位姿生成准确 K/T | 否：使用理想标定，尚无实测标定流程 |
| 全局定位 | 两个相机各自 PnP、合并位置；具体脚本保留经验坐标转换 | global_camera.py:141 进行 PnP 检查，再用已知顶部高度射线求交、估计 yaw、双机融合 | 部分：同为粗定位，不是原版算法，也不是完整双目三角化 |
| 手眼/工具外参 | fulladjustment.py:390 使用约 72/37 mm 平移及经验坐标约定；本次读取文件未见通用手眼求解流程 | wrist_camera.py:18 的相机平移和 E_FROM_D 为配置的准确仿真变换 | 否：没有真实手眼、TCP 和标记到接口标定 |
| 腕部修正 | fine_adjust 根据 ArUco 像素误差反复移动机械臂，读取深度并调整转角 | 渲染深度拟合顶面，仅修正平移一次，保留全局 yaw | 部分：尚非迭代完整姿态视觉伺服 |
| 机械臂执行 | xArm SDK 的 set_position_aa、set_servo_angle、夹爪命令 | task_space_control.py 使用阻尼雅可比控制 MuJoCo 关节执行器 | 任务类似，后端不同；尚无实机执行适配 |
| 普通器件抓放 | 实机夹爪开闭和真实物体搬运 | 自由刚体、凸接触、摩擦，检查双侧承力、抬升、支撑后释放 | 仿真对应已接通，尺寸/质量/摩擦未实测 |
| 放置复查 | 原论文描述视觉复查、超过 0.5 mm 重新摆放；旧脚本不能单凭函数名视为已封装此协议 | 独立真值评价；机械 XY 门限仍 10 mm，0.5 mm 研究目标单独记录；完整入口不强制 precision_goal | 否：没有放置后的视觉重抓纠偏闭环 |
| FAT 取用与归还 | FAT.py 含工具 ID 抓取；FATfeedback.py 主程序含接近和电机命令，归还部分注释关闭 | FAT 使用 mocap，相对工具坐标随动并插值归还 | 否：普通器件的接触抓取不能证明 FAT 已物理抓牢 |
| FAT 接近 | calculate_approach_vector 使用经验补偿；FATfeedback.py:1189/1239 识别白色目标并修正接近 | fat_vision 识别侧面 ArUco；用配置外参估算接口，孔前停 1.5 mm | 部分：不是接口孔边缘/接合深度反馈，没有插入接触 |
| 电机与调节机构 | FATfeedback.py:1295 发送 move,motor,steps；FAT.py 有步进脉冲函数 | 内部镜架为固定几何，没有旋钮/镜面关节或步数到角度映射 | 缺失 |
| 光斑反馈 | beam_aligment.py:468/478 使用灰度最大值定位光点，并通过机械臂移动相机使光点居中 | LightPipes 输出探测面强度、通过率、质心，仅验收，不向控制器输出动作 | 缺失闭环；相机追光与镜面调光也须区分 |

标记参数也不同：旧 fulladjustment.py 使用 DICT_6X6_250、示例标记边长 62 mm；本项目为 DICT_4X4_50，顶标记 24 mm、侧标记 12 mm。它们在本项目内部自洽，但不能直接用于对方实物标记。

## 不能直接复制的内容

FAT.py:1115 主入口在电机测试后显式 `raise KeyError`，之后的演示代码不会直接执行。interferometer.py 的主程序保留固定坐标，FAT 操作多处注释关闭。FATfeedback.py 虽有接近反馈和双电机命令，不等于它已封装可直接运行的任意镜架光学优化器。

这些文件适合作为步骤和接口参考，不能复制其 IP、COM 端口、经验平移、像素换算和电机步数当成本设备的标定值。未在本次读取文件中找到某功能，不等于断言整个项目其他版本或未公开代码不存在该功能。

## 建议下一步

保留三件简单链路，先补“传感—动作—再测量”闭环。任务分为两类：

1. 三件空间滤波装配的位置失准：增加独立光学探测面/虚拟光斑相机，输出像素、质心、功率和有效性；结合放置后视觉复查，以有界微移/重抓修正位置。控制只用观测，实际状态单独评价，不让优化器直接读取目标位姿误差。
2. FAT 旋钮角度调节：先做单镜架验证单元。增加可驱动的 tip/tilt 机构、声明步数到旋钮/镜面角度映射、正确的光学转向模型，再接光斑反馈。当前三件透镜/针孔模型只处理薄透镜偏心和小角孔径投影，不能用来证明镜面转角调光。

FAT 同时需从 mocap 改为可接触抓取的自由刚体；分清“抓牢”“对准”“接合”“转动”四个状态，之后才可在完整链路宣称 FAT 光学精调。机构参数未知时允许使用显式暂定参数验证软件接口，但不得称为实物标定。

推荐分步验收：先让固定光学探测面看到的光斑随一次合法微移变化；随后用观测驱动动作改善功率；再加入有限重试和失败停止；最后迁移到真实 FAT 机构。无需扩展到复杂干涉仪或重新跑多种随机布局。

当前约 9.65% 的实际通过率属于既定参数下的失败证据。不能通过放宽验收、把光学中心直接设回设计值或增加理想六轴台来消除该失败。

## 本次主要来源

- [fulladjustment.py](https://github.com/AutoArm/AUTO-Arm/blob/main/fulladjustment.py#L357)，Git blob `ee2a5176852562db2eee7ba8f65ac4100d0b350e`。
- [FAT.py](https://github.com/AutoArm/AUTO-Arm/blob/main/FAT.py#L1043)，Git blob `62a19ac682a5bd7e6fe6fa085ebc42ca8f3730f9`。
- [FATfeedback.py](https://github.com/AutoArm/AUTO-Arm/blob/main/FATfeedback.py#L1189)，Git blob `94306e47bd21d5434d18d4e8c95ac5a72e6974c8`。
- [beam_aligment.py](https://github.com/AutoArm/AUTO-Arm/blob/main/beam_aligment.py#L468)，Git blob `2529a8efb48e4334cd3fb40df2acd131d0465e5e`。
- [interferometer.py](https://github.com/AutoArm/AUTO-Arm/blob/main/interferometer.py#L1141)，Git blob `9a12dd70b0c40ec39918ecd1de590a39d1f58d6b`。
- [OptSetGen 验证函数](https://github.com/sachin4594/OptSetGen/blob/main/MODULE_validation_functions.py)。
