# xArm 7 实机 SDK 联调

控制箱 IPv4 为 `192.168.1.212`。本目录用于实机联调，SDK 使用独立的 Python 环境。首轮脚本读取固件版本、七轴角度、TCP 位姿和控制器状态。

## Ubuntu 有线直连

以下配置假设控制箱掩码为 `255.255.255.0`，且 `192.168.1.210` 在这条链路上可用。用网线连接 Ubuntu 的专用有线网口与已通电的控制箱。若 `ip -br link` 里没有有线网口，先接入 USB 转网口或检查内置网口是否被系统识别。

1. 打开 **设置 → 网络 → 有线 → 齿轮 → IPv4**，选择 **手动**。
2. 地址填 `192.168.1.210`，子网掩码填 `255.255.255.0`，网关和 DNS 留空。
3. 保存并重新启用该有线连接。在终端检查到控制箱的路由和连通性。

```bash
ip -4 route get 192.168.1.212
ping -c 3 -W 1 192.168.1.212
```

路由输出应显示 `dev` 指向接控制箱的有线网口，`src` 为 `192.168.1.210`。如果选中了 Wi-Fi 或 VPN，先排除网段重叠。若路由正确但 ping 无应答，可继续用下面的 SDK 脚本验证连接。

## 读取实机反馈

从仓库根目录运行。安装 SDK 时需要访问 Python 包索引。

```bash
python3 -m venv hardware/.venv
hardware/.venv/bin/python -m pip install -r hardware/requirements.txt
hardware/.venv/bin/python hardware/read_xarm.py
```

默认以约 2 Hz 采样 20 次状态。需要留档时：

```bash
mkdir -p hardware/logs
set -o pipefail
hardware/.venv/bin/python hardware/read_xarm.py 2>&1 | tee "hardware/logs/xarm-read-$(date +%Y%m%d-%H%M%S).log"
```

读通的判据是能得到固件版本、七个关节角、六个 TCP 位姿值和连续状态采样。若有非零错误或警告，记录并在 UFACTORY Studio 核对，再进入运动验收。

## 七轴和 Gripper G2 联测

实机脚本 [joint_micro_motion.py](joint_micro_motion.py) 在启动时读取当前七轴角度作为参考。请在 Studio 确认机械臂已经实际停在预定测试起点，已处于真实机器人位置控制模式（Mode 0），并用只读脚本确认 `state=2 error=0 warning=0`。保存 Initial Position 后，以现场实际姿态为准。开始前确认静止急停检查已完成，整条机械臂和末端工具在每一段往返中都有净空。

在文件开头编辑 `JOINT_DELTAS_DEG`。键 `1` 到 `7` 对应 J1 到 J7。正负号决定方向，单位是度，`0` 跳过。例如 `+5.0` 和 `-5.0` 分别表示正、负方向的 5°。实机批量执行前，按当前台架逐轴确认符号和幅度。关节步长上限、速度和加速度分别由 `MAX_DELTA_DEG`、`SPEED_DEG_S` 和 `ACCEL_DEG_S2` 控制。

从仓库根目录运行：

```bash
# 预览 J1–J7，读取当前位置并检查目标关节限位
hardware/.venv/bin/python hardware/joint_micro_motion.py

# 只执行 J7，换其他数字即可选择对应关节
hardware/.venv/bin/python hardware/joint_micro_motion.py --joints 7 --execute

# 一次运行逐个执行 J1–J7
hardware/.venv/bin/python hardware/joint_micro_motion.py --execute

# 预览七轴和 G2
hardware/.venv/bin/python hardware/joint_micro_motion.py --gripper

# 七轴和 G2 联测
hardware/.venv/bin/python hardware/joint_micro_motion.py --gripper --execute
```

每个关节先微动，再回到该轴本次微动前的实测角度。两个动作各自等待完成并核对七轴反馈，全部通过后才进入下一轴。两次回读都满足脚本容差后会打印 `J<n> PASS`。关节动作期间若有错误或中断，脚本中止批次并尝试发送手臂软件停止请求。现场有异常时按实验室停机流程操作，确认实机已停稳后再排查。

使用 `--gripper` 时，先在 Studio 确认 G2 已可控，夹爪空载，夹口有净空。脚本在预览阶段读一次 G2 开口，完成所有关节后重新读取开口并确认手臂就绪。夹爪按 `G2_STEP_MM` 张开，再回到起始开口，速度和力度分别由 `G2_SPEED_MM_S` 和 `G2_FORCE_PERCENT` 控制。两个方向都核对开口反馈，容差 ±1 mm，同时核对手臂保持原姿态。全部通过时会打印 `G2 PASS` 和总结果。

## 实机验收记录

xArm 7 固件 `v2.8.2`，Python SDK `1.18.4`。七轴单关节往返全部通过，日志中关节目标与读回的最大偏差约为 `0.062°`，低于 `0.10°` 容差。Gripper G2 空载从 `39 mm` 张开到目标 `54 mm` 再返回，分别读回 `53 mm` 和 `38 mm`，均在 `±1 mm` 容差内。笛卡尔运动、软件停止和物理急停待单独验收。

- [UFACTORY Gripper G2 控制手册](https://docs.accessories.ufactory.cc/xArm_Gripper_G2/3.Control.html)
- [UFACTORY 硬件安装手册](https://docs.xarm.ufactory.cc/zhHans/2.hardware_installation.html)
- [UFACTORY Python SDK](https://github.com/xArm-Developer/xArm-Python-SDK)
