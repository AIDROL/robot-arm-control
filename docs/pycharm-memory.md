# PyCharm 内存告警诊断（2026-10-28）

本机 16 GiB RAM。PyCharm CE 2024.3 当前自定义 JVM 堆上限原为 `-Xmx2048m`。近期日志尾部存在 1236 条 `LowMemoryWatcher` 记录，其中多条明确为垃圾回收后的低内存信号。这支持 IDE 堆空间不足的诊断，但尚不能唯一确定每次卡顿或崩溃的根因。

仿真产物 `runs/` 当前约 208 MB。生成输出目录应排除索引，避免每次保存图像、深度和 GIF 后触发不必要的项目分析；这是一项减负措施，尚无内存快照证明它是主要根因。[JetBrains 索引说明](https://www.jetbrains.com/help/pycharm/project-analysis.html)

已在项目 `.idea/AI driven robot arm.iml` 中明确内容根为项目根目录，并排除 `.venv/`、`runs/`、`simulation/__pycache__/` 与 `tests/__pycache__/`。代码与仿真结果均未删除。

用户明确批准后，已将 `~/Library/Application Support/JetBrains/PyCharmCE2024.3/pycharm.vmoptions` 的堆上限改为 `-Xmx3072m`。原配置备份在同目录 `pycharm.vmoptions.codex-backup-20261028`。没有自动重启 IDE：保存工作后重新打开 PyCharm 才生效。16 GiB 机器先用 3 GiB 观察，不直接给 IDE 分配 8 GiB。堆上限是 IDE JVM 的配置，并不等于 Python/MuJoCo 进程的内存限额。[JetBrains 堆设置说明](https://www.jetbrains.com/help/pycharm/increasing-memory-heap.html)

先用 macOS Terminal、从项目根目录运行 `./.venv/bin/python -m simulation.vision_pick_place --seed 7 --output runs/wrist-terminal-demo --view`。暂不同时录制 GIF，一次只运行一个实例；仿真结束后关闭保留的窗口再开始下一次。避免在 Python Console 中反复导入执行带图形上下文的脚本。终端运行可以把 IDE 与仿真进程分开，但不意味着绕过模型所需的系统内存。

若重启后仍出现告警，保留新的时间点、IDE 日志和内存快照继续定位；若终端进程也退出，则需单独检查 Python/MuJoCo 的错误输出和 macOS 崩溃记录，不能一概归因于 IDE 堆。此前的进程快照中未发现运行中的本项目仿真进程，未结束任何其他进程。
