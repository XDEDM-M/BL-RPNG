# HR-from-zero 对照：端到端成本与 HR-dev 结果

日期：2026-10-15。仅分析冻结开发集；模型 seed=0、support seed=271829、m=4，边界族为 high_frequency、amplitude_ood、localized_bump、piecewise_smooth。未增加种子，没有读取 query/test。

## 每个边界族

| task | 边界族 | path-source HR-dev | HR-from-zero 最佳 HR-dev | 最佳轮次 | HR-from-zero GPU-h | 误差比 | 到达 path-source 阈值 |
|---:|---|---:|---:|---:|---:|---:|---|
| 00 | high_frequency | 0.068318 | 0.640716 | 150 | 0.660 | 9.38× | 500 轮内未达到 |
| 01 | amplitude_ood | 0.044733 | 0.300194 | 159 | 0.611 | 6.71× | 500 轮内未达到 |
| 03 | localized_bump | 0.060216 | 0.485944 | 229 | 0.684 | 8.07× | 500 轮内未达到 |
| 05 | piecewise_smooth | 0.051360 | 0.341559 | 324 | 0.687 | 6.65× | 500 轮内未达到 |

四个任务的 path-source HR-dev 平均为 **0.056157**；HR-from-zero 的 per-task 最佳值平均为 **0.442103**（约 **7.87×** 更高）。HR-from-zero 用 HR-dev 每轮选最佳 checkpoint；path-source 用 LR validation 选 checkpoint，因此此处选择规则偏向直接 HR 基线。尽管有此优势，四项都未触及 path-source 的对应 HR-dev 误差；TTM 均为未达到/右删失，不能报告有限的等精度 speedup。

## 时间与标签生成成本

- path-source 源预训练：**2.068 GPU-h**；四项 target LR 微调合计 **0.487 GPU-h**。LR 支持数据的已有共享池覆盖 6 个族、96 个 cases，生成墙钟 **349.981 s**（本比较按整个共享池计入）。
- HR-from-zero 四项 500 轮训练合计 **2.642 GPU-h**。其 16 个 HR 支持 rollouts（4 族 × m=4）标签生成并打包需 **1187.548 s CPU 墙钟**；320/320 转移收敛，solver 时间和为 405.171 s。该 HR 标签生成时间已明确计入比较。
- 在单卡串行等效口径下，把 CPU 墙钟小时加到 GPU-h：path-source 为 **2.652 h**，HR-from-zero 为 **2.972 h**，成本比约 **1.12×**（path-source 低约 10.8%）。这是资源成本的条件性比较，不是观察到的总墙钟加速，也不是等精度 speedup；source HR 数据集的历史生成时间无可追溯记录，故无法纳入，单独标注为缺失。
- 两种路线最终都用同一 256×256 FNO 推理，公共推理耗时不计入差值。两卡并行时 task03 与 GPU1 上的不同任务重叠；逐任务 elapsed 相加用于 GPU-h，不代表 campaign wall time。

## 结论与限制

这次单模型 seed、单 support seed、四族 HR-dev 对照支持一个有限结论：在冻结的 500 epoch 与 m=4 条件下，path-source 路线以略低的 GPU 训练成本取得明显更低的 HR-dev 误差；HR-from-zero 没有在预算内到达匹配误差。它不能给出“达到同一精度快了几倍”的有限数字，也不能证明跨种子/边界总体优势。源 HR 训练数据的历史生成成本缺失；path-source 的 HR-dev 使用方式也比 scratch 更保守。

任务报告与 checkpoint 通过配置摘要、support/validation index 哈希、m=4 源索引、checkpoint SHA、完整 500 epoch history 和 path-source HR-dev 报告哈希核对。第一次汇总因 runner 比较字段名错误而中断；修正为报告真实字段 `config_sha256` 后仅重用现有报告完成聚合，没有重训。

机器可读明细：`artifacts/reports/phase1_hr_scratch_end_to_end_20261015.json`。canonical run summary：`results/phase1/HRSCRATCHDEV/hr_scratch_from_zero_hr256_m4_s0_ss271829/summary.json`。官方 ledger 完成后为 **45.989272 GPU-h**；120 GPU-h 总剩余 **74.010728 GPU-h**，其中 exploration 余量 **26.010728 GPU-h**、确认/控制 reserve 48 GPU-h 未动。
