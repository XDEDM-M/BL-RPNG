# BL-RPNG

Boundary-Lifted Residual-Progress Newton Guidance（BL-RPNG，也称
Boundary-Lifted Newton Path Fine-Tuning）是一个独立研究项目，用于验证
“高分辨率路径预训练 + 低分辨率
路径微调 + 高分辨率零样本推理”。

## 独立性与环境

本项目不 import 原 `FNOtest` 的 `models/`、`scripts/`、`train_sufno.py` 或
`rollout_utils.py`，也不使用软/硬链接。生成数据、模型和求解器临时文件写入
`artifacts/`，逐任务训练输出写入本地 `results/`；项目路径守卫拒绝写出项目根目录。
旧式嵌套checkout仍会审计外层 worktree；独立checkout不依赖任何父级Git仓库。

不修改 Conda 环境。训练父进程使用 `fnotest`，求解 worker 由父进程调用已有
`sfepy_env`，只输出临时 `.npz + JSON`，再由 `fnotest` Python 转换为 `.pt`。

```bash
source /opt/miniconda3/bin/activate fnotest
cd /gemini/code/BL-RPNG

python -m blrpng.cli.environment_check
python -m blrpng.cli.generate --config configs/smoke.yaml
python -m blrpng.cli.verify_data --config configs/smoke.yaml
python -m blrpng.cli.train --config configs/smoke.yaml
python -m blrpng.cli.evaluate --config configs/smoke.yaml
python -m pytest tests
```

共享 `fnotest` 环境没有第三方 `pytest`，项目内的 `pytest.py` 是只依赖标准库
`unittest` 的兼容入口，因此上述命令无需安装包，也不会改变环境。

## 当前实验快照（2026-10-16）

Phase-1 已完成多边界路径预训练、未见边界 LR 适配、256² HR-dev 推理的
三种子/六边界开发矩阵、目标路径与 endpoint 对照、HR 专家比较，以及普通
迁移基线的学习率搜索和 seed1 复现。当前最强的正向信号是预训练迁移显著优于
低分辨率 scratch；路径源候选在开发矩阵中排名第一，但相对充分调优的普通迁移
基线优势仍未确认。完整结果、实验过程、结论边界和预算状态见
[`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md)、
[`reports/PHASE1_FINAL.md`](reports/PHASE1_FINAL.md)、
[`docs/LOG.md`](docs/LOG.md) 和 [`docs/STATE.md`](docs/STATE.md)。

早期冻结的 Allen–Cahn seed-42 query 结果保留为独立历史评测；它未用于本轮
基线调优。当前新确认 query 仍关闭。Git 版本控制保留源码、配置、脚本、实验卡、
追加式 ledger、结论报告及精选汇总 JSON；逐任务训练历史、原始 PDE 数据、模型
参数和优化器状态保留在本地 `results/` 与 `artifacts/`。详细策略见
[`docs/REPOSITORY_SCOPE.md`](docs/REPOSITORY_SCOPE.md)。

## 方法

对静态非齐次 Dirichlet 数据 \(g\)，先在每个离散网格上求离散调和延拓
\(E_hg\)，再令

\[
v_h=u_h-E_hg,\qquad
\mathcal G_\theta:[v_t,E_hg,x,y]\mapsto \widehat v_{t+1}.
\]

输出在边界上投影为零，并重构
\(\widehat u_{t+1}=\widehat v_{t+1}+E_hg\)。因此模型对每个分辨率都精确满足
Dirichlet 边界，而不是用有限权重的边界 penalty 近似满足。

SfePy worker 捕获经过 line search 接受的 Newton 状态。若残差为 \(r_k\)，使用

\[
p_k=\frac{\log r_0-\log r_k}{\log r_0-\log r_*}
\]

定义尺度无关的求解进度，选择首个跨过 \(p=0.25\) 的 accepted state
\(u^{(k)}\)。`layer1:hook0` 预测从初始 Newton state 到该状态的累计路径位移
\(\delta_{0\to k}=u^{(k)}-u^{(0)}\)，损失为

\[
1-\cos_h(\widehat\delta,\delta)
+0.1\,\mathrm{SmoothL1}\!\left[
\log\frac{\|\widehat\delta\|_h+\epsilon}{\|\delta\|_h+\epsilon}
\right].
\]

每个 batch 分别计算最终损失和路径损失在 path-active 参数上的梯度范数，
动态选择路径权重，使路径梯度贡献/最终梯度贡献为 `0.15`。这避免了固定
lambda 随网格、方程尺度和训练阶段漂移。双阶段消融另保存 \(p=0.75\)
cumulative correction，在 `layer3:hook0` 监督。

理论上，lifting 把不同边界的仿射解空间 \(E_hg+H^1_0\) 对齐到共同切空间
\(H^1_0\)；每个局部 Newton correction 近似
\(-J(u)^{-1}R(u)\)，累计 target 是 accepted corrections 沿求解路径的和。
累计定义很关键：probe 只以时间步输入为条件，无法预测依赖未观测中间 iterate
的单个局部 correction；从初始 state 出发的路径位移则是条件良定义的。因此
路径目标不是任意 hidden-feature matching，而是给中间表示注入逆算子求解路径
几何。跨分辨率假设可被清楚拆为两部分：调和 lifting
的一致性，以及 FNO 对齐后的零迹残差算子的网格迁移。

## BL-RPNG v2：低保真目标边界适配

v2 将单场接口推广为 `C` 个场：lifted 输入固定为
`[u-Eg, Eg, grid_x, grid_y]`（`2C+2` 通道），每个场可使用独立 Dirichlet
mask；无 Dirichlet 的压力场使用零均值 gauge projection。除 progress-0.25
路径监督外，v2 加入 64→128 的 restriction-commuting loss、仅在 final/path
梯度余弦为正时开启的 conflict gate、随 epoch 退火的路径梯度比，以及不读取
HR solution 的 fine-grid PDE defect。LoRA 和 neuron-linear transformation (NLT)
适配均以相对 source checkpoint 的 delta 保存。

截至2026-08-31，主候选已收束为 **BL-RPNG-NLT**：Newton path 用于源预训练，
目标任务默认采用 NLT final-only 少样本适配。修正后的目标 NLT+path 已能产生
非零共享梯度和有效 gate，但新盲测上未稳定超过 final-only，因此不升级主方法。方法定义见
`METHOD.md`，当前可引用的 seed42 阶段结论见
`docs/EXPERIMENT_STATUS.md` 和
`docs/experiment_records/allen_cahn_current_state_seed42.json`。
严格区分代数事实、建模假设和可证伪机制预测的理论契约见 `THEORY.md`。

2026-08-31 的全新六边界族盲测含48个 target tasks 和768个成对 query
rollouts。在256²、20-step、m=4上，BL-RPNG-NLT 平均 relL2 为 **5.223%**；
相对同 LR support 的 lifted scratch 降低 **90.50%**（95% CI
[89.86%, 91.08%]），相对 final-only source pretrain + NLT 降低 **13.72%**
（95% CI [11.88%, 15.53%]），两组均在48/48 tasks胜出。LR setup 与
resource-capped m256 HR specialist setup 的观测加速为1,350×（95% CI
[1,288×, 1,437×]）。该 HR specialist 未达数据/优化饱和，因此这些结果
不支持 saturated-oracle parity 主张；其余 seeds 0/1/2/3 仍待确证。

冻结该阶段的命令为：

```bash
python -m blrpng.cli.summarize_reconfirmation_query \
  --budget 4 --resolution 256 --horizon 20 --replicates 50000
python -m blrpng.cli.summarize_reconfirmation_data_efficiency_query \
  --resolution 256 --replicates 50000
python -m blrpng.cli.summarize_reconfirmation_cost --resolution 256
python -m blrpng.cli.visualize_reconfirmation --resolution 256
CUDA_VISIBLE_DEVICES=0 python -m blrpng.cli.visualize_reconfirmation_fields \
  --resolution 256
python -m blrpng.cli.freeze_reconfirmation_stage
```

在进入 Poisson 负对照前，从冻结 seed42 报告和预先选定的六个代表
rollout 生成只含 PNG 的论文级图组：

```bash
CUDA_VISIBLE_DEVICES=0 python -m blrpng.cli.visualize_reconfirmation_paper
```

该命令生成总体表现、48任务成对改善、768-rollout ECDF、物理/频谱诊断、
accuracy–cost、六类边界 profile/lift、三方法物理场对比、50步误差增长，
以及每类边界的 step 0/1/5/10/20/35/50 时间演化图。所有代表样本选择在
本命令之前已冻结；作图推理不会重选 checkpoint。图组索引见 `PAPER_FIGURES.md`。

维护与多物理场 smoke 命令：

```bash
python -m blrpng.cli.consolidate_results
python -m blrpng.cli.cleanup_artifacts                 # dry-run only
python -m blrpng.cli.generate_poisson --resolution 32 --cases 2
python -m blrpng.cli.generate_gray_scott_v4 --resolution 16 --steps 1
python -m blrpng.cli.generate_hyperelastic --resolution 9
python -m blrpng.cli.generate_navier_stokes --resolution 9 --steps 1
python -m blrpng.cli.visualize_nlt_causal
python -m blrpng.cli.freeze_reconfirmation_manifest \
  --config configs/allen_cahn_reconfirmation.yaml
python -m blrpng.cli.audit_stage
```

Gray--Scott 的全新 128² 确认集保留严格论文门控；若主方法达到预注册的
点估计开发门槛、但 75% 改善的 bootstrap CI 下界尚未通过，只允许用以下
独立 amendment 继续预先冻结的 256² 跨分辨率检查：

```bash
python -m blrpng.cli.freeze_gray_scott_256_continuation
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m blrpng.cli.generate_gray_scott_confirmation --resolution 256 --workers 2
python -m blrpng.cli.verify_gray_scott_confirmation --resolution 256
CUDA_VISIBLE_DEVICES=0 python -m blrpng.cli.evaluate_gray_scott_confirmation --resolution 256
python -m blrpng.cli.summarize_gray_scott_confirmation --resolution 256
```

该 continuation gate 不得解释为论文门控通过，也不授权多随机种子或下一 PDE。

正式 Gray--Scott、超弹性和 Navier--Stokes 数据分别受前一阶段的新门控
manifest约束，不再接受旧的未饱和Allen--Cahn oracle gate。
新盲测 manifest 只冻结边界描述符与 support/query 随机流；在饱和 oracle、方法配置
和 checkpoint 规则冻结前，`query_opened` 保持为 `false`，不会生成或读取新 HR query。

HR specialist 的首轮收敛审计使用：

```bash
python -m blrpng.cli.run_oracle_convergence \
  --config configs/allen_cahn_oracle_convergence.yaml \
  --tasks 0 --budget 64 --learning-rates 0.0003 \
  --initializations scratch,ordinary,path --epochs 100 --max-parallel 2
python -m blrpng.cli.summarize_oracle_convergence
python -m blrpng.cli.summarize_extended_oracle
```

若 m64 未达到数据饱和，以下命令为代表任务新增独立冻结的 m128/m256 support；
它不会覆盖原 m64 manifest 或 shard，并在生成前执行 300 GiB 容量门控：

```bash
python -m blrpng.cli.extend_oracle_support --preflight-only
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m blrpng.cli.extend_oracle_support --workers 12
```

训练过程每个 epoch 原子更新
`artifacts/reports/progress/<checkpoint-stem>.json`。所有 JSON 写入采用同目录临时
文件加原子替换，并行命令还会在 `artifacts/reports/workspace_audits/` 保留各自的
唯一审计记录，环境中断不会再破坏已有进度文件。

截至 2026-08-25 的恢复性探索结果：task 0、m64、100 epoch、lr=3e-4 时，
path-pretrained HR specialist 的冻结 validation/query-256-20 relL2 分别为
1.306%/2.300%，ordinary-pretrained 为 1.479%/2.451%，scratch 为
3.482%/6.915%。该结果证明原 HR 对照确实存在优化不足，并显示 path
初始化也改善 HR specialist；但当前 task 0 NLT query 仍约为 4.18%，所以
`saturated_hr_oracle` 与 `oracle parity` 门控仍未通过，不能用于等价精度主张。

只有代表任务的优化曲线平台化、并且至少三个数据预算上连续两次翻倍的改善均低于
1%，报告才允许写出 `saturated_hr_oracle=true`。
`summarize_extended_oracle` 只比较相同 path 初始化在同一冻结 query 初值上的
m64/m128/m256 结果，并将 validation 指标单独保存，避免跨集合比较。

## SfePy Taylor--Hood Navier--Stokes 阶段

新的正式 NS 管线与旧 MAC smoke 完全隔离。它使用独立 SfePy 环境中的
P2/P1 Taylor--Hood、BDF1、完整隐式对流与 line-search Newton；父进程只接收
`.npz + JSON` 并转换为 v4 原子 shard。速度使用离散 Stokes lift，压力保存为
零均值 gauge；输入为三场状态、网格和 `[viscosity, dt, force_x, force_y]`，
lifted FNO 内部共12通道。

```bash
python -m blrpng.cli.ns_prescreen \
  --config configs/navier_stokes_sfepy_seed42.yaml --stage dt64
python -m blrpng.cli.ns_visualize
python -m blrpng.cli.ns_generate --stage development_train --preflight-only
python -m blrpng.cli.ns_generate --stage development_train --workers 1
CUDA_VISIBLE_DEVICES=0 python -m blrpng.cli.ns_train_source \
  --objective ordinary --architecture small --development
CUDA_VISIBLE_DEVICES=1 python -m blrpng.cli.ns_train_source \
  --objective newton --architecture small --development
python -m blrpng.cli.ns_adapt --task 0 --budget 4 \
  --source-checkpoint artifacts/checkpoints/navier_stokes_sfepy/<source>.pt
```

所有生成与训练命令首先读取
`artifacts/manifests/navier_stokes_sfepy_solver_gate.json`。截至2026-09-19，
预注册的 Re={100,400,800}、dt={0.005,0.01,0.02,0.04} 预筛没有找到合格
设置：时间精度、非线性残差、边界和 FEM 弱连续性通过，但 progress-0.25
Newton target 相对 endpoint increment 的聚合中位差仅约0.000025%–0.30%，
远低于5%门槛。因此 gate 保持关闭，大规模128²/256²数据和神经网络训练不会
被误启动。冻结报告与PNG位于
`artifacts/reports/navier_stokes_sfepy_seed42/` 和
`artifacts/figures/navier_stokes_sfepy_seed42/`。这一结论表示当前时间精确的
cavity离散不适合作为“非冗余Newton路径”机制证据，并不否定其作为 lifting/NLT
边界适配任务的价值。

### 瞬态 NS field-wise Newton-information gate（2026-09-21）

上述关闭结论继续适用于旧的 `progress=0.25` 标签；该规则在本问题中会跳过
第一次 accepted update，选到几乎等于 endpoint 的下一状态。后续门控没有改变
Newton 初值：每一步仍从上一物理时刻状态自然 warm-start。新门控记录整条
accepted path，并分别统计速度和零均值压力的 endpoint discrepancy、cosine、
正交分量、路径长度及 residual progress。

当前通过的**有限候选区间**是 BDF1、`dt=0.07`、`Re≈800`、grad-div=100，
配合幅值约0.05且作为条件输入的散度自由体力。64²和128²的8步 `dt/2`
时间精度与压力路径门控均通过；速度路径仍高度冗余，信息主要来自压力—连续性
耦合。静态体力的20步长时检查中，两种边界分别有50%和35%的时间步满足
信息门槛，故正式训练必须使用 `path_information_mask`，并将完全相同的 mask
用于 matched-endpoint 对照。64²目标适配保持 final-only，不使用低分辨率压力
路径。

```bash
python -m blrpng.cli.ns_transient_information_gate \
  --dts 0.07 --reynolds 800 --tasks 0 6 --initials 2 \
  --steps 8 --resolution 64 --forcing-amplitude 0.05 \
  --namespace bdf1_dt007_candidate64
python -m blrpng.cli.ns_freeze_transient_gate
```

综合证据、限制及下一阶段协议冻结于
`artifacts/reports/navier_stokes_sfepy_transient_gate/frozen_gate_and_protocol.json`，
候选配置为 `configs/navier_stokes_transient_candidate.yaml`。这只表示“20步、
mask-aware 数值候选通过”；尚未进行256²瞬态基准和神经网络 causal matrix，
不得表述为 BL-RPNG 已在瞬态 NS 上取得模型性能提升。

### 稳态 Reynolds 延续 NS 数值试验集

为获得真正非冗余的 Newton 路径，另建与上述瞬态数据隔离的稳态
NS 参数延续试验集。轨迹轴是 `Re=50→100→200→400`，**不是物理时间**；
主版本每一步直接从前一 Reynolds 稳态解 warm-start；首次状态为 Stokes lift。
另保留 `Stokes lift + 0.5 × (前一稳态解 − Stokes lift)` 作为预测器敏感性消融。
求解使用
SfePy P2/P1 Taylor–Hood、grad-div 稳定化和 line-search Newton。
路径标签是第一次被接受的 Newton 更新；匹配的普通辅助标签是同一个
预测状态到收敛解的增量 `predictor_endpoint_increment`。两者必须使用相同
预测状态比较，不可改用相邻 Reynolds 状态差。

```bash
source /opt/miniconda3/bin/activate fnotest
cd /gemini/code/BL-RPNG
python -m blrpng.cli.ns_generate_steady_pilot --tasks 0 6 --resolutions 64 128 256
python -m blrpng.cli.ns_freeze_steady_pilot
python -m blrpng.cli.ns_steady_predictor_sensitivity
python -m pytest tests/test_navier_stokes_sfepy.py
```

生成器仅启动一个 SfePy worker，限制 BLAS 线程、单进程地址空间至 8 GiB，
单 case 超时 180 秒，并可复用已通过校验的 shard。正式数据、索引和门控报告
分别位于 `artifacts/data/navier_stokes_sfepy_steady_projected/`、
`artifacts/manifests/navier_stokes_sfepy_steady_natural_*.index.json` 和
`artifacts/reports/navier_stokes_sfepy_steady_pilot/natural_warmstart_gate.json`。
混合预测器版本的门控另存为同目录下的 `numerical_gate.json`。
输出网格上的速度经保边界的最小改动散度投影；报告同时保留投影前 FEM
积分点散度及投影幅度，避免把网格后处理误称为原始 FEM 精度。

截至 2026-09-19，此集只有两个边界、各一个初始状态、三个分辨率，属于
**数值可行性试验集**。两边界的 128²/256² 解均满足数值门控，且 Newton
混合预测器路径相对匹配终点增量的中位差超过 18%；自然 warm-start 的
两个边界在 256² 的中位差分别为 11.36% 与 11.62%，Re=400 时为
32.02% 与 33.58%。但尚未训练 endpoint/Newton
神经网络对照，不能由此声称 BL-RPNG 在 NS 上已有性能收益。扩展 A/B
数据前，应先用该集验证路径标签和模型接口，再冻结任务划分及训练协议。

若明确在 m256 停止扩展，task 0/2/7 的最终 query 全部完成后运行：

```bash
python -m blrpng.cli.freeze_resource_capped_oracle
python -m blrpng.cli.freeze_method_gate \
  --oracle-gate artifacts/manifests/allen_cahn_oracle_resource_capped_gate.json
python -m blrpng.cli.prepare_reconfirmation_split --role support
python -m blrpng.cli.prepare_reconfirmation_split --role validation
python -m blrpng.cli.generate_reconfirmation --role support --resolution 64 --preflight-only
python -m blrpng.cli.generate_reconfirmation --role support --resolution 64 --workers 16
python -m blrpng.cli.generate_reconfirmation --role validation --resolution 64 --workers 16
python -m blrpng.cli.verify_reconfirmation
CUDA_VISIBLE_DEVICES=0 python -m blrpng.cli.train_reconfirmation \
  --task 0 --budget 4 --method nlt --seed 42
python -m blrpng.cli.summarize_reconfirmation_validation
python -m blrpng.cli.freeze_reconfirmation_query_protocol
python -m blrpng.cli.summarize_reconfirmation_efficiency
python -m blrpng.cli.freeze_reconfirmation_scratch
python -m blrpng.cli.generate_reconfirmation --role query --resolution 128 --workers 8
python -m blrpng.cli.generate_reconfirmation --role query --resolution 256 --workers 8
python -m blrpng.cli.verify_reconfirmation_query --resolution 128
python -m blrpng.cli.verify_reconfirmation_query --resolution 256
CUDA_VISIBLE_DEVICES=0 python -m blrpng.cli.evaluate_reconfirmation \
  --task 0 --budget 4 --method nlt --resolution 256 --horizon 20
python -m blrpng.cli.summarize_reconfirmation_query \
  --budget 4 --resolution 256 --horizon 20 --replicates 50000
```

该路径只授权进入新盲测，始终保存 `saturated_hr_oracle=false`；报告必须称其为
`resource-capped m256 HR reference`，不得用于饱和 oracle 或5%等价精度主张。
通过后才可冻结方法并准备 split manifest：

```bash
python -m blrpng.cli.freeze_method_gate
python -m blrpng.cli.prepare_reconfirmation_split --role support
python -m blrpng.cli.prepare_reconfirmation_split --role validation
# query/query_512 还会额外检查方法冻结 gate；命令只准备 ID，不生成解。
```

Allen–Cahn A1 的冻结 seed42 流程为：

```bash
python -m blrpng.cli.generate_calibration --config configs/allen_cahn_v2_calibration.yaml --stage manifest
python -m blrpng.cli.generate_calibration --config configs/allen_cahn_v2_calibration.yaml --stage support
python -m blrpng.cli.generate_calibration --config configs/allen_cahn_v2_calibration.yaml --stage query128
python -m blrpng.cli.generate_calibration --config configs/allen_cahn_v2_calibration.yaml --stage query256
python -m blrpng.cli.generate_calibration --config configs/allen_cahn_v2_calibration.yaml --stage oracle

python -m blrpng.cli.run_calibration --config configs/allen_cahn_v2_calibration.yaml \
  --budget 4 --seed 42 --skip-existing
python -m blrpng.cli.run_oracle_parallel --config configs/allen_cahn_v2_calibration.yaml \
  --seed 42 --skip-existing
python -m blrpng.cli.quick_evaluate_matrix --resolution 256 --horizon 20 \
  --output artifacts/reports/v2_seed42_query256_h20.json
python -m blrpng.cli.evaluate_oracles --resolution 256 --horizon 20
python -m blrpng.cli.summarize_calibration
python -m blrpng.cli.visualize --summary artifacts/reports/v2_seed42_calibration_summary.json
```

`summarize_calibration` 执行 50,000 次 seed→boundary→initial-condition 配对分层
bootstrap、5% non-inferiority/TOST、目标边际成本和 10× speedup 门控。只有其
`gate.passed=true` 后才允许生成 120-task 正式族或 Gray–Scott/CFD 数据。

## 数据与 checkpoint

旧数据继续兼容 schema v3。v2 使用 schema v4 原子 rollout shard：每个 shard
只保存一次 `trajectory[T+1,C,H,W]`、边界/lift/mask 与 Newton corrections，
grid 按需构造；support/query 与 LR/HR 关系由带 SHA256 的 index manifest 管理。
数据包含状态、轨迹、边界、lifting、Newton correction、
残差进度、有效 mask、case/pair/boundary/IC ID、fidelity 和收敛诊断。loader
会校验：

- schema、有限值、rollout-major 顺序和 Fourier modes 容量；
- 输入/目标/lifting 的边界精度及 correction 的零边界迹；
- coarse/fine 的 manifest SHA 和四类配对 ID；
- frozen manifest 的 A/B 边界隔离、初值隔离和允许的 seen-boundary 共享。

checkpoint 保存模型与表示配置、loss 配置、数据/manifest SHA256、初始化
SHA256、seed、最佳粗网格分层 rollout 指标、raw/EMA 权重及恢复训练状态。

## 正式流程与盲测锁

开发和正式 manifest 可先独立冻结：

```bash
python -m blrpng.cli.generate --config configs/development.yaml --stage manifest
python -m blrpng.cli.generate --config configs/allen_cahn_formal.yaml --stage manifest
```

生成 A 和 B 粗网格数据：

```bash
BLRPNG_GENERATION_WORKERS=24 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python -m blrpng.cli.generate \
  --config configs/allen_cahn_formal.yaml --stage a
python -m blrpng.cli.generate --config configs/allen_cahn_formal.yaml --stage b-coarse
```

`BLRPNG_GENERATION_WORKERS` 只将冻结 manifest 的连续 case 区间并行求解，父进程
按原顺序合并并统一检查返回码；它不改变 case、ID、数值参数或数据顺序。每个 worker
的 stdout/stderr 写入 `artifacts/tmp/` 独立日志，避免 pipe backpressure 使并行任务
串行化。未设置该变量时保持单 worker 行为。

开发门控证据 JSON 必须含 `data_reports`、`gradient_ratios`、
`seen_blrpng`、`seen_pretrained`、`unseen_blrpng`、`unseen_pretrained`：

```bash
python -m blrpng.cli.gate_formal \
  --development-config configs/development.yaml \
  --formal-config configs/allen_cahn_formal.yaml \
  --evidence artifacts/reports/development_gate_evidence.json
```

只有门控全部通过，以下命令才会解锁成对细网格和 blind splits：

```bash
python -m blrpng.cli.generate \
  --config configs/allen_cahn_formal.yaml --stage blind
```

冻结 120 个微调任务和额外消融：

```bash
python -m blrpng.cli.run_matrix --config configs/allen_cahn_formal.yaml
```

加入 `--execute` 会先运行 5 seeds × 2 个 A 预训练，再按相同
seed/budget 的初始化、batch 顺序和步数运行 120 个因果任务，并持续写入
`artifacts/reports/matrix_progress.json`。由于正式矩阵计算量很大，默认仅冻结
任务清单。

先执行单个 seed、稍后再补其余 seed 时，冻结矩阵仍保持完整 120 个任务：

```bash
python -m blrpng.cli.run_matrix \
  --config configs/allen_cahn_formal.yaml --execute --seeds 42 --skip-existing
```

`--initializations`、`--skip-pretraining` 与 `--skip-ablations` 可将同一冻结矩阵分配
到不同 GPU/阶段执行，不会建立新的实验矩阵。完成一个正式 seed 后自动汇总：

```bash
python -m blrpng.cli.summarize_formal \
  --config configs/allen_cahn_formal.yaml --seed 42
```

分层 bootstrap：

```bash
python -m blrpng.cli.analyze \
  --records artifacts/reports/paired_records.json \
  --method blrpng_pretrain_blrpng_finetune \
  --baseline scratch_final_only
```

统计单位固定为 seed → boundary → initial condition，默认 50,000 次配对
重采样。Allen–Cahn 五项论文门控通过后，才允许生成 Gray–Scott 确认数据：

```bash
python -m blrpng.cli.gate_paper \
  --evidence artifacts/reports/paper_claim_evidence.json
python -m blrpng.cli.generate_gray_scott \
  --config configs/gray_scott_confirmation.yaml
```

## 主要输出

- `artifacts/manifests/`：冻结数据与实验矩阵；
- `artifacts/data/`：schema v3 `.pt` 与 schema v4 原子 rollout shards；
- `artifacts/checkpoints/`：最佳与 recovery checkpoint；
- `artifacts/reports/`：验证、workspace audit、训练、评估与统计报告；
- `artifacts/figures/`、`artifacts/wandb/`：论文图和本地 W&B 文件。

细网格评估报告包含 20 步 per-rollout relative L2、MSE、one-step、
near-boundary/interior、PDE residual、Fourier shell error、边界误差和粗求解器
插值误差下限。结果的科研主张仍必须以完整五随机种子正式矩阵为准；smoke
结果仅验证实现链路，不能作为方法有效性的证据。

## Poisson 线性负对照（seed42）

Poisson 阶段把每个 target task 定义为一条固定、预训练未见的 Dirichlet 边界，
support/validation/query 只改变 forcing。正式协议包含500/100条128² source
train/validation、30个目标边界、每边界8/4/16个64² support、64² validation和
128/256² query forcing。所有 split 和分辨率配对在生成前冻结：

```bash
python -m blrpng.cli.freeze_poisson_protocol
python -m blrpng.cli.generate_poisson_formal --section source
python -m blrpng.cli.generate_poisson_formal --section target_lr
python -m blrpng.cli.verify_poisson_formal --scope lr
python -m blrpng.cli.train_poisson_source --objective final_only --seed 42
python -m blrpng.cli.train_poisson_source --objective blrpng --seed 42
python -m blrpng.cli.freeze_poisson_methods
python -m blrpng.cli.train_poisson_task --task 0 --budget 4 --method ordinary_nlt
python -m blrpng.cli.train_poisson_task --task 0 --budget 4 --method path_nlt
python -m blrpng.cli.freeze_poisson_adapters
python -m blrpng.cli.generate_poisson_formal --section query128
python -m blrpng.cli.generate_poisson_formal --section query256
python -m blrpng.cli.verify_poisson_formal --scope query
python -m blrpng.cli.summarize_poisson_negative_control
python -m blrpng.cli.visualize_poisson_negative_control
```

seed42的256²/480个paired query结果为：final-pretrain→NLT 0.820% relL2，
path-pretrain→NLT 0.849%，训练至validation收敛的lifted LR scratch 54.74%，
boundary-value+mask LR scratch 134.83%。path相对ordinary的改善为−3.44%
（95% CI −5.94%至−0.97%），因此通过“绝对改善不超过5%”的线性负对照门控；
严格±5% TOST尚未通过（ratio单侧上界1.0555），不得写成统计等价。
ordinary NLT相对收敛型lifted scratch提升98.50%（95% CI 98.35%–98.63%）。
这些仍是seed42结论，论文主张需要补齐多随机种子。

冻结汇总与PNG位于：

- `artifacts/experiments/poisson_negative_control/seed42_summary.json`
- `artifacts/figures/poisson_negative_control/poisson_negative_control_comparison.png`
- `artifacts/figures/poisson_negative_control/poisson_representative_fields_256.png`
