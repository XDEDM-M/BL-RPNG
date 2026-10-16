# BL-RPNG：从性能改进到“低保真适配替代高保真重训练”

## 1. 论文主张重新定义

当前结果证明了 BL-RPNG 的表示迁移价值，但不足以单独证明工程意义上的边界条件泛化。论文的中心问题应改为：

> 对一个预训练未见的目标 Dirichlet 边界函数，仅使用该边界上的少量低分辨率 trajectory 和 Newton path，能否得到一个高分辨率 surrogate，其精度与在该目标边界上使用大量高分辨率数据训练到饱和的 specialist oracle 等价，同时显著降低目标任务的边际成本？

对应三个必须同时成立的可证伪主张：

1. **High-fidelity equivalence**：BL-RPNG low-fidelity adaptation 的 128/256-grid query error 相对 target-specific HR-full oracle 的误差比不超过 1.05。
2. **Economical adaptation**：不计 evaluation ground truth，目标任务数据生成、微调和推理的总边际成本相对 HR-full retraining 至少降低 10 倍；另报告包含预训练成本后的 amortized break-even。
3. **Low-resolution insufficiency**：相同低分辨率 support 上 scratch 模型显著更差，预期保持当前约 88% 的相对误差下降。

“高分辨率完整训练”不能由任意固定样本数定义。HR oracle 的训练量应从 4/8/16/32/64 rollouts 递增，直到样本量翻倍带来的 query error 改善小于 1%，再把该平台点定义为 HR-full oracle。

## 2. 当前实验与目标之间的关键缺口

1. 当前 B-budget 是联合训练时采用的边界函数数量，不是“对一个新边界分别进行 few-shot adaptation”的 support size。
2. 当前 primary `blind_unseen` 没有经过该边界上的适配；它验证 zero-shot boundary extrapolation，却不是工程目标中的 target-boundary adaptation。
3. 当前没有 target-specific HR-full oracle，因而 10.49% 只表示相对普通预训练的优势，不能证明接近高保真训练上限。
4. 当前没有严格记录 coarse solve、fine solve、数据转换、微调和推理的端到端成本，也没有把 A 预训练成本摊销到下游边界任务数量。
5. FNO 参数可以跨网格运行不等于误差自动离散无关；必须显式测量 64→128→256 的 commutator/discretization error。

## 3. 方法升级：BL-RPNG v2

### 3.1 Affine boundary reduction

对每个 Dirichlet 边界函数计算离散调和 lift `E_h g`，令

`v_h = u_h - E_h g ∈ V_{h,0}`。

模型只预测齐次迹空间中的更新并在输出端投影为零边界。因此 boundary feasibility 是结构性结论，而不是统计现象。

### 3.2 Newton tangent representation

对隐式时间步残差

`F_h(v_h^{n+1}; v_h^n, E_h g) = 0`

监督 accepted Newton correction

`δ_{h,k} = -J_h(v_{h,k})^{-1} r_h(v_{h,k})`。

归一化方向和 log-magnitude 监督学习的不是某个绝对场，而是边界条件索引的局部 inverse-Jacobian action。论文中应把它表述为 **boundary-conditioned tangent solver prior**，而不是泛称“多一个辅助损失”。

### 3.3 Resolution-commuting path loss

从 A 的 fine trajectory 构造限制后的多分辨率视图，并加入

`L_comm = ||R_{h→H} Phi_h(v,E_hg) - Phi_H(Rv,R E_hg)||`

以及 Newton correction 的方向一致性损失。小规模真实 coarse/fine paired solve 用于校准限制误差。该项直接约束低分辨率适配后在高分辨率运行时的离散交换误差。

### 3.4 Conflict-gated annealed fine-tuning

固定 0.15 路径梯度比例已在 seed42 的 budget 2/4/8 上暴露目标冲突。v2 使用

`lambda_t ∝ (1 - t/T)^gamma · max(0, cosine(grad_final, grad_path))`

并仅在 active batch 中保持路径/最终梯度范数比 0.15。这样 Newton path 作为早期优化几何先验，在其与最终 rollout 目标冲突时自动退出。

### 3.5 Unlabelled fine-grid defect calibration

目标边界适配仍不使用任何 fine-grid solution label，但允许在 128/256 网格上计算少量 PDE residual：

`L_defect = ||F_h(u_hat^{n+1}; u_hat^n,g)||_{H^{-1}}`。

一次 residual/JVP 计算远便宜于完整 nonlinear solve。该项用于消除 coarse adaptation 与 fine inference 的 fidelity gap，并应单独报告它的时间和消融。为避免混淆，论文同时报告纯数据版 BL-RPNG 和加入 defect calibration 的 BL-RPNG+。

## 4. 决定性 target-task 实验

### 4.1 数据单元

每个目标边界 `g_j` 是一个独立 adaptation task，且完全不出现在 A：

- support-LR：64×64，`m ∈ {1,2,4,8}` 个初值 rollout，每条 10 步；
- support-HR learning curve：128×128 和 256×256，`m ∈ {4,8,16,32,64}`；
- query-HR：与 support 初值严格隔离，每个边界至少 16 个初值，每条 20/50 步；
- 每个 method/seed/task 使用相同初始化分布、batch 顺序和更新步数。

开发阶段每组 8 个目标边界，正式阶段每组至少 30 个目标边界。边界组包括：

1. in-family hard：frequency 5/6、amplitude 0.35；
2. frequency OOD：7/8；
3. amplitude OOD：0.65；
4. family shift：corner-compatible Matérn/GP、局部 boundary bump、混合 Fourier 边界。

所有主结论先限定为静态非齐次 Dirichlet 边界和固定矩形区域；不要在没有实验时扩张到 Neumann、动态边界或新几何。

### 4.2 对照模型

| 组别 | 目标边界数据 | 作用 |
|---|---|---|
| HR-full lifted FNO specialist | 饱和数量 HR labels | 精度 oracle |
| HR-few lifted FNO | 与 LR 相同数量 HR labels | 分离分辨率与样本数量效应 |
| LR scratch lifted FNO | few LR labels | 用户目标 3 |
| A final-pretrain → LR final | few LR labels | 普通迁移基线 |
| A BL-path-pretrain → LR final | few LR labels | 路径预训练机制 |
| A BL-path-pretrain → fixed path | few LR labels | 已知失败模式对照 |
| BL-RPNG v2 | few LR labels/path | 主模型 |
| BL-RPNG+ | few LR labels/path + fine residual，无 HR labels | 实用增强版 |

还应至少加入一个现代 few-shot neural-operator transfer baseline，以及一个 hard-BC baseline。Newton-informed neural operator 作为机制邻近基线；如果其时间步 rollout 接口无法公平复现，应明确说明差异而不是省略。

### 4.3 Primary endpoint

主指标改为：**经过该目标边界 support 适配后，在相同边界的新初值 HR query 上的 20-step per-rollout relative L2**。未适配边界上的 zero-shot error 保留为 secondary endpoint。

精度等价用 one-sided non-inferiority/TOST，而不是普通显著性检验：

- `error_BL / error_HR-full` 的分层 bootstrap 95% CI 上界 `< 1.05`；
- 分层顺序为 seed → target boundary → query initial condition；
- 至少 4/5 seeds 达到 pointwise win/non-inferiority；
- 同时报告 near-boundary、interior、PDE residual、Fourier shell、50-step stability。

## 5. 成本与工程价值协议

目标任务成本定义为：

`C_target = C_data_generation + C_adaptation_training + Q · C_inference`。

分别报告：

1. marginal cost：A 已预训练时，新边界的增量成本；
2. amortized cost：`C_A / K + C_target`，其中 K 是复用该预训练模型的目标边界数量；
3. break-even：使 BL-RPNG 总成本首次低于 HR retraining 和 HR numerical solve 的 `K*` 与 query 数 `Q*`。

计时必须包含 mesh/assembly、Newton iterations、I/O、训练、checkpoint selection 和推理；evaluation ground truth 生成单独列出，不计入部署成本。64/128/256 网格使用同一硬件、相同 solver tolerance 和线程数，至少 3 次重复，报告 wall-clock、GPU-hours、峰值显存和能耗。速度主张要求 paired speedup 的 95% CI 下界大于 10×。

最终绘制 accuracy–cost Pareto frontier，而不是只给单个误差表。需要诚实报告大规模 A 的离线成本；工程价值来自其在多个目标边界和大量 query 上的复用，而不是隐藏预训练成本。

## 6. 机制证据

为把方法从经验技巧提升为机制工作，需要新增：

- coarse/fine Newton correction cosine、log-norm error 和 `J δ + r` defect；
- 64→128→256 operator commutator error；
- path/final 梯度 cosine 随 epoch 和 support size 的变化，解释 fixed path 失效及 gating 生效；
- rollout Jacobian spectral radius/local Lipschitz proxy，验证长期稳定性；
- error-vs-boundary-distance 与 boundary-frequency/amplitude 曲线；
- probe 表示的 principal-angle/CKA 分析，检验预训练是否保留了可迁移 Newton tangent subspace。

## 7. 理论结果目标

论文理论部分至少形成三个命题：

1. **Exact trace proposition**：在 lift 和零迹投影离散一致时，任意参数和任意 rollout 长度均精确满足目标 Dirichlet trace。
2. **Tangent-transfer proposition**：在 Jacobian 一致稳定、离散算子具有 p 阶一致性时，coarse/fine Newton action 的 restriction commutator 由 `O(H^p+h^p)` 和边界 lift 逼近误差控制。
3. **Rollout bound**：T-step fine-grid error 由表示误差、目标适配误差、commutator error、fine-grid defect 和离散误差的稳定性加权和控制。

这三个结果分别对应 boundary correctness、low-to-high transfer 和长期 rollout accuracy，能够与实验中的每个度量一一对应。

## 8. 执行优先级

1. 先利用当前 8 个 B 边界把训练改成“每个边界一个 task”，生成 HR specialist learning curve，验证 parity 是否可达。
2. 在 seed42 上做 v2 的最小三项消融：commutator、conflict-gated annealing、fine defect。
3. 只有当 seed42 同时达到 `<5% oracle gap` 和 `>10× marginal speedup` 时，再扩展至 30+ target boundaries 和 seeds 0/1/2/3。
4. Allen–Cahn 的跨 seed 主张成立后，再用 Gray–Scott 做 2/4-shot confirmation；否则先修复机制，不扩大 PDE 数量。
