# Phase-1 统计与确认协议复核

审计日期：2026-10-12（Asia/Shanghai）  
范围：仅审阅任务书、配置和代码接口。未打开 HR query rollout 数值/张量，未读取测试结果，未运行生成、训练、评估或测试命令。

## 判定摘要

当前 Allen–Cahn reconfirmation 管线适合作为数据生成和 gate 的骨架，但不能原样承担任务书要求的多种子确认。它目前把一个 target task 对应到一个 LR support 集和一个 `seed=42` 训练运行；没有独立 support 重抽样维度，也没有 5 个完整 model seed × 至少 3 个 support seed 的交叉矩阵。现有 bootstrap 的重复单位是 `seed × boundary × initial condition`，既没有 support-seed 字段，也没有 boundary-family/task 两层泛化结构。因此，现有 seed-42 报告只能作为历史/探索证据，不能填充新的确认主表。

## 冻结的确认设计

在生成任何新确认 HR query 前，建立全新、不可覆盖的 campaign ID 和 manifest。不得复用 `allen_cahn_reconfirmation_tasks_20260820.json`、它的 query manifest/index、报告或 checkpoint 路径；新 manifest 应记录独立 boundary 生成主种子、六个预设边界族、各族任务数、支持/验证/query 规则、数据分辨率和 rollout horizon，并保存协议 hash、源/开发边界排除审计及每个 descriptor 的 hash。冻结命令遇到同名资产时应失败，不能静默覆盖。

确认统计单位建议固定为：

```text
campaign / family / target_task / model_seed / support_seed / query_initial_condition
```

- **边界任务**：每个确认 task 是一个未用于源 HR 训练或超参选择的边界函数；`family` 和 `target_task_id` 均写进每条训练/评价记录。保持六个预设 family 的任务数平衡，并公布逐 family 结果。
- **模型种子**：完整确认矩阵采用任务书规定的 5 个 model seeds `[0, 1, 2, 3, 42]`。对预训练方法，seed 应对应独立的源训练初始化/训练运行及其适配；普通迁移对照也按各自冻结的训练流程独立运行。只对同一个 seed-42 source checkpoint 更换 adapter RNG，不能声称已经覆盖 5 个完整模型种子，应明确标作条件于该 source checkpoint 的适配随机性分析。
- **support 抽样**：冻结至少 3 个 support seeds。每个 `(task, support_seed)` 使用独立生成的真正 LR solver rollouts，独立于 validation 与 HR query。对每个 support seed 固定一条嵌套 support 顺序，以前缀形成 `m=1,2,4,8,16`（若资源只支持较少预算，须在 query 解封前冻结主预算及删减理由）。同一 `(task, support_seed, m)` 数据对所有方法和 model seeds 共享，保证方法比较配对；模型 seed 和 support seed 做完整笛卡尔积，不抽成 15 个“独立 seeds”。
- **query**：每个 task 冻结同一组 HR query 初始条件，和所有 support/validation 初始条件全局不相交；128/256（以及若执行的其他分辨率）使用同一 query case/pair IDs。query 仅用于锁定配置、检查点选择和统计方案之后的最终评测，不能选方法、seed、support 抽样或超参。
- **角色隔离**：manifest 明示 `model_seed` 与 `support_seed` 是不同字段；每条 LR rollout 含 `campaign_id, family, task_id, support_seed, support_rank, initial_condition_id, boundary_id, role, resolution, steps`。所有 case/pair/IC ID 不重复；同 task 同 support seed 的 m 子集严格嵌套；不同 support seeds 的 IC 不交叉；support 与 validation/query IC 不交叉。

在解封 HR query 前，应依次冻结：新 boundary/task manifest → support/validation manifests 和 LR 数据 gate → 所有确认方法与训练配置、checkpoint/早停规则、query 主终点、成本口径、种子矩阵和分析代码 hash → HR query manifest/数据生成 gate。当前 HR query 已打开的历史资产只可用于历史复算或背景，不进入新选择流程。

## 统计与 bootstrap

### 目标估计量

对预先指定的完整 rollout 主指标，先在每个 query rollout 上计算定义一致的 20-step per-rollout relative L2，再按预先冻结的权重汇总；建议各 task 等权、各 family 等权，避免 query 数量改变 family 权重。主比较对同一 `family/task/model_seed/support_seed/query_initial_condition` 的方法结果做配对，报告差值或误差比、95% CI、逐族值及尾部退化/失败率。

### 必须实现的重抽样层级

写一个新的配对 crossed-cluster bootstrap，不要直接复用当前 helper 来伪装交叉设计。每次 replicate 应：

1. 在 family 层按预先声明的目标总体做 cluster 重抽样；在抽中的 family 内按 target task cluster 重抽样。
2. 独立重抽 model-seed 轴与 support-seed 轴，再取完整笛卡尔积；不可把 5×3 组合展平成 15 个可互换 seed。
3. 在 task 内重抽 query initial-condition cluster；对候选与基线使用完全相同的 family、task、两个 seed 轴和 query IC 抽样索引。
4. 每次 replicate 用相同聚合权重重算配对增益/比值，固定 bootstrap RNG seed、replicate 数和区间算法，并在输出中保留这些信息。

六个 family cluster 数仍少，跨 family 总体 CI 的精度有限。应同时给出六个逐族效应、按 family 等权的主估计、leave-one-family-out 敏感性，并把外推范围限制在采样到的边界族。若 family 被视作固定分层而非随机总体，也要明确 CI 是条件于这六族；不要把时间步或 15 个交叉 seed cell 当作独立任务扩充样本量。

五个 model seeds 与三个 support seeds 是任务书的最低确认规模；尤其 support-seed cluster 只有三个时，bootstrap 区间会很不稳定。预算允许时增加 support draws；无论如何都展示每个 seed 轴的边际效应/leave-one-seed-out 敏感性，并把基于最低种子数的 CI 解释为有限精度，而不把名义上的 15 个交叉格子当作有效独立 cluster 数。

## 现有可复用代码与缺口

| 位置 | 可复用内容 | 确认协议所需修改 |
|---|---|---|
| `blrpng/data/blind_boundaries.py`：`sample_boundary_descriptor`、`freeze_reconfirmation_manifest`、`write_reconfirmation_split_manifest` | 边界族描述符、确定性 boundary/IC 生成、manifest 内容 hash、角色 ID offset 的实现模式 | 新 campaign 独立 seed/path；冻结不可覆盖；生成多个 support-seed 子集；增加 support-seed / rank 字段；检查 descriptor hash 与源/开发边界排除；不要沿用静态 `20260820` 文件名 |
| `blrpng/cli/freeze_reconfirmation_manifest.py`、`prepare_reconfirmation_split.py` | freeze/prepare 分离、split 不包含解、query 前置 gate 的命令框架 | 接收 campaign manifest/output path；新增 `--support-seed` 或独立 support-draw ID；support 不再只有单一 manifest；冻结 support-seed 列表及完整预算；固定路径现会覆盖同名资产 |
| `blrpng/cli/generate_reconfirmation.py`：`_generate`、`_write_task_indices` | 复用真实 solver worker、v4 rollout shard、index hash、容量检查、workspace audit、LR/HR 角色的网格/horizon 设定 | 给 campaign 与 support draw 路径加参数；每个 support seed 独立 index 与 m 前缀；index 已存在时除 shard hash 外还必须核对请求的 manifest hash/role/resolution；新 campaign 单独数据根目录，避免复用旧 index |
| `blrpng/cli/verify_reconfirmation.py`、`verify_reconfirmation_query.py` | shard schema、solver 收敛、correction mask、边界 trace、split 独立性、跨分辨率 pair 检查 | 去掉固定 384/192/768 与 48-task 假设；逐 support seed 验证 counts、嵌套关系和跨 seed 不相交；查询 gate 也绑定新 campaign manifest hash 和 method freeze hash |
| `blrpng/cli/freeze_method_gate.py`、`freeze_reconfirmation_query_protocol.py` | 冻结方法/权重 hash、冻结 query 主终点、声明不得从 query 选择 | 当前 gate 固定 seed-42 checkpoint、NLT final-only、单一 support budget；改为记录完整模型种子与方法 checkpoint 清单、路径 FT 和强基线的选择规则、LR-only checkpoint/早停依据、crossed seed/support 设计和新 manifest hash |
| `blrpng/cli/train_reconfirmation.py`、`evaluate_reconfirmation.py`、`evaluate_reconfirmation_matrix.py` | 训练/验证/HR query 的清晰分层、checkpoint hash、共同 query bundle 的配对评测模式 | 添加 campaign、model-seed、support-seed 参数；读取对应 support index；checkpoint/report 文件名纳入两种 seed；报告显式输出 family/task/model_seed/support_seed/query IC；移除固定 `_s42_` 与旧 gate/manifest 依赖 |
| `blrpng/analysis/bootstrap.py`：`paired_hierarchical_bootstrap`、`paired_ratio_bootstrap` | 完整配对检查、层次重抽样和区间报告模板 | 当前索引键只有 `(seed, boundary_id, initial_condition_id)`，层级为 `seed→boundary→IC`。不能表示 task→family 和独立 support seed 交叉轴；新增函数和唯一键检查，不要把 `(model_seed,support_seed)` 编码成 seed 后调用旧实现 |

### 具体风险

1. `write_reconfirmation_split_manifest()` 对一个 task 每个 role 只取一个 `split_seeds[role]`；现有 support 是单样本抽法。三次调用若不改 ID/path，会互相覆盖或把相同 prefix 误当独立 support。
2. 当前 support manifest/index 和训练 runner 都只需 `task, budget, method, seed`，没有 support draw ID。相同 checkpoint 路径可能被多次写入，破坏可审计性。
3. `generate_reconfirmation._generate()` 发现目标 index 已存在时只校验其自身 hash 后直接复用；尚未验证它绑定的是本次传入的 manifest hash。新 campaign 必须绑定并核对 manifest hash。
4. 当前 `verify_reconfirmation*` 的硬编码 task/rollout counts 只适配旧设计，无法检查完整交叉矩阵的支持集配对和 support-seed 独立性。
5. 将交叉的 15 个 `(model_seed,support_seed)` 组合拼成单个 seed 字段会让旧 bootstrap 以为有 15 个独立训练种子，丢失两个 seed 轴的共享/交叉相关性，CI 可能偏窄。

## 最小实现建议

1. 新增版本化 protocol/config 和 campaign 专属目录；旧确认/历史资产只读保留。新增 `freeze_confirmation_manifest` 与 `verify_confirmation_manifest`，冻结时写完整种子/角色计划和 hash，不能覆盖已存在文件。
2. 一份 task manifest 描述新边界任务；每个 support seed 产生独立 support case 清单，或一个含 `support_seed` 字段的组合清单。validation/query 清单独立，ID 唯一且 query 仅在配置 freeze gate 通过后生成。
3. 扩展生成与审计命令，使 support-seed 成为输入和输出路径的一部分；验证真正 LR solver 分辨率、物理时刻/边界配对、角色隔离、不同抽样不交叉以及任务完整性。
4. 训练/评估 runner 的输出键至少包含 `(campaign, method, task, model_seed, support_seed, budget)`，并保存支持集/查询 index hash、训练 config hash、checkpoint hash 和选择来源。
5. 新增 `paired_crossed_cluster_bootstrap`：family→task→query IC 嵌套重抽样，model seed 与 support seed 独立 crossed 重抽样，所有方法共享抽样索引。旧 bootstrap 留给历史单 support 协议复算，避免改变历史结果定义。
6. 在启动正式计算前由主线程统筹资源；此审计没有运行任何计算任务。

## 可直接复用接口索引

- 数据格式/哈希：`blrpng.data.manifest.read_manifest`、`write_manifest`；`blrpng.data.shards.read_index`、`write_index`、`subset_index`、`write_v4_rollouts_from_arrays`。
- 数据隔离：`blrpng.data.tasks.validate_task_isolation` 是支持/验证/query IC 和 task 对齐的参考实现，但需要扩成多 support draw。
- solver 生成：`blrpng.cli.generate_reconfirmation._generate` 与 `blrpng.data.generate.workspace_audit`。
- 前置 gate：`blrpng.data.capacity.require_prerequisite_gate`、`blrpng.paths.sha256_file` / `sha256_json`。
- 配对统计起点：`blrpng.analysis.bootstrap._index` 和 `paired_hierarchical_bootstrap`；保留严格方法配对检查，但改用包含 family/task/model/support/query 多维 key 的新 helper。

---

## R1 当前结果与账本独立复核（2026-10-13）

### 范围与结论

本次是只读统计复核；未启动训练、生成、评估或测试，也未读取任何 query/test 数值或张量。检查范围为 `docs/EXPERIMENTS.csv`、`docs/BUDGET.md`、`docs/STATE.md`、R1 配置/runner/汇总代码、两组 R1 LR-development 原始 task JSON、checkpoint/index/manifest 哈希及已有输入完整性审计。

结论：`docs/STATE.md` 声明的两个四任务均值和 **18.39%** 相对下降均可从原始 LR-development 输出精确重算，哈希链在当前文件状态下闭合；没有发现 query/test 指标混入本次 R1 排名，也没有发现 support 与 LR validation 初值交叉。但这仍是单 model seed、单 support seed、四个开发 family 的早期 screen，并且同一 LR validation 同时承担逐任务 checkpoint epoch 选择和配置排名，不能作为确认结果、HR 推理优势或路径微调机制证据。预算总额有一个约 0.10 秒的账本差异，ledger 另有一个未用同一 trial ID 闭合的 RUNNING 行。

### 数字重算

选择规则与 `configs/phase1_r1.yaml`、`docs/experiment_cards/R1.md` 一致：对 task 0/1/3/5 的各自最佳 64²、20-step LR-validation rollout relative-L2 做不加权算术平均；同均值时以 worst-task 和 trial ID 排序。两组均使用 model seed 42、support seed 271829、m=4、相同四个 task 以及相同 support/validation index。

| 候选 | task 0 | task 1 | task 3 | task 5 | 独立重算均值 | worst task |
|---|---:|---:|---:|---:|---:|---:|
| path-source pretrain + final-only FT, `lr=1e-3/cosine100` | 0.027361116488464177 | 0.033740171708632256 | 0.025245579797774552 | 0.025317266315687453 | 0.02791603357763961 | 0.033740171708632256 |
| ordinary final-only pretrain + final-only FT, `lr=1e-3/cosine100` | 0.03478868951788172 | 0.037237829144578426 | 0.03361704026465304 | 0.03118929336778819 | 0.034208213073725346 | 0.037237829144578426 |

重算式：

```text
absolute reduction = 0.034208213073725346 - 0.02791603357763961
                   = 0.006292179496085737
relative reduction = 0.006292179496085737 / 0.034208213073725346
                   = 0.18393768427848 = 18.393768427848%
```

因此四舍五入后的 **18.39% lower** 正确。逐任务相对下降为 21.3505%、9.3928%、24.9024%、18.8271%，四个已筛选 family 上方向一致。该比例的分母是当前 ordinary-transfer screen leader，不是 scratch、HR expert 或确认集结果。

`docs/STATE.md` 中另外两个 ordinary-transfer 均值也与 ledger 的四任务值一致：`lr=1e-4/cosine100` 为 0.03902231，`lr=3e-4/cosine100` 为 0.03596554。当前仅完成 ordinary-transfer 预设九配置中的三个，故 `lr=1e-3/cosine100` 只能称“best so far/provisional leader”，不能称充分调优的最终强基线。

### 原始输出与哈希闭环

- 主候选 summary 的当前 SHA-256 为 `a9777df64dc676badbb0844d46bc2e520f944b050d8a699ca666c3603f9c46dc`，与 `STATE` 声明一致；ordinary summary SHA-256 为 `5589ee35c27aba25ade488709377c6f2c0d79a7568efd0ff5637840eed86f90a`。
- 冻结 R1 YAML 的文件 SHA-256 为 `d829bcebf152b0b5f39b125ced68d432ce363362b73299d9aa1f41bc5da05771`，两组 summary 的 `trial_spec_sha256` 均一致。`a62c...` 和 `21db...` 是各自 canonical descriptor hash，而不是 YAML 文件 hash；ledger、config、task 输出内部一致。
- path-source 与 ordinary source checkpoint 的当前 SHA-256 分别为 `a85ed6185c26f730d6096e94ca37d6905ddb3c5f64194066ae01c111881d5c87` 和 `0bcefb090a53c6639c08ea08754d76035d68360c8dca1ba7512a09b0ae6655f6`，与 descriptor/ledger 一致。
- 八个 task 输出的 `best_validation_metric` 与 summary、ledger 完全一致；八个 compact adapter checkpoint 的当前文件 SHA-256 均与对应 task JSON 记录一致。两种方法在每个 task 上记录的 support/validation index 路径与 whole-file SHA-256 相同。
- 输入审计的 per-task `*_index_sha256` 是 index JSON 内嵌的 canonical content digest，例如 task 0 support 为 `50a83b...`；task 输出记录的是整个 index 文件的字节 SHA-256，例如同一文件为 `ccc45a...`。二者分别与当前文件的内嵌字段和 whole-file digest 相符，不是资产变化，但字段同名容易误读，后续报告应显式写成 `index_content_sha256` 与 `index_file_sha256`。

### 选择纪律与泄漏复核

- `scripts/run_phase1_trials.py` 明确将训练路径绑定到 per-task LR support index、验证路径绑定到 LR validation index；该 runner 不构造 HR/query 路径。原始 task JSON 的 `query_metric` 均为 `null`，summary 的 `query_opened_or_loaded` 为 `false`，输入审计的 `query_read` 为 `false`。
- 输入审计报告四个 task 的 support/validation initial-condition overlap 均为 0，且两者都是 fidelity 0、64²、20 steps；已有 campaign 完整性审计还记录 support、LR validation、HR-development validation 三个角色间 IC overlap 均为 0。
- 在本次可审计的代码路径和产物中未发现 query/test 泄漏。该判断是 provenance/code-path 审计结论，不等价于系统调用级文件访问证明。
- 当前 LR validation 的 8 rollouts/task 同时用于 epoch 选择和 36 配置搜索/排名，存在开发集复用带来的选择乐观偏差，但这不是 query 泄漏。必须按冻结规则只将其用于 screen，再在六 family 的独立 HR-development validation 上晋级；最终结论仍需新 confirmation 边界与未打开 query。
- 18.39% 比较隔离的是不同源预训练目标在同一 final-only 目标适配配置下的差异；它没有测试 target path FT。两个 source checkpoint 的 source-HR validation 也略有差异（0.0089858848 vs 0.0090430082），因此不能把全部差值直接归因为 accepted-path 机制。

### Ledger 异常

1. `R1-source-path-continuation-s42` 有 RUNNING 行但没有同 trial ID 的终态。相同配置 hash `96e26d...` 的终态被记为 `R1-source-path-warm_start-s42`（INTERRUPTED，2.489248444306 GPU-hours）。从注释可推断二者是同一原始作业，但机器可读 ledger 无法自动闭合。建议仅追加一条明确 supersede/alias 的终态说明，不修改旧行。
2. 第一次 scratch smoke 的 DONE 行后来被 `INVALID_OUTPUT` 推翻；其输出目录随后被新 config hash `ebbc...` 的 rerun 复用。收费说明保留了第一次 DONE 的 0.002625693647 GPU-hours且 INVALID_OUTPUT 记 0，计算上没有双扣，但旧 DONE/INVALID_OUTPUT 行的 `raw_output_path` 现在指向 rerun 文件，无法单靠该路径复现原零字节状态。未来无效重跑应使用新 namespace，或记录不可变 incident artifact/hash。
3. append-only 的 RUNNING→DONE/FAILED 双行模式本身符合当前 runner 设计；排名必须取终态并尊重后续 INVALID_OUTPUT/supersede，不能把所有 DONE 行无条件当有效结果。

### 预算重算与差异

按 `docs/EXPERIMENTS.csv` 所有非 RUNNING 终态行的 `elapsed_gpu_hours` 求和（包括失败、被中断和需收费的无效尝试；INVALID_OUTPUT 行本身为 0）：

| 状态 | 行数 | GPU-hours |
|---|---:|---:|
| DONE | 24 | 1.076302732727 |
| FAILED | 3 | 1.950514362668 |
| INTERRUPTED | 3 | 3.678137333206 |
| INVALID_OUTPUT | 1 | 0 |
| **合计** | **31** | **6.704954428600** |

这也等于 incident 前终态收费 `5.459638603512` + 两个 incident reservation `1.188888888900` + 本次四任务 trial `0.056426936188`。`BUDGET.md`/`STATE.md` 声明 `6.70492694`，比 ledger 精确和少 `0.000027488600` GPU-hours（约 0.099 秒）。差异对资源决策不实质，但账本应以 ledger 精确值更正；对应 exploration balance 应为 `72 - 6.704954428600 = 65.295045571400` GPU-hours，而不是 `65.29507306`。此外，`BUDGET.md` 提及 `<0.01` GPU-hours 的 HR-development inference overhead，却未给精确值或 ledger 行；在发生实际 HR-dev 推理后应单列准确收费，避免累计口径含糊。

### 可支持的表述

当前证据支持：“在 seed 42、support seed 271829、m=4 和四个 LR-development task 的早期 screen 上，path-source + final-only FT 的未加权均值比目前已完成的最佳 ordinary-transfer 候选低 18.39%，四个 task 均改善。”不支持：“已确认强迁移优势”“HR 推理误差下降 18.39%”“target path fine-tuning 有效”“达到 T2”或“已优于充分调优基线”。

---

## R1 新增 trial、R2 commutator 与 HR-development 复核（2026-10-13 18:09 快照）

本节继续只读复核；未启动训练、生成、GPU 或 query/test 评估，也未读取 query/test 文件。快照时 `R2-comm256_lr1e-3_cw005_r2` 正在运行，仅其已终态 smoke 进入 ledger 总额；按主线程要求未等待该作业。

### R1 新增 LR-development trial

以下均为 seed 42、support seed 271829、m=4、task 0/1/3/5 的 64²、20-step LR-development validation 未加权任务均值。每个均值均已从四个原始 `task*.json` 重新求和，和 summary 精确一致。

| trial | 四任务均值 | 相对 paired final-only 变化 | 判定 |
|---|---:|---:|---|
| `final_lr1e-3_cosine100` | 0.02791603357763961 | 基准 | 当前 path-source final-only LR leader |
| `path_lr1e-3_ratio005` | 0.02792729793436592 | +0.040350849611% | 略差 |
| `path_lr1e-3_ratio015` | 0.028044917760416864 | +0.461685154586% | 差 |
| `path_lr1e-3_ratio030` | 0.02819331257342128 | +0.993260718829% | 差 |
| `final_lr3e-4_cosine100` | 0.0281588135738275 | 相对 `final_lr1e-3` +0.869679410267% | 差 |
| `path_lr3e-4_ratio005` | 0.028188174283422997 | 相对 paired `final_lr3e-4` +0.104268276497% | 略差 |
| `scratch_lr1e-3_cosine100` | 0.6009033954935148 | — | scratch 候选 |
| `scratch_lr3e-4_cosine100` | 0.5381263811839745 | — | scratch 候选 |
| `scratch_lr1e-4_cosine100` | 0.5347863933304324 | — | 三个已跑 scratch 中最佳 |

因此 target true-path FT 在当前两个已配对学习率上均未优于 final-only，且 ratio 增大时 `lr=1e-3` 均值单调变差。三个 scratch 的最佳值仍比 `final_lr1e-3` 高很多；`final_lr1e-3` 相对当前最佳 scratch 的 LR-development 均值下降 94.779965622575%。这只能说明当前四任务开发 screen 的差异，不能替代 scratch 的完整九配置调优或确认统计。

三组 `lr=1e-3` path trial 的 summary SHA-256 依次为：ratio .05 `caf1f8d4229012ea683cbcdad39644f9553924d9db3754c805419aee298888f2`、ratio .15 `706029013f9cb90d031c31dffbd7891a77854d46dbeee7c8f145019627231ebd`、ratio .30 `0d32c5a5ee0431a4e7236016ffc610aa44f4a896fcc2796606a795937aa1a2e6`。`final_lr3e-4`、`path_lr3e-4_ratio005`、scratch 1e-3/3e-4/1e-4 的 summary SHA-256 分别为 `3a79d81b9aeee35d5fd7d3107ed11ed37408038287171aa65e9131cec25b78b4`、`4398384a6bc7fcaa51043711519bca43d69f7bee3c3a6f7bf79ff0ba8a43465c`、`9872289576938996f2b16c9e5c16130e2099e383a2eba90c0f4a0645d4d60fb5`、`c171619b8e3640cf07629ce0c92942290c634ed3859dadb73fb18a1caf8c895b`、`a3cfe262a69b8e406340ee92b2af89f2594781b953c6073c9165dc37da4f2ea3`。

### 四任务 HR-development 报告

当前完整可复核的是 `final_lr1e-3`、`ordinary_lr1e-3`、`scratch_lr1e-4` 和 `final_lr3e-4` 各四个报告；尚无本节 path-FT 或 R2 commutator 的 HR-development 报告。每个 task 有 8 条独立 256²、20-step HR-development rollout。逐 task 的 `rollout_rel_l2` 均已直接由报告中的 8 个 `per_rollout_rel_l2` 重算，16/16 与报告值精确一致。

| 方法 | task 0 | task 1 | task 3 | task 5 | 四任务等权均值 |
|---|---:|---:|---:|---:|---:|
| `final_lr1e-3` | 0.07056848717620597 | 0.03756627095863223 | 0.06324082425562665 | 0.05272268553962931 | 0.05602456698252354 |
| `ordinary_lr1e-3` | 0.07688996323850006 | 0.04815320051275194 | 0.07744551125215367 | 0.04629440136486664 | 0.06219576909206808 |
| `scratch_lr1e-4` | 0.697299804724753 | 0.5150294010527432 | 0.7734444940462708 | 0.4554726143367589 | 0.6103115785401314 |
| `final_lr3e-4` | 0.06808910763356835 | 0.03689212269382551 | 0.06679480229504406 | 0.054146695462986824 | 0.056480682021356185 |

`final_lr1e-3` 相对 ordinary 的四任务 HR-development 均值下降 9.922221719631%（绝对差 0.006171202109544541），相对 scratch 下降 90.820333588208%。但逐任务相对 ordinary 的变化为 task 0/1/3 改善 8.2215%/21.9859%/18.3415%，task 5 **退化 13.8857%**；所以不能把 LR screen 的“四任务均改善”外推到 HR development，也尚未达到任务书 T2 的确认门槛。`final_lr3e-4` 的 HR-development 均值比 `final_lr1e-3` 高 0.814133983356%，与 LR-development 排名方向一致，但个别 task 排序不同。

这些只是四个已筛选 family、单 model seed 和单 support seed 的 development 结果；未覆盖冻结开发 manifest 的六个 family，更不是新 confirmation query。当前也没有生成聚合 summary 文件，上表是本次从 16 个 task 报告重算所得。

### R2 `comm_lr1e-3_cw005_r1`

R2 trial 的四任务原始值为 0.02753081985283643、0.03356807844247669、0.02534724119468592、0.02549606805550866，重算均值 `0.027985551886376928`，与 summary 精确一致。相对相同 path-source 初始化、`lr=1e-3/cosine100` 的 `final_lr1e-3`，均值高 `0.000069518308737318`，即 **+0.249026454793%（更差）**；相对当前最佳 true-path FT ratio .05 也高 0.208591436765%。逐任务相对 final-only 的变化为 task 0 +0.6202%、task 1 -0.5101%（改善）、task 3 +0.4027%、task 5 +0.7062%。因此当前 one-step、128 calibration、weight .05 的 commutator arm 没有形成平均收益。

R2 YAML 文件 SHA-256 为 `8ec11a6605021b9e922faf517d4699bcb09b80c26acc91d4e354695ecc51de5b`，与 descriptor 的 `trial_spec_sha256` 一致；descriptor/config hash 为 `6d1502f8966f...`，summary SHA-256 为 `e591f3ca5655880d4150a25172d1d54a30d33ed990eff7e7a0d937c96d09fad9`。descriptor 记录的 runner/engine/adapter SHA-256 `486ca3b7...`/`d0b5109e...`/`8f7e91a3...` 均与当前对应文件一致，source checkpoint hash `a85ed618...` 也一致。四个 task 的 compact checkpoint、support index 和 LR-validation index whole-file hash 均与当前文件一致，`query_metric=null`；summary 的 `query_opened_or_loaded=false`。

### 哈希链与 query 纪律

- 对上述 8 个新增 R1 完整 trial，共核对 8 个 config↔summary descriptor hash、32 个 task descriptor hash、32 个 compact checkpoint 文件 hash：72/72 通过；32 个 task 的 `query_metric` 均为 `null`，8 个 summary 的 `query_opened_or_loaded` 均为 `false`。所有 trial 绑定冻结 R1 YAML hash `d829bcef...` 和 source checkpoint `a85ed618...`；scratch 的 source checkpoint/hash 均为 null。
- 对 16 个 HR-development 报告，`query_read=false` 和 `training_or_within_task_checkpoint_selection_used_hr=false` 均为 16/16；checkpoint 与 task-result whole-file hash 共 32/32 通过。四个 HR index 的报告 hash 是 index 内嵌 canonical content digest（task 0/1/3/5 为 `2faa...`/`558c...`/`e04b...`/`ca86...`），均与当前 index 内嵌 seal 一致；它们不是 whole-file SHA-256。task manifest 同理记录 canonical unsigned digest `bb50...`，whole-file digest 是 `d101...`。
- HR evaluator 验证 index role=`validation_hr`、fidelity=1、256²、20 steps、8 rollouts/task 和 frozen task manifest，并只以已由 LR validation 选定的 checkpoint 推理。未发现 query/test 指标或资产进入这些代码路径。
- 缺口：HR-development 报告未记录 evaluator/rollout 代码 hash、git commit、elapsed time 或 GPU-hours。因此数据/权重/task-result 哈希链闭合，但执行代码 provenance 不完整；`BUDGET.md` 所称“HR development inference overhead is tracked with its reports”目前不可由这些报告精确复算。后续报告应加入 evaluator 与 rollout module hash、开始/结束时间及同步后的 GPU 占用时间。

### Ledger 终态 GPU 小时

截至本节快照，所有非 RUNNING 终态行的精确和为：

| 状态 | 行数 | GPU-hours |
|---|---:|---:|
| DONE | 63 | 1.791109441272805 |
| FAILED | 3 | 1.950514362667519 |
| INTERRUPTED | 3 | 3.678137333205526 |
| INVALID_OUTPUT | 1 | 0 |
| **合计** | **70** | **7.419761137145857** |

其中 R2 commutator smoke 为 0.002843671386945 GPU-hours，正式四任务合计 0.096200467431667；随后已终态的 direct256 smoke 为 0.004820272182500。正在运行的 direct256 正式 task 没有终态 elapsed 值，按 append-only 终态口径未计入，也未等待其完成。

`BUDGET.md`/`STATE.md` 仍停在 `6.917708176803`，比当前终态 ledger 少 `0.502052960342857` GPU-hours；按 ledger 的当前 exploration balance 应为 `64.580238862854143` GPU-hours。该差额主要包括后续 R1 final/path/scratch、R2 commutator 与 direct256 smoke，不能继续沿用旧余额。另有 HR-development inference 未提供可复算时间，故上述 7.419761137145857 仍不含无法从报告精确恢复的推理开销。

---

## R2 direct256 与带计时 HR-development 复核（2026-10-13 18:39 快照）

本节为只读审计。未运行训练、生成、GPU 或 query/test；审计期间 R3 layer-3 path-FT 正式 trial 正在运行，本节不等待、不读取其中间结果，也不将其未终态时间计费。

### LR-development 四族完整性

`comm256_lr1e-3_cw005_r2` 已完整覆盖冻结 screen 的四个 family：task 0 high-frequency、task 1 amplitude-OOD、task 3 localized-bump、task 5 piecewise-smooth。四个 LR-development 值为：

| task / family | LR rollout relative-L2 |
|---|---:|
| 0 / high-frequency | 0.027716420806245878 |
| 1 / amplitude-OOD | 0.03350050075678155 |
| 3 / localized-bump | 0.025499118183506652 |
| 5 / piecewise-smooth | 0.025723299192031843 |
| **四任务等权均值** | **0.02810983473464148** |

均值从四个原始 task JSON 重算后与 summary 精确一致；LR tail 是 amplitude-OOD `0.03350050075678155`。相对 R1 `final_lr1e-3` 的 LR 均值 `0.02791603357763961`，direct256 高 0.694228843302%（更差）；相对 R2 64→128 one-step arm 的 `0.027985551886376928` 也高 0.444096470812%。因此它的价值来自独立 HR-development 表现，而不是 LR selection 指标改善。

### HR-development 重算、相对变化与 tail

四个报告各含相同冻结 task 上的 8 个 256²、20-step HR-development rollout。逐报告重新平均 `per_rollout_rel_l2` 后，4/4 与其 `metrics.rollout_rel_l2` 精确一致：

| task / family | direct256 | R1 final-only | ordinary transfer | direct256 vs R1 | direct256 vs ordinary |
|---|---:|---:|---:|---:|---:|
| 0 / high-frequency | 0.06769019482890144 | 0.07056848717620597 | 0.07688996323850006 | -4.0787219090% | -11.9648495358% |
| 1 / amplitude-OOD | 0.03647126234136521 | 0.03756627095863223 | 0.04815320051275194 | -2.9148717435% | -24.2599412853% |
| 3 / localized-bump | 0.05997935163322836 | 0.06324082425562665 | 0.07744551125215367 | -5.1572266187% | -22.5528366157% |
| 5 / piecewise-smooth | 0.0473763178801164 | 0.05272268553962931 | 0.04629440136486664 | -10.1405450136% | **+2.3370353290%** |
| **四任务等权均值** | **0.052879281670902854** | **0.05602456698252354** | **0.06219576909206808** | **-5.614118021835%** | **-14.979294503737%** |

绝对均值差为：相对 R1 `0.0031452853116206853`，相对 ordinary `0.009316487421165226`。direct256 对 R1 的四个 family 全部改善；相对 R2 64→128 one-step arm 的 HR 均值 `0.0543033644760726` 下降 2.622457777542%，四个 task 也分别改善 1.9670%、1.0273%、2.8186%、4.4761%。

HR tail 仍是 high-frequency `0.06769019482890144`，但比 R1 的同族值低 4.0787%。必须同时保留的负面 tail 是：piecewise-smooth 虽较 R1 改善，却比 ordinary 高 2.3370%；所以“全四族改善”只能明确写成“相对 R1 final-only 全四族改善”，不能写成相对 ordinary 全胜。

### 14.9793% 的统计边界

`14.979294503737%` 是 **seed 42、单一 support seed 271829、m=4、四个 development family** 上相对当前 ordinary 候选的描述性均值下降。它不能宣称确认 T2 达到，理由同时包括：

1. 精确改善 `14.9792945%` 本身略低于预设的 `≥15%` 门槛；不能通过四舍五入写成达到 15%。
2. 这是 HR development，不是冻结 confirmation query；只覆盖四个 screen family，未覆盖完整六族或新的未见确认边界。
3. 只有一个完整 model seed 42 和一个 support draw，没有任务书要求的 5 model seeds × ≥3 support seeds，也没有配对 95% CI。
4. ordinary baseline 仍是当前开发候选，不是已完成全部预分配调优和确认的强基线；且 piecewise-smooth 上 direct256 仍退化 2.3370%。

当前可支持的表述是：“在 seed-42 四族 HR-development 上，direct256 相对 R1 final-only 均值下降 5.61%，相对 ordinary 均值下降 14.98%，并相对 R1 四族均改善。”不得写成“确认 T2 已达到”或“确认集强迁移优势 ≥15%”。

### 数据、权重与描述符哈希链

- direct256 summary SHA-256 为 `e6d081ffed582a0eb07f2e15c6ab9fa2fdb9c62c7201aa1c4757a6d1fc32f2e0`；descriptor/config hash `af78ef97...` 在 config、summary 和四个 task 输出中一致。冻结 YAML SHA-256 `8a1636e8336a210a739a7bd104bd9866b3bd37a31dff836fac425ecef16f249a` 与 descriptor 的 `trial_spec_sha256` 一致。
- 四个 task 的 compact checkpoint、support index、LR-validation index whole-file hash 共 12/12 与当前文件一致；四个 HR 报告的 checkpoint、task-result whole-file hash及 HR-index canonical seal 共 12/12 一致。四个报告使用与 R1/ordinary 相同的 per-task HR index，因而 task/initial-condition 比较是配对的。
- 训练 descriptor 记录 runner `486ca3b7...`、engine `d0b5109e...`、adapter `8f7e91a3...` 和 source checkpoint `a85ed618...`。审计时当前 runner/engine 已因后续 R3 工作变为其他 hash，但保留的 git blobs `b9c50844...`/`34cfa37c...` 可重建字节 SHA-256 `486ca3b7...`/`d0b5109e...`，所以 direct256 的历史代码 provenance 仍闭合，不能用当前工作树 hash误判为资产漂移。
- 四个训练 task 的 `query_metric=null`，summary `query_opened_or_loaded=false`；四个 HR 报告 `query_read=false` 且 `training_or_within_task_checkpoint_selection_used_hr=false`。未发现 query/test 进入训练、checkpoint 选择或本次开发评估。
- HR 报告记录 evaluator SHA-256 `82011cb1...` 和 rollout module SHA-256 `d656f053...`，均与审计时文件一致，修复了早期 HR 报告缺执行代码 hash 的缺口。

### HR 计时 scope 与 ledger 去重

四次 HR-development 计时分别为 2.6771711330、2.7030446580、2.6884545980、2.7019881480 秒，合计 `10.770658537003328` 秒，即 `0.002991849593612` GPU-hours；每条 ledger 值都精确等于报告 `elapsed_seconds/3600`，4/4 匹配。四条 HRDEV terminal trial ID 和输出文件均唯一，没有与训练行或彼此重复收费。

计时实现先 `torch.cuda.synchronize()` 并 reset peak memory，再开始墙钟；结束前再次 synchronize。计时范围包括 task-result/index/manifest 校验、checkpoint hash、模型加载、数据读取以及未预热的 8-rollout HR-development 推理；不包括 Python import/CLI 初始化、计时前 CUDA context 初始化、报告 durable write 和 ledger append。每次 peak allocation 均为 1,576,464,384 bytes。`gpu_hours=elapsed wall time/3600` 表示该已同步评估 scope 的设备占用时间，不是由 SM utilization 积分得到的能耗/忙碌时间。

因此这组计时适合预算收费和复核本次开发评估开销，但**不是** Stage 3 所需的预热、多次重复、纯 deployment inference benchmark，不能直接代入 `C_inferHR` 或盈亏平衡公式。早期 R1/ordinary/R2-one-step 的 HR-development 报告没有计时，当前也没有补造或重复计费；它们的小额开销仍作为已披露、不可精确复算的历史 overhead。

direct256 正式训练四任务 ledger 合计 `0.494812728538332` GPU-hours，direct256 smoke 为 `0.004820272182500`，带计时 HR-development 四任务为 `0.002991849593612`；三者是不同 scope，没有重复。审计还检查了所有同 trial-ID/status 的 terminal 行：唯一历史重复仍是此前已标记 INVALID_OUTPUT 的 scratch smoke，与 direct256/HRDEV 无关。

截至 18:39 快照，终态 ledger 总和为 `7.920229697540023` GPU-hours（DONE 72 行 `2.291578001666972`，FAILED 3 行 `1.950514362667519`，INTERRUPTED 3 行 `3.678137333205526`，INVALID_OUTPUT 1 行 0）。该总额包括已终态 R3 smoke `0.002663982262222`，不包括当时运行中的 R3 正式 task。`BUDGET.md` 的 `7.667461615332` 明示为 18:29、仅到 direct256 task 0/1 的快照，不是重复收费错误；相对本审计快照少 `0.252768082208023`，对应当前 exploration balance 为 `64.079770302459977` GPU-hours，另仍需披露早期未计时 HR overhead。

## 2026-10-13 R3 layer mapping and R2 H3 rollout audit

Read-only audit of the latest R3 layer-3 accepted-path screen and R2 fixed-H3 rollout screen; no training, evaluation, tests, or query data were opened.

- **R3 A1 layer 3:** Recomputed from the four task outputs, LR-validation mean is `0.027998546817980238` (tasks 0/1/3/5: `0.0274400254 / 0.0337812933 / 0.0252729762 / 0.0254998924`), exactly matching `summary.json`. It is 0.30% worse than R1 final-only `0.02791603` and 0.26% worse than the R1 layer-1 accepted-path screen `0.02792730`; this seed-42 screen does not show a layer-mapping gain. The summary records `query_opened_or_loaded=false`; all task `query_metric` values are null. Checkpoint, support-index, and LR-validation-index hashes match the four on-disk files (12/12). Descriptor/config/source-checkpoint fields agree across config, summary, and task results. No R3 independent HR-development reports exist yet.
- **R3 A2/replication gap:** The layer-3 endpoint control currently has only task 0 marked `RUNNING`; there is no summary or task-results file yet. Further endpoint-direction/path controls and paired HR-development checks remain outstanding. This is one model seed and one support draw, so any completed result remains exploratory, not confirmation. A1 records runner/engine hashes `f742cb0a…`/`55b1e103…`; current files have since changed, so retain the result's recorded code provenance and do not relabel it with working-tree hashes.
- **R2 H3:** Recomputed mean LR-validation rollout error is `0.026429235535033513` (tasks `0.0244942647 / 0.0338103163 / 0.0240077542 / 0.0234046070`), matching the summary and about 5.3% below R1 final-only. Recomputed independent HR-development mean from each report's eight per-rollout errors is `0.055966493886080576`, versus R1 final-only `0.05602456698252354` (about 0.10% lower) and R2 direct256 `0.052879281670902854` (about 5.84% higher). All four HR reports specify development-only HR data, `query_read=false`, and no HR-based checkpoint selection. The H3 outputs have matching on-disk checkpoint/index hashes and use the same task-specific HR-development indexes as the paired reference reports.
- **Fair-comparison gap:** `configs/phase1_r2_rollout.yaml` defines `ordinary_rollout3_lr1e-3_cos100` and `scratch_rollout3_lr1e-4_cos100`, but no completed outputs or ledger rows exist for those arms. Since H3 rollout supervision is a general training improvement, the HR evidence does not yet establish an advantage over the strongest baseline receiving the same improvement. The current H3 results support a development-screen observation only.
- **Budget reconciliation:** The current terminal ledger sum is `8.111946287553081` GPU-hours (DONE `2.4832945916800284`, FAILED `1.950514362667519`, INTERRUPTED `3.678137333205526`, INVALID_OUTPUT `0`), while `docs/BUDGET.md` still states the earlier `7.920229697540` snapshot at 18:39. The difference is `0.191716590013` GPU-hours; it comprises later terminal rows and excludes currently RUNNING tasks. Exploration headroom against 72 GPU-hours is `63.888053712447` before charging active jobs; the 48 GPU-hour reserve is untouched. Update the budget snapshot after current active tasks reach terminal status.

---

## R3 path/endpoint/direction controls and endpoint-L1 review (2026-10-13 late snapshot)

This was a read-only review. No training, generation, GPU process, or query/test
asset was started or opened. The requested HR-development batch had already
reached terminal `DONE` rows for all 16 task reports by the time of this audit;
the review therefore uses the completed files rather than extrapolating from an
in-progress subset.

### LR-development values from raw task results

The source directories are:

- `results/phase1/R3/path_l3_lr1e-3_ratio005_m4_s42_ss271829/`
- `results/phase1/R3/endpoint_l3_lr1e-3_ratio005_m4_s42_ss271829/`
- `results/phase1/R3/direction_l3_lr1e-3_ratio005_m4_s42_ss271829/`
- `results/phase1/R3/endpoint_l1_lr1e-3_ratio005_m4_s42_ss271829/`

For every arm, the task 0/1/3/5 values below were read from `task00.json`,
`task01.json`, `task03.json`, and `task05.json`. Each value has exactly one
matching `DONE` row with the same raw-output path and validation metric. The
unweighted means independently recompute to the saved summaries exactly.

| arm | task 0 | task 1 | task 3 | task 5 | four-task LR mean |
|---|---:|---:|---:|---:|---:|
| path L3 | 0.027440025389660148 | 0.03378129332559183 | 0.025272976187989116 | 0.025499892368679864 | **0.027998546817980238** |
| endpoint L3 | 0.027394687040941793 | 0.03379058443242684 | 0.02527489108615555 | 0.025361303746467452 | **0.027955366576497907** |
| direction L3, contaminated | 0.027392623655032367 | 0.03378633530810475 | 0.02527249978738837 | 0.025340516195865345 | **0.027947993736597707** |
| endpoint L1 | 0.027323787583736703 | 0.03372475238284096 | 0.025231312843970958 | 0.025455227564089 | **0.027933770093659403** |

Relative to R1 final-only LR mean `0.02791603357763961`, the path-L3,
endpoint-L3, direction-L3, and endpoint-L1 means are respectively 0.2955765%,
0.1408975%, 0.1144867%, and 0.0635352% higher (worse). All are lower than the
direct256 LR mean `0.02810983473464148`, but direct256 was already selected for
its HR behavior rather than its LR score. Endpoint L3 is 0.1542232% below path
L3, with mixed task directions; endpoint L1 is only 0.0772534% below endpoint
L3 and is worse on task 5. These sub-percent, single-seed LR differences are
not a promotion signal.

The direction-L3 mean is **not a valid formal-trial statistic**. Its task 0 has
a later append-only `EXCLUDED_RESOURCE_CONTENTION` row because it overlapped the
interrupted first-path job on the same physical GPU. Although its original
`DONE` row and summary remain preserved, the whole arm must stay excluded from
ranking until a fresh serial four-task attempt exists. The fact that tasks 1/3/5
were clean does not repair a predeclared four-task mean.

### Completed HR-development files

The 16 reports are under
`artifacts/reports/phase1_hrdev_evaluations/{path_l3,endpoint_l3,direction_l3,endpoint_l1}_lr1e-3_ratio005_task{00,01,03,05}_s42_ss271829.json`.
Re-averaging each report's eight `per_rollout_rel_l2` values reproduces all 16
reported task metrics. The four-task results are:

| arm | task 0 | task 1 | task 3 | task 5 | four-task HR mean | vs R1 | vs direct256 | vs ordinary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| path L3 | 0.07046930438373238 | 0.037628301035147166 | 0.062055647175293424 | 0.050090377102606005 | **0.05506090742419474** | -1.7201% | +4.1257% | -11.4716% |
| endpoint L3 | 0.07053278799867259 | 0.037631486705504355 | 0.06203592764213681 | 0.049890035972930485 | **0.05502255957981106** | -1.7885% | +4.0532% | -11.5333% |
| direction L3, contaminated | 0.07052015593508258 | 0.037620715657249094 | 0.06201707634609192 | 0.052109041204676035 | **0.05556674728577490** | -0.8172% | +5.0823% | -10.6583% |
| endpoint L1 | 0.07057485375553370 | 0.03756344172870740 | 0.06335286794928834 | 0.052815040817949924 | **0.05607655106286984** | +0.0928% | +6.0464% | -9.8386% |

Here a negative table entry means lower error than the named comparator. The
reference means are R1 `0.05602456698252354`, direct256
`0.052879281670902854`, and ordinary `0.06219576909206808`.

Endpoint L3 is only `0.000038347844383682` (0.0696462%) below path L3 in HR,
again with mixed per-task directions; this does not support a path-geometry
advantage. Endpoint L1 is 1.9155624% worse than endpoint L3 in HR despite its
slightly lower LR mean, including 2.1229% and 5.8629% regressions on tasks 3 and
5. All three valid arms are worse than direct256; endpoint L1 is also slightly
worse than R1. The direction HR mean is descriptive only because its task-0
checkpoint is exactly the resource-contention-excluded checkpoint. Its four
HRDEV ledger rows currently say `DONE`, but those rows do not restore eligibility
of the underlying model arm and must not enter an aggregate ranking.

### Mechanism-screen decision and risks

- The valid path-L3/endpoint-L3 pair is worth retaining as a **negative
  mechanism control**: endpoint is marginally better in both LR and HR, and the
  effect is tiny and mixed by task. It provides no evidence that the selected
  Newton-path target adds useful information beyond the endpoint target.
- None of path L3, endpoint L3, or endpoint L1 merits performance promotion:
  all lose to direct256 HR development by 4.05%--6.05%. Direct256 remains the
  development leader among these comparators.
- Endpoint L1 is useful only as a mapping control. Its LR/HR ranking reversal
  and task-3/task-5 HR regressions are a warning against screening mechanisms on
  LR error alone.
- Direction L3 must be rerun serially before any mechanism statement. Evaluating
  the contaminated checkpoint on HR created a reproducible report, not a valid
  experiment. No mean or apparent comparison from that arm should guide the
  next mechanism choice.
- These are still seed 42, support seed 271829, m=4 development screens. They
  neither establish confirmation performance nor alter the prior conclusion
  that the 14.9793% direct256-vs-ordinary development improvement does not
  confirm T2.

### Provenance and leakage checks

Summary whole-file SHA-256 values are path L3
`38447a38a4c377f26e5df7ab35c849aee69294b9e2da90e20f064117a8c93aa3`,
endpoint L3 `29ceec7cc57d1632e93461edc4da9e8c3247ebacbd85c80dd0e0d2322144bf03`,
direction L3 `a0320f91dfe14ca5f7e3677a79290efbd53f8b3932056fdff1a27e569ce80f10`,
and endpoint L1 `7f7025b9cff6ae831e7a3fc2451258a96fbd6673c75d09103f7d7ab4fb810df7`.
Their descriptor/config hashes are respectively `c982d68f...`, `b7276d30...`,
`5eca80b8...`, and `6bf40155...`.

The common config file `configs/phase1_r3_path_controls.yaml` has SHA-256
`3da829a7b853890e13055bbede900aaaff36eb2b8e053f5f32f4be6798c92f73`,
matching every descriptor's `trial_spec_sha256`. Path L3 records runner/engine
hashes `f742cb0a...`/`55b1e103...`; endpoint L3 records
`f8c86eb0...`/`4c38203e...`; direction L3 and endpoint L1 record
`f8c86eb0...`/`1fbbd244...`. The current runner and engine exactly match the
latter hashes, and adapter SHA `8f7e91a3...` is common and current. The path-L3
engine is reproducible from recorded git commit `6ccea77` (`55b1e103...`), but
its recorded runner `f742cb0a...` is not the runner at that commit or the current
runner; retain the recorded runtime hash as a provenance caveat. More broadly,
the path/endpoint/direction arms were not all run under one engine hash, so the
very small A1/A2/A3 differences are not a strictly code-identical contrast.

Across the four arms, 16/16 task descriptor hashes agree with their summaries,
16/16 compact checkpoint hashes verify, and all task-wise support and LR-index
canonical seals verify. The arms use identical per-task support and validation
indices. All 16 task outputs have `query_metric=null`, and all four summaries
have `query_opened_or_loaded=false`. For the 16 HR reports, checkpoint and task-
result whole-file hashes, HR-index canonical seals, eight-rollout means, and
`query_read=false` all verify. The HR reports use the same frozen per-task
HR-development indices as R1/direct256/ordinary. No query/test data was used or
opened in this audit.

---

## R3 first-accepted serial2 five-arm review (2026-10-14)

Read-only audit only: no training, generation, GPU work, or query/test access was
performed. The independent `direction_l3_lr1e-3_ratio005-serial2` retry was
still training during the audit and was not evaluated or included below. The
five completed arms reviewed here are the two first-accepted ratios, their two
same-data endpoint controls, and the first-norm endpoint-direction control.

### Raw LR/HR recomputation

The raw training outputs are in
`results/phase1/R3/<trial>_m4_s42_ss271829/{task00,task01,task03,task05,summary}.json`;
the HR reports are in
`artifacts/reports/phase1_hrdev_evaluations/<trial>_task{00,01,03,05}_s42_ss271829.json`.
For each arm, the LR mean below was recomputed from the four task JSON best
validation metrics, and the HR task values were first recomputed from each
report's eight per-rollout errors before taking the four-family mean. All five
LR means exactly equal their summaries, and all 20 HR task recomputations equal
their reports.

| serial2 arm | LR task 0 / 1 / 3 / 5 | LR mean | HR task 0 / 1 / 3 / 5 | HR mean |
|---|---|---:|---|---:|
| firstpath .05 | .0273102200 / .0337425439 / .0252774641 / .0253208766 | **.027912776126** | .0705823197 / .0375711563 / .0622014499 / .0517976526 | **.055538144617** |
| firstpath .15 | .0272936007 / .0340278570 / .0253606610 / .0254617890 | **.028035976936** | .0706229562 / .0375895703 / .0627809365 / .0496080202 | **.055150370795** |
| endpoint .05, firstpath data | .0273237876 / .0337247524 / .0252313128 / .0254552276 | **.027933770094** | .0705748538 / .0375634417 / .0633528679 / .0528150408 | **.056076551063** |
| endpoint .15, firstpath data | .0272755908 / .0340342475 / .0254093014 / .0254320630 | **.028037800660** | .0706080737 / .0375427640 / .0626472274 / .0499501458 | **.055187052739** |
| endpoint-direction first-norm .15 | .0273101271 / .0340197980 / .0253907562 / .0255208439 | **.028060381291** | .0706039410 / .0376075070 / .0622865744 / .0522427690 | **.055685197838** |

Reference means are R1 final-only LR/HR
`.027916033578/.056024566983`, progress-.25 path .05
`.027927297934/.055249937711`, progress-.25 path .15
`.028044917760/.055791560697`, progress-.25 endpoint .05
`.027933770094/.056076551063`, and direct256
`.028109834735/.052879281671`. The `.05` arms use progress-path .05 as their
ratio-matched path reference; all `.15` arms use progress-path .15. Only a .05
progress endpoint reference exists.

The following are percent changes `(serial2-reference)/reference`; negative is
better. Vectors are ordered task 0/1/3/5 and expose the family-level effects
rather than only the aggregate.

| arm | LR vs R1 final | LR vs matched progress path | LR vs progress endpoint .05 | LR vs direct256 |
|---|---|---|---|---|
| firstpath .05 | [-.186, +.007, +.126, +.014]% | [-.121, +.063, +.080, -.261]% | [-.050, +.053, +.183, -.528]% | [-1.466, +.723, -.869, -1.564]% |
| firstpath .15 | [-.247, +.853, +.456, +.571]% | [-.031, +.218, -.208, -.189]% | [-.110, +.899, +.513, +.026]% | [-1.526, +1.574, -.543, -1.017]% |
| endpoint .05 | [-.136, -.046, -.057, +.545]% | [-.071, +.010, -.103, +.268]% | [0, 0, 0, 0]% | [-1.417, +.669, -1.050, -1.042]% |
| endpoint .15 | [-.313, +.872, +.649, +.453]% | [-.097, +.237, -.017, -.306]% | [-.176, +.918, +.705, -.091]% | [-1.591, +1.593, -.352, -1.132]% |
| direction first-norm .15 | [-.186, +.829, +.575, +.804]% | [+.029, +.194, -.090, +.042]% | [-.050, +.875, +.632, +.258]% | [-1.466, +1.550, -.425, -.787]% |

| arm | HR vs R1 final | HR vs matched progress path | HR vs progress endpoint .05 | HR vs direct256 |
|---|---|---|---|---|
| firstpath .05 | [+.020, +.013, -1.644, -1.755]% | [+.091, +.045, -.112, +2.254]% | [+.011, +.021, -1.817, -1.926]% | [+4.273, +3.016, +3.705, +9.332]% |
| firstpath .15 | [+.077, +.062, -.727, -5.908]% | [+.008, -.010, +.173, -5.116]% | [+.068, +.070, -.903, -6.072]% | [+4.333, +3.066, +4.671, +4.711]% |
| endpoint .05 | [+.009, -.008, +.177, +.175]% | [+.081, +.024, +1.737, +4.262]% | [0, 0, 0, 0]% | [+4.262, +2.995, +5.624, +11.480]% |
| endpoint .15 | [+.056, -.063, -.939, -5.259]% | [-.013, -.134, -.041, -4.461]% | [+.047, -.055, -1.114, -5.424]% | [+4.311, +2.938, +4.448, +5.433]% |
| direction first-norm .15 | [+.050, +.110, -1.509, -.910]% | [-.019, +.038, -.616, -.076]% | [+.041, +.117, -1.683, -1.084]% | [+4.305, +3.115, +3.847, +10.272]% |

At the four-family mean level, firstpath .05/.15 improve on R1 HR by
0.8682%/1.5604%, endpoint .05 is 0.0928% worse, endpoint .15 improves by
1.4949%, and direction first-norm .15 improves by 0.6058%. Against direct256,
however, all five are worse by 4.2949%--6.0464%, and all four task families are
worse for every arm. The apparent gains over R1/progress are primarily from
tasks 3 and especially 5; tasks 0/1 are essentially flat or slightly worse.

The clean within-serial2 mechanism contrasts are small or adverse:

- At ratio .05, firstpath is 0.0752% lower in LR and 0.9601% lower in HR than
  endpoint, but the HR task effects are +0.0106%, +0.0205%, -1.8175%, -1.9263%:
  the mean benefit is confined to tasks 3/5 rather than uniform.
- At ratio .15, firstpath is only 0.0065% lower in LR and 0.0665% lower in HR
  than endpoint. HR task changes are +0.0211%, +0.1247%, +0.2134%, -0.6849%,
  so task 5 alone reverses three small regressions.
- Endpoint-direction first-norm .15 is 0.9026% worse than endpoint .15 and
  0.9698% worse than firstpath .15 in HR, driven by a 4.5898%/5.3111% task-5
  regression. It does not isolate a beneficial direction-only mechanism.
- Endpoint .05 reproduces the earlier progress endpoint task metrics bit-for-
  bit in both LR and HR despite pointing at the regenerated firstpath data root.
  This is an expected deterministic control because endpoint supervision does
  not use the first-accepted label; it also validates the matched rerun.

Thus the serial2 evidence does not establish a robust first-accepted geometry
advantage. Ratio .15 firstpath is the best of these five HR means, but it is
nearly tied with endpoint .15, loses on three of four paired tasks, and remains
4.2949% worse than direct256. This is mechanism-development evidence only, not
a performance promotion or confirmation result.

### Hash chain and leakage discipline

Summary whole-file SHA-256 values are, in table order:
`e69aebdfd6be140c903bf04ac5b9e1c494f85b38d7f8e329a6055ce9470f2ad8`,
`91b3d8b78b5b030ae23c91fdb3c6780f8544407592d3b4786848812292982630`,
`2ca5ee4f4358eb8fd9f5436901bd1468532cc0d877174cc85ff2e0026f3233e4`,
`624a526d437b7250136ed7e46e8c6ee87cbfec7e3849469b752dc1866cd24808`,
and `bc57bffefd35a0216289461869372328fd5355ace649a6b2bb8d30c23290e6f2`.
Their descriptor hashes are respectively `adf5c0ff...`, `d71403f4...`,
`97cf4ff5...`, `10d0b898...`, and `9a3a5d6a...`.

The common config SHA is
`67c4ee911c5d436f0e07bbcb25349deda9d4fb8e5c617b430457b838681c5bd4`;
all five descriptors bind it and identically record runner
`767504eec62b80c912db6d0f288ec0a291bf6b066366e9f627530a3738e84a15`,
engine `1fbbd244332448762dfa887dfdd4afdf9b18dae7c0c061ccd5c869220e44e73d`,
adapter `8f7e91a3465a1f88769726610ad1bb29ca137c3a6fb91f654a51a88af08a9137`,
and source checkpoint `a85ed6185c26f730d6096e94ca37d6905ddb3c5f64194066ae01c111881d5c87`.
The runner/engine/adapter hashes match the audited working-tree files.

HR report whole-file SHA-256 values, ordered task 0/1/3/5, are:

- firstpath .05: `1260e113e9a2be5e3d48cfe81776494433912b8f98084b312a1d02d9f1e41343`,
  `3987469b9078adf9879d6e67749bde1530c15c988e2aa750651f7ca51f87b00e`,
  `78a34d9b2eff039f0a50de8874d196246ccdd1d20fb270fe6fb642ba9beebc46`,
  `e6cc6a6c27fa62d75cbbf40e574b36c042e0cb66cabe6a833b4f9ed39580a938`;
- firstpath .15: `4fcbe0951d22cd725fc50c3382b368b9abe82406a87b61c7fe7bee1754cd4545`,
  `fb5f69bc5833fa9cf33ceaf4959cece09e0ced65f46318322c13056887923ed4`,
  `582d0fed5ef2ba5c88232a403016c5d2bd6a7cfe1c3615b9d415ff9100e9734d`,
  `413955dac73b4045b197d8fab2bca41dd9875b8cccef401ef40e490d9417dd09`;
- endpoint .05: `2661fb524f5f32759b8f4d7193b018d00d8c77ac8411ccde7ec29bb70b05374b`,
  `1eea6adc59a16d79330ee49b06d271b3e257a44401203bac3b0ba92e3599e8ce`,
  `7c41771e9c940b83e0bd150959562fed403d447687c4e6f857477e1224bf7af0`,
  `2ad28a4c4e2a015d4a9d04e1f7c6854eb85a19e53ffafb955f21599df6707dfe`;
- endpoint .15: `6ec49d405a588ae932a95b7939071d878cd1cb51e36dc2833b47b1a3911388f5`,
  `7a6a5b750fd8601bdec126d88f056162580e01026e40a1f2da6db4bdc1f09501`,
  `cb207ef075e5d45213dd476d3de6d8ef6745ecac50e0dbccf42cd70023014a0e`,
  `8344f6cab8c9003ac323851efbb3916f3e3299832b44c9740e7618bfa1945497`;
- endpoint-direction first-norm .15:
  `60880615858d09110223e128cbf4ba07111d6b3a941115be433d5c73dade3f29`,
  `bd9a76b6620fbe33615a5feb2233dc0bb98142ca355ae0b172ab73fd2457cfe2`,
  `c0b784e089415d1eac201044e00a987cfa95253b25c8b0da2683559aaa0f9e95`,
  `ae5aadae43dfad23d01e80c97c31573e569e8d8c788480435340e0dae8cacfb2`.

For all 20 reports, checkpoint whole-file hash, task-result whole-file hash, and
HR-index canonical seal verify (60/60); the embedded eight-rollout mean verifies
(20/20); `query_read=false` and
`training_or_within_task_checkpoint_selection_used_hr=false` verify (40/40).
All 20 training task JSONs have `query_metric=null`, and all five summaries have
`query_opened_or_loaded=false`. No query/test file was read. These results use
only model seed 42, support seed 271829, m=4, and four development families;
there are no replicate seeds, support draws, confidence intervals, or
confirmation-query evidence here.

---

## Clean direction-L3 serial2 review (2026-10-14)

This is a read-only audit of
`direction_l3_lr1e-3_ratio005-serial2_m4_s42_ss271829`; no experiment setting,
GPU process, or query/test asset was accessed or changed.

### Independent LR and HR recomputation

The four task JSON values are task 0/1/3/5
`0.027392623655032367 / 0.03378633530810475 / 0.02527249978738837 /
0.025340516195865345`. Their unweighted LR-development mean is
**`0.027947993736597707`**, exactly equal to `summary.json`. The corresponding
HR-development task values, each independently recomputed from eight rollout
errors, are `0.07052015593508258 / 0.037620715657249094 /
0.06201707634609192 / 0.052109041204676035`, giving the four-family mean
**`0.05556674728577490`**.

| paired reference | LR task changes, 0/1/3/5 | LR mean change | HR task changes, 0/1/3/5 | HR mean change |
|---|---|---:|---|---:|
| path L3 .05 | -0.1727% / +0.0149% / -0.0019% / -0.6250% | **-0.1806%** | +0.0722% / -0.0202% / -0.0622% / **+4.0300%** | **+0.9187%** |
| endpoint L3 .05 | -0.0075% / -0.0126% / -0.0095% / -0.0820% | **-0.0264%** | -0.0179% / -0.0286% / -0.0304% / **+4.4478%** | **+0.9890%** |
| direct256 | -1.1683% / +0.8532% / -0.8887% / -1.4881% | **-0.5757%** | **+4.1808% / +3.1517% / +3.3974% / +9.9896%** | **+5.0823%** |
| R1 final-only | +0.1152% / +0.1368% / +0.1066% / +0.0918% | **+0.1145%** | -0.0685% / +0.1449% / -1.9351% / -1.1639% | **-0.8172%** |

Changes are `(direction-reference)/reference`, so negative is lower error. The
direction target is marginally lower than endpoint/path on LR, yet is about
0.92%--0.99% worse on HR. Against both same-layer controls, tasks 0/1/3 are
essentially tied while task 5 degrades by 4.03%--4.45%; the aggregate difference
is therefore neither broad nor consistent with the LR ranking. Direct256 is
better on every HR family and by 5.08% in the mean, with its largest advantage
on task 5. The clean direction arm is 0.82% below R1 final-only HR, but that does
not make it competitive with the current development leader.

The clean serial2 run exactly reproduces all four LR and all four HR metrics of
the earlier resource-contention-excluded direction arm. It does **not** reuse
the old files: all four serial2 task-result and checkpoint hashes differ, and
the HR reports bind the serial2 descriptor and task-result hashes. This clean
replication makes the numerical result rankable while leaving the earlier arm
excluded.

### Integrity and provenance

- Summary SHA-256 is
  `1e4c1b6d1d302ccb7d9376d74386a28aac6d1a63cbab7681639c8bf43ed63149`;
  descriptor/config hash is
  `3913158321cab54a2057f177f4c06131effa0859ee6473ab791574711db48dc9`.
  The four training task JSONs each have one matching terminal `DONE` ledger row
  with an identical validation metric.
- Checkpoint hashes, in task order, are
  `df8c4a7418f2fcee9e2677ae76eeeb9f3533c513ed37a14f45205cf467f05b8e`,
  `71942408cbdd71e7d26176419c164f3b0f03a59d36b4f482d65d2c0ca6f0e164`,
  `466a67fc45e051f8bd0a368c12a6af8508c7bbc4ff57f92c48ab44a33dec4b16`,
  and `950914989482bc55e3081890308e8d3326a64e2679f908eaf9300d9ebe147c32`;
  all four verify against the current files. All four task descriptor hashes,
  support-index canonical seals, and LR-validation-index canonical seals also
  verify.
- HR report whole-file hashes are
  `06549498db8a60851dfdc321b64e0f8e8163c6d6aa8dbc9d41ac3886e035b20c`,
  `471437cdb538bda1a9483ef5a925c36de5acedf6af668d1ed99c5cb4610513d8`,
  `f346fb96f6214f86761c827885f080aa4f79437fb6bc185220aee8aa477a68b0`,
  and `75d4cb275fc1c607dc2d3b74fbe692817c13c075c563c9fe765e06cd38d9868f`.
  Checkpoint, task-result, and HR-index hashes/seals verify 12/12; report means
  verify 4/4. All reports have `query_read=false` and
  `training_or_within_task_checkpoint_selection_used_hr=false`; all training
  tasks have `query_metric=null`, and the summary has
  `query_opened_or_loaded=false`.
- The descriptor records runner `767504ee...`, engine `1fbbd244...`, adapter
  `8f7e91a3...`, source checkpoint `a85ed618...`, and config YAML
  `3da829a7...`; each matches the audited file. Evaluator `82011cb1...` and
  rollout module `d656f053...` also match current files. Four timed HR reports
  total `10.808318431998487` seconds or `0.003002310675555135` GPU-hours, and
  each ledger charge equals its report time divided by 3600.

No broken asset/hash chain or leakage flag was found. Two provenance limits
remain: the descriptor's `git_commit=6ccea77` is not a sufficient description
of the later runtime tree without the recorded file hashes; and the historical
path-L3/endpoint-L3 comparators were run under different runner/engine hashes.
Thus their sub-percent contrasts with clean direction are useful exploratory
development comparisons, not a strictly code-identical ablation. This is still
one model seed, one support seed, m=4, and four HR-development families, with no
replicate confidence interval or confirmation-query evidence.

---

## R2B H3 + direct256-commutator paired review (2026-10-14)

This section independently reviews the completed serial1 path-source and
ordinary-source arms using development assets only. No settings were changed
and no query/test data was opened.

### Recomputed task and mean results

Each HR task metric below was recomputed from the report's eight
`per_rollout_rel_l2` values before taking an unweighted four-family mean.

| arm | task 0 high-frequency | task 1 amplitude-OOD | task 3 localized-bump | task 5 piecewise-smooth | HR mean |
|---|---:|---:|---:|---:|---:|
| R2B pathsource H3+comm256 | 0.06836318153655158 | 0.034265823266468945 | 0.06209227186627686 | 0.050672593386843795 | **0.05384846751403529** |
| R2B ordinary H3+comm256 | 0.07468463547993451 | 0.04103024525102228 | 0.06834128308109938 | 0.050270848209038384 | **0.058581753005273636** |
| direct256 | 0.06769019482890144 | 0.03647126234136521 | 0.05997935163322836 | 0.04737631788011640 | **0.052879281670902854** |
| R2 fixed-H3 | 0.07060510865412653 | 0.03474461433943361 | 0.06408738776808605 | 0.054428864782676105 | **0.055966493886080576** |
| R1 ordinary | 0.07688996323850006 | 0.04815320051275194 | 0.07744551125215367 | 0.04629440136486664 | **0.06219576909206808** |

Taskwise percent changes `(R2B-reference)/reference` are:

| R2B arm / reference | task 0 | task 1 | task 3 | task 5 | mean change |
|---|---:|---:|---:|---:|---:|
| pathsource vs direct256 | +0.9942% | **-6.0471%** | +3.5227% | +6.9576% | **+1.8328%** |
| pathsource vs fixed-H3 | **-3.1753%** | **-1.3780%** | **-3.1131%** | **-6.9012%** | **-3.7845%** |
| pathsource vs R1 ordinary | -11.0896% | -28.8400% | -19.8246% | **+9.4573%** | **-13.4210%** |
| ordinary vs direct256 | +10.3330% | +12.5002% | +13.9414% | +6.1097% | **+10.7839%** |
| ordinary vs fixed-H3 | +5.7779% | +18.0910% | +6.6376% | **-7.6394%** | **+4.6729%** |
| ordinary vs R1 ordinary | -2.8682% | -14.7923% | -11.7557% | **+8.5895%** | **-5.8107%** |

The two R2B arms form the properly matched initialization comparison: identical
data, task/support draws, H3 objective, commutator resolution/weight, schedule,
runtime code, and evaluation indices, differing in source initialization.
Pathsource is 8.0798% lower in mean HR error than matched ordinary and improves
tasks 0/1/3 by 8.4642%/16.4864%/9.1438%, but is 0.7992% worse on task 5. Thus
the source advantage under the combined recipe is substantial on the mean but
still not uniform across families.

For context, the LR-development means from the four raw task JSONs are
`0.026608289874275215` for pathsource and `0.032575267943320795` for ordinary,
each exactly matching its summary. Pathsource is 18.3175% lower than matched
ordinary in LR. However, pathsource's HR mean remains 1.8328% above direct256
despite an LR mean 5.3417% below direct256, another clear LR/HR ranking
reversal. The combined recipe does improve over fixed-H3 on every HR task and
by 3.7845% overall, but it does not improve the direct256 HR leader.

### Integrity checks

The pathsource and ordinary summary SHA-256 values are respectively
`f3d075858b00f2f466a00e6617598327bc380f42dd6d66be891a088e5ec2f3a7`
and `5dab346e831940245fd9fcc53d1bb46ea3759acd120e07076da191d3a0b308c0`;
descriptor hashes are `63313abd...` and `a1c41676...`. Both bind config
`configs/phase1_r2_h3_comm256.yaml` SHA-256
`eb30c183807ed4c1ca8771dd8f5696e49a0ef01ed558756daaa5942feaf4a354`
and the same runner/engine/adapter hashes `767504ee...`/`1fbbd244...`/
`8f7e91a3...`. Their intended initialization hashes differ: pathsource
`a85ed618...`, ordinary `0bcefb09...`.

Pathsource checkpoint hashes, task 0/1/3/5, are
`b4564790ec22e0897f6bb8bc076f9fbb18300c0af730f33f21cefb578dda2b60`,
`2b9bcab1d693ddd677b5a4d6ae371f2d246874b9f20a6e1edbca91cef8b68530`,
`0f6b71263d06d39aaf46f76a029d89dbc3584a8c21fe13284ddd43a2b3c355e5`,
and `efc140c200ef3533c13e11d4be05893fbcde30ef8fae8dcf29ca9d05e53e2398`.
Ordinary checkpoint hashes are
`1eb8760a16c60b1e757fb3ceead2db8dce0e1dda0789ce342a590f5a171cd9d3`,
`20c5527becb4f0a90a52ff7f5bcbd0ebec190df4cfae0da053066e1e36c6f437`,
`5679e90382cd40d8f5abfa6263005be5751e802582d44b1b9cdc548ae65f7349`,
and `14e26a4258c3f8f1ff107506be39a2fe8942bb88e95eaed711cf642cbcdf6570`.
All eight verify against disk.

HR report whole-file hashes are:

- pathsource: `2d504ca2d5625e9ead6f2cee67777723257f1e8aeecc72bbc19ce4b5898d25d1`,
  `5d8cbc4e53d30bfba855d5a25341e1616781adc37338092f8b3e13a9de34ec33`,
  `264b78dd8162cbfcc05190add50b31e33dce4279912f9ec41065253ccb2044c8`,
  `bcbc4770da240ef07bfc31fa6ae60da0ec4b177c932fb7aaa10ed91939ab5941`;
- ordinary: `c5cd373d08c8b11926cebf54931c4f16c5010089b3592b5b89f607b594ca40b9`,
  `3eb9fb787389d124ea84150f51df67be2d55ad43fb0a670404a196b7c3f0f16d`,
  `7b2af039e3330617d6078f9c78f35a3a57a1874f0af765d3469d4745ad717d4b`,
  `e9281b7489d91c33187d127f69abd1d287e253eec051aba8047f6767ed763cf8`.

Across the eight reports, checkpoint whole-file hash, task-result whole-file
hash, and HR-index canonical seal verify 24/24; per-rollout means verify 8/8;
`query_read=false` and
`training_or_within_task_checkpoint_selection_used_hr=false` verify 16/16.
All eight training tasks have `query_metric=null`, and both summaries have
`query_opened_or_loaded=false`. Evaluator `82011cb1...` and rollout module
`d656f053...` match current files. Timed HR evaluation totals are
`10.841587937997247 s = 0.0030115522049992353 GPU-h` for pathsource and
`10.90819247699983 s = 0.003030053465833286 GPU-h` for ordinary; all ledger
charges exactly equal report seconds/3600. No broken hash chain or leakage flag
was found.

### Follow-up decision

The pathsource combined arm should enter a **limited replicated development
comparison** against direct256 and fixed-H3 because it improves fixed-H3 on all
four families and the matched ordinary arm by 8.08% on the mean. It should not
replace direct256 or advance directly to confirmation: direct256 remains 1.83%
better overall and wins three of four tasks, while all evidence is one model
seed, one support seed, m=4, and four development families.

The ordinary combined arm has completed its role as the fair matched baseline.
It should remain in any paired replication needed to estimate the source-
initialization effect, but it should **not** be promoted as a performance
candidate: it is 10.78% worse than direct256 and 4.67% worse than fixed-H3.
Neither arm supports a confirmation or T2 claim; task 5 remains the explicit
negative family for both comparisons with R1 ordinary and for the pathsource-
versus-matched-ordinary source advantage.

## Ordinary direct256 baseline interim audit (2026-10-14 01:32 CST)

Independent read-only audit of the active arm
`ordinary_comm256_lr1e-3_cw005_r2-baseline1_m4_s42_ss271829` verified its
descriptor/config/code hashes, ordinary source checkpoint SHA
`0bcefb09...`, task-matched support and LR-validation index seals and shards,
and checkpoint hashes for tasks 0 and 1. Their LR-dev minima (`0.034440441569`
and `0.037655809789`) are the unique minima of their 100-epoch LR validation
histories (best epochs 98/99); the runner selects from the LR validation index
only. Both outputs record `query_metric=null`, and no HR report existed at
audit time. No query/test or HR assets were read. The descriptor commit is
older than the runtime tree, but exact runtime file hashes match. Tasks 3 and
5 plus independent HR-dev reports remain pending this interim review.

## Ordinary direct256 baseline final audit (2026-10-14)

Independent recomputation verified all four LR values
`0.0344404415693/0.0376558097894/0.0344900323777/0.0309849444195`
(mean `0.0343928070390`; unique best epochs 98/99/99/16) and all four HR values
`0.0724135997472/0.0405240448774/0.0717075194814/0.0439702435513`
(mean `0.0571538519143`). Checkpoint/task-result/HR-index hashes pass 12/12;
all task `query_metric` fields are null, all HR reports have
`query_read=false`, and all within-task selection flags exclude HR. The
summarizer's one-report-per-path aggregation reproduces the 8-rollout means.

Compared with path-source comm256/direct256 HR `0.0528792816709`, this matched
ordinary initialization is 8.0836% worse by mean, losing on tasks 0/1/3 and
winning on task 5. Equivalently, the path-source result is 7.4791% lower than
ordinary direct256. Ordinary direct256 still improves on the earlier ordinary
NLT HR mean `0.06219577` by 8.1065%.

The three duplicate HRDEV rows for tasks 1/3/5 all reference the same trial
ID, report path, evaluator hash, and metric as the final reports. The file
reports match the second ledger timings; the summarizer reads each report path
once and its max-by-ID accounting charges one logical evaluation. However,
actual ledger execution rows total `0.005613611168` GPU-h versus
`0.003248571189` logical-charge GPU-h, so the current summary undercounts
`0.002365039979` GPU-h (`8.514 s`) if both racing evaluator calls ran. Preserve
both rows and add this compute overhead to campaign accounting; future HR-dev
evaluation now shares the GPU0 training lock. A valid full-FT smoke also
preceded the scored capacity queue: `ordinary_full_lr3e-5_cos100-capacity-smoke1-smoke`,
task 0, 2 epochs, finite LR-dev `0.037213062355`, query-free.

## Ordinary full-FT capacity audit (2026-10-14)

Independent read-only audit of `ordinary_full_lr3e-5_cos100-capacity1` and
`ordinary_full_lr1e-4_cos100-capacity1` verified four-task LR means
`0.030261243111` and `0.029712766977`, respectively, from the task outputs.
The `1e-4` arm is 1.8125% lower by mean LR error. Best epochs are unique
LR-validation argmins over the 100-epoch runs: `95/31/41/68` and
`25/12/22/48` (tasks 0/1/3/5). All eight checkpoint hashes, 96 support and
validation shard hashes, source/config/code/index hashes, and one-DONE-row-per-
task ledger bindings pass. `query_metric` is null for every task; selection is
LR-only and no HR/query data was opened. These remain baseline LR-screen
results; HR ranking awaits the frozen HR-development evaluation.

## Scratch 500-epoch capacity task-0 audit (2026-10-14)

Read-only audit of `scratch_lr1e-4_cos500_wd1e-4-capacity1` task 0 passed.
The immutable descriptor has null initialization/source checkpoint and the
engine resolves this to `kind:scratch`; the configured full-model optimizer is
LR `1e-4`, cosine-500, minimum LR `1e-6`, weight decay `1e-4`. The unique
minimum across all 500 LR-validation history entries is epoch 365 at
`0.54202954098582268` (next best epoch 313: `0.54212468257173896`), matching
the task JSON and DONE ledger row. Config/descriptor hashes, checkpoint SHA,
support and LR-validation index seals and all 12 shard hashes match. The task
JSON and `task_results.json` entry match exactly; one RUNNING/DONE row pair
closes to `0.101571864809723` GPU-h. Training/selection use only the task's
LR support and LR validation; `query_metric=null`, and no HR/query paths are
present in the task evidence. No leakage or integrity anomaly was found.

## Ordinary full-FT HR-dev and scratch LR 3e-4 audit (2026-10-14)

Independent read-only audit of the eight full-FT HR-development reports passed.
For LR `3e-5`, task HR values are
`0.07764795989496634/0.04306753039127215/0.07136852937983350/0.05982957256492227`
(mean `0.06297839805774856`); for LR `1e-4`, they are
`0.07721036945004016/0.04191738310037180/0.06953901535598561/0.05948388455435634`
(mean `0.06203766311518848`). Each report binds the LR-selected checkpoint,
task result, HR validation index seal and rollout shards; eight per-rollout
means and eight report-DONE ledger rows match. All four LR selection epochs
per arm are unique validation minima. Every report has `query_read=false`,
`training_or_within_task_checkpoint_selection_used_hr=false`, and
`hr_validation_is_development_only=true`; both training summaries have
`query_opened_or_loaded=false`. No integrity or leakage issue was found.
The reported HR-index SHA is the protocol's canonical self-seal rather than a
bytewise JSON-file hash; its seal and shard bindings verify.

The four-task scratch `3e-4`, cosine-500, WD-`1e-4` audit also passed. LR-dev
values are `0.5476257042959333/0.3805401546414942/0.3996937176212669/0.4049696068279445`
(mean `0.43320729584665973`), with unique minima at epochs `207/314/371/478`.
All 500-point histories, checkpoint and data hashes, index seals/shards,
descriptor/config hashes and one-DONE-row-per-task ledger bindings match;
`query_metric` is null and the summary confirms the query was not opened.
Despite a lower mean than scratch LR `1e-4`, this remains far behind transfer
and is not a competitive core baseline.

## R2C ordinary full-FT + direct256 independent audit (2026-10-14)

Read-only verification of R2C and the three requested comparator arms. No GPU work was run and no query/test data was opened.

- **R2C protocol and provenance:** `ordinary_full_comm256_lr1e-4_cw005-fullcomm1`, tasks 0/1/3/5, seed 42, support draw 271829, m=4; full FT from ordinary source checkpoint, direct-256 two-step consistency weight 0.05, LR `1e-4`, cosine, 100 epochs. Config descriptor hash `f857e579…` agrees in `config.json`, summary, task results and all four ledger DONE rows. Frozen YAML SHA / `trial_spec_sha256` is `6ddd0ecf…`; base calibration config SHA is `841d6e24…`. Source `source_final_s42_resume.pt` SHA `0bcefb09…` and data-manifest whole-file SHA `d1013c91…` match disk. Runner/engine/adapter hashes recorded by the descriptor (`767504ee…`/`1fbbd244…`/`8f7e91a3…`) match current files. Summary SHA is `11dcc7c7…`.
- **Selection and LR hashes:** All four task histories contain 100 validation entries and have a unique LR-validation argmin exactly at the recorded best epoch: task 0/1/3/5 = epochs 33/12/10/18. Each task JSON, task-results entry, summary and ledger metric agrees. On-disk checkpoint, source-checkpoint, support-index and LR-validation-index hashes pass 16/16 checks. The same task-specific m=4 support and LR-validation index seals are used by all four comparison arms.
- **Recomputed results:** LR validation task values 0/1/3/5 are `0.03398692 / 0.03109127 / 0.03063397 / 0.02538508`; mean `0.03027431`, worst family high-frequency `0.03398692`. HR development uses eight 256², 20-step rollouts per task. Recomputed per-task means are `0.07017276 / 0.03706850 / 0.06422707 / 0.05203564`; equal-task mean `0.05587599`, worst family high-frequency `0.07017276` (largest individual rollout error `0.14854686` in task 0).
- **Comparisons (LR mean / worst family; HR-dev mean / worst family):** ordinary full-FT final-only: `0.02971277 / 0.03268298 (high-frequency); 0.06203766 / 0.07721037 (high-frequency)`. Ordinary NLT direct256: `0.03439281 / 0.03765581 (amplitude_ood); 0.05715385 / 0.07241360 (high-frequency)`. Path NLT direct256: `0.02810983 / 0.03350050 (amplitude_ood); 0.05287928 / 0.06769019 (high-frequency)`. R2C is 9.93% lower HR-dev error than ordinary full-FT final-only, but 1.89% worse on LR validation; versus ordinary NLT direct256 it is 2.24% lower HR-dev mean and 11.97% lower LR mean, with piecewise-smooth HR error 18.34% higher; versus path NLT direct256 it is 5.67% worse on HR-dev mean and 7.70% worse on LR validation, and loses on all four HR families. R2C’s final-only comparison is the closest same-initialization/same-LR contrast; NLT/path-NLT arms also differ in adaptation strategy, LR, and (for the latter) source initialization, so those are system-level comparisons, not isolated causal effects.
- **HR report seals and query discipline:** Four R2C HR-dev reports bind the LR-selected checkpoint and task JSON, match the reported eight-rollout means and HRDEV ledger metrics, and verify checkpoint/task-result SHA, canonical HR-index seals, all 32 listed HR-validation shard hashes, and the canonical task-manifest seal. Evaluator hash `5109505b…` matches the HRDEV ledger config field. All four report `query_read=false`, `training_or_within_task_checkpoint_selection_used_hr=false`, and development-only HR validation; the training summary says `query_opened_or_loaded=false`, and all task `query_metric` fields are null. The HR index hash is its embedded canonical seal, not the bytewise JSON-file hash.
- **Ledger and limits:** R2C has one RUNNING/DONE history pair per training task; its four DONE charges sum to `0.554417839611` GPU-h and equal the task JSON charges. Its four HRDEV charges sum to `0.002771358202` GPU-h and each equals the report timing; no duplicate R2C rows were found. The separate smoke is not included in these four-task scores. The ordinary NLT direct256 comparator has duplicate DONE ledger rows for the same task-1/3/5 HRDEV trial IDs and output paths, with identical metrics but two distinct time charges each. The retained reports match only one charge in each pair; its accuracy evidence remains reproducible, but do not use the duplicated rows as an unadjusted cost total. These are four-family, one-model-seed/one-support-draw development results, not confirmatory evidence.

## Stage-2 pilot and new Phase-1 confirmation pipeline audit (2026-10-14)

Read-only audit of the frozen pilot YAML/card, source/target GPU lock use, new
confirmation boundary manifest/generator, and available Phase-1 evaluation
entrypoints. No training, rollout generation, HR query, or query/test data read.

**Pilot launch blocker:** `configs/phase1_stage2_pilot.yaml` sets
`experiment.round: S2PILOT`, while `scripts/run_phase1_trials.py` only accepts
round names beginning with `R` (lines 159–161). The configured target run will
raise `ValueError` before any adaptation task. This needs a code/config
amendment before the first target dispatch. The pilot card lists seeds
`[0,1,42]`, but the YAML contains only `screen_model_seed: 42` and the runner
accepts one `--seed` per invocation; the three-seed matrix therefore needs
three explicit, logged invocations per arm. Also clarify the YAML's
`primary_selection_metric` (LR validation) as the within-task checkpoint
selection metric: the card and `summarize_phase1_hrdev.py` rank candidate arms
by HR-development results after LR selection. The current name otherwise
conflicts with the stated promotion rule.

**Source lock/provenance:** source training, target training, and HR-development
evaluation all use the same `phase1_training_gpu0` advisory lock, so the
serialization protocol is consistent. The reused source runner is nevertheless
hard-coded to label reports/ledger rows and checkpoints as `R1` (including its
experiment name), although these fresh-scratch pretrains serve Stage 2. Its
checkpoint output names also have no Stage-2 namespace; existing-file refusal
prevents overwrite, but the source ledger/report namespace should be explicitly
mapped to the pilot in an amendment or recorded as a documented provenance
alias. This is an audit-trail issue, not a lock race.

**Confirmation boundary freeze passed:** the new manifest seal verifies; it
contains 48 unique boundary IDs, exactly eight tasks in each of the six frozen
families, and zero exact boundary-descriptor overlap with the six-task
HR-development manifest. It records `boundaries_frozen_rollouts_not_generated`.
These checks concern boundary descriptors only; no rollout assets were opened.

**Query gate is bypassable and currently does not enforce the stated freeze:**
`scripts/prepare_phase1_development.py` only requires an explicit `--roles`
argument when `--campaign-type confirmation` is selected (lines 395–399). It
then permits `validation_hr` in that same call without checking for a frozen
checkpoint/analysis manifest or a completed LR-data gate. More seriously,
`--campaign-type` defaults to `development`, `--task-manifest` accepts any
sealed manifest, and the code does not validate manifest schema/campaign against
the requested campaign type. A caller can pass the new confirmation
`tasks.json` with the default development type; the generator will treat
`validation_hr` as ordinary development validation, place it under the
`phase1_development` root, and report `query_generated: false`. This is a real
mislabeling path around the intended query boundary. Before any confirmation
rollout generation, bind campaign type to manifest schema/root, reject
confirmation manifests under development mode, and require a durable
checkpoint/analysis freeze plus explicit one-shot authorization before
`validation_hr` generation.

**Final confirmation evaluation is not yet executable as a frozen 5×3 study:**
the available new Phase-1 assets are the boundary freezer and generic role
generator. I found no Phase-1 confirmation-specific config, complete seed/support
matrix manifest, data verifier, checkpoint-freeze gate, query evaluator, or
statistical aggregator. The existing `scripts/evaluate_phase1_hrdev.py` and
`scripts/summarize_phase1_hrdev.py` are hard-wired to the development manifest
and cannot serve as confirmation evaluation. The generator exposes one
`--support-seed` per invocation but does not freeze the mission's five model
seeds crossed with at least three support seeds, nor enforce pairing or
selection rules. Freeze those dimensions, LR-only checkpoint selection, query
units, estimands/CI and method/checkpoint hashes before producing any HR query.
The existing old Allen–Cahn reconfirmation gates are tied to their historical
manifest and are not a substitute for this new campaign.

A smaller pilot protocol gap: `evaluate_phase1_hrdev.py` has no campaign-wide
release gate proving all LR-selected cells are complete before the first HR-dev
report; `--output` can also be changed to repeat an evaluation. The card's
“evaluate once after the complete LR screen” remains operator-enforced. A
sealed readiness manifest or a documented no-inspection queue should precede
HR-dev evaluation; this can be fixed while source training continues.

## Revised Stage-2 startup and confirmation gate review (2026-10-14)

Static read-only review of `run_phase1_trials.py`,
`prepare_phase1_development.py`, `freeze_phase1_confirmation_matrix.py`, and
`freeze_phase1_confirmation_query.py`. No tests, training, data generation, or
query/test access.

**Resolved:** the target runner now accepts the frozen `S2PILOT` namespace.
Confirmation generation checks the task-manifest seal and matching campaign ID;
confirmation mode requires the canonical campaign task manifest and its
confirmation schema, while development mode rejects a confirmation schema.
Generating `validation_hr` now requires the campaign's query gate, whose
self-seal and recorded preregistration, matrix, config, checkpoint, analysis,
generator, and freezer file hashes are rechecked. I found no remaining direct
`--task-manifest`/`--campaign-type` bypass through this generator path.

**Remaining query-use gap in the training runner:** `scripts/run_phase1_trials.py`
formats support/validation index paths from config and records their bytewise
hashes, but never reads/verifies index metadata or role. It passes the support
path to `train_data_path` and validation path to `validation_data_paths`
(lines 359–410). A confirmation config can therefore point either slot at a
`validation_hr` index and use the query for fitting or checkpoint selection;
`selection_data` is only a free-form label. Enforce `support` and
`validation_lr` roles, expected campaign/task IDs, resolution/horizon, and
support count before training, and reject HR/query roles in this runner.

**Checkpoint-matrix integrity is not tied to training evidence:** the query
freezer checks every expected `(method, model_seed, support_seed, task)` tuple,
rejects repeated tuples, and verifies that each listed file matches its
`checkpoint_sha256`. It does not require distinct checkpoint paths across
cells, nor bind a cell to a hashed task-result/selection record, the method's
frozen config hash, support/validation index hashes, or a unique LR-validation
argmin. Consequently one checkpoint (or an unrelated in-project file) can be
listed for multiple cells and satisfy the complete grid. Require each cell's
runner task-result evidence and validate its seed/task/support/config/checkpoint
bindings, LR-only selection history, and role-checked data hashes; reject
checkpoint-path reuse across cells.

**Query generation parameters remain changeable after unlock:** the matrix
states a 20-step metric, but the query gate does not freeze or check
`--hr-steps` or `--hr-validation-rollouts`. The generator accepts both as
runtime overrides after validating the gate. Thus the first query generation
can use a different horizon or rollout count from the preregistered endpoint
without changing the matrix/gate hashes. Freeze a query-protocol manifest
(resolution, fidelity, horizon, rollouts per task, IC seed rule and solver
settings) into the gate and require exact matches before generating
`validation_hr`.

**Additional fail-closed details:** the query freezer verifies the task
manifest's self-seal and campaign ID but does not re-check its confirmation
schema; it accepts any in-project `--matrix-manifest` path with matching sealed
contents rather than requiring the canonical campaign `matrix.json`. The
matrix freezer also sorts task IDs without rejecting duplicate IDs or checking
family counts. Pin the task/matrix schemas and roots in the query freezer and
validate unique task IDs and the frozen family/task counts before sealing.

No pilot startup blocker remains in the reviewed round-name path. The query
unlocker is intentionally unavailable until both required files
`scripts/evaluate_phase1_confirmation.py` and
`scripts/analyze_phase1_confirmation.py` exist; they were absent at this
read-only review, so the final query path is not yet executable end-to-end.

## Confirmation gate hardening follow-up (2026-10-14)

Static source review only; no scripts/tests, training, data generation, or
query/test access. The Stage-2 namespace check now accepts `S2PILOT`. The
training runner reads and verifies both target indices, requires the expected
support/validation roles, LR resolution/fidelity/horizon, task manifest/task and
boundary, support seed and exact support count, and disjoint support versus
validation IC IDs. Confirmation HR generation is restricted to the standalone
`validation_hr` role; it checks the sealed gate and matches runtime HR rollout
count/horizon to the sealed matrix. The matrix freezes 16 support, 8 LR
validation and 8 HR query rollouts, all at 20 steps. The query freezer now
requires canonical task/matrix/checkpoint-manifest paths, full unique matrix
cells, distinct checkpoint and task-result paths, LR-history unique minima,
role-checked index seals, checkpoint/config/result/index hashes, and a frozen
ledger. The previous direct generator bypasses appear closed.

Remaining issues:

1. **LR validation sample count and within-role IC uniqueness are not enforced.**
   `run_phase1_trials.py::_validate_target_lr_indices` requires at least one
   validation shard but does not require the matrix's eight validation
   rollouts. It rejects overlap between support and validation IC sets, but does
   not reject duplicate IC IDs within either role. The query freezer repeats
   role/resolution/horizon checks, but likewise does not check exact support m,
   eight validation rows, or within-role IC uniqueness. Such indices can pass
   validation and produce a task result whose `support_budget` claims m=4 even
   when the actual inputs contain fewer effective samples. Validate exact
   expected LR counts and unique `initial_condition_id`s in both the runner and
   final checkpoint gate.

2. **The query gate does not verify the source checkpoint bytes.** The task
   result carries `source_checkpoint` and `source_checkpoint_sha256`, but the
   query freezer and generator only hash the adapted checkpoint, task result,
   and LR indices. For compact NLT/adapter checkpoints the source checkpoint is
   an inference dependency; a changed or missing base source can therefore
   invalidate the frozen inference model without failing the gate. Verify and
   record each non-null source checkpoint path/hash (and reject inconsistent
   source hashes for shared method/seed cells).

3. **Run-config/task-result linkage can be tightened.** The freezer checks the
   run-config file hash and selected descriptor fields, and separately checks
   the task-result hash and LR argmin. It does not compare the run-config's
   `config_sha256` with the task-result's `config_sha256` (or recompute it from
   the descriptor), nor require `descriptor.trial.id` to equal the matrix
   method. Bind these records directly so a correct-looking result and an
   unrelated run config cannot be paired in the checkpoint manifest.

No remaining pilot startup blocker or direct query-generation bypass was found
in the reviewed paths. The items above concern final-sample adequacy and the
proof that frozen artifacts are exactly those selected by the recorded LR run.

## Final confirmation provenance follow-up (2026-10-14)

Static source review only. I did not execute scripts/tests, run generation or
training, or inspect query/test artifacts.

The three previously flagged checkpoint-provenance issues are now closed in the
reviewed code:

1. `freeze_phase1_confirmation_checkpoints.py::_verify_lr_index` requires the
   exact support and LR-validation shard counts and unique initial-condition
   IDs within each role, then checks cross-role disjointness. The query freezer
   independently repeats exact-count, within-role uniqueness, and cross-role
   checks before unlocking.
2. The checkpoint freezer verifies each source checkpoint's bytes against the
   run-config descriptor and binds the task result to that path/hash. The query
   freezer rechecks the source bytes and records the path/hash in the gate;
   the generator rechecks them again before HR generation.
3. The checkpoint freezer recomputes the run-config digest from its descriptor,
   checks the descriptor's round/method/seed/support/config/task-manifest
   bindings, and requires the result's config and descriptor hashes to match.
   It also verifies the unique LR-history argmin and best epoch. The query
   freezer repeats the key config/result/argmin checks.

Remaining protocol risks:

1. **The HR solver implementation is not frozen by the query gate.** The gate
   records hashes for `prepare_phase1_development.py`, the gate freezer, and
   analysis entry points, but not `blrpng/solvers/sfepy_worker.py` or its
   generation dependencies. The wrapper launches that worker with the query
   solver parameters at `prepare_phase1_development.py:308-315`; after gate
   creation, a worker change can alter the generated HR endpoint while the
   generator and gate hashes still pass. Freeze and recheck the worker and
   relevant solver/data-generation source hashes (and record the effective
   solver parameters) before query generation.
2. **`query_opened: false` is asserted, not evidenced.** The checkpoint freezer
   writes this constant into its manifest, without checking whether query
   artifacts already exist. Also, `blrpng/solvers/sfepy_worker.py:159-170`
   accepts a manifest and split and generates its cases without consulting the
   confirmation gate; calling that worker directly bypasses the wrapper gate.
   The gate therefore protects the supported `prepare_phase1_development.py`
   path, but cannot certify that HR query generation was absent before unlock.
   Either make the worker reject confirmation query manifests unless given a
   verified gate, or document and enforce the wrapper as the only authorized
   execution path and add a pre-freeze check for existing query outputs.

## Worker-side query-gate follow-up (2026-10-14)

Static source review only. No scripts/tests, generation, training, or query/test
artifacts were executed or read.

The new shared verifier closes the ordinary direct-worker path for a correctly
tagged confirmation query manifest: `sfepy_worker.generate()` calls
`_authorize_confirmation_query_manifest()` before solving; that function
verifies the canonical frozen task manifest and gate, then checks query role,
256/257 output/mesh resolution, fidelity 1, full task × rollout coverage,
boundary binding, and unique IC IDs. The wrapper calls the same verifier, and
the gate now records hashes for the generator, gate freezer, worker, and
verifier. Changes to those files are detected as long as the verifier itself
remains the reviewed implementation.

Remaining issues:

1. **A role-label downgrade still bypasses the worker gate before unlock.**
   `_authorize_confirmation_query_manifest()` returns immediately unless
   `evaluation_role == "query"` or the metadata role is `validation_hr` with a
   confirmation task-manifest path (`sfepy_worker.py:145-153`). A manifest
   bound to confirmation tasks but labeled `support` or `validation_lr` does
   not enter the gate path. The worker accepts caller-supplied resolution,
   fidelity, and step arguments, so such a manifest can be sent directly to
   the worker at HR/fidelity 1 while the query gate is absent. Apply the gate
   to every HR/fidelity-1 solve bound to confirmation tasks, then require the
   canonical query role and its frozen protocol.
2. **The worker does not bind the case manifest or all solver arguments to the
   freeze.** The verifier freezes query count and steps, while worker checks
   role, resolution, mesh, fidelity, case range, task coverage, boundary IDs,
   and unique IC IDs (`sfepy_worker.py:175-210`). It does not require the
   canonical `validation_hr.json` case-manifest path/hash or reproduce the
   preregistered query IC seed/case construction. A different self-sealed
   manifest with the same tasks/counts but altered ICs can pass after unlock.
   Direct worker arguments for `dt`, `epsilon`, `gamma`, progress threshold,
   Newton limits, and tolerances are not checked against the wrapper's fixed
   values either. Freeze and validate the exact case-manifest identity and the
   full numerical solver protocol in the gate/worker.
3. **Verifier hash enforcement is self-referential.** The worker imports and
   calls the verifier; the verifier is the only code that checks its recorded
   hash (`confirmation_query_gate.py:95-112`). If that verifier is edited after
   freeze to weaken or remove the checks, the unchanged worker imports the
   weakened implementation and does not independently reject it. Hashing
   `blrpng/paths.py` dependencies is also absent, although the verifier relies
   on its path-resolution and SHA helpers. Add an independent verification
   anchor (for example, a frozen source/build identity checked by trusted
   bootstrap code) and include relevant imported generation/hash dependencies.
4. **Existing query indices can be reused without worker provenance.**
   `_generate_role()` returns an existing index when manifest hash, split,
   steps, resolution, and fidelity match (`prepare_phase1_development.py`,
   existing-index branch around lines 275-295); the index metadata does not
   bind the worker hash or effective solver settings. A pre-existing index
   produced before the current gate checks could therefore be reused after
   unlock. Require generation provenance in the index/audit and make query
   unlock fail closed if an unproven query output already exists.

### Stage-2 pilot runner follow-up (2026-10-14)

Static review of the new `S2PILOT` runner path only; I did not repeat the
parent's CPU-only provenance preflight. `run_phase1_trials.py` now accepts the
exact `S2PILOT` round namespace (`:229-233`) and applies the frozen eight-row
LR-validation requirement for both `S2PILOT` and `CONFIRM` (`:438-449`). Its
index validator checks role, LR resolution/fidelity, 20-step horizon, frozen
task-manifest/task/boundary, task-only rows, support seed and support count;
it rejects HR/query roles and requires the expected validation count
(`:58-100`). It also rejects repeated initial-condition IDs within either
split and overlap between support and LR validation (`:102-109`). I found no
missing constraint in the requested S2PILOT/CONFIRM validation-count and
split-uniqueness changes.

### Latest confirmation query worker and pre-unlock review (2026-10-14)

Static source review only. No tests or scripts were run, no GPU work was done,
and no query/test artifacts or data were read.

The latest worker change closes the previously identified simple metadata
downgrade: a case manifest under the confirmation-manifest root is treated as
confirmation-bound, and a referenced task manifest is also recognized by its
confirmation schema. An HR/fidelity-1 request through either route must use
the canonical case manifest and pass the frozen gate. The worker checks the
gate's worker, verifier, generator, freezer, and runtime source hashes with
standard-library hashing before lazily importing the shared verifier. Matrix
checks bind the query case path, deterministic IC seed/case IDs, NumPy version,
and full solver argument set. Query index reuse also requires matching
worker/gate/protocol/NumPy/IC-seed metadata and verifies shard hashes. I found
no ordinary stale-index reuse path around those checks.

The new pre-unlock check is present and consistently required: the freezer
fails if the canonical HR index, shard directory, or wrapper `.npz`/`.json`
temporary outputs exist; it writes the checked path list and absence flag into
the gate, and the verifier requires both. This is useful evidence for those
four canonical locations. It is still an assertion by the freezer, not a
filesystem snapshot or signature, and it does not cover arbitrary output paths
from direct worker invocations.

Two residual trust-boundary issues remain:

1. **An unknown-path copy can still evade confirmation classification.** The
   worker returns before gate verification when all three signals are absent:
   the supplied case path is outside the confirmation root, its metadata
   `task_manifest` neither names a confirmation-root path nor resolves to an
   existing confirmation-schema task manifest, and `evaluation_role` is
   relabeled away from `query`. A copied confirmation case manifest with
   those metadata fields changed and its outer self-seal recomputed will then
   proceed through generic `read_manifest()`; that reader checks only schema
   and self-seal. Direct HR/fidelity-1 generation can therefore use the copied
   confirmation cases without the query gate, writing to an arbitrary output
   path that the pre-unlock check does not inspect. Reject unregistered HR
   manifests or bind their content to a trusted task/case-manifest registry,
   rather than relying only on mutable path and metadata labels.
2. **The direct worker's self-hash is not an independent execution anchor.**
   A direct `python -m blrpng.solvers.sfepy_worker` invocation executes the
   worker file whose hash it later checks, so a modified worker can alter or
   omit its own check. Also, hash-bound project modules (`paths`, `manifest`,
   `lifting`, `blind_boundaries`, and `progress`) are imported at module load
   before the check runs. The supported wrapper verifies the worker before
   spawning it, but that protection does not cover direct worker invocation.
   If direct invocation must be an enforced trust boundary, move authorization
   to a separately trusted stdlib-only launcher and defer project imports
   until after source checks; otherwise make the wrapper the only trusted
   entry point and prevent direct use operationally.

Reproducibility note: the frozen matrix pins NumPy but not Python, SfePy,
SciPy, or the BLAS/LAPACK implementation. Those versions can affect solver
behavior; record and validate the effective solver environment if numerically
stable HR replication is required.

### Boundary-identity and numerical-runtime follow-up (2026-10-14)

Static review only; no scripts/tests, generation, GPU work, or query/test data
access.

The new boundary scan closes the straightforward copy-and-relabel case as
long as the copied cases retain their frozen `boundary_id` or exact
`boundary_descriptor`: a sealed confirmation task manifest is scanned, and
any HR/fidelity-1 request matching either identity is routed to the canonical
gate. The runtime-environment path is internally connected: matrix freeze
captures Python/NumPy/SciPy/SfePy versions and a NumPy/SciPy build-config
fingerprint in the SfePy environment; the query gate carries that object; the
verifier checks it against the matrix and gate; and the worker recomputes and
compares the full object before solving. The wrapper's early verifier call
only supplies its NumPy version, but any new HR solve still reaches the worker
check before problem construction.

One simple boundary-identity bypass remains. Copy the confirmation case
manifest, change each case's top-level `boundary_id` and its nested
`boundary_descriptor.boundary_id` to fresh values, relabel the task-manifest
reference and evaluation role, then recompute the outer manifest self-seal.
`_references_frozen_confirmation_boundary()` no longer finds the changed IDs
or descriptor hashes. Those two ID fields do not affect
`evaluate_case_boundary()`; the remaining descriptor fields still produce
the exact frozen confirmation boundary, and the embedded descriptor `sha256`
is not validated by the generic manifest reader or boundary evaluator. Such
an HR/fidelity-1 worker request can therefore miss gate routing. Compare a
canonicalized effective boundary identity that excludes non-semantic IDs and
embedded seal fields, or validate each case descriptor's own seal and source
task binding before allowing generic HR generation.

The global scan can also conservatively classify a non-confirmation HR case
when its numeric boundary ID collides with any confirmation task ID, even if
the physical descriptor differs. This fails closed but may block legitimate
development HR generation when ID ranges overlap; prefer a semantic match or
an explicit campaign/namespace-aware ID binding.

### Boundary trace fallback follow-up (2026-10-14)

Static review only; no scripts/tests/solver/GPU/query/test data were accessed.

The recursive semantic hash correctly closes the prior ID-only edit: it drops
only `boundary_id` and `sha256` recursively, so changing those fields leaves
the semantic descriptor match intact. However, the numerical fallback samples
33 points per edge, which is not a collision-resistant fingerprint of the
boundary trace used by the 257-vertex solver mesh. For a Fourier boundary
descriptor, add 1024 to each frequency and change the case's top-level
`boundary_id` to a fresh value. On the solver mesh, perimeter coordinates are
`s = k + j/256`; each Fourier phase changes by
`1024 * pi/2 * s = 512*pi*k + 2*pi*j`, so the boundary values at every mesh
vertex are unchanged. The worker's initial-field extension also evaluates the
boundary at projected mesh coordinates, so it sees the same values. On the
fallback's 33-point grid (`s = k + j/33`), the phase shift is generally not a
multiple of `2*pi`, so the trace comparison rejects the match. The semantic
hash also differs because the frequencies changed. This gives the same
discrete HR solve while evading the current boundary scan, provided the
affected Fourier coefficients are nonzero.

Use a canonical signature over the actual solver mesh boundary values (and
the projected boundary samples used by initial-field construction), or make
the canonical descriptor/task binding mandatory for any HR case whose
effective mesh trace matches a frozen confirmation task. A small fixed sample
grid should not serve as the sole semantic identity check.

### Canonical development HR allowlist follow-up (2026-10-14)

Static review only; no scripts/tests/solver/GPU/query/test data were accessed.

The new top-level policy is fail-closed for unknown HR manifests, and the
canonical development case validator binds every case's task ID, boundary ID,
descriptor, family, and split to the sealed development task manifest. It
blocks ordinary copies of confirmation cases because the confirmation scan
still runs before development is allowed. However, the development task seal
is only a recomputable content hash; `_is_canonical_development_hr_manifest()`
does not verify that its tasks were produced by the frozen development task
generator or exclude task descriptors copied from confirmation.

The remaining finite-grid alias can be carried through this allowlist: make a
new task manifest under `phase1_development/<campaign>/tasks.json` using the
development schema, copy a confirmation Fourier task, change its boundary ID,
add 1024 to its descriptor frequencies, and recompute the task self-seal.
Create the canonical `validation_hr.json` with cases bound exactly to those
forged tasks and recompute its self-seal. The development validator accepts
the path, schemas, hashes, and per-case equality. The confirmation scan sees
changed IDs and semantic hashes; its 33-point trace generally differs, so the
HR request follows the development allowlist. On the 257-vertex solver mesh,
the shifted Fourier descriptor has identical boundary values and identical
projected-boundary warm-start values, producing the same discrete solve as the
confirmation task. Recompute development tasks from their recorded seed using
the canonical task-generation code (and validate task inventory/uniqueness),
or compare actual solver-mesh boundary/effective initial-state values against
frozen confirmation tasks before admitting a development HR manifest.

### Pinned development allowlist and resolution-threshold review (2026-10-14)

Static source review only; no scripts, tests, solver/GPU work, or confirmation
query rollout access.

The previous forged-development-task bypass is closed in the current worker:
the one allowed development campaign, task-manifest SHA, and HR case-manifest
SHA are hardcoded, in addition to canonical paths and self-seals. A newly
forged development task manifest cannot enter that allowlist without matching
the pinned SHA-256 values.

One practical query-read path remains through the high-resolution threshold.
Unknown/copy manifests are rejected only when `high_resolution_request` is
true, which currently depends on `split == "validation_hr"`,
`output_resolution >= 256`, or `fidelity == 1`. `fidelity` is only recorded in
the output and does not change the PDE solve. A copied confirmation manifest
can have its cases relabeled to `validation_lr`, metadata role/evaluation role
changed away from `query`, and outer seal recomputed; a direct worker request
with `output_resolution=255`, `mesh_resolution=257`, `fidelity=0`, and the
same 20-step solver parameters then has `high_resolution_request == false`.
The unknown-manifest rejection and confirmation boundary scan are both
skipped. This runs the same finite-element mesh solve and emits a 255×255
sampling of the query trajectory, which can be interpolated to the official
256×256 output grid. Lower output resolutions with the same 257-node solve
also bypass the guard. Classify protected cases independently of caller
resolution/fidelity labels, or at minimum treat the frozen HR mesh resolution
as high-resolution and gate query-bound cases at every fidelity.

### Direct-worker query-route threshold follow-up (2026-10-14)

Static review only; no tests/scripts/solver/GPU work and no query rollout
access.

The prior near-HR `255` output / `257` mesh request is now rejected: the
high-resolution predicate checks output above 64, mesh above 65, steps above
10, fidelity 1, or the HR split. Query identity routing and confirmation
manifest-path/task-manifest routing are also no longer conditioned on that
predicate, so ordinary split relabeling does not evade recognition.

An adjacent low-resolution direct-worker path remains. The predicate uses
`>` thresholds rather than requiring the exact permitted LR protocol, and
the confirmation boundary scan compares only 65- and 257-node edge grids,
not the requested mesh. A copied/reconstructed query case can change its
top-level boundary ID, relabel its case split and metadata away from query,
and add 128 to Fourier descriptor frequencies. On a 33-node mesh, the
perimeter nodes are `s = k + j/32`; the added frequency changes phase by
`64*pi*k + 2*pi*j`, preserving boundary and projected warm-start values on
that mesh. The fixed 65/257 scan generally does not match the original trace.
With output 64, mesh 33, steps 10, fidelity 0, and split `validation_lr`, the
request is below every threshold; an unknown copied manifest is not rejected
and can produce a lower-resolution, shorter-horizon confirmation-bound solve.
This is not the official HR endpoint, but it is still query-rollout access and
can expose partial/low-resolution target behavior. Compare identity on the
actual requested mesh or require the exact canonical LR protocol plus a
registered LR manifest; do not let arbitrary below-LR settings fall through
as an unguarded non-HR request.

### Requested-mesh confirmation identity review (2026-10-14)

Static source review only; no scripts, tests, solver/GPU work, or query rollout
access.

The reported `mesh_resolution=33`, `+128` Fourier alias path is closed in the
current worker. The confirmation-boundary scan now runs before route selection
regardless of split or resolution labels, and compares sealed confirmation
tasks against edge traces at the requested mesh as well as the canonical 65
and 257 node grids. With a 33-node mesh, the prior frequency shift agrees at
the requested FE edge vertices, so the copied/relabelled case is routed to
the canonical confirmation gate. The predicate also treats output above 64,
mesh above 65, steps above 10, fidelity 1, or the HR split as HR requests;
unknown HR manifests are rejected, while the pinned development HR manifest
must use the frozen solver protocol. Confirmation path/task-manifest labels
continue to route to the gate independently of these thresholds.

I found no remaining simple copied-manifest/label/requested-mesh alias route
to confirmation output under the current worker source. This conclusion
assumes the frozen worker and helper code remain intact: a direct Python entry
point cannot provide an external root of trust against an operator who edits
the worker's own authorization code and its self-check.

## Stage-2 recovery, pilot matrix, and query-gate snapshot (2026-10-14 15:33 Asia/Shanghai)

Read-only audit only; no training, GPU, solver, test, or query operation was initiated.

### Seed-1 path-source recovery is live

At 15:33:39 CST, `ps` showed the detached supervisor PID `418560` running `artifacts/workflows/phase1_stage2_complete.sh` and its child PID `418565` running:

```text
python -u scripts/run_phase1_source.py --objective path --seed 1 --resume-from-scratch \
  --resume-from artifacts/checkpoints/source_path_s1_scratch_retry1_recovery.pt --attempt-suffix recovery1
```

The worker is on GPU0 (the point-in-time `nvidia-smi` sample was 94% utilization; GPU1 was idle). Progress file `artifacts/reports/progress/source_path_s1_resume_scratch_recovery1.json` was updated at 15:32:17 CST and reports epoch 44/60, best/latest source-HR validation `0.010447250982630062`, 112,500 optimizer steps, zero stale epochs, and 629.693 seconds in this resumed attempt. Its SHA-256 at audit time is `8e64f86b8296770c64c36bc4c29b00f96d695c95590950f3803854feb1cce51d`. The recovery input is present with SHA-256 `629194c90e10e981825f532244248ebe550894ac1166fe150c7b642baa75433e`. The intended final checkpoint `artifacts/checkpoints/phase1/R1/source_path_s1_resume_scratch_recovery1.pt` and terminal report were not present at this snapshot, so completion and final metric remain unknown.

The queue status file has only `source_seed1_path_recovery` as RUNNING; it has not recorded a completed source alias or started subsequent seeds/target jobs. Thus the earlier `STATE.md`/pilot-card epoch-42 and retry1-active descriptions are stale. Update those records from a later live snapshot before relying on them.

### Append-only R1 ledger reconciliation

Rows 418–426 of `docs/EXPERIMENTS.csv` were checked by trial ID and config hash:

- Seed-0 path source: RUNNING row 418 is paired with DONE row 419, 2.068093906690 GPU-h, source validation `0.0072757053154928145`.
- Seed-0 final-only source: RUNNING row 420 is paired with DONE row 421, 1.402507711082 GPU-h, source validation `0.00776299830264179`.
- Seed-1 initial path attempt: RUNNING row 422 is paired with INTERRUPTED row 423, 0.1838888889 GPU-h conservative charge; excluded from ranking.
- Seed-1 `retry1`: RUNNING row 424 is paired with INTERRUPTED row 425, 1.5008333333 GPU-h conservative charge; excluded from ranking. Its documented epoch-39 recovery is the input to the current resumed attempt.
- Current seed-1 `recovery1`: row 426 is RUNNING with config SHA `535abe00410330bd6f222d572f94c3c10e207818e74d8940f446bf31d547803b`, matching the live attempt namespace; it has no terminal ledger row or elapsed charge yet.

The repeated RUNNING rows are append-only start records and are not separate successful results. Both earlier seed-1 attempts have explicit same-ID terminal INTERRUPTED rows; the current attempt is correctly still open. No R1 DONE row exists yet for seed-1 recovery.

### Frozen Stage-2 pilot cell count

`configs/phase1_stage2_pilot.yaml` currently has SHA-256 `91bfeb45993305dde3a42fc5348036d86e19749fc554989bb992279b0ea882a2`. Parsing the frozen file gives six arms:

`pilot_pathsource_comm256`, `pilot_pathsource_firstpath_ratio015`, `pilot_pathsource_endpoint_ratio015`, `pilot_pathsource_endpoint_direction_ratio015`, `pilot_ordinary_full_comm256`, and `pilot_lifted_scratch_lr3e-4_cos500_wd1e-4`.

The frozen target seeds are `[0, 1, 42]`, tasks are `[0, 1, 2, 3, 4, 5]`, `m=4`, and support seed is `271829`. Therefore the planned LR matrix is `6 × 3 × 6 = 108` arm/seed/task cells; the subsequent one-per-selected-checkpoint HR-development pass is also 108 evaluations. This is the configured plan, not a claim that any target cell has completed. At this snapshot, queue status shows the source recovery only, and the matrix audit/HR-dev phases have not started.

### New confirmation query gate remains unopened

The current campaign is `ac_phase1_confirmation_20261014`. Its manifest directory contains only `tasks.json` (48 boundary descriptors; file SHA `c1da9541b80c21b8b37979ae12af06d8db116c04cee21b05e54e46a04c24acf8`). There is no campaign `matrix.json`, `query_gate.json`, checkpoint manifest, or `artifacts/data/phase1_confirmation/` directory. `docs/prereg/CONFIRMATORY.md` still says “No amendments yet” and requires the selected method/config IDs and frozen checkpoint matrix before gate unlock. This is consistent with the absent gate and data assets. Historical query data under the separate `artifacts/data/reconfirmation/` campaign does exist and was previously opened; do not interpret this audit as saying no query has ever been accessed.

These filesystem and process checks establish the gate state and artifacts visible in this workspace at the snapshot, not an OS-level proof that no process ever attempted a read. No such confirmation query access is evidenced here; any broader access history remains unknown.

## R4 pre-evaluation static audit (2026-10-16 03:32 Asia/Shanghai)

Independent review of `docs/experiment_cards/R4_PATHCOMM_INTERACTION.md`, `scripts/run_phase1_r4_hrdev_queue.py`, and `scripts/summarize_phase1_r4_pathcomm.py` found the frozen 1%/three-seed/four-of-six-family rule, paired seed-macro t interval with df=2, queue gate, and query isolation consistent. The interval is descriptive because this comparison uses one support draw and one task per family. At audit time there were no R4 HR-dev output files.

One machine-readable per-cell improvement key named `improvement_candidate_minus_reference` while the stored value was `reference − candidate`; this was renamed `improvement_reference_minus_candidate`. The paired summarizer also now compares support, LR-validation, and HR-validation index paths/seals and source-checkpoint path/hash across each seed×task arm pair, then stores that provenance on each paired cell. The threshold and statistical unit are unchanged. In the completed seed0/task00 A0/A1 records inspected during review, those data seals and source-checkpoint hashes matched. Query data was not accessed.

## R4 HR-dev results independent audit (2026-10-16)

Read-only recomputation from the paired JSON, its Markdown report, and raw
HR-dev JSON reports; no GPU, test, or query data was used. I checked all 36 R4
candidate reports (A1/A2) and 18 A0 comparator reports, including the frozen
seed-1 `recovery1` alias. The task named
`reports/PHASE1_R4_PATHCOMM_20261016.md`; that path is absent. The generated
report exists at `artifacts/reports/PHASE1_R4_PATHCOMM_20261016.md`, matching
the summarizer output path.

All 54 raw-report hashes agree with their paired-summary entries, all
task-result hashes agree with on-disk task JSON, and all report metrics match
the paired cell values. Every report has `query_read=false`,
`training_or_within_task_checkpoint_selection_used_hr=false`, and
`smoke_only=false`; all task results have `query_metric=null`. For every cell,
the report LR-validation metric equals the selected task result's best
validation metric, and the report checkpoint path/hash matches the selected
task-result checkpoint. The checkpoint and source-checkpoint byte hashes
match their recorded values. I independently recomputed each support, LR
validation, and HR validation index's canonical JSON seal (rather than the
serialized file's byte hash); seals and development split/resolution/fidelity/
horizon roles match. In all 18 seed×task cells, A0/A1/A2 have identical
support, LR-validation, and HR-validation paths/seals and source-checkpoint
path/hash. No integrity or pairing mismatch was found.

Independent macro means from raw reports are A0 `0.05347208853`, A1
`0.05387774431`, and A2 `0.05363236246`. Seed means in order 0/1/42 are:

- A0: `0.05736147829 / 0.05100475291 / 0.05205003438`
- A1: `0.05810307870 / 0.05162976656 / 0.05190038767`
- A2: `0.05732694328 / 0.05178591948 / 0.05178422463`

For paired improvements, positive means lower error for the candidate named
first in the comparison. Recomputed macro improvement, relative improvement,
seed-level paired differences (0/1/42), seed wins, family wins, and paired t
interval over the three seed means (df=2) are:

| Comparison | Absolute | Relative | Seed differences | Wins (seed/family) | 95% t interval |
|---|---:|---:|---|---:|---:|
| A1 vs A0 | `-0.000405655783` | `-0.75863%` | `-0.000741600412 / -0.000625013645 / +0.000149646708` | `1/3`, `1/6` | `[-0.001609037221, 0.000797725654]` |
| A1 vs A2 | `-0.000245381851` | `-0.45753%` | `-0.000776135427 / +0.000156152918 / -0.000116163044` | `1/3`, `2/6` | `[-0.001436249370, 0.000945485668]` |
| A2 vs A0 | `-0.000160273932` | `-0.29973%` | `+0.000034535015 / -0.000781166563 / +0.000265809752` | `2/3`, `3/6` | `[-0.001526555874, 0.001206008010]` |

Family differences (same sign convention; order high-frequency, amplitude_ood,
matern_gp, localized_bump, mixed_fourier, piecewise_smooth) are:

- A1 vs A0: `-0.000156416961, -0.001472694275, +0.000502929348, -0.000002135523, -0.001226587535, -0.000079029752`.
- A1 vs A2: `+0.000338916993, -0.000993004985, +0.000567744199, -0.000003187968, -0.001250668949, -0.000132090397`.
- A2 vs A0: `-0.000495333954, -0.000479689291, -0.000064814855, +0.000001052445, +0.000024081414, +0.000053060645`.

The Markdown values agree with the independent calculations to displayed
rounding, and the frozen A1 rule is correctly reported as failed: A1 is not at
least 1% better than either comparator, does not win all three seed means, and
wins only 1/6 and 2/6 family means against A0 and A2. All three paired
intervals include zero. The intervals are descriptive as documented; the
statistical unit is the three seed-level six-family means.

One wording ambiguity was resolved in `artifacts/reports/PHASE1_R4_PATHCOMM_20261016.md`: it now states that the six tasks have task-specific HR-support indices under one common support seed, with each task's index paired across A0/A1/A2.

## HRSCRATCHM16 task01 read-only audit (2026-10-16)

Audited only the completed task01 record and its static development artifacts;
did not inspect task03/GPU state, run tests, or access query data. The task
report SHA-256 is `8a2da6cb777f091f6dad49afaa8879da0fe1fb9e79a8944fd9ec18ce980c561f`.
Its selected inference checkpoint exists and its bytewise SHA-256 matches
`3e946e6e54e52a5c0d2c39f166edf70e5e7809e477fea75760369f423bf9c0f1`. The
HR-support m=16 canonical seal is
`9285f576fb3d69a9c376e1edd117621435e45bdf4a41e0a11d4ef880b88948d7` and the
HR-dev validation seal is
`558c3fc8a9218434f3747b03bae79682a746ae0e9b73051b5eeaec8beadc79f2`; both
were independently recomputed from canonical index JSON. The support index
has 16 rank-0..15 HR shards for task 1/boundary 5000001; validation has eight
HR-dev shards, same task/boundary, with no support-ID overlap. Both are
256-resolution, fidelity-1, 20-step development indices.

The saved 500-epoch history independently extracts best validation
`0.0955920296953991` at epoch 416 and endpoint validation `0.09693536814302207`
at epoch 499; the task report matches the history minimum/epoch. The training
engine saves the best raw/EMA state and full-checkpoint compaction retains that
selected EMA state. The task report's matched path-source reference is A0
`pilot_pathsource_comm256`, task01/seed0 HR-dev; its actual hash matches the
recorded `a90f1872108df932af526d22f5502b98588ed9cce00ba414ee6a7df6bd452eb7`,
and its metric `0.04473285377025604` matches the task record. Both use the same
HR-dev validation path and seal. Scratch best/endpoint errors are 2.137×/2.167×
that reference; no progress snapshot reached the reference, so recorded
`first_matching_epoch` and `time_to_match_pathsource_hrdev_seconds` are
correctly null.

The task is random initialized with no source checkpoint, m=16 HR support, and
HR-dev checkpoint selection; `query_read=false` and `query_metric=null`. The
ledger has the expected append-only RUNNING then DONE pair. The DONE row's
config SHA, output path, best validation metric, and `2.0513671777391673`
GPU-h charge agree with the report (`elapsed_seconds/3600`). No audit mismatch
was found.

## HRSCRATCHM16 task03 and four-task summary audit (2026-10-16)

Read-only audit of task03 and the completed four-task summary; no query/test
access and no GPU, training, or test process was run. The task03 report
SHA-256 is `01729fd3ba7e4e20e7cfcaa703ad0d1e42512dd57c3ea67d9903be06eac5c299`;
its 500-record history is identical to the full training report
(`2142044f3757c841811feaa805c86a88e5d9899afa49da02645604ea71e72911`). The
history minimum and report both give best HR-dev `0.30002690374385566` at epoch
491; epoch 499 endpoint is `0.300297031365335`. The compact inference
checkpoint exists and its byte SHA matches the reported
`421164b77cdfadcb5477681348d426df4c55ae50f8b03d39e161fe8c22ccd934`. Source
inspection confirms the engine saves its best EMA state and inference
compaction retains that state.

Task03 support seal `6c7cdf3065dcfccfd1349fc341bc32ca97234966aae0e5030038380b12777fa6`
and validation seal `e04bfdf4f4b4c6a2fa479f30a1ce053c42d882df3dc4babaafe7a9bebfe7ebc3`
were recomputed from canonical index JSON and match the report. They describe
16 task03/boundary5000003 HR support shards and eight same-task HR-dev shards,
at 256 resolution, fidelity 1 and 20 steps. The matched path-source report is
A0 `pilot_pathsource_comm256`, task03/seed0; its actual hash matches the
recorded `1528ddc0838c3590af9f7bd33e30532240052f188014384c386bef93b0cb8e86`,
and its HR-dev metric `0.060216126055456703` equals the stored reference. It
uses the same HR-dev validation index path/seal. Scratch best error is 4.983×
that reference, no progress snapshot reaches it, and both match-time fields are
correctly null. The report marks random initialization, HR-dev checkpoint
selection, `query_read=false`, and `query_metric=null`.

Recomputed best-error mean across tasks `[0,1,3,5]` is `0.16426903249230235`,
matching `summary.json`; the four matched path-source references average
`0.056156671792268756` (scratch is 2.925× that mean). Raw task GPU charges sum
to `8.693980632270554` GPU-h; the summary and four terminal ledger rows each
sum to `8.693980632270556` (floating-point rounding only). Each DONE ledger
metric and charge matches its task report. The summary lists all four tasks,
the same per-task metrics, and `query_opened_or_loaded=false`. It separately
reports HR-support generation wall time `4786.623911857605` s and solver time
`1609.028743789917` s; these are CPU generation costs, not included in the
four-task GPU training total. No audit mismatch was found.

## HRFULLFTM16 task00 attempt-fit1 read-only audit (2026-10-16)

Audited only `task00_attempt-fit1.json`, its task00 checkpoint/indices/reference,
and the exact task00 ledger rows. I did not inspect the active task01 or run
GPU, training, tests, or query/test access. The task report SHA-256 is
`1296eeec02e87321ebbf6ea9b0af5bde0d5e81f3ca757ab6e25eddaec99ca26e`. Its
250-epoch history gives the same best value/epoch as the report:
`0.0718351761577651` at epoch 233; endpoint is `0.07195750554092228` at epoch
249. The compact inference checkpoint's actual SHA matches the recorded
`15b81ff2f0765896be9377b9b000f23bb1cbde7fc28af5a4deabfa460bcf4176`.

The recorded path-source initialization is
`artifacts/checkpoints/phase1/R1/source_path_s0_scratch.pt`; its bytewise SHA
matches `b1e49eb96c558500fd735fba07bd35ee196cd9ee8993846601fda801f5199fbf`.
Support and HR-dev validation canonical seals recompute and match the report:
`6106a9a2cb4be40806c2ed40393f1a0e4b70617ec06a3decd7cdc4e12fbe656a` and
`2faa0874f0c00beb7c55a07f27746ecabb00a41c882ef264ae9629b67e363903`. They
describe 16 task00/boundary5000000 HR support shards and eight same-task HR-dev
shards, both 256-resolution/fidelity-1/20-step, with no overlap.

The matched path-source report hash matches its recorded seal and reports
`0.06831792555749415` on the same task00/seed0 HR-dev validation index. The
full-FT best did not reach that reference (5.15% higher error), consistent with
null `first_matching_epoch` and time-to-match. The report specifies
`path_source_pretrained`, HR-dev selection over all 250 epochs,
`query_read=false`, and `query_metric=null`. The ledger has one RUNNING and one
DONE row for this attempt; the DONE row's metric, config SHA, output path, and
`1.0165096217108334` GPU-h charge match the report and elapsed-seconds
calculation. No audit mismatch was found.

## HRFULLFTM16 task01 attempt-fit1 read-only audit (2026-10-16)

Audited only `task01_attempt-fit1.json`, the task01 checkpoint/indices/reference,
the task01 entry in `task_results.json`, and its exact ledger rows. I did not
inspect the active task03, run GPU/training/tests, or access query/test data.
The report SHA-256 is
`0fe5048815361ad3a0828a767eed48a7f99a3883b33de5b69b84b4ba961bd82c`. Its
history and progress snapshots each cover all 250 epochs. Recomputed minimum
is HR-dev `0.05558206179412081` at epoch 231, matching the report; the epoch
249 endpoint is `0.05571089384611695`. The compact inference checkpoint's
actual SHA matches the reported
`8c7f28729575d6afff913e83a7de48270b1178a863b728289c8d62da4117236c`.

The path-source initialization file's actual SHA matches the recorded
`b1e49eb96c558500fd735fba07bd35ee196cd9ee8993846601fda801f5199fbf`.
Canonical support and HR-dev validation index seals recompute and match the
report: `9285f576fb3d69a9c376e1edd117621435e45bdf4a41e0a11d4ef880b88948d7`
and `558c3fc8a9218434f3747b03bae79682a746ae0e9b73051b5eeaec8beadc79f2`.
They describe 16 task01 HR support shards and eight task01 HR-dev validation
shards at 256 resolution, fidelity 1, 20 steps; support and validation have no
initial-condition overlap.

The matched seed0 path-source A0 report's actual SHA matches its recorded
`a90f1872108df932af526d22f5502b98588ed9cce00ba414ee6a7df6bd452eb7`; its
metric is `0.04473285377025604` on the same HR-dev validation index. Full-FT
best error is higher, so no snapshot reaches the reference and null
first-match/time-to-match fields are consistent. The report records
`path_source_pretrained`, HR-dev selection over the 250-epoch trajectory,
`query_read=false`, and `query_metric=null`. The task01 `task_results.json`
entry exactly matches the report. The attempt has RUNNING and DONE ledger rows;
the DONE metric, config SHA and output path match, and its
`1.0158786718052741` GPU-h charge equals the report and elapsed-seconds
conversion. No audit mismatch was found.

## HRFULLFTM16 task03 attempt-fit1 read-only audit (2026-10-16)

Audited `task03_attempt-fit1.json`, its task03 checkpoint/indices/reference,
the task03 `task_results.json` entry, and its exact ledger rows. No task05
report, checkpoint, training history, or query/test artifact was opened. A brief
`head` preview of the aggregate results descriptor incidentally displayed only
the task05 support/validation index seal digests; those values were not used in
this audit.

The task03 report SHA-256 is
`ba36283ca9e53b02fe594e5e131d96a0e643faedaf16765b2774c1db8b0a2f78`. Its
history and progress snapshots each contain all 250 epochs. Recomputed minimum
is HR-dev `0.09273512583458797` at epoch 0, matching the report; the epoch 249
endpoint is `0.1500226510106586`. The compact inference checkpoint exists and
its actual SHA matches the report:
`3994371ceaa3cfcd9b1f1b7156ecbbcc56f7c27ff177231f4cc4fee48eb8447b`. The
path-source initialization file's actual SHA matches
`b1e49eb96c558500fd735fba07bd35ee196cd9ee8993846601fda801f5199fbf`.

Support and validation index seals recompute from canonical JSON with the
embedded seal field excluded and match the report:
`6c7cdf3065dcfccfd1349fc341bc32ca97234966aae0e5030038380b12777fa6` and
`e04bfdf4f4b6c4a2fa479f30a1ce053c42d882df3dc4babaafe7a9bebfe7ebc3`.
They describe 16 support and eight validation HR shards for task03 / boundary
5000003 at resolution 256, fidelity 1 and 20 steps; the support index marks
`query_used=false`, and support/validation initial conditions do not overlap.

The matched seed0 path-source reference is A0 `pilot_pathsource_comm256`;
its actual report SHA matches the recorded
`1528ddc0838c3590af9f7bd33e30532240052f188014384c386bef93b0cb8e86`. It is
for task03 / localized_bump / seed0, has HR-dev metric `0.060216126055456703`,
`query_read=false`, and uses the same validation index path/seal. The full-FT
best is worse than this reference, consistent with null first-match/time-to-
match fields. The report specifies HR-dev selection across all 250 epochs,
`query_read=false`, and `query_metric=null`. The task03 entry in
`task_results.json` exactly equals the report. Its DONE ledger row's metric,
config SHA, output path, and `1.0150413266719462` GPU-h charge match; the charge
also equals report elapsed seconds divided by 3600. No artifact or accounting
mismatch was found.

## HRFULLFTM16 task05 attempt-fit1 read-only audit (2026-10-16)

Audited task05's report, compact checkpoint, source initialization, task05 HR
indices and A0 reference, task05 `task_results.json` entry, and exact ledger
rows. All task05 seals below were independently recomputed from task05's own
index JSON with its embedded `sha256` field excluded; no earlier-seen digest
was used as an input. No confirmation query/test data or unrelated task
artifacts were accessed.

The report SHA-256 is
`6d44c676e12bf12467f9690afd9b24049eb5fe55c30d9db721a088dbc4e561fd`. Its
history and progress snapshots each contain all 250 epochs. Recomputed best is
HR-dev `0.06261736205779016` at epoch 218, matching the report; endpoint epoch
249 is `0.06299680741503835`. The compact inference checkpoint's actual SHA
matches the report:
`4e58e5c81f2c8e2cd049ffb4b569a4cce5042cc2f5eabdb91d6b1a318914175e`. The
path-source initialization file's actual SHA matches
`b1e49eb96c558500fd735fba07bd35ee196cd9ee8993846601fda801f5199fbf`.

The support seal independently recomputes to
`11c105287192db6bbe056d1194611d7e394cda7887703f162768337fdfeef5d0`; the
validation seal recomputes to
`ca860c75b774864b30035f81786833d94e0f3cd51143713b8f8844ab5b8ce9b4`. They
cover 16 support and eight validation HR shards for task05 / boundary 5000005
at resolution 256, fidelity 1 and 20 steps. The support index records
`query_used=false`; support and validation initial conditions do not overlap.

The matched seed0 path-source A0 reference is `pilot_pathsource_comm256`; its
actual report SHA matches the recorded
`428089271f4347c38cbf6ffe614f94cbed0150c26efac22e2cece46e1427b186`. It is
task05 / piecewise_smooth / seed0, has HR-dev metric
`0.051359781785868114`, `query_read=false`, and uses the same HR-dev validation
index path and seal. Full-FT best remains above the reference, consistent
with null first-match/time-to-match fields. The report specifies HR-dev
selection over the full trajectory, `query_read=false`, and `query_metric=null`.
The task05 `task_results.json` entry exactly equals the report. Its DONE ledger
row's metric, config SHA, output path, and `1.1411138838005557` GPU-h charge
match; the charge equals report elapsed seconds divided by 3600. No audit or
accounting mismatch was found.

## HR full-FT m=16 end-to-end report audit (2026-10-16)

Read-only review of `artifacts/reports/PHASE1_HR_FULLFT_M16_EXPERT_END_TO_END_20261016.md`,
the full-FT and HRSCRATCHM16 aggregate summaries, the four matched task/seed0
`pilot_pathsource_comm256` HR-dev reports, exact full-FT ledger rows, the cited
progress snapshot, and the earlier task audit entries in this file. I did not
open any per-task report, checkpoint, history, or index in this audit, and did
not access query/test data or run GPU/training/tests.

The full-FT summary lists task metrics `{0: 0.0718351761577651, 1:
0.05558206179412081, 3: 0.09273512583458797, 5: 0.06261736205779016}` and
mean `0.07069243146106602`; the markdown table rounds these correctly. The
matched reference metrics are `{0: 0.06831792555749415, 1:
0.04473285377025604, 3: 0.060216126055456703, 5: 0.051359781785868114}` with
mean `0.05615667179226875`. Their per-task ratios are approximately
`1.051484, 1.242533, 1.540038, 1.219191`; mean ratio is `1.258843` and the
full-FT mean is `25.8843%` above the reference. The HRSCRATCHM16 aggregate
mean is `0.16426903249230235`; `(scratch_mean - fullft_mean) / scratch_mean`
is `56.96545%`, so the reported 56.97% reduction is correct. Task03's cited
endpoint `0.15002265` agrees with the prior independent task03 audit's exact
endpoint `0.1500226510106586`.

The matched A0 reports are all task/family/seed aligned, use the task-matched
HR-dev validation indices, and mark `query_read=false`; each records
`training_or_within_task_checkpoint_selection_used_hr=false` and an LR
validation metric. Full-FT and scratch summaries mark query access false and
selection on HR-dev. The four full-FT DONE ledger rows match summary task
metrics, config SHA and output paths; their GPU-hour charges sum exactly to
`4.18854350398861`, equal to the full-FT summary. The CPU support-generation
wall/solver times in the report also match the summary.

Two report corrections are needed. First, task01's displayed ratio `1.242×`
truncates `0.05558206179412081 / 0.04473285377025604 = 1.242533...`; using the
same three-decimal rounding style as the other cells gives `1.243×`. Second,
the cited `phase1_progress_after_hr_fullft_m16_complete_20261016_1153.json`
snapshot has status `IN_PROGRESS`, no HR-development or training record for
this m=16 / seed0 / support-seed-271829 full-FT run, and its embedded ledger
SHA differs from the current ledger. It therefore does not corroborate this
run's completion or costs; the matching `DONE` rows in `docs/EXPERIMENTS.csv`
and the task audits above do. Apart from these display/citation issues, the
summary arithmetic, task03 endpoint, 56.97% calculation, and ledger charges
are consistent.

## PHASE1 baseline LR3E4 screen report audit (2026-10-16)

Read-only audit of `artifacts/reports/PHASE1_BASELINE_LR3E4_SCREEN_20261016.md`,
the six R5 and paired seed0 1e-4 task-result scalar/provenance fields, their
12 HR-dev reports, the two matching summaries, the stated config, matching
ledger rows, and the cited 12:50 progress snapshot. I did not inspect task
histories, checkpoint files, or index files, and did not access query/test data
or run GPU/training/tests. The prior three-seed Stage-2 aggregate claim in the
report was outside this audit's allowed evidence and was not rechecked.

The six R5 LR-validation metrics are
`[0.16078083941247315, 0.11531357355415821, 0.0902660969295539,
0.1060682518640533, 0.12062085208017378, 0.1531291013583541]`, with selected
epochs `[2,3,1,0,8,2]`; these match the report, its summary, and the six DONE
ledger metrics. Their equal-task mean is `0.12436311919979441`. The paired
seed0 1e-4 LR metrics average `0.040232126989091434`, giving ratio
`3.091139557`, consistent with `3.091×` in the report.

All six pairs match on task/family, seed 0, support seed 271829, m=4, support
index path/seal, and 64x64 LR-validation path/seal. In each arm, the HR-dev
report's LR metric equals the corresponding task result's selected LR metric,
and its checkpoint hash matches that task result's checkpoint hash. Each
paired HR report uses the same task-specific HR-dev validation path/seal; all
12 reports record `query_read=false` and no HR-based checkpoint selection.
The R5 HR metrics are
`[0.17564816002268344, 0.09549712450243533, 0.11483408904168756,
0.16877616327255962, 0.20089715875219552, 0.12378177298232912]`; the paired
1e-4 values are
`[0.07271281073335557, 0.048056444781832394, 0.0417717567528598,
0.06366512620588766, 0.08776810296112672, 0.04268694879719988]`. Their means
are `0.14657241142898178` and `0.059443531705377`; ratio of means is
`2.465741978`, or `146.5742%` higher. Per-family ratios
`[2.416,1.987,2.749,2.651,2.289,2.900]` round to the report table. R5 and
paired summaries both mark query data unopened; task results have null query
metrics, and the ledger notes query=false.

The six R5 DONE charges total `0.9245288309494369` GPU-h; six HR-dev
EVALUATION rows total `0.004192632427782681`, for `0.9287214633772196` GPU-h,
which rounds to the report's training, evaluation, and combined costs. Ledger
metrics, config fingerprint, and output paths match the result artifacts.
The cited snapshot contains the exact R5 training total and mean, the HR mean,
and all six HR report hashes; its summary hash matches the actual R5 summary.
Its terminal ledger sum `69.70597208354533` rounds as reported, and the stated
`2.294028` headroom is arithmetically consistent with a 72-GPU-h exploration
cap after preserving 48 GPU-h. The snapshot explicitly warns that terminal
sum excludes active jobs and early untimed HR evaluations, so headroom is a
point-in-time estimate; the report's “if no GPU work is active” condition does
not mention the untimed-evaluation exclusion. The snapshot's ledger hash is
historical and differs from the subsequently appended current ledger; its
R5-specific summary/report hashes still match.

Configuration fingerprint `645a9ef6037f9833b9f8859eaab043e0717ae930ba55fe9b94e68b14094b9172`
recomputes from the run's canonical descriptor and matches the summary, task
results and ledger. The raw YAML file byte hash is
`a77ee30bb8ed08a11ce35d183d990857049a3995257e7024d72a29888fc4ec0b`; the
report would be clearer if it labels `645a...` as the resolved descriptor
fingerprint rather than leaving readers to interpret it as the YAML byte hash.
The config itself specifies R5 tasks 0–5, seed 0/support seed 271829, m=4,
LR 3e-4, 100 epochs, cosine minimum LR 1e-6, full adaptation, and commutator
weight 0.05 over two steps, consistent with the report. No discrepancy was
found in the six-family metrics, paired comparison, selection/query protocol,
or cost arithmetic.

## Independent audit: R5 LR=3e-5 screen (2026-10-16)

Read-only audit of `artifacts/reports/PHASE1_BASELINE_LR3E5_SCREEN_20261016.md`, the six low-side task results and HR-dev reports, matched seed-0 LR=1e-4/3e-4 development results, `docs/EXPERIMENTS.csv`, and `artifacts/reports/phase1_progress_after_r5_complete_20261016_1349.json`. No query/test data, histories, GPU, training, or tests were accessed. All six R5 tasks report 100 epochs; their LR best epochs are `[3,11,18,6,6,13]`, and HR-dev reports confirm LR-only checkpoint selection. Across the three arms, all six comparisons match task/family, model seed 0, support seed 271829, m=4, and support/LR-validation/HR-validation paths and seals. HR reports point to the corresponding selected checkpoint, report the same LR metric as the task result, and set `query_read=false`; checkpoint byte hashes, task-report hashes referenced by HR reports, and all six HR-report hashes referenced by the snapshot recompute correctly. R5 summary/config fingerprints and snapshot summary/report hashes also match.

Recomputed LR means for LR `[3e-5,1e-4,3e-4]` are `[0.02485536744, 0.04023212699, 0.12436311920]`; relative to 1e-4, 3e-5 reduces mean LR error by 38.22%, while 3e-4 has 3.091x the 1e-4 error. HR-dev means are `[0.04472520453, 0.05944353171, 0.14657241143]`; 3e-5 is 24.76% below 1e-4 and 69.49% below 3e-4. Per-family HR 3e-4/3e-5 ratios round to `[2.416,1.987,2.749,2.651,2.289,2.900]`, matching the report. Six 3e-5 training charges sum to `0.9246865112` GPU-h and their six HR-dev charges to `0.0042212534`, total `0.9289077646`, matching report rounding and DONE ledger rows. The report correctly states this is a single model seed/support draw and cannot establish or upgrade a three-seed champion. Its caveat that the snapshot budget/headroom excludes early untimed HR-dev evaluations is accurate.

**Initial raw-sum discrepancy (resolved by the follow-up below):** the snapshot's then-current `terminal_ledger_gpu_hours=70.6348798481` matched its recorded ledger SHA-256 (`0ee708ab...`) and the current ledger file SHA, while summing every nonempty `elapsed_gpu_hours` cell in that CSV across statuses yielded `71.0576142692`. The following budget-ledger audit reconciles the difference by identifying distinct retries omitted by trial-ID-only max and duplicate R3 exclusion annotations counted by a raw row sum. No other material inconsistency was found in the audited R5 report or its permitted evidence.

### Budget-ledger aggregation follow-up (2026-10-16)

The prior R5-screen note's comparison against the raw sum of every nonempty ledger cell was not a valid terminal-charge comparison: it double-counted four R3 `EXCLUDED_RESOURCE_CONTENTION` annotations that repeat the corresponding `DONE` row for the exact same `(trial_id, config_sha256, raw_output_path)` and GPU-hour amount. The corrected aggregation rule takes the maximum charge for that attempt key, while adding every `HRDEV` `DONE` execution. Independent recomputation over `docs/EXPERIMENTS.csv` gives `70.75322368983164` GPU-h for non-HR attempt keys plus `0.18230948400722424` GPU-h across 244 HRDEV DONE executions, total `70.93553317383886` GPU-h. This matches `artifacts/reports/phase1_progress_after_r5_complete_20261016_1355_corrected.json`; snapshot SHA-256 is `ed21cc8adecd08d15e00b5e4c3e650fb60f35abdda90a2be52a7ce542aa3d2e1`, and its ledger SHA `0ee708ab1d9f181a06b0709e1db6f44301e620e9293ab80ff63cd392253bace7` matches the current CSV.

The old trial-ID-only maximum (with HRDEV executions additive) reproduces the prior snapshot total `70.63487984814115`. Its `0.30065332569771` GPU-h undercount is exactly the cost of three distinct retry attempts: R1 source-final resume's separate-config DONE attempt (`0.29799960917638`; the FAILED attempt at `1.22213748918497` remains separately charged), plus two R1 smoke reruns under different config hashes (`0.00002802287468` for ordinary smoke after its failed attempt, and `0.00262569364665` for scratch smoke before its successful rerun). Thus the two smoke retry cases add `0.00265371652133` GPU-h. Each R3 contention-excluded row matches a DONE row on the full attempt key and charge; the four annotations total `0.12208109540361` GPU-h but add zero under max-per-attempt aggregation. Consequently the naïve all-row sum `71.05761426924` exceeds the corrected terminal total by exactly those duplicate annotations, while the old snapshot was lower because it omitted the three distinct retries. This reconciles the earlier `0.42273442110` raw-sum versus old-snapshot gap; it is not an unexplained accounting discrepancy. No query/test, GPU, or test execution was involved.

## Independent audit: seed-1 R5 LR=3e-5 replication (2026-10-16)

Read-only audit of the six seed-1 R5 task reports/configs/checkpoints, six HR-dev reports, their ledger rows, the dated replication card, and final snapshot `artifacts/reports/phase1_progress_after_seed1_hrdev_complete_20261016.json`. No checkpoint was loaded; no query/test data, GPU, or tests were accessed. The replication YAML byte SHA-256 is `e3bb759291ed462f81dfa2adb4dfcbd19a53c5b7b20ac225413cd31df2ac6ddd`, matching the task descriptors' `trial_spec_sha256`; the base calibration YAML hash is `841d6e24a37515460f1ed1a7e681433759f17126d77a0cd6bbf5aba82e96f119`. For every task, the canonical descriptor hash recomputes and matches `config.json`, task JSON, summary JSON, and R5 ledger. All six task JSONs report 100 epochs with history epochs 0–99; each `best_epoch` is the argmin of LR-development validation history, and the selected metric equals the recorded best value. Raw checkpoint-file hashes match all six compact-checkpoint seals. All tasks use seed 1, support seed 271829, m=4, full FT, LR `3e-5`, cosine 100 epochs/min-LR `1e-6`, and direct 64-to-256 two-step consistency at weight `.05`; query metric is null.

The task-specific support-index and LR-validation-index `sha256` fields are canonical seals, not raw file-byte hashes. Recomputing `sha256_json(index_without_sha256)` for all 12 index files matches the seals in their task JSONs. The same canonical-seal check passes for all six HR-validation indices. All six tasks point to the same seed-1 final-only source checkpoint, whose raw SHA-256 recomputes to `8b9509f482b2029c0c12302449c93509762991e72a3343978a2bd2b55273507b`; each task descriptor and task report agree on that path/hash. Each HR report's task-result hash and compact-checkpoint hash recompute from the referenced files, its LR metric equals the corresponding task's selected LR metric, its model config hash matches the task config, and its report SHA matches the final progress snapshot.

| Task / family | LR-dev best epoch | LR-dev metric | HR-dev metric | Train GPU-h |
|---|---:|---:|---:|---:|
| 00 / high_frequency | 5 | 0.0251744271 | 0.0599293678 | 0.1543900661 |
| 01 / amplitude_ood | 8 | 0.0296606313 | 0.0300229727 | 0.1544305793 |
| 02 / matern_gp | 5 | 0.0240534448 | 0.0312742609 | 0.1543726039 |
| 03 / localized_bump | 6 | 0.0260514874 | 0.0522114087 | 0.1543514565 |
| 04 / mixed_fourier | 4 | 0.0252136014 | 0.0491959505 | 0.1543851205 |
| 05 / piecewise_smooth | 10 | 0.0243845930 | 0.0438281372 | 0.1543622083 |

For each HR report, the mean over eight per-rollout values recomputes to `metrics.rollout_rel_l2`, the per-step vector has 160 values (20 rollout steps × 8 rollouts), and HR selection is explicitly false. All reports are development-only at 256 resolution with `query_read=false`. Recomputed six-task LR mean is `0.02575636418090047`; HR mean is `0.044410349642081806`. The six R5 DONE charges sum to `0.926292034616664` GPU-h; the six timed HRDEV DONE charges sum to `0.004196965066665952`, total `0.93048899968333` GPU-h. Every charge and metric matches its unique task/HRDEV DONE ledger row; ledger query-metric cells are empty. The card's overall cost estimate `0.931556` GPU-h remains conservative in aggregate, although task 01's `0.1544305793` training charge exceeds its stated prior per-task maximum `0.15439786` by `0.00003272` GPU-h.

The final snapshot's SHA-256 is `783bee79b54f71304ae069b123b310884e7d32afc178d3cd63a1c8a7139b8d05`; its ledger SHA `37148f2ca48448dc2aaf8e17d1236b6270405df456192527bb2abb75fe746d81` matches the current CSV. All six task summary hashes and all six HR-report hashes in the snapshot match the actual files. Recomputing terminal usage with max-per-`(trial_id, config_sha256, raw_output_path)` for non-HR attempts and additive charges for each HRDEV DONE gives `71.8660221735222` GPU-h, matching the snapshot. It reports `query_read=false` and `confirmation_evidence=false`; nominal room under the 72 GPU-h exploration cap is `0.1339778264778` GPU-h before historical untimed HR-dev overhead.

The card provides the LR runner command template, and the task descriptors/ledger show the documented attempt suffix `lr3e5seed1rep-task00` through `task05`. For literal reproduction, clarify that the suffix's task number must be zero-padded to two digits (the numeric `--task` value remains 0–5). The card requires HR-dev evaluation after all six fits but does not give its CLI command; the recorded outputs are reproducible through `scripts/evaluate_phase1_hrdev.py --task-result <taskXX.json> --task N --device cuda`, which defaults to the matching HR index and report-name template. Existing outputs require a fresh `--output` path for a rerun. Apart from these command-template clarity points and the noted tiny single-task envelope exceedance, no material artifact, metric, provenance, query-discipline, or ledger discrepancy was found.
