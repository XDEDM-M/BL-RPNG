# Stage‑2 HR-dev 结果汇总与论文主张判读

**状态：冻结 Stage‑2 已完成。** 结果只来自既定模型种子 `[0, 1, 42]`、support seed `271829`、`m=4` 和六个固定开发边界族；没有增加种子、临时种子或更改配置/划分。此交付前没有启动任何额外实验。每个单元包含 8 条 rollout，共 864 条 HR-dev rollout。所有误差均为 256×256、20 步 rollout relative L2，越低越好；每个 task 的 HR-dev 指标是 8 条 rollout 的等权均值。HR-dev 仅用于本轮开发评估，没有读取 query/test 数据，也没有用 HR 选择 checkpoint。

## 协议与完成情况

- 六个候选 × 三个模型种子 × 六个边界族，共 **108/108** 个 LR→HR-dev 单元完成。冻结 support/LR-validation/HR-dev 清单合计 96/48/48 个 shard，三角色 pairwise disjoint。LR provenance audit 为 108/108 且 `query_access=false`；HR-dev 独立审计从 108 份原始 JSON 重新计算每格指标、候选均值、seed 均值、family 均值和排名，均一致。
- checkpoint 只按 64×64、20 步 LR validation 选择；独立复核确认 task result、配置 descriptor、checkpoint SHA、HR index canonical seal 和所有相关 shard SHA 均匹配，报告声明 `query_read=false`、HR 未参与训练或单任务 checkpoint selection。
- 汇总脚本曾把 index 的 canonical content seal 误校验成 JSON 文件字节 SHA，导致汇总中断。现已按 `read_index()` 的协议校验内嵌 seal，并继续逐个核验 shard 哈希；没有重跑训练、HR 推理或改动任何数据/指标。脚本修正在 [`scripts/summarize_phase1_hrdev.py`](../../scripts/summarize_phase1_hrdev.py)。
- 完整 raw reports、重试/恢复映射和 LR 审计见 [`phase1_hrdev_evaluations/`](phase1_hrdev_evaluations/)、[`phase1_stage2_attempt_aliases_20261015.json`](phase1_stage2_attempt_aliases_20261015.json) 与 [`phase1_stage2_lr_retry_audit_20261015.json`](phase1_stage2_lr_retry_audit_20261015.json)。

## 候选含义与总体排名

排名按 18 个 seed×task HR-dev cell 的不加权均值排序；worst-family 是跨三个模型种子的族均值最大值。候选定义与冻结配置见 [`configs/phase1_stage2_pilot.yaml`](../../configs/phase1_stage2_pilot.yaml)。

| 排名 | 简称 | 候选 | HR-dev 均值 | 最差族均值 | Seed 0 | Seed 1 | Seed 42 | 相对 Ordinary full |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | C256 | `pilot_pathsource_comm256` | 0.053472089 | 0.067265023 (mixed_fourier) | 0.057361478 | 0.051004753 | 0.052050034 | +2.57% |
| 2 | A2 endpoint | `pilot_pathsource_endpoint_ratio015` | 0.054025624 | 0.072187248 (mixed_fourier) | 0.056375292 | 0.053174043 | 0.052527537 | +1.56% |
| 3 | A1 firstpath | `pilot_pathsource_firstpath_ratio015` | 0.054093897 | 0.072543684 (mixed_fourier) | 0.056337479 | 0.053496924 | 0.052447287 | +1.43% |
| 4 | A3 direction | `pilot_pathsource_endpoint_direction_ratio015` | 0.054230212 | 0.072121338 (mixed_fourier) | 0.056300071 | 0.053555958 | 0.052834606 | +1.19% |
| 5 | Ordinary full | `pilot_ordinary_full_comm256` | 0.054881172 | 0.076181195 (mixed_fourier) | 0.059443532 | 0.053190336 | 0.052009648 | +0.00% |
| 6 | Scratch | `pilot_lifted_scratch_lr3e-4_cos500_wd1e-4` | 0.509043588 | 0.688090617 (high_frequency) | 0.496913402 | 0.514209018 | 0.516008344 | -827.54% |

“相对 Ordinary full”一列为相对误差下降率，正数表示更低误差，负数表示更高误差。

`C256` 为 path-source 初始化 + NLT 适配器 + commutator-only 256 校准目标（weight 0.05，2-step）；`A1` 为 path-source + NLT + first accepted Newton correction path target（ratio 0.15）；`A2` 为同配置的 endpoint increment；`A3` 为同配置的 endpoint-direction/first-path-norm target；`Ordinary full` 为 ordinary final-only source + full fine-tuning + 同 commutator 目标；`Scratch` 为 lifted scratch、全参数训练、500 epoch、cosine、weight decay `1e-4`。

## LR validation 指标：按 seed × 边界族

下表报告 LR checkpoint selection 使用的 64×64、20 步 validation relative L2 最小值；模型 seed、任务族和支持集与 HR 表完全匹配。LR audit 将 immutable recovery/retry alias 映射回对应冻结候选。

| 候选 | Seed | HF | Amplitude OOD | Matérn GP | Localized bump | Mixed Fourier | Piecewise smooth | 六族均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C256 | 0 | 0.020945114 | 0.026371587 | 0.018362553 | 0.027411289 | 0.020474365 | 0.022568278 | 0.022688864 |
| C256 | 1 | 0.022703295 | 0.028957645 | 0.018329907 | 0.024261603 | 0.019176767 | 0.020289750 | 0.022286495 |
| C256 | 42 | 0.020091916 | 0.022276101 | 0.018359359 | 0.018155601 | 0.017623860 | 0.020054568 | 0.019426901 |
| A2 endpoint | 0 | 0.021029377 | 0.026970081 | 0.018284000 | 0.027264112 | 0.019895661 | 0.021945669 | 0.022564817 |
| A2 endpoint | 1 | 0.021927400 | 0.029363005 | 0.018380045 | 0.023912198 | 0.018857211 | 0.019657496 | 0.022016226 |
| A2 endpoint | 42 | 0.019298135 | 0.022194927 | 0.018241798 | 0.018060290 | 0.017688813 | 0.019811998 | 0.019215994 |
| A1 firstpath | 0 | 0.021023234 | 0.027059846 | 0.018328955 | 0.027136980 | 0.019998141 | 0.021951105 | 0.022583044 |
| A1 firstpath | 1 | 0.021914836 | 0.028852287 | 0.018432301 | 0.023898068 | 0.018917417 | 0.019671181 | 0.021947682 |
| A1 firstpath | 42 | 0.019179032 | 0.022204238 | 0.018384904 | 0.018102728 | 0.017651276 | 0.019762907 | 0.019214181 |
| A3 direction | 0 | 0.021027999 | 0.026972036 | 0.018267683 | 0.027177955 | 0.019849618 | 0.021935164 | 0.022538409 |
| A3 direction | 1 | 0.021965420 | 0.029615376 | 0.018404979 | 0.023910166 | 0.019073132 | 0.019670389 | 0.022106577 |
| A3 direction | 42 | 0.019310375 | 0.022171943 | 0.018360098 | 0.018114645 | 0.017720870 | 0.019796519 | 0.019245742 |
| Ordinary full | 0 | 0.043375408 | 0.048047667 | 0.033220853 | 0.036945639 | 0.042129779 | 0.037673416 | 0.040232127 |
| Ordinary full | 1 | 0.037029018 | 0.045844316 | 0.037664394 | 0.039230308 | 0.032245910 | 0.046430797 | 0.039740790 |
| Ordinary full | 42 | 0.039015362 | 0.037214914 | 0.030638060 | 0.031110722 | 0.036433897 | 0.042092109 | 0.036084177 |
| Scratch | 0 | 0.518354410 | 0.350545394 | 0.424989895 | 0.408128392 | 0.471626012 | 0.392609412 | 0.427708919 |
| Scratch | 1 | 0.538656096 | 0.393020513 | 0.474808149 | 0.410299672 | 0.545446345 | 0.402029603 | 0.460710063 |
| Scratch | 42 | 0.547625704 | 0.380540155 | 0.461562402 | 0.399693718 | 0.541567013 | 0.404969607 | 0.455993100 |

LR 全部 108 格的输入为 [`phase1_stage2_lr_retry_audit_20261015.json`](phase1_stage2_lr_retry_audit_20261015.json)，其中同时保留 checkpoint SHA、候选 alias 和 query-access 记录。
## 每个 seed × 边界族的完整矩阵

每格为该 seed、该候选、该边界族的 HR-dev rollout relative L2；每行均值为六族等权平均。

| 候选 | Seed | HF | Amplitude OOD | Matérn GP | Localized bump | Mixed Fourier | Piecewise smooth | 行均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C256 | 0 | 0.068317926 | 0.044732854 | 0.048678163 | 0.060216126 | 0.070864019 | 0.051359782 | 0.057361478 |
| C256 | 1 | 0.051846488 | 0.028369401 | 0.047545823 | 0.060144286 | 0.064788322 | 0.053334198 | 0.051004753 |
| C256 | 42 | 0.050400067 | 0.032495493 | 0.051912263 | 0.058069980 | 0.066142729 | 0.053279674 | 0.052050034 |
| A2 endpoint | 0 | 0.070338371 | 0.036444097 | 0.043638185 | 0.062603172 | 0.073696611 | 0.051531319 | 0.056375292 |
| A2 endpoint | 1 | 0.064623795 | 0.031922379 | 0.039823964 | 0.063207775 | 0.069459528 | 0.050006815 | 0.053174043 |
| A2 endpoint | 42 | 0.058629482 | 0.034186478 | 0.039625111 | 0.058677122 | 0.073405605 | 0.050641421 | 0.052527537 |
| A1 firstpath | 0 | 0.070551518 | 0.036368077 | 0.043462314 | 0.062624250 | 0.073609427 | 0.051409285 | 0.056337479 |
| A1 firstpath | 1 | 0.063887169 | 0.033867150 | 0.039884928 | 0.063143236 | 0.070370520 | 0.049828542 | 0.053496924 |
| A1 firstpath | 42 | 0.057279045 | 0.034240667 | 0.039796000 | 0.058727543 | 0.073651106 | 0.050989360 | 0.052447287 |
| A3 direction | 0 | 0.070517811 | 0.036396089 | 0.043864089 | 0.062606167 | 0.072932739 | 0.051483528 | 0.056300071 |
| A3 direction | 1 | 0.064292904 | 0.034245418 | 0.039972528 | 0.063191088 | 0.069654203 | 0.049979609 | 0.053555958 |
| A3 direction | 42 | 0.059364114 | 0.034221652 | 0.039736173 | 0.058989340 | 0.073777071 | 0.050919286 | 0.052834606 |
| Ordinary full | 0 | 0.072712811 | 0.048056445 | 0.041771757 | 0.063665126 | 0.087768103 | 0.042686949 | 0.059443532 |
| Ordinary full | 1 | 0.049243947 | 0.044800503 | 0.045952828 | 0.055644007 | 0.070984630 | 0.052516102 | 0.053190336 |
| Ordinary full | 42 | 0.069165507 | 0.038717799 | 0.036239220 | 0.049139127 | 0.069790853 | 0.049005380 | 0.052009648 |
| Scratch | 0 | 0.669769894 | 0.330470004 | 0.529949575 | 0.623059860 | 0.432060051 | 0.396171026 | 0.496913402 |
| Scratch | 1 | 0.688140552 | 0.351953430 | 0.557699739 | 0.626570374 | 0.458225144 | 0.402664871 | 0.514209018 |
| Scratch | 42 | 0.706361405 | 0.328197116 | 0.576051287 | 0.629170219 | 0.433165381 | 0.423104655 | 0.516008344 |

### 跨 seed 的边界族均值

| 候选 | HF | Amplitude OOD | Matérn GP | Localized bump | Mixed Fourier | Piecewise smooth |
|---|---:|---:|---:|---:|---:|---:|
| C256 | 0.056854827 | 0.035199249 | 0.049378750 | 0.059476797 | 0.067265023 | 0.052657884 |
| A2 endpoint | 0.064530549 | 0.034184318 | 0.041029087 | 0.061496023 | 0.072187248 | 0.050726518 |
| A1 firstpath | 0.063905911 | 0.034825298 | 0.041047747 | 0.061498343 | 0.072543684 | 0.050742396 |
| A3 direction | 0.064724943 | 0.034954386 | 0.041190930 | 0.061595532 | 0.072121338 | 0.050794141 |
| Ordinary full | 0.063707422 | 0.043858249 | 0.041321268 | 0.056149420 | 0.076181195 | 0.048069477 |
| Scratch | 0.688090617 | 0.336873516 | 0.554566867 | 0.626266818 | 0.441150192 | 0.407313517 |

## 配对比较与论文主张

表中 `Δ = 右侧 comparator − 左侧候选`，正数表示左侧误差更低。名义 95% t 区间使用三个 seed 的六族均值配对差（df=2），固定六任务与单一 support draw 不变；它只描述 seed 间变化，不是 confirmation CI，也不能把 18 个交叉格当成 18 个独立边界样本。

| 配对对照 | Δ（HR relative L2） | A1 相对误差差幅 | seed×族格 A1 胜/负 | 三 seed 名义 95% CI |
|---|---:|---:|---:|---:|
| A1 vs A2 endpoint increment | −0.0000683 | A1 高 0.126% | 8 / 10 | [−0.0006185, +0.0004820] |
| A1 vs A3 endpoint direction | +0.0001363 | A1 低 0.251% | 11 / 7 | [−0.0004168, +0.0006894] |
| A1 vs C256 reference | −0.0006218 | A1 高 1.163% | 6 / 12 | [−0.0050158, +0.0037722] |
| C256 vs Ordinary full | +0.0014091 | C256 低 2.568% | 9 / 9 | [−0.0017118, +0.0045300] |

A1 与机制控制的逐族配对均值差（`A1 − comparator`，负值有利 A1）：

| 对照 | HF | Amplitude OOD | Matérn GP | Localized bump | Mixed Fourier | Piecewise smooth |
|---|---:|---:|---:|---:|---:|---:|
| A1 − A2 endpoint | −0.0006246 | +0.0006410 | +0.0000187 | +0.0000023 | +0.0003564 | +0.0000159 |
| A1 − A3 direction | −0.0008190 | −0.0001291 | −0.0001432 | −0.0000972 | +0.0004223 | −0.0000518 |

- **强普通迁移基线：** 冻结矩阵中最强 ordinary full + comm256 均值为 `0.054881172`；当前最佳 C256 为 `0.053472089`，只低 **2.57%**，nominal 三 seed 配对区间跨过 0，且 seed 42 的 ordinary full 略优（`0.052009648` vs C256 `0.052050034`）。Mission 的 T2 目标是确认集上至少 15% 且配对 95% CI 支持优势；Stage‑2 开发结果远未达到该门槛，不能写成确认优势。
- **目标路径监督：** A1 对 endpoint A2 实际略差 `0.126%`；对 A3 仅略好 `0.251%`，且主要由 seed 42 拉动。效果量很小，区间均包含 0；现有结果不支持 first accepted path 提供稳定的额外 Newton 路径几何收益，也不证明 A1/A2/A3 等效。
- **路径适配相对 C256：** A1 比 C256 高 `1.16%`，但 C256 使用不同目标（commutator），因此不是纯粹的“关闭目标路径微调”A0。缺少匹配的 A0，不能识别目标路径微调本身相对 final-only 的因果收益；缺少 A4。
- **Scratch：** 最佳 transfer 候选 C256 的均值 `0.05347`，相对冻结 scratch `0.50904` 低约 **89.49%**。这说明本设置下 source transfer 明显优于该 scratch 配置；它不替代对强 ordinary-transfer 的比较。
- **尾部：** 五个 transfer 候选的最差族均为 mixed Fourier（C256 为 `0.067265`）；scratch 最差为 high-frequency（`0.688091`）。未删去难族。

## 证据边界与结论

这组结果支持“多边界 path-source 预训练后，少样本目标适配器可以在六个固定未见边界开发族上完成 HR 推理，并明显胜过本轮冻结 scratch 方案”。相对普通 full-transfer 的优势只有约 2.6%，单一 support draw、三个模型种子、六个固定开发任务也不足以支持跨边界族泛化、确认性显著性或完整 Newton trajectory 因果机制主张。当前证据更适合支持一个有竞争力的边界适配系统；Newton 路径特有机制仍未建立。汇总 JSON 中的 `promoted_candidate` 仅指本轮六臂总体均值第一名；按预注册的候选类别进入后续阶段时，应分别保留最佳 path-source、ordinary-transfer 和 scratch 候选。

协议复核者从 raw reports 独立重算全部 108 格、候选/seed/family 汇总及排名，并核对逐步误差、aliases、唯一 LR 最小值、source/checkpoint SHA、冻结 evaluator/rollout 代码哈希、index/shard 哈希和 query discipline；机制复核独立判读配对差异。完整原始计算保存在 JSON 汇总中，每格包含原始 HR 报告路径、报告 SHA、checkpoint SHA 与 LR validation metric。

### 复现入口

- 冻结配置：`configs/phase1_stage2_pilot.yaml`
- HR-dev 汇总 JSON：`artifacts/reports/phase1_stage2_pilot_hrdev_summary_20261015.json`
- LR provenance audit：`artifacts/reports/phase1_stage2_lr_retry_audit_20261015.json`
- aliases 与排除记录：`artifacts/reports/phase1_stage2_attempt_aliases_20261015.json`
- HR 原始报告目录：`artifacts/reports/phase1_hrdev_evaluations/`
- summarizer：`scripts/summarize_phase1_hrdev.py`；主日志：`artifacts/logs/phase1_stage2_recovery/hrdev_summary.log`（第一次汇总因索引哈希校验语义问题失败，修复后使用同一 108 报告重算）。
