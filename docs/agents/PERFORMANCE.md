# 性能研究员审计

审计日期：2026-10-12。范围限于 Allen–Cahn 的代码入口、冻结记录和可复用资产；未启动训练/求解器作业、未访问 GPU、未改主代码。

## 恢复点与边界

- `docs/STATE.md`、`docs/BEST_CONFIG.md`、`docs/BUDGET.md`、`docs/LOG.md`、`docs/EXPERIMENTS.csv` 和 `docs/prereg/CONFIRMATORY.md` 当前均不存在。不能从文件推断本轮已消耗的 GPU 小时或剩余预算。任务书的 120 GPU 小时只是默认上限；主线程应先探测硬件并落盘实际预算，至少预留 48 GPU 小时用于确认与控制。
- `docs/EXPERIMENT_STATUS.md` 和 `docs/experiment_records/allen_cahn_current_state_seed42.json` 表明最近可信结果是 2026-08-31 冻结的 seed-42 Allen–Cahn 重确认。该历史结果可以复核，不能作为新确认集上的多种子证据。
- 正式路径预训练权重 `artifacts/checkpoints/allen_cahn_formal_a_blrpng_seed42.pt` 存在，大小 1,020,276,497 bytes；SHA-256 与 `artifacts/manifests/allen_cahn_method_freeze_gate.json` 中冻结值一致（`ed9c0aaf…bafbdcf9`）。对应 final-only 源权重 `artifacts/checkpoints/allen_cahn_formal_a_final_only_seed42.pt` 也存在（1,020,274,393 bytes）。不要覆盖这两个冻结文件；所有续训、适配候选用新文件名和新配置。
- 源 HR 数据可复用：`artifacts/data/allen_cahn_formal_a_train_fine.pt`（约 6.06 GB）、`artifacts/data/allen_cahn_formal_a_validation_fine.pt`（约 1.21 GB）。目标 LR support/validation 也已落盘：`artifacts/data/reconfirmation/` 有 48 个 m=4 support 索引和相应 validation 索引；每个目标确实来自 64 网格求解数据。这里的 48 个边界任务适合继续做 LR-only 开发。

## 训练、适配、评价入口

| 阶段 | 入口 | 审计结论 |
|---|---|---|
| 多边界 HR 源训练 | `python -m blrpng.cli.train` → `blrpng/training/engine.py::train` | `--objective final_only/blrpng/...`、`--initialize-from`、`--train-data`、`--validation-data`、`--output` 均已支持。调度器支持 multistep、cosine、plateau；支持 EMA、梯度裁剪、早停和恢复快照。可在不改训练器的情况下做延长训练与路径权重配置实验。 |
| 目标 LR 适配 | `python -m blrpng.cli.train_reconfirmation` | `nlt` = 路径预训练 checkpoint + final-only NLT；`nlt_path` = 同一源初始化 + `blrpng_v2` 目标路径损失 + NLT；`ordinary_nlt` = 普通 final-only 源初始化 + NLT；`lr_scratch` = lifted LR 全量训练。训练输入只取 m=1/2/4/8 的 LR support，checkpoint 选择只看目标 LR validation。`--tag` 可为探索结果隔离文件名。 |
| 旧校准任务适配 | `python -m blrpng.cli.train_task` | 支持 `nlt_path`、`nlt_mix`、`lora`、`ordinary_nlt`、`lr_scratch` 等，并有 `--lr`、`--epochs`、`--adapter-rank`、`--run-tag`。旧任务 support/validation/query 文件位于 `artifacts/data/tasks/`；其 HR query 已进入历史评测，后续不可拿来挑新 checkpoint。 |
| 验证选择与查询汇总 | `blrpng.cli.summarize_reconfirmation_validation` / `blrpng.cli.evaluate_reconfirmation` / `blrpng.cli.summarize_reconfirmation_query` | validation 汇总明确记录 `query_opened=False`；HR query evaluator 需要冻结 gate。当前协议禁止根据已打开的 HR query 重新选择方法、权重或 checkpoint。新确认必须换独立边界任务，并先冻结协议。 |

“目标路径微调”实现是真实可用的：`train_reconfirmation.py::METHODS` 将 `nlt_path` 映射到目标 `blrpng_v2` 目标和 `nlt` 适配器；它不是只训练 source path head。`nlt` 与 `nlt_path` 共用 `source_blrpng_checkpoint`，因而已有路径开启/关闭的同初始化对照。

## 当前最强可复用结果

- 冻结主方法：lifted harmonic boundary lift + 多边界 HR Newton-path 预训练 + unseen boundary 上的 LR final-only NLT，支持 m=4。
- `artifacts/reports/allen_cahn_reconfirmation_seed42_stage_20260831.json`：256²、20 步、48 个 unseen boundary tasks、768 条配对 query rollout，`nlt` 平均 relL2 **0.052226**；普通预训练+NLT **0.060533**；lifted LR scratch **0.549589**。相对普通迁移下降 **13.72%**（95% CI 11.88–15.53%），相对 LR scratch 下降 **90.50%**（95% CI 89.86–91.08%）；与两者各赢 48/48 个任务。它仍是单模型 seed 42。
- 该记录显示目标 `nlt_path` / commutator / defect 版本没有通过预设 LR-validation 晋级标准；HR 查询上的小差异不能拿来继续挑选。路径微调代码与对照已经存在，但“路径预训练 + 路径目标微调优于 final-only”仍未得到证据。
- 资源受限的 HR 专家不是饱和 oracle；冻结记录明确禁止声称专家等精度。旧 256² HR query 已打开，所有旧 query 结果均仅作历史评测。

## 预算与资产复用建议

- 有历史计算量依据但无完整预算账：seed-42 正式源预训练在 `artifacts/reports/v2_seed42_calibration_summary.json` 中记录 3,358.87 秒；重确认 m=4 NLT 每个 task 平均训练 10.65 秒，`nlt_path` 平均 17.33 秒，普通 NLT 10.59 秒，LR scratch 训练权重最大。它们是历史耗时，不代表当前机器的未来耗时。
- 可直接复用 frozen source checkpoint 与 HR 数据，不必重新生成 A 集。重确认 NLT adapter 每个约 107 KB；`nlt_path` adapter 约 111 KB。每种方法 48 个现存 m=4 adapter，适合核对训练/恢复协议；不要覆盖其文件。
- 旧 `configs/allen_cahn_adapter_convergence.yaml` 已有 100 epoch、cosine、lr=3e-4、min-lr=1e-6 设置；`artifacts/reports/convergence_findings_task0_exploratory.json` 记载旧试验中延长 NLT 虽改善 LR validation，却损害当前 256 网格迁移，commutator 只能部分缓和。故不建议把“多训 NLT”单独当作已证明改进；探索可复用，但选择不能再依赖那批旧 HR query。

## 建议的三轮可运行搜索

以下命令只用现有 64² support/validation 训练与筛选，不访问旧 HR query。筛选数据的 boundary IDs 是历史任务，结果按开发证据记录；正式确认必须在新边界任务与新 query 上执行。先由主线程完成 GPU 检查、设置本轮实际 GPU-hour cap，并将所有输出放在新目录。

### R1：补足源训练的优化时间

理由：`allen_cahn_formal_a_{blrpng,final_only}_seed42_training.json` 两条 20-epoch 历史曲线都在最后两个 epoch 继续明显下降（path 验证 relL2 从 epoch 17 的 0.05027 降到 epoch 19 的 0.01720；final-only 从 0.03558 降到 0.01707）。对 path 与 final-only 分别做 60 epoch、cosine、1e-4 到 1e-6 的 warm-start 续训；保持原 HR 源训练和验证集不变。续训用新 optimizer，是从 checkpoint warm-start，不是恢复旧 optimizer。

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path('configs/allen_cahn_formal.yaml').read_text())
cfg['experiment']['name'] = 'allen_cahn_phase1_r1_source'
cfg['training'].update(epochs=60, lr=1e-4, lr_scheduler='cosine', minimum_lr=1e-6,
                      early_stopping_patience=12, minimum_epochs=30)
Path('configs/phase1_r1_source.yaml').parent.mkdir(parents=True, exist_ok=True)
Path('configs/phase1_r1_source.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
python3 -m blrpng.cli.train --config configs/phase1_r1_source.yaml --seed 42 --objective blrpng \
  --initialize-from artifacts/checkpoints/allen_cahn_formal_a_blrpng_seed42.pt \
  --train-data artifacts/data/allen_cahn_formal_a_train_fine.pt \
  --validation-data artifacts/data/allen_cahn_formal_a_validation_fine.pt \
  --output artifacts/checkpoints/phase1/r1_path_source_s42.pt
python3 -m blrpng.cli.train --config configs/phase1_r1_source.yaml --seed 42 --objective final_only \
  --initialize-from artifacts/checkpoints/allen_cahn_formal_a_final_only_seed42.pt \
  --train-data artifacts/data/allen_cahn_formal_a_train_fine.pt \
  --validation-data artifacts/data/allen_cahn_formal_a_validation_fine.pt \
  --output artifacts/checkpoints/phase1/r1_final_source_s42.pt
```

先看 source HR validation 是否稳定改善；然后用固定、同预算 LR m=4 开发适配检查跨网格转移。source HR validation 单独改善不能代替下游 HR 指标。

### R2：同初始化下调目标 LR 路径微调，并给强基线调参机会

理由：主对照必须是同一 path-pretrained initialization 的 `nlt` 与 `nlt_path`；同时以 `ordinary_nlt` 和 `lr_scratch` 做实用竞争基线。选 6 个家族各一个任务（0/8/16/24/32/40），固定 m=4，screen seed=42，分别扫描学习率 1e-4/3e-4/1e-3；目标最长 100 epoch cosine schedule 使用现成配置。只按 64² LR validation 选择；不运行 `evaluate_reconfirmation`。

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path('configs/allen_cahn_adapter_convergence.yaml').read_text())
cfg['experiment']['name'] = 'allen_cahn_phase1_r2_adapter'
cfg['data']['source_blrpng_checkpoint'] = 'artifacts/checkpoints/phase1/r1_path_source_s42.pt'
cfg['data']['source_final_checkpoint'] = 'artifacts/checkpoints/phase1/r1_final_source_s42.pt'
Path('configs/phase1_r2_adapter.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
for task in 0 8 16 24 32 40; do
  for method in nlt nlt_path ordinary_nlt lr_scratch; do
    for lr in 0.0001 0.0003 0.001; do
      tag="r2_${method}_lr${lr/./p}_e100"
      python3 -m blrpng.cli.train_reconfirmation \
        --training-config configs/phase1_r2_adapter.yaml \
        --task "$task" --budget 4 --method "$method" --seed 42 \
        --epochs 100 --lr "$lr" --tag "$tag" \
        2>&1 | tee "artifacts/experiments/phase1/${tag}_task${task}.log"
    done
  done
done
```

`phase1_r2_adapter.yaml` 将 R1 source checkpoint 路径复制到新配置，不改原配置。每个候选输出行都含 `best_validation_metric`；按 task 汇总，不根据任何 HR query 结果重新排名。scratch 和 ordinary migration 同样要从 validation 上独立选各自 LR/训练预算，不能只拿历史单点当强基线。

### R3：定位路径表征的目标层与辅助梯度强度

理由：历史 `nlt_path` 在默认 layer=1、path gradient ratio=0.15 时未晋级；当前引擎支持通过配置扫监督层和梯度比例，无需改模型代码。固定 R2 选出的源权重、目标 LR、训练时长与 NLT；对 layer∈{1,3}、ratio∈{0.05,0.15,0.30} 做 6 个主方法配置，每个覆盖上述六任务；final-only NLT 使用完全相同的 source initialization 与同一 validation 作为配对控制。路径目标继续使用真实 64 网格 accepted Newton correction。

```bash
python3 - <<'PY'
from pathlib import Path
import copy, yaml
base = yaml.safe_load(Path('configs/phase1_r2_adapter.yaml').read_text())
base['data']['source_blrpng_checkpoint'] = 'artifacts/checkpoints/phase1/r1_path_source_s42.pt'
for layer in (1, 3):
    for ratio in (0.05, 0.15, 0.30):
        cfg = copy.deepcopy(base)
        ratio_tag = f'{ratio:g}'.replace('.', 'p')
        cfg['experiment']['name'] = f'allen_cahn_phase1_r3_l{layer}_r{ratio_tag}'
        cfg['training']['path_layer'] = layer
        cfg['training']['path_gradient_ratio'] = ratio
        out = Path(f'configs/phase1_r3_l{layer}_r{ratio_tag}.yaml')
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
for cfg in configs/phase1_r3_l*.yaml; do
  stem=$(basename "$cfg" .yaml)
  for task in 0 8 16 24 32 40; do
    python3 -m blrpng.cli.train_reconfirmation \
      --training-config "$cfg" --task "$task" --budget 4 \
      --method nlt_path --seed 42 --epochs 100 --lr 0.0003 \
      --tag "${stem}_task${task}" \
      2>&1 | tee "artifacts/experiments/phase1/${stem}_task${task}.log"
  done
done
```

入选配置须在至少三个 model seeds 与三个独立 support seeds 上复跑；若 HR dev validation 不支持改进，就保留 final-only NLT champion，不开启 blind query 来救候选。为了估计正式效果，必须新冻结一批未参与上述筛选的 boundary families/support/query，按 `docs/MISSION_PHASE1.md` 先写 prereg amendment，再做确认。以上三轮的筛选权重和差异均不得写回旧的 method/checkpoint/query freeze manifests。

## 交给主线程的行动点

1. 先创建并更新 `docs/STATE.md`、`docs/BUDGET.md`、trial ledger；硬件和已消耗预算在本审计中未探测。
2. 把 R1/R2/R3 产物全部放入新目录；确认配置哈希、数据 index/manifest 哈希、Git commit、随机种子与完整 stdout 日志。
3. 新 confirmation 必须重新划分 boundary families。现有 48-task HR query 和 8-task calibration query 均不是未来的 blind selection 集。
4. 先筛 seed 42，再对短名单做多模型种子和 support 重抽；不可根据最好 seed 决策。主 checkpoint 以独立开发验证规则选择。

## R1 revised design audit (2026-10-13)

Read-only review of `configs/phase1_r1.yaml` and the current R1 runner, input audit, LR summary, and HR-dev evaluator. No GPU job or query asset was accessed.

- The screen is internally aligned: 36 configs (9 each for path FT, source-final-only FT, ordinary transfer, and lifted scratch), m=4, and tasks 0/1/3/5 map to four distinct families in the frozen six-family dev manifest. The R1 input audit enforces 64², genuine LR fidelity 0, 20 steps, per-task support/validation counts, and support-vs-validation IC separation. The HR-dev evaluator restricts its index role to independent 256²/20-step validation, so the staged LR screen → HR-dev finalist promotion follows the mission's selection hierarchy.
- **Dispatch blocker: stale-result reuse can mislabel experiments.** `run_phase1_trials.py` names output directories only by trial ID, m, model seed, and support seed, then reuses existing task results without comparing their stored descriptor/config hash (lines 237–255). It also rewrites `config.json` before reuse. `summarize_phase1_r1.py` checks trial ID and task coverage but does not compare result/task config hashes to the current config. An edited config with the same trial ID can therefore be reported as the current run while retaining old checkpoints/metrics. Before launch, make reuse fail closed on descriptor/config-hash mismatch (and verify per-task config hash), or namespace outputs by immutable config hash.
- **Promotion is not yet an executable aggregate.** The runner accepts `--task`/`--seed`, and the HR evaluator emits one task report, but the current summarizer is hardwired to the four screen tasks and one screen seed/support draw. There is no checked all-six-family × ≥3-model-seed finalist HR-dev aggregate or promotion report. Add/prepare that aggregation before calling a candidate promoted; include per-family and seed summaries, checkpoint/index hashes, and reject smoke or missing cells. Keep independent support redraws for the later confirmatory matrix as required by the frozen protocol.
- The 8 LR-validation rollouts select each task's checkpoint epoch and also contribute to ranking the 36 configurations. That's suitable as an exploratory screen but reuses the same validation observations twice and can favor noisy candidates; report it as LR selection evidence, then make finalist choice on all-six-family HR-dev only, with a separate untouched confirmation boundary set.

No other family/horizon/fidelity mismatch found in the reviewed configuration and scripts. Existing HR-dev task manifest explicitly marks these as development-only and says not to reuse them as confirmation tasks.

## R1 resume performance audit (2026-10-13, 13:32 Asia/Shanghai)

Scope was strictly read-only for code/results/data descriptors: no GPU job, training, solver generation, HR development evaluation, or query/test read was performed. This update only records the next dispatch recommendation and implementation risks.

### Evidence-backed current ranking

- The promoted source checkpoints are extremely close on source HR validation: path source `0.0089858848` versus final-only source `0.0090430082` (path lower by about `0.63%`). Evidence: `artifacts/reports/phase1_R1-source-path-resume-s42-resume50.json` and `artifacts/reports/phase1_R1-source-final-resume-s42.json`.
- On the four frozen LR development tasks, path-source + final-only target adaptation at `lr=1e-3`, cosine-100 has mean `0.0279160336`, worst `0.0337401717`; ordinary-source + final-only at the same target schedule has mean `0.0342082131`, worst `0.0372378291`. The paired screen mean is therefore `18.39%` lower for the path source. Evidence: `results/phase1/R1/final_lr1e-3_cosine100_m4_s42_ss271829/summary.json` and `results/phase1/R1/ordinary_lr1e-3_cosine100_m4_s42_ss271829/summary.json`.
- The ordinary-transfer LR sweep is monotone in its three completed cosine-100 learning rates at the aggregate level: `1e-4 -> 0.03902231`, `3e-4 -> 0.03596554`, `1e-3 -> 0.03420821`. Tasks 0/1/3 at `1e-3` select epoch 99, while task 5 selects epoch 17. Thus `1e-3` is the strongest tested common LR, but the first three families still show training-horizon/schedule sensitivity; do not call the ordinary baseline fully tuned until its remaining frozen schedules or an evidence-based local extension are checked.
- The current path-source final-only run selects epochs `96/96/29/37` on tasks `0/1/3/5`, so a fixed shorter horizon would hurt the two hardest early-converging-to-late tasks. Keep the declared 100 epochs for the paired path-FT test.

### Highest-value next R1 experiment

Run `path_lr1e-3_ratio015` across all four frozen screen tasks, one process on one GPU. It is the highest-value next trial because it (1) uses the best tested target LR, (2) uses the historically/default supported path ratio, and (3) supplies the mission-required same-initialization comparison against the already completed `final_lr1e-3_cosine100`. It should precede more ordinary/scratch grid completion because target path FT is still the missing core R1 arm.

Recommended command from the repository root (main thread chooses the currently idle physical device):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 scripts/run_phase1_trials.py \
  --config configs/phase1_r1.yaml \
  --trial path_lr1e-3_ratio015 \
  2>&1 | tee artifacts/experiments/phase1/R1_path_lr1e-3_ratio015_20261013.log
```

Do not add `--task`; a partial-task descriptor occupies the same run namespace and prevents later reuse as the intended four-task run. Do not add `--force`; see the overwrite risk below. After completion, compare paired per-task metrics, mean, and worst task against `final_lr1e-3_cosine100`, and inspect the recorded history for `path_gate_active`, `gradient_cosine`, `path_target_ratio`, and achieved `gradient_ratio`. A nominal ratio alone is not the actual path exposure because `blrpng_v2` applies conflict gating and quadratic annealing.

If ratio `0.15` is technically valid, next test ratios `0.05` and `0.30` at the same `1e-3` LR before moving to lower learning rates. If all three fail to improve, diagnose gate-active fraction/gradient conflict and per-family behavior before spending the full 3x3 path grid. Continue the frozen ordinary and scratch baseline searches afterward; the scratch arm currently has only a two-epoch implementation smoke and is not yet a performance baseline.

### Implementation and interpretation risks

1. **Old path smoke is not current ranking evidence.** `results/phase1/R1/path_lr1e-4_ratio015-smoke/` predates the aligned support-seed namespace and its history includes a nonzero commutator term; the later aligned path-FT smoke was interrupted during the host incident. Neither answers the current frozen `commutator_weight=0`, promoted-source, 100-epoch path-FT question.
2. **Path exposure can silently be much lower than the label suggests.** In `blrpng/training/engine.py`, `blrpng_v2` multiplies the target ratio by `(1-progress)^2` and sets path weight to zero when the shared final/path gradient cosine is negative. Report realized gate-active fraction and achieved gradient ratio for every path trial. If the gate is mostly off, a negative result tests the gated schedule more than path supervision itself.
3. **The path signal has narrow adapter reach by design.** At `path_layer: 1` with NLT, balancing uses the trainable NLT affine parameters through layers 0–1; the probe head also receives path gradients, while later NLT parameters and the projection head are final-loss-only. This is a real target-path update, but a failure can indicate insufficient path-controlled capacity rather than useless labels. That diagnosis should motivate R3 layer/capacity work, not an immediate rejection of the path hypothesis.
4. **Avoid `--force` on scored namespaces.** The runner's normal reuse checks now fail closed on descriptor and artifact hashes, but `--force` deliberately skips loading old results and writes the same `config.json`, task JSON, summary, and checkpoint namespace. This can overwrite scored artifacts and violate the mission's append-only/history-preservation rule. For any intentional rerun, create a new trial ID/round namespace instead.
5. **Ledger round is hard-coded.** `scripts/run_phase1_trials.py` writes ledger field `round: R1` even though it accepts a configurable `experiment.round`. This is harmless for the present R1 command but must be fixed before reusing the runner for R2/R3, or later trials will be mislabeled in `docs/EXPERIMENTS.csv`.
6. **Failure artifact durability is weaker than success artifact durability.** Success JSON uses atomic replacement, directory `fsync`, and read-back verification; the `.failure.txt` path uses plain `write_text`. This does not affect completed scores, but after another host interruption the ledger may reference a missing/truncated failure detail. Preserve console logs and consider routing failure text through the durable writer before a broader campaign.

No query/test access is needed for any recommendation above. HR development validation remains a shortlist promotion step only after the LR screen, and the current single-seed LR leader is not a campaign champion.

## R2 cross-resolution / rollout design audit (2026-10-13)

Scope: read-only inspection of the new R1 LR and HR-development reports, existing non-query data assets, training engine, shard loader, rollout evaluator, and tests. No GPU/training/generation job was started and no query/test asset was read.

### Diagnosis from the new R1 evidence

- Path-source + final-only `lr=1e-3` has four-family HR-development mean `0.05602457`, versus ordinary transfer `0.06219577` (relative reduction `9.92%`). Per-family reductions are `8.22%` (task 0), `21.99%` (task 1), and `18.34%` (task 3), but task 5 is `13.89%` worse (`0.05272269` versus `0.04629440`). Evidence: `artifacts/reports/phase1_hrdev_evaluations/{final,ordinary}_lr1e-3_cosine100_task{00,01,03,05}_s42_ss271829.json`.
- The path-source `lr=3e-4` target adapter is nearly tied at HR-development mean `0.05648068`; learning-rate tuning alone is therefore unlikely to produce the required R2 gain.
- Error accumulation is material. For the current main arm, mean HR-development error from step 1 to step 20 grows `2.44x` on task 0 (`0.03578 -> 0.08747`), `2.04x` on task 3 (`0.03527 -> 0.07209`), and `1.73x` on task 5 (`0.03410 -> 0.05904`). Task 1 grows only `1.29x`. This supports short autoregressive training rather than another one-step-only schedule sweep.
- Existing target support shards already contain complete true-solver 20-step LR trajectories: `artifacts/data/phase1_development/hrdev_20261012_t20_ss271829/*support*.index.json`, four independent rollouts per task at m=4. `RolloutShardDataset` currently discards this sequence structure during training and returns only one transition. No new data generation is required to expose 2–3 step windows.
- Existing code already implements exact refined harmonic lifts, differentiable restriction commutator loss, multiple refined resolutions, and recursive commutator rollouts in `blrpng/training/engine.py`; `tests/test_v2.py::test_exact_multiscale_lift_and_two_step_commutator` exercises the core path. R1 deliberately disabled it, so it is a clean R2 intervention.

### Candidate 1 (run first): direct 64-to-256 two-step consistency adaptation

Use the current path-source checkpoint and final-only NLT target adapter, but set objective `commutator_only`, `commutator_weight` in `{0.01, 0.05}`, `calibration_resolution: 256`, and `commutator_rollout_steps: 2`. Keep `lr=1e-3`, cosine-100 initially so the comparison differs from the R1 main arm only by the unlabeled deployment-resolution consistency term. The 256 branch starts from bilinearly prolonged LR state but recomputes the harmonic lift from its boundary trace; it does **not** claim true 256 labels.

Why first: it directly targets the 64-adaptation to 256-inference gap, is already implemented and tested, needs no new labels, and can be isolated with two weights. The old task-0 `reconfirmation_task00_m4_s42_nlt_commutator_training.json` was essentially neutral on LR validation (`0.034497` versus NLT `0.034461`) and used the older setup/default 128 calibration; it is a warning against a broad weight grid, not evidence against direct 256 HR-development consistency.

Expected changes:

- Add `configs/phase1_r2.yaml` with a new immutable R2 trial namespace and two `commutator_only` trials. Reuse the R1 data manifest/support indices and promoted source checkpoint; do not edit `configs/phase1_r1.yaml`.
- Fix `scripts/run_phase1_trials.py` ledger row from hard-coded `"round": "R1"` to `round_name` before using the generic runner for R2. The runner already forwards singular `calibration_resolution`, `commutator_rollout_steps`, and `commutator_weight`; no engine change is required for this candidate.
- Add/extend an R2 input audit or descriptor check so objective, 256 calibration resolution, two-step setting, source hash, support hash, and `query_opened_or_loaded=false` are explicit.

Smoke acceptance (task 0, two epochs, non-ranking):

1. Existing CPU unit `test_exact_multiscale_lift_and_two_step_commutator` passes.
2. History contains finite positive `commutator_64_to_256`, `commutator_step2_64_to_256`, and `commutator_loss`; `path_loss == 0` and the compact NLT checkpoint round-trips.
3. A 256 forward pass has exact boundary enforcement (max constrained trace error exactly zero), finite gradients, no query path, and peak allocation below a conservative 22 GiB gate. If batch 4 breaches the gate, freeze R2 batch size at 2 in the new base config rather than weakening the loss.
4. The smoke must not be ranked. Full screening should start on tasks 0 and 5 (largest accumulation and the regression family), then cover 0/1/3/5 only if technically sound.

### Candidate 2: true LR short-rollout NLT adaptation (horizon 3)

Add a final-only autoregressive training loss over three consecutive states from each existing 20-step support rollout. Use normalized weights, initially equal `1/3` per step (or configurable discount with weights normalized to sum one), feed each prediction into the next step, and backpropagate through all three steps. Keep the same source checkpoint, NLT parameterization, `lr=1e-3`, cosine-100, and LR full-rollout validation selection. This tests rollout stability without conflating it with the nearly endpoint-equivalent path labels.

Expected changes:

- `blrpng/data/rollout_windows.py`: wrap `RolloutShardDataset` without modifying it; expose only valid start windows and return `future_targets [H,C,H,W]`. Preserve exact current behavior at `H=1`. The initial implementation supports static conditions/boundaries only and fails explicitly on time dependence.
- `blrpng/training/engine.py`: when `supervised_rollout_steps > 1`, recursively call the model, accumulate normalized per-step relative-L2 losses, log `rollout_step1_loss` ... `rollout_stepH_loss`, and use this sum as the final loss. Fail closed if a non-v4 dataset requests H>1. Apply path/physics auxiliaries only if explicitly designed; the first R2 trial should be `final_only`.
- `scripts/run_phase1_trials.py`: forward `supervised_rollout_steps` and `rollout_loss_discount` into the local training config and include them in the immutable descriptor through the trial object. Also apply the R2 ledger-round fix above.
- `tests/test_phase1_rollout_windows.py` plus engine-focused tests: add synthetic v4-window shape/order tests, H=1 numerical-equivalence, H=3 gradient-through-time, explicit time-dependence rejection, and exact boundary enforcement at every predicted step.
- `configs/phase1_r2.yaml`: add one horizon-3 trial. Do not start with a large horizon grid; m=4 provides only 80 transitions, so H=3 is a controlled bias toward stability without collapsing the number of valid windows.

Smoke acceptance (task 0, two epochs, non-ranking):

1. CPU tests establish H=1 equivalence and correct ordered targets/conditions for H=3.
2. GPU history has finite nonzero losses for steps 1/2/3, gradients reach NLT parameters through the recursive path, all predicted boundaries are exact, and peak allocation stays below 12 GiB at 64 resolution.
3. Validation remains the unchanged independent 20-step LR rollout; no support loss or teacher-forced metric is used for selection, and no HR/query metric is loaded by the training runner.
4. Screen tasks 0 and 5 first. Promote to all four development families only if task 0 full-rollout LR error improves without material task-5 degradation; then use the already independent HR-development split for shortlist promotion.

### Order and decision rule

Run Candidate 1 first because it is configuration-only apart from the ledger fix and directly matches 256 deployment. Run Candidate 2 next because the observed step-wise growth gives it a strong mechanistic basis but it requires a small, reviewable engine/data change. Keep them separate for attribution; combine the winning consistency weight with horizon 3 only if at least one isolated arm improves HR-development mean.

Before execution, freeze an exploratory promotion rule such as: at least `3%` lower four-family HR-development mean than `0.05602457`, no family worse by more than `5%`, and explicit reporting of task 5. LR validation remains the checkpoint selector; HR development is only the candidate-level R2 promotion metric. Neither arm may read the historical query/test set.

## R2 rollout-window data helper implementation (2026-10-13)

Implemented the data-only part of Candidate 2 without modifying the existing shard loader, engine, runner, or any data asset:

- New `blrpng/data/rollout_windows.py` defines `RolloutWindowDataset(base, horizon=H)`, where `base` must be a `RolloutShardDataset`.
- It preserves `index`, `index_path`, `cache_rollouts`, and a filtered `(shard_index, step)` `samples` list, so the existing `RolloutBatchSampler` can operate on the wrapper after engine integration.
- H=1 preserves the base length, order, `x`, `y`, and every original label/ID/boundary field exactly, adding only `future_targets = y[None]` with shape `[1,C,H,W]`.
- H>1 keeps only starts satisfying `step + H <= shard.steps` and slices `trajectory[step+1:step+H+1]` from that same shard. It cannot cross a rollout boundary. Original `y` remains the first future target for backward compatibility.
- Multi-step access explicitly rejects v4 `condition_fields` with a time dimension. It also rejects any non-static boundary tensor if a future schema allows one; the current v4 validator already fails such boundary tensors before they can be silently reused.
- Invalid/nonintegral horizons and horizons unavailable in every rollout fail closed.

Integration interface for the main thread:

```python
base = RolloutShardDataset(index_path, cache_rollouts=max(batch_size, 2))
# Apply any boundary/task filtering to base.samples before wrapping.
train_dataset = RolloutWindowDataset(base, horizon=supervised_rollout_steps)
loader = DataLoader(
    train_dataset,
    batch_sampler=RolloutBatchSampler(
        train_dataset, batch_size=batch_size, seed=seed, start_epoch=start_epoch
    ),
)
```

The collated engine batch shapes are `x: [B,C+2+K,H,W]`, `y: [B,C,H,W]`, and `future_targets: [B,rollout_horizon,C,H,W]`. For the first engine integration, only static conditions/boundaries are permitted, so later autoregressive inputs may safely reuse `x[:, C:]`; do not relax the wrapper's explicit rejection until per-step auxiliary inputs are implemented.

New isolated tests in `tests/test_phase1_rollout_windows.py` use only temporary synthetic v4 shards. Five CPU tests pass under the `fnotest` environment with CUDA hidden:

```text
CUDA_VISIBLE_DEVICES='' /opt/miniconda3/envs/fnotest/bin/python \
  -m unittest tests.test_phase1_rollout_windows -v
Ran 5 tests in 0.012s — OK
```

They cover H=1 field-by-field equality, H=3 same-rollout filtering and target order, DataLoader shape `[B,H,C,height,width]`, explicit time-dependent-condition rejection, and invalid/unavailable horizon handling. Both new Python files also pass `py_compile`.

Engine-side tests still required during integration:

1. H=1 loss and parameter gradients must be numerically identical to the current one-step path for a fixed synthetic batch/model state.
2. H=3 must produce finite per-step losses and nonzero gradients on trainable NLT parameters through the recursive prediction at steps 2/3; compare against a detached-recursion negative control so the test proves gradient-through-time rather than only three independent forwards.
3. Every predicted step must equal the lift on the constrained boundary, and the normalized rollout weights must sum to one.
4. A non-v4/materialized dataset requesting H>1 must fail closed rather than silently reverting to one-step training.

No GPU/training/generation job ran, and no query/test data asset was read for this implementation.

## Read-only review of integrated H3 supervision (2026-10-13)

Reviewed `blrpng/training/engine.py`, `blrpng/training/rollout_supervision.py`, `blrpng/data/rollout_windows.py`, `scripts/run_phase1_trials.py`, `configs/phase1_r2_rollout.yaml`, and the focused CPU tests. No GPU/solver/training command was run and no query/test asset was accessed. The main thread reports 40 CPU tests already passing; this review did not rerun them.

### Correctness findings

- **True recursion and BPTT: PASS.** The caller computes step 1 once; `supervised_rollout_loss` feeds each predicted `state` directly into the next model call, with no teacher forcing and no `detach`. The three scalar losses are stacked and averaged. `test_h3_recursive_gradient_not_detached` checks the analytic recursive gradient against a detached-style negative value, and the real-FNO test verifies finite nonzero gradients.
- **Boundary enforcement: PASS for the current static-boundary Allen–Cahn data.** Every recursive step calls `DirichletOperator(..., boundary_lift, boundary_mask)`, so the model reapplies exact constraints. The focused test checks all three predicted frames. `RolloutWindowDataset` rejects time-dependent conditions for H>1; current v4 validation also rejects non-static boundary tensors, preventing silent geometry reuse.
- **Same-rollout windows: PASS.** The wrapper filters `(shard_index, step)` by `step + H <= shard.steps` and slices only that shard's `trajectory`. It cannot cross rollouts. The existing `RolloutBatchSampler` consumes the wrapper's filtered `.samples`, groups by shard, and still batches distinct rollouts in its cache-friendly order.
- **H=1 compatibility: PASS.** Engine wrapping and `supervised_rollout_loss` are both bypassed when `rollout_horizon <= 1`, leaving the original one-step path unchanged. The helper's direct H=1 test also proves original fields/order plus `future_targets[0] == y`, and the loss helper verifies exact value and gradient equality.
- **Fixed-H3 objective: PASS.** For `joint_relative_l2`, the initial one-step loss is replaced by the equal-weight mean of true steps 1/2/3. Validation remains the unchanged independent 20-step LR rollout, so checkpoint selection is not teacher-forced.
- **Curriculum transition arithmetic: PASS but semantics qualified.** With 100 epochs, `1 + int(epoch_progress * 3)` yields approximately 33 epochs H1, 33 H2, and 34 H3. Resume uses the global epoch index, so it returns to the correct phase.
- **Provenance routing: PASS.** The runner forwards horizon/curriculum, hashes the engine/window/supervision modules into the descriptor, uses the R2 namespace and ledger round, and retains LR-only selection text plus `query_metric=None`.

No blocking correctness bug was found for the fixed-H3 GPU smoke.

**Scored-run provenance blocker:** the runner descriptor hashes `phase1_r2_rollout.yaml` but not its referenced `experiment.base_training_config`. Although `load_config()` computes the base YAML hash, `_training_config()` replaces `_config_sha256` with the trial descriptor hash. Editing `configs/allen_cahn_v2_calibration.yaml` can therefore leave the R2 descriptor unchanged and permit stale-result reuse. Before any ranked R2 run, add the base config path and `base["_config_sha256"]` (or its file SHA-256) to the immutable descriptor and persisted task result. This does not block a clearly non-ranking smoke.

### Required qualification before scoring curriculum

The curriculum always constructs an H3-filtered dataset before training. Each 20-step rollout therefore has only starts 0–17, even while `active_horizon == 1`; the first curriculum phase omits the final two one-step transitions that the R1 H1 baseline saw. It is correctly a **horizon curriculum on a fixed H3-window population**, but it is not a strict full-data H1 → H2 → H3 curriculum.

This also changes target exposure: under fixed H3, middle physical targets appear in up to three overlapping windows per epoch, while the first/last targets appear once. There are 72 windows and 216 supervised state targets per task/epoch, versus 80 transitions/targets for H1. Thus H3 uses about 2.7x as many target-loss evaluations and nearly 3x model forwards per epoch despite equal nominal epochs.

Actions:

1. The fixed-H3 smoke may proceed unchanged.
2. Do not describe the current curriculum as reproducing the old H1 stage. Either keep the precise fixed-population wording, or rebuild the dataset/loader at the H1/H2/H3 phase boundaries so each stage uses all valid starts for its active horizon. A dynamic implementation must preserve deterministic sampler epoch/resume state.
3. Report optimizer steps, supervised target evaluations, model forwards, GPU time, and energy alongside nominal epochs. R1 H1 versus H3 is a performance comparison with extra training compute, not a fixed-budget mechanism comparison.

### Fair-baseline audit

- `ordinary_rollout3_lr1e-3_cos100` is a correct same-H3, same-LR, same-epoch direct competitor using the final-only source checkpoint. This makes the main path-source versus ordinary-source comparison fair under the new general rollout technique.
- `scratch_rollout3_lr1e-4_cos100` uses the best completed R1 scratch LR (`0.534786` at `1e-4`, narrowly ahead of `0.538126` at `3e-4` and well ahead of `0.600903` at `1e-3`), so it is a defensible first H3 scratch baseline.
- If `rollout3_curriculum` becomes the main winner, add an **ordinary-source curriculum** arm with identical phase rule. The current config has only ordinary fixed-H3, so a winning curriculum would otherwise mix source-pretraining benefit with a general training trick not granted to the strongest direct competitor.
- Reuse of the existing R1 H1 main/ordinary results is valid for the independently tuned performance table, because H=1 code/data behavior is unchanged. For a fixed-compute mechanism table, add an H1 control with comparable GPU time/forward evaluations or explicitly state the H3 compute multiplier.

### Nonblocking issues to fix before full R2 promotion

1. `supervised_rollout_loss` returns only the mean, so history logs `one_step_final_loss` and the aggregate but not step-2/step-3 losses. Add per-step detached metrics before the full screen; otherwise the smoke cannot distinguish a healthy recursive objective from one dominated by a late-step explosion.
2. The compact checkpoint's standalone metadata does not explicitly retain `rollout_horizon` or `rollout_curriculum`. The runner descriptor/config preserves them, and curriculum history indirectly records the active horizon, but add both to checkpoint `loss_config` and compact-adapter metadata for robust reconstruction.
3. `best_metric_name` still says `ten_step_projected_rollout_rel_l2` while this campaign's validation indices contain 20 steps. The numerical selector uses the actual full validation rollout, so scores are not wrong, but the saved label is misleading and should be corrected before formal R2 reports.
4. Add one CPU integration test that constructs `RolloutWindowDataset(H=3)` and passes it through `RolloutBatchSampler`, asserting each batch's windows remain within their shard and rollout IDs are distinct when batch size permits. Current unit coverage proves each component separately but not their engine wiring.
5. The runner error text still says “R1 runner” for unsupported adapter strategies during R2. This is cosmetic but should be generalized.

### GPU-smoke acceptance recommendation

Proceed with only task 0, two epochs, non-ranking, fixed `rollout3_lr1e-3_cos100`. Require:

- history horizon exactly 3, finite aggregate and per-step losses (after adding diagnostics), nonzero optimizer updates, and no query/HR load;
- exact boundary trace at each recursive step, compact adapter reconstruction, and unchanged 20-step LR validation;
- peak memory below the declared safety gate and GPU time recorded;
- 72 H3 windows per epoch for m=4 × 20-step support, with no unexpected sample-count drift.

After this smoke, screen fixed H3 before curriculum. Curriculum is not a blocker for validating the core recursive implementation.

## Read-only recomputation: H3, R3 mapping, and baseline capacity (2026-10-13)

I recomputed the metrics from the per-task result JSON and independent HR-development reports for tasks 0/1/3/5, seed 42, m=4, support seed 271829. LR values are the unweighted mean of each task's saved `best_validation_metric`; HR values are each report's mean of its eight raw `per_rollout_rel_l2` values, then the equal-task mean. The HR reports agree with their stored `rollout_rel_l2`. No training, GPU work, data generation, or query/test access was performed.

- **R2 fixed-H3 rollout:** LR mean is `0.02642924` (per task `0.02449426/0.03381032/0.02400775/0.02340461`), exactly matching `summary.json`. This is **5.33% below** R1 path-source H1 final-only (`0.02791603`) and **22.74% below** the current ordinary H1 screen (`0.03420821`). Recomputed HR-development mean is `0.05596649`, essentially tied with R1 path-source H1 (`0.05602457`, only 0.10% lower). By family versus R1 H1, H3 improves amplitude-OOD by 7.51% but is worse on high-frequency (+0.05%), localized-bump (+1.34%), and piecewise-smooth (+3.24%). It therefore improves LR rollout fit without materially closing the LR→HR gap. Evidence: `results/phase1/R2/rollout3_lr1e-3_cos100_m4_s42_ss271829/{summary,task00,task01,task03,task05}.json` and the four matching `artifacts/reports/phase1_hrdev_evaluations/rollout3_lr1e-3_cos100_task*_s42_ss271829.json`.
- **Best measured HR-development arm remains direct-256 consistency:** the R2 two-step, weight-0.05 arm recomputes to `0.02810983` LR and `0.05287928` HR-dev, a 5.61% HR-dev reduction from R1 path-source H1 and a 5.84% reduction from H3. H3 is the better LR-only result; direct-256 is the better measured HR-dev result. A combined H3 + direct-256 arm is the clearest method-specific next experiment because the isolated terms lead on different metrics; keep LR-only checkpoint selection and use HR-dev only for promotion.
- **R3 layer-3 path mapping:** raw task JSON recomputes to LR mean `0.02799855`, 0.30% worse than R1 path-source H1 and 0.26% worse than the R1 layer-1 ratio-0.05 path arm (`0.02792730`). The layer-3 signal was active on average 45.8% of batches, but the achieved gradient ratio averaged `0.0079` against mean annealed target ratio `0.0168`. There are no R3 HR-development reports yet. The endpoint-control directory contains only its config and its ledger row remains RUNNING, so A1 versus A2/A3 is not yet a completed matched comparison. Evidence: `results/phase1/R3/path_l3_lr1e-3_ratio005_m4_s42_ss271829/{summary,task00,task01,task03,task05}.json`, `results/phase1/R3/endpoint_l3_lr1e-3_ratio005_m4_s42_ss271829/config.json`, and `docs/EXPERIMENTS.csv`.
- **Symmetric H3 baseline opportunity is required before ranking the H3 path-source arm.** The frozen H3 config defines ordinary and scratch candidates, but their scored result directories are absent; only the path-source fixed-H3 trial has completed. Run fixed-H3 with the best ordinary-transfer initialization (currently ordinary NLT, `lr=1e-3`) and with lifted LR scratch, under the same support, horizon, 100-epoch schedule, and LR-only selection. Ordinary full FT (`lr=3e-5/1e-4`) must first be screened per `BASELINE_CAPACITY.md`; if it beats ordinary NLT, grant that winner the same H3 opportunity. Likewise, current scratch evidence is only 100-epoch cosine (`best lr=1e-4`, LR mean `0.53479`, HR-dev `0.61031`); complete the prescribed 500-epoch `lr=1e-4/3e-4`, weight-decay `1e-4` convergence screen before calling it a strong scratch baseline, then give the best converged scratch arm H3. The configs are not evidence that these runs occurred: `results/phase1/R1B` and ordinary/scratch H3 result folders are currently absent.

**Recommendation:** do not promote H3 from LR alone or cite its gain as a clear HR improvement. First obtain the symmetric ordinary and converged-scratch H3 results and finish the matched R3 endpoint controls; the most promising main-method continuation is then the direct-256 consistency + H3 combination, evaluated on the same four HR-development families. This prioritizes the only isolated intervention with a meaningful HR-dev reduction while testing whether H3 recovers its LR benefit. All comparisons remain exploratory, single-seed, and development-only.

## Stage-2 seed-0 path-source pretraining review (2026-10-14)

Read-only review of the frozen pilot config/card, source progress artifact, source manifest, and completed seed-42 source-continuation histories. No GPU/training was run and no query/test rollout data was read.

- **Frozen setting:** `configs/phase1_stage2_pilot.yaml` requests six source pretrains (path/final × seeds 0/1/42), each from random initialization, with the existing source architecture and HR train/validation sets. `configs/phase1_r1_source.yaml` supplies 60 epochs, minimum 30, patience 12, batch size 4, Adam learning rate `1e-4`, cosine schedule to `1e-6`, and weight decay `1e-6`. The source manifest has 1,000 `a_train` trajectories/boundaries and 200 `a_validation` trajectories/boundaries; the files are 5.7 GB and 1.2 GB. Progress indicates 2,500 optimizer updates per epoch, i.e. 150,000 updates (about 60 passes over the 10,000 one-step examples) at the cap.
- **Progress record needs reconciliation:** `docs/STATE.md`/`docs/LOG.md` report epoch 9/60, 25,000 updates, best `0.049681636885`, and three stale epochs. The on-disk `artifacts/reports/progress/source_path_s0_scratch.json` instead reports epoch 10/60, 27,500 updates, best/current `0.047623343658`, and zero stale epochs. Treat the progress file as a newer improvement only after the main thread reconciles this discrepancy against the active process/recovery metadata; either record gives no basis for early stopping.
- **Comparison to prior source curves:** existing seed-42 path and final-only runs were *warm starts from the frozen historical HR checkpoints*, so their absolute errors cannot be compared directly to random-init seed 0. Their late-epoch trend is useful for stopping-rule calibration: path validation improved from `0.01571` at epoch 0 to `0.00943` at 39 and best `0.0089859` at 56; final-only improved from `0.01578` to `0.00959` at 39 and reached its best `0.0090430` at 59. Their longest consecutive non-improvement spans were only 4 and 3 epochs. Thus a 60-epoch cap can still censor late gains, while patience 12 did not prematurely stop either recorded continuation.
- **Assessment:** 60 epochs with minimum 30 and patience 12 is a reasonable fixed *pilot budget/stopping rule*. The supplied epoch-9 snapshot has only three stale epochs—well below patience—and the on-disk epoch-10 snapshot has improved again. The early scratch curve drops from roughly `0.07` to below `0.05`, which is healthy validation progress, not an undertraining or optimizer-failure diagnosis. There is no committed per-epoch training-loss, gradient, or LR trace for this active scratch run, so train/validation divergence and optimization pathologies cannot yet be assessed. The older warm-start runs' late improvements do mean that reaching epoch 60 should not automatically be called saturation.
- **Budget and advice:** at the active run's recorded throughput (about 1,361 s for 10 epochs), 60 epochs project to roughly 2.27 GPU-hours per source run, or about 13.6 GPU-hours for all six source objective/seed pairs. This fits the current `60.29` GPU-hour exploration balance, leaving the confirmation reserve untouched; the estimate is approximate and excludes already active elapsed time. Keep the frozen 60/12 rule across the pilot, preserve best-source-validation checkpoints, and report any run that reaches epoch 60 still improving (or before patience expires) as right-censored rather than converged. If an extension is later justified by the completed curves, use a fresh namespace and extend the relevant objective/seed cells symmetrically; first archive validation, training loss, LR, gradient/path diagnostics, and best/stale history. Three stale epochs alone do not justify changing the schedule or extending now.

## 120-GPU-hour confirmation feasibility and compute allocation (2026-10-14)

Read-only budget estimate from `docs/MISSION_PHASE1.md`, the frozen confirmation preregistration, `docs/BUDGET.md`/`docs/EXPERIMENTS.csv`, current `STATE.md`, and timed task JSONs. No GPU or query/test data was accessed. The target costs below are the recorded training `gpu_hours` divided by the four development tasks; they exclude source pretraining and HR evaluation.

### Measured cost of the smallest preregistered primary set

The preregistration selects at most three primary methods: path-source candidate, strongest ordinary-transfer baseline, and lifted LR scratch. At `m=4`, a full confirmation cell is 48 tasks × 5 model seeds × 3 crossed support draws = 720 task adaptations per method.

| Method proxy (single-seed, four-task development run) | Measured GPU-h/task | 720 adaptations |
|---|---:|---:|
| Path source + direct-256 two-step consistency (`R2`) | 0.123703 | 89.07 GPU-h |
| Ordinary full-FT + direct-256 consistency (`R2C`, current strongest measured ordinary arm) | 0.138604 | 99.80 GPU-h |
| Lifted scratch, 500 epochs (`R1B`) | 0.101563 | 73.13 GPU-h |
| **Three-method target-adaptation subtotal** | — | **262.00 GPU-h** |

The underlying four-task training totals are 0.494813, 0.554418, and 0.406252 GPU-h. These are optimistic for confirmation: they are single-seed estimates from four development families, and do not include source runs, HR query evaluation, or any HR expert. Even the path-source arm alone needs about 89 GPU-h before source training and evaluation, exceeding the currently projected post-pilot balance.

At the active seed-0 source throughput, one 60-epoch source run projects to about 2.27 GPU-h. The frozen pilot's six source runs therefore project to 13.62 GPU-h; its 108 task adaptations project to about 7.51 GPU-h by applying measured costs for the three R3 path controls, path-source direct-256, ordinary full-FT direct-256, and 500-epoch scratch to their six-task × three-seed design. After the already terminal 11.706 GPU-h, the projected pilot leaves about 87.16 GPU-h of the 120-hour cap. This is an estimate, not a booked charge. Reusing the pilot's source checkpoints for seeds 0/1/42, confirmation still needs four additional source runs for seeds 2/3 and the path/final-only objectives (about 9.08 GPU-h). Thus three-method primary confirmation requires about 271 GPU-h of target/source training after the pilot, plus roughly 1.6 GPU-h for one comparable query-evaluation pass if timed `HRDEV` throughput carries over; experts and m-curves add more. This is about 273 GPU-h, approximately 186 over the projected 87.16 GPU-h remaining. The 48-hour confirmation reserve is not sufficient even for one full path-source arm.

The HR expert increment cannot be timed credibly from this ledger: there are no HR-expert training rows in `EXPERIMENTS.csv`; existing `HRDEV` rows are only short evaluations (about 0.00075 GPU-h per task/checkpoint), not HR training. E2-b's two expert types over 4–8 tasks and doubling rollout budgets therefore add unknown, positive training cost. They cannot fit on top of a primary matrix that already exceeds the cap.

For E2-c, a simple fixed-epoch throughput extrapolation assumes adaptation work grows roughly in proportion to the number of nested support rollouts. Across 48 tasks, 3 model seeds, and the three methods, the measured `m=4` point alone is about 52.4 GPU-h. The five-point curve `m={1,2,4,8,16}` is about 406 GPU-h including `m=4`; reusing `m=4` primary cells still leaves roughly 354 GPU-h for the other four points. On four preselected representative tasks, the same estimate is about 33.8 GPU-h total (about 29.5 incremental GPU-h after reusing `m=4`); on eight tasks, about 67.7 total. These are throughput projections, not measured curves, and do not include HR experts. The taskbook permits a clearly labeled screening curve on a smaller task set; it does not permit representing that curve as the 48-task primary table.

### Priority and budget recommendation

1. **P0 — finish the already frozen six-family pilot and select no more than the preregistered three arms.** The projected 21.1 GPU-h for its six source runs plus 108 adaptations is within the remaining exploration balance. Do not add broad hyperparameter sweeps before the candidate freeze.
2. **P0 — preserve every planned primary cell (48 tasks, five model seeds, three support draws) at `m=4`.** At measured throughput the minimum three-arm matrix is not feasible under 120 GPU-h. Do not quietly cut seeds, support draws, tasks, or a baseline. Before dispatching it, require a development-only, symmetric runtime optimization to bring the forecast under the real remaining balance; otherwise record the budget as the blocker and do not open query. With the pilot forecast, the primary target adaptations alone would need about a 3.0× speedup to fit the remaining 87 GPU-h, and about 3.4× once the four extra confirmation source runs are included. The task-adaptation budget after those source runs is about 78 GPU-h, or 0.036 GPU-h per cell averaged across the three methods, versus the measured 0.102–0.139.
3. **P0 — minimal HR expert and m-curve evidence only after the primary matrix is covered.** Use the taskbook's 4-task representative screen for expert saturation and the same preselected set for an exploratory three-method m curve; reuse already trained `m=4` cells. Continue doubling expert data until the specified saturation rule or the true budget limit, and report right-censoring if it is not reached. The projected 4-task m curve alone costs about 34 GPU-h before HR experts, leaving roughly 53 GPU-h of the post-pilot balance; it cannot be scheduled alongside the current full primary estimate without a demonstrated runtime reduction or an explicit budget increase.
4. **P1 — defer broad mechanism arms and expanded expert/curve coverage.** The pilot already compares six candidate/control arms on six development families and three model seeds; any smaller follow-up must be described as mechanism evidence on that subset, not as a substitute for the primary confirmation matrix.

### Compute savings that preserve the primary sample matrix

- Train each source objective once per model seed and reuse that exact checkpoint across all 48 tasks and all three support draws. The pilot's seeds 0/1/42 can be reused if their frozen source configuration is selected; never repeat source pretraining per task or support draw.
- Generate and cache each frozen support/validation rollout and the nested support indices once. Pair methods on the same task, model seed, and support draw. Reuse the core `m=4` checkpoints as curve points instead of retraining them.
- Use LR-validation-driven early stopping only after calibrating a fixed minimum epoch and patience on the full development histories, then apply the declared rule to every confirmation cell. Current best epochs show possible savings for ordinary full-FT (10–33 of 100) and some path-source tasks (28 of 100), but path-source tasks also peak at 82 and 99; a short blind patience could censor real minima. The 500-epoch scratch minima range from 290 to 498, so no one shortened schedule is supported by these runs. Save the best LR-validation checkpoint and the actual update count either way.
- Benchmark implementation-level savings on the pilot without changing the objective: mixed precision with numerical/metric parity checks, caching static lifts and repeated calibration inputs, and batching independent task models if it preserves separate parameters and optimizer states. If changing cadence or approximating the 256-grid commutator, treat it as a new development candidate and give the same general improvement to the strongest ordinary baseline before freezing; do not assume the speedup or its accuracy until measured.
- Keep full-resolution HR evaluation to the LR-selected checkpoint, as in the current protocol. Record GPU-hours per cell and per source job so the final ledger can use observed costs rather than these projections.

**Conclusion:** full 5×3×48 confirmation for the three preregistered primary methods, plus HR experts and m-curves, does not fit the 120 GPU-hour cap at measured costs. The shortfall is already decisive before the two additions; exact HR-expert cost remains unmeasured. The clean options are a validated several-fold reduction in per-cell training cost or a larger authorized compute budget. Reducing the confirmation matrix silently would invalidate the frozen evidence target.

## Updated pilot and confirmation budget from observed per-task throughput (2026-10-14)

This update supersedes the earlier coarse source-cost projection above. Read-only calculation from current `STATE.md`/`BUDGET.md`, the frozen six-arm pilot YAML, completed source-job charges, and development target-task JSONs. No GPU, solver/data generation, query access, or configuration change was made. Pilot target tasks have not run yet, so their cost remains a projection from matched development cells.

### Assumptions and pilot forecast

- Stage 2 is six arms × three model seeds × six development tasks: 18 target-adaptation cells per arm, 108 total. All target support/validation is m=4; source training is shared by objective and model seed, not repeated for each target cell.
- Measured proxies: path-source direct-256 target adaptation is 445.6 s/cell; the three path-target arms are 63–83 s/cell; ordinary full-FT + direct-256 is 499 s/cell; 500-epoch scratch is 365.6 s/cell. The last two and direct-256 are the costly pilot arms; the path-target range applies to firstpath/endpoint/direction adaptation, not the path-source direct-256 arm.
- Completed source pretrains from scratch cost 2.0681 GPU-h for seed0/path and 1.4025 GPU-h for seed0/final, both at 60 epochs. Assuming the same objective-specific throughput for the remaining source checkpoints, the four successful source runs still needed for the pilot (seed1/path retry, seed1/final, seed42/path, seed42/final) cost about 6.94 GPU-h. The failed seed1/path attempt's 0.1839 GPU-h is already in the current terminal ledger, `15.3605` GPU-h; it is not charged again here. This forecast assumes the retry uses a full run and that source-seed throughput is similar.

| Stage-2 target arms | Basis | Projected GPU-h |
|---|---|---:|
| Path source + direct-256 | 18 × 445.6 s | 2.23 |
| Three path-target arms together | 54 × 63–83 s | 0.95–1.25 |
| Ordinary full-FT + direct-256 | 18 × 499 s | 2.50 |
| Lifted scratch, 500 epochs | 18 × 365.6 s | 1.83 |
| **108 target adaptations** | | **7.50–7.80** |
| Four remaining source jobs | measured seed0 path/final throughput proxy | **6.94** |
| **Pilot work remaining from the current ledger** | | **14.44–14.74** |

After the pilot, expected campaign use is about 29.80–30.10 GPU-h and total remaining capacity about 89.90–90.20 GPU-h. In the original 72+48 split, this is approximately 41.9–42.2 exploration GPU-h plus the untouched 48 GPU-h confirmation reserve. These estimates exclude HR-dev evaluation overhead for the pilot and any additional retries.

### Frozen 5 × 3 × 48 target-adaptation costs

Each method has 5 model seeds × 3 support draws × 48 tasks = **720 target cells**. Current observed throughput gives:

| Pilot candidate | GPU seconds per target cell | 720-cell adaptation cost | Fits the post-pilot 89.9–90.2 GPU-h alone? |
|---|---:|---:|---|
| Path-source + direct-256 | 445.6 | 89.12 GPU-h | No, after its two new path-source seed pretrains |
| Each path-source + firstpath/endpoint/direction target arm | 63–83 | 12.60–16.60 GPU-h | Yes, alone |
| Ordinary full-FT + direct-256 | 499 | 99.80 GPU-h | No |
| Lifted scratch, 500 epochs | 365.6 | 73.12 GPU-h | Yes, alone |

“Alone” here means no other target method is being confirmed. For a complete method set, the two new confirmation model seeds `[2,3]` require two path-source and two final-only source pretrains; using the observed objective-specific rates, that shared source cost is about 6.94 GPU-h. The existing pilot checkpoints for seeds `[0,1,42]` are reusable only if their selected frozen source configurations are retained.

The smallest three-method confirmation that selects one of the cheaper path-target arms, the strongest measured ordinary baseline, and scratch costs **185.52–189.52 GPU-h** for target adaptation alone. Adding the four new source runs brings it to **192.46–196.46 GPU-h**, before HR evaluation, HR experts, or m-curves. Against the projected 89.90–90.20 GPU-h remaining, the shortfall is about **102–107 GPU-h** even with the cheap path-target candidate.

If the selected path-source method is the direct-256 candidate, the three-arm target subtotal is **262.04 GPU-h**; adding the shared source jobs raises it to about **268.98 GPU-h**, roughly **179 GPU-h beyond** post-pilot capacity. Confirming all six pilot candidates would cost about **299.84–311.84 GPU-h** in target adaptations and **306.78–318.78 GPU-h** including those four new source pretrains. These are training-only lower bounds; evaluations and experts are additional.

### Budget conclusion and hard conflict

At the current observed rates, only a single cheap path-target arm or the scratch arm can fit as a 720-cell matrix within the projected post-pilot balance. The ordinary full-FT + direct-256 matrix alone exceeds that balance; the path-source direct-256 matrix leaves effectively no room and also requires new source runs. Neither can be paired with the other required primary methods. Even the least expensive preregistered three-method set is more than twice the available post-pilot budget. Thus the full frozen 5×3×48 comparison cannot finish inside the present 72+48 cap after already charged work and the frozen pilot. It remains impossible before adding the HR experts and m-curves.

The arithmetic assumes: (1) one task adaptation uses the measured development protocol and same epoch budget; (2) per-cell cost transfers to all 48 frozen boundaries and five seeds; (3) one source checkpoint per objective/seed is reused across all tasks and supports; (4) confirmation source seeds 0/1/42 can reuse pilot checkpoints; and (5) the 63–83 s range belongs only to path-target adaptation arms. If any candidate's frozen training configuration changes, its cost must be remeasured before confirmation. Do not interpret these projections as measured pilot results or use them to reduce the frozen task/seed/support matrix silently.

## Stage-2 live queue and budget reconciliation (2026-10-14)

Read-only update from `docs/STATE.md`, the queue status/progress files, the append-only experiment ledger, frozen pilot/confirmation configs, and prior timed task outputs. No GPU was launched and no training code, frozen protocol, or query data was changed.

### Live state and corrected pilot forecast

- At 15:32 Asia/Shanghai, serialized supervisor PID `418560` and worker PID `418565` were active. The worker was resuming seed-1/path source training from the full-state epoch-39 checkpoint; the latest progress was epoch 44/60, 112,500 optimizer steps, best/latest source-HR validation `0.01044725`, and `629.7 s` in this recovery process. The queue had not yet started target adaptation or HR-development evaluation. Its frozen order remains recovery, three other source runs, 108 LR target cells, LR-matrix audit, then 108 HR-development evaluations.
- Terminal ledger use is `16.86137` GPU-h, including `0.18389` h for the first interrupted seed-1/path attempt and `1.50083` h charged conservatively for its retry. The active recovery adds `0.17491` h so far, outside the terminal ledger. Thus the estimated exploration balance is `54.96372` h (`72 - 16.86137 - 0.17491`); the separate 48 h confirmation reserve is still untouched. The recovery resume is incremental work after the interrupted-attempt charge.
- A current-rate projection for the remaining 16 recovery epochs is about `0.56` GPU-h, using the observed ~126 s/epoch over epochs 40–44. The three later source jobs (seed-1/final and seed-42/path/final) project to `4.87` h using the *measured* seed-0/path and seed-0/final run totals (`2.06809` and `1.40251` h). These are forecasts, not charges. Together with the active recovery remainder, the source work still ahead is about `5.43` h; do not add the already charged interrupted attempts a second time.
- The 108 frozen LR target cells remain a projection of earlier measured task costs: `7.50–7.80` GPU-h. The additional 108 HR-development checks project to about `0.08` h from prior timed HRDEV records. Accordingly, expected work from the current live point through pilot completion is about `13.01–13.31` h. Including the `16.86137` terminal and `0.17491` active hours, the projected campaign balance after the pilot is about `89.65–89.95` GPU-h total, of which roughly `41.65–41.95` h is exploration and 48 h remains the protected confirmation allocation. The earlier estimate of `6.94` h for four remaining source jobs is stale for this live point: it treated the recovering seed-1/path source as a new full run despite its already charged attempts and resumed checkpoint.

### Confirmation feasibility without changing its sample matrix

The confirmation protocol still requires 48 boundaries × 5 model seeds × 3 crossed support draws at `m=4` for each selected method. At the measured per-cell costs, the cheapest plausible path-target candidate plus ordinary full-FT/direct-256 and lifted scratch costs `185.5–189.5` GPU-h for target adaptation alone. New source pretraining for confirmation seeds 2 and 3 is estimated at another `6.94` h; HR query evaluation projects to roughly `1.6` h from timed development evaluators. This lower bound is already `194.1–198.0` h against about `89.8` h projected after the pilot, before expert training or m-curves. It requires about a `2.3×` reduction in target-adaptation GPU time to fit even this primary matrix and its source/evaluation overhead. If path-source direct-256 remains the selected path candidate, target adaptation rises to about `262.0` h and requires about `3.2×` target-cell acceleration. These are projections from development costs; no confirmation cell has been run.

The full completion list is harder than the primary matrix alone: the four-task, three-method five-point `m`-curve has a prior estimate of about `33.8` h including `m=4`, or about `29.5` incremental hours if those primary cells are reused. HR-expert training cost has no measured ledger entry. Thus an optimization that merely brings the primary matrix just under budget would not establish that all core items fit; expert cost remains unknown and positive.

### Measured optimization signal and next decision

- The existing ordinary full-FT/direct-256 development runs all trained 100 epochs (four tasks, `498.6–499.9 s` each), although their unique LR-validation minima were at epochs `10, 12, 18, 33`. If a development-calibrated rule used a 12-epoch patience after the minimum, the *retrospective* stopping points would be about `22–45` epochs, a roughly `3.3×` cost reduction on those four tasks. This is a plausible early-stopping opportunity, not a measured speedup: the rule has not been run, the task set is small, and confirmation must use an LR-only rule frozen before its query is opened.
- The path-source/direct-256 development history has minima at epochs `28, 28, 82, 99`, so the same illustrative patience would stop at about `40, 40, 94, 100` epochs (only ~`1.46×` average reduction). The scratch-500 minima were `207, 314, 371, 478`, implying about `1.41×` under that same retrospective rule. The cheap path-target first-path examples also have late minima at `96–98` on two of four tasks. Therefore the ordinary-baseline early-stop signal cannot be applied as a blanket `3.3×` campaign speedup.
- Continue the already-running frozen queue through the six-family/three-seed pilot; it is within the remaining exploration allocation, and it determines the selected path candidate and actual cell costs. After the pilot, use its complete LR histories for a development-only, symmetric stopping-rule calibration across all selected arms and the strongest baseline, then run a fresh development replay to measure actual cost and metric changes. Profile and benchmark implementation-preserving options such as BF16/AMP with numerical and metric parity, cached static inputs/lifts, and fused or compiled repeated operators. Record real GPU-hours and preserve LR-validation selection. Any changed training schedule/objective must be evaluated as a fresh development candidate and offered symmetrically to the strongest ordinary baseline before freezing exact confirmation configs.
- Do not start the frozen confirmation matrix at current throughput: the smallest three-method matrix exceeds remaining capacity by roughly `104–108` h before secondary evidence. Do not drop tasks, seeds, supports, HR experts, or curve points to make the arithmetic look feasible. Keep the query gate closed until a validated optimization forecast covers the complete frozen checkpoint matrix. If implementation-level benchmarks cannot establish the required reduction within the budget, preserve the evidence and report a budget-limited partial campaign; do not label a smaller matrix as confirmation.

## Reforecast after model-seed amendment (2026-10-15)

Amendment 2 to `docs/prereg/CONFIRMATORY.md` changes the confirmatory model seeds to the already-used `[0,1,42]`, per the user's instruction. It retains all 48 unseen tasks, all three crossed support draws, `m=4`, the three selected methods, and the query roles. This section supersedes prior five-seed feasibility estimates for current planning; those estimates remain above as the original-protocol record.

- The direct-256 primary set's measured target-adaptation subtotal was about `262.04 GPU-h` at five model seeds. Linear scaling to three model seeds gives about `157.22 GPU-h`. HR-query inference was estimated at `1.6 GPU-h` for five seeds; scaling by 3/5 gives about `0.96 GPU-h`. The extra source pretraining estimated for seeds 2/3 (`6.94 GPU-h`) is no longer needed because selected seeds 0/1/42 already have source checkpoints. Thus the amended primary matrix is projected at about `158.2 GPU-h`, not including confirmation-scale CPU data generation or final audits.
- The 4-task, three-method nested `m`-curve remains about `29.5 GPU-h` incremental after reusing its `m=4` primary cells, giving a known subtotal near `187.7 GPU-h`. This does not include HR-expert training, dedicated B2/M1 path-FT confirmation beyond the three selected methods, or mechanism-control extensions.
- Against the current `74.01 GPU-h` total remaining, the revised primary shortfall is about `84.2 GPU-h`; primary plus the representative `m` curve is short by about `113.7 GPU-h`. At two independently saturated GPUs, the known subtotal is about `93.9 wall-hours`, versus an ideal 37.0 wall-hours available from the remaining GPU budget. These are proportional projections, not measured confirmation runs. The query gate remains closed.

## Live budget review (2026-10-16 03:39 Asia/Shanghai)

The terminal ledger was `56.750629 GPU-h` at 03:35, with R4 A2/seed42 task00 and HR-scratch m=16 task01 active. Two stale `RUNNING` rows had no process/final report: R1 source continuation epoch50/60 (`8961.294399 s`) and R3 first-path task03 epoch72/100 (`41.025256 s`). The main thread appended `INTERRUPTED` terminal rows charging these observed progress-time lower bounds, `2.500644 GPU-h` combined; current official total is `59.251273 GPU-h`, exploration remainder `12.748727 GPU-h`, reserve48 intact. R4 has six cells remaining (~1.2281 GPU-h) plus 36 HR-dev reports (~0.0266 GPU-h); m=16 task01 active at epoch4/500 and task03 queued, with ~5.11–5.30 GPU-h expected. Frozen queued work leaves an estimated `6.19–6.38 GPU-h` exploration balance before unquantified early HR-dev evaluation overhead.

Recommendation remains conditional on R4's frozen rule. Complete R4 and m=16 first. If A1 wins against both A0 and A2 by the card's ≥1% / all-three-seeds / ≥4-of-6-families criterion, the highest-value next development check is one new support draw with A1 frozen across 3 model seeds × 6 tasks (~3.68 GPU-h training plus ~0.013 GPU-h HR-dev, plus CPU support generation), leaving roughly 2 GPU-h buffer. If A1 fails the rule, do not spend more on a path-specific arm; reconcile reports and budget. That support-draw repeat would still be development evidence and would not substitute for the frozen confirmation matrix or three-support core.

## HR full-adaptation expert budget follow-up (2026-10-16)

Read-only review after the R4 paired HR-development evaluation completed; no experiment, test, or query data was run or read. R4 does not justify additional path-specific tuning: its frozen HR-dev means are A0 `0.05347209`, A1 first-path `0.05387774`, and A2 endpoint `0.05363236`. A1 is 0.7586% worse than A0 and 0.4575% worse than A2, and fails the preregistered repeatability/practical-effect rule. See `artifacts/reports/PHASE1_R4_PATHCOMM_20261016.md`.

### Budget available after the frozen m=16 queue

- The official terminal ledger after the 36 R4 HR-dev reports is `60.497069 GPU-h`; exploration balance is `11.502931 GPU-h`, while the 48 GPU-h confirmation/control reserve remains protected. The 36 HR-dev evaluations cost `0.028250 GPU-h` in total.
- The m=16 random-init task01 progress is at epoch 284/500 with `4221.715 s` elapsed. Linear epoch scaling projects its complete 500-epoch fit at about `2.065 GPU-h`. Task03 has no m=16 timing yet; its same-task m=4 cost was `0.683930 GPU-h`, or `2.736 GPU-h` when scaled by the fourfold m=16 transitions. The completed m=16 fits cost `2.057014` (task00) and `2.545329 GPU-h` (task05); using the latter for task03 instead gives a lower `2.545 GPU-h` projection. Thus task01+task03 are expected to book approximately `4.61–4.80 GPU-h`, leaving about `6.70–6.89 GPU-h` of nominal exploration headroom. These are runtime projections; the old untimed HR-dev overhead in `BUDGET.md` remains an unquantified downward risk.
- The seed-0 multi-boundary path-source checkpoint already exists at `artifacts/checkpoints/phase1/R1/source_path_s0_scratch.pt`; its measured source-training cost (`2.068094 GPU-h`) is already in the ledger. All 64 m=16 target HR support rollouts are also already generated (CPU wall `4786.624 s`), so the proposed expert needs no new HR support solves. The m=16 validation manifest is the same development-only HR-dev set used by the scratch controls; no query role is needed.

### Recommendation: a 250-epoch expert point is worthwhile and nominally fits

The supplied 450-epoch four-task estimate of `7.5–8.3 GPU-h` exceeds the projected post-m=16 exploration headroom by about `0.61–1.60 GPU-h`; do not schedule 450 epochs. Scaling the same full-network workload linearly to 250 epochs gives `4.17–4.61 GPU-h`. It would leave about `2.09–2.72 GPU-h` of nominal exploration capacity after task01/task03 finish, before any unmeasured historical HR-dev overhead. This makes a fixed 250-epoch run the best affordable next expert comparison, provided the ledger is refreshed after the scratch queue terminates and the measured remaining headroom still covers the upper cost estimate plus a useful margin.

This four-task, seed-0, support-seed-271829, m=16 source-initialized HR full-FT endpoint would be scientifically useful: it directly tests whether multi-boundary HR initialization can exploit the already available target HR support better than random-init HR scratch on the same four development tasks. The current scratch m=16 best HR-dev errors remain above the matched path-source values on the two completed tasks: task00 `0.15876050` vs `0.06831793`, task05 `0.10269669` vs `0.05135978`; task01's best so far at epoch284 is `0.10330813`. The expert could therefore give a meaningful HR-transfer and cost comparison even if it fails to reach the path-source result.

Label the result as a **250-epoch budget-limited HR expert development point**, select the best checkpoint only on the pinned HR-dev validation, and report task-level results, GPU cost, and the trajectory's endpoint. It is not an experienced-saturation curve, an oracle, or an equivalence result: it uses one model seed, one support draw, four development families, and the mission's saturation criterion requires additional HR-data doubling and evidence that longer optimization no longer helps. If validation is still improving at epoch250, report that expert as right-censored rather than reducing the source model or weakening the comparison.

Implementation note: `scripts/run_phase1_hr_scratch_m16_dev.py` currently hard-requires random initialization and 500 epochs, so the source-initialized 250-epoch expert needs its own immutable config/run namespace and a narrow runner update before dispatch. The estimate assumes the same four tasks `[0,1,3,5]`, full-FNO updates, m=16 support, and 250 epochs. The R4 outcome argues against spending the remaining budget on more A1/A2/path-supervision searches; this expert instead addresses the mission's still-open HR full-adaptation comparison.

## Static audit: source-initialized HR full-FT m=16 runner (2026-10-16)

Read-only audit of the frozen config, card, runner, active scratch runner/protocol, and pinned artifacts; no training, tests, or query data were accessed.

- **Clean-run protocol/provenance: PASS.** The YAML and card agree on tasks `[0,1,3,5]`, seed0/support seed271829, nested m=16 HR support, 250 full-adapter epochs, final-only loss, and HR-dev-only checkpoint selection. The runner pins the source checkpoint (on-disk SHA-256 matches), task and validation manifests, support-generation report, and per-task support index hashes. All four canonical support-index hashes match the runner constants; the matched path-source HR-dev reference reports and validation indices exist. `train()` accepts the source initialization and full-adapter settings. The result/checkpoint namespace and ledger IDs/round (`HRFULLFTM16`) are distinct from the random-init scratch namespace (`HRSCRATCHM16`).
- **Dispatch gate remains active.** At audit time the m=16 scratch runner was still running on GPU1; the new expert is configured for GPU0. The card/STATE require waiting for the scratch queue to become terminal, refreshing the official ledger, and confirming at least 5.6 GPU-h exploration headroom before launch. No static first-launch path or hash mismatch was found.
- **Concrete retry blocker.** The runner appends a `RUNNING` ledger row and then, outside its protected `try` block, aborts if the fixed per-task progress JSON already exists. The training engine writes that progress file every epoch and leaves it in place. Thus an interrupted/failed task cannot be retried in the same namespace: the stale progress file triggers `FileExistsError` before a terminal failure row is written. A retry needs an explicit recovery/attempt-path procedure or a code fix before it is attempted; this does not block a clean first launch when the fresh progress paths are absent.

### Retry-hardening addendum (2026-10-16)

Read-only follow-up of the runner's `--attempt-suffix` changes; no tests, GPU jobs, or query data were accessed. Attempt-specific checkpoint, progress, successful task-report, and ledger trial-ID paths are collision-checked before appending `RUNNING`. Thus a failed/interrupted attempt can be retried with a new suffix while preserving its stale progress/checkpoint artifacts. The shared `task_results.json` remains coherent for resume: it is keyed by task and the immutable protocol descriptor, so previously completed tasks with matching descriptor and `query_read=false` are reused, while an unfinished task is trained under the new suffix; the summary aggregates those task reports and each report records its suffix.

One residual artifact-path issue remains: the `FAILED` ledger row and failure text still use the fixed `taskXX.failure.txt` path, without the attempt suffix. Repeated failures for the same task can overwrite the prior failure detail even though their ledger trial IDs differ. Make the failure artifact path attempt-specific before relying on repeated-failure history. Re-running the same suffix is intentionally rejected when its immutable attempt artifacts exist; increment the suffix for a retry.

Runner artifact-path follow-up: the residual failure-detail path is now also attempt-specific (`taskXX_attempt-<suffix>.failure.txt`) and the `FAILED` ledger row points to that same unique file. This preserves repeated failure diagnostics across suffixes; the static retry/resume path is now internally consistent. No experiment or test was run for this source-only correction.

## R5 ordinary full-FT + direct256 low-LR bracket decision (2026-10-16)

Read-only strategy review of the mission, current STATE/budget, R5 high-LR card/config, low-side card/config, and existing LR=`1e-4` results. No GPU, training, test, or query work was run/accessed.

- **Recommendation: run the frozen six-family LR=`3e-5` screen only after R5's LR fits and all six HR-dev reports are terminal.** The mission calls for a strong, independently tuned direct baseline, and its suggested LR neighborhood around a current winner includes the `0.3×` low side. Existing ordinary full-FT + direct256 at `1e-4/cosine-100` is the six-family, three-model-seed baseline (HR-dev mean `0.054881171919`); the R5 `3e-4` arm tests only the high side. Prior `3e-5` ordinary full-FT evidence is final-only, four-family, and seed42, so it does not answer whether the direct256 objective shifts the useful LR bracket.
- **Budget gate passes conditionally at the stated ~2.3 GPU-h post-R5 headroom.** The frozen low-side card estimates `0.94 GPU-h` for training plus HR-dev and requires at least `1.8 GPU-h` fresh exploration headroom, retaining `0.9 GPU-h` for uncertainty and preserving the 48 GPU-h reserve. A 2.3 estimate leaves about `1.36 GPU-h` after the expected charge, including the required buffer. Recompute the official ledger once the high-side and HR-dev queue is terminal; dispatch only if the measured exploration remainder is still ≥1.8 GPU-h and there is no concurrent GPU work. Otherwise stop the bracket at `{1e-4,3e-4}`.
- **Interpretation stays local.** This adds a preselected seed0/support271829 one-seed screen on tasks `[0,1,2,3,4,5]`, m=4, same frozen manifests, 100-epoch cosine and full FT, changing only LR to `3e-5`. Select checkpoints on LR validation, then report six per-task and equal-family LR/HR-dev metrics. It can complete a useful local `{3e-5,1e-4,3e-4}` bracket but cannot replace or override the three-seed `1e-4` baseline on one-seed HR-dev evidence; if `3e-5` appears better, validate that setting across the existing model seeds before promoting it. Query/test remains closed.
