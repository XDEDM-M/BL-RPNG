#!/usr/bin/env python3
"""Summarize the single frozen NS-PIMPLE method-V1 query pass (JSON only)."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts/reports/ns_pimple_method_v1"
DATA_INDEX = ROOT / "artifacts/data/ns_pimple_method_v1/dataset_index.json"
ARMS = ("selected_pimple_path", "final_only", "matched_endpoint_control", "lifted_lr_scratch")
ARM_LABELS = {
    "selected_pimple_path": "Selected PIMPLE path",
    "final_only": "Final only",
    "matched_endpoint_control": "Matched endpoint control",
    "lifted_lr_scratch": "LR scratch",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def mean(rows: list[dict], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def f(value: float) -> str:
    return f"{value:.4f}"


def main() -> int:
    result = read_json(REPORTS / "frozen_query_result.json")
    access = read_json(REPORTS / "query_access_log.json")
    freeze = read_json(REPORTS / "global_checkpoint_freeze.json")
    source = read_json(REPORTS / "source_training.json")
    target = read_json(REPORTS / "target_training.json")
    index = read_json(DATA_INDEX)
    if access.get("status") != "complete" or access.get("sealed_test_opened") is not True:
        raise ValueError("The frozen query pass is not complete.")
    if result.get("freeze_sha256") != access.get("freeze_sha256"):
        raise ValueError("Query result and access log refer to different freezes.")

    csv_path = REPORTS / "query_metrics.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "split", "task", "arm", "pair_id", "rollout_rel_l2_mean",
            "rollout_rel_l2_final", "teacher_forced_rel_l2_mean",
            "teacher_forced_rel_l2_final", "prediction_to_reference_norm_ratio_mean",
            "cell_center_divergence_rms_pred_mean", "cell_center_divergence_rms_reference_mean",
        ))
        writer.writeheader()
        for task, task_result in result["tasks"].items():
            split = "development" if task.startswith("F5_") else "confirmation"
            for arm, arm_result in task_result["arms"].items():
                for row in arm_result["per_query"]:
                    writer.writerow({
                        "split": split, "task": task, "arm": arm,
                        "pair_id": row["pair_id"],
                        "rollout_rel_l2_mean": row["rollout_rel_l2_mean"],
                        "rollout_rel_l2_final": row["rollout_rel_l2_final"],
                        "teacher_forced_rel_l2_mean": row["teacher_forced_rel_l2_mean"],
                        "teacher_forced_rel_l2_final": row["teacher_forced_rel_l2_final"],
                        "prediction_to_reference_norm_ratio_mean": row["prediction_to_reference_norm_ratio_mean"],
                        "cell_center_divergence_rms_pred_mean": row["cell_center_divergence_rms_pred_mean"],
                        "cell_center_divergence_rms_reference_mean": row["cell_center_divergence_rms_reference_mean"],
                    })

    split_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for split, prefix in (("development", "F5_"), ("confirmation", "F6_")):
        tasks = [(name, row) for name, row in result["tasks"].items() if name.startswith(prefix)]
        split_metrics[split] = {}
        for arm in ARMS:
            arms = [row["arms"][arm] for _, row in tasks]
            queries = [query for row in arms for query in row["per_query"]]
            split_metrics[split][arm] = {
                "rollout": statistics.mean(row["mean_relative_l2"] for row in arms),
                "final": statistics.mean(row["final_step_relative_l2_mean"] for row in arms),
                "teacher": mean(queries, "teacher_forced_rel_l2_mean"),
                "norm_ratio": mean(queries, "prediction_to_reference_norm_ratio_mean"),
                "div_pred": mean(queries, "cell_center_divergence_rms_pred_mean"),
                "div_ref": mean(queries, "cell_center_divergence_rms_reference_mean"),
            }

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
        x = list(range(len(ARMS)))
        width = 0.36
        for offset, (split, color) in enumerate((("development", "#276FBF"), ("confirmation", "#D1495B"))):
            values = [split_metrics[split][arm]["rollout"] for arm in ARMS]
            axes[0].bar([index + (offset - 0.5) * width for index in x], values,
                        width=width, label=split.title(), color=color)
        axes[0].set_xticks(x, [ARM_LABELS[arm] for arm in ARMS], rotation=18, ha="right")
        axes[0].set_ylabel("Mean 20-step relative L2")
        axes[0].set_title("Frozen HR query rollout error")
        axes[0].legend(frameon=False)
        axes[0].grid(axis="y", alpha=0.25)

        diagnostic_names = ("teacher", "rollout", "final")
        diagnostic_labels = ("Teacher-forced", "Autoregressive mean", "Final step")
        x2 = [0, 1]
        width2 = 0.23
        for offset, (key, label, color) in enumerate(zip(
            diagnostic_names, diagnostic_labels, ("#4C956C", "#276FBF", "#D1495B")
        )):
            values = [split_metrics[split]["selected_pimple_path"][key] for split in ("development", "confirmation")]
            axes[1].bar([position + (offset - 1) * width2 for position in x2], values,
                        width=width2, label=label, color=color)
        axes[1].set_xticks(x2, ("F5 development", "F6 confirmation"))
        axes[1].set_ylabel("Relative L2")
        axes[1].set_title("Where selected-path error grows")
        axes[1].legend(frameon=False)
        axes[1].grid(axis="y", alpha=0.25)
        fig.suptitle("NS-PIMPLE Method V1 — one frozen query pass")
        fig.savefig(REPORTS / "ns_method_v1_rollout_diagnostics.png", dpi=180)
        plt.close(fig)
        plot_note = "Figure: `ns_method_v1_rollout_diagnostics.png`."
    except ImportError:
        plot_note = "Plot omitted because matplotlib is unavailable."

    source_lines = []
    for arm in ARMS[:3]:
        row = source["arms"][arm]
        source_lines.append(
            f"| {ARM_LABELS[arm]} | {row['training_pair_count']} | {row['training_transition_count']} "
            f"| {row['epochs_run']} | {row['optimizer_updates']} | {f(row['best_source_validation_rollout_rel_l2'])} "
            f"| {row['best_epoch']} |"
        )

    target_lines = []
    for task in sorted(target["tasks"]):
        values = target["tasks"][task]
        target_lines.append(
            f"| {task} | {f(values['selected_pimple_path']['best_lr_validation_rollout_rel_l2'])} "
            f"| {f(values['final_only']['best_lr_validation_rollout_rel_l2'])} "
            f"| {f(values['matched_endpoint_control']['best_lr_validation_rollout_rel_l2'])} "
            f"| {f(values['lifted_lr_scratch']['best_lr_validation_rollout_rel_l2'])} |"
        )

    query_lines = []
    for split in ("development", "confirmation"):
        table = result[f"{split}_mean"]
        query_lines.append(
            f"| {split.title()} | {f(table['selected_pimple_path'])} | {f(table['final_only'])} "
            f"| {f(table['matched_endpoint_control'])} | {f(table['lifted_lr_scratch'])} |"
        )

    task_lines = []
    for task in sorted(result["tasks"]):
        arms = result["tasks"][task]["arms"]
        task_lines.append(
            f"| {task} | {f(arms['selected_pimple_path']['mean_relative_l2'])} "
            f"| {f(arms['final_only']['mean_relative_l2'])} | "
            f"{f(arms['matched_endpoint_control']['mean_relative_l2'])} | "
            f"{f(arms['lifted_lr_scratch']['mean_relative_l2'])} |"
        )

    development = split_metrics["development"]["selected_pimple_path"]
    confirmation = split_metrics["confirmation"]["selected_pimple_path"]
    md = f"""# NS-PIMPLE Method V1 结果

## 判定

筛选后的 PIMPLE 路径数据通过路径门控，完整 source/target 数据角色、网格桥接和训练流程也通过审计。但模型结果没有通过预注册的有效性判据：selected-PIMPLE-path 只在 F5 development 上取得很小优势，在 F6 confirmation 上同时落后于 final-only 和 matched-endpoint control。因此，这轮结果**不支持**“所选 solver-path 信息改善 NS source-to-target 迁移”的结论。该结论限于此 2D cavity、两种目标边界族、4 个任务和 seed 42。

## 数据与训练审计

- 首个完整非端点 PIMPLE 状态作为每个物理转移的唯一路径标签；其余迭代节点未作为训练标签。
- 源训练集：28 条 HR 轨迹、560 个转移，其中 7 条 V0 `ic_a` 和 21 条独立新增 IC；source validation 为 14 条 IC 不重叠的 HR query。
- 每个目标任务：4 条 LR support（80 个转移）、4 条独立 LR validation（80 个转移）、4 条 HR query；4 个任务共 16 条 HR query。
- source 新增轨迹门控：21/21 通过；V0 manifest 未改；源训练最小首钩 D/S 中位数为 {f(index['bridge_gate']['minimum_source_first_hook_D_median'])}/{f(index['bridge_gate']['minimum_source_first_hook_S_median'])}。有限体积边界 lift 最大残差为 {index['bridge_gate']['maximum_source_lift_residual']:.3e}，原生 cell-centre 网格未重采样。
- 全部 3 个 source arm 各训练 30 轮、2,100 次更新；target HR query 在全局冻结前未读取。全局冻结校验通过，之后只执行了一次 HR query pass。

### Source 预训练

| 臂 | HR 轨迹 | 转移 | 轮数 | 更新数 | Source validation rollout L2 | 最佳轮 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(source_lines)}

### Target LR validation 最优 20-step rollout L2

| Task | Selected path | Final only | Matched endpoint | LR scratch |
|---|---:|---:|---:|---:|
{chr(10).join(target_lines)}

四个任务的 transfer arm 最佳轮大多落在训练早期；训练损失下降但 LR rollout validation 常常变差，显示明显泛化/递推误差问题。Checkpoint 始终由各自 LR validation 选择。

## 冻结 HR query 结果

主指标为每条 query 的 20 个物理步相对 L2 均值，再先对每个任务的 4 条 IC 求均值、再对两个任务求均值。

| Split | Selected path | Final only | Matched endpoint | LR scratch |
|---|---:|---:|---:|---:|
{chr(10).join(query_lines)}

| Task | Selected path | Final only | Matched endpoint | LR scratch |
|---|---:|---:|---:|---:|
{chr(10).join(task_lines)}

- F5 development：selected path {f(result['development_mean']['selected_pimple_path'])}，比 final-only 低 {100*result['development_relative_reduction_vs']['final_only']:.2f}%，比 matched endpoint 低 {100*result['development_relative_reduction_vs']['matched_endpoint_control']:.2f}%。
- F6 confirmation：selected path {f(result['confirmation_mean']['selected_pimple_path'])}，比 final-only **高** {100*(-result['confirmation_relative_reduction_vs']['final_only']):.2f}%，比 matched endpoint **高** {100*(-result['confirmation_relative_reduction_vs']['matched_endpoint_control']):.2f}%。

## 高误差诊断

Selected-path 在 F5 上的 teacher-forced 单步误差均值为 {f(development['teacher'])}，20-step autoregressive 均值为 {f(development['rollout'])}，最后一步为 {f(development['final'])}；F6 分别为 {f(confirmation['teacher'])}、{f(confirmation['rollout'])}、{f(confirmation['final'])}。这表明单步空间映射已有较大误差，递推又进一步放大，F6 后期尤其明显。

预测/参考速度范数比均值在 F5/F6 为 {f(development['norm_ratio'])}/{f(confirmation['norm_ratio'])}，显示整体速度幅值偏低。cell-centred finite-difference divergence RMS 诊断中，selected path 预测/参考均值为 F5 {f(development['div_pred'])}/{f(development['div_ref'])}、F6 {f(confirmation['div_pred'])}/{f(confirmation['div_ref'])}；这是定位用的诊断，并非有限体积连续性误差。

逐 query 结果显示每个任务中新增的 `query_q1/q2` 通常比 V0 `ic_b/ic_c` 难。配置上，target validation 的初值频率是 `(1,4),(2,3),(3,2),(4,1)`、幅值 0.04，而新增 HR query 是 `(2,4),(4,2)`、幅值 0.035。这种频谱差异可能造成验证集与查询集难度不匹配，是需要在下一轮数据设计中修正的风险；它不能解释或抵消本轮确认集上的负结果。

## 可复核文件

- Frozen result: `artifacts/reports/ns_pimple_method_v1/frozen_query_result.json`
- Query access log: `artifacts/reports/ns_pimple_method_v1/query_access_log.json`
- Global checkpoint freeze: `artifacts/reports/ns_pimple_method_v1/global_checkpoint_freeze.json` (SHA256 `{access['freeze_sha256']}`)
- Source training history: `artifacts/reports/ns_pimple_method_v1/source_training.json`
- Target training history: `artifacts/reports/ns_pimple_method_v1/target_training.json`
- Per-query CSV: `artifacts/reports/ns_pimple_method_v1/query_metrics.csv`
- {plot_note}

## 下一轮应优先检查

1. 让 LR validation IC 与 sealed query 的初值频谱/幅值分布一致，同时保持 IC 完全不重叠。
2. 对 20-step 推理采用 NS 适配的多步训练窗口；窗口必须逐步更新 physical history、boundary lifts 和 `dt/nu` 条件，不能直接复用静态条件的 Allen–Cahn rollout wrapper。
3. 再核查离散不可压缩约束，并在新的独立 HR query 上重做冻结确认。当前 query 已使用，不应再作为新模型的 sealed test。
"""
    (REPORTS / "NS_METHOD_V1_RESULTS.md").write_text(md)
    print(json.dumps({
        "report": str(REPORTS / "NS_METHOD_V1_RESULTS.md"),
        "csv": str(csv_path),
        "plot": str(REPORTS / "ns_method_v1_rollout_diagnostics.png"),
        "status": "complete",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
