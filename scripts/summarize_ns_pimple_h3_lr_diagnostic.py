#!/usr/bin/env python3
"""Write an LR-only summary for the exploratory NS H3 follow-up."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts/reports/ns_pimple_method_v1"
ARMS = ("selected_pimple_path", "final_only", "matched_endpoint_control", "lifted_lr_scratch")
LABELS = {
    "selected_pimple_path": "Selected PIMPLE path",
    "final_only": "Final only",
    "matched_endpoint_control": "Matched endpoint control",
    "lifted_lr_scratch": "LR scratch",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    h3 = read(REPORTS / "h3_lr_diagnostic.json")
    h1 = read(REPORTS / "target_training.json")
    if h3.get("target_hr_query_used") is not False:
        raise ValueError("H3 diagnostic unexpectedly used target HR query data.")

    h1_means, h3_means = {}, h3["split_mean_lr_validation_rollout_rel_l2"]
    task_rows = []
    for split, prefix in (("development", "F5_"), ("confirmation", "F6_")):
        names = [name for name in h3["tasks"] if name.startswith(prefix)]
        h1_means[split] = {
            arm: sum(h1["tasks"][name][arm]["best_lr_validation_rollout_rel_l2"] for name in names) / len(names)
            for arm in ARMS
        }
        for name in sorted(names):
            fields = [f"| {name} | {split.title()} |"]
            for arm in ARMS:
                a = h1["tasks"][name][arm]["best_lr_validation_rollout_rel_l2"]
                b = h3["tasks"][name][arm]["best_lr_validation_rollout_rel_l2"]
                fields.append(f" {a:.4f} | {b:.4f} |")
            task_rows.append("".join(fields))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)
    colors = ("#276FBF", "#D1495B")
    x = list(range(len(ARMS)))
    width = 0.36
    for axis, split in zip(axes, ("development", "confirmation")):
        for offset, (horizon, values, color) in enumerate((
            ("H1", h1_means[split], colors[0]),
            ("H3", h3_means[split], colors[1]),
        )):
            axis.bar([i + (offset - 0.5) * width for i in x], [values[arm] for arm in ARMS],
                     width=width, label=horizon, color=color)
        axis.set_xticks(x, [LABELS[arm] for arm in ARMS], rotation=18, ha="right")
        axis.set_ylabel("Best 20-step LR validation relative L2")
        axis.set_title("F5 development" if split == "development" else "F6 confirmation")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("NS target adaptation: H1 versus NS-aware H3 supervision")
    fig.savefig(REPORTS / "h3_lr_validation_comparison.png", dpi=180)
    plt.close(fig)

    rows = []
    for split in ("development", "confirmation"):
        for arm in ARMS:
            rows.append(
                f"| {split.title()} | {LABELS[arm]} | {h1_means[split][arm]:.4f} "
                f"| {h3_means[split][arm]:.4f} | {h3_means[split][arm]-h1_means[split][arm]:+.4f} |"
            )
    md = f"""# NS-PIMPLE H3 LR-only 诊断

这是在 V1 sealed HR query 已完成一次评估后的 exploratory follow-up。它只使用预先冻结的 LR support 和 LR validation；`target_hr_query_used=false`。报告中的 H3 checkpoint 不经过 HR query 评价，不能替代 V1 冻结 HR 结果，也不能作为论文确认集结果。

H3 窗口使用 3 个连续物理转移，并在每一步递归更新预测当前速度、previous-state history、当前/下一时刻 boundary lift、粘度和 `dt`。所有 4 个 target arm 均运行 100 轮，best checkpoint 只按 20-step LR autoregressive validation L2 选择。预注册和代码哈希见 `docs/prereg/ns_pimple_method_v1_h3_lr_diagnostic.json` 与 `h3_lr_diagnostic.json`。

## Split 平均

| Split | Arm | H1 | H3 | H3 − H1 |
|---|---|---:|---:|---:|
{chr(10).join(rows)}

## Task 级比较

| Task | Split | Selected path H1 | H3 | Final-only H1 | H3 | Matched H1 | H3 | Scratch H1 | H3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(task_rows)}

H3 对部分 final-only / matched-control 任务带来小幅 LR rollout 改善，但 selected path 在 F5/F6 的 split mean 仍都高于 final-only 和 matched control。F6 path 从 H1 0.931 降到 H3 0.895，仍高于 final-only 0.834 和 matched control 0.782。H3 没有扭转路径预训练的相对排序，因此当前高误差不能只归因于 H1 target loss。

下一轮若继续验证，应先改进 source path 表征/边界和不可压缩约束，并让 validation 与新 query 的 IC 频谱一致；随后用全新的 HR query 做一次性冻结评估。V1 HR query 已打开，不应再用于 H3 sealed evaluation。

![H1 与 H3 的 LR validation 对比](/gemini/code/BL-RPNG/artifacts/reports/ns_pimple_method_v1/h3_lr_validation_comparison.png)
"""
    (REPORTS / "H3_LR_DIAGNOSTIC.md").write_text(md)
    print(json.dumps({
        "report": str(REPORTS / "H3_LR_DIAGNOSTIC.md"),
        "plot": str(REPORTS / "h3_lr_validation_comparison.png"),
        "target_hr_query_used": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
