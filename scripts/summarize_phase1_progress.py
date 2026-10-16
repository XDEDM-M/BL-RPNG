"""Recompute exploratory trial and HR-development tables without query access."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blrpng.paths import PROJECT_ROOT, sha256_file, sha256_json, write_json


def summarize() -> dict:
    ledger_path = PROJECT_ROOT / "docs/EXPERIMENTS.csv"
    ledger_rows = list(csv.DictReader(ledger_path.open(newline="", encoding="utf-8")))
    excluded_trials = set()
    for ledger_row in ledger_rows:
        if ledger_row.get("status") not in {"EXCLUDED_RESOURCE_CONTENTION", "EXCLUDED_INVALID"}:
            continue
        trial_id = str(ledger_row.get("trial_id", ""))
        round_name, separator, remainder = trial_id.partition("-")
        if separator and round_name.startswith("R"):
            excluded_trials.add(remainder.split("-task", 1)[0])

    training = []
    for path in sorted((PROJECT_ROOT / "results/phase1").glob("R*/*/summary.json")):
        summary = json.loads(path.read_text())
        if summary.get("trial_id") in excluded_trials:
            continue
        descriptor = summary["descriptor"]
        if descriptor.get("smoke_only"):
            continue
        if summary.get("query_opened_or_loaded") is not False:
            raise ValueError(f"Query discipline missing: {path}")
        config_sha = sha256_json(descriptor)
        if config_sha != summary["config_sha256"]:
            raise ValueError(f"Descriptor seal mismatch: {path}")
        values = []
        charges = []
        for task in descriptor["task_ids"]:
            task_path = path.parent / f"task{int(task):02d}.json"
            result = json.loads(task_path.read_text())
            if result["config_sha256"] != config_sha or result.get("query_metric") is not None:
                raise ValueError(f"Task descriptor/query mismatch: {task_path}")
            values.append(float(result["best_validation_metric"]))
            charges.append(float(result["gpu_hours"]))
        mean = statistics.mean(values)
        if not math.isclose(mean, summary["mean_task_lr_validation_rollout_rel_l2"], abs_tol=1e-12):
            raise ValueError(f"LR mean mismatch: {path}")
        training.append({
            "round": descriptor["round"], "trial": summary["trial_id"],
            "model_seed": descriptor["model_seed"], "support_seed": descriptor["support_seed"],
            "m": descriptor["support_budget"], "task_ids": descriptor["task_ids"],
            "lr_mean": mean, "lr_worst": max(values), "gpu_hours": sum(charges),
            "summary": str(path.relative_to(PROJECT_ROOT)), "summary_sha256": sha256_file(path),
        })
    development = []
    for row in training:
        cells = []
        for task in row["task_ids"]:
            budget_tag = "" if int(row["m"]) == 4 else f"_m{row['m']}"
            report_path = PROJECT_ROOT / "artifacts/reports/phase1_hrdev_evaluations" / (
                f"{row['trial']}{budget_tag}_task{int(task):02d}_s{row['model_seed']}_ss{row['support_seed']}.json"
            )
            if not report_path.exists():
                continue
            report = json.loads(report_path.read_text())
            if int(report["support_budget"]) != int(row["m"]) or report["model_trial_id"] != row["trial"]:
                raise ValueError(f"HR support-budget/trial pairing mismatch: {report_path}")
            if report.get("query_read") is not False or report.get("training_or_within_task_checkpoint_selection_used_hr") is not False:
                raise ValueError(f"HR selection discipline missing: {report_path}")
            for key, hash_key in (("checkpoint", "checkpoint_sha256"), ("task_result", "task_result_sha256")):
                if sha256_file(PROJECT_ROOT / report[key]) != report[hash_key]:
                    raise ValueError(f"HR provenance mismatch: {report_path}/{key}")
            metrics = report["metrics"]
            values = metrics["per_rollout_rel_l2"]
            mean = statistics.mean(values)
            if not math.isclose(mean, metrics["rollout_rel_l2"], abs_tol=2e-7):
                raise ValueError(f"HR aggregation mismatch: {report_path}")
            # projected_rollout_index appends each step's batch values in order.
            n = len(values)
            steps = metrics["per_step_rel_l2"]
            if len(steps) != n * int(report["rollout_steps"]):
                raise ValueError(f"HR flattened step layout mismatch: {report_path}")
            cells.append({
                "task": task, "family": report["family"], "mean": mean,
                "one_step": statistics.mean(steps[:n]), "last_step": statistics.mean(steps[-n:]),
                "report": str(report_path.relative_to(PROJECT_ROOT)),
                "report_sha256": sha256_file(report_path),
            })
        if cells:
            development.append({
                "trial": row["trial"], "model_seed": row["model_seed"], "support_seed": row["support_seed"],
                "m": row["m"], "complete_for_screen": len(cells) == len(row["task_ids"]),
                "hr_mean": statistics.mean(c["mean"] for c in cells),
                "hr_worst_family": max(c["mean"] for c in cells), "cells": cells,
            })
    terminal_hours = terminal_gpu_hours(ledger_rows)
    return {
        "schema": "phase1_exploratory_progress_v1", "query_read": False,
        "status": "IN_PROGRESS", "confirmation_evidence": False,
        "training": training, "hr_development": development,
        "terminal_ledger_gpu_hours": terminal_hours,
        "ledger_sha256": sha256_file(ledger_path),
        "limitations": ["Screens are exploratory and use few families/seeds.",
                       "Terminal sum excludes active jobs and early untimed HR evaluations.",
                        "Trials flagged EXCLUDED_RESOURCE_CONTENTION are omitted from performance summaries.",
                       "Time steps and crossed seeds are not independent generalization tasks."],
    }


def terminal_gpu_hours(ledger_rows: list[dict[str, str]]) -> float:
    terminal_charge_by_attempt = {}
    hrdev_execution_hours = 0.0
    for row in ledger_rows:
        if row["status"] == "RUNNING" or not row["elapsed_gpu_hours"]:
            continue
        trial_id = row["trial_id"]
        if row.get("round") == "HRDEV" and row["status"] == "DONE":
            # Each DONE evaluator row is real GPU work, even if a race reused
            # the logical report ID/path. Keep report aggregation deduplicated,
            # but account for every evaluation execution.
            hrdev_execution_hours += float(row["elapsed_gpu_hours"])
            continue
        # A logical task can have multiple real attempts (for example a FAILED
        # row followed by a resumed DONE row). Charge distinct attempt records
        # separately, while collapsing append-only annotations that repeat an
        # existing attempt. The old trial-ID-only max undercounted retries with
        # a new config hash or output path.
        attempt_key = (
            trial_id,
            row.get("config_sha256", ""),
            row.get("raw_output_path", ""),
        )
        terminal_charge_by_attempt[attempt_key] = max(
            terminal_charge_by_attempt.get(attempt_key, 0.0),
            float(row["elapsed_gpu_hours"]),
        )
    return sum(terminal_charge_by_attempt.values()) + hrdev_execution_hours


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = PROJECT_ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Use a fresh progress output: {output}")
    result = summarize()
    write_json(output, result)
    print(json.dumps({"output": str(output), "training_trials": len(result["training"]),
                      "hr_development_trials": len(result["hr_development"]),
                      "terminal_ledger_gpu_hours": result["terminal_ledger_gpu_hours"]}, indent=2))


if __name__ == "__main__":
    main()
