from __future__ import annotations

import json

from blrpng.data.generate import workspace_audit
from blrpng.paths import resolve_in_project, sha256_file, write_json


SOURCE = "artifacts/manifests/gray_scott_seed42_confirmation128_evaluation_gate.json"
OUTPUT = "artifacts/manifests/gray_scott_seed42_256_continuation_gate.json"


def main() -> None:
    """Authorize the frozen 256-grid development check without changing the paper gate.

    The original 128-grid summary predates selection of ``safe_path_nlt_gated`` as
    the main candidate and checks the final-only safe-path ablation instead.  This
    amendment is deliberately narrower than a paper gate: it permits only the
    already-precommitted 256-grid evaluation and records that the 75% CI-lower
    requirement is still unmet.
    """
    source_path = resolve_in_project(SOURCE, must_exist=True)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not source.get("frozen") or source.get("query_checkpoint_selection_used") is not False:
        raise PermissionError("The 128-grid confirmation report is not a frozen blind evaluation.")

    comparisons = source["comparisons"]
    scratch_h20 = comparisons["safe_path_nlt_gated_vs_lifted_scratch_h20"]
    ordinary_h20 = comparisons["safe_path_nlt_gated_vs_ordinary_nlt_h20"]
    ordinary_h50 = comparisons["safe_path_nlt_gated_vs_ordinary_nlt_h50"]
    development_checks = {
        "main_point_improvement_vs_scratch_at_least_75pct":
            float(scratch_h20["relative_improvement"]) >= 0.75,
        "main_better_than_ordinary_h20":
            float(ordinary_h20["relative_improvement"]) > 0.0,
        "main_better_than_ordinary_h50":
            float(ordinary_h50["relative_improvement"]) > 0.0,
        "all_four_shift_family_means_better_than_ordinary_h20": all(
            float(methods["safe_path_nlt_gated"]) < float(methods["ordinary_nlt"])
            for methods in source["h20_family_means"].values()
        ),
    }
    paper_checks = {
        "main_ci_lower_improvement_vs_scratch_at_least_75pct":
            float(scratch_h20["ci95"][0]) >= 0.75,
    }
    development_passed = all(development_checks.values())
    paper_claim_passed = development_passed and all(paper_checks.values())
    payload = {
        "schema": "blrpng_gray_scott_256_continuation_gate_v1",
        "frozen": True,
        "passed": development_passed,
        "paper_claim_passed": paper_claim_passed,
        "seed": 42,
        "main_method": "safe_path_nlt_gated",
        "source_gate": SOURCE,
        "source_gate_sha256": sha256_file(source_path),
        "source_gate_passed": bool(source.get("passed")),
        "development_checks": development_checks,
        "paper_checks": paper_checks,
        "scientific_scope": (
            "Authorizes only generation and evaluation of the precommitted 256-grid "
            "Gray--Scott confirmation query. It does not authorize a 75%-CI paper "
            "claim, multiseed confirmation, or the next PDE stage."
        ),
        "authorized_next_experiment": (
            "Generate the frozen 256-grid Gray--Scott confirmation query."
            if development_passed else None
        ),
    }
    with workspace_audit("freeze_gray_scott_256_continuation") as audit:
        output = write_json(OUTPUT, payload)
        audit.update({"output": str(output), "passed": development_passed})
    print(json.dumps({"output": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
