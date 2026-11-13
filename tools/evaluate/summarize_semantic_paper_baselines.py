#!/usr/bin/env python3
"""Summarize paper semantic baselines from existing oracle-box evaluations."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


EVAL_ROOT = Path("/output/yolo11_baselines/evaluations")
OUTPUT_ROOT = EVAL_ROOT / "semantic_paper_baselines"

# Detection-only checkpoints are deliberately excluded: they contain no semantic
# heads.  The independent joint/detached models are the single-dataset semantic
# references; the multi-dataset entries are the paper's main shared controls.
RUNS = (
    ("independent", "joint", 0, "FineAir", "fineair_joint_seed0_test.json"),
    ("independent", "detached", 0, "FineAir", "fineair_detached_seed0_test.json"),
    ("independent", "joint", 0, "RarePlanes", "rareplanes_joint_seed0_test.json"),
    ("independent", "detached", 0, "RarePlanes", "rareplanes_detached_seed0_test.json"),
    ("shared", "strict_detached", 0, "FineAir", "aircraft_dota_aircraft_strict_detached_fineair_test.json"),
    ("shared", "strict_detached", 0, "RarePlanes", "aircraft_dota_aircraft_strict_detached_rareplanes_test.json"),
    ("shared", "strict_detached", 1, "FineAir", "aircraft_dota_aircraft_strict_detached_seed1_fineair_test.json"),
    ("shared", "strict_detached", 1, "RarePlanes", "aircraft_dota_aircraft_strict_detached_seed1_rareplanes_test.json"),
    ("shared", "strict_detached", 2, "FineAir", "aircraft_dota_aircraft_strict_detached_seed2_fineair_test.json"),
    ("shared", "strict_detached", 2, "RarePlanes", "aircraft_dota_aircraft_strict_detached_seed2_rareplanes_test.json"),
    ("shared", "dota_adapter", 0, "FineAir", "aircraft_dota_aircraft_dota_adapter_fineair_test.json"),
    ("shared", "dota_adapter", 0, "RarePlanes", "aircraft_dota_aircraft_dota_adapter_rareplanes_test.json"),
    ("shared", "dota_adapter", 1, "FineAir", "aircraft_dota_aircraft_dota_adapter_seed1_fineair_test.json"),
    ("shared", "dota_adapter", 1, "RarePlanes", "aircraft_dota_aircraft_dota_adapter_seed1_rareplanes_test.json"),
    ("shared", "dota_adapter", 2, "FineAir", "aircraft_dota_aircraft_dota_adapter_seed2_fineair_test.json"),
    ("shared", "dota_adapter", 2, "RarePlanes", "aircraft_dota_aircraft_dota_adapter_seed2_rareplanes_test.json"),
)


def wilson_interval(correct: int, total: int) -> tuple[float | None, float | None]:
    if not total:
        return None, None
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - margin, center + margin


def classification_row(scope: str, variant: str, seed: int, dataset: str,
                       task: str, values: dict, source: Path) -> dict:
    per_class = values.get("per_class", {})
    supports = [entry["total"] for entry in per_class.values()]
    recalls = [entry["accuracy"] for entry in per_class.values()
               if entry.get("accuracy") is not None]
    total = int(values.get("total", 0))
    correct = int(values.get("correct", 0))
    low, high = wilson_interval(correct, total)
    return {
        "model_scope": scope,
        "variant": variant,
        "seed": seed,
        "dataset": dataset,
        "task": task,
        "metric": "accuracy",
        "value": values.get("accuracy"),
        "macro_recall": sum(recalls) / len(recalls) if recalls else None,
        "support": total,
        "classes_present": len(supports),
        "majority_baseline": max(supports) / total if total and supports else None,
        "uniform_chance_baseline": 1 / len(supports) if supports else None,
        "wilson_95_low": low,
        "wilson_95_high": high,
        "source": str(source),
    }


def regression_row(scope: str, variant: str, seed: int, dataset: str,
                   task: str, values: dict, source: Path) -> dict:
    return {
        "model_scope": scope,
        "variant": variant,
        "seed": seed,
        "dataset": dataset,
        "task": task,
        "metric": "original_scale_mae",
        "value": values.get("original_scale_mae"),
        "macro_recall": None,
        "support": values.get("total", 0),
        "classes_present": None,
        "majority_baseline": None,
        "uniform_chance_baseline": None,
        "wilson_95_low": None,
        "wilson_95_high": None,
        "source": str(source),
    }


def fmt(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    rows = []
    missing = []
    for scope, variant, seed, dataset, filename in RUNS:
        path = EVAL_ROOT / filename
        if not path.is_file():
            missing.append(str(path))
            continue
        result = json.loads(path.read_text())
        for task, values in result.get("classification", {}).items():
            if values.get("total", 0):
                rows.append(classification_row(
                    scope, variant, seed, dataset, task, values, path))
        for task, values in result.get("regression", {}).items():
            if values.get("total", 0):
                rows.append(regression_row(
                    scope, variant, seed, dataset, task, values, path))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (OUTPUT_ROOT / "semantic_paper_baselines.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_ROOT / "semantic_paper_baselines.json").write_text(json.dumps({
        "protocol": "oracle_box_center_scale_selected_pyramid_level",
        "notes": [
            "Majority and uniform references are computed on the same test labels.",
            "Macro recall is reported because several RarePlanes attributes are strongly imbalanced.",
            "Detection-only checkpoints are inapplicable because they have no semantic heads.",
        ],
        "missing_evaluations": missing,
        "rows": rows,
    }, indent=2) + "\n")

    display = [row for row in rows if row["metric"] == "accuracy" and
               row["task"] in {"fine-class", "very-fine-class", "role"}]
    lines = [
        "# Semantic paper baselines",
        "",
        "All learned metrics use oracle ground-truth boxes. Majority and uniform baselines use the identical test-label support.",
        "",
        "| Scope | Variant | Seed | Dataset | Task | Accuracy | Macro recall | Majority | Uniform | N |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in display:
        lines.append("| " + " | ".join(fmt(row[key]) for key in (
            "model_scope", "variant", "seed", "dataset", "task", "value",
            "macro_recall", "majority_baseline", "uniform_chance_baseline",
            "support")) + " |")
    if missing:
        lines.extend(("", "## Missing evaluations", ""))
        lines.extend(f"- `{path}`" for path in missing)
    (OUTPUT_ROOT / "semantic_paper_baselines.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"rows": len(rows), "missing": len(missing),
                      "output": str(OUTPUT_ROOT)}, indent=2))


if __name__ == "__main__":
    main()
