#!/usr/bin/env python3
"""Create a seed-aggregated paper table for independent and shared detectors."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path


SOURCE = Path("docs/baseline_results.csv")
CSV_OUTPUT = Path("docs/detection_paper_baselines.csv")
MD_OUTPUT = Path("docs/detection_paper_baselines.md")
DATASETS = ("FineAir", "RarePlanes", "DOTA-aircraft")
INDEPENDENT_VARIANTS = {
    "detection_only": 0,
    "detection_only_seed1": 1,
    "detection_only_seed2": 2,
}
SHARED_VARIANTS = {
    "strict-detached": {
        "strict_detached": 0,
        "strict_detached_seed1": 1,
        "strict_detached_seed2": 2,
    },
    "DOTA adapter": {
        "dota_adapter": 0,
        "dota_adapter_seed1": 1,
        "dota_adapter_seed2": 2,
    },
}
SHARED_SCOPE = "FineAir+RarePlanes+DOTA-aircraft"
SHARED_COLUMNS = {
    "FineAir": "test_fineair_map50_95",
    "RarePlanes": "test_rareplanes_map50_95",
    "DOTA-aircraft": "test_dota_aircraft_map50_95",
}


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    rows = list(csv.DictReader(SOURCE.open()))
    independent: dict[str, dict[int, float]] = {}
    for dataset in DATASETS:
        independent[dataset] = {
            INDEPENDENT_VARIANTS[row["variant"]]: float(row["test_map50_95"])
            for row in rows
            if row["dataset_scope"] == dataset
            and row["variant"] in INDEPENDENT_VARIANTS
        }
        if set(independent[dataset]) != {0, 1, 2}:
            raise RuntimeError(f"Missing independent seeds for {dataset}")

    output = []
    for dataset in DATASETS:
        values = [independent[dataset][seed] for seed in range(3)]
        mean, sd = mean_sd(values)
        output.append({
            "model": "Independent detectors",
            "dataset": dataset,
            "seed0": values[0], "seed1": values[1], "seed2": values[2],
            "mean": mean, "sd": sd,
            "delta_vs_independent_mean": 0.0,
            "delta_sd": 0.0,
        })

    for label, variants in SHARED_VARIANTS.items():
        selected = {
            variants[row["variant"]]: row for row in rows
            if row["dataset_scope"] == SHARED_SCOPE and row["variant"] in variants
        }
        if set(selected) != {0, 1, 2}:
            raise RuntimeError(f"Missing shared seeds for {label}")
        for dataset in DATASETS:
            values = [float(selected[seed][SHARED_COLUMNS[dataset]]) for seed in range(3)]
            deltas = [values[seed] - independent[dataset][seed] for seed in range(3)]
            mean, sd = mean_sd(values)
            delta_mean, delta_sd = mean_sd(deltas)
            output.append({
                "model": label,
                "dataset": dataset,
                "seed0": values[0], "seed1": values[1], "seed2": values[2],
                "mean": mean, "sd": sd,
                "delta_vs_independent_mean": delta_mean,
                "delta_sd": delta_sd,
            })

    with CSV_OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    lines = [
        "# Three-seed detector comparison",
        "",
        "Test mAP50-95, reported as mean ± sample standard deviation over seeds 0-2. Deltas are seed-paired against the corresponding independent detector.",
        "",
        "| Model | Dataset | Seed 0 | Seed 1 | Seed 2 | Mean ± SD | Paired delta ± SD |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in output:
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['seed0']:.4f} | "
            f"{row['seed1']:.4f} | {row['seed2']:.4f} | "
            f"{row['mean']:.4f} ± {row['sd']:.4f} | "
            f"{row['delta_vs_independent_mean']:+.4f} ± {row['delta_sd']:.4f} |"
        )
    lines.extend((
        "",
        "With only three seeds, this table supports effect-size and stability comparisons, not a strong claim of statistical significance. Neither shared model consistently improves all three datasets over independent training.",
    ))
    MD_OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"updated {CSV_OUTPUT} and {MD_OUTPUT}")


if __name__ == "__main__":
    main()
