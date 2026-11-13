#!/usr/bin/env python3
"""Refresh permanent CSV and Markdown summaries for aircraft chip experiments."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import torch


ROOT = Path("/output/aircraft_chip_mtl")
CSV_OUTPUT = Path("/workspace/satellitepy/docs/aircraft_chip_mtl_results.csv")
MD_OUTPUT = Path("/workspace/satellitepy/docs/aircraft_chip_mtl_results.md")


def average(metrics: dict, dataset: str, suffix: str):
    values = [value for key, value in metrics.items()
              if key.startswith(f"{dataset}:") and key.endswith(suffix)]
    return sum(values) / len(values) if values else None


def family(name: str) -> str:
    for suffix in ("_seed0", "_seed1", "_seed2"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def fmt(value, scale=100):
    return "—" if value is None else f"{scale * value:.2f}"


def main():
    rows = []
    metric_names = set()
    for result_path in sorted(ROOT.glob("*/test_metrics.json")):
        folder = result_path.parent
        config_path = folder / "config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text())
        metrics = json.loads(result_path.read_text())
        checkpoint = torch.load(folder / "best.pt", map_location="cpu",
                                weights_only=False)
        row = {
            "run": folder.name,
            "family": family(folder.name),
            "seed": config.get("seed"),
            "datasets": "+".join(config.get("datasets", [])),
            "mask_route": config.get("mask_route"),
            "mask_weight": config.get("mask_weight"),
            "best_epoch": checkpoint.get("epoch"),
            "fineair_mean_accuracy": average(metrics, "fineair", "/accuracy"),
            "fineair_mean_macro_recall": average(metrics, "fineair", "/macro_recall"),
            "fineair_mean_macro_f1": average(metrics, "fineair", "/macro_f1"),
            "rareplanes_mean_accuracy": average(metrics, "rareplanes", "/accuracy"),
            "rareplanes_mean_macro_recall": average(metrics, "rareplanes", "/macro_recall"),
            "rareplanes_mean_macro_f1": average(metrics, "rareplanes", "/macro_f1"),
            "dota_mask_dice": metrics.get("dota/mask_dice"),
            "dota_mask_iou": metrics.get("dota/mask_iou"),
            "mean_primary": metrics.get("mean_primary"),
            "output": str(folder),
        }
        row.update(metrics)
        metric_names.update(metrics)
        rows.append(row)

    base_fields = [
        "run", "family", "seed", "datasets", "mask_route", "mask_weight",
        "best_epoch", "fineair_mean_accuracy", "fineair_mean_macro_recall",
        "fineair_mean_macro_f1", "rareplanes_mean_accuracy",
        "rareplanes_mean_macro_recall", "rareplanes_mean_macro_f1",
        "dota_mask_dice", "dota_mask_iou", "mean_primary", "output",
    ]
    fields = base_fields + sorted(metric_names - set(base_fields))
    with CSV_OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Aircraft chip MTL results",
        "",
        "Held-out scene-level test results. Classification columns are unweighted "
        "means across the available tasks within each dataset. DOTA Dice is the "
        "pixel-level F1 score.",
        "",
        "| Run | Seed | Mask route | Weight | FineAir acc. | FineAir macro R | "
        "FineAir macro F1 | RarePlanes acc. | RarePlanes macro R | "
        "RarePlanes macro F1 | DOTA Dice/F1 | DOTA IoU |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['seed']} | {row['mask_route']} | "
            f"{row['mask_weight']:.2f} | {fmt(row['fineair_mean_accuracy'])} | "
            f"{fmt(row['fineair_mean_macro_recall'])} | "
            f"{fmt(row['fineair_mean_macro_f1'])} | "
            f"{fmt(row['rareplanes_mean_accuracy'])} | "
            f"{fmt(row['rareplanes_mean_macro_recall'])} | "
            f"{fmt(row['rareplanes_mean_macro_f1'])} | "
            f"{fmt(row['dota_mask_dice'])} | {fmt(row['dota_mask_iou'])} |")

    lines.extend(("", "## Three-seed summary", "",
                  "Values are mean ± sample standard deviation in percentage points.", "",
                  "| Family | FineAir acc. | FineAir macro R | FineAir macro F1 | "
                  "RarePlanes acc. | RarePlanes macro R | RarePlanes macro F1 | "
                  "DOTA Dice/F1 | DOTA IoU |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"))
    grouped = {}
    for row in rows:
        grouped.setdefault(row["family"], []).append(row)
    for name in ("fineair_independent", "rareplanes_independent",
                 "dota_mask_independent", "dota_mask_independent_early",
                 "shared_conditional_semantics",
                 "shared_layer1_mask", "shared_early_mask", "shared_layer3_mask",
                 "shared_full_mask", "shared_early_mask_detached"):
        group = grouped.get(name, [])
        if len(group) != 3:
            continue
        values = []
        for key in ("fineair_mean_accuracy", "fineair_mean_macro_recall",
                    "fineair_mean_macro_f1", "rareplanes_mean_accuracy",
                    "rareplanes_mean_macro_recall", "rareplanes_mean_macro_f1",
                    "dota_mask_dice", "dota_mask_iou"):
            samples = [row[key] for row in group if row[key] is not None]
            values.append("—" if not samples else
                          f"{100 * statistics.mean(samples):.2f} ± "
                          f"{100 * statistics.stdev(samples):.2f}")
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    MD_OUTPUT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"runs": len(rows), "csv": str(CSV_OUTPUT),
                      "markdown": str(MD_OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
