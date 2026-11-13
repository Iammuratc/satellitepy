#!/usr/bin/env python3
"""Build a compact comparison of the three FineAir YOLO11 baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


RUNS = (
    (
        "fully_joint",
        "/output/yolo11_baselines/separate/fineair_seed0/best.pt",
        "/output/yolo11_baselines/evaluations/fineair_joint_seed0_test.json",
        "fineair-y11s-sep-s0",
    ),
    (
        "classification_detached",
        "/output/yolo11_baselines/detached/fineair_seed0/best.pt",
        "/output/yolo11_baselines/evaluations/fineair_detached_seed0_test.json",
        "fineair-y11s-det-s0",
    ),
    (
        "detection_only",
        "/output/yolo11_baselines/detection_only/fineair_seed0/best.pt",
        "/output/yolo11_baselines/evaluations/fineair_detection_only_seed0_test.json",
        "fineair-y11s-detonly-s0",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/output/yolo11_baselines/evaluations"),
    )
    return parser.parse_args()


def load_row(name: str, checkpoint: str, evaluation: str, wandb_id: str) -> dict[str, Any]:
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    test_data = json.loads(Path(evaluation).read_text())
    classification = test_data.get("classification", {})
    fine = classification.get("fine-class", {})
    very_fine = classification.get("very-fine-class", {})
    validation = checkpoint_data["validation_metrics"]
    return {
        "model": name,
        "wandb_id": wandb_id,
        "checkpoint": checkpoint,
        "checkpoint_epoch": test_data["checkpoint_epoch"],
        "val_map50": float(validation["map50"]),
        "val_map50_95": float(validation["map50_95"]),
        "test_images": int(test_data["images"]),
        "test_map50": float(test_data["detection"]["map50"]),
        "test_map50_95": float(test_data["detection"]["map50_95"]),
        "test_fine_accuracy": fine.get("accuracy"),
        "test_fine_objects": fine.get("total"),
        "test_very_fine_accuracy": very_fine.get("accuracy"),
        "test_very_fine_objects": very_fine.get("total"),
    }


def markdown_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [load_row(*run) for run in RUNS]

    detection_only = next(row for row in rows if row["model"] == "detection_only")
    for row in rows:
        row["test_map50_95_delta_vs_detection_only"] = (
            row["test_map50_95"] - detection_only["test_map50_95"]
        )

    json_path = args.output_dir / "fineair_baseline_comparison.json"
    csv_path = args.output_dir / "fineair_baseline_comparison.csv"
    markdown_path = args.output_dir / "fineair_baseline_comparison.md"
    json_path.write_text(json.dumps({"runs": rows}, indent=2) + "\n")

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    columns = (
        "model",
        "checkpoint_epoch",
        "val_map50_95",
        "test_map50",
        "test_map50_95",
        "test_map50_95_delta_vs_detection_only",
        "test_fine_accuracy",
        "test_very_fine_accuracy",
    )
    labels = (
        "Model",
        "Best epoch",
        "Val mAP50-95",
        "Test mAP50",
        "Test mAP50-95",
        "Test delta vs det-only",
        "Fine accuracy",
        "Very-fine accuracy",
    )
    lines = [
        "# FineAir YOLO11s baseline comparison",
        "",
        "All test metrics use the same untouched 213-image split. Classification metrics use oracle ground-truth boxes.",
        "",
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---" for _ in labels) + " |",
    ]
    lines.extend(
        "| " + " | ".join(markdown_value(row[column]) for column in columns) + " |"
        for row in rows
    )
    markdown_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "markdown": str(markdown_path)}))


if __name__ == "__main__":
    main()
