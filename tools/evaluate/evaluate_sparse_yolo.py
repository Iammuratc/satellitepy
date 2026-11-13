#!/usr/bin/env python3
"""Evaluate OBB detection and all configured oracle-box semantic tasks."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from satellitepy.data.utils import get_task_dict
from satellitepy.models.yolo_mtl.data import sparse_mtl_collate
from satellitepy.models.yolo_mtl.factory import build_model, load_config
from tools.train.yolo_mtl import OBBMAPEvaluator, make_dataset


def gather_object_predictions(logits, target, strides):
    """Mirror training's pyramid-level selection and center sampling."""
    if not len(target["target"]):
        return None, None
    device = logits[0].device
    batch_indices = target["batch_idx"].to(device)
    xywh = target["xywh"].to(device)
    values = target["target"].to(device)
    image_size = logits[0].shape[-1] * strides[0]
    object_size = xywh[:, 2:].amax(1) * image_size
    nominal_size = strides * 4
    levels = (torch.log2(object_size[:, None].clamp_min(1) / nominal_size[None])
              .abs().argmin(1))
    gathered, ordered = [], []
    for level_index, level_logits in enumerate(logits):
        selected = levels == level_index
        if not selected.any():
            continue
        height, width = level_logits.shape[-2:]
        centers = xywh[selected, :2]
        xs = (centers[:, 0] * width).long().clamp_(0, width - 1)
        ys = (centers[:, 1] * height).long().clamp_(0, height - 1)
        gathered.append(level_logits[batch_indices[selected], :, ys, xs])
        ordered.append(values[selected])
    return torch.cat(gathered), torch.cat(ordered)


def classification_result(correct, total, confusion, task):
    inverse = {}
    for name, index in get_task_dict(task).items():
        inverse.setdefault(int(index), str(name))
    per_class = {}
    for true_index, values in sorted(confusion.items()):
        class_total = sum(values.values())
        per_class[inverse.get(true_index, str(true_index))] = {
            "correct": values.get(true_index, 0),
            "total": class_total,
            "accuracy": values.get(true_index, 0) / class_total if class_total else None,
            "predictions": {
                inverse.get(predicted, str(predicted)): count
                for predicted, count in sorted(values.items())
            },
        }
    return {
        "protocol": "oracle_box_center_scale_selected_pyramid_level",
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "per_class": per_class,
    }


def regression_result(error_sum, squared_error_sum, total, task):
    metadata = get_task_dict(task)
    scale = float(metadata["max"] - metadata["min"])
    normalized_mae = error_sum / total if total else None
    normalized_rmse = math.sqrt(squared_error_sum / total) if total else None
    return {
        "protocol": "oracle_box_center_scale_selected_pyramid_level",
        "total": total,
        "normalized_mae": normalized_mae,
        "normalized_rmse": normalized_rmse,
        "original_scale_mae": normalized_mae * scale if total else None,
        "original_scale_rmse": normalized_rmse * scale if total else None,
        "original_scale_min": float(metadata["min"]),
        "original_scale_max": float(metadata["max"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    dataset = make_dataset([args.split], config, False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        collate_fn=sparse_mtl_collate, persistent_workers=args.workers > 0)
    model = build_model(config).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    geometry_name = (dataset.geometry_family
                     if model.geometry_families else "obboxes")
    geometry_domain = dataset.geometry_domain
    evaluator = OBBMAPEvaluator(
        model.geometry_heads[geometry_name], config["train"].get("map"))

    classification_tasks = [
        name for name, spec in model.task_specs.items()
        if spec.kind == "classification"
    ]
    regression_tasks = [
        name for name, spec in model.task_specs.items()
        if spec.kind == "regression"
    ]
    correct = defaultdict(int)
    total = defaultdict(int)
    confusion = {
        name: defaultdict(lambda: defaultdict(int))
        for name in classification_tasks
    }
    error_sum = defaultdict(float)
    squared_error_sum = defaultdict(float)

    with torch.no_grad():
        for batch in tqdm(loader, desc=args.checkpoint.parent.name, dynamic_ncols=True):
            images = batch["img"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                outputs = model(images, sample_domains=batch['geometry_domain'])
            geometry_outputs = (outputs.get("geometry_by_domain", {}).get(
                geometry_domain, outputs["geometry"]))
            evaluator.update(geometry_outputs[geometry_name],
                             batch["geometry"]["obboxes"], batch["image_path"])
            for task in classification_tasks:
                logits, targets = gather_object_predictions(
                    outputs["tasks"][task], batch["tasks"][task], model.stride)
                if logits is None:
                    continue
                targets = targets.long()
                predictions = logits.argmax(1)
                correct[task] += int((predictions == targets).sum())
                total[task] += len(targets)
                for truth, prediction in zip(targets.cpu().tolist(),
                                             predictions.cpu().tolist()):
                    confusion[task][truth][prediction] += 1
            for task in regression_tasks:
                logits, targets = gather_object_predictions(
                    outputs["tasks"][task], batch["tasks"][task], model.stride)
                if logits is None:
                    continue
                errors = logits.squeeze(-1).float() - targets.float()
                total[task] += len(targets)
                error_sum[task] += errors.abs().sum().item()
                squared_error_sum[task] += errors.square().sum().item()

    result = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "split": str(args.split),
        "images": len(dataset),
        "detection": evaluator.compute(),
        "classification": {
            task: classification_result(correct[task], total[task], confusion[task], task)
            for task in classification_tasks
        },
        "regression": {
            task: regression_result(
                error_sum[task], squared_error_sum[task], total[task], task)
            for task in regression_tasks
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({
        "checkpoint_epoch": result["checkpoint_epoch"],
        "images": result["images"],
        "detection": result["detection"],
        "classification_accuracy": {
            name: values["accuracy"]
            for name, values in result["classification"].items()
        },
        "regression_normalized_mae": {
            name: values["normalized_mae"]
            for name, values in result["regression"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
