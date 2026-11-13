#!/usr/bin/env python3
"""Evaluate FineAir OBB detection and oracle-box semantic classification.

Detection metrics use decoded predictions and NMS. Classification metrics use
the ground-truth object center and scale to sample the corresponding dense task
logits. The latter isolates classifier quality from detection matching and is
therefore reported as ``oracle_box`` accuracy.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from satellitepy.data.utils import get_task_dict
from satellitepy.models.yolo_mtl.data import sparse_mtl_collate
from satellitepy.models.yolo_mtl.factory import build_model, load_config
from tools.train.yolo_mtl import OBBMAPEvaluator, make_dataset


def gather_object_logits(logits, target, strides):
    """Mirror the training loss's scale selection and center sampling."""
    if not len(target["target"]):
        return None, None
    device = logits[0].device
    batch_indices = target["batch_idx"].to(device)
    xywh = target["xywh"].to(device)
    values = target["target"].to(device).long()
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
        inverse.setdefault(int(index), name)
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
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    evaluator = OBBMAPEvaluator(
        model.geometry_heads["obboxes"], config["train"].get("map"))
    semantic_tasks = [
        name for name in ("fine-class", "very-fine-class")
        if name in model.task_heads
    ]
    correct = defaultdict(int)
    total = defaultdict(int)
    confusion = {name: defaultdict(lambda: defaultdict(int)) for name in semantic_tasks}

    with torch.no_grad():
        for batch in tqdm(loader, desc=args.checkpoint.parent.name, dynamic_ncols=True):
            images = batch["img"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                outputs = model(images, sample_domains=batch["geometry_domain"])
            evaluator.update(outputs["geometry"]["obboxes"],
                             batch["geometry"]["obboxes"], batch["image_path"])
            for task in semantic_tasks:
                logits, targets = gather_object_logits(
                    outputs["tasks"][task], batch["tasks"][task], model.stride)
                if logits is None:
                    continue
                predictions = logits.argmax(1)
                correct[task] += int((predictions == targets).sum())
                total[task] += len(targets)
                for truth, prediction in zip(targets.cpu().tolist(),
                                             predictions.cpu().tolist()):
                    confusion[task][truth][prediction] += 1

    result = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "split": str(args.split),
        "images": len(dataset),
        "detection": evaluator.compute(),
        "classification": {
            task: classification_result(correct[task], total[task], confusion[task], task)
            for task in semantic_tasks
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({
        "checkpoint_epoch": result["checkpoint_epoch"],
        "images": result["images"],
        "detection": result["detection"],
        "classification": {
            name: {"accuracy": values["accuracy"], "total": values["total"]}
            for name, values in result["classification"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
