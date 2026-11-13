#!/usr/bin/env python3
"""Calculate validation foreground-mask IoU for selected YOLO MTL checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from satellitepy.models.yolo_mtl.data import sparse_mtl_collate
from satellitepy.models.yolo_mtl.factory import build_model, load_config
from tools.train.yolo_mtl import make_dataset


DATASETS = {"dota": "dota", "potsdam": "potsdam-vaihingen"}


def selected_checkpoints(root: Path):
    evaluated = root / "first_run_checkpoint_map.csv"
    choices = {}
    with evaluated.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["domain"], row["run"]
            if key not in choices or float(row["map50_95"]) > float(choices[key]["map50_95"]):
                choices[key] = row
    selected = []
    for (domain, condition), row in sorted(choices.items()):
        selected.append((domain, condition, 0,
                         root / domain / "runs" / condition / row["checkpoint"]))
        selected.append((domain, condition, 1,
                         root / domain / "runs" / f"{condition}_seed1" / "best.pt"))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/output/yolo_mask_effect"))
    parser.add_argument("--patch-root", type=Path, default=Path("/output/yolo_mtl_patches"))
    parser.add_argument("--config", type=Path,
                        default=Path("configs/yolo_mtl/yolo11s_mask_effect_seed1.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    selected = selected_checkpoints(args.root)
    selected = [item for index, item in enumerate(selected)
                if index % args.num_shards == args.shard_index]
    loaders = {}
    model = build_model(config).to(device).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        fields = ["dataset", "condition", "seed", "checkpoint", "epoch",
                  "mask_iou", "intersection", "union", "supervised_images"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for domain, condition, seed, checkpoint_path in selected:
            if domain not in loaders:
                val_root = args.patch_root / DATASETS[domain] / "rgb" / "val"
                dataset = make_dataset([val_root], config, False)
                loaders[domain] = DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=device.type == "cuda",
                    collate_fn=sparse_mtl_collate, persistent_workers=args.workers > 0)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            intersection = 0
            union = 0
            supervised_images = 0
            with torch.no_grad():
                for batch in tqdm(loaders[domain],
                                  desc=f"{domain}/{condition}/seed{seed}", dynamic_ncols=True):
                    available = batch["tasks"]["masks"]["image_mask"]
                    if not available.any():
                        continue
                    images = batch["img"].to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16,
                                        enabled=device.type == "cuda"):
                        logits = model(
                            images, sample_domains=batch["geometry_domain"]
                        )["tasks"]["masks"]
                    predictions = logits[available.to(device)].sigmoid() >= args.threshold
                    targets = batch["tasks"]["masks"]["mask"][available].to(device) >= 0.5
                    intersection += int((predictions & targets).sum().item())
                    union += int((predictions | targets).sum().item())
                    supervised_images += int(available.sum())
            iou = intersection / union if union else float("nan")
            writer.writerow({
                "dataset": domain, "condition": condition, "seed": seed,
                "checkpoint": str(checkpoint_path), "epoch": int(checkpoint["epoch"]) + 1,
                "mask_iou": iou, "intersection": intersection, "union": union,
                "supervised_images": supervised_images,
            })
            stream.flush()
            print(domain, condition, seed, iou, supervised_images, flush=True)


if __name__ == "__main__":
    main()
