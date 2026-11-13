#!/usr/bin/env python3
"""Calculate rotated validation mAP for saved sparse-MTL YOLO checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from satellitepy.models.yolo_mtl.data import sparse_mtl_collate
from satellitepy.models.yolo_mtl.factory import build_model, load_config
from tools.train.yolo_mtl import OBBMAPEvaluator, make_dataset


DATASETS = {"dota": "dota", "potsdam": "potsdam-vaihingen"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/output/yolo_mask_effect"))
    parser.add_argument("--patch-root", type=Path, default=Path("/output/yolo_mtl_patches"))
    parser.add_argument("--config", type=Path,
                        default=Path("configs/yolo_mtl/yolo11s_mask_effect_seed1.yaml"))
    parser.add_argument("--domains", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--checkpoints", nargs="+", default=["best.pt", "last.pt"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    completed = set()
    if args.output.exists():
        with args.output.open(newline="") as stream:
            completed = {(row["domain"], row["run"], row["checkpoint"])
                         for row in csv.DictReader(stream)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists()

    with args.output.open("a", newline="") as stream:
        fields = ["domain", "run", "checkpoint", "epoch", "map50", "map50_95"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for domain in args.domains:
            val_root = args.patch_root / DATASETS[domain] / "rgb" / "val"
            dataset = make_dataset([val_root], config, False)
            loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                pin_memory=device.type == "cuda", collate_fn=sparse_mtl_collate,
                persistent_workers=args.workers > 0)
            run_folders = sorted(
                path for path in (args.root / domain / "runs").glob("cgc*")
                if not path.name.endswith("_seed1"))
            model = build_model(config).to(device).eval()
            for run_folder in run_folders:
                for checkpoint_name in args.checkpoints:
                    key = (domain, run_folder.name, checkpoint_name)
                    checkpoint_path = run_folder / checkpoint_name
                    if key in completed or not checkpoint_path.exists():
                        continue
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    model.load_state_dict(checkpoint["model"])
                    evaluator = OBBMAPEvaluator(
                        model.geometry_heads["obboxes"], config["train"].get("map"))
                    progress = tqdm(loader, desc=f"{domain}/{run_folder.name}/{checkpoint_name}",
                                    dynamic_ncols=True)
                    with torch.no_grad():
                        for batch in progress:
                            images = batch["img"].to(device, non_blocking=True)
                            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                                enabled=device.type == "cuda"):
                                outputs = model(
                                    images, sample_domains=batch["geometry_domain"])
                            evaluator.update(outputs["geometry"]["obboxes"],
                                             batch["geometry"]["obboxes"],
                                             batch["image_path"])
                    metrics = evaluator.compute()
                    writer.writerow({
                        "domain": domain,
                        "run": run_folder.name,
                        "checkpoint": checkpoint_name,
                        "epoch": int(checkpoint["epoch"]) + 1,
                        **metrics,
                    })
                    stream.flush()
                    print(domain, run_folder.name, checkpoint_name, metrics, flush=True)


if __name__ == "__main__":
    main()
