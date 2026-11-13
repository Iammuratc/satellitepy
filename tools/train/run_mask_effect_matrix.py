#!/usr/bin/env python3
"""Schedule partial-mask experiments, one independent run per GPU."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import torch


CONDITIONS = [
    "cgc100_mask000", "cgc100_mask010", "cgc100_mask025", "cgc100_mask050",
    "cgc100_mask100", "cgc050_mask000", "cgc050_mask010", "cgc050_mask025",
    "cgc050_mask050", "cgc025_mask000", "cgc025_mask010", "cgc025_mask025",
    "cgc010_mask000", "cgc010_mask010",
]
DATASETS = {
    "dota": "dota",
    "potsdam": "potsdam-vaihingen",
}


def checkpoint_epoch(path: Path):
    if not path.exists():
        return 0
    try:
        return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"]) + 1
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/output/yolo_mask_effect"))
    parser.add_argument("--patch-root", type=Path, default=Path("/output/yolo_mtl_patches"))
    parser.add_argument("--config", type=Path,
                        default=Path("configs/yolo_mtl/yolo11s_mask_effect.yaml"))
    parser.add_argument("--domains", nargs="+", choices=DATASETS, default=["dota", "potsdam"])
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-key-file", type=Path)
    args = parser.parse_args()
    wandb_key = None
    if args.wandb_key_file:
        wandb_key = args.wandb_key_file.read_text().strip()
        if not wandb_key:
            raise SystemExit("The W&B key file is empty")

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise SystemExit("No CUDA GPU is available")
    slots = list(range(min(gpu_count, args.max_parallel)))
    queue = [(domain, condition) for domain in args.domains for condition in args.conditions]
    running = {}

    while queue or running:
        for gpu in slots:
            if gpu in running or not queue:
                continue
            domain, condition = queue.pop(0)
            run_name = f"{condition}{args.run_suffix}"
            output = args.root / domain / "runs" / run_name
            output.mkdir(parents=True, exist_ok=True)
            checkpoint = output / "last.pt"
            completed = checkpoint_epoch(checkpoint)
            if completed >= args.epochs:
                print(f"skip {domain}/{condition}: epoch {completed}", flush=True)
                continue
            train_folder = args.root / domain / "data" / condition / "train"
            val_folder = args.patch_root / DATASETS[domain] / "rgb" / "val"
            command = [
                "python", "tools/train/yolo_mtl.py",
                "--train-folders", str(train_folder),
                "--val-folders", str(val_folder),
                "--config", str(args.config),
                "--out-folder", str(output),
                "--epochs", str(args.epochs),
                "--seed", str(args.seed),
            ]
            if completed:
                command.extend(("--resume", str(checkpoint)))
            if args.wandb_project:
                command.extend((
                    "--wandb-project", args.wandb_project,
                    "--wandb-group", domain,
                    "--wandb-name", run_name,
                    "--wandb-id", f"{domain}-{run_name}",
                ))
            log = (output / "train.log").open("ab")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["PYTHONPATH"] = str(Path.cwd())
            if wandb_key:
                environment["WANDB_API_KEY"] = wandb_key
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=environment)
            running[gpu] = process, log, domain, condition
            print(f"start gpu={gpu} {domain}/{condition} resume_epoch={completed}", flush=True)

        time.sleep(10)
        for gpu, (process, log, domain, condition) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log.close()
            del running[gpu]
            print(f"finish gpu={gpu} {domain}/{condition} exit={return_code} ", flush=True)
            if return_code:
                raise SystemExit(f"Experiment failed: {domain}/{condition}; see its train.log")


if __name__ == "__main__":
    main()
