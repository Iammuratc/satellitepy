#!/usr/bin/env python3
"""Re-evaluate completed aircraft-chip checkpoints with current test metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.train.train_aircraft_chip_mtl import (
    AircraftChipDataset, AircraftChipMTL, collate, evaluate,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/output/aircraft_chip_mtl"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    device = torch.device(args.device)

    for marker in sorted(args.root.glob("*/completed.ok")):
        folder = marker.parent
        config_path, checkpoint_path = folder / "config.json", folder / "best.pt"
        if not config_path.is_file() or not checkpoint_path.is_file():
            continue
        config = json.loads(config_path.read_text())
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        vocabs = checkpoint["vocabs"]
        dataset = AircraftChipDataset(Path(config["data"]), "test", set(config["datasets"]))
        loader = DataLoader(dataset, batch_size=config.get("batch_size", 64),
                            shuffle=False, num_workers=args.workers, collate_fn=collate,
                            pin_memory=True, persistent_workers=False)
        model = AircraftChipMTL(vocabs, config["mask_route"], pretrained=False).to(device)
        model.load_state_dict(checkpoint["model"])
        metrics = evaluate(model, loader, vocabs, device)
        (folder / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps({"run": folder.name, "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()
