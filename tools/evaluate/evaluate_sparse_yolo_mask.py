#!/usr/bin/env python3
"""Evaluate foreground-mask IoU for one sparse YOLO checkpoint and split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from satellitepy.models.yolo_mtl.data import sparse_mtl_collate
from satellitepy.models.yolo_mtl.factory import build_model, load_config
from tools.train.yolo_mtl import make_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    model = build_model(config).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    dataset = make_dataset([args.split], config, False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == 'cuda',
        collate_fn=sparse_mtl_collate, persistent_workers=args.workers > 0)

    intersection = union = supervised_images = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=args.split.parent.name, dynamic_ncols=True):
            available = batch['tasks']['masks']['image_mask']
            if not available.any():
                continue
            images = batch['img'].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == 'cuda'):
                logits = model(
                    images, sample_domains=batch['geometry_domain'])['tasks']['masks']
            predictions = logits[available.to(device)].sigmoid() >= args.threshold
            targets = batch['tasks']['masks']['mask'][available].to(device) >= 0.5
            valid = batch['tasks']['masks']['valid_mask'][available].to(device).bool()
            intersection += int((predictions & targets & valid).sum().item())
            union += int(((predictions | targets) & valid).sum().item())
            supervised_images += int(available.sum())

    result = {
        'checkpoint_epoch': int(checkpoint['epoch']) + 1,
        'images': len(dataset),
        'supervised_images': supervised_images,
        'threshold': args.threshold,
        'mask_iou': intersection / union if union else float('nan'),
        'intersection': intersection,
        'union': union,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
