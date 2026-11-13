#!/usr/bin/env python3
"""Create a training-only Fourier Domain Adaptation DOTA-aircraft dataset.

DOTA training images keep their phase and high-frequency amplitude, while the
central low-frequency amplitude square is replaced by that of a deterministic
FineAir or RarePlanes reference patch. Validation and test images remain
native DOTA so evaluation measures performance on the original target domain.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dota-root', type=Path,
                        default=Path('/output/yolo_mtl_patches/dota_aircraft/rgb'))
    parser.add_argument('--fineair-root', type=Path,
                        default=Path('/output/yolo_mtl_patches/fineair/rgbir'))
    parser.add_argument('--rareplanes-root', type=Path,
                        default=Path('/output/yolo_mtl_patches/rareplanes/rgb'))
    parser.add_argument('--out-folder', type=Path,
                        default=Path('/output/yolo_mtl_patches_fda/dota_aircraft/rgb'))
    parser.add_argument('--beta', type=float, default=0.01,
                        help='Half-width of swapped spectrum region as an image-size fraction.')
    parser.add_argument('--amplitude-mix', type=float, default=1.0,
                        help='Target amplitude weight inside the low-frequency region.')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def images(folder: Path):
    return sorted(path for path in folder.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def read_rgb(path: Path):
    # IMREAD_COLOR deliberately drops FineAir's IR/alpha channel: FDA aligns
    # only the three visible channels used by the DOTA detector.
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Could not read {path}')
    return image


def fda(source: np.ndarray, target: np.ndarray, beta: float, mix: float):
    height, width = source.shape[:2]
    target = cv2.resize(target, (width, height), interpolation=cv2.INTER_AREA)
    source_float = source.astype(np.float32)
    target_float = target.astype(np.float32)
    source_fft = np.fft.fftshift(np.fft.fft2(source_float, axes=(0, 1)), axes=(0, 1))
    target_fft = np.fft.fftshift(np.fft.fft2(target_float, axes=(0, 1)), axes=(0, 1))
    source_amplitude = np.abs(source_fft)
    target_amplitude = np.abs(target_fft)
    radius = max(1, int(min(height, width) * beta))
    cy, cx = height // 2, width // 2
    region = np.s_[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1, :]
    source_amplitude[region] = ((1.0 - mix) * source_amplitude[region]
                                + mix * target_amplitude[region])
    adapted_fft = source_amplitude * np.exp(1j * np.angle(source_fft))
    adapted = np.fft.ifft2(np.fft.ifftshift(adapted_fft, axes=(0, 1)),
                           axes=(0, 1)).real
    return np.clip(adapted, 0, 255).astype(np.uint8)


def link(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        destination.symlink_to(source)


def main():
    args = parse_args()
    if not 0 < args.beta <= 0.5:
        raise ValueError('--beta must be in (0, 0.5]')
    if not 0 <= args.amplitude_mix <= 1:
        raise ValueError('--amplitude-mix must be in [0, 1]')
    references = {
        'fineair': images(args.fineair_root / 'train/images'),
        'rareplanes': images(args.rareplanes_root / 'train/images'),
    }
    if not all(references.values()):
        raise ValueError('Both FDA reference pools must contain training images')

    rng = random.Random(args.seed)
    train_sources = images(args.dota_root / 'train/images')
    train_out = args.out_folder / 'train'
    completed_path = train_out / 'completed_images.txt'
    train_out.mkdir(parents=True, exist_ok=True)
    completed = set(completed_path.read_text().splitlines()) if completed_path.exists() else set()
    assignments = []
    for index, source_path in enumerate(train_sources):
        domain = 'fineair' if index % 2 == 0 else 'rareplanes'
        reference_path = rng.choice(references[domain])
        assignments.append({'image': source_path.name, 'reference_domain': domain,
                            'reference': reference_path.name})
        destination = train_out / 'images' / source_path.name
        if source_path.name not in completed:
            adapted = fda(read_rgb(source_path), read_rgb(reference_path),
                          args.beta, args.amplitude_mix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), adapted):
                raise OSError(f'Failed to write {destination}')
            with completed_path.open('a') as handle:
                handle.write(f'{source_path.name}\n')
        label = args.dota_root / 'train/labels' / f'{source_path.stem}.json'
        link(label, train_out / 'labels' / label.name)

    manifest = {
        'method': 'Fourier Domain Adaptation (training only)',
        'source_domain': 'DOTA-aircraft',
        'reference_domains': ['FineAir RGB', 'RarePlanes RGB'],
        'reference_assignment': 'deterministic alternating 50/50 by sorted source image',
        'beta': args.beta,
        'amplitude_mix': args.amplitude_mix,
        'seed': args.seed,
        'training_images': len(train_sources),
        'validation_test_policy': 'native DOTA images and labels; no FDA',
        'assignments': assignments,
    }
    (train_out / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    for split in ('val', 'test'):
        for kind in ('images', 'labels'):
            for source in sorted((args.dota_root / split / kind).iterdir()):
                if source.is_file():
                    link(source, args.out_folder / split / kind / source.name)
        split_manifest = dict(manifest, split=split, transformed=False,
                              images=len(images(args.dota_root / split / 'images')))
        (args.out_folder / split).mkdir(parents=True, exist_ok=True)
        (args.out_folder / split / 'manifest.json').write_text(
            json.dumps(split_manifest, indent=2))
    (args.out_folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: value for key, value in manifest.items()
                      if key != 'assignments'}, indent=2))


if __name__ == '__main__':
    main()
