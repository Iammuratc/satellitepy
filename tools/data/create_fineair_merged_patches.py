#!/usr/bin/env python3
"""Create leakage-safe, class-balanced FineAir RGB+NIR patches.

The original TIFFs are read window-by-window. Bands 1--3 are saved in OpenCV
BGR order and band 4 is retained as NIR in the fourth PNG channel. Original
scenes, rather than patches from the same scene, are assigned to train/val/test.
Patches intersecting a cut-off airplane are rejected, so partial objects never
become either positive examples or unlabelled background.

This file is directly executable from any working directory.  It only assumes
that it remains inside the SatellitePy checkout and that the Python dependencies
of SatellitePy are installed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

# Make ``python /path/to/create_fineair_merged_patches.py`` work without an
# editable SatellitePy installation or a manually configured PYTHONPATH.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import cv2
import numpy as np
import rasterio

from satellitepy.data.labels import read_label
from satellitepy.data.patch import (
    get_pad_size,
    get_patch_start_coords,
)
from satellitepy.data.utils import (
    fineair_band_stretches,
    get_all_satellitepy_keys,
    get_satellitepy_dict_values,
    init_satellitepy_label,
    read_fineair_rgb_ir,
    set_image_keys,
)


def matched_scenes(root: Path):
    images = {path.stem: path for path in (root / 'images').glob('*.tif')}
    label_root = root / 'labels_fineair' / 'role_th_50'
    labels = {path.stem: path for path in label_root.glob('*.geojson')}
    names = sorted(images.keys() & labels.keys())
    if not names:
        raise ValueError(f'No matching FineAir images and labels under {root}')
    return [(images[name], labels[name]) for name in names]


def _scene_counts(label_path: Path):
    labels = read_label(label_path, 'fineair')
    return Counter(value for value in get_satellitepy_dict_values(labels, 'fine-class')
                   if value is not None)


def balanced_scene_split(scenes, seed=42, ratios=(0.8, 0.1, 0.1), trials=20000):
    """Random-search scene assignments with the notebook's class-balance goal."""
    names = ('train', 'val', 'test')
    counts = [_scene_counts(label) for _, label in scenes]
    class_names = sorted({name for count in counts for name in count})
    matrix = np.asarray([[count[name] for name in class_names] for count in counts],
                        dtype=np.float64)
    totals = matrix.sum(0).clip(min=1)
    scene_targets = np.asarray([round(len(scenes) * ratio) for ratio in ratios])
    scene_targets[0] += len(scenes) - int(scene_targets.sum())
    rng = random.Random(seed)
    best = None
    best_score = float('inf')
    indices = list(range(len(scenes)))
    for _ in range(trials):
        rng.shuffle(indices)
        assignments = {}
        offset = 0
        score = 0.0
        for split_index, (name, count, ratio) in enumerate(zip(names, scene_targets, ratios)):
            selected = indices[offset:offset + count]
            offset += count
            assignments[name] = tuple(selected)
            observed = matrix[selected].sum(0) if selected else np.zeros_like(totals)
            score += float(np.mean(np.abs(observed / totals - ratio)))
            # Strongly discourage a class disappearing from validation/test.
            if split_index and np.any((totals >= 10) & (observed == 0)):
                score += 1.0
        if score < best_score:
            best_score, best = score, assignments
    return {name: tuple(scenes[index] for index in best[name]) for name in names}


def _patch_labels(labels, x0, y0, patch_size):
    """Return labels for a safe patch, or ``None`` if an object is cut off."""
    result = init_satellitepy_label()
    result['_provided_geometry'] = ['obboxes']
    result['_provided_tasks'] = [
        'coarse-class', 'fine-class', 'very-fine-class', 'fineair-class',
        'role', 'source',
    ]
    tasks = get_all_satellitepy_keys()
    included = []
    for index, corners in enumerate(labels['obboxes']):
        points = np.asarray(corners)
        minimum = points.min(0)
        maximum = points.max(0)
        intersects = not (
            maximum[0] <= x0 or minimum[0] >= x0 + patch_size or
            maximum[1] <= y0 or minimum[1] >= y0 + patch_size)
        if not intersects:
            continue
        contained = (
            minimum[0] >= x0 and maximum[0] <= x0 + patch_size and
            minimum[1] >= y0 and maximum[1] <= y0 + patch_size)
        if not contained:
            return None, []
        set_image_keys(tasks, result, labels, index)
        result['obboxes'][-1] = (np.asarray(result['obboxes'][-1]) - [x0, y0]).tolist()
        included.append(index)
    return result, included


def write_scene(image_path, label_path, destination, patch_size=1024,
                patch_overlap=100, keep_empty=False):
    labels = read_label(label_path, 'fineair')
    stretches = fineair_band_stretches(image_path)
    with rasterio.open(image_path) as source:
        x_padded = source.width + get_pad_size(source.width, patch_size, patch_overlap)
        y_padded = source.height + get_pad_size(source.height, patch_size, patch_overlap)
    xs = get_patch_start_coords(x_padded, patch_size, patch_overlap)
    ys = get_patch_start_coords(y_padded, patch_size, patch_overlap)
    image_folder = destination / 'images'
    label_folder = destination / 'labels'
    image_folder.mkdir(parents=True, exist_ok=True)
    label_folder.mkdir(parents=True, exist_ok=True)
    written = 0
    covered = set()
    visited = set()

    def write_window(x0, y0):
        nonlocal written
        coordinates = (int(x0), int(y0))
        if coordinates in visited:
            return
        visited.add(coordinates)
        x0, y0 = coordinates
        patch_labels, included = _patch_labels(labels, x0, y0, patch_size)
        if patch_labels is None:
            return False
        if patch_labels['obboxes'] or keep_empty:
            stem = f'{image_path.stem}_x_{x0}_y_{y0}'
            image_output = image_folder / f'{stem}.png'
            label_output = label_folder / f'{stem}.json'
            if image_output.is_file() and label_output.is_file():
                covered.update(included)
                written += 1
                return True
            image = read_fineair_rgb_ir(
                image_path, window=(x0, y0, patch_size, patch_size),
                stretches=stretches, output_size=(patch_size, patch_size))
            if not np.any(image):
                return False
            if not cv2.imwrite(str(image_output), image):
                raise OSError(f'Could not write FineAir patch {stem}')
            label_output.write_text(json.dumps(patch_labels, separators=(',', ':')))
            covered.update(included)
            written += 1
        return True

    for x0 in xs:
        for y0 in ys:
            write_window(x0, y0)

    # Grid boundaries can intersect an object even with overlap. Add a window
    # centred on every still-uncovered aircraft so no source annotation is
    # silently lost merely to avoid a truncated positive.
    while len(covered) < len(labels['obboxes']):
        index = next(i for i in range(len(labels['obboxes'])) if i not in covered)
        corners = np.asarray(labels['obboxes'][index], dtype=np.float64)
        minimum = corners.min(0)
        maximum = corners.max(0)
        if np.any(maximum - minimum > patch_size):
            raise ValueError(
                f'Aircraft {index} in {label_path} exceeds patch size {patch_size}')
        before = len(covered)
        # Search several valid placements containing the target. A centred
        # window can itself truncate a neighbouring aircraft in dense scenes.
        x_low = max(0, int(np.ceil(maximum[0] - patch_size)))
        x_high = min(int(np.floor(minimum[0])), max(0, x_padded - patch_size))
        y_low = max(0, int(np.ceil(maximum[1] - patch_size)))
        y_high = min(int(np.floor(minimum[1])), max(0, y_padded - patch_size))
        xs_target = np.unique(np.linspace(x_low, x_high, 9, dtype=int))
        ys_target = np.unique(np.linspace(y_low, y_high, 9, dtype=int))
        center = (minimum + maximum) / 2
        preferred_x = int(np.clip(round(center[0] - patch_size / 2), x_low, x_high))
        preferred_y = int(np.clip(round(center[1] - patch_size / 2), y_low, y_high))
        candidates = [(preferred_x, preferred_y)] + [
            (int(x0), int(y0)) for x0 in xs_target for y0 in ys_target]
        for x0, y0 in candidates:
            write_window(x0, y0)
            if index in covered:
                break
        if len(covered) == before:
            raise RuntimeError(
                f'Could not create a non-truncated patch for aircraft {index} '
                f'in {label_path}; try a larger --patch-size')
    return written


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path('/data/FineAir/FR24_dataset'))
    parser.add_argument('--out-folder', type=Path,
                        default=Path('/output/yolo_mtl_patches/fineair/rgbir'))
    parser.add_argument('--patch-size', type=int, default=1024)
    parser.add_argument('--patch-overlap', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--balance-trials', type=int, default=20000)
    parser.add_argument('--keep-empty', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.patch_size <= 0:
        raise ValueError('--patch-size must be positive')
    if not 0 <= args.patch_overlap < args.patch_size:
        raise ValueError('--patch-overlap must be in [0, patch-size)')
    scenes = matched_scenes(args.root)
    splits = balanced_scene_split(scenes, args.seed, trials=args.balance_trials)
    manifest = {}
    for split, selected in splits.items():
        split_counts = Counter()
        for _, label in selected:
            split_counts.update(_scene_counts(label))
        manifest[split] = {
            'scenes': [image.stem for image, _ in selected],
            'fine_class_counts': dict(sorted(split_counts.items())),
        }
        print(f'{split}: {len(selected)} scenes')
        print(f'{split}: fine classes {dict(sorted(split_counts.items()))}')
        if args.dry_run:
            continue
        count = 0
        for image, label in selected:
            count += write_scene(image, label, args.out_folder / split,
                                 args.patch_size, args.patch_overlap,
                                 args.keep_empty)
        print(f'{split}: wrote {count} patches')
    if not args.dry_run:
        payload = {'seed': args.seed, 'splits': manifest}
        (args.out_folder / 'split_manifest.json').write_text(
            json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
