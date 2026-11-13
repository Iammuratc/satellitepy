#!/usr/bin/env python3
"""Create DOTA-aircraft patches with coarse native imagery upscaled to a GSD cap.

Images already finer than the requested target are left unchanged. DOTA headers
below ``--metric-gsd-min`` are treated as angular/non-metric metadata and are
also left unchanged. Resampling happens before patching, so enlarged aircraft
near native patch boundaries are not clipped or duplicated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon, box

from satellitepy.data.labels import read_label
from satellitepy.data.patch import get_pad_size, get_patch_start_coords


IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dota-root', type=Path, default=Path('/data/DOTA'))
    parser.add_argument('--dota-v2-root', type=Path,
                        default=Path('/output/dota_v2_annotations'))
    parser.add_argument('--out-folder', type=Path,
                        default=Path('/output/yolo_mtl_patches_gsd030/dota_aircraft/rgb'))
    parser.add_argument('--target-gsd', type=float, default=0.30)
    parser.add_argument('--metric-gsd-min', type=float, default=0.05)
    parser.add_argument('--patch-size', type=int, default=1024)
    parser.add_argument('--patch-overlap', type=int, default=100)
    parser.add_argument('--truncated-object-thr', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def native_sources(dota_root: Path, dota_v2_root: Path):
    result = []
    for official_split in ('train', 'val'):
        images = {
            path.stem: path
            for suffix in IMAGE_SUFFIXES
            for path in (dota_root / official_split / 'images').glob(f'*{suffix}')
        }
        labels = dota_v2_root / official_split / official_split / 'annotations' / 'version2.0'
        for label in sorted(labels.glob('*.txt')):
            image = images.get(label.stem)
            native_label = dota_root / official_split / 'bounding_boxes' / label.name
            if image is not None and native_label.exists():
                result.append((image, label, native_label))
    return result


def split_sources(sources, seed: int):
    ordered = sorted(sources, key=lambda sample: hashlib.sha256(
        f'{seed}:{sample[0].stem}'.encode()).digest())
    train_end = round(len(ordered) * 0.8)
    val_end = train_end + round(len(ordered) * 0.1)
    return {
        'train': ordered[:train_end],
        'val': ordered[train_end:val_end],
        'test': ordered[val_end:],
    }


def read_gsd(path: Path):
    for line in path.read_text(errors='ignore').splitlines()[:3]:
        if line.lower().startswith('gsd:'):
            try:
                return float(line.split(':', 1)[1].strip())
            except ValueError:
                return None
    return None


def has_aircraft(path: Path):
    return any(len(fields := line.split()) >= 9 and fields[-2 if len(fields) == 10 else -1] == 'plane'
               for line in path.read_text(errors='ignore').splitlines())


def filter_aligned(value, selected, object_count):
    if isinstance(value, list) and len(value) == object_count:
        return [value[index] for index in selected]
    if isinstance(value, dict):
        return {key: filter_aligned(child, selected, object_count)
                for key, child in value.items()}
    return value


def patch_labels(labels, patch_polygon, x0, y0, threshold):
    boxes = labels['obboxes']
    selected = []
    for index, corners in enumerate(boxes):
        if corners is None:
            continue
        polygon = Polygon(corners)
        if polygon.area and polygon.intersection(patch_polygon).area / polygon.area > threshold:
            selected.append(index)
    if not selected or not any(labels['coarse-class'][index] == 'airplane'
                               for index in selected):
        return None
    result = filter_aligned(labels, selected, len(boxes))
    for corners in result['obboxes']:
        if corners is not None:
            for point in corners:
                point[0] -= x0
                point[1] -= y0
    return result


def process_source(image_path, label_path, native_label_path, destination,
                   target_gsd, metric_gsd_min, patch_size, overlap, threshold):
    native_gsd = read_gsd(native_label_path)
    metric = native_gsd is not None and native_gsd >= metric_gsd_min
    scale = max(1.0, native_gsd / target_gsd) if metric else 1.0
    labels = read_label(label_path, 'dota', rescaling=scale)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Could not read {image_path}')
    if scale != 1.0:
        width = round(image.shape[1] * scale)
        height = round(image.shape[0] * scale)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    image_folder = destination / 'images'
    label_folder = destination / 'labels'
    image_folder.mkdir(parents=True, exist_ok=True)
    label_folder.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[:2]
    padded_width = width + get_pad_size(width, patch_size, overlap)
    padded_height = height + get_pad_size(height, patch_size, overlap)
    xs = get_patch_start_coords(padded_width, patch_size, overlap)
    ys = get_patch_start_coords(padded_height, patch_size, overlap)
    counts = Counter(source_images=1, aircraft_objects=sum(
        value == 'airplane' for value in labels['coarse-class']))
    for x0 in xs:
        for y0 in ys:
            selected = patch_labels(labels, box(x0, y0, x0 + patch_size, y0 + patch_size),
                                    x0, y0, threshold)
            if selected is None:
                continue
            crop = np.zeros((patch_size, patch_size, 3), dtype=image.dtype)
            x1, y1 = min(x0 + patch_size, width), min(y0 + patch_size, height)
            crop[:y1 - y0, :x1 - x0] = image[y0:y1, x0:x1]
            name = f'{image_path.stem}_x_{x0}_y_{y0}'
            if not cv2.imwrite(str(image_folder / f'{name}.png'), crop):
                raise OSError(f'Failed to write patch {name}')
            boxes = selected['obboxes']
            classes = selected['coarse-class']
            aircraft = [i for i, value in enumerate(classes) if value == 'airplane']
            ignored = [i for i in range(len(boxes)) if i not in aircraft]
            filtered = filter_aligned(selected, aircraft, len(boxes))
            filtered['ignore_obboxes'] = [boxes[i] for i in ignored]
            filtered['ignore_coarse_classes'] = [classes[i] for i in ignored]
            filtered['_subset'] = {
                'name': 'dota-aircraft-gsd030',
                'source_label': str(label_path),
                'native_gsd': native_gsd,
                'target_gsd': target_gsd,
                'scale': scale,
                'interpolation': 'bicubic',
                'supervised_class': 'airplane',
                'ignored_objects': len(ignored),
            }
            (label_folder / f'{name}.json').write_text(
                json.dumps(filtered, separators=(',', ':')))
            counts['patches'] += 1
            counts['supervised_aircraft'] += len(aircraft)
            counts['ignored_objects'] += len(ignored)
    counts['upscaled_source_images'] = int(scale > 1.0)
    counts['native_gsd_nonmetric_or_missing'] = int(not metric)
    return counts, native_gsd, scale


def main():
    args = parse_args()
    if args.target_gsd <= 0 or args.metric_gsd_min <= 0:
        raise ValueError('GSD values must be positive')
    sources = native_sources(args.dota_root, args.dota_v2_root)
    splits = split_sources(sources, args.seed)
    summary = {
        'target_gsd_m': args.target_gsd,
        'policy': 'upscale only: scale=max(1,native_gsd/target_gsd)',
        'nonmetric_policy': f'gsd < {args.metric_gsd_min} or missing is left unchanged',
        'interpolation': 'OpenCV INTER_CUBIC; no additional blur',
        'split_seed': args.seed,
        'splits': {},
    }
    for split, samples in splits.items():
        destination = args.out_folder / split
        completed_path = destination / 'completed_sources.txt'
        destination.mkdir(parents=True, exist_ok=True)
        completed = set(completed_path.read_text().splitlines()) if completed_path.exists() else set()
        totals = Counter()
        scales = []
        aircraft_samples = [sample for sample in samples if has_aircraft(sample[1])]
        for image, label, native_label in aircraft_samples:
            if image.stem in completed:
                continue
            counts, gsd, scale = process_source(
                image, label, native_label, destination, args.target_gsd,
                args.metric_gsd_min, args.patch_size, args.patch_overlap,
                args.truncated_object_thr)
            totals.update(counts)
            scales.append({'image': image.stem, 'native_gsd': gsd, 'scale': scale})
            with completed_path.open('a') as handle:
                handle.write(f'{image.stem}\n')
        manifest = {
            'eligible_source_images': len(aircraft_samples),
            'new_counts': dict(totals),
            'new_scales': scales,
            'total_patches_on_disk': len(list((destination / 'labels').glob('*.json'))),
        }
        (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        summary['splits'][split] = manifest
        print(split, json.dumps(manifest, sort_keys=True), flush=True)
    (args.out_folder / 'manifest.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
