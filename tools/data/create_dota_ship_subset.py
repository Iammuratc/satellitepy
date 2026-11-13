#!/usr/bin/env python3
"""Create a single-class DOTA patch view while retaining other objects as ignores."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path


def filter_aligned(value, selected, object_count):
    """Filter object-aligned lists, including lists nested under attributes."""
    if isinstance(value, list) and len(value) == object_count:
        return [value[index] for index in selected]
    if isinstance(value, dict):
        return {
            key: filter_aligned(child, selected, object_count)
            for key, child in value.items()
        }
    return value


def link_image(source: Path, destination: Path):
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def process_split(source: Path, destination: Path, supervised_class: str, subset_name: str):
    image_source = source / 'images'
    label_source = source / 'labels'
    image_destination = destination / 'images'
    label_destination = destination / 'labels'
    image_destination.mkdir(parents=True, exist_ok=True)
    label_destination.mkdir(parents=True, exist_ok=True)

    images = {path.stem: path for path in image_source.iterdir() if path.is_file()}
    counts = Counter()
    for label_path in sorted(label_source.glob('*.json')):
        labels = json.loads(label_path.read_text())
        boxes = labels.get('obboxes', [])
        classes = labels.get('coarse-class', [])
        if len(classes) != len(boxes):
            raise ValueError(
                f'{label_path}: {len(classes)} coarse classes for {len(boxes)} boxes')
        selected_indices = [
            index for index, value in enumerate(classes) if value == supervised_class
        ]
        if not selected_indices:
            continue
        ignored_indices = [
            index for index in range(len(boxes)) if index not in selected_indices
        ]
        filtered = {
            key: filter_aligned(value, selected_indices, len(boxes))
            for key, value in labels.items()
        }
        filtered['ignore_obboxes'] = [boxes[index] for index in ignored_indices]
        filtered['ignore_coarse_classes'] = [classes[index] for index in ignored_indices]
        filtered['_subset'] = {
            'name': subset_name,
            'source_label': str(label_path),
            'supervised_class': supervised_class,
            'ignored_objects': len(ignored_indices),
        }
        image_path = images.get(label_path.stem)
        if image_path is None:
            raise FileNotFoundError(f'No image matching {label_path}')
        link_image(image_path, image_destination / image_path.name)
        (label_destination / label_path.name).write_text(
            json.dumps(filtered, separators=(',', ':')))
        counts['patches'] += 1
        counts[supervised_class] += len(selected_indices)
        counts['ignored_objects'] += len(ignored_indices)
        counts.update(f'ignored_{classes[index]}' for index in ignored_indices)

    manifest = {
        'source': str(source),
        'selection': f'patches containing at least one {supervised_class}',
        'supervision': f'{supervised_class} obboxes only',
        'other_object_policy': 'retained in ignore_obboxes; requires ignore-aware loss',
        'counts': dict(counts),
    }
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=Path('/output/yolo_mtl_patches/dota/rgb'))
    parser.add_argument('--destination', type=Path,
                        default=Path('/output/yolo_mtl_patches/dota_ships/rgb'))
    parser.add_argument('--supervised-class', default='ship')
    parser.add_argument('--subset-name', default='dota-ships')
    args = parser.parse_args()
    summary = {}
    for split in ('train', 'val', 'test'):
        summary[split] = process_split(
            args.source / split, args.destination / split,
            args.supervised_class, args.subset_name)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
