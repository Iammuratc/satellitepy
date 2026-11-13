#!/usr/bin/env python3
"""Create nested object-level CGC/mask subsets for mask-effect experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


CONDITIONS = {
    100: (0, 10, 25, 50, 100),
    50: (0, 10, 25, 50),
    25: (0, 10, 25),
    10: (0, 10),
}
OBJECT_LIST_KEYS = {
    "hbboxes", "obboxes", "masks", "coarse-class", "role", "fine-class",
    "very-fine-class", "fineair-class", "difficulty", "source", "dota-class",
}


def stable_digest(seed: int, label_path: Path, index: int):
    return hashlib.sha256(f"{seed}:{label_path}:{index}".encode()).digest()


def slice_labels(labels: dict, selected: list[int], masked: set[int]):
    count = len(labels.get("obboxes", labels.get("hbboxes", ())))
    result = {}
    for key, value in labels.items():
        if key in OBJECT_LIST_KEYS and isinstance(value, list) and len(value) == count:
            result[key] = [value[index] for index in selected]
        elif key == "attributes" and isinstance(value, dict):
            result[key] = {
                group: {name: ([values[index] for index in selected]
                               if isinstance(values, list) and len(values) == count else values)
                        for name, values in group_values.items()}
                for group, group_values in value.items()
            }
        else:
            result[key] = value

    source_masks = labels.get("masks", [None] * count)
    result["masks"] = [source_masks[index] if index in masked else None for index in selected]
    provided = set(labels.get("_provided_tasks", ()))
    if any(mask is not None for mask in result["masks"]):
        provided.add("masks")
    else:
        provided.discard("masks")
    result["_provided_tasks"] = sorted(provided)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Dataset split containing images/ and labels/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = []
    by_class = defaultdict(list)
    image_by_stem = {path.stem: path for path in (args.source / "images").glob("*")}
    for label_path in sorted((args.source / "labels").glob("*.json")):
        labels = json.loads(label_path.read_text())
        count = len(labels.get("obboxes", labels.get("hbboxes", ())))
        classes = labels.get("coarse-class", [None] * count)
        masks = labels.get("masks", [None] * count)
        record_index = len(records)
        records.append((label_path, labels, image_by_stem[label_path.stem]))
        for object_index, coarse_class in enumerate(classes):
            if coarse_class not in {"airplane", "vehicle", "ship"}:
                continue
            by_class[coarse_class].append((
                stable_digest(args.seed, label_path, object_index),
                record_index, object_index, masks[object_index] is not None,
            ))

    ranks = {}
    class_counts = {}
    for coarse_class, objects in by_class.items():
        objects.sort(key=lambda item: item[0])
        class_counts[coarse_class] = len(objects)
        for rank, (_, record_index, object_index, mask_available) in enumerate(objects):
            ranks[(record_index, object_index)] = (coarse_class, rank, mask_available)

    for cgc_percent, mask_percents in CONDITIONS.items():
        selected_limits = {name: round(count * cgc_percent / 100)
                           for name, count in class_counts.items()}
        for mask_percent in mask_percents:
            mask_limits = {name: round(count * mask_percent / 100)
                           for name, count in class_counts.items()}
            condition = args.output / f"cgc{cgc_percent:03d}_mask{mask_percent:03d}" / "train"
            image_folder, label_folder = condition / "images", condition / "labels"
            image_folder.mkdir(parents=True, exist_ok=True)
            label_folder.mkdir(parents=True, exist_ok=True)
            selected_counts = defaultdict(int)
            masked_counts = defaultdict(int)
            patch_count = 0
            for record_index, (label_path, labels, image_path) in enumerate(records):
                count = len(labels.get("obboxes", labels.get("hbboxes", ())))
                selected, masked = [], set()
                for object_index in range(count):
                    ranked = ranks.get((record_index, object_index))
                    if ranked is None:
                        continue
                    coarse_class, rank, mask_available = ranked
                    if rank >= selected_limits[coarse_class]:
                        continue
                    selected.append(object_index)
                    selected_counts[coarse_class] += 1
                    if rank < mask_limits[coarse_class] and mask_available:
                        masked.add(object_index)
                        masked_counts[coarse_class] += 1
                if not selected:
                    continue
                destination_image = image_folder / image_path.name
                if not destination_image.exists():
                    os.link(image_path, destination_image)
                destination_label = label_folder / label_path.name
                temporary = destination_label.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(
                    slice_labels(labels, selected, masked), separators=(",", ":")))
                temporary.replace(destination_label)
                patch_count += 1
            print(condition.parent.name, "patches", patch_count,
                  "objects", dict(selected_counts), "masks", dict(masked_counts))


if __name__ == "__main__":
    main()
