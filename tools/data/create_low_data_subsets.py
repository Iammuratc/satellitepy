#!/usr/bin/env python3
"""Create deterministic, stratified symlink subsets of patch datasets."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def nested_values(label: dict, path: tuple[str, ...]) -> list:
    value = label
    for key in path:
        value = value.get(key, {})
    return value if isinstance(value, list) else []


def stratum(dataset: str, label: dict) -> str:
    if dataset == "fineair":
        values = nested_values(label, ("fine-class",))
        return "fine:" + str(next((v for v in values if v is not None), "__empty__"))
    if dataset == "rareplanes":
        values = nested_values(label, ("role",))
        return "role:" + str(next((v for v in values if v is not None), "__empty__"))
    count = len([box for box in label.get("obboxes", []) if box is not None])
    upper = min(8, 2 ** math.ceil(math.log2(max(1, count))))
    return f"objects:{upper if count else 0}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("fineair", "rareplanes", "dota_aircraft"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")

    images = {path.stem: path for path in (args.source / "images").iterdir() if path.is_file()}
    labels = {path.stem: path for path in (args.source / "labels").glob("*.json")}
    if set(images) != set(labels):
        raise RuntimeError("Image/label stems differ")

    groups: dict[str, list[str]] = defaultdict(list)
    for stem, path in labels.items():
        groups[stratum(args.dataset, json.loads(path.read_text()))].append(stem)

    rng = random.Random(args.seed)
    selected: list[str] = []
    allocation = {}
    for name, stems in sorted(groups.items()):
        rng.shuffle(stems)
        amount = min(len(stems), max(1, round(len(stems) * args.fraction)))
        selected.extend(stems[:amount])
        allocation[name] = {"available": len(stems), "selected": amount}

    for folder in (args.destination / "images", args.destination / "labels"):
        folder.mkdir(parents=True, exist_ok=True)
    for stem in sorted(selected):
        image_link = args.destination / "images" / images[stem].name
        label_link = args.destination / "labels" / labels[stem].name
        if not image_link.exists():
            image_link.symlink_to(images[stem])
        if not label_link.exists():
            label_link.symlink_to(labels[stem])

    manifest = {
        "dataset": args.dataset,
        "source": str(args.source),
        "fraction_requested": args.fraction,
        "seed": args.seed,
        "available": len(images),
        "selected": len(selected),
        "fraction_actual": len(selected) / len(images),
        "strata": allocation,
        "stems": sorted(selected),
    }
    (args.destination / "subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in (
        "dataset", "available", "selected", "fraction_actual", "seed")}, indent=2))


if __name__ == "__main__":
    main()
