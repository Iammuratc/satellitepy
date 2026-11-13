#!/usr/bin/env python3
"""Create complete target-prompted vessel chips from official FAIR1M XMLs.

Source images are clustered by perceptual similarity before a class-balanced
scene-group split. Chips are generated only after the split and incomplete
source-boundary vessels are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


VESSEL_CLASSES = {
    "Passenger Ship", "Motorboat", "Fishing Boat", "Tugboat", "other-ship",
    "Engineering Ship", "Liquid Cargo Ship", "Dry Cargo Ship", "Warship",
}


@dataclass(frozen=True)
class Vessel:
    object_index: int
    box: np.ndarray
    fine_class: str


@dataclass
class Scene:
    name: str
    image_path: Path
    label_path: Path
    vessels: list[Vessel]
    perceptual_hash: int = 0


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def parse_box(element: ET.Element) -> np.ndarray:
    values = []
    for point in element.findall("./points/point")[:4]:
        x, y = point.text.split(",")[:2]
        values.append([float(x), float(y)])
    return np.asarray(values, dtype=np.float32)


def read_scenes(image_root: Path, label_root: Path) -> list[Scene]:
    scenes = []
    for label_path in sorted(label_root.glob("*.xml"), key=lambda p: int(p.stem)):
        root = ET.parse(label_path).getroot()
        filename = root.findtext("./source/filename") or f"{label_path.stem}.tif"
        image_path = image_root / filename
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        vessels = []
        for index, element in enumerate(root.findall("./objects/object")):
            category = element.findtext("./possibleresult/name")
            if category not in VESSEL_CLASSES:
                continue
            box = parse_box(element)
            if box.shape == (4, 2) and np.isfinite(box).all() and cv2.contourArea(box) > 0:
                vessels.append(Vessel(index, box, category))
        scenes.append(Scene(label_path.stem, image_path, label_path, vessels))
    return scenes


def difference_hash(image_path: Path) -> int:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unreadable image: {image_path}")
    small = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def duplicate_groups(scenes: list[Scene], max_hamming: int) -> list[list[int]]:
    for index, scene in enumerate(scenes):
        scene.perceptual_hash = difference_hash(scene.image_path)
        if (index + 1) % 100 == 0:
            print(f"hashed {index + 1}/{len(scenes)} scenes", flush=True)
    union = UnionFind(len(scenes))
    for left in range(len(scenes)):
        for right in range(left + 1, len(scenes)):
            if (scenes[left].perceptual_hash ^ scenes[right].perceptual_hash).bit_count() <= max_hamming:
                union.union(left, right)
    grouped = {}
    for index in range(len(scenes)):
        grouped.setdefault(union.find(index), []).append(index)
    return list(grouped.values())


def group_counts(group: list[int], scenes: list[Scene]) -> Counter:
    return Counter(v.fine_class for index in group for v in scenes[index].vessels)


def balanced_split(groups: list[list[int]], scenes: list[Scene], seed: int,
                   trials: int) -> dict[str, list[int]]:
    names, ratios = ("train", "val", "test"), (0.8, 0.1, 0.1)
    total_scenes = len(scenes)
    total_classes = sum((group_counts(g, scenes) for g in groups), Counter())
    rng = random.Random(seed)
    best_assignment, best_score = None, float("inf")
    for _ in range(trials):
        order = groups.copy(); rng.shuffle(order)
        assigned = {name: [] for name in names}
        scene_counts = Counter()
        for group in order:
            deficits = {name: ratios[i] * total_scenes - scene_counts[name]
                        for i, name in enumerate(names)}
            target = max(names, key=lambda name: deficits[name])
            assigned[target].extend(group); scene_counts[target] += len(group)
        score = 0.0
        for ratio, name in zip(ratios, names):
            observed = Counter(v.fine_class for i in assigned[name]
                               for v in scenes[i].vessels)
            score += abs(len(assigned[name]) / total_scenes - ratio)
            score += sum(abs(observed[key] / max(total_classes[key], 1) - ratio)
                         for key in total_classes)
            if name != "train":
                score += 10 * sum(total_classes[key] >= 10 and observed[key] == 0
                                  for key in total_classes)
        if score < best_score:
            best_assignment, best_score = assigned, score
    return best_assignment


def affine_for_box(box: np.ndarray, size: int, context_scale: float,
                   minimum_source_size: float) -> tuple[np.ndarray, float, float]:
    edges = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(edges, axis=1)
    direction = edges[int(np.argmax(lengths))] / max(float(lengths.max()), 1e-6)
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    center = box.mean(axis=0)
    source_size = max(minimum_source_size, context_scale * float(lengths.max()))
    scale = size / source_size
    matrix = np.asarray([
        [scale * direction[0], scale * direction[1],
         size / 2 - scale * np.dot(direction, center)],
        [scale * normal[0], scale * normal[1],
         size / 2 - scale * np.dot(normal, center)],
    ], dtype=np.float32)
    return matrix, source_size, float(np.degrees(np.arctan2(direction[1], direction[0])))


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.transform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)


def write_scene(scene: Scene, split: str, output: Path, size: int,
                context_scale: float, minimum_source_size: float,
                source_margin: float, chip_margin: float,
                exclusions: Counter) -> list[dict]:
    image = cv2.imread(str(scene.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(scene.image_path)
    height, width = image.shape[:2]
    records = []
    for vessel in scene.vessels:
        box = vessel.box
        boundary = min(float(box[:, 0].min()), float(box[:, 1].min()),
                       width - 1 - float(box[:, 0].max()),
                       height - 1 - float(box[:, 1].max()))
        if boundary < source_margin:
            exclusions["source_boundary_or_truncated"] += 1
            continue
        matrix, source_size, angle = affine_for_box(
            box, size, context_scale, minimum_source_size)
        chip_box = transform(box, matrix)
        actual_margin = min(float(chip_box[:, 0].min()), float(chip_box[:, 1].min()),
                            size - 1 - float(chip_box[:, 0].max()),
                            size - 1 - float(chip_box[:, 1].max()))
        if actual_margin < chip_margin:
            raise AssertionError(f"{scene.name}/{vessel.object_index}: {actual_margin}")
        chip = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REFLECT_101)
        prompt = np.zeros((size, size), dtype=np.uint8)
        cv2.fillPoly(prompt, [np.rint(chip_box).astype(np.int32)], 255)
        identity = f"{scene.name}|{vessel.object_index}|{box.tolist()}"
        sample_id = "fair1m_" + hashlib.sha1(identity.encode()).hexdigest()[:16]
        base = output / split
        image_path = base / "images" / f"{sample_id}.png"
        prompt_path = base / "prompts" / f"{sample_id}.png"
        label_path = base / "labels" / f"{sample_id}.json"
        for path in (image_path, prompt_path, label_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_path), chip) or not cv2.imwrite(str(prompt_path), prompt):
            raise OSError(sample_id)
        role = None if vessel.fine_class == "other-ship" else (
            "Warship" if vessel.fine_class == "Warship" else "Merchant Ship")
        semantic = {"fine-class": vessel.fine_class}
        if role is not None:
            semantic["role"] = role
        record = {
            "id": sample_id, "dataset": "fair1m", "split": split,
            "scene": scene.name, "source_image": str(scene.image_path),
            "source_label": str(scene.label_path),
            "source_object_index": vessel.object_index,
            "source_obbox": box.tolist(), "chip_obbox": chip_box.tolist(),
            "semantic_labels": semantic, "source_boundary_margin_px": boundary,
            "source_crop_size_px": source_size, "source_long_axis_degrees": angle,
            "chip_box_margin_px": actual_margin, "output_size": size,
            "image": str(image_path.relative_to(output)),
            "prompt": str(prompt_path.relative_to(output)),
            "target_mask": None, "ignore_mask": None,
        }
        label_path.write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path,
                        default=Path("/data/fair1m/train/images"))
    parser.add_argument("--labels", type=Path, default=Path(
        "/data/fair1m_official/repository/data/labelXmls"))
    parser.add_argument("--output", type=Path, default=Path(
        "/output/vessel_semantic_chips/fair1m_official_rgb_256_split_seed42"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--context-scale", type=float, default=1.5)
    parser.add_argument("--minimum-source-size", type=float, default=48.0)
    parser.add_argument("--source-margin", type=float, default=2.0)
    parser.add_argument("--chip-margin", type=float, default=8.0)
    parser.add_argument("--duplicate-hamming", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-trials", type=int, default=10000)
    parser.add_argument("--max-scenes", type=int,
                        help="Development-only scene limit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes = read_scenes(args.images, args.labels)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    groups = duplicate_groups(scenes, args.duplicate_hamming)
    assignment = balanced_split(groups, scenes, args.split_seed, args.split_trials)
    exclusions = Counter()
    manifest = {
        "schema": "target_prompted_fair1m_vessel_chips_v1",
        "official_labels": str(args.labels), "source_images": str(args.images),
        "output_size": args.size, "context_scale": args.context_scale,
        "minimum_source_size": args.minimum_source_size,
        "source_margin_px": args.source_margin, "required_chip_margin_px": args.chip_margin,
        "split_seed": args.split_seed,
        "split_policy": "source-scene split after perceptual near-duplicate clustering",
        "duplicate_hamming_threshold": args.duplicate_hamming,
        "source_scenes": len(scenes), "duplicate_groups": len(groups), "splits": {},
    }
    for split in ("train", "val", "test"):
        records = []
        for scene_index in sorted(assignment[split], key=lambda i: int(scenes[i].name)):
            records.extend(write_scene(scenes[scene_index], split, args.output,
                                       args.size, args.context_scale,
                                       args.minimum_source_size, args.source_margin,
                                       args.chip_margin, exclusions))
        counts = Counter(r["semantic_labels"]["fine-class"] for r in records)
        manifest["splits"][split] = {
            "scenes": len(assignment[split]), "samples": len(records),
            "fine_class_counts": dict(sorted(counts.items())),
            "minimum_chip_margin_px": min((r["chip_box_margin_px"] for r in records),
                                           default=None),
        }
        print(f"{split}: {len(records)} vessels from {len(assignment[split])} scenes",
              flush=True)
    manifest["excluded"] = dict(exclusions)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
