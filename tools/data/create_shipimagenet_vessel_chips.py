#!/usr/bin/env python3
"""Create complete, target-prompted vessel chips from ShipImageNet.

The official training split remains training data. The labelled official
validation images are divided into balanced validation and test subsets. Each
non-truncated vessel is aligned to its longest oriented-box edge and scaled
isotropically into a square chip, with an enforced margin around every vertex.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from satellitepy.data.utils import get_shipnet_categories


@dataclass(frozen=True)
class Vessel:
    source_split: str
    scene: str
    image_path: Path
    label_path: Path
    object_index: int
    box: np.ndarray
    role: str | None
    fine_class: str | None
    very_fine_class: str | None


def text_int(element: ET.Element, key: str) -> int:
    value = element.findtext(key)
    if value is None:
        raise ValueError(f"Missing {key}")
    return int(value)


def polygon(element: ET.Element) -> np.ndarray:
    node = element.find("polygon")
    if node is None:
        raise ValueError("Missing vessel polygon")
    return np.asarray([[float(node.findtext(f"x{i}")),
                        float(node.findtext(f"y{i}"))]
                       for i in range(1, 5)], dtype=np.float32)


def labels_for(element: ET.Element) -> tuple[str | None, str | None, str | None]:
    role = {1: None, 2: "Warship", 3: "Merchant Ship"}.get(
        text_int(element, "level_1"))
    fine = get_shipnet_categories()[text_int(element, "level_2")]
    if fine in {"Other Ship", "Other Warship", "Other Merchant"}:
        fine = None
    name = element.findtext("name")
    very_fine = name if fine is not None and name != fine else None
    return role, fine, very_fine


def read_scene(split_root: Path, source_split: str, label_path: Path,
               excluded: Counter) -> list[Vessel]:
    root = ET.parse(label_path).getroot()
    filename = root.findtext("filename") or f"{label_path.stem}.bmp"
    image_path = split_root / source_split / "images" / filename
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    vessels = []
    for index, element in enumerate(root.findall("object")):
        if text_int(element, "level_0") != 1:
            excluded["non_vessel"] += 1
            continue
        if element.findtext("truncted", "0") == "1":
            excluded["source_truncated"] += 1
            continue
        box = polygon(element)
        if not np.isfinite(box).all() or cv2.contourArea(box) <= 0:
            excluded["invalid_polygon"] += 1
            continue
        role, fine, very_fine = labels_for(element)
        vessels.append(Vessel(source_split, label_path.stem, image_path,
                              label_path, index, box, role, fine, very_fine))
    return vessels


def collect(split_root: Path, source_split: str,
            excluded: Counter) -> dict[str, list[Vessel]]:
    result = {}
    folder = split_root / source_split / "labels"
    for label_path in sorted(folder.glob("*.xml")):
        result[label_path.stem] = read_scene(
            split_root, source_split, label_path, excluded)
    return result


def scene_signature(vessels: list[Vessel]) -> Counter:
    return Counter(v.fine_class or v.role or "unlabelled" for v in vessels)


def balanced_halves(by_scene: dict[str, list[Vessel]], seed: int,
                    trials: int) -> tuple[set[str], set[str]]:
    scenes = sorted(by_scene)
    target_size = len(scenes) // 2
    total = sum((scene_signature(by_scene[s]) for s in scenes), Counter())
    rng = random.Random(seed)
    best, best_score = None, float("inf")
    for _ in range(trials):
        rng.shuffle(scenes)
        left = scenes[:target_size]
        observed = sum((scene_signature(by_scene[s]) for s in left), Counter())
        score = sum(abs(observed[key] / max(total[key], 1) - 0.5)
                    for key in total)
        if score < best_score:
            best, best_score = left.copy(), score
    validation = set(best)
    return validation, set(by_scene) - validation


def affine_for_box(box: np.ndarray, output_size: int, context_scale: float,
                   minimum_source_size: float) -> tuple[np.ndarray, float, float]:
    edges = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(edges, axis=1)
    direction = edges[int(np.argmax(lengths))] / max(float(lengths.max()), 1e-6)
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    center = box.mean(axis=0)
    source_size = max(minimum_source_size, context_scale * float(lengths.max()))
    scale = output_size / source_size
    matrix = np.asarray([
        [scale * direction[0], scale * direction[1],
         output_size / 2 - scale * np.dot(direction, center)],
        [scale * normal[0], scale * normal[1],
         output_size / 2 - scale * np.dot(normal, center)],
    ], dtype=np.float32)
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    return matrix, source_size, angle


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.transform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)


def write_vessel(vessel: Vessel, split: str, output: Path, size: int,
                 context_scale: float, minimum_source_size: float,
                 required_margin: float) -> dict:
    image = cv2.imread(str(vessel.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {vessel.image_path}")
    matrix, source_size, angle = affine_for_box(
        vessel.box, size, context_scale, minimum_source_size)
    chip_box = transform(vessel.box, matrix)
    actual_margin = float(min(chip_box[:, 0].min(), chip_box[:, 1].min(),
                              size - 1 - chip_box[:, 0].max(),
                              size - 1 - chip_box[:, 1].max()))
    if actual_margin < required_margin:
        raise AssertionError(
            f"Vessel {vessel.scene}/{vessel.object_index} margin "
            f"{actual_margin:.2f} < {required_margin:.2f}")
    chip = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REFLECT_101)
    prompt = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(prompt, [np.rint(chip_box).astype(np.int32)], 255)

    identity = f"{vessel.scene}|{vessel.object_index}|{vessel.box.tolist()}"
    sample_id = "shipimagenet_" + hashlib.sha1(identity.encode()).hexdigest()[:16]
    base = output / split
    image_path = base / "images" / f"{sample_id}.png"
    prompt_path = base / "prompts" / f"{sample_id}.png"
    label_path = base / "labels" / f"{sample_id}.json"
    for path in (image_path, prompt_path, label_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), chip) or not cv2.imwrite(str(prompt_path), prompt):
        raise OSError(f"Could not write {sample_id}")
    semantic = {key: value for key, value in {
        "role": vessel.role,
        "fine-class": vessel.fine_class,
        "very-fine-class": vessel.very_fine_class,
    }.items() if value is not None}
    record = {
        "id": sample_id, "dataset": "shipimagenet", "split": split,
        "scene": vessel.scene, "source_split": vessel.source_split,
        "source_image": str(vessel.image_path), "source_label": str(vessel.label_path),
        "source_object_index": vessel.object_index,
        "source_obbox": vessel.box.tolist(), "chip_obbox": chip_box.tolist(),
        "semantic_labels": semantic, "source_truncated": False,
        "source_crop_size_px": source_size, "source_long_axis_degrees": angle,
        "chip_box_margin_px": actual_margin, "output_size": size,
        "image": str(image_path.relative_to(output)),
        "prompt": str(prompt_path.relative_to(output)),
        "target_mask": None, "ignore_mask": None,
    }
    label_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/data/ship_image_net/split"))
    parser.add_argument("--output", type=Path, default=Path(
        "/output/vessel_semantic_chips/shipimagenet_rgb_256_split_seed42"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--context-scale", type=float, default=1.5)
    parser.add_argument("--minimum-source-size", type=float, default=48.0)
    parser.add_argument("--required-margin", type=float, default=8.0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-trials", type=int, default=10000)
    parser.add_argument("--max-per-split", type=int,
                        help="Development-only limit after scene assignment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size <= 0 or args.context_scale <= 1 or args.required_margin < 0:
        raise ValueError("size must be positive, context-scale > 1, margin non-negative")
    excluded = Counter()
    train_scenes = collect(args.input, "train", excluded)
    heldout_scenes = collect(args.input, "val", excluded)
    validation_scenes, test_scenes = balanced_halves(
        heldout_scenes, args.split_seed, args.split_trials)
    assignments = {
        "train": set(train_scenes), "val": validation_scenes, "test": test_scenes,
    }
    pools = train_scenes | heldout_scenes
    manifest = {
        "schema": "target_prompted_shipimagenet_vessel_chips_v1",
        "output_size": args.size, "context_scale": args.context_scale,
        "minimum_source_size": args.minimum_source_size,
        "required_chip_margin_px": args.required_margin,
        "split_seed": args.split_seed,
        "split_policy": "official train; class-balanced half of official val for val/test",
        "source_truncated_policy": "excluded",
        "excluded": dict(excluded), "splits": {},
    }
    for split in ("train", "val", "test"):
        vessels = [v for scene in sorted(assignments[split]) for v in pools[scene]]
        if args.max_per_split is not None:
            vessels = vessels[:args.max_per_split]
        records = [write_vessel(v, split, args.output, args.size,
                                args.context_scale, args.minimum_source_size,
                                args.required_margin) for v in vessels]
        margins = [record["chip_box_margin_px"] for record in records]
        manifest["splits"][split] = {
            "scenes": len(assignments[split]), "samples": len(records),
            "minimum_observed_chip_margin_px": min(margins) if margins else None,
            "fine_classes": len({r["semantic_labels"].get("fine-class")
                                 for r in records
                                 if r["semantic_labels"].get("fine-class")}),
            "very_fine_classes": len({r["semantic_labels"].get("very-fine-class")
                                      for r in records
                                      if r["semantic_labels"].get("very-fine-class")}),
        }
        print(f"{split}: {len(records)} vessels from {len(assignments[split])} scenes")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
