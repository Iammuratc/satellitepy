#!/usr/bin/env python3
"""Create target-prompted aircraft chips for sparse multi-task learning.

Every unique aircraft becomes one sample.  Copies caused by overlapping source
patches are deduplicated in source-scene coordinates, and the copy with the
largest edge margin is used.  The target is centred, aligned by its longest box
edge, and isotropically resized into a square chip.  Other aircraft remain in
the RGB image but are not assigned the target's labels.  Where instance masks
exist (DOTA/iSAID), their pixels are written as an ignore mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


PATCH_NAME = re.compile(r"^(?P<scene>.+)_x_(?P<x>\d+)_y_(?P<y>\d+)$")
DEFAULT_DATASETS = {
    "fineair": Path("/output/yolo_mtl_patches/fineair/rgbir"),
    "rareplanes": Path("/output/yolo_mtl_patches/rareplanes/rgb"),
    "dota": Path("/output/yolo_mtl_patches/dota_aircraft/rgb"),
}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Candidate:
    dataset: str
    split: str
    scene: str
    offset: tuple[int, int]
    image_path: Path
    label_path: Path
    object_index: int
    box: np.ndarray
    global_box: np.ndarray
    edge_margin: float
    labels: dict


def patch_identity(stem: str) -> tuple[str, int, int]:
    match = PATCH_NAME.match(stem)
    if match is None:
        return stem, 0, 0
    return match.group("scene"), int(match.group("x")), int(match.group("y"))


def image_for_label(label_path: Path) -> Path:
    folder = label_path.parent.parent / "images"
    for suffix in (".png", ".tif", ".tiff", ".jpg", ".jpeg"):
        candidate = folder / f"{label_path.stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No image corresponding to {label_path}")


def object_value(value, index: int):
    if isinstance(value, list):
        return value[index] if index < len(value) else None
    if isinstance(value, dict):
        result = {key: object_value(item, index) for key, item in value.items()}
        return {key: item for key, item in result.items() if item is not None}
    return None


def semantic_labels(document: dict, index: int) -> dict:
    excluded = {"hbboxes", "obboxes", "masks", "ignore_obboxes",
                "ignore_coarse_classes", "_provided_geometry",
                "_provided_tasks", "_subset"}
    result = {}
    for key, value in document.items():
        if key in excluded or key.startswith("_"):
            continue
        selected = object_value(value, index)
        if selected not in (None, {}, []):
            result[key] = selected
    return result


def canonical_key(scene: str, box: np.ndarray) -> tuple:
    """Stable identity for the same object copied into overlapping patches."""
    points = np.round(box * 2).astype(np.int64)  # half-pixel tolerance
    ordered = tuple(sorted(map(tuple, points.tolist())))
    return scene, ordered


def collect_candidates(dataset: str, root: Path, split: str) -> list[Candidate]:
    selected = {}
    label_folder = root / split / "labels"
    if not label_folder.is_dir():
        raise FileNotFoundError(label_folder)
    for label_path in sorted(label_folder.glob("*.json")):
        document = json.loads(label_path.read_text())
        image_path = image_for_label(label_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not read {image_path}")
        height, width = image.shape[:2]
        scene, x_offset, y_offset = patch_identity(label_path.stem)
        for index, coordinates in enumerate(document.get("obboxes", [])):
            if coordinates is None:
                continue
            box = np.asarray(coordinates, dtype=np.float32)
            if box.shape != (4, 2):
                continue
            global_box = box + np.asarray([x_offset, y_offset], dtype=np.float32)
            margin = float(min(box[:, 0].min(), box[:, 1].min(),
                               width - box[:, 0].max(), height - box[:, 1].max()))
            candidate = Candidate(
                dataset, split, scene, (x_offset, y_offset), image_path,
                label_path, index, box, global_box, margin,
                semantic_labels(document, index))
            key = canonical_key(scene, global_box)
            if key not in selected or margin > selected[key].edge_margin:
                selected[key] = candidate
    return list(selected.values())


def has_instance_mask(candidate: Candidate) -> bool:
    document = json.loads(candidate.label_path.read_text())
    masks = document.get("masks", [])
    if candidate.object_index >= len(masks):
        return False
    entry = masks[candidate.object_index]
    return isinstance(entry, dict) and bool(entry.get("polygons"))


def collect_dataset_pool(dataset: str, root: Path) -> list[Candidate]:
    """Combine inherited splits and deduplicate again in global coordinates."""
    selected = {}
    for inherited_split in SPLITS:
        for candidate in collect_candidates(dataset, root, inherited_split):
            key = canonical_key(candidate.scene, candidate.global_box)
            if key not in selected or candidate.edge_margin > selected[key].edge_margin:
                selected[key] = candidate
    return list(selected.values())


def scene_resplit(candidates: list[Candidate], seed: int, trials: int,
                  ratios=(0.8, 0.1, 0.1)) -> dict[str, list[Candidate]]:
    """Assign whole scenes while approximating target object-count ratios."""
    by_scene = {}
    for candidate in candidates:
        by_scene.setdefault(candidate.scene, []).append(candidate)
    scenes = sorted(by_scene)
    if len(scenes) < len(SPLITS):
        raise ValueError(f"Need at least {len(SPLITS)} scenes, found {len(scenes)}")
    counts = {scene: len(by_scene[scene]) for scene in scenes}
    total_objects = sum(counts.values())
    target_scene_counts = np.asarray(
        [round(len(scenes) * ratio) for ratio in ratios], dtype=int)
    target_scene_counts[0] += len(scenes) - int(target_scene_counts.sum())
    rng = random.Random(seed)
    best_order = None
    best_score = float("inf")
    for _ in range(trials):
        order = scenes.copy()
        rng.shuffle(order)
        offset = 0
        score = 0.0
        for count, ratio in zip(target_scene_counts, ratios):
            group = order[offset:offset + count]
            offset += count
            observed = sum(counts[scene] for scene in group)
            score += abs(observed / total_objects - ratio)
        if score < best_score:
            best_score, best_order = score, order
    result = {split: [] for split in SPLITS}
    offset = 0
    for split, count in zip(SPLITS, target_scene_counts):
        assigned = best_order[offset:offset + count]
        offset += count
        for scene in assigned:
            result[split].extend(replace(item, split=split)
                                 for item in by_scene[scene])
    return result


def affine_for_box(box: np.ndarray, output_size: int, context_scale: float,
                   minimum_source_size: float) -> tuple[np.ndarray, float, float]:
    edges = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(edges, axis=1)
    long_index = int(np.argmax(lengths))
    direction = edges[long_index] / max(lengths[long_index], 1e-6)
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
    return matrix, source_size, float(np.degrees(np.arctan2(direction[1], direction[0])))


def transform_points(points, matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.transform(array, matrix).reshape(-1, 2)


def fill_polygons(canvas: np.ndarray, mask_entry, matrix: np.ndarray,
                  value: int = 255) -> None:
    if not isinstance(mask_entry, dict):
        return
    for polygon in mask_entry.get("polygons", []):
        transformed = np.rint(transform_points(polygon, matrix)).astype(np.int32)
        if len(transformed) >= 3:
            cv2.fillPoly(canvas, [transformed], value)


def write_candidate(candidate: Candidate, output_root: Path, output_size: int,
                    context_scale: float, minimum_source_size: float) -> dict:
    document = json.loads(candidate.label_path.read_text())
    image = cv2.imread(str(candidate.image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read {candidate.image_path}")
    # The first experiment is deliberately RGB-only. OpenCV stores these as BGR.
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] > 3:
        image = image[:, :, :3]

    matrix, source_size, angle = affine_for_box(
        candidate.box, output_size, context_scale, minimum_source_size)
    chip = cv2.warpAffine(image, matrix, (output_size, output_size),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REFLECT_101)
    prompt = np.zeros((output_size, output_size), dtype=np.uint8)
    transformed_box = transform_points(candidate.box, matrix)
    cv2.fillPoly(prompt, [np.rint(transformed_box).astype(np.int32)], 255)

    target_mask = np.zeros_like(prompt)
    ignore_mask = np.zeros_like(prompt)
    masks = document.get("masks", [])
    for index, mask_entry in enumerate(masks):
        if index == candidate.object_index:
            fill_polygons(target_mask, mask_entry, matrix)
        else:
            fill_polygons(ignore_mask, mask_entry, matrix)
    ignore_mask[target_mask > 0] = 0

    identity = (f"{candidate.dataset}|{candidate.split}|{candidate.scene}|"
                f"{canonical_key(candidate.scene, candidate.global_box)[1]}")
    sample_id = f"{candidate.dataset}_{hashlib.sha1(identity.encode()).hexdigest()[:16]}"
    split_root = output_root / candidate.split
    paths = {
        "image": split_root / "images" / f"{sample_id}.png",
        "prompt": split_root / "prompts" / f"{sample_id}.png",
        "label": split_root / "labels" / f"{sample_id}.json",
    }
    if np.any(target_mask):
        paths["target_mask"] = split_root / "target_masks" / f"{sample_id}.png"
        paths["ignore_mask"] = split_root / "ignore_masks" / f"{sample_id}.png"
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(paths["image"]), chip):
        raise OSError(paths["image"])
    if not cv2.imwrite(str(paths["prompt"]), prompt):
        raise OSError(paths["prompt"])
    if "target_mask" in paths:
        cv2.imwrite(str(paths["target_mask"]), target_mask)
        cv2.imwrite(str(paths["ignore_mask"]), ignore_mask)

    record = {
        "id": sample_id,
        "dataset": candidate.dataset,
        "split": candidate.split,
        "scene": candidate.scene,
        "source_image": str(candidate.image_path),
        "source_label": str(candidate.label_path),
        "source_object_index": candidate.object_index,
        "source_patch_offset": list(candidate.offset),
        "source_obbox": candidate.box.tolist(),
        "global_obbox": candidate.global_box.tolist(),
        "chip_obbox": transformed_box.tolist(),
        "semantic_labels": candidate.labels,
        "has_target_mask": bool(np.any(target_mask)),
        "source_crop_size_px": source_size,
        "source_long_axis_degrees": angle,
        "output_size": output_size,
        "image": str(paths["image"].relative_to(output_root)),
        "prompt": str(paths["prompt"].relative_to(output_root)),
        "target_mask": (str(paths["target_mask"].relative_to(output_root))
                        if "target_mask" in paths else None),
        "ignore_mask": (str(paths["ignore_mask"].relative_to(output_root))
                        if "ignore_mask" in paths else None),
    }
    paths["label"].write_text(json.dumps(record, indent=2) + "\n")
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fineair-root", type=Path, default=DEFAULT_DATASETS["fineair"])
    parser.add_argument("--rareplanes-root", type=Path, default=DEFAULT_DATASETS["rareplanes"])
    parser.add_argument("--dota-root", type=Path, default=DEFAULT_DATASETS["dota"])
    parser.add_argument("--output", type=Path,
                        default=Path("/output/aircraft_semantic_chips/rgb_256"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--context-scale", type=float, default=2.25)
    parser.add_argument("--minimum-source-size", type=float, default=64.0)
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Seed for chip-specific scene-level splits")
    parser.add_argument("--split-trials", type=int, default=10000)
    parser.add_argument("--dota-masked-only", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Build all DOTA splits only from aircraft with real masks")
    parser.add_argument("--max-per-dataset-split", type=int,
                        help="Development-only limit applied after deduplication")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size <= 0 or args.context_scale <= 1 or args.minimum_source_size <= 0:
        raise ValueError("size/minimum-source-size must be positive and context-scale > 1")
    roots = {
        "fineair": args.fineair_root,
        "rareplanes": args.rareplanes_root,
        "dota": args.dota_root,
    }
    manifest = {
        "schema": "target_prompted_aircraft_chips_v1",
        "output_size": args.size,
        "context_scale": args.context_scale,
        "minimum_source_size": args.minimum_source_size,
        "split_seed": args.split_seed,
        "split_policy": "chip-specific scene-level 80/10/10",
        "dota_masked_only": args.dota_masked_only,
        "input_channels": "BGR plus separate binary oriented-box prompt",
        "datasets": {},
    }
    for dataset, root in roots.items():
        manifest["datasets"][dataset] = {}
        pool = collect_dataset_pool(dataset, root)
        if dataset == "dota" and args.dota_masked_only:
            pool = [candidate for candidate in pool if has_instance_mask(candidate)]
        split_candidates = scene_resplit(
            pool, args.split_seed, args.split_trials)
        for split in SPLITS:
            candidates = split_candidates[split]
            candidates.sort(key=lambda item: (item.scene, item.object_index,
                                               str(item.label_path)))
            if args.max_per_dataset_split is not None:
                candidates = candidates[:args.max_per_dataset_split]
            records = [write_candidate(candidate, args.output, args.size,
                                       args.context_scale,
                                       args.minimum_source_size)
                       for candidate in candidates]
            masked = sum(record["has_target_mask"] for record in records)
            manifest["datasets"][dataset][split] = {
                "samples": len(records), "target_masks": masked,
                "scenes": sorted({record["scene"] for record in records}),
            }
            print(f"{dataset}/{split}: {len(records)} chips, {masked} masks")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
