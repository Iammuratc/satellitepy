#!/usr/bin/env python3
"""Match iSAID instance polygons to SatellitePy DOTA patch objects.

The DOTA boxes and iSAID masks describe the same source imagery but use
separate annotation files.  This tool matches instances by source image,
coarse class, and bounding-box IoU, then writes the cropped rasterized mask
coordinates into each patch label.  Writes are atomic so an interrupted run
cannot leave truncated JSON files.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


PATCH_NAME = re.compile(r"^(?P<source>.+)_x_(?P<x>\d+)_y_(?P<y>\d+)$")
TARGET_CLASSES = {
    "plane": "airplane",
    "Large_Vehicle": "vehicle",
    "Small_Vehicle": "vehicle",
    "ship": "ship",
}


def load_isaid(paths: list[Path]):
    instances = defaultdict(list)
    for path in paths:
        data = json.loads(path.read_text())
        images = {item["id"]: Path(item["file_name"]).stem for item in data["images"]}
        categories = {item["id"]: item["name"] for item in data["categories"]}
        for annotation in data["annotations"]:
            coarse_class = TARGET_CLASSES.get(categories[annotation["category_id"]])
            if coarse_class is None or annotation.get("iscrowd", 0):
                continue
            instances[images[annotation["image_id"]]].append({
                "class": coarse_class,
                "bbox": annotation["bbox"],
                "segmentation": annotation["segmentation"],
            })
    return instances


def candidate_arrays(candidates):
    boxes = np.asarray([item["bbox"] for item in candidates], dtype=np.float32).reshape(-1, 4)
    if len(boxes):
        boxes[:, 2:] += boxes[:, :2]
    classes = np.asarray([item["class"] for item in candidates], dtype=object)
    return boxes, classes


def box_ious(box, boxes):
    if not len(boxes):
        return np.empty(0, dtype=np.float32)
    intersection_min = np.maximum(np.asarray(box[:2]), boxes[:, :2])
    intersection_max = np.minimum(np.asarray(box[2:]), boxes[:, 2:])
    intersection_size = np.maximum(0.0, intersection_max - intersection_min)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    area = (box[2] - box[0]) * (box[3] - box[1])
    other_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area + other_area - intersection, 1e-9)


def local_polygons(segmentation, offset_x, offset_y, patch_size):
    polygons = []
    for polygon in segmentation:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        points -= np.asarray([offset_x, offset_y], dtype=np.float32)
        # Discard polygons wholly outside this patch. OpenCV clips the others
        # at rasterization time, so their original boundary stays intact.
        if (points[:, 0].max() < 0 or points[:, 1].max() < 0
                or points[:, 0].min() >= patch_size or points[:, 1].min() >= patch_size):
            continue
        polygons.append(np.rint(points).astype(np.int32).tolist())
    return {"polygons": polygons} if polygons else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-root", type=Path, required=True)
    parser.add_argument("--isaid", type=Path, nargs="+", required=True)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--splits", nargs="+", default=("train", "val", "test"),
        help="Patch splits to process (default: train val test).",
    )
    parser.add_argument(
        "--clear-only", action="store_true",
        help="Remove masks from the selected splits without attempting matches.",
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    instances = load_isaid(args.isaid)
    arrays = {source: candidate_arrays(candidates) for source, candidates in instances.items()}
    totals = defaultdict(int)
    ious = []
    unmatched_examples = []
    label_paths = (
        label_path
        for split in args.splits
        for label_path in sorted((args.patch_root / split / "labels").glob("*.json"))
    )
    for label_path in label_paths:
        match = PATCH_NAME.match(label_path.stem)
        if match is None:
            totals["bad_patch_name"] += 1
            continue
        source = match.group("source")
        offset_x, offset_y = int(match.group("x")), int(match.group("y"))
        labels = json.loads(label_path.read_text())
        masks = [None] * len(labels.get("obboxes", ()))
        if args.clear_only:
            if args.write:
                labels["masks"] = masks
                provided = set(labels.get("_provided_tasks", ()))
                provided.discard("masks")
                labels["_provided_tasks"] = sorted(provided)
                temporary = label_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(labels, separators=(",", ":")))
                temporary.replace(label_path)
            totals["patches"] += 1
            continue
        candidates = instances.get(source, ())
        candidate_boxes, candidate_classes = arrays.get(
            source, (np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=object)))
        used = set()
        for index, (obbox, coarse_class) in enumerate(zip(
                labels.get("obboxes", ()), labels.get("coarse-class", ()))):
            if obbox is None or coarse_class not in TARGET_CLASSES.values():
                continue
            points = np.asarray(obbox, dtype=np.float32)
            points += np.asarray([offset_x, offset_y], dtype=np.float32)
            box = (*points.min(axis=0), *points.max(axis=0))
            scores = box_ious(box, candidate_boxes)
            eligible = candidate_classes == coarse_class
            if used:
                eligible[np.fromiter(used, dtype=int)] = False
            scores[~eligible] = -1.0
            candidate_index = int(scores.argmax()) if len(scores) else -1
            score = float(scores[candidate_index]) if candidate_index >= 0 else 0.0
            candidate = candidates[candidate_index] if candidate_index >= 0 else None
            totals["objects"] += 1
            if score < args.iou_threshold:
                totals["unmatched"] += 1
                if len(unmatched_examples) < 20:
                    unmatched_examples.append((str(label_path), index, coarse_class, score))
                continue
            mask = None
            if args.write:
                mask = local_polygons(candidate["segmentation"], offset_x, offset_y, args.patch_size)
                if mask is None:
                    totals["cropped_empty"] += 1
                    continue
                masks[index] = mask
            used.add(candidate_index)
            ious.append(score)
            totals["matched"] += 1

        if args.write:
            labels["masks"] = masks
            provided = set(labels.get("_provided_tasks", ()))
            if any(mask is not None for mask in masks):
                provided.add("masks")
            else:
                provided.discard("masks")
            labels["_provided_tasks"] = sorted(provided)
            temporary = label_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(labels, separators=(",", ":")))
            temporary.replace(label_path)
        totals["patches"] += 1

    print(dict(totals))
    if ious:
        print("matched_iou", {
            "min": min(ious), "mean": float(np.mean(ious)),
            "median": float(np.median(ious)), "max": max(ious),
        })
    print("unmatched_examples", unmatched_examples)


if __name__ == "__main__":
    main()
