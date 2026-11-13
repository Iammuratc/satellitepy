"""Convert merged BBAVector original-image JSON results to DOTA Task1 files."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from shapely import make_valid
from shapely.geometry import Polygon
from shapely.strtree import STRtree

DOTA_CLASSES = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter',
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-folder', type=Path, required=True,
                        help='Folder containing one merged prediction JSON per original test image')
    parser.add_argument('--out-folder', type=Path, required=True)
    parser.add_argument('--score-threshold', type=float, default=0.01)
    parser.add_argument('--nms-threshold', type=float, default=0.1)
    parser.add_argument('--num-shards', type=int, default=1,
                        help='Split original images deterministically across this many exporter processes.')
    parser.add_argument('--shard-index', type=int, default=0,
                        help='Zero-based exporter shard handled by this process.')
    return parser.parse_args()


PATCH_NAME = re.compile(
    r'^(?P<scale>0\.5|1)__(?P<image>.+)_x_(?P<x>\d+)_y_(?P<y>\d+)$')


def polygon_nms(detections, threshold):
    detections.sort(key=lambda item: item[0], reverse=True)
    if not detections:
        return []
    polygons = []
    for _, box in detections:
        polygon = Polygon(box)
        polygons.append(make_valid(polygon) if not polygon.is_valid else polygon)
    tree = STRtree(polygons)
    suppressed = np.zeros(len(detections), dtype=bool)
    kept = []
    for index, best in enumerate(detections):
        if suppressed[index]:
            continue
        kept.append(best)
        best_polygon = polygons[index]
        # STRtree restricts exact polygon comparisons to intersecting spatial
        # envelopes. Because detections are score-sorted, only later indices
        # can be suppressed by the current candidate.
        for candidate_index in tree.query(best_polygon):
            candidate_index = int(candidate_index)
            if candidate_index <= index or suppressed[candidate_index]:
                continue
            polygon = polygons[candidate_index]
            intersection = best_polygon.intersection(polygon).area
            if not intersection:
                continue
            union = best_polygon.area + polygon.area - intersection
            if union and intersection / union > threshold:
                suppressed[candidate_index] = True
    return kept


def main():
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Require num_shards >= 1 and 0 <= shard_index < num_shards')
    args.out_folder.mkdir(parents=True, exist_ok=True)
    class_to_index = {name: index for index, name in enumerate(DOTA_CLASSES)}
    index_to_class = {value: key for key, value in class_to_index.items()}
    # Group paths first, then read and merge one original image at a time. A
    # low threshold with K=1000 can otherwise retain tens of millions of patch
    # candidates and exhaust RAM before NMS starts.
    image_paths = defaultdict(list)
    for result_path in sorted(args.result_folder.glob('*.json')):
        match = PATCH_NAME.match(result_path.stem)
        image_name = match.group('image') if match else result_path.stem
        image_paths[image_name].append((result_path, match))

    handles = {name: (args.out_folder / f'Task1_{name}.txt').open('w')
               for name in class_to_index}
    try:
        selected_images = sorted(image_paths)[args.shard_index::args.num_shards]
        for image_name in selected_images:
            detections = defaultdict(list)
            for result_path, match in image_paths[image_name]:
                result = json.loads(result_path.read_text())['det_labels']
                probabilities = np.asarray(result.get('dota-class', []), dtype=float)
                boxes = np.asarray(result.get('obboxes', []), dtype=float)
                if not len(probabilities):
                    continue
                if match:
                    scale = float(match.group('scale'))
                    boxes += np.asarray([float(match.group('x')), float(match.group('y'))])
                    boxes /= scale
                indices = probabilities.argmax(axis=1)
                scores = probabilities.max(axis=1)
                for box, index, score in zip(boxes, indices, scores):
                    if score >= args.score_threshold:
                        detections[index_to_class[int(index)]].append(
                            (float(score), box.tolist()))
            for name, candidates in detections.items():
                for score, box in polygon_nms(candidates, args.nms_threshold):
                    coords = ' '.join(f'{coordinate:.2f}' for point in box for coordinate in point)
                    handles[name].write(f'{image_name} {score:.8f} {coords}\n')
    finally:
        for handle in handles.values():
            handle.close()


if __name__ == '__main__':
    main()
