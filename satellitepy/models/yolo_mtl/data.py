"""Dataset and collation for sparse object-level multi-task supervision."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from ultralytics.utils.ops import xyxyxyxy2xywhr

from satellitepy.data.utils import get_satellitepy_dict_values, get_task_dict


def _object_count(labels: dict, tasks: Iterable[str]) -> int:
    values = [labels.get('hbboxes', ()), labels.get('obboxes', ())]
    values.extend(get_satellitepy_dict_values(labels, task) for task in tasks)
    return max((len(value) for value in values), default=0)


def _at(values, index):
    return values[index] if index < len(values) else None


def _reference_box(labels: dict, index: int):
    """Return supplied geometry for locating an object's semantic targets."""
    obbox = _at(labels.get('obboxes', ()), index)
    if obbox is not None:
        return obbox
    return _at(labels.get('hbboxes', ()), index)


class SparseMTLDataset(Dataset):
    """Read SatellitePy patch JSON without fabricating missing targets."""

    def __init__(
        self,
        image_folder: Path,
        label_folder: Path,
        tasks: Iterable[str],
        image_size: int = 640,
        input_channels: int = 4,
        ir_dropout: float = 0.25,
        training: bool = True,
        geometry_family: str | None = None,
        geometry_domain: str | None = None,
    ):
        self.image_folder = Path(image_folder)
        self.label_folder = Path(label_folder)
        labels = {path.stem: path for path in self.label_folder.glob('*.json')}
        self.items = [(path, labels[path.stem]) for path in sorted(self.image_folder.glob('*'))
                      if path.is_file() and path.stem in labels]
        if not self.items:
            raise ValueError(f'No matching image/label patches in {image_folder} and {label_folder}')
        self.tasks = tuple(tasks)
        self.object_tasks = tuple(task for task in self.tasks if task != 'masks')
        self.task_dicts = {task: get_task_dict(task) for task in self.object_tasks}
        self.image_size = image_size
        self.input_channels = input_channels
        self.ir_dropout = ir_dropout
        self.training = training
        self.geometry_family = geometry_family
        self.geometry_domain = geometry_domain
        if input_channels not in (3, 4):
            raise ValueError('input_channels must be 3 or 4')

    def __len__(self):
        return len(self.items)

    def _read_image(self, path: Path) -> torch.Tensor:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f'Could not read {path}')
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        if self.input_channels == 4:
            if image.shape[2] >= 4:
                infrared = image[:, :, 3:4]
            else:
                infrared = np.zeros((*image.shape[:2], 1), dtype=image.dtype)
            if self.training and random.random() < self.ir_dropout:
                infrared = np.zeros_like(infrared)
            image = np.concatenate((rgb, infrared), axis=2)
        else:
            image = rgb
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        if image.ndim == 2:
            image = image[:, :, None]
        return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)

    def __getitem__(self, index):
        image_path, label_path = self.items[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f'Could not read {image_path}')
        height, width = image.shape[:2]
        labels = json.loads(label_path.read_text())
        count = _object_count(labels, self.object_tasks)

        geometry = {'hbboxes': [], 'obboxes': []}
        semantic = {task: [] for task in self.object_tasks}
        for object_index in range(count):
            hbbox = _at(labels.get('hbboxes', ()), object_index)
            if hbbox is not None:
                points = np.asarray(hbbox, dtype=np.float32)
                xmin, ymin = points.min(axis=0)
                xmax, ymax = points.max(axis=0)
                geometry['hbboxes'].append([
                    (xmin + xmax) / (2 * width), (ymin + ymax) / (2 * height),
                    (xmax - xmin) / width, (ymax - ymin) / height,
                ])

            obbox = _at(labels.get('obboxes', ()), object_index)
            if obbox is not None:
                points = torch.as_tensor(obbox, dtype=torch.float32).view(1, 4, 2)
                xywhr = xyxyxyxy2xywhr(points)[0]
                xywhr[:4] /= torch.tensor([width, height, width, height])
                geometry['obboxes'].append(xywhr.tolist())

            reference = _reference_box(labels, object_index)
            if reference is None:
                continue
            points = np.asarray(reference, dtype=np.float32)
            xmin, ymin = points.min(axis=0)
            xmax, ymax = points.max(axis=0)
            location = [
                (xmin + xmax) / (2 * width), (ymin + ymax) / (2 * height),
                (xmax - xmin) / width, (ymax - ymin) / height,
            ]
            for task in self.object_tasks:
                value = _at(get_satellitepy_dict_values(labels, task), object_index)
                if value is None:
                    continue
                task_dict = self.task_dicts[task]
                if 'min' in task_dict and 'max' in task_dict:
                    target = (float(value) - task_dict['min']) / (task_dict['max'] - task_dict['min'])
                else:
                    # A field name can be reused with a dataset-specific
                    # vocabulary (for example FineAir's role-like strings vs.
                    # RarePlanes roles). Unknown values are sparse/missing
                    # supervision for this shared head, not valid class IDs.
                    if value not in task_dict:
                        continue
                    target = int(task_dict[value])
                semantic[task].append({'target': target, 'xywh': location})

        provided = set(labels.get('_provided_geometry', ()))
        # Legacy patch JSON lacks provenance. A non-null box proves that the
        # representation was supplied, but an empty legacy patch remains
        # unlabelled rather than being incorrectly used as a negative.
        for name in geometry:
            if geometry[name]:
                provided.add(name)
        tasks = semantic
        if 'masks' in self.tasks:
            mask = np.zeros((height, width), dtype=np.float32)
            valid_mask = np.ones((height, width), dtype=np.float32)
            mask_values = labels.get('masks', ())
            for object_index, coordinates in enumerate(mask_values):
                if coordinates is None:
                    # An unmatched instance is unknown, not background. Ignore
                    # its OBB footprint in the dense pixel loss.
                    reference = _reference_box(labels, object_index)
                    if reference is not None:
                        polygon = np.rint(np.asarray(reference)).astype(np.int32)
                        cv2.fillPoly(valid_mask, [polygon], 0.0)
                    continue
                if isinstance(coordinates, dict):
                    polygons = [np.asarray(points, dtype=np.int32)
                                for points in coordinates.get('polygons', ())]
                    if polygons:
                        cv2.fillPoly(mask, polygons, 1.0)
                    continue
                xs, ys = coordinates
                xs = np.asarray(xs, dtype=int).clip(0, width - 1)
                ys = np.asarray(ys, dtype=int).clip(0, height - 1)
                mask[ys, xs] = 1.0
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            valid_mask = cv2.resize(
                valid_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            tasks['masks'] = {
                'mask': torch.from_numpy(mask).unsqueeze(0),
                'valid_mask': torch.from_numpy(valid_mask).unsqueeze(0),
                'available': 'masks' in labels.get('_provided_tasks', ()),
            }
        return {
            'image': self._read_image(image_path),
            'geometry': geometry,
            'geometry_available': {name: name in provided for name in geometry},
            'tasks': semantic,
            'image_path': str(image_path),
            'geometry_family': self.geometry_family,
            'geometry_domain': self.geometry_domain,
        }


def sparse_mtl_collate(samples: list[dict]) -> dict:
    """Collate variable sparse annotations and retain image-level masks."""
    batch = {
        'img': torch.stack([sample['image'] for sample in samples]),
        'geometry': {},
        'tasks': {},
        'image_path': [sample['image_path'] for sample in samples],
        'geometry_family': [sample['geometry_family'] for sample in samples],
        'geometry_domain': [sample['geometry_domain'] for sample in samples],
    }
    for name, dimensions in (('hbboxes', 4), ('obboxes', 5)):
        boxes, batch_indices = [], []
        for batch_index, sample in enumerate(samples):
            values = sample['geometry'][name]
            boxes.extend(values)
            batch_indices.extend([batch_index] * len(values))
        batch['geometry'][name] = {
            'bboxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, dimensions),
            'batch_idx': torch.tensor(batch_indices, dtype=torch.long),
            # YOLO's objectness-class target is always zero. Semantic classes
            # are supervised by independent heads below.
            'cls': torch.zeros(len(boxes), dtype=torch.long),
            'image_mask': torch.tensor([
                sample['geometry_available'][name] for sample in samples
            ], dtype=torch.bool),
        }
    for task in samples[0]['tasks']:
        if task == 'masks':
            batch['tasks'][task] = {
                'mask': torch.stack([sample['tasks'][task]['mask'] for sample in samples]),
                'valid_mask': torch.stack([
                    sample['tasks'][task]['valid_mask'] for sample in samples]),
                'image_mask': torch.tensor([
                    sample['tasks'][task]['available'] for sample in samples
                ], dtype=torch.bool),
            }
            continue
        targets, locations, batch_indices = [], [], []
        for batch_index, sample in enumerate(samples):
            values = sample['tasks'][task]
            targets.extend(value['target'] for value in values)
            locations.extend(value['xywh'] for value in values)
            batch_indices.extend([batch_index] * len(values))
        is_float = any(isinstance(value, float) for value in targets)
        batch['tasks'][task] = {
            'target': torch.tensor(targets, dtype=torch.float32 if is_float else torch.long),
            'xywh': torch.tensor(locations, dtype=torch.float32).reshape(-1, 4),
            'batch_idx': torch.tensor(batch_indices, dtype=torch.long),
        }
    return batch
