"""Configuration helpers for sparse MTL YOLO models."""

from pathlib import Path

import yaml

from satellitepy.data.utils import get_task_dict
from .model import SparseMTLYOLO, TaskSpec


def _merge_config(base, overrides):
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path):
    path = Path(path)
    with path.open() as handle:
        config = yaml.safe_load(handle)
    parent = config.pop('extends', None)
    if parent is None:
        return config
    return _merge_config(load_config(path.parent / parent), config)


def task_specs_from_config(config: dict) -> list[TaskSpec]:
    specs = []
    for name, options in config.get('tasks', {}).items():
        kind = options.get('kind', 'classification')
        if kind in {'regression', 'segmentation'}:
            outputs = int(options.get('outputs', 1))
        else:
            task_dict = get_task_dict(name)
            outputs = int(options.get('outputs', max(task_dict.values()) + 1))
        specs.append(TaskSpec(
            name=name,
            outputs=outputs,
            kind=kind,
            loss_weight=float(options.get('loss_weight', 1.0)),
        ))
    return specs


def build_model(config: dict) -> SparseMTLYOLO:
    options = config['model']
    return SparseMTLYOLO(
        task_specs_from_config(config),
        geometry_tasks=options.get('geometry_tasks', ('obboxes',)),
        geometry_families=options.get('geometry_families'),
        geometry_loss_weights=options.get('geometry_loss_weights'),
        weights=options.get('weights', 'yolo11s-obb.pt'),
        input_channels=int(options.get('input_channels', 3)),
        detach_task_features=bool(options.get('detach_task_features', False)),
        task_feature_gate_initial=options.get('task_feature_gate_initial'),
        task_feature_gate_regularization=float(
            options.get('task_feature_gate_regularization', 0.0)),
        geometry_feature_gate_initial=options.get('geometry_feature_gate_initial'),
        geometry_feature_domains=options.get('geometry_feature_domains'),
        geometry_domain_gate_initial=options.get('geometry_domain_gate_initial'),
        geometry_domain_adapter_type=options.get(
            'geometry_domain_adapter_type', 'scalar_1x1'),
        geometry_domain_bottleneck_ratio=int(
            options.get('geometry_domain_bottleneck_ratio', 4)),
        geometry_fpn_domains=options.get('geometry_fpn_domains'),
        geometry_fpn_start=int(options.get('geometry_fpn_start', 11)),
        mask_feature_domains=options.get('mask_feature_domains'),
        mask_detach_base_features=bool(options.get('mask_detach_base_features', False)),
        normalization_domains=options.get('normalization_domains'),
        normalization_module_end=int(options.get('normalization_module_end', 0)),
        backbone_feature_domains=options.get('backbone_feature_domains'),
        backbone_domain_end=int(options.get('backbone_domain_end', 0)),
        frozen_backbone_modules=options.get('frozen_backbone_modules'),
    )
