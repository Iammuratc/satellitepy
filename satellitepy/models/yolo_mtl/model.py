"""YOLO11 backbone with independently selectable geometry and task heads.

The stock YOLO detection head couples one geometry representation to one class
taxonomy.  SatellitePy datasets instead contain sparse combinations such as
HBB+coarse-class and OBB+coarse-class+role.  This module keeps those outputs
independent so losses can be masked according to annotation availability.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import nn

try:
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.modules import Conv
    from ultralytics.nn.modules.head import Detect, OBB
except ImportError as error:  # pragma: no cover - exercised only without extra
    raise ImportError(
        'SparseMTLYOLO requires ultralytics>=8.4,<8.5. '
        'Install the project requirements first.'
    ) from error


GEOMETRY_TASKS = frozenset({'hbboxes', 'obboxes'})


class DomainSpecificBatchNorm2d(nn.Module):
    """Batch normalization with independent parameters and statistics per dataset.

    Convolutional weights stay shared. Mixed-domain minibatches are partitioned
    only for normalization and restored to their original sample order.
    """

    def __init__(self, source: nn.BatchNorm2d, domains: Iterable[str]):
        super().__init__()
        domains = tuple(domains)
        if not domains or len(set(domains)) != len(domains):
            raise ValueError('Domain-specific normalization needs unique domains')
        self.norms = nn.ModuleDict({domain: copy.deepcopy(source) for domain in domains})
        self.domains = domains

    def forward(self, x: torch.Tensor, sample_domains: Iterable[str]) -> torch.Tensor:
        sample_domains = tuple(sample_domains)
        if len(sample_domains) != x.shape[0]:
            raise ValueError('One normalization domain is required per image')
        unknown = set(sample_domains) - set(self.domains)
        if unknown:
            raise ValueError(f'Unknown normalization domains: {sorted(unknown)}')
        output = torch.zeros_like(x)
        for domain in dict.fromkeys(sample_domains):
            indices = torch.tensor(
                [index for index, value in enumerate(sample_domains) if value == domain],
                device=x.device, dtype=torch.long)
            normalized = self.norms[domain](x.index_select(0, indices))
            output = output.index_copy(0, indices, normalized)
        return output


class DomainSpecificNormContext:
    """Temporarily supplies dataset tags to nested normalization modules."""

    def __init__(self, modules, sample_domains):
        self.modules = modules
        self.sample_domains = tuple(sample_domains) if sample_domains is not None else None

    def __enter__(self):
        if self.modules and self.sample_domains is None:
            raise ValueError('Dataset domains are required with domain-specific normalization')
        for module in self.modules:
            module._sample_domains = self.sample_domains

    def __exit__(self, exc_type, exc_value, traceback):
        for module in self.modules:
            module._sample_domains = None


class ContextualDomainSpecificBatchNorm2d(DomainSpecificBatchNorm2d):
    """DSBN adapter compatible with unmodified Ultralytics graph modules."""

    def __init__(self, source: nn.BatchNorm2d, domains: Iterable[str]):
        super().__init__(source, domains)
        self._sample_domains = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x, self._sample_domains)


@dataclass(frozen=True)
class TaskSpec:
    """A categorical or continuous object-level task head."""

    name: str
    outputs: int
    kind: str = 'classification'
    loss_weight: float = 1.0

    def __post_init__(self):
        if self.outputs < 1:
            raise ValueError(f'{self.name}: outputs must be positive')
        if self.kind not in {'classification', 'regression', 'segmentation'}:
            raise ValueError(f'{self.name}: unsupported task kind {self.kind!r}')
        if self.loss_weight < 0:
            raise ValueError(f'{self.name}: loss_weight must be non-negative')


class SharedTaskTower(nn.Module):
    """A shared semantic tower keeps the many-task model genuinely small."""

    def __init__(self, channels: Iterable[int], hidden: int = 128):
        super().__init__()
        channels = tuple(channels)
        self.layers = nn.ModuleList(
            nn.Sequential(
                Conv(channel, hidden, 3),
                Conv(hidden, hidden, 3),
            )
            for channel in channels
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        return [layer(feature) for layer, feature in zip(self.layers, features)]


class DenseTaskHead(nn.Module):
    """Cheap task-specific projections from the shared semantic tower."""

    def __init__(self, levels: int, hidden: int, outputs: int):
        super().__init__()
        self.layers = nn.ModuleList(nn.Conv2d(hidden, outputs, 1) for _ in range(levels))

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        return [layer(feature) for layer, feature in zip(self.layers, features)]


class GatedDenseTaskHead(nn.Module):
    """Task adapter with a learned, regularized route into shared features.

    The identity feature value is detached. Only the gated residual adapter can
    send semantic gradients into the shared tower/backbone. This gives each
    task and pyramid level a continuous choice between detached and joint
    training without changing the detector path.
    """

    def __init__(self, levels: int, hidden: int, outputs: int, initial_gate: float):
        super().__init__()
        if not 0 < initial_gate < 1:
            raise ValueError('initial_gate must be strictly between zero and one')
        self.adapters = nn.ModuleList(
            nn.Sequential(nn.Conv2d(hidden, hidden, 1), nn.SiLU())
            for _ in range(levels)
        )
        self.layers = nn.ModuleList(nn.Conv2d(hidden, outputs, 1) for _ in range(levels))
        initial_logit = math.log(initial_gate / (1.0 - initial_gate))
        self.gate_logits = nn.Parameter(torch.full((levels,), initial_logit))

    def gates(self) -> torch.Tensor:
        return self.gate_logits.sigmoid()

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        gates = self.gates()
        return [
            projection(feature.detach() + gates[level] * adapter(feature))
            for level, (feature, adapter, projection) in enumerate(
                zip(features, self.adapters, self.layers))
        ]


class GatedGeometryAdapter(nn.Module):
    """Detection-loss-trained residual specialization at each FPN level."""

    def __init__(self, channels: Iterable[int], initial_gate: float):
        super().__init__()
        if not 0 < initial_gate < 1:
            raise ValueError('initial_gate must be strictly between zero and one')
        channels = tuple(channels)
        self.adapters = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channel, channel, 1), nn.SiLU())
            for channel in channels
        )
        initial_logit = math.log(initial_gate / (1.0 - initial_gate))
        self.gate_logits = nn.Parameter(torch.full((len(channels),), initial_logit))

    def gates(self) -> torch.Tensor:
        return self.gate_logits.sigmoid()

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        gates = self.gates()
        return [
            feature + gates[level] * adapter(feature)
            for level, (feature, adapter) in enumerate(zip(features, self.adapters))
        ]


class DynamicSpatialChannelGeometryAdapter(nn.Module):
    """Image-conditioned spatial/channel residual expert for one domain.

    A bottleneck depthwise spatial path models domain-specific texture while a
    squeeze-style gate selects channels independently for every image. The
    zero-initialized expansion starts training at the shared identity route.
    """

    def __init__(self, channels: Iterable[int], initial_gate: float,
                 bottleneck_ratio: int = 4):
        super().__init__()
        if not 0 < initial_gate < 1:
            raise ValueError('initial_gate must be strictly between zero and one')
        if bottleneck_ratio < 1:
            raise ValueError('bottleneck_ratio must be positive')
        channels = tuple(channels)
        self.adapters = nn.ModuleList()
        self.gate_networks = nn.ModuleList()
        initial_logit = math.log(initial_gate / (1.0 - initial_gate))
        for channel in channels:
            hidden = max(16, channel // bottleneck_ratio)
            expansion = nn.Conv2d(hidden, channel, 1)
            nn.init.zeros_(expansion.weight)
            nn.init.zeros_(expansion.bias)
            self.adapters.append(nn.Sequential(
                Conv(channel, hidden, 1),
                Conv(hidden, hidden, 3, g=hidden),
                expansion,
            ))
            gate_hidden = max(16, channel // 16)
            gate_output = nn.Conv2d(gate_hidden, channel, 1)
            nn.init.zeros_(gate_output.weight)
            nn.init.constant_(gate_output.bias, initial_logit)
            self.gate_networks.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel, gate_hidden, 1),
                nn.SiLU(),
                gate_output,
                nn.Sigmoid(),
            ))
        self.register_buffer('_last_gates', torch.full((len(channels),), initial_gate))

    def gates(self) -> torch.Tensor:
        return self._last_gates

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs = []
        gate_means = []
        for feature, adapter, gate_network in zip(
                features, self.adapters, self.gate_networks):
            gate = gate_network(feature)
            outputs.append(feature + gate * adapter(feature))
            gate_means.append(gate.detach().mean())
        self._last_gates.copy_(torch.stack(gate_means))
        return outputs


class MaskHead(nn.Module):
    """Lightweight foreground-mask decoder from the P3 feature map."""

    def __init__(self, channels: int, outputs: int = 1):
        super().__init__()
        hidden = min(256, channels)
        self.layers = nn.Sequential(
            Conv(channels, hidden, 3),
            Conv(hidden, hidden, 3),
            nn.Conv2d(hidden, outputs, 1),
        )

    def forward(self, features: list[torch.Tensor], output_size) -> torch.Tensor:
        logits = self.layers(features[0])
        return torch.nn.functional.interpolate(
            logits, size=output_size, mode='bilinear', align_corners=False)


class SparseMTLYOLO(nn.Module):
    """YOLO11 feature pyramid with sparse HBB, OBB, and attribute heads.

    Parameters
    ----------
    task_specs:
        Object-level categorical or regression task definitions. Geometry is
        configured separately and must not appear here.
    geometry_tasks:
        Any non-empty subset of ``{'hbboxes', 'obboxes'}``.
    weights:
        Ultralytics checkpoint used for the backbone/neck and geometry head
        initialization. ``yolo11s-obb.pt`` is the intended small baseline.
    input_channels:
        Three for RGB or four for RGB+IR. When four, the pretrained RGB stem is
        expanded by initializing the IR kernel from the mean RGB kernel.
    """

    def __init__(
        self,
        task_specs: Iterable[TaskSpec],
        geometry_tasks: Iterable[str] = ('obboxes',),
        geometry_families: Iterable[str] | None = None,
        geometry_loss_weights: Mapping[str, float] | None = None,
        weights: str = 'yolo11s-obb.pt',
        input_channels: int = 3,
        detach_task_features: bool = False,
        task_feature_gate_initial: float | None = None,
        task_feature_gate_regularization: float = 0.0,
        geometry_feature_gate_initial: float | None = None,
        geometry_feature_domains: Iterable[str] | None = None,
        geometry_domain_gate_initial: float | None = None,
        geometry_domain_adapter_type: str = 'scalar_1x1',
        geometry_domain_bottleneck_ratio: int = 4,
        geometry_fpn_domains: Iterable[str] | None = None,
        geometry_fpn_start: int = 11,
        mask_feature_domains: Mapping[str, str] | None = None,
        mask_detach_base_features: bool = False,
        normalization_domains: Iterable[str] | None = None,
        normalization_module_end: int = 0,
        backbone_feature_domains: Iterable[str] | None = None,
        backbone_domain_end: int = 0,
        frozen_backbone_modules: Iterable[int] | None = None,
    ):
        super().__init__()
        task_specs = tuple(task_specs)
        self.task_specs = {spec.name: spec for spec in task_specs}
        if len(self.task_specs) != len(task_specs):
            raise ValueError('Task names must be unique')
        if GEOMETRY_TASKS.intersection(self.task_specs):
            raise ValueError('Geometry belongs in geometry_tasks, not task_specs')

        self.geometry_tasks = frozenset(geometry_tasks)
        self.geometry_families = tuple(geometry_families or ())
        if self.geometry_families and self.geometry_tasks != {'obboxes'}:
            raise ValueError('geometry_families currently supports OBB geometry only')
        geometry_loss_weights = geometry_loss_weights or {}
        weighted_names = self.geometry_families or tuple(self.geometry_tasks)
        self.geometry_loss_weights = {
            name: float(geometry_loss_weights.get(
                name, geometry_loss_weights.get('obboxes', 1.0)))
            for name in weighted_names
        }
        if any(weight < 0 for weight in self.geometry_loss_weights.values()):
            raise ValueError('Geometry loss weights must be non-negative')
        unknown = self.geometry_tasks - GEOMETRY_TASKS
        if unknown or not self.geometry_tasks:
            raise ValueError(
                f'geometry_tasks must be a non-empty subset of {sorted(GEOMETRY_TASKS)}; '
                f'got unknown tasks {sorted(unknown)}')

        source = YOLO(weights).model
        # Checkpoint metadata contains only a few overrides; merge it with the
        # installed version's complete loss defaults (box/cls/dfl/angle gains).
        self.ultralytics_args = get_cfg(overrides=copy.deepcopy(source.args))
        source_head = source.model[-1]
        if not isinstance(source_head, OBB):
            raise TypeError(f'{weights!r} is not an Ultralytics OBB model')

        # Keep the graph-routing metadata on every backbone/neck layer. The
        # final stock head is replaced by independent heads below.
        self.backbone_neck = copy.deepcopy(source.model[:-1])
        self.backbone_neck.requires_grad_(True)
        self.save = frozenset(source.save)
        self.feature_from = tuple(source_head.f)
        channels = tuple(layer[0].conv.in_channels for layer in source_head.cv2)
        self.feature_channels = channels
        self.register_buffer('stride', source_head.stride.detach().clone())

        if input_channels not in (3, 4):
            raise ValueError('input_channels must be 3 (RGB) or 4 (RGB+IR)')
        self.input_channels = input_channels
        self.detach_task_features = bool(detach_task_features)
        if self.detach_task_features and task_feature_gate_initial is not None:
            raise ValueError('detach_task_features and task feature gating are mutually exclusive')
        self.task_feature_gate_initial = task_feature_gate_initial
        self.task_feature_gate_regularization = float(task_feature_gate_regularization)
        if self.task_feature_gate_regularization < 0:
            raise ValueError('task_feature_gate_regularization must be non-negative')
        if input_channels == 4:
            self._expand_input_stem()
        self.backbone_feature_domains = tuple(backbone_feature_domains or ())
        self.backbone_domain_end = int(backbone_domain_end)
        if self.backbone_feature_domains:
            if len(set(self.backbone_feature_domains)) != len(self.backbone_feature_domains):
                raise ValueError('backbone_feature_domains must be unique')
            if not 0 < self.backbone_domain_end < len(self.backbone_neck):
                raise ValueError('backbone_domain_end must split the backbone')
        elif self.backbone_domain_end:
            raise ValueError('backbone_domain_end requires backbone_feature_domains')
        self.backbone_domain_routes = nn.ModuleDict({
            domain: copy.deepcopy(self.backbone_neck[:self.backbone_domain_end])
            for domain in self.backbone_feature_domains
        })
        # The shared prefix is retained only as graph metadata/checkpoint
        # provenance. Every accepted dataset is routed through a private copy.
        if self.backbone_domain_routes:
            self.backbone_neck[:self.backbone_domain_end].requires_grad_(False)
        self.normalization_domains = tuple(normalization_domains or ())
        self.normalization_module_end = int(normalization_module_end)
        if self.normalization_domains and self.backbone_feature_domains:
            raise ValueError('Early backbone routing and domain-specific normalization '
                             'cannot be enabled together')
        if self.normalization_domains:
            if not 0 < self.normalization_module_end <= len(self.backbone_neck):
                raise ValueError('normalization_module_end must select backbone modules')
            for module in self.backbone_neck[:self.normalization_module_end]:
                self._replace_batch_norm(module)
        elif self.normalization_module_end:
            raise ValueError('normalization_module_end requires normalization_domains')
        if geometry_feature_gate_initial is not None and not self.geometry_families:
            raise ValueError('geometry feature gating requires geometry_families')
        self.geometry_feature_adapters = nn.ModuleDict({
            name: GatedGeometryAdapter(channels, float(geometry_feature_gate_initial))
            for name in self.geometry_families
        } if geometry_feature_gate_initial is not None else {})
        self.geometry_feature_domains = tuple(geometry_feature_domains or ())
        if self.geometry_feature_domains and geometry_domain_gate_initial is None:
            raise ValueError('geometry_feature_domains requires geometry_domain_gate_initial')
        if geometry_domain_gate_initial is not None and not self.geometry_feature_domains:
            raise ValueError('geometry_domain_gate_initial requires geometry_feature_domains')
        domain_adapter_types = {
            'scalar_1x1': GatedGeometryAdapter,
            'dynamic_spatial_channel': DynamicSpatialChannelGeometryAdapter,
        }
        if geometry_domain_adapter_type not in domain_adapter_types:
            raise ValueError(f'Unknown geometry_domain_adapter_type: '
                             f'{geometry_domain_adapter_type!r}')
        domain_adapter_class = domain_adapter_types[geometry_domain_adapter_type]
        self.geometry_domain_adapter_type = geometry_domain_adapter_type
        self.geometry_domain_adapters = nn.ModuleDict({
            name: (domain_adapter_class(
                channels, float(geometry_domain_gate_initial),
                int(geometry_domain_bottleneck_ratio))
                if domain_adapter_class is DynamicSpatialChannelGeometryAdapter
                else domain_adapter_class(channels, float(geometry_domain_gate_initial)))
            for name in self.geometry_feature_domains
        })
        self.geometry_fpn_domains = tuple(geometry_fpn_domains or ())
        overlap = set(self.geometry_fpn_domains) & set(self.geometry_feature_domains)
        if overlap:
            raise ValueError(f'Domains cannot use both an adapter and a separate FPN: {overlap}')
        self.geometry_fpn_start = int(geometry_fpn_start)
        if self.geometry_fpn_domains and not 0 < self.geometry_fpn_start < len(self.backbone_neck):
            raise ValueError('geometry_fpn_start must split the backbone and neck')
        self.geometry_domain_necks = nn.ModuleDict({
            name: copy.deepcopy(self.backbone_neck[self.geometry_fpn_start:])
            for name in self.geometry_fpn_domains
        })

        # A routed mask is a geometry auxiliary: it consumes a selected
        # domain-specific FPN directly and is optimized with the detector.
        self.mask_feature_domains = dict(mask_feature_domains or {})
        unknown_mask_tasks = set(self.mask_feature_domains) - set(self.task_specs)
        non_mask_tasks = {
            name for name in self.mask_feature_domains
            if name in self.task_specs and self.task_specs[name].kind != 'segmentation'
        }
        available_mask_domains = (
            set(self.geometry_fpn_domains) | set(self.geometry_feature_domains) | {'shared'}
        )
        unknown_mask_domains = set(self.mask_feature_domains.values()) - available_mask_domains
        if unknown_mask_tasks:
            raise ValueError(f'Unknown routed mask tasks: {sorted(unknown_mask_tasks)}')
        if non_mask_tasks:
            raise ValueError(f'Only segmentation tasks can use mask feature routes: '
                             f'{sorted(non_mask_tasks)}')
        if unknown_mask_domains:
            raise ValueError(f'Unknown mask feature domains: '
                             f'{sorted(unknown_mask_domains)}')
        self.mask_detach_base_features = bool(mask_detach_base_features)
        self.detector_task_names = frozenset(self.mask_feature_domains)

        self.geometry_heads = nn.ModuleDict()
        self.geometry_target_names = {}
        obbox_head_names = self.geometry_families or (('obboxes',)
                                                       if 'obboxes' in self.geometry_tasks else ())
        for head_name in obbox_head_names:
            head = OBB(nc=1, ch=channels)
            head.cv2.load_state_dict(copy.deepcopy(source_head.cv2.state_dict()))
            head.cv4.load_state_dict(copy.deepcopy(source_head.cv4.state_dict()))
            head.stride = self.stride.clone()
            head.bias_init()
            self.geometry_heads[head_name] = head
            self.geometry_target_names[head_name] = 'obboxes'
        if 'hbboxes' in self.geometry_tasks:
            head = Detect(nc=1, ch=channels)
            head.cv2.load_state_dict(copy.deepcopy(source_head.cv2.state_dict()))
            head.stride = self.stride.clone()
            head.bias_init()
            self.geometry_heads['hbboxes'] = head
            self.geometry_target_names['hbboxes'] = 'hbboxes'

        semantic_hidden = 128
        self.shared_task_tower = SharedTaskTower(channels, semantic_hidden)
        head_class = (GatedDenseTaskHead
                      if task_feature_gate_initial is not None else DenseTaskHead)
        self.task_heads = nn.ModuleDict({
            name: (head_class(len(channels), semantic_hidden, spec.outputs,
                              float(task_feature_gate_initial))
                   if task_feature_gate_initial is not None
                   else head_class(len(channels), semantic_hidden, spec.outputs))
            for name, spec in self.task_specs.items()
            if spec.kind != 'segmentation'
        })
        self.mask_heads = nn.ModuleDict({
            name: MaskHead(channels[0] if name in self.mask_feature_domains
                           else semantic_hidden, spec.outputs)
            for name, spec in self.task_specs.items()
            if spec.kind == 'segmentation'
        })
        self.frozen_backbone_modules = tuple(
            sorted(set(int(index) for index in (frozen_backbone_modules or ()))))
        invalid_frozen = [index for index in self.frozen_backbone_modules
                          if not 0 <= index < len(self.backbone_neck)]
        if invalid_frozen:
            raise ValueError(f'Invalid frozen backbone modules: {invalid_frozen}')
        for index in self.frozen_backbone_modules:
            self.backbone_neck[index].requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # A true freeze includes BatchNorm running statistics, not only its
        # affine parameters. Keep each selected subtree in evaluation mode.
        if mode:
            for index in self.frozen_backbone_modules:
                self.backbone_neck[index].eval()
        return self

    def _expand_input_stem(self):
        stem = self.backbone_neck[0].conv
        if stem.in_channels != 3:
            raise ValueError(f'Expected a three-channel pretrained stem, got {stem.in_channels}')
        expanded = nn.Conv2d(
            4, stem.out_channels, stem.kernel_size, stem.stride, stem.padding,
            stem.dilation, stem.groups, stem.bias is not None, stem.padding_mode)
        with torch.no_grad():
            expanded.weight[:, :3].copy_(stem.weight)
            expanded.weight[:, 3:4].copy_(stem.weight.mean(dim=1, keepdim=True))
            if stem.bias is not None:
                expanded.bias.copy_(stem.bias)
        self.backbone_neck[0].conv = expanded

    def _replace_batch_norm(self, module):
        for name, child in tuple(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                setattr(module, name, ContextualDomainSpecificBatchNorm2d(
                    child, self.normalization_domains))
            else:
                self._replace_batch_norm(child)

    def _domain_norm_modules(self):
        return tuple(module for module in self.modules()
                     if isinstance(module, ContextualDomainSpecificBatchNorm2d))

    def _run_graph(self, modules, x, values):
        for module in modules:
            if module.f != -1:
                if isinstance(module.f, int):
                    x = values[module.f]
                else:
                    x = [x if index == -1 else values[index] for index in module.f]
            x = module(x)
            values.append(x if module.i in self.save else None)
        return x, values

    def _select_features(self, x, values) -> list[torch.Tensor]:
        return [x if index == -1 else values[index] for index in self.feature_from]

    def extract_features(self, image: torch.Tensor) -> list[torch.Tensor]:
        """Run the Ultralytics graph and return the P3/P4/P5 head inputs."""
        x, values = self._run_graph(self.backbone_neck, image, [])
        return self._select_features(x, values)

    def _run_domain_backbone_prefix(self, image: torch.Tensor, sample_domains):
        """Run each dataset subset through its private early-backbone copy."""
        if not self.backbone_domain_routes:
            return self._run_graph(
                self.backbone_neck[:self.backbone_domain_end], image, [])
        if sample_domains is None:
            raise ValueError('Dataset domains are required with early backbone routing')
        sample_domains = tuple(sample_domains)
        if len(sample_domains) != image.shape[0]:
            raise ValueError('One backbone domain is required per image')
        unknown = set(sample_domains) - set(self.backbone_feature_domains)
        if unknown:
            raise ValueError(f'Unknown early-backbone domains: {sorted(unknown)}')

        merged_x = None
        merged_values = None
        for domain in dict.fromkeys(sample_domains):
            indices = torch.tensor(
                [index for index, value in enumerate(sample_domains) if value == domain],
                device=image.device, dtype=torch.long)
            domain_x, domain_values = self._run_graph(
                self.backbone_domain_routes[domain], image.index_select(0, indices), [])
            if merged_x is None:
                merged_x = domain_x.new_empty((image.shape[0], *domain_x.shape[1:]))
                merged_values = [
                    (value.new_empty((image.shape[0], *value.shape[1:]))
                     if value is not None else None)
                    for value in domain_values
                ]
            merged_x.index_copy_(0, indices, domain_x)
            for merged, value in zip(merged_values, domain_values):
                if merged is not None:
                    merged.index_copy_(0, indices, value)
        return merged_x, merged_values

    def extract_features_with_domains(self, image: torch.Tensor, sample_domains=None):
        """Share the backbone, then run the common and domain-specific FPNs."""
        if not self.backbone_domain_routes and not self.geometry_domain_necks:
            return self.extract_features(image), {}
        split = (self.geometry_fpn_start if self.geometry_domain_necks
                 else len(self.backbone_neck))
        if self.backbone_domain_routes:
            if self.backbone_domain_end > split:
                raise ValueError('backbone_domain_end cannot follow geometry_fpn_start')
            prefix_x, prefix_values = self._run_domain_backbone_prefix(
                image, sample_domains)
            prefix_x, prefix_values = self._run_graph(
                self.backbone_neck[self.backbone_domain_end:split], prefix_x,
                list(prefix_values))
        else:
            prefix_x, prefix_values = self._run_graph(
                self.backbone_neck[:split], image, [])
        shared_x, shared_values = self._run_graph(
            self.backbone_neck[split:], prefix_x, list(prefix_values))
        domain_features = {}
        for domain, neck in self.geometry_domain_necks.items():
            domain_x, domain_values = self._run_graph(
                neck, prefix_x, list(prefix_values))
            domain_features[domain] = self._select_features(domain_x, domain_values)
        return self._select_features(shared_x, shared_values), domain_features

    def forward(self, image: torch.Tensor,
                sample_domains: Iterable[str] | None = None) -> Mapping[str, object]:
        with DomainSpecificNormContext(self._domain_norm_modules(), sample_domains):
            features, fpn_domain_features = self.extract_features_with_domains(
                image, sample_domains)
        routed_features = dict(fpn_domain_features)
        routed_features.update({
            domain: adapter(features)
            for domain, adapter in self.geometry_domain_adapters.items()
        })
        task_features = ([feature.detach() for feature in features]
                         if self.detach_task_features else features)
        semantic_features = self.shared_task_tower(task_features)

        def mask_features(name):
            domain = self.mask_feature_domains.get(name)
            if domain is None:
                return semantic_features
            base = ([feature.detach() for feature in features]
                    if self.mask_detach_base_features else features)
            if domain == 'shared':
                return base
            if domain in self.geometry_domain_adapters:
                # Reuse the detection adapter parameters, but optionally cut
                # only the mask gradient at the shared feature boundary.
                return self.geometry_domain_adapters[domain](base)
            domain_features = routed_features[domain]
            return ([feature.detach() for feature in domain_features]
                    if self.mask_detach_base_features else domain_features)

        outputs = {
            'geometry': {
                # Keep raw per-level tensors in both train and validation;
                # decoding/NMS is an explicit inference concern.
                name: head.forward_head(
                    self.geometry_feature_adapters[name](features)
                    if name in self.geometry_feature_adapters else features,
                    **head.one2many)
                for name, head in self.geometry_heads.items()
            },
            'tasks': {
                name: head(semantic_features)
                for name, head in self.task_heads.items()
            } | {
                name: head(mask_features(name), image.shape[-2:])
                for name, head in self.mask_heads.items()
            },
            'features': features,
        }
        if routed_features:
            outputs['geometry_by_domain'] = {
                domain: {
                    name: head.forward_head(domain_features, **head.one2many)
                    for name, head in self.geometry_heads.items()
                }
                for domain, domain_features in routed_features.items()
            }
        gated_heads = [
            head for head in self.task_heads.values()
            if isinstance(head, GatedDenseTaskHead)
        ]
        if gated_heads:
            outputs['task_gate_regularization'] = (
                torch.cat([head.gates() for head in gated_heads]).mean()
                * self.task_feature_gate_regularization
            )
            outputs['task_gate_values'] = {
                name: head.gates().detach()
                for name, head in self.task_heads.items()
                if isinstance(head, GatedDenseTaskHead)
            }
        return outputs
