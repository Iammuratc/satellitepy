"""Masked geometry and semantic losses for :mod:`satellitepy.models.yolo_mtl`."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.utils.loss import v8DetectionLoss, v8OBBLoss


class _CriterionModelProxy:
    """Minimal Ultralytics model interface for a standalone geometry head."""

    def __init__(self, head: nn.Module, args):
        self.model = [head]
        self.args = args
        self.class_weights = None

    def parameters(self):
        return self.model[0].parameters()


def _slice_predictions(predictions: dict, indices: torch.Tensor) -> dict:
    result = {}
    for key, value in predictions.items():
        if key == 'feats':
            result[key] = [feature.index_select(0, indices) for feature in value]
        elif torch.is_tensor(value):
            result[key] = value.index_select(0, indices)
        else:
            result[key] = value
    return result


class SparseMTLLoss(nn.Module):
    """Apply every loss only to images/objects carrying that annotation."""

    def __init__(self, model, balance_options=None):
        super().__init__()
        args = model.ultralytics_args
        self.criteria = {
            name: (v8OBBLoss if model.geometry_target_names[name] == 'obboxes'
                   else v8DetectionLoss)(
                _CriterionModelProxy(head, args))
            for name, head in model.geometry_heads.items()
        }
        self.task_specs = model.task_specs
        self.geometry_loss_weights = model.geometry_loss_weights
        self.geometry_target_names = model.geometry_target_names
        self.register_buffer('strides', model.stride.clone())
        balance_options = balance_options or {}
        self.balance_mode = balance_options.get('mode', 'none')
        if self.balance_mode not in {'none', 'ema'}:
            raise ValueError(f'Unsupported loss balance mode: {self.balance_mode}')
        self.balance_momentum = float(balance_options.get('momentum', 0.98))
        self.balance_target = float(balance_options.get('target', 1.0))
        self.balance_targets = {
            name: float(target)
            for name, target in balance_options.get('targets', {}).items()
        }
        self.balance_min_scale = float(balance_options.get('min_scale', 0.05))
        self.balance_max_scale = float(balance_options.get('max_scale', 100.0))
        self._balance_names = tuple(model.geometry_heads) + tuple(model.task_specs)
        for index, _ in enumerate(self._balance_names):
            self.register_buffer(f'_loss_ema_{index}', torch.ones(()))
            self.register_buffer(f'_loss_ema_ready_{index}', torch.zeros((), dtype=torch.bool))

    def _balance(self, losses):
        if self.balance_mode == 'none':
            return losses
        balanced = {}
        for index, name in enumerate(self._balance_names):
            value = losses[name]
            ema = getattr(self, f'_loss_ema_{index}')
            ready = getattr(self, f'_loss_ema_ready_{index}')
            magnitude = value.detach().abs()
            if self.training and torch.isfinite(magnitude) and magnitude > 0:
                if ready:
                    ema.mul_(self.balance_momentum).add_(
                        magnitude, alpha=1.0 - self.balance_momentum)
                else:
                    ema.copy_(magnitude)
                    ready.fill_(True)
            if ready:
                target = self.balance_targets.get(name, self.balance_target)
                scale = (target / ema.clamp_min(1e-8)).clamp(
                    self.balance_min_scale, self.balance_max_scale)
                balanced[name] = value * scale
            else:
                balanced[name] = value
        return balanced

    @staticmethod
    def _geometry_batch(target: dict, supervised: torch.Tensor, device) -> dict:
        old_to_new = torch.full(
            (len(target['image_mask']),), -1, device=device, dtype=torch.long)
        old_to_new[supervised] = torch.arange(len(supervised), device=device)
        object_mask = torch.isin(target['batch_idx'].to(device), supervised)
        old_indices = target['batch_idx'].to(device)[object_mask]
        return {
            'batch_idx': old_to_new[old_indices],
            'cls': target['cls'].to(device)[object_mask],
            'bboxes': target['bboxes'].to(device)[object_mask],
        }

    def _geometry_loss(self, name, predictions, target):
        device = predictions['boxes'].device
        supervised = target['image_mask'].to(device).nonzero(as_tuple=False).flatten()
        if not len(supervised):
            zero = predictions['boxes'].sum() * 0
            if 'angle' in predictions:
                zero = zero + predictions['angle'].sum() * 0
            return zero, {}
        selected = _slice_predictions(predictions, supervised)
        geometry_batch = self._geometry_batch(target, supervised, device)
        components, detached = self.criteria[name].loss(selected, geometry_batch)
        # Ultralytics multiplies detection loss by its local batch size. Undo
        # that convention here so geometry and mean-reduced semantic losses
        # have compatible per-sample scales in the joint objective.
        loss = components.sum() / len(supervised)
        return loss * self.geometry_loss_weights[name], detached

    def _task_loss(self, name, logits, target):
        spec = self.task_specs[name]
        if spec.kind == 'segmentation':
            supervised = target['image_mask'].to(logits.device)
            if not supervised.any():
                return logits.sum() * 0
            targets = target['mask'].to(logits.device)[supervised]
            valid = target['valid_mask'].to(logits.device)[supervised]
            pixel_loss = F.binary_cross_entropy_with_logits(
                logits[supervised], targets, reduction='none')
            return ((pixel_loss * valid).sum() / valid.sum().clamp_min(1)
                    * spec.loss_weight)
        if not len(target['target']):
            return sum(level.sum() * 0 for level in logits)
        device = logits[0].device
        batch_indices = target['batch_idx'].to(device)
        xywh = target['xywh'].to(device)
        target_values = target['target'].to(device)

        # Select the pyramid level whose nominal receptive object size is
        # closest to the target's largest side. This keeps semantic inference
        # tied to the same center/scale representation as YOLO detections.
        image_size = logits[0].shape[-1] * self.strides[0]
        object_size = xywh[:, 2:].amax(1) * image_size
        nominal_size = self.strides * 4
        levels = (torch.log2(object_size[:, None].clamp_min(1) / nominal_size[None])
                  .abs().argmin(1))

        gathered, ordered_targets = [], []
        for level_index, level_logits in enumerate(logits):
            mask = levels == level_index
            if not mask.any():
                continue
            height, width = level_logits.shape[-2:]
            centers = xywh[mask, :2]
            xs = (centers[:, 0] * width).long().clamp_(0, width - 1)
            ys = (centers[:, 1] * height).long().clamp_(0, height - 1)
            gathered.append(level_logits[batch_indices[mask], :, ys, xs])
            ordered_targets.append(target_values[mask])
        predictions = torch.cat(gathered)
        targets = torch.cat(ordered_targets)
        if spec.kind == 'classification':
            loss = F.cross_entropy(predictions, targets.long())
        else:
            loss = F.smooth_l1_loss(predictions.squeeze(-1), targets.float())
        return loss * spec.loss_weight

    def forward(self, outputs: dict, batch: dict):
        raw_losses, details = {}, {}
        geometry_domain_losses = {}
        geometry_domain_counts = {}
        geometry_predictions = outputs['geometry']
        domain_predictions = outputs.get('geometry_by_domain')
        for name, predictions in geometry_predictions.items():
            target_name = self.geometry_target_names[name]
            target = dict(batch['geometry'][target_name])
            if name != target_name:
                family_mask = torch.tensor(
                    [family == name for family in batch['geometry_family']],
                    dtype=torch.bool)
                target['image_mask'] = target['image_mask'] & family_mask
            raw_losses[name], details[name] = self._geometry_loss(
                name, predictions, target)
        if batch['geometry_domain']:
            routed_losses = {}
            for name in self.geometry_target_names:
                target_name = self.geometry_target_names[name]
                weighted_losses, total_supervised = [], 0
                for domain in dict.fromkeys(batch['geometry_domain']):
                    predictions_by_family = (domain_predictions or {}).get(
                        domain, geometry_predictions)
                    target = dict(batch['geometry'][target_name])
                    domain_mask = torch.tensor(
                        [value == domain for value in batch['geometry_domain']],
                        dtype=torch.bool)
                    if name != target_name:
                        family_mask = torch.tensor(
                            [family == name for family in batch['geometry_family']],
                            dtype=torch.bool)
                        domain_mask &= family_mask
                    target['image_mask'] = target['image_mask'] & domain_mask
                    supervised = int(target['image_mask'].sum())
                    loss, _ = self._geometry_loss(
                        name, predictions_by_family[name], target)
                    geometry_domain_losses[domain] = (
                        geometry_domain_losses.get(domain, 0) + loss)
                    geometry_domain_counts[domain] = max(
                        geometry_domain_counts.get(domain, 0), supervised)
                    weighted_losses.append(loss * supervised)
                    total_supervised += supervised
                if total_supervised:
                    routed_losses[name] = sum(weighted_losses) / total_supervised
                else:
                    routed_losses[name] = sum(weighted_losses)
            # Routed adapters need the domain-specific predictions in the
            # ordinary objective. An unrouted detector keeps its exact legacy
            # geometry loss; the per-domain copies above exist only for PCGrad.
            if domain_predictions:
                raw_losses.update(routed_losses)
        for name, logits in outputs['tasks'].items():
            raw_losses[name] = self._task_loss(name, logits, batch['tasks'][name])
        losses = self._balance(raw_losses)
        if 'task_gate_regularization' in outputs:
            gate_regularization = outputs['task_gate_regularization']
            losses['task_gate_regularization'] = gate_regularization
            raw_losses['task_gate_regularization'] = gate_regularization
        total = sum(losses.values())
        details['raw_losses'] = raw_losses
        # Expose per-dataset detection objectives for optional gradient surgery.
        # These retain their graphs; ordinary training simply ignores them.
        details['geometry_domain_losses'] = geometry_domain_losses
        details['geometry_domain_counts'] = geometry_domain_counts
        return total, losses, details
