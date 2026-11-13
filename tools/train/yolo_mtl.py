"""Train sparse-supervision YOLO MTL on one or more patch datasets.

Launch on four GPUs with, for example::

    torchrun --standalone --nproc_per_node=4 tools/train/yolo_mtl.py \
      --train-folders /patches/merged/train \
      --val-folders /patches/merged/val \
      --out-folder /output/yolo11s_sparse_mtl
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from ultralytics.utils import nms
from ultralytics.utils.metrics import DetMetrics, batch_probiou

from satellitepy.models.yolo_mtl.data import SparseMTLDataset, sparse_mtl_collate
from satellitepy.models.yolo_mtl.factory import build_model, load_config
from satellitepy.models.yolo_mtl.loss import SparseMTLLoss


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=Path('configs/yolo_mtl/yolo11s_sparse_mtl.yaml'))
    parser.add_argument('--train-folders', type=Path, nargs='+', required=True,
                        help='Split roots containing images/ and labels/.')
    parser.add_argument('--val-folders', type=Path, nargs='+', required=True)
    parser.add_argument('--out-folder', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--init-weights', type=Path,
                        help='Initialize model weights only; optimizer, scaler, epoch, and '
                             'early-stopping state start fresh.')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wandb-project')
    parser.add_argument('--wandb-group')
    parser.add_argument('--wandb-name')
    parser.add_argument('--wandb-id')
    parser.add_argument('--wandb-notes',
                        help='Short W&B run description shown in the run overview.')
    return parser.parse_args()


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_initial_model_weights(model, checkpoint_path):
    """Load a detector checkpoint, adapting only an RGB/RGB+IR stem mismatch."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = dict(checkpoint['model'])
    target = model.state_dict()
    stem_key = 'backbone_neck.0.conv.weight'
    if stem_key in state and stem_key in target and state[stem_key].shape != target[stem_key].shape:
        source_weight, target_weight = state[stem_key], target[stem_key].clone()
        if source_weight.shape[1] == 3 and target_weight.shape[1] == 4:
            target_weight[:, :3].copy_(source_weight)
            target_weight[:, 3:4].copy_(source_weight.mean(dim=1, keepdim=True))
        elif source_weight.shape[1] == 4 and target_weight.shape[1] == 3:
            target_weight.copy_(source_weight[:, :3])
        else:
            raise ValueError(
                f'Unsupported input-stem transfer {tuple(source_weight.shape)} -> '
                f'{tuple(target_weight.shape)}')
        state[stem_key] = target_weight
        print(f'adapted input stem while loading {checkpoint_path}: '
              f'{source_weight.shape[1]} -> {target_weight.shape[1]} channels', flush=True)
    model.load_state_dict(state)
    return checkpoint


class OBBMAPEvaluator:
    """Accumulate single-class rotated-box AP using Ultralytics matching."""

    def __init__(self, head, options=None):
        options = options or {}
        self.head = head
        self.confidence = float(options.get('confidence', 0.001))
        self.nms_iou = float(options.get('nms_iou', 0.7))
        self.max_detections = int(options.get('max_detections', 1000))
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.metrics = DetMetrics({0: 'object'})

    def _match(self, pred_classes, true_classes, iou):
        thresholds = self.iouv.to(iou.device)
        correct = np.zeros((len(pred_classes), len(thresholds)), dtype=bool)
        if not len(pred_classes) or not len(true_classes):
            return correct
        iou = (iou * (true_classes[:, None] == pred_classes)).cpu().numpy()
        thresholds = thresholds.cpu().numpy()
        matched = np.zeros((iou.shape[0], len(thresholds)), dtype=bool)
        for detection in np.flatnonzero((iou >= thresholds.min()).any(0)):
            available = np.where(matched, 0, iou[:, detection, None])
            target = available.argmax(0)
            correct[detection] = available[target, range(len(thresholds))] >= thresholds
            matched[target, range(len(thresholds))] |= correct[detection]
        return correct

    def update(self, predictions, target, image_names):
        device_type = predictions['boxes'].device.type
        with torch.autocast(device_type=device_type, dtype=torch.float16,
                            enabled=device_type == 'cuda'):
            decoded = self.head._inference(predictions)
        detections = nms.non_max_suppression(
            decoded, self.confidence, self.nms_iou, nc=1, multi_label=True,
            max_det=self.max_detections, rotated=True)
        image_mask = target['image_mask'].to(decoded.device)
        batch_indices = target['batch_idx'].to(decoded.device)
        boxes = target['bboxes'].to(decoded.device)
        height, width = predictions['feats'][0].shape[-2:]
        image_height = height * int(self.head.stride[0])
        image_width = width * int(self.head.stride[0])
        scale = boxes.new_tensor([image_width, image_height, image_width, image_height, 1.0])
        for image_index, prediction in enumerate(detections):
            if not image_mask[image_index]:
                continue
            selected = batch_indices == image_index
            target_boxes = boxes[selected] * scale
            target_classes = torch.zeros(len(target_boxes), device=decoded.device)
            pred_boxes = torch.cat((prediction[:, :4], prediction[:, 6:7]), 1)
            pred_classes = prediction[:, 5]
            if len(target_boxes) and len(pred_boxes):
                correct = self._match(
                    pred_classes, target_classes, batch_probiou(target_boxes, pred_boxes))
            else:
                correct = np.zeros((len(prediction), len(self.iouv)), dtype=bool)
            target_numpy = target_classes.cpu().numpy()
            self.metrics.update_stats({
                'tp': correct,
                'conf': prediction[:, 4].detach().cpu().numpy(),
                'pred_cls': pred_classes.detach().cpu().numpy(),
                'target_cls': target_numpy,
                'target_img': np.unique(target_numpy),
                'im_name': Path(image_names[image_index]).name,
            })

    def compute(self):
        if not self.metrics.stats['target_cls']:
            return {'map50': 0.0, 'map50_95': 0.0}
        self.metrics.process()
        results = self.metrics.results_dict
        return {
            'map50': results['metrics/mAP50(B)'],
            'map50_95': results['metrics/mAP50-95(B)'],
        }


def distributed_context():
    distributed = 'LOCAL_RANK' in os.environ
    if distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
        return True, local_rank, dist.get_rank(), dist.get_world_size()
    device = 0 if torch.cuda.is_available() else None
    return False, device, 0, 1


def make_dataset(folders, config, training):
    train_config = config['train']
    tasks = tuple(config['tasks'])
    input_channels = int(config['model'].get('input_channels', 3))
    family_mapping = config['model'].get('dataset_geometry_families', {})
    datasets = [SparseMTLDataset(
        folder / 'images', folder / 'labels', tasks,
        image_size=int(train_config.get('image_size', 1024)),
        input_channels=input_channels,
        ir_dropout=float(train_config.get('rgb_ir_channel_dropout', 0.0)) if training else 0.0,
        training=training,
        geometry_family=family_mapping.get(folder.parents[1].name),
        geometry_domain=folder.parents[1].name,
    ) for folder in folders]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


class DistributedWeightedSampler(Sampler):
    """Deterministic weighted replacement sampling split across DDP ranks."""

    def __init__(self, weights, rank=0, world_size=1, seed=0):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.num_samples = math.ceil(len(self.weights) / world_size)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.num_samples * self.world_size,
            replacement=True, generator=generator)
        return iter(indices[self.rank::self.world_size].tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def make_train_sampler(dataset, train_config, distributed, rank, world_size, seed):
    sampling = train_config.get('dataset_sampling', 'natural')
    if sampling == 'natural':
        return DistributedSampler(dataset, shuffle=True) if distributed else None
    if sampling not in {'balanced', 'weighted'}:
        raise ValueError(f'Unsupported dataset_sampling mode: {sampling}')
    if not isinstance(dataset, ConcatDataset):
        raise ValueError(f'{sampling.capitalize()} dataset sampling requires multiple train folders')
    probabilities = train_config.get('dataset_sampling_weights')
    if sampling == 'balanced':
        probabilities = [1.0] * len(dataset.datasets)
    elif probabilities is None or len(probabilities) != len(dataset.datasets):
        raise ValueError('Weighted dataset sampling requires one dataset_sampling_weights '
                         'entry per train folder')
    probabilities = torch.as_tensor(probabilities, dtype=torch.double)
    if not torch.isfinite(probabilities).all() or (probabilities <= 0).any():
        raise ValueError('dataset_sampling_weights must be finite and strictly positive')
    probabilities /= probabilities.sum()
    weights = torch.cat([
        torch.full((len(source),), probability.item() / len(source), dtype=torch.double)
        for source, probability in zip(dataset.datasets, probabilities)
    ])
    return DistributedWeightedSampler(
        weights, rank=rank, world_size=world_size, seed=seed)


def reduce_mean(value, world_size):
    value = value.detach().clone()
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= world_size
    return value


def _all_reduce_gradients(parameters, world_size):
    if world_size <= 1:
        return
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def _pcgrad_step(domain_losses, domain_counts, detector_parameters, semantic_loss,
                 optimizers, world_size):
    """Apply PCGrad to dataset-specific detection gradients.

    Gradients are first aggregated across ranks using supervised-image counts,
    then projected. Semantic parameters are disjoint under strict-detached
    optimization and are averaged normally.
    """
    if semantic_loss.requires_grad:
        semantic_loss.backward(retain_graph=True)

    local_domains = sorted(domain_losses)
    if world_size > 1:
        gathered_domains = [None] * world_size
        dist.all_gather_object(gathered_domains, local_domains)
        domains = sorted(set().union(*gathered_domains))
    else:
        domains = local_domains

    task_gradients = []
    for domain in domains:
        local_count = torch.tensor(
            float(domain_counts.get(domain, 0)), device=semantic_loss.device)
        if domain in domain_losses and local_count.item() > 0:
            computed = torch.autograd.grad(
                domain_losses[domain], detector_parameters, retain_graph=True,
                allow_unused=True)
            gradients = [
                (torch.zeros_like(parameter) if gradient is None else gradient)
                * local_count
                for parameter, gradient in zip(detector_parameters, computed)
            ]
        else:
            gradients = [torch.zeros_like(parameter)
                         for parameter in detector_parameters]
        flat_gradient = torch.cat([gradient.reshape(-1) for gradient in gradients])
        if world_size > 1:
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(flat_gradient, op=dist.ReduceOp.SUM)
        if local_count.item() > 0:
            task_gradients.append(flat_gradient / local_count)

    diagnostics = {}
    pair_cosines = []
    for left_index in range(len(task_gradients)):
        for right_index in range(left_index + 1, len(task_gradients)):
            left, right = task_gradients[left_index], task_gradients[right_index]
            cosine = torch.dot(left, right) / (
                left.norm() * right.norm()).clamp_min(1e-12)
            pair_cosines.append(cosine)
            diagnostics[
                f'pcgrad/cosine_{domains[left_index]}_{domains[right_index]}'
            ] = cosine.detach().item()
    if pair_cosines:
        diagnostics['pcgrad/conflict_fraction'] = torch.stack([
            (cosine < 0).float() for cosine in pair_cosines]).mean().item()

    projected = [gradient.clone() for gradient in task_gradients]
    eps = 1e-12
    for task_index, task in enumerate(projected):
        for other_index, other in enumerate(task_gradients):
            if task_index == other_index:
                continue
            dot = torch.dot(task, other)
            if dot < 0:
                norm = torch.dot(other, other)
                coefficient = dot / norm.clamp_min(eps)
                task.sub_(other, alpha=coefficient.item())

    merged = torch.stack(projected).mean(0)
    offset = 0
    for parameter in detector_parameters:
        count = parameter.numel()
        parameter.grad = merged[offset:offset + count].view_as(parameter).clone()
        offset += count
    _all_reduce_gradients(
        [parameter for group in optimizers['semantic'].param_groups
         for parameter in group['params']], world_size)
    optimizers['detector'].step()
    optimizers['semantic'].step()
    return diagnostics


def run_epoch(model, criterion, loader, device, optimizers, scalers, epoch, rank, world_size,
              tracker=None, log_interval=10, map_options=None,
              gradient_conflict_control='none'):
    training = optimizers is not None
    model.train(training)
    criterion.train(training)
    total = torch.zeros((), device=device)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    evaluator = None
    mask_intersection = torch.zeros((), device=device, dtype=torch.float64)
    mask_union = torch.zeros((), device=device, dtype=torch.float64)
    mask_threshold = float((map_options or {}).get('mask_threshold', 0.5))
    if not training and 'obboxes' in raw_model.geometry_heads:
        evaluator = OBBMAPEvaluator(raw_model.geometry_heads['obboxes'], map_options)
    progress = tqdm(loader, disable=rank != 0,
                    desc=f'{"train" if training else "val"} epoch {epoch + 1}', dynamic_ncols=True)
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for step, batch in enumerate(progress):
            images = batch['img'].to(device, non_blocking=True)
            if training:
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == 'cuda'):
                outputs = model(images, sample_domains=batch['geometry_domain'])
                loss, losses, details = criterion(outputs, batch)
            if not training and evaluator is None and raw_model.geometry_families:
                families = set(batch['geometry_family'])
                if len(families) != 1 or None in families:
                    raise ValueError(f'Validation loader mixes geometry families: {families}')
                family = next(iter(families))
                evaluator = OBBMAPEvaluator(raw_model.geometry_heads[family], map_options)
            if evaluator is not None:
                geometry_outputs = outputs['geometry']
                if outputs.get('geometry_by_domain'):
                    domains = set(batch['geometry_domain'])
                    if len(domains) != 1 or None in domains:
                        raise ValueError(f'Validation loader mixes geometry domains: {domains}')
                    geometry_outputs = outputs['geometry_by_domain'].get(
                        next(iter(domains)), outputs['geometry'])
                geometry_name = ('obboxes' if 'obboxes' in outputs['geometry']
                                 else next(iter(set(batch['geometry_family']))))
                if geometry_name not in geometry_outputs:
                    geometry_name = next(iter(set(batch['geometry_family'])))
                evaluator.update(geometry_outputs[geometry_name], batch['geometry']['obboxes'],
                                 batch['image_path'])
            if not training and 'masks' in outputs['tasks']:
                available_cpu = batch['tasks']['masks']['image_mask']
                if available_cpu.any():
                    available = available_cpu.to(device)
                    predicted_masks = outputs['tasks']['masks'][available].sigmoid() >= mask_threshold
                    target_masks = batch['tasks']['masks']['mask'][available_cpu].to(device) >= 0.5
                    valid_masks = batch['tasks']['masks']['valid_mask'][available_cpu].to(device).bool()
                    mask_intersection += (predicted_masks & target_masks & valid_masks).sum()
                    mask_union += ((predicted_masks | target_masks) & valid_masks).sum()
            if training:
                if gradient_conflict_control == 'pcgrad':
                    semantic_loss = sum(
                        value for name, value in losses.items()
                        if (name not in raw_model.geometry_heads
                            and name not in raw_model.detector_task_names))
                    detector_parameters = [
                        parameter for group in optimizers['detector'].param_groups
                        for parameter in group['params']]
                    gradient_diagnostics = _pcgrad_step(
                        details['geometry_domain_losses'],
                        details['geometry_domain_counts'], detector_parameters,
                        semantic_loss, optimizers, world_size)
                elif set(optimizers) == {'detector', 'semantic'}:
                    geometry_loss = sum(
                        value for name, value in losses.items()
                        if (name in raw_model.geometry_heads
                            or name in raw_model.detector_task_names))
                    semantic_loss = sum(
                        value for name, value in losses.items()
                        if (name not in raw_model.geometry_heads
                            and name not in raw_model.detector_task_names))
                    # A single backward keeps DDP's reducer lifecycle intact.
                    # The parameter sets are disjoint, so each optimizer still
                    # receives gradients scaled only by its own AMP scaler.
                    (scalers['detector'].scale(geometry_loss)
                     + scalers['semantic'].scale(semantic_loss)).backward()
                    scalers['detector'].step(optimizers['detector'])
                    scalers['semantic'].step(optimizers['semantic'])
                    scalers['detector'].update()
                    scalers['semantic'].update()
                else:
                    scalers['all'].scale(loss).backward()
                    scalers['all'].step(optimizers['all'])
                    scalers['all'].update()
            total += loss.detach()
            if rank == 0:
                if tracker is not None and (step % log_interval == 0 or step + 1 == len(loader)):
                    phase = 'train' if training else 'val'
                    tracker.log({
                        'epoch': epoch + 1,
                        f'{phase}/batch': step + 1,
                        f'{phase}/loss': loss.detach().item(),
                        **{f'{phase}/{name}': value.detach().item()
                           for name, value in losses.items()},
                        **{f'{phase}_raw/{name}': value.detach().item()
                           for name, value in details['raw_losses'].items()},
                        **(gradient_diagnostics
                           if training and gradient_conflict_control == 'pcgrad'
                           else {}),
                    })
                progress.set_postfix({
                    'loss': f'{loss.detach().item():.3f}',
                    **{name: f'{value.detach().item():.2f}' for name, value in losses.items()},
                }, refresh=True)
    metrics = {}
    if evaluator is not None:
        if world_size > 1:
            gathered = [None] * world_size if rank == 0 else None
            dist.gather_object(evaluator.metrics.stats, gathered, dst=0)
            if rank == 0:
                for shard in gathered[1:]:
                    for name, values in shard.items():
                        evaluator.metrics.stats[name].extend(values)
                metrics = evaluator.compute()
        else:
            metrics = evaluator.compute()
    if not training:
        intersection = reduce_mean(mask_intersection, world_size) * world_size
        union = reduce_mean(mask_union, world_size) * world_size
        metrics['mask_iou'] = ((intersection / union).item() if union.item() else float('nan'))
    return reduce_mean(total / max(len(loader), 1), world_size).item(), metrics


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    distributed, local_rank, rank, world_size = distributed_context()
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', local_rank)
    config = load_config(args.config)
    train_config = config['train']
    epochs = args.epochs or int(train_config.get('epochs', 100))

    model = build_model(config).to(device)
    optimizer_options = {
        'lr': float(train_config.get('learning_rate', 2e-4)),
        'weight_decay': float(train_config.get('weight_decay', 5e-4)),
    }
    strict_detached = bool(train_config.get('strict_detached_optimization', False))
    gradient_conflict_control = train_config.get('gradient_conflict_control', 'none')
    if gradient_conflict_control not in {'none', 'pcgrad'}:
        raise ValueError(f'Unsupported gradient_conflict_control: '
                         f'{gradient_conflict_control}')
    if gradient_conflict_control != 'none' and not strict_detached:
        raise ValueError('Gradient conflict control currently requires '
                         'strict_detached_optimization=true')
    if strict_detached:
        if not model.detach_task_features:
            raise ValueError('strict_detached_optimization requires detach_task_features=true')
        semantic_parameters = (
            list(model.shared_task_tower.parameters())
            + list(model.task_heads.parameters())
            + [parameter
               for name, head in model.mask_heads.items()
               if name not in model.detector_task_names
               for parameter in head.parameters()])
        semantic_parameters = [parameter for parameter in semantic_parameters
                               if parameter.requires_grad]
        semantic_ids = {id(parameter) for parameter in semantic_parameters}
        detector_parameters = [
            parameter for parameter in model.parameters()
            if id(parameter) not in semantic_ids and parameter.requires_grad
        ]
        optimizers = {
            'detector': torch.optim.AdamW(detector_parameters, **optimizer_options),
            'semantic': torch.optim.AdamW(semantic_parameters, **optimizer_options),
        }
    else:
        optimizers = {'all': torch.optim.AdamW(model.parameters(), **optimizer_options)}
    early_stopping_metric = train_config.get('early_stopping_metric', 'validation_loss')
    maximize_metric = early_stopping_metric.endswith(('map50', 'map50_95'))
    start_epoch, best_validation, stale_epochs = 0, float('inf'), 0
    best_metric = -float('inf') if maximize_metric else float('inf')
    best_mask_iou = -float('inf')
    best_dataset_metrics = {}
    criterion = SparseMTLLoss(model, train_config.get('loss_balance')).to(device)
    if args.resume and args.init_weights:
        raise ValueError('--resume and --init-weights are mutually exclusive')
    if args.init_weights:
        checkpoint = load_initial_model_weights(model, args.init_weights)
        print(f'initialized model weights from {args.init_weights}; optimizer, scaler, epoch, '
              'and early-stopping state reset', flush=True)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        saved_optimizers = checkpoint.get('optimizers')
        if saved_optimizers is not None:
            if set(saved_optimizers) != set(optimizers):
                raise ValueError('Checkpoint optimizer layout does not match current config')
            for name, optimizer in optimizers.items():
                optimizer.load_state_dict(saved_optimizers[name])
        elif set(optimizers) == {'all'}:
            optimizers['all'].load_state_dict(checkpoint['optimizer'])
        else:
            raise ValueError('Strict-detached resume requires a strict-detached checkpoint')
        start_epoch = checkpoint['epoch'] + 1
        if 'criterion' in checkpoint:
            criterion.load_state_dict(checkpoint['criterion'])
            best_validation = checkpoint.get('best_validation', best_validation)
            if checkpoint.get('early_stopping_metric') == early_stopping_metric:
                best_metric = checkpoint.get('best_metric', best_metric)
                stale_epochs = checkpoint.get('stale_epochs', 0)
            else:
                print(f'early-stopping metric changed to {early_stopping_metric}; '
                      'resetting its best value and patience counter', flush=True)
            best_mask_iou = checkpoint.get('best_mask_iou', best_mask_iou)
            best_dataset_metrics = checkpoint.get('best_dataset_metrics', {})
        else:
            print('resume checkpoint predates loss balancing; resetting validation baseline',
                  flush=True)
    if distributed and gradient_conflict_control == 'none':
        # Sparse supervision means a batch may not exercise every task head.
        model = DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )

    train_dataset = make_dataset(args.train_folders, config, True)
    val_datasets = [make_dataset([folder], config, False) for folder in args.val_folders]
    val_names = [folder.parents[1].name for folder in args.val_folders]
    if len(set(val_names)) != len(val_names):
        raise ValueError(f'Validation dataset names must be unique, got {val_names}')
    train_sampler = make_train_sampler(
        train_dataset, train_config, distributed, rank, world_size, args.seed)
    val_samplers = [
        DistributedSampler(dataset, shuffle=False) if distributed else None
        for dataset in val_datasets
    ]
    batch_size = int(train_config.get('batch_size', 16)) // world_size
    if batch_size < 1:
        raise ValueError('Configured global batch size is smaller than the GPU count')
    loader_options = dict(
        batch_size=batch_size,
        num_workers=int(train_config.get('workers', 8)),
        pin_memory=device.type == 'cuda',
        collate_fn=sparse_mtl_collate,
        persistent_workers=int(train_config.get('workers', 8)) > 0,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed + rank),
    )
    train_loader = DataLoader(train_dataset, sampler=train_sampler,
                              shuffle=train_sampler is None, **loader_options)
    val_loaders = [
        DataLoader(dataset, sampler=sampler, shuffle=False, **loader_options)
        for dataset, sampler in zip(val_datasets, val_samplers)
    ]
    scalers = {
        name: torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
        for name in optimizers
    }
    if args.resume and 'scalers' in checkpoint:
        if set(checkpoint['scalers']) != set(scalers):
            raise ValueError('Checkpoint AMP scaler layout does not match current config')
        for name, scaler in scalers.items():
            scaler.load_state_dict(checkpoint['scalers'][name])
    validation_interval = int(train_config.get('validation_interval', 1))
    early_stopping_patience = int(train_config.get('early_stopping_patience', 0))
    early_stopping_min_delta = float(train_config.get('early_stopping_min_delta', 0.0))
    tracker = None
    if rank == 0:
        args.out_folder.mkdir(parents=True, exist_ok=True)
        print(f'train images={len(train_dataset)}, '
              f'val images={sum(map(len, val_datasets))} ({dict(zip(val_names, map(len, val_datasets))) }), '
              f'world_size={world_size}, global_batch={batch_size * world_size}', flush=True)
        if args.wandb_project:
            import wandb
            tracker = wandb.init(
                project=args.wandb_project,
                group=args.wandb_group,
                name=args.wandb_name,
                notes=args.wandb_notes,
                id=args.wandb_id,
                resume='allow',
                config={
                    **config,
                    'train_images': len(train_dataset),
                    'val_images': sum(map(len, val_datasets)),
                    'val_images_by_dataset': dict(zip(val_names, map(len, val_datasets))),
                    'world_size': world_size,
                    'global_batch_size': batch_size * world_size,
                    'seed': args.seed,
                },
                dir=str(args.out_folder),
            )

    for epoch in range(start_epoch, epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        train_loss, _ = run_epoch(model, criterion, train_loader, device, optimizers,
                                  scalers, epoch, rank, world_size, tracker,
                                  gradient_conflict_control=gradient_conflict_control)
        validation_loss = None
        validation_metrics = {}
        if (epoch + 1) % validation_interval == 0:
            dataset_losses = []
            dataset_metrics = {}
            for dataset_name, val_loader in zip(val_names, val_loaders):
                dataset_loss, metrics = run_epoch(
                    model, criterion, val_loader, device, None, scalers, epoch, rank,
                    world_size, tracker, map_options=train_config.get('map'))
                dataset_losses.append(dataset_loss)
                dataset_metrics[dataset_name] = metrics
            validation_loss = sum(dataset_losses) / len(dataset_losses)
            if len(dataset_metrics) == 1:
                validation_metrics = next(iter(dataset_metrics.values()))
            else:
                for dataset_name, metrics in dataset_metrics.items():
                    validation_metrics.update({
                        f'{dataset_name}_{metric}': value
                        for metric, value in metrics.items()
                    })
                # Detection statistics are gathered to rank zero. Other ranks
                # deliberately have no mAP dictionary and only participate in
                # the synchronized early-stopping broadcast below.
                if rank == 0:
                    for metric in ('map50', 'map50_95'):
                        values = [metrics[metric] for metrics in dataset_metrics.values()]
                        validation_metrics[f'mean_{metric}'] = sum(values) / len(values)
        should_stop = torch.zeros(1, dtype=torch.uint8, device=device)
        if rank == 0:
            raw_model = (model.module
                         if isinstance(model, DistributedDataParallel) else model)
            if validation_loss is not None:
                best_validation = min(best_validation, validation_loss)
            current_metric = (validation_metrics.get(early_stopping_metric)
                              if maximize_metric else validation_loss)
            if maximize_metric:
                is_best = (current_metric is not None and
                           current_metric > best_metric + early_stopping_min_delta)
            else:
                is_best = (current_metric is not None and
                           current_metric < best_metric - early_stopping_min_delta)
            if is_best:
                best_metric = current_metric
                stale_epochs = 0
            elif current_metric is not None:
                stale_epochs += 1
            current_mask_iou = validation_metrics.get('mask_iou')
            is_best_iou = (current_mask_iou is not None and
                           current_mask_iou > best_mask_iou)
            if is_best_iou:
                best_mask_iou = current_mask_iou
            dataset_improvements = {}
            for dataset_name in val_names:
                metric_name = f'{dataset_name}_map50_95'
                metric_value = validation_metrics.get(metric_name)
                if metric_value is not None and metric_value > best_dataset_metrics.get(
                        dataset_name, -float('inf')):
                    best_dataset_metrics[dataset_name] = metric_value
                    dataset_improvements[dataset_name] = True
            state = {
                'epoch': epoch,
                'model': raw_model.state_dict(),
                'optimizers': {
                    name: optimizer.state_dict() for name, optimizer in optimizers.items()
                },
                'scalers': {name: scaler.state_dict() for name, scaler in scalers.items()},
                'criterion': criterion.state_dict(),
                'train_loss': train_loss,
                'validation_loss': validation_loss,
                'best_validation': best_validation,
                'best_metric': best_metric,
                'best_mask_iou': best_mask_iou,
                'best_dataset_metrics': best_dataset_metrics,
                'early_stopping_metric': early_stopping_metric,
                'validation_metrics': validation_metrics,
                'stale_epochs': stale_epochs,
                'config': config,
            }
            torch.save(state, args.out_folder / 'last.pt')
            if is_best:
                torch.save(state, args.out_folder / 'best.pt')
            if is_best_iou:
                torch.save(state, args.out_folder / 'best_iou.pt')
            for dataset_name in dataset_improvements:
                torch.save(state, args.out_folder / f'best_{dataset_name}.pt')
            print(f'epoch={epoch + 1} train={train_loss:.5f} '
                  f'validation={validation_loss if validation_loss is not None else "skipped"} '
                  f'map50={validation_metrics.get("map50", "skipped")} '
                  f'map50_95={validation_metrics.get("map50_95", "skipped")} '
                  f'mean_map50_95={validation_metrics.get("mean_map50_95", "skipped")} '
                  f'mask_iou={validation_metrics.get("mask_iou", "skipped")}', flush=True)
            if tracker is not None:
                epoch_log = {
                    'epoch': epoch + 1,
                    'epoch/train_loss': train_loss,
                    'epoch/validation_loss': validation_loss,
                    'epoch/best_validation_loss': best_validation,
                    'epoch/val_map50': validation_metrics.get('map50'),
                    'epoch/val_map50_95': validation_metrics.get('map50_95'),
                    **{f'epoch/val_{name}': value for name, value in validation_metrics.items()
                       if name.endswith(('map50', 'map50_95'))},
                    'epoch/val_mask_iou': validation_metrics.get('mask_iou'),
                    'epoch/best_early_stopping_metric': best_metric,
                    'epoch/best_mask_iou': best_mask_iou,
                    'epoch/early_stopping_stale_epochs': stale_epochs,
                }
                for task_name, head in raw_model.task_heads.items():
                    if hasattr(head, 'gates'):
                        for level, value in enumerate(head.gates().detach().cpu().tolist(), 3):
                            epoch_log[f'epoch/task_gate/{task_name}/p{level}'] = value
                for family_name, adapter in raw_model.geometry_feature_adapters.items():
                    for level, value in enumerate(
                            adapter.gates().detach().cpu().tolist(), 3):
                        epoch_log[f'epoch/detection_gate/{family_name}/p{level}'] = value
                for domain_name, adapter in raw_model.geometry_domain_adapters.items():
                    for level, value in enumerate(
                            adapter.gates().detach().cpu().tolist(), 3):
                        epoch_log[f'epoch/domain_gate/{domain_name}/p{level}'] = value
                tracker.log(epoch_log)
            if (early_stopping_patience > 0 and validation_loss is not None and
                    stale_epochs >= early_stopping_patience):
                print(f'early stopping at epoch={epoch + 1}: no improvement greater than '
                      f'{early_stopping_min_delta:g} in {early_stopping_metric} for '
                      f'{stale_epochs} validations', flush=True)
                should_stop.fill_(1)
        if distributed:
            # Rank zero owns checkpoint selection and the patience counter, but
            # every DDP rank must leave the epoch loop together. Otherwise the
            # remaining ranks enter the next all-reduce and eventually time out.
            dist.broadcast(should_stop, src=0)
        if should_stop.item():
            break

    if tracker is not None:
        tracker.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
