#!/usr/bin/env python3
"""Train independent or shared target-prompted object-chip models."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import ResNet18_Weights, resnet18


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

TASKS = {
    "fineair": (
        "fine-class", "very-fine-class",
    ),
    "rareplanes": (
        "role", "attributes.engines.no-engines",
        "attributes.engines.propulsion", "attributes.fuselage.canards",
        "attributes.wings.wing-shape", "attributes.wings.wing-position",
        "attributes.tail.no-tail-fins",
    ),
    "fair1m": ("role", "fine-class"),
    "shipimagenet": ("role", "fine-class", "very-fine-class"),
}


def nested_value(record: dict, path: str):
    value = record.get("semantic_labels", {})
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def safe_name(value: str) -> str:
    return value.replace(".", "__").replace("-", "_")


class AircraftChipDataset(Dataset):
    def __init__(self, root: Path, split: str, datasets: set[str],
                 dataset_roots: dict[str, Path] | None = None,
                 max_per_dataset: int | None = None):
        self.root = root
        self.records = []
        roots = dataset_roots or {dataset: root for dataset in datasets}
        for dataset in sorted(datasets):
            dataset_records = []
            dataset_root = roots.get(dataset, root)
            for path in sorted((dataset_root / split / "labels").glob("*.json")):
                record = json.loads(path.read_text())
                if record["dataset"] != dataset:
                    continue
                record["_label_path"] = str(path)
                record["_root"] = str(dataset_root)
                dataset_records.append(record)
            if max_per_dataset is not None:
                dataset_records = dataset_records[:max_per_dataset]
            self.records.extend(dataset_records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        root = Path(record.get("_root", self.root))
        image = cv2.imread(str(root / record["image"]), cv2.IMREAD_COLOR)
        prompt = cv2.imread(str(root / record["prompt"]), cv2.IMREAD_GRAYSCALE)
        if image is None or prompt is None:
            raise ValueError(f"Unreadable sample {record['id']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - np.asarray([0.485, 0.456, 0.406], np.float32)) / np.asarray(
            [0.229, 0.224, 0.225], np.float32)
        tensor = torch.from_numpy(np.moveaxis(image, -1, 0))
        tensor = torch.cat((tensor, torch.from_numpy(prompt[None].astype(np.float32) / 255)), 0)
        target_mask = ignore_mask = None
        if record.get("target_mask"):
            target_mask = cv2.imread(str(root / record["target_mask"]),
                                     cv2.IMREAD_GRAYSCALE)
            ignore_mask = cv2.imread(str(root / record["ignore_mask"]),
                                     cv2.IMREAD_GRAYSCALE)
            target_mask = torch.from_numpy(target_mask[None].astype(np.float32) / 255)
            ignore_mask = torch.from_numpy(ignore_mask[None].astype(np.float32) / 255)
        return tensor, record, target_mask, ignore_mask


def collate(items):
    images, records, target_masks, ignore_masks = zip(*items)
    return torch.stack(images), list(records), list(target_masks), list(ignore_masks)


def build_vocabs(dataset: AircraftChipDataset) -> dict[str, list]:
    values = defaultdict(set)
    for record in dataset.records:
        for task in TASKS.get(record["dataset"], ()):
            value = nested_value(record, task)
            if value is not None:
                values[f"{record['dataset']}:{task}"].add(value)
    return {task: sorted(items, key=str) for task, items in sorted(values.items())}


class AircraftChipMTL(nn.Module):
    def __init__(self, vocabs: dict[str, list], mask_route: str, pretrained: bool):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        old = backbone.conv1
        backbone.conv1 = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride,
                                   old.padding, bias=False)
        with torch.no_grad():
            backbone.conv1.weight[:, :3] = old.weight
            backbone.conv1.weight[:, 3:].zero_()
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu,
                                  backbone.maxpool)
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4
        self.heads = nn.ModuleDict({safe_name(task): nn.Linear(512, len(classes))
                                    for task, classes in vocabs.items()})
        self.mask_route = mask_route
        if mask_route == "layer1":
            self.mask_decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 4, 2, 1))
        elif mask_route in {"early", "early_detached"}:
            self.mask_decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 4, 2, 1))
        elif mask_route == "layer3":
            self.mask_decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 4, 2, 1))
        elif mask_route == "full":
            self.mask_decoder = nn.Sequential(
                nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(16, 1, 4, 2, 1))

    def forward(self, image):
        x1 = self.layer1(self.stem(image))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        pooled = F.adaptive_avg_pool2d(x4, 1).flatten(1)
        logits = {task: head(pooled) for task, head in self.heads.items()}
        mask = None
        if self.mask_route == "layer1":
            mask = self.mask_decoder(x1)
        elif self.mask_route == "early":
            mask = self.mask_decoder(x2)
        elif self.mask_route == "early_detached":
            mask = self.mask_decoder(x2.detach())
        elif self.mask_route == "layer3":
            mask = self.mask_decoder(x3)
        elif self.mask_route == "full":
            mask = self.mask_decoder(x4)
        return logits, mask


def semantic_loss(logits, records, vocabs, device):
    losses = []
    for task, classes in vocabs.items():
        dataset, task_name = task.split(":", 1)
        indices, targets = [], []
        lookup = {value: index for index, value in enumerate(classes)}
        for index, record in enumerate(records):
            value = nested_value(record, task_name) if record["dataset"] == dataset else None
            if value in lookup:
                indices.append(index)
                targets.append(lookup[value])
        if indices:
            prediction = logits[safe_name(task)][torch.as_tensor(indices, device=device)]
            losses.append(F.cross_entropy(prediction,
                                          torch.as_tensor(targets, device=device)))
    return sum(losses) / max(len(losses), 1) if losses else torch.zeros((), device=device)


def segmentation_loss(prediction, target_masks, ignore_masks, device):
    indices = [i for i, target in enumerate(target_masks) if target is not None]
    if not indices or prediction is None:
        return torch.zeros((), device=device)
    pred = prediction[torch.as_tensor(indices, device=device)]
    target = torch.stack([target_masks[i] for i in indices]).to(device)
    ignore = torch.stack([ignore_masks[i] for i in indices]).to(device)
    valid = 1 - ignore
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp_min(1)
    probability = pred.sigmoid() * valid
    target = target * valid
    intersection = (probability * target).sum((1, 2, 3))
    dice = 1 - ((2 * intersection + 1) /
                (probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)).mean()
    return bce + dice


@torch.no_grad()
def evaluate(model, loader, vocabs, device):
    model.eval()
    task_counts = {task: Counter() for task in vocabs}
    mask_intersection = mask_union = mask_dice_num = mask_dice_den = 0.0
    for images, records, targets, ignores in loader:
        logits, masks = model(images.to(device))
        for task, classes in vocabs.items():
            dataset, task_name = task.split(":", 1)
            lookup = {value: index for index, value in enumerate(classes)}
            predictions = logits[safe_name(task)].argmax(1).cpu().tolist()
            for index, record in enumerate(records):
                value = nested_value(record, task_name) if record["dataset"] == dataset else None
                if value in lookup:
                    target = lookup[value]
                    predicted = predictions[index]
                    task_counts[task][("total", target)] += 1
                    task_counts[task][("predicted", predicted)] += 1
                    task_counts[task][("correct", target)] += int(predicted == target)
        if masks is not None:
            for index, target in enumerate(targets):
                if target is None:
                    continue
                pred = masks[index].sigmoid().cpu() >= 0.5
                truth = target >= 0.5
                valid = ignores[index] < 0.5
                pred, truth = pred & valid, truth & valid
                inter = float((pred & truth).sum())
                pred_sum, truth_sum = float(pred.sum()), float(truth.sum())
                mask_intersection += inter
                mask_union += float((pred | truth).sum())
                mask_dice_num += 2 * inter
                mask_dice_den += pred_sum + truth_sum
    metrics = {}
    for task, counts in task_counts.items():
        totals = [counts[("total", index)] for index in range(len(vocabs[task]))]
        correct = [counts[("correct", index)] for index in range(len(vocabs[task]))]
        metrics[f"{task}/accuracy"] = sum(correct) / max(sum(totals), 1)
        recalls = [right / total for right, total in zip(correct, totals) if total]
        metrics[f"{task}/macro_recall"] = sum(recalls) / max(len(recalls), 1)
        f1_scores = []
        for index, (right, total) in enumerate(zip(correct, totals)):
            if not total:
                continue
            predicted = counts[("predicted", index)]
            precision = right / predicted if predicted else 0.0
            recall = right / total
            f1_scores.append(2 * precision * recall / (precision + recall)
                             if precision + recall else 0.0)
        metrics[f"{task}/macro_f1"] = sum(f1_scores) / max(len(f1_scores), 1)
    if mask_dice_den:
        metrics["dota/mask_dice"] = mask_dice_num / mask_dice_den
        metrics["dota/mask_iou"] = mask_intersection / max(mask_union, 1)
    metrics["mean_primary"] = sum(
        value for key, value in metrics.items()
        if key.endswith("/macro_recall") or key == "dota/mask_dice") / max(
            sum(key.endswith("/macro_recall") or key == "dota/mask_dice"
                for key in metrics), 1)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path("/output/aircraft_semantic_chips/rgb_256_chip_split_seed42"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=("fineair", "rareplanes", "dota",
                                                            "fair1m", "shipimagenet"),
                        required=True)
    parser.add_argument("--fair1m-data", type=Path)
    parser.add_argument("--shipimagenet-data", type=Path)
    parser.add_argument("--dota-data", type=Path)
    parser.add_argument("--max-per-dataset-split", type=int,
                        help="Development-only deterministic sample limit")
    parser.add_argument("--mask-route",
                        choices=("none", "layer1", "early", "early_detached",
                                 "layer3", "full"), default="none")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-weight", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="aircraft-chip-mtl")
    parser.add_argument("--run-name")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-notes")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    datasets = set(args.datasets)
    roots = {
        "fair1m": args.fair1m_data or args.data,
        "shipimagenet": args.shipimagenet_data or args.data,
        "dota": args.dota_data or args.data,
        "fineair": args.data,
        "rareplanes": args.data,
    }
    train_data = AircraftChipDataset(args.data, "train", datasets, roots,
                                     args.max_per_dataset_split)
    val_data = AircraftChipDataset(args.data, "val", datasets, roots,
                                   args.max_per_dataset_split)
    test_data = AircraftChipDataset(args.data, "test", datasets, roots,
                                    args.max_per_dataset_split)
    missing = datasets - {record["dataset"] for record in train_data.records}
    if missing:
        raise ValueError(f"No training records found for: {sorted(missing)}")
    vocabs = build_vocabs(train_data)
    if not vocabs and args.mask_route == "none":
        raise ValueError("No supervised semantic task and mask route is disabled")
    counts = Counter(record["dataset"] for record in train_data.records)
    weights = [1 / counts[record["dataset"]] for record in train_data.records]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader = DataLoader(train_data, args.batch_size, sampler=sampler,
                              num_workers=args.workers, collate_fn=collate,
                              pin_memory=True, persistent_workers=False)
    loader_options = dict(batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, collate_fn=collate,
                          pin_memory=True, persistent_workers=False)
    val_loader, test_loader = DataLoader(val_data, **loader_options), DataLoader(test_data, **loader_options)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AircraftChipMTL(vocabs, args.mask_route, args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(vars(args) | {"data": str(args.data),
        "output": str(args.output), "vocabs": vocabs}, indent=2, default=str) + "\n")
    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.run_name,
                               group=args.wandb_group, notes=args.wandb_notes,
                               config=vars(args))
    except Exception as error:
        print(f"W&B disabled: {error}")
    best, stale, start_epoch = -math.inf, 0, 0
    last_path = args.output / "last.pt"
    if last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("vocabs") != vocabs:
            raise ValueError("Existing checkpoint vocabulary does not match this dataset")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        best = checkpoint["best"]
        stale = checkpoint["stale"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"resuming from epoch {start_epoch}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        for images, records, targets, ignores in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, masks = model(images.to(device, non_blocking=True))
                loss = semantic_loss(logits, records, vocabs, device)
                loss = loss + args.mask_weight * segmentation_loss(
                    masks, targets, ignores, device)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running += float(loss.detach())
        metrics = evaluate(model, val_loader, vocabs, device)
        metrics["train/loss"] = running / max(len(train_loader), 1)
        metrics["epoch"] = epoch
        print(json.dumps(metrics, sort_keys=True))
        if wandb_run:
            wandb_run.log(metrics, step=epoch)
        if metrics["mean_primary"] > best:
            best, stale = metrics["mean_primary"], 0
            torch.save({"model": model.state_dict(), "vocabs": vocabs,
                        "epoch": epoch, "metrics": metrics}, args.output / "best.pt")
        else:
            stale += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "vocabs": vocabs,
                    "epoch": epoch, "best": best, "stale": stale}, last_path)
        if stale >= args.patience:
            break
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, vocabs, device)
    (args.output / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2) + "\n")
    if wandb_run:
        wandb_run.log({f"test/{key}": value for key, value in test_metrics.items()})
        wandb_run.finish()
    print(json.dumps({"best_epoch": checkpoint["epoch"], "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
