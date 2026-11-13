"""Create reproducible patches directly from the datasets in orig_data_paths.csv.

Splits are made at original-image level, before patch creation. Existing dataset
splits are retained; datasets without usable splits receive a seeded 80/10/10
split. VEDAI emits both RGB and four-channel RGB+IR variants.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff'}
DEFAULT_CLASSES = ('vehicle', 'ship', 'airplane')


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path
    mask: Path | None = None
    ir: Path | None = None


@dataclass(frozen=True)
class DatasetPart:
    name: str
    split: str | None
    label_format: str
    samples: tuple[Sample, ...]
    modality: str = 'rgb'


def _files(folder: Path, suffixes=None) -> list[Path]:
    suffixes = suffixes or IMAGE_SUFFIXES
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir()
                  if path.is_file() and path.suffix.lower() in suffixes)


def _match(image_folder: Path, label_folder: Path, label_suffixes,
           image_key=lambda path: path.stem,
           label_key=lambda path: path.stem) -> tuple[Sample, ...]:
    labels = {label_key(path): path for path in _files(label_folder, label_suffixes)}
    result = []
    for image in _files(image_folder):
        label = labels.get(image_key(image))
        if label is not None:
            result.append(Sample(image, label))
    return tuple(result)


def _parts(root: Path, dota_v2_root: Path | None = None) -> list[DatasetPart]:
    """Describe supported layouts without modifying the original datasets."""
    name = root.name.lower()
    parts: list[DatasetPart] = []
    if name == 'split' and root.parent.name.lower() == 'dior':
        for split in ('train', 'val', 'test'):
            samples = _match(root/split/'images', root/split/'labels', {'.xml'})
            if samples:
                parts.append(DatasetPart('dior', split, 'dior', samples))
    elif name == 'dota':
        for split in ('train', 'val'):
            label_folder = root / split / 'bounding_boxes'
            if dota_v2_root is not None:
                candidate = dota_v2_root / split / split / 'annotations' / 'version2.0'
                if candidate.is_dir():
                    label_folder = candidate
            samples = _match(root/split/'images', label_folder, {'.txt'})
            if samples:
                parts.append(DatasetPart('dota', split, 'dota', samples))
    elif name == 'fair1m':
        for split in ('train', 'val'):
            samples = _match(root/split/'images', root/split/'bounding_boxes', {'.xml'})
            if samples:
                parts.append(DatasetPart('fair1m', split, 'fair1m', samples))
    elif 'isprs_semantic_labeling' in name:
        samples = _match(root/'top', root/'gts_disconnected_components', IMAGE_SUFFIXES)
        parts.append(DatasetPart('potsdam-vaihingen', None, 'isprs', samples))
    elif name == 'train' and root.parent.name.lower() == 'rareplanes':
        samples = _match(root/'tiled_images_only', root/'split_labels'/'labels', {'.json'})
        parts.append(DatasetPart('rareplanes', None, 'rareplanes_real', samples))
    elif name == 'vedai':
        labels = {p.stem: p for p in _files(root/'Annotations1024', {'.txt'})}
        irs = {p.stem.removesuffix('_ir'): p for p in _files(root/'Vehicules1024')
               if p.stem.endswith('_ir')}
        samples = tuple(Sample(p, labels[p.stem.removesuffix('_co')],
                               ir=irs.get(p.stem.removesuffix('_co')))
                        for p in _files(root/'Vehicules1024') if p.stem.endswith('_co')
                        and p.stem.removesuffix('_co') in labels)
        parts.append(DatasetPart('vedai', None, 'vedai', samples, 'rgb'))
        parts.append(DatasetPart('vedai', None, 'vedai', samples, 'rgbir'))
    elif name == 'vhr-10-nwpu':
        samples = _match(root/'positive image set', root/'ground truth', {'.txt'})
        parts.append(DatasetPart('vhr10', None, 'vhr', samples))
    elif name == 'split' and root.parent.name.lower() == 'ship_image_net':
        for split in ('train', 'val', 'test'):
            samples = _match(root/split/'images', root/split/'labels', {'.xml'})
            if samples:
                parts.append(DatasetPart('ship-image-net', split, 'ship_net', samples))
    elif name == 'ucas_aod':
        samples = _match(root/'CAR', root/'CAR_labels', {'.txt'})
        samples += _match(root/'PLANE', root/'PLANE_labels', {'.txt'})
        parts.append(DatasetPart('ucas-aod', None, 'ucas', samples))
    elif name == 'xview':
        samples = _match(root/'train_images'/'train_images',
                         root/'train_labels'/'satpy_labels', {'.json'})
        parts.append(DatasetPart('xview', None, 'satellitepy', samples))
    return [part for part in parts if part.samples]


def _container_path(csv_path: Path, raw_path: str, data_root: Path) -> Path:
    path = Path(raw_path)
    try:
        relative = path.relative_to('/mnt/2tb-0/satellitepy/data')
    except ValueError:
        return path
    return data_root / relative


def read_manifest(csv_path: Path, data_root: Path,
                  dota_v2_root: Path | None = None) -> list[DatasetPart]:
    parts = []
    with csv_path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            parts.extend(_parts(_container_path(csv_path, row['path'], data_root),
                                dota_v2_root=dota_v2_root))
    return parts


def split_samples(samples: tuple[Sample, ...], seed: int,
                  ratios=(0.8, 0.1, 0.1)) -> dict[str, tuple[Sample, ...]]:
    """Deterministically split samples without depending on input ordering."""
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError('split ratios must sum to one')
    ordered = sorted(samples, key=lambda sample: hashlib.sha256(
        f'{seed}:{sample.image.stem}'.encode()).digest())
    count = len(ordered)
    train_end = round(count * ratios[0])
    val_end = train_end + round(count * ratios[1])
    return {
        'train': tuple(ordered[:train_end]),
        'val': tuple(ordered[train_end:val_end]),
        'test': tuple(ordered[val_end:]),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=Path('docs/orig_data_paths.csv'))
    parser.add_argument('--data-root', type=Path, default=Path('/data'))
    parser.add_argument(
        '--dota-v2-label-root', type=Path,
        default=Path('/output/dota_v2_annotations'),
        help='Extracted DOTA-v2 annotation root. Matching v1 image IDs use these relabelled annotations.')
    parser.add_argument('--out-folder', type=Path, default=Path('outputs/patches'))
    parser.add_argument('--datasets', nargs='*', help='Dataset names to process')
    parser.add_argument('--patch-size', type=int, default=1024)
    parser.add_argument('--patch-overlap', type=int, default=100)
    parser.add_argument('--truncated-object-thr', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--keep-empty', action='store_true')
    parser.add_argument('--keep-official-splits', action='store_true',
                        help='Retain existing splits instead of applying 80/10/10 to all labelled sources')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    requested = set(args.datasets or ())
    parts = read_manifest(args.manifest, args.data_root, args.dota_v2_label_root)
    parts = [part for part in parts if not requested or part.name in requested]
    if not parts:
        raise SystemExit('No matching labelled datasets were found')

    if not args.dry_run:
        # Keep manifest discovery usable on machines that do not have the
        # optional image-processing stack installed yet.
        from satellitepy.data.tools import save_patches_from_paths

    grouped = {}
    for part in parts:
        grouped.setdefault((part.name, part.modality, part.label_format), []).append(part)

    for (name, modality, label_format), dataset_parts in grouped.items():
        if args.keep_official_splits and all(part.split for part in dataset_parts):
            split_map = {part.split: part.samples for part in dataset_parts}
        else:
            samples = tuple(sample for part in dataset_parts for sample in part.samples)
            split_map = split_samples(samples, args.seed)
        for split, samples in split_map.items():
            if not samples:
                continue
            destination = args.out_folder / name / modality / split
            logging.info('%s/%s/%s: %d source images -> %s',
                         name, modality, split, len(samples), destination)
            if args.dry_run:
                continue
            destination.mkdir(parents=True, exist_ok=True)
            save_patches_from_paths(
                [sample.image for sample in samples],
                [sample.label for sample in samples],
                label_format,
                destination,
                args.truncated_object_thr,
                args.patch_size,
                args.patch_overlap,
                mask_paths=[sample.mask for sample in samples],
                ir_paths=([sample.ir for sample in samples]
                          if modality == 'rgbir' else None),
                keep_empty=args.keep_empty,
                include_only_cgc=DEFAULT_CLASSES,
                workers=args.workers,
            )


if __name__ == '__main__':
    main()
