"""Prepare the all-class DOTA-v1.0 data used by the original BBAVector model."""

import argparse
from pathlib import Path

def matched(image_folder, label_folder=None):
    images = sorted(image_folder.glob('*.png'))
    if label_folder is None:
        return images, [None] * len(images)
    labels = {path.stem: path for path in label_folder.glob('*.txt')}
    pairs = [(image, labels[image.stem]) for image in images if image.stem in labels]
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dota-root', type=Path, default=Path('/data/DOTA'))
    parser.add_argument('--out-folder', type=Path,
                        default=Path('outputs/patches/DOTA_bbavector'))
    parser.add_argument('--workers', type=int)
    parser.add_argument('--scales', type=float, nargs='+', default=(1.0, 0.5),
                        choices=(1.0, 0.5))
    parser.add_argument('--splits', nargs='+', default=('trainval', 'test'),
                        choices=('train', 'val', 'trainval', 'test'))
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    train_images, train_labels = matched(args.dota_root/'train'/'images',
                                         args.dota_root/'train'/'bounding_boxes')
    val_images, val_labels = matched(args.dota_root/'val'/'images',
                                     args.dota_root/'val'/'bounding_boxes')
    test_images, test_labels = matched(args.dota_root/'test'/'images')
    print(f'train originals: {len(train_images)}; val originals: {len(val_images)}; '
          f'test originals: {len(test_images)}')
    if args.dry_run:
        return
    from satellitepy.data.tools import save_patches_from_paths

    # The authors used 600x600 DOTA crops at scales 1.0 and 0.5 with a
    # 100-pixel gap (500-pixel sliding-window stride), then resized to 608
    # inside the network. Prefixes prevent the two scales from colliding.
    for scale in args.scales:
        scale_prefix = '1' if scale == 1.0 else '0.5'
        for split, images, labels, keep_empty, source_names in (
            ('train', train_images, train_labels, False, None),
            ('val', val_images, val_labels, False, None),
            # Train and validation reuse source IDs. Include their source split
            # in combined-dataset names so patches cannot overwrite each other.
            ('trainval', train_images + val_images, train_labels + val_labels,
             False, ['train'] * len(train_images) + ['val'] * len(val_images)),
            ('test', test_images, test_labels, True, None),
        ):
            if split not in args.splits:
                continue
            destination = args.out_folder / split
            destination.mkdir(parents=True, exist_ok=True)
            prefixes = ([scale_prefix] * len(images) if source_names is None else
                        [f'{scale_prefix}__{source}' for source in source_names])
            save_patches_from_paths(
                images, labels, 'dota_all', destination,
                truncated_object_thr=0.5,
                patch_size=600,
                patch_overlap=100,
                rescaling=scale,
                keep_empty=keep_empty,
                include_only_cgc=None,
                workers=args.workers,
                name_prefixes=prefixes,
            )


if __name__ == '__main__':
    main()
