import configargparse
import random
from pathlib import Path
from satellitepy.utils.path_utils import create_folder, init_logger, get_default_log_path, get_default_log_config
from satellitepy.data.tools import copy_files
import logging
import logging.config

def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-image-folder', type=Path, required=True,
                        help='Folder of original images. The images in this folder will be processed.')
    parser.add_argument('--in-label-folder', type=Path, required=True,
                        help='Folder of original labels. The labels in this folder will be used to create patches.')
    parser.add_argument('--in-mask-folder', type=Path, required=False,
                        help='Folder of original masks. The masks in this folder will be processed.')
    parser.add_argument('--train-out-folder', type=Path, required=True,
                        help='Save folder of train set. Images and corresponding labels will be saved into '
                             '<out-folder>/images and <out-folder>/labels.')
    parser.add_argument('--val-out-folder', type=Path, required=True,
                        help='Save folder of val set. Images and corresponding labels will be saved into '
                             '<out-folder>/images and <out-folder>/labels.')
    parser.add_argument('--test-out-folder', type=Path, required=True,
                        help='Save folder of test set. Images and corresponding labels will be saved into '
                             '<out-folder>/images and <out-folder>/labels.')
    parser.add_argument('--randomise', type=bool, required=False, default=False, 
                        help='Determines if the files should be randomised before copying them to new locations. Default=False')
    parser.add_argument('--val-split', type=float, required=False, default=0.1,
                        help='Which percantage of the split should be used for val. Default=0.1')
    parser.add_argument('--test-split', type=float, required=False, default=0.2,
                        help='Which percantage of the split should be used for test. Default=0.2')
    parser.add_argument('--train-split', type=float, required=False, default=0.7,
                        help='Which percantage of the split should be used for train. Default=0.7')
    parser.add_argument('--log-config-path', default=None, type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, default=None, help='Log path.')
    
    return parser


def run(parser):
    args = parser.parse_args()
    log_config_path = get_default_log_config() if args.log_config_path == None else Path(args.log_config_path)
    log_path = get_default_log_path(Path(__file__).resolve().stem) if args.log_path == None else Path(args.log_path)
    init_logger(config_path=log_config_path, log_path=log_path)

    assert args.val_split + args.test_split + args.train_split == 1, 'Percentages of val, test and train splits are not equal to 1!'

    src_image_folder = Path(args.in_image_folder)
    src_label_folder = Path(args.in_label_folder)
    
    target_train_image_folder = Path(args.train_out_folder)/'images/'
    target_train_label_folder = Path(args.train_out_folder)/'labels/'
    target_test_image_folder = Path(args.test_out_folder)/'images/'
    target_test_label_folder = Path(args.test_out_folder)/'labels/'
    target_val_image_folder = Path(args.val_out_folder)/'images/'
    target_val_label_folder = Path(args.val_out_folder)/'labels/'

    assert create_folder(target_train_image_folder)
    assert create_folder(target_train_label_folder)
    assert create_folder(target_test_image_folder)
    assert create_folder(target_test_label_folder)
    assert create_folder(target_val_image_folder)
    assert create_folder(target_val_label_folder)
    
    image_files = sorted([f for f in src_image_folder.iterdir() if f.is_file()])
    annotation_files = sorted([f for f in src_label_folder.iterdir() if f.is_file()])
    
    if args.in_mask_folder:
        src_mask_folder = Path(args.in_mask_folder)
        mask_files = sorted([f for f in src_mask_folder.iterdir() if f.is_file()])
        target_val_mask_folder = Path(args.val_out_folder)/'masks/'
        target_train_mask_folder = Path(args.train_out_folder)/'masks/'
        target_test_mask_folder = Path(args.test_out_folder)/'masks/'
        assert create_folder(target_val_mask_folder)
        assert create_folder(target_train_mask_folder)
        assert create_folder(target_test_mask_folder)
    else:
        mask_files = [None] * len(image_files)
    
    
    assert len(image_files) == len(annotation_files) == len(mask_files), "Number of image, annotation and mask files should be the same."

    # Shuffle files
    paired_files = list(zip(image_files, annotation_files, mask_files))
    if args.randomise:
        random.shuffle(paired_files)
    image_files, annotation_files, mask_files = zip(*paired_files)

    # Split files
    total_files = len(image_files)
    train_end = round(total_files * args.train_split)
    val_end = train_end + round(total_files * args.val_split)

    train_images, val_images, test_images = image_files[:train_end], image_files[train_end:val_end], image_files[val_end:]
    train_labels, val_labels, test_labels = annotation_files[:train_end], annotation_files[train_end:val_end], annotation_files[val_end:]
    train_masks, val_masks, test_masks = mask_files[:train_end], mask_files[train_end:val_end], mask_files[val_end:]

    copy_files(train_images, src_image_folder, target_train_image_folder)
    copy_files(train_labels, src_label_folder, target_train_label_folder)
    copy_files(val_images, src_image_folder, target_val_image_folder)
    copy_files(val_labels, src_label_folder, target_val_label_folder)
    copy_files(test_images, src_image_folder, target_test_image_folder)
    copy_files(test_labels, src_label_folder, target_test_label_folder)
    
    if args.in_mask_folder:
        copy_files(train_masks, src_mask_folder, target_train_mask_folder)
        copy_files(test_masks, src_mask_folder, target_test_mask_folder)
        copy_files(val_masks, src_mask_folder, target_val_mask_folder)
    

if __name__ == '__main__':
    args = get_args()
    run(args)