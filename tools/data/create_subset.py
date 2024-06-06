import os
import shutil
import configargparse
import random
from pathlib import Path
from satellitepy.utils.path_utils import create_folder

def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-image-folder', type=Path, required=True,
                        help='Folder of original images. The images in this folder will be processed.')
    parser.add_argument('--in-label-folder', type=Path, required=True,
                        help='Folder of original labels. The labels in this folder will be used to create patches.')
    parser.add_argument('--train-out-folder', type=Path, required=True,
                        help='Save folder of train set. Images and corresponding labels will be saved into '
                             '<out-folder>/images and <out-folder>/labels.')
    parser.add_argument('--val-out-folder', type=Path, required=True,
                        help='Save folder of val set. Images and corresponding labels will be saved into '
                             '<out-folder>/images and <out-folder>/labels.')
    parser.add_argument('--test-out-folder', type=Path, required=True,
                        help='Save folder of test set. Images and corresponding labels will be saved into '
                             '<out-folder>/images and <out-folder>/labels.')
    return parser

def copy_files(files, src_folder, dst_folder):
    for file in files:
        print(f'Copying {src_folder / file.name} to {dst_folder / file.name}')
        shutil.copy(src_folder / file.name, dst_folder / file.name)

def run(parser):
    args = parser.parse_args()

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

    assert len(image_files) == len(annotation_files), "Number of image files and annotation files should be the same."

    # Shuffle files
    paired_files = list(zip(image_files, annotation_files))
    random.shuffle(paired_files)
    image_files, annotation_files = zip(*paired_files)

    # Split files
    total_files = len(image_files)
    train_end = round(total_files * 0.7)
    val_end = train_end + round(total_files * 0.1)

    train_images, val_images, test_images = image_files[:train_end], image_files[train_end:val_end], image_files[val_end:]
    train_labels, val_labels, test_labels = annotation_files[:train_end], annotation_files[train_end:val_end], annotation_files[val_end:]

    copy_files(train_images, src_image_folder, target_train_image_folder)
    copy_files(train_labels, src_label_folder, target_train_label_folder)
    copy_files(val_images, src_image_folder, target_val_image_folder)
    copy_files(val_labels, src_label_folder, target_val_label_folder)
    copy_files(test_images, src_image_folder, target_test_image_folder)
    copy_files(test_labels, src_label_folder, target_test_label_folder)
    

if __name__ == '__main__':
    args = get_args()
    run(args)