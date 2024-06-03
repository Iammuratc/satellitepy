import os
import shutil
from pathlib import Path
from satellitepy.utils.path_utils import get_file_paths, create_folder

if __name__ == '__main__':
    src_image_folder = '/raid/userdata/j21k0907/Patches/images/'
    src_label_folder = '/raid/userdata/j21k0907/Patches/labels/'

    target_train_image_folder = '/raid/userdata/j21k0907/Patches/train/images/'
    target_train_label_folder = '/raid/userdata/j21k0907/Patches/train/labels/'

    target_val_image_folder = '/raid/userdata/j21k0907/Patches/val/images/'
    target_val_label_folder = '/raid/userdata/j21k0907/Patches/val/labels/'

    assert create_folder(Path(target_train_image_folder))
    assert create_folder(Path(target_train_label_folder))

    assert create_folder(Path(target_val_image_folder))
    assert create_folder(Path(target_val_label_folder))

    src_image_directory = os.fsencode(src_image_folder)

    val_cnt = 100

    for file in os.listdir(src_image_directory):
        val_cnt -= 1

        name = os.fsdecode(file).split('.')[0]

        src_image_file = os.path.join(src_image_folder, Path(name + '.png'))
        src_label_file = os.path.join(src_label_folder, Path(name + '.json'))

        if val_cnt >= 0:
            shutil.copy(src_image_file, target_val_image_folder)
            shutil.copy(src_label_file, target_val_label_folder)
        else:
            shutil.copy(src_image_file, target_train_image_folder)
            shutil.copy(src_label_file, target_train_label_folder)