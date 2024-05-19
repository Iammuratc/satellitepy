from pathlib import Path
from satellitepy.utils.path_utils import get_file_paths, create_folder


def create_symbolic_paths(add_dataset_name=True):
    def get_file_paths_recursively(folders):
        recursive_file_paths = []
        for folder in folders:
            file_paths = get_file_paths(folder)
            for file_path in file_paths:
                recursive_file_paths.append(file_path)
        return recursive_file_paths

    def save_symbolic_links(in_paths, out_folder):
        for i, in_path in enumerate(in_paths):
            file_name = in_path.name
            out_file_name = f'{i}_{file_name}'
            out_path = Path(out_folder) / out_file_name
            if not out_path.is_symlink():
                out_path.symlink_to(in_path)

    train_folders = [
        '/mnt/2tb-0/satellitepy/patches/fair1m/train/',
        '/mnt/2tb-0/satellitepy/patches/ship_net/train/',
        '/mnt/2tb-0/satellitepy/patches/DOTA/train/',
        '/mnt/2tb-0/satellitepy/patches/xview/train/',
        '/mnt/2tb-0/satellitepy/patches/DIOR/train/',
        '/mnt/2tb-0/satellitepy/patches/DIOR/test/',
        '/mnt/2tb-0/satellitepy/patches/Potsdam/',
        '/mnt/2tb-0/satellitepy/patches/Rareplanes/train/',
        '/mnt/2tb-0/satellitepy/patches/Rareplanes_synthetic_subset/train/',
        '/mnt/2tb-0/satellitepy/patches/ucas/CAR/',
        '/mnt/2tb-0/satellitepy/patches/ucas/PLANE/',
        '/mnt/2tb-0/satellitepy/patches/VHR/',
        '/mnt/2tb-0/satellitepy/patches/VEDAI/'
    ]
    train_label_folders = [Path(train_folder) / 'labels' for train_folder in train_folders]
    train_image_folders = [Path(train_folder) / 'images' for train_folder in train_folders]

    val_folders = [
        '/mnt/2tb-0/satellitepy/full_satpy/fair1m/val/',
        '/mnt/2tb-0/satellitepy/full_satpy/ship_net/val/',
        '/mnt/2tb-0/satellitepy/full_satpy/DOTA/val/',
        '/mnt/2tb-0/satellitepy/full_satpy/DIOR/val/',
        '/mnt/2tb-0/satellitepy/full_satpy/Rareplanes_synth_subset/test/',
        '/mnt/2tb-0/satellitepy/full_satpy/Rareplanes/test/',
    ]
    val_label_folders = [Path(val_folder) / 'labels' for val_folder in val_folders]
    val_image_folders = [Path(val_folder) / 'images' for val_folder in val_folders]

    train_image_paths = get_file_paths_recursively(train_image_folders)
    train_label_paths = get_file_paths_recursively(train_label_folders)

    val_image_paths = get_file_paths_recursively(val_image_folders)
    val_label_paths = get_file_paths_recursively(val_label_folders)

    train_image_out_folder = Path('/mnt/2tb-0/satellitepy/patches/all/train/images')
    train_label_out_folder = Path('/mnt/2tb-0/satellitepy/patches/all/train/labels')
    val_image_out_folder = Path('/mnt/2tb-0/satellitepy/patches/all/val/images')
    val_label_out_folder = Path('/mnt/2tb-0/satellitepy/patches/all/val/labels')

    assert create_folder(train_image_out_folder)
    assert create_folder(train_label_out_folder)
    assert create_folder(val_image_out_folder)
    assert create_folder(val_label_out_folder)

    save_symbolic_links(train_image_paths, out_folder=train_image_out_folder)
    save_symbolic_links(train_label_paths, out_folder=train_label_out_folder)
    save_symbolic_links(val_image_paths, out_folder=val_image_out_folder)
    save_symbolic_links(val_label_paths, out_folder=val_label_out_folder)


if __name__ == '__main__':
    create_symbolic_paths()
