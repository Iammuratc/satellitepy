import os
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
        for i,in_path in enumerate(in_paths):
            file_name = in_path.name
            out_file_name = f'{i}_{file_name}'
            out_path =  Path(out_folder) / out_file_name
            if not out_path.is_symlink():
                out_path.symlink_to(in_path)

    train_folders = [
        '/mnt/2tb-0/satellitepy/patches/fair1m/train/patch600_100/',
        # '/mnt/2tb-0/satellitepy/patches/ship_net/train/patch600_100/',
        # '/mnt/2tb-0/satellitepy/patches/DOTA/train/patch600_100/',
        '/mnt/2tb-0/satellitepy/patches/xview/train/patch600_100/',
        '/mnt/2tb-0/satellitepy/patches/DIOR/train/patch600_100/',
        '/mnt/2tb-0/satellitepy/patches/DIOR/test/patch600_100/',
        '/mnt/2tb-0/satellitepy/patches/Potsdam/patch600_100/',
        '/mnt/2tb-0/satellitepy/patches/Rareplanes/train/patch600_100/',
        '/mnt/2tb-0/satellitepy/patches/Rareplanes_synthetic_subset/train/',
        '/mnt/2tb-0/satellitepy/patches/ucas/CAR/patch600_100/',
        '/mnt/2tb-0/satellitepy/patches/ucas/PLANE/patch600_100/',
        # '/mnt/2tb-0/satellitepy/patches/VHR/patch600_100/',
        ]
    train_label_folders = [Path(train_folder) / 'labels' for train_folder in train_folders]
    train_image_folders = [Path(train_folder) / 'images' for train_folder in train_folders]
    


    val_folders = [
        '/mnt/2tb-0/satellitepy/patches/fair1m/val/patch600_100/',
        # '/mnt/2tb-0/satellitepy/patches/ship_net/val/patch600_100/',
        # '/mnt/2tb-0/satellitepy/patches/DOTA/val/patch600_100/',
        '/mnt/2tb-0/satellitepy/patches/DIOR/val/patch600_100/',
        '/mnt/2tb-0/satellitepy/patches/Rareplanes_synthetic_subset/test/',
        '/mnt/2tb-0/satellitepy/patches/Rareplanes/test/patch600_100/',
    ]
    val_label_folders = [Path(val_folder) / 'labels' for val_folder in val_folders]
    val_image_folders = [Path(val_folder) / 'images' for val_folder in val_folders]


    # Train 
    train_image_paths = get_file_paths_recursively(train_image_folders)
    train_label_paths = get_file_paths_recursively(train_label_folders)

    # Val
    val_image_paths = get_file_paths_recursively(val_image_folders)
    val_label_paths = get_file_paths_recursively(val_label_folders)

    train_image_out_folder = Path('/mnt/2tb-0/satellitepy/patches/all-noDOTAshipnetVHR/train/images')
    train_label_out_folder = Path('/mnt/2tb-0/satellitepy/patches/all-noDOTAshipnetVHR/train/labels')
    val_image_out_folder = Path('/mnt/2tb-0/satellitepy/patches/all-noDOTAshipnetVHR/val/images')
    val_label_out_folder = Path('/mnt/2tb-0/satellitepy/patches/all-noDOTAshipnetVHR/val/labels')

    assert create_folder(train_image_out_folder)
    assert create_folder(train_label_out_folder)
    assert create_folder(val_image_out_folder)
    assert create_folder(val_label_out_folder)

    save_symbolic_links(train_image_paths,out_folder=train_image_out_folder)  
    save_symbolic_links(train_label_paths,out_folder=train_label_out_folder)  
    save_symbolic_links(val_image_paths,out_folder=val_image_out_folder)
    save_symbolic_links(val_label_paths,out_folder=val_label_out_folder)  

if __name__ == '__main__':
    create_symbolic_paths()
    # args = parse_args()
    #     '/mnt/2tb-0/satellitepy/new-patches-murat/fair1m/train/images/',
    #     '/mnt/2tb-0/satellitepy/new-patches-murat/ship_net/train/images/',
    #     '/mnt/2tb-0/satellitepy/new-patches-murat/dota/train/images/',
    #     '/mnt/2tb-0/satellitepy/new-patches-murat/xview/train/images/',
    #     '/mnt/2tb-0/satellitepy/new-patches/DIOR/patch600_100/images/',
    #     '/mnt/2tb-0/satellitepy/new-patches/Potsdam/patch600_100/images/',
    #     '/mnt/2tb-0/satellitepy/new-patches/Rareplanes/patch600_100/images/',
    #     '/mnt/2tb-0/satellitepy/new-patches/Rareplanes_synthetic/patch600_100/images/',
    #     '/mnt/2tb-0/satellitepy/new-patches/ucas/patch600_100/images/'
    #     '/mnt/2tb-0/satellitepy/new-patches/VHR/patch600_100/images/',
    #     ]
    # train_dataset_names = [
    #     'fair1m',
    #     'ship_net',
    #     'dota',
    #     'xview',
    #     'DIOR',
    #     'Potsdam',
    #     'Rareplanes',
    #     'Rareplanes_synthetic_subset',
    #     'ucas',
    #     'VHR'
    # ]
    # Val 
    # val_dataset_names = [
    #     'fair1m',
    #     'ship_net',
    #     'dota',
    #     'DIOR',
    #     'Rareplanes_synthetic_subset'
    # ]