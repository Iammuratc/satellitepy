import logging
from pathlib import Path
from satellitepy.utils.path_utils import create_folder, init_logger, get_project_folder
from satellitepy.data.cutout.cutout import Cutout
import configargparse

def get_args():
    """Arguments parser."""
    project_folder = get_project_folder()

    parser = configargparse.ArgumentParser(description = __doc__)
    parser.add_argument('--src-label-format', type=str, required=True,
                        help='Label file format. e.g., dota, fair1m')
    parser.add_argument('--src-folder', type=Path, required=True,
                        help='Input folder which contains the images.')
    parser.add_argument('--out-folder', type=Path, required=True,
                        help='Save folder for chips. Images will be saved in <output-folder>/images and corresponding labels in <output-folder>/labels_satellitepy')
    parser.add_argument('--margin-size', type=Path, required=True,
                        help='Margin size of the chip to be created.')
    parser.add_argument('--log-config-path', default=project_folder /
                    Path("configs/log.config"), type=Path, help='Log config file.')
    
    args = parser.parse_args()
    return args

def run(args):
    src_label_format = args.src_label_format
    src_folder = Path(args.src_folder)
    
    src_folder_train = src_folder / Path("train")
    src_folder_train_images = src_folder_train / Path("img/images")
    src_folder_train_labels = src_folder_train / Path("labels_hbb")

    src_folder_val = src_folder / Path("val")
    src_folder_val_images = src_folder_val / Path("img/images")
    src_folder_val_labels = src_folder_val / Path("labels_hbb")

    out_folder = Path(args.out_folder)

    out_folder_train = out_folder / Path("train/cutouts")
    out_folder_train_images = out_folder_train / Path("images")
    out_folder_train_orthogonal_image = out_folder_train / Path("orthogonal_images")
    out_folder_train_orthogonal_padded_image = out_folder_train / Path("orthogonal_images_unet_padded")
    out_folder_train_labels = out_folder_train / Path("labels")
    
    out_folder_val = out_folder / Path("val/cutouts")
    out_folder_val_images = out_folder_val / Path("images")
    out_folder_val_orthogonal_image = out_folder_val / Path("orthogonal_images")
    out_folder_val_orthogonal_padded_image = out_folder_val / Path("orthogonal_images_unet_padded")
    out_folder_val_labels = out_folder_val / Path("labels")

    assert create_folder(out_folder)
    assert create_folder(out_folder_train)
    assert create_folder(out_folder_train_images)
    assert create_folder(out_folder_train_orthogonal_image)
    assert create_folder(out_folder_train_orthogonal_padded_image)
    assert create_folder(out_folder_train_labels)
    assert create_folder(out_folder_val)
    assert create_folder(out_folder_val_images)
    assert create_folder(out_folder_val_orthogonal_image)
    assert create_folder(out_folder_val_orthogonal_padded_image)
    assert create_folder(out_folder_val_labels)
    
    margin_size = Path(args.margin_size)

    log_path = Path(out_folder) / 'chip_creation.log'
    init_logger(config_path=args.log_config_path, log_path=log_path)
    logger = logging.getLogger(__name__)

    logger.info(
        f'No log path is given, the default log path will be used: {log_path}')
    logger.info(f'Saving chips with margin-size: {margin_size}')

    settings = {
        'project_folder' : get_project_folder(),
        'dataset_name' : src_label_format,
        'tasks' : ['bbox'],
        'dataset_parts' : ['train', 'val'],
        'instance_names' : {
            'Boeing787' : 1, # fair1m dataset
            'Boeing737' : 2, # fair1m dataset
            'Boeing747' : 3, # fair1m dataset
            'Boeing777' : 4, # fair1m dataset
            'A220' : 5, # fair1m dataset
            'A321' : 6, # fair1m dataset
            'A330' : 7, # fair1m dataset 
            'A350' : 8, # fair1m dataset
            'ARJ21' : 9, # fair1m dataset
            'C919' : 10, # fair1m dataset
            'other-airplane' : 11, # ?
            'plane' : 12 # this is for the dota dataset
        },
        'original' : {
            'train' : {
                'base_folder' : src_folder_train,
                'image_folder' : src_folder_train_images,
                'bounding_box_folder' : src_folder_train_labels
            },
            'val' : {
                'base_folder' : src_folder_val,
                'image_folder' : src_folder_val_images,
                'bounding_box_folder' : src_folder_val_labels
            }
        },
        'cutout' : {
            'train' : {
                'root_folder' : out_folder_train,
                'image_folder': out_folder_train_images,
                'orthogonal_image_folder': out_folder_train_orthogonal_image,
                'orthogonal_zoomed_image_folder': out_folder_train_orthogonal_padded_image,
                'label_folder': out_folder_train_labels
            },
            'val' : {
                'root_folder' : out_folder_val,
                'image_folder': out_folder_val_images,
                'orthogonal_image_folder': out_folder_val_orthogonal_image,
                'orthogonal_zoomed_image_folder': out_folder_val_orthogonal_padded_image,
                'label_folder': out_folder_val_labels
            }
        },
        'bbox_rotation' : 'clockwise'
    }

    cutout = Cutout(settings, 'val')
    cutout.get_cutouts(save = True, plot = False, multi_process = False)


if __name__ == "__main__":
    args = get_args()
    run(args)
