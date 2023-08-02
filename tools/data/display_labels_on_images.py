import configargparse
from pathlib import Path
from satellitepy.data.tools import show_labels_on_image
from satellitepy.utils.path_utils import create_folder, init_logger, get_project_folder
import logging
"""
Show labels (e.g., bounding boxes) on an image
"""

project_folder = get_project_folder()

def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-image-folder', type=Path, required=True,
                        help='Images that should be displayed')
    parser.add_argument('--in-label-folder', type=Path, required=True,
                        help='Labels that corresponds to the given images')
    parser.add_argument('--in-mask-folder', type=Path, required=False,
                        help='Path to masks')
    parser.add_argument('--label-format', type=str, required=True,
                        help='Label file format. e.g. dota, fair1m.')
    parser.add_argument('--out-folder', type=Path, required=True,
                        help='Folder where the generated image should be saved to.')
    parser.add_argument('--tasks', type=str, nargs='+',
                        help='Which information to show on generated images. E.g.: bboxes, masks, labels')
    parser.add_argument('--log-config-path', default=project_folder /
                        Path("configs/log.config"), type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, default=None, help='Log file path.')
    args = parser.parse_args()
    return args


def run(args):
    image_path = Path(args.in_image_folder)
    label_path = Path(args.in_label_folder)
    label_format = str(args.label_format)
    output_folder = Path(args.out_folder)
    
    assert create_folder(output_folder)
    
    tasks = args.tasks
    mask_path = args.in_mask_folder

    if mask_path != None:
        mask_path = Path(mask_path)
    log_path = output_folder / f'display_labels.log' if args.log_path == None else args.log_path

    init_logger(config_path=args.log_config_path, log_path=log_path)
    logger = logging.getLogger(__name__)
    logger.info(
        f'No log path is given, the default log path will be used: {log_path}')
    
    if 'masks' in tasks and mask_path == None:
        logger.error('No mask path given!')
        exit(1)

    logger.info(f'Displaying labels on {image_path.name}')

    show_labels_on_image(image_path,label_path,label_format,output_folder,tasks, mask_path)
    
if __name__ == '__main__':
    args = get_args()
    run(args)
