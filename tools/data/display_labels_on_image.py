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
    parser.add_argument('--image-path', type=Path, required=True,
                        help='Image that should be displayed')
    parser.add_argument('--label-path', type=Path, required=True,
                        help='Label that corresponds to the given image')
    parser.add_argument('--label-format', type=str, required=True,
                        help='Label file format. e.g., dota, fair1m.')
    parser.add_argument('--output-folder', type=Path, required=False, default= project_folder / Path('docs/labels_on_images/'),
                        help='Folder where the generated image should be saved to. Default: satellitepy/docs/labels_on_images')
    parser.add_argument('--show-bboxes', type=str, default='True', required=False,
                        help='Whether to show bounding boxes on the generated image')
    parser.add_argument('--log-config-path', default=project_folder /
                        Path("configs/log.config"), type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, default=None, help='Log file path.')
    args = parser.parse_args()
    return args


def run(args):
    image_path = Path(args.image_path)
    label_path = Path(args.label_path)
    label_format = str(args.label_format)
    output_folder = Path(args.output_folder)
    assert create_folder(output_folder)
    show_bboxes = args.show_bboxes.lower() == 'true'
    log_path = output_folder / f'display_labels.log' if args.log_path == None else args.log_path

    init_logger(config_path=args.log_config_path, log_path=log_path)
    logger = logging.getLogger(__name__)
    logger.info(
        f'No log path is given, the default log path will be used: {log_path}')
    
    logger.info(f'Displaying labels on {image_path.name}')

    show_labels_on_image(image_path,label_path,label_format,output_folder,show_bboxes)
    
if __name__ == '__main__':
    args = get_args()
    run(args)
