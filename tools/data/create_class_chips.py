from pathlib import Path

import configargparse

from satellitepy.data.tools import save_class_chips
from satellitepy.utils.path_utils import get_project_folder, init_logger, create_folder

project_folder = get_project_folder()

def get_args():
    """Arguments parser."""

    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-label-format', type=str, required=True,
                        help='Label file format. e.g., dota, fair1m')
    parser.add_argument('--in-image-folder', type=Path, required=True,
                        help='Input folder which contains the images.')
    parser.add_argument('--in-label-folder', type=Path, required=True,
                        help='Input folder which contains the labels.')
    parser.add_argument('--task', type=str, default='fine-class', help='Task by which the chips are sorted. Default is fine-class.')
    parser.add_argument('--out-folder', type=Path, required=True,
                        help='Save folder for chips. Images will be saved in <output-folder>/<class-name>/images.')
    parser.add_argument('--log-config-path', default=project_folder /
                        Path("configs/log.config"), type=Path, help='Log config file.')
    parser.add_argument('--in-mask-folder', type=Path, required=False,
                        help='Folder of original mask images. The mask images in this folder will be used to set mask '
                             'pixel coordinates in out labels.')

    args = parser.parse_args()
    return args


def run(args):
    in_label_format = args.in_label_format
    in_image_folder = Path(args.in_image_folder)
    in_label_folder = Path(args.in_label_folder)

    if args.in_mask_folder is not None:
        in_mask_folder = Path(args.in_mask_folder)
    else:
        in_mask_folder = None

    out_folder = Path(args.out_folder)
    assert create_folder(out_folder)

    task = args.task

    log_path = Path(out_folder) / 'chip_creation.log'
    init_logger(config_path=args.log_config_path, log_path=log_path)

    save_class_chips(
        label_format=in_label_format,
        image_folder=in_image_folder,
        label_folder=in_label_folder,
        out_folder=out_folder,
        task=task,
        mask_folder=in_mask_folder
    )


if __name__ == '__main__':
    args = get_args()
    run(args)
