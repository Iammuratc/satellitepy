"""
This script is to fix the naming issues in fineair, e.g., CRJ2 >> CRJ-200.
"""

import configargparse
from pathlib import Path
import json
from collections import Counter
import logging

from satellitepy.utils.path_utils import create_folder, init_logger, get_default_log_path, get_default_log_config, get_file_paths
from satellitepy.data.labels import read_label, read_fineair_label
from satellitepy.data.utils import get_satellitepy_dict_values, count_unique_values


def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-label-folder', type=Path, required=True)
    parser.add_argument('--log-config-path', default=None, type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, default=None, help='Log path.')
    parser.add_argument('--out-folder', type=Path, required=True, help="Output folder.")
    return parser


def run(parser):
    """Application entry point."""
    args = parser.parse_args()

    out_folder = Path(args.out_folder)
    assert create_folder(out_folder)
    out_label_folder = out_folder / 'labels'
    assert create_folder(out_label_folder)

    log_config_path = get_default_log_config() if args.log_config_path == None else Path(args.log_config_path)
    log_path = get_default_log_path(Path(__file__).resolve().stem) if args.log_path == None else Path(args.log_path)

    init_logger(config_path=log_config_path, log_path=log_path)

    logger = logging.getLogger('')
    logger.info('Saving patches from original images...')
    logger.info(f'Log will be stored at: {log_path}')

    config_path = out_folder / f"{Path(__file__).resolve().stem}.ini"
    parser.write_config_file(args, [str(config_path)])
    logger.info(f'Configs will be stored at {config_path}')

    in_label_folder = Path(args.in_label_folder)
    fix_naming(
        in_label_folder,
        logger=logger,
        out_label_folder=out_label_folder)



def fix_naming(in_label_folder,logger,out_label_folder):

    label_paths = get_file_paths(in_label_folder)

    for label_path in label_paths:
        with open(label_path, 'r') as f:
            labels = json.load(f)
        logger.info(f"Processing {label_path.stem}")
        for label in labels["features"]:
            fgc = label["properties"]["Type"]
            ftgc = label["properties"]["Subtype"]
            # List the naming changes here:
            if fgc=='CRJ' and ftgc=='900LR':
                label["properties"]["Type"] = 'CRJ-900'
                label["properties"]["SubType"] = 'LR'
            elif fgc=='CRJ' and ftgc=='900ER':
                label["properties"]["Type"] = 'CRJ-900'
                label["properties"]["SubType"] = 'ER'
            elif fgc=='CRJ' and ftgc=='900':
                label["properties"]["Type"] = 'CRJ-900'
                label["properties"]["SubType"] = None
            elif fgc=='CRJ9' and ftgc=='LR':
                label["properties"]["Type"] = 'CRJ-900'
                label["properties"]["SubType"] = 'LR'
            elif fgc=='CRJ9' and ftgc==None:
                label["properties"]["Type"] = 'CRJ-900'
                label["properties"]["SubType"] = None
            elif fgc=='CRJ' and ftgc=='701ER':
                label["properties"]["Type"] = 'CRJ-700'
                label["properties"]["SubType"] = '701ER'
            elif fgc=='CRJ7' and ftgc=='500':
                label["properties"]["Type"] = 'CRJ-700'
                label["properties"]["SubType"] = '500'
            elif fgc=='CRJ7' and ftgc=='SE':
                label["properties"]["Type"] = 'CRJ-700'
                label["properties"]["SubType"] = 'SE'
            elif fgc=='CRJ' and ftgc=='550':
                label["properties"]["Type"] = 'CRJ-550'
                label["properties"]["SubType"] = None
            elif fgc=='CRJ' and ftgc=='200LR':
                label["properties"]["Type"] = 'CRJ-200'
                label["properties"]["SubType"] = 'LR'
            elif fgc=='CRJ' and ftgc=='200ER':
                label["properties"]["Type"] = 'CRJ-200'
                label["properties"]["SubType"] = 'ER'
            elif fgc=='CRJ' and ftgc=='200':
                label["properties"]["Type"] = 'CRJ-200'
                label["properties"]["SubType"] = None
            elif fgc=='CRJ2' and ftgc=='200LR':
                label["properties"]["Type"] = 'CRJ-200'
                label["properties"]["SubType"] = 'LR'
            elif fgc=='CRJ2' and ftgc==None:
                label["properties"]["Type"] = 'CRJ-200'
                label["properties"]["SubType"] = None
            elif fgc=='CRJ' and ftgc=='1000':
                label["properties"]["Type"] = 'CRJ-1000'
                label["properties"]["SubType"] = None
            ############
            elif fgc=='ERJ' and ftgc=='145XR':
                label["properties"]["Type"] = 'E145'
                label["properties"]["SubType"] = 'XR'
            elif fgc=='ERJ' and ftgc=='145':
                label["properties"]["Type"] = 'E145'
                label["properties"]["SubType"] = None
            elif fgc=='ERJ' and ftgc==None:
                logger.error('FIX THIS LATER!')  
                label["properties"]["Type"] = 'E145'
                label["properties"]["SubType"] = None
            elif fgc=='R175' and ftgc=='LR':
                label["properties"]["Type"] = 'E175'
                label["properties"]["SubType"] = 'LR'
            ############
            elif fgc=='Cessna' and ftgc=='560':
                label["properties"]["Type"] = 'Cessna560'
                label["properties"]["SubType"] = None
            ###########
            elif fgc=='B752' and ftgc=='232':
                label["properties"]["Type"] = 'B757'
                label["properties"]["SubType"] = '232'
            ###########
            elif fgc=='Beechcraft' and ftgc=='400XP':
                label["properties"]["Type"] = 'BE40'
                label["properties"]["SubType"] = '400XP'
            elif fgc=='Bombardier Global' and ftgc=='6000':
                label["properties"]["Type"] = 'Bombardier-Global'
                label["properties"]["SubType"] = '6000'
                

        ## Save the new labels
        out_label_path = str(out_label_folder / label_path.name)
        with open(out_label_path, 'w') as file:
            json.dump(labels, file, indent=4)

if __name__ == '__main__':
    args = get_args()
    run(args)
