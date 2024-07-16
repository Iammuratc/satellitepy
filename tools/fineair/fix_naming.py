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

    fix_labels(in_label_folder,out_label_folder,logger)


def fix_labels(in_label_folder, out_label_folder, logger):
    label_paths = get_file_paths(in_label_folder)
    
    skipped_ftgc_names = []
    for label_path in label_paths:
        with open(label_path, 'r') as f:
            labels = json.load(f)
        logger.info(f"Processing {label_path.stem}")

        labels = fix_fgc(labels,logger)
        labels, skipped_names = fix_ftgc(labels)

        ## Save the new labels
        out_label_path = str(out_label_folder / label_path.name)
        with open(out_label_path, 'w') as file:
            json.dump(labels, file, indent=4)

        for name in skipped_names:
            skipped_ftgc_names.append(name)

    logger.info('The Full names below are skipped by the ftgc corrections!')
    for name in sorted(set(skipped_ftgc_names)):
        logger.info(name)


def fix_fgc(labels,logger):
    for label in labels["features"]:
        fgc = label["properties"]["Type"]
        ftgc = label["properties"]["Subtype"]
        # List the naming changes here:
        if fgc=='CRJ' and ftgc=='900LR':
            label["properties"]["Type"] = 'CRJ-900'
            label["properties"]["Subtype"] = 'LR'
        elif fgc=='CRJ' and ftgc=='900ER':
            label["properties"]["Type"] = 'CRJ-900'
            label["properties"]["Subtype"] = 'ER'
        elif fgc=='CRJ' and ftgc=='900':
            label["properties"]["Type"] = 'CRJ-900'
            label["properties"]["Subtype"] = None
        elif fgc=='CRJ9' and ftgc=='LR':
            label["properties"]["Type"] = 'CRJ-900'
            label["properties"]["Subtype"] = 'LR'
        elif fgc=='CRJ9' and ftgc==None:
            label["properties"]["Type"] = 'CRJ-900'
            label["properties"]["Subtype"] = None
        elif fgc=='CRJ' and ftgc=='701ER':
            label["properties"]["Type"] = 'CRJ-701'
            label["properties"]["Subtype"] = 'ER'
        elif fgc=='CRJ7' and ftgc=='500':
            label["properties"]["Type"] = 'CRJ-700'
            label["properties"]["Subtype"] = '500'
        elif fgc=='CRJ7' and ftgc=='SE':
            label["properties"]["Type"] = 'CRJ-700'
            label["properties"]["Subtype"] = 'SE'
        elif fgc=='CRJ' and ftgc=='550':
            label["properties"]["Type"] = 'CRJ-550'
            label["properties"]["Subtype"] = None
        elif fgc=='CRJ' and ftgc=='200LR':
            label["properties"]["Type"] = 'CRJ-200'
            label["properties"]["Subtype"] = 'LR'
        elif fgc=='CRJ' and ftgc=='200ER':
            label["properties"]["Type"] = 'CRJ-200'
            label["properties"]["Subtype"] = 'ER'
        elif fgc=='CRJ' and ftgc=='200':
            label["properties"]["Type"] = 'CRJ-200'
            label["properties"]["Subtype"] = None
        elif fgc=='CRJ2' and ftgc=='200LR':
            label["properties"]["Type"] = 'CRJ-200'
            label["properties"]["Subtype"] = 'LR'
        elif fgc=='CRJ2' and ftgc==None:
            label["properties"]["Type"] = 'CRJ-200'
            label["properties"]["Subtype"] = None
        elif fgc=='CRJ' and ftgc=='1000':
            label["properties"]["Type"] = 'CRJ-1000'
            label["properties"]["Subtype"] = None
        ############
        elif fgc=='Cessna560':
            label["properties"]["Type"] = 'Cessna-Citation'
            label["properties"]["Subtype"] = '560'
        elif fgc=='Cessna' and ftgc=='560':
            label["properties"]["Type"] = 'Cessna-Citation'
            label["properties"]["Subtype"] = '560'
        ###########
  
        ############
        elif fgc=='ERJ' and ftgc=='135':
            label["properties"]["Type"] = 'E135'
            label["properties"]["Subtype"] = None
        elif fgc=='ERJ' and ftgc=='145XR':
            label["properties"]["Type"] = 'E145'
            label["properties"]["Subtype"] = 'XR'
        elif fgc=='ERJ' and ftgc=='145':
            label["properties"]["Type"] = 'E145'
            label["properties"]["Subtype"] = None
        elif fgc=='ERJ' and ftgc==None:
            logger.error('FIX THIS LATER!')  
            label["properties"]["Type"] = 'E145'
            label["properties"]["Subtype"] = None
        elif fgc=='R175' and ftgc=='LR':
            label["properties"]["Type"] = 'E175'
            label["properties"]["Subtype"] = 'LR'
        elif fgc=='Embraer' and ftgc=='Praetor 500':
            label["properties"]["Type"] = 'Embraer-Praetor'
            label["properties"]["Subtype"] = '500'
        ############
        elif fgc=='B752' and ftgc=='232':
            label["properties"]["Type"] = 'B757'
            label["properties"]["Subtype"] = '232'
        ###########
        elif fgc=='Beechcraft' and ftgc=='400XP':
            label["properties"]["Type"] = 'BE40'
            label["properties"]["Subtype"] = '400XP'
        ###########
        elif fgc=='Bombardier Global' and ftgc=='6000':
            label["properties"]["Type"] = 'Bombardier-Global'
            label["properties"]["Subtype"] = '6000'
        elif fgc=='GLEX' and ftgc=='6000':
            label["properties"]["Type"] = 'Bombardier-Global'
            label["properties"]["Subtype"] = '6000'
        elif fgc=='Global' and ftgc=='5500':
            label["properties"]["Type"] = 'Bombardier-Global'
            label["properties"]["Subtype"] = '5500'
        elif fgc=='Global' and ftgc=='6500':
            label["properties"]["Type"] = 'Bombardier-Global'
            label["properties"]["Subtype"] = '6500'
        elif fgc=='Gulfstream' and ftgc=='G280':
            label["properties"]["Type"] = 'Gulfstream-Global'
            label["properties"]["Subtype"] = '280'
    return labels


def fix_ftgc(labels):
    skipped_names = []
    for label in labels["features"]:
        fgc = label["properties"]["Type"]
        ftgc = label["properties"]["Subtype"]
        full_name = f"{fgc}--{ftgc}"
        skip_fulltype = False
        if full_name.startswith('A220--1'):
            label["properties"]["Subtype"] = f'{fgc}--100'
        elif full_name.startswith('A220--3'):
            label["properties"]["Subtype"] = f'{fgc}--300'
        elif full_name.startswith('A319--1'):
            label["properties"]["Subtype"] = f'{fgc}--100'
        elif full_name.startswith('A320--2'):
            label["properties"]["Subtype"] = f'{fgc}--200'        
        elif full_name.startswith('A321--2'):
            label["properties"]["Subtype"] = f'{fgc}--200'        
        elif full_name.startswith('A330--2'):
            label["properties"]["Subtype"] = f'{fgc}--200'
        elif full_name.startswith('A330--3'):
            label["properties"]["Subtype"] = f'{fgc}--300'
        elif full_name.startswith('A330--9'):
            label["properties"]["Subtype"] = f'{fgc}--900'
        elif full_name.startswith('A350--9'):
            label["properties"]["Subtype"] = f'{fgc}--900'
        elif full_name.startswith('A380--9'):
            label["properties"]["Subtype"] = f'{fgc}--900'
        elif full_name.startswith('A380--8'):
            label["properties"]["Subtype"] = f'{fgc}--800'
        elif full_name.startswith('B717--2'):
            label["properties"]["Subtype"] = f'{fgc}--200'
        elif full_name.startswith('B737--7'):
            label["properties"]["Subtype"] = f'{fgc}--700'
        elif full_name.startswith('B737--8'):
            label["properties"]["Subtype"] = f'{fgc}--800'
        elif full_name.startswith('B737--9'):
            label["properties"]["Subtype"] = f'{fgc}--900'
        elif full_name.startswith('B737--M'): # Max 8, MAX9
            label["properties"]["Subtype"] = f'{fgc}--MAX{ftgc[-1]}'
            label["properties"]["Fulltype"] = f'{fgc}--MAX{ftgc[-1]}'
            skip_fulltype = True
        elif full_name.startswith('B747--4'):
            label["properties"]["Subtype"] = f'{fgc}--400'
        elif full_name.startswith('B747--8'):
            label["properties"]["Subtype"] = f'{fgc}--800'
        elif full_name.startswith('B757--2'):
            label["properties"]["Subtype"] = f'{fgc}--200'
        elif full_name.startswith('B757--3'):
            label["properties"]["Subtype"] = f'{fgc}--300'
        elif full_name.startswith('B767--2'):
            label["properties"]["Subtype"] = f'{fgc}--200'
        elif full_name.startswith('B767--3'):
            label["properties"]["Subtype"] = f'{fgc}--300'
        elif full_name.startswith('B767--4'):
            label["properties"]["Subtype"] = f'{fgc}--400'
        elif full_name.startswith('B777--2'):
            label["properties"]["Subtype"] = f'{fgc}--200'
        elif full_name.startswith('B777--3'):
            label["properties"]["Subtype"] = f'{fgc}--300'
        elif full_name.startswith('B777--F'):
            label["properties"]["Subtype"] = f'{fgc}--F'
        elif full_name.startswith('B777--M'): # Max 8, MAX9
            label["properties"]["Subtype"] = f'{fgc}--MAX{ftgc[-1]}'
            label["properties"]["Fulltype"] = f'{fgc}--MAX{ftgc[-1]}'
            skip_fulltype = True
        elif full_name.startswith('B787--10'):
            label["properties"]["Subtype"] = f'{fgc}--10'
        elif full_name.startswith('B787--9') or full_name.startswith('B787--Dreamliner9'):
            label["properties"]["Subtype"] = f'{fgc}--9'
        elif full_name.startswith('B787--8'):
            label["properties"]["Subtype"] = f'{fgc}--8'
        else:
            label["properties"]["Subtype"] = full_name
            skipped_names.append(full_name)
        if not skip_fulltype:
            label["properties"]["Fulltype"] = full_name

    return labels, skipped_names
if __name__ == '__main__':
    args = get_args()
    run(args)
