"""
This script is to preprocess the fineair annotation files.
"""

import configargparse
from pathlib import Path
import json
from collections import Counter
import logging

from satellitepy.utils.path_utils import create_folder, init_logger, get_default_log_path, get_default_log_config, get_file_paths
from satellitepy.data.labels import read_label, read_fineair_label
from satellitepy.data.utils import get_satellitepy_dict_values, count_unique_values, get_fineair_roles


def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-label-folder', type=Path, required=True)
    parser.add_argument('--role-merge-threshold', type=int, help='If a class has a lower number of instances than the threshold,' 
                        'the role will be assigned to its experiment/competition class.')
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
    set_roles(
        in_label_folder,
        role_merge_threshold=args.role_merge_threshold,
        logger=logger,
        out_label_folder=out_label_folder)



def set_roles(in_label_folder,role_merge_threshold,logger,out_label_folder):

    label_paths = get_file_paths(in_label_folder)

    roles = get_fineair_roles()

    # Count instances
    count_instances = get_instances_per_task(in_label_folder,task='fine-class')


    for label_path in label_paths:
        with open(label_path, 'r') as f:
            labels = json.load(f)
        logger.info(f"Processing {label_path.stem}")
        for label in labels["features"]:
            label["properties"]["fineairtype"] = None
            fgc = label["properties"]["Type"]
            role = label["properties"]["Role"]

            # ftgc = label["properties"]["Subtype"]
            # print(fgc)
            if fgc is None:
                label["properties"]["fineairtype"] = label["properties"]["Role"].split('-')[0]
            elif count_instances[fgc] < role_merge_threshold:
                for role, fg_classes in roles.items():
                    if fgc in fg_classes:
                        label["properties"]["fineairtype"] = role
                        label["properties"]["Role"] = role
                        continue
            else:
                label["properties"]["fineairtype"] = fgc
                for role, fg_classes in roles.items():
                    if fgc in fg_classes:
                        label["properties"]["Role"] = role

            # if role is None:
                
            #         fac = label["properties"]["fineairtype"]
            #         if fac in fg_classes:
            #             label["properties"]["Role"] = role
            #             continue
            ## Check if a fineair class is really assigned
            fineair_class = label["properties"]["fineairtype"] 
            if fineair_class == 'None' or fineair_class is None:
                id = label["properties"]["id"]
                logger.error(f'The object with the id {id} has an issue!')
                assert Exception(f'{fineair_class} is None')
        ## Save the new labels
        out_label_path = str(out_label_folder / label_path.name)
        with open(out_label_path, 'w') as file:
            json.dump(labels, file, indent=4)


def get_instances_per_task(label_folder,task):
    label_paths = get_file_paths(label_folder)
    # Count instances
    count_instances = {}
    for label_path in label_paths:
        # label = read_label(label_path, label_format='fr24')
        label = read_fineair_label(label_path, include_fineair_class=False)
        # If task is very-fine-class, merge fine-class and very-fine-class first
        values = get_satellitepy_dict_values(label, task)
        count_instances = count_unique_values(satellitepy_values = values, instances=count_instances)
    return count_instances


# def count_unique_strings(strings):
#     # Count occurrences of each string in the list
#     string_counts = Counter(strings)
    
#     # Print the count number for each unique string
#     for string, count in string_counts.items():
#         print(f"{string}: {count}")

if __name__ == '__main__':
    args = get_args()
    run(args)
