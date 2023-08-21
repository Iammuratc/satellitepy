import configargparse
from pathlib import Path
from satellitepy.utils.path_utils import init_logger, get_project_folder, get_file_paths
from satellitepy.data.labels import read_label, init_satellitepy_label
import logging
import os
import json
import numpy as np

project_folder = get_project_folder()

def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label-folder', type=Path, required=True,
                        help='The folder which contains the labels to analyze')
    parser.add_argument('--tasks', help='Which attributes to analyze. E.g. classes_0', nargs='+')
    parser.add_argument('--log-config-path', default=project_folder /
                        Path("configs/log.config"), type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, default=None, help='Log file path.')
    parser.add_argument('--label-format', default='satellitepy', help='Format the label files are written in')
    args = parser.parse_args()
    return args


def run(args):
    label_folder = Path(args.label_folder)
    label_format = args.label_format.lower()
    tasks = args.tasks
    log_path = project_folder / f'analyze_labels_{label_format}.log' if args.log_path == None else args.log_path
    init_logger(config_path=args.log_config_path, log_path=log_path)
    logger = logging.getLogger(__name__)
    logger.info(
        f'No log path is given, the default log path will be used: {log_path}')
    
    logger.info('Analyzing labels')

    label_paths = get_file_paths(label_folder)
    analyse_label_paths(label_paths,label_format,tasks,logger)



def analyse_label_paths(label_paths,label_format,tasks,logger):
    label_dict = init_satellitepy_label()
    count_instances = {task:{} for task in tasks}
    for label_path in label_paths:
        # logger.info(f'Following file will be analyzed: {label_path}')
        label = read_label(label_path,label_format)
        for task in tasks:
            keys = task.split('_')
            if len(keys)==1:
                values = label[keys[0]]
            elif len(keys)==2:
                values = label[keys[0]][keys[1]]
            elif len(keys)==3:
                values = label[keys[0]][keys[1]][keys[2]]
            count_satellitepy_values(count_instances,task,values)

    logger.info('The results from the analyzed labels:')
    for task in count_instances.keys():
        for key, value in count_instances[task].items():
            logger.info(f"{key}:{value}")

def count_satellitepy_values(count_instances,task,values):
    for value in values:
        if isinstance(value,str) or isinstance(value,int):
            if value not in count_instances[task].keys():
                count_instances[task][value] = 0
            count_instances[task][value] += 1
        elif isinstance(value,list):
            if 'count' not in count_instances[task].keys():
                count_instances[task]['count'] = 0
            count_instances[task]['count'] += 1
        elif isinstance(value,float):
            if 'max' not in count_instances[task].keys():
                 count_instances[task]['max'] = 0
                 count_instances[task]['min'] = np.inf
            if value>count_instances[task]['max']:
                count_instances[task]['max'] = value  
            if value<count_instances[task]['min']:
                count_instances[task]['min'] = value  
    return count_instances

if __name__ == '__main__':
    args = get_args()
    run(args)
