import configargparse
from pathlib import Path
from path_utils import init_logger, get_project_folder
from satellitepy.data.labels import read_label
import logging
import os
import json
import pprint

def get_args():
    """Arguments parser."""
    project_folder = get_project_folder()
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label-folder', type=Path, required=True,
                        help='The folder which contains the labels to analyze')
    parser.add_argument('--tasks', help='Which attributes to analyze. E.g. classes_0', nargs='+')
    parser.add_argument('--log-config-path', default=project_folder /
                        Path("configs/log.config"), type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, help='Log file path.')
    parser.add_argument('--label-format', default='satellitepy', help='Format the label files are written in')
    args = parser.parse_args()
    return args


def run(args):
    label_folder = Path(args.label_folder)
    label_format = args.label_format.lower()
    tasks = args.tasks
    log_path = Path(
        label_folder) / 'analyze_labels.log' if args.log_path == None else args.log_path
    init_logger(config_path=args.log_config_path, log_path=log_path)
    logger = logging.getLogger(__name__)
    logger.info(
        f'No log path is given, the default log path will be used: {log_path}')
    
    logger.info('Analyzing labels')

    json_file_names = [filename for filename in os.listdir(label_folder) if filename.endswith('.json') or filename.endswith('.geojson')]

    dictionary = {
        'classes': {}
    }

    features = ['obboxes', 'hbboxes', 'masks', 'difficulty']
    
    for json_file_name in json_file_names:
        labels = read_label(os.path.join(label_folder, json_file_name), label_format)
        for i in range(0, 3):
            if len(labels['classes'][str(i)]) > 0 and labels['classes'][str(i)] != [None]:
                text = ''.join(labels['classes'][str(i)])
                if not str(i) in dictionary['classes']:
                    dictionary['classes'][str(i)] = {}
                if text in dictionary['classes'][str(i)]:
                    dictionary['classes'][str(i)][text] += 1
                else:
                    dictionary['classes'][str(i)][text] = 1

        for feature in features:
            if len(labels[feature]) > 0 and labels[feature] != [None]:
                if feature in dictionary:
                    dictionary[feature] += 1
                else: 
                    dictionary[feature] = 1

    pp = pprint.PrettyPrinter(indent=2)
    if tasks:
        for task in tasks:
            keys = task.split('_')
            logger.info(f'{task}: ')
            try:
                if len(keys)==1:
                    pp.pprint(dictionary[keys[0]])
                elif len(keys)==2:
                    pp.pprint(dictionary[keys[0]][keys[1]])
                elif len(keys)==3:
                    pp.pprint(dictionary[keys[0]][keys[1]][keys[2]])
            except KeyError:
                logger.error(f'No occurences for {keys} found!')
    else:
        pp.pprint(dictionary)

    logger.info('Finished analysing')


if __name__ == '__main__':
    args = get_args()
    run(args)
