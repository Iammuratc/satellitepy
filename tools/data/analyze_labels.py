import configargparse
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt

from satellitepy.utils.path_utils import create_folder, init_logger, get_default_log_path, get_default_log_config, get_file_paths
from satellitepy.data.labels import read_label, init_satellitepy_label
from satellitepy.data.utils import get_satellitepy_dict_values

def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', is_config_file=True, help='Path to the configuration file')
    parser.add_argument('--in-label-folder', type=Path, required=True,
                        help='The folder which contains the labels to analyze')
    parser.add_argument('--in-label-format', default='satellitepy', help='Label file format')
    parser.add_argument('--out-folder', type=Path,
                        help='Analysis will be written to a text file under <out-folder>.')
    parser.add_argument('--task', help='Task to analyze. E.g. coarse-class')
    parser.add_argument('--log-config-path', default=None, type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, default=None, help='Log file path.')
    parser.add_argument('--plot-bar', action='store_true', help='Bar chart will be displayed')
    return parser


def run(parser):
    """Application entry point."""
    args = parser.parse_args()

    out_folder = Path(args.out_folder)
    assert create_folder(out_folder)


    # Logger configs
    log_config_path = get_default_log_config() if args.log_config_path==None else Path(args.log_config_path)
    log_path = get_default_log_path(Path(__file__).resolve().stem) if args.log_path==None else Path(args.log_path)

    # Initiate logger
    init_logger(config_path=log_config_path,log_path=log_path)
    logger = logging.getLogger(__name__)
    logger.info('Saving patches from original images...')
    logger.info(f'Log will be stored at: {log_path}')
    
    # configargparse config
    config_path = out_folder / f"{Path(__file__).resolve().stem}.ini"
    parser.write_config_file(args, [str(config_path)])
    logger.info(f"Configs will be stored at {config_path}")

    analyse_label_paths(label_folder=args.in_label_folder,
        label_format=args.in_label_format,
        task=args.task,
        logger=logger,
        plot_bar=args.plot_bar,
        out_folder=out_folder)


def analyse_label_paths(label_folder,label_format,task,logger,plot_bar,out_folder):
    label_paths = get_file_paths(label_folder)
    count_instances = {}
    for label_path in label_paths:
        label = read_label(label_path,label_format)
        values = get_satellitepy_dict_values(label,task)
        count_satellitepy_values(count_instances,values)

    logger.info('The results from the analyzed labels:')
    for key, value in sorted(count_instances.items()):
        logger.info(f"{key}:{value}")

    if plot_bar:
        count_instances_sorted =  dict(reversed(sorted(count_instances.items(), key=lambda item: item[1])))
        fig,ax = plt.subplots(1)
        instances = list(count_instances_sorted.keys())
        counts = list(count_instances_sorted.values())
        ax.bar(instances,counts)
        ax.set_xticks(np.arange(len(instances)))
        ax.set_xticklabels(instances, rotation=45, fontsize=16)
        plt.savefig(str(out_folder/f'{task}_bar_chart.png'))
        plt.show()
        # plt.close()



def count_satellitepy_values(count_instances,values):
    for value in values:
        if isinstance(value,str) or isinstance(value,int):
            if value not in count_instances.keys():
                count_instances[value] = 0
            count_instances[value] += 1
        elif isinstance(value,list):
            if 'count' not in count_instances.keys():
                count_instances['count'] = 0
            count_instances['count'] += 1
        elif isinstance(value,float):
            if 'max' not in count_instances.keys():
                 count_instances['max'] = 0
                 count_instances['min'] = np.inf
            if value>count_instances['max']:
                count_instances['max'] = value  
            if value<count_instances['min']:
                count_instances['min'] = value
        elif value == None:
            if 'None' not in count_instances.keys():
                count_instances['None'] = 0
            count_instances['None'] += 1
    return count_instances

if __name__ == '__main__':
    args = get_args()
    run(args)
