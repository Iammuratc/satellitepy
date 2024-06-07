import configargparse
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt

from satellitepy.utils.path_utils import create_folder, init_logger, get_default_log_path, \
    get_default_log_config, get_file_paths
from satellitepy.data.labels import read_label
from satellitepy.data.utils import get_satellitepy_dict_values, count_unique_values


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
    parser.add_argument('--print-none', action='store_true', help='If True, None values in dict values (e.g., none as the annotation source) will be included.')
    parser.add_argument('--max-class-name-length', default=0, type=int, help='Shorten the class names for a visually better plot bar.')
    return parser


def run(parser):
    """Application entry point."""
    args = parser.parse_args()

    out_folder = Path(args.out_folder)
    assert create_folder(out_folder)

    log_config_path = get_default_log_config() if args.log_config_path is None else Path(args.log_config_path)
    log_path = get_default_log_path(Path(__file__).resolve().stem) if args.log_path is None else Path(args.log_path)

    init_logger(config_path=log_config_path, log_path=log_path)
    logger = logging.getLogger('')
    logger.info('Saving patches from original images...')
    logger.info(f'Log will be stored at: {log_path}')

    config_path = out_folder / f'{Path(__file__).resolve().stem}.ini'
    parser.write_config_file(args, [str(config_path)])
    logger.info(f'Configs will be stored at {config_path}')

    analyse_label_paths(label_folder=args.in_label_folder,
                        label_format=args.in_label_format,
                        task=args.task,
                        logger=logger,
                        plot_bar=args.plot_bar,
                        out_folder=out_folder,
                        max_class_name_length=args.max_class_name_length,
                        print_none=args.print_none)


def analyse_label_paths(label_folder, label_format, task, logger, plot_bar, out_folder, max_class_name_length, print_none):
    label_paths = get_file_paths(label_folder)
    count_instances = {}
    for label_path in label_paths:
        label = read_label(label_path, label_format)
        values = get_satellitepy_dict_values(label, task)
        count_unique_values(satellitepy_values = values, instances=count_instances)

    logger.info('The results from the analyzed labels:')
    for key, value in sorted(count_instances.items()):
        logger.info(f'{key}:{value}')
    
    if not print_none and 'None' in count_instances.keys():
        del count_instances['None'] 

    if max_class_name_length > 0:
        for key in list(count_instances.keys()):
            if len(key)>max_class_name_length:
                key_shortened = f"{key[:max_class_name_length]}."
                count_instances[key_shortened] = count_instances.pop(key)
                logger.info(f"{key} is abbreviated with {key_shortened}")

    if plot_bar:
        count_instances_sorted = dict(reversed(sorted(count_instances.items(), key=lambda item: item[1])))
        fig, ax = plt.subplots(1)
        instances = list(count_instances_sorted.keys())
        counts = list(count_instances_sorted.values())
        ax.bar(instances, counts)
        ax.set_xticks(np.arange(len(instances)))
        ax.set_xticklabels(instances, rotation=45, fontsize=35)
        ax.tick_params(axis='y', labelsize=35)
        fig.set_size_inches(40, 25)
        plot_bar_path = str(out_folder / f'{task}_bar_chart.png')
        plt.savefig(plot_bar_path)
        logger.info(f"Plot bar is saved at: {plot_bar_path}")
        # plt.show()


if __name__ == '__main__':
    args = get_args()
    run(args)
