import configargparse
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

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
    parser.add_argument('--remove-other', action='store_true', help='If set True, Other will be removed in the bar chart.')
    parser.add_argument('--print-none', action='store_true', help='If True, None values in dict values (e.g., none as the annotation source) will be included.')
    parser.add_argument('--group-into-other', type=int, default=0, help='If larger than 0, the classes, that have less'
                        'instances than <group-into-other>, will be grouped into a class called other.')
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
                        print_none=args.print_none,
                        group_into_other_threshold=args.group_into_other,
                        remove_other=args.remove_other)


def analyse_label_paths(label_folder, 
    label_format, 
    task, 
    logger, 
    plot_bar, 
    out_folder, 
    max_class_name_length, 
    print_none, 
    group_into_other_threshold,
    remove_other):
    label_paths = get_file_paths(label_folder)
    count_instances = {}
    for label_path in label_paths:
        label = read_label(label_path, label_format)
        # If task is very-fine-class, merge fine-class and very-fine-class first
        if task == 'very-fine-class':
            ftgc = get_satellitepy_dict_values(label,task='very-fine-class')
            fgc = get_satellitepy_dict_values(label,task='fine-class')
            values = [f"{fgc_i}--{ftgc_i}" for fgc_i, ftgc_i in zip(fgc,ftgc)]
        else:
            values = get_satellitepy_dict_values(label, task)
        count_unique_values(satellitepy_values = values, instances=count_instances)

    

    logger.info('The results from the analyzed labels:')
    for key, value in sorted(count_instances.items()):
        logger.info(f'{key}:{value}')
    
    # Adjust this to FtGC
    # if not print_none and 'None' in count_instances.keys():
    if not print_none:
        count_instances = remove_none_keys(count_instances)

    if group_into_other_threshold > 0:

        others,count_instances = group_into_other(count_instances,group_into_other_threshold)
        logger.info(f"Following classes are grouped into other: {','.join(others)}")

    if max_class_name_length > 0:
        for key in list(count_instances.keys()):
            if len(key)>max_class_name_length:
                key_shortened = f"{key[:max_class_name_length]}."
                count_instances[key_shortened] = count_instances.pop(key)
                logger.info(f"{key} is abbreviated with {key_shortened}")

    if task == 'role':
        logger.info(f"Military objects will be merged to their civilian matches, e.g., Airliner-Military to Airliner")
        print(count_instances)
        count_instances = group_into_civilian_role(count_instances)            
        print(count_instances)

    if remove_other:
        del count_instances['Other']

    if plot_bar:
        count_instances_sorted = dict(reversed(sorted(count_instances.items(), key=lambda item: item[1])))
        fig, ax = plt.subplots(1)
        # ax.yaxis.set_major_locator(MultipleLocator(50))  # Adjust the interval to shorten y-axis spacing
        # ax.xaxis.set_major_locator(MultipleLocator(20))
        instances = list(count_instances_sorted.keys())
        counts = list(count_instances_sorted.values())
        ax.bar(instances, counts)
        ax.set_xticks(np.arange(len(instances)))
        ax.set_xticklabels(instances, rotation=45, fontsize=40)
        ax.tick_params(axis='y', labelsize=40)
        fig.set_size_inches(100, 25)
        plot_bar_path = str(out_folder / f'{task}_bar_chart.png')
        plt.savefig(plot_bar_path)
        logger.info(f"Plot bar is saved at: {plot_bar_path}")
        # plt.show()


def remove_none_keys(input_dict):
    result_dict = {}
    for key, value in input_dict.items():
        key_strings = key.split('-')
        if 'None' in key_strings:
            continue
        else:
            result_dict[key] = value    
    return result_dict


def group_into_civilian_role(input_dict):
    # Initialize a new dictionary for the result
    result_dict = {key:0 for key in input_dict.keys() if not key.endswith('Military')}
    
    for key, value in input_dict.items():
        if key.endswith('Military'):
            # Add the value to the "other" sum if it's below the threshold
            key_civilian = key.split('-')[0]
            result_dict[key_civilian] += value
        else:
            # Otherwise, keep the key-value pair in the result dictionary
            result_dict[key] += value
    
    return result_dict    

def group_into_other(input_dict, threshold):
    # Initialize a new dictionary for the result
    result_dict = {}
    # Initialize the sum for the "other" key
    other_sum = 0
    others = []
    
    for key, value in input_dict.items():
        if value < threshold:
            # Add the value to the "other" sum if it's below the threshold
            other_sum += value
            others.append(key)
        else:
            # Otherwise, keep the key-value pair in the result dictionary
            result_dict[key] = value
    
    # Add the "other" key to the result dictionary if there's any sum to add
    if other_sum > 0:
        result_dict["Other"] = other_sum

    
    return others,result_dict

if __name__ == '__main__':
    args = get_args()
    run(args)
