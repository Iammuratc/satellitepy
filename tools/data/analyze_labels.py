import configargparse
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from satellitepy.utils.path_utils import create_folder, init_logger, get_default_log_path, \
    get_default_log_config, get_file_paths
from satellitepy.data.labels import read_label, read_fineair_label
from satellitepy.data.utils import get_satellitepy_dict_values, count_unique_values, get_fineair_roles


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
    parser.add_argument('--plot-sunburst-bar', action='store_true', help='Sunburst bar chart will be displayed.')
    parser.add_argument('--plot-horizontal-bar', action='store_true', help='Horizontal bar chart will be displayed.')
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
    logger.info('Analyzing label files...')
    logger.info(f'Log will be stored at: {log_path}')

    config_path = out_folder / f'{Path(__file__).resolve().stem}.ini'
    parser.write_config_file(args, [str(config_path)])
    logger.info(f'Configs will be stored at {config_path}')

    analyse_label_paths(label_folder=args.in_label_folder,
                        label_format=args.in_label_format,
                        task=args.task,
                        logger=logger,
                        plot_bar=args.plot_bar,
                        plot_sunburst_bar=args.plot_sunburst_bar,
                        plot_horizontal_bar=args.plot_horizontal_bar,
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
    plot_sunburst_bar,
    plot_horizontal_bar,
    out_folder, 
    max_class_name_length, 
    print_none, 
    group_into_other_threshold,
    remove_other,
    remove_zero=False):


    label_paths = get_file_paths(label_folder)
    count_instances = {}
    for label_path in label_paths:
        label = read_label(label_path, label_format)
        values = get_satellitepy_dict_values(label, task)
        count_instances = count_unique_values(satellitepy_values = values, instances=count_instances)

    logger.info('The results from the analyzed labels:')
    for key, value in sorted(count_instances.items()):
        key_ratio = value / sum(count_instances.values())
        logger.info(f'{key}:{value} ({key_ratio:.2f})')
    logger.info(f'There are {len(count_instances)} classes and {sum(count_instances.values())} instances in total.')
    
    # Adjust this to FtGC
    # if not print_none and 'None' in count_instances.keys():
    if not print_none:
        count_instances = remove_none_keys(count_instances)

    if remove_zero:
        count_instances = remove_zero_values(count_instances)

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

    if plot_sunburst_bar:
        instance_names = get_instance_names(label_paths,label_format)

        # print(instance_names)
        df = pd.DataFrame(instance_names)
        df = df.dropna()
        fig = px.sunburst(df, 
            path=list(instance_names.keys()), 
            title='Distribution of Classes at Three Levels')
        fig.update_layout(
            title_font_size=36,
            sunburstcolorway=["#636efa","#ef553b","#00cc96"],
            margin=dict(t=0, l=0, r=0, b=0),
            uniformtext=dict(minsize=20, mode='hide')
            )
        fig.show()


    if plot_horizontal_bar:

        roles = get_fineair_roles()
        label_paths = get_file_paths(label_folder)
        count_instances_by_task = {
            'role':{},
            'fineair-class':{},
            'very-fine-class':{}}
        tasks = count_instances_by_task.keys()
        for label_path in label_paths:
            label = read_label(label_path, label_format)
            for task in tasks:
                values = get_satellitepy_dict_values(label, task)
                count_instances_by_task[task] = count_unique_values(satellitepy_values = values, instances=count_instances_by_task[task], merge_military=True)
        
        ## Drop None--None from very-fine-class
        very_fine_class_without_none = {}
        for class_name, class_count in count_instances_by_task['very-fine-class'].items():
            if not (class_name.startswith('None') or class_name.endswith('None')):
                very_fine_class_without_none[class_name] = class_count
        count_instances_by_task['very-fine-class'] = very_fine_class_without_none
        len_unique_ftgc = len(count_instances_by_task['very-fine-class'].keys())
        print(list(count_instances_by_task['very-fine-class'].keys()))
        print(len_unique_ftgc)
        len_ftgc = sum(count_instances_by_task['very-fine-class'].values())
        print(f'Number of airplanes with FtGC: {len_ftgc}')
        ## Merge classes with low instance numbers to the role
        count_instances_by_task = merge_into_role(count_instances_by_task,th=15,roles=roles)


        # Create the bar chart
        ## Color map
        ## Three roles are assigned to three colors
        instance_names = get_instance_names(label_paths, label_format)
        levels = ', '.join(instance_names.keys())
        logger.info(f"3 levels will be plotted: {levels}")

        ## Color role dict
        colors_role_dict = {role:color for role, color in zip(roles.keys(),['blue','orange','green'])}
        colors = []
        for class_name in instance_names['role']:
            colors.append(colors_role_dict[class_name])
        # instance_names['colors'] = colors

        ## Color dict
        color_dict = {task:{} for task in tasks}
        
        for task in tasks:
            for class_name in sorted(count_instances_by_task[task].keys()):
                if task == 'very-fine-class':
                    class_name_to_ind = class_name.split('--')[0]
                    # if class_name_to_ind.startswith('None') or class_name_to_ind.endswith('None'):
                    #     continue
                    if class_name_to_ind in instance_names['fine-class']:
                        class_name_ind = instance_names['fine-class'].index(class_name_to_ind)
                    elif class_name_to_ind in instance_names['role']:
                        class_name_ind = instance_names['role'].index(class_name_to_ind)
                    else:
                        print('MISTAKE!')
                        return 0
                else:
                    class_name_ind = instance_names[task].index(class_name)
                # Get the item from the second list at that index
                color = colors[class_name_ind]
                color_dict[task][class_name] = color

        len_ftgc = sum(count_instances_by_task['very-fine-class'].values())
        print(f'Number of airplanes with FtGC: {len_ftgc}')

        ### Merge small very-fine-class into the parent role
        fig = make_subplots(
            rows=len(list(tasks)), cols=1,
            subplot_titles=['Role', 'FineAir Class', 'Finest-grained Class'],
            shared_xaxes=False,
            vertical_spacing=0.1
            )
        for i, task in enumerate(list(tasks)):
            # fig = go.Figure()
            for class_name, class_count in count_instances_by_task[task].items():
                fig.add_trace(
                    go.Bar(
                        y=[task],
                        x=[class_count],
                        text=class_name,
                        marker_line=dict(width=2, color='black'),
                        textposition='inside',
                        insidetextanchor='middle',
                        orientation='h',
                        marker=dict(color=color_dict[task][class_name],opacity=0.5)
                    ),
                    row=i+1,
                    col=1,
                )
            # Update layout
        fig.update_annotations(font_size=36)
        fig.update_layout(
            barmode='stack',
            # title='Distribution of Classes A, B, and C',
            # xaxis_title='Size',
            # yaxis_title=task,
            # yaxis=dict(tickvals=['Distribution'], ticktext=['Classes A, B, and C']),
            # legend_title='Classes'
            showlegend=False,
            polar = dict(angularaxis = dict(showticklabels = False)),
            # tickfont = dict(size=24),
            xaxis=dict(tickfont=dict(size=36)),  # First subplot
            xaxis2=dict(tickfont=dict(size=36)),  # Second subplot
            xaxis3=dict(tickfont=dict(size=36)),  # Third subplot
            font=dict(size=24)
        )
        # fig.update_xaxes(title_font=dict(size=24))
        fig.update_yaxes(title='y', visible=False, showticklabels=False)
        # plt.savefig('/home/murat/Projects/satellitepy/docs/fineair_dist.png')
        fig.show()

    return count_instances

def merge_into_role(count_instances_by_task,th,roles):
    result_dict = {role:0 for role in roles.keys()}
    for class_name, class_count in count_instances_by_task['very-fine-class'].items():
        if class_count < th:
            role_defined = False
            for role, values in roles.items():
                if class_name.split('--')[0] in values:

                    role_defined = True
                    result_dict[role] += class_count
                    print(f"Class {class_name} is merged to {role}!")
                    continue
            if not role_defined:
                print(f"No role for {class_name.split('--')[0]}!")
        else:
            result_dict[class_name] = class_count
    count_instances_by_task['very-fine-class'] = result_dict
    return count_instances_by_task
             


def get_instance_names(label_paths, label_format):
    instance_names = {
        'role':[],
        'fineair-class':[],
        'fine-class':[],
        'very-fine-class':[]}
    tasks = instance_names.keys()
    for label_path in label_paths:
        label = read_label(label_path, label_format)
        for task in tasks:
            values = get_satellitepy_dict_values(label, task)
            for value in values:
                if value is None:
                    instance_names[task].append(None)
                elif value.endswith('Military'):
                    value_civilian = value.split('-')[0]
                    instance_names[task].append(value_civilian)
                elif value.endswith('None'):
                    instance_names[task].append(None)
                else:
                    instance_names[task].append(value)
    return instance_names

def remove_none_keys(input_dict):
    result_dict = {}
    for key, value in input_dict.items():
        key_strings = key.split('-')
        if 'None' in key_strings:
            continue
        else:
            result_dict[key] = value    
    return result_dict

def remove_zero_values(input_dict):
    result_dict = {}
    for key, value in input_dict.items():
        if value == 0:
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
