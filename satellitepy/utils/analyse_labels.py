import configargparse
from pathlib import Path
from path_utils import init_logger, get_project_folder, get_file_paths
from satellitepy.data.labels import read_label, init_satellitepy_label
import logging
import os
import json
import pprint

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
    # temp_count=0
    for label_path in label_paths:
        label = read_label(label_path,label_format)
        # print(label_path)
        # print(label)
        # temp_count += 1
        # if temp_count ==3:
        #     break
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
    logger.info(count_instances)

def count_satellitepy_values(count_instances,task,values):
    for value in values:
        if isinstance(value,str):
            if value not in count_instances[task].keys():
                count_instances[task][value] = 0
            count_instances[task][value] += 1
        elif isinstance(value,list) or isinstance(value,float) or isinstance(value,int):
            if 'count' not in count_instances[task].keys():
                count_instances[task]['count'] = 0
            count_instances[task]['count'] += 1
    return count_instances
    # json_file_names = [filename for filename in os.listdir(label_folder) if filename.endswith('.json') or filename.endswith('.geojson')]

    # global dictionary
    # dictionary = {
    #     'classes': {},
    #     'attributes': {
    #         'engines': {
    #             'no-engines': 0,
    #             'propulsion': {
    #                 'unpowered': 0,
    #                 'jet': 0,
    #                 'propeller': 0
    #             }
    #         },
    #         'fuselage': {
    #             'canards': 0,
    #             'length': 0
    #         },
    #         'wings': {
    #             'wing-span': 0,
    #             'wing-shape': {
    #                 'swept': 0,
    #                 'straight': 0,
    #                 'delta': 0,
    #                 'variable_swept': 0
    #             },
    #             'wing-position': {
    #                 'low_mounted': 0,
    #                 'high_mounted': 0
    #             }
    #         },
    #         'no-tail-fins': 0,
    #         'role': {
    #             'civil': {
    #                 'large_transport': 0,
    #                 'medium_transport': 0,
    #                 'small_transport': 0
    #             },
    #             'military': {
    #                 'fighter': 0,
    #                 'bomber': 0,
    #                 'transport': 0,
    #                 'trainer': 0
    #             }
    #         }
    #     }
    # }

    # features = ['obboxes', 'hbboxes', 'masks', 'difficulty']
    
    # for json_file_name in json_file_names:
    #     labels = read_label(os.path.join(str(label_folder), json_file_name), label_format)
    #     for i in range(0, 3):
    #         if len(labels['classes'][str(i)]) > 0 and labels['classes'][str(i)] != [None]:
    #             text = ''.join(labels['classes'][str(i)])
    #             if not str(i) in dictionary['classes']:
    #                 dictionary['classes'][str(i)] = {}
    #             if text in dictionary['classes'][str(i)]:
    #                 dictionary['classes'][str(i)][text] += 1
    #             else:
    #                 dictionary['classes'][str(i)][text] = 1

    #     for feature in features:
    #         if len(labels[feature]) > 0 and labels[feature] != [None]:
    #             if feature in dictionary:
    #                 dictionary[feature] += 1
    #             else: 
    #                 dictionary[feature] = 1
        
    #     if label_format == 'satellitepy' or label_format == 'rareplanes':

    #         if section_handler(["attributes", "engines", "no-engines"], labels):
    #             dictionary['attributes']['engines']['no-engines'] += labels["attributes"]["engines"]["no-engines"][0]
    #             dictionary['attributes']['engines']['propulsion'][labels["attributes"]["engines"]['propulsion'][0]] += 1
            
    #         if section_handler(["attributes", "fuselage", "canards"], labels):
    #             if labels["attributes"]["fuselage"]["canards"][0]:
    #                 dictionary["attributes"]["fuselage"]["canards"] += 1

    #         section_handler(["attributes", "fuselage", "length"], labels, add_one=True)
            
    #         section_handler(["attributes", "wings", "wing-span"], labels, add_one=True)    

    #         if section_handler(["attributes", "wings", "wing-shape"], labels):
    #             dictionary['attributes']['wings']['wing-shape'][labels["attributes"]["wings"]['wing-shape'][0]] += 1

    #         if section_handler(["attributes", "wings", "wing-position"], labels):
    #             dictionary['attributes']['wings']['wing-position'][labels["attributes"]["wings"]['wing-position'][0]] += 1

    #         if section_handler(["attributes", "tail", "no-tail-fins"], labels):
    #             dictionary["attributes"]["no-tail-fins"] += labels["attributes"]["tail"]["no-tail-fins"][0]

    #         if section_handler(["attributes", "role", "civil"], labels):
    #             dictionary['attributes']['role']['civil'][labels["attributes"]["role"]['civil'][0]] += 1

    #         if section_handler(["attributes", "role", "military"], labels):
    #             dictionary['attributes']['role']['military'][labels["attributes"]["role"]['military'][0]] += 1


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


# def section_handler(keys, label, add_one=False):
#     global dictionary
#     for key in keys:
#         label = label[key]
    
#     exists = (label != [None] and len(label) >0)

#     if exists and add_one:
#         dictionary[keys[0]][keys[1]][keys[2]] += 1
#         return

#     return exists


if __name__ == '__main__':
    args = get_args()
    run(args)
