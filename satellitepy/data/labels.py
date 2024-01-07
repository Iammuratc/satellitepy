import logging
from builtins import print

import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from satellitepy.data.utils import get_vedai_classes, set_mask, parse_potsdam_labels, set_satellitepy_dict_values, get_shipnet_categories
from satellitepy.data.bbox import BBox
import json
import numpy as np

from satellitepy.data.bbox import BBox

def read_label(label_path,label_format, mask_path = None):
    logger = logging.getLogger(__name__)

    if isinstance(label_path,Path):
        label_path = str(label_path)
    if label_format=='dota' or label_format=='DOTA':
        return read_dota_label(label_path,mask_path)
    elif label_format=='fair1m':
        return read_fair1m_label(label_path)
    elif label_format=='satellitepy':
        return read_satellitepy_label(label_path)
    elif label_format=="dior" or label_format=="DIOR":
        return read_dior_label(label_path)
    elif label_format=="vhr" or label_format=="VHR":
        return read_VHR_label(label_path)
    elif label_format == 'rareplanes_real' or label_format == 'rarePlanes_real':
        return read_rareplanes_real_label(label_path)
    elif label_format == 'rareplanes_synthetic' or label_format == 'rarePlanes_synthetic':
        return read_rareplanes_synthetic_label(label_path, mask_path)
    elif label_format == 'ship_net':
        return read_ship_net_label(label_path)
    elif label_format == 'ucas':
        return read_ucas_label(label_path)
    elif label_format == 'xview':
        logger.info('Please run tools/data/split_xview_into_satellitepy_labels.py to get the satellitepy labels.'
              ' Then pass label_format as satellitepy for those labels.')
        exit(1)
    elif label_format == 'isprs':
        return read_isprs_label(label_path)
    elif label_format == "results":
        return read_result_label(label_path)
    elif label_format == "vedai":
        return read_vedai_label(label_path)
    else:
        logger.info('---Label format is not defined---')
        exit(1)

def get_all_satellitepy_keys():
    """
    Get all possible satellitepy keys
    Returns
    -------
    all_keys : list of str
        E.g. ['bboxes','mask-indices','coarse-class','attributes_engines_propulsion']
    """

    labels = init_satellitepy_label()

    all_keys = []

    for key_0, value_0 in labels.items():
        if isinstance(value_0,list):
            all_keys.append(key_0)
        else:
            for key_1, value_1 in value_0.items():
                if isinstance(value_1,list):
                    all_keys.append(f"{key_0}_{key_1}")
                else:
                    for key_2, value_2 in value_1.items():
                        if isinstance(value_2,list):
                            all_keys.append(f"{key_0}_{key_1}_{key_2}")
    return all_keys

def set_image_keys(all_satellitepy_keys, sub_image_labels, gt_labels, i):
    """
    Set object labels for the patch 
    Parameters
    ----------
    sub_image_labels : dict of str
        Dict in satellitepy format. sub_image could be patch or chip
    gt_labels : dict of str
        Dict in satellitepy format 
    gt_label_i : int
        Index of object in gt_labels
    Returns
    -------
    sub_image_labels : dict of str
        Dict in satellitepy format. Only the objects within the patch and the chip
    """
    for task in all_satellitepy_keys:
        keys = task.split("_")
        if len(keys) == 1:
            sub_image_labels[keys[0]].append(gt_labels[keys[0]][i])
        elif len(keys) == 2:
            sub_image_labels[keys[0]][keys[1]].append(gt_labels[keys[0]][keys[1]][i])
        elif len(keys) == 3:
            sub_image_labels[keys[0]][keys[1]][keys[2]].append(gt_labels[keys[0]][keys[1]][keys[2]][i])
    return sub_image_labels

def fill_none_to_empty_keys(labels,not_available_tasks):
    """
    Append None to non existing tasks for one object; Hbboxes are calculated by obboxes if missing
    Parameters
    ----------
    labels : dict of str
        Dict in satellitepy format 
    not_available_tasks : list of str
        Tasks are not available within the dataset. E.g., [masks','attributes_engines_propulsion','attributes_engines_no-engines']
    Returns
    -------
    labels : dict of str
        None appended dict in satellitepy format
    """

    for task in not_available_tasks:
        keys = task.split('_')
        if len(keys)==1:
            labels[keys[0]].append(None)
        elif len(keys)==2:
            labels[keys[0]][keys[1]].append(None)
        elif len(keys)==3:
            labels[keys[0]][keys[1]][keys[2]].append(None)
    return labels

def satellitepy_labels_empty(labels, ignore: list[str] = []):
    """
    This function checks whether the labels dict in satellitepy format is empty.
    Returns
    -------
    bool: Whether it is empty
    """
    # every key in the dict has ultimately a list as value
    # we recursively go through the dict until the value is a list
    # we collect booleans telling us whether the found lists are empty or not
    # the labels dict is empty if every list value is empty
    def inner(d):
        # recursion stop
        if isinstance(d, list):
            return len(d)==0
        # recursion
        elif isinstance(d, dict):
             for k, v in d.items():
                return inner(v)
        # unexpected => raise error
        else:
            raise ValueError("expected labels dict in satellitepy format to only contain keys and values of type list")
    return inner(labels)



def init_satellitepy_label():
    """
    This function creates an empty labels dict in satellitepy format.
    WARNING: Do not use underdash "_" in key names, because "_" will be 
    used in parsing the nested task names (e.g., attributes_engines_no-engines) within other functions.
    Returns
    -------
    labels : dict of str
        hbboxes : list
            Horizontal bounding box corners for every object
        obboxes : list
            Oriented bounding box corners for every object
        masks : list
            Mask pixel coordinates with the shape [<number-of-objects>,2]. The order is x,y
        coarse-class : list of str
            coarse grained classes. It has to be one of these three types: airplane,ship,vehicle
        fine-class : list of str 
            fine grained classes (e.g., A220, passenger ship)
        role : list of str 
            object roles. For example, Small Civil Transport/Utility, Military Fighter/Interceptor/Attack
        very-fine-class : list of str
            very fine grained classes (e.g., A220-100)
        difficulty : list of int
            Detection difficulty of the object. Only DOTA provides this. 
            For example, clouds make the detection task difficult. 
        attributes : dict of str
            This value only serves for Rareplanes at the moment. 
            Please check the rareplanes paper for details.
            'engines' : dict of str
                'no-engines' : list of int
                    Number of engines
                'propulsion' : list of str
                    unpowered, jet, propeller
            'fuselage' : dict of str
                'canards' : list of bool
                'length' : list of float
            'wings' : dict of str
                'wing-span' : list of float
                'wing-shape' : list of str
                    swept, straight, delta, variable_swept
                'wing-position' : list of str
                    low_mounted, high_mounted
            'tail' : dict of str
                'no-tail-fins' : list of int
                    1, 2
            'role' : list of str
                large_transport, medium_transport, small_transport
                fighter, bomber, transport, trainer
    """
    labels={
        'hbboxes':[],
        'obboxes': [],
        'masks':[],
        'coarse-class':[],
        'role':[],
        'fine-class':[],
        'very-fine-class':[],
        # 'merged-class':[], # This is a concatenated version of all class values
        'difficulty':[],
        'attributes':{
            'engines':{
                'no-engines':[],
                'propulsion':[]
            },
            'fuselage':{
                'canards':[],
                'length':[]
            },
            'wings':{
                'wing-span':[],
                'wing-shape':[],
                'wing-position':[]
            },
            'tail':{
                'no-tail-fins':[]
            }
        }
    }
    return labels    

def merge_satpy_label_dict(target_satpy_labels: dict, to_add: dict):
    for k in target_satpy_labels.keys():
        if isinstance(target_satpy_labels[k], dict):
            merge_satpy_label_dict(target_satpy_labels[k], to_add[k])
        else:
            # we expect final values to always be lists
            target_satpy_labels[k] += to_add[k]

def read_dota_label(label_path, mask_path=None):
    labels = init_satellitepy_label()
    # Get all not available tasks so we can append None to those tasks
    ## Default available tasks for dota
    available_tasks=['hbboxes', 'obboxes', 'difficulty', 'coarse-class', 'role']
    mask_exists = True if mask_path else False
    if mask_exists:
        available_tasks.append('masks')
    ## All possible tasks
    all_tasks = get_all_satellitepy_keys()
    ## Not available tasks
    not_available_tasks = [task for task in all_tasks if not task in available_tasks or available_tasks.remove(task)]

    with open(label_path, 'r') as f:
        for line in f.readlines():
            bbox_line = line.split(' ') # Corner points, category, [difficulty]
            # Original DOTA dataset has some metadata in first two lines
            # Check if the line matches with the DOTA format
            len_bbox_line = len(bbox_line)
            if len_bbox_line==10:
                # Difficulty defined
                difficulty = bbox_line[-1].rstrip()
                labels['difficulty'].append(difficulty)
                category_i = -2
            elif len_bbox_line==9:
                # No difficulty defined
                category_i = -1
                labels['difficulty'].append(None)
            else:
                continue

            # Classes
            category = bbox_line[category_i].rstrip()
            ## large-vehicle and small-vehicle should be handled individually
            ### class_coarse-class = vehicle, class_fine-class = large-vehicle
            category_words = category.split('-')
            if len(category_words) == 2 and category_words[1]=='vehicle':
                labels['coarse-class'].append(category_words[1]) # vehicle
                if category_words[0]=='small':
                    labels['role'].append('Small Vehicle') 
                elif category_words[0]=='large':
                    labels['role'].append('Large Vehicle')
                else:
                    labels['role'].append(None)
            elif category=='plane' or category=='ship' or category=='helicopter':
                # Airplane is the common word
                category = 'airplane' if category == 'plane' else category
                labels['coarse-class'].append(category) # plane, ship
                labels['role'].append(None) #
            else:
                labels['coarse-class'].append('other') #
                labels['role'].append(None) #
            # BBoxes
            bbox_corners_flatten = [[float(corner) for corner in bbox_line[:category_i]]]
            bbox_corners = np.reshape(bbox_corners_flatten, (4, 2)).tolist()
            labels['obboxes'].append(bbox_corners)
            labels['hbboxes'].append(BBox.get_hbb_from_obb(bbox_corners))
            fill_none_to_empty_keys(labels,not_available_tasks)

        # Mask
        if mask_exists:
            labels = set_mask(labels,mask_path,bbox_type='obboxes', mask_type='DOTA')


    return labels

def read_fair1m_label(label_path):
    labels = init_satellitepy_label()
    # Get all not available tasks so we can append None to those tasks
    ## Default available tasks for dota
    available_tasks=['hbboxes', 'obboxes','coarse-class','fine-class','role']
    ## All possible tasks
    all_tasks = get_all_satellitepy_keys()
    ## Not available tasks
    not_available_tasks = [task for task in all_tasks if not task in available_tasks or available_tasks.remove(task)]

    root = ET.parse(label_path).getroot()

    file_name = root.findall('./source/filename')[0].text

    # Instance names
    instance_names = root.findall(
        './objects/object/possibleresult/name')
    for instance_name in instance_names:
        if instance_name.text in ['Boeing747','Boeing787','A330','Boeing777','A350']:
            labels['coarse-class'].append('airplane')
            labels['fine-class'].append(instance_name.text)
            labels['role'].append('Large Civil Transport/Utility')
        elif instance_name.text in ['A321', 'A220', 'ARJ21', 'Boeing737','C919']:
            labels['coarse-class'].append('airplane')
            labels['fine-class'].append(instance_name.text)
            labels['role'].append('Medium Civil Transport/Utility')  
        elif instance_name.text in ['other-airplane']:
            labels['coarse-class'].append('airplane')
            labels['fine-class'].append(None)
            labels['role'].append(None)
        elif instance_name.text in ['Cargo Truck','Dump Truck','Excavator','Bus','Truck Tractor','Tractor','Trailer']:
            labels['coarse-class'].append('vehicle')
            labels['fine-class'].append(instance_name.text)
            labels['role'].append('Large Vehicle')
        elif instance_name.text in ['Small Car','Van']:
            labels['coarse-class'].append('vehicle')
            labels['fine-class'].append(instance_name.text)
            labels['role'].append('Small Vehicle')
        elif instance_name.text in ['other-vehicle']:
            labels['coarse-class'].append('vehicle')
            labels['fine-class'].append(None)
            labels['role'].append(None)
        elif instance_name.text in ['Liquid Cargo Ship','Passenger Ship','Dry Cargo Ship','Motorboat',
                                    'Engineering Ship','Tugboat','Fishing Boat']:
            labels['coarse-class'].append('ship')
            labels['fine-class'].append(instance_name.text)
            labels['role'].append('Merchant Ship')
        elif instance_name.text in ['Warship']:
            labels['coarse-class'].append('ship')
            labels['fine-class'].append(None)
            labels['role'].append('Warship')
        elif instance_name.text in ['other-ship']:
            labels['coarse-class'].append('ship')
            labels['fine-class'].append(None)
            labels['role'].append(None)
        else:
            labels['coarse-class'].append('other')
            labels['fine-class'].append(instance_name.text)
            labels['role'].append(None)

    # BBOX CCORDINATES
    point_spaces = root.findall('./objects/object/points')
    for point_space in point_spaces:
        # remove the last coordinate points
        my_points = point_space.findall('point')[:4]  
        coords = []
        for my_point in my_points:
            coord = []
            for point in my_point.text.split(','):
                coord.append(float(point))
            coords.append(coord)
        labels['obboxes'].append(coords)
        labels['hbboxes'].append(BBox.get_hbb_from_obb(coords))
        fill_none_to_empty_keys(labels,not_available_tasks)

    return labels


def read_rareplanes_real_label(label_path):
    labels = init_satellitepy_label()

    # ## Available tasks for rareplanes_real
    available_tasks = ['hbboxes', 'obboxes', 'coarse-class', 'role', 'attributes_engines_no-engines', 'attributes_engines_propulsion',
                       'attributes_fuselage_canards', 'attributes_fuselage_length', 'attributes_wings_wing-span',
                       'attributes_wings_wing-shape', 'attributes_wings_wing-position', 'attributes_tail_no-tail-fins']

    # ## All possible tasks
    all_tasks = get_all_satellitepy_keys()

    # ## Not available tasks
    not_available_tasks = [task for task in all_tasks if not task in available_tasks or available_tasks.remove(task)]

    with open(label_path, 'r') as f:
        file = json.load(f)

    for annotation in file['annotations']:
        points = annotation['segmentation'][0]

        A = (points[0], points[1])
        B = (points[2], points[3])
        C = (points[4], points[5])
        D = (points[6], points[7])
        # converting polygon-annotations to bounding box
        vecBD = tuple(np.subtract(D, B))
        middle = tuple(np.add(B, np.divide(vecBD, 2)))
        vecToC = tuple(np.subtract(C, middle))
        vecToA = tuple(np.subtract(A, middle))

        corners = [np.add(D, vecToA).tolist(), np.add(D, vecToC).tolist(), np.add(B, vecToC).tolist(),
                   np.add(B, vecToA).tolist()]
        labels['obboxes'].append(corners)
        labels['hbboxes'].append(BBox.get_hbb_from_obb(corners))
        labels['coarse-class'].append('airplane')
        labels['attributes']['engines']['no-engines'].append(int(annotation['num_engines']))
        labels['attributes']['engines']['propulsion'].append(annotation['propulsion'])
        canards =  annotation['canards']
        if canards=='yes':
            labels['attributes']['fuselage']['canards'].append(True)
        elif canards=='no':
            labels['attributes']['fuselage']['canards'].append(False)     
        labels['attributes']['fuselage']['length'].append(float(annotation['length']))
        labels['attributes']['wings']['wing-span'].append(float(annotation['wingspan']))
        labels['attributes']['wings']['wing-shape'].append(annotation['wing_type'])
        labels['attributes']['wings']['wing-position'].append(annotation['wing_position'])
        labels['attributes']['tail']['no-tail-fins'].append(int(annotation['num_tail_fins']))
        role = annotation['role']
        labels['role'].append(role)

        fill_none_to_empty_keys(labels, not_available_tasks)
    return labels


def read_rareplanes_synthetic_label(label_path, mask_path):
    labels = init_satellitepy_label()

    # ## Available tasks for rareplanes_synthetic
    available_tasks = ['hbboxes', 'obboxes', 'coarse-class', 'fine-class', 'very-fine-class', 'role']

    mask_exists = True if mask_path else False
    if mask_exists:
        available_tasks.append('masks')

    # ## All possible tasks
    all_tasks = get_all_satellitepy_keys()

    # ## Not available tasks
    not_available_tasks = [task for task in all_tasks if not task in available_tasks or available_tasks.remove(task)]

    with open(label_path, 'r') as f:
        file = json.load(f)

    for annotation in file['annotations']:
        points = annotation['segmentation'][0]

        A = (points[0], points[1])
        B = (points[2], points[3])
        C = (points[4], points[5])
        D = (points[6], points[7])
        # converting polygon-annotations to bounding box
        vecBD = tuple(np.subtract(D, B))
        middle = tuple(np.add(B, np.divide(vecBD, 2)))
        vecToC = tuple(np.subtract(C, middle))
        vecToA = tuple(np.subtract(A, middle))

        corners = [np.add(D, vecToA).tolist(), np.add(D, vecToC).tolist(), np.add(B, vecToC).tolist(),
                   np.add(B, vecToA).tolist()]

        # masks missing

        labels['obboxes'].append(corners)
        labels['hbboxes'].append(BBox.get_hbb_from_obb(corners))
        labels['coarse-class'].append('airplane')

        name = '_'.join(annotation['full'].split('_')[3:])
        fine = name.split('-')[0]
        labels['fine-class'].append(fine)

        if name.split('-').__len__() > 1:
            labels['very-fine-class'].append(name)
        else:
            labels['very-fine-class'].append(None)

        role = annotation['category_id']

        if role == 1:  # Small Civil Transport/Utility
            labels['role'].append('Small Civil Transport/Utility')
        elif role ==2:  # Medium Civil Transport/Utility
            labels['role'].append('Medium Civil Transport/Utility')
        elif role ==3:  # Large Civil Transport/Utility
            labels['role'].append('Large Civil Transport/Utility')
        else:
            raise Exception(f'Unexpected role found: {role}')

        fill_none_to_empty_keys(labels, not_available_tasks)

    if mask_exists:
        labels = set_mask(labels,mask_path, bbox_type='obboxes', mask_type='rareplanes')

    return labels

def read_VHR_label(label_path):
    labels = init_satellitepy_label()
    # Get all not available tasks so we can append None to those tasks
    ## Default available tasks for VHR
    available_tasks=['hbboxes', 'coarse-class', 'fine-class']
    ## All possible tasks
    all_tasks = get_all_satellitepy_keys()
    ## Not available tasks
    not_available_tasks = [task for task in all_tasks if not task in available_tasks or available_tasks.remove(task)]
    
    lut = { # Look-up Table for fine-class
                "1": None,
                "2": None,
                "3": 'storagetank',
                "4": 'baseballdiamond',
                "5": 'tennis court',
                "6": 'basketballcourt',
                "7": 'groundtrackfield',
                "8": 'harbor',
                "9": 'bridge',
               "10": None,
                }
    files = Path(label_path).glob('*.txt')

    handler = open(label_path, "r")
    for line in handler.readlines():
        if line == "\n":
            continue

        vals = line.replace('(','').replace(')','').replace('\n', '').split(',')
        vals = [int(x) for x in vals] # x1, y1, x2, y2

        labels['hbboxes'].append([[vals[0], vals[1]], [vals[2], vals[1]], [vals[2], vals[3]], [vals[0], vals[3]]])

        typ = str(vals[-1])
        if typ == "1":
            labels['coarse-class'].append('airplane')
        elif typ == "2":
            labels['coarse-class'].append('ship')
        elif typ == "10":
            labels['coarse-class'].append('vehicle')
        else:
            labels['coarse-class'].append('other')

        labels['fine-class'].append(lut[typ])
    handler.close()
            
    fill_none_to_empty_keys(labels,not_available_tasks)
    return labels


def read_dior_label(label_path):
    labels = init_satellitepy_label()
    # Get all not available tasks so we can append None to those tasks
    ## Default available tasks for VHR
    available_tasks=['hbboxes', 'obboxes', "difficulty", 'coarse-class', "fine-class"]
    ## All possible tasks
    all_tasks = get_all_satellitepy_keys()
    ## Not available tasks
    not_available_tasks = [task for task in all_tasks if not task in available_tasks or available_tasks.remove(task)]
    
    handler = open(label_path, "r")
    root = ( ET.parse(handler) ).getroot()
    for elem in root.findall("object"):
            typ = elem.find("name").text
            if typ == "ship" or typ == "vehicle" or typ == "airplane" : # object of interest
                    labels['coarse-class'].append(typ)
                    labels['fine-class'].append(None)
            else :
                    labels['coarse-class'].append("other")
                    labels['fine-class'].append(typ)

            difficulty = str((elem.find("difficult").text))
            labels['difficulty'].append(difficulty)
            bndbox = elem.find("robndbox")
            x_left_top = int(bndbox.find("x_left_top").text)
            y_left_top = int(bndbox.find("y_left_top").text)
            x_left_bottom = int(bndbox.find("x_left_bottom").text)
            y_left_bottom = int(bndbox.find("y_left_bottom").text)
            x_right_bottom = int(bndbox.find("x_right_bottom").text)
            y_right_bottom = int(bndbox.find("y_right_bottom").text)
            x_right_top = int(bndbox.find("x_right_top").text)
            y_right_top = int(bndbox.find("y_right_top").text)

            corners = [[x_left_top, y_left_top], [x_left_bottom, y_left_bottom], [x_right_bottom, y_right_bottom], [x_right_top, y_right_top]]
            labels['obboxes'].append(corners)
            labels['hbboxes'].append(BBox.get_hbb_from_obb(corners))
            fill_none_to_empty_keys(labels,not_available_tasks)
    handler.close()

    return labels

def read_ship_net_label(label_path):
    labels = init_satellitepy_label()
    # classes = get_shipnet_classes()
    # Get all not available tasks so we can append None to those tasks
    ## Default available tasks for dota
    available_tasks=['hbboxes', 'obboxes', 'difficulty', 'coarse-class', 'fine-class', 'very-fine-class', 'role']
    ## All possible tasks
    all_tasks = get_all_satellitepy_keys()
    ## Not available tasks
    not_available_tasks = [task for task in all_tasks if not task in available_tasks or available_tasks.remove(task)]
    
    root = ET.parse(label_path).getroot()
    # Instance names
    objects = root.findall('./object')
    coarse_classes = {1:'ship',2:'other'} # 2 is dock, and other in our framework
    roles = {1:'Other Ship',2:'Warship',3:'Merchant Ship',4:'Dock'}
    fine_classes = get_shipnet_categories()
    for ship_object in objects:
        # print(ship_object.)
        # Difficulty
        labels['difficulty'].append(ship_object.find('difficult').text)

        # Coarse class
        coarse_class_ind = int(ship_object.find('level_0').text)
        coarse_class = coarse_classes[coarse_class_ind]
        labels['coarse-class'].append(coarse_class)
        if coarse_class == 'other':
            labels['role'].append(None)
            labels['fine-class'].append(None)
            labels['very-fine-class'].append(None)
            continue

        # Roles
        role_ind = int(ship_object.find('level_1').text)
        role = roles[role_ind]
        if role == 'Other Ship':
            labels['role'].append(None)
            labels['fine-class'].append(None)
            labels['very-fine-class'].append(None)
            continue

        labels['role'].append(role)
        # Fine-grained class
        fine_class_ind = int(ship_object.find('level_2').text)
        fine_class = fine_classes[fine_class_ind]
        if fine_class in ['Other Merchant', 'Other Warship']:
            labels['fine-class'].append(None)
            labels['very-fine-class'].append(None)
            continue
        labels['fine-class'].append(fine_class)

        # Very fine classes
        very_fine_class = ship_object.find('name').text
        if very_fine_class == fine_class:
            labels['very-fine-class'].append(None)
            continue
        labels['very-fine-class'].append(very_fine_class)

    # BBOX CCORDINATES
    point_spaces = root.findall('./object/polygon')
    for point_space in point_spaces:
        # remove the last coordinate points
        my_points = point_space.findall('.//')
        coords = []
        corner = []
        for my_point in my_points:
            corner.append(int(float(my_point.text)))
            if 'y' in my_point.tag:
                coords.append(corner)
                corner = []
        labels['obboxes'].append(coords)
        labels['hbboxes'].append(BBox.get_hbb_from_obb(coords))
        fill_none_to_empty_keys(labels,not_available_tasks)
    return labels

def read_ucas_label(label_path):
    labels = init_satellitepy_label()
    # Get all not available tasks so we can append None to those tasks
    ## Default available tasks for dota
    available_tasks=['hbboxes', 'obboxes', 'coarse-class']
    ## All possible tasks
    all_tasks = get_all_satellitepy_keys()
    ## Not available tasks
    not_available_tasks = [task for task in all_tasks if not task in available_tasks or available_tasks.remove(task)]

    file = open(label_path, 'r')
    for line in file.readlines():
        bbox = line.split()[:8]
        coords_x = bbox[0::2]
        coords_y = bbox[1::2]
        coords = []
        corner = []
        for i in range(0, len(coords_x)):
            corner.append(int(float(coords_x[i])))
            corner.append(int(float(coords_y[i])))
            coords.append(corner)
            corner = []
        labels['obboxes'].append(coords)
        labels['hbboxes'].append(BBox.get_hbb_from_obb(coords))

        # Using label path to determine object type
        if 'CAR' in str(label_path):
            labels['coarse-class'].append('vehicle')
        elif 'PLANE' in str(label_path):
            labels['coarse-class'].append('airplane')
        else:
            labels['coarse-class'].append('other')

        fill_none_to_empty_keys(labels,not_available_tasks)
    return labels

def read_satellitepy_label(label_path):
    with open(label_path,'r') as f:
        labels = json.load(f)
    return labels

def read_isprs_label(label_path):
    labels = init_satellitepy_label()
    # Get all not available tasks so we can append None to those tasks
    ## Default available tasks for dota
    available_tasks=['hbboxes', 'coarse-class', 'masks']
    ## All possible tasks
    all_tasks = get_all_satellitepy_keys()
    ## Not available tasks
    not_available_tasks = [task for task in all_tasks if not task in available_tasks or available_tasks.remove(task)]

    hbboxes, masks = parse_potsdam_labels(label_path)

    labels['masks'] = masks

    for i in range(len(hbboxes)):
        labels['coarse-class'].append('vehicle')
        labels['hbboxes'].append(hbboxes[i])
        fill_none_to_empty_keys(labels, not_available_tasks)

    return labels

def read_vedai_label(label_path):
    labels = init_satellitepy_label()
    classes = get_vedai_classes()
    available_tasks=['hbboxes', 'obboxes', 'coarse-class', 'fine-class']
    all_tasks = get_all_satellitepy_keys()
    ## Not available tasks
    not_available_tasks = [task for task in all_tasks if not task in available_tasks or available_tasks.remove(task)]

    file = open(label_path, 'r')
    for line in file.readlines():
        separated = line.split()[3:]
        if len(separated) < 9:
            continue
        coords_x = separated[3:7]
        coords_y = separated[7:11]
        coords = []
        corner = []
        
        for i in range(0, len(coords_x)):
            corner.append(int(coords_x[i]))
            corner.append(int(coords_y[i]))
            coords.append(corner)
            corner = []
        
        labels['obboxes'].append(coords)
        labels['hbboxes'].append(BBox.get_hbb_from_obb(coords))
        
        fine_class = classes.get(int(separated[0]))
        
        if fine_class == "boat":
            labels['coarse-class'].append("ship")
            labels['fine-class'].append(None)
        elif fine_class == "plane":
            labels['coarse-class'].append("airplane")
            labels['fine-class'].append(None)
        elif fine_class == "other" or fine_class == "large":
            labels['coarse-class'].append("other")
            labels['fine-class'].append(None)
        else:
            labels['coarse-class'].append("vehicle")
            labels['fine-class'].append(fine_class)
            
        fill_none_to_empty_keys(labels, not_available_tasks)
        
    return labels
