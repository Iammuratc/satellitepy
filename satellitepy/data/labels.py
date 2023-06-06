import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from satellitepy.data.utils import get_xview_classes, set_mask
import json

from satellitepy.data.cutout.geometry import BBox

def read_label(label_path,label_format, mask_path = None):
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
        return read_vhr_label(label_path)
    elif label_format == 'rareplanes_real' or label_format == 'rarePlanes_real':
        return read_rareplanes_real_label(label_path)
    elif label_format == 'rareplanes_synthetic' or label_format == 'rarePlanes_synthetic':
        return read_rareplanes_synthetic_label(label_path)
    elif label_format == 'ship_net':
        return read_ship_net_label(label_path)
    elif label_format == 'ucas':
        return read_ucas_label(label_path)
    elif label_format == 'xview':   
        print('Please run tools/data/split_xview_into_satellitepy_labels.py to get the satellitepy labels.'
              ' Then pass label_format as satellitepy for those labels.')
        exit(1)
    else:
        print('---Label format is not defined---')
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


def fill_none_to_empty_keys(labels,not_available_tasks):
    """
    Append None to non existing tasks for one object 
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
        object-role : list of str 
            INSERT HERE
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
            'role' : dict of str
                'civil' : list of str
                    large_transport, medium_transport, small_transport
                'military' : list of str
                    fighter, bomber, transport, trainer
    """
    labels={
        'hbboxes':[],
        'obboxes': [],
        'masks':[],
        'coarse-class':[],
        'fine-class':[],
        'object-role':[],
        'very-fine-class':[],
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
            },
            'role':{
                'civil':[],
                'military':[]
            }
        }
    }
    return labels    


def read_dota_label(label_path, mask_path=None):
    labels = init_satellitepy_label()
    # Get all not available tasks so we can append None to those tasks
    ## Default available tasks for dota
    available_tasks=['obboxes','difficulty','coarse-class','fine-class']
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
            ### class_0 = vehicle, class_1 = large-vehicle
            category_words = category.split('-')
            if len(category_words) == 2 and category_words[1]=='vehicle':
                labels['coarse-class'].append(category_words[1]) # vehicle
                labels['fine-class'].append(category) # small-vehicle
            elif category=='plane' or category=='ship' or category=='helicopter':
                # Airplane is the common word
                category = 'airplane' if category == 'plane' else category
                labels['coarse-class'].append(category) # plane, ship
                labels['fine-class'].append(None) #
            else:
                labels['coarse-class'].append('object') #
                labels['fine-class'].append(category) #
            # BBoxes
            bbox_corners_flatten = [[float(corner) for corner in bbox_line[:category_i]]]
            bbox_corners = np.reshape(bbox_corners_flatten, (4, 2)).tolist()
            labels['obboxes'].append(bbox_corners)
            fill_none_to_empty_keys(labels,not_available_tasks)

        # Mask
        if mask_exists:
            labels = set_mask(labels,mask_path,bbox_type='obboxes')


    return labels

def read_fair1m_label(label_path):
    labels = init_satellitepy_label()
    # Get all not available tasks so we can append None to those tasks
    ## Default available tasks for dota
    available_tasks=['obboxes','coarse-class','fine-class']
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
        if instance_name.text in ['A321','A220','other-airplane','ARJ21','Boeing737','Boeing747','Boeing787','A330','Boeing777','C919','A350']:
            labels['coarse-class'].append('airplane')
            labels['fine-class'].append(instance_name.text)
        elif instance_name.text in ['Cargo Truck','Small Car','Dump Truck','Van','Excavator','Bus','other-vehicle','Truck Tractor','Tractor','Trailer']:
            labels['coarse-class'].append('vehicle')
            labels['fine-class'].append(instance_name.text)
        elif instance_name.text in ['Liquid Cargo Ship','Passenger Ship','Dry Cargo Ship','Motorboat','Engineering Ship','Tugboat','Fishing Boat','other-ship','Warship']:
            labels['coarse-class'].append('ship')
            labels['fine-class'].append(instance_name.text)
        else:
            labels['coarse-class'].append('object')
            labels['fine-class'].append(instance_name.text)

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
        fill_none_to_empty_keys(labels,not_available_tasks)

    return labels


def read_rareplanes_real_label(label_path):
    labels = init_satellitepy_label()

    # ## Available tasks for rareplanes_real
    available_tasks = ['hbboxes', 'obboxes', 'coarse-class', 'attributes_engines_no-engines', 'attributes_engines_propulsion',
                       'attributes_fuselage_canards', 'attributes_fuselage_length', 'attributes_wings_wing-span',
                       'attributes_wings_wing-shape', 'attributes_wings_wing-position', 'attributes_tail_no-tail-fins',
                       'attributes_role_civil', 'attributes_role_military']

    # ## All possible tasks
    all_tasks = get_all_satellitepy_keys()

    # ## Not available tasks
    not_available_tasks = [task for task in all_tasks if not task in available_tasks or available_tasks.remove(task)]

    with open(label_path, 'r') as f:
        file = json.load(f)

    for annotation in file['annotations']:
        points = annotation['segmentation'][0]

        labels['hbboxes'].append(points)

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
        if role == 'Small Civil Transport/Utility':
            labels['attributes']['role']['civil'].append(role)
            labels['attributes']['role']['military'].append(None)
        elif role== 'Medium Civil Transport/Utility':
            labels['attributes']['role']['civil'].append(role)
            labels['attributes']['role']['military'].append(None)
        elif role== 'Large Civil Transport/Utility':
            labels['attributes']['role']['civil'].append(role)
            labels['attributes']['role']['military'].append(None)
        elif role=='Military Transport/Utility/AWAC':
            labels['attributes']['role']['military'].append(role)
            labels['attributes']['role']['civil'].append(None)
        elif role=='Military Fighter/Interceptor/Attack':
            labels['attributes']['role']['military'].append(role)
            labels['attributes']['role']['civil'].append(None)
        elif role=='Military Trainer':
            labels['attributes']['role']['military'].append(role)
            labels['attributes']['role']['civil'].append(None)
        elif role=='Military Bomber':
            labels['attributes']['role']['military'].append(role)
            labels['attributes']['role']['civil'].append(None)
        else:
            raise Exception(f'Unexpected role found: {role}')

        fill_none_to_empty_keys(labels, not_available_tasks)
    return labels


def read_rareplanes_synthetic_label(label_path):
    print('read_synthetic_rareplanes')
    labels = init_satellitepy_label()

    # ## Available tasks for rareplanes_synthetic
    available_tasks = ['hbboxes'  'obboxes', 'coarse-class', 'attributes_engines_no-engines', 'attributes_engines_propulsion',
     'attributes_fuselage_canards', 'attributes_fuselage_length', 'attributes_wings_wing-span',
     'attributes_wings_wing-shape', 'attributes_wings_wing-position', 'attributes_tail_no-tail-fins',
     'attributes_role_civil', 'attributes_role_military']

    # ## All possible tasks
    all_tasks = get_all_satellitepy_keys()

    # ## Not available tasks
    not_available_tasks = [task for task in all_tasks if not task in available_tasks or available_tasks.remove(task)]

    with open(label_path, 'r') as f:
        file = json.load(f)

    for annotation in file['annotations']:
        points = annotation['segmentation'][0]

        labels['hbboxes'].append(points)

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
        labels['coarse-class'].append('airplane')
        labels['attributes']['engines']['no-engines'].append(int(annotation['num_engines']))
        labels['attributes']['engines']['propulsion'].append(annotation['propulsion'])
        canards = annotation['canards']
        if canards == 'yes':
            labels['attributes']['fuselage']['canards'].append(True)
        elif canards == 'no':
            labels['attributes']['fuselage']['canards'].append(False)
        labels['attributes']['fuselage']['length'].append(float(annotation['length']))
        labels['attributes']['wings']['wing-span'].append(float(annotation['wingspan']))
        labels['attributes']['wings']['wing-shape'].append(annotation['wing_type'])
        labels['attributes']['wings']['wing-position'].append(annotation['wing_position'])
        labels['attributes']['tail']['no-tail-fins'].append(int(annotation['num_tail_fins']))
        role = annotation['category_id']
        if role == 1:  # Small Civil Transport/Utility
            role = 'Small_Civil_Transport/Utility'
            labels['attributes']['role']['civil'].append(role)
            labels['attributes']['role']['military'].append(None)
        elif role ==2:  # Medium Civil Transport/Utility
            role = 'Medium_Civil_Transport/Utility'
            labels['attributes']['role']['civil'].append(role)
            labels['attributes']['role']['military'].append(None)
        elif role ==3:  # Large Civil Transport/Utility
            role = 'Large_Civil_Transport/Utility'
            labels['attributes']['role']['civil'].append(role)
            labels['attributes']['role']['military'].append(None)
        else:
                raise Exception(f'Unexpected role found: {role}')

        fill_none_to_empty_keys(labels, not_available_tasks)
    return labels
 
def read_VHR_label(label_path):
    labels = init_satellitepy_label()
    # Get all not available tasks so we can append None to those tasks
    ## Default available tasks for VHR
    available_tasks=['bboxes', 'coarse-class', "fine-class"]
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
    for file in files:
        handler = open(file, "r")
        for line in handler.readlines():
            vals = line.replace('(','').replace(')','').split(',')
            
            labels['bboxes'].append([vals[0:2],vals[2:4]])
            # The first two values are the top-right, the next two the bottom left corner
            
            typ = vals[-1].strip()
            if typ == "1":
                    labels['coarse-class'].append('airplane')
            elif typ == "2":
                    labels['coarse-class'].append('ship')
            elif typ == "10":
                    labels['coarse-class'].append('vehicle')
                                   
            else:
                    labels['coarse-class'].append('object')
                
            labels['fine-class'].append(lut[typ])
        handler.close()
            
    fill_none_to_empty_keys(labels,not_available_tasks)
    return labels


def read_dior_label(label_path):
    labels = init_satellitepy_label()
    # Get all not available tasks so we can append None to those tasks
    ## Default available tasks for VHR
    available_tasks=['bboxes', "difficulty", 'coarse-class', "fine-class"]
    ## All possible tasks
    all_tasks = get_all_satellitepy_keys()
    ## Not available tasks
    not_available_tasks = [task for task in all_tasks if not task in available_tasks or available_tasks.remove(task)]
    
    handler = open(file, "r")
    root = ( ET.parse(handler) ).getroot()
    for elem in root.findall("object"):
            typ = elem.find("name").text
            if typ == "ship" or typ == "vehicle" or typ == "airplane" : # object of interest
                    labels['coarse-class'].append(typ)
                    labels['fine-class'].append(None)
            else :
                    labels['coarse-class'].append("object")
                    labels['fine-class'].append(typ)
            bndbox = elem.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            xmax = int(bndbox.find("xmax").text)
            ymin = int(bndbox.find("ymin").text)
            ymax = int(bndbox.find("ymax").text)
            labels['bboxes'].append([[xmin,ymin],[xmax,ymax]])        
    handler.close()

    fill_none_to_empty_keys(labels,not_available_tasks)
    return labels

def read_ship_net_label(label_path):
    labels = init_satellitepy_label()
    # Get all not available tasks so we can append None to those tasks
    ## Default available tasks for dota
    available_tasks=['obboxes', 'difficulty', 'coarse-class','fine-class']

    ## All possible tasks
    all_tasks = get_all_satellitepy_keys()
    ## Not available tasks
    not_available_tasks = [task for task in all_tasks if not task in available_tasks or available_tasks.remove(task)]
    root = ET.parse(label_path).getroot()
    # Instance names
    instance_names = root.findall('./object/name')
    for instance_name in instance_names:
        labels['coarse-class'].append('ship')
        labels['fine-class'].append(instance_name.text)
    instance_difficulties = root.findall('./object/difficult')
    for instance_difficulty in instance_difficulties:
        labels['difficulty'].append(instance_difficulty.text)

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
        fill_none_to_empty_keys(labels,not_available_tasks)
    return labels

def read_ucas_label(label_path):
    labels = init_satellitepy_label()
    # Get all not available tasks so we can append None to those tasks
    ## Default available tasks for dota
    available_tasks=['obboxes', 'coarse-class']
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

        # Using label path to determine object type
        if 'CAR' in str(label_path):
            labels['coarse-class'].append('car')
        elif 'PLANE' in str(label_path):
            labels['coarse-class'].append('airplane')
        else:
            labels['coarse-class'].append(None)

        fill_none_to_empty_keys(labels,not_available_tasks)
    return labels

def read_satellitepy_label(label_path):
    with open(label_path,'r') as f:
        labels = json.load(f)
    return labels

