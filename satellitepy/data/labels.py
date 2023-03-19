import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import json

def read_label(label_path,label_format):
    if isinstance(label_path,Path):
        label_path = str(label_path)
    if label_format=='dota' or label_format=='DOTA':
        return read_dota_label(label_path)
    elif label_format=='fair1m':
        return read_fair1m_label(label_path)
    elif label_format=='satellitepy':
        return read_satellitepy_label(label_path)
    else:
        print('---Label format is not defined---')
        exit(1)


def init_satellitepy_label():
    """
    This function creates an empty labels dict in satellitepy format.
    Returns
    -------
    labels : dict of str
        bboxes : list
            Bounding box corners for every object
        mask : list of Path
            Path to segmentation mask of objects
        classes : dict of str
            '0' : coarse grained classes (e.g., airplane,ship)
            '1' : fine grained classes (e.g., A220, passenger ship)
            '2' : very fine grained classes (e.g., A220-100)
        difficulty : list of int
            Detection difficulty of the object. Only DOTA provides this. 
            For example, clouds make the detection task difficult. 
        attributes : dict of str
            This value only serves for Rareplanes at the moment. 
            Please check the rareplanes paper for details.
            'engines' : dict of str
                'no_engines' : int
                    Number of engines
                'propulsion' : str
                    unpowered, jet, propeller
            'fuselage' : dict of str
                'canards' : bool
                'length' : float
            'wings' : dict of str
                'wing_span' : float
                'wing_shape' : str
                    swept, straight, delta, variable_swept
                'wing_position' : str
                    low_mounted, high_mounted
            'tail' : dict of str
                'no_tail_fins' : int
                    1, 2
            'role' : dict of str
                'civil' : str
                    large_transport, medium_transport, small_transport
                'military' : str
                    fighter, bomber, transport, trainer
    """
    labels={
        'bboxes':[],
        'classes':{
            '0':[],
            '1':[],
            '2':[]
        },
        'difficulty':[],
        'attributes':{
            'engines':{
                'no_engines':[],
                'propulsion':[]
            },
            'fuselage':{
                'canards':[],
                'length':[]
            },
            'wings':{
                'wing_span':[],
                'wing_shape':[],
                'wing_position':[],
            },
            'tail':{
                'no_tail_fins':[]
            },
            'role':{
                'civil':[],
                'military':[]
            }
        }
    }
    return labels    


def read_dota_label(label_path):
    labels = init_satellitepy_label()
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
            else:
                continue

            category = bbox_line[category_i].rstrip()

            bbox_corners_flatten = [[float(corner) for corner in bbox_line[:category_i]]]
            bbox_corners = np.reshape(bbox_corners_flatten, (4, 2)).tolist()

            labels['bboxes'].append(bbox_corners)
            labels['instance_names'].append(category)
    return labels


def read_rareplanes_label(label_path):
    labels = init_satellitepy_label()

    return labels

def read_fair1m_label(label_path):
    labels = init_satellitepy_label()
    root = ET.parse(label_path).getroot()

    file_name = root.findall('./source/filename')[0].text

    # INSTANCE NAMES
    instance_names = root.findall(
        './objects/object/possibleresult/name')  # [0].text
    for instance_name in instance_names:
        labels['instance_names'].append(instance_name.text)

    # BBOX CCORDINATES
    point_spaces = root.findall('./objects/object/points')
    for point_space in point_spaces:
        # remove the last coordinate points
        my_points = point_space.findall('point')[:4]  
        coords = []
        for my_point in my_points:
            # [[[x1,y1],[x2,y2]],[[x1,y1]]]
            coord = []
            for point in my_point.text.split(','):
                coord.append(float(point))
            coords.append(coord)
        labels['bboxes'].append(coords)
    return labels

def read_satellitepy_label(label_path):
    labels = init_satellitepy_label()

    with open(label_path,'r') as f:
        labels_file = json.load(f)

    for key in labels.keys():
        labels[key] = labels_file[key]
    return labels

if __name__=='__main__':
    # label_path = './data/DOTA/train/bounding_boxes/P0023.txt'
    # label_path = './data/fair1m/train/patches/patches_512/labels_binary_dota/13894_x_0_y_288.txt'
    # print(read_dota_label(label_path))
    label_path = './data/fair1m/train/bounding_boxes/13341.xml'
    print(read_fair1m_label(label_path))