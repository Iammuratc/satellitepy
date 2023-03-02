import numpy as np
import xml.etree.ElementTree as ET


def read_label(label_path,label_format):
    if label_format=='dota' or label_format=='DOTA':
        return read_dota_label(label_path)
    elif label_format=='fair1m':
        return read_fair1m_label(label_path)
    else:
        print('---Label format is not defined---')
        exit(1)



def read_dota_label(label_path):
    labels={
        'bboxes':[],
        'instance_names':[],
        'difficulty':[]}
    with open(label_path, 'r') as f:
        for line in f.readlines():
            # bbox_labels = line[:-1].split(' ')[:-1]
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


def read_fair1m_label(label_path):
    labels={
        'bboxes':[],
        'instance_names':[]
    }

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

if __name__=='__main__':
    # label_path = './data/DOTA/train/bounding_boxes/P0023.txt'
    # label_path = './data/fair1m/train/patches/patches_512/labels_binary_dota/13894_x_0_y_288.txt'
    # print(read_dota_label(label_path))
    label_path = './data/fair1m/train/bounding_boxes/13341.xml'
    print(read_fair1m_label(label_path))