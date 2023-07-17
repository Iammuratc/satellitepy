import cv2
import numpy as np

def set_mask(labels,mask_path,bbox_type):
    """
    Set the masks key in the satellitepy dict by using the bboxes in the dict
    Parameters
    ----------
    labels : dict
        Satellitepy dict
    mask_path : Path
        Mask image path
    Returns
    -------
    labels : dict
        Satellitepy dict
    """
    mask = cv2.cvtColor(cv2.imread(str(mask_path)), cv2.COLOR_RGB2GRAY)
    empty_mask = np.zeros((mask.shape[0],mask.shape[1]))
    for bbox in labels[bbox_type]:
        mask_0 = empty_mask.copy()
        cv2.fillPoly(mask_0, [np.array(bbox,dtype=int)], 1)
        coords = np.argwhere((mask_0 == 1) & (mask != 0)).T.tolist() # y,x
        labels['masks'].append([coords[1],coords[0]]) # x,y
    return labels

def get_xview_classes():
    classes={
            'vehicles':{
                17: 'Passenger Vehicle',
                23: 'Truck',
                53: 'Engineering Vessel',
                18: 'Small Car',
                19: 'Bus',
                20: 'Pickup Truck',
                21: 'Utility Truck',
                24: 'Cargo Truck',
                25: 'Truck w/Box',
                26: 'Truck Tractor',
                28: 'Truck w/Flatbed', 
                29: 'Truck w/Liquid',
                54: 'Tower Crane',
                55: 'Container Crane',
                56: 'Reach Stacker',
                57: 'Straddle Carrier',
                59: 'Mobile Crane',
                60: 'Dump Truck',
                61: 'Haul Truck',
                62: 'Scraper/Tractor',
                63: 'Front Loader',
                64: 'Excavator',
                65: 'Cement Mixer',
                66: 'Ground Grader',
                32: 'Crane Truck',
                33: 'Railway Vehicle',
                34: 'Passenger Car',
                35: 'Cargo Car',
                36: 'Flat Car',
                37: 'Tank Car',
                38: 'Locomotive',

            },
            'ships':{
                40: 'Maritime Vessel',
                41: 'Motorboat',
                42: 'Sailboat',
                44: 'Tugboat',
                45: 'Barge',
                47: 'Fishing Vessel',
                49: 'Ferry',
                50: 'Yacht',
                51: 'Container Ship',
                52: 'Oil Tanker'
            },
            'airplanes':{
                11: 'Fixed-Wing Aircraft',
                12: 'Small Aircraft',
                13: 'Cargo Plane'
            },
            'helicopter': {
                15: 'Helicopter'
            },
            'objects': {
                73: 'Building',
                71: 'Hut/Tent',
                72: 'Shed',
                74: 'Aircraft Hangar',
                76: 'Damaged Building',
                77: 'Facility',	  
                84: 'Helipad',
                93: 'Pylon',
                91: 'Shipping Container',
                89: 'Shipping Container Lot',
                85: 'Storage Tank',
                83: 'Vehicle Lot',
                79: 'Construction Site',
                94: 'Tower Structure'
            }
        }
    return classes

def get_satellitepy_table():
    """
    This function returns an indexing/mapping table for the satellitepy dict
    Returns
    -------
    satellitepy_table : dict of dict
        If values are the instance of string or int, then the index number
        If values are the instance of float, then max and min values
    """
    satellitepy_table = {
        'coarse-class':{
            'airplane':0,
            'ship':1,
            'vehicle':2,
            'helicopter':3,
            'other':4},
        'fine-class':{
            'small-vehicle':0,
            'large-vehicle':1,
            'harbor':2,
            'tennis-court':3,
            'soccer-ball-field':4,
            'baseball-diamond':5,
            'ground-track-field':6,
            'swimming-pool':7,
            'roundabout':8,
            'basketball-court':9,
            'storage-tank':10,
            'bridge':11},
        'very-fine-class':{},
        'role':{
            'Small Civil Transport/Utility':0,
            'Medium Civil Transport/Utility':1,
            'Large Civil Transport/Utility':2,
            'Military Transport/Utility/AWAC':3,
            'Military Fighter/Interceptor/Attack':4,
            'Military Bomber':5,
            'Military Trainer':6,},
        'difficulty':{
            '0':0,
            '1':1},
        # calculated from rareplanes train
        'attributes':{
            'engines':{
                'no-engines':{
                    0:0,
                    1:1,
                    2:2,
                    3:3,
                    4:4
                    },
                'propulsion':{
                    'propeller':0,
                    'jet':1,
                    'unpowered':2
                    }
            },
            'fuselage':{
                'canards':{
                    False:0,
                    True:1},
                'length':{
                    'max':82.5,
                    'min':4.0}
            },
            'wings':{
                'wing-span':{
                    'max':80.0,
                    'min':4.0
                    },
                'wing-shape':{
                    'swept':0,
                    'straight':1,
                    'variable swept':2,
                    'delta':3
                    },
                'wing-position':{
                    'mid/low mounted':0,
                    'high mounted':1
                    }
            },
            'tail':{
                'no-tail-fins':{
                    1:0,
                    2:1
                    }
            }
        }
    } 

    # Add the merged class
    # For example, this is a solution to be able to train the bbavector on the original dota dataset
    len_coarse_class = len(satellitepy_table['coarse-class'])
    len_fine_class = len(satellitepy_table['fine-class'])
    satellitepy_table['merged-class'] = satellitepy_table['coarse-class'] 
    for key,value in satellitepy_table['fine-class'].items():
        satellitepy_table['merged-class'][key] = value+len_coarse_class 
    for key,value in satellitepy_table['very-fine-class'].items():
        satellitepy_table['merged-class'][key] = value+len_coarse_class+len_fine_class
    return satellitepy_table


def get_satellitepy_dict_values(satellitepy_dict,task):
    '''
    Get the satellitepy dict values by parsing task.
    Parameters
    ----------
        satellitepy_dict : dict
            Satellitepy formatted dict
        task : str
            Task name. E.g. attributes_tail_no-tail-fins
    Returns
    -------
        values : list or dict
            The values of the corresponding task
    '''

    keys = task.split('_')
    if len(keys)==1:
        return satellitepy_dict[keys[0]]
    elif len(keys)==2:
        return satellitepy_dict[keys[0]][keys[1]]
    elif len(keys)==3:
        return satellitepy_dict[keys[0]][keys[1]][keys[2]]
    else:
        return 0

# def set_merged_task_values(satellitepy_dict,merged_task_name):
#     '''
#     Merge the satellitepy dict task values. 
#     For example, this function can merge coarse-class and fine-class values into one list, 
#     so the models can train be trained on this list. <merged_task_name> must have the tasks in the order of fine to coarse
    
#     Parameters
#     ----------
#         satellitepy_dict : dict
#             Satellitepy formatted dict
#         merged_task_name : str
#             Task names. E.g. fine-class--coarse-class
#     Returns
#     -------
#         satellitepy_dict : dict
#             This dict includes a key whose values are the merged task values
#     '''

#     satellitepy_dict_values = {}
#     merged_task = []
#     tasks = merged_task_name.split('--')

#     for task in tasks:
#         satellitepy_dict_values[task] = get_satellitepy_dict_values(satellitepy_dict,task)
    

#     len_tasks = len(tasks)
#     len_values = len(satellitepy_dict_values[tasks[0]])

#     for i_value in range(len_values):
#         for i_task in range(len(tasks)):
#             task_value = satellitepy_dict_values[tasks[i_task]][i_value]
#             if task_value != None:
#                 merged_task.append(task_value)
#                 break
    
#     satellitepy_dict[merged_task_name] = merged_task
#     return satellitepy_dict

