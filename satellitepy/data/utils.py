import cv2
import numpy as np
from satellitepy.data.bbox import BBox

from satellitepy.data.bbox import BBox

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

    for bbox in labels[bbox_type]:
        h = BBox.get_bbox_limits(np.array(bbox))
        mask_0 = np.zeros((mask.shape[0],mask.shape[1]))
        cv2.fillPoly(mask_0, [np.array(bbox,dtype=int)], 1)
        coords = np.argwhere((mask_0[h[2]:h[3], h[0]:h[1]] == 1) & (mask[h[2]:h[3], h[0]:h[1]] != 0)).T.tolist() # y,x
        labels['masks'].append([coords[1] + h[0],coords[0] + h[2]]) # x,y

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
            'airplane': 0,
            'ship': 1,
            'vehicle': 2,
            'helicopter': 3,
            'other': 4
        },
        'fine-class':
        {
			'A220' 						: 0,    # Fair1m
			'A321' 						: 1,    # Fair1m
			'A330' 						: 2,    # Fair1m
			'A350' 						: 3,    # Fair1m
			'AOE' 						: 4,    # Ship Net
			'ARJ21'						: 5,    # Fair1m
			'Arleigh Burke DD'			: 6,    # Ship Net
			'Asagiri DD'				: 7,    # Ship Net
            'AirbusA300'                : 98,   # Rareplanes_synthetic
            'AirbusA-319'               : 99,    # Rareplanes_synthetic
            'Airbus_A320'               : 100,
            'Airbus_A330'               : 101,
            'Airbus_A'                  : 102,
            'Airbus_A380'               : 103,
			'Atago DD'					: 8,    # Ship Net
			'Austin LL'					: 9,    # Ship Net
            'ATR_ATR'                   : 104,
            'BAE_146'                   : 105,
			'Barge'						: 10,   # Ship Net
            'Boeing_707'                : 106,
            'Boeing_717'                : 107,
            'Boeing_727'                : 108,
			'Boeing737'					: 11,   # Fair1m, Rareplanes_synthetic
			'Boeing747'					: 12,   # Fair1m, Rareplanes_synthetic
            'Boeing_757'                : 109,
            'Boeing_767'                : 110,
			'Boeing777'					: 13,   # Fair1m, Rareplanes_synthetic
			'Boeing787'					: 14,   # Fair1m
            'Boeing_BBJ'                : 111,
            'Bombardier_BD'             : 112,
            'Bombardier_Challenger'     : 113,
            'Bombardier_CRJ'            : 114,
            'Bombardier_Learjet'        : 115,
			'Bus'						: 15,   # Fair1m, Xview
			'C919'						: 16,   # Fair1m
			'Cargo Car'					: 17,   # Xview
			'Cargo Plane'				: 18,   # Xview
			'Cargo Truck'				: 19,   # Fair1m, Xview
			'Cement Mixer'				: 20,   # Xview
			'Commander'					: 21,   # Ship Net
			'Container Ship'			: 22,   # Ship Net, Xview
			'Crane Truck'				: 23,   # Xview
            'Cessna'                    : 116,
            'Cessna_170'                : 117,
            'Cessna_172'                : 118,
            'Cessna_310'                : 119,
            'Cessna_Citation'           : 120,
            'Dassault_Falcon'           : 121,
            'DeHavillandCanada_DHC'     : 122,
			'Dry Cargo Ship'			: 24,   # Fair1m
			'Dump Truck'				: 25,   # Fair1m, Xview
            'Embraer_ERJ'               : 123,
            'Embraer_Legacy'            : 124,
			'Engineering Ship'			: 26,   # Fair1m
			'Engineering Vessel'		: 27,   # Xview
			'EPF'						: 28,   # Ship Net
            'Enterprise'                : 96,   # Ship Net
			'Ferry'						: 29,   # Ship Net, Xview
			'Fishing Boat'				: 30,   # Fair1m
			'Fishing Vessel'			: 30,   # Ship Net, Xview
			'Fixed-Wing Aircraft'		: 31,   # Xview
			'Flat Car'					: 32,   # Xview
            'Fokker_100'                : 125,
			'Front Loader'				: 33,   # Xview
			'Ground Grader'				: 34,   # Xview
            'Gulfstream_G200'           : 126,
            'Gulfstream_GIII'           : 127,
			'Hatsuyuki DD'				: 35,   # Ship Net
			'Haul Truck'				: 36,   # Xview
            'HBC_Hawker'                : 128,
			'Hovercraft'				: 37,   # Ship Net
			'Hyuga DD'					: 38,   # Ship Net
			'large-vehicle'				: 39,   # Dota
            'Let_L'                     : 129,
			'LHA LL'					: 40,   # Ship Net
			'Liquid Cargo Ship'			: 41,   # Fair1m
            'LockheedCorp_L'            : 130,
			'Locomotive'				: 42,   # Xview
			'LSD 41 LL'					: 43,   # Ship Net
			'Maritime Vessel'			: 22,   # Xview
			'Masyuu AS'					: 44,   # Ship Net
            'McDonnellDouglas_DC'       : 131,
            'McDonnellDouglas_MD'       : 132,
			'Medical Ship'				: 45,   # Ship Net
            'Midway'                    : 95,   # Ship Net
            'Mobile Crane'              : 98,   # Xview
			'Motorboat'					: 46,   # Ship Net, Fair1m, Xview
			'Nimitz'					: 47,   # Ship Net
			'Oil Tanker'				: 48,   # Xview
			'Osumi LL'					: 49,   # Ship Net
			'Other Aircraft Carrier'	: 50,   # Ship Net
			'Other Auxiliary Ship'		: 51,   # Ship Net
			'Other Destroyer'			: 52,   # Ship Net
			'Other Frigate'				: 53,   # Ship Net
			'Other Landing'				: 54,   # Ship Net
			'Other Merchant'			: 55,   # Ship Net
			'Other Ship'				: 56,   # Ship Net
			'Other Warship'				: 57,   # Ship Net
			'other-airplane'			: 58,   # Fair1m
			'other-ship'				: 56,   # Fair1m
			'other-vehicle'				: 59,   # Fair1m
			'Passenger Car'				: 60,   # Xview
			'Passenger Ship'			: 61,   # Fair1m
			'Passenger Vehicle'			: 62,   # Xview
			'Patrol'					: 63,   # Ship Net
			'Perry FF'					: 64,   # Ship Net
			'Pickup Truck'				: 65,	# Xview
            'PiperAircraft_PA'          : 133,
			'Railway Vehicle'			: 66,   # Xview
			'Reach Stacker'				: 67,   # Xview
			'RoRo'						: 68,   # Ship Net
			'Sailboat'					: 69,   # Xview
			'Sanantonio AS'				: 70,   # Ship Net
			'Scraper/Tractor'			: 71,   # Xview
			'Small Aircraft'			: 72,   # Xview
			'Small Car'					: 73,   # Fair1m, Xview
			'small-vehicle'				: 73,   # Dota
			'Straddle Carrier'			: 75,   # Xview
			'Submarine'					: 76,   # Ship Net
            'SudAviation_Caravelle'     : 134,
			'Tank Car'					: 77,   # Xview
			'Test Ship'					: 78,   # Ship Net
			'Ticonderoga'				: 79,   # Ship Net
			'Tractor'					: 80,   # Fair1m
            'Trailer'                   : 97,   # Fair1m
			'Training Ship'				: 81,   # Ship Net
			'Truck'						: 19,   # Xview
			'Truck Tractor'				: 82,   # Fair1m, Xview
			'Truck w/Box'				: 83,   # Xview
			'Truck w/Flatbed'			: 84,   # Xview
			'Truck w/Liquid'			: 85,   # Xview
			'Tugboat'					: 86,   # Ship Net, Fair1m, Xview
            'Tupolev_154'               : 135,
			'Utility Truck'				: 87,   # Xview
			'Van'						: 88,   # Fair1m
			'Warship'					: 57,   # Fair1m
			'Wasp LL'					: 89,   # Ship Net
			'Yacht'						: 90,   # Ship Net, Xview
			'YuDao LL'					: 91,   # Ship Net
			'YuDeng LL'					: 92,   # Ship Net
			'YuTing LL'					: 93,   # Ship Net
			'YuZhao LL'					: 94    # Ship Net
		 },
        'very-fine-class':{
        },
        'role':{
            'Small Civil Transport/Utility': 0,
            'Medium Civil Transport/Utility': 1,
            'Large Civil Transport/Utility': 2,
            'Military Transport/Utility/AWAC': 3,
            'Military Fighter/Interceptor/Attack': 4,
            'Military Bomber': 5,
            'Military Trainer': 6,
        },
        'difficulty':{
            '0': 0,
            '1': 1
        },
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

def get_task_dict(task):
    satellitepy_table = get_satellitepy_table()
    task_dict = get_satellitepy_dict_values(satellitepy_table,task)
    return task_dict


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

