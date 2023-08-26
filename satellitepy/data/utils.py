import cv2
import numpy as np
from satellitepy.data.bbox import BBox
from scipy.ndimage import generate_binary_structure, label, find_objects

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
                18: 'Small Car',
                19: 'Bus',
                20: 'Pickup Truck',
                21: 'Utility Truck',
                24: 'Cargo Truck',
                25: 'Truck w/Box',
                26: 'Truck Tractor',
                28: 'Truck w/Flatbed', 
                29: 'Truck w/Liquid',
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
                52: 'Oil Tanker',
                53: 'Engineering Vessel',
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
                94: 'Tower Structure',
                54: 'Tower Crane',
                55: 'Container Crane'
            }
        }
    return classes

def get_shipnet_classes():
	classes = {
		1 : ["Other Ship","Other Ship"],
		2 : ["Other Warship","Other Warship"],
		3 : ["Submarine","Submarine"],
		4 : ["Aircraft Carrier","Other Aircraft Carrier"],
		5 : ["Aircraft Carrier","Enterprise"],
		6 : ["Aircraft Carrier","Nimitz"],
		7 : ["Aircraft Carrier","Midway"],
		8 : ["Cruiser","Ticonderoga"],
		9 : ["Destroyer","Other Destroyer"],
		10: ["Destroyer","Atago DD"],
		11: ["Destroyer","Arleigh Burke DD"],
		12: ["Destroyer","Hatsuyuki DD"],
		13: ["Destroyer","Hyuga DD"],
		14: ["Destroyer","Asagiri DD"],
		15: ["Frigate","Other Frigate"],
		16: ["Frigate","Perry FF"],
		17: ["Patrol","Patrol"],
		18: ["Landing","Other Landing"],
		19: ["Landing","YuTing LL"],
		20: ["Landing","YuDeng LL"],
		21: ["Landing","YuDao LL"],
		22: ["Landing","Yuzhao LL"],
		23: ["Landing","Austin LL"],
		24: ["Landing","Asumi LL"],
		25: ["Landing","Wasp LL"],
		26: ["Landing","LSD 41 LL"],
		27: ["Landing","LHA LL"],
		28: ["Commander","Commander"],
		29: ["Auxiliary Ship","Other Auxiliary Ship"],
		30: ["Auxiliary Ship","Medical Ship"],
		31: ["Auxiliary Ship","Test Ship"],
		32: ["Auxiliary Ship","Training Ship"],
		33: ["Auxiliary Ship","AOE"],
		34: ["Auxiliary Ship","Masyuu AS"],
		35: ["Auxiliary Ship","Sanantonio AS"],
		36: ["Auxiliary Ship","EPF"],
		37: ["Other Merchant","Other Merchant"],
		38: ["Container Ship","Container SHip"],
		39: ["RoRo","RoRo"],
		40: ["Cargo","Cargo"],
		41: ["Barge","Barge"],
		42: ["Tugboat","Tugboat"],
		43: ["Ferry","Ferry"],
		44: ["Yacht","Yacht"],
		45: ["Sailboat","Sailboat"],
		46: ["Fishing Vessel","Fishing Vessel"],
		47: ["Oil Tanker","Oil Tanker"],
		48: ["Hovercraft","Hovercraft"],
		49: ["Motorboat","Motorboat"]
	}
	return classes


def parse_potsdam_labels(label_path):
    """
    Parses the potsdam images to extract the label data
    Parameters
    ----------
    label_path : string
        Path to the
    Returns
    -------
    objs : list of slices
        Horizontal bounding boxes of the objects
    """
    img = cv2.imread(label_path)
    
    # bleaching every color except yellow
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    lower=np.array([20,100,100])
    upper=np.array([40,255,255])

    mask=cv2.inRange(hsv,lower,upper)

    s = generate_binary_structure(2, 2)

    # figuring out the hbboxes for the yellow segements
    labeled_image, num_features = label(mask, structure=s)
    objs = find_objects(labeled_image)

    masks = []
    hbboxes = []

    for obj in objs:
        hbbox = [[obj[1].start, obj[0].start], [obj[1].stop, obj[0].start], [obj[1].stop, obj[0].stop], [obj[1].start, obj[0].stop]]
        h = BBox.get_bbox_limits(np.array(hbbox))

        coords = np.argwhere((mask[h[2]:h[3], h[0]:h[1]] != 0)).T.tolist()

        masks.append([(coords[1] + h[0]).tolist(), (coords[0] + h[2]).tolist()])
        hbboxes.append(hbbox)

    return (hbboxes, masks)
  
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
            'Airbus_A300'               : 99,   # Rareplanes_synthetic
            'Airbus_A-319'              : 98,   # Rareplanes_synthetic
            'Airbus_A320'               : 0,	# Rareplanes_synthetic
            'Airbus_A330'               : 2,	# Rareplanes_synthetic
            'Airbus_A'                  : 100,	# Rareplanes_synthetic
            'Airbus_A380'               : 101,	# Rareplanes_synthetic
			'Atago DD'					: 8,    # Ship Net
			'Austin LL'					: 9,    # Ship Net
            'ATR_ATR'                   : 102,	# Rareplanes_synthetic
            'BAE_146'                   : 103,	# Rareplanes_synthetic
			'Barge'						: 10,   # Ship Net
            'Boeing_707'                : 104,  # Rareplanes_synthetic
            'Boeing_717'                : 105,	# Rareplanes_synthetic
            'Boeing_727'                : 106,	# Rareplanes_synthetic
			'Boeing_737'				: 11,	# Rareplanes_synthetic
			'Boeing737'					: 11,   # Fair1m
			'Boeing_747'				: 12,	# Rareplanes_synthetic
			'Boeing747'					: 12,   # Fair1m
            'Boeing_757'                : 107,	# Rareplanes_synthetic
            'Boeing_767'                : 108,	# Rareplanes_synthetic
			'Boeing_777'				: 13,	# Rareplanes_synthetic
			'Boeing777'					: 13,   # Fair1m
			'Boeing787'					: 14,   # Fair1m
            'Boeing_BBJ'                : 109,	# Rareplanes_synthetic
            'Bombardier_BD'             : 110,	# Rareplanes_synthetic
            'Bombardier_Challenger'     : 111,	# Rareplanes_synthetic
            'Bombardier_CRJ'            : 112,	# Rareplanes_synthetic
            'Bombardier_Learjet'        : 113,	# Rareplanes_synthetic
			'Bus'						: 15,   # Fair1m, Xview
			'C919'						: 16,   # Fair1m
            'Cargo'                     : 135,  # Ship Net
			'Cargo Car'					: 17,   # Xview
			'Cargo Plane'				: 18,   # Xview
			'Cargo Truck'				: 19,   # Fair1m, Xview
			'Cement Mixer'				: 20,   # Xview
			'Commander'					: 21,   # Ship Net
			'Container Ship'			: 22,   # Ship Net, Xview
			'Crane Truck'				: 23,   # Xview
            'Cessna'                    : 114,	# Rareplanes_synthetic
            'Cessna_170'                : 115,	# Rareplanes_synthetic
            'Cessna_172'                : 116,	# Rareplanes_synthetic
            'Cessna_310'                : 117,	# Rareplanes_synthetic
            'Cessna_Citation'           : 118,	# Rareplanes_synthetic
            'Dassault_Falcon'           : 119,	# Rareplanes_synthetic
            'DeHavillandCanada_DHC'     : 120,	# Rareplanes_synthetic
			'Dry Cargo Ship'			: 24,   # Fair1m
			'Dump Truck'				: 25,   # Fair1m, Xview
            'Embraer_ERJ'               : 121,	# Rareplanes_synthetic
            'Embraer_Legacy'            : 122,	# Rareplanes_synthetic
			'Engineering Ship'			: 26,   # Fair1m
			'Engineering Vessel'		: 27,   # Xview
			'EPF'						: 28,   # Ship Net
            'Enterprise'                : 96,   # Ship Net
            'Excavator'                 : 136,  # Xview
			'Ferry'						: 29,   # Ship Net, Xview
			'Fishing Boat'				: 30,   # Fair1m
			'Fishing Vessel'			: 30,   # Ship Net, Xview
			'Fixed-Wing Aircraft'		: 31,   # Xview
			'Flat Car'					: 32,   # Xview
            'Fokker_100'                : 123,	# Rareplanes_synthetic
			'Front Loader'				: 33,   # Xview
			'Ground Grader'				: 34,   # Xview
            'Gulfstream_G200'           : 124,	# Rareplanes_synthetic
            'Gulfstream_GIII'           : 125,	# Rareplanes_synthetic
            'harbor'                    : 134,  # Dota
			'Hatsuyuki DD'				: 35,   # Ship Net
			'Haul Truck'				: 36,   # Xview
            'HBC_Hawker'                : 126,	# Rareplanes_synthetic
			'Hovercraft'				: 37,   # Ship Net
			'Hyuga DD'					: 38,   # Ship Net
			'large-vehicle'				: 39,   # Dota
            'Let_L'                     : 127,	# Rareplanes_synthetic
			'LHA LL'					: 40,   # Ship Net
			'Liquid Cargo Ship'			: 41,   # Fair1m
            'LockheedCorp_L'            : 128,	# Rareplanes_synthetic
			'Locomotive'				: 42,   # Xview
			'LSD 41 LL'					: 43,   # Ship Net
			'Maritime Vessel'			: 22,   # Xview
			'Masyuu AS'					: 44,   # Ship Net
            'McDonnellDouglas_DC'       : 129,	# Rareplanes_synthetic
            'McDonnellDouglas_MD'       : 130,	# Rareplanes_synthetic
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
            'PiperAircraft_PA'          : 131,	# Rareplanes_synthetic
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
            'SudAviation_Caravelle'     : 132,	# Rareplanes_synthetic
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
            'Tupolev_154'               : 133,	# Rareplanes_synthetic
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
			'Airbus_A330-300'					: 0,	# Rareplanes_synthetic
			'Airbus_A-340'						: 1,	# Rareplanes_synthetic
			'Airbus_A380-800'					: 2,	# Rareplanes_synthetic
			'ATR_ATR-72'						: 3,	# Rareplanes_synthetic
			'BAE_146-100'						: 4,	# Rareplanes_synthetic
			'BAE_146-300'						: 5,	# Rareplanes_synthetic
			'Boeing_717-200'					: 6,	# Rareplanes_synthetic
			'Boeing_727-100'					: 7,	# Rareplanes_synthetic
			'Boeing_737-200'					: 8,	# Rareplanes_synthetic
			'Boeing_737-300'					: 9,	# Rareplanes_synthetic
			'Boeing_747-200'					: 10,	# Rareplanes_synthetic
			'Boeing_747-400'					: 11,	# Rareplanes_synthetic
			'Boeing_757-300'					: 12,	# Rareplanes_synthetic
			'Boeing_767-200'					: 13,	# Rareplanes_synthetic
			'Boeing_767-400'					: 14,	# Rareplanes_synthetic
			'Boeing_777-300'					: 15,	# Rareplanes_synthetic
			'Boeing_BBJ-2'						: 16,	# Rareplanes_synthetic
			'Bombardier_BD-700-GlobalExpress'	: 17,	# Rareplanes_synthetic
			'Bombardier_Challenger-300'			: 18,	# Rareplanes_synthetic
			'Bombardier_Challenger-604'			: 19,	# Rareplanes_synthetic
			'Bombardier_CRJ-705'				: 20,	# Rareplanes_synthetic
			'Cessna_Citation-CJ4'				: 21,	# Rareplanes_synthetic
			'Dassault_Falcon-100'				: 22,	# Rareplanes_synthetic
			'Dassault_Falcon-2000'				: 23,	# Rareplanes_synthetic
			'Dassault_Falcon-900'				: 24,	# Rareplanes_synthetic
			'DeHavillandCanada_DHC-2-Beaver'	: 25,	# Rareplanes_synthetic
			'DeHavillandCanada_DHC-3-Otter'		: 26,	# Rareplanes_synthetic
			'Embraer_ERJ-135'					: 27,	# Rareplanes_synthetic
			'HBC_Hawker-4000'					: 28,	# Rareplanes_synthetic
			'LockheedCorp_L-1011-TriStar'		: 29,	# Rareplanes_synthetic
			'Let_L-200Morova'					: 30,	# Rareplanes_synthetic
			'McDonnellDouglas_DC-9-30 '			: 31,	# Rareplanes_synthetic
			'McDonnellDouglas_MD-11'			: 32,	# Rareplanes_synthetic
			'PiperAircraft_PA-28'				: 33	# Rareplanes_synthetic
        },
        'role':{
            'Small Civil Transport/Utility': 0,
            'Medium Civil Transport/Utility': 1,
            'Large Civil Transport/Utility': 2,
            'Military Transport/Utility/AWAC': 3,
            'Military Fighter/Interceptor/Attack': 4,
            'Military Bomber': 5,
            'Military Trainer': 6,
            'Small Vehicle' : 7,
            'Large Vehicle': 8,
            'Warship': 9,
            'Merchant Ship': 10,
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
    satellitepy_table.setdefault("merged-class", {})
    for key,value in satellitepy_table['coarse-class'].items():
        satellitepy_table['merged-class'][key] = value
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

def set_satellitepy_dict_values(satellitepy_dict,task, value):
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
        satellitepy_dict[keys[0]] = value
    elif len(keys)==2:
        satellitepy_dict[keys[0]][keys[1]] = value
    elif len(keys)==3:
        satellitepy_dict[keys[0]][keys[1]][keys[2]] = value

    return satellitepy_dict

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

