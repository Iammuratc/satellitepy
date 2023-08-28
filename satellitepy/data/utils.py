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

def get_shipnet_categories():
    # Categories rely on the original shipnet categories. Check out the paper
    categories = {
        #'Other Ship'
        # 1: {
        #     1: {
        1:'Other Ship',
            # },
            # # Warship
            # 2: {
        2:'Other Warship',
        3:'Submarine',
        4:'Aircraft Carrier',
        5:'Cruiser',
        6:'Destroyer',
        7:'Frigate',
        8:'Patrol',
        9:'Landing',
        10:'Commander',
        11:'Auxiliary Ship',
            # },
            # # Merchant
            # 3: {
        12:'Other Merchant',
        13:'Container Ship',
        14:'RoRo',
        15:'Cargo',
        16:'Barge',
        17:'Tugboat',
        18:'Ferry',
        19:'Yacht',
        20:'Sailboat',
        21:'Fishing Vessel',
        22:'Oil Tanker',
        23:'Hovercraft',
        24:'Motorboat',
        #     }
        # },
        # Dock
        # 2: {
        #     4: {
        25:'Dock'
        #     }
        # }
    }
    return categories

def get_vedai_classes():
    classes={
        1:"car",
        2:"truck",
        3:"tractor", 
        4:"camping car", 
        5:"motorcycle", 
        6:"bus", 
        9:"van", 
        10:"other", 
        11:"pickup", 
        12:"large",
        23:"boat", 
        31:"plane"
    }
    return classes

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
                11: 'Fixed-Wing Aircraft', # role small aircraft
                12: 'Small Aircraft', # role medium aircraft
                13: 'Cargo Plane' # role large aircraft
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
			'ARJ21'						: 5,    # Fair1m
            'Motorboat'                 : 6,    # ShipNet
            'Hovercraft'                : 7,    # ShipNet
            'Patrol'                    : 8,    # ShipNet
            'Destroyer'                 : 9,    # ShipNet
            'Commander'                 : 10,   # ShipNet
            'Ferry'                     : 134,  # ShipNet, Xview
            'Submarine'                 : 135,  # ShipNet
            'Landing'                   : 21,   # ShipNet
            'Cruiser'                   : 28,   # ShipNet
            'Frigate'                   : 29,   # ShipNet
            'Auxiliary Ship'            : 31,   # ShipNet
            'Barge'                     : 35,   # ShipNet
            'Aircraft Carrier'          : 37,   # ShipNet
            'Yacht'                     : 38,   # ShipNet, Xview
            'Tugboat'                   : 39,   # ShipNet, Fair1m, Xview
            'Cargo'                     : 40,   # ShipNet
            'Container Ship'            : 43,   # ShipNet, Xview
            'Oil Tanker'                : 44,   # ShipNet
            'RoRo'                      : 45,   # ShipNet
            'Sailboat'                  : 46,   # ShipNet
            'Fishing Vessel'            : 30,   # ShipNet, Xview
            'Airbus_A300'               : 99,   # Rareplanes_synthetic
            'Airbus_A-319'              : 98,   # Rareplanes_synthetic
            'Airbus_A320'               : 4,	# Rareplanes_synthetic
            'Airbus_A330'               : 2,	# Rareplanes_synthetic
            'Airbus_A'                  : 100,	# Rareplanes_synthetic
            'Airbus_A380'               : 101,	# Rareplanes_synthetic
            'ATR_ATR'                   : 102,	# Rareplanes_synthetic
            'BAE_146'                   : 103,	# Rareplanes_synthetic
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
			'Cargo Car'					: 17,   # Xview
			'Cargo Plane'				: 18,   # Xview
			'Cargo Truck'				: 46,   # Fair1m, Xview
			'Cement Mixer'				: 20,   # Xview
			'Fishing Boat'				: 30,   # Fair1m
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
			'Engineering Vessel'		: 26,   # Xview
            'Excavator'                 : 136,  # Xview
			'Flat Car'					: 32,   # Xview
            'Fokker_100'                : 123,	# Rareplanes_synthetic
			'Front Loader'				: 33,   # Xview
			'Ground Grader'				: 34,   # Xview
            'Gulfstream_G200'           : 124,	# Rareplanes_synthetic
            'Gulfstream_GIII'           : 125,	# Rareplanes_synthetic
			'Haul Truck'				: 36,   # Xview
            'HBC_Hawker'                : 126,	# Rareplanes_synthetic
            'Let_L'                     : 127,	# Rareplanes_synthetic
			'Liquid Cargo Ship'			: 41,   # Fair1m
            'LockheedCorp_L'            : 128,	# Rareplanes_synthetic
			'Locomotive'				: 42,   # Xview
			'Maritime Vessel'			: 22,   # Xview
            'McDonnellDouglas_DC'       : 129,	# Rareplanes_synthetic
            'McDonnellDouglas_MD'       : 130,	# Rareplanes_synthetic
            'Mobile Crane'              : 44,   # Xview
			'Oil Tanker'				: 48,   # Xview
			'Passenger Car'				: 60,   # Xview
			'Passenger Ship'			: 61,   # Fair1m
			'Passenger Vehicle'			: 62,   # Xview
			'Pickup Truck'				: 65,	# Xview
            'PiperAircraft_PA'          : 131,	# Rareplanes_synthetic
			'Railway Vehicle'			: 66,   # Xview
			'Reach Stacker'				: 67,   # Xview
			'Sailboat'					: 69,   # Xview
			'Scraper/Tractor'			: 71,   # Xview
			'Small Car'					: 60,   # Fair1m, Xview
			'Straddle Carrier'			: 75,   # Xview
            'SudAviation_Caravelle'     : 132,	# Rareplanes_synthetic
			'Tank Car'					: 77,   # Xview
			'Tractor'					: 80,   # Fair1m
            'Trailer'                   : 97,   # Fair1m
			'Truck'						: 19,   # Xview
			'Truck Tractor'				: 82,   # Fair1m, Xview
			'Truck w/Box'				: 83,   # Xview
			'Truck w/Flatbed'			: 84,   # Xview
			'Truck w/Liquid'			: 85,   # Xview
            'Tupolev_154'               : 133,	# Rareplanes_synthetic
			'Utility Truck'				: 87,   # Xview
			'Van'						: 88,   # Fair1m
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
			'PiperAircraft_PA-28'				: 33,	# Rareplanes_synthetic
            'Other Destroyer'                   : 34,   # ShipNet
            'Atago DD'                          : 35,   # ShipNet
            'Arleigh Burke DD'                  : 36,   # ShipNet
            'LSD 41 LL'                         : 37,   # ShipNet
            'Ticonderoga'                       : 38,   # ShipNet
            'Hatsuyuki DD'                      : 39,   # ShipNet
            'Asagiri DD'                        : 40,   # ShipNet
            'Hyuga DD'                          : 41,   # ShipNet
            'Perry FF'                          : 42,   # ShipNet
            'AOE'                               : 43,   # ShipNet
            'Austin LL'                         : 44,   # ShipNet
            'Other Frigate'                     : 45,   # ShipNet
            'Enterprise'                        : 46,   # ShipNet
            'Other Landing'                     : 47,   # ShipNet
            'Other Auxiliary Ship'              : 48,   # ShipNet
            'Osumi LL'                          : 49,   # ShipNet
            'Nimitz'                            : 50,   # ShipNet
            'Sanantonio AS'                     : 51,   # ShipNet
            'Other Aircraft Carrier'            : 52,   # ShipNet
            'EPF'                               : 53,   # ShipNet
            'Masyuu AS'                         : 54,   # ShipNet
            'Wasp LL'                           : 55,   # ShipNet
            'LHA LL'                            : 56,   # ShipNet
            'Test Ship'                         : 57,   # ShipNet
            'Training Ship'                     : 58,   # ShipNet
            'YuTing LL'                         : 59,   # ShipNet
            'Medical Ship'                      : 60,   # ShipNet
            'YuDeng LL'                         : 61,   # ShipNet
            'YuDao LL'                          : 62,   # ShipNet
            'YuZhao LL'                         : 63,   # ShipNet
            'Midway'                            : 64,   # ShipNet

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
    # len_coarse_class = len(satellitepy_table['coarse-class'])
    # len_fine_class = len(satellitepy_table['fine-class'])
    # satellitepy_table['merged-class'] = satellitepy_table['coarse-class'].copy()
    # for key,value in satellitepy_table['fine-class'].items():
    #     satellitepy_table['merged-class'][key] = value+len_coarse_class
    # for key,value in satellitepy_table['very-fine-class'].items():
    #     satellitepy_table['merged-class'][key] = value+len_coarse_class+len_fine_class
    return satellitepy_table

def analyze_satellitepy_table():
    satellitepy_table = get_satellitepy_table()
    unique_indices = set(list(satellitepy_table['fine-class'].values()))
    max_indices = range(max(satellitepy_table['fine-class'].values())+1)
    empty_indices = set(max_indices) - unique_indices
    print('Empty indices are:', empty_indices)

    group_by_index = {ind:[] for ind in max_indices}
    for key, value in sorted(satellitepy_table['fine-class'].items()):
        group_by_index[value].append(key)

    print('Indices:Values in satellitepy table')
    for key, value in group_by_index.items():
        print(key, value)

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

# def get_shipnet_classes():
# 	classes = {
# 		1 : ["Other Ship","Other Ship"],
# 		2 : ["Other Warship","Other Warship"],
# 		3 : ["Submarine","Submarine"],
# 		4 : ["Aircraft Carrier","Other Aircraft Carrier"],
# 		5 : ["Aircraft Carrier","Enterprise"],
# 		6 : ["Aircraft Carrier","Nimitz"],
# 		7 : ["Aircraft Carrier","Midway"],
# 		8 : ["Cruiser","Ticonderoga"],
# 		9 : ["Destroyer","Other Destroyer"],
# 		10: ["Destroyer","Atago DD"],
# 		11: ["Destroyer","Arleigh Burke DD"],
# 		12: ["Destroyer","Hatsuyuki DD"],
# 		13: ["Destroyer","Hyuga DD"],
# 		14: ["Destroyer","Asagiri DD"],
# 		15: ["Frigate","Other Frigate"],
# 		16: ["Frigate","Perry FF"],
# 		17: ["Patrol","Patrol"],
# 		18: ["Landing","Other Landing"],
# 		19: ["Landing","YuTing LL"],
# 		20: ["Landing","YuDeng LL"],
# 		21: ["Landing","YuDao LL"],
# 		22: ["Landing","Yuzhao LL"],
# 		23: ["Landing","Austin LL"],
# 		24: ["Landing","Asumi LL"],
# 		25: ["Landing","Wasp LL"],
# 		26: ["Landing","LSD 41 LL"],
# 		27: ["Landing","LHA LL"],
# 		28: ["Commander","Commander"],
# 		29: ["Auxiliary Ship","Other Auxiliary Ship"],
# 		30: ["Auxiliary Ship","Medical Ship"],
# 		31: ["Auxiliary Ship","Test Ship"],
# 		32: ["Auxiliary Ship","Training Ship"],
# 		33: ["Auxiliary Ship","AOE"],
# 		34: ["Auxiliary Ship","Masyuu AS"],
# 		35: ["Auxiliary Ship","Sanantonio AS"],
# 		36: ["Auxiliary Ship","EPF"],
# 		37: ["Other Merchant","Other Merchant"],
# 		38: ["Container Ship","Container SHip"],
# 		39: ["RoRo","RoRo"],
# 		40: ["Cargo","Cargo"],
# 		41: ["Barge","Barge"],
# 		42: ["Tugboat","Tugboat"],
# 		43: ["Ferry","Ferry"],
# 		44: ["Yacht","Yacht"],
# 		45: ["Sailboat","Sailboat"],
# 		46: ["Fishing Vessel","Fishing Vessel"],
# 		47: ["Oil Tanker","Oil Tanker"],
# 		48: ["Hovercraft","Hovercraft"],
# 		49: ["Motorboat","Motorboat"]
# 	}
# 	return classes
