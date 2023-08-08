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
			'AOE' 						: 4,    # Shit Net
			'ARJ21'						: 5,    # Fair1m
			'Arleigh Burke DD'			: 6,    # Shit Net
			'Asagiri DD'				: 7,    # Shit Net
			'Atago DD'					: 8,    # Shit Net
			'Austin LL'					: 9,    # Shit Net
			'Barge'						: 10,   # Shit Net
			'Boeing737'					: 11,   # Fair1m
			'Boeing747'					: 12,   # Fair1m
			'Boeing777'					: 13,   # Fair1m
			'Boeing787'					: 14,   # Fair1m
			'Bus'						: 15,   # Fair1m, Xview
			'C919'						: 16,   # Fair1m
			'Cargo Car'					: 17,   # Xview
			'Cargo Plane'				: 18,   # Xview
			'Cargo Truck'				: 19,   # Fair1m, Xview
			'Cement Mixer'				: 20,   # Xview
			'Commander'					: 21,   # Shit Net
			'Container Ship'			: 22,   # Shit Net, Xview
			'Crane Truck'				: 23,   # Xview
			'Dry Cargo Ship'			: 24,   # Fair1m
			'Dump Truck'				: 25,   # Fair1m, Xview
			'Engineering Ship'			: 26,   # Fair1m
			'Engineering Vessel'		: 27,   # Xview
			'EPF'						: 28,   # Shit Net
            'Enterprise'                : 96,   # Shit Net
			'Ferry'						: 29,   # Shit Net, Xview
			'Fishing Boat'				: 30,   # Fair1m
			'Fishing Vessel'			: 30,   # Shit Net, Xview
			'Fixed-Wing Aircraft'		: 31,   # Xview
			'Flat Car'					: 32,   # Xview
			'Front Loader'				: 33,   # Xview
			'Ground Grader'				: 34,   # Xview
			'Hatsuyuki DD'				: 35,   # Shit Net
			'Haul Truck'				: 36,   # Xview
			'Hovercraft'				: 37,   # Shit Net
			'Hyuga DD'					: 38,   # Shit Net
			'large-vehicle'				: 39,   # Dota
			'LHA LL'					: 40,   # Shit Net
			'Liquid Cargo Ship'			: 41,   # Fair1m
			'Locomotive'				: 42,   # Xview
			'LSD 41 LL'					: 43,   # Shit Net
			'Maritime Vessel'			: 22,   # Xview
			'Masyuu AS'					: 44,   # Shit Net
			'Medical Ship'				: 45,   # Shit Net
            'Midway'                    : 95,   # Shit Net   
            'Mobile Crane'              : 98,   # Xview
			'Motorboat'					: 46,   # Shit Net, Fair1m, Xview
			'Nimitz'					: 47,   # Shit Net
			'Oil Tanker'				: 48,   # Xview
			'Osumi LL'					: 49,   # Shit Net
			'Other Aircraft Carrier'	: 50,   # Shit Net
			'Other Auxiliary Ship'		: 51,   # Shit Net
			'Other Destroyer'			: 52,   # Shit Net
			'Other Frigate'				: 53,   # Shit Net
			'Other Landing'				: 54,   # Shit Net
			'Other Merchant'			: 55,   # Shit Net
			'Other Ship'				: 56,   # Shit Net
			'Other Warship'				: 57,   # Shit Net
			'other-airplane'			: 58,   # Fair1m
			'other-ship'				: 56,   # Fair1m
			'other-vehicle'				: 59,   # Fair1m
			'Passenger Car'				: 60,   # Xview
			'Passenger Ship'			: 61,   # Fair1m
			'Passenger Vehicle'			: 62,   # Xview
			'Patrol'					: 63,   # Shit Net
			'Perry FF'					: 64,   # Shit Net
			'Pickup Truck'				: 65,	# Xview
			'Railway Vehicle'			: 66,   # Xview
			'Reach Stacker'				: 67,   # Xview
			'RoRo'						: 68,   # Shit Net
			'Sailboat'					: 69,   # Xview
			'Sanantonio AS'				: 70,   # Shit Net
			'Scraper/Tractor'			: 71,   # Xview
			'Small Aircraft'			: 72,   # Xview
			'Small Car'					: 73,   # Fair1m, Xview
			'small-vehicle'				: 73,   # Dota
			'Straddle Carrier'			: 75,   # Xview
			'Submarine'					: 76,   # Shit Net
			'Tank Car'					: 77,   # Xview
			'Test Ship'					: 78,   # Shit Net
			'Ticonderoga'				: 79,   # Shit Net
			'Tractor'					: 80,   # Fair1m
            'Trailer'                   : 97,   # Fair1m
			'Training Ship'				: 81,   # Shit Net
			'Truck'						: 19,   # Xview
			'Truck Tractor'				: 82,   # Fair1m, Xview
			'Truck w/Box'				: 83,   # Xview
			'Truck w/Flatbed'			: 84,   # Xview
			'Truck w/Liquid'			: 85,   # Xview
			'Tugboat'					: 86,   # Shit Net, Fair1m, Xview
			'Utility Truck'				: 87,   # Xview
			'Van'						: 88,   # Fair1m
			'Warship'					: 57,   # Fair1m
			'Wasp LL'					: 89,   # Shit Net
			'Yacht'						: 90,   # Shit Net, Xview
			'YuDao LL'					: 91,   # Shit Net
			'YuDeng LL'					: 92,   # Shit Net
			'YuTing LL'					: 93,   # Shit Net
			'YuZhao LL'					: 94    # Shit Net
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
    return satellitepy_table