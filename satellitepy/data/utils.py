import cv2
import numpy as np
from satellitepy.data.cutout.geometry import BBox

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
            'airplane':0,
            'ship':1,
            'vehicle':2,
            'helicopter':3,
            'other':4},
        'fine-class':{},
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
    return satellitepy_table
