import cv2
import numpy as np

def read_mask_image(label_dict,mask_path):
    """
    Set the masks key in the satellitepy dict
    Parameters
    ----------
    label_dict : dict
        Satellitepy dict
    mask_path : Path
        Mask image path
    Returns
    -------
    label_dict : dict
        Satellitepy dict
    """
    mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_RGB2GRAY)

    obbox_exist = True if label_dict['obboxes']

    tmp_mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    pts = np.array([bbox_corners], dtype = np.int32)
    cv2.fillPoly(tmp_mask, pts, 255)
    coord = np.argwhere((tmp_mask == 255) & (img != 0)).tolist()
    labels['mask-indices'].append(coord)



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