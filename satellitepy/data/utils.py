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

def get_shipnet_classes():
	classes = {
		1 : ["Other Ship","Other Ship"],
		2 : ["Other Warship","Other Warship"],
		3 : ["Submarine","Submarine"],
		4 : ["Aircraft Carrier","Other Aircraft Carrier"],
		5 : ["Aircraft Carrier","Enterprise"],
		6 : ["Aircraft Carrier","Nimitz"],
		7 : ["Aircraft Carrier","Midway"],
		8 : ["cruiser","Ticonderoga"],
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
