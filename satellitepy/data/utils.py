import cv2
import numpy as np

from scipy.ndimage import generate_binary_structure, label, find_objects

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

def parse_potsdam_labels(label_path):
    """
    Parses the potsdam images to extract the label data
    Parameters
    ----------
    label_path : string
    Path to the 
    Returns
    -------
    labels : dict
        Satellitepy dict
    """
    img = cv2.imread(label_path)
    
    # bleaching every color except yellow
    hsv= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    yellow = np.uint8([[[0, 255, 255]]])
    hsvYellow = cv2.cvtColor(yellow, cv2.COLOR_BGR2HSV)

    lower=np.array([20,100,100])
    upper=np.array([40,255,255])

    mask=cv2.inRange(hsv,lower,upper)

    s = generate_binary_structure(2, 2)

    # figuring out the hbboxes for the yellow segements
    labeled_image, num_features = label(mask, structure=s)
    objs = find_objects(labeled_image)

    return objs