import cv2
import numpy as np
import rasterio
from scipy.ndimage import generate_binary_structure, label, find_objects
import logging

from satellitepy.data.bbox import BBox

logger = logging.getLogger('')

def rescale_labels(labels,rescaling):
    '''
    Rescale the tasks obboxes, hbooxes and masks, if defined.
    Parameters
    ----------
    labels : dict
        Label dictionary in the satellitepy format.
    Returns
    -------
    labels_rescaled : dict
        Label dictionary in the satellitepy format. Relevant tasks are rescaled.
    '''
    labels_rescaled = labels.copy()
    for task in ['obboxes', 'hbboxes', 'masks', 'attributes_fuselage_length', 'attributes_wings_wing-span']:
        rescaled_values = []
        for i, value in enumerate(get_satellitepy_dict_values(labels,task)):
            if value is None:
                rescaled_values.append(value)
            else:
                rescaled_value = np.array(value) * rescaling
                rescaled_values.append(rescaled_value.tolist())
        set_satellitepy_dict_values(satellitepy_dict=labels_rescaled,task=task,value=rescaled_values)
    return labels_rescaled

def read_img(img_path, module='cv2', rescaling=1.0, interpolation_method=cv2.INTER_LINEAR):
    """
    Read image.
    Parameters
    ----------
    img_path : str
        Image path
    module : str
        Use this module to read the image. rasterio is suggested for large TIF images
    Returns
    -------
    img : np.ndarray
        Image in the opencv format,  (rows, columns, bands). If image has four bands, the alpha channel is ignored.
    """

    if module == 'cv2':
        img = cv2.imread(img_path)
    elif module == 'rasterio':
        with rasterio.open(img_path) as src:
            img_array = src.read()
            img = np.transpose(img_array, axes=(1, 2, 0))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            lo, hi = np.percentile(img, (1, 99))
            img = (img.astype(float) - lo) * (255 / (hi - lo))
            img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        logger.error(f'{module} is not a valid argument!')
        return 0
    
    if rescaling != 1.0:
        interpolation_dict = {
            'INTER_NEAREST':cv2.INTER_NEAREST,
            'INTER_LINEAR':cv2.INTER_LINEAR,
            'INTER_CUBIC':cv2.INTER_CUBIC,
            'INTER_AREA':cv2.INTER_AREA
            }
        img = cv2.resize(img, (0, 0), fx = rescaling, fy = rescaling, interpolation = interpolation_dict[interpolation_method])
    return img


def set_mask(labels, mask_path, bbox_type, mask_type):
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
    if mask_type == 'DOTA':
        mask = cv2.cvtColor(cv2.imread(str(mask_path)), cv2.COLOR_RGB2GRAY)
    elif mask_type == 'rareplanes':
        mask = cv2.imread(str(mask_path))
    else:
        raise Exception(f'mask_type {mask_type} not defined')

    for bbox in labels[bbox_type]:
        h = BBox.get_bbox_limits(np.array(bbox))
        mask_0 = np.zeros((mask.shape[0], mask.shape[1]))
        cv2.fillPoly(mask_0, [np.array(bbox, dtype=int)], 1)

        if mask_type == 'DOTA':
            coords = np.argwhere((mask_0[h[2]:h[3], h[0]:h[1]] == 1) & (mask[h[2]:h[3], h[0]:h[1]] != 0)).T
            labels['masks'].append(np.array([coords[1] + h[0], coords[0] + h[2]]).tolist())  # x,y
        if mask_type == 'rareplanes':
            center = (int((h[0] + h[1]) / 2), int((h[2] + h[3]) / 2))

            h = np.clip(h, a_min=0, a_max=1919)


            if 0 <= center[0] < mask.shape[1] and 0 <= center[1] < mask.shape[0]:
                color = mask[center[::-1]]

                coords = np.array(np.argwhere(np.all((mask[h[2]:h[3], h[0]:h[1]] == color), axis=-1)).T.tolist())

                labels['masks'].append([(coords[1] + h[0]).tolist(), (coords[0] + h[2]).tolist()])
            else:
                labels['masks'].append(None)

    return labels


def get_shipnet_categories():
    categories = {
        1: 'Other Ship',
        2: 'Other Warship',
        3: 'Submarine',
        4: 'Aircraft Carrier',
        5: 'Cruiser',
        6: 'Destroyer',
        7: 'Frigate',
        8: 'Patrol',
        9: 'Landing',
        10: 'Commander',
        11: 'Auxiliary Ship',
        12: 'Other Merchant',
        13: 'Container Ship',
        14: 'RoRo',
        15: 'Cargo',
        16: 'Barge',
        17: 'Tugboat',
        18: 'Ferry',
        19: 'Yacht',
        20: 'Sailboat',
        21: 'Fishing Vessel',
        22: 'Oil Tanker',
        23: 'Hovercraft',
        24: 'Motorboat',
        25: 'Dock'
    }
    return categories


def get_vedai_classes():
    classes = {
        1: 'Car',
        2: 'Truck',
        3: 'Tractor',
        4: 'Camping Car',
        5: 'Motorcycle',
        6: 'Bus',
        9: 'Van',
        10: 'other',
        11: 'Pickup Truck',
        12: 'large',
        23: 'boat',
        31: 'plane'
    }
    return classes


def get_xview_classes():
    classes = {
        'vehicles': {
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
        'ships': {
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
        'airplanes': {
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

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([20, 100, 100])
    upper = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    s = generate_binary_structure(2, 2)

    labeled_image, num_features = label(mask, structure=s)
    objs = find_objects(labeled_image)

    masks = []
    hbboxes = []

    for obj in objs:
        hbbox = [[obj[1].start, obj[0].start], [obj[1].stop, obj[0].start], [obj[1].stop, obj[0].stop],
                 [obj[1].start, obj[0].stop]]
        h = BBox.get_bbox_limits(np.array(hbbox))

        coords = np.argwhere((mask[h[2]:h[3], h[0]:h[1]] != 0)).T.tolist()

        masks.append([(coords[1] + h[0]).tolist(), (coords[0] + h[2]).tolist()])
        hbboxes.append(hbbox)

    return (hbboxes, masks)


def get_satellitepy_table():
    """"
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
            'helicopter':3},
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
        'fine-class':
        {
            'A220'       	                :	0	,	   # Fair1m FR24
            'A321'       	                :	1	,	   # Fair1m FR24
            'A330'       	                :	2	,	   # Fair1m FR24
            'Airbus_A330'               	:	2	,	   # Rareplanes_synthetic
            'A350'                      	:	3	,	   # Fair1m FR24
            'Airbus_A320'               	:	4	,	   # Rareplanes_synthetic
            'ARJ21'      	                :	5	,	   # Fair1m
            'Motorboat'                 	:	6	,	   # ShipNet
            'Hovercraft'                	:	7	,	   # ShipNet
            'Patrol'                    	:	8	,	   # ShipNet
            'Destroyer'                 	:	9	,	   # ShipNet
            'Commander'                 	:	10	,	   # ShipNet		
            'Boeing_737'    	            :	11	,	   # Rareplanes_synthetic
            'Boeing737'     	            :	11	,	   # Fair1m
            'Boeing_747'    	            :	12	,	   # Rareplanes_synthetic
            'Boeing747'     	            :	12	,	   # Fair1m
            'Boeing_777'    	            :	13	,	   # Rareplanes_synthetic
            'Boeing777'     	            :	13	,	   # Fair1m
            'Boeing787'     	            :	14	,	   # Fair1m
            'Bus'                       	:	15	,	   # Fair1m	 Xview	Vedai
            'C919'      	                :	16	,	   # Fair1m
            'Cargo Car'     	            :	17	,	   # Xview
            'Cargo Plane'    	            :	18	,	   # Xview
            'Truck'      	                :	19	,	   # Xview, Vedai
            'Cement Mixer'    	            :	20	,	   # Xview
            'Landing'                   	:	21	,	   # ShipNet		
            'Maritime Vessel'   	        :	22	,	   # Xview
            'Crane Truck'    	            :	23	,	   # Xview
            'Dry Cargo Ship'            	:	24	,	   # Fair1m
            'Dump Truck'    	            :	25	,	   # Fair1m	 Xview
            'Engineering Ship'          	:	26	,	   # Fair1m
            'Engineering Vessel'        	:	26	,	   # Xview
            'Cruiser'                   	:	27	,	   # ShipNet		
            'Frigate'                   	:	28	,	   # ShipNet		
            'Fishing Vessel'            	:	29	,	   # ShipNet	 Xview	
            'Fishing Boat'              	:	29	,	   # Fair1m
            'Auxiliary Ship'            	:	30	,	   # ShipNet		
            'Flat Car'                  	:	31	,	   # Xview
            'Front Loader'    	            :	32	,	   # Xview
            'Ground Grader'             	:	33	,	   # Xview
            'Barge'                     	:	34	,	   # ShipNet		
            'Haul Truck'    	            :	35	,	   # Xview
            'Aircraft Carrier'          	:	36	,	   # ShipNet		
            'Yacht'                     	:	37	,	   # ShipNet	 Xview	
            'Tugboat'                   	:	38	,	   # ShipNet	 Fair1m	 Xview
            'Cargo'                     	:	39	,	   # ShipNet		
            'Liquid Cargo Ship'   	        :	40	,	   # Fair1m
            'Locomotive'    	            :	41	,	   # Xview
            'Container Ship'            	:	42	,	   # ShipNet	 Xview	
            'Oil Tanker'                	:	43	,	   # ShipNet	 Xview	
            'Mobile Crane'              	:	44	,	   # Xview		
            'RoRo'                      	:	45	,	   # ShipNet		
            'Sailboat'                  	:	46	,	   # ShipNet	Xview	
            'Cargo Truck'    	            :	47	,	   # Fair1m	 Xview
            'Passenger Car'             	:	48	,	   # Xview
            'Small Car'     	            :	49	,	   # Fair1m	 Xview
            'Passenger Ship'   	            :	50	,	   # Fair1m
            'Passenger Vehicle'   	        :	51	,	   # Xview
            'Pickup Truck'    	            :	52	,	   # Xview    Vedai
            'Railway Vehicle'   	        :	53	,	   # Xview
            'Reach Stacker'    	            :	54	,	   # Xview
            'Scraper/Tractor'   	        :	55	,	   # Xview
            'Straddle Carrier'   	        :	56	,	   # Xview
            'Tank Car'     	                :	57	,	   # Xview
            'Tractor'     	                :	58	,	   # Fair1m, Vedai
            'Truck Tractor'             	:	59	,	   # Fair1m	 Xview
            'Truck w/Box'    	            :	60	,	   # Xview
            'Truck w/Flatbed'   	        :	61	,	   # Xview
            'Truck w/Liquid'   	            :	62	,	   # Xview
            'Utility Truck'             	:	63	,	   # Xview
            'Van'      	                    :	64	,	   # Fair1m	Vedai
            'Trailer'                   	:	65	,	   # Fair1m
            'Airbus_A-319'              	:	66	,	   # Rareplanes_synthetic		
            'Airbus_A300'               	:	67	,	   # Rareplanes_synthetic		
            'Airbus_A'                  	:	68	,	   # Rareplanes_synthetic
            'Airbus_A380'               	:	69	,	   # Rareplanes_synthetic
            'ATR_ATR'                   	:	70	,	   # Rareplanes_synthetic
            'BAE_146'                   	:	71	,	   # Rareplanes_synthetic
            'Boeing_707'                	:	72	,	   # Rareplanes_synthetic
            'Boeing_717'                	:	73	,	   # Rareplanes_synthetic
            'Boeing_727'                	:	74	,	   # Rareplanes_synthetic
            'Boeing_757'                	:	75	,	   # Rareplanes_synthetic
            'Boeing_767'                	:	76	,	   # Rareplanes_synthetic
            'Boeing_BBJ'                	:	77	,	   # Rareplanes_synthetic
            'Bombardier_BD'             	:	78	,	   # Rareplanes_synthetic
            'Bombardier_Challenger'     	:	79	,	   # Rareplanes_synthetic
            'Bombardier_CRJ'            	:	80	,	   # Rareplanes_synthetic
            'Bombardier_Learjet'        	:	81	,	   # Rareplanes_synthetic
            'Cessna'                    	:	82	,	   # Rareplanes_synthetic FR24
            'Cessna_170'                	:	83	,	   # Rareplanes_synthetic
            'Cessna_172'                	:	84	,	   # Rareplanes_synthetic
            'Cessna_310'                	:	85	,	   # Rareplanes_synthetic
            'Cessna_Citation'           	:	86	,	   # Rareplanes_synthetic
            'Dassault_Falcon'           	:	87	,	   # Rareplanes_synthetic
            'DeHavillandCanada_DHC'     	:	88	,	   # Rareplanes_synthetic
            'Embraer_ERJ'               	:	89	,	   # Rareplanes_synthetic
            'Embraer_Legacy'            	:	90	,	   # Rareplanes_synthetic
            'Fokker_100'                	:	91	,	   # Rareplanes_synthetic
            'Gulfstream_G200'           	:	92	,	   # Rareplanes_synthetic
            'Gulfstream_GIII'           	:	93	,	   # Rareplanes_synthetic
            'HBC_Hawker'                	:	94	,	   # Rareplanes_synthetic
            'Let_L'                     	:	95	,	   # Rareplanes_synthetic
            'LockheedCorp_L'            	:	96	,	   # Rareplanes_synthetic
            'McDonnellDouglas_DC'       	:	97	,	   # Rareplanes_synthetic
            'McDonnellDouglas_MD'       	:	98	,	   # Rareplanes_synthetic
            'PiperAircraft_PA'          	:	99	,	   # Rareplanes_synthetic
            'SudAviation_Caravelle'     	:	100	,	   # Rareplanes_synthetic
            'Tupolev_154'               	:	101	,	   # Rareplanes_synthetic
            'Ferry'                     	:	102	,	  # ShipNet	 Xview
            'Submarine'                 	:	103	,	  # ShipNet		
            'Excavator'                 	:	104	,	  # Xview		
            'Car'                           :   105,      # Vedai
            'Camping Car'                   :   106,      # Vedai
            'Motorcycle'                    :   107,      # Vedai
            'B787'                          :   14,      # FR24
            'A320'                          :   4,      # FR24
            'E190'                          :   110,      # FR24
            'E195'                          :   111,      # FR24
            'B777'                          :   13,      # FR24
            'B737'                          :   11,      # FR24
            'E175'                          :   114,      # FR24
            'B747'                          :   12,      # FR24
            'A319'                          :   66,      # FR24
            'B767'                          :   76,      # FR24
            'A380'                          :   69,      # FR24
            'B757'                          :   75,      # FR24
            'E170'                          :   120,      # FR24
            'CRJ7'                          :   121,      # FR24
            'E145'                          :   122,      # FR24
            'CRJ2'                          :   123,      # FR24
            'MD11'                          :   124,      # FR24
            'B717'                          :   73,      # FR24
            'A340'                          :   126,      # FR24
            'CRJ9'                          :   127,      # FR24
            'CRJ-900'                       :   128,      # FR24
            'CRJ-701'                       :   129,      # FR24
            'CRJ-700'                       :   130,      # FR24
            'R175'                          :   131,      # FR24
            'Beechcraft'                    :   132,      # FR24
            'ERJ'                           :   133,      # FR24
            'CRJ'                           :   134,      # FR24
            'Falcon'                        :   135,      # FR24
            'Sukhoi-100'                    :   136,      # FR24
            'Embraer'                       :   137,      # FR24
            'Bombardier-Global'             :   138,      # FR24
            'E135'                          :   139,      # FR24
            'Cessna560'                     :   140,      # FR24
            'H25B'                          :   141,      # FR24
            'Cessna-Citation'               :    86,      # FR24
            'Gulfstream'                    :   142,      # FR24  
            'Gulfstream-Global'             :   143,      # FR24
            'Embraer-Praetor'               :   144,      # FR24
            'Embraer-Phenom'                :   145,      # FR24
            'DeHavilland-Dash-8'            :   146,      # FR24
        },
        'very-fine-class':{
			'Airbus_A330-300'					: 0,	# Rareplanes_synthetic
			'Airbus_A-340'						: 1,	# Rareplanes_synthetic
			'Airbus_A380-800'					: 2,	# Rareplanes_synthetic
            'Airbus_A-319'                      : 65,   # Rareplanes_synthetic
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
			'McDonnellDouglas_DC-9-30'			: 31,	# Rareplanes_synthetic
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
        'difficulty': {
            '0': 0,
            '1': 1
        },
        'attributes':{
            'engines':{
                'no-engines': {
                    0:0,
                    1:1,
                    2:2,
                    3:3,
                    4:4
                    },
                'propulsion': {
                    'propeller': 0,
                    'jet':       1,
                    'unpowered': 2
                    }
            },
            'fuselage': {
                'canards': {
                    False: 0,
                    True:  1
                },
                'length': {
                    'max': 82.5,
                    'min': 4.0}
            },
            'wings': {
                'wing-span': {
                    'max': 80.0,
                    'min': 4.0
                    },
                'wing-shape': {
                    'swept':            0,
                    'straight':         1,
                    'variable swept':   2,
                    'delta':            3
                    },
                'wing-position': {
                    'mid/low mounted': 0,
                    'high mounted':    1
                    }
            },
            'tail': {
                'no-tail-fins': {
                    1: 0,
                    2: 1
                    }
            }
        }
    }
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
    """
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
    """

    keys = task.split('_')
    if len(keys) == 1:
        return satellitepy_dict[keys[0]]
    elif len(keys) == 2:
        return satellitepy_dict[keys[0]][keys[1]]
    elif len(keys) == 3:
        return satellitepy_dict[keys[0]][keys[1]][keys[2]]
    else:
        return 0


def set_satellitepy_dict_values(satellitepy_dict,task, value):
    """
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
    """

    keys = task.split('_')
    if len(keys) == 1:
        satellitepy_dict[keys[0]] = value
    elif len(keys) == 2:
        satellitepy_dict[keys[0]][keys[1]] = value
    elif len(keys) == 3:
        satellitepy_dict[keys[0]][keys[1]][keys[2]] = value

    return satellitepy_dict


def get_task_dict(task):
    satellitepy_table = get_satellitepy_table()
    task_dict = get_satellitepy_dict_values(satellitepy_table, task)
    return task_dict

def count_unique_values(satellitepy_values, instances={}):
    for value in satellitepy_values:
        if isinstance(value, str) or isinstance(value, int):
            if value not in instances.keys():
                instances[value] = 0
            instances[value] += 1
        elif isinstance(value, list):
            if 'count' not in instances.keys():
                instances['count'] = 0
            instances['count'] += 1
        elif isinstance(value, float):
            if 'max' not in instances.keys():
                instances['max'] = 0
                instances['min'] = np.inf
            if value > instances['max']:
                instances['max'] = value
            if value < instances['min']:
                instances['min'] = value
        elif value is None:
            if 'None' not in instances.keys():
                instances['None'] = 0
            instances['None'] += 1
    return instances