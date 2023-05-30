import configargparse
from pathlib import Path
from satellitepy.data.tools import show_labels_on_image
from satellitepy.utils.path_utils import create_folder, init_logger, get_project_folder
import logging
"""
Show labels (e.g., bounding boxes) on an image
"""

# TODO: 
#	Accept arguments from terminal

if __name__ == '__main__':
    # args = get_args()
    # run(args)
	# img_path = '/home/murat/Projects/satellitepy/data/fair1m/train/images/16483.tif'
	img_path = '/home/murat/Projects/satellitepy/data/fair1m/train/images/16487.tif'
	
	label_path = '/home/murat/Projects/satellitepy/data/fair1m/train/bounding_boxes/16487.xml'
	label_format = 'fair1m'

	show_labels_on_image(img_path,label_path,label_format)