from pathlib import Path

def add_shared_args(parser):
	"""
	Add shared arguments of the parsers
	Parameter
	---------
	parser : configargparse.ArgumentParser
		Parser.
	Returns
	-------
	parser : configargparse.ArgumentParser
		Parser.		
	"""
	parser.add_argument('--config-path', required=True, help='Path to MMRotate config file. Please refer to https://github.com/open-mmlab/mmrotate for more details.')
	parser.add_argument('--weights-path', required=True, help='Path to MMRotate model weights file. Please refer to https://github.com/open-mmlab/mmrotate for more details.')
	parser.add_argument('--nms-on-multiclass-thr', default=0.5, type=float, help='nms_on_multiclass_thr value is used to filter out the overlapping bounding boxes with lower scores, and keep the best. Set 0.0 to cancel it.')
	parser.add_argument('--device', default='cuda:0', help='Device to load the model.')
	parser.add_argument('--class-names', required=True, type=str, help='Class names. MMRotate does not include class names in config files, but class indexes. Be sure that the indices match with the class names')
	parser.add_argument('--log-config-path', default=Path("./configs/log.config") ,type=Path, help='Log config file.')
	parser.add_argument('--in-image-folder', required=True, help='Test image folder. The images in this folder will be tested.')
	parser.add_argument('--in-label-folder', required=True, help='Test label folder. The labels in this folder will be used for evaluation purposes.')
	parser.add_argument('--in-label-format', required=True, help='Test label file format. e.g., dota, fair1m.')
	return parser
