import urllib.request
import patoolib
import logging
import configargparse
from satellitepy.utils.path_utils import init_logger, create_folder
from pathlib import Path


DATASET_NAME = "VHR-10_dataset_coco"
DATASET_URL = "https://drive.google.com/uc?id=1--foZ3dV5OCsqXQXT84UeKtrAqc5CkAE&export=download&confirm=true"

def get_args():
	"""Arguments parser."""
	parser = configargparse.ArgumentParser(description=__doc__)
	parser.add_argument('--in-folder', type=Path, required=True,
						help='Folder where the dataset should be downloaded to.')
	parser.add_argument('--log-config', type=Path, required=True,
						help='Whether to write to log to file.')
	parser.add_argument('--log-path', type=Path, required=False,
						help='Where the log config should be saved.')
	parser.add_argument('--unzip', type=bool, required=True, help="If datasets with compressed files should be automatically unzipped")
	args = parser.parse_args()
	return args

def download_VHR_10_dataset_coco(in_path, log_config, log_path=None, unzip=True):
	if (type(in_path) != str or type(unzip) != bool or type(log_config) != bool): raise TypeError() # Sanity-checking inputs
	if (log_config and (type(log_path) != str and log_path != None) ): raise TypeError()
	in_path = str(Path(in_path) / DATASET_NAME)
	if not (Path(in_path).is_dir()):
		create_folder(in_path) #creates directory if not already existing
	if (log_config):
		if (log_path == None): log_path =  str(Path(in_path) / "download.log")
		init_logger(config_path, log_path)
	if (log_path != None): log_path = str(Path(in_path) / "download.log") # Set log file location
	try:
		logging.info("Attempting to download archive from server.")
		file = urllib.request.urlretrieve(DATASET_URL, str(Path(in_path) / "NWPU_VHR-10_dataset.rar"))[0]
	except Exception as e:
		if (logging): logging.error("Failed to download dataset with the following error:\n"+ str(e))
	else:
		if (logging): logging.info("Download complete.")
		if (unzip):
			try: patoolib.extract_archive(file, outdir=in_path)
			except patoolib.util.PatoolError:
				if (logging): logging.warning("Failed to extract .rar file. It is likely that you are missing the non-free software required to handle the archive format.")
			else:
				if (logging): logging.info("Archive extracted.")

if __name__ == '__main__':
	args = get_args()
	download_VEDAI(args.in_folder, args.log_config, args.log_path, args.unzip)
