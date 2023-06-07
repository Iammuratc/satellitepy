import configargparse
from pathlib import Path
from satellitepy.utils.path_utils import create_folder, init_logger, get_project_folder
import logging
import subprocess
import zipfile
import fnmatch
try:
    import gdown
except ImportError:
    subprocess.check_call(["pip", "install", "gdown"])
    import gdown

"""
Download the Dota-Dataset v1.5
"""

def get_args():
    """Argument Parser"""
    project_folder = get_project_folder()
    parser = configargparse.ArgumentParser(description = __doc__)
    parser.add_argument('--in-folder', default = project_folder/Path('in_folder/dota/'), type = Path,
                        help = 'Which folder to save the dataset into')
    parser.add_argument('--unpack', action = 'store_true', help = 'Unpack the dataset')
    parser.add_argument('--log-config-path', default = project_folder/Path("configs/log.config"),
                        type=Path, help='Log config file.')
    parser.add_argument('--log-path',  type=Path, help='Log file.')
    args = parser.parse_args()
    return args

def main(args):

    parent_folder = Path(args.in_folder)
    assert create_folder(parent_folder)
    train_folder = parent_folder/Path("train")
    assert create_folder(train_folder)
    train_img_folder = train_folder/Path("img")
    assert create_folder(train_img_folder)
    train_label_folder = train_folder/Path("labels")
    assert create_folder(train_label_folder)
    train_label_folder_hbb = train_folder/Path("labels_hbb")
    assert create_folder(train_label_folder_hbb)
    val_folder = parent_folder/Path("val")
    assert create_folder(val_folder)
    val_img_folder = val_folder/Path("imgs")
    assert create_folder(val_img_folder)
    val_label_folder = val_folder/Path("labels")
    assert create_folder(val_label_folder)
    val_label_folder_hbb = val_folder/Path("labels_hbb")
    assert create_folder(val_label_folder_hbb)
    test_folder = parent_folder/Path("test")
    assert create_folder(test_folder)
    test_img_folder = test_folder/Path("imgs")
    assert create_folder(test_img_folder)

    train_img_file_ids = ["1BlaGYNNEKGmT6OjZjsJ8HoUYrTTmFcO2",
                    "1JBWCHdyZOd9ULX0ng5C9haAt3FMPXa3v",
                    "1pEmwJtugIWhiwgBqOtplNUtTG2T454zn"]
    train_label_file_id = "12uPWoADKggo9HGaqGh2qOmcXXn-zKjeX"
    train_label_hbb_file_id = "1-vLCMhIW9CV2cmCPPBbDR9_hdecf5bLb"
    val_img_file_id = "1uCCCFhFQOJLfjBpcL5MC0DHJ9lgOaXWP"
    val_label_file_id = "1FkCSOCy4ieNg1UZj1-Irfw6-Jgqa37cC"
    val_label_hbb_file_id = "1XDWNx3FkH9layL8jVUkEHJ_-CY8K4zse"
    test_img_file_ids = ["1fwiTNqRRen09E-O9VSpcMV2e6_d4GGVK",
                        "1wTwmxvPVujh1I6mCMreoKURxCUI8f-qv"]
    test_json_id = "1nQokIxSy3DEHImJribSCODTRkWlPJLE3"
    
    #train data
    for id in train_img_file_ids:
        url = f"https://drive.google.com/uc?id={id}"
        output_path = train_folder/Path(f"{id}.zip")
        output = str(output_path)
        if Path(output).is_file() == False:
            gdown.download(url, output, quiet = False)

    url = f"https://drive.google.com/uc?id={train_label_file_id}"
    output_path = train_folder/Path("train_labels.zip")
    output = str(output_path)
    if Path(output).is_file() == False:
        gdown.download(url, output, quiet = False)

    url = f"https://drive.google.com/uc?id={train_label_hbb_file_id}"
    output_path = train_folder/Path("train_labels_hbb.zip")
    output = str(output_path)
    if Path(output).is_file() == False:
        gdown.download(url, output, quiet = False)
    
    if args.unpack:
        for file in Path(train_folder).iterdir():
            if Path(file).is_file():
                if fnmatch.fnmatch(file, "*/train_labels.zip"):
                    with zipfile.ZipFile(file) as zip_ref:
                        zip_ref.extractall(train_label_folder)
                elif fnmatch.fnmatch(file, "*/train_labels_hbb.zip"):
                    with zipfile.ZipFile(file) as zip_ref:
                        zip_ref.extractall(train_label_folder)
                else:
                    with zipfile.ZipFile(train_folder/Path(file)) as zip_ref:
                        zip_ref.extractall(train_img_folder)

    #val data
    url = f"https://drive.google.com/uc?id={val_img_file_id}"
    output_path = val_folder/Path(f"{val_img_file_id}.zip")
    output = str(output_path)
    if Path(output).is_file() == False:
        gdown.download(url, output, quiet = False)

    url = f"https://drive.google.com/uc?id={val_label_file_id}"
    output_path = val_folder/Path("val_labels.zip")
    output = str(output_path)
    if Path(output).is_file() == False:
        gdown.download(url, output, quiet = False)
    
    url = f"https://drive.google.com/uc?id={val_label_hbb_file_id}"
    output_path = val_folder/Path("val_labels_hbb.zip")
    output = str(output_path)
    if Path(output).is_file() == False:
        gdown.download(url, output, quiet = False)

    if args.unpack:
        for file in Path(val_folder).iterdir():
            if Path(file).is_file():
                if fnmatch.fnmatch(file, "*/val_labels.zip"):
                    with zipfile.ZipFile(file) as zip_ref:
                        zip_ref.extractall(val_label_folder)
                elif fnmatch.fnmatch(file, "*/val_labels_hbb.zip"):
                    with zipfile.ZipFile(file) as zip_ref:
                        zip_ref.extractall(val_label_folder_hbb)
                else:
                    with zipfile.ZipFile(file) as zip_ref:
                        zip_ref.extractall(val_img_folder)

    #test data
    url = f"https://drive.google.com/uc?id={test_json_id}"
    output_path = test_folder/Path("test_info.json")
    output = str(output_path)
    if Path(output).is_file() == False:
        gdown.download(url, output, quiet = False)

    for id in test_img_file_ids:
        url = f"https://drive.google.com/uc?id={id}"
        output_path = test_folder/Path(f"{id}.zip")
        output = str(output_path)
        if Path(output).is_file() == False:
            gdown.download(url, output, quiet = False)

    if args.unpack:
        for file in Path(test_folder).iterdir():
            if Path(file).is_file():
                if fnmatch.fnmatch(file, "*.zip"):
                    with zipfile.ZipFile(file) as zip_ref:
                        zip_ref.extractall(test_img_folder)

    # Init logger
    log_path = Path(
        parent_folder) / 'download_dota.log' if args.log_path == None else args.log_path
    init_logger(config_path=args.log_config_path, log_path=log_path)
    logger = logging.getLogger(__name__)
    if args.log_path == None:
        logger.info(
            f'No log path is given, the default log path will be used: {log_path}')
    logger.info('Downloaded dota datatset')
    if args.unpack:
        logger.info('Dataset unpacked')

if __name__ == '__main__':
    args = get_args()
    main(args)