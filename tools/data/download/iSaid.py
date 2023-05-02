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
Download the iSaid-Dataset
"""

def get_args():
    """Argument Parser"""
    project_folder = get_project_folder()
    parser = configargparse.ArgumentParser(description = __doc__)
    parser.add_argument('--in-folder', default = project_folder/Path('in_folder/iSaid/'), type = Path,
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
    train_semanticMask_img_folder = train_folder/Path("semanticMask_img")
    assert create_folder(train_semanticMask_img_folder)
    train_instanceMask_img_folder = train_folder/Path("instanceMask_img")
    assert create_folder(train_instanceMask_img_folder)
    val_folder = parent_folder/Path("val")
    assert create_folder(val_folder)
    val_semanticMask_img_folder = val_folder/Path("semanticMask_imgs")
    assert create_folder(val_semanticMask_img_folder)
    val_instanceMask_img_folder = val_folder/Path("instanceMask_imgs")
    assert create_folder(val_instanceMask_img_folder)
    test_folder = parent_folder/Path("test")
    assert create_folder(test_folder)
    test_img_folder = test_folder/Path("imgs")
    assert create_folder(test_img_folder)

    train_semanticMask_img_file_id = "1YLjZ1cmA9PH3OfzMF-eq6T-O9FTGvSrx"
    train_instanceMask_img_file_id = "12XhSgEt_Xw4FJQxLJZgMutw3awoAq2Ve"
    train_annotion_id = "1-PYSXak2JBg3xuZWzAWVXfKQO1TkPCqF"
    val_semanticMask_img_file_id = "1_PJy7kDVnp9tjETUbvwNjkLGpzDfqYh6"
    val_instanceMask_img_file_id = "1GCExuFqEKOY5Hyp1WSAmdW6I4RAaxBGG"
    val_annotion_id = "1QDKeAQ8Ka6_wxoN3Ld5t3EP9UHk7Fw98"
    test_img_file_ids = ["1fwiTNqRRen09E-O9VSpcMV2e6_d4GGVK","1wTwmxvPVujh1I6mCMreoKURxCUI8f-qv"]
    test_annotion_id = "1nQokIxSy3DEHImJribSCODTRkWlPJLE3"

    #train data
    url = f"https://drive.google.com/uc?id={train_semanticMask_img_file_id}"
    output_path = train_folder/Path("train_semanticMask.zip")
    output = str(output_path)
    if Path(output).is_file() == False:
        gdown.download(url, output, quiet = False)
    

    url = f"https://drive.google.com/uc?id={train_instanceMask_img_file_id}"
    output_path = train_folder/Path("train_instanceMask.zip")
    output = str(output_path)
    if Path(output).is_file() == False:
        gdown.download(url, output, quiet = False)

    url = f"https://drive.google.com/uc?id={train_annotion_id}"
    output_path = train_folder/Path("train.json")
    output = str(output_path)
    if Path(output).is_file() == False:
        gdown.download(url, output, quiet = False)
    
    if args.unpack:
        for file in Path(train_folder).iterdir():
            if Path(file).is_file():
                if fnmatch.fnmatch(file, "*/train_semanticMask.zip"):
                    with zipfile.ZipFile(file) as zip_ref:
                        zip_ref.extractall(train_semanticMask_img_folder)
                elif fnmatch.fnmatch(file, "*/train_instanceMask.zip"):
                    with zipfile.ZipFile(file) as zip_ref:
                        zip_ref.extractall(train_instanceMask_img_folder)

    #val data
    url = f"https://drive.google.com/uc?id={val_semanticMask_img_file_id}"
    output_path = val_folder/Path("val_semanticMask.zip")
    output = str(output_path)
    if Path(output).is_file() == False:
        gdown.download(url, output, quiet = False)

    url = f"https://drive.google.com/uc?id={val_instanceMask_img_file_id}"
    output_path = val_folder/Path("val_instanceMask.zip")
    output = str(output_path)
    if Path(output).is_file() == False:
        gdown.download(url, output, quiet = False)

    url = f"https://drive.google.com/uc?id={val_annotion_id}"
    output_path = val_folder/Path("val.json")
    output = str(output_path)
    if Path(output).is_file() == False:
        gdown.download(url, output, quiet = False)

    if args.unpack:
        for file in Path(val_folder).iterdir():
            if Path(file).is_file():
                if fnmatch.fnmatch(file, "*/val_semanticMask.zip"):
                    with zipfile.ZipFile(file) as zip_ref:
                        zip_ref.extractall(val_semanticMask_img_folder)
                elif fnmatch.fnmatch(file, "*/val_instanceMask.zip"):
                    with zipfile.ZipFile(file) as zip_ref:
                        zip_ref.extractall(val_instanceMask_img_folder)

    #test data
    for id in test_img_file_ids:
        url = f"https://drive.google.com/uc?id={id}"
        output_path = test_folder/Path(f"{id}.zip")
        output = str(output_path)
        if Path(output).is_file() == False:
            gdown.download(url, output, quiet = False)
        
    url = f"https://drive.google.com/uc?id={test_annotion_id}"
    output_path = test_folder/Path("test.json")
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
        parent_folder) / 'download_iSaid.log' if args.log_path == None else args.log_path
    init_logger(config_path=args.log_config_path, log_path=log_path)
    logger = logging.getLogger(__name__)
    if args.log_path == None:
        logger.info(
            f'No log path is given, the default log path will be used: {log_path}')
    logger.info('Downloaded iSaid datatset')
    if args.unpack:
        logger.info('Dataset unpacked')

if __name__ == '__main__':
    args = get_args()
    main(args)