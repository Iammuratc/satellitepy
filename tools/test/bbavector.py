import argparse
import os
from pathlib import Path
import logging
import torch


from satellitepy.utils.path_utils import create_folder, init_logger, get_project_folder
from satellitepy.models.bbavector import ctrbox_net, decoder, train_model
from satellitepy.data.utils import get_satellitepy_table, get_satellitepy_dict_values
from satellitepy.dataset.bbavector.dataset_bbavector import BBAVectorDataset



def parse_args():
    project_folder = get_project_folder()

    parser = argparse.ArgumentParser(description='BBAVectors Implementation in satellitepy')
    parser.add_argument('--num-epoch', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--input-h', type=int, default=608, help='Resized image height')
    parser.add_argument('--input-w', type=int, default=608, help='Resized image width')
    parser.add_argument('--conf-thresh', type=float, default=0.18, help='Confidence threshold, 0.1 for general evaluation')
    parser.add_argument('--model-path', type=Path, help='Model to be tested')
    parser.add_argument('--valid-image-folder', type=Path,
                        help='Image folder. The images in this folder will be used to validate the model.')
    parser.add_argument('--valid-label-folder', type=Path,
                        help='Label folder. The labels in this folder will be used to validate the model.')
    parser.add_argument('--valid-label-format', type=Path,
                        help='Label file format. e.g., dota, fair1m, satellitepy.')
    parser.add_argument('--log-config-path', default=project_folder /
                        Path("configs/log.config"), type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, required=False, help='Log path.')
    parser.add_argument('--task-name', default='coarse-class', type=str, help='The model will be trained for the given task.' 
        'Find the other task names at satellitepy.data.utils.get_satellitepy_table.'
        'If it is fine-class or very-fine class, None values in those keys will be filled from one upper level')
    parser.add_argument('--out-folder',
                        type=Path,
                        help='Save folder of experiments. The trained weights will be saved under this folder.')
    parser.add_argument('--K', type=int, default=500, help='Maximum of objects')

    args = parser.parse_args()
    return args

def test_bbavector(args):
    valid_image_folder = Path(args.valid_image_folder)
    valid_label_folder = Path(args.valid_label_folder)
    valid_label_format = str(args.valid_label_format)
    input_h = args.input_h
    input_w = args.input_w
    model_path = args.model_path
    down_ratio = 4
    task = args.task_name
    K = args.K
    conf_thresh = args.conf_thresh

    num_epoch = args.num_epoch
    batch_size = args.batch_size
    num_workers = args.num_workers
    # Data output
    out_folder = Path(args.out_folder)
    assert create_folder(out_folder)

    # Init logger
    log_path = Path(
        out_folder) / 'train_bbavector.log' if args.log_path == None else args.log_path
    init_logger(config_path=args.log_config_path, log_path=log_path)
    logger = logging.getLogger(__name__)
    logger.info(
        f'No log path is given, the default log path will be used: {log_path}')
    logger.info('Initiating the training of the BBAVector model...')

    # Dataset
    satellitepy_table = get_satellitepy_table()
    task_dict = get_satellitepy_dict_values(satellitepy_table,task)

    num_classes = len(task_dict)
    heads = {'hm': num_classes,
             'wh': 10,
             'reg': 2,
             'cls_theta': 1
             }
    model = ctrbox_net.CTRBOX(heads=heads,
                              pretrained=True,
                              down_ratio=down_ratio,
                              final_kernel=1,
                              head_conv=256)

    model_decoder = decoder.DecDecoder(K=K,
                                 conf_thresh=conf_thresh,
                                 num_classes=num_classes)

    valid_dataset = BBAVectorDataset(
        valid_image_folder,
        valid_label_folder,
        valid_label_format,
        task,
        task_dict,
        input_h,
        input_w,
        down_ratio)

    ctrbox_obj = train_model.TrainModule(
        train_dataset=valid_dataset,
        valid_dataset=valid_dataset,
        model=model,
        decoder=model_decoder,
        down_ratio=down_ratio,
        out_folder=out_folder,
        init_lr=1e-5,
        num_epoch=num_epoch,
        batch_size=batch_size,
        num_workers=num_workers,
        conf_thresh=conf_thresh,
        ngpus=2,
        resume_train=model_path,
        patience=10
        )

    ctrbox_obj.test_network()
    


if __name__ == '__main__':
    args = parse_args()
    test_bbavector(args)
