import argparse
import os
from pathlib import Path
import logging
import torch


from satellitepy.utils.path_utils import create_folder, init_logger, get_project_folder
from satellitepy.models.bbavector import train_model
from satellitepy.models.bbavector.tools import get_model
from satellitepy.dataset.bbavector.dataset_bbavector import BBAVectorDataset
from satellitepy.data.utils import get_task_dict



def parse_args():
    project_folder = get_project_folder()

    parser = argparse.ArgumentParser(description='BBAVectors Implementation in satellitepy')
    parser.add_argument('--num-epoch', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--init-lr', type=float, default=1.25e-4, help='Initial learning rate')
    parser.add_argument('--input-h', type=int, default=608, help='Resized image height')
    parser.add_argument('--input-w', type=int, default=608, help='Resized image width')
    parser.add_argument('--K', type=int, default=500, help='Maximum of objects')
    parser.add_argument('--conf-thresh', type=float, default=0.18, help='Confidence threshold, 0.1 for general evaluation')
    parser.add_argument('--ngpus', type=int, default=1, help='Number of gpus, ngpus>1 for multigpu')
    parser.add_argument('--patience', type=int, default=10, help='Number of Patience epochs. If the valid loss does not improve for <patience> times, the training will stop. ')
    parser.add_argument('--resume-train', type=Path, help='Weights resumed in training')
    parser.add_argument('--train-image-folder', type=Path,
                        help='Image folder. The images in this folder will be used to train the model.')
    parser.add_argument('--train-label-folder', type=Path,
                        help='Label folder. The labels in this folder will be used to train the model.')
    parser.add_argument('--train-label-format', type=Path,
                        help='Label file format. e.g., dota, fair1m, satellitepy.')
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

    # parser.add_argument('--resume', type=str, default='model_50.pth', help='Weights resumed in testing and evaluation')
    # parser.add_argument('--dataset', type=str, default='dota', help='Name of dataset')
    # parser.add_argument('--data_dir', type=str, default='../Datasets/dota', help='Data directory')

    # parser.add_argument('--phase', type=str, default='test', help='Phase choice= {train, test, eval}')
    # parser.add_argument('--wh_channels', type=int, default=8, help='Number of channels for the vectors (4x2)')
    args = parser.parse_args()
    return args

def train_bbavector(args):

    # Data input
    train_image_folder = Path(args.train_image_folder)
    train_label_folder = Path(args.train_label_folder)
    train_label_format = str(args.train_label_format)
    valid_image_folder = Path(args.valid_image_folder)
    valid_label_folder = Path(args.valid_label_folder)
    valid_label_format = str(args.valid_label_format)
    input_h = args.input_h
    input_w = args.input_w
    down_ratio = 4
    patience = args.patience
    task = args.task_name
    # Training input
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    num_workers = args.num_workers
    init_lr = args.init_lr
    conf_thresh = args.conf_thresh
    K = args.K
    ngpus = args.ngpus
    checkpoint_path = args.resume_train

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

    # Model
    model = get_model(task,down_ratio,checkpoint_path)

    task_dict = get_task_dict(task)
    num_classes = len(task_dict)
    
    train_dataset = BBAVectorDataset(
        train_image_folder,
        train_label_folder,
        train_label_format,
        task,
        task_dict,
        input_h,
        input_w,
        down_ratio)

    if args.valid_image_folder:
        valid_dataset = BBAVectorDataset(
            valid_image_folder,
            valid_label_folder,
            valid_label_format,
            task,
            task_dict,
            input_h,
            input_w,
            down_ratio)
    else:
        valid_dataset = None


    ctrbox_obj = train_model.TrainModule(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        model=model,
        # decoder=model_decoder,
        down_ratio=down_ratio,
        out_folder=out_folder,
        init_lr=init_lr,
        num_epoch=num_epoch,
        batch_size=batch_size,
        num_workers=num_workers,
        conf_thresh=conf_thresh,
        ngpus=ngpus,
        resume_train=checkpoint_path,
        patience=patience
        )

    ctrbox_obj.train_network()


if __name__ == '__main__':
    args = parse_args()
    train_bbavector(args)
    # sample_temp = next(iter(my_dataloader))
    # print(sample_temp.keys())
    # import matplotlib.pyplot as plt
    # import matplotlib
    # import numpy as np
    # matplotlib.use('TKAgg')
    # for i in range(15):
    #     sample_heatmap = sample_temp['hm'][0][i,:,:].cpu().detach().numpy()
    #     hm_max = np.amax(sample_heatmap)
    #     print(hm_max)
    #     if hm_max==0:
    #         continue
    #     else:
    #         plt.imshow(sample_heatmap*255)
    #         plt.show()
