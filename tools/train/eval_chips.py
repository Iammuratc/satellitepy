import argparse
import logging
from pathlib import Path

import numpy as np

import torch
import torchvision

from satellitepy.data.utils import get_satellitepy_table
from satellitepy.dataset.chip_dataset import get_train_val_dataloaders
from satellitepy.models.chips.chip_models import get_model
from satellitepy.models.chips.train_model import TrainModule
from satellitepy.utils.path_utils import get_project_folder, init_logger, create_folder


def parse_args():
    project_folder = get_project_folder()

    parser = argparse.ArgumentParser(description='BBAVectors Implementation in satellitepy')
    parser.add_argument('--backbone', type=str, default='resnet34', help='Name of the backbone to use. Check "satellitepy/models/chips/chip_models.py" for options.')
    parser.add_argument('--num-epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Number of batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--init-lr', type=float, default=1.25e-4, help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of Patience epochs. If the valid loss does not improve for <patience> times, '
                             'the training will stop. ')
    parser.add_argument('--train-image-folder', type=Path,
                        help='Image folder. The images in this folder will be used to train the model.')
    parser.add_argument('--train-label-folder', type=Path,
                        help='Label folder. The labels in this folder will be used to train the model.')
    parser.add_argument('--val-image-folder', type=Path,
                        help='Image folder. The images in this folder will be used to train the model.')
    parser.add_argument('--val-label-folder', type=Path,
                        help='Label folder. The labels in this folder will be used to train the model.')
    parser.add_argument('--eval-by-source', type=bool, default=False, required=False)
    parser.add_argument('--verbose-output', type=bool, default=False, required=False)

    parser.add_argument('--log-config-path', default=project_folder /
                                                     Path('configs/log.config'), type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, required=False, help='Log path.')
    parser.add_argument('--out-folder',
                        type=Path,
                        help='Save folder of experiments. The trained weights will be saved under this folder.')

    args = parser.parse_args()
    return args

def train_chips(args):
    train_image_path = Path(args.train_image_folder)
    train_label_path = Path(args.train_label_folder)

    val_image_path = Path(args.val_image_folder)
    val_label_path = Path(args.val_label_folder)

    out_folder = Path(args.out_folder)
    assert create_folder(out_folder)
    classes = list(get_satellitepy_table()['fineair-class'].keys())

    log_path = Path(
        out_folder) / 'train_chips.log' if args.log_path is None else args.log_path
    init_logger(config_path=args.log_config_path, log_path=log_path)
    logger = logging.getLogger('')
    logger.info(
        f'No log path is given, the default log path will be used: {log_path}')
    logger.info('Initiating the training of the BBAVector model...')

    transform = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomRotation(180),
                 torchvision.transforms.RandomVerticalFlip()]

    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epochs = args.num_epoch

    eval_by_source = args.eval_by_source
    verbose_output = args.verbose_output

    train_dataloader, val_dataloader = get_train_val_dataloaders(
        train_image_path, train_label_path, val_image_path, val_label_path, classes, train_batch_size=batch_size, val_batch_size=1, num_workers=num_workers, transform=transform)

    backbone_name = args.backbone
    init_lr = args.init_lr
    model = get_model(backbone_name, len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)
    loss = torch.nn.CrossEntropyLoss()
    patience = args.patience

    train_module = TrainModule(model, train_dataloader, val_dataloader, optimizer, scheduler, loss, num_epochs, out_folder, classes, patience=patience, val_by_source=eval_by_source, verbose_output=verbose_output)

    train_module.train_network()

if __name__ == '__main__':
    args = parse_args()
    train_chips(args)