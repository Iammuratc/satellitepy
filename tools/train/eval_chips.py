import argparse
import logging
from pathlib import Path

import numpy as np

import torch
import torchvision
from torch import manual_seed
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from satellitepy.data.utils import get_satellitepy_table
from satellitepy.dataset.chip_dataset import get_train_val_dataloaders, ChipDataset
from satellitepy.models.chips.chip_models import get_model, Classifier
from satellitepy.models.chips.train_model import TrainModule
from satellitepy.models.utils import EarlyStopping
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
                        help='Image folder. The images in this folder will be used to validate the model.')
    parser.add_argument('--val-label-folder', type=Path,
                        help='Label folder. The labels in this folder will be used to validate the model.')
    parser.add_argument('--test-image-folder', type=Path,
                        help='Image folder. The images in this folder will be used to test the model.')
    parser.add_argument('--test-label-folder', type=Path,
                        help='Label folder. The labels in this folder will be used to test the model.')
    parser.add_argument('--eval-by-source', type=bool, default=False, required=False)
    parser.add_argument('--verbose-output', type=bool, default=False, required=False)

    parser.add_argument('--log-config-path', default=project_folder /
                                                     Path('configs/log.config'), type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, required=False, help='Log path.')
    parser.add_argument('--out-folder',
                        type=Path,
                        help='Save folder of experiments. The trained weights will be saved under this folder.')
    parser.add_argument('--manual-seed', type=int, help='Seed for splitting data in train and val sets')

    args = parser.parse_args()
    return args

def train_chips(args):
    train_image_path = Path(args.train_image_folder)
    train_labels_path = Path(args.train_label_folder)

    val_image_path = Path(args.val_image_folder)
    val_labels_path = Path(args.val_label_folder)

    test_image_path = Path(args.test_image_folder)
    test_labels_path = Path(args.test_label_folder)

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

    transform = [torchvision.transforms.RandomVerticalFlip()]

    # seed = args.manual_seed if args.manual_seed else np.random.randint(4242)
    batch_size = args.batch_size
    train_batch_size = batch_size
    val_batch_size = 1 #batch_size
    test_batch_size = 1 #batch_size

    num_workers = args.num_workers
    num_epochs = args.num_epoch

    eval_by_source = args.eval_by_source
    verbose_output = args.verbose_output

    train_dataset = ChipDataset(train_image_path, train_labels_path, classes, transform)
    val_dataset = ChipDataset(val_image_path, val_labels_path, classes)
    test_dataset = ChipDataset(test_image_path, test_labels_path, classes)


    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    backbone_name = args.backbone
    model = get_model(backbone_name, len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss = torch.nn.CrossEntropyLoss()

    train_module = TrainModule(model, train_dataloader, val_dataloader, test_dataloader, optimizer, scheduler, loss, num_epochs, out_folder, classes, backbone_name, patience=10, val_by_source=eval_by_source, verbose_output=verbose_output)

    train_module.train_network()
    train_module.test()

if __name__ == '__main__':
    args = parse_args()
    train_chips(args)