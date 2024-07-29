import argparse
from pathlib import Path

import numpy as np

import torch
import torchvision
from torch import manual_seed

from tqdm import tqdm

from satellitepy.data.utils import get_satellitepy_table
from satellitepy.dataset.chip_dataset import get_train_val_dataloaders
from satellitepy.models.chips.chip_models import get_model, Classifier
from satellitepy.models.chips.train_model import TrainModule
from satellitepy.models.utils import EarlyStopping
from satellitepy.utils.path_utils import get_project_folder


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
    parser.add_argument('--image-folder', type=Path,
                        help='Image folder. The images in this folder will be used to train the model.')
    parser.add_argument('--label-folder', type=Path,
                        help='Label folder. The labels in this folder will be used to train the model.')
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
    in_image_path = Path(args.image_folder)
    in_labels_path =Path(args.label_folder)
    save_path = args.out_folder
    classes = list(get_satellitepy_table()['fineair-class'].keys())

    transform = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomRotation(180),
                 torchvision.transforms.RandomVerticalFlip()]

    manual_seed = args.manual_seed if args.manual_seed else np.random.randint()
    batch_size = args.batch_size
    num_workers = args.num_workers

    train_dataloader, val_dataloader = get_train_val_dataloaders(
        in_image_path, in_labels_path, classes, train_batch_size=batch_size, val_batch_size=1, num_workers=num_workers, transform=transform, manual_seed=manual_seed)

    backbone_name = args.backbone
    model = get_model(backbone_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss = torch.nn.CrossEntropyLoss()

    train_module = TrainModule(model, train_dataloader, val_dataloader, optimizer, scheduler, loss, 50, save_path, classes, patience=10, val_by_source=True, verbose_output=True)

    train_module.train_network()

if __name__ == '__main__':
    args = parse_args()
    train_chips(args)

    # in_image_path = 'S:\\UniBw\\chips\\images'
    # in_labels_path = 'S:\\UniBw\\chips\\labels'
