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
from tools.data.analyze_labels import analyse_label_paths

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.01, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            noise = torch.randn(tensor.size()) * self.std + self.mean
            return tensor + noise
        return tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class RandomChannelShuffle:
    def __call__(self, img):
        if not isinstance(img, torch.Tensor) or img.ndimension() != 3:
            raise TypeError("Input must be a 3D PyTorch tensor with shape [C, H, W]")
            # Shuffle channels
        perm = torch.randperm(img.size(0))  # Random permutation of [0, 1, 2]
        return img[perm, :, :]  # Reorder channels


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
    parser.add_argument('--task', type=str, required=True, help='Task to evaluate')
    parser.add_argument('--augmentation-factor', type=float, default=0, help='Percentage by which lower instance class are augmented. 0 means no instance number based augmentation, '
                                                                             '1 results in almost completely balanced classes.')
    parser.add_argument('--augmentation-percentage', type=float, default=0.2, help='Probability of using augmentations. Final p = aug_factor*aug_percentage, including augmentation_factor.')
    parser.add_argument('--eval-by-source', type=bool, default=False, required=False)
    parser.add_argument('--verbose-output', type=bool, default=False, required=False)

    parser.add_argument('--log-config-path', default=project_folder /
                                                     Path('configs/log.config'), type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, required=False, help='Log path.')
    parser.add_argument('--out-folder',
                        type=Path,
                        help='Save folder of experiments. The trained weights will be saved under this folder.')
    parser.add_argument('--keep-classes', type=str, default='all', help='If not all, contains all class names that are trained on. Every class not in this list will not be considered.')

    args = parser.parse_args()
    return args

def train_chips(args):
    logger = logging.getLogger('')

    train_image_path = Path(args.train_image_folder)
    train_labels_path = Path(args.train_label_folder)

    val_image_path = Path(args.val_image_folder)
    val_labels_path = Path(args.val_label_folder)

    test_image_path = Path(args.test_image_folder)
    test_labels_path = Path(args.test_label_folder)

    out_folder = Path(args.out_folder)
    assert create_folder(out_folder)

    task = args.task

    logger.info('Analyzing labels')
    class_distribution = analyse_label_paths(train_labels_path,
                                             label_format='satellitepy',
                                             task=task,
                                             logger=logger,
                                             plot_bar=False,
                                             plot_sunburst_bar=False,
                                             plot_horizontal_bar=False,
                                             out_folder=out_folder,
                                             max_class_name_length=0,
                                             print_none=True,
                                             group_into_other_threshold=0,
                                             remove_other=False,
                                             remove_zero=True)

    keep_classes = [c.strip() for c in args.keep_classes.split(",")] if args.keep_classes != 'all' else 'all'
    if keep_classes != 'all':
        keep_dict = {}
        for k, v in class_distribution.items():
            if k in keep_classes:
                keep_dict[k] = v
        class_distribution = keep_dict
    logger.info(f'class distribution: {class_distribution}')

    augmentation_factor = args.augmentation_factor
    augmentation_percentage = args.augmentation_percentage * augmentation_factor
    num_all = np.sum(list(class_distribution.values()))
    val_max = np.max(list(class_distribution.values()))

    multiplier = {}
    for k, v in class_distribution.items():
        multiplier[k] = np.rint(augmentation_factor * (val_max / class_distribution[k]))

    logger.info(f'using class multipliers: {multiplier}')

    classes = list(class_distribution.keys())

    log_path = Path(
        out_folder) / 'train_chips.log' if args.log_path is None else args.log_path
    init_logger(config_path=args.log_config_path, log_path=log_path)
    logger = logging.getLogger('')
    logger.info(
        f'No log path is given, the default log path will be used: {log_path}')
    logger.info('Initiating the training of the BBAVector model...')

    transform = [torchvision.transforms.RandomHorizontalFlip(p=0.5),
                 torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=augmentation_percentage),
                 torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=augmentation_percentage),
                 torchvision.transforms.RandomApply([torchvision.transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05))], p=augmentation_percentage),
                 torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5))], p=augmentation_percentage),
                 AddGaussianNoise(mean=0.0, std=0.01, p=augmentation_percentage),
                 torchvision.transforms.RandomApply([RandomChannelShuffle()], p=augmentation_percentage)]

    batch_size = args.batch_size
    train_batch_size = batch_size
    val_batch_size = 1
    test_batch_size = 1

    num_workers = args.num_workers
    num_epochs = args.num_epoch

    eval_by_source = args.eval_by_source
    verbose_output = args.verbose_output

    train_dataset = ChipDataset(train_image_path, train_labels_path, classes, task, transform, multiplier, keep_classes=keep_classes)
    val_dataset = ChipDataset(val_image_path, val_labels_path, classes, task, keep_classes=keep_classes)
    test_dataset = ChipDataset(test_image_path, test_labels_path, classes, task, keep_classes=keep_classes)


    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    backbone_name = args.backbone
    model = get_model(backbone_name, len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss = torch.nn.CrossEntropyLoss()

    train_module = TrainModule(model, train_dataloader, val_dataloader, test_dataloader, optimizer,
                               scheduler, loss, num_epochs, out_folder, classes, backbone_name, patience=5, val_by_source=eval_by_source, verbose_output=verbose_output)

    train_module.train_network()
    train_module.test()

if __name__ == '__main__':
    args = parse_args()
    train_chips(args)