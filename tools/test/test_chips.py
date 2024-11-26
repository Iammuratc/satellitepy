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

def parse_args():
    project_folder = get_project_folder()

    parser = argparse.ArgumentParser(description='BBAVectors Implementation in satellitepy')
    parser.add_argument('--backbone', type=str, default='resnet34', help='Name of the backbone to use. Check "satellitepy/models/chips/chip_models.py" for options.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--test-image-folder', type=Path,
                        help='Image folder. The images in this folder will be used to test the model.')
    parser.add_argument('--test-label-folder', type=Path,
                        help='Label folder. The labels in this folder will be used to test the model.')
    parser.add_argument('--task', type=str, required=True, help='Task to evaluate')
    parser.add_argument('--eval-by-source', type=bool, default=False, required=False)
    parser.add_argument('--verbose-output', type=bool, default=False, required=False)

    parser.add_argument('--log-config-path', default=project_folder /
                                                     Path('configs/log.config'), type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, required=False, help='Log path.')
    parser.add_argument('--classes', type=str, default='all', help='If not all, contains all class names that are tested. Must be the same the model is trained on')
    parser.add_argument('--out-folder',
                        type=Path,
                        help='Choose the out-folder of the training. Model must be called model_best.pth')

    args = parser.parse_args()
    return args

def test_chips(args):
    logger = logging.getLogger('')

    test_image_path = Path(args.test_image_folder)
    test_labels_path = Path(args.test_label_folder)

    out_folder = Path(args.out_folder)
    assert create_folder(out_folder)

    pred_folder = Path(out_folder)/ 'predictions'
    assert create_folder(pred_folder, ask_permission=False)

    log_path = Path(out_folder) / 'train_chips.log' if args.log_path is None else args.log_path
    init_logger(config_path=args.log_config_path, log_path=log_path)

    task = args.task

    classes = [c.strip() for c in args.classes.split(",")]
    logger.info(f'Testing on classes: {classes}')

    logger.info('Initiating testing')

    test_batch_size = 1

    num_workers = args.num_workers

    eval_by_source = args.eval_by_source
    verbose_output = args.verbose_output

    test_dataset = ChipDataset(test_image_path, test_labels_path, classes, task, keep_classes=classes)

    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    backbone_name = args.backbone
    model = get_model(backbone_name, len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss = torch.nn.CrossEntropyLoss()

    train_module = TrainModule(model, None, None, test_dataloader, optimizer,
                               scheduler, loss, -1, out_folder, classes, backbone_name, task, patience=-1, val_by_source=eval_by_source, verbose_output=verbose_output)

    train_module.test()

if __name__ == '__main__':
    args = parse_args()
    test_chips(args)