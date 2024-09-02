import json
import logging
from glob import glob

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

logger = logging.getLogger('')


def compute_mean_and_std(chip_paths):
    mean = np.zeros(3)
    std = np.zeros(3)
    num_images = len(chip_paths)

    for path in tqdm(chip_paths, desc='Computing mean and std'):
        # Load image
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = np.array(image) / 255.0  # Normalize to [0, 1]

        # Calculate mean and std per channel
        for i in range(3):
            mean[i] += image[:, :, i].mean()
            std[i] += image[:, :, i].std()

    # Calculate the mean and std over all images
    mean /= num_images
    std /= num_images

    return mean, std


def get_train_val_dataloaders(train_chip_path, train_label_path, val_chip_path, val_label_path, classes, train_batch_size, val_batch_size, transform=[], shuffle=True, num_workers=4, seed=42424242):
    train_dataset = ChipDataset(train_chip_path, train_label_path, classes, transform)
    val_dataset = ChipDataset(val_chip_path, val_label_path, classes, transform)

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader


class ChipDataset(Dataset):
    def __init__(self, chip_path, label_path, classes, transform=[]):
        chip_paths = [f for f in sorted(glob(str(chip_path) + "/*.png"))]
        label_paths = [f for f in sorted(glob(str(label_path) + "/*.json"))]

        self.mean, self.std = compute_mean_and_std(chip_paths)

        self.transform = v2.Compose(transform)

        self.classes = classes
        self.items = list(zip(chip_paths, label_paths))

    def __len__(self):
        return len(self.items)

    def _preprocess(self, img):
        defaults = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(self.mean, self.std)])
        img = defaults(img)
        return self.transform(img)

    def __getitem__(self, idx):
        chip_path, label_path = self.items[idx]

        chip = cv2.imread(chip_path, cv2.IMREAD_COLOR)
        with open(label_path, 'r') as f:
            file = json.load(f)
        label_idx = self.classes.index(file['fineair-class'][0])
        source = file['source'][0]

        if source == 'FR24':
            s = 0
        elif source == 'Mask':
            s = 1
        elif source is None or source == 'None':
            s = 2
        else:
            raise Exception(f'source {source} is unknown')

        label = np.zeros([len(self.classes)])
        label[label_idx] = 1

        chip = self._preprocess(chip)

        return chip, label, s


