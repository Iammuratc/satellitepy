import json
from glob import glob

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.transforms import v2
from tqdm import tqdm


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


def get_train_val_dataloaders(chip_path, label_path, classes, train_batch_size, val_batch_size, transform=[], get_source=False, shuffle=True, num_workers=4):
    dataset = ChipDataset(chip_path, label_path, classes, transform)

    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator)

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader


class ChipDataset(Dataset):
    def __init__(self, chip_path, label_path, classes, task, transform=[], multiplier=None):
        if not multiplier:
            chip_paths = [f for f in sorted(glob(str(chip_path) + "/*.png"))]
            label_paths = [f for f in sorted(glob(str(label_path) + "/*.json"))]
        else:
            chip_paths = []
            label_paths = []
            for c, l in zip(sorted(glob(str(chip_path) + "/*.png")), sorted(glob(str(label_path)+ "/*.json"))):
                with open(l, 'r') as f:
                    file = json.load(f)
                class_name = file[task][0]
                multiply = int(1 + np.rint(np.max([multiplier[class_name]-1, 0])))

                chip_paths += [c]*multiply
                label_paths += [l]*multiply

        self.mean, self.std = compute_mean_and_std(chip_paths)

        self.transform = v2.Compose(transform)
        self.task = task
        self.classes = classes
        self.items = list(zip(chip_paths, label_paths))

    def __len__(self):
        return len(self.items)

    def _preprocess(self, img):
        defaults = v2.Compose([v2.ToImageTensor(), v2.ToDtype(torch.float32), v2.Normalize(self.mean, self.std)])
        img = defaults(img)
        return self.transform(img)

    def __getitem__(self, idx):
        chip_path, label_path = self.items[idx]

        chip = cv2.imread(chip_path, cv2.IMREAD_COLOR)
        with open(label_path, 'r') as f:
            file = json.load(f)
        class_idx = file[self.task][0]

        label_idx = self.classes.index(class_idx)
        source = file['source'][0]

        if source == 'FR24':
            s = 0
        elif source == 'Mask':
            s = 1
        else:
            s = 2       # source = None

        label = np.zeros([len(self.classes)])
        label[label_idx] = 1

        chip = self._preprocess(chip)

        return chip, label, s


