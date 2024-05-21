from torch.utils.data.dataset import Dataset
import cv2
import torch
from pathlib import Path

from satellitepy.utils.path_utils import zip_matched_files
from satellitepy.data.labels import read_label


class MTLDataset(Dataset):
    """
    The dataset for satellitepy's multitask learning approach. 
    Simply reads the images and labels corresponding to the given parameters.
    Parameters
    ----------
    image_folders: list[str]
        The paths to the image folders, that shall be loaded by this Dataset.
    label_folders : list[str]
        The paths to the label folders, corresponding to the image_folders.
    label_formats : list[str]
        The label format identifier for each label folder instance.
    image_transforms: 
        Transformation(s) that accept an image tensor (C, H, W) and returns an image tensor of the same shape.
    label_transforms:
        Transformation(s) that accept a label object in the satellitepy format and transforms it.
    """

    def __init__(self,
                 image_folders: list[str],
                 label_folders: list[str],
                 label_formats: list[str],
                 image_transforms=None,
                 label_transforms=None
                 ):
        self.items = []
        self.image_transforms = image_transforms
        self.label_transforms = label_transforms

        for image_folder, label_folder, label_format in zip(image_folders, label_folders, label_formats):
            for img_path, label_path in zip_matched_files(Path(image_folder), Path(label_folder)):
                self.items.append((img_path, label_path, label_format))

    def __getitem__(self, idx):
        img_path, label_path, label_format = self.items[idx]
        cv2_image = cv2.imread(img_path.absolute().as_posix())
        image = torch.from_numpy(cv2_image).permute((2, 0, 1))

        if image.max() > 1:
            image = image.float() / 255.0

        gt_labels = read_label(label_path, label_format)

        if self.image_transforms:
            image = self.image_transforms(image)
        if self.label_transforms:
            gt_labels = self.label_transforms(gt_labels)

        return image, gt_labels

    def __len__(self):
        return len(self.items)
