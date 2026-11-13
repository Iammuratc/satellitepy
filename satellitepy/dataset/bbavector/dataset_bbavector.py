import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import hashlib

from satellitepy.data.labels import read_label
from satellitepy.dataset.bbavector.utils import Utils
from satellitepy.data.utils import get_satellitepy_dict_values, get_task_dict
from satellitepy.utils.path_utils import get_file_paths, zip_matched_files


class BBAVectorDataset(Dataset):
    def __init__(self,
                 in_image_folder,
                 in_label_folder,
                 in_label_format,
                 tasks,
                 input_h,
                 input_w,
                 down_ratio,
                 target_task='coarse-class',
                 augmentation=False,
                 validate_dataset=True,
                 K=1000,
                 random_seed=12,
                 input_channels=3):
        super(BBAVectorDataset, self).__init__()

        self.utils = Utils(tasks, input_h, input_w, down_ratio, K, augmentation, input_channels)
        self.input_channels = input_channels
        self.tasks = tasks
        self.target_task = target_task
        self.items = []
        self.augmentation = augmentation
        self.random_seed = random_seed
        self.testing = not in_label_folder

        self.grouped_tasks = []
        for t in self.tasks:
            if t not in ['obboxes', 'hbboxes', 'masks']:
                task_dict = get_task_dict(t)
                if 'min' in task_dict.keys() and 'max' in task_dict.keys():
                    assert t != target_task, f'target-task can not be a regression task but is {target_task}.'
                    self.grouped_tasks.append('reg_' + t)
                else:
                    self.grouped_tasks.append('cls_' + t)
            else:
                self.grouped_tasks.append(t)

        if not in_label_folder:
            for img_path in get_file_paths(in_image_folder):
                self.items.append((img_path, None, None))
        else:
            if validate_dataset:
                total = len(os.listdir(in_image_folder))
                removed = 0
                pbar = tqdm(zip_matched_files(in_image_folder, in_label_folder), total=total, desc='validating data')

                for img_path, label_path in pbar:
                    hash_str = str(img_path) + str(label_path) + str(self.random_seed)
                    hash_bytes = hashlib.sha256(bytes(hash_str, 'utf-8')).digest()[:4]
                    np.random.seed(int.from_bytes(hash_bytes[:4], 'little'))
                    image = cv2.imread(img_path.absolute().as_posix(), cv2.IMREAD_UNCHANGED)
                    if image.ndim == 2:
                        image = image[:, :, np.newaxis]
                    if self.input_channels == 3 and image.shape[2] == 4:
                        image = image[:, :, :3]
                    if image.shape[2] != self.input_channels:
                        raise ValueError(f'Expected {self.input_channels} channels: {img_path}')
                    labels = read_label(label_path, in_label_format)
                    image_h, image_w, c = image.shape
                    annotation = self.utils.prepare_annotations(labels, image_w, image_h)
                    image, annotation = self.utils.data_transform(image, annotation, self.augmentation)

                    if self.all_tasks_available(annotation):
                        self.items.append((img_path, label_path, in_label_format))
                    else:
                        removed += 1
                        pbar.set_description(f'validating data (removed: {removed})')
            else:
                for img_path, label_path in zip_matched_files(in_image_folder, in_label_folder):
                    self.items.append((img_path, label_path, in_label_format))

    def __len__(self):
        return len(self.items)

    def all_tasks_available(self, annotation: dict):
        """
        Checks if labels had a bad format or annotations are missing after random crop
        """
        existing = list(annotation.keys())
        for k in self.grouped_tasks:
            if k not in existing:
                return False
        return True

    def __getitem__(self, idx):
        img_path, label_path, label_format = self.items[idx]

        if self.testing:
            image = cv2.imread(img_path.absolute().as_posix(), cv2.IMREAD_UNCHANGED)
            if image.ndim == 2:
                image = image[:, :, np.newaxis]
            if self.input_channels == 3 and image.shape[2] == 4:
                image = image[:, :, :3]
            image_h, image_w, c = image.shape
            # DOTA follows the original BBAVector recipe: 600x600 source
            # crops are resized to the 608x608 network input. Keep the source
            # dimensions below so decoded boxes are mapped back correctly.
            if image_h != self.utils.input_h or image_w != self.utils.input_w:
                image = cv2.resize(
                    image, (self.utils.input_w, self.utils.input_h),
                    interpolation=cv2.INTER_LINEAR)
            return {
                'input': torch.from_numpy(image / 255).permute((2, 0, 1)),
                'img_path': str(img_path),
                'img_w': image_w,
                'img_h': image_h
            }

        data_dict = self.utils.get_data_dict(img_path, label_path, label_format, self.target_task)
        return data_dict

    def visualize_masks(self, image, masks, labels):
        mask_image = np.array(image)
        image_h, image_w, _ = image.shape
        mask = np.zeros((image_h, image_w, 1), dtype=np.uint8)
        for m in np.moveaxis(masks, -1, 0):
            test_m = m.astype(np.uint8)[:, :, np.newaxis]
            mask = cv2.bitwise_or(mask, test_m)

        mask_image = cv2.bitwise_and(mask_image, mask_image, mask=mask)
        vis_image = np.concatenate((mask_image, image), axis=1)
        title = ",".join(list(set([value for value in get_satellitepy_dict_values(labels, self.task)])))
        cv2.imshow(title, vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
