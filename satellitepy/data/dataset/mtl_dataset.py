from torch.utils.data.dataset import Dataset
import cv2
import torch
from pathlib import Path

from satellitepy.utils.path_utils import zip_matched_files
from satellitepy.data.labels import read_label, init_satellitepy_label, merge_satpy_label_dict, torchify_satpy_label_dict

# for testing purposes, remove when proper map is create
DOTA_IDX_MAP = {
    "object": 0,
    "vehicle": 1,
    "airplane": 2,
    "ship": 3,
    "helicopter": 4
}

def prepare_image(image_path: Path):
    """
    Loads an image from the given path, transforms it to a (C, H, W) tensor and 
    normalizes its values between [0, 1] if its max value is above 1.
    
    Parameters
    ----------
    image_path: str
        The path to the image.
    """
    cv2_image = cv2.imread(image_path.absolute().as_posix())
    # todo remove
    cv2_image = cv2.resize(cv2_image, (400, 400))
    image = torch.from_numpy(cv2_image).permute((2, 0, 1))

    if image.max() > 1:
        image = image.float() / 255.0

    return image

class TaskSpecificDataset(Dataset):
    """
    A task specific dataset. Created by splitting an MTLDataset.

    Parameters
    ----------
    data: dict
        A dictionary containing the dataset_id:[(image_path, label_path, label_format)] mapping.
    task_specific_data_mapping: list
        A list of task specific item mappings.
    """
    def __init__(self, data: dict, task_id: str, task_specific_data_mapping: list):
        super().__init__()
        self.data = data
        self.task_id = task_id
        self.task_specific_data_mapping = task_specific_data_mapping
        self.length = len(self.task_specific_data_mapping)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns a item depending on the idx.

        Parameters
        ----------
        idx: int
            The item idx.
        """
        dataset_id, dataset_idx = self.task_specific_data_mapping[idx % self.length]
        image_path, label_path, label_format = self.data[dataset_id][dataset_idx]
        image = prepare_image(image_path)
        label = read_label(label_path, label_format)
        label = access_label(label, self.task_id)
        return (image, label)

def access_label(satpy_dict: dict, task_label: str):
    access_list = task_label.split("_")

    if len(access_list) == 0:
        return None

    item = satpy_dict[access_list[0]]

    for a in access_list[1:]:
        item = item[a]

    if isinstance(item, list) and len(item) == 1:
        return item[0]

    return item

class MTLDataset(Dataset):
    """
    The dataset for satellitepy's multitask learning approach. 
    Made for satellite chips. Default pytorch collate might not work, if 
    not used with chips.

    Parameters
    ----------
    cfg : dict
        The configuration dict. See "Configuration Format" below.
    image_transforms: 
        Transformation(s) that accept an image tensor (C, H, W) and returns an image tensor of the same shape.
    label_transforms:
        Transformation(s) that accept a label object in the satellitepy format and transforms it.

    Configuration Format:
    {
        "datasets":{
            <id>: { # id to be referenced in tasks mapping
                "image_folder": <image folder path>,
                "label_folder": <label folder path>,
                "label_format": <the label format (e.g. dota)>
            }
        },
        "tasks": {
            <id>: { # task id
                "dataset_ids": [<id>, ...], # which datasets from the datasets mapping provide data for this task
                "label": <label_id> # task target from the satpy label dict format
            }
        },
        "align_len": <'max'|'min'>, # whether the length of this dataset is dictated by the 
                                       minimal or maximal length of all (task, ConcatDataset) tuples.
                                       Repeats items if maximal is chosen. Default: 'min'
    }
    """
    def __init__(self, 
            cfg: dict
        ):
        self.datasets = cfg["datasets"]
        self.tasks = cfg["tasks"]
        self.align_len = cfg["align_len"]
        self.data = {}
        # task_data_mapping:
        # "task_id": [(dataset_id, dataset_idx, task_label)]
        self.task_data_mapping = {}
        self.task_sample_counts = {}

        for dataset_id, dataset_cfg in self.datasets.items():
            self.data[dataset_id] = self.get_dataset_items(
                    dataset_cfg["image_folder"],
                    dataset_cfg["label_folder"],
                    dataset_cfg["label_format"]
            )

        for task_id, dataset_ids in self.tasks.items():
            idx_mappings = []
            
            # order of dataset_ids is important
            # maximizing overlap between tasks -> minimizing dataloading
            for dataset_id in dataset_ids:
                for dataset_idx in range(len(self.data[dataset_id])):
                    idx_mappings.append((dataset_id, dataset_idx))

            self.task_data_mapping[task_id] = idx_mappings
            self.task_sample_counts[task_id] = len(idx_mappings)

    def get_dataset_items(self, image_folder: str, label_folder: str, label_format: str):
        """
        Loads a list of tuples containing (image_path, label_path, label_format) correponding to 
        the parameters.

        Parameters
        ----------
        image_folder : str
            The path to the image folder.
        label_folder: str
            The path to the label folder.
        label_format: str
            The expected label format.
        """
        items = []

        for img_path, label_path in zip_matched_files(Path(image_folder),Path(label_folder)):
            items.append((img_path, label_path, label_format))

        return items

    def split_into_task_specific_datasets(self):
        """
        Splits this MTL Dataset into multiple TaskSpecificDatasets.
        
        Returns
        -------
        dict: A mapping task:TaskSpecificDataset
        """
        task_specific = {}

        for task_id in self.tasks.keys():
            task_specific[task_id] = TaskSpecificDataset(self.data, task_id, self.task_data_mapping[task_id])

        return task_specific

    def get_by_global_idx_task_map(self, global_idx):
        """
        Returns a task_id:item map depending on the global_idx.

        Parameters
        ----------
        global_idx: int
            The item idx.
        """
        mapping = {}

        # task_data_mapping:
        # "task_id": [(dataset_id, dataset_idx, task_label)]
        for k, v in self.task_data_mapping.items():
            dataset_id, dataset_idx = v[global_idx % self.task_sample_counts[k]]
            image_path, label_path, label_format = self.data[dataset_id][dataset_idx]
            image = prepare_image(image_path)
            label = read_label(label_path, label_format)
            label = access_label(label, k)
            mapping[k] = (image, label)

        return mapping

    def get_by_global_idx_satpy(self, global_idx: int):
        """
        Returns a list of tuples (image, satpy_labels).
        Warning: satpy_labels can contain None/NaN values, needs to be considered in loss.

        Parameters
        ----------
        global_idx: int
            The item idx.
        """
        images = []
        labels = init_satellitepy_label()
        seen = []

        for k,v in self.task_data_mapping.items():
            dataset_id, dataset_idx = v[global_idx % self.task_sample_counts[k]]
            image_path, label_path, label_format = self.data[dataset_id][dataset_idx]

            image_key = f"{dataset_id}__MTL_DS__{dataset_idx}"

            if image_key in seen:
                continue

            merge_satpy_label_dict(labels, read_label(label_path, label_format))
            images.append(prepare_image(image_path))

        return torch.stack(images, dim=0), torchify_satpy_label_dict(labels)

    def __getitem__(self, idx):
        # int <-> access by global_idx return satpy style (with None values)
        return self.get_by_global_idx_satpy(idx)

    def __len__(self):
        if self.align_len == 'max':
            return max(self.task_sample_counts.values())
        else:
            return min(self.task_sample_counts.values())

