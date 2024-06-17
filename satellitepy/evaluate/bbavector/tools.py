import cv2

from satellitepy.models.bbavector.utils import load_checkpoint, get_model_decoder
from satellitepy.dataset.bbavector.dataset_bbavector import BBAVectorDataset
from satellitepy.dataset.bbavector.utils import Utils as BBAVectorDatasetUtils
from satellitepy.data.utils import read_img
from satellitepy.data.tools import read_label
from satellitepy.data.patch import get_patches, merge_patch_results
from satellitepy.utils.path_utils import create_folder, get_file_paths
from satellitepy.evaluate.utils import match_gt_and_det_bboxes
from satellitepy.evaluate.bbavector.utils import get_patch_result

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import logging
from pathlib import Path
import json
import numpy as np


def save_patch_results(
        out_folder,
        in_image_folder,
        in_label_folder,
        in_label_format,
        checkpoint_path,
        device,
        tasks,
        target_task,
        num_workers,
        input_h,
        input_w,
        conf_thresh,
        down_ratio,
        K
):
    """
    Pass patch images to a bbavector model and save the detected bounding boxes as json files in satellitepy format
    Parameters
    ----------
    out_folder : Path
        Results will be saved here
    in_image_folder : Path
        Test image folder
    in_label_folder : Path
        Test label folder
    in_label_format : str
        Test label file format (e.g., dota, fair1m)
    checkpoint_path : Path
        BBAVector model weights path
    device : str
        cpu or cuda:0
    task : str
        Task name.
    Returns
    -------
    None
    """
    logger = logging.getLogger('')

    model, optimizer, epoch, valid_loss = load_checkpoint(checkpoint_path, down_ratio)
    model.to(device)
    model.eval()

    model_decoder = get_model_decoder(tasks,
                                      K,
                                      conf_thresh, target_task)

    dataset = BBAVectorDataset(
        in_image_folder=in_image_folder,
        in_label_folder=in_label_folder,
        in_label_format=in_label_format,
        tasks=tasks,
        input_h=input_h,
        input_w=input_w,
        down_ratio=down_ratio,
        augmentation=False,
        validate_dataset=False,
    )

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True)

    patch_result_folder = Path(out_folder) / 'results' / 'predictions'
    assert create_folder(patch_result_folder)

    if 'masks' in tasks:
        patch_mask_folder = Path(out_folder) / 'results' / 'masks'
        assert create_folder(patch_mask_folder)

    for data_dict in tqdm(data_loader):
        img_name = Path(data_dict['img_path'][0]).stem

        save_dict = get_patch_result(
            model,
            model_decoder,
            data_dict,
            device,
            input_h,
            input_w,
            down_ratio
        )

        if in_label_folder:
            gt_labels = read_label(data_dict['label_path'][0], in_label_format)
            matches = match_gt_and_det_bboxes(gt_labels, save_dict)
            save_dict['gt_labels'] = gt_labels
            save_dict['matches'] = matches

        if 'masks' in tasks:
            mask = save_dict['masks']
            path = str(patch_mask_folder.joinpath(f'{img_name}.png'))
            assert np.max(mask) <= 1.0, 'mask value > 1.0!'
            cv2.imwrite(path, mask * 255.0, [cv2.IMWRITE_JPEG_QUALITY, 100])
            del save_dict['masks']

        with open(Path(patch_result_folder) / f'{img_name}.json', 'w') as f:
            json.dump(save_dict, f, indent=4)


def save_original_image_results(
        out_folder,
        in_image_folder,
        in_label_folder,
        in_mask_folder,
        in_label_format,
        checkpoint_path,
        truncated_object_threshold,
        patch_size,
        patch_overlap,
        device,
        tasks,
        num_workers,
        input_h,
        input_w,
        conf_thresh,
        down_ratio,
        K,
        img_read_module='cv2',
        target_task='coarse-class'
):
    logger = logging.getLogger('')

    result_folder = Path(out_folder) / 'results' / 'predictions'
    assert create_folder(result_folder)

    if 'masks' in tasks:
        mask_folder = Path(out_folder) / 'results' / 'masks'
        assert create_folder(mask_folder)

    model, optimizer, epoch, valid_loss = load_checkpoint(checkpoint_path, down_ratio)
    model.to(device)
    model.eval()

    model_decoder = get_model_decoder(tasks,
                                      K,
                                      conf_thresh,
                                      target_task)

    img_paths = get_file_paths(in_image_folder)
    label_paths = get_file_paths(in_label_folder) if in_label_folder is not None else len(img_paths) * [None]
    mask_paths = get_file_paths(in_mask_folder) if in_mask_folder is not None else len(img_paths) * [None]
    try:
        assert len(img_paths) == len(label_paths) == len(mask_paths)
    except AssertionError:
        logger.error('The number of files does not match.')
        logger.error(
            f'There are {len(img_paths)} images, {len(label_paths)} label files and {len(mask_paths)} mask images.')
        return 0

    bbavector_dataset_utils = BBAVectorDatasetUtils(tasks=tasks,
                                                    input_h=input_h,
                                                    input_w=input_w,
                                                    down_ratio=down_ratio,
                                                    K=K,
                                                    augmentation=False)

    for img_path, label_path, mask_path in tqdm(zip(img_paths, label_paths, mask_paths), total=len(img_paths)):

        img_name = img_path.stem
        img = read_img(img_path=str(img_path), module=img_read_module)

        gt_labels = read_label(label_path, in_label_format, mask_path)

        patch_dict = get_patches(
            img=img,
            gt_labels=gt_labels,
            truncated_object_thr=truncated_object_threshold,
            patch_size=patch_size,
            patch_overlap=patch_overlap
        )
        patch_dict['det_labels'] = []
        patch_dict['masks'] = []

        for patch_img, patch_labels in zip(patch_dict['images'], patch_dict['labels']):
            image_h, image_w, c = patch_img.shape
            annotation = bbavector_dataset_utils.prepare_annotations(patch_labels, image_w, image_h)
            patch_img, annotation = bbavector_dataset_utils.data_transform(patch_img, annotation,
                                                                           bbavector_dataset_utils.augmentation)
            data_dict = bbavector_dataset_utils.generate_ground_truth(patch_img, annotation, target_task)
            data_dict['input'] = torch.Tensor(data_dict['input']).unsqueeze(0)
            data_dict['img_w'] = torch.from_numpy(np.array(image_w)).unsqueeze(0)
            data_dict['img_h'] = torch.from_numpy(np.array(image_h)).unsqueeze(0)
            save_dict = get_patch_result(
                model,
                model_decoder,
                data_dict,
                device,
                input_h,
                input_w,
                down_ratio
            )

            if 'masks' in tasks:
                patch_dict['masks'].append(save_dict['masks'])
            if 'masks' in save_dict.keys():
                del save_dict['masks']
            patch_dict['det_labels'].append(save_dict)

        merged_det_labels, mask = merge_patch_results(patch_dict, patch_size, img.shape)

        # matches = match_gt_and_det_bboxes(gt_labels, merged_det_labels)

        result = {
            'gt_labels': gt_labels,
            'det_labels': merged_det_labels,
            # 'matches': matches
        }

        json_path = Path(result_folder) / f'{img_name}.json'
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=4)

        if 'masks' in tasks and mask.any:
            path = str(mask_folder.joinpath(f'{img_name}.png'))
            max = np.max(mask)
            mask *= 255.0
            assert max <= 1.0, 'mask value > 1.0!'
            cv2.imwrite(path, mask, [cv2.IMWRITE_JPEG_QUALITY, 100])
