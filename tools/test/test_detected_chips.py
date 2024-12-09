import argparse
import logging
from pathlib import Path
import cv2 
import numpy as np

import torch
import torchvision
from torch import manual_seed
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

from tqdm import tqdm
import json
from matplotlib import pyplot as plt

from satellitepy.data.utils import get_satellitepy_table, read_img
from satellitepy.data.labels import read_label
from satellitepy.data.chip import create_chip
from satellitepy.models.chips.chip_models import get_model
from satellitepy.utils.path_utils import get_project_folder, init_logger, create_folder, get_file_paths

def parse_args():
    project_folder = get_project_folder()

    parser = argparse.ArgumentParser(description='BBAVectors Implementation in satellitepy')
    parser.add_argument('--backbone', type=str, default='resnet34', help='Name of the backbone to use. Check "satellitepy/models/chips/chip_models.py" for options.')
    parser.add_argument('--chip-model-path', type=Path, help='Model weights path.')
    parser.add_argument('--chip-dataset-params', type=Path, help='mean and std values in .npy file.')
    parser.add_argument('--chip-size', type=int, default=256, help='Chip size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--test-image-folder', type=Path,
                        help='Image folder. The images in this folder will be used to test the model.')
    parser.add_argument('--test-result-folder', type=Path,
                        help='Detection result folder. The detected bounding boxes in this folder will be used to create chips'
                        'followed by being tested by the chip model.')
    # parser.add_argument('--task', type=str, required=True, help='Task to evaluate')
    parser.add_argument('--classes', type=str, default='all', help='If not all, contains all class names that are tested. Must be the same the model is trained on')
    parser.add_argument('--verbose-output', type=bool, default=False, required=False)
    parser.add_argument('--log-config-path', default=project_folder /
                                                     Path('configs/log.config'), type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, required=False, help='Log path.')
    parser.add_argument('--out-folder',
                        type=Path,
                        help='Updated result files will be saved under this folder.')
    parser.add_argument('--mask-background', action='store_true', help='If True, background around bboxes will be masked out.')

    args = parser.parse_args()
    return args

def test_detected_chips(args):
    logger = logging.getLogger('')

    test_image_path = Path(args.test_image_folder)

    out_folder = Path(args.out_folder)
    assert create_folder(out_folder)

    log_path = Path(out_folder) / 'train_chips.log' if args.log_path is None else args.log_path
    init_logger(config_path=args.log_config_path, log_path=log_path)

    # task = args.task

    classes = [c.strip() for c in args.classes.split(",")]
    logger.info(f'Testing on classes: {classes}')

    logger.info('Initiating testing')

    test_batch_size = 1

    num_workers = args.num_workers
    # Model
    checkpoint = torch.load(args.chip_model_path)
    model = get_model(args.backbone, len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Inference model
    inference_chips_from_detected_results(
        model,
        in_result_folder=args.test_result_folder, 
        img_folder=args.test_image_folder,
        chip_size=args.chip_size,
        mask_background=args.mask_background,
        chip_dataset_params=args.chip_dataset_params,
        device=device,
        classes=classes,
        out_folder=out_folder,
        logger=logger)

def inference_chips_from_detected_results(
    model,
    in_result_folder, 
    img_folder,
    chip_size,
    mask_background,
    chip_dataset_params,
    device,
    classes,
    out_folder,
    logger):

    # from satellitepy.evaluate.utils import match_gt_and_det_bboxes
    img_paths = get_file_paths(img_folder)
    result_paths = get_file_paths(in_result_folder)

    model.eval()

    # Define the pad size to ensure that all chips have a fixed size
    pad_size = int(chip_size)
    pad_width = ((pad_size, pad_size), (pad_size, pad_size), (0, 0)) # y,x,ch

    # Define dataset params
    mean, std = np.load(chip_dataset_params)
    defaults = v2.Compose([v2.ToImageTensor(), v2.ToDtype(torch.float32), v2.Normalize(mean, std)])

    with torch.no_grad():
        for img_path, result_path in zip(img_paths, result_paths):
            logger.info(f"Reading {img_path.stem}...")
            img = read_img(str(img_path))
            img = np.pad(img, pad_width, mode='constant', constant_values=0)

            result = read_label(result_path,label_format='satellitepy')
            det_bboxes = result['det_labels']['obboxes']
            
            # Create conf score and det_labels['fineair-class] in result
            result['det_labels']['fac-confidence-scores'] = []
            result['det_labels']['fineair-class'] = []
            # det_conf = result['det_labels']['obboxes']
            det_bboxes = np.array(det_bboxes) + np.array([pad_width[1][0], pad_width[0][0]])
            ########
            # gt_bboxes = np.array(result['gt_labels']['obboxes']) + np.array([pad_width[1][0], pad_width[0][0]])
            # print({'obboxes':gt_bboxes})
            # matches = match_gt_and_det_bboxes(gt_labels={'obboxes':gt_bboxes},det_labels={'obboxes':det_bboxes})
            # print(matches)
            # break
            ##########
            instance_dict = get_satellitepy_table()['fineair-class']


            for i, det_bbox in enumerate(det_bboxes):
                chip_img, center = create_chip(img=img,
                    bbox=np.array(det_bbox).astype(int), 
                    chip_size=chip_size, 
                    bbox_mask_margin=0.2, # %10 on both sides
                    draw_corners=False,
                    orient_objects=False,
                    mask_background=mask_background)
                # cv2.imwrite(f"/home/murat/Projects/satellitepy/data/fineair/experiments/fac_from_detections/chips/{img_path.stem}_{i}.png",chip_img)
                chip_img = defaults(chip_img)
                chip_img = chip_img.unsqueeze(0)  # Add batch dimension
                # print(chip_img.shape)

                # Inference the model
                y_hat = model(chip_img.to(device))
                pred_int = torch.argmax(y_hat, dim=1).cpu().numpy()
                det_conf_score = float(torch.max(y_hat, dim=1).values.cpu().numpy())#.tolist()
                det_class_name = classes[int(pred_int)]
                det_class_ind = instance_dict[det_class_name]
                result['det_labels']['fac-confidence-scores'].append(det_conf_score)
                result['det_labels']['fineair-class'].append(det_class_ind)
            
            out_result_path = str(out_folder/result_path.name)
            logger.info(f"Writing new results to {out_result_path}...")
            with open(out_result_path, 'w') as f:
                json.dump(result,f,indent=4)
            # break







if __name__ == '__main__':
    args = parse_args()
    test_detected_chips(args)