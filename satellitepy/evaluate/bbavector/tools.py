from satellitepy.models.bbavector.tools import get_model, get_model_decoder
from satellitepy.models.bbavector.utils import load_checkpoint, decode_predictions #, collater
from satellitepy.dataset.bbavector.dataset_bbavector import BBAVectorDataset
from satellitepy.data.utils import get_task_dict
from satellitepy.data.tools import read_label
import satellitepy.models.bbavector.loss as loss_utils
from satellitepy.utils.path_utils import create_folder
from satellitepy.evaluate.utils import match_gt_and_det_bboxes #, nms_rotated
from satellitepy.data.bbox import BBox

from tqdm import tqdm
import torch
import logging
from pathlib import Path
from mmcv.ops import nms_rotated
import json 
import numpy as np


def save_patch_results(
    out_folder,
    in_image_folder,
    in_label_folder,
    in_label_format,
    checkpoint_path,
    device,
    task,
    num_workers,
    input_h,
    input_w,
    conf_thresh,
    down_ratio,
    K,
    # nms_on_multiclass_thr
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
    logger = logging.getLogger(__name__)
    # Model
    model = get_model(task,down_ratio)
    model, optimizer, epoch, valid_loss = load_checkpoint(model, checkpoint_path)
    model.to(device)
    model.eval()

    model_decoder = get_model_decoder(task,
    K,
    conf_thresh)


    # Task dict
    task_dict = get_task_dict(task)
    num_classes = len(task_dict)

    # Dataset
    dataset = BBAVectorDataset(
        in_image_folder,
        in_label_folder,
        in_label_format,
        task,
        task_dict,
        input_h,
        input_w,
        down_ratio)
    # Dataloader
    data_loader = torch.utils.data.DataLoader(dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,)
        # collate_fn=collater)


    # Create result patch folder
    patch_result_folder = Path(out_folder) / 'results' / 'patch_labels'
    assert create_folder(patch_result_folder)

    # criterion = loss_utils.LossAll()

    for data_dict in tqdm(data_loader):
        img_name = Path(data_dict['img_path'][0]).stem
        with torch.no_grad():
            pred = model(data_dict['input'].to(device))
        
        predictions = model_decoder.ctdet_decode(pred)
        bboxes, scores, classes = decode_predictions(
            predictions = predictions, 
            orig_h = data_dict['img_h'].numpy(), 
            orig_w = data_dict['img_w'].numpy(),
            input_h = input_h, 
            input_w = input_w, 
            down_ratio = down_ratio)


        bboxes_nms, keep_ind = nms_rotated(
            # dets=torch.from_numpy(result_squeezed[:,:5]),
            # scores=torch.from_numpy(result_squeezed[:,-1]),
            # iou_threshold= nms_iou_thr,
            # labels=torch.from_numpy(labels_squeezed))
            dets=torch.Tensor([BBox(corners=corners).params for corners in bboxes]),
            scores=torch.Tensor(scores),
            iou_threshold= 0.5,
            labels=torch.Tensor(classes))

        if keep_ind is not None:
            classes_nms = [classes[i] for i in keep_ind]
            scores_nms = [scores[i].astype(float) for i in keep_ind]

            det_labels = {
                task:[key for pred_class in classes_nms for key,value in task_dict.items() if value==pred_class],
                'obboxes':[BBox(params=params.tolist()).corners for params in bboxes_nms[:,:5]],
                'confidence_scores':scores_nms
            }

        else:
            det_labels = {
                task:[key for pred_class in classes for key,value in task_dict.items() if value==pred_class],
                'obboxes':bboxes,
                'confidence_scores':scores
            }


        gt_labels = read_label(data_dict['label_path'][0],in_label_format)

        matches = match_gt_and_det_bboxes(gt_labels,det_labels)

        result = {
            'gt_labels':gt_labels,
            'det_labels':det_labels,
            'matches':matches
                    }

        # Save results with the corresponding ground truth
        # # Save labels to json file
        with open(Path(patch_result_folder) / f"{img_name}.json",'w') as f:
            json.dump(result, f, indent=4)