import cv2

from satellitepy.models.bbavector.tools import get_model, get_model_decoder
from satellitepy.models.bbavector.utils import load_checkpoint, decode_predictions #, collater
from satellitepy.dataset.bbavector.dataset_bbavector import BBAVectorDataset
from satellitepy.data.utils import get_task_dict
from satellitepy.data.tools import read_label
import satellitepy.models.bbavector.loss as loss_utils
from satellitepy.utils.path_utils import create_folder
from satellitepy.evaluate.utils import match_gt_and_det_bboxes #, nms_rotated
from satellitepy.data.bbox import BBox
from torch.utils.data import DataLoader

from tqdm import tqdm
import torch
import logging
from pathlib import Path
from mmcv.ops import nms_rotated, nms
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
    model = get_model(tasks,down_ratio)
    model, optimizer, epoch, valid_loss = load_checkpoint(model, checkpoint_path)
    model.to(device)
    model.eval()

    model_decoder = get_model_decoder(tasks,
    K,
    conf_thresh)

    # Dataset
    dataset = BBAVectorDataset(
        in_image_folder,
        in_label_folder,
        in_label_format,
        tasks,
        input_h,
        input_w,
        down_ratio,
        False)
    # Dataloader
    data_loader = DataLoader(dataset,
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
            pred = model(data_dict['input'].to(device).float())
        
        predictions = model_decoder.ctdet_decode(pred)
        dec_pred = decode_predictions(
            predictions = predictions, 
            orig_h = data_dict['img_h'].numpy(),
            orig_w = data_dict['img_w'].numpy(),
            input_h = input_h, 
            input_w = input_w, 
            down_ratio = down_ratio
        )

        save_dict = dict()

        if "mask" in dec_pred:
            mask_folder = Path(out_folder)/ 'results' / 'masks'
            assert create_folder(mask_folder)
            im = cv2.normalize(dec_pred["mask"], None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(str(Path(mask_folder) / f"{img_name}.jpg"), im)

        if "obboxes" in dec_pred:
            bboxes_nms, keep_ind = nms_rotated(
                 dets=torch.Tensor([BBox(corners=corners).params for corners in dec_pred["obboxes"]]),
                 scores=torch.Tensor(dec_pred["confidence-scores"]),
                 iou_threshold= 0.5,
                 labels=torch.Tensor(dec_pred["coarse-class"])
            )
            if keep_ind is not None:
                save_dict["obboxes"] = [BBox(params=params.tolist()).corners for params in bboxes_nms[:,:5]]
                for k, v in dec_pred.items():
                    if k != "obboxes":
                        save_dict[k] = np.asarray(v)[keep_ind.cpu().numpy()].tolist()
            else:
                save_dict["obboxes"] = dec_pred["obboxes"]
                for k, v in dec_pred.items():
                    if k != "obboxes":
                        if isinstance(v, list):
                            save_dict[k] = v
                        else:
                            save_dict[k] = v.tolist()

        # if obboxes is not in dec_pred, hbboxes must be in dec_pred
        else:
            bboxes_nms, keep_ind = nms(
                 boxes=torch.Tensor(dec_pred["hbboxes"]),
                 scores=torch.Tensor(dec_pred["confidence-scores"]),
                 iou_threshold= 0.5
            )
            if keep_ind is not None:
                save_dict["hbboxes"] = [BBox(params=params.tolist()).corners for params in bboxes_nms[:,:5]]
                for k, v in dec_pred.items():
                    if k != "hbboxes":
                        save_dict[k] = np.asarray(v)[keep_ind].tolist()
            else:
                save_dict["hbboxes"] = dec_pred["hbboxes"]
                for k, v in dec_pred.items():
                    if k != "hbboxes":
                        if isinstance(v, list):
                            save_dict[k] = v
                        else:
                            save_dict[k] = v.tolist()

        if in_label_folder:
            gt_labels = read_label(data_dict['label_path'][0],in_label_format)
            matches = match_gt_and_det_bboxes(gt_labels,save_dict)
            save_dict["gt_labels"] = gt_labels
            save_dict["matches"] = matches

        # Save results with the corresponding ground truth
        # # Save labels to json file
        with open(Path(patch_result_folder) / f"{img_name}.json",'w') as f:
            json.dump(save_dict, f, indent=4)
