from satellitepy.models.bbavector.utils import decode_predictions, decode_masks  
from satellitepy.data.bbox import BBox

from mmrotate.core.bbox import rbbox_overlaps

from mmcv.ops import nms_rotated, nms
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_patch_result(
    model,
    model_decoder,
    data_dict,
    device,
    input_h,
    input_w,
    down_ratio):
    """
    Infer the model on a patch
    Parameters
    ----------
    model : satellitepy.models.bbavector.ctrbox_net.CTRBOX
        torch model
    model_decoder : satellitepy.models.bbavector.decoder.DecDecoder
        Decoder part of model
    data_dict : dict
        Item in torch.utils.data.DataLoader generator
    device : str
        cpu or cuda:0
    Returns
    -------
    save_dict : dict
        Results for the patch
    """

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

    return dec_pred

def nms(bboxes_0, bboxes_1, scores, iou_th):
    """
    NMS for rotated and horizontal bounding boxes
    Parameters
    ----------
    bboxes_0 : np.ndarray
        Bounding box parameters. (N,5)
    bboxes_1 : np.ndarray
        Bounding box parameters. (N,5)
    scores : np.ndarray
        Scores to filter out the boxes. For example confidence scores. (N)
    iou_th : float
        IoU threshold.
    Returns
    -------
    inds : np.ndarray
        Indices of bounding boxes after NMS (K).
    """
    if iou_th<=0:
        logger.info("IoU threshold must be larger than 0!")
        return 0
    iou = rbbox_overlaps(torch.FloatTensor(bboxes_0), torch.FloatTensor(bboxes_1)) # ROW: bboxes_0 # COL: bboxes_1
    iou_upper_triangular = np.triu(iou,1) # set the lower triangular to zero
    inds_bboxes = np.nonzero(iou_upper_triangular>iou_th) # Get IOUs that are larger than IOU threshold

    # Get scores for the bboxes that have greater IOU
    scores_bboxes_0 = np.array([scores[ind] for ind in inds_bboxes[0]])
    scores_bboxes_1 = np.array([scores[ind] for ind in inds_bboxes[1]])

    # Get mask for confidence score conditions
    inds_to_remove = np.unique(inds_bboxes[1][scores_bboxes_0 > scores_bboxes_1])
    # Remove the overlapping bboxes (high IOU) with lower confidence scores
    all_inds = np.arange(len(scores))
    mask = np.ones_like(all_inds, dtype=bool)
    mask[inds_to_remove] = False
    inds = all_inds[mask]

    return inds




def apply_nms(det_labels, nms_iou_threshold=0.5):
    """
    Apply nms to labels, e.g., dec_pred, merged_det_labels
    Parameters
    ----------
    det_labels : dict
        Dict with satellitepy keys and satellitepy result keys, e.g., obboxes, hbboxes, confidence-scores, coarse-class
    """

    save_dict = dict()
    if "mask" in det_labels.keys():
        save_dict["mask"] = det_labels["mask"]

    ###########
    bbox_params = [BBox(corners=corners).params for corners in det_labels["obboxes"]]
    conf_scores = det_labels["confidence-scores"]
    # print(f"BBOXES Before : {len(bbox_params)}")
    nms_inds = nms(bbox_params,bbox_params,scores=np.array(det_labels["confidence-scores"]),iou_th=nms_iou_threshold)#[0]
    # bbox_nms_inds = inds_nms[0]
    # print(f"After : {len(nms_inds)}")
    # print(inds_nms)
    det_labels_keys = [key for key in list(det_labels.keys()) if key not in ["mask"]]
    if nms_inds is not None:
        # save_dict["obboxes"] = [BBox(params=params.tolist()).corners for params in bboxes_nms[:,:5]]
        for k in det_labels_keys:
            save_dict[k] = np.asarray(det_labels[k])[nms_inds].tolist()
    else:
        # save_dict["obboxes"] = det_labels["obboxes"]
        for k in det_labels_keys:
            v = det_labels[k]
            if isinstance(v, list):
                save_dict[k] = v
            else:
                save_dict[k] = v.tolist()
    return save_dict

