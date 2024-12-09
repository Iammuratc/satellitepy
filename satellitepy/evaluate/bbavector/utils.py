from satellitepy.models.bbavector.utils import decode_predictions
from satellitepy.data.bbox import BBox

from mmrotate.core.bbox import rbbox_overlaps

import torch
import numpy as np
import logging

logger = logging.getLogger('')


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
        predictions=predictions,
        orig_h=data_dict['img_h'].numpy(),
        orig_w=data_dict['img_w'].numpy(),
        input_h=input_h,
        input_w=input_w,
        down_ratio=down_ratio
    )

    return dec_pred


def nms(bboxes, scores, iou_th):
    # pass
    """
    NMS for rotated and horizontal bounding boxes
    Parameters
    ----------
    bboxes: np.ndarray
        Bounding box parameters. (N,5)
    scores (torch.Tensor)
        Confidence scores for each bounding box/polygon, of shape (N,).
    iou_threshold (float)
        IoU threshold for NMS.
    Returns
    -------
    torch.Tensor: Indices of the bounding boxes/polygons to keep.
    """
    if iou_th <= 0:
        logger.info('IoU threshold must be larger than 0!')
        return 0
    iou_matrix = rbbox_overlaps(torch.FloatTensor(bboxes), torch.FloatTensor(bboxes))
    assert iou_matrix.shape[0] == scores.shape[0]
    
    # Get the indices of the boxes sorted by scores (highest to lowest)
    _, indices = scores.sort(descending=True)
    
    keep_indices = []
    
    while indices.numel() > 0:
        # Select the index of the current best box
        current = indices[0].item()
        keep_indices.append(current)
    
        if indices.numel() == 1:
            break
    
        # Compute IoU of the selected box with the rest
        current_iou = iou_matrix[current, indices[1:]]
    
        # Keep indices where IoU is below the threshold
        remaining_indices = indices[1:][current_iou <= iou_th]
    
        # Update the indices
        indices = remaining_indices
    
    # return torch.tensor(keep_indices, dtype=torch.long)
    return keep_indices


def apply_nms(det_labels, nms_iou_threshold=0.5, target_task="coarse-class", no_probability=False):
    # pass
    """
    Apply nms to labels, e.g., dec_pred, merged_det_labels
    Parameters
    ----------
    det_labels : dict
        Dict with satellitepy keys and satellitepy result keys, e.g., obboxes, hbboxes, confidence-scores, coarse-class
    """
    #
    save_dict = dict()
    
    bbox_params = [BBox(corners=corners).params for corners in det_labels['obboxes']]
    # bbox_params = [BBox(corners=corners.astype(np.float32)).get_params_cv2() for corners in det_labels['obboxes']]
    if no_probability:
        conf_scores = det_labels['confidence-scores']
    else:
        conf_scores = np.max(det_labels[target_task], axis=1) if len(det_labels[target_task]) > 0 else []
    nms_inds = nms(bbox_params, scores=torch.Tensor(conf_scores), iou_th=nms_iou_threshold)  # [0]
    
    det_labels_keys = [key for key in list(det_labels.keys()) if key not in ['masks']]
    if nms_inds is not None:
        for k in det_labels_keys:
            save_dict[k] = np.asarray(det_labels[k])[nms_inds].tolist()
    else:
        for k in det_labels_keys:
            v = det_labels[k]
            if isinstance(v, list):
                save_dict[k] = v
            else:
                save_dict[k] = v.tolist()
    return save_dict
