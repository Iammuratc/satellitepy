from satellitepy.models.bbavector.utils import decode_predictions, decode_masks  
from satellitepy.data.bbox import BBox

from mmcv.ops import nms_rotated, nms
import torch
import numpy as np

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

    save_dict = dict()
    mask = None

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
                if k == "mask":
                    mask = v
                elif k != "obboxes":
                    save_dict[k] = np.asarray(v)[keep_ind.cpu().numpy()].tolist()
        else:
            save_dict["obboxes"] = dec_pred["obboxes"]
            for k, v in dec_pred.items():
                if k == "mask":
                    mask = v
                elif k != "obboxes":
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
                if k == "mask":
                    mask = v
                elif k != "hbboxes":
                    mask = v
        else:
            save_dict["hbboxes"] = dec_pred["hbboxes"]
            for k, v in dec_pred.items():
                if k != "hbboxes":
                    if k == "mask":
                        mask = v
                    elif isinstance(v, list):
                        save_dict[k] = v
                    else:
                        save_dict[k] = v.tolist()
    return save_dict, mask
