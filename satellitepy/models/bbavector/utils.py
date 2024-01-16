import cv2
import torch
import logging
import numpy as np

from satellitepy.data.bbox import BBox

logger = logging.getLogger(__name__)

def save_model(path, epoch, model, optimizer):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        # 'loss': loss
    }, path)

def load_checkpoint(model, checkpoint_path, init_lr=1e-3):
    checkpoint = torch.load(checkpoint_path)
    logger.info('loaded weights from {}, epoch {}'.format(checkpoint_path, checkpoint['epoch']))

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(model.module.parameters(), lr=init_lr)
    else:
        # we crop the weight sizes in first dimension if necessary
        for k, v in model.state_dict().items():
            saved_weights = checkpoint["model_state_dict"][k]
            if v.shape != saved_weights.shape:
                checkpoint["model_state_dict"][k] = saved_weights[:v.shape[0], ...]
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    valid_loss = checkpoint['loss']
    # model.to(device)
    return model, optimizer, epoch, valid_loss

def decode_predictions(predictions, orig_h, orig_w, input_h, input_w, down_ratio):
    if "obboxes" in predictions:
        predictions["obboxes"] = decode_obboxes(predictions["obboxes"], orig_w, orig_h, input_w, input_h, down_ratio)
    if "hbboxes" in predictions:
        predictions["hbboxes"] = decode_hbboxes(predictions["hbboxes"], orig_w, orig_h, input_w, input_h, down_ratio)
    return predictions

def decode_obboxes(boxes, orig_w, orig_h, input_w, input_h, down_ratio):
    points = []
    for box in boxes:
        cen_pt = np.asarray([box[0], box[1]], np.float32)
        tt = np.asarray([box[2], box[3]], np.float32)
        rr = np.asarray([box[4], box[5]], np.float32)
        bb = np.asarray([box[6], box[7]], np.float32)
        ll = np.asarray([box[8], box[9]], np.float32)
        tl = tt + ll - cen_pt
        bl = bb + ll - cen_pt
        tr = tt + rr - cen_pt
        br = bb + rr - cen_pt
        pts = np.asarray([tl, bl,br, tr,], np.float32)
        pts[:, 0] = pts[:, 0] * down_ratio / input_w * orig_w
        pts[:, 1] = pts[:, 1] * down_ratio / input_h * orig_h
        points.append(pts)
    return points

def decode_masks(bbox, mask):
    h = BBox.get_bbox_limits(np.array(bbox))
    mask_0 = np.zeros((mask.shape[0], mask.shape[1]))
    cv2.fillPoly(mask_0, [np.array(bbox, dtype=int)], 1)

    coords = np.argwhere((mask_0[h[2]:h[3], h[0]:h[1]] == 1) & (mask[h[2]:h[3], h[0]:h[1]] == 255)).T.tolist()
    mask_res = [(coords[1] + h[0]).tolist(), (coords[0] + h[2]).tolist()]

    return mask_res


def decode_hbboxes(boxes, orig_w, orig_h, input_w, input_h, down_ratio):
    points = []
    for box in boxes:
        cen_pt = np.asarray([box[0], box[1]], np.float32)
        width, height = box[2], box[3]
        tl = cen_pt + np.asarray([-width/2, -height/2], np.float32)
        tr = cen_pt + np.asarray([width/2, -height/2], np.float32)
        br = cen_pt + np.asarray([width/2, height/2], np.float32)
        bl = cen_pt + np.asarray([-width/2, height/2], np.float32)
        pts = np.asarray([tl, bl,br, tr,], np.float32)
        pts[:, 0] = pts[:, 0] * down_ratio / input_w * orig_w
        pts[:, 1] = pts[:, 1] * down_ratio / input_h * orig_h
        points.append(pts)
    return points

# def collater(data):
#     out_data_dict = {}
#     for name in data[0]:
#         out_data_dict[name] = []
#     for sample in data:
#         for name in sample:
#             out_data_dict[name].append(torch.from_numpy(sample[name]))
#     for name in out_data_dict:
#         out_data_dict[name] = torch.stack(out_data_dict[name], dim=0)
#     return out_data_dict

