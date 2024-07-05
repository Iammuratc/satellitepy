import cv2
import torch
import logging
import numpy as np

from satellitepy.models.bbavector import ctrbox_net, decoder
from satellitepy.data.utils import get_task_dict
from satellitepy.data.bbox import BBox

logger = logging.getLogger('')


def get_model(tasks, down_ratio):
    heads = {}

    for task in tasks:
        if task == 'obboxes':
            heads['obboxes_params'] = 10
            heads['obboxes_offset'] = 2
            heads['obboxes_theta'] = 1
        elif task == 'hbboxes':
            heads['hbboxes_params'] = 2
            heads['hbboxes_offset'] = 2
        elif task == 'masks':
            heads[task] = 1
        else:
            td = get_task_dict(task)
            if 'max' and 'min' in td.keys():
                heads['reg_' + task] = 1
            else:
                heads['cls_' + task] = len(set(td.values()))

    model = ctrbox_net.CTRBOX(heads=heads,
                              pretrained=True,
                              down_ratio=down_ratio,
                              final_kernel=1,
                              head_conv=256)
    return model


def save_model(path, epoch, model, optimizer):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

<<<<<<< Updated upstream

def load_checkpoint(checkpoint_path, down_ratio, init_lr=1e-3):
    checkpoint = torch.load(checkpoint_path)
    logger.info(f'loaded weights from {checkpoint_path}, epoch {checkpoint["epoch"]}')
=======
def load_checkpoint(checkpoint_path, down_ratio, init_lr=1e-3):
    checkpoint = torch.load(checkpoint_path)
    logger.info('loaded weights from {}, epoch {}'.format(checkpoint_path, checkpoint['epoch']))
>>>>>>> Stashed changes

    keys = checkpoint['model_state_dict'].keys()
    bbox_keys = list(set([key[:7] for key in keys if 'bboxes' in key]))
    cls_keys = list(set([key[4:].split('.')[0] for key in keys if 'cls_' in key]))
    reg_keys = list(set([key[4:].split('.')[0] for key in keys if 'reg_' in key]))
    mask_key = list(set([key.split('.')[0] for key in keys if 'masks' in key]))
    all_keys = bbox_keys + cls_keys + reg_keys + mask_key

<<<<<<< Updated upstream
    model = get_model(all_keys, down_ratio)
=======
    model = get_model(all_keys,down_ratio)
>>>>>>> Stashed changes

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(model.module.parameters(), lr=init_lr)
    else:
        for k, v in model.state_dict().items():
            saved_weights = checkpoint['model_state_dict'][k]
            if v.shape != saved_weights.shape:
                checkpoint['model_state_dict'][k] = saved_weights[:v.shape[0], ...]
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    valid_loss = checkpoint['loss']

    return model, optimizer, epoch, valid_loss


def decode_predictions(predictions, orig_h, orig_w, input_h, input_w, down_ratio):
    if 'obboxes' in predictions:
        predictions['obboxes'] = decode_obboxes(predictions['obboxes'], orig_w, orig_h, input_w, input_h, down_ratio)
    if 'hbboxes' in predictions:
        predictions['hbboxes'] = decode_hbboxes(predictions['hbboxes'], orig_w, orig_h, input_w, input_h, down_ratio)
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
        pts = np.asarray([tl, bl, br, tr, ], np.float32)
        pts[:, 0] = pts[:, 0] * down_ratio / input_w * orig_w
        pts[:, 1] = pts[:, 1] * down_ratio / input_h * orig_h
        points.append(pts.tolist())
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
        tl = cen_pt + np.asarray([-width / 2, -height / 2], np.float32)
        tr = cen_pt + np.asarray([width / 2, -height / 2], np.float32)
        br = cen_pt + np.asarray([width / 2, height / 2], np.float32)
        bl = cen_pt + np.asarray([-width / 2, height / 2], np.float32)
        pts = np.asarray([tl, bl, br, tr, ], np.float32)
        pts[:, 0] = pts[:, 0] * down_ratio / input_w * orig_w
        pts[:, 1] = pts[:, 1] * down_ratio / input_h * orig_h
        points.append(pts.tolist())
    return points


def get_model_decoder(tasks, K, conf_thresh, target_task):
    model_decoder = decoder.DecDecoder(K=K,
                                       conf_thresh=conf_thresh,
                                       tasks=tasks, target_task=target_task)
    return model_decoder
