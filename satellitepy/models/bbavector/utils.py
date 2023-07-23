import torch
import logging
import numpy as np
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
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    valid_loss = checkpoint['loss']
    # model.to(device)
    return model, optimizer, epoch, valid_loss

def decode_predictions(predictions, orig_h, orig_w, input_h, input_w, down_ratio):
    predictions = predictions[0, :, :]
    # ori_image = dsets.load_image(dsets.img_ids.index(img_id))
    # h, w, c = ori_image.shape

    # pts0 = {cat: [] for cat in dsets.category}
    # scores0 = {cat: [] for cat in dsets.category}
    points = []
    scores = []
    classes = []

    for pred in predictions:
        cen_pt = np.asarray([pred[0], pred[1]], np.float32)
        tt = np.asarray([pred[2], pred[3]], np.float32)
        rr = np.asarray([pred[4], pred[5]], np.float32)
        bb = np.asarray([pred[6], pred[7]], np.float32)
        ll = np.asarray([pred[8], pred[9]], np.float32)
        tl = tt + ll - cen_pt
        bl = bb + ll - cen_pt
        tr = tt + rr - cen_pt
        br = bb + rr - cen_pt
        score = pred[10]
        clse = pred[11]
        pts = np.asarray([tl, bl,br, tr,], np.float32)
        pts[:, 0] = pts[:, 0] * down_ratio / input_w * orig_w
        pts[:, 1] = pts[:, 1] * down_ratio / input_h * orig_h
        # pts0[dsets.category[int(clse)]].append(pts)
        # scores0[dsets.category[int(clse)]].append(score)
        points.append(pts)
        classes.append(clse)
        scores.append(score)
    # return pts0, scores0
    return points, scores, classes


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

