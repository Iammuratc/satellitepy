import numpy
import torch.nn.functional as F
import torch
import numpy as np

from satellitepy.data.torchify import untorchify_continuous_values


class DecDecoder(object):
    def __init__(self, K, tasks, target_task):
        self.K = K
        assert "obboxes" in tasks or "hbboxes" in tasks, "tasks must contain obboxes and/or hbboxes"
        self.tasks = tasks
        self.target_task = target_task

    def _topk(self, scores):
        batch, cat, height, width = scores.size()
        scores = scores.view(batch, cat, -1)

        own_topk_scores, own_topk_inds = torch.topk(scores[0, :, :].flatten(), self.K)
        own_inds = np.array(np.unravel_index(own_topk_inds.cpu().numpy(), scores.shape[1:])).T
        y = own_inds[:, 1]

        # implementation without numpy, check again with proper predictions
        y2 = own_topk_inds // cat

        topk_scores, topk_inds = torch.topk(scores, self.K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), self.K)
        topk_clses = (topk_ind // self.K).int()
        topk_inds = self._gather_feat( topk_inds.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, self.K)

        full_scores = scores[0, :, y]

        return topk_score, full_scores, topk_inds, topk_clses, topk_ys, topk_xs


    def _nms(self, heat, kernel=3):
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def decode_obboxes(self, box_params, box_offsets, box_theta_cls, heatmap):
        batch, c, height, width = heatmap.size()
        heat = self._nms(heatmap)

        scores, _, inds, clses, ys, xs = self._topk(heat)
        reg = self._tranpose_and_gather_feat(box_offsets, inds)
        reg = reg.view(batch, self.K, 2)
        xs = xs.view(batch, self.K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, self.K, 1) + reg[:, :, 1:2]
        wh = self._tranpose_and_gather_feat(box_params, inds)
        wh = wh.view(batch, self.K, 10)
        # add
        cls_theta = self._tranpose_and_gather_feat(box_theta_cls, inds)
        cls_theta = cls_theta.view(batch, self.K, 1)
        mask = (cls_theta>0.8).float().view(batch, self.K, 1)
        #
        tt_x = (xs+wh[..., 0:1])*mask + (xs)*(1.-mask)
        tt_y = (ys+wh[..., 1:2])*mask + (ys-wh[..., 9:10]/2)*(1.-mask)
        rr_x = (xs+wh[..., 2:3])*mask + (xs+wh[..., 8:9]/2)*(1.-mask)
        rr_y = (ys+wh[..., 3:4])*mask + (ys)*(1.-mask)
        bb_x = (xs+wh[..., 4:5])*mask + (xs)*(1.-mask)
        bb_y = (ys+wh[..., 5:6])*mask + (ys+wh[..., 9:10]/2)*(1.-mask)
        ll_x = (xs+wh[..., 6:7])*mask + (xs-wh[..., 8:9]/2)*(1.-mask)
        ll_y = (ys+wh[..., 7:8])*mask + (ys)*(1.-mask)
        return torch.cat([xs,                        # cen_x
                            ys,                      # cen_y
                            tt_x,
                            tt_y,
                            rr_x,
                            rr_y,
                            bb_x,
                            bb_y,
                            ll_x,
                            ll_y],
                           dim=2)

    def decode_hbboxes(self, box_params, box_offsets, heatmap):
        batch, c, height, width = heatmap.size()
        heat = self._nms(heatmap)

        scores, _, inds, clses, ys, xs = self._topk(heat)
        reg = self._tranpose_and_gather_feat(box_offsets, inds)
        reg = reg.view(batch, self.K, 2)
        xs = xs.view(batch, self.K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, self.K, 1) + reg[:, :, 1:2]
        wh = self._tranpose_and_gather_feat(box_params, inds)
        wh = wh.view(batch, self.K, 2)
        return torch.cat([
            xs,                        # cen_x
            ys,                      # cen_y
            wh # widht, height
        ], dim=2)

    def ctdet_decode(self, pr_decs):
        heat = pr_decs['cls_' + self.target_task]
        scores, all_scores, idx_2d, target, _, _ = self._topk(heat)
        result = {
            self.target_task: all_scores.cpu().numpy().T.tolist()
        }

        if "obboxes" in self.tasks:
            obb_detections = self.decode_obboxes(
                pr_decs["obboxes_params"],
                pr_decs["obboxes_offset"],
                pr_decs["obboxes_theta"],
                heat
            )
            result["obboxes"] = obb_detections.squeeze(0).cpu().numpy()
        if "hbboxes" in self.tasks:
            hbb_detections = self.decode_hbboxes(
                pr_decs["hbboxes_params"],
                pr_decs["hbboxes_offset"],
                heat
            )
            result["hbboxes"] = hbb_detections.squeeze(0).cpu().numpy()
        
        for k, v in pr_decs.items():
            # ignore bounding boxes and coarse class (heatmap)
            if (
                k == "cls_" + self.target_task or
                (k[:3] != "cls" and k[:3] != "reg" and k != "masks")
            ):
                continue

            arr_val = self._tranpose_and_gather_feat(v, idx_2d)

            if k == "masks":
                result[k[:4]] = v.squeeze(0).squeeze(0).cpu().numpy()
            # classification -> we save the confidence scores for each class
            elif k[:3] == "cls":
                result[k[4:]] = arr_val.squeeze(0).cpu().numpy().tolist()
            # regression -> there is only one value, we squeeze
            else:
                det = arr_val.squeeze(0).cpu().numpy()
                result[k[4:]] = untorchify_continuous_values(k, det).tolist()

        return result
