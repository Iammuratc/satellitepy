import torch
import torch.nn as nn
import torch.nn.functional as F

from satellitepy.data.utils import get_task_dict

EPSILON = 1e-9

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

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

    def forward(self, output, mask, ind, target):
        # torch.Size([1, 1, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 1])
        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 1])
        if mask.sum():
            nan_mask = torch.isnan(target) == False
            _mask = mask & nan_mask
            loss = F.cross_entropy(pred[_mask],
                                          target[_mask].long(),
                                          reduction='mean')
            if torch.any(torch.isnan(loss)):
                return 0.
            return loss
        else:
            return 0.
class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

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

    def forward(self, output, mask, ind, target):
        # torch.Size([1, 1, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 1])
        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 1])
        if mask.sum():
            nan_mask = torch.isnan(target) == False
            _mask = mask & nan_mask.squeeze(-1)
            loss = F.binary_cross_entropy(pred[_mask],
                                          target[_mask],
                                          reduction='mean')
            if torch.any(torch.isnan(loss)):
                return 0.
            return loss
        else:
            return 0.

class OffSmoothL1Loss(nn.Module):
    def __init__(self):
        super(OffSmoothL1Loss, self).__init__()

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

    def forward(self, output, mask, ind, target):
        # torch.Size([1, 2, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 2])
        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 2])
        if mask.sum():
            nan_mask = torch.isnan(target) == False
            if len(nan_mask.shape) > 2:
                # if gt is none, than all regression targets will be None
                nan_mask = nan_mask[:,:,0]
    
            _mask = mask & nan_mask
            loss = F.smooth_l1_loss(pred[_mask],
                                    target[_mask],
                                    reduction='mean')
            if torch.any(torch.isnan(loss)):
                return 0.
            return loss
        else:
            return 0.

class FocalLoss(nn.Module):
  def __init__(self):
    super(FocalLoss, self).__init__()

  def forward(self, pred, gt):
      pos_inds = gt.eq(1).float()
      neg_inds = gt.lt(1).float()
      # Simon 18/07/23: prevent log(0)
      pred = torch.clamp(pred, min=EPSILON, max=1-EPSILON)

      neg_weights = torch.pow(1 - gt, 4)

      loss = 0

      pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
      neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

      num_pos  = pos_inds.float().sum()
      pos_loss = pos_loss.sum()
      neg_loss = neg_loss.sum()

      if num_pos == 0:
        loss = loss - neg_loss
      else:
        loss = loss - (pos_loss + neg_loss) / num_pos
      return loss

def isnan(x):
    return x != x
  
class LossAll(torch.nn.Module):
    def __init__(self, tasks):
        super(LossAll, self).__init__()
        #self.L_hm = FocalLoss()
        #self.L_wh =  OffSmoothL1Loss()
        #self.L_off = OffSmoothL1Loss()
        #self.L_cls_theta = BCELoss()
        self.tasks_losses = nn.ModuleDict()
        for t in tasks:
            if t == "masks": 
                self.tasks_losses[t] = BCELoss()
            elif t == "obboxes":
                self.tasks_losses[t + "_params"] = OffSmoothL1Loss()
                self.tasks_losses[t + "_offset"] = OffSmoothL1Loss()
                self.tasks_losses[t + "_theta"] = BCELoss()
            elif t == "hbboxes":
                self.tasks_losses[t + "_params"] = OffSmoothL1Loss()
                self.tasks_losses[t + "_offset"] = OffSmoothL1Loss()
            elif t == "coarse-class": # this is the heatmap loss
                self.tasks_losses["cls_" + t] = FocalLoss() 
            else:
                td = get_task_dict(t)
                if 'max' in td.keys() and 'min' in td.keys():
                    self.tasks_losses["reg_" + t] = OffSmoothL1Loss()
                else:
                    self.tasks_losses["cls_" + t] = CELoss()

    def forward(self, pr_decs, gt_batch):
        #hm_loss  = self.L_hm(pr_decs['hm'], gt_batch['hm'])
        #wh_loss  = self.L_wh(pr_decs['reg_wh'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['wh'])
        #off_loss = self.L_off(pr_decs['reg'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['reg'])
        ## add
        #cls_theta_loss = self.L_cls_theta(pr_decs['cls_theta'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['cls_theta'])

        loss_dict = dict()

        for task, loss_fnc in self.tasks_losses.items():
            if task == "cls_coarse-class": 
                loss_dict[task] = loss_fnc(pr_decs[task], gt_batch[task])
            else:
                loss_dict[task] = loss_fnc(
                    pr_decs[task], 
                    gt_batch["reg_mask"], 
                    gt_batch["ind"], 
                    gt_batch[task]
                )

        return loss_dict
