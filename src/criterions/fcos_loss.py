import mlconfig
import math
from functools import partial

import torch
import torch.nn as nn


@mlconfig.register
class Fcos_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def _reshape_cat_out(preds):
        preds_reshape = [torch.reshape(pred.permute(0, 2, 3, 1), [pred.shape[0], -1, pred.shape[1]]) for pred in preds]
        return torch.cat(preds_reshape, dim=1)
    
    def forward(self, cls_logits, reg_preds, cnt_logits, cls_targets, cnt_targets, reg_targets):
        mask_pos = cnt_targets[:, :, -1]!=0
        cls_loss = self.compute_cls_loss(cls_logits, cls_targets[:, :, :-1], mask_pos)
        cnt_loss = self.compute_cnt_loss(cnt_logits, cnt_targets[:, :, :-1], mask_pos)
        reg_loss = self.compute_reg_loss(reg_preds, reg_targets[:, :, :-1], mask_pos)

        total_loss = cls_loss + cnt_loss + reg_loss
        return total_loss
    
    def compute_cls_loss(self, preds, targets, mask, gamma=2.0, alpha=0.25):
        #--------------------#
        #   计算batch_size
        #   计算种类数量
        #--------------------#
        batch_size      = targets.shape[0]
        num_classes     = preds[0].shape[1]
        
#         mask            = mask.unsqueeze(dim = -1)
        #--------------------#
        #   计算正样本数量
        #--------------------#
        num_pos         = torch.sum(mask, dim = 1).clamp_(min = 1).float()
#         num_pos         = torch.sum(mask, dim = [1, 2]).clamp_(min = 1).float()
        preds = self._reshape_cat_out(preds)
        assert preds.shape[:2]==targets.shape[:2]
        
        #--------------------#
        #   对计算损失
        #--------------------#
        loss = 0
        for batch_index in range(batch_size):
            pred_pos    = torch.sigmoid(preds[batch_index])
            target_pos  = targets[batch_index]
            #--------------------#
            #   生成one_hot标签
            #--------------------#
            target_pos  = (torch.arange(0, num_classes, device=target_pos.device)[None,:] == target_pos).float()
            
            #--------------------#
            #   计算focal_loss
            #--------------------#
            pt          = pred_pos * target_pos + (1.0 - pred_pos) * (1.0 - target_pos)
            w           = alpha * target_pos + (1.0 - alpha) * (1.0 - target_pos)
            batch_loss  = -w * torch.pow((1.0 - pt), gamma) * pt.log()
            batch_loss  = batch_loss.sum()
            loss += batch_loss
            
        return loss / torch.sum(num_pos)

    def compute_cnt_loss(self, preds, targets, mask):
        #------------------------#
        #   计算batch_size
        #   计算center长度（1）
        #------------------------#
        batch_size  = targets.shape[0]
        c           = targets.shape[-1]
        
#         mask            = mask.unsqueeze(dim = -1)
        #--------------------#
        #   计算正样本数量
        #--------------------#
        num_pos         = torch.sum(mask, dim = 1).clamp_(min = 1).float()
#         num_pos         = torch.sum(mask, dim = [1, 2]).clamp_(min = 1).float()
        preds = self._reshape_cat_out(preds)
        assert preds.shape==targets.shape
        
        #--------------------#
        #   对计算损失
        #--------------------#
        loss = 0
        for batch_index in range(batch_size):
            pred_pos    = preds[batch_index][mask[batch_index]]
            target_pos  = targets[batch_index][mask[batch_index]]
            batch_loss  = nn.functional.binary_cross_entropy_with_logits(input=pred_pos,target=target_pos,reduction='sum').view(1)
            loss += batch_loss
            
        return torch.sum(loss, dim=0) / torch.sum(num_pos)

    def giou_loss(self, preds, targets):
        #------------------------#
        #   左上角和右下角
        #------------------------#
        lt_min  = torch.min(preds[:, :2], targets[:, :2])
        rb_min  = torch.min(preds[:, 2:], targets[:, 2:])
        #------------------------#
        #   重合面积计算
        #------------------------#
        wh_min  = (rb_min + lt_min).clamp(min=0)
        overlap = wh_min[:, 0] * wh_min[:, 1]#[n]
        
        #------------------------------#
        #   预测框面积和实际框面积计算
        #------------------------------#
        area1   = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
        area2   = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])
        
        #------------------------------#
        #   计算交并比
        #------------------------------#
        union   = (area1 + area2 - overlap)
        iou     = overlap / union

        #------------------------------#
        #   计算外包围框
        #------------------------------#
        lt_max  = torch.max(preds[:, :2],targets[:, :2])
        rb_max  = torch.max(preds[:, 2:],targets[:, 2:])
        wh_max  = (rb_max + lt_max).clamp(0)
        G_area  = wh_max[:, 0] * wh_max[:, 1]

        #------------------------------#
        #   计算GIOU
        #------------------------------#
        giou    = iou - (G_area - union) / G_area.clamp(1e-10)
        loss    = 1. - giou
        return loss.sum()
        
    def compute_reg_loss(self, preds, targets, mask):
        #------------------------#
        #   计算batch_size
        #   计算回归参数长度（4）
        #------------------------#
        batch_size  = targets.shape[0]
        c           = targets.shape[-1]
        
        num_pos     = torch.sum(mask, dim=1).clamp_(min=1).float()#[batch_size,]
        preds = self._reshape_cat_out(preds)
        assert preds.shape==targets.shape
        
        loss = 0
        for batch_index in range(batch_size):
            pred_pos    = preds[batch_index][mask[batch_index]]
            target_pos  = targets[batch_index][mask[batch_index]]
            batch_loss  = self.giou_loss(pred_pos, target_pos).view(1)
            loss += batch_loss
        return torch.sum(loss, dim=0) / torch.sum(num_pos)