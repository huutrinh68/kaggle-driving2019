import torch 
import torch.nn as nn
from utils.logger import log


class CustomLoss(nn.Module):
    __name__ = 'custom_loss'

    def __init__(self, cfg, eps=1e-12, size_average=True):
        super().__init__()
        self.eps = eps
        self.size_average = size_average
    
    def forward(self, prediction, mask, regr):
        # Binary mask loss
        pred_mask = torch.sigmoid(prediction[:, 0])
    #     mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
        mask_loss = mask * torch.log(pred_mask + self.eps) + (1 - mask) * torch.log(1 - pred_mask + self.eps)
        mask_loss = -mask_loss.mean(0).sum()
        
        # Regression L1 loss
        pred_regr = prediction[:, 1:]
        regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
        regr_loss = regr_loss.mean(0)
        
        # Sum
        loss = mask_loss + regr_loss
        if not self.size_average:
            loss *= prediction.shape[0]
        return loss



def get_criterion(cfg):
    return CustomLoss(cfg)