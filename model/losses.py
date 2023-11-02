import torch
from torch import nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        B, c = output.size()
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        ce_loss = - torch.sum(log_preds * target, dim=1)
        if self.reduction == 'mean':
            ce_loss = ce_loss.mean()
        return loss * self.eps / c + (1 - self.eps) * ce_loss
