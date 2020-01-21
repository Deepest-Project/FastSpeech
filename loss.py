import torch
import torch.nn as nn


class TransformerLoss(nn.Module):
    def __init__(self):
        super(TransformerLoss, self).__init__()
        
    def forward(self, pred, target):
        mel_out, duration_out = pred
        mel_target, duration_target = target
        
        mel_loss = nn.L1Loss()(mel_out, mel_target)
        duration_loss = nn.MSELoss()(duration_out, duration_target)
        
        return mel_loss, duration_loss