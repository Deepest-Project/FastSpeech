import torch
import torch.nn as nn
from utils.utils import get_mask_from_lengths

class TransformerLoss(nn.Module):
    def __init__(self):
        super(TransformerLoss, self).__init__()
        
    def forward(self, pred, target, lengths):
        mel_out, duration_out = pred
        mel_target, duration_target = target
        text_lengths, mel_lengths = lengths
        
        mel_mask = ~get_mask_from_lengths(mel_lengths)
        duration_mask = ~get_mask_from_lengths(text_lengths)
        
        mel_target = mel_target.masked_select(mel_mask.unsqueeze(1))
        mel_out = mel_out.masked_select(mel_mask.unsqueeze(1))

        duration_target = duration_target.masked_select(duration_mask)
        duration_out = duration_out.masked_select(duration_mask)
        
        mel_loss = nn.L1Loss()(mel_out, mel_target)
        duration_loss = nn.MSELoss()(duration_out, duration_target)
        
        return mel_loss, duration_loss