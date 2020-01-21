import torch
import torch.nn as nn
import torch.nn.functional as F
from init_layer import *
from transformer import *


class CBAD(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 kernel_size,
                 stride,
                 padding,
                 bias,
                 activation,
                 dropout):
        super(CBAD, self).__init__()
        self.conv = Conv1d(in_dim,
                           out_dim,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           bias=bias,
                           w_init_gain=activation)

        self.bn = nn.BatchNorm1d(out_dim)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        out = self.dropout(x)

        return out

    
class Prenet_E(nn.Module):
    def __init__(self, hp):
        super(Prenet_E, self).__init__()

        self.conv1 = CBAD(in_dim=hp.symbols_embedding_dim,
                          out_dim=hp.hidden_dim,
                          kernel_size=5,
                          stride=1,
                          padding=2,
                          bias=False,
                          activation='relu',
                          dropout=0.5)
        self.conv2 = CBAD(in_dim=hp.hidden_dim,
                          out_dim=hp.hidden_dim,
                          kernel_size=5,
                          stride=1,
                          padding=2,
                          bias=False,
                          activation='relu',
                          dropout=0.5)
        self.conv3 = CBAD(in_dim=hp.hidden_dim,
                          out_dim=hp.hidden_dim,
                          kernel_size=5,
                          stride=1,
                          padding=2,
                          bias=False,
                          activation='relu',
                          dropout=0.5)

        self.center = Linear(hp.hidden_dim, hp.hidden_dim)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x).transpose(1, 2).contiguous()
        out = self.center(x)

        return out

    
class Duration(nn.Module):
    def __init__(self, hp):
        super(Duration, self).__init__()
        self.conv1 = Conv1d(hp.hidden_dim,
                            hp.duration_dim,
                            kernel_size=3,
                            padding=1)
        
        self.conv2 = Conv1d(hp.duration_dim,
                            hp.duration_dim,
                            kernel_size=3,
                            padding=1)
        
        self.ln1 = nn.LayerNorm(hp.duration_dim)
        self.ln2 = nn.LayerNorm(hp.duration_dim)
        self.dropout = nn.Dropout(0.1)
        
        self.linear = nn.Linear(hp.duration_dim, 1)

    def forward(self, hidden_states):
        x = F.relu(self.conv1(hidden_states.permute(1,2,0).contiguous()))
        x = self.dropout(self.ln1(x.transpose(1,2).contiguous()))
        x = F.relu(self.conv2(x.transpose(1,2).contiguous()))
        x = self.dropout(self.ln2(x.transpose(1,2).contiguous()))
        out = self.linear(x)
        
        return out.squeeze(-1)

    
class Model(nn.Module):
    def __init__(self, hp):
        super(Model, self).__init__()
        self.hp = hp
        self.Embedding = nn.Embedding(hp.n_symbols, hp.symbols_embedding_dim)
        self.Prenet_E = Prenet_E(hp)
        self.pe = PositionalEncoding(hp.hidden_dim).pe
        self.alpha = nn.Parameter(torch.ones(1))

        self.Encoder = nn.ModuleList([FFTblock(d_model=hp.hidden_dim,
                                               nhead=hp.n_heads,
                                               dim_feedforward=hp.ff_dim)
                                      for _ in range(hp.n_layers)])
        
        self.Decoder = nn.ModuleList([FFTblock(d_model=hp.hidden_dim,
                                               nhead=hp.n_heads,
                                               dim_feedforward=hp.ff_dim)
                                      for _ in range(hp.n_layers)])
        
        self.Duration = Duration(hp)
        self.Projection = Linear(hp.hidden_dim, hp.n_mel_channels)
        
    def forward(self, text, alignments, text_lengths=None, mel_lengths=None):
        ### Size ###
        B, L = text.size()
        T = alignments.size(1)
        
        ### Positional embedding ###
        position_embedding = text.new_tensor(self.pe, dtype=torch.float)
        
        ### Prepare Inputs ###
        embedded_input = self.Embedding(text)
        encoder_input = self.Prenet_E(embedded_input).transpose(0,1).contiguous()
        encoder_input += self.alpha*position_embedding[:L].unsqueeze(1)
        
        ### Prepare Masks ###
        text_mask = get_mask_from_lengths(text_lengths)
        mel_mask = get_mask_from_lengths(mel_lengths)
        
        ### Speech Synthesis ###
        hidden_states = encoder_input
        for layer in self.Encoder:
            hidden_states = layer(hidden_states,
                                  src_key_padding_mask=text_mask)
            
        durations = self.align2duration(alignments, mel_mask)
        hidden_states_expanded = self.LR(hidden_states, durations)
        hidden_states_expanded += position_embedding[:T].unsqueeze(1)

        for layer in self.Decoder:
            hidden_states_expanded = layer(hidden_states_expanded,
                                           src_key_padding_mask=mel_mask)
        
        mel_out = self.Projection(hidden_states_expanded.transpose(0,1).contiguous())
        mel_out = mel_out.masked_fill_(mel_mask.unsqueeze(-1), 0)
        mel_out = mel_out.transpose(1, 2).contiguous()
        
        ### Duration Predictor ###
        duration_out = self.Duration(hidden_states)
        duration_out = duration_out.masked_fill_(text_mask, 0)
        
        return mel_out, durations ,duration_out
    
    
    def inference(self, text, alpha=1.0):
        ### Prepare Inference ###
        text_lengths = torch.tensor([1, text.size(1)])
        position_embedding = text.new_tensor(self.pe, dtype=torch.float)
        
        ### Prepare Inputs ###
        embedded_input = self.Embedding(text)
        encoder_input = self.Prenet_E(embedded_input).transpose(0,1).contiguous()
        encoder_input += self.alpha*position_embedding[:text.size(1)].unsqueeze(1)
        
        ### Speech Synthesis ###
        hidden_states = encoder_input
        text_mask = text.new_zeros(1,text.size(1)).to(torch.bool)
        for layer in self.Encoder:
            hidden_states = layer(hidden_states,
                                  src_key_padding_mask=text_mask)
            
        ### Duration Predictor ###
        durations = self.Duration(hidden_states)
        hidden_states_expanded = self.LR(hidden_states, durations, alpha)
        hidden_states_expanded += position_embedding[:hidden_states_expanded.size(0)].unsqueeze(1)
        mel_mask = text.new_zeros(1, hidden_states_expanded.size(0)).to(torch.bool)
        
        for layer in self.Decoder:
            hidden_states_expanded = layer(hidden_states_expanded,
                                           src_key_padding_mask=mel_mask)
        
        mel_out = self.Projection(hidden_states_expanded.transpose(0,1).contiguous())
        mel_out = mel_out.transpose(1,2).contiguous()
        
        return mel_out, durations

    
    def align2duration(self, alignments, mel_mask):
        ids = alignments.new_tensor( torch.arange(alignments.size(2)) )
        max_ids = torch.max(alignments, dim=2)[1].unsqueeze(-1)
        one_hot = 1.0*(ids==max_ids)
        one_hot.masked_fill_(mel_mask.unsqueeze(2), 0)
        
        durations = torch.sum(one_hot, dim=1)

        return durations
    
    
    def LR(self, hidden_states, durations, alpha=1.0):
        L, B, D = hidden_states.size()
        durations = torch.round(durations*alpha).to(torch.long)
        T=int(torch.sum(durations, dim=-1).max().item())
        expanded = hidden_states.new_zeros(T, B, D)
        
        for i, d in enumerate(durations):
            mel_len = torch.sum(d).item()
            expanded[:mel_len, i] = torch.repeat_interleave(hidden_states[:, i],
                                                            d,
                                                            dim=0)
            
        return expanded
    
def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = lengths.new_tensor(torch.arange(0, max_len))
    mask = (lengths.unsqueeze(1) <= ids).to(torch.bool)
    return mask