import torch
import torch.nn as nn
import torch.nn.functional as F
from .init_layer import *
from .transformer import *
from utils.utils import get_mask_from_lengths


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
        x = F.relu(self.conv1(hidden_states))
        x = self.dropout(self.ln1(x.transpose(1,2)))
        x = F.relu(self.conv2(x.transpose(1,2)))
        x = self.dropout(self.ln2(x.transpose(1,2)))
        out = self.linear(x)
        
        return out.squeeze(-1)

    
class Model(nn.Module):
    def __init__(self, hp):
        super(Model, self).__init__()
        self.hp = hp
        self.Embedding = nn.Embedding(hp.n_symbols, hp.symbols_embedding_dim)
        
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.register_buffer('pe', PositionalEncoding(hp.hidden_dim).pe)
        self.dropout = nn.Dropout(0.1)

        self.Encoder = nn.ModuleList([TransformerEncoderLayer(d_model=hp.hidden_dim,
                                                              nhead=hp.n_heads,
                                                              dim_feedforward=hp.ff_dim)
                                      for _ in range(hp.n_layers)])
        
        self.Decoder = nn.ModuleList([TransformerEncoderLayer(d_model=hp.hidden_dim,
                                                              nhead=hp.n_heads,
                                                              dim_feedforward=hp.ff_dim)
                                      for _ in range(hp.n_layers)])
        
        self.Duration = Duration(hp)
        self.Projection = Linear(hp.hidden_dim, hp.n_mel_channels)
        
    def outputs(self, text, alignments, text_lengths, mel_lengths):
        ### Size ###
        B, L, T = text.size(0), text.size(1), alignments.size(1)
        
        ### Prepare Inputs ###
        encoder_input = self.Embedding(text).transpose(0,1)
        encoder_input += self.alpha1*(self.pe[:L].unsqueeze(1))
        encoder_input = self.dropout(encoder_input)
        
        ### Prepare Masks ###
        text_mask = get_mask_from_lengths(text_lengths)
        mel_mask = get_mask_from_lengths(mel_lengths)
        
        ### Speech Synthesis ###
        hidden_states = encoder_input
        for layer in self.Encoder:
            hidden_states, _ = layer(hidden_states,
                                     src_key_padding_mask=text_mask)
        
        durations = self.align2duration(alignments, mel_mask)
        hidden_states_expanded = self.LR(hidden_states, durations)
        hidden_states_expanded += self.alpha2*(self.pe[:T].unsqueeze(1))
        hidden_states_expanded = self.dropout(hidden_states_expanded)

        for layer in self.Decoder:
            hidden_states_expanded, _ = layer(hidden_states_expanded,
                                              src_key_padding_mask=mel_mask)
        
        mel_out = self.Projection(hidden_states_expanded.transpose(0,1)).transpose(1, 2)
        duration_out = self.Duration(hidden_states.permute(1,2,0))
        
        return mel_out, duration_out, durations
    
    
    def forward(self, text, melspec, alignments, text_lengths, mel_lengths, criterion):
        ### Size ###
        text = text[:,:text_lengths.max().item()]
        melspec = melspec[:,:,:mel_lengths.max().item()]
        alignments = alignments[:,:mel_lengths.max().item(),:text_lengths.max().item()]
        mel_out, duration_out, durations = self.outputs(text, alignments, text_lengths, mel_lengths)
        
        mel_loss, duration_loss = criterion((mel_out, duration_out),
                                            (melspec, durations),
                                            (text_lengths, mel_lengths))
        
        return mel_loss, duration_loss
    
    
    def inference(self, text, alpha=1.0):
        ### Prepare Inference ###
        text_lengths = torch.tensor([1, text.size(1)])
        
        ### Prepare Inputs ###
        encoder_input = self.Embedding(text).transpose(0,1)
        encoder_input += self.alpha1*(self.pe[:text.size(1)].unsqueeze(1))
        
        ### Speech Synthesis ###
        hidden_states = encoder_input
        text_mask = text.new_zeros(1,text.size(1)).to(torch.bool)
        for layer in self.Encoder:
            hidden_states, _ = layer(hidden_states,
                                     src_key_padding_mask=text_mask)
            
        ### Duration Predictor ###
        durations = self.Duration(hidden_states.permute(1,2,0))
        hidden_states_expanded = self.LR(hidden_states, durations, alpha)
        hidden_states_expanded += self.alpha2*self.pe[:hidden_states_expanded.size(0)].unsqueeze(1)
        mel_mask = text.new_zeros(1, hidden_states_expanded.size(0)).to(torch.bool)
        
        for layer in self.Decoder:
            hidden_states_expanded, _ = layer(hidden_states_expanded,
                                              src_key_padding_mask=mel_mask)
        
        mel_out = self.Projection(hidden_states_expanded.transpose(0,1)).transpose(1,2)
        
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
        durations[durations<=0]=1
        T=int(torch.sum(durations, dim=-1).max().item())
        expanded = hidden_states.new_zeros(T, B, D)
        
        for i, d in enumerate(durations):
            mel_len = torch.sum(d).item()
            expanded[:mel_len, i] = torch.repeat_interleave(hidden_states[:, i],
                                                            d,
                                                            dim=0)
            
        return expanded
    
