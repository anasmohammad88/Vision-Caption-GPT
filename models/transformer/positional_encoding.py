import math
import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        # A tensor of shape(max_seq_length, d_model)
        pe = torch.zeros(max_seq_length, d_model)
        
        # (max_seq_length,) -> (max_seq_length, 1)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # (1, max_seq_length, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # (batch_size, seq_length, d_model)
        return x + self.pe[:, :x.size(1)]      
