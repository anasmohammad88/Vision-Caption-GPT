from torch import nn
import torch

from models.transformer.attention import MultiHeadAttention
from models.transformer.feedforward import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff_sublayer = FeedForward(d_model, hidden_dim, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.cross_gate = nn.Parameter(torch.tensor(2.0))

    def forward(self, captions, image_features, src_mask=None):
        # Masked Self Attention
        residual = captions
        norm1_x = self.norm1(captions)
        attn_output = self.self_attn(norm1_x, norm1_x, norm1_x, mask=src_mask, causal=True)
        x = residual + self.dropout(attn_output)
        
        # Cross Attention
        residual = x
        norm2_x = self.norm2(x)
        attn_output = self.cross_attn(query=norm2_x, key=image_features, value=image_features, causal=False)
        gate = torch.sigmoid(self.cross_gate)
        x = residual + gate * self.dropout(attn_output)
        
        # FeedForward
        residual = x
        norm3_x = self.norm3(x)
        ff_output = self.ff_sublayer(norm3_x)
        x = residual + self.dropout(ff_output)
        
        return x