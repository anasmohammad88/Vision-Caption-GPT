import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        self.attention_dropout = nn.Dropout(dropout)
        
        self.query_linear = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.key_linear = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.value_linear = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.output_linear = nn.Linear(in_features= d_model, out_features=d_model) 
        
    def split_heads(self, x, batch_size):
        seq_length = x.size(1)
        # (batch_size, seq_length, num_heads, head_dim)
        x = x.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
         # return (batch_size, num_heads, seq_length, head_dim)
        return x.permute(0, 2, 1, 3)

    def compute_attention(self, query, key, value, mask=None, causal=False):
        # query @ key^T
        # (batch_size, num_heads, seq_length, head_dim) --> (..., head_dim, seq_length)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # causal mask
        if causal:
            seq_len = query.size(-2)
            causal_mask = torch.tril(torch.ones(seq_len,seq_len,device=query.device)).bool()
            scores = scores.masked_fill(~causal_mask, float('-inf'))
        
        # padding mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_output = self.attention_dropout(F.softmax(scores, dim=-1))
        # (batch_size, num_heads, seq_length, seq_length)
        return torch.matmul(attention_output, value)
    
    def combine_heads(self, x, batch_size):
        # (batch_size, seq_length, num_heads, head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, -1, self.d_model)
    
    def forward(self, query, key, value, mask=None, causal=False):
        batch_size = query.size(0)

        query = self.split_heads(self.query_linear(query), batch_size)
        key = self.split_heads(self.key_linear(key), batch_size)
        value = self.split_heads(self.value_linear(value), batch_size)

        attention_weights = self.compute_attention(query, key, value, mask, causal)

        output = self.combine_heads(attention_weights, batch_size)
        return self.output_linear(output)