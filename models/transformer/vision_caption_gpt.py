from torch import nn

from models.transformer.decoder import DecoderLayer
from models.transformer.embeddings import InputEmbeddings
from models.transformer.positional_encoding import PositionalEncoding

class VisionCaptionGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, hidden_dim, dropout, max_seq_length):
        super().__init__()
        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, hidden_dim, dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, captions, image_features, mask):
        x = self.embedding(captions)
        x = self.dropout(self.positional_encoding(x))
        
        for layer in self.layers:
            x = layer(x, image_features, mask)
            
        x = self.final_norm(x)
        
        return x