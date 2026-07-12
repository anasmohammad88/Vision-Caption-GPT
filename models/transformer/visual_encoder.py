from torch import nn


class VisualEncoder(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, num_layers, dropout):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,   # Silences the norm_first warning
        )

    def forward(self, x):
        return self.encoder(x)
