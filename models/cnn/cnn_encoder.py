import torch
import torch.nn as nn
from models.transformer.positional_encoding import PositionalEncoding
from training.config import Config

class CNNEncoder(nn.Module):

    class Block(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()

            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.act = nn.GELU()
            self.dropout = nn.Dropout2d(Config.DROPOUT)
            
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut = nn.Identity()

        def forward(self, x):
            residual = self.shortcut(x)
            
            x = self.act(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))

            x = x + residual
            return self.dropout(self.act(x))

    class Downsample(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()

            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )

        def forward(self, x):
            return self.down(x)

    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(3,2,1)
        )
        
        in_channels = 32
        model_layers = []
        stages = [(64,2), (128,2), (256,2), (512,1)]
        in_channels = 32
        
        for out_channels, num_blocks in stages:
            for _ in range(num_blocks):
                model_layers.append(self.Block(in_channels, out_channels))
                in_channels = out_channels

            if out_channels <= 128:
                model_layers.append(self.Downsample(out_channels,out_channels))

        self.features = nn.Sequential(*model_layers)
        
        self.projection = nn.Conv2d(512, Config.D_MODEL, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((14,14))
        self.dropout = nn.Dropout(0.2)
        self.token_norm = nn.LayerNorm(Config.D_MODEL)
        self.pos_encoding = PositionalEncoding(d_model=Config.D_MODEL, max_seq_length=512)

    def forward(self, x):
        x = self.dropout(self.pool(self.projection(self.features(self.stem(x)))))
        x = x.flatten(2).transpose(1,2)
        return self.pos_encoding(self.token_norm(x))