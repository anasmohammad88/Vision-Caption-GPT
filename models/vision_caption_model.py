from torch import nn
import torch

from models.cnn.cnn_encoder import CNNEncoder
from models.transformer.transformer_head import TransformerHead
from models.transformer.vision_caption_gpt import VisionCaptionGPT
from models.transformer.visual_encoder import VisualEncoder

class VisionCaptionModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, hidden_dim, dropout, max_seq_length,):
        super().__init__()
        
        # Vision Encoder
        self.encoder = CNNEncoder()
        self.image_norm = nn.LayerNorm(d_model)
        
        # Transformer Decoder
        self.decoder = VisionCaptionGPT(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_seq_length=max_seq_length,
        )

        # Language Modeling Head
        self.lm_head = TransformerHead(
            vision_caption_gpt=self.decoder,
            d_model=d_model,
            vocab_size=vocab_size,
        )
        
        # visual Transformer encoder
        self.visual_encoder = VisualEncoder(
            d_model=d_model,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout
        )

    def forward(self, images, captions, attention_mask=None):
        """
        Args:
            images: Tensor (B, 3, H, W)
            captions: Tensor (B, SeqLen)
            attention_mask: Tensor (B, SeqLen)

        Returns:
            logits: Tensor (B, SeqLen, vocab_size)
        """

        # Encode images
        image_features = self.encode_images(images)

        # Decode captions conditioned on image
        decoder_output = self.decoder(
            captions=captions,
            image_features=image_features,
            mask=attention_mask,
        )

        # Vocabulary prediction
        logits = self.lm_head.classifier(decoder_output)

        return logits
    
    
    def encode_images(self, images):
        image_features = self.encoder(images)
        image_features = self.image_norm(image_features)

        # Refine visual tokens
        image_features = self.visual_encoder(image_features)

        # Build global token from refined features
        global_token = image_features.mean(dim=1, keepdim=True)
        image_features = torch.cat([global_token, image_features], dim=1)

        return image_features