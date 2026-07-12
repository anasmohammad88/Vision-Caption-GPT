from torch import nn
import torch
from models.cnn.cnn_encoder import CNNEncoder
from models.transformer.transformer_head import TransformerHead
from models.transformer.vision_caption_gpt import VisionCaptionGPT
from models.transformer.visual_encoder import VisualEncoder

class VisionCaptionModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, hidden_dim,
                 dropout, max_seq_length, num_concepts=0):
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

        # Visual Transformer encoder
        self.visual_encoder = VisualEncoder(
            d_model=d_model,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

        # Auxiliary grounding head: predicts which visual concepts are present
        self.num_concepts = num_concepts
        self.grounding_head = nn.Linear(d_model, num_concepts) if num_concepts > 0 else None

    def forward(self, images, captions, attention_mask=None, return_grounding=False):
        """
        Returns:
            logits: (B, SeqLen, vocab_size)
            grounding_logits: (B, num_concepts)   [only if return_grounding=True]
        """
        image_features = self.encode_images(images)

        decoder_output = self.decoder(
            captions=captions,
            image_features=image_features,
            mask=attention_mask,
        )
        logits = self.lm_head.classifier(decoder_output)

        if return_grounding and self.grounding_head is not None:
            global_token = image_features[:, 0]  # (B, d_model)
            grounding_logits = self.grounding_head(global_token)
            return logits, grounding_logits

        return logits

    def encode_images(self, images):
        image_features = self.encoder(images)
        image_features = self.image_norm(image_features)
        image_features = self.visual_encoder(image_features)
        global_token = image_features.mean(dim=1, keepdim=True)
        image_features = torch.cat([global_token, image_features], dim=1)
        return image_features

    @torch.no_grad()
    def cross_gate_values(self):
        """Per-layer sigmoid(cross_gate). If these drift toward 0 during training,
        the model is learning to IGNORE the image - watch them."""
        return [torch.sigmoid(layer.cross_gate).item() for layer in self.decoder.layers]
