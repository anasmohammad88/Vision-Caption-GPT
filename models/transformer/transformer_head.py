from torch import nn

class TransformerHead(nn.Module):
    def __init__(self, vision_caption_gpt, d_model, vocab_size):
        super().__init__()
        self.vision_caption_gpt = vision_caption_gpt
        self.classifier = nn.Linear(d_model, vocab_size, bias=False)
        self.classifier.weight = self.vision_caption_gpt.embedding.embedding.weight
        
    def forward(self, x, attention_mask):
        x = self.vision_caption_gpt(x, attention_mask)
        logits = self.classifier(x)
        return logits