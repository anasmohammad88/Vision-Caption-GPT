import torch
from torch import nn
import torch.nn.functional as F

from training.config import Config


class CaptionGenerator:
    """Beam Search Caption Generator"""

    def __init__(self, model, vocabulary, device):
        self.model = model
        self.vocabulary = vocabulary
        self.device = device

        self.pad_id = vocabulary.word_to_index[vocabulary.PAD_TOKEN]
        self.bos_id = vocabulary.word_to_index[vocabulary.BOS_TOKEN]
        self.eos_id = vocabulary.word_to_index[vocabulary.EOS_TOKEN]

    @torch.no_grad()
    def generate(self, images, max_length=30, beam_size=3):

        self.model.eval()
        images = images.to(self.device)
        image_features = self.model.encode_images(images)
        
        captions = []

        for i in range(images.size(0)):
            image = image_features[i:i + 1]
            beams = [(torch.tensor([[self.bos_id]], device=self.device), 0.0)]
            completed = []
            
            for _ in range(max_length):
                candidates = []
                for seq, score in beams:
                    # finished sequence
                    if seq[0, -1] == self.eos_id:
                        completed.append((seq, score))
                        continue

                    attention_mask = (seq != self.pad_id).long()
                    decoder_output = self.model.decoder(captions=seq, image_features=image, mask=attention_mask)
                    logits = self.model.lm_head.classifier(decoder_output)
                    log_probs = F.log_softmax(logits[:, -1], dim=-1)
                    top_scores, top_tokens = torch.topk(log_probs, beam_size, dim=-1)
                    
                    for j in range(beam_size):
                        token = top_tokens[0, j].view(1, 1)
                        new_seq = torch.cat([seq, token], dim=1)
                        new_score = score + top_scores[0, j].item()
                        candidates.append((new_seq, new_score))

                if len(candidates) == 0:
                    break

                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_size]

            if len(completed) == 0:
                completed = beams

            # Length normalization
            best_seq = max(completed, key=lambda x: x[1] / (x[0].size(1) ** 0.7))[0]
            words = []

            for token in best_seq.squeeze().tolist():
                if token == self.bos_id: continue
                if token == self.eos_id: break
                if token == self.pad_id: continue
                words.append(self.vocabulary.index_to_word[token])
            captions.append(" ".join(words))

        return captions