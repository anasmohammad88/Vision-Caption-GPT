import torch
import torch.nn.functional as F

class CaptionGenerator:
    """Beam Search Caption Generator."""

    def __init__(self, model, vocabulary, device):
        self.model = model
        self.vocabulary = vocabulary
        self.device = device
        self.pad_id = vocabulary.word_to_index[vocabulary.PAD_TOKEN]
        self.bos_id = vocabulary.word_to_index[vocabulary.BOS_TOKEN]
        self.eos_id = vocabulary.word_to_index[vocabulary.EOS_TOKEN]

    def _lp(self, seq, alpha):
        """Length penalty. seq is (1, L) including BOS; count generated tokens only."""
        gen_len = max(1, seq.size(1) - 1)
        return gen_len ** alpha

    def _decode(self, seq):
        words = []
        for token in seq.squeeze(0).tolist():
            if token in (self.bos_id, self.pad_id): continue
            if token == self.eos_id: break
            words.append(self.vocabulary.index_to_word[token])
        return " ".join(words)

    @torch.no_grad()
    def generate(self, images, max_length=30, beam_size=5, length_penalty=0.7, min_length=3):
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
                    # already finished
                    if seq[0, -1].item() == self.eos_id:
                        completed.append((seq, score))
                        continue

                    # No padding mask needed during generation
                    decoder_output = self.model.decoder(captions=seq, image_features=image, mask=None)
                    logits = self.model.lm_head.classifier(decoder_output)
                    log_probs = F.log_softmax(logits[:, -1], dim=-1)

                    # Block EOS until the caption reaches a minimum length
                    gen_len = seq.size(1) - 1  # exclude BOS
                    if gen_len < min_length:
                        log_probs[0, self.eos_id] = float("-inf")

                    top_scores, top_tokens = torch.topk(log_probs, beam_size, dim=-1)
                    for j in range(beam_size):
                        token = top_tokens[0, j].view(1, 1)
                        new_seq = torch.cat([seq, token], dim=1)
                        new_score = score + top_scores[0, j].item()
                        candidates.append((new_seq, new_score))

                if len(candidates) == 0:
                    break

                # Prune with the LENGTH-NORMALISED score
                candidates.sort(key=lambda x: x[1] / self._lp(x[0], length_penalty), reverse=True)
                beams = candidates[:beam_size]
                if len(completed) >= beam_size:
                    break

            if len(completed) == 0:
                completed = beams

            best_seq = max(completed, key=lambda x: x[1] / self._lp(x[0], length_penalty))[0]
            captions.append(self._decode(best_seq))

        return captions
