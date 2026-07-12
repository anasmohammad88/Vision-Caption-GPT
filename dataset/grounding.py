from collections import Counter
import torch
from dataset.preprocessing import clean_text

# Function words carry no visual grounding signal.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "being", "been",
    "in", "on", "at", "of", "and", "to", "with", "this", "that", "these",
    "those", "it", "its", "his", "her", "their", "for", "as", "by", "he",
    "she", "they", "we", "you", "i", "him", "them", "from", "up", "out",
    "into", "over", "near", "next", "down", "then", "there", "here", "while",
    "who", "which", "some", "one", "other", "another", "also", "very",
    "just", "around", "through", "front", "behind", "off", "about",
}


class GroundingVocab:
    """Fixed set of visual 'concept' words from TRAIN captions -> multi-hot
    target per image. Used as weak supervision so the CNN's global token must
    predict which concepts are present. Still fully from scratch (Flickr8k only)."""

    def __init__(self, num_concepts=300, min_word_len=3):
        self.num_concepts = num_concepts
        self.min_word_len = min_word_len
        self.word_to_idx = {}
        self.doc_freq = None # fraction of TRAIN images containing each concept

    def _content_words(self, caption):
        for w in clean_text(caption).split():
            if len(w) >= self.min_word_len and w not in STOPWORDS:
                yield w

    def build(self, image_to_captions, train_images):
        df = Counter()
        for img in train_images:
            words = set()
            for cap in image_to_captions[img]:
                words.update(self._content_words(cap))
            df.update(words)                                   # document frequency
        common = [w for w, _ in df.most_common(self.num_concepts)]
        self.word_to_idx = {w: i for i, w in enumerate(common)}
        n_img = max(1, len(train_images))
        self.doc_freq = torch.tensor([df[w] / n_img for w in common], dtype=torch.float)

    def target(self, captions):
        vec = torch.zeros(len(self.word_to_idx), dtype=torch.float)
        for cap in captions:
            for w in self._content_words(cap):
                idx = self.word_to_idx.get(w)
                if idx is not None:
                    vec[idx] = 1.0
        return vec

    def pos_weight(self, cap=10.0):
        """BCEWithLogitsLoss pos_weight = (#neg / #pos) per concept, clamped.
        This stops the head from coasting on base rates (the ~0.14 plateau)."""
        f = self.doc_freq.clamp(min=1e-4)
        return ((1.0 - f) / f).clamp(max=cap)

    def __len__(self):
        return len(self.word_to_idx)