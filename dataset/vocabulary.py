from collections import Counter

from dataset.preprocessing import clean_text


class Vocabulary:
    def __init__(self, vocab_size=20000, min_frequency=1):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.BOS_TOKEN = "<BOS>"
        self.EOS_TOKEN  = "<EOS>"
        self.word_to_index = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1, self.BOS_TOKEN: 2, self.EOS_TOKEN :3}
        self.index_to_word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN, 2:self.BOS_TOKEN, 3: self.EOS_TOKEN}

    def build_vocabulary(self, texts, tokenizer):
        """
        Build vocabulary from dataset texts.
        """
        counter = Counter()
        for text in texts:
            text = clean_text(text)
            tokens = tokenizer.tokenize(text)
            counter.update(tokens)
            
        common_words = counter.most_common(self.vocab_size - 4)
        
        index = len(self.word_to_index)
        for word, frequency in common_words:
            if frequency < self.min_frequency:
                continue
            if word in self.word_to_index:
                continue
            self.word_to_index[word] = index
            self.index_to_word[index] = word
            index +=1

    def encode(self, tokens):
        """
        Convert tokens to indices.
        """
        encoded_tokens = []
        
        for token in tokens:
            # Try to get token index. If token does NOT exist, use UNK index instead
            index = self.word_to_index.get(token, self.word_to_index[self.UNK_TOKEN])
            encoded_tokens.append(index)
        
        return encoded_tokens
    
    def decode(self, indices):
        """
        Convert indices back to tokens.
        """
        decoded_tokens = []
        
        for index in indices:
            token = self.index_to_word.get(int(index), self.UNK_TOKEN)
            decoded_tokens.append(token)
        
        return decoded_tokens
    
    def __len__(self):
        return len(self.word_to_index)