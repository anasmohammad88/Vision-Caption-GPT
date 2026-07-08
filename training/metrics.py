from torchmetrics.text import BLEUScore

class CaptionMetrics:
    """ BLEU evaluation for image captioning  """

    def __init__(self):
        self.bleu1 = BLEUScore(n_gram=1)
        self.bleu2 = BLEUScore(n_gram=2)
        self.bleu3 = BLEUScore(n_gram=3)
        self.bleu4 = BLEUScore(n_gram=4)

    def compute_bleu(self, predictions, references):
        """
        Input:
            predictions: List[str]
            references: List[List[str]]

        Returns:
            dict containing BLEU-1 ... BLEU-4
        """
        scores = {
            "bleu1": self.bleu1(predictions, references).item(),
            "bleu2": self.bleu2(predictions, references).item(),
            "bleu3": self.bleu3(predictions, references).item(),
            "bleu4": self.bleu4(predictions, references).item(),
        }
        return scores