import matplotlib.pyplot as plt
import torch
from PIL import Image
from inference.caption_generator import CaptionGenerator
from models.vision_caption_model import VisionCaptionModel
from training.config import Config
from training.transform import VALID_TRANSFORM
import argparse
import pickle

class ImageCaptionPredictor:

    def __init__(self, model_path: str, vocabulary):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = VisionCaptionModel(
            vocab_size=len(vocabulary),
            d_model=Config.D_MODEL,
            num_layers=Config.NUM_LAYERS,
            num_heads=Config.NUM_HEADS,
            hidden_dim=Config.HIDDEN_DIM,
            dropout=Config.DROPOUT,
            max_seq_length=Config.MAX_SEQUENCE_LENGTH
        )
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        self.generator = CaptionGenerator(model=self.model, vocabulary=vocabulary, device=self.device)

    @torch.no_grad()
    def predict(self, image_path: str, beam_size: int = Config.BEAM_SIZE, show_image: bool = True) -> str:

        image = Image.open(image_path).convert("RGB")
        image_tensor = VALID_TRANSFORM(image).unsqueeze(0).to(self.device)

        caption = self.generator.generate(
            images=image_tensor,
            max_length=Config.MAX_SEQUENCE_LENGTH,
            beam_size=beam_size,
            length_penalty=Config.LENGTH_PENALTY,
            min_length=Config.MIN_GEN_LENGTH,
        )[0]

        if show_image:
            plt.figure(figsize=(8, 8))
            plt.imshow(image)
            plt.axis("off")
            plt.title(caption, fontsize=14)
            plt.show()

        return caption
    
def main():

    parser = argparse.ArgumentParser(description="Image Caption Prediction")
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--model", default="checkpoints/vision_caption_gpt_best_bleu.pth", help="Model checkpoint")
    parser.add_argument("--vocab", default="checkpoints/vocabulary.pkl",help="Vocabulary file")
    parser.add_argument("--beam",type=int, default=Config.BEAM_SIZE,help="Beam size")
    args = parser.parse_args()

    with open(args.vocab, "rb") as f: vocab = pickle.load(f)
    predictor = ImageCaptionPredictor(model_path=args.model, vocabulary=vocab)

    caption = predictor.predict(image_path=args.image, beam_size=args.beam, show_image=True)
    print(f"\nPrediction:\n{caption}")


if __name__ == "__main__":
    main()