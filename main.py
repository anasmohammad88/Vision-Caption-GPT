import torch
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch import nn, optim
from dataset.flickr8k_dataset import create_dataloader
from dataset.preprocessing import TextPreprocessor, clean_text
from dataset.tokenizer import Tokenizer
from dataset.vocabulary import Vocabulary
from models.vision_caption_model import VisionCaptionModel
from training.config import Config
from training.train import Trainer
from training.transform import TRAIN_TRANSFORM, VALID_TRANSFORM
from training.metrics import CaptionMetrics
from inference.caption_generator import CaptionGenerator
from collections import defaultdict

def load_text_file(file_path:str) -> list:
    """ Read a text file and return a list of lines """
    with open(file=file_path, mode="r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]
    
    
def main():
    """ Main training pipeline """
    
    # 1. Load dataset
    print("1. Load dataset...")
    dataset = load_text_file(Path(__file__).parent / "data" / "captions.txt")
    dataset = dataset[1:] # Skip the header line
    
    # Group all captions by image
    image_to_captions = defaultdict(list)
    for line in dataset:
        image_name, caption = line.split(",", maxsplit=1)
        image_to_captions[image_name].append(caption)
    
    image_names = list(image_to_captions.keys())
    
    # Split images instead of caption lines
    train_images, valid_images = train_test_split(image_names, test_size=0.2, random_state=42, shuffle=True)
    
    
    # 2. Initialize tokenizer
    print("2. Initialize tokenizer...")
    tokenizer = Tokenizer()
    
    # 3. Build vocabulary
    print("3. Build vocabulary...")
    
    # Rebuild caption lists
    train_captions = [ caption for image in train_images for caption in image_to_captions[image]]
    vocab = Vocabulary(Config.VOCAB_SIZE, Config.MIN_FREQUENCY)
    vocab.build_vocabulary(train_captions , tokenizer)
    
    sample = train_captions[random.randint(0, len(train_captions) - 1)]
    print(f"RAW: {sample}")

    cleaned = clean_text(sample)
    print(f"CLEANED: {cleaned}")

    tokens = tokenizer.tokenize(cleaned)
    print(f"TOKENS: {tokens}")

    encoded = vocab.encode(tokens)
    print(f"ENCODED: {encoded}")

    decoded = vocab.decode(encoded)
    print(f"DECODED: {decoded}")
    
    print(f"VOCAB SIZE: {len(vocab)}")
    print(f"TRAIN IMAGES: {len(train_images)}")
    print(f"VALID IMAGES: {len(valid_images)}")
    print(f"TRAIN CAPTIONS: {len(train_captions)}")

    # 4. Create preprocessor
    print("4. Create preprocessor...")
    preprocessor = TextPreprocessor(tokenizer=tokenizer, vocabulary=vocab)
    
    # 5. Create datasets
    print("5. Create datasets...")
    train_loader = create_dataloader(train_images, preprocessor, image_to_captions, TRAIN_TRANSFORM, Config.BATCH_SIZE, shuffle=True, random_caption=True)
    valid_loader = create_dataloader(valid_images, preprocessor, image_to_captions, VALID_TRANSFORM, Config.BATCH_SIZE, random_caption=False)

   
    # 6. Initialize model
    print("6. Initialize model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vision_caption_model = VisionCaptionModel(
        vocab_size=len(vocab), 
        d_model=Config.D_MODEL,
        num_layers=Config.NUM_LAYERS,
        num_heads=Config.NUM_HEADS,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT, 
        max_seq_length=Config.MAX_SEQUENCE_LENGTH
    ).to(device)
    
    caption_generator = CaptionGenerator(model=vision_caption_model,vocabulary=vocab,device=device)
    metrics = CaptionMetrics()
    
    # 7. Initialize optimizer and loss
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = optim.AdamW(params=vision_caption_model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-2)
    
    # 9. Initialize trainer
    trainer = Trainer(
        model=vision_caption_model, 
        train_loader=train_loader, 
        valid_loader=valid_loader, 
        optimizer=optimizer, 
        criterion=criterion, 
        caption_generator=caption_generator,
        metrics=metrics,
        base_lr=Config.LEARNING_RATE, 
        warmup_ratio=Config.WARMUP_RATIO
        )
    
    # 10. Start training
    trainer.train(Config.EPOCHS)
    
if __name__ == "__main__":
    main()