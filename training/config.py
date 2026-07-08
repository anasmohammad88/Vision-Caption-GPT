from pathlib import Path

class Config:
    IMAGE_DIR = Path(__file__).parent.parent / "data" / "Images"
    VOCAB_SIZE = 7500
    MIN_FREQUENCY = 1
    IMAGE_SIZE = 224
    D_MODEL = 128
    NUM_HEADS = 4
    NUM_LAYERS = 3
    HIDDEN_DIM = 512
    MAX_SEQUENCE_LENGTH = 30
    BATCH_SIZE = 128
    LEARNING_RATE = 5e-4
    DROPOUT = 0.2
    WARMUP_RATIO = 0.05
    MIN_SCALE = 0.3
    EPOCHS = 100