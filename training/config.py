from pathlib import Path

class Config:
    IMAGE_DIR = Path(__file__).parent.parent / "data" / "Images"

    # Vocabulary
    VOCAB_SIZE = 5000
    MIN_FREQUENCY = 1

    # Image
    IMAGE_SIZE = 224

    # Model capacity
    D_MODEL = 256
    NUM_HEADS = 8
    NUM_LAYERS = 4
    HIDDEN_DIM = 1024
    MAX_SEQUENCE_LENGTH = 30

    # Optimisation
    BATCH_SIZE = 128
    LEARNING_RATE = 5e-4       
    DROPOUT = 0.15
    WEIGHT_DECAY = 1e-2
    WARMUP_RATIO = 0.05
    MIN_SCALE = 0.05
    EPOCHS = 100

    # Evaluation
    BLEU_EVAL_INTERVAL = 20
    BLEU_SUBSET_SIZE = 500

    # Beam search
    BEAM_SIZE = 5
    LENGTH_PENALTY = 0.7 
    MIN_GEN_LENGTH = 3
    
    # Auxiliary grounding loss
    USE_GROUNDING = True
    NUM_CONCEPTS = 300 
    GROUNDING_WEIGHT = 0.3