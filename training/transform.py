from torchvision import transforms
from training.config import Config

TRAIN_TRANSFORM = transforms.Compose([      
        transforms.Resize((Config.IMAGE_SIZE,Config.IMAGE_SIZE)),
        transforms.ToTensor()
    ])

VALID_TRANSFORM = transforms.Compose([      
        transforms.Resize((Config.IMAGE_SIZE,Config.IMAGE_SIZE)),
        transforms.ToTensor()
    ])


TEST_TRANSFORM = transforms.Compose([      
        transforms.Resize((Config.IMAGE_SIZE,Config.IMAGE_SIZE)),
        transforms.ToTensor()
    ])