from torchvision import transforms
from training.config import Config

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
                    transforms.RandomResizedCrop(Config.IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                    transforms.RandomHorizontalFlip(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(_MEAN, _STD),
                ])


VALID_TRANSFORM = transforms.Compose([
                    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(_MEAN, _STD),
                ])

TEST_TRANSFORM = VALID_TRANSFORM