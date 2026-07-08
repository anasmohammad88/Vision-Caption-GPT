import torch
import random
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from dataset.preprocessing import clean_text
from training.config import Config

class Flickr8kDataset(Dataset):
    def __init__(self, images, preprocessor, transform, image_to_captions, random_caption=True):
        self.images = images
        self.preprocessor = preprocessor
        self.transform = transform
        self.image_dir = Config.IMAGE_DIR
        self.image_to_captions = image_to_captions
        self.random_caption = random_caption
        self.max_sequence_length = Config.MAX_SEQUENCE_LENGTH
        # Used only for deterministic validation
        self.caption_indices = {image_name: 0 for image_name in self.images}
        self.current_epoch = 0
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # Read images
        image_name = self.images[index] # Get data by index
        image = Image.open(self.image_dir / image_name).convert('RGB')
        
        if self.transform is not None: image = self.transform(image)
        
        captions = self.image_to_captions[image_name]
        
        if self.random_caption: caption = captions[(self.current_epoch + index) % len(captions)] # Training
        else: caption = captions[0] # Validation
            
        encoded_caption = self.preprocessor.process(caption)
        encoded_caption = self.preprocessor.pad_sequence(encoded_caption, self.max_sequence_length)
        
        return {
            "image": image, 
            "caption": torch.tensor(encoded_caption, dtype=torch.long), 
            "references": [clean_text(caption) for caption in self.image_to_captions[image_name]], 
            "image_name": image_name
            }
    

def flickr_collate_fn(batch):
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "caption": torch.stack([item["caption"] for item in batch]),
        "references": [item["references"] for item in batch],
        "image_name": [item["image_name"] for item in batch],
    }
    
def create_dataloader(images, preprocessor, image_to_captions, transform=None, batch_size=128, num_workers=2, shuffle=False, random_caption=True):
    dataset = Flickr8kDataset(images=images, preprocessor=preprocessor, transform=transform, image_to_captions=image_to_captions, random_caption=random_caption)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=flickr_collate_fn)