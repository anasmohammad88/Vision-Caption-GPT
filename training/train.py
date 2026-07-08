import math
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from PIL import Image
from training.config import Config
from training.transform import VALID_TRANSFORM

class Trainer:
    def __init__(self, model, train_loader, valid_loader, optimizer, criterion, caption_generator, metrics, base_lr, warmup_ratio):
        self.device =  torch.device("cuda" if torch.cuda.is_available()  else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.warmup_ratio = warmup_ratio
        self.caption_generator = caption_generator
        self.metrics = metrics
        self.base_lr = base_lr
        self.global_step = 0
        self.writer = SummaryWriter(log_dir="logs")

    def get_lr_scale(self, step: int, warmup_steps: int, total_steps: int) -> float:
        # Linear warmup
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        
        # Cosine decay
        progress = ((step - warmup_steps)/ max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    def train(self, epochs):
        """
        Full training loop for caption generation.
        Tracks:
        - Train Loss
        - Validation Loss
        - Train Perplexity
        - Validation Perplexity
        """
        best_bleu4 = 0.0
        self.total_steps = len(self.train_loader) * epochs
        self.warmup_steps = int(self.warmup_ratio * self.total_steps)
        
        print(f"Total Steps   : {self.total_steps}")
        print(f"Warmup Steps  : {self.warmup_steps}")
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            self.train_loader.dataset.current_epoch = epoch
            
            for batch in self.train_loader:
                images = batch["image"].to(self.device)
                captions = batch["caption"].to(self.device)
                
                # causal mask forcing
                input_ids = captions[:, :-1]
                target_ids = captions[:, 1:]
                attention_mask = (input_ids != 0).long()
                
                 # LR Schedule 
                scale = self.get_lr_scale(step=self.global_step, warmup_steps=self.warmup_steps, total_steps=self.total_steps)
                current_lr = self.base_lr * scale
                for pg in self.optimizer.param_groups:
                    pg["lr"] = current_lr
                
                # Reset gradients
                self.optimizer.zero_grad()

                # Forward pass
                logits = self.model(images=images, captions=input_ids, attention_mask=attention_mask)
    
                # Compute loss
                # (Batch Size, Sequence Length, Vocabulary Size) --> (Batch Size × Sequence Length, Vocabulary Size)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

                # Backpropagation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Update weights
                self.optimizer.step()
                self.global_step += 1
                 
                # Accumulate loss
                train_loss += loss.item()
            
            # Average train loss
            train_loss /= len(self.train_loader)
            
            # Average train loss
            train_perplexity = math.exp(train_loss)
            
            # =========== Validation =========
            self.model.eval()
            valid_loss = 0.0
            predictions = []
            references = []
            compute_bleu = ((epoch + 1) % 50 == 0 or epoch == epochs - 1)
            
            with torch.no_grad():
                for batch in self.valid_loader:
                    images = batch["image"].to(self.device)
                    captions = batch["caption"].to(self.device)
                    batch_references  = batch["references"]
                    
                    # causal mask forcing
                    input_ids = captions[:, :-1]
                    target_ids = captions[:, 1:]
                    attention_mask = (input_ids != 0).long()
                
                    # Forward pass
                    logits = self.model(images=images, captions=input_ids, attention_mask=attention_mask)

                    # Compute loss
                    loss = self.criterion(logits.reshape(-1, logits.size(-1)),target_ids.reshape(-1))

                    # Accumulate loss
                    valid_loss += loss.item()
                    
                    # Caption Generation every 5 epochs
                    if compute_bleu:
                        generated = self.caption_generator.generate(images, max_length=Config.MAX_SEQUENCE_LENGTH, beam_size=5)
                        predictions.extend(generated)
                        references.extend(batch_references) # BLEU expects List[List[str]]

            # Average validation loss
            valid_loss /= len(self.valid_loader)
            valid_perplexity = math.exp(valid_loss)
            
            if compute_bleu:
                bleu_scores = self.metrics.compute_bleu(predictions=predictions, references=references)
            else: bleu_scores = {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
            
            self.writer.add_scalars("Loss", {"Train": train_loss,"Eval": valid_loss}, epoch)
            self.writer.add_scalars("Perplexity", {"Train": train_perplexity,"Eval": valid_perplexity}, epoch)
            if compute_bleu:
                self.writer.add_scalars("BLEU", {
                        "BLEU-1": bleu_scores["bleu1"],
                        "BLEU-2": bleu_scores["bleu2"],
                        "BLEU-3": bleu_scores["bleu3"],
                        "BLEU-4": bleu_scores["bleu4"],
                    }, epoch )
            
            
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            if compute_bleu:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"T-Loss: {train_loss:.4f} | "
                    f"V-Loss: {valid_loss:.4f} | "
                    f"T-PPL: {train_perplexity:.4f} | "
                    f"V-PPL: {valid_perplexity:.4f} | "
                    f"B-1: {bleu_scores['bleu1']:.4f} | "
                    f"B-2: {bleu_scores['bleu2']:.4f} | "
                    f"B-3: {bleu_scores['bleu3']:.4f} | "
                    f"B-4: {bleu_scores['bleu4']:.4f} | "
                    f"LR: {current_lr:.6f}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"T-Loss: {train_loss:.4f} | "
                    f"V-Loss: {valid_loss:.4f} | "
                    f"T-PPL: {train_perplexity:.4f} | "
                    f"V-PPL: {valid_perplexity:.4f} | "
                    f"LR: {current_lr:.6f}"
                )
                
            if compute_bleu:
                self.debug_single_image(Config.IMAGE_DIR / "1089181217_ee1167f7af.jpg")
            
            # Save best model
            if compute_bleu and bleu_scores["bleu4"] > best_bleu4:
                best_bleu4 = bleu_scores["bleu4"]
                torch.save(self.model.state_dict(), "vision_caption_gpt.pth")
                print(" *** Saved")
                
        self.writer.close()
        print(f"\nBest Validation BLEU-4: {best_bleu4:.4f}")
        
    
    def debug_single_image(self, image_path):
        self.model.eval()
        image = Image.open(image_path).convert("RGB")
        image = VALID_TRANSFORM(image).unsqueeze(0)

        prediction = self.caption_generator.generate(images=image, max_length=Config.MAX_SEQUENCE_LENGTH, beam_size=5)[0]


        print("\n==============================")
        print(f"Image: {Path(image_path).name}")
        print(f"Prediction: {prediction}")

        image_name = Path(image_path).name
        dataset = self.valid_loader.dataset

        print("\nGround Truth:")
        for i, caption in enumerate(dataset.image_to_captions[image_name], start=1):
            print(f"{i}. {caption}")

        print("==============================\n")