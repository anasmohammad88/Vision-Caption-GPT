import math
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from PIL import Image
from training.config import Config
from training.transform import VALID_TRANSFORM

class Trainer:
    def __init__(self, model, train_loader, valid_loader, optimizer, criterion, caption_generator, metrics, base_lr, warmup_ratio, checkpoint_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.criterion = criterion # Smoothed CE
        self.eval_criterion = nn.CrossEntropyLoss(ignore_index=0) # Unsmoothed label
        self.grounding_criterion = nn.BCEWithLogitsLoss()
        self.grounding_weight = Config.GROUNDING_WEIGHT if Config.USE_GROUNDING else 0.0
        self.warmup_ratio = warmup_ratio
        self.caption_generator = caption_generator
        self.metrics = metrics
        self.base_lr = base_lr
        self.global_step = 0
        self.writer = SummaryWriter(log_dir="logs")
        self.checkpoint_dir = checkpoint_dir

    def get_lr_scale(self, step, warmup_steps, total_steps):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        
        # Cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return Config.MIN_SCALE + (1.0 - Config.MIN_SCALE) * cosine

    def train(self, epochs):
        best_bleu4 = 0.0
        best_val_ce = float("inf")
        self.total_steps = len(self.train_loader) * epochs
        self.warmup_steps = int(self.warmup_ratio * self.total_steps)
            
        print(f"Total Steps   : {self.total_steps}")
        print(f"Warmup Steps  : {self.warmup_steps}")

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_ce = 0.0
            train_gloss = 0.0
            self.train_loader.dataset.current_epoch = epoch

            for batch in self.train_loader:
                images = batch["image"].to(self.device)
                captions = batch["caption"].to(self.device)

                input_ids = captions[:, :-1]
                target_ids = captions[:, 1:]
                attention_mask = (input_ids != 0).long()

                scale = self.get_lr_scale(self.global_step, self.warmup_steps, self.total_steps)
                current_lr = self.base_lr * scale
                for pg in self.optimizer.param_groups: pg["lr"] = current_lr

                self.optimizer.zero_grad()

                use_grounding = self.grounding_weight > 0 and "grounding" in batch
                if use_grounding: logits, g_logits = self.model(images=images, captions=input_ids, attention_mask=attention_mask, return_grounding=True)
                else: logits = self.model(images=images, captions=input_ids, attention_mask=attention_mask)

                ce = self.criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

                if use_grounding:
                    g_target = batch["grounding"].to(self.device)
                    gloss = self.grounding_criterion(g_logits, g_target)
                    loss = ce + self.grounding_weight * gloss
                    train_gloss += gloss.item()
                else: loss = ce

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.global_step += 1

                train_loss += loss.item()
                train_ce += ce.item()

            n = len(self.train_loader)
            train_loss /= n
            train_ce /= n
            train_gloss /= n
            train_perplexity = math.exp(train_ce)

            # =========== Validation =========
            self.model.eval()
            valid_loss = 0.0
            valid_ce = 0.0
            predictions = []
            references = []
            compute_bleu = ((epoch + 1) % Config.BLEU_EVAL_INTERVAL == 0) or (epoch == epochs - 1)
            bleu_images_done = 0
            subset = Config.BLEU_SUBSET_SIZE

            with torch.no_grad():
                for batch in self.valid_loader:
                    images = batch["image"].to(self.device)
                    captions = batch["caption"].to(self.device)
                    batch_references = batch["references"]

                    input_ids = captions[:, :-1]
                    target_ids = captions[:, 1:]
                    attention_mask = (input_ids != 0).long()

                    logits = self.model(images=images, captions=input_ids, attention_mask=attention_mask)
                    flat_logits = logits.reshape(-1, logits.size(-1))
                    flat_targets = target_ids.reshape(-1)

                    valid_loss += self.criterion(flat_logits, flat_targets).item()
                    valid_ce += self.eval_criterion(flat_logits, flat_targets).item()

                    if compute_bleu and (subset is None or bleu_images_done < subset):
                        generated = self.caption_generator.generate(
                            images,
                            max_length=Config.MAX_SEQUENCE_LENGTH,
                            beam_size=Config.BEAM_SIZE,
                            length_penalty=Config.LENGTH_PENALTY,
                            min_length=Config.MIN_GEN_LENGTH,
                        )
                        predictions.extend(generated)
                        references.extend(batch_references)
                        bleu_images_done += images.size(0)

            valid_loss /= len(self.valid_loader)
            valid_ce /= len(self.valid_loader)
            valid_perplexity = math.exp(valid_ce)

            if compute_bleu:
                bleu_scores = self.metrics.compute_bleu(predictions=predictions, references=references)
                avg_pred_len = sum(len(p.split()) for p in predictions) / max(1, len(predictions))
                avg_ref_len = sum(sum(len(r.split()) for r in refs) / max(1, len(refs)) for refs in references) / max(1, len(references))
            else:
                bleu_scores = {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
                avg_pred_len = avg_ref_len = 0.0

            gates = self.model.cross_gate_values()   # image-usage monitor

            # TensorBoard
            self.writer.add_scalars("Loss", {"Train": train_loss, "Eval_smoothed": valid_loss, "Eval_CE": valid_ce}, epoch)
            self.writer.add_scalar("GroundingLoss/Train", train_gloss, epoch)
            self.writer.add_scalars("Perplexity", {"Train": train_perplexity, "Eval": valid_perplexity}, epoch)
            self.writer.add_scalars("CrossGate", {f"layer{i}": g for i, g in enumerate(gates)}, epoch)
            if compute_bleu:
                self.writer.add_scalars("BLEU", {
                    "BLEU-1": bleu_scores["bleu1"], "BLEU-2": bleu_scores["bleu2"],
                    "BLEU-3": bleu_scores["bleu3"], "BLEU-4": bleu_scores["bleu4"],
                }, epoch)
                self.writer.add_scalars("Length", {"pred": avg_pred_len, "ref": avg_ref_len}, epoch)

            current_lr = self.optimizer.param_groups[0]["lr"]
            gate_str = "/".join(f"{g:.2f}" for g in gates)

            if compute_bleu:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"T-Loss: {train_loss:.4f} | T-CE: {train_ce:.4f} | G: {train_gloss:.4f} | "
                    f"V-CE: {valid_ce:.4f} | V-PPL: {valid_perplexity:.4f} | "
                    f"B-1: {bleu_scores['bleu1']:.4f} | B-2: {bleu_scores['bleu2']:.4f} | "
                    f"B-3: {bleu_scores['bleu3']:.4f} | B-4: {bleu_scores['bleu4']:.4f} | "
                    f"len p/r: {avg_pred_len:.1f}/{avg_ref_len:.1f} | gate: {gate_str} | LR: {current_lr:.6f}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"T-Loss: {train_loss:.4f} | T-CE: {train_ce:.4f} | G: {train_gloss:.4f} | "
                    f"V-CE: {valid_ce:.4f} | V-PPL: {valid_perplexity:.4f} | "
                    f"gate: {gate_str} | LR: {current_lr:.6f}"
                )

            if compute_bleu:
                self.debug_single_image(Config.IMAGE_DIR / "1089181217_ee1167f7af.jpg")

            # ---- Checkpointing (two independent signals) ----
            if compute_bleu and bleu_scores["bleu4"] > best_bleu4:
                best_bleu4 = bleu_scores["bleu4"]
                torch.save(self.model.state_dict(), self.checkpoint_dir / "vision_caption_gpt_best_bleu.pth")
                print(" *** Saved (best BLEU-4)")

            if valid_ce < best_val_ce:
                best_val_ce = valid_ce
                torch.save(self.model.state_dict(), self.checkpoint_dir / "vision_caption_gpt_best_ce.pth")

        self.writer.close()
        print(f"\nBest Validation BLEU-4 : {best_bleu4:.4f}")
        print(f"Best Validation CE     : {best_val_ce:.4f}")
        
        
    def debug_single_image(self, image_path):
        self.model.eval()
        image = Image.open(image_path).convert("RGB")
        image = VALID_TRANSFORM(image).unsqueeze(0)

        prediction = self.caption_generator.generate(
            images=image,
            max_length=Config.MAX_SEQUENCE_LENGTH,
            beam_size=Config.BEAM_SIZE,
            length_penalty=Config.LENGTH_PENALTY,
            min_length=Config.MIN_GEN_LENGTH,
        )[0]

        print("\n==============================")
        print(f"Image: {Path(image_path).name}")
        print(f"Prediction: {prediction}")

        image_name = Path(image_path).name
        dataset = self.valid_loader.dataset
        print("\nGround Truth:")
        for i, caption in enumerate(dataset.image_to_captions[image_name], start=1):
            print(f"{i}. {caption}")
        print("==============================\n")