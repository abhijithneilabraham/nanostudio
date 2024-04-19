import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import time
from .model import wrap_model_ddp


class Trainer:
    def __init__(self, config, model, data_loader):
        self.config = config
        self.model = model.to(self.config['device'])
        self.data_loader = data_loader
        self.optimizer = self.configure_optimizer()
        self.scaler = GradScaler(enabled=(self.config['dtype'] == 'float16'))
        self.is_master = True  # Default to True for non-DDP setups

        if self.config['ddp']:
            self.model = wrap_model_ddp(self.model, self.config['ddp_local_rank'])
            torch.distributed.init_process_group(backend=self.config['backend'])
            self.is_master = torch.distributed.get_rank() == 0

    def configure_optimizer(self):
        """Configure the AdamW optimizer with hyperparameters from the config."""
        return optim.AdamW(self.model.parameters(),
                           lr=self.config['learning_rate'],
                           betas=(self.config['beta1'], self.config['beta2']),
                           weight_decay=self.config['weight_decay'])

    def train_epoch(self):
        """Train the model for one epoch, iterating over the data loader."""
        self.model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(self.data_loader.get_batches()):
            x, y = x.to(self.config['device']), y.to(self.config['device'])
            self.optimizer.zero_grad()

            with autocast(enabled=self.config['dtype'] == 'float16'):
                logits = self.model(x)
                loss = self.loss_fn(logits, y)

            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item()

        average_loss = total_loss / len(self.data_loader)
        if self.is_master:
            print(f"Avg Training Loss: {average_loss:.4f}")
        return average_loss

    def loss_fn(self, logits, labels):
        """Compute the loss function."""
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    def train(self):
        """Execute the full training loop over the specified number of epochs."""
        for epoch in range(self.config['epochs']):
            epoch_start_time = time.time()
            epoch_duration = time.time() - epoch_start_time

            if self.is_master:
                print(f"Epoch {epoch + 1}/{self.config['epochs']} completed in {epoch_duration:.2f} seconds.")
                self.save_model(epoch)

    def save_model(self, epoch):
        """Save model checkpoints."""
        checkpoint_path = f"{self.config['out_dir']}/model_epoch_{epoch + 1}.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def cleanup(self):
        """Cleanup distributed training resources if used."""
        if self.config['ddp']:
            torch.distributed.destroy_process_group()

    def evaluate(self):
        """Evaluate the model on the validation set."""
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in self.data_loader.get_validation_batches():
                x, y = x.to(self.config['device']), y.to(self.config['device'])
                with autocast(enabled=self.config['dtype'] == 'float16'):
                    logits = self.model(x)
                    loss = self.loss_fn(logits, y)
                val_loss += loss.item()
        average_val_loss = val_loss / len(self.data_loader.validation_data)
        if self.is_master:
            print(f"Validation Loss: {average_val_loss:.4f}")
        return average_val_loss
