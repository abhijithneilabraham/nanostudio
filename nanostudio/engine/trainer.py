import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, config, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.device = config['device']
        self.scaler = GradScaler()  # For automatic scaling of gradients
        self.model.to(self.device)

        if config['ddp']:
            self.model = DDP(model, device_ids=[config['ddp_local_rank']], output_device=config['ddp_local_rank'])

        self.optimizer = AdamW(model.parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self):
        self.model.train()
        for epoch in range(self.config['epochs']):
            total_loss = 0
            progress_bar = tqdm(range(self.config['batch_size']), desc=f'Epoch {epoch}', leave=True)

            for _ in progress_bar:
                inputs, targets = next(self.data_loader.get_batch('train'))
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                with autocast():
                    outputs, loss = self.model(inputs, targets=targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss += loss.item()
                progress_bar.set_postfix(loss=f'{loss.item():.4f}')

            avg_loss = total_loss / self.config['batch_size']
            print(f'Epoch {epoch}: Average Loss {avg_loss:.4f}')

            # Save checkpoint at the end of each epoch
            self.save_checkpoint(epoch, avg_loss)

    def save_checkpoint(self, epoch, loss):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')
