import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm  # Import tqdm for progress bars


class Trainer:
    def __init__(self, config, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.device = config['device']
        self.scaler = GradScaler()  # GradScaler for managing scaling of gradients automatically
        
        self.model.to(self.device)

        if config['ddp']:  # Setup for Distributed Data Parallel if enabled
            self.model = DDP(model, device_ids=[config['ddp_local_rank']], output_device=config['ddp_local_rank'])

        self.optimizer = AdamW(model.parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])

    def train(self):
        self.model.train()
        for epoch in range(self.config['epochs']):
            total_loss = 0
            progress_bar = tqdm(range(self.config['batch_size']), desc=f'Epoch {epoch}', leave=True)  # Initialize progress bar
            for _ in progress_bar:
                inputs, targets = next(self.data_loader.get_batch('train'))
                inputs = inputs.to(self.device)  # Ensure inputs are on the correct device
                targets = targets.to(self.device)  # Ensure targets are on the correct device
                
                self.optimizer.zero_grad()
                # Using autocast for the scope of this forward pass (automatic mixed precision)
                with autocast():
                    outputs, loss = self.model(inputs, targets=targets)
                
                # Scale loss to prevent gradient underflow
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)  # Adjust optimizer with scaled gradients
                self.scaler.update()  # Update the scale for next iteration
                total_loss += loss.item()

                progress_bar.set_postfix(loss=f'{loss.item():.4f}')  # Update progress bar with current loss

            avg_loss = total_loss / self.config['batch_size']
            print(f'Epoch {epoch}: Average Loss {avg_loss:.4f}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')
