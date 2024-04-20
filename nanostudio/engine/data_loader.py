import numpy as np
import torch
import os

class DataLoader:
    def __init__(self, data_dir='data', block_size=1024, batch_size=12, device='cpu'):
        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.data_cache = {}

    def load_data(self, split):
        if split not in self.data_cache:
            file_path = os.path.join(self.data_dir, f'{split}.bin')
            data_array = np.memmap(file_path, dtype=np.uint16, mode='r')
            if len(data_array) < self.block_size:
                raise ValueError("Dataset is too small for the defined block_size.")
            self.data_cache[split] = data_array
        return self.data_cache[split]

    def get_batch(self, split='train'):
        data = self.load_data(split)
        num_sequences = len(data) - self.block_size
        while True:
            start_indices = np.random.randint(0, num_sequences, size=(self.batch_size,))
            x = [data[i:i+self.block_size] for i in start_indices]
            y = [data[i+1:i+self.block_size+1] for i in start_indices]
            x_tensor = torch.tensor(x, dtype=torch.long).to(self.device, non_blocking=True)
            y_tensor = torch.tensor(y, dtype=torch.long).to(self.device, non_blocking=True)
            yield x_tensor, y_tensor
