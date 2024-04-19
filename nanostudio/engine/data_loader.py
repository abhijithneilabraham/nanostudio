import numpy as np
import torch
import os

class DataLoader:
    def __init__(self, data_dir='data', block_size = 1024, batch_size = 12, device='cpu'):
        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def load_data(self, split):
        """
        Load data from a numpy memmap file based on the split.
        """
        file_path = os.path.join(self.data_dir, f'{split}.bin')
        return np.memmap(file_path, dtype=np.uint16, mode='r')

    def get_batch(self, data):
        """
        Randomly sample batch indices and extract sequences from the data array.
        """
        # Calculate the number of possible sequences based on data length and block size
        num_sequences = len(data) - self.block_size
        # Randomly sample indices for batch sequences
        indices = torch.randint(0, num_sequences, (self.batch_size,))
        # Extract sequences from the data array
        x = torch.stack([torch.from_numpy(data[int(index):int(index) + self.block_size]).long() for index in indices])
        y = torch.stack([torch.from_numpy(data[int(index) + 1:int(index) + self.block_size + 1]).long() for index in indices])
        # Move data to the specified device, e.g., GPU or CPU
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        return x, y

    def fetch_batch(self, split='train'):
        """
        Convenience method to fetch a batch from the specified split.
        """
        data = self.load_data(split)
        return self.get_batch(data)
