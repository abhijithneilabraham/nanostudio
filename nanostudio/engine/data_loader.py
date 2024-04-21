import numpy as np
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer
import os

class DataLoader:
    def __init__(self, block_size=1024, batch_size=12, device='cpu', local_data_dir=None, hf_dataset_name=None, hf_dataset_split='train'):
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.data_cache = {}
        
        if hf_dataset_name:
            # Load dataset from Hugging Face
            self.dataset = load_dataset(hf_dataset_name, split=hf_dataset_split)
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  
            # Set the tokenizer padding token to EOS token which is common for GPT-2
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.preprocess_hf_data()
        elif local_data_dir:
            # Load local data
            self.load_local_data(local_data_dir, hf_dataset_split)
        else:
            raise ValueError("Either local_data_dir or hf_dataset_name must be specified.")

    def preprocess_hf_data(self):
        # Convert Hugging Face dataset to format suitable for batching
        tokenized_data = []
        for item in self.dataset:
            # Tokenize the text entry and ensure it's in the correct format for batching
            tokens = self.tokenizer(
                item['text'],
                return_tensors="np",
                max_length=self.block_size,
                truncation=True,
                padding="max_length"
            )['input_ids'].squeeze()
            tokenized_data.append(tokens)
        
        # Flatten the list of token arrays into a single array
        flattened_data = np.concatenate(tokenized_data)
        self.data_cache['train'] = flattened_data

    def load_local_data(self, data_dir, split):
        file_path = os.path.join(data_dir, f'{split}.bin')
        data_array = np.memmap(file_path, dtype=np.uint16, mode='r')
        if len(data_array) < self.block_size:
            raise ValueError(f"Dataset split '{split}' is too small for the defined block_size.")
        self.data_cache[split] = data_array

    def get_batch(self, split='train'):
        if split not in self.data_cache:
            raise ValueError(f"No data loaded for split: {split}")

        data = self.data_cache[split]
        num_sequences = len(data) - self.block_size
        while True:
            start_indices = np.random.randint(0, num_sequences, size=self.batch_size)
            x = [data[i:i+self.block_size] for i in start_indices]
            y = [data[i+1:i+self.block_size+1] for i in start_indices]
            x_tensor = torch.tensor(x, dtype=torch.long).to(self.device, non_blocking=True)
            y_tensor = torch.tensor(y, dtype=torch.long).to(self.device, non_blocking=True)
            yield x_tensor, y_tensor
