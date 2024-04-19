import os
import yaml
import json
import logging
from collections import ChainMap


class Config:
    model_defaults = {
        'vocab_size': 50257,
        'n_layer': 12,
        'n_head': 12,
        'n_embd': 768,
        'block_size': 1024,
        'dropout': 0.1,
        'bias': True
    }
    
    training_defaults = {
        'batch_size': 64,
        'learning_rate': 1e-3,
        'epochs': 50,
        'save_every_n_steps': 1000,
        'print_every_n_steps': 100,
        'device': 'cpu',
        'beta1': 0.9,
        'beta2': 0.999,
        'weight_decay': 0.01,
        'dtype': 'float32',
        'ddp': False,
        'ddp_local_rank': 0,
        'backend': 'nccl',
        'out_dir': 'output'
    }

    def __init__(self, config_path=None, model_config={}, training_config={}):
        self.config_path = config_path
        base_config = self.load_config()
        self.model_config = ChainMap(model_config, base_config.get('MODEL_CONFIG', {}), self.model_defaults)
        self.training_config = ChainMap(training_config, base_config.get('TRAINING_CONFIG', {}), self.training_defaults)
        self.post_process()
    def load_config(self):
        # Loads configuration from a YAML or JSON file
        if self.config_path:
            try:
                with open(self.config_path, 'r') as file:
                    user_config = yaml.safe_load(file) if self.config_path.endswith(('.yaml', '.yml')) else json.load(file)
                    return user_config
            except Exception as e:
                logging.error(f"Error loading configuration file: {e}")
                return {}
        return {}

    def post_process(self):
        # Ensure all paths are absolute and directories are created
        for key in ['MODEL_DIR', 'DATA_DIR', 'OUTPUT_DIR']:
            path = self.training_config.get(key, '')
            if not os.path.isabs(path):
                path = os.path.join(os.getcwd(), path)
            os.makedirs(path, exist_ok=True)
            self.training_config[key] = path



# Configure logging
logging.basicConfig(level=logging.INFO)
