import os
import yaml
import json
import logging
from collections import ChainMap

class Config:
    # Default configurations
    defaults = {
        'ROOT_DIR': os.path.dirname(os.path.abspath(__file__)),
        'MODEL_DIR': 'models',
        'DATA_DIR': 'datasets',
        'OUTPUT_DIR': 'output',
        'MODEL_CONFIG': {
            'vocab_size': 256,
            'embed_dim': 64,
            'num_heads': 2,
            'num_layers': 2,
            'dropout_rate': 0.1
        },
        'TRAINING_CONFIG': {
            'batch_size': 64,
            'learning_rate': 1e-3,
            'epochs': 50,
            'save_every_n_steps': 1000,
            'print_every_n_steps': 100
        }
    }

    def __init__(self, config_path=None):
        self.config_path = config_path
        self.config = self.load_config()
        self.post_process()

    def load_config(self):
        if self.config_path:
            try:
                with open(self.config_path, 'r') as file:
                    user_config = yaml.safe_load(file) if self.config_path.endswith(('.yaml', '.yml')) else json.load(file)
                    # Merge defaults with loaded configuration
                    return ChainMap(user_config, self.defaults)
            except Exception as e:
                logging.error(f"Error loading configuration file: {e}")
        return self.defaults

    def post_process(self):
        # Ensure all paths are absolute and directories are created
        for key in ['MODEL_DIR', 'DATA_DIR', 'OUTPUT_DIR']:
            path = self.config[key]
            if not os.path.isabs(path):
                path = os.path.join(self.config['ROOT_DIR'], path)
            os.makedirs(path, exist_ok=True)
            self.config[key] = path

    def update(self, updates):
        # Allows dynamic updating of the configuration
        self.config.update(updates)
        self.post_process()

    def get(self, key, default=None):
        # Get a configuration value with a default
        return self.config.get(key, default)

# Configure logging
logging.basicConfig(level=logging.INFO)
