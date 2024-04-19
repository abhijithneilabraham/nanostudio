# config.py
import os
import yaml
import json


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
        # Load user configuration if provided, otherwise load defaults
        self.config = self.load_config(config_path) if config_path else self.defaults
        self.post_process()

    def load_config(self, config_path):
        # Load configuration from a YAML or JSON file
        try:
            with open(config_path, 'r') as file:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(file)
                elif config_path.endswith('.json'):
                    return json.load(file)
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            return self.defaults

    def post_process(self):
        # Ensure all paths are absolute and directories are created
        for key in ['MODEL_DIR', 'DATA_DIR', 'OUTPUT_DIR']:
            if not os.path.isabs(self.config[key]):
                self.config[key] = os.path.join(self.config['ROOT_DIR'], self.config[key])
            os.makedirs(self.config[key], exist_ok=True)

