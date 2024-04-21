# examples/train_example.py
import torch
from nanostudio.engine import Trainer, DataLoader, GPT, GPTConfig, Config

def main():
    model_config_dict = {
        'vocab_size': 50257,
        'n_layer': 8,
        'n_head': 8,
        'n_embd': 512,
        'block_size': 128,
        'dropout': 0.1,
        'bias': True
    }

    training_config_dict = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 5,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'beta1': 0.9,
        'beta2': 0.999,
        'weight_decay': 0.01,
        'dtype': 'float16',
        'ddp': False,
        'ddp_local_rank': 0,
        'backend': 'nccl',
        'out_dir': 'output'
    }

    config = Config(model_config=model_config_dict, training_config=training_config_dict)

    model_config = GPTConfig(**config.model_config)
    model = GPT(model_config)

    data_loader = DataLoader(hf_dataset_name="ag_news", block_size=model_config.block_size, 
                             batch_size=config.training_config['batch_size'], 
                             device=config.training_config['device'])

    trainer = Trainer(config=config.training_config, model=model, data_loader=data_loader)

    trainer.train()

    output_dir = config.training_config.get('OUTPUT_DIR', './')
    torch.save(model.state_dict(), f'{output_dir}/trained_model.pt')
    print("Training complete and model saved!")

if __name__ == "__main__":
    main()
