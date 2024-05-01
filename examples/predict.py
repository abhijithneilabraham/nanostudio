import torch
from nanostudio.engine import Evaluator, GPT, GPTConfig, Config
from transformers import GPT2Tokenizer

def main():
    # Load configuration and model
    model_config_dict = {
        'vocab_size': 50257,
        'n_layer': 8,
        'n_head': 8,
        'n_embd': 512,
        'block_size': 128,
        'dropout': 0.1,
        'bias': True
    }

    model_config = GPTConfig(**model_config_dict)
    model = GPT(model_config)

    # Assume the model was trained and saved as 'trained_model.pt' in the 'output' directory
    model_path = 'trained_model.pt'
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Initialize evaluator
    evaluator = Evaluator(model_path=model_path)

    # Generate text using a start prompt
    generated_texts = evaluator.generate_text(start_text="Oil")
    for text in generated_texts:
        print(text)
        
main()