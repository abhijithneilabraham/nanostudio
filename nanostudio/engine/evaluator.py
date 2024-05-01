import torch
from .model import GPT, GPTConfig
from transformers import GPT2Tokenizer

class Evaluator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = 'cpu'  # Adjust if necessary
        self.num_samples = 5  # Fewer samples for demonstration
        self.max_new_tokens = 100
        self.temperature = 0.8
        self.top_k = 50
        self.model = self.load_model()

    def load_model(self):
        # Load the model from state dictionary
        model_config = GPTConfig(
            vocab_size=50257, n_layer=8, n_head=8, n_embd=512,
            block_size=128, dropout=0.1, bias=True
        )
        model = GPT(model_config)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def generate_text(self, start_text="\n"):
        # Prepare tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        # Encode the input text
        input_ids = tokenizer.encode(start_text, return_tensors='pt').to(self.device)
        
        # Generate text
        generated_texts = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_texts.append(generated_text)

        return generated_texts