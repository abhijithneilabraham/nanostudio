import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Model


class GPTConfig:
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size, dropout, bias):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.dropout = dropout
        self.bias = bias


class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.model = GPT2Model(config=config)

    def forward(self, x, labels=None):
        return self.model(x, labels=labels)


def initialize_model(model_args, device, init_from='scratch'):
    # Create a GPTConfig from the given model_args
    config = GPTConfig(**model_args)

    if init_from == 'scratch':
        # Initialize model from scratch
        model = GPT(config)
    elif init_from == 'resume':
        # Resume model from checkpoint
        model = load_model_from_checkpoint(model_args, device)
    elif init_from.startswith('gpt2'):
        # Load a pretrained GPT-2 model
        model = GPT.from_pretrained(init_from, config=model_args)

    model.to(device)
    return model


def load_model_from_checkpoint(model_args, device):
    ckpt_path = model_args.get('checkpoint_path')
    checkpoint = torch.load(ckpt_path, map_location=device)
    # Create model config from checkpoint, then initialize model
    model_config = GPTConfig(**checkpoint['model_args'])
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model'])
    return model


def wrap_model_ddp(model, device_id):
    return DDP(model, device_ids=[device_id])
