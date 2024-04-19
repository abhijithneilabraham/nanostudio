import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2Model


class GPTConfig:
    # Default values for GPTConfig to allow optional parameters
    def __init__(self, vocab_size=50257,
                 n_layer=12,
                 n_head=12,
                 n_embd=768,
                 block_size=1024,
                 dropout=0.1,
                 bias=True):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.dropout = dropout
        self.bias = bias

    def to_transformers_config(self):
        # Converts custom config to a transformers library compatible config
        return GPT2Config(
            vocab_size=self.vocab_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            n_positions=self.block_size,
            n_ctx=self.block_size,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_bias=self.bias
        )


class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        # Ensure the config passed is a GPTConfig instance
        assert isinstance(config, GPTConfig), "config must be a GPTConfig instance"
        transformers_config = config.to_transformers_config()
        self.model = GPT2Model(config=transformers_config)

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
        model = GPT.from_pretrained(init_from,
                                    config=config.to_transformers_config())

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
    # Wrap the model with DistributedDataParallel
    return DDP(model, device_ids=[device_id])
