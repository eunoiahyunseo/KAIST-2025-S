import os
import time
import math
import pickle
from contextlib import nullcontext
import yaml

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.nn.functional as F
import uuid

# -----------------------------------------------------------------------------
# These configs will be overridden by the config file and so their values here do not matter.
out_dir = 'out'

run_name = 'gpt2' # 'run' + str(time.time())

# data
dataset = 'text8'
batch_size = 64
block_size = 256

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
qk_layernorm = True
do_x1_sc = False

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
data_dir = '/path/to/datasets/text8' #  directory should contain meta.pkl

# sampling
total_samples = 128
dt = 0.001
max_t = 0.98
argmax_final = True
noise = 0.0
x1_temp = 1.0
use_different_x1_sc_temp = False
x1_sc_temp = 1.0
ignore_x1_sc = False # If true, even if the model is self conditioned, we just put in the mask condition every iteration anyway

model_type = 'flow' # flow, d3pm

do_purity_sampling = False
purity_temp = 1.0

ckpt_path = 'out/ckpt.pt'

# d3pm settings
timesteps = 1000

# Flow Matching sampling settings
num_flow_steps = int(max_t / dt)  # Number of discretization steps
div_free = 0.0  # Divergence-free component
dtype_categorical = torch.float32  # Precision for categorical sampling
return_intermediates = False  # Whether to return intermediate states

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

assert model_type in ['flow', 'd3pm']

hash = str(uuid.uuid1()).split("-")[0]
samples_dir = os.path.join(out_dir, 'samples_' + time.strftime('%Y-%m-%d-%H-%M-%S') + '_' + hash)
os.mkdir(samples_dir)
with open(os.path.join(samples_dir, 'config.yaml'), 'w') as f:
    yaml.dump(config, f, sort_keys=False)

with open(os.path.join(samples_dir, f'run_name_{run_name}.txt'), 'w') as f:
    f.write(f'{run_name}')


from flow_model import GPT, GPTConfig

# Flow Matching imports
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper

# attempt to derive vocab_size from the dataset
data_dir = os.path.join('data', dataset)
meta_path = os.path.join(data_dir, 'meta.json')
assert os.path.exists(meta_path)

import json
with open(meta_path, 'r') as f:
    meta = json.load(f)
meta_vocab_size = meta['vocab_size']

print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

mask_token_id = meta_vocab_size - 1


device_type = 'cuda'
device = 'cuda:0'


def load_model(ckpt_path):
    # resume training from a checkpoint.
    print(f"Loading network from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    model_args['vocab_size'] = meta_vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    return model, checkpoint

model, checkpoint = load_model(ckpt_path)

# save the model information to the sample directory
model_information = {
    'model_args': checkpoint['model_args'],
    'iter_num': checkpoint['iter_num'],
    'best_val_loss': checkpoint['best_val_loss'],
    'config': checkpoint['config'],
}
torch.save(model_information, os.path.join(samples_dir, 'model_information.pt'))
checkpoint = None
model.eval()
model.to(device)

if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model) # requires PyTorch 2.0

# Flow Matching setup
class FlowMatchingModelWrapper(ModelWrapper):
    """Wrapper for GPT model to work with Flow Matching framework"""
    def __init__(self, gpt_model):
        super().__init__(gpt_model)
        self.gpt_model = gpt_model
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        """
        Forward pass for Flow Matching
        Args:
            x: input tokens, shape (batch_size, seq_len)
            t: time, shape (batch_size,)
        Returns:
            logits: output logits, shape (batch_size, seq_len, vocab_size)
        """
        # Convert time to proper shape for GPT model
        # t_expanded = t.unsqueeze(-1).expand(-1, x.shape[1])  # (batch_size, seq_len)
        logits, _ = self.gpt_model(x, t)
        logits = torch.softmax(logits, dim=-1)

        return logits


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return torch.softmax(self.model(x, t), dim=-1)
    
    
scheduler = PolynomialConvexScheduler(n=1.0)  # Linear scheduler
prob_path = MixtureDiscreteProbPath(scheduler=scheduler)
wrapped_probability_denoiser = FlowMatchingModelWrapper(model)
solver = MixtureDiscreteEulerSolver(
    model=wrapped_probability_denoiser,
    path=prob_path,
    vocabulary_size=meta_vocab_size
)


torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
print(torch.__version__)
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ----------------- SAMPLING CODE --------------=-

S = meta_vocab_size
B = batch_size
D = block_size

# write an empty file to store the samples eventually
with open(os.path.join(samples_dir, 'samples.txt'), 'w') as f:
    pass

assert total_samples % B == 0

with torch.no_grad():
    with ctx:
        for batch_idx in range(total_samples // B):
            print(f"Processing batch {batch_idx + 1}/{total_samples // B}")

            if model_type == 'flow':
                # Initialize with mask tokens (source distribution)
                x_init = torch.full((B, D), mask_token_id, device=device, dtype=torch.long)
                
                print(f"Sampling using Flow Matching solver with {num_flow_steps} steps...")
                print(f"Step size: {dt}, Max time: {max_t}")
                
                # Use Flow Matching solver for sampling
                if return_intermediates:
                    # Define time grid for intermediate sampling
                    time_grid = torch.linspace(0.0, max_t, steps=10, device=device)
                    samples = solver.sample(
                        x_init=x_init,
                        step_size=dt,
                        div_free=div_free,
                        dtype_categorical=dtype_categorical,
                        time_grid=time_grid,
                        return_intermediates=True,
                        verbose=True
                    )
                    
                    # Save intermediate states if needed
                    print(f"Intermediate samples shape: {samples.shape}")
                    samples = samples[-1]  # Take final samples
                else:
                    samples = solver.sample(
                        x_init=x_init,
                        step_size=dt,
                        div_free=div_free,
                        dtype_categorical=dtype_categorical,
                        return_intermediates=False,
                        verbose=True
                    )
                
                print(f"Sampling completed. Final samples shape: {samples.shape}")
                
                # Apply final argmax if requested
                if argmax_final:
                    print("Applying final argmax to remaining mask tokens...")
                    # Get final predictions
                    t_final = torch.ones((B,), device=device) * max_t
                    logits = wrapped_probability_denoiser(samples, t_final)
                    
                    # Only update positions that are still mask tokens
                    sample_is_mask = (samples == mask_token_id).float()
                    num_remaining_masks = sample_is_mask.sum().item()
                    print(f"Remaining mask tokens: {num_remaining_masks}/{B*D}")
                    
                    if num_remaining_masks > 0:
                        argmax_samples = torch.argmax(logits, dim=-1)
                        samples = (argmax_samples * sample_is_mask + 
                                  samples * (1 - sample_is_mask)).long()

                samples_np = samples.cpu().detach().numpy() # (B, D)
                
            # Save samples to file
            for sample_idx in range(samples_np.shape[0]):
                with open(os.path.join(samples_dir, 'samples.txt'), 'a') as f:
                    sample_line = ' '.join(map(str, samples_np[sample_idx]))
                    f.write(sample_line + '\n')
                    
            print(f'Batch {batch_idx + 1} completed: {samples_np.shape[0]} samples written to file.')
        
        print('All samples have been written to file.')

with open(os.path.join(samples_dir, 'finished_sampling.txt'), 'w') as f:
    f.write('finished sampling\n')