"""
modified with: eunoia_hyunseo heart2002101@knu.ac.kr
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import yaml

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from pathlib import Path

from flow_model import GPT, GPTConfig

# Flow Matching imports
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL

# -----------------------------------------------------------------------------
# these values will be overridden by the config file so their values here don't matter.
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())

wandb_id = 'blank'
is_repeat = False

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
overfit_batch = False
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
qk_layernorm = False
proper_timestep_emb = False
do_x1_sc = False
x1_sc_prob = 0.5

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster

data_dir = 'data/text8' #  directory should contain train.bin, val.bin, meta.pkl
warm_start_ckpt = None
resume_dir = None

model_type = 'flow' # flow, d3pm

d3pm_loss_weighting = False
d3pm_loss_weighting_maxT = 1000
timesteps = 1000

min_t = 0.0

bonus_seed_offset = 0

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and (isinstance(v, (int, float, bool, str)) or v is None) ]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------import os


assert model_type in ['flow', 'd3pm']

if resume_dir is None:
    if wandb_id == 'blank':
        out_dir = os.path.join(out_dir, time.strftime('%Y-%m-%d-%H-%M-%S') + '_' + wandb_run_name)
    else:
        out_dir = os.path.join(out_dir, str(wandb_id) + '_' + wandb_run_name)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

else:
    out_dir = resume_dir

assert (resume_dir is not None) == is_repeat


# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    print("ddp run")
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    print("not ddp run")
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

shared_generator = torch.Generator(device).manual_seed(42) # for use when we want the random numbers to be the same across processes
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size

print(f"tokens per iteration will be: {tokens_per_iter:,}")

torch.manual_seed(1337 + seed_offset + bonus_seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

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


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, qk_layernorm=qk_layernorm,
                  do_x1_sc=do_x1_sc, mask_token_id=mask_token_id, proper_timestep_emb=proper_timestep_emb,
                  d3pm_loss_weighting=d3pm_loss_weighting, d3pm_loss_weighting_maxT=d3pm_loss_weighting_maxT)
    
model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

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
        return logits

# Initialize Flow Matching components
scheduler = PolynomialConvexScheduler(n=1.0)  # Linear scheduler
prob_path = MixtureDiscreteProbPath(scheduler=scheduler)
wrapped_model = FlowMatchingModelWrapper(model)
solver = MixtureDiscreteEulerSolver(
    model=wrapped_model,
    path=prob_path,
    vocabulary_size=meta_vocab_size
)

# Flow Matching loss function
flow_loss_fn = MixturePathGeneralizedKL(path=prob_path)

def corrupt_data_flow_matching(x_0, x_1, times):
    """
    Flow Matching data corruption using MixtureDiscreteProbPath
    Args:
        x_0: source data (usually masked/random)
        x_1: target data (clean data)  
        times: time values
    Returns:
        path_sample: sampled intermediate states
    """
    path_sample = prob_path.sample(x_0=x_0, x_1=x_1, t=times)
    return path_sample

def corrupt_data(data, times):
    """Legacy corruption for d3pm compatibility"""
    b = times.shape[0]
    t = data.shape[1]

    assert times.shape == (b,)
    assert data.shape == (b, t)

    u = torch.rand((batch_size, block_size), device=times.device)
    target_mask = u < (1.0 - times.view(batch_size, 1))
    data[target_mask] = mask_token_id # random masking
    return data, target_mask

# data loader
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split, times=None):
    data = train_data if split == 'train' else val_data

    if not overfit_batch:
        ix = torch.randint(len(data) - block_size, (batch_size,)) # start index
    else:
        ix = torch.zeros((batch_size,), dtype=torch.int64)
        
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    
    if times is None:
        times = torch.rand((batch_size,)) * (1.0 - min_t) + min_t
    else:
        assert times.shape == (batch_size,)

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, times = x.pin_memory().to(device, non_blocking=True), \
            y.pin_memory().to(device, non_blocking=True), \
            times.pin_memory().to(device, non_blocking=True)
    else:
        x, y, times = x.to(device), y.to(device), times.to(device)
    return x, y, times


def calc_loss(X, Y, times, target_mask, infill_probs, num_ones_in_mask):
    # Use Flow Matching framework
    # Sample from probability path
    path_sample = prob_path.sample(x_0=X, x_1=Y, t=times)
    
    # Get model predictions
    logits = wrapped_model(path_sample.x_t, times)
    
    # Compute Flow Matching loss
    loss = flow_loss_fn(
        logits=logits,
        x_t=path_sample.x_t,
        x_1=path_sample.x_1,
        t=times
    )
    
    return loss



# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, times = get_batch(split)
            
            # Create source data (masked version)
            X_source = torch.full_like(X, mask_token_id)  # Start from all mask tokens
            with ctx:
                loss = calc_loss(X_source, Y, times, None, None, None)

            losses[k] = loss.item()
        out[split] = losses.mean() # train/val loss
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# @torch.no_grad()
# def sample_from_flow_model(num_samples=1, seq_length=None, num_steps=50):
#     """
#     Sample from the trained Flow Matching model
#     Args:
#         num_samples: number of sequences to generate
#         seq_length: length of sequences (default: block_size)
#         num_steps: number of diffusion steps
#     Returns:
#         generated sequences
#     """
#     if seq_length is None:
#         seq_length = block_size
        
#     model.eval()
    
#     # Start from all mask tokens
#     x_init = torch.full((num_samples, seq_length), mask_token_id, device=device)
    
#     # Sample using the solver
#     samples = solver.sample(
#         x_init=x_init,
#         step_size=1.0/num_steps,
#         return_intermediates=False,
#         verbose=True
#     )
    
#     model.train()
#     return samples


# logging
if wandb_log and master_process:
    import wandb
    if wandb_id == 'blank' :
        wandb.init(project=wandb_project, name=wandb_run_name, config=config, id=None,
            resume=is_repeat)
    else:
        wandb.init(project=wandb_project, name=wandb_run_name, config=config, id=wandb_id,
            resume=is_repeat)


# training loop
X, Y, times = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            try:
                times = 0.85 * torch.ones((batch_size,))

                X, Y, times = get_batch('train', times)
                
                X_source = torch.full_like(X, mask_token_id)
                with torch.no_grad():
                    print("running forward pass for logging...")
                    # Sample from path for visualization
                    path_sample = prob_path.sample(x_0=X_source, x_1=Y, t=times)
                    logits = wrapped_model(path_sample.x_t, times)  # (B, T, V)
                    
                predictions = torch.argmax(logits, dim=-1)
                samples = torch.multinomial(torch.softmax(logits, dim=-1).view(-1, meta_vocab_size), num_samples=1)[:, 0].view(batch_size, -1)
                
                # Calculate accuracy on the intermediate state
                matches = (samples == Y)  # (B, T)
                acc = matches.float().mean()
                
                # Get some token logits for monitoring
                first_logit_0 = logits[0, 0, 0]
                first_logit_1 = logits[0, 0, 1] if meta_vocab_size > 1 else torch.tensor(0.0)
                first_logit_2 = logits[0, 0, 2] if meta_vocab_size > 2 else torch.tensor(0.0)
                    
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "first_logit_0": first_logit_0,
                    "first_logit_1": first_logit_1,
                    "first_logit_2": first_logit_2,
                    "acc": acc,
                }, step=iter_num)
            except Exception as e:
                print(f"logging failed: {e}")

        def save_checkpoint(file_path):
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {file_path}")
                torch.save(checkpoint, file_path)

        save_checkpoint(os.path.join(out_dir, 'current_ckpt.pt'))

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            save_checkpoint(os.path.join(out_dir, 'best_ckpt.pt'))

    # decide whether to do self-conditioning loop
    if do_x1_sc and torch.rand(1, generator=shared_generator, device=device) < x1_sc_prob:
        do_self_conf_loop = True
    else:
        do_self_conf_loop = False

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        # Flow Matching approach - use clean source/target pairs
        X_source = torch.full_like(X, mask_token_id)  # Start from all mask tokens
        # start forward pass in GPU
        with ctx:
            loss = calc_loss(X_source, Y, times, None, None, None)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

        # while gpu forward pass, cpu can prepare the next batch
        X, Y, times = get_batch('train')

        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
        

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        try:
            wandb.log({"train/iter_loss": lossf}, step=iter_num)
        except Exception as e:
            print(e)
            
            
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()