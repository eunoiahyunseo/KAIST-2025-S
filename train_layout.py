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
from pathlib import Path
from torch import nn, Tensor

from flow_model2 import GPT, GPTConfig

# Flow Matching imports
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.solver import MixtureDiscreteEulerSolver

import matplotlib.pyplot as plt


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
dataset = 'layout'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 128  # layout에 맞게 축소 (bounding box 개수에 맞춰 조정)
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

data_dir = 'data/layout' #  directory should contain train.npz, val.npz, meta.json
warm_start_ckpt = None
resume_dir = None

model_type = 'flow' # flow, d3pm

d3pm_loss_weighting = False
d3pm_loss_weighting_maxT = 1000
timesteps = 1000

min_t = 0.0

bonus_seed_offset = 0
use_coordinate_embedding = False
mask_token_id = None

coupling = 'C'

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and (isinstance(v, (int, float, bool, str)) or v is None) ]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

if resume_dir is None:
    if wandb_id == 'blank':
        out_dir = os.path.join(out_dir, time.strftime('%Y-%m-%d-%H-%M-%S') + '_' + wandb_run_name)
    else:
        out_dir = os.path.join(out_dir, str(wandb_id) + '_' + wandb_run_name)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

else:
    out_dir = resume_dir

assert (resume_dir is not None) == is_repeat


shared_generator = torch.Generator(device).manual_seed(42) # for use when we want the random numbers to be the same across processes

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Layout specific configuration
max_bbox_value = 50*50  # bounding box 좌표의 최대값 (이미지 크기에 따라 조정)
num_bbox_tokens = max_bbox_value + 1  # 0~512까지의 값들
padding_token_id = 0
mask_token_id = num_bbox_tokens  # padding 다음에 mask token

source_distribution = 'mask' # 'mask' or 'uniform', determines how the x_0 is initialized
meta_vocab_size = num_bbox_tokens + 1  # padding token + bbox tokens

# attempt to derive vocab_size from the dataset
# For layout, we don't need meta.json as we define vocab size based on bbox coordinates
print(f"Using layout dataset with vocab_size: {meta_vocab_size}")
print(f"Max bbox value: {max_bbox_value}, Padding token: {padding_token_id}, Mask token: {mask_token_id}")

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 1
best_val_loss = 1e9

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=meta_vocab_size, dropout=dropout, qk_layernorm=qk_layernorm,
                  mask_token_id=mask_token_id, pad_token_id=padding_token_id)


gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

class WrappedModel(ModelWrapper):
    def forward(self, x_t: torch.Tensor, times: torch.Tensor, **extras):
        return self.model(x_t=x_t, times=times, **extras)

class WrappedSamperModel(ModelWrapper):
    def __init__(self, model, temperature: float = 1.0): # add temperature 
        super().__init__(model)
        self.temperature = temperature

    def forward(self, x_t: torch.Tensor, times: torch.Tensor, **extras):
        return F.softmax(self.model(x_t=x_t, times=times), dim=-1)



S = meta_vocab_size
B = batch_size
D = block_size

# mask distribution을 위한 초기 분포 설정
x0 = torch.full((S,), mask_token_id, dtype=torch.long)

scheduler = PolynomialConvexScheduler(n=1.0) 
prob_path = MixtureDiscreteProbPath(scheduler=scheduler)
wrapped_probability_denoiser = WrappedModel(model)
wrapped_sampler_probability_denoiser = WrappedSamperModel(model)
flow_loss_fn = MixturePathGeneralizedKL(path=prob_path)
solver = MixtureDiscreteEulerSolver(
    model=wrapped_sampler_probability_denoiser,
    path=prob_path,
    vocabulary_size=meta_vocab_size,
    source_distribution_p=x0.to(device)
)

train_data = np.load(os.path.join(data_dir, 'train.npz'), allow_pickle=True)
val_data = np.load(os.path.join(data_dir, 'val.npz'), allow_pickle=True)


def get_batch(split, times=None):
    data = train_data if split == 'train' else val_data
    data_keys = list(data.files)

    if not overfit_batch:
        selected_keys = np.random.choice(data_keys, size=batch_size, replace=True)
    else:
        selected_keys = [data_keys[0]] * batch_size
    
    batch_layouts = []
    for key in selected_keys:
        layout_data = data[key]  # .item()으로 numpy array에서 리스트 추출
        
        layout_tokens = torch.tensor(layout_data, dtype=torch.long)
        
        if len(layout_tokens) > block_size:
            layout_tokens = layout_tokens[:block_size]
        elif len(layout_tokens) < block_size:
            padding_needed = block_size - len(layout_tokens)
            layout_tokens = torch.cat([layout_tokens, torch.full((padding_needed,), padding_token_id, dtype=torch.long)])
        batch_layouts.append(layout_tokens)

    batch_layouts = torch.stack(batch_layouts, dim=0)
    x_1 = batch_layouts

    if times is None:
        times = torch.rand((batch_size,)) * (1.0 - min_t) + min_t
    else:
        assert times.shape == (batch_size,)

    if device_type == 'cuda':
        x_1, times = x_1.pin_memory().to(device, non_blocking=True), \
            times.pin_memory().to(device, non_blocking=True)
    else:
        x_1, times = x_1.to(device), times.to(device)
    
    return x_1, times
        


def calc_loss(x_0, x_1, times):
    path_sample = prob_path.sample(x_0=x_0, x_1=x_1, t=times)
    # print(path_sample.x_t[0])
    logits = wrapped_probability_denoiser(x_t=path_sample.x_t, times=times)

    flow_loss = flow_loss_fn(
        logits=logits,
        x_t=path_sample.x_t,
        x_1=path_sample.x_1,
        t=times
    )    

    return flow_loss

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_1, times = get_batch(split)
            
            # mask distribution 사용 - 모든 토큰을 mask token으로 초기화
            x_0 = torch.full_like(x_1, mask_token_id)
            # padding token 위치는 그대로 유지
            x_0[x_1.eq(padding_token_id)] = padding_token_id
            
            with ctx:
                loss = calc_loss(x_0, x_1, times)

            losses[k] = loss.item()

        out[split] = losses.mean()
    model.train()
    return out



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


# logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config, id=wandb_id,resume=is_repeat)


x_1, times = get_batch('train')
print('x_1 shape:', x_1.shape, x_1[0])

local_iter_num = 1
t0 = time.time()
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    

    optimizer.zero_grad()
    for micro_step in range(gradient_accumulation_steps):
        x_1, times = get_batch('train')
        # print(x_1[0])

        # mask distribution 사용 - 모든 토큰을 mask token으로 초기화
        x_0 = torch.full_like(x_1, mask_token_id)
        # padding token 위치는 그대로 유지
        x_0[x_1.eq(padding_token_id)] = padding_token_id

        with ctx:
            loss = calc_loss(x_0, x_1, times)
            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        

    scaler.step(optimizer)
    scaler.update()
    model.zero_grad()
    
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
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