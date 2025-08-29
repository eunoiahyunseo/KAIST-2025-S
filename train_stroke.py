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

from flow_model3 import GPT, GPTConfig

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

layout_data_dir = 'data/layout' #  directory should contain train.npz, val.npz, meta.json
stroke_data_dir = 'data/layout_stroke' #  directory should contain train.npz, val.npz, meta.json
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


bbox_vocab_size = 2500  # 1\~2500 for coordinates
stroke_vocab_size = 293 # Example size
padding_token_id = 0 # Let's use 0 for padding universally

# Create a unified vocabulary
# 0: pad
# 1 \~ 2500: bbox tokens
# 2501 \~ 2793: stroke tokens 293
# 2794: mask token

BBOX_OFFSET = 1
STROKE_OFFSET = BBOX_OFFSET + bbox_vocab_size # = 2501
meta_vocab_size = STROKE_OFFSET + stroke_vocab_size # = 2794
mask_token_id = meta_vocab_size # mask is 2794
meta_vocab_size += 1 # meta_vocab_size = 2795

source_distribution = 'mask' # 'mask' or 'uniform', determines how the x_0 is initialized

BBOX_TYPE_ID = 0
STROKE_TYPE_ID = 1
print('mask_token_id:', mask_token_id)

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



S = meta_vocab_size
B = batch_size
D = block_size

# mask distribution을 위한 초기 분포 설정
x0 = torch.full((S,), mask_token_id, dtype=torch.long)

scheduler = PolynomialConvexScheduler(n=1.0) 
prob_path = MixtureDiscreteProbPath(scheduler=scheduler)
wrapped_probability_denoiser = WrappedModel(model)
flow_loss_fn = MixturePathGeneralizedKL(path=prob_path, padding_token_id=padding_token_id)

# Bbox 데이터 로드
train_data_bbox = np.load(os.path.join(layout_data_dir, 'train.npz'), allow_pickle=True)
val_data_bbox = np.load(os.path.join(layout_data_dir, 'val.npz'), allow_pickle=True)

# Stroke 데이터 로드
train_data_stroke = np.load(os.path.join(stroke_data_dir, 'train.npz'), allow_pickle=True)
val_data_stroke = np.load(os.path.join(stroke_data_dir, 'val.npz'), allow_pickle=True)


def get_batch(split, times=None):
    # 1. split에 따라 올바른 데이터 소스 쌍을 선택합니다.
    bbox_data = train_data_bbox if split == 'train' else val_data_bbox
    stroke_data = train_data_stroke if split == 'train' else val_data_stroke

    # 두 .npz 파일은 동일한 키를 가지고 있다고 가정합니다.
    data_keys = list(bbox_data.files)

    # 2. 배치에 사용할 키를 무작위로 선택합니다.
    if not overfit_batch:
        selected_keys = np.random.choice(data_keys, size=batch_size, replace=True)
    else:
        selected_keys = [data_keys[0]] * batch_size

    batch_x1 = []
    batch_token_types = []

    for key in selected_keys:
        # 3. 동일한 키로 각 소스에서 bbox와 stroke 데이터를 가져옵니다.
        bbox_tokens_np = bbox_data[key]
        stroke_tokens_np = stroke_data[key]
        # print(bbox_tokens_np, stroke_tokens_np)
        # 4. 텐서로 변환하고 각 토큰 유형에 맞는 오프셋을 더해줍니다.
        bbox_tokens = torch.tensor(bbox_tokens_np, dtype=torch.long) + BBOX_OFFSET
        stroke_tokens = torch.tensor(stroke_tokens_np, dtype=torch.long) + STROKE_OFFSET
        
        # 5. bbox와 stroke 시퀀스를 하나로 합치고, 토큰 타입도 생성합니다.
        x1_seq = torch.cat([bbox_tokens, stroke_tokens])
        
        bbox_types = torch.full_like(bbox_tokens, BBOX_TYPE_ID)
        stroke_types = torch.full_like(stroke_tokens, STROKE_TYPE_ID)
        token_types_seq = torch.cat([bbox_types, stroke_types])

        # 6. 시퀀스 길이를 block_size에 맞게 패딩 또는 절단합니다.
        seq_len = len(x1_seq)
        if seq_len > block_size:
            x1_seq = x1_seq[:block_size]
            token_types_seq = token_types_seq[:block_size]
        elif seq_len < block_size:
            padding_needed = block_size - seq_len
            x1_seq = F.pad(x1_seq, (0, padding_needed), 'constant', padding_token_id)
            # 패딩 부분의 타입은 0 (BBOX_TYPE_ID와 동일)으로 설정하거나 별도의 PAD_TYPE_ID를 사용
            token_types_seq = F.pad(token_types_seq, (0, padding_needed), 'constant', 0)
        
        batch_x1.append(x1_seq)
        batch_token_types.append(token_types_seq)

    # 7. 최종 배치를 스택하고 디바이스로 이동시킵니다.
    x_1 = torch.stack(batch_x1, dim=0)
    token_types = torch.stack(batch_token_types, dim=0)

    if times is None:
        times = torch.rand((batch_size,)) * (1.0 - min_t) + min_t
    else:
        assert times.shape == (batch_size,)

    if device_type == 'cuda':
        x_1, times, token_types = x_1.pin_memory().to(device, non_blocking=True), \
                                  times.pin_memory().to(device, non_blocking=True), \
                                  token_types.pin_memory().to(device, non_blocking=True)
    else:
        x_1, times, token_types = x_1.to(device), times.to(device), token_types.to(device)
        
    return x_1, times, token_types


def calc_loss(x_0, x_1, times, token_tpypes=None):
    path_sample = prob_path.sample(x_0=x_0, x_1=x_1, t=times)
    # print(path_sample.x_t[0])
    # x0: [c c c m m m 0 0 0 0]
    # x1: [c c c s s s 0 0 0 0] 
    # xt: [c c c s m m 0 0 0 0] <-- denoise to [c c c s s s 0 0 0 0]
    logits = wrapped_probability_denoiser(x_t=path_sample.x_t, times=times, token_types=token_tpypes)

    flow_loss = flow_loss_fn(
        logits=logits,
        x_t=path_sample.x_t,
        x_1=path_sample.x_1,
        t=times
    )
    return flow_loss


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


x_1, times, token_types = get_batch('val')
print('x_1[0]:',  x_1[0])
print('x_1[1]:',  x_1[1])
print('x_1[2]:',  x_1[2])
# print('x_1 shape:', x_1.shape, x_1[0])
# print(token_types.shape, token_types[0])
# assert False
local_iter_num = 1
t0 = time.time()
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        if iter_num % eval_interval == 0 and iter_num > 0:
            print(f"\n--- [Validation] Running one-step prediction at iteration {iter_num} ---")
            model.eval()
            with torch.no_grad():
                # 1. 검증 데이터 배치 가져오기
                x1_val, times_val, token_types_val = get_batch('val')
                
                # 2. 첫 번째 샘플을 사용하여 비교
                x1_sample = x1_val[0]
                time_sample = times_val[0]
                token_types_sample = token_types_val[0]

                # 3. 노이즈가 섞인 입력 x_t 만들기 (Forward Process)
                #    DFM의 목표는 x_0(조건)과 x_1(정답)을 보고 x_t를 예측하는 것이 아니라,
                #    x_t를 보고 x_1을 예측하는 것이므로, x_0는 정답(x_1)으로 설정
                path_sample = prob_path.sample(x_0=x1_sample.unsqueeze(0), x_1=x1_sample.unsqueeze(0), t=time_sample.unsqueeze(0))
                x_t_sample = path_sample.x_t
                
                # 4. 모델에 x_t를 통과시켜 로짓 예측
                predicted_logits = model(x_t=x_t_sample, times=time_sample.unsqueeze(0), token_types=token_types_sample.unsqueeze(0))
                
                # 5. argmax로 가장 확률 높은 토큰 예측
                predicted_tokens = torch.argmax(predicted_logits, dim=-1).squeeze(0)
                
                # 6. 원본 토큰과 예측 토큰 비교 출력
                stroke_mask = token_types_sample.eq(STROKE_TYPE_ID)
                original_stroke_tokens = x1_sample[stroke_mask].cpu().numpy()
                predicted_stroke_tokens = predicted_tokens[stroke_mask].cpu().numpy()
                
                print(f"Time (t): {time_sample.item():.4f}")
                print(f"Original Stroke Tokens: \n{original_stroke_tokens}")
                print(f"Predicted (argmax) Tokens: \n{predicted_stroke_tokens}\n")

    

    optimizer.zero_grad()
    for micro_step in range(gradient_accumulation_steps):
        x_1, times, token_types = get_batch('val')
        x_0 = torch.full_like(x_1, mask_token_id)
        condition_mask = token_types.eq(BBOX_TYPE_ID)
        x_0[condition_mask] = x_1[condition_mask]
        x_0[x_1.eq(padding_token_id)] = padding_token_id

        # x0: [c c c m m m 0 0 0 0]
        # x1: [c c c s s s 0 0 0 0]

        with ctx:
            loss = calc_loss(x_0, x_1, times, token_types)
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


print(f"--- Training Finished ---")
final_checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'iter_num': iter_num,
    'best_val_loss': best_val_loss,
    'config': config,
}

out_dir = './checkpoints/stroke_dfm'  # Ensure the output directory is set correctly
torch.save(final_checkpoint, os.path.join(out_dir, 'final_ckpt.pt'))
print(f"✅ Final checkpoint saved to {os.path.join(out_dir, 'final_ckpt.pt')}")