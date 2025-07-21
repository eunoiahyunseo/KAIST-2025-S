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
from torch import nn, Tensor

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

# attempt to derive vocab_size from the dataset
data_dir = os.path.join('data', dataset)
meta_path = os.path.join(data_dir, 'meta.json')
assert os.path.exists(meta_path)

import json
with open(meta_path, 'r') as f:
    meta = json.load(f)
meta_vocab_size = meta['vocab_size']

print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    
# add mask_token 
eos_token_id = meta_vocab_size # 256

if dataset == 'sketch':
    meta_vocab_size += 1
    # add eos token
    mask_token_id = meta_vocab_size # 257

    meta_vocab_size += 1
    # add padding token    
    padding_token_id = meta_vocab_size # 258
    
    meta_vocab_size += 1 # total token


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 1
best_val_loss = 1e9
if dataset == 'sketch':
    use_coordinate_embedding = True
    n_embd = n_embd * 2

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, qk_layernorm=qk_layernorm,
                  do_x1_sc=do_x1_sc, proper_timestep_emb=proper_timestep_emb,
                  d3pm_loss_weighting=d3pm_loss_weighting, d3pm_loss_weighting_maxT=d3pm_loss_weighting_maxT, use_coordinate_embedding=use_coordinate_embedding,
                  eos_token_id=eos_token_id, mask_token_id=mask_token_id, pad_token_id=padding_token_id)

    

model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
model_args
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, attention_mask=None, **extras):
        logits = self.model._run_net(idx=x, time=t, attn_mask=attention_mask)
        return logits
        

scheduler = PolynomialConvexScheduler(n=1.0) 
prob_path = MixtureDiscreteProbPath(scheduler=scheduler)
wrapped_probability_denoiser = WrappedModel(model)

flow_loss_fn = MixturePathGeneralizedKL(path=prob_path)

if dataset == 'graph':
    # data loader
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
if dataset == 'sketch':
    # data loader
    train_data = np.load(os.path.join(data_dir, 'train.npz'), allow_pickle=True)
    val_data = np.load(os.path.join(data_dir, 'val.npz'), allow_pickle=True)


def _coordinate_transform(idx, padding_token=None, eos_token=None):
    """
    좌표 데이터를 [x1, y1, x2, y2, eos, eos, pad, pad, pad, pad] 형태에서 [(x1, x2, eos, pad, pad), (y1, y2, eos, pad, pad)] 형태로 변환
    idx: (batch_size, seq_len) - interleaved x,y coordinates with padding
    """
    b, t = idx.size()
    padding_token = 258
    eos_token = 256 # 257
    
    transformed_list = []
    transformed_attn_mask = []
    
    for batch_idx in range(b):
        sequence = idx[batch_idx]
        num_eos = 2
        non_pad_tokens = sequence[sequence != padding_token]
        
        num_coords = len(non_pad_tokens) - num_eos # not include (eos), (pad) tokens
        num_pads = t - len(non_pad_tokens)

        x_coords = non_pad_tokens[0:num_coords//2].tolist() + [eos_token] + [padding_token] * (num_pads // 2)
        y_coords = non_pad_tokens[num_coords//2:num_coords].tolist() + [eos_token] + [padding_token] * (num_pads // 2)
        transformed_x_attn_mask = [1] * (len(x_coords) - (num_pads // 2) - 1) + [2] + [0] * (num_pads // 2)
        transformed_y_attn_mask = [1] * (len(y_coords) - (num_pads // 2) - 1) + [2] + [0] * (num_pads // 2)
        transformed_list.append(x_coords + y_coords)
        transformed_attn_mask.append(transformed_x_attn_mask + transformed_y_attn_mask)

    return torch.tensor(transformed_list).to(idx.device), torch.tensor(transformed_attn_mask).to(idx.device)


def get_batch(split, times=None):
    data = train_data if split == 'train' else val_data
    
    if dataset == 'text8':
        if not overfit_batch:
            ix = torch.randint(len(data) - block_size, (batch_size,)) # start index
        else:
            ix = torch.zeros((batch_size,), dtype=torch.int64)
        x_1 = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        
    elif dataset == 'sketch':
        # NPZ 데이터에서 랜덤하게 키 선택
        data_keys = list(data.files)
        
        if not overfit_batch:
            # 랜덤하게 batch_size만큼 키 선택
            selected_keys = np.random.choice(data_keys, size=batch_size, replace=True)
        else:
            # overfit_batch일 때는 같은 키들을 반복 사용
            selected_keys = [data_keys[0]] * batch_size
        
        # 각 키에 대해 스케치 데이터 로드
        batch_sketches = []
        for key in selected_keys:
            sketch_tokens = np.array(data[key])  # (seq_len,) 형태
            sketch_tokens = sketch_tokens.reshape(-1)
            batch_sketches.append(sketch_tokens)
        
        # batch_sketches: (b, x, y)
        x_1, _ = collate_fn(batch_sketches)
        x_1, attention_mask = _coordinate_transform(x_1)

    if times is None:
        times = torch.rand((batch_size,)) * (1.0 - min_t) + min_t
    else:
        assert times.shape == (batch_size,)

    if device_type == 'cuda':
        x_1, times = x_1.pin_memory().to(device, non_blocking=True), \
            times.pin_memory().to(device, non_blocking=True)
        if dataset == 'sketch':
            attention_mask = attention_mask.pin_memory().to(device, non_blocking=True)
    else:
        x_1, times = x_1.to(device), times.to(device)
        if dataset == 'sketch':
            attention_mask = attention_mask.to(device)


    # sketch 데이터셋의 경우 attention_mask도 반환
    if dataset == 'sketch':
        return x_1, times, attention_mask
    else:
        return x_1, times

def collate_fn(batch_sketches):
    # 각 스케치의 유효한 길이 계산 (패딩 토큰 제외) + EOS 토큰 고려
    valid_lengths = []
    for sketch in batch_sketches:
        # 패딩 토큰이 아닌 부분의 길이 계산
        if dataset == 'sketch':
            # 마스크 토큰(256)이 아닌 토큰들의 개수 + EOS 토큰(2개)
            valid_len = len(sketch) - np.sum(sketch == mask_token_id)
            valid_lengths.append(max(2, valid_len + 2))  # +2 for EOS token
        else:
            valid_lengths.append(len(sketch))
    
    # 배치 내 최대 길이 결정 (block_size를 넘지 않도록)
    max_len = min(max(valid_lengths), block_size)
    
    
    padded_sequences = []
    attention_masks = []
    
    for i, sketch in enumerate(batch_sketches):
        sketch = sketch.astype(np.int64)
        
        # 유효한 토큰과 패딩 토큰 구분
        if dataset == 'sketch':
            # 마스크 토큰이 아닌 부분을 유효한 토큰으로 간주
            valid_mask = sketch != mask_token_id
            valid_tokens = sketch[valid_mask]
            
            # 유효한 토큰이 max_len-1보다 길면 자르기 (EOS 토큰 공간 확보)
            if len(valid_tokens) > max_len - 2:
                valid_tokens = valid_tokens[:max_len - 2]
            
            # EOS 토큰 pair 추가
            tokens_with_eos = np.concatenate([valid_tokens, [eos_token_id] * 2])
            
            # 패딩 추가
            num_padding = max_len - len(tokens_with_eos)
            if num_padding > 0:
                padded_seq = np.concatenate([tokens_with_eos, 
                                           np.full(num_padding, padding_token_id)])
            else:
                padded_seq = tokens_with_eos
                
            # 어텐션 마스크 생성 (유효한 토큰과 EOS는 1, 패딩은 0)
            mask = np.concatenate([np.ones(len(tokens_with_eos)), 
                                 np.zeros(num_padding)])
        else:
            # 다른 데이터셋의 경우 기존 로직 유지
            if len(sketch) > max_len:
                sketch = sketch[:max_len]
                
            num_padding = max_len - len(sketch)
            if num_padding > 0:
                padded_seq = np.concatenate([sketch, 
                                           np.full(num_padding, padding_token_id)])
            else:
                padded_seq = sketch
                
            mask = np.concatenate([np.ones(len(sketch)), 
                                 np.zeros(max(0, num_padding))])
        
        padded_sequences.append(padded_seq)
        attention_masks.append(mask)
    
    # 리스트를 numpy 배열로 변환 후 텐서로 변환
    padded_sequences = np.array(padded_sequences)
    attention_masks = np.array(attention_masks)
    
    return torch.tensor(padded_sequences, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.float)




def calc_loss(x_0, x_1, times, attention_mask=None):
    path_sample = prob_path.sample(x_0=x_0, x_1=x_1, t=times)
    # print('path_sample.x_t.shape', path_sample.x_t.shape, path_sample.x_t)
    
    # attention_mask를 모델에 전달
    logits = wrapped_probability_denoiser(path_sample.x_t, times, attention_mask=attention_mask)
    # print('logits', logits[0])
    loss = flow_loss_fn(
        logits=logits,
        x_t=path_sample.x_t,
        x_1=path_sample.x_1,
        attention_mask=attention_mask,
        t=times
    )

    return loss

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if dataset == 'sketch':
                x_1, times, attention_mask = get_batch(split)
                x_0 = torch.full_like(x_1, padding_token_id) # start from all mask token
                x_0[attention_mask.to(torch.bool)] = mask_token_id
                
                with ctx:
                    loss = calc_loss(x_0, x_1, times, attention_mask)
            else:
                x_1, times = get_batch(split)
                x_0 = torch.full_like(x_1, mask_token_id) # start from all mask token
                with ctx:
                    loss = calc_loss(x_0, x_1, times)

            losses[k] = loss.item()
        out[split] = losses.mean() # train/val loss
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

# 초기 배치 로드
if dataset == 'sketch':
    x_1, times, attention_mask = get_batch('train')
    # print('x1 check', x_1.shape)
    # print('x_1 check: ', x_1[0], x_1.shape)
    # print('attention_mask check:', attention_mask[0])
    # print('attention_mask check:', attention_mask)
    # print('check time', times[0])
else:
    x_1, times = get_batch('train')
    attention_mask = None

local_iter_num = 1
t0 = time.time()  # 시간 측정을 위한 초기값 

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            try:
                times_fixed = 0.85 * torch.ones((batch_size,))
                
                if dataset == 'sketch':
                    x_1_log, times_log, attention_mask_log = get_batch('train', times_fixed)
                    x_0_log = torch.full_like(x_1_log, mask_token_id)
                    
                    with torch.no_grad():
                        path_sample = prob_path.sample(x_0=x_0_log, x_1=x_1_log, t=times_log)
                        logits = wrapped_probability_denoiser(path_sample.x_t, times_log)
                        
                        # 어텐션 마스크 적용
                        mask_expanded = attention_mask_log.unsqueeze(-1).expand_as(logits)
                        masked_logits = logits * mask_expanded + (1 - mask_expanded) * (-1e9)
                        
                        predictions = torch.argmax(masked_logits, dim=-1)
                        samples = torch.multinomial(torch.softmax(masked_logits, dim=-1).view(-1, meta_vocab_size), num_samples=1)[:, 0].view(batch_size, -1)
                        
                        # 유효한 토큰에 대해서만 정확도 계산
                        matches = (samples == x_1_log) * attention_mask_log
                        acc = matches.sum() / attention_mask_log.sum() if attention_mask_log.sum() > 0 else 0.0
                else:
                    x_1_log, times_log = get_batch('train', times_fixed)
                    x_0_log = torch.full_like(x_1_log, mask_token_id)
                    
                    with torch.no_grad():
                        path_sample = prob_path.sample(x_0=x_0_log, x_1=x_1_log, t=times_log)
                        logits = wrapped_probability_denoiser(path_sample.x_t, times_log)
                        
                    predictions = torch.argmax(logits, dim=-1)
                    samples = torch.multinomial(torch.softmax(logits, dim=-1).view(-1, meta_vocab_size), num_samples=1)[:, 0].view(batch_size, -1)
                    
                    matches = (samples == x_1_log)
                    acc = matches.float().mean()
                
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "acc": acc,
                }, step=iter_num)
            except Exception as e:
                print(f"logging failed: {e}")

        def save_checkpoint(file_path):
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
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

    for micro_step in range(gradient_accumulation_steps):
        if dataset == 'sketch':
            x_0 = torch.full_like(x_1, padding_token_id) # start from all mask token
            x_0[attention_mask.to(torch.bool)] = mask_token_id
            # print('x_1: ', x_1[0])
            # print('x_0: ', x_0[0])
            with ctx:
                # print('in microstep: ', x_1.shape, x_1[0])
                loss = calc_loss(x_0, x_1, times, attention_mask)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

            # 다음 배치 준비
            x_1, times, attention_mask = get_batch('train')
        else:
            x_0 = torch.full_like(x_1, mask_token_id)  # Start from all mask tokens
            with ctx:
                loss = calc_loss(x_0, x_1, times)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

            # 다음 배치 준비
            x_1, times = get_batch('train')

        scaler.scale(loss).backward()

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    
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