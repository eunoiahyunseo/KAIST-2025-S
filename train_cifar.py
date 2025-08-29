"""
modified with: eunoia_hyunseo heart2002101@knu.ac.kr
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import yaml
import logging
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
from torch import nn, Tensor
from torch.nn.modules import Module
from tqdm import tqdm
from calc_fid import calculate_fid

from flow_matching.path.scheduler.corrector_scheduler import CorrectorScheduler
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
import shutil
from cifar_model.cifar_model_config import instantiate_model
from cifar_model.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from cifar_model.transform_cifar10 import get_train_transform
from cifar_model.ema import EMA
from cifar_model.load_and_save import save_model, load_model
from torchvision.utils import save_image
from torch.utils.data import Subset # Subset을 import합니다.

from flow_model2 import GPT, GPTConfig

# Flow Matching imports
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL
# from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.solver.discrete_solver2 import MixtureDiscreteEulerSolver
from Unet import UNetModel
from Unet2 import AttentionUNet
from torchmetrics.aggregation import MeanMetric
from torchmetrics.image.fid import FrechetInceptionDistance


import matplotlib.pyplot as plt


out_dir = 'out_cifar10'
eval_interval = 50
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'


wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())

wandb_id = 'blank'
is_repeat = False

dataset = 'layout'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 128  # layout에 맞게 축소 (bounding box 개수에 맞춰 조정)
overfit_batch = False

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
qk_layernorm = False

learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
backend = 'nccl' # 'nccl', 'gloo', etc.
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
max_t = 0.999

bonus_seed_offset = 0
use_coordinate_embedding = False
mask_token_id = None

coupling = 'C'

fid_eval_interval = 50
num_fid_samples = 100
num_vis_samples = 16
generation_steps = 100
dt = 1.0 / generation_steps

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


cifar10_data_dir = 'data/cifar10' # CIFAR-10 데이터 저장 경로
source_distribution = 'mask' # 'mask' or 'uniform', determines how the x_0 is initialized

iter_num = 1
best_val_loss = 1e9

# cifar 10 (temp overload)
# 0~255 vocab, mask: 256
block_size = 32 * 32 * 3
cifar_vocab_size = 256
mask_token_id = cifar_vocab_size
meta_vocab_size = cifar_vocab_size + 1
logger = logging.getLogger(__name__)

use_ema = True

model = instantiate_model('cifar10', is_discrete=True, use_ema=use_ema)
model.to(device)
print(f"Model parameter count: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, betas=(beta1, beta2))

class SamplingWrappedModel(ModelWrapper):
    def __init__(self, model: Module):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(
            self, x: torch.Tensor, t: torch.Tensor, cfg_scale: float, label: torch.Tensor
            ):
        t = torch.zeros(x.shape[0], device=x.device) + t
        if cfg_scale != 0.0:
            with torch.cuda.amp.autocast(), torch.no_grad():
                conditional = self.model(x, t, extra={"label": label})
                conditional_free = self.model(x, t, extra={})
            result = (1.0 + cfg_scale) * conditional - cfg_scale * conditional_free
        else:
            with torch.cuda.amp.autocast(), torch.no_grad():
                result = self.model(x, t, extra={"label": label})

        self.nfe_counter += 1
        return torch.softmax(result.to(dtype=torch.float32), dim=-1)
    
    def reset_nfe_counter(self):
        self.nfe_counter = 0
    
    def get_nfe(self) -> int:
        return self.nfe_counter
        
        

corrector_scheduler = CorrectorScheduler(alpha_param=12.0, a=2.0, b=0.2)
scheduler = PolynomialConvexScheduler(n=3.0)
prob_path = MixtureDiscreteProbPath(scheduler=scheduler)
sampling_denoiser = SamplingWrappedModel(model)
flow_loss_fn = MixturePathGeneralizedKL(path=prob_path)



train_transform = get_train_transform()
train_set = torchvision.datasets.CIFAR10(root=cifar10_data_dir, train=True, download=True, transform=train_transform)
# val_set = torchvision.datasets.CIFAR10(root=cifar10_data_dir, train=False, download=True, transform=transform)
data_loader_train = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
# data_loader_val = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
iter_train = iter(data_loader_train)
# iter_val = iter(data_loader_val)

num_train_samples = 1000
indices = [0] * num_train_samples
subset_train_set = Subset(train_set, indices)
sub_data_loader_train = torch.utils.data.DataLoader(
    subset_train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

        
# def calc_loss(x_0, x_1, times):
#     path_sample = prob_path.sample(x_0=x_0, x_1=x_1, t=times)
#     # (B, C, H, W) -> (B, S, V) logit
#     logits_bsv = training_denoiser(x_t=path_sample.x_t, t=times, extra=conditioning)

#     x_1_flat = rearrange(path_sample.x_1, 'b c h w -> b (c h w)')
#     x_t_flat = rearrange(path_sample.x_t, 'b c h w -> b (c h w)')

#     flow_loss = flow_loss_fn(
#         logits=logits_bsv,
#         x_t=x_t_flat,
#         x_1=x_1_flat,
#         t=times
#     )    
#     return flow_loss



if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config, id=wandb_id,resume=is_repeat)


total_epochs = 1000
lr_schedule = torch.optim.lr_scheduler.ConstantLR(
    optimizer,
    total_iters=total_epochs,
    factor=1.0
)

class_drop_prob = 1.0
cfg_scale = 0.0
fid_samples = 32
sampling_dtype = "float32"
loss_scaler = NativeScaler()

for epoch in range(1, total_epochs + 1):
    model.train(True)
    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    print('epoch:', epoch)
    pbar = tqdm(data_loader_train, desc=f"Epoch {epoch}/{total_epochs}")


    for data_iter_step, (samples, labels) in enumerate(pbar):
        if data_iter_step % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
            batch_loss.reset()

        samples = samples.to(device)
        labels = labels.to(device)

        if torch.rand(1) < class_drop_prob:
            conditioning = {}
        else:
            conditioning = {"label": labels}

        samples = (samples * 255.0).to(torch.long)
        t = torch.torch.rand(samples.shape[0]).to(device)

        x_0 = (
            torch.zeros(samples.shape, dtype=torch.long, device=device) + mask_token_id
        )

        path_sample = prob_path.sample(x_0=x_0, x_1=samples, t=t)

        logits = model(path_sample.x_t, t=t, extra=conditioning)
        loss = torch.nn.functional.cross_entropy(logits.reshape([-1, 257]), samples.reshape([-1])).mean()

        loss_value = loss.item()
        batch_loss.update(loss)
        epoch_loss.update(loss)

        loss /= gradient_accumulation_steps
        apply_update = (iter_num + 1) % gradient_accumulation_steps == 0


        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=apply_update,
        )

        if apply_update and isinstance(model, EMA):
            model.update_ema()

        pbar.set_postfix(loss=f"{loss_value:.4f}")
        lr = optimizer.param_groups[0]['lr']
    
    lr_schedule.step()
    stats = {"loss": float(epoch_loss.compute().detach().cpu())}

    log_stats = {
        **{f"train_{k}": v for k, v in stats.items()},
        "epoch": epoch,
    }

    if epoch % 5 == 0:
        save_model(
            output_dir=out_dir,
            epoch=epoch,
            model=model,
            model_without_ddp=model,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            loss_scaler=loss_scaler if dtype != 'float16' else None, # float16인 경우 None으로 설정
        )


    cfg_scaled_model = SamplingWrappedModel(model=model)
    cfg_scaled_model.train(False)


    source_distribution_p = torch.zeros(meta_vocab_size, device=device)
    source_distribution_p[mask_token_id] = 1.0

    solver = MixtureDiscreteEulerSolver(
        path=prob_path,
        model=cfg_scaled_model,
        vocabulary_size=meta_vocab_size,
        source_distribution_p=source_distribution_p,
    )

    fid_metric = FrechetInceptionDistance(normalize=True).to(device)
    snapshots_saved = False

    if out_dir is not None:
        (Path(out_dir) / "snapshots").mkdir(parents=True, exist_ok=True)

    num_synthetic = 0
    for data_iter_step, (samples, labels) in enumerate(data_loader_train):
        samples = samples.to(device)
        labels = labels.to(device)
        # fid_metric.update(samples, real=True)

        if num_synthetic < fid_samples:
            cfg_scaled_model.reset_nfe_counter()
            x_0 = (
                torch.zeros(samples.shape, dtype=torch.long, device=device) + mask_token_id
            )
            synthetic_samples = solver.sample(
                x_init=x_0,
                step_size=1.0/generation_steps,
                div_free=corrector_scheduler,
                label=labels,
                cfg_scale=cfg_scale,
                verbose=False)
            
            synthetic_samples = synthetic_samples.to(torch.float32) / 255.0

            if num_synthetic + synthetic_samples.shape[0] > fid_samples:
                synthetic_samples = synthetic_samples[: fid_samples - num_synthetic]

            # fid_metric.update(synthetic_samples, real=False)
            num_synthetic += synthetic_samples.shape[0]

            if not snapshots_saved and out_dir is not None:
                save_image(
                    synthetic_samples,
                    fp=Path(out_dir)
                    / "snapshots"
                    / f"{epoch}_{data_iter_step}-samples.png",
                )
                snapshots_saved = True
                print('save snapshot')
        else:
            break

        # eval_stats = {"fid": float(fid_metric.compute().detach().cpu())}
        # print(eval_stats)
        # log_stats.update({f"eval_{k}": v for k, v in eval_stats.items()})



            






    

    