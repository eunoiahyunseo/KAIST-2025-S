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
from torch import nn, Tensor

# These configs will be overridden by the config file and so their values here do not matter.
out_dir = 'out'

run_name = 'gpt2' # 'run' + str(time.time())

dataset = 'text8'
batch_size = 64
block_size = 256

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
qk_layernorm = True
do_x1_sc = False

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
data_dir = '/path/to/datasets/text8'

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
model_type = 'flow'
source_distribution = 'mask'
ckpt_path = 'out/ckpt.pt'

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


from flow_model3 import GPT, GPTConfig

# Flow Matching imports
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler, CorrectorScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper

# attempt to derive vocab_size from the dataset
data_dir = os.path.join('data', dataset)
# meta_path = os.path.join(data_dir, 'meta.json')
# assert os.path.exists(meta_path)

# import json
# with open(meta_path, 'r') as f:
#     meta = json.load(f)

# meta_vocab_size = meta['vocab_size'] + 1 if source_distribution == 'mask' else meta['vocab_size']

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

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 1
best_val_loss = 1e9

device_type = 'cuda'
device = 'cuda:0'


def load_model(ckpt_path):
    # resume training from a checkpoint.
    print(f"Loading network from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    print('model_args: ', model_args)
    print('meta_vocab_size: ', meta_vocab_size)
    model_args['vocab_size'] = meta_vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    
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

class ConditionalModelWrapper(ModelWrapper):
    def __init__(self, model, condition_tokens, token_types):
        super().__init__(model)
        # condition_tokens = [m m m | m m m | m m m m]
        self.condition_tokens = condition_tokens
        # token_types = [0 0 0 | 1 1 1 | 0 0 0 0]
        self.token_types = token_types
        # 조건(bbox)에 해당하는 부분은 True인 마스크 생성
        # condition_mask = [True True True | False False False | True True True True]
        self.condition_mask = (self.token_types == BBOX_TYPE_ID)

    def forward(self, x_t: torch.Tensor, times: torch.Tensor, **extras):
        # 모델에 입력을 넣기 전, 조건 부분을 깨끗한 원본으로 강제 교체
        x_t_conditioned = torch.where(self.condition_mask, self.condition_tokens, x_t)

        
        # 이제 모델은 항상 깨끗한 bbox를 보게 됨
        return F.softmax(self.model(x_t=x_t_conditioned, times=times, token_types=self.token_types), dim=-1)


S = meta_vocab_size
B = batch_size
D = block_size
coupling = 'C'

temperature = 1

x0 = torch.zeros((S)) + mask_token_id



torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
print(torch.__version__)
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# write an empty file to store the samples eventually
with open(os.path.join(samples_dir, 'samples.txt'), 'w') as f:
    pass

assert total_samples % B == 0

val_data_bbox = np.load(os.path.join(data_dir, 'val.npz'), allow_pickle=True)
val_data_keys = val_data_bbox.files

# --- 샘플링 시작 ---
print(f"Starting conditional sampling for {total_samples} samples...")
all_generated_samples = []
generated_stroke_dict = {}

with torch.no_grad():
    with ctx:
        for i in range(0, min(total_samples, len(val_data_keys))):
            print(f"--- Generating sample {i+1}/{total_samples} ---")
            
            # 1. 조건으로 사용할 bbox 토큰 시퀀스 로드
            key = val_data_keys[i]
            bbox_tokens_np = val_data_bbox[key]
            bbox_tokens = torch.tensor(bbox_tokens_np, dtype=torch.long, device=device) + BBOX_OFFSET
            num_bbox = len(bbox_tokens)
            
            num_stroke_to_gen = num_bbox
            num_padding = block_size - num_bbox - num_stroke_to_gen

            
            stroke_mask_part = torch.full((num_stroke_to_gen,), mask_token_id, device=device, dtype=torch.long)
            padding_part = torch.full((num_padding,), padding_token_id, device=device, dtype=torch.long)
            CLEAN_REFERENCE_TEMPLATE = torch.cat([bbox_tokens, stroke_mask_part, padding_part]).unsqueeze(0)



            # x0 [m m m | m m m | m m m m m]
            x_init = torch.full((1, block_size), mask_token_id, device=device, dtype=torch.long)


            # 3. 토큰 타입 시퀀스 생성
            # [0 0 0 | 1 1 1 | 0 0 0 0 0]
            bbox_types = torch.full((num_bbox,), BBOX_TYPE_ID, device=device, dtype=torch.long)
            stroke_types = torch.full((num_stroke_to_gen,), STROKE_TYPE_ID, device=device, dtype=torch.long)


            etc = torch.full((block_size - num_bbox * 2,), 0, device=device, dtype=torch.long)
            token_types = torch.cat([bbox_types, stroke_types, etc]).unsqueeze(0)

            # 4. 조건부 모델 래퍼 생성
            # print('x_init shape:', x_init.shape, token_types.shape)
            wrapped_sampler = ConditionalModelWrapper(model, condition_tokens=CLEAN_REFERENCE_TEMPLATE, token_types=token_types)

            # 5. Solver 준비 (매번 새로운 래퍼로 초기화)
            solver = MixtureDiscreteEulerSolver(
                model=wrapped_sampler,
                path=MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=1.0)),
                vocabulary_size=meta_vocab_size
            )

            # 6. 샘플링 실행 (x_init 자체가 조건이므로 여기서 다시 전달)
            samples = solver.sample(x_init=x_init, step_size=dt) # num_steps는 적절히 조절
            
            # 생성된 전체 시퀀스 저장
            generated_sequence = samples.squeeze(0).cpu().numpy()

            

            start_index = num_bbox
            end_index = num_bbox + num_bbox # bbox 개수와 동일한 개수
            
            # 추출된 스트로크 토큰 (오프셋을 다시 빼서 원래 토큰 ID로 복원)
            extracted_stroke_tokens = generated_sequence[start_index:end_index] - STROKE_OFFSET
            
            # 8. 결과를 딕셔너리에 저장
            generated_stroke_dict[key] = extracted_stroke_tokens


            all_generated_samples.append(generated_sequence)
            print(f"Generated sequence for {key}: {generated_sequence}")

            # (선택) 생성된 샘플을 텍스트 파일로 저장
            with open(os.path.join(samples_dir, 'samples.txt'), 'a') as f:
                f.write(f"# Sample for {key}\n")
                f.write(' '.join(map(str, generated_sequence)) + '\n\n')

### [핵심 수정] 추출된 스트로크 토큰만 .npz 파일로 저장 ###
print("\n--- Saving generated stroke tokens to NPZ file ---")
save_path = os.path.join(samples_dir, 'generated_strokes.npz')
np.savez_compressed(save_path, **generated_stroke_dict)
print(f"✅ Generated stroke tokens saved to {save_path}")