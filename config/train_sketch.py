out_dir = './output/sketch_apple_c_coupling'
eval_interval = 200
eval_iters = 100

log_interval = 2

data_dir = './data/full_sketch'

warm_start_ckpt = None
init_from = 'scratch'
resume_dir = None

always_save_checkpoint = False

wandb_log = True
wandb_project = 'KAIST'
wandb_run_name = 'sketch'
wandb_id = 'apple_c_coupling'

is_repeat = False

gradient_accumulation_steps = 8
batch_size = 32
block_size = 45
overfit_batch = False

n_layer = 12
n_head = 16
n_embd = 1024
dropout = 0
qk_layernorm = True

learning_rate = 1e-4
max_iters = 30000
lr_decay_iters = 15000
min_lr = 1e-5
beta2 = 0.99

warmup_iters = 1000

min_t = 0.0
source_distribution='mask'