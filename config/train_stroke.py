out_dir = './checkpoints/stroke_dfm'
eval_interval = 500
eval_iters = 100

log_interval = 2

layout_data_dir = './data/layout'
stroke_data_dir = './data/layout_stroke'

warm_start_ckpt = None
init_from = 'scratch'
resume_dir = None

always_save_checkpoint = False

wandb_log = False
wandb_project = 'KAIST'
wandb_run_name = 'sketch'
wandb_id = 'apple_c_coupling'

is_repeat = False

gradient_accumulation_steps = 1
batch_size = 10
block_size = 20
overfit_batch = False

n_layer = 6
n_head = 8
n_embd = 512
dropout = 0
qk_layernorm = True

# learning_rate = 1e-4
# max_iters = 30000
# lr_decay_iters = 15000
# min_lr = 1e-5
# beta2 = 0.99
learning_rate = 1e-4
max_iters = 3000
lr_decay_iters = 2000
min_lr = 1e-5
beta2 = 0.99

warmup_iters = 1000

min_t = 0.0
source_distribution='mask'