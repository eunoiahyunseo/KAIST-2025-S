out_dir = './checkpoints/mnist'
eval_iters = 100

eval_interval = 100
fid_eval_interval = 100000
num_fid_samples = 64
num_vis_samples = 10
generation_steps = 100

log_interval = 2

warm_start_ckpt = None
init_from = 'scratch'
resume_dir = None

always_save_checkpoint = False

wandb_log = True
wandb_project = 'KAIST'
wandb_run_name = 'mnist-dfm'
wandb_id = 'guidance'

is_repeat = False

gradient_accumulation_steps = 1
batch_size = 64
block_size = 28 * 28
overfit_batch = False

n_layer = 6
n_head = 8
n_embd = 512
dropout = 0
qk_layernorm = True

learning_rate = 1e-4
max_iters = 200000
lr_decay_iters = 100000
min_lr = 1e-5
beta2 = 0.999

warmup_iters = 2000

min_t = 0.0
source_distribution='mask'

