out_dir = './output/sketch'
eval_interval = 500
eval_iters = 200
log_interval = 10

data_dir = './data/sketch'

warm_start_ckpt = None
init_from = 'scratch'
resume_dir = None

always_save_checkpoint = False

wandb_log = True
wandb_project = 'KAIST'
wandb_run_name = 'graph_dfm'
wandb_id = 'knu-prmi-org'
is_repeat = False

dataset = 'sketch'
gradient_accumulation_steps = 4
batch_size = 12
block_size = 120
overfit_batch = False

n_layer = 12
n_head = 16
n_embd = 1024
dropout = 0.1
qk_layernorm = True
proper_timestep_emb = False
do_x1_sc = False
x1_sc_prob = 0.5

model_type = 'flow'

learning_rate = 1e-4
max_iters = 3000
lr_decay_iters = 10000
min_lr = 1e-5
beta2 = 0.99

warmup_iters = 1000

min_t = 0.0