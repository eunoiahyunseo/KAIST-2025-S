
out_dir = './output/graph'
ckpt_path = './output/graph/knu-prmi-org_base/best_ckpt.pt'
data_dir = './data/graph'

run_name = 'base'

dataset = 'graph'
batch_size = 10
block_size = 400 # context of up to 256 previous characters

n_layer = 6
n_head = 6
n_embd = 768
dropout = 0.0
qk_layernorm = True
do_x1_sc = True

total_samples = 10
dt = 0.001
max_t = 0.98
argmax_final = True
noise = 15.0
x1_temp = 1.0
use_different_x1_sc_temp = False
x1_sc_temp = 1.0
ignore_x1_sc = False

do_purity_sampling = False
purity_temp = 1.0

model_type = 'flow'