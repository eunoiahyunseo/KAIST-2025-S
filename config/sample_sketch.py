
out_dir = './output/sketch'
ckpt_path = './output/sketch/knu-prmi-org_graph_dfm/current_ckpt.pt'
data_dir = './data/sketch'

run_name = 'base'

dataset = 'sketch'
batch_size = 1
block_size = 100 # context of up to 256 previous characters

n_layer = 12
n_head = 16
n_embd = 1024
dropout = 0.1
qk_layernorm = True
do_x1_sc = False

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

div_free = 0.000001
