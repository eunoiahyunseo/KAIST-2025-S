
out_dir = './checkpoints/stroke_dfm'
ckpt_path = './checkpoints/stroke_dfm/final_ckpt.pt'
data_dir = './data/layout' # layout dir for load bounding box condition

run_name = 'base'

dataset = 'layout'
batch_size = 10
block_size = 20

n_layer = 6
n_head = 8
n_embd = 512
dropout = 0
qk_layernorm = True

total_samples = 10
dt = 0.001
max_t = 0.98
argmax_final = True

div_free = 0
source_distribution = 'mask'
