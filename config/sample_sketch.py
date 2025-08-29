
out_dir = './output/sketch_stroke'
ckpt_path = './output/sketch_apple_c_coupling/2025-07-30-16-06-47_sketch_apple_c_coupling/current_ckpt.pt'
data_dir = './data/full_sketch'

run_name = 'base'

dataset = 'full_sketch'
batch_size = 10
block_size = 45

n_layer = 12
n_head = 16
n_embd = 1024
dropout = 0
qk_layernorm = True

total_samples = 10
dt = 0.001
max_t = 0.98
argmax_final = True

div_free = 0
source_distribution = 'mask'
