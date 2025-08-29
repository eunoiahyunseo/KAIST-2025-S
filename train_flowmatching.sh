#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python submitit_train.py \
--dataset=cifar10 \
--nodes=1 \
--discrete_flow_matching \
--batch_size=32 \
--accum_iter=1 \
--cfg_scale=0.0 \
--use_ema \
--epochs=3000 \
--class_drop_prob=1.0 \
--compute_fid \
--sym_func