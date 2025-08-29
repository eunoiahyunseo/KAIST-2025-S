#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python decode_sketch.py \
    --bbox_layout_path ./data/layout/val.npz \
    --stroke_layout_path /home/heart20021010/workspace/KAIST-2025-S/checkpoints/stroke_dfm/samples_2025-08-21-11-02-35_e802deac/generated_strokes.npz \
    --bbox_layout_model_path ./checkpoints/vq-vae-training/bbox_vqvae_layout.pth \
    --stroke_model_path ./checkpoints/vq-vae-training/stroke_vqae.pth \
    # --stroke_layout_path /home/heart20021010/workspace/KAIST-2025-S/checkpoints/stroke_dfm/samples_2025-08-20-15-55-37_ad683e5e/generated_strokes.npz \
