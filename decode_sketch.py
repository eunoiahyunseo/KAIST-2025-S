# 파일 이름: decode_sketch.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm

# --- 모델 정의 파일 임포트 ---
from encoder_decoder_modify import MultimodalVQVAE, BoundingBoxVQVAE, BoundingBoxVQVAE_Dual

# --- argparse ---
parser = argparse.ArgumentParser(description="Decode and visualize tokenized sketches.")
parser.add_argument('--bbox_layout_path', type=str, required=True, help="Path to the layout (bbox) tokens .npz file (e.g., ./data/layout_bbox/val.npz)")
parser.add_argument('--stroke_layout_path', type=str, required=True, help="Path to the stroke tokens .npz file (e.g., ./data/layout_stroke/val2.npz)")
parser.add_argument('--bbox_layout_model_path', type=str, required=True, help="Path to the trained size VQ-VAE weights (.pth)")
parser.add_argument('--stroke_model_path', type=str, required=True, help="Path to the trained stroke VQ-VAE weights (.pth)")
parser.add_argument('--output_dir', type=str, default='./decoded_sketches', help="Directory to save the output image")
args = parser.parse_args()

# --- 하이퍼파라미터 (모델 생성 및 토큰화 때와 일치해야 함) ---
config = {
    "canvas_size": 256.0, "max_stroke_len": 48,
    "d_model": 256, "d_img": 128, "d_seq": 128, "embedding_dim": 256,
    "num_codes": 293, "commitment_cost": 0.25, "n_head": 8, "num_layers": 6,
    "bbox_hidden_dim": 64, "bbox_codebook_dim": 128,
    "bbox_loc_codebook_size": 50, "bbox_size_codebook_size": 50,
}

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. 모든 훈련된 모델 불러오기 ---
    print("--- Loading all pretrained models ---")
    model = BoundingBoxVQVAE_Dual(
        # input_dim=4, # (cx, cy)
        num_loc_embeddings=config["bbox_loc_codebook_size"],
        num_size_embeddings=config["bbox_size_codebook_size"],
        loc_latent_dim=config["bbox_codebook_dim"] // 2,
        size_latent_dim=config["bbox_codebook_dim"] // 2,
        commitment_cost=config["commitment_cost"]
    ).to(device)
    model.load_state_dict(torch.load(args.bbox_layout_model_path, map_location=device))
    model.eval()
    # Stroke 모델
    stroke_vqvae = MultimodalVQVAE(
        d_model=config["d_model"], d_img=config["d_img"], d_seq=config["d_seq"],
        n_head=config["n_head"], num_layers=config["num_layers"],
        num_codes=config["num_codes"], embedding_dim=config["embedding_dim"],
        commitment_cost=config["commitment_cost"], udf_res=64, # udf_res는 임의값
    ).to(device)
    stroke_vqvae.load_state_dict(torch.load(args.stroke_model_path, map_location=device))
    stroke_vqvae.eval()
    print("✅ All models loaded successfully.")

    # --- 2. 토큰 데이터 로드 ---
    bbox_tokens_data = np.load(args.bbox_layout_path)
    stroke_tokens_data = np.load(args.stroke_layout_path)
    drawing_keys = bbox_tokens_data.files
    print(f"\nFound {len(drawing_keys)} drawings to decode.")
    stroke_cmap = plt.cm.get_cmap('viridis', config["num_codes"])
    stroke_norm = plt.Normalize(vmin=0, vmax=config["num_codes"])
    print(bbox_tokens_data[drawing_keys[0]])
    print(stroke_tokens_data[drawing_keys[0]])


    # --- 3. 모든 스케치에 대해 반복하여 디코딩 실행 ---
    with torch.no_grad():
        for drawing_key in tqdm(drawing_keys, desc="Decoding all sketches"):
            bbox_token_sequence = torch.tensor(bbox_tokens_data[drawing_key], dtype=torch.long).to(device)
            stroke_token_sequence = torch.tensor(stroke_tokens_data[drawing_key], dtype=torch.long).to(device)

            # --- Bbox 디코딩 (시퀀스 전체를 한 번에) ---
            loc_indices = bbox_token_sequence // config["bbox_size_codebook_size"]
            size_indices = bbox_token_sequence % config["bbox_size_codebook_size"]
            
            reconstructed_bbox = model.decode(loc_indices, size_indices)
            decoded_bboxes_norm = torch.cat([reconstructed_bbox[:, :2], reconstructed_bbox[:, 2:]], dim=1)

            # --- Stroke 디코딩 (시퀀스 전체를 한 번에) ---
            print(stroke_token_sequence.shape, stroke_token_sequence)
            print(bbox_token_sequence.shape, bbox_token_sequence)
            list_of_decoded_strokes = stroke_vqvae.decode_from_token_sequence(stroke_token_sequence)

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.set_title(f"Decoded Sketch for '{drawing_key}'")
            ax.set_xlim(0, config["canvas_size"]); ax.set_ylim(0, config["canvas_size"])
            ax.invert_yaxis(); ax.set_aspect('equal', adjustable='box'); ax.grid(True, alpha=0.3)
            
            # ### 스트로크와 bbox를 함께 그리는 루프 ###
            for i, stroke_token_id in enumerate(stroke_token_sequence):
                color = stroke_cmap(stroke_norm(stroke_token_id.item()))

                # --- 1. Bbox 좌표 및 크기 복원 (Un-Normalization) ---
                cx_norm, cy_norm, w_norm, h_norm = decoded_bboxes_norm[i].cpu().numpy()
                cx, cy = (cx_norm + 0.5) * config["canvas_size"], (cy_norm + 0.5) * config["canvas_size"]
                w, h = w_norm * config["canvas_size"], h_norm * config["canvas_size"]
                x_min, y_min = cx - w / 2, cy - h / 2
                
                rect = plt.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor=color, facecolor='none', alpha=0.6, linestyle='--')
                ax.add_patch(rect)

                # --- 2. Stroke 좌표 복원 (Un-Normalization) ---
                full_decoded_stroke = list_of_decoded_strokes[i]
                
                pen_states_ar = full_decoded_stroke[:, 2] > 0.5
                
                zero_indices = (pen_states_ar == 0).nonzero(as_tuple=True)[0]
                valid_len = zero_indices[0].item() + 1 if len(zero_indices) > 0 else config["max_stroke_len"]
                print('valid_len: ', valid_len)
                valid_stroke_normalized = full_decoded_stroke[:valid_len].cpu().numpy()[:, :2]

                ### [핵심 수정] 올바른 등방적(Isotropic) 스트로크 복원 로직 ###
                
                # a) [0.15, 0.85] 범위를 -> [0, 1] 범위로 되돌리기
                stroke_0_to_1 = (valid_stroke_normalized - 0.15) / 0.7
                
                # b) [0, 1] 스트로크의 자체적인 너비와 높이를 계산
                stroke_min_coords = np.min(stroke_0_to_1, axis=0)
                stroke_max_coords = np.max(stroke_0_to_1, axis=0)
                stroke_w = stroke_max_coords[0] - stroke_min_coords[0]
                stroke_h = stroke_max_coords[1] - stroke_min_coords[1]
                if stroke_w < 1e-9: stroke_w = 1.0
                if stroke_h < 1e-9: stroke_h = 1.0

                # c) 너비 비율과 높이 비율을 계산하여, 더 작은 쪽을 최종 스케일로 선택
                #    이렇게 하면 종횡비가 유지되면서 bbox를 벗어나지 않음
                scale_ratio = min(w / stroke_w, h / stroke_h)
                
                # d) 스트로크를 새로운 스케일로 확대하고, bbox 중앙에 위치하도록 평행이동
                scaled_stroke_w = stroke_w * scale_ratio
                scaled_stroke_h = stroke_h * scale_ratio
                offset_x = (w - scaled_stroke_w) / 2
                offset_y = (h - scaled_stroke_h) / 2
                
                final_stroke = (stroke_0_to_1 - stroke_min_coords) * scale_ratio + [x_min + offset_x, y_min + offset_y]
                
                ax.plot(final_stroke[:, 0], final_stroke[:, 1], color=color, linewidth=2.0)

            save_path = os.path.join(args.output_dir, f"{drawing_key}_decoded.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)






