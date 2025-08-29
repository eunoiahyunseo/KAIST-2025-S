import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from encoder_decoder_modify import MultimodalVQVAE, MultimodalVAE

from tqdm import tqdm
import ndjson
import numpy as np
import lpips
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from transformers import get_cosine_schedule_with_warmup
import argparse
import os
import math
from rdp import rdp

import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='모델 체크포인트 경로 (기본값: vae_model.pth)')

parser.add_argument('--run_name', type=str, default="vae-training",
                    help='W&B에 기록될 실행(run) 이름')

args = parser.parse_args()

def distance_point_to_segment_squared(points, p1, p2):
    segment_vec = p2 - p1
    segment_length_sq = np.sum(segment_vec**2)
    if segment_length_sq < 1e-9:
        return np.sum((points - p1)**2, axis=-1)
    points_vec = points - p1
    t = np.sum(points_vec * segment_vec, axis=-1) / segment_length_sq
    t_clamped = np.clip(t, 0, 1)
    closest_points_on_segment = p1 + t_clamped[..., np.newaxis] * segment_vec
    return np.sum((points - closest_points_on_segment)**2, axis=-1)

def create_udf_from_stroke(stroke_points, resolution=64, gamma=50.0):
    if stroke_points.shape[0] < 2:
        return np.zeros((resolution, resolution))
    grid_coords = np.linspace(0, 1, resolution)
    grid_x, grid_y = np.meshgrid(grid_coords, grid_coords)
    grid_points = np.stack([grid_x, grid_y], axis=-1)
    final_udf = np.zeros((resolution, resolution))
    for i in range(len(stroke_points) - 1):
        p1, p2 = stroke_points[i], stroke_points[i+1]
        dist_sq = distance_point_to_segment_squared(grid_points, p1, p2)
        exp_dist = np.exp(-gamma * dist_sq)
        final_udf = np.maximum(final_udf, exp_dist)
    return final_udf

def visualize_stroke_and_udf(stroke_points, udf_image, title="Stroke Visualization", save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    if len(stroke_points) > 0:
        x_coords, y_coords = stroke_points[:, 0], stroke_points[:, 1]
        ax1.plot(x_coords, y_coords, 'b-', linewidth=2, marker='o', markersize=3)
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.invert_yaxis()
        ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3); ax1.set_title('Original Stroke')
        ax1.plot(x_coords[0], y_coords[0], 'go', markersize=8, label='Start')
        ax1.plot(x_coords[-1], y_coords[-1], 'ro', markersize=8, label='End')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'Empty Stroke', ha='center', va='center'); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    if isinstance(udf_image, torch.Tensor): udf_image = udf_image.squeeze().cpu().numpy()
    im = ax2.imshow(udf_image, cmap='hot', interpolation='bilinear')
    ax2.set_title('UDF Image'); ax2.axis('off'); plt.colorbar(im, ax=ax2, shrink=0.8)
    plt.suptitle(title); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # plt.show() # wandb 사용 시 자동 실행을 위해 show()는 주석 처리하는 것이 좋음
    return fig

def visualize_batch_data(dataset, num_samples=5):
    print(f"Visualizing {num_samples} samples from the dataset...")
    for i in range(min(num_samples, len(dataset))):
        stroke_seq, stroke_mask, udf_image = dataset[i]
        valid_length = int(stroke_mask.sum().item())
        valid_stroke = stroke_seq[:valid_length, :2].numpy()
        fig = visualize_stroke_and_udf(valid_stroke, udf_image, title=f"Sample {i+1} - Valid Length: {valid_length}", save_path=f"sample_{i+1}_visualization.png")
        plt.close(fig) # Figure 객체를 닫아 메모리 관리
        # ... (print 문들은 그대로)

class QuickDrawDataset(Dataset):
    def __init__(self, dic_item, max_stroke_len=48, udf_res=64, gamma=50.0, rdp_epsilon=0.2, cache_path="./data/preprocessed_strokes.pt"):
        self.max_stroke_len = max_stroke_len
        self.udf_res = udf_res
        self.gamma = gamma
        self.rdp_epsilon = rdp_epsilon

        # self.all_strokes, self.all_udfs = self._preprocess(dic_item)

        cache_dir = os.path.dirname(cache_path)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            
        if os.path.exists(cache_path):
            print(f"✅ 전처리된 데이터 캐시 발견. 캐시에서 불러옵니다 2: {cache_path}")
            # 저장된 딕셔너리를 불러옵니다.
            # 클래스 객체가 아닌 리스트/numpy 배열이므로 weights_only=False가 필요합니다.
            cached_data = torch.load(cache_path, weights_only=False)
            self.all_strokes = cached_data['strokes']
            self.all_udfs = cached_data['udfs']
            print(f"캐시 로딩 완료. 총 {len(self.all_strokes)}개의 스트로크.")
        else:
            print(f"⏳ 전처리된 데이터 캐시 없음. 전처리를 시작합니다.")
            # 캐시가 없으면 전처리 함수를 호출합니다.
            self.all_strokes, self.all_udfs = self._preprocess(dic_item)
            
            # 다음 사용을 위해 전처리된 데이터를 딕셔너리 형태로 저장합니다.
            print(f"💾 전처리된 데이터를 캐시에 저장합니다: {cache_path}")
            data_to_save = {'strokes': self.all_strokes, 'udfs': self.all_udfs}
            torch.save(data_to_save, cache_path)

    def _preprocess(self, dic_item):
        print("데이터 전처리 시작...")
        processed_strokes = []
        processed_udfs = []
        for drawing in tqdm(dic_item):
            strokes_np = [np.array(stroke).T.astype(np.float32) for stroke in drawing]
            if not strokes_np: continue
            for stroke in strokes_np:
                if len(stroke) < 2: continue
                simplified_stroke = rdp(stroke, epsilon=self.rdp_epsilon) if self.rdp_epsilon > 0 else stroke
                if len(simplified_stroke) < 2: continue
                min_coords, max_coords = np.min(simplified_stroke, axis=0), np.max(simplified_stroke, axis=0)
                center = (min_coords + max_coords) / 2.0
                scale = (max_coords - min_coords).max()
                if scale < 1e-9: scale = 1.0
                centered_and_scaled = (simplified_stroke - center) / scale
                normalized_stroke = (centered_and_scaled + 0.5) * 0.7 + 0.15
                if not np.isnan(normalized_stroke).any(): processed_strokes.append(normalized_stroke)
                udf_image = create_udf_from_stroke(normalized_stroke, self.udf_res, self.gamma)
                processed_udfs.append(udf_image)
        print(f"전처리 완료. 총 {len(processed_strokes)}개의 스트로크 추출.")
        return processed_strokes, processed_udfs
    def __len__(self):
        return len(self.all_strokes)
    def __getitem__(self, idx):
        stroke = self.all_strokes[idx]
        # udf_image_tensor = torch.from_numpy(create_udf_from_stroke(stroke, self.udf_res, self.gamma)).float().unsqueeze(0)
        udf_image = self.all_udfs[idx]
        udf_image_tensor = torch.from_numpy(udf_image).float().unsqueeze(0)
        stroke_len = min(len(stroke), self.max_stroke_len)
        stroke_with_pen_state = np.ones((stroke_len, 3), dtype=np.float32)
        stroke_with_pen_state[:, :2] = stroke[:stroke_len]
        if stroke_len > 0: stroke_with_pen_state[-1, 2] = 0
        padded_stroke = np.zeros((self.max_stroke_len, 3), dtype=np.float32)
        padded_stroke[:stroke_len] = stroke_with_pen_state
        stroke_mask = np.zeros(self.max_stroke_len, dtype=np.float32)
        stroke_mask[:stroke_len] = 1.0
        return torch.from_numpy(padded_stroke), torch.from_numpy(stroke_mask), udf_image_tensor


def get_quickdraw_dataset(categories, config, cache_path="./data/preprocessed_quickdraw.pt"):
    """
    QuickDraw 데이터셋을 불러오거나, 캐시된 파일이 있으면 사용합니다.

    Args:
        categories (list): 불러올 카테고리 이름 리스트
        config (dict): 데이터 처리 관련 설정값
        cache_path (str): 전처리된 데이터셋을 저장하고 불러올 경로

    Returns:
        QuickDrawDataset: 생성되거나 캐시에서 불러온 데이터셋 객체
    """
    # 캐시 파일이 저장될 디렉토리가 없으면 생성
    cache_dir = os.path.dirname(cache_path)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    # 1. 캐시된 파일이 있는지 확인
    if os.path.exists(cache_path):
        print(f"✅ 전처리된 데이터셋 발견. 캐시에서 불러옵니다: {cache_path}")
        dataset = torch.load(cache_path, weights_only=False)
        return dataset

    # 2. 캐시 파일이 없으면 전처리 시작
    print(f"⏳ 전처리된 데이터셋 없음. 데이터 전처리를 시작합니다...")
    
    all_drawings = []
    for category in categories:
        data_path = f"./data/quickdraw/{category}.ndjson"
        print(f"  -> {data_path} 처리 중...")
        try:
            with open(data_path, 'r') as f:
                data = ndjson.load(f)
            
            # 'recognized'가 True인 그림만 필터링
            recognized_drawings = [
                item['drawing'] for item in data if item.get('recognized', False) is True
            ]
            
            # 카테고리별로 지정된 개수만큼 데이터 선택
            all_drawings += recognized_drawings[:config["data_size_per_category"]]
            
            

        except FileNotFoundError:
            print(f"  ⚠️ 경고: {data_path} 파일을 찾을 수 없습니다. 건너뜁니다.")
            continue

    if not all_drawings:
        raise RuntimeError("데이터를 전혀 불러오지 못했습니다. 데이터 경로를 확인해주세요.")

    # 3. 데이터셋 객체 생성
    print("데이터셋 객체를 생성합니다...")
    dataset = QuickDrawDataset(
        all_drawings, 
        max_stroke_len=config["max_stroke_len"], 
        udf_res=config["udf_resolution"], 
        rdp_epsilon=config["rdp_epsilon"],
        gamma=config["gamma"]
    )

    # 4. 다음 사용을 위해 데이터셋 객체를 파일로 저장
    print(f"💾 전처리된 데이터셋을 저장합니다: {cache_path}")
    torch.save(dataset, cache_path)
    
    return dataset
# --- 하이퍼파라미터 ---
config = {
    "max_stroke_len": 48,
    "udf_resolution": 64,
    "d_model": 256,
    "d_img": 128,
    "d_seq": 128,
    "embedding_dim": 256,
    "batch_size": 256,
    "learning_rate": 1e-4,
    "2-epochs": 1000,
    "n_head": 8,
    "num_layers": 6,
    "data_size_per_category": 50,
    "loss_weight": {
        "coord": 1.0,
        "pen": 0.1,
        "img": 0.1,
        "kl": 0.001
    },
}

if __name__ == "__main__":

    wandb.init(
            project="stroke-vq-vae-pooling", 
            name=args.run_name,
            config=config
        )
    checkpoint_dir = f"./checkpoints/{wandb.run.name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"체크포인트는 '{checkpoint_dir}' 디렉토리에 저장됩니다.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CATEGORIES = ["umbrella", "apple"] # 더 많은 카테고리 사용 권장
    
    dataset = get_quickdraw_dataset(CATEGORIES, config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True)
    fixed_dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    fixed_batch = next(iter(fixed_dataloader))

    # 1. VAE 모델 생성
    vae_model = MultimodalVAE(
        d_model=config["d_model"],
        d_img=config["d_img"],
        d_seq=config["d_seq"],
        n_head=config["n_head"],
        num_layers=config["num_layers"],
        embedding_dim=config["embedding_dim"],
        udf_res=config["udf_resolution"],
    ).to(device)

    

    num_training_steps = config["2-epochs"] * len(dataloader)
    num_warmup_steps = int(num_training_steps * 0.1)

    trainable_params = filter(lambda p: p.requires_grad, vae_model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=config["learning_rate"])

    loss_fn_alex = lpips.LPIPS(net='vgg').to(device)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    recon_loss_coord_fn = nn.L1Loss(reduction='none') 

        
    print("=== VAE Pre-training Start ===")
    global_step=0
    for epoch in range(config["2-epochs"]):
        epoch_total_loss, epoch_coord_loss, epoch_pen_loss, epoch_img_loss, epoch_vq_loss, epoch_diff_loss, epoch_perplexity, epoch_kl_loss = 0, 0, 0, 0, 0, 0, 0, 0
        num_batches = 0
        loop = tqdm(dataloader, leave=True)
        for batch_idx, (stroke_seq, stroke_mask, udf_image) in enumerate(loop):
            stroke_seq, stroke_mask, udf_image = stroke_seq.to(device), stroke_mask.to(device), udf_image.to(device)
            optimizer.zero_grad()

            recon_coords, recon_udf, kl_loss = vae_model(
                stroke_seq, stroke_mask, udf_image
            )

            recon_pen_logits = recon_coords[:, :, 2]
            recon_coords = recon_coords[:, :, :2]


            target_coords = stroke_seq[:, :, :2]
            target_pen_state = stroke_seq[:, :, 2]
            
            recon_pen_probs = recon_pen_logits 

            loss_pen = F.binary_cross_entropy(recon_pen_probs, target_pen_state.float(), reduction='none')
            loss_pen = (loss_pen * stroke_mask).sum() / stroke_mask.sum().clamp(min=1e-9)
            
            loss_coords_raw = recon_loss_coord_fn(recon_coords, target_coords)
            weight_mask = torch.ones_like(stroke_mask)
            weight_mask[:, 0] = 2.0
            loss_coords = (loss_coords_raw.mean(dim=-1) * weight_mask * stroke_mask).sum() / (stroke_mask * weight_mask).sum().clamp(min=1e-9)

            loss_l1_img = F.l1_loss(recon_udf, udf_image)
            recon_udf_3ch, udf_image_3ch = recon_udf.repeat(1, 3, 1, 1), udf_image.repeat(1, 3, 1, 1)
            loss_percep = loss_fn_alex(recon_udf_3ch, udf_image_3ch).mean()
            loss_img = loss_l1_img + loss_percep
            
            total_loss = (
                config['loss_weight']['coord'] * loss_coords + \
                config['loss_weight']['pen'] * loss_pen + \
                config['loss_weight']['kl'] * kl_loss + \
                config['loss_weight']['img'] * loss_img
            )
            
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_total_loss += total_loss.item()
            epoch_coord_loss += loss_coords.item()
            epoch_pen_loss += loss_pen.item()
            epoch_kl_loss += kl_loss.item()
            epoch_img_loss += loss_img.item()

            loop.set_description(f"Epoch [{epoch+1}/{config['2-epochs']}]")
            loop.set_postfix(loss=total_loss.item())

        num_batches += 1
            
        avg_total_loss = epoch_total_loss / num_batches
        avg_coord_loss = epoch_coord_loss / num_batches
        avg_pen_loss = epoch_pen_loss / num_batches
        avg_img_loss = epoch_img_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches

        # --- [WANDB] 3. wandb.log()로 에폭마다 지표 기록 ---
        wandb.log({
            "epoch": epoch + 1,
            "avg_total_loss": avg_total_loss,
            "avg_coord_loss": avg_coord_loss,
            "avg_pen_loss": avg_pen_loss,
            "avg_img_loss": avg_img_loss,
            "avg_kl_loss": avg_kl_loss,
            "learning_rate": scheduler.get_last_lr()[0]
        }, step=epoch)
    
        print('epoch', epoch)
        if (epoch + 1) % 100 == 0:
            print(f"\nEpoch {epoch+1}: 시각화 및 모델 체크포인트 저장 중...")
            vae_model.eval()
            with torch.no_grad():
                stroke_seq_fixed, stroke_mask_fixed, udf_image_fixed = [x.to(device) for x in fixed_batch]
                predicted_points_tf, predicted_udf_fixed, kl_loss = vae_model(
                                stroke_seq_fixed, stroke_mask_fixed, udf_image_fixed
                            )

                # predicted_points_tf, predicted_udf_fixed, _ = vqae_model(
                #     stroke_seq_fixed, stroke_mask_fixed, udf_image_fixed
                # )


                # --- [변경점] Autoregressive 방식의 생성 결과 얻기 ---
                generated_strokes_ar, _ = vae_model.generate(
                    stroke_seq_fixed, stroke_mask_fixed, udf_image_fixed, 
                    max_len=config["max_stroke_len"]
                )


                # W&B에 로깅할 이미지들을 담을 리스트
                log_images = []
                # --- [변경점] 샘플 수를 5개에서 2개로 변경 ---
                num_samples_to_log = min(5, config["batch_size"])

                for i in range(num_samples_to_log):
                    # --- [변경점] 2x3 그리드 생성 ---
                    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                    fig.suptitle(f'Epoch {epoch+1} - Sample {i+1} Reconstruction', fontsize=16)

                    # Axes 핸들 정의
                    ax_orig_stroke = axes[0, 0]
                    ax_tf_stroke = axes[0, 1]
                    ax_ar_stroke = axes[0, 2]
                    ax_orig_udf = axes[1, 0]
                    ax_recon_udf = axes[1, 1]
                    fig.delaxes(axes[1, 2]) # 비어있는 마지막 subplot 제거

                    # --- Plot 1: 원본 스트로크 ---
                    valid_len_orig = int(stroke_mask_fixed[i].sum().item())
                    orig_stroke = stroke_seq_fixed[i, :valid_len_orig, :2].cpu().numpy()
                    ax_orig_stroke.plot(orig_stroke[:, 0], orig_stroke[:, 1], 'b-', lw=2)
                    ax_orig_stroke.set_title('Original Stroke')
                    ax_orig_stroke.set_xlim(0, 1); ax_orig_stroke.set_ylim(0, 1); ax_orig_stroke.invert_yaxis()
                    ax_orig_stroke.set_aspect('equal'); ax_orig_stroke.grid(True, alpha=0.3)
                    
                    # --- Plot 2: 복원된 스트로크 (Teacher Forcing) ---
                    # forward()의 출력을 사용
                    recon_stroke_tf = predicted_points_tf[i, :valid_len_orig, :2].cpu().numpy()
                    ax_tf_stroke.plot(recon_stroke_tf[:, 0], recon_stroke_tf[:, 1], 'g-', lw=2)
                    ax_tf_stroke.set_title('Recon (Teacher Forcing)')
                    ax_tf_stroke.set_xlim(0, 1); ax_tf_stroke.set_ylim(0, 1); ax_tf_stroke.invert_yaxis()
                    ax_tf_stroke.set_aspect('equal'); ax_tf_stroke.grid(True, alpha=0.3)

                    # --- [신규] Plot 3: 생성된 스트로크 (Autoregressive) ---
                    # generate()의 출력을 사용
                    pen_states_ar = (generated_strokes_ar[i, :, 2] > 0.5)
                    zero_indices = (pen_states_ar == 0).nonzero(as_tuple=True)[0]
                    valid_len_ar = zero_indices[0].item() + 1 if len(zero_indices) > 0 else config["max_stroke_len"]
                    recon_stroke_ar = generated_strokes_ar[i, :valid_len_ar, :2].cpu().numpy()
                    ax_ar_stroke.plot(recon_stroke_ar[:, 0], recon_stroke_ar[:, 1], 'r-', lw=2)
                    ax_ar_stroke.set_title('Generated (Autoregressive)')
                    ax_ar_stroke.set_xlim(0, 1); ax_ar_stroke.set_ylim(0, 1); ax_ar_stroke.invert_yaxis()
                    ax_ar_stroke.set_aspect('equal'); ax_ar_stroke.grid(True, alpha=0.3)

                    # --- Plot 4: 원본 UDF ---
                    original_udf_img = udf_image_fixed[i].squeeze().cpu().numpy()
                    im_orig = ax_orig_udf.imshow(original_udf_img, cmap='hot', vmin=0, vmax=1)
                    ax_orig_udf.set_title('Original UDF')
                    ax_orig_udf.axis('off')
                    fig.colorbar(im_orig, ax=ax_orig_udf, shrink=0.8)

                    # --- Plot 5: 복원된 UDF ---
                    reconstructed_udf_img = predicted_udf_fixed[i].squeeze().cpu().numpy()
                    im_recon = ax_recon_udf.imshow(reconstructed_udf_img, cmap='hot', vmin=0, vmax=1)
                    ax_recon_udf.set_title('Reconstructed UDF')
                    ax_recon_udf.axis('off')
                    fig.colorbar(im_recon, ax=ax_recon_udf, shrink=0.8)

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    
                    log_images.append(wandb.Image(fig, caption=f"Sample {i+1}"))
                    plt.close(fig)
                
                wandb.log({"reconstruction_examples": log_images}, step=epoch)
                # 주기적인 모델 저장 로직 (이전과 동일)
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
                torch.save(vae_model.state_dict(), checkpoint_path)
                artifact = wandb.Artifact(f'model-{wandb.run.name}', type='model')
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact, aliases=[f"epoch_{epoch+1}"])
                print(f"Epoch {epoch+1}: 모델 체크포인트 저장 완료 -> {checkpoint_path}")
                

            vae_model.train() # 모델을 다시 훈련 모드로
            
    print("--- 훈련 완료 ---")
    
    model_path = './checkpoints/vae-training/pretrained_vae_model.pth'
    torch.save(vae_model.state_dict(), model_path)
    artifact = wandb.Artifact('vae-model', type='model')
    artifact.add_file(model_path)
    wandb.run.log_artifact(artifact)
    print(f"Model saved and logged to W&B Artifacts as '{model_path}'")
