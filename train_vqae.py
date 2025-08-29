import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from encoder_decoder import StrokeTokenizerVQVAE
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
                        help='모델 체크포인트 경로 (기본값: vqae_model.pth)')
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
    def __init__(self, dic_item, max_stroke_len=48, udf_res=64, gamma=50.0, rdp_epsilon=0.2):
        self.max_stroke_len = max_stroke_len
        self.udf_res = udf_res
        self.gamma = gamma
        self.rdp_epsilon = rdp_epsilon
        self.all_strokes, self.all_udfs = self._preprocess(dic_item)
    def _preprocess(self, dic_item):
        print("데이터 전처리 시작...")
        processed_strokes = []
        processed_udfs = []
        for key, drawing in tqdm(dic_item):
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




# --- 하이퍼파라미터 ---
config = {
    "max_stroke_len": 48,
    "udf_resolution": 64,
    "d_model": 256,
    "d_img": 256,
    "d_seq": 256,
    "embedding_dim": 256,
    "num_embeddings": 1024,
    "batch_size": 256,
    "learning_rate": 5e-4,
    "1-epochs": 1000,
    "2-epochs": 1500,
    "gamma": 100.0,
    "rdp_epsilon": 0.0,
    "commitment_cost": 0.4,
    "nhead": 4,
    "num_layers": 4,
    "d_num_layers": 2,
    "overfit_size": 40,
    "data_size_per_category": 2000,
    "loss_weight": {
        "coord": 1.0,
        "diff": 1.0,
        "pen": 0.1,
        "vq": 0.5
    }
}

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        # nn.init.normal_(m.weight, mean=0, std=0.02)
        bound = math.sqrt(1.0 / config["num_embeddings"])
        nn.init.uniform_(m.weight, -bound, bound)


if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dic_item = []

    for category in ["moon", "apple"]:
        data_path = f"./data/quickdraw/{category}.ndjson"
        try:
            with open(data_path, 'r') as f: data = ndjson.load(f)
            dic = {}
            cnt = 0
            for i in data:
                if i['recognized'] is True:
                    dic[str(cnt)] = i['drawing']
                    cnt += 1
                    if cnt >= config["data_size_per_category"]: break
            dic_item += list(dic.items())
        except FileNotFoundError:
            print(f"Warning: Data file not found at {data_path}")

    dataset = QuickDrawDataset(
        dic_item, max_stroke_len=config["max_stroke_len"], udf_res=config["udf_resolution"], 
        rdp_epsilon=config["rdp_epsilon"], gamma=config["gamma"])

    OVERFIT = False
    if OVERFIT:
        from torch.utils.data import Subset
        overfit_indices = list(range(min(config["overfit_size"], len(dataset))))
        dataset = Subset(dataset, overfit_indices)
        print(f"Overfitting mode: Using only {len(dataset)} samples")
        config["batch_size"] = min(config["batch_size"], len(dataset))

    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    fixed_dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    fixed_batch = next(iter(fixed_dataloader))


    vqae_model = StrokeTokenizerVQVAE(
        max_stroke_len=config["max_stroke_len"],
        d_model=config["d_model"],
        d_img=config["d_img"],
        d_seq=config["d_seq"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        num_embeddings=config["num_embeddings"],
        embedding_dim=config["embedding_dim"],
        commitment_cost=config["commitment_cost"],
        udf_resolution=config["udf_resolution"],
        d_num_layers=config["d_num_layers"]
    ).to(device)

    

    wandb.init(
        project="stroke-vq-vae-2-stage", 
        name="stage2-stroke-VQ-autoencoder",
        config=config
    )

    if args.checkpoint is None:
        print("--- 훈련 모드 시작 ---")
        vqae_model.apply(weights_init)
        # vqae_model.initialize_codebook_with_kmeans(dataloader, device)
        vae_model_path = "image_spatial_vae_pretrained.pth"
        vqae_model.load_pretrained_image_modules(vae_model_path)

        num_training_steps = config["2-epochs"] * len(dataloader)
        num_warmup_steps = int(num_training_steps * 0.1)

        trainable_params = filter(lambda p: p.requires_grad, vqae_model.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=config["learning_rate"])

        loss_fn_alex = lpips.LPIPS(net='vgg').to(device)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        recon_loss_coord_fn = nn.L1Loss(reduction='none') 

        print("=== VQ-AE 학습 시작 ===")
        global_step = 0
        
        for epoch in range(config["2-epochs"]):
            epoch_total_loss, epoch_coord_loss, epoch_pen_loss, epoch_img_loss, epoch_vq_loss, epoch_diff_loss = 0, 0, 0, 0, 0, 0
            num_batches = 0
            
            loop = tqdm(dataloader, leave=True)
            for batch_idx, (stroke_seq, stroke_mask, udf_image) in enumerate(loop):
                stroke_seq, stroke_mask, udf_image = stroke_seq.to(device), stroke_mask.to(device), udf_image.to(device)
                optimizer.zero_grad()

                recon_coords, recon_pen_logits, recon_udf, vq_loss, indices_stroke, indices_img, full_mask = vqae_model(
                    stroke_seq, stroke_mask, udf_image
                )

                # recon_coords, recon_pen_logits, recon_udf, vq_loss, _, _ = vqae_model(stroke_seq, stroke_mask, udf_image)
                print('recon_coords:', recon_coords[0, :20, 0])
                print('target:', stroke_seq[0, :20, 0])
                print('indices_stroke: ', indices_stroke[0, :20])
                
                target_coords = stroke_seq[:, :, :2]
                target_pen_state = stroke_seq[:, :, 2]
                
                loss_pen = F.binary_cross_entropy_with_logits(recon_pen_logits, target_pen_state.float(), reduction='none')
                loss_pen = (loss_pen * stroke_mask).sum() / stroke_mask.sum().clamp(min=1e-9)
                
                loss_coords_raw = recon_loss_coord_fn(recon_coords, target_coords)
                weight_mask = torch.ones_like(stroke_mask)
                weight_mask[:, 0] = 5.0
                loss_coords = (loss_coords_raw.mean(dim=-1) * weight_mask * stroke_mask).sum() / (stroke_mask * weight_mask).sum().clamp(min=1e-9)
                
                target_diffs = target_coords[:, 1:, :] - target_coords[:, :-1, :]
                recon_diffs = recon_coords[:, 1:, :] - recon_coords[:, :-1, :]
                diff_mask = stroke_mask[:, 1:]
                loss_diff = recon_loss_coord_fn(recon_diffs, target_diffs)
                loss_diff = (loss_diff.mean(dim=-1) * diff_mask).sum() / diff_mask.sum().clamp(min=1e-9)
                
                # loss_l1_img = F.l1_loss(recon_udf, udf_image)
                # with torch.no_grad():
                #     recon_udf_3ch, udf_image_3ch = recon_udf.repeat(1, 3, 1, 1), udf_image.repeat(1, 3, 1, 1)
                #     loss_percep = loss_fn_alex(recon_udf_3ch, udf_image_3ch).mean()
                # loss_img = loss_l1_img + loss_percep

                total_loss = (
                    config['loss_weight']['coord'] * loss_coords + \
                    config['loss_weight']['diff'] * loss_diff + \
                    config['loss_weight']['pen'] * loss_pen + \
                    config['loss_weight']['vq'] * vq_loss
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(vqae_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1

                epoch_total_loss += total_loss.item()
                epoch_coord_loss += loss_coords.item()
                epoch_pen_loss += loss_pen.item()
                epoch_vq_loss += vq_loss.item()
                epoch_diff_loss += loss_diff.item()
                num_batches += 1
                
                loop.set_description(f"Epoch [{epoch+1}/{config['2-epochs']}]")
                loop.set_postfix(loss=total_loss.item())
            
            avg_total_loss = epoch_total_loss / num_batches
            avg_coord_loss = epoch_coord_loss / num_batches
            avg_pen_loss = epoch_pen_loss / num_batches
            avg_vq_loss = epoch_vq_loss / num_batches
            avg_diff_loss = epoch_diff_loss / num_batches
            
            # --- [WANDB] 3. wandb.log()로 에폭마다 지표 기록 ---
            wandb.log({
                "epoch": epoch + 1,
                "avg_total_loss": avg_total_loss,
                "avg_coord_loss": avg_coord_loss,
                "avg_pen_loss": avg_pen_loss,
                "avg_vq_loss": avg_vq_loss,
                "avg_diff_loss": avg_diff_loss,
                "learning_rate": scheduler.get_last_lr()[0]
            }, step=epoch)
        
            print('epoch', epoch)
            if (epoch + 1) % 25 == 0 or (OVERFIT and (epoch + 1) % 10 == 0):
                print(f"\nEpoch {epoch+1} Summary: Total Loss: {avg_total_loss:.6f}")
                vqae_model.eval()
                with torch.no_grad():
                    stroke_seq_fixed, stroke_mask_fixed, udf_image_fixed = [x.to(device) for x in fixed_batch]
                    
                    # 모델 forward pass에서 모든 결과물 받기
                    # 참고: 2-Codebook 모델을 사용하신다면 indices_img도 반환됩니다.
                    recon_coords, recon_pen_logits, recon_udf, _, indices_stroke, _, full_mask = vqae_model(
                        stroke_seq_fixed, stroke_mask_fixed, udf_image_fixed
                    )
                    
                    # generate_from_sequence 대신 reconstruct_from_latents를 사용해야 할 수 있습니다.
                    # 여기서는 forward pass의 recon_coords를 직접 사용하겠습니다.
                    reconstructed_strokes_fixed = torch.cat([recon_coords, torch.sigmoid(recon_pen_logits).unsqueeze(-1)], dim=-1)

                    # 시각화할 샘플 인덱스 (0번째)
                    i = 0
                    
                    # --- [수정] 2x2 그리드 생성 ---
                    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                    fig.suptitle(f'Epoch {epoch+1} - Reconstruction Progress', fontsize=16)

                    ax_orig_stroke = axes[0, 0]
                    ax_recon_stroke = axes[0, 1]
                    ax_orig_udf = axes[1, 0]
                    ax_recon_udf = axes[1, 1]

                    # --- Plot 1: 원본 스트로크 ---
                    valid_len_orig = int(stroke_mask_fixed[i].sum().item())
                    orig_stroke = stroke_seq_fixed[i, :valid_len_orig, :2].cpu().numpy()
                    ax_orig_stroke.plot(orig_stroke[:, 0], orig_stroke[:, 1], 'b-', lw=2)
                    ax_orig_stroke.set_title('Original Stroke')
                    ax_orig_stroke.set_xlim(0, 1); ax_orig_stroke.set_ylim(0, 1); ax_orig_stroke.invert_yaxis()
                    ax_orig_stroke.set_aspect('equal'); ax_orig_stroke.grid(True, alpha=0.3)
                    
                    # --- Plot 2: 복원된 스트로크 ---
                    pen_states = (torch.sigmoid(recon_pen_logits[i, :]) > 0.5).float()
                    zero_indices = (pen_states == 0).nonzero(as_tuple=True)[0]
                    valid_len_recon = zero_indices[0].item() + 1 if len(zero_indices) > 0 else config["max_stroke_len"]
                    recon_stroke = reconstructed_strokes_fixed[i, :valid_len_recon, :2].cpu().numpy()
                    ax_recon_stroke.plot(recon_stroke[:, 0], recon_stroke[:, 1], 'r-', lw=2)
                    ax_recon_stroke.set_title('Reconstructed Stroke')
                    ax_recon_stroke.set_xlim(0, 1); ax_recon_stroke.set_ylim(0, 1); ax_recon_stroke.invert_yaxis()
                    ax_recon_stroke.set_aspect('equal'); ax_recon_stroke.grid(True, alpha=0.3)

                    # --- [신규] Plot 3: 원본 UDF ---
                    original_udf_img = udf_image_fixed[i].squeeze().cpu().numpy()
                    im_orig = ax_orig_udf.imshow(original_udf_img, cmap='hot', vmin=0, vmax=1)
                    ax_orig_udf.set_title('Original UDF')
                    ax_orig_udf.axis('off')
                    fig.colorbar(im_orig, ax=ax_orig_udf, shrink=0.8)

                    # --- [신규] Plot 4: 복원된 UDF ---
                    reconstructed_udf_img = recon_udf[i].squeeze().cpu().numpy()
                    im_recon = ax_recon_udf.imshow(reconstructed_udf_img, cmap='hot', vmin=0, vmax=1)
                    ax_recon_udf.set_title('Reconstructed UDF')
                    ax_recon_udf.axis('off')
                    fig.colorbar(im_recon, ax=ax_recon_udf, shrink=0.8)

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    
                    # wandb에 2x2 이미지 로깅
                    wandb.log({"reconstruction_progress": wandb.Image(fig)}, step=epoch)
                    plt.close(fig) # 창이 뜨지 않도록 닫아줌

                vqae_model.train() # 모델을 다시 훈련 모드로
                
        print("--- 훈련 완료 ---")
        
        model_path = 'vqae_model.pth'
        torch.save(vqae_model.state_dict(), model_path)
        artifact = wandb.Artifact('vqae-model', type='model')
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)
        print(f"Model saved and logged to W&B Artifacts as '{model_path}'")
    else:
        print(f"--- 테스트 모드 시작 ---")
        if os.path.exists(args.checkpoint):
            print(f"'{args.checkpoint}' 에서 훈련된 모델을 불러옵니다...")
            vqae_model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        else:
            print(f"에러: 체크포인트 파일 '{args.checkpoint}'을(를) 찾을 수 없습니다.")
            exit()

    print("\n=== 학습 완료 후 모델 성능 테스트 ===")
    vqae_model.eval()

    test_dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    with torch.no_grad():
        try:
            test_batch = next(iter(test_dataloader))
        except StopIteration:
            print("Test dataloader is empty. Skipping test.")
            test_batch = None

        if test_batch:
            stroke_seq, stroke_mask, udf_image = [x.to(device) for x in test_batch]
            
            # 1. 모델 forward pass를 통해 Teacher Forcing 결과와 잠재 코드를 모두 얻음
            recon_coords_tf, recon_pen_logits_tf, recon_udf, _, indices_stroke, _, _ = vqae_model(
                stroke_seq, stroke_mask, udf_image
            )
            # Teacher Forcing 결과를 (x, y, p) 형태로 결합
            reconstructed_strokes_tf = torch.cat([
                recon_coords_tf, 
                torch.sigmoid(recon_pen_logits_tf).unsqueeze(-1)
            ], dim=-1)

            # 2. 얻어진 이산 토큰으로 스트로크를 자기회귀적으로 생성
            reconstructed_strokes_ar = vqae_model.reconstruct_from_latents(indices_stroke)

            # --- 시각화 ---
            num_samples_to_show = min(10, config['batch_size']) # 샘플 수 조절
            print(f"결과를 이미지 파일로 저장합니다 ({num_samples_to_show}개 샘플)...")

            for i in range(num_samples_to_show):
                # --- [수정] 2x3 그리드 생성 ---
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                fig.suptitle(f'Sample {i+1} - Original vs Reconstructed', fontsize=16)
                
                # Axes 이름 재정의
                ax_orig_stroke = axes[0, 0]
                ax_tf_stroke = axes[0, 1]
                ax_ar_stroke = axes[0, 2]
                ax_orig_udf = axes[1, 0]
                ax_recon_udf = axes[1, 1]
                fig.delaxes(axes[1, 2]) # 비어있는 마지막 subplot 제거

                # --- Plot 1: 원본 스트로크 ---
                valid_length_orig = int(stroke_mask[i].sum().item())
                original_stroke = stroke_seq[i, :valid_length_orig, :2].cpu().numpy()
                ax_orig_stroke.plot(original_stroke[:, 0], original_stroke[:, 1], 'b-', lw=2)
                ax_orig_stroke.set_title('Original Stroke')
                ax_orig_stroke.set_xlim(0, 1); ax_orig_stroke.set_ylim(0, 1); ax_orig_stroke.invert_yaxis()
                ax_orig_stroke.set_aspect('equal'); ax_orig_stroke.grid(True, alpha=0.3)

                # --- [신규] Plot 2: Teacher Forcing 복원 스트로크 ---
                pen_states_tf = (reconstructed_strokes_tf[i, :, 2] > 0.5).float()
                # 원본과 동일한 길이만큼만 시각화
                recon_stroke_tf = reconstructed_strokes_tf[i, :valid_length_orig, :2].cpu().numpy()
                ax_tf_stroke.plot(recon_stroke_tf[:, 0], recon_stroke_tf[:, 1], 'g-', lw=2)
                ax_tf_stroke.set_title('Recon (Teacher Forcing)')
                ax_tf_stroke.set_xlim(0, 1); ax_tf_stroke.set_ylim(0, 1); ax_tf_stroke.invert_yaxis()
                ax_tf_stroke.set_aspect('equal'); ax_tf_stroke.grid(True, alpha=0.3)

                # --- Plot 3: 자기회귀(Autoregressive) 복원 스트로크 ---
                pen_states_ar = (reconstructed_strokes_ar[i, :, 2] > 0.5).float()
                zero_indices = (pen_states_ar == 0).nonzero(as_tuple=True)[0]
                valid_len_recon = zero_indices[0].item() + 1 if len(zero_indices) > 0 else config["max_stroke_len"]
                recon_stroke_ar = reconstructed_strokes_ar[i, :valid_len_recon, :2].cpu().numpy()
                ax_ar_stroke.plot(recon_stroke_ar[:, 0], recon_stroke_ar[:, 1], 'r-', lw=2)
                ax_ar_stroke.set_title('Recon (Autoregressive)')
                ax_ar_stroke.set_xlim(0, 1); ax_ar_stroke.set_ylim(0, 1); ax_ar_stroke.invert_yaxis()
                ax_ar_stroke.set_aspect('equal'); ax_ar_stroke.grid(True, alpha=0.3)
                
                # --- Plot 4: 원본 UDF ---
                original_udf_img = udf_image[i].squeeze().cpu().numpy()
                im_orig = ax_orig_udf.imshow(original_udf_img, cmap='hot', vmin=0, vmax=1)
                ax_orig_udf.set_title('Original UDF (Input)')
                ax_orig_udf.axis('off')
                fig.colorbar(im_orig, ax=ax_orig_udf, shrink=0.8)

                # --- Plot 5: 복원된 UDF ---
                reconstructed_udf_img = recon_udf[i].squeeze().cpu().numpy()
                im_recon = ax_recon_udf.imshow(reconstructed_udf_img, cmap='hot', vmin=0, vmax=1)
                ax_recon_udf.set_title('Reconstructed UDF')
                ax_recon_udf.axis('off')
                fig.colorbar(im_recon, ax=ax_recon_udf, shrink=0.8)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                save_path = f'./vqae_output/reconstruction_result_{i+1}.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"Sample {i+1}의 복원 결과가 '{save_path}'에 저장되었습니다.")

    print("\n=== 모든 테스트 완료 ===")