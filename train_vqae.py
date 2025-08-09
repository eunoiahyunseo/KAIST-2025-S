
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# from encoder_decoder import StrokeFusionVQAE
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
from rdp import rdp




parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='모델 체크포인트 경로 (기본값: vqae_model.pth)')
args = parser.parse_args()

def get_scheduled_epsilon(current_step, total_steps, decay_ratio=0.8, initial_epsilon=1.0, final_epsilon=0.0):
    """
    전체 훈련 기간의 특정 지점(decay_ratio)까지만 epsilon을 감소시키고,
    그 이후에는 final_epsilon 값을 유지하는 스케줄러.
    """
    # 1. Epsilon 감소가 끝나는 실제 스텝(step) 위치를 계산합니다.
    # decay_ratio=0.8 이면, 전체 스텝의 80% 지점에서 감소가 끝납니다.
    decay_end_step = int(total_steps * decay_ratio)

    # 2. 현재 스텝이 감소 기간을 지났는지 확인합니다.
    if current_step >= decay_end_step:
        return final_epsilon  # 감소가 끝났으므로 최종 epsilon 값을 반환합니다.

    # 3. 아직 감소 기간 내에 있다면, 선형적으로 값을 감소시킵니다.
    #    진행률(progress)을 전체 기간이 아닌 '감소 기간'에 맞춰 다시 계산합니다.
    progress = current_step / decay_end_step
    epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * progress
    
    return epsilon


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
    """스트로크와 UDF 이미지를 시각화합니다."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. 스트로크 시각화
    if len(stroke_points) > 0:
        x_coords = stroke_points[:, 0]
        y_coords = stroke_points[:, 1]
        ax1.plot(x_coords, y_coords, 'b-', linewidth=2, marker='o', markersize=3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.invert_yaxis()  # y축 뒤집기 (QuickDraw 스타일)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Original Stroke')
        
        # 시작점과 끝점 표시
        ax1.plot(x_coords[0], y_coords[0], 'go', markersize=8, label='Start')
        ax1.plot(x_coords[-1], y_coords[-1], 'ro', markersize=8, label='End')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'Empty Stroke', ha='center', va='center')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
    
    # 2. UDF 이미지 시각화
    if isinstance(udf_image, torch.Tensor):
        udf_image = udf_image.squeeze().cpu().numpy()
    
    im = ax2.imshow(udf_image, cmap='hot', interpolation='bilinear')
    ax2.set_title('UDF Image')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, shrink=0.8)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def visualize_batch_data(dataset, num_samples=5):
    """데이터셋에서 몇 개의 샘플을 시각화합니다."""
    print(f"Visualizing {num_samples} samples from the dataset...")
    
    for i in range(min(num_samples, len(dataset))):
        stroke_seq, stroke_mask, udf_image = dataset[i]
        
        # 유효한 부분만 추출 (패딩 제거)
        valid_length = int(stroke_mask.sum().item())
        valid_stroke = stroke_seq[:valid_length, :2].numpy()  # 좌표만 (pen state 제외)
        
        visualize_stroke_and_udf(
            valid_stroke, 
            udf_image, 
            title=f"Sample {i+1} - Valid Length: {valid_length}",
            save_path=f"sample_{i+1}_visualization.png"
        )
        
        print(f"Sample {i+1}:")
        print(f"  - Stroke shape: {stroke_seq.shape}")
        print(f"  - Valid length: {valid_length}")
        print(f"  - UDF shape: {udf_image.shape}")
        print(f"  - Coordinate range: x[{valid_stroke[:, 0].min():.3f}, {valid_stroke[:, 0].max():.3f}], y[{valid_stroke[:, 1].min():.3f}, {valid_stroke[:, 1].max():.3f}]")
        print()

class QuickDrawDataset(Dataset):
    def __init__(self, dic_item, max_stroke_len=48, udf_res=64, gamma=50.0, rdp_epsilon=0.2):
        self.max_stroke_len = max_stroke_len
        self.udf_res = udf_res
        self.gamma = gamma
        self.rdp_epsilon = rdp_epsilon
        self.all_strokes = self._preprocess(dic_item)


    def _preprocess(self, dic_item):
        """
        데이터를 모델에 맞는 형태로 변환합니다.
        - 종횡비를 유지하며(Isotropic) 스케일링합니다.
        - 스트로크의 중심을 기준으로 정렬합니다.
        """
        print("데이터 전처리 시작 (종횡비 유지 및 중심 정렬)...")
        processed_strokes = []
        i = 0
        for key, drawing in tqdm(dic_item):
            strokes_np = [np.array(stroke).T.astype(np.float32) for stroke in drawing]
            
            if not strokes_np: continue

            for stroke in strokes_np:
                if len(stroke) < 2:
                    continue
                

                simplified_stroke = rdp(stroke, epsilon=self.rdp_epsilon)


                if len(simplified_stroke) < 2:
                    continue

                # 1. 스트로크의 중심점과 크기(scale) 계산
                min_coords = np.min(simplified_stroke, axis=0)
                max_coords = np.max(simplified_stroke, axis=0)
                center = (min_coords + max_coords) / 2.0
                
                # 종횡비 유지를 위해 가로/세로 중 더 큰 쪽을 스케일 기준으로 삼음
                scale = (max_coords - min_coords).max()
                if scale < 1e-9:
                    scale = 1.0

                # 2. 중심 이동 및 등방성 스케일링
                # (stroke - center) -> 중심을 (0,0)으로 이동
                # / scale -> 가장 긴 축의 길이가 1이 되도록 축소. 범위는 [-0.5, 0.5]가 됨
                centered_and_scaled = (simplified_stroke - center) / scale
                
                # 3. 최종 범위 [0.15, 0.85]로 이동
                # + 0.5 -> [-0.5, 0.5] 범위를 [0, 1] 범위로 이동
                normalized_stroke = centered_and_scaled + 0.5
                normalized_stroke = (normalized_stroke * 0.7) + 0.15

                if np.isnan(normalized_stroke).any():
                    continue

                # if i == 0:
                #     normalized_stroke = normalized_stroke[20:-1]
                # i += 1
                    
                processed_strokes.append(normalized_stroke)
        
        print(f"전처리 완료. 총 {len(processed_strokes)}개의 스트로크 추출.")
        return processed_strokes

    def __len__(self):
        return len(self.all_strokes)

    def __getitem__(self, idx):
        stroke = self.all_strokes[idx]
        
        udf_image = create_udf_from_stroke(stroke, self.udf_res, self.gamma)
        udf_image_tensor = torch.from_numpy(udf_image).float().unsqueeze(0) # (1, H, W)
        
        stroke_len = len(stroke)

        if stroke_len > self.max_stroke_len:
            stroke = stroke[:self.max_stroke_len]
            stroke_len = self.max_stroke_len
        
        stroke_with_pen_state = np.ones((stroke_len, 3), dtype=np.float32) 
        stroke_with_pen_state[:, :2] = stroke
        
        if stroke_len > 0:
            stroke_with_pen_state[-1, 2] = 0 

        padded_stroke = np.zeros((self.max_stroke_len, 3), dtype=np.float32)

        padded_stroke[:stroke_len] = stroke_with_pen_state
        
        stroke_mask = np.zeros(self.max_stroke_len, dtype=np.float32)
        stroke_mask[:stroke_len] = 1.0
        
        return torch.from_numpy(padded_stroke), torch.from_numpy(stroke_mask), udf_image_tensor


def weights_init(m):
    """
    모델의 각 레이어에 대해 적절한 가중치 초기화를 적용하는 함수.
    model.apply(weights_init) 형태로 사용됩니다.
    """
    classname = m.__class__.__name__
    
    # Linear, Conv 레이어에 대한 초기화 (Kaiming He Normal)
    # ReLU 활성화 함수와 함께 사용할 때 성능이 좋다고 알려져 있음
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    # BatchNorm 레이어에 대한 초기화
    # Weight는 1, Bias는 0으로 초기화하여 정규화 효과를 초기에 유지
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
    # Embedding 레이어(코드북)에 대한 초기화
    # 작은 분산을 갖는 정규분포로 초기화하여 안정적인 학습 유도
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.02)


# --- 하이퍼파라미터 ---
MAX_STROKE_LEN = 48
UDF_RESOLUTION = 64 # udf 해상도

D_MODEL= 256 # transformer에서 사용되는 embedding크기
D_IMG = 64
D_SEQ = 64
D_F = 256 # 코드북 임베딩 크기, fusion vector 크기
NUM_EMBEDDINGS = 128 # 코드북 크기

BATCH_SIZE = 128
LEARNING_RATE_AE = 5e-4
EPOCHS_AE = 1000 # 예시용 에폭

DATA_SIZE = 10 # QuickDraw 데이터에서 사용할 최대 샘플 수


device = "cuda" if torch.cuda.is_available() else "cpu"
dic_item = []

category = "moon"
data_path = f"./data/quickdraw/{category}.ndjson"

with open(data_path, 'r') as f:
    data = ndjson.load(f)

dic = {}
cnt = 0
for i in data: 
    if i['recognized'] is True:
        dic[str(cnt)] = i['drawing']
        cnt += 1
        if cnt >= DATA_SIZE:
            break

dic_item += list(dic.items())

category = "apple"
data_path = f"./data/quickdraw/{category}.ndjson"

with open(data_path, 'r') as f:
    data = ndjson.load(f)

dic = {}
cnt = 0
for i in data: 
    if i['recognized'] is True:
        dic[str(cnt)] = i['drawing']
        cnt += 1
        if cnt >= DATA_SIZE:
            break

dic_item += list(dic.items())


print(len(dic_item[0]))

RDP_EPSILON = 0

dataset = QuickDrawDataset(
    dic_item, max_stroke_len=MAX_STROKE_LEN, udf_res=UDF_RESOLUTION, rdp_epsilon=RDP_EPSILON)

# 전처리된 데이터 시각화
print("=== 전처리된 데이터 시각화 ===")
visualize_batch_data(dataset, num_samples=10)

# Overfitting을 위한 작은 데이터셋 생성
OVERFIT = True  # True로 설정하면 소수의 데이터로만 학습
OVERFIT_SIZE = 20  # overfitting할 데이터 개수

if OVERFIT:
    # 처음 몇 개의 데이터만 사용
    from torch.utils.data import Subset
    overfit_indices = list(range(min(OVERFIT_SIZE, len(dataset))))
    dataset = Subset(dataset, overfit_indices)
    print(f"Overfitting mode: Using only {len(dataset)} samples")
    BATCH_SIZE = min(BATCH_SIZE, len(dataset))  # 배치 사이즈 조정

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

vqae_model = StrokeTokenizerVQVAE(
    max_stroke_len=MAX_STROKE_LEN,
    d_model=D_MODEL,
    d_img=D_IMG,
    d_seq=D_SEQ,
    nhead=4,
    num_layers=6,
    num_embeddings=NUM_EMBEDDINGS,
    embedding_dim=D_F,
    commitment_cost=0.25
).to(device)


if args.checkpoint is None:
    print("--- 훈련 모드 시작 ---")
    vqae_model.apply(weights_init)

    vqae_model.initialize_codebook_with_kmeans(dataloader, device)


    num_training_steps = EPOCHS_AE * len(dataloader)
    num_warmup_steps = int(num_training_steps * 0.1) # 훈련 스텝의 10%를 워밍업에 사용


    optimizer = torch.optim.Adam(vqae_model.parameters(), lr=LEARNING_RATE_AE)
    loss_fn_alex = lpips.LPIPS(net='vgg').to(device)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    recon_loss_coord_fn = nn.L1Loss(reduction='none') 

    # 학습 과정 추적을 위한 리스트
    train_losses = []
    coord_losses = []
    pen_losses = []
    img_losses = []
    vq_losses = []

    print("=== VQ-AE 학습 시작 ===")
    print(f"Total epochs: {EPOCHS_AE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE_AE}")
    print(f"Dataset size: {len(dataset)}")
    print()

    global_step = 0
    vq_warmup_steps = int(num_training_steps * 0.25)

    for epoch in range(EPOCHS_AE):
        
        epoch_total_loss = 0
        epoch_coord_loss = 0
        epoch_pen_loss = 0
        epoch_img_loss = 0
        epoch_vq_loss = 0
        num_batches = 0
        
        loop = tqdm(dataloader, leave=True)
        DECAY_RATIO = 0.8  # 전체 훈련의 80% 시점까지만 epsilon을 감소시킴

        for batch_idx, (stroke_seq, stroke_mask, udf_image) in enumerate(loop):
            stroke_seq, stroke_mask, udf_image = stroke_seq.to(device), stroke_mask.to(device), udf_image.to(device)
            vq_weight = min(1.0, global_step / vq_warmup_steps)

            # print('asdfadsf', stroke_mask[0, :-1])
            # print(stroke_seq[0, :-1, :2])

            optimizer.zero_grad()

            epsilon = get_scheduled_epsilon(
                    global_step, 
                    num_training_steps, 
                    decay_ratio=DECAY_RATIO
                )
            recon_coords, recon_pen_logits, recon_udf, vq_loss, i = vqae_model(
                stroke_seq, stroke_mask, udf_image)
            # recon_coords, recon_pen_logits, vq_loss, i = vqae_model(
            #     stroke_seq, stroke_mask, udf_image)
            print('i', i[:3, :30])
            
            target_coords = stroke_seq[:, :, :2]  # (B, Np, 2)
            target_pen_state = stroke_seq[:, :, 2] # (B, Np)
            


            loss_pen = F.binary_cross_entropy_with_logits(recon_pen_logits, target_pen_state.float(), reduction='none')
            loss_pen = (loss_pen * stroke_mask).sum() / stroke_mask.sum().clamp(min=1e-9)

            loss_coords = recon_loss_coord_fn(recon_coords, target_coords) # (B, Np, 2)
            
            weight_mask = torch.ones_like(stroke_mask)
            weight_mask[:, 0] = 5
            weight_mask = weight_mask.unsqueeze(-1)  # (B, Np, 1)

            weighted_loss_per_point = loss_coords * weight_mask

            loss_coords = (weighted_loss_per_point * stroke_mask.unsqueeze(-1)).sum() / (stroke_mask.unsqueeze(-1) * weight_mask).sum().clamp(min=1e-9)


            target_diffs = target_coords[:, 1:, :] - target_coords[:, :-1, :]
            recon_diffs = recon_coords[:, 1:, :] - recon_coords[:, :-1, :]

            diff_mask = stroke_mask[:, 1:]
            loss_diff = recon_loss_coord_fn(recon_diffs, target_diffs) # (B, Np-1, 2)
            loss_diff = (loss_diff.mean(dim=-1) * diff_mask).sum() / diff_mask.sum().clamp(min=1e-9)


            # Overfitting 모드에서는 더 자주 출력
            if OVERFIT and epoch % 1 == 0 and batch_idx == 0:
                sigmoid = nn.Sigmoid()
                print(f'\nEpoch {epoch+1} - Batch {batch_idx+1}:')
                print('recon_coords (first 10):', recon_coords[0, :35, 0].detach().cpu().numpy())
                print('target_coords (first 10):', target_coords[0, :35, 0].detach().cpu().numpy())
                print('recon_pen (first 10):', torch.where(sigmoid(recon_pen_logits[0, :35]) > 0.5, 1, 0).detach().cpu().numpy())
                print('target_pen (first 10):', target_pen_state[0, :35].detach().cpu().numpy())
            elif not OVERFIT and epoch % 5 == 0 and batch_idx == 0:
                sigmoid = nn.Sigmoid()
                print('recon_stroke:', recon_coords[0, :49, 0] * stroke_mask[0, :49])
                print('target_stroke:', stroke_seq[0, :49, 0])
                
                print('recon_coords:', torch.where(sigmoid(recon_pen_logits[0, :49] * stroke_mask[0, :49]) > 0.5 , 1, 0))
                print('target_coords:', target_pen_state[0, :49])

            loss_l1_img = F.l1_loss(recon_udf, udf_image)
            recon_udf_3ch = recon_udf.repeat(1, 3, 1, 1)
            udf_image_3ch = udf_image.repeat(1, 3, 1, 1)
            loss_percep = loss_fn_alex(recon_udf_3ch, udf_image_3ch).mean()
            loss_img = 1 * loss_l1_img + 1 * loss_percep


            # avg_probs = torch.mean(soft_one_hot, dim=0) # 배치 내 평균 코드 사용 확률
            # # 엔트로피 계산 (엔트로피를 최대화해야 하므로, 음수를 취해 손실로 만듦)
            # entropy_loss = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
            # print('entropy_loss: ', entropy_loss.item())

            total_loss =    1.0 * loss_coords + \
                            0.2 * loss_pen + \
                            0.1 * loss_img + \
                            0.5 * vq_loss 
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vqae_model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            global_step += 1


            # 손실 누적
            epoch_total_loss += total_loss.item()
            epoch_coord_loss += loss_coords.item()
            epoch_pen_loss += loss_pen.item()
            epoch_img_loss += loss_img.item()
            epoch_vq_loss += vq_loss.item()
            num_batches += 1
            
            loop.set_description(f"Epoch [Gumbel VQ-AE] {epoch+1}/{EPOCHS_AE}")
            loop.set_postfix(
                total_loss=total_loss.item(),
                coord_loss=loss_coords.item(),
                recon_loss_udf=loss_img.item(),
                pen_loss=loss_pen.item(),
                loss_diff=loss_diff.item(),
                vq_loss=vq_loss.item(),
            )
        
        # 에폭별 평균 손실 저장
        avg_total_loss = epoch_total_loss / num_batches
        avg_coord_loss = epoch_coord_loss / num_batches
        avg_pen_loss = epoch_pen_loss / num_batches
        avg_img_loss = epoch_img_loss / num_batches
        avg_vq_loss = epoch_vq_loss / num_batches
        
        train_losses.append(avg_total_loss)
        coord_losses.append(avg_coord_loss)
        pen_losses.append(avg_pen_loss)
        img_losses.append(avg_img_loss)
        vq_losses.append(avg_vq_loss)

        # 주기적으로 손실 출력
        if (epoch + 1) % 10 == 0 or OVERFIT:
            print(f"\nEpoch {epoch+1}/{EPOCHS_AE} Summary:")
            print(f"  Total Loss: {avg_total_loss:.6f}")
            print(f"  Coord Loss: {avg_coord_loss:.6f}")
            print(f"  Pen Loss: {avg_pen_loss:.6f}")
            print(f"  Image Loss: {avg_img_loss:.6f}")
            # print(f"  VQ Loss: {avg_vq_loss:.6f}")
            print()
            
    print("--- 1단계: VQ-AE 훈련 완료 ---")

    # 학습 곡선 시각화
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(train_losses)
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(coord_losses)
    plt.title('Coordinate Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(pen_losses)
    plt.title('Pen State Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(img_losses)
    plt.title('Image Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(vq_losses)
    plt.title('VQ Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Training curves saved to: training_curves.png")

    # 훈련된 모델 저장
    torch.save(vqae_model.state_dict(), 'vqae_model.pth')
else:
    print(f"--- 테스트 모드 시작 ---")
    if os.path.exists(args.checkpoint):
        print(f"'{args.checkpoint}' 에서 훈련된 모델을 불러옵니다...")
        vqae_model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print(f"에러: 체크포인트 파일 '{args.checkpoint}'을(를) 찾을 수 없습니다.")
        exit()



# =======================================================================================
# ===== 테스트 및 시각화 섹션 (이 부분이 수정되었습니다) =====
# =======================================================================================
print("\n=== 학습 완료 후 모델 성능 테스트 ===")
vqae_model.eval()

test_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

with torch.no_grad():
    try:
        test_batch = next(iter(test_dataloader))
    except StopIteration:
        print("Test dataloader is empty. Skipping test.")
        test_batch = None

    if test_batch:
        stroke_seq, stroke_mask, udf_image = [x.to(device) for x in test_batch]

        # --- 1. 모델을 통과시켜 복원된 결과물과 이산 토큰(indices) 얻기 ---
        # recon_udf: 복원된 UDF 이미지
        # indices_seq: 스트로크 생성을 위한 이산 토큰 시퀀스
        _, _, recon_udf, _, indices_seq = vqae_model(stroke_seq, stroke_mask, udf_image)
        print('indices_seq shape:', indices_seq.shape)
        print('indices_seq shape:', indices_seq)
        
        # --- 2. 얻어진 이산 토큰으로 스트로크를 자기회귀적으로 생성 ---
        reconstructed_strokes = vqae_model.generate_from_indices(indices_seq)
        
        num_samples_to_show = min(20, BATCH_SIZE)
        print(f"결과를 이미지 파일로 저장합니다 ({num_samples_to_show}개 샘플)...")

        for i in range(num_samples_to_show):
            # --- 시각화를 위한 2x2 그리드 생성 ---
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            fig.suptitle(f'Sample {i+1} - Original vs Reconstructed', fontsize=16)
            
            ax_orig_stroke = axes[0, 0]
            ax_recon_stroke = axes[0, 1]
            ax_orig_udf = axes[1, 0]
            ax_recon_udf = axes[1, 1]

            # --- Plot 1: 원본 스트로크 ---
            valid_length_orig = int(stroke_mask[i].sum().item())
            original_stroke = stroke_seq[i, :valid_length_orig, :2].cpu().numpy()
            ax_orig_stroke.plot(original_stroke[:, 0], original_stroke[:, 1], 'b-', lw=2)
            ax_orig_stroke.set_title('Original Stroke')
            ax_orig_stroke.set_xlim(0, 1); ax_orig_stroke.set_ylim(0, 1); ax_orig_stroke.invert_yaxis()
            ax_orig_stroke.set_aspect('equal'); ax_orig_stroke.grid(True, alpha=0.3)
            print('original stroke: ', stroke_seq[i, :35, 0])

            # --- Plot 2: 복원된 스트로크 ---
            pen_states = reconstructed_strokes[i, :, 2]
            ideal_cumsum = torch.arange(1, len(pen_states) + 1, device=pen_states.device)
            comparison = (torch.cumsum(pen_states, dim=0) == ideal_cumsum)
            valid_length_recon = torch.sum(comparison)

            print('reconstructed stroke: ', reconstructed_strokes[i, :35, 0])
            reconstructed_stroke = reconstructed_strokes[i, :valid_length_recon, :2].cpu().numpy()

            ax_recon_stroke.plot(reconstructed_stroke[:, 0], reconstructed_stroke[:, 1], 'r-', lw=2)
            ax_recon_stroke.set_title('Reconstructed Stroke')
            ax_recon_stroke.set_xlim(0, 1); ax_recon_stroke.set_ylim(0, 1); ax_recon_stroke.invert_yaxis()
            ax_recon_stroke.set_aspect('equal'); ax_recon_stroke.grid(True, alpha=0.3)

            
            # --- Plot 3: 원본 UDF ---
            original_udf_img = udf_image[i].squeeze().cpu().numpy()
            im_orig = ax_orig_udf.imshow(original_udf_img, cmap='hot', vmin=0, vmax=1)
            ax_orig_udf.set_title('Original UDF (Input)')
            ax_orig_udf.axis('off')
            fig.colorbar(im_orig, ax=ax_orig_udf, shrink=0.8)

            # --- Plot 4: 복원된 UDF ---
            reconstructed_udf_img = recon_udf[i].squeeze().cpu().numpy()
            im_recon = ax_recon_udf.imshow(reconstructed_udf_img, cmap='hot', vmin=0, vmax=1)
            ax_recon_udf.set_title('Reconstructed UDF')
            ax_recon_udf.axis('off')
            fig.colorbar(im_recon, ax=ax_recon_udf, shrink=0.8)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # suptitle과 겹치지 않게 조정
            save_path = f'reconstruction_result_{i+1}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Sample {i+1}의 복원 결과가 '{save_path}'에 저장되었습니다.")

print("\n=== 모든 테스트 완료 ===")