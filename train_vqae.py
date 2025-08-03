import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from encoder_decoder import StrokeFusionVQAE
from tqdm import tqdm
import ndjson
import numpy as np
import lpips

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

class QuickDrawDataset(Dataset):
    def __init__(self, dic_item, max_stroke_len=48, udf_res=64, gamma=50.0):
        self.max_stroke_len = max_stroke_len
        self.udf_res = udf_res
        self.gamma = gamma
        self.all_strokes = self._preprocess(dic_item)

    # QuickDrawDataset 클래스의 _preprocess 함수를 아래 코드로 교체하세요.

    def _preprocess(self, dic_item):
        """
        데이터를 모델에 맞는 형태로 변환합니다.
        - 종횡비를 유지하며(Isotropic) 스케일링합니다.
        - 스트로크의 중심을 기준으로 정렬합니다.
        """
        print("데이터 전처리 시작 (종횡비 유지 및 중심 정렬)...")
        processed_strokes = []
        for key, drawing in tqdm(dic_item):
            strokes_np = [np.array(stroke).T.astype(np.float32) for stroke in drawing]
            
            if not strokes_np: continue

            for stroke in strokes_np:
                if len(stroke) < 2:
                    continue

                # 1. 스트로크의 중심점과 크기(scale) 계산
                min_coords = np.min(stroke, axis=0)
                max_coords = np.max(stroke, axis=0)
                center = (min_coords + max_coords) / 2.0
                
                # 종횡비 유지를 위해 가로/세로 중 더 큰 쪽을 스케일 기준으로 삼음
                scale = (max_coords - min_coords).max()
                if scale < 1e-9:
                    scale = 1.0

                # 2. 중심 이동 및 등방성 스케일링
                # (stroke - center) -> 중심을 (0,0)으로 이동
                # / scale -> 가장 긴 축의 길이가 1이 되도록 축소. 범위는 [-0.5, 0.5]가 됨
                centered_and_scaled = (stroke - center) / scale
                
                # 3. 최종 범위 [0.15, 0.85]로 이동
                # + 0.5 -> [-0.5, 0.5] 범위를 [0, 1] 범위로 이동
                normalized_stroke = centered_and_scaled + 0.5

                if np.isnan(normalized_stroke).any():
                    continue
                    
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





# --- 하이퍼파라미터 ---
# VQ-AE용
MAX_STROKE_LEN = 48
UDF_RESOLUTION = 64
D_F = 256
NUM_EMBEDDINGS = 256 # 코드북 크기
BATCH_SIZE = 128
LEARNING_RATE_AE = 2e-4
EPOCHS_AE = 500 # 예시용 에폭


LAMBDA_L1_IMG = 1.0
LAMBDA_PERCEP = 1.0
LAMBDA_PEN = 0.1
LAMBDA_COORD = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"
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
        if cnt >= 70000:
            break
dic_item = list(dic.items())
print(len(dic_item[0]))


dataset = QuickDrawDataset(dic_item, max_stroke_len=MAX_STROKE_LEN, udf_res=UDF_RESOLUTION)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

vqae_model = StrokeFusionVQAE(
    max_stroke_len=MAX_STROKE_LEN,
    d_f=D_F,
    num_embeddings=NUM_EMBEDDINGS
).to(device)

optimizer = torch.optim.Adam(vqae_model.parameters(), lr=LEARNING_RATE_AE)
loss_fn_alex = lpips.LPIPS(net='vgg').to(device)


recon_loss_coord_fn = nn.L1Loss(reduction='none') 

for epoch in range(EPOCHS_AE):
    loop = tqdm(dataloader, leave=True)
    for stroke_seq, stroke_mask, udf_image in loop:
        stroke_seq, stroke_mask, udf_image = stroke_seq.to(device), stroke_mask.to(device), udf_image.to(device)

        optimizer.zero_grad()
        
        recon_coords, recon_pen_logits, recon_udf, vq_loss = vqae_model(stroke_seq, stroke_mask, udf_image)
        
        target_coords = stroke_seq[:, :, :2]  # (B, Np, 2)
        target_pen_state = stroke_seq[:, :, 2].long() # (B, Np)

        loss_pen = F.binary_cross_entropy_with_logits(recon_pen_logits, target_pen_state.float(), reduction='none')
        loss_pen = (loss_pen * stroke_mask).sum() / stroke_mask.sum().clamp(min=1e-9)

        loss_coords = recon_loss_coord_fn(recon_coords, target_coords) # (B, Np, 2)
        loss_coords = (loss_coords.mean(dim=-1) * stroke_mask).sum() / stroke_mask.sum().clamp(min=1e-9)

        sigmoid = nn.Sigmoid()
        if epoch % 5 == 0:
            print('recon_stroke:', recon_coords[0, :49, 0] * stroke_mask[0, :49])
            print('target_stroke:', stroke_seq[0, :49, 0])
            
            print('recon_coords:', torch.where(sigmoid(recon_pen_logits[0, :49] * stroke_mask[0, :49]) > 0.5 , 1, 0))
            print('target_coords:', target_pen_state[0, :49])

        loss_l1_img = F.l1_loss(recon_udf, udf_image)

        # 4. Perceptual 손실
        # LPIPS는 3채널 이미지를 기대하므로 채널을 복제
        recon_udf_3ch = recon_udf.repeat(1, 3, 1, 1)
        udf_image_3ch = udf_image.repeat(1, 3, 1, 1)
        loss_percep = loss_fn_alex(recon_udf_3ch, udf_image_3ch).mean() # .mean()으로 스칼라 값으로 만듦

        # 5. 최종 이미지 손실
        loss_img = LAMBDA_L1_IMG * loss_l1_img + LAMBDA_PERCEP * loss_percep
        
        # 6. 최종 손실 합산 (가중치 람다는 여기서는 1로 가정)
        # 논문: Ls = λ_CE*L_CE + λ_L1*L_1 + λ_img*L_img + λ_KL*L_KL
        # 우리: total = loss_pen + loss_coords + recon_loss_udf + vq_loss
        total_loss = loss_coords + 0.1 * loss_pen + 0.1 * loss_img + vq_loss


        total_loss.backward()
        optimizer.step()
        
        loop.set_description(f"Epoch [VQ-AE] {epoch+1}/{EPOCHS_AE}")
        loop.set_postfix(
            total_loss=total_loss.item(),
            coord_loss=loss_coords.item(),
            recon_loss_udf=loss_img.item(),
            pen_loss=loss_pen.item(),
            vq_loss=vq_loss.item()
        )
        
print("--- 1단계: VQ-AE 훈련 완료 ---")

# 훈련된 모델 저장
torch.save(vqae_model.state_dict(), 'vqae_model.pth')