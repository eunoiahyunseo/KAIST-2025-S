import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from encoder_decoder_modify import MultimodalVQVAE, MultimodalVAE, BoundingBoxVQVAE, BoundingBoxVQVAE_Dual

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

parser.add_argument('--run_name', type=str, default="vq-vae-training",
                    help='W&B에 기록될 실행(run) 이름')

parser.add_argument('--model_type', type=str, default='stroke', choices=['stroke', 'bbox'],
                    help='어떤 모델을 훈련할지 선택 (stroke 또는 bbox)')

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
        fig = visualize_stroke_and_udf(valid_stroke, udf_image, title=f"Sample {i+1} - Valid Length: {valid_length}", save_path=f"./vis_batch/sample_{i+1}_visualization.png")
        plt.close(fig) # Figure 객체를 닫아 메모리 관리

class QuickDrawDataset(Dataset):
    def __init__(self, dic_item, max_stroke_len=48, udf_res=64, gamma=50.0, rdp_epsilon=0.2, cache_path="./data/preprocessed_strokes.pt"):
        self.max_stroke_len = max_stroke_len
        self.udf_res = udf_res
        self.gamma = gamma
        self.rdp_epsilon = rdp_epsilon
        self.canvas_size = 256.0


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

            self.all_bboxes = cached_data['bboxes']
            print(f"캐시 로딩 완료. 총 {len(self.all_strokes)}개의 스트로크.")
        else:
            print(f"⏳ 전처리된 데이터 캐시 없음. 전처리를 시작합니다.")
            # 캐시가 없으면 전처리 함수를 호출합니다.
            self.all_strokes, self.all_udfs, self.all_bboxes = self._preprocess(dic_item)
            
            # 다음 사용을 위해 전처리된 데이터를 딕셔너리 형태로 저장합니다.
            print(f"💾 전처리된 데이터를 캐시에 저장합니다: {cache_path}")
            data_to_save = {'strokes': self.all_strokes, 'udfs': self.all_udfs, 'bboxes': self.all_bboxes}
            torch.save(data_to_save, cache_path)

    def _preprocess(self, dic_item):
        print("데이터 전처리 시작...")
        processed_strokes = []
        processed_udfs = []
        processed_bboxes = []

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

                

                # bounding box
                x_min, y_min = np.min(simplified_stroke, axis=0)
                x_max, y_max = np.max(simplified_stroke, axis=0)

                width = x_max - x_min
                height = y_max - y_min             

                center_x = x_min + width / 2
                center_y = y_min + height / 2 # 뺄셈(-)을 덧셈(+)으로, 정수 나눗셈(//)을 일반 나눗셈(/)으로 수정
                print('width:', width, 'height:', height)

                center_x = (center_x / self.canvas_size) - 0.5
                center_y = (center_y / self.canvas_size) - 0.5
                width = width / self.canvas_size
                height = height / self.canvas_size
                print('center_x:', center_x, 'center_y:', center_y)

                if width == 0: width = 1.0
                if height == 0: height = 1.0
                bbox = np.array([center_x, center_y, width, height], dtype=np.float32)

                normalized_stroke = (centered_and_scaled + 0.5) * 0.7 + 0.15

                if not np.isnan(normalized_stroke).any(): processed_strokes.append(normalized_stroke)
                udf_image = create_udf_from_stroke(normalized_stroke, self.udf_res, self.gamma)
                processed_udfs.append(udf_image)
                processed_bboxes.append(bbox)

        print(f"전처리 완료. 총 {len(processed_strokes)}개의 스트로크 추출.")
        return processed_strokes, processed_udfs, processed_bboxes
    def __len__(self):
        return len(self.all_strokes)
    def __getitem__(self, idx):
        stroke = self.all_strokes[idx]
        # udf_image_tensor = torch.from_numpy(create_udf_from_stroke(stroke, self.udf_res, self.gamma)).float().unsqueeze(0)
        udf_image = self.all_udfs[idx]
        udf_image_tensor = torch.from_numpy(udf_image).float().unsqueeze(0)

        bbox = self.all_bboxes[idx]
        stroke_len = min(len(stroke), self.max_stroke_len)
        stroke_with_pen_state = np.ones((stroke_len, 3), dtype=np.float32)
        stroke_with_pen_state[:, :2] = stroke[:stroke_len]
        if stroke_len > 0: stroke_with_pen_state[-1, 2] = 0
        padded_stroke = np.zeros((self.max_stroke_len, 3), dtype=np.float32)
        padded_stroke[:stroke_len] = stroke_with_pen_state
        stroke_mask = np.zeros(self.max_stroke_len, dtype=np.float32)
        stroke_mask[:stroke_len] = 1.0
        return torch.from_numpy(padded_stroke), torch.from_numpy(stroke_mask), udf_image_tensor, torch.from_numpy(bbox).float()


class QuickDrawDrawingDataset(Dataset):
    def __init__(self, dic_item, rdp_epsilon=0.2, canvas_size=256.0):
        self.rdp_epsilon = rdp_epsilon
        self.canvas_size = canvas_size
        self.drawings_with_bboxes = self._preprocess(dic_item)

    def _preprocess(self, dic_item):
        processed_drawings = []
        for drawing_data in tqdm(dic_item, desc="Preprocessing Drawings into Sequences"):
            bboxes_in_drawing = []
            raw_strokes = [np.array(stroke).T.astype(np.float32) for stroke in drawing_data]
            if not raw_strokes: continue
            
            for stroke in raw_strokes:
                if len(stroke) < 2: continue
                simplified_stroke = rdp(stroke, epsilon=self.rdp_epsilon) if self.rdp_epsilon > 0 else stroke
                if len(simplified_stroke) < 2: continue
                
                x_min, y_min = np.min(simplified_stroke, axis=0)
                x_max, y_max = np.max(simplified_stroke, axis=0)

                w = x_max - x_min; h = y_max - y_min
                if w < 1e-9: w = 1.0
                if h < 1e-9: h = 1.0
                

                # 정규화된 (cx, cy, w, h)
                cx = ((x_min + w / 2) / self.canvas_size) - 0.5
                cy = ((y_min + h / 2) / self.canvas_size) - 0.5

                w_norm = w / self.canvas_size
                h_norm = h / self.canvas_size
                bboxes_in_drawing.append([cx, cy, w_norm, h_norm])
            
            if bboxes_in_drawing:
                processed_drawings.append(np.array(bboxes_in_drawing, dtype=np.float32))
        return processed_drawings

    def __len__(self): return len(self.drawings_with_bboxes)
    def __getitem__(self, idx): return self.drawings_with_bboxes[idx]

class QuickDrawDrawingDataset2(Dataset):
    def __init__(self, dic_item, config):
        self.config = config
        # __init__에서 필요한 config 값들을 미리 꺼내놓으면 더 좋습니다.
        self.rdp_epsilon = config['rdp_epsilon']
        self.udf_resolution = config['udf_resolution']
        self.gamma = config['gamma']
        
        self.drawings = self._preprocess(dic_item)

    def _preprocess(self, dic_item):
        processed_drawings = []
        for drawing_data in tqdm(dic_item, desc="Preprocessing Drawings with UDFs"):
            # ### [추가] 'udfs'를 저장할 공간을 딕셔너리에 추가 ###
            drawing_info = {'strokes': [], 'bboxes': [], 'udfs': []}
            raw_strokes = [np.array(stroke).T.astype(np.float32) for stroke in drawing_data]
            if not raw_strokes: continue

            for stroke in raw_strokes:
                if len(stroke) < 2: continue
                simplified_stroke = rdp(stroke, epsilon=self.rdp_epsilon) if self.rdp_epsilon > 0 else stroke
                if len(simplified_stroke) < 2: continue

                min_coords, max_coords = np.min(simplified_stroke, axis=0), np.max(simplified_stroke, axis=0)
                center = (min_coords + max_coords) / 2.0
                scale = (max_coords - min_coords).max()
                if scale < 1e-9: scale = 1.0
                
                # 스트로크 정규화 (이전과 동일)
                normalized_stroke = ((simplified_stroke - center) / scale + 0.5) * 0.7 + 0.15
                
                drawing_info['strokes'].append(normalized_stroke)

                ### [핵심 추가] 정규화된 스트로크로 UDF를 생성하여 저장 ###
                udf_image = create_udf_from_stroke(normalized_stroke, self.udf_resolution, self.gamma)
                drawing_info['udfs'].append(udf_image)

            if drawing_info['strokes']:
                processed_drawings.append(drawing_info)
        return processed_drawings

    def __len__(self):
        return len(self.drawings)

    def __getitem__(self, idx):
        return self.drawings[idx]
    

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


config = {
    "max_stroke_len": 48,
    "udf_resolution": 64,
    "d_model": 256,
    "d_img": 128,
    "d_seq": 128,
    "embedding_dim": 256,
    "num_embeddings": 293,
    "batch_size": 256,
    "learning_rate": 1e-4,
    "1-epochs": 1000,
    "2-epochs": 1500,
    "3-epochs": 1000,
    "gamma": 100.0,
    "rdp_epsilon": 0.0,
    "commitment_cost": 0.25,
    "n_head": 8,
    "num_layers": 6,
    "overfit_size": 10,
    "data_size_per_category": 50,
    "loss_weight": {
        "coord": 1.0,
        "pen": 0.1,
        "img": 0.1,
        "vq": 0.8,
        "kl": 0.01
    },
    "image_num_layer": 4,
    "bbox_hidden_dim": 64,
    "bbox_codebook_size": 50,
    "bbox_codebook_dim": 128
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

    # CATEGORIES = ["airplane", "apple", "bus", "cat", "chair", "face", "fish", "moon", "pizza", "shoe", "spider", "television", "train", "umbrella"]
    CATEGORIES = ["umbrella", "apple"]


    dataset = get_quickdraw_dataset(CATEGORIES, config)

    print(f"\n총 {len(dataset)}개의 데이터 로딩 완료!")

    OVERFIT = False
    if OVERFIT:
        from torch.utils.data import Subset
        overfit_indices = list(range(min(config["overfit_size"], len(dataset))))
        dataset = Subset(dataset, overfit_indices)
        print(f"Overfitting mode: Using only {len(dataset)} samples")
        config["batch_size"] = min(config["batch_size"], len(dataset))
        # config['loss_weight']['kl'] = 0

    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True)
    fixed_dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    last_batch = None
    for batch in fixed_dataloader:
        last_batch = batch

    # 마지막으로 저장된 배치를 로깅용으로 사용
    fixed_batch = last_batch

    wandb.init(
        project="stroke-vq-vae-pooling", 
        name=args.run_name,
        config=config
    )
    
    if args.model_type == 'bbox' and args.checkpoint is None:

        model = BoundingBoxVQVAE_Dual(
            # input_dim=4, # (cx, cy)
            num_loc_embeddings=config["bbox_codebook_size"],
            num_size_embeddings=config["bbox_codebook_size"],
            loc_latent_dim=config["bbox_codebook_dim"] // 2,
            size_latent_dim=config["bbox_codebook_dim"] // 2,
            commitment_cost=config["commitment_cost"]
        ).to(device)



        model.apply(weights_init)
        model.initialize_codebooks(dataloader, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

        for epoch in range(config["2-epochs"]):
            loop = tqdm(dataloader, leave=True)
            for batch_idx, (batch_stroke_seq, batch_stroke_mask, batch_udf_image, bbox) in enumerate(loop):
                bbox = bbox.to(device)
                optimizer.zero_grad()

                recon_bbox, losses, perplexities, indices = model(bbox)
                print('recon_bbox shape:', recon_bbox.shape)
                print('bbox shape:', bbox.shape)
                reconstruction_loss = F.mse_loss(recon_bbox, bbox)
                total_loss = reconstruction_loss + losses['vq_loss']

                total_loss.backward()
                optimizer.step()

                loop.set_postfix(
                    loss=total_loss.item(),
                    perp_loc=perplexities['loc'].item(),
                    perp_size=perplexities['size'].item()
                )

            if (epoch + 1) % 1000 == 0: # 10 에폭마다 시각화
                
                # 시각화를 위해 데이터를 CPU로 이동하고 numpy 배열로 변환
                original_bboxes = bbox.cpu().numpy()
                reconstructed_bboxes = recon_bbox.detach().cpu().numpy()
                
                # reconstructed_bboxes1 = recon_bbox1.detach().cpu().numpy()
                # reconstructed_bboxes2 = recon_bbox2.detach().cpu().numpy()
                
                num_to_show = min(4, original_bboxes.shape[0]) # 최대 4개 샘플 시각화

                # ### [수정] 시각화에 사용할 컬러맵 정의 ###
                # 'rainbow' 컬러맵에서 16개의 고유한 색상을 가져옵니다.
                colors = plt.cm.get_cmap('rainbow', num_to_show)

                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                fig.suptitle(f'Epoch {epoch+1}: BBox Reconstruction')

                ### [수정] 올바른 서브플롯 설정 방식 ###
                # 원본 바운딩 박스 플롯 설정
                axes[0].set_title('Original Bounding Boxes')
                axes[0].set_xlim(-0.6, 0.6)
                axes[0].set_ylim(-0.6, 0.6)
                axes[0].set_aspect('equal', adjustable='box')
                axes[0].grid(True, alpha=0.3)

                # 복원된 바운딩 박스 플롯 설정
                axes[1].set_title('Reconstructed Bounding Boxes')
                axes[1].set_xlim(-0.6, 0.6)
                axes[1].set_ylim(-0.6, 0.6)
                axes[1].set_aspect('equal', adjustable='box')
                axes[1].grid(True, alpha=0.3)

                for i in range(num_to_show):
                    # i번째 샘플에 대한 색상 선택
                    color = colors(i)

                    ### [수정] 잘못된 구문(\) 제거 ###
                    # Original Bbox 그리기
                    center_x, center_y, w, h = original_bboxes[i]
                    x_min = center_x - w / 2
                    y_min = center_y - h / 2
                    rect = plt.Rectangle((x_min, y_min), w, h, linewidth=1.5, edgecolor=color, facecolor='none', alpha=0.8)
                    axes[0].add_patch(rect)

                    # Reconstructed Bbox 그리기
                    center_x_r, center_y_r, = reconstructed_bboxes[i, :2]
                    w_r, h_r = reconstructed_bboxes[i, 2:]

                    x_min_r = center_x_r - w_r / 2
                    y_min_r = center_y_r - h_r / 2
                    rect_r = plt.Rectangle((x_min_r, y_min_r), w_r, h_r, linewidth=1.5, edgecolor=color, facecolor='none', alpha=0.8)
                    axes[1].add_patch(rect_r)
                
                # 로컬 파일로 저장
                save_dir = "./bbox_visualizations"
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"epoch_{epoch+1}_reconstruction.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                
                # 메모리 관리를 위해 plot을 닫아줍니다.
                plt.close(fig)
                
                print(f"✅ Visualization saved to {save_path}")

                model_path = './vq-vae-training/vqae_model.pth'
        
        torch.save(model.state_dict(), './checkpoints/vq-vae-training/bbox_vqvae_layout.pth')

        print("\n--- Preparing QuickDraw Drawing Dataset ---")
        all_drawings_raw = []
        for category in CATEGORIES:
            with open(f"./data/quickdraw/{category}.ndjson", 'r') as f:
                data = ndjson.load(f)
            recognized = [item['drawing'] for item in data if item.get('recognized', True)]
            all_drawings_raw += recognized[:config["data_size_per_category"]]
        
        drawing_dataset = QuickDrawDrawingDataset(all_drawings_raw)
        
        print(f"총 {len(drawing_dataset)}개의 그림 데이터셋 생성 예정")
        layout_token_sequences = []
        with torch.no_grad():
            for drawing_bboxes in tqdm(drawing_dataset, desc="Tokenizing Layouts"):
                bboxes = torch.from_numpy(drawing_bboxes).to(device)
                locations = bboxes[:, :2]; sizes = bboxes[:, 2:]

                loc_indices, size_indices = model.encode(bboxes)

                final_indices = loc_indices * config["bbox_codebook_size"] + size_indices
                layout_token_sequences.append(final_indices.cpu().numpy())


        print("\n--- Stage 3: Splitting and saving to .npz format ---")
        val_size = len(layout_token_sequences) // 10
        train_sequences = layout_token_sequences[:-val_size]
        val_sequences = layout_token_sequences[-val_size:]

        train_data_dict = {f'drawing_{i}': seq for i, seq in enumerate(train_sequences)}
        val_data_dict = {f'drawing_{i}': seq for i, seq in enumerate(val_sequences)}
        print(val_data_dict)

        np.savez_compressed('./data/layout/train.npz', **train_data_dict)
        np.savez_compressed('./data/layout/val.npz', **val_data_dict)

        print(f"\n✅ Data processing complete.")
        print(f"Train data saved to 'train.npz' ({len(train_sequences)} drawings)")
        print(f"Validation data saved to 'val.npz' ({len(val_sequences)} drawings)")
        
        
    if args.model_type == 'stroke' and args.checkpoint is None:

        vqae_model = MultimodalVQVAE(
            d_model=config["d_model"],
            d_img=config["d_img"],
            d_seq=config["d_seq"],
            n_head=config["n_head"],
            num_layers=config["num_layers"],
            num_codes=config["num_embeddings"],
            embedding_dim=config["embedding_dim"],
            commitment_cost=config["commitment_cost"],
            udf_res=config["udf_resolution"],
            image_num_layer=config["image_num_layer"]
        ).to(device)
            
        print("--- 훈련 모드 시작 ---")
        vqae_model.apply(weights_init)
        vae_model_path = "./checkpoints/vae-training/pretrained_vae_model.pth"
        # vqae_model.load_pretrained_image_modules(vae_model_path)
        vae_weights = torch.load(vae_model_path, map_location=device)
        vqae_model.load_state_dict(vae_weights, strict=False)
        print("✅ VAE weights successfully transplanted to VQ-VAE.")
        vqae_model.initialize_codebook_with_kmeans(dataloader, device)

        checkpoint_dir = f"./checkpoints/{wandb.run.name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"체크포인트는 '{checkpoint_dir}' 디렉토리에 저장됩니다.")

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
            epoch_total_loss, epoch_coord_loss, epoch_pen_loss, epoch_img_loss, epoch_vq_loss, epoch_diff_loss, epoch_perplexity, epoch_kl_loss = 0, 0, 0, 0, 0, 0, 0, 0
            num_batches = 0
            
            loop = tqdm(dataloader, leave=True)
            for batch_idx, (stroke_seq, stroke_mask, udf_image, bbox) in enumerate(loop):
                stroke_seq, stroke_mask, udf_image, bbox = stroke_seq.to(device), stroke_mask.to(device), udf_image.to(device), bbox.to(device)
                optimizer.zero_grad()
                # print('stroke_seq: ', stroke_seq[0, :-1, 0])
                # print('stroke_mask: ', stroke_mask[0, :-1])

                recon_coords, recon_udf, vq_loss, indices_stroke, perplexity = vqae_model(
                    stroke_seq, stroke_mask, udf_image
                )

                # recon_coords, recon_udf, kl_loss = vqae_model(
                #     stroke_seq, stroke_mask, udf_image
                # )

                recon_pen_logits = recon_coords[:, :, 2]
                recon_coords = recon_coords[:, :, :2]

                # print('recon_coords:', recon_coords[0, :20, 0])
                # print('target:', stroke_seq[0, :20, 0])
                # print()
                # print('recon_pen_state:', recon_pen_logits[0, :20])
                # print('target:', stroke_seq[0, :20, 2])
                
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
                    config['loss_weight']['vq'] * vq_loss + \
                    config['loss_weight']['img'] * loss_img
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
                epoch_img_loss += loss_img.item()
                epoch_perplexity += perplexity.item()


                loop.set_description(f"Epoch [{epoch+1}/{config['2-epochs']}]")
                loop.set_postfix(loss=total_loss.item())
            
            num_batches += 1
            
            avg_total_loss = epoch_total_loss / num_batches
            avg_coord_loss = epoch_coord_loss / num_batches
            avg_pen_loss = epoch_pen_loss / num_batches
            avg_vq_loss = epoch_vq_loss / num_batches
            avg_img_loss = epoch_img_loss / num_batches
            avg_perplexity = epoch_perplexity / num_batches
            avg_kl_loss = epoch_kl_loss / num_batches


            # --- [WANDB] 3. wandb.log()로 에폭마다 지표 기록 ---
            wandb.log({
                "epoch": epoch + 1,
                "avg_total_loss": avg_total_loss,
                "avg_coord_loss": avg_coord_loss,
                "avg_pen_loss": avg_pen_loss,
                "avg_vq_loss": avg_vq_loss,
                "avg_img_loss": avg_img_loss,
                "avg_perplexity": avg_perplexity,
                "avg_kl_loss": avg_kl_loss,
                "learning_rate": scheduler.get_last_lr()[0]
            }, step=epoch)
        
            print('epoch', epoch)
            if (epoch + 1) % 100 == 0:
                print(f"\nEpoch {epoch+1}: 시각화 및 모델 체크포인트 저장 중...")
                vqae_model.eval()
                with torch.no_grad():
                    stroke_seq_fixed, stroke_mask_fixed, udf_image_fixed, bbox_fixed = [x.to(device) for x in fixed_batch]
                    predicted_points_tf, predicted_udf_fixed, _, _, _ = vqae_model(
                        stroke_seq_fixed, stroke_mask_fixed, udf_image_fixed
                    )

                    
                    # predicted_points_tf, predicted_udf_fixed, _ = vqae_model(
                    #     stroke_seq_fixed, stroke_mask_fixed, udf_image_fixed
                    # )


                    # --- [변경점] Autoregressive 방식의 생성 결과 얻기 ---
                    generated_strokes_ar, _ = vqae_model.generate(
                        stroke_seq_fixed, stroke_mask_fixed, udf_image_fixed, 
                        max_len=config["max_stroke_len"]
                    )


                    # W&B에 로깅할 이미지들을 담을 리스트
                    log_images = []
                    # --- [변경점] 샘플 수를 5개에서 2개로 변경 ---
                    num_samples_to_log = min(20, config["batch_size"])

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
                        valid_len_orig = int(stroke_mask_fixed[-1 - i].sum().item())
                        orig_stroke = stroke_seq_fixed[-1 - i, :valid_len_orig, :2].cpu().numpy()
                        ax_orig_stroke.plot(orig_stroke[:, 0], orig_stroke[:, 1], 'b-', lw=2)
                        ax_orig_stroke.set_title('Original Stroke')
                        ax_orig_stroke.set_xlim(0, 1); ax_orig_stroke.set_ylim(0, 1); ax_orig_stroke.invert_yaxis()
                        ax_orig_stroke.set_aspect('equal'); ax_orig_stroke.grid(True, alpha=0.3)
                        
                        # --- Plot 2: 복원된 스트로크 (Teacher Forcing) ---
                        # forward()의 출력을 사용
                        recon_stroke_tf = predicted_points_tf[-1 - i, :valid_len_orig, :2].cpu().numpy()
                        ax_tf_stroke.plot(recon_stroke_tf[:, 0], recon_stroke_tf[:, 1], 'g-', lw=2)
                        ax_tf_stroke.set_title('Recon (Teacher Forcing)')
                        ax_tf_stroke.set_xlim(0, 1); ax_tf_stroke.set_ylim(0, 1); ax_tf_stroke.invert_yaxis()
                        ax_tf_stroke.set_aspect('equal'); ax_tf_stroke.grid(True, alpha=0.3)

                        # --- [신규] Plot 3: 생성된 스트로크 (Autoregressive) ---
                        # generate()의 출력을 사용
                        pen_states_ar = (generated_strokes_ar[-1 - i, :, 2] > 0.5)
                        zero_indices = (pen_states_ar == 0).nonzero(as_tuple=True)[0]
                        valid_len_ar = zero_indices[0].item() + 1 if len(zero_indices) > 0 else config["max_stroke_len"]
                        recon_stroke_ar = generated_strokes_ar[-1 - i, :valid_len_ar, :2].cpu().numpy()
                        ax_ar_stroke.plot(recon_stroke_ar[:, 0], recon_stroke_ar[:, 1], 'r-', lw=2)
                        ax_ar_stroke.set_title('Generated (Autoregressive)')
                        ax_ar_stroke.set_xlim(0, 1); ax_ar_stroke.set_ylim(0, 1); ax_ar_stroke.invert_yaxis()
                        ax_ar_stroke.set_aspect('equal'); ax_ar_stroke.grid(True, alpha=0.3)

                        # --- Plot 4: 원본 UDF ---
                        original_udf_img = udf_image_fixed[-1 - i].squeeze().cpu().numpy()
                        im_orig = ax_orig_udf.imshow(original_udf_img, cmap='hot', vmin=0, vmax=1)
                        ax_orig_udf.set_title('Original UDF')
                        ax_orig_udf.axis('off')
                        fig.colorbar(im_orig, ax=ax_orig_udf, shrink=0.8)

                        # --- Plot 5: 복원된 UDF ---
                        reconstructed_udf_img = predicted_udf_fixed[-1 - i].squeeze().cpu().numpy()
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
                    torch.save(vqae_model.state_dict(), checkpoint_path)
                    artifact = wandb.Artifact(f'model-{wandb.run.name}', type='model')
                    artifact.add_file(checkpoint_path)
                    wandb.log_artifact(artifact, aliases=[f"epoch_{epoch+1}"])
                    print(f"Epoch {epoch+1}: 모델 체크포인트 저장 완료 -> {checkpoint_path}")
                    

                vqae_model.train() # 모델을 다시 훈련 모드로
                
        print("--- 훈련 완료 ---")
        
        model_path = './checkpoints/vq-vae-training/stroke_vqae.pth'
        torch.save(vqae_model.state_dict(), model_path)
        artifact = wandb.Artifact('vqae-model', type='model')
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)

        vqae_model.eval()

        # Drawing 단위로 데이터를 처리하기 위해 새로운 데이터셋 인스턴스 생성
        all_drawings_raw = []
        for category in CATEGORIES:
            with open(f"./data/quickdraw/{category}.ndjson", 'r') as f:
                data = ndjson.load(f)
            recognized = [item['drawing'] for item in data if item.get('recognized', True)]
            all_drawings_raw += recognized[:config["data_size_per_category"]]
        

        drawing_dataset = QuickDrawDrawingDataset2(all_drawings_raw, config)
        
        
        stroke_token_sequences = []
        with torch.no_grad():
            for drawing_info in tqdm(drawing_dataset, desc="Tokenizing Strokes"):
                tokens_for_drawing = []
                # 한 drawing에 속한 각 stroke와 udf를 순서대로 토큰화
                for stroke_np, udf_image_np in zip(drawing_info['strokes'], drawing_info['udfs']):
                    # 데이터셋의 __getitem__ 로직을 활용하여 stroke, mask, udf 텐서 생성
                    stroke_len = min(len(stroke_np), config["max_stroke_len"])
                    
                    stroke_with_pen = np.ones((stroke_len, 3), dtype=np.float32)
                    stroke_with_pen[:, :2] = stroke_np[:stroke_len]
                    if stroke_len > 0: stroke_with_pen[-1, 2] = 0
                    
                    padded_stroke = np.zeros((config["max_stroke_len"], 3), dtype=np.float32)
                    padded_stroke[:stroke_len] = stroke_with_pen
                    
                    stroke_mask = np.zeros(config["max_stroke_len"], dtype=np.float32)
                    stroke_mask[:stroke_len] = 1.0

                    
                    # 배치 차원(1)을 추가하고 디바이스로 이동
                    s_tensor = torch.from_numpy(padded_stroke).unsqueeze(0).to(device)
                    m_tensor = torch.from_numpy(stroke_mask).unsqueeze(0).to(device)
                    u_tensor = torch.from_numpy(udf_image_np).float().unsqueeze(0).unsqueeze(0).to(device)
                    # print('s_tensor: ', s_tensor[0][:-1, 0])
                    # print('m_tensor: ', m_tensor[0][:-1])


                    # 모델의 .encode() 함수로 단일 토큰 생성
                    token_index = vqae_model.encode(s_tensor, m_tensor, u_tensor)
                    tokens_for_drawing.append(token_index.item())
                
                if tokens_for_drawing:
                    stroke_token_sequences.append(np.array(tokens_for_drawing, dtype=np.int32))

        # 3. 토큰 시퀀스를 .npz 파일로 분할 저장
        print("\n--- Splitting and saving to .npz format ---")
        val_size = len(stroke_token_sequences) // 10
        train_sequences = stroke_token_sequences[:-val_size]
        val_sequences = stroke_token_sequences[-val_size:]

        train_data_dict = {f'drawing_{i}': seq for i, seq in enumerate(train_sequences)}
        val_data_dict = {f'drawing_{i}': seq for i, seq in enumerate(val_sequences)}

        save_dir = './data/layout_stroke'
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(os.path.join(save_dir, 'train.npz'), **train_data_dict)
        np.savez_compressed(os.path.join(save_dir, 'val.npz'), **val_data_dict)

        print(f"\n✅ Stroke token data processing complete.")
        print(f"Train data saved to '{os.path.join(save_dir, 'train.npz')}' ({len(train_sequences)} drawings)")
        print(f"Validation data saved to '{os.path.join(save_dir, 'val.npz')}' ({len(val_sequences)} drawings)")
        print(val_data_dict)

