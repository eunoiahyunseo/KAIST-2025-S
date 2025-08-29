import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
import ndjson
from tqdm import tqdm

# 기존 스크립트에서 필요한 클래스들을 import 합니다.
# 'your_main_script_name'은 실제 메인 스크립트 파일명으로 변경해주세요.
from train_vqae import QuickDrawDataset, config as main_config
from encoder_decoder import ImageSpatialVAE

def visualize_tsne_local(latent_vectors):
    """로컬 특징 벡터들을 t-SNE로 시각화하는 함수"""
    print("Running t-SNE on local features... (This will be slower)")
    # 데이터가 많으므로, 일부만 샘플링하여 t-SNE를 실행하는 것이 효율적일 수 있습니다.
    # 예: 5000개만 무작위로 선택
    if len(latent_vectors) > 5000:
        indices = np.random.choice(len(latent_vectors), 5000, replace=False)
        latent_vectors = latent_vectors[indices]

    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42, verbose=1)
    tsne_results = tsne.fit_transform(latent_vectors)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.3, s=5) # s는 점 크기
        
    plt.title("VAE Local Feature Latent Space (t-SNE)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.savefig("tsne_local_features_visualization.png", dpi=150)
    print("t-SNE visualization saved to tsne_local_features_visualization.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='학습된 ImageSpatialVAE 모델(.pth) 경로')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 1. 모델 불러오기 ---
    model = ImageSpatialVAE(
        d_img=main_config["d_img"], 
        udf_resolution=main_config["udf_resolution"]
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # --- 2. 데이터 및 레이블 준비 ---
    all_latents = []
    all_labels = []
    categories = ["moon", "apple"]
    all_local_latents = []

    
    for label_idx, category in enumerate(categories):
        print(f"Loading data for category: {category}")
        # 각 카테고리별로 데이터셋을 따로 만듦
        data_path = f"./data/quickdraw/{category}.ndjson"
        # ... (기존 스크립트의 dic_item 생성 로직) ...
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
                        if cnt >= main_config["data_size_per_category"]: break
                dic_item += list(dic.items())
            except FileNotFoundError:
                print(f"Warning: Data file not found at {data_path}")
        
        dataset = QuickDrawDataset(
            dic_item, max_stroke_len=main_config["max_stroke_len"], udf_res=main_config["udf_resolution"],
            rdp_epsilon=main_config["rdp_epsilon"], gamma=main_config["gamma"])
        dataloader = DataLoader(dataset, batch_size=main_config["batch_size"], shuffle=False)

        with torch.no_grad():
            for _, _, udf_image in tqdm(dataloader, desc=f"Extracting latents for {category}"):
                udf_image = udf_image.to(device)
                
                # VAE 모델을 통해 mu_seq 추출
                _, mu_seq, _ = model(udf_image)
                
                # 공간적 특징을 대표하는 전역 벡터로 평균
                all_local_latents.append(mu_seq.reshape(-1, mu_seq.size(-1)).cpu())
                

    final_local_latents = torch.cat(all_local_latents, dim=0).numpy()
    visualize_tsne_local(final_local_latents)
