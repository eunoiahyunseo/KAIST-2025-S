import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import wandb
import matplotlib.pyplot as plt
import ndjson
import os

from encoder_decoder import ImageSpatialVAE
import torch.nn.functional as F
from train_vqae import QuickDrawDataset, config as main_config


config = {
    "stage": 1,
    "learning_rate": 1e-3,
    "epochs": 1000,
    "batch_size": main_config['batch_size'],
    "d_img": main_config['d_img'],
    "udf_resolution": main_config['udf_resolution'],
    "kl_weight": 0.001,
    "data_size_per_category": main_config['data_size_per_category'],
    "max_stroke_len": main_config['max_stroke_len'],
    "rdp_epsilon": main_config['rdp_epsilon'],
    "gamma": main_config['gamma'],
}

if __name__ == "__main__":
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

    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    fixed_dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    fixed_batch = next(iter(fixed_dataloader))

    model = ImageSpatialVAE(
        d_img=config["d_img"], udf_resolution=config["udf_resolution"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    recon_loss_fn = nn.MSELoss(reduction='sum') # VAE는 보통 sum을 사용

    wandb.init(
        project="stroke-vq-vae-2-stage", 
        name="stage1-image-VAE",
        config=config
    )

    print("--- Stage 1: Image Variational Autoencoder Training ---")
    best_loss = float('inf') # 최고 성능 모델을 저장하기 위한 변수

    # 체크포인트를 저장할 폴더 생성
    CHECKPOINT_DIR = "./checkpoints_vae"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(config["epochs"]):
        model.train()
        total_recon_loss = 0
        total_kl_loss = 0
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for _, _, udf_image in loop:
            udf_image = udf_image.to(device)
            optimizer.zero_grad()
            
            recon_udf, mu, log_var = model(udf_image)
            
            recon_loss = F.mse_loss(recon_udf, udf_image, reduction='mean') * udf_image.numel() / udf_image.shape[0]
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=[1, 2]).mean()
            
            loss = recon_loss + config["kl_weight"] * kl_loss
            
            loss.backward()
            optimizer.step()
            
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            loop.set_postfix(recon_loss=recon_loss.item(), kl_loss=kl_loss.item())

        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)
        avg_total_loss = avg_recon_loss + config["kl_weight"] * avg_kl_loss
        
        wandb.log({
            "epoch": epoch, 
            "reconstruction_loss": avg_recon_loss,
            "kl_divergence_loss": avg_kl_loss,
            "total_loss": avg_total_loss
        })

        # --- [신규] 100 에폭마다 체크포인트 저장 ---
        if (epoch + 1) % 100 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"vae_checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"\nCheckpoint saved at epoch {epoch+1} to {checkpoint_path}")
            
            # Wandb Artifact에도 주기적으로 저장 (선택적)
            artifact = wandb.Artifact(f'image-spatial-vae-epoch-{epoch+1}', type='model')
            artifact.add_file(checkpoint_path)
            wandb.run.log_artifact(artifact)

        # --- [개선] 최고 성능 모델(best) 저장 ---
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            best_model_path = os.path.join(CHECKPOINT_DIR, "vae_best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at epoch {epoch+1} with loss {best_loss:.4f}")

        # ... (10 에폭마다 wandb에 이미지 로깅하는 부분은 기존과 동일) ...


    # --- [수정] 학습 완료 후 최종 모델(latest) 및 최고 성능 모델(best) 저장 ---
    final_model_path = os.path.join(CHECKPOINT_DIR, "vae_latest_model.pth")
    torch.save(model.state_dict(), final_model_path)

    # Wandb Artifact에 latest와 best 버전을 저장하여 관리
    artifact = wandb.Artifact('image-spatial-vae', type='model')
    artifact.add_file(final_model_path, name="latest_model.pth")
    artifact.add_file(best_model_path, name="best_model.pth") # 최고 성능 모델 추가
    wandb.run.log_artifact(artifact)

    print(f"Final model saved to {final_model_path}")
    print(f"Best model saved to {best_model_path}")
    print("Training complete and models saved as W&B Artifacts.")

    wandb.finish()