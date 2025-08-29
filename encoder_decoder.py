
# encoder_decoder.py (이 코드로 파일 전체를 교체하세요)

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Tuple
from sklearn.cluster import KMeans
from vqtorch.nn import VectorQuant

class ImageSpatialVAE(nn.Module):
    def __init__(self, d_img=128, udf_resolution=64):
        super().__init__()
        self.encoder = ImageEncoder(d_img=d_img)
        
        self.fc_mu = nn.Linear(d_img, d_img)
        self.fc_log_var = nn.Linear(d_img, d_img)
        
        self.decoder = UDFDecoder(input_dim=d_img, output_res=udf_resolution)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        z_img_seq = self.encoder(x)
        mu_seq = self.fc_mu(z_img_seq)
        log_var_seq = self.fc_log_var(z_img_seq)
        z_sampled_seq = self.reparameterize(mu_seq, log_var_seq)
        recon_x = self.decoder(z_sampled_seq)    
        return recon_x, mu_seq, log_var_seq
    
class PositionalEncoding(nn.Module):
    # (이전과 동일, 변경 없음)
    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class VectorEncoder(nn.Module):
    # (이전과 동일, Mean Pooling 없음)
    def __init__(self, input_dim=3, d_model=256, d_seq=32, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True, dim_feedforward=d_model*4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, d_seq)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        batch_size = src.shape[0]
        embedded = self.embedding(src) * math.sqrt(self.d_model)
        sos_token = torch.zeros(batch_size, 1, self.d_model, device=src.device)
        full_seq = torch.cat([sos_token, embedded], dim=1)

        pos_encoded = self.pos_encoder(full_seq.permute(1, 0, 2)).permute(1, 0, 2)
        
        full_mask = torch.cat([torch.ones(batch_size, 1, device=src.device), src_mask], dim=1)
        encoded_seq = self.transformer_encoder(pos_encoded, src_key_padding_mask=(full_mask == 0))

        projected_seq = self.output_proj(encoded_seq)
        return projected_seq, full_mask
    

class ImageEncoder(nn.Module):
    """
    UDF 이미지를 공간 정보가 담긴 피처맵 시퀀스로 인코딩합니다.
    """
    def __init__(self, in_channels=1, d_img=128, start_res=64):
        super().__init__()
        
        layers = []
        in_c = in_channels
        
        # start_res가 8x8이 될 때까지 다운샘플링
        num_downsamples = int(math.log2(start_res / 8))
        
        # 채널 수를 점진적으로 늘림
        out_c = 64
        for i in range(num_downsamples):
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(True))
            in_c = out_c
            out_c *= 2 # 다음 레이어 채널 2배
        
        # 최종 채널 수를 d_img로 맞춤
        layers.append(nn.Conv2d(in_c, d_img, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(d_img))
        layers.append(nn.ReLU(True))
        
        self.encoder = nn.Sequential(*layers)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: (B, 1, 64, 64)
        feature_map = self.encoder(src) # -> (B, d_img, 8, 8)
        
        B, C, H, W = feature_map.shape
        # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        # 최종적으로 (B, 64, d_img) 형태의 시퀀스로 반환
        return feature_map.reshape(B, C, H * W).permute(0, 2, 1)
    
class CrossAttentionFusion(nn.Module):
    def __init__(self, d_seq: int, d_img: int, embedding_dim: int, nhead: int):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_seq, 
            kdim=d_img, 
            vdim=d_img, 
            num_heads=nhead, 
            batch_first=True
        )
        self.gate_linear = nn.Linear(d_seq, d_seq)

        self.norm1 = nn.LayerNorm(d_seq)
        self.norm2 = nn.LayerNorm(d_seq)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_seq, d_seq * 4),
            nn.ReLU(),
            nn.Linear(d_seq * 4, d_seq)
        )
        
        self.final_proj = nn.Linear(d_seq, embedding_dim)
        self.final_norm = nn.LayerNorm(embedding_dim)

    def forward(self, z_seq: torch.Tensor, z_img: torch.Tensor) -> torch.Tensor:
        z_img_seq = z_img
        attn_output, _ = self.attention(query=z_seq, key=z_img_seq, value=z_img_seq)

        gate = torch.sigmoid(self.gate_linear(z_seq))
        fused_info = (1 - gate) * z_seq + gate * attn_output
        x = self.norm1(fused_info)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        output = self.final_norm(self.final_proj(x))
        return output
    

class SequentialVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float, temperature: float = 1.0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.commitment_cost = commitment_cost
        self.temperature = temperature

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, Np, D = inputs.shape

        valid_indices = torch.where(mask.reshape(-1) == 1)[0]
        valid_inputs = inputs.reshape(-1, D)[valid_indices]

        distances = torch.sum(valid_inputs**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(valid_inputs, self.embedding.weight.t())
        encoding_indices_valid = torch.argmin(distances, dim=1)

        quantized_valid = F.embedding(encoding_indices_valid, self.embedding.weight)
        
        loss = F.mse_loss(quantized_valid, valid_inputs.detach()) + self.commitment_cost * F.mse_loss(valid_inputs, quantized_valid.detach())
        
        quantized_valid_sg = valid_inputs + (quantized_valid - valid_inputs).detach()
        
        quantized_seq = torch.zeros_like(inputs)
        quantized_seq.reshape(-1, D)[valid_indices] = quantized_valid_sg
        
        indices_seq = torch.full((B * Np,), -1, device=inputs.device, dtype=torch.long)
        indices_seq[valid_indices] = encoding_indices_valid
        indices_seq = indices_seq.reshape(B, Np)


        return loss, quantized_seq, indices_seq

class VectorDecoderAR(nn.Module):
    def __init__(self, output_dim=3, d_model=256, nhead=8, num_layers=6, embedding_dim=256):
        super().__init__()
        self.d_model = d_model
        self.input_embedding = nn.Linear(output_dim, d_model)
        self.input_dropout = nn.Dropout(0.1) # <-- Dropout 레이어 추가

        self.pos_encoder = PositionalEncoding(d_model)
        self.memory_proj = nn.Linear(embedding_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True, dim_feedforward=d_model*4)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.coord_head = nn.Sequential(nn.Linear(d_model, 2), nn.Sigmoid())
        self.pen_state_head = nn.Linear(d_model, 1)

    def forward(self, tgt_seq: torch.Tensor, memory: torch.Tensor, memory_padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tgt_embedded = self.input_embedding(tgt_seq) * math.sqrt(self.d_model)
        tgt_embedded = self.input_dropout(tgt_embedded)

        tgt_pos = self.pos_encoder(tgt_embedded.permute(1, 0, 2)).permute(1, 0, 2)
        memory_projected = self.memory_proj(memory)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq.size(1)).to(tgt_seq.device)
        decoded_features = self.transformer_decoder(tgt_pos, memory_projected, tgt_mask=causal_mask, memory_key_padding_mask=memory_padding_mask)
        coords = self.coord_head(decoded_features)
        pen_state_logit = self.pen_state_head(decoded_features).squeeze(-1)
        return coords, pen_state_logit


class UDFDecoder(nn.Module):
    """
    피처맵 시퀀스를 입력받아 UDF 이미지로 복원합니다.
    """
    def __init__(self, input_dim: int, output_channels: int = 1, output_res: int = 64):
        super().__init__()
        
        # 최종 피처맵 크기(8x8)를 기준으로 업샘플링 횟수 계산
        start_res = 8 
        num_upsamples = int(math.log2(output_res / start_res))
        
        layers = []
        in_channels = input_dim # ImageEncoder의 출력 채널(d_img)
        
        # 동적으로 업샘플링 레이어 생성
        for i in range(num_upsamples):
            out_channels = max(in_channels // 2, 32) # 최소 채널 32 보장
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(True))
            in_channels = out_channels
            
        # 최종 출력 레이어 (3x3 Conv로 더 부드러운 결과 생성)
        layers.append(nn.Conv2d(in_channels, output_channels, kernel_size=3, padding=1))
        layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*layers)

    def forward(self, z_img_seq: torch.Tensor) -> torch.Tensor:
        # z_img_seq shape: (B, H*W, D), 예: (B, 64, 128)
        
        # 1. 시퀀스 -> 피처맵 형태로 복원
        B, HW, D = z_img_seq.shape
        H = W = int(math.sqrt(HW))
        # (B, H*W, D) -> (B, D, H*W) -> (B, D, H, W)
        feature_map = z_img_seq.permute(0, 2, 1).reshape(B, D, H, W)
        
        # 2. 디코더(업샘플러)를 통해 이미지 복원
        recon_udf = self.decoder(feature_map) # -> (B, 1, 64, 64)
        return recon_udf
    
class StrokeTokenizerVQVAE(nn.Module):
    """(수정) UDF 융합 기능이 포함된 최종 토크나이저"""
    def __init__(self, max_stroke_len=32, d_model=256, d_seq=32, d_img=128, nhead=8, num_layers=6, d_num_layers=2,
                 num_embeddings=512, embedding_dim=256, commitment_cost=0.25, udf_resolution=64):
        super().__init__()
        self.max_len = max_stroke_len
        
        self.vector_encoder = VectorEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers, d_seq=d_seq)
        self.fusion_encoder = CrossAttentionFusion(
            d_seq=d_seq,
            d_img=d_img, 
            embedding_dim=embedding_dim, 
            nhead=nhead
        )
        self.quantizer = SequentialVectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = VectorDecoderAR(d_model=d_model, nhead=nhead, num_layers=d_num_layers, embedding_dim=embedding_dim)

        self.image_vae = ImageSpatialVAE(d_img=d_img, udf_resolution=udf_resolution)
        # self.image_encoder = ImageEncoder(d_img=d_img)
        # self.udf_decoder = UDFDecoder(input_dim=d_img, output_res=udf_resolution)


    def load_pretrained_image_modules(self, vae_checkpoint_path):
        self.image_vae.load_state_dict(torch.load(vae_checkpoint_path))        
        for param in self.image_vae.parameters():
            param.requires_grad = False

    def forward(self, stroke_seq: torch.Tensor, stroke_mask: torch.Tensor, udf_image: torch.Tensor):
        z_seq, full_mask = self.vector_encoder(stroke_seq, stroke_mask)
        
        with torch.no_grad():
            recon_udf_from_vae, mu_seq, _ = self.image_vae(udf_image)
            print('check udf loss: ', F.mse_loss(recon_udf_from_vae, udf_image))
        
        fused_seq = self.fusion_encoder(z_seq, mu_seq)
        vq_loss, quantized_fused, indices_stroke = self.quantizer(fused_seq, full_mask)
        print('check indices: ', indices_stroke[0, :30])
        
        sos_token = torch.zeros(stroke_seq.shape[0], 1, 3, device=stroke_seq.device)
        decoder_input = torch.cat([sos_token, stroke_seq[:, :-1, :]], dim=1)
        recon_coords, recon_pen_logits = self.decoder(decoder_input, memory=quantized_fused, memory_padding_mask=(full_mask==0))
        
        return recon_coords, recon_pen_logits, recon_udf_from_vae, vq_loss, indices_stroke, fused_seq, full_mask


    def encode_to_latents(self, stroke_seq: torch.Tensor, stroke_mask: torch.Tensor, udf_image: torch.Tensor) -> torch.Tensor:
        """
        (신규 헬퍼 메서드)
        입력 데이터를 받아 양자화 직전의 융합된 잠재 벡터(fused_seq)를 반환합니다.
        """
        z_seq, full_mask = self.vector_encoder(stroke_seq, stroke_mask)
        _, mu_seq, _ = self.image_vae(udf_image)
        fused_seq = self.fusion_encoder(z_seq, mu_seq)
        return fused_seq, full_mask
    
    def initialize_codebook_with_kmeans(self, data_loader, device):
        """
        (신규 메인 메서드)
        데이터로더의 일부 데이터를 사용해 K-Means로 코드북을 초기화합니다.
        """
        print("Initializing codebook with K-Means...")
        
        latent_vectors_list = []
        masks_list = []
        
        self.eval() # 모델을 평가 모드로 설정
        with torch.no_grad():
            # 보통 5~10 배치 정도면 충분합니다.
            for i, (stroke_seq, stroke_mask, udf_image) in enumerate(data_loader):
                if i >= 10: break
                
                stroke_seq, stroke_mask, udf_image = stroke_seq.to(device), stroke_mask.to(device), udf_image.to(device)
                
                # `encode_to_latents`를 호출하여 잠재 벡터 추출
                latent_vectors, full_mask = self.encode_to_latents(stroke_seq, stroke_mask, udf_image)
                
                latent_vectors_list.append(latent_vectors.cpu())
                masks_list.append(stroke_mask.cpu())

        all_latents = torch.cat(latent_vectors_list, dim=0)
        all_masks = torch.cat(masks_list, dim=0)
        
        # 마스크를 사용해 유효한 잠재 벡터만 추출
        B, Np, D = all_latents.shape
        valid_indices = torch.where(all_masks.view(-1) == 1)[0]
        valid_latents = all_latents.view(-1, D)[valid_indices]
        valid_latents_np = valid_latents.numpy()
        
        print(f"Running K-Means on {valid_latents_np.shape[0]} valid latent vectors...")
        
        num_codes = self.quantizer.embedding.num_embeddings
        kmeans = KMeans(n_clusters=num_codes, n_init='auto', random_state=0)
        kmeans.fit(valid_latents_np)
        
        centroids = torch.from_numpy(kmeans.cluster_centers_).to(device, dtype=torch.float32)
        
        # 양자화기(quantizer)의 임베딩 가중치를 K-Means 결과로 덮어쓰기
        with torch.no_grad():
            self.quantizer.embedding.weight.copy_(centroids)
            
        self.train() # 모델을 다시 훈련 모드로 설정
        print("K-Means initialization complete.")
    
    
    @torch.no_grad()
    def reconstruct_from_latents(self, z_stroke_indices: torch.Tensor):
        """
        양자화된 스트로크 인덱스 시퀀스로부터 스트로크를 자기회귀적으로 복원합니다.
        """
        # 1. 인덱스 -> 양자화된 벡터 시퀀스로 변환
        quantized_fused = self.quantizer.embedding(torch.clamp(z_stroke_indices, min=0))
        
        B, Np, _ = quantized_fused.shape
        device = z_stroke_indices.device
        
        # 2. 자기회귀 생성을 위한 <SOS> 토큰 초기화
        generated_seq = torch.zeros(B, 1, 3, device=device)
        
        # 3. 자기회귀 루프
        # Np는 SOS 토큰이 포함된 길이이므로, 실제 스트로크 길이만큼만 생성 (Np - 1)
        for _ in range(Np - 1):
            coords, logits = self.decoder(
                generated_seq, 
                memory=quantized_fused, 
                memory_padding_mask=(z_stroke_indices == -1)
            )
            next_coord = coords[:, -1, :]
            next_logit = logits[:, -1]
            next_pen = (torch.sigmoid(next_logit) > 0.5).float().unsqueeze(1)
            
            next_point = torch.cat([next_coord, next_pen], dim=1).unsqueeze(1)
            generated_seq = torch.cat([generated_seq, next_point], dim=1)
            
        return generated_seq[:, 1:, :] # <SOS> 토큰 제외


