
# encoder_decoder.py (이 코드로 파일 전체를 교체하세요)

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Tuple
from sklearn.cluster import KMeans


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
        return projected_seq

class ImageEncoder(nn.Module):
    # (이전과 동일, UDF 이미지 인코딩 담당)
    def __init__(self, in_channels=1, d_img=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 2, 1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(512, d_img)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(src).squeeze(-1).squeeze(-1)
        return self.fc(encoded)
    
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
        z_img_seq = z_img.unsqueeze(1)
        attn_output, _ = self.attention(query=z_seq, key=z_img_seq, value=z_img_seq)
        x = self.norm1(z_seq + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        output = self.final_norm(self.final_proj(x))
        return output
    
class FusionEncoder(nn.Module):
    """(신규) Vector 시퀀스와 Image 벡터를 결합하는 융합 모듈"""
    def __init__(self, d_model: int, d_img: int, embedding_dim: int):
        super().__init__()
        self.fusion_layer = nn.Linear(d_model + d_img, embedding_dim)
        self.fusion_layer2 = nn.Linear(d_model, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, z_seq: torch.Tensor, z_img: torch.Tensor) -> torch.Tensor:
        Np = z_seq.size(1)
        z_img_expanded = z_img.unsqueeze(1).repeat(1, Np, 1)
        # print('sex', z_img_expanded[0, :20, :10])
        # print('sex2', z_seq[0, :20, :10])
        # fused_seq = torch.cat([z_seq, z_img_expanded], dim=-1)
        # output_seq = self.fusion_layer(fused_seq)
        fused_seq = torch.cat([z_seq], dim=-1)
        # output_seq = self.fusion_layer2(fused_seq)
        return self.norm(fused_seq)

class SequentialVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.commitment_cost = commitment_cost

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, Np, D = inputs.shape


        valid_indices = torch.where(mask.view(-1) == 1)[0]
        valid_inputs = inputs.view(-1, D)[valid_indices]

        distances = torch.sum(valid_inputs**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(valid_inputs, self.embedding.weight.t())
        encoding_indices_valid = torch.argmin(distances, dim=1)

        quantized_valid = F.embedding(encoding_indices_valid, self.embedding.weight)
        
        loss = F.mse_loss(quantized_valid, valid_inputs.detach()) + self.commitment_cost * F.mse_loss(valid_inputs, quantized_valid.detach())
        
        quantized_valid_sg = valid_inputs + (quantized_valid - valid_inputs).detach()
        
        quantized_seq = torch.zeros_like(inputs)
        quantized_seq.view(-1, D)[valid_indices] = quantized_valid_sg
        
        indices_seq = torch.full((B * Np,), -1, device=inputs.device, dtype=torch.long)
        indices_seq[valid_indices] = encoding_indices_valid
        indices_seq = indices_seq.view(B, Np)
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
        

        tgt_pos = self.pos_encoder(tgt_embedded.permute(1, 0, 2)).permute(1, 0, 2)
        memory_projected = self.memory_proj(memory)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq.size(1)).to(tgt_seq.device)
        decoded_features = self.transformer_decoder(tgt_pos, memory_projected, tgt_mask=causal_mask, memory_key_padding_mask=memory_padding_mask)
        coords = self.coord_head(decoded_features)
        pen_state_logit = self.pen_state_head(decoded_features).squeeze(-1)
        return coords, pen_state_logit

class UDFDecoder(nn.Module):
    def __init__(self, embedding_dim: int, output_channels: int = 1, output_res: int = 64):
        super().__init__()
        self.start_dim = 4 # 4x4에서 업샘플링 시작
        
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 512 * self.start_dim * self.start_dim),
            nn.ReLU(True)
        )
        
        self.decoder = nn.Sequential(
            # Input: (B, 512, 4, 4)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # -> (B, 256, 8, 8)
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> (B, 128, 16, 16)
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (B, 64, 32, 32)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1), # -> (B, 1, 64, 64)
            nn.Sigmoid() # UDF 값을 [0, 1] 범위로 출력
        )

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        z_global = z_seq.mean(dim=1)
        x = self.fc(z_global)
        x = x.view(x.size(0), 512, self.start_dim, self.start_dim)
        recon_udf = self.decoder(x)
        return recon_udf
    
class StrokeTokenizerVQVAE(nn.Module):
    """(수정) UDF 융합 기능이 포함된 최종 토크나이저"""
    def __init__(self, max_stroke_len=32, d_model=256, d_seq=32, d_img=128, nhead=8, num_layers=6,
                 num_embeddings=512, embedding_dim=256, commitment_cost=0.25):
        super().__init__()
        self.max_len = max_stroke_len
        
        self.vector_encoder = VectorEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers, d_seq=d_seq)
        self.image_encoder = ImageEncoder(d_img=d_img)
        # self.fusion_encoder = FusionEncoder(d_model=d_model, d_img=d_img, embedding_dim=embedding_dim)
        self.fusion_encoder = CrossAttentionFusion(
            d_seq=d_seq,
            d_img=d_img, 
            embedding_dim=embedding_dim, 
            nhead=nhead
        )
        self.quantizer = SequentialVectorQuantizer(num_embeddings, embedding_dim, commitment_cost) # (B, Np + 1, embedding_dim)

        self.decoder = VectorDecoderAR(d_model=d_model, nhead=nhead, num_layers=num_layers, embedding_dim=embedding_dim)
        self.udf_decoder = UDFDecoder(embedding_dim=embedding_dim, output_res=64)


    def forward(self, stroke_seq: torch.Tensor, stroke_mask: torch.Tensor, udf_image: torch.Tensor):
        fused_seq, full_mask = self.encode_to_latents(stroke_seq, stroke_mask, udf_image)
        vq_loss, quantized_seq, indices_seq = self.quantizer(fused_seq, full_mask)
        
        sos_token = torch.zeros(stroke_seq.shape[0], 1, 3, device=stroke_seq.device)
        decoder_input = torch.cat([sos_token, stroke_seq[:, :-1, :]], dim=1)
        print(decoder_input.shape)
        recon_coords, recon_pen_logits = self.decoder(decoder_input, memory=quantized_seq, memory_padding_mask=(full_mask==0))


        recon_udf = self.udf_decoder(quantized_seq)

        
        return recon_coords, recon_pen_logits, recon_udf, vq_loss, indices_seq


    def encode_to_latents(self, stroke_seq: torch.Tensor, stroke_mask: torch.Tensor, udf_image: torch.Tensor) -> torch.Tensor:
        """
        (신규 헬퍼 메서드)
        입력 데이터를 받아 양자화 직전의 융합된 잠재 벡터(fused_seq)를 반환합니다.
        """
        z_seq = self.vector_encoder(stroke_seq, stroke_mask)
        z_img = self.image_encoder(udf_image)
        batch_size = stroke_seq.shape[0]
        full_mask = torch.cat([torch.ones(batch_size, 1, device=stroke_seq.device), stroke_mask], dim=1)

        fused_seq = self.fusion_encoder(z_seq, z_img)
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
    def generate_from_indices(self, z_indices: torch.Tensor):
        B, Np = z_indices.shape
        device = z_indices.device
        valid_z_indices = torch.clamp(z_indices, min=0) 

        quantized_seq = self.quantizer.embedding(valid_z_indices)
        # print('z_indices', z_indices[:2, :30])
        print('quantized_seq: ', quantized_seq.shape, quantized_seq[0, :30, :5])
        print('quantized_seq: ', quantized_seq.shape, quantized_seq[3, :30, :])
        
        # 2. 자기회귀 생성을 위한 <SOS> 토큰 초기화
        generated_seq = torch.zeros(B, 1, 3, device=device)

        for _ in range(Np):
            coords, logits = self.decoder(generated_seq, memory=quantized_seq, memory_padding_mask=(z_indices == -1))
            next_coord = coords[:, -1, :]
            print('next-coord', next_coord[0])
            next_logit = logits[:, -1]
            next_pen = (torch.sigmoid(next_logit) > 0.2).float().unsqueeze(1)
            
            # print('next_pen: ', next_pen.shape, next_pen)
            
            next_point = torch.cat([next_coord, next_pen], dim=1).unsqueeze(1)
            generated_seq = torch.cat([generated_seq, next_point], dim=1)
            
        return generated_seq[:, 1:, :] # <SOS> 토큰 제외


