import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Tuple


class PositionalEncoding(nn.Module):
    """Transformer에서 사용하는 Sinusoidal Positional Encoding을 구현합니다."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch_size, d_model)
        Returns:
            (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class VectorEncoder(nn.Module):
    def __init__(self, input_dim=3, d_model=64, nhead=8, num_layers=6, d_seq=32):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, d_seq)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): 스트로크 시퀀스. Shape: (B, Np, 3) (x, y, pen_state)
            src_mask (torch.Tensor): 패딩 마스크. Shape: (B, Np)

        Returns:
            torch.Tensor: 인코딩된 시퀀스 잠재 벡터. Shape: (B, d_seq)
        """
        embedded = self.embedding(src) * math.sqrt(self.d_model) # (B, Np, d_model)
        pos_encoded = self.pos_encoder(embedded.permute(1, 0, 2)).permute(1, 0, 2) # (B, Np, d_model)
        
        padding_mask = (src_mask == 0)

        encoded = self.transformer_encoder(pos_encoded, src_key_padding_mask=padding_mask) # (B, Np, d_model)

        encoded = encoded * src_mask.unsqueeze(-1)
        masked_sum = encoded.sum(dim=1)
        valid_tokens = src_mask.sum(dim=1).unsqueeze(-1)
        mean_pooled = masked_sum / valid_tokens.clamp(min=1e-9)

        z_seq = self.output_layer(mean_pooled) # (B, d_seq)
        return z_seq

class ImageEncoder(nn.Module):
    def __init__(self, in_channels=1, d_img=64):
        super().__init__()
        self.encoder = nn.Sequential(
            self._make_block(in_channels, 64),  # 64x64 -> 32x32
            self._make_block(64, 128),          # 32x32 -> 16x16
            self._make_block(128, 256),         # 16x16 -> 8x8
            self._make_block(256, 512),         # 8x8 -> 4x4
            self._make_block(512, 512),         # 4x4 -> 2x2
            self._make_block(512, 512, pool=False) # 2x2 -> 2x2
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, d_img)

    def _make_block(self, in_c, out_c, pool=True):
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): UDF 이미지. Shape: (B, 1, 64, 64)

        Returns:
            torch.Tensor: 인코딩된 이미지 잠재 벡터. Shape: (B, d_img)
        """
        encoded = self.encoder(src) # (B, 512, 2, 2)
        pooled = self.avg_pool(encoded) # (B, 512, 1, 1)
        flattened = torch.flatten(pooled, 1) # (B, 512)
        z_img = self.fc(flattened) # (B, d_img)
        return z_img

class StrokeUDFEncoder(nn.Module):
    """Dual-Modal 인코더: Vector와 Image 인코더를 결합하여 최종 잠재 벡터 생성"""
    def __init__(self, vector_encoder, image_encoder, d_seq=32, d_img=64, d_f=128):
        super().__init__()
        self.vector_encoder = vector_encoder
        self.image_encoder = image_encoder
        self.fusion_layer = nn.Linear(d_seq + d_img, d_f)

    def forward(self, stroke_seq: torch.Tensor, stroke_mask: torch.Tensor, udf_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stroke_seq (torch.Tensor): 스트로크 시퀀스. Shape: (B, Np, 3)
            stroke_mask (torch.Tensor): 스트로크 패딩 마스크. Shape: (B, Np)
            udf_image (torch.Tensor): UDF 이미지. Shape: (B, 1, 64, 64)

        Returns:
            torch.Tensor: 융합된 최종 잠재 벡터. Shape: (B, d_f)
        """
        z_seq = self.vector_encoder(stroke_seq, stroke_mask) # (B, d_seq)
        z_img = self.image_encoder(udf_image) # (B, d_img)
        
        fused = torch.cat([z_seq, z_img], dim=1) # (B, d_seq + d_img)
        z_e = self.fusion_layer(fused) # (B, d_f)
        return z_e


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        """
        Vector Quantization 레이어.
        Args:
            num_embeddings (int): 코드북의 크기 (K).
            embedding_dim (int): 각 코드북 벡터의 차원. Encoder의 출력 차원과 같아야 함.
            commitment_cost (float): Commitment 손실의 가중치 (β).
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # 코드북을 nn.Embedding으로 정의. (K, D)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # 코드북 벡터를 [-1, 1] 범위에서 균등 분포로 초기화
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 입력 z_e의 shape: (B, D)
        # 코드북의 shape: (K, D)
        
        # 1. 가장 가까운 코드북 벡터 찾기
        # 거리 계산: ||z_e - e_k||^2 = ||z_e||^2 - 2*z_e*e_k^T + ||e_k||^2
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(inputs, self.embedding.weight.t()))
        
        # 가장 가까운 코드의 인덱스 찾기
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # (B, 1)
        
        # 인덱스를 one-hot 벡터로 변환 (B, K)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 2. 양자화된 벡터 z_q 생성
        quantized = torch.matmul(encodings, self.embedding.weight) # (B, D)
        
        # 3. 손실 계산
        # 코드북 손실: 코드북 벡터가 인코더 출력에 가까워지도록 학습
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        # Commitment 손실: 인코더가 선택된 코드북 벡터에 전념하도록 학습
        q_latent_loss = F.mse_loss(inputs, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # 4. Straight-Through Estimator: Decoder로 가는 그래디언트를 인코더로 복사
        quantized = inputs + (quantized - inputs).detach()
        
        return loss, quantized, encoding_indices.squeeze()

class VectorDecoder(nn.Module):
    """잠재 벡터로부터 스트로크 시퀀스를 복원하는 Transformer 기반 디코더 [cite: 181]"""
    def __init__(self, output_dim=3, d_model=64, nhead=8, num_layers=6, d_f=128, max_len=48):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.latent_proj = nn.Linear(d_f, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, output_dim)
        self.coord_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
            nn.Sigmoid()
        )
        self.pen_state_head = nn.Linear(d_model, 1) # 활성화 함수 없이 로짓을 출력


    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        B = z_q.shape[0]
        memory = self.latent_proj(z_q).unsqueeze(1).repeat(1, self.max_len, 1) # (B, Np, d_model)
        
        # 타겟 시퀀스를 생성하기 위한 초기 입력 (0으로 채움)
        tgt = torch.zeros(B, self.max_len, self.d_model, device=z_q.device)
        tgt_pos = self.pos_encoder(tgt.permute(1, 0, 2)).permute(1, 0, 2)
        
        decoded_features = self.transformer_decoder(tgt_pos, memory) # (B, Np, d_model)

        coords = self.coord_head(decoded_features) # (B, Np, 2)
        pen_state_logit = self.pen_state_head(decoded_features).squeeze(-1) # (B, Np, 1) -> (B, Np)


        return coords, pen_state_logit


class ImageDecoder(nn.Module):
    """잠재 벡터로부터 UDF 이미지를 복원하는 Transposed CNN 기반 디코더 [cite: 183]"""
    def __init__(self, out_channels=1, d_f=128):
        super().__init__()
        self.latent_proj = nn.Linear(d_f, 512 * 2 * 2)
        # [cite_start]6개의 Transposed Convolutional Layer로 구성 [cite: 183]
        self.decoder = nn.Sequential(
            self._make_block(512, 512),       # 2x2 -> 4x4
            self._make_block(512, 256),       # 4x4 -> 8x8
            self._make_block(256, 128),       # 8x8 -> 16x16
            self._make_block(128, 64),        # 16x16 -> 32x32
            self._make_block(64, 32),         # 32x32 -> 64x64
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid() # UDF 값은 0~1 사이
        )

    def _make_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, z_f: torch.Tensor) -> torch.Tensor:
        x = self.latent_proj(z_f) # (B, 512*2*2)
        x = x.view(x.shape[0], 512, 2, 2) # (B, 512, 2, 2)
        reconstructed_udf = self.decoder(x) # (B, 1, 64, 64)
        return reconstructed_udf

class StrokeUDFDecoder(nn.Module):
    """Dual-Modal 디코더: 잠재 벡터로부터 Vector와 Image를 모두 복원"""
    def __init__(self, vector_decoder, image_decoder):
        super().__init__()
        self.vector_decoder = vector_decoder
        self.image_decoder = image_decoder

    def forward(self, z_q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        recon_coords, recon_pen_logits = self.vector_decoder(z_q)
        recon_udf = self.image_decoder(z_q)
        return recon_coords, recon_pen_logits, recon_udf

class StrokeFusionVQAE(nn.Module):
    """StrokeFusion의 인코더와 디코더를 VQ-VAE 구조로 결합한 모델"""
    def __init__(self, max_stroke_len=48, d_model=64, d_seq=32, d_img=64, d_f=128,
                 num_embeddings=512, commitment_cost=0.25):
        super().__init__()
        # 인코더 부분
        vec_enc = VectorEncoder(d_model=d_model, d_seq=d_seq)
        img_enc = ImageEncoder(d_img=d_img)
        self.encoder = StrokeUDFEncoder(vec_enc, img_enc, d_seq=d_seq, d_img=d_img, d_f=d_f)
        
        # 양자화 부분
        self.quantizer = VectorQuantizer(num_embeddings, d_f, commitment_cost)
        
        # 디코더 부분
        vec_dec = VectorDecoder(max_len=max_stroke_len, d_model=d_model, d_f=d_f)
        img_dec = ImageDecoder(d_f=d_f)
        self.decoder = StrokeUDFDecoder(vec_dec, img_dec)

    def forward(self, stroke_seq: torch.Tensor, stroke_mask: torch.Tensor, udf_image: torch.Tensor) -> Tuple:
        z_e = self.encoder(stroke_seq, stroke_mask, udf_image)
        vq_loss, z_q, _ = self.quantizer(z_e)
        recon_coords, recon_pen_logits, recon_udf = self.decoder(z_q)
        return recon_coords, recon_pen_logits, recon_udf, vq_loss



# --- 모델 테스트 ---
if __name__ == '__main__':
    BATCH_SIZE = 32
    MAX_STROKE_LEN = 48
    UDF_RESOLUTION = 64
    D_MODEL = 64
    D_SEQ = 32
    D_IMG = 64
    D_F = 128            # Encoder 출력 차원, Quantizer 코드북 벡터 차원
    NUM_EMBEDDINGS = 256 # 코드북 크기 (K)

    # 모델 초기화
    model = StrokeFusionVQAE(
        max_stroke_len=MAX_STROKE_LEN,
        d_model=D_MODEL,
        d_seq=D_SEQ,
        d_img=D_IMG,
        d_f=D_F,
        num_embeddings=NUM_EMBEDDINGS
    )

    # 더미 입력 데이터 생성
    dummy_stroke_seq = torch.rand(BATCH_SIZE, MAX_STROKE_LEN, 3)
    dummy_stroke_mask = torch.ones(BATCH_SIZE, MAX_STROKE_LEN)
    dummy_stroke_mask[0, 40:] = 0
    dummy_stroke_mask[1, 30:] = 0
    dummy_stroke_seq = dummy_stroke_seq * dummy_stroke_mask.unsqueeze(-1)
    dummy_udf_image = torch.rand(BATCH_SIZE, 1, UDF_RESOLUTION, UDF_RESOLUTION)

    # 모델 forward pass
    recon_stroke, recon_udf, vq_loss = model(dummy_stroke_seq, dummy_stroke_mask, dummy_udf_image)
    
    # 최종 학습 손실 계산 예시
    recon_loss_stroke = F.mse_loss(recon_stroke, dummy_stroke_seq)
    recon_loss_udf = F.mse_loss(recon_udf, dummy_udf_image)
    total_loss = recon_loss_stroke + recon_loss_udf + vq_loss

    # 출력 Shape 확인
    print("--- StrokeFusion VQ-VAE Test ---")
    print(f"Input Stroke Sequence Shape: {dummy_stroke_seq.shape}")
    print(f"Input UDF Image Shape:       {dummy_udf_image.shape}")
    print("-" * 35)
    print(f"Reconstructed Stroke Shape:  {recon_stroke.shape}")
    print(f"Reconstructed UDF Shape:     {recon_udf.shape}")
    print(f"VQ Loss (scalar):            {vq_loss.item():.4f}")
    print(f"Total Loss (scalar):         {total_loss.item():.4f}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Trainable Parameters: {total_params:,}")