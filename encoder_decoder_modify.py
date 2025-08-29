

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Tuple
from sklearn.cluster import KMeans
from vqtorch.nn import VectorQuant
import numpy as np

class PositionalEncoding(nn.Module):
    # (이전과 동일, 변경 없음)
    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 5000):
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return x
    
class SelfAttentionPooling(nn.Module):
    """
    어텐션 스코어를 학습하여 시퀀스를 풀링하는 레이어.
    """
    def __init__(self, input_dim, attention_dim=512):
        super().__init__()
        # 각 토큰 임베딩을 하나의 스코어(어텐션 로짓)로 변환하는 선형 레이어
        self.attention_transform = nn.Linear(input_dim, attention_dim, bias=True)
        self.attention_net = nn.Linear(attention_dim, 1, bias=False)


    def forward(self, sequence, mask):
        """
        sequence: (B, S, D) - 트랜스포머 인코더의 출력
        mask: (B, S) - 패딩 마스크 (1: 실제 토큰, 0: 패딩 토큰)
        """
        # (B, S, D) -> (B, S, 1)
        proj = torch.tanh(self.attention_transform(sequence))
        attention_logits = self.attention_net(proj)

        # 마스크를 적용하여 패딩된 위치의 로짓을 매우 작은 값으로 만듭니다.
        # 이렇게 하면 softmax 계산 시 해당 위치의 가중치가 0에 가까워집니다.
        mask_expanded = mask.unsqueeze(2) # (B, S, 1)
        attention_logits = attention_logits.masked_fill(mask_expanded == 0, -1e9)

        # Softmax를 통해 어텐션 가중치를 계산합니다.
        attention_weights = F.softmax(attention_logits, dim=1) # (B, S, 1)

        # 어텐션 가중치를 적용하여 시퀀스의 가중합을 계산합니다.
        # (B, S, D) * (B, S, 1) -> (B, S, D)
        # torch.sum(..., dim=1) -> (B, D)
        pooled_output = torch.sum(sequence * attention_weights, dim=1)
        
        return pooled_output
    
class VectorEncoder(nn.Module):
    def __init__(self, input_dim=3, d_model=256, d_seq=32, n_head=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, batch_first=True, dim_feedforward=d_model*4, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pooling = SelfAttentionPooling(input_dim=d_model)
        self.output_proj = nn.Linear(d_model, d_seq)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(src) * math.sqrt(self.d_model)
        pos_encoded = self.pos_encoder(embedded.permute(1, 0, 2)).permute(1, 0, 2)
        encoded_seq = self.transformer_encoder(pos_encoded, src_key_padding_mask=(src_mask == 0))
        pooled_output = self.pooling(encoded_seq, src_mask)
        projected_output = self.output_proj(pooled_output) # (B, d_seq)
        return projected_output
    
def ConvBlock(in_channels, out_channels, num_repeats):
    """Conv2d -> BatchNorm2d -> ReLU 블록을 생성하는 헬퍼 함수"""
    layers = []
    for i in range(num_repeats):
        # 첫 번째 Conv만 채널 수를 변경하고, 나머지는 채널 수를 유지
        cin = in_channels if i == 0 else out_channels
        layers.append(nn.Conv2d(cin, out_channels, kernel_size=3, padding=1))
        # layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)

class ImageEncoder(nn.Module):
    def __init__(self, in_channels=1, d_img=128):
        super().__init__()
        # 다이어그램의 'Co' (좌표 채널 추가)를 반영
        self.in_conv = nn.Conv2d(in_channels + 2, 64, kernel_size=3, padding=1) # +2 for coord channels
        
        # Encoder Path (Downsampling)
        self.down_block1 = ConvBlock(64, 64, num_repeats=1)
        self.down_block2 = ConvBlock(64, 128, num_repeats=1)
        self.down_block3 = ConvBlock(128, 256, num_repeats=2)
        self.down_block4 = ConvBlock(256, 512, num_repeats=2)
        
        self.pool = nn.MaxPool2d(2) # 2x2 Max Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projection = nn.Linear(512, d_img)



    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: (B, 1, 64, 64)
        B, _, H, W = src.shape
        
        # 좌표 채널 생성 및 입력과 결합 (CoordConv)
        coords_x = torch.linspace(-1, 1, W, device=src.device).view(1, 1, 1, W).expand(B, 1, H, W)
        coords_y = torch.linspace(-1, 1, H, device=src.device).view(1, 1, H, 1).expand(B, 1, H, W)
        x = torch.cat([src, coords_x, coords_y], dim=1) # (B, 3, 64, 64)
        
        x = self.in_conv(x) # -> (B, 64, 64, 64)
        
        x = self.down_block1(x)
        x = self.pool(x) # -> (B, 64, 32, 32)
        
        x = self.down_block2(x)
        x = self.pool(x) # -> (B, 128, 16, 16)
        
        x = self.down_block3(x)
        x = self.pool(x) # -> (B, 256, 8, 8)
        
        x = self.down_block4(x) # -> (B, 512, 8, 8)

        x = self.global_pool(x) # -> (B, 512, 1, 1)
        x = x.view(B, -1) # Flatten
        z_img = self.projection(x) # -> (B, d_img)
        
        return z_img
    
class SimpleFusion(nn.Module):
    def __init__(self, d_seq: int, d_img: int, embedding_dim: int, hidden_dim_ratio: int = 2):
        """
        d_seq: 시퀀스 인코더의 출력 차원
        d_img: 이미지 인코더의 출력 차원
        embedding_dim: 최종 출력 벡터의 차원
        hidden_dim_ratio: 은닉층의 크기를 조절하는 비율
        """
        super().__init__()
        
        # 입력 차원은 두 벡터의 차원을 합한 것
        input_dim = d_seq + d_img
        # 중간 은닉층의 차원
        hidden_dim = input_dim * hidden_dim_ratio

        # 간단한 2층 MLP (FC Layers) 구조
        self.fusion_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )

        self.norm_seq = nn.LayerNorm(d_seq)
        self.norm_img = nn.LayerNorm(d_img)

        self.final_norm = nn.LayerNorm(embedding_dim)

    def forward(self, z_seq: torch.Tensor, z_img: torch.Tensor) -> torch.Tensor:
        """
        z_seq: (B, d_seq) 모양의 시퀀스 벡터
        z_img: (B, d_img) 모양의 이미지 벡터
        """
        z_seq_norm = self.norm_seq(z_seq)
        z_img_norm = self.norm_img(z_img)
        # 1. 두 벡터를 dim=1 (피처 차원) 기준으로 이어 붙입니다.
        # (B, d_seq) + (B, d_img) -> (B, d_seq + d_img)
        fused_vector = torch.cat([z_seq_norm, z_img_norm], dim=1)

        # 2. MLP를 통과시켜 정보를 융합하고 최종 차원으로 변환합니다.
        output = self.fusion_net(fused_vector)
        
        # 3. 최종 정규화를 거쳐 반환합니다.
        return self.final_norm(output)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.commitment_cost = commitment_cost

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        inputs: (B, D) - 이미 퓨전(풀링)된 단일 벡터
        """
        # --- 퓨전 벡터와 코드북 간의 양자화 수행 ---

        # (B, D) 모양의 입력 벡터와 코드북 간의 거리를 계산
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) +
                     torch.sum(self.embedding.weight**2, dim=1) -
                     2 * torch.matmul(inputs, self.embedding.weight.t()))
        
        # 각 입력 벡터당 하나의 가장 가까운 코드 인덱스를 찾음
        indices = torch.argmin(distances, dim=1) # (B,)
        num_unique_indices = len(torch.unique(indices))
        # print('indicies:', indices)

        print(f"Unique indices used in this batch: {num_unique_indices}")


        # 인덱스를 이용해 양자화된 벡터를 가져옴
        quantized_vector = F.embedding(indices, self.embedding.weight) # (B, D)
        
        encodings = F.one_hot(indices, self.num_embeddings).float()
        avg_probs = torch.mean(encodings, dim=0)
        # 엔트로피 계산 (log(0) 방지를 위해 작은 값 추가)
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        perplexity = torch.exp(entropy)
        # VQ 손실 계산
        loss = (F.mse_loss(quantized_vector, inputs.detach()) +
                self.commitment_cost * F.mse_loss(inputs, quantized_vector.detach()))
        
        # Straight-Through Estimator 적용
        quantized_vector_sg = inputs + (quantized_vector - inputs).detach()
        
        # 최종적으로 단일 손실, 단일 양자화 벡터, 단일 인덱스를 반환
        return loss, quantized_vector_sg, indices, perplexity


class VectorQuantizerSeq(nn.Module):
    """
    시퀀스 입력을 지원하는 VQ 레이어.
    입력: (B, C, N) 또는 (B, N, C)
    출력: quantized는 입력과 동일한 형태, indices는 (B, N)
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1/num_embeddings, 1/num_embeddings)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # x: (B, C, N) 또는 (B, N, C)
        transpose_back = False
        if x.dim() != 3:
            raise ValueError(f"expected 3D tensor, got {x.shape}")

        if x.size(1) == self.embedding_dim:
            # (B, C, N) -&gt; (B, N, C)
            x = x.permute(0, 2, 1).contiguous()
            transpose_back = True
        elif x.size(2) == self.embedding_dim:
            # already (B, N, C)
            pass
        else:
            raise ValueError(f"embedding_dim mismatch: got {x.shape}, expected C or last dim = {self.embedding_dim}")

        B, N, C = x.shape
        flat_x = x.view(B * N, C)  # (B·N, C)

        # 거리 계산
        # (B·N, 1) + (num_embeddings,) - 2·(B·N, num_embeddings)
        distances = (flat_x.pow(2).sum(dim=1, keepdim=True)
                     + self.embedding.weight.pow(2).sum(dim=1)
                     - 2 * flat_x @ self.embedding.weight.t())

        indices = torch.argmin(distances, dim=1)                 # (B·N,)
        quantized = F.embedding(indices, self.embedding.weight)  # (B·N, C)
        quantized = quantized.view(B, N, C)

        # 손실(코드북 + 커밋)
        # codebook: || sg[x] - e ||^2, commitment: || x - sg[e] ||^2
        # (stopgrad는 detach로)
        codebook_loss = F.mse_loss(quantized, x.detach())
        commit_loss   = F.mse_loss(x, quantized.detach())
        vq_loss = codebook_loss + self.commitment_cost * commit_loss

        # 마스크가 있으면 유효 토큰에만 가중 손실을 줄 수도 있음 (옵션)
        if mask is not None:
            # mask: (B, N) with 1(valid)/0(pad)
            m = mask.view(B, N, 1).float()
            # 가중 평균으로 재계산 (선택사항)
            codebook_loss = ((quantized - x.detach()).pow(2) * m).sum() / (m.sum() + 1e-8)
            commit_loss   = ((x - quantized.detach()).pow(2) * m).sum() / (m.sum() + 1e-8)
            vq_loss = codebook_loss + self.commitment_cost * commit_loss

        # straight-through estimator
        quantized_st = x + (quantized - x).detach()

        # 원래 형태로 복원
        if transpose_back:
            quantized_st = quantized_st.permute(0, 2, 1).contiguous()  # (B, C, N)
        indices = indices.view(B, N)

        # perplexity(선택): 토큰 사용 다양성 확인
        with torch.no_grad():
            encodings = F.one_hot(indices.view(-1), num_classes=self.num_embeddings).float()
            avg_probs = encodings.mean(dim=0)
            entropy = -(avg_probs * (avg_probs + 1e-10).log()).sum()
            perplexity = entropy.exp()

        return vq_loss, quantized_st, indices, perplexity
    
class VectorDecoderAR(nn.Module):
    def __init__(self, output_dim=3, d_model=256, n_head=8, num_layers=6, embedding_dim=256):
        super().__init__()
        self.d_model = d_model
        self.input_embedding = nn.Linear(output_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Quantized Vector를 Transformer 메모리로 사용하기 위한 프로젝션 레이어
        self.memory_proj = nn.Linear(embedding_dim, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_head, batch_first=True, dim_feedforward=d_model*4, dropout=0.1)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # 최종 출력 헤드
        self.output_head = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.Sigmoid()
        )

    def forward(self, tgt_seq: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        tgt_seq: (B, T, 3) - 현재까지 생성된 타겟 시퀀스
        memory: (B, D_emb) - Quantizer로부터 온 단일 컨텍스트 벡터
        """
        # 1. 타겟 시퀀스 임베딩 및 포지셔널 인코딩
        tgt_embedded = self.input_embedding(tgt_seq) * math.sqrt(self.d_model)
        tgt_pos = self.pos_encoder(tgt_embedded.permute(1, 0, 2)).permute(1, 0, 2)
        
        # 2. 메모리(컨텍스트 벡터) 준비
        # (B, D_emb) -> (B, D_model)
        memory_projected = self.memory_proj(memory)
        # Cross-Attention을 위해 시퀀스처럼 차원을 확장: (B, D_model) -> (B, 1, D_model)
        memory_projected = memory_projected.unsqueeze(1)
        seq_len = tgt_seq.size(1)
        memory_tiled = memory_projected.repeat(1, seq_len, 1) 

        # 3. 디코더에 입력
        # 미래 정보를 볼 수 없도록 하는 Causal Mask 생성
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt_seq.device)
        decoded_features = self.transformer_decoder(tgt_pos, memory_tiled, tgt_mask=causal_mask)
        
        # 4. 다음 스텝의 포인트를 예측
        output = self.output_head(decoded_features)
        
        return output
    
class ImageDecoder(nn.Module):
    """다이어그램 구조에 대칭되는 디코더 클래스"""
    def __init__(self, embedding_dim: int, output_channels: int = 1, output_res: int = 64):
        super().__init__()
        
        # 1. 입력 벡터를 2D 피처맵으로 변환 (Encoder의 Flatten&MLP와 대칭)
        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512 * 8 * 8)
        )
        
        # Decoder Path (Upsampling)
        self.up_block1 = ConvBlock(512, 256, num_repeats=2)
        self.up_block2 = ConvBlock(256, 128, num_repeats=2)
        self.up_block3 = ConvBlock(128, 64, num_repeats=1)
        self.up_block4 = ConvBlock(64, 64, num_repeats=1)
        
        # MaxPool의 역연산으로 Upsample 사용
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # 최종 출력 레이어
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z_fusion: torch.Tensor) -> torch.Tensor:
        # z_fusion: (B, embedding_dim)
        x = self.input_proj(z_fusion)
        x = x.view(z_fusion.size(0), 512, 8, 8) # Reshape to feature map
        
        x = self.up_block1(x)
        x = self.upsample(x) # -> (B, 256, 16, 16)
        
        x = self.up_block2(x)
        x = self.upsample(x) # -> (B, 128, 32, 32)
        
        x = self.up_block3(x)
        x = self.upsample(x) # -> (B, 64, 64, 64)
        
        x = self.up_block4(x)
        
        recon_udf = self.out_conv(x) # -> (B, 1, 64, 64)
        
        return recon_udf

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_seq: int, d_img: int, n_head: int, output_dim: int):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_seq, kdim=d_img, vdim=d_img, 
                                                     num_heads=n_head, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_seq)
        self.output_proj = nn.Linear(d_seq, output_dim)

    def forward(self, z_seq: torch.Tensor, z_img: torch.Tensor) -> torch.Tensor:
        # 어텐션 입력을 위해 z_seq, z_img에 sequence 차원을 추가 (길이 1)
        z_seq_q = z_seq.unsqueeze(1) # Query: (B, 1, D_seq)
        z_img_kv = z_img.unsqueeze(1) # Key/Value: (B, 1, D_img)

        # Cross-Attention 수행
        attn_output, _ = self.cross_attention(query=z_seq_q, key=z_img_kv, value=z_img_kv)

        # Residual connection 및 LayerNorm
        fused = self.layer_norm(z_seq_q + attn_output)

        # 최종 차원으로 프로젝션
        return self.output_proj(fused.squeeze(1))

# 사용 시: self.fusion = SimpleFusion(...) 대신

    
class MultimodalVQVAE(nn.Module):
    def __init__(self,
                 d_seq: int = 128,
                 d_img: int = 128,
                 embedding_dim: int = 256,
                 d_model: int = 256,
                 num_codes: int = 1024,
                 commitment_cost: float = 0.25,
                 udf_res: int = 64,
                 image_num_layer: int = 3,
                 n_head: int = 8,
                 num_layers: int = 6,
                 ):
        """
        지금까지 만든 모든 모듈을 통합하여 VQ-VAE 모델을 구성합니다.
        
        Args:
            d_seq (int): 벡터 인코더의 출력 차원
            d_img (int): 이미지 인코더의 출력 차원
            embedding_dim (int): 퓨전 및 양자화에 사용될 임베딩 차원
            num_codes (int): 코드북의 크기 (코드의 개수)
            commitment_cost (float): VQ-VAE의 commitment 손실 가중치
            udf_output_res (int): UDF 디코더의 최종 출력 해상도
            udf_num_upsamples (int): UDF 디코더의 업샘플링 횟수
        """
        super().__init__()
        
        # 1. 인코더 모듈
        self.vec_encoder = VectorEncoder(d_model=d_model, d_seq=d_seq, n_head=n_head, num_layers=num_layers)
        self.img_encoder = ImageEncoder(d_img=d_img)

        # 2. 퓨전 모듈
        self.fusion = SimpleFusion(d_seq=d_seq, d_img=d_img, embedding_dim=embedding_dim)
        # self.fusion = CrossAttentionFusion(d_seq=d_seq, d_img=d_img, n_head=n_head, output_dim=embedding_dim)

        
        # 3. 양자화 모듈
        self.quantizer = VectorQuantizer(num_embeddings=num_codes, 
                                         embedding_dim=embedding_dim, 
                                         commitment_cost=commitment_cost)
        
        # 4. 디코더 모듈
        self.vec_decoder = VectorDecoderAR(d_model=d_model, n_head=n_head, embedding_dim=embedding_dim)
        self.udf_decoder = ImageDecoder(embedding_dim=embedding_dim,
                                             output_res=udf_res)
        self.sos_token = nn.Parameter(torch.randn(1, 1, 3))


    def initialize_codebook_with_kmeans(self, dataloader, device, num_batches=50):
        """
        데이터로더의 일부 데이터를 사용하여 K-Means 클러스터링으로 코드북을 초기화합니다.
        """
        print("⏳ Initializing codebook with K-Means...")
        self.eval() # 모델을 평가 모드로 설정
        encoder_outputs = []
        cluster_num = 10
        
        # 1. 훈련 데이터 일부로 인코더 출력 수집
        with torch.no_grad():
            for i, (stroke_seq, stroke_mask, udf_image, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                stroke_seq, stroke_mask, udf_image = stroke_seq.to(device), stroke_mask.to(device), udf_image.to(device)
                
                z_seq = self.vec_encoder(stroke_seq, stroke_mask)
                z_img = self.img_encoder(udf_image)
                fused_vector = self.fusion(z_seq, z_img)
                encoder_outputs.append(fused_vector.cpu().numpy())

        encoder_outputs = np.concatenate(encoder_outputs, axis=0)

        # 2. K-Means 클러스터링으로 중심점 계산
        print(f"Running K-Means on {encoder_outputs.shape[0]} samples to find {self.quantizer.num_embeddings} clusters...")
        kmeans = KMeans(n_clusters=self.quantizer.num_embeddings, n_init='auto', random_state=0)

        kmeans.fit(encoder_outputs)
        centroids = torch.from_numpy(kmeans.cluster_centers_).to(device)

        # 3. 코드북 가중치를 계산된 중심점으로 설정
        self.quantizer.embedding.weight.data.copy_(centroids)
        self.train() # 모델을 다시 훈련 모드로 설정
        print("✅ Codebook initialized successfully.")

    def forward(self, 
                src_points: torch.Tensor, 
                src_mask: torch.Tensor, 
                src_image: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        학습(training) 시에 사용되는 forward pass 입니다.
        Teacher Forcing 방식으로 디코더를 학습합니다.
        """
        # 1. 인코딩: 각 modality의 입력을 단일 벡터로 인코딩
        z_seq = self.vec_encoder(src_points, src_mask)
        z_img = self.img_encoder(src_image)
        
        # 2. 퓨전: 두 벡터를 결합하여 공유 잠재 공간으로 프로젝션
        fused_vector = self.fusion(z_seq, z_img)
        
        # 3. 양자화: 퓨전된 벡터를 이산적인 코드로 변환
        vq_loss, quantized_vector, indices, perplexity = self.quantizer(fused_vector)
        

        tgt_points = src_points  # 디코더의 입력으로 사용할 타겟 시퀀스
        batch_size = tgt_points.size(0)
        
        # 2. SOS 토큰을 배치 크기에 맞게 복제합니다.
        sos = self.sos_token.expand(batch_size, -1, -1)
        decoder_input = torch.cat([sos, tgt_points[:, :-1, :]], dim=1)

        
        # 벡터 드로잉 복원
        predicted_points = self.vec_decoder(decoder_input, quantized_vector)
        # UDF 이미지 복원
        predicted_udf = self.udf_decoder(quantized_vector)
        
        return predicted_points, predicted_udf, vq_loss, indices, perplexity
    



    @torch.no_grad()
    def generate(self, 
                 src_points: torch.Tensor = None, 
                 src_mask: torch.Tensor = None, 
                 src_image: torch.Tensor = None,
                 max_len: int = 150,
                 quantized_vector: torch.Tensor = None
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        추론(inference) 시에 사용되는 생성 함수입니다.
        벡터 드로잉은 자기회귀적으로, UDF는 한 번에 생성합니다.
        """
        
        if  quantized_vector is None:
            # 1-3. 양자화된 컨텍스트 벡터 얻기 (forward와 동일)
            z_seq = self.vec_encoder(src_points, src_mask)
            z_img = self.img_encoder(src_image)
            fused_vector = self.fusion(z_seq, z_img)
            _, quantized_vector, indices, perplexity = self.quantizer(fused_vector)
            
            # 4. UDF 이미지 생성
            generated_udf = self.udf_decoder(quantized_vector)
            batch_size = src_points.size(0)

        else:
            quantized_vector = quantized_vector
            generated_udf = self.udf_decoder(quantized_vector)
            batch_size = 1
        # 5. 벡터 드로잉 자기회귀적 생성
        # 시작을 알리는 SOS(Start-of-Sequence) 포인트 생성
        generated_seq = self.sos_token.expand(batch_size, -1, -1)

        for _ in range(max_len):
            output_points = self.vec_decoder(generated_seq, quantized_vector)
            next_point = output_points[:, -1:, :]
            generated_seq = torch.cat([generated_seq, next_point], dim=1)
        
        
        return generated_seq[:, 1:, :], generated_udf
    
    @torch.no_grad()
    def encode(self, 
               src_points: torch.Tensor, 
               src_mask: torch.Tensor, 
               src_image: torch.Tensor
               ) -> torch.Tensor:
        """
        입력 데이터를 받아 양자화된 인덱스(토큰)만 반환합니다.
        """
        z_seq = self.vec_encoder(src_points, src_mask)
        z_img = self.img_encoder(src_image)
        fused_vector = self.fusion(z_seq, z_img)
        
        # quantizer의 forward는 (loss, quantized, indices, perplexity)를 반환
        # 이 중 indices만 필요함
        _, _, indices, _ = self.quantizer(fused_vector)
        self.train() # 다시 훈련 모드
        return indices
    

    @torch.no_grad()
    def decode_from_token(self, token_index: torch.Tensor, max_len: int = 48) -> torch.Tensor:
        """
        [기존 함수] 단일 스트로크 토큰 인덱스(스칼라 텐서)를 받아 스트로크 하나를 생성합니다.
        """
        quantized_vector = self.quantizer.embedding(token_index).unsqueeze(0)
        generated_stroke, _ = self.generate(quantized_vector=quantized_vector, max_len=max_len)
        # print('generated_stroke: ', generated_stroke[:10])
        return generated_stroke.squeeze(0)
    
    @torch.no_grad()
    def decode_from_token_sequence(self, token_indices: torch.Tensor, max_len: int = 48) -> list:
        """
        스트로크 토큰 인덱스의 시퀀스(1D 텐서)를 입력받아,
        각 토큰을 스트로크로 디코딩한 후 그 리스트를 반환합니다.
        """
        decoded_strokes = []
        # 입력된 토큰 시퀀스를 순회하는 루프
        for token_index in token_indices:
            # 각 토큰에 대해 단일 스트로크 생성 함수 호출
            stroke_tensor = self.decode_from_token(token_index, max_len=max_len)
            decoded_strokes.append(stroke_tensor)
        return decoded_strokes




class MultimodalVAE(nn.Module):
    def __init__(self,
                 d_seq: int = 128,
                 d_img: int = 128,
                 embedding_dim: int = 256,
                 d_model: int = 256,
                 udf_res: int = 64,
                 n_head: int = 8,
                 num_layers: int = 6,
                 point_dim: int = 3):
        super().__init__()
        
        # 1. 인코더 및 퓨전 모듈 (이전과 동일)
        self.vec_encoder = VectorEncoder(input_dim=point_dim, d_model=d_model, d_seq=d_seq, n_head=n_head, num_layers=num_layers)
        self.img_encoder = ImageEncoder(d_img=d_img)
        self.fusion = SimpleFusion(d_seq=d_seq, d_img=d_img, embedding_dim=embedding_dim)
        
        # --- [변경점] VectorQuantizer를 mu, log_var 레이어로 교체 ---
        self.fc_mu = nn.Linear(embedding_dim, embedding_dim)
        self.fc_log_var = nn.Linear(embedding_dim, embedding_dim)
        
        # 4. 디코더 모듈 (이전과 동일)
        self.vec_decoder = VectorDecoderAR(output_dim=point_dim, d_model=d_model, n_head=n_head, embedding_dim=embedding_dim)
        self.udf_decoder = ImageDecoder(embedding_dim=embedding_dim, output_res=udf_res)
        
        self.sos_token = nn.Parameter(torch.randn(1, 1, point_dim))

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization Trick을 사용하여 z를 샘플링"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, 
                src_points: torch.Tensor, 
                src_mask: torch.Tensor, 
                src_image: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # 1. 인코딩 및 퓨전 (이전과 동일)
        z_seq = self.vec_encoder(src_points, src_mask)
        z_img = self.img_encoder(src_image)
        fused_vector = self.fusion(z_seq, z_img)
        
        # --- [변경점] VQ 대신 VAE 잠재 공간 처리 ---
        # 2. mu와 log_var 계산
        mu = self.fc_mu(fused_vector)
        log_var = self.fc_log_var(fused_vector)
        
        # 3. Reparameterization Trick으로 잠재 벡터 z 샘플링
        z = self.reparameterize(mu, log_var)
        z = mu

        # have to modify just for overfitting test

        
        # 4. KL Divergence 손실 계산
        # N(mu, var)와 N(0, 1) 사이의 KL Divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

        # 5. 디코딩 (입력으로 양자화된 벡터 대신 샘플링된 z 사용)
        tgt_points = src_points
        batch_size = tgt_points.size(0)
        sos = self.sos_token.expand(batch_size, -1, -1)
        decoder_input = torch.cat([sos, tgt_points[:, :-1, :]], dim=1)
        
        predicted_points = self.vec_decoder(decoder_input, z)
        predicted_udf = self.udf_decoder(z)
        
        return predicted_points, predicted_udf, kl_loss

    @torch.no_grad()
    def generate(self, 
                 src_points: torch.Tensor, 
                 src_mask: torch.Tensor, 
                 src_image: torch.Tensor,
                 max_len: int = 150
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        
        z_seq = self.vec_encoder(src_points, src_mask)
        z_img = self.img_encoder(src_image)
        fused_vector = self.fusion(z_seq, z_img)
        
        mu = self.fc_mu(fused_vector)
        log_var = self.fc_log_var(fused_vector)
        z = self.reparameterize(mu, log_var)
        z = mu
        
        generated_udf = self.udf_decoder(z)
        generated_seq = self.sos_token.expand(z.size(0), -1, -1)
        for _ in range(max_len):
            output_points = self.vec_decoder(generated_seq, z)
            next_point = output_points[:, -1:, :]
                        
            generated_seq = torch.cat([generated_seq, next_point], dim=1)
        
        return generated_seq[:, 1:, :], generated_udf

class BoundingBoxVQVAE(nn.Module):
    def __init__(self,
                 input_dim=4,
                 latent_dim=64,
                 num_embeddings=256,
                 commitment_cost=0.25):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

        # 사용자님의 VectorQuantizer 인스턴스 사용
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, input_dim)
        )
        self.num_embeddings = num_embeddings

    def forward(self, bbox_data: torch.Tensor):
        # bbox_data: (batch_size, input_dim)

        # 1. 인코딩
        latents = self.encoder(bbox_data) # (batch_size, latent_dim)

        # 2. 양자화 (VectorQuantizer의 반환 값 순서에 맞게 변수명 변경)
        vq_loss, quantized_latents, min_encoding_indices, perplexity = self.quantizer(latents)

        # 3. 디코딩
        reconstructed_bbox = self.decoder(quantized_latents) # (batch_size, input_dim)

        return reconstructed_bbox, vq_loss, perplexity, min_encoding_indices

    def encode(self, data: torch.Tensor):
        latents = self.encoder(data)
        # VectorQuantizer의 forward는 (loss, quantized_vector_sg, indices, perplexity) 반환
        # 여기서 indices (토큰 ID)만 필요하므로 인덱스 2를 사용
        _, _, min_encoding_indices, _ = self.quantizer(latents) 
        return min_encoding_indices

    def decode(self, token_indices: torch.Tensor):
        # self.quantizer.embedding은 VectorQuantizer 내부의 nn.Embedding 객체
        quantized_latents = self.quantizer.embedding(token_indices)
        reconstructed_bbox = self.decoder(quantized_latents)
        return reconstructed_bbox
    

       ### [수정] 위치(Location) 코드북 초기화 함수 ###
    def initialize_location_codebook(self, dataloader, device, num_batches=50):
        print("⏳ Initializing Location Codebook with K-Means...")
        self.to(device)
        self.eval()
        loc_latents_list = []
        
        with torch.no_grad():
            for i, (_, _, _, bbox) in enumerate(dataloader):
                if i >= num_batches: break
                bbox = bbox.to(device)
                locations = bbox[:, :2]
                loc_latents = self.encoder(locations)
                loc_latents_list.append(loc_latents.cpu().numpy())

        loc_latents_all = np.concatenate(loc_latents_list, axis=0)
        print(f"Running K-Means for Location on {loc_latents_all.shape[0]} samples...")
        kmeans_loc = KMeans(n_clusters=self.num_embeddings, n_init='auto', random_state=0)
        kmeans_loc.fit(loc_latents_all)
        loc_centroids = torch.from_numpy(kmeans_loc.cluster_centers_).to(device)
        self.quantizer.embedding.weight.data.copy_(loc_centroids)
        print("✅ Location codebook initialized.")
        self.train()

    ### [수정] 크기(Size) 코드북 초기화 함수 ###
    def initialize_size_codebook(self, dataloader, device, num_batches=50):
        print("⏳ Initializing Size Codebook with K-Means...")
        self.to(device)
        self.eval()
        size_latents_list = []

        with torch.no_grad():
            for i, (_, _, _, bbox) in enumerate(dataloader):
                if i >= num_batches: break
                bbox = bbox.to(device)
                sizes = bbox[:, 2:]
                size_latents = self.encoder(sizes)
                size_latents_list.append(size_latents.cpu().numpy())

        size_latents_all = np.concatenate(size_latents_list, axis=0)
        print(f"Running K-Means for Size on {size_latents_all.shape[0]} samples...")
        kmeans_size = KMeans(n_clusters=self.num_embeddings, n_init='auto', random_state=0)
        kmeans_size.fit(size_latents_all)
        size_centroids = torch.from_numpy(kmeans_size.cluster_centers_).to(device)
        self.quantizer.embedding.weight.data.copy_(size_centroids)
        print("✅ Size codebook initialized.")
        self.train()


class BoundingBoxVQVAE_Dual(nn.Module):
    def __init__(self,
                 loc_input_dim=2,      # (cx, cy)
                 size_input_dim=2,     # (w, h)
                 loc_latent_dim=32,    # 위치 latent 차원
                 size_latent_dim=32,   # 크기 latent 차원
                 num_loc_embeddings=128,  # 위치 코드북 크기
                 num_size_embeddings=128, # 크기 코드북 크기
                 commitment_cost=0.25):
        super().__init__()
        
        # --- 1. 분리된 인코더 정의 ---
        self.loc_encoder = nn.Sequential(
            nn.Linear(loc_input_dim, loc_latent_dim * 2),
            nn.ReLU(),
            nn.Linear(loc_latent_dim * 2, loc_latent_dim)
        )
        self.size_encoder = nn.Sequential(
            nn.Linear(size_input_dim, size_latent_dim * 2),
            nn.ReLU(),
            nn.Linear(size_latent_dim * 2, size_latent_dim)
        )
        
        # --- 2. 분리된 VQ 레이어 정의 ---
        self.loc_quantizer = VectorQuantizer(num_loc_embeddings, loc_latent_dim, commitment_cost)
        self.size_quantizer = VectorQuantizer(num_size_embeddings, size_latent_dim, commitment_cost)
        
        # --- 3. 통합된 디코더 정의 ---
        # 결합된 latent 차원을 입력으로 받음 (loc_latent_dim + size_latent_dim)
        combined_latent_dim = loc_latent_dim + size_latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(combined_latent_dim, combined_latent_dim * 2),
            nn.ReLU(),
            # 4차원 (cx, cy, w, h) 전체를 복원
            nn.Linear(combined_latent_dim * 2, loc_input_dim + size_input_dim) 
        )
        
        # 하이퍼파라미터 저장
        self.num_loc_embeddings = num_loc_embeddings
        self.num_size_embeddings = num_size_embeddings

    def forward(self, bbox_data: torch.Tensor):
        # bbox_data: (batch_size, 4) -&gt; (cx, cy, w, h)
        
        # --- 데이터 분리 ---
        loc_data = bbox_data[:, :2]
        size_data = bbox_data[:, 2:]
        
        # --- 개별 인코딩 ---
        loc_latents = self.loc_encoder(loc_data)
        size_latents = self.size_encoder(size_data)
        
        # --- 개별 양자화 ---
        vq_loss_loc, q_loc, loc_indices, perplexity_loc = self.loc_quantizer(loc_latents)
        vq_loss_size, q_size, size_indices, perplexity_size = self.size_quantizer(size_latents)
        
        # --- 정보 통합 ---
        combined_q = torch.cat([q_loc, q_size], dim=1)
        
        # --- 통합 디코딩 ---
        reconstructed_bbox = self.decoder(combined_q)
        
        # 손실 결합
        total_vq_loss = vq_loss_loc + vq_loss_size
        
        # 전체 반환 값 구성
        losses = {'vq_loss': total_vq_loss, 'vq_loss_loc': vq_loss_loc, 'vq_loss_size': vq_loss_size}
        perplexities = {'loc': perplexity_loc, 'size': perplexity_size}
        indices = {'loc': loc_indices, 'size': size_indices}
        
        return reconstructed_bbox, losses, perplexities, indices

    def encode(self, bbox_data: torch.Tensor):
        """ Bbox 데이터를 받아 위치와 크기 토큰을 각각 반환합니다. """
        loc_data = bbox_data[:, :2]
        size_data = bbox_data[:, 2:]

        loc_latents = self.loc_encoder(loc_data)
        size_latents = self.size_encoder(size_data)
        
        _, _, loc_indices, _ = self.loc_quantizer(loc_latents)
        _, _, size_indices, _ = self.size_quantizer(size_latents)
        
        return loc_indices, size_indices

    def decode(self, loc_indices: torch.Tensor, size_indices: torch.Tensor):
        """ 위치와 크기 토큰을 받아 전체 Bbox를 복원합니다. """
        q_loc = self.loc_quantizer.embedding(loc_indices)
        q_size = self.size_quantizer.embedding(size_indices)
        
        combined_q = torch.cat([q_loc, q_size], dim=1)
        reconstructed_bbox = self.decoder(combined_q)
        return reconstructed_bbox

    def initialize_codebooks(self, dataloader, device, num_batches=50):
        print("⏳ Initializing Dual Codebooks with K-Means...")
        self.to(device)
        self.eval()
        loc_latents_list, size_latents_list = [], []
        
        with torch.no_grad():
            for i, (_, _, _, bbox) in enumerate(dataloader):
                if i >= num_batches: break
                bbox = bbox.to(device)
                locations = bbox[:, :2]
                sizes = bbox[:, 2:]
                
                loc_latents_list.append(self.loc_encoder(locations).cpu().numpy())
                size_latents_list.append(self.size_encoder(sizes).cpu().numpy())

        # 위치 코드북 초기화
        loc_latents_all = np.concatenate(loc_latents_list, axis=0)
        print(f"Running K-Means for Location on {loc_latents_all.shape[0]} samples...")
        kmeans_loc = KMeans(n_clusters=self.num_loc_embeddings, n_init='auto', random_state=0)
        kmeans_loc.fit(loc_latents_all)
        loc_centroids = torch.from_numpy(kmeans_loc.cluster_centers_).float().to(device)
        self.loc_quantizer.embedding.weight.data.copy_(loc_centroids)
        print("✅ Location codebook initialized.")

        # 크기 코드북 초기화
        size_latents_all = np.concatenate(size_latents_list, axis=0)
        print(f"Running K-Means for Size on {size_latents_all.shape[0]} samples...")
        kmeans_size = KMeans(n_clusters=self.num_size_embeddings, n_init='auto', random_state=0)
        kmeans_size.fit(size_latents_all)
        size_centroids = torch.from_numpy(kmeans_size.cluster_centers_).float().to(device)
        self.size_quantizer.embedding.weight.data.copy_(size_centroids)
        print("✅ Size codebook initialized.")
        
        self.train()