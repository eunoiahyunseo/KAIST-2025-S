import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

# --- Helper Modules (변경 없음) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = x
        h = F.silu(self.norm1(h))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return self.shortcut(x) + h

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4, head_channels=64):
        super().__init__()
        self.num_heads = num_heads
        hidden_dim = num_heads * head_channels
        self.scale = head_channels ** -0.5
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, hidden_dim * 3, 1)
        self.proj_out = nn.Conv2d(hidden_dim, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (nh ch) h w -> b nh (h w) ch', nh=self.num_heads)
        k = rearrange(k, 'b (nh ch) h w -> b nh (h w) ch', nh=self.num_heads)
        v = rearrange(v, 'b (nh ch) h w -> b nh (h w) ch', nh=self.num_heads)
        attention = torch.einsum('bnqc,bnkc->bnqk', q, k) * self.scale
        attention = F.softmax(attention, dim=-1)
        out = torch.einsum('bnqk,bnkc->bnqc', attention, v)
        out = rearrange(out, 'b nh (h w) ch -> b (nh ch) h w', h=h, w=w)
        return x + self.proj_out(out)

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

# --- Main U-Net Model (안정적인 구조로 전면 재구성) ---

class UNetModel(nn.Module):
    def __init__(
        self,
        num_classes=257,
        channels=96,
        depth=5,
        channel_mults=(1, 2, 4, 4),
        num_heads=4,
        head_channels=64,
        attention_resolutions=(16,),
        dropout=0.4,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(num_classes, channels)
        self.init_conv = nn.Conv2d(3 * channels, channels, 3, padding=1)
        
        # --- [FIXED] 명확한 레벨별 모듈 구성 ---
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        
        # 채널 크기 결정
        ch_mults = [1] + list(channel_mults)
        dims = [channels * m for m in ch_mults]
        
        # --- Downsampling 경로 ---
        for i in range(depth):
            in_ch = dims[i-1] if i > 0 else channels
            out_ch = dims[i]
            
            # 각 레벨은 2개의 블록(ResBlock/AttnBlock)과 다운샘플러로 구성
            block1 = ResidualBlock(in_ch, out_ch, dropout)
            block2 = ResidualBlock(out_ch, out_ch, dropout)
            
            # Attention 적용 여부 확인
            current_res = 32 // (2**i)
            use_attn = current_res in attention_resolutions
            
            down_block = nn.ModuleList([block1, block2])
            if use_attn:
                down_block.append(AttentionBlock(out_ch, num_heads, head_channels))
            
            # 다운샘플러 추가 (마지막 레벨 제외)
            if i < depth - 1:
                down_block.append(Downsample(out_ch))
            
            self.down_blocks.append(down_block)

        # --- Middle 경로 ---
        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, dropout)
        self.mid_attn = AttentionBlock(mid_dim, num_heads, head_channels)
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, dropout)

        # --- Upsampling 경로 ---
        for i in reversed(range(depth)):
            in_ch = dims[i+1] if i < depth - 1 else mid_dim
            out_ch = dims[i]
            skip_ch = dims[i]
            
            # 각 레벨은 2개의 블록과 업샘플러로 구성
            block1 = ResidualBlock(in_ch + skip_ch, out_ch, dropout)
            
            # [BUG FIX] block2의 입력 채널은 block1의 출력 채널인 out_ch가 되어야 합니다.
            block2 = ResidualBlock(out_ch, out_ch, dropout) # <-- 이 부분이 수정되었습니다.

            current_res = 32 // (2**i)
            use_attn = current_res in attention_resolutions

            up_block = nn.ModuleList([block1, block2])
            if use_attn:
                up_block.append(AttentionBlock(out_ch, num_heads, head_channels))

            # 업샘플러 추가 (첫 레벨 제외)
            if i > 0:
                up_block.append(Upsample(out_ch))
            
            self.up_blocks.append(up_block)

        # --- Output 경로 ---
        self.out_norm = nn.GroupNorm(32, channels)
        self.out_conv = nn.Conv2d(channels, 3 * num_classes, 1)

# 기존 UNetModel 클래스 내의 forward 함수만 수정합니다.

    def forward(self, x):
        # --- 입력 처리 부분 (변경 없음) ---
        B, C, H, W = x.shape
        x_emb = self.embedding(x)
        h = rearrange(x_emb, 'b c h w d -> b (c d) h w')
        h = self.init_conv(h)
        
        # --- Downsampling 경로 (변경 없음) ---
        skips = [h]
        for block_group in self.down_blocks:
            for layer in block_group:
                h = layer(h)
            skips.append(h)

        # --- Middle 경로 (변경 없음) ---
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        
        # --- Upsampling 경로 (아래 로직으로 교체) ---
        for block_group in self.up_blocks:
            # 먼저 스킵 연결 텐서를 가져옵니다.
            s = skips.pop()
            
            # h와 스킵 연결 s를 채널 차원으로 합칩니다.
            # 이 시점에서 h와 s는 공간적 크기(H, W)가 같아야 합니다.
            h = torch.cat([h, s], dim=1)
            
            # 해당 레벨의 모든 레이어(ResBlocks, AttnBlock, Upsampler)를 순서대로 통과시킵니다.
            # block_group에 Upsample 레이어가 있다면, 가장 마지막에 실행되어 h의 크기를 키웁니다.
            for layer in block_group:
                h = layer(h)

        # --- Output 경로 (변경 없음) ---
        h = self.out_norm(h)
        h = F.silu(h)
        logits_4d = self.out_conv(h)
        logits_bsv = rearrange(logits_4d, 'b (c v) h w -> b (c h w) v', c=C)
        
        return logits_bsv
if __name__ == '__main__':
    # --- Sanity Check ---
    paper_params = {
        "num_classes": 257, "channels": 96, "depth": 5, "channel_mults": (1, 2, 4, 4), 
        "num_heads": 4, "head_channels": 64, "attention_resolutions": (16,), "dropout": 0.4,
    }
    model = UNetModel(**paper_params).cuda()
    dummy_input = torch.randint(0, 257, (2, 3, 32, 32)).cuda()
    output = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    assert output.shape == (2, 3072, 257)
    print("Sanity check passed!")
