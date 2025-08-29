import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange # einops가 필요합니다. pip install einops
import torchvision.transforms as transforms

# --- Helper Modules (ResBlock, UpConv, Attention2D)은 변경 없음 ---
class ResBlock(nn.Module):
    # ... (이전과 동일)
    def __init__(self, in_ch, out_ch, dropout=0.4):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        if in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        else:
            self.res_conv = None
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        res = x if self.res_conv is None else self.res_conv(x)
        return out + res

class UpConv(nn.Module):
    # ... (이전과 동일)
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.up(x)

class Attention2D(nn.Module):
    # ... (이전과 동일)
    def __init__(self, channels, head_channels=64, dropout=0.4):
        super(Attention2D, self).__init__()
        num_heads = max(1, channels // head_channels)
        num_heads = min(num_heads, channels)
        if channels % num_heads != 0:
            num_heads = 1
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.proj = nn.Linear(channels, channels)
    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.view(B, C, H*W).permute(0, 2, 1)
        attn_out, _ = self.attn(seq, seq, seq)
        attn_out = self.proj(attn_out)
        out = self.norm(seq + attn_out)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out

class AttentionUNet(nn.Module):
    def __init__(self, img_ch=3, vocab_size=257, embedding_dim=96, base_ch=96, channel_mults=(1,2,4,4,8), depth=5, head_channels=64, attention_resolutions=(16,), dropout=0.4):
        super(AttentionUNet, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        chs = [base_ch * m for m in channel_mults]
        assert len(chs) >= depth, "channel_mults must have at least 'depth' elements"
        chs = chs[:depth]

        in_ch0 = img_ch * self.embedding_dim
        self.enc_blocks = nn.ModuleList()
        prev_ch = in_ch0
        for i in range(depth):
            out_ch = chs[i]
            self.enc_blocks.append(nn.Sequential(
                ResBlock(prev_ch, out_ch, dropout=dropout),
                ResBlock(out_ch, out_ch, dropout=dropout)
            ))
            prev_ch = out_ch

        res_sizes = [32 // (2**i) for i in range(depth)]
        self.attentions = nn.ModuleDict()
        for i, sz in enumerate(res_sizes):
            if sz in attention_resolutions:
                self.attentions[str(i)] = Attention2D(chs[i], head_channels=head_channels, dropout=dropout)

        self.bottleneck = nn.Sequential(
            ResBlock(chs[-1], chs[-1]*2, dropout=dropout),
            ResBlock(chs[-1]*2, chs[-1], dropout=dropout)
        )

        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        # [수정] Decoder 블록 구성 시 인코더의 채널 구성을 역순으로 사용합니다.
        for i in reversed(range(depth)):
            # 현재 레벨의 채널 (Decoder의 출력 채널)
            out_ch = chs[i]
            # 이전(더 깊은) 레벨의 채널 (Decoder의 입력 채널)
            in_ch = chs[i+1] if i < depth - 1 else chs[-1]
            
            # 마지막 레벨(가장 해상도 높은)을 제외하고 UpConv 추가
            if i < depth - 1:
                self.up_convs.append(UpConv(in_ch, out_ch))
                # Decoder 블록은 (업샘플링된 채널 + 스킵연결 채널) -> 출력 채널
                self.dec_blocks.append(nn.Sequential(
                    ResBlock(out_ch * 2, out_ch, dropout=dropout),
                    ResBlock(out_ch, out_ch, dropout=dropout)
                ))
        
        self.final_conv = nn.Conv2d(chs[0], self.vocab_size * img_ch, kernel_size=1)
        
        # 가중치 초기화 함수를 호출합니다.
        self.apply(self._init_weights)
        print("✅ AttentionUNet weights initialized.")

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x_emb = self.embedding(x)
        x_in = rearrange(x_emb, 'b c h w d -> b (c d) h w')

        skips = []
        out = x_in
        for i, block in enumerate(self.enc_blocks):
            out = block(out)
            if str(i) in self.attentions:
                out = self.attentions[str(i)](out)
            skips.append(out)
            if i < len(self.enc_blocks) - 1:
                out = self.MaxPool(out)

        out = self.bottleneck(out)

        # [수정] Decoder 로직을 재구성합니다.
        for i, (up_conv, dec_block) in enumerate(zip(self.up_convs, self.dec_blocks)):
            out = up_conv(out)
            # 스킵 연결 텐서를 역순으로 가져옵니다.
            skip = skips[-(i + 2)] # skips 리스트의 끝에서부터 두 번째 요소부터 사용

            # [BUG FIX] 크기가 다를 경우, skip 텐서를 out 텐서 크기에 맞춰 잘라냅니다.
            if out.shape[2:] != skip.shape[2:]:
                target_h, target_w = out.shape[2], out.shape[3]
                skip = transforms.functional.center_crop(skip, [target_h, target_w])
            
            out = torch.cat([skip, out], dim=1)
            out = dec_block(out)

        logits = self.final_conv(out)
        logits = rearrange(logits, 'b (c v) h w -> b (c h w) v', c=C, v=self.vocab_size)
        return logits