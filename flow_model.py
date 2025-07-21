"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # if not self.flash:
        #     print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        #     # causal mask to ensure that attention is only applied to the left in the input sequence
        #     self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                                 .view(1, 1, config.block_size, config.block_size))


        if config.qk_layernorm:
            self.q_layernorm = LayerNorm(config.n_embd // self.n_head, bias=config.bias)
            self.k_layernorm = LayerNorm(config.n_embd // self.n_head, bias=config.bias)

        self.qk_layernorm = config.qk_layernorm

    def forward(self, x, attn_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # (b, t, embed) -> (b, t, 3 * embed)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.qk_layernorm:
            q = self.q_layernorm(q)
            k = self.k_layernorm(k)

        final_attn_mask = None
        if attn_mask is not None:
            attn_mask = attn_mask.to(torch.bool)
            valid_mask_2d = attn_mask.unsqueeze(1) & attn_mask.unsqueeze(2) # (B, T, T)
            valid_mask_full = valid_mask_2d.unsqueeze(1).expand(B, self.n_head, T, T) # (B, H, T, T)

            final_attn_mask = torch.zeros_like(valid_mask_full, dtype=x.dtype)
            final_attn_mask = final_attn_mask.masked_fill_(~valid_mask_full, float('-inf'))
            final_attn_mask = final_attn_mask.to(torch.bool).to(torch.float)
            
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=final_attn_mask,
                dropout_p=self.dropout if self.training else 0)
            

        else:
            raise NotImplementedError
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # assumes timesteps is in the range 0 to 1000

    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    qk_layernorm: bool = False
    do_x1_sc: bool = False
    mask_token_id: int = 0
    proper_timestep_emb: bool = False
    d3pm_loss_weighting: bool = False
    d3pm_loss_weighting_maxT: int = 1000
    # 좌표 임베딩 관련 설정
    use_coordinate_embedding: bool = False  # x, y 좌표를 따로 임베딩할지 여부
    eos_token_id: int = 256  # end of sequence token id
    mask_token_id: int = 257
    pad_token_id: int = 258
class GPT(nn.Module):

    def __init__(self, config):
        """
        config.vocab_size should include a mask token 
        """
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config


        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # 좌표별 임베딩 설정
        if config.use_coordinate_embedding:
            # print('config.n_embd // 2', config.n_embd // 2)
            self.x_embedding = nn.Embedding(config.vocab_size, (config.n_embd // 2), padding_idx=config.pad_token_id) # include eos, pad tokens, pad tokens will masks in attention process
            self.y_embedding = nn.Embedding(config.vocab_size, (config.n_embd // 2), padding_idx=config.pad_token_id)
            self.coord_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # include padding token
        self.x_output_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.y_output_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying


        if config.do_x1_sc:
            self.xt_x1_proj = nn.Linear(2 * config.n_embd, config.n_embd, bias=config.bias)

        # init all weights
        self.apply(self._init_weights)
        
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def _get_coordinate_embedding(self, idx):
        """
        좌표 데이터를 x, y로 분리하여 각각 임베딩 처리
        idx: (batch_size, seq_len) - [x1, x2, y1, y2, eos, eos, pad, pad, pad, pad] 형태
        """
        # idx = self._coordinate_transform(idx)

        b, t = idx.size()
        coord_len = t // 2
  
        x_coords = idx[:, :coord_len]
        y_coords = idx[:, coord_len:]
        
        x_emb = self.x_embedding(x_coords) # (b, coord_len, n_embd // 2)
        y_emb = self.y_embedding(y_coords) # (b, coord_len, n_embd // 2)

        tok_emb = torch.cat([x_emb, y_emb], dim=-1) # (b, coord_len, n_embd)
        return tok_emb  
    

    def _run_net(self, idx, time, x1=None, attn_mask=None):
        device = idx.device
        b, t = idx.size()

        n_embd = self.config.n_embd
        pos = torch.arange(0, t // 2, dtype=torch.long, device=device) # shape (t)
        
        if self.config.use_coordinate_embedding:    
            tok_emb = self._get_coordinate_embedding(idx) # (b, t // 2, n_embd)
        else:
            tok_emb = self.transformer.wte(idx)
            

        pos_emb = self.transformer.wpe(pos) # (t//2,) -> position embeddings of shape (t // 2, n_embd)
        
        if self.config.proper_timestep_emb:
            time_emb = transformer_timestep_embedding(time * 1000, n_embd)
        else:
            time_emb = transformer_timestep_embedding(time, n_embd)
        assert time_emb.shape == (b, n_embd)
        
        # (b, t//2, n_embd * 2) + (1, t//2, n_embd * 2) + (b, 1, n_embd * 2)
        res = tok_emb.view(b, t // 2, n_embd) + pos_emb.view(1, t // 2, n_embd) + time_emb.view(b, 1, n_embd)

        x = self.transformer.drop(res)
        for i, block in enumerate(self.transformer.h):
            x = block(x, attn_mask=attn_mask[:, :t//2])
        x = self.transformer.ln_f(x)
        
        x_head_logits = self.x_output_head(x)  # (b, t // 2, vocab_size)
        y_head_logits = self.y_output_head(x)  # (b, t // 2, vocab_size)
        logits = torch.cat([x_head_logits, y_head_logits], dim=1) # (b, t, vocab_size * 2)

        return logits


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

