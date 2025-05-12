import torch
import torch.nn as nn
from typing import Optional
from utils import apply_rope, precompute_freq_cis
from config import ModelConfig

class MultiLatentAttention(nn.Module):

    def __init__(self,cfg):
        super().__init__()
        self.dim = cfg.dim
        self.kv_down_dim = cfg.kv_down_dim
        self.n_heads = cfg.n_head
        self.head_dim = self.dim // self.n_heads
        self.kv_down_proj = nn.Linear(self.dim,self.kv_down_dim)

        self.Wq = nn.Linear(self.dim,self.head_dim * self.n_heads)
        self.W_uk = nn.Linear(self.kv_down_dim, self.head_dim * self.n_heads)
        self.W_uv = nn.Linear(self.kv_down_dim, self.head_dim * self.n_heads)

        self.Wo = nn.Linear(self.n_heads * self.head_dim, self.dim)
        self.softmax_scale = self.dim ** -0.5

        self.register_buffer('kv_cache',torch.zeros(cfg.max_batch_size,cfg.max_seq_len,cfg.kv_down_dim))
        self.register_buffer('rope_cache',torch.zeros(cfg.max_batch_size,cfg.max_seq_len, self.head_dim))

    def forward(self,x:torch.Tensor, start_pos:int, freqs_cis:torch.Tensor, mask:Optional[torch.Tensor]=None):
        batch_size, seq_len, _ = x.size()

        L_kv = self.kv_down_proj(x)

        q = self.Wq(x)
        k = self.W_uk(L_kv)
        v =self.W_uv(L_kv)
        print(f'q shape {q.shape}\n')
        q = q.reshape(batch_size,seq_len,self.n_heads,-1).transpose(1,2)
        print(f'q shape {q.shape}\n')
        k = k.reshape(batch_size,seq_len,self.n_heads,-1).transpose(1,2)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)
        
        attn_score = torch.softmax((q * k) * self.softmax_scale, dim=-1)

        if mask:
            attn_score = attn_score.masked_fill(mask)
        
        v = v.reshape(*v.shape[:-1],self.n_heads,self.head_dim).transpose(1,2)
        print(f'attn score   shape: {attn_score.shape}, v shape: {v.shape}')
        attn_weights = attn_score * v
        print(f'attn_weights shape: {attn_weights.shape}')
        print(f'self.wo shape: {self.Wo.weight.shape}')
        attn_weights = attn_weights.reshape(batch_size,seq_len,-1)
        out = self.Wo(attn_weights)
        print(f'out.shape: {out.shape}')
        return out
    
cfg = ModelConfig()
mla = MultiLatentAttention(cfg)
x = torch.rand(4,12,cfg.dim)
freq_cis = precompute_freq_cis(cfg)
x_ = mla(x,0,freq_cis)