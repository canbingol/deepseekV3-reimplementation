import torch
import torch.nn as nn
import math 


class RMSNorm(nn.Module):

    def __init__(self,dim:int, eps:float=1e-6):
        super().__init__()
        self.eps = eps
        self.gama = nn.Parameter(torch.ones(dim))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        normalized_x = x * torch.rsqrt(x ** 2 + self.eps) * self.gama
        return normalized_x
    

def precompute_freq_cis(cfg):
    assert cfg.dim % cfg.n_head == 0, f'dim({cfg.dim}) should be divisibl by n_head ({cfg.n_head})'
    dim = cfg.dim // cfg.n_head 
    base = cfg.base
    beta_slow = cfg.beta_slow
    beta_fast = cfg.beta_fast
    max_seq_len = 8 #cfg.max_seq_len
    rope_factor = cfg.rope_factor
    
    def find_correct_dim(num_rotate,dim,base,max_seq_len):
        return dim * math.log(max_seq_len / (2 * math.pi *  num_rotate)) / (2 * math.log(base))
    
    def find_correct_range(low_rot,high_rot,dim,base,max_seq_len):
        low = math.floor(find_correct_dim(low_rot,dim,base,max_seq_len))
        high = math.ceil(find_correct_dim(low_rot,dim,base,max_seq_len))
        return max(0, high), min(high,dim-1)
    
    def linear_ramp(min,max,dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim,dtype=torch.float32)- min) / (max - min)
        ramp_func = torch.clamp(linear_func,0,1)
        return ramp_func
    
    freqs = 1.0/ (base ** (torch.arange(0,dim,2) / dim))

    if max_seq_len > cfg.original_seq_len:
        low, high = find_correct_range(beta_slow,beta_fast,dim,base,max_seq_len)
        smooth = 1 - linear_ramp(low,high ,dim // 2)
        freqs = freqs / rope_factor * (1 - smooth) + freqs * smooth
    
    pos = torch.arange(max_seq_len)
    freqs = torch.outer(pos, freqs)
    freqs = torch.polar(torch.ones_like(freqs),freqs)
    return freqs


def apply_rope(x:torch.Tensor, freqs_cis:torch.Tensor):
    print(f'x shape {x.shape}')
    assert x.shape[-1] % 2 == 0, "Rotary dim must be divisible by 2!"
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).reshape(*x.shape[:-1], -1)
    print(f'y.shape {y.shape}')
    return y.to(dtype)