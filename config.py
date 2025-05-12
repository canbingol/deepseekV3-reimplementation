from dataclasses import dataclass

@dataclass
class ModelConfig:
    dim:int = 1024
    base:float = 10_000
    n_head:int = 8
    original_seq_len:int=256
    beta_fast:int = 32
    beta_slow:int = 1
    max_seq_len:int = 512
    rope_factor:float = 0.7
    kv_down_dim:int= 576
    max_batch_size:int = 16


