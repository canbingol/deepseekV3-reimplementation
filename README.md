# DeepSeekV3 Reimplementation (MLA + RoPE)

This repository contains a minimal reimplementation of the **Multi-Latent Attention (MLA)** mechanism and **Rotary Positional Embeddings (RoPE)** based on the DeepSeekV3 architecture.

## Components
**Note:** Other parts will be added over time.
- `MLA.py`: Implements the `MultiLatentAttention` class using latent key/value projections and rotary positional encodings.
- `utils.py`: Includes `RMSNorm`, RoPE-related functions (`apply_rope`, `precompute_freq_cis`).
- `config.py`: Stores model hyperparameters using a `dataclass`.

##  How to Run (Currently only MLA can be run)

Run the following inside `MLA.py` to test the attention module:

```bash
python MLA.py
