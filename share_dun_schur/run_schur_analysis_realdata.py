"""
Layerwise Schur complement analysis for GPT-3 Large (24L, 1536d, 16 heads)
with residual_gelu query mode.

Usage:
    python run_schur_analysis_gpt3_large.py --checkpoint path/to/checkpoint.pt --output results.csv

Dependencies:
    pip install torch numpy scikit-learn

The script uses synthetic random data (no need to download WikiText-2).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
import argparse
import gc
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schur_stats import schur_complement_stats, permutation_null


# =============================================================================
# Model definition (nanoGPT-based, matching Jiwon's GPT-3 Large config)
# =============================================================================

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.num_heads == 0
        self.config = config

        if self.config.query_mode in ["identity", "residual_gelu"]:
            # Only 2 * n_embd (K, V) — Q comes from the residual MLP or identity
            self.c_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        elif self.config.query_mode in ["residual", "original"]:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        if self.config.query_mode == "residual_gelu":
            self.residual_rmsnorm_input = nn.RMSNorm(config.n_embd, eps=1e-5)
            self.residual_layernorm_output = LayerNorm(config.n_embd, bias=config.bias)
            self.residual_in = nn.Linear(config.n_embd, config.n_embd // 2, bias=config.bias)
            self.residual_out = nn.Linear(config.n_embd // 2, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.num_heads = config.num_heads
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.config = config

        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        if self.config.query_mode in ["identity", "residual_gelu"]:
            k, v = self.c_attn(x).split(self.n_embd, dim=2)
            if self.config.query_mode == "identity":
                q = x
            elif self.config.query_mode == "residual_gelu":
                q = x + self.residual_layernorm_output(
                    self.residual_out(F.gelu(self.residual_in(self.residual_rmsnorm_input(x))))
                )
        else:
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            if self.config.query_mode == "residual":
                q = x + q

        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=True, scale=self.config.scale
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.mlp_hidden_size, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.mlp_hidden_size, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.normalizing_coef = 1.0

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x * self.normalizing_coef


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTConfig:
    def __init__(self, **kwargs):
        # GPT-3 Large defaults (24L, 1536d, 16 heads, d_head=96)
        self.block_size = 1024
        self.vocab_size = 50304
        self.n_layer = 24
        self.num_heads = 16
        self.n_embd = 1536
        self.head_size = 96
        self.mlp_hidden_size = 1536 * 4
        self.tie_weights = True
        self.query_mode = "residual_gelu"
        self.dropout = 0.0
        self.bias = False
        self.batch_size = 12
        self.accumulation_size = 40
        self.scale = 1 / (2 * math.sqrt(1536 // 16))
        self.learning_rate = 2.55e-3
        self.min_lr = 2e-5
        self.weight_decay = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0
        self.decay_lr = True
        self.warmup_iters = 2000
        self.lr_decay_iters = 60002
        self.save_checkpoint_steps = None
        self.max_iters = 60002
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_weights:
            self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"Number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
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

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss


# =============================================================================
# Feature extraction hooks
# =============================================================================

def extract_model_features(model, layer_idx, data_loader, n_embd, max_batches=5, device="cuda"):
    """Extract Q, K, V activations from a specific layer."""
    q_store, k_store, v_store = [], [], []
    attn_module = model.transformer.h[layer_idx].attn

    def attn_hook(_m, _i, out):
        x = _i[0]
        if attn_module.config.query_mode in ["identity", "residual_gelu"]:
            k, v = out.split(n_embd, dim=2)
            if attn_module.config.query_mode == "identity":
                q = x
            else:  # residual_gelu
                q = x + attn_module.residual_layernorm_output(
                    attn_module.residual_out(
                        F.gelu(attn_module.residual_in(attn_module.residual_rmsnorm_input(x)))
                    )
                )
        else:
            q, k, v = out.split(n_embd, dim=2)
            if attn_module.config.query_mode == "residual":
                q = x + q

        q_store.append(q.detach().reshape(-1, n_embd).cpu())
        k_store.append(k.detach().reshape(-1, n_embd).cpu())
        v_store.append(v.detach().reshape(-1, n_embd).cpu())

    h = attn_module.c_attn.register_forward_hook(attn_hook)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if max_batches is not None and i >= max_batches:
                break
            x = batch.to(device)
            model(x)

    h.remove()

    f_arr = torch.cat(q_store).numpy()  # Q = the nonlinear query
    p_arr = torch.cat(k_store).numpy()  # K = always linear
    q_arr = torch.cat(v_store).numpy()  # V = always linear
    return f_arr, p_arr, q_arr


# =============================================================================
# Schur complement statistics: imported from shared module schur_stats.py
# =============================================================================

# =============================================================================
# Synthetic data loader (no need for WikiText-2)
# =============================================================================

def make_synthetic_loader(batch_size=4, seq_len=1024, vocab_size=50304, num_batches=10,
                         bin_path="data/openwebtext/val.bin", seed=0):
    """REAL DATA: sample real token windows from OWT val.bin (not random)."""
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num_batches):
        ix = rng.integers(0, len(data) - seq_len, size=batch_size)
        batch = np.stack([data[i:i+seq_len].astype(np.int64) for i in ix])
        out.append(torch.from_numpy(batch))
    return out


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the checkpoint .pt file")
    parser.add_argument("--output", type=str, default="schur_results.csv",
                        help="Output CSV path")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--max_batches", type=int, default=5,
                        help="Number of batches to run")
    parser.add_argument("--n_perm", type=int, default=3,
                        help="Number of permutations for null baseline")
    parser.add_argument("--max_rows", type=int, default=20000,
                        help="Max rows for stats computation")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load checkpoint ---
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint if not isinstance(checkpoint, dict) else checkpoint.get('model', checkpoint)

    # --- Build model ---
    config = GPTConfig()
    model = GPT(config).to(device)
    print(f"Model: {config.n_layer}L, {config.n_embd}d, {config.num_heads} heads")

    # Clean state dict keys (handle _orig_mod. prefix and transformer. prefix)
    cleaned = {}
    for k, v in state_dict.items():
        k2 = k
        if k2.startswith("_orig_mod."):
            k2 = k2[len("_orig_mod."):]
        if not k2.startswith("transformer.") and k2 != "lm_head.weight":
            k2 = "transformer." + k2
        cleaned[k2] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    print("  Checkpoint loaded.")

    # --- Data ---
    data_loader = make_synthetic_loader(
        batch_size=4, seq_len=1024, vocab_size=config.vocab_size,
        num_batches=args.max_batches
    )
    print(f"Data: {args.max_batches} batches of random tokens")

    # --- Per-layer analysis ---
    n_layers = config.n_layer
    rng = np.random.default_rng(0)
    all_rows = []

    for L in range(n_layers):
        print(f"\nLayer {L}/{n_layers - 1}...", end=" ", flush=True)

        F_arr, P_arr, Q_arr = extract_model_features(
            model, L, data_loader, config.n_embd, max_batches=args.max_batches, device=device
        )
        B_arr = np.concatenate([P_arr, Q_arr], axis=1)
        del P_arr, Q_arr
        gc.collect()

        # Subsample if needed
        if F_arr.shape[0] > args.max_rows:
            keep = rng.choice(F_arr.shape[0], size=args.max_rows, replace=False)
            F_eval, B_eval = F_arr[keep], B_arr[keep]
        else:
            F_eval, B_eval = F_arr, B_arr

        stats = schur_complement_stats(F_eval, B_eval)
        null = permutation_null(F_eval, B_eval, n_perm=args.n_perm)

        row = {
            "layer": L,
            "nu_eff": stats["nu_eff"],
            "nu_eff_null_mean": null["nu_eff_null_mean"],
            "nu_eff_null_std": null["nu_eff_null_std"],
            "nu_frac": stats["nu_frac"],
            "nu_frac_null_mean": null["nu_frac_null_mean"],
            "nu_frac_null_std": null["nu_frac_null_std"],
            "ridge_lambda": stats["ridge_lambda"],
            "cond_Sigma_bb": stats["cond_Sigma_bb"],
            "rank_Rf_holdout@1e-6": stats["rank_Rf_holdout@1e-06"],
        }
        all_rows.append(row)
        print(f"nu_frac={stats['nu_frac']:.6f}")

        del F_arr, B_arr, F_eval, B_eval, stats, null
        gc.collect()

    # --- Save ---
    import csv
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nResults saved to: {args.output}")
    print("Done!")


if __name__ == "__main__":
    main()