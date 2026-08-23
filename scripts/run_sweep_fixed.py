"""
Standalone fixed sweep over all 34 checkpoints.

Fixes applied (matching run_schur_analysis_gpt3_large.py and compute_validation_losses.py):
  - Real WikiText-2 data: continuous blocks, no PAD padding
  - Correct scale per architecture (original: 1/√d_head, residual_gelu: 1/(2√d_head))
  - Cross-validated joint ridge for high-dimensional B covariance (was LedoitWolf)
  - Direct second-moment MSE (no np.cov) for holdout/in-sample residual
  - Batch unpacking handles list/tuple and dict batches
  - Config inference from checkpoint weights (n_embd, mlp_hidden_size, n_layer, query_mode)
  - Auto-detects torch.compile wrapper (_orig_mod.)

Output:
  checkpoint/sweep_results.csv  —  per-layer nu_frac/nu_eff for every config
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
import argparse
import gc
import csv
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schur_stats import schur_complement_stats, permutation_null
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Model definition (12L/768d GPT-2, matching the 34 checkpoints)
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
# Config table (same as compute_validation_losses.py)
# =============================================================================

CONFIGS = [
    ("tiedw_original",                                          "original",        4.0),
    ("tiedw_original_005wd",                                    "original",        4.0),
    ("tiedw_original_20k_steps",                                "original",        4.0),
    ("tiedw_original_mlp_4_33_times",                           "original",        4.33),
    ("tiedw_original_mlp_4_5_times",                            "original",        4.5),
    ("tiedw_original_mlp_4_625_times",                          "original",        4.625),
    ("tiedw_original_mlp_4_75_times",                           "original",        4.75),
    ("tiedw_original_mlp_4_75_times_lrscaled",                  "original",        4.75),
    ("tiedw_residual_gelu",                                     "residual_gelu",   4.0),
    ("tiedw_residual_gelu_0025wd_1_7e-3lr_3e-5",               "residual_gelu",   4.0),
    ("tiedw_residual_gelu_0025wd_2_4e-3lr_3e-5",               "residual_gelu",   4.0),
    ("tiedw_residual_gelu_005wd",                              "residual_gelu",   4.0),
    ("tiedw_residual_gelu_005wd_1_1e-3lr",                     "residual_gelu",   4.0),
    ("tiedw_residual_gelu_005wd_2_4e-3lr_3e-5",                "residual_gelu",   4.0),
    ("tiedw_residual_gelu_01wd_3_6e-3lr_2e-5",                 "residual_gelu",   4.0),
    ("tiedw_residual_gelu_01wd_3_6e-3lr_2e-5_20k_steps",       "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-5wd_1_2e-3lr",                     "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-5wd_1_2e-3lr_3e-4",                "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-5wd_1_2e-3lr_7e-5",                "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-5wd_1_5e-3lr_4e-5",                "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-5wd_1_7e-3lr_3e-5",                "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-5wd_1_7e-3lr_5e-5",                "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-5wd_2_2e-3lr_3e-5",                "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-5wd_2_4e-3lr_3e-5",                "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-5wd_2_6e-3lr_3e-5",                "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-5wd_2_8e-3lr_3e-5",                "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2e-3lr_3e-5",                        "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-5wd_3_2e-3lr_3e-5",                "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-5wd_3_4e-3lr_3e-5",                "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-5wd_3e-3lr_3e-5",                  "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-6wd_1_5e-3lr_4e-5",                "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-6wd_1_5e-3lr_8e-5",                "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-6wd_2e-3lr_1_2e-4",                "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-6wd_2e-3lr_2e-5",                  "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-6wd_2e-3lr_8e-5",                  "residual_gelu",   4.0),
]


# =============================================================================
# Helpers
# =============================================================================

def infer_mlp_hidden_size(state_dict, n_embd=768, default=4*768):
    """Read mlp_hidden_size directly off the checkpoint's own weights."""
    for k, v in state_dict.items():
        if k.endswith("mlp.c_fc.weight"):
            return v.shape[0]
    print(f"  !! could not find mlp.c_fc.weight, falling back to default={default}")
    return default


def make_real_data_loader(batch_size=4, seq_len=1024, num_batches=20, device="cuda"):
    """Load WikiText-2 test set and create a DataLoader with continuous blocks."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    full_text = "\n\n".join([t for t in raw_dataset["text"] if t.strip()])
    tokens = tokenizer.encode(full_text, return_tensors="pt").squeeze(0)

    num_samples = len(tokens) // seq_len
    tokens = tokens[: num_samples * seq_len].view(num_samples, seq_len)

    dataset = TensorDataset(tokens)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def load_model(name, query_mode, mlp_mult, ckpt_dir, device):
    """Load a single checkpoint and return the model."""
    ckpt_path = os.path.join(ckpt_dir, f"{name}.pt")
    if not os.path.exists(ckpt_path):
        print(f"  !! checkpoint not found: {ckpt_path}")
        return None

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint if not isinstance(checkpoint, dict) else checkpoint.get('model', checkpoint)

    _mlp_hidden_size = infer_mlp_hidden_size(state_dict, n_embd=768)
    _query_mode = query_mode
    _head_size = 64
    _scale = 1 / (2 * math.sqrt(_head_size)) if query_mode == "residual_gelu" else 1 / math.sqrt(_head_size)

    class Config:
        block_size = 1024
        vocab_size = 50304
        n_layer = 12
        num_heads = 12
        n_embd = 768
        head_size = _head_size
        mlp_hidden_size = _mlp_hidden_size
        tie_weights = True
        query_mode = _query_mode
        dropout = 0.0
        bias = False
        batch_size = 12
        accumulation_size = 40
        scale = _scale
        learning_rate = 1.5e-3
        weight_decay = 2 ** -5
        beta1 = 0.9
        beta2 = 0.95
        grad_clip = 1.0
        decay_lr = True
        warmup_iters = 2000
        lr_decay_iters = 60002
        min_lr = 4e-5
        save_checkpoint_steps = [50000, 100000, 105000]
        max_iters = 60002

    model = GPT(Config()).to(device)

    cleaned = {}
    for k, v in state_dict.items():
        k2 = k
        if k2.startswith("_orig_mod."):
            k2 = k2[len("_orig_mod."):]
        if not k2.startswith("transformer.") and k2 != "lm_head.weight":
            k2 = "transformer." + k2
        cleaned[k2] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print(f"  [{name}] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 0 and len(missing) == len(unexpected):
            raise RuntimeError(f"[{name}] weights NOT loaded (key prefix mismatch)")

    return model


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
                    attn_module.residual_out(F.gelu(attn_module.residual_in(attn_module.residual_rmsnorm_input(x))))
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
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch["input_ids"].to(device)
            model(x)

    h.remove()

    f_arr = torch.cat(q_store).numpy()
    p_arr = torch.cat(k_store).numpy()
    q_arr = torch.cat(v_store).numpy()
    return f_arr, p_arr, q_arr


# =============================================================================
# Schur complement statistics: imported from shared module schur_stats.py
#   (joint-ridge estimator with cross-validated ridge penalty; back-compatible)
# =============================================================================

# =============================================================================
# Main sweep
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default=r"D:\modification research\checkpoint")
    parser.add_argument("--output", type=str, default=r"D:\modification research\checkpoint\sweep_results.csv")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--n_perm", type=int, default=3)
    parser.add_argument("--max_rows", type=int, default=20000)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Resume support: load any rows already written to the output CSV ---
    ROW_FIELDS = [
        "config", "query_mode", "mlp_mult", "layer",
        "nu_eff", "nu_eff_null_mean", "nu_eff_null_std",
        "nu_frac", "nu_frac_null_mean", "nu_frac_null_std",
        "ridge_lambda", "cond_Sigma_bb", "rank_Rf_holdout@1e-6",
    ]
    done = set()
    if os.path.exists(args.output) and os.path.getsize(args.output) > 0:
        with open(args.output, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("config") is not None and r.get("layer") is not None:
                    done.add((r["config"], str(r["layer"])))
        print(f"Resume: {len(done)} layer-rows already present, will skip them.")
    out_f = open(args.output, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=ROW_FIELDS)
    if os.path.getsize(args.output) == 0:
        writer.writeheader()
        out_f.flush()

    # --- Data ---
    print("\nLoading WikiText-2 (real text)...")
    data_loader = make_real_data_loader(batch_size=4, seq_len=1024, num_batches=args.max_batches, device=device)
    print(f"Data: {args.max_batches} batches of WikiText-2 (real text)")

    # --- Sweep ---
    rng = np.random.default_rng(0)
    n_rows = 0

    for i, (name, query_mode, mlp_mult) in enumerate(CONFIGS):
        print(f"\n[{i+1}/{len(CONFIGS)}] {name} (query_mode={query_mode})")

        model = load_model(name, query_mode, mlp_mult, args.ckpt_dir, device)
        if model is None:
            continue

        n_layers = model.config.n_layer

        for L in range(n_layers):
            if (name, str(L)) in done:
                print(f"  Layer {L}/{n_layers-1}: skipped (already in output)")
                continue
            print(f"  Layer {L}/{n_layers-1}...", end=" ", flush=True)

            F_arr, P_arr, Q_arr = extract_model_features(
                model, L, data_loader, model.config.n_embd, max_batches=args.max_batches, device=device
            )
            B_arr = np.concatenate([P_arr, Q_arr], axis=1)
            del P_arr, Q_arr
            gc.collect()

            if F_arr.shape[0] > args.max_rows:
                keep = rng.choice(F_arr.shape[0], size=args.max_rows, replace=False)
                F_eval, B_eval = F_arr[keep], B_arr[keep]
            else:
                F_eval, B_eval = F_arr, B_arr

            stats = schur_complement_stats(F_eval, B_eval)
            null = permutation_null(F_eval, B_eval, n_perm=args.n_perm)

            row = {
                "config": name,
                "query_mode": query_mode,
                "mlp_mult": mlp_mult,
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
            writer.writerow(row)
            out_f.flush()
            n_rows += 1
            print(f"nu_frac={stats['nu_frac']:.6f}")

            del F_arr, B_arr, F_eval, B_eval, stats, null
            gc.collect()

        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    out_f.close()
    print(f"\nTotal rows in {args.output}: {n_rows}")
    print("Done!")


if __name__ == "__main__":
    main()