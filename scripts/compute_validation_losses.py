"""
Compute validation losses for all 34 nanoGPT checkpoints on WikiText-2.

This is independent of the Schur analysis — it just runs inference and records
the cross-entropy loss so we can correlate nu_frac with actual LM performance.

Usage:
    python compute_validation_losses.py

Output:
    checkpoint/validation_losses.csv  —  config_name, val_loss, query_mode, mlp_mult

Dependencies:
    pip install torch numpy datasets transformers
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
import gc
import csv
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Model definition (nanoGPT-based, matching the 34 checkpoints)
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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss


# =============================================================================
# Config table (same as beyondlinearity_sweep)
# =============================================================================

CONFIGS = [
    # name,                                                     query_mode,       mlp_mult
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
    ("tiedw_residual_gelu_2-5wd_2e-3lr_3e-5",                  "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-5wd_3_2e-3lr_3e-5",                "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-5wd_3_4e-3lr_3e-5",                "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-5wd_3e-3lr_3e-5",                  "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-6wd_1_5e-3lr_4e-5",                "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-6wd_1_5e-3lr_8e-5",                "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-6wd_2e-3lr_1_2e-4",                "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-6wd_2e-3lr_2e-5",                  "residual_gelu",   4.0),
    ("tiedw_residual_gelu_2-6wd_2e-3lr_8e-5",                  "residual_gelu",   4.0),
]


def infer_mlp_hidden_size(state_dict, n_embd=768, default=4*768):
    """Read mlp_hidden_size directly off the checkpoint's own weights."""
    for k, v in state_dict.items():
        if k.endswith("mlp.c_fc.weight"):
            return v.shape[0]
    print(f"  !! could not find mlp.c_fc.weight, falling back to default={default}")
    return default


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
    # residual_gelu uses HALF scale: 1/(2*sqrt(d_head)) instead of 1/sqrt(d_head)
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

    # Clean state dict keys
    unwanted_prefix = "_orig_mod."
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(unwanted_prefix):
            k = k[len(unwanted_prefix):]
        elif ("." + unwanted_prefix) in k:
            k = k.replace(unwanted_prefix, "")
        cleaned_state_dict[k] = v

    new_state_dict = {}
    for k, v in cleaned_state_dict.items():
        if k == 'lm_head.weight' or k.startswith('lm_head.'):
            new_key = k
        elif k.startswith('transformer.'):
            new_key = k
        else:
            new_key = 'transformer.' + k
        new_state_dict[new_key] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing or unexpected:
        print(f"  [{name}] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 0 and len(missing) == len(unexpected):
            raise RuntimeError(f"[{name}] weights NOT loaded (key prefix mismatch)")

    return model


@torch.no_grad()
def compute_val_loss(model, data_loader, device, max_batches=20):
    """Run the model on validation data and return average cross-entropy loss.

    Uses standard autoregressive LM setup:
      - inputs:  x  = tokens[:, :-1]  (first T-1 tokens)
      - targets: y  = tokens[:, 1:]   (shifted right by 1)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(data_loader):
        if max_batches is not None and i >= max_batches:
            break
        tokens = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch["input_ids"].to(device)
        x = tokens[:, :-1]   # input: all but last token
        y = tokens[:, 1:]    # target: all but first token (predict next token)
        _, loss = model(x, targets=y)
        n_tokens = y.numel()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / total_tokens


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    CKPT_DIR = r"D:\modification research\checkpoint"
    OUTPUT_PATH = r"D:\modification research\checkpoint\validation_losses.csv"

    # --- Data ---
    print("\nLoading WikiText-2...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Concatenate all text into one long sequence, then chunk into non-overlapping blocks
    full_text = "\n\n".join([t for t in raw_dataset["text"] if t.strip()])
    tokens = tokenizer(full_text, return_tensors="pt", truncation=False).input_ids.squeeze(0)

    seq_len = 1024
    num_samples = len(tokens) // seq_len
    tokens = tokens[: num_samples * seq_len].view(num_samples, seq_len)

    dataset = TensorDataset(tokens)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    print(f"  {num_samples} sequences loaded")

    # --- Run inference on each checkpoint ---
    results = []

    for i, (name, query_mode, mlp_mult) in enumerate(CONFIGS):
        print(f"\n[{i+1}/{len(CONFIGS)}] {name} (query_mode={query_mode})")

        model = load_model(name, query_mode, mlp_mult, CKPT_DIR, device)
        if model is None:
            continue

        loss = compute_val_loss(model, val_loader, device, max_batches=20)
        print(f"  val_loss = {loss:.4f}")

        results.append({
            "config": name,
            "query_mode": query_mode,
            "mlp_mult": mlp_mult,
            "val_loss": loss,
        })

        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # --- Save ---
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "query_mode", "mlp_mult", "val_loss"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} results to {OUTPUT_PATH}")

    # --- Summary ---
    print("\n=== Summary ===")
    for r in results:
        print(f"  {r['config']:50s}  {r['query_mode']:15s}  loss={r['val_loss']:.4f}")


if __name__ == "__main__":
    main()