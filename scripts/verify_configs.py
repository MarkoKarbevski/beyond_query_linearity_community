"""Verify that every checkpoint's model_args/config matches the CONFIGS table
used by the analysis scripts (run_sweep_fixed.py, compute_validation_losses.py,
beyondlinearity_sweep notebook), and that the GPT-3 Large checkpoint matches
run_schur_analysis_gpt3_large.py.

This is the ground-truth check for the claim that "the analysis model is created
and trained consistently with the configs": each checkpoint stores the exact
model_args it was built with, so the trained architecture is compared directly
against the configs the analysis assumes.

Usage:
    python scripts/verify_configs.py
    python scripts/verify_configs.py --ckpt-dir "D:/modification research/checkpoint"
    python scripts/verify_configs.py --gpt3-large "D:/modification research/gpt3large_residual_gelu_ckpt60000_modelonly.pt"

Exit code 0 = all checks pass, 1 = mismatch found.
"""

import argparse
import csv
import math
import os
import sys

import torch

# CONFIGS table -- MUST stay identical to run_sweep_fixed.py / compute_validation_losses.py
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

# Architecture fields every GPT-2 Medium checkpoint must carry.
EXPECTED_ARCH = {
    "block_size": 1024,
    "vocab_size": 50304,
    "n_layer": 12,
    "num_heads": 12,
    "n_embd": 768,
    "head_size": 64,
    "tie_weights": True,
    "dropout": 0.0,
    "bias": False,
}

GPT3_LARGE_EXPECTED = {
    "block_size": 1024,
    "vocab_size": 50304,
    "n_layer": 24,
    "num_heads": 16,
    "n_embd": 1536,
    "head_size": 96,
    "tie_weights": True,
    "query_mode": "residual_gelu",
    "mlp_hidden_size": 6144,
    "learning_rate": 2.55e-3,
    "weight_decay": 0.1,
    "max_iters": 60002,
}


def expected_scale(query_mode, n_embd, num_heads):
    d_head = n_embd // num_heads
    if query_mode == "residual_gelu":
        return 1.0 / (2.0 * math.sqrt(d_head))
    return 1.0 / math.sqrt(d_head)


def expected_mlp_hidden(mlp_mult, n_embd):
    # mlp_mult labels are approximate (e.g. "4.33" -> actual 3328 = 4.333x);
    # the caller allows a small tolerance around the nominal value.
    return int(round(mlp_mult * n_embd))


def parse_lr_wd(name, query_mode):
    """Best-effort parse of learning_rate / weight_decay / min_lr from the config
    name. Returns dict of (field -> expected value) or empty when the name does
    not encode the field."""
    out = {}
    tail = name.replace("tiedw_original_", "").replace("tiedw_residual_gelu_", "")
    if tail == "tiedw_original" or tail == "tiedw_residual_gelu":
        # base configs carry no hyperparameter suffix
        out["learning_rate"] = 0.0006 if query_mode == "original" else 0.0008
        out["weight_decay"] = 0.1
        out["min_lr"] = 6e-5 if query_mode == "original" else 8e-5
        return out
    # weight decay tokens
    if "2-5wd" in tail:
        out["weight_decay"] = 2 ** -5
    elif "2-6wd" in tail:
        out["weight_decay"] = 2 ** -6
    elif "005wd" in tail:
        out["weight_decay"] = 0.05
    elif "0025wd" in tail:
        out["weight_decay"] = 0.025
    elif "01wd" in tail:
        out["weight_decay"] = 0.1
    # learning-rate tokens: e.g. 1_7e-3lr -> 1.7e-3, 3e-3lr -> 3e-3, 2_4e-3lr -> 2.4e-3
    import re
    m = re.search(r"(\d+)(?:_(\d+))?e-3lr", tail)
    if m:
        whole = int(m.group(1))
        frac = int(m.group(2)) if m.group(2) else 0
        out["learning_rate"] = float(f"{whole}.{frac}e-3")
    # min-lr token at the end: _3e-5, _2e-5, _4e-5, _7e-5, _8e-5, _1_2e-4
    m = re.search(r"_(\d+)(?:_(\d+))?e-([45])$", tail)
    if m:
        whole = int(m.group(1))
        frac = int(m.group(2)) if m.group(2) else 0
        exp = int(m.group(3))
        val = float(f"{whole}.{frac}e-{exp}") if frac else float(f"{whole}e-{exp}")
        out["min_lr"] = val
    return out


def check_model_args(name, ma, mode, mlp_mult, errors, info, strict_hyperparams):
    """Compare checkpoint model_args against the CONFIGS entry. Appends failures
    to errors, informational mismatches to info."""
    label = f"{name}:"
    n_embd = ma.get("n_embd", EXPECTED_ARCH["n_embd"])
    n_heads = ma.get("num_heads", EXPECTED_ARCH["num_heads"])

    if ma.get("query_mode") != mode:
        errors.append(f"{label} query_mode: checkpoint={ma.get('query_mode')} expected={mode}")

    exp_scale = expected_scale(mode, n_embd, n_heads)
    if abs(ma.get("scale", -1.0) - exp_scale) > 1e-9:
        errors.append(f"{label} scale: checkpoint={ma.get('scale')} expected={exp_scale}")

    exp_mlp = expected_mlp_hidden(mlp_mult, n_embd)
    if abs(ma.get("mlp_hidden_size", -1) - exp_mlp) > 8:
        errors.append(
            f"{label} mlp_hidden_size: checkpoint={ma.get('mlp_hidden_size')} "
            f"expected~{exp_mlp} (mlp_mult={mlp_mult}, ratio={ma.get('mlp_hidden_size', 0) / n_embd:.4f})"
        )

    for field, exp in EXPECTED_ARCH.items():
        if ma.get(field) != exp:
            errors.append(f"{label} {field}: checkpoint={ma.get(field)} expected={exp}")

    # 20k-step configs must carry max_iters=20000
    if "20k" in name and ma.get("max_iters") != 20000:
        errors.append(f"{label} max_iters: checkpoint={ma.get('max_iters')} expected=20000")
    if "20k" not in name and ma.get("max_iters") != 60002:
        errors.append(f"{label} max_iters: checkpoint={ma.get('max_iters')} expected=60002")

    # training hyperparameters (informational unless --strict-hyperparams)
    expected = parse_lr_wd(name, mode)
    for field, exp in expected.items():
        actual = ma.get(field)
        ok = actual is not None and abs(actual - exp) < 1e-12
        if ok:
            info.append(f"{label} {field}={actual} OK")
        elif strict_hyperparams:
            errors.append(f"{label} {field}: checkpoint={actual} expected={exp} (from name)")
        else:
            info.append(f"{label} {field}: checkpoint={actual} expected={exp} (from name) -- NOTE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=r"D:\modification research\checkpoint")
    ap.add_argument("--gpt3-large", default=r"D:\modification research\gpt3large_residual_gelu_ckpt60000_modelonly.pt")
    ap.add_argument("--strict-hyperparams", action="store_true",
                    help="fail on training-hyperparameter mismatches parsed from config names")
    ap.add_argument("--sweep-csv", default=os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), "checkpoint", "sweep_results.csv"),
                    help="path to sweep_results.csv (for completeness check)")
    ap.add_argument("--val-loss-csv", default=os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), "checkpoint", "validation_losses.csv"),
                    help="path to validation_losses.csv (for completeness check)")
    args = ap.parse_args()

    errors, info = [], []

    print("=" * 78)
    print("Verify GPT-2 Medium sweep checkpoints vs CONFIGS table")
    print("=" * 78)
    missing = 0
    for name, mode, mlp_mult in CONFIGS:
        path = os.path.join(args.ckpt_dir, name + ".pt")
        if not os.path.exists(path):
            errors.append(f"MISSING checkpoint: {path}")
            missing += 1
            continue
        ck = torch.load(path, map_location="cpu", weights_only=True)
        ma = ck.get("model_args", {})
        if not isinstance(ma, dict):
            errors.append(f"{name}: no model_args in checkpoint (top-level keys={list(ck.keys())})")
            continue
        check_model_args(name, ma, mode, mlp_mult, errors, info, args.strict_hyperparams)
    print(f"  checked {len(CONFIGS) - missing} checkpoints, {missing} missing")

    # Completeness check: every checkpoint should be present in both CSVs
    if os.path.exists(args.sweep_csv) and os.path.exists(args.val_loss_csv):
        sw = {r["config"] for r in csv.DictReader(open(args.sweep_csv, encoding="utf-8"))}
        vl = {r["config"] for r in csv.DictReader(open(args.val_loss_csv, encoding="utf-8"))}
        in_dir = {name for name, _, _ in CONFIGS}
        print()
        print("Completeness of analysis CSVs vs checkpoint directory:")
        print(f"  checkpoints: {len(in_dir)}   sweep_results.csv: {len(sw)}   "
              f"validation_losses.csv: {len(vl)}")
        missing_sweep = sorted(in_dir - sw)
        missing_val = sorted(in_dir - vl)
        if missing_sweep:
            info.append(f"checkpoints missing from sweep_results.csv: {missing_sweep}")
            print(f"  WARNING: not in sweep_results.csv: {missing_sweep}")
        else:
            print("  all checkpoints present in sweep_results.csv")
        if missing_val:
            errors.append(f"checkpoints missing from validation_losses.csv: {missing_val}")
        else:
            print("  all checkpoints present in validation_losses.csv")
    else:
        print("  (completeness check skipped: sweep/val-loss CSV not found)")

    print()
    print("=" * 78)
    print("Verify GPT-3 Large checkpoint")
    print("=" * 78)
    if os.path.exists(args.gpt3_large):
        ck = torch.load(args.gpt3_large, map_location="cpu", weights_only=True)
        ma = ck.get("model_args", {}) if isinstance(ck, dict) else {}
        if not isinstance(ma, dict):
            errors.append("GPT-3 Large: no model_args")
        else:
            for field, exp in GPT3_LARGE_EXPECTED.items():
                actual = ma.get(field)
                if isinstance(exp, str):
                    ok = actual == exp
                elif isinstance(exp, bool):
                    ok = actual is exp
                else:
                    ok = actual is not None and abs(actual - exp) < 1e-9
                if not ok:
                    errors.append(
                        f"GPT-3 Large {field}: checkpoint={actual} expected={exp}"
                    )
                else:
                    info.append(f"GPT-3 Large {field}={actual} OK")
            exp_scale = expected_scale("residual_gelu", 1536, 16)
            if abs(ma.get("scale", -1.0) - exp_scale) > 1e-9:
                errors.append(f"GPT-3 Large scale: {ma.get('scale')} expected={exp_scale}")
            else:
                info.append(f"GPT-3 Large scale={ma.get('scale')} OK (1/(2*sqrt(96))={exp_scale})")
    else:
        errors.append(f"MISSING GPT-3 Large checkpoint: {args.gpt3_large}")

    # Optional prototype checkpoints (used by V3 / EmpiricalValidation notebooks)
    for label, path, mode in (
        ("linear (prototype)", r"D:\modification research\linear\ckpt_50000.pt", "original"),
        ("non_linear (prototype)", r"D:\modification research\non_linear\ckpt_50000.pt", "residual_gelu"),
    ):
        if os.path.exists(path):
            ck = torch.load(path, map_location="cpu", weights_only=True)
            ma = ck.get("model_args", {})
            if isinstance(ma, dict):
                n_embd = ma.get("n_embd", 768)
                n_heads = ma.get("num_heads", 12)
                exp_scale = expected_scale(mode, n_embd, n_heads)
                if ma.get("query_mode") != mode or abs(ma.get("scale", -1.0) - exp_scale) > 1e-9:
                    errors.append(
                        f"{label}: query_mode/scale checkpoint=({ma.get('query_mode')}, {ma.get('scale')}) "
                        f"expected=({mode}, {exp_scale})"
                    )
                else:
                    info.append(f"{label} query_mode={ma.get('query_mode')} scale={ma.get('scale')} OK")
                if ma.get("mlp_hidden_size") != 3072:
                    errors.append(f"{label}: mlp_hidden_size={ma.get('mlp_hidden_size')} expected=3072")

    print()
    if errors:
        print(f"FAILED: {len(errors)} problem(s)")
        for e in errors:
            print("  [ERROR] " + e)
        sys.exit(1)
    print("ALL CHECKS PASSED")
    if info:
        print("informational details:")
        for i in info:
            print("  " + i)
    sys.exit(0)


if __name__ == "__main__":
    main()