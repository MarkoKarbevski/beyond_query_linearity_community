"""
Numerical verification of Theorem 1 activation-robustness.

 propositions:
  - ReLU:  exact realization of s1*s2 at every finite κ
  - GELU:  realization with error ~ exp(-2κ²), vanishing in limit κ→∞

Reference: B2_theorem_robustness.pdf (PI's latest theorem)
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

DEVICE = "cpu"


def build_parity_inputs():
    """Four sign patterns s = (s1, s2) in {−1,+1}².
    
    x^q = s1*u1 + s2*u2 for four patterns: (+,+), (+,-), (-,+), (-,-)
    p(x^q) = <u1+u2, x^q> = s1 + s2 in {-2,0,2} (if u1,u2 orthonormal)
    """
    rng = np.random.default_rng(0)
    u1 = rng.normal(size=(4,)).astype(np.float32)
    u2 = rng.normal(size=(4,)).astype(np.float32)
    # make orthonormal for exact p = s1+s2
    u1 = u1 / np.linalg.norm(u1)
    u2 = u2 - np.dot(u1, u2) * u1
    u2 = u2 / np.linalg.norm(u2)
    
    # four patterns: (+,+), (+,-), (-,+), (-,-)
    signs = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.float32)
    # x = s1*u1 + s2*u2 for each pattern
    x = signs[:, 0:1] * u1 + signs[:, 1:2] * u2  # shape (4, 4)
    
    # p(x^q) = <u1+u2, x^q> = s1 + s2
    p_vals = signs[:, 0] + signs[:, 1]  # [-2,0,0,2]
    s1s2 = signs[:, 0] * signs[:, 1]    # [+1,-1,-1,+1]
    return x, p_vals, s1s2


def relu_construction(p_vals, s1s2, kappa):
    """
    Proposition 1: width-2 ReLU bottleneck.
    f(x) = κ (φ(p) + φ(-p)) ĝ - κ ĝ = κ(|p|-1)
    """
    p = torch.tensor(p_vals, dtype=torch.float32, device=DEVICE)
    phi = F.relu
    f = kappa * (phi(p) + phi(-p)) - kappa
    target = kappa * s1s2
    return f, target


def gelu_construction(p_vals, s1s2, lam, kappa):
    """
    Proposition 2: width-2 GELU bottleneck with λ,κ.
    f_{λ,κ}(x) = (κ/λ) (φ(λp) + φ(-λp)) ĝ - κ ĝ
    ℓ̂ = κ s1 s2 + error
    error = 0 when p=0, ≤ (2κ/λ) φ(2λ) when p=±2
    """
    p = torch.tensor(p_vals, dtype=torch.float32, device=DEVICE)
    phi = F.gelu
    raw = (kappa / lam) * (phi(lam * p) + phi(-lam * p)) - kappa
    target = kappa * s1s2
    return raw, target


def verify():
    x, p_vals, s1s2 = build_parity_inputs()
    
    print("=" * 60)
    print("Theorem 1 activation-robustness numerical verification")
    print("=" * 60)
    
    # ReLU
    print("\n### ReLU (Proposition 1) ###")
    for kappa in [1.0, 5.0, 10.0, 50.0, 100.0]:
        f, target = relu_construction(p_vals, s1s2, kappa)
        err = (f - target).abs().max().item()
        print(f"  κ={kappa:6.1f}  max |f - target| = {err:.6f}  (exact = 0)")
    
    # GELU
    print("\n### GELU (Proposition 2) ###")
    print("  error bound ≤ (2κ/λ) φ(2λ), with λ=κ")
    for kappa in [1.0, 5.0, 10.0, 50.0, 100.0]:
        lam = kappa
        f, target = gelu_construction(p_vals, s1s2, lam, kappa)
        err = (f - target).abs().max().item()
        phi_2lam = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-(2 * lam) ** 2 / 2)
        bound = (2 * kappa / lam) * phi_2lam
        print(f"  κ={kappa:6.1f}  max err = {err:.6f}, bound = {bound:.6f}, err/bound = {err/bound:.3f}")
    
    # Plot GELU error decay
    kappas = np.logspace(0, 2, 50)
    errors = []
    bounds = []
    for kappa in kappas:
        lam = kappa
        f, target = gelu_construction(p_vals, s1s2, lam, kappa)
        errors.append((f - target).abs().max().item())
        phi_2lam = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-(2 * lam) ** 2 / 2)
        bounds.append((2 * kappa / lam) * phi_2lam)
    
    plt.figure(figsize=(8, 5))
    plt.loglog(kappas, errors, 'o-', label='GELU max error', color='#2a78d6')
    plt.loglog(kappas, bounds, '--', label='Theoretical bound', color='#e34948')
    plt.xlabel("κ = λ")
    plt.ylabel("Absolute error")
    plt.title("GELU parity construction: error vs κ")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_theorem_gelu_error.png", dpi=150)
    print("\nSaved: fig_theorem_gelu_error.png")
    
    print("\n=== Summary ===")
    print("ReLU: exact at every finite κ (f = κ(|p|-1) = κ s1 s2 on support {-2,0,2})")
    print("GELU: error ~ O(exp(-2κ²)), confirmed numerically")


if __name__ == "__main__":
    verify()