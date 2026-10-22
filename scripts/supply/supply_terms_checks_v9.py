#!/usr/bin/env python3
"""Checks behind Section 4 and Appendix B of the v9 write-up (derived supply terms).

usage (from the repository root):
    python3 -I docs/writeup/v9/calc/supply_terms_checks_v9.py > docs/writeup/v9/calc/supply_terms_checks_v9.out.txt
    add --no-ledger to skip part I (it calls docs/writeup/calc/ledger_pairs_v1.py on the saved official JSONs)

Light: one process, < 200 MB, well under a minute. It never runs the estimator; part A only reads its text.

Parts
  A  shipped strengths, read from the linked estimator (sha8 dd17f7f4) with line numbers
  B  physical coefficients by exact first-order quadrature (smooth activation, 3 neurons)
  C  engine normalisation: newborn-leg edits against ARC's factored path terms (small n)
  D  polarization rows of a quartic source (the K31F coefficients 1, 3/4, 1/2, 1/4)
  E  the r = 1 trace normalisation 6/(n+4) and the mean-metric gate gamma
  F  power counting of the tabulated Edgeworth mean (ARC block costs) and a scaling illustration
  G  tmd-lite: the squared weight-row norm has mean 2 under He initialisation
  H  cost of the dense alternative to the O1 fold
  J  one source's factored third cumulant, its D21 slabs, the O1 contraction and the fold
  I  official paired changes recomputed from the saved evaluation JSONs (printed last)
"""
from __future__ import annotations

import hashlib
import itertools
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
EST = ROOT / ("experiments/claude_merge_20261014/package_c2eko10S2H8D3ahF10CR3GsHV4RxDnoT78cf4g065/"
              "package/estimator.py")
LEDGER = ROOT / "docs/writeup/calc/ledger_pairs_v1.py"


def hdr(t):
    print("\n" + "=" * 8 + " " + t + " " + "=" * 8)


def ok(flag):
    return "OK" if flag else "MISMATCH"


# ----------------------------------------------------------------------------------------------
# A. shipped strengths
# ----------------------------------------------------------------------------------------------
def part_a():
    hdr("A. shipped strengths in the linked estimator")
    raw = EST.read_bytes()
    print("estimator", EST.relative_to(ROOT), "sha256", hashlib.sha256(raw).hexdigest()[:16])
    lines = raw.decode().split("\n")

    def find(pattern, first=True):
        hits = [(i + 1, l.strip()) for i, l in enumerate(lines) if re.search(pattern, l)]
        return hits[:1] if first else hits

    probes = [
        ("O1 strength tau (derived 1)", r'_CF_O1P_TAU = float\(_os\.environ\.get\("CI_O1P_TAU", "'),
        ("O1 coefficient -tau*6/(n+4)/3", r"c = -_CF_O1P_TAU \* 6\.0 / \(n \+ 4\.0\) / 3\.0"),
        ("O1 vector e = W_all[:,i12] - 2 m w1", r"fnp\.add\(e0, W_all\[:, core\._i12\], out=e0\)"),
        ("ACC carry factor g (derived 1)", r'_CF_O1A = float\(_os\.environ\.get\("CI_O1P_ACC", "'),
        ("ACC gamma = g 2/(n+4) sum(w1 - m d)", r"gam = fnp\.multiply\(fnp\.sum\(t\), _CF_O1A \* 2\.0 / \(n \+ 4\.0\)\)"),
        ("K31F tau31 (derived 1)", r'_CI_K31F_TAU = float\(_os\.environ\.get\("CI_K31F_TAU", "'),
        ("K31F layers", r'_CI_K31F_LI = frozenset'),
        ("K31F U_A weights", r"_CI_K31F_WA = "),
        ("K31F U_P weights", r"_CI_K31F_WB = "),
        ("K31F column factor -tau g_s/3 (g_s = star gain)", r"c = float\(-tau \* _CI_G\['star'\]\[li\] / 3\.0\)"),
        ("C2E tau_birth (derived 1)", r'_CI_C2E_TAU = float\(_os\.environ\.get\("CI_C2E_TAU", "'),
        ("C2E column 1/2 tau r31", r"v = \(fnp\.multiply\(r31, 0\.5 \* _CI_C2E_TAU\)"),
        ("feedback-birth strength 0.35 g_fbb (derived 1/2)", r"fnp\.multiply\(T1, 0\.35 \* _CI_G\['fbb'\]\[li\], out=T1\)"),
        ("feedback-birth M-leg compensation", r'S_sep = fnp\.multiply\(\(e_b\)\[:, None\], a_b\.T, out=NN\("ssep"\)\)'),
        ("gk v3 gate on/off", r'_GK_GATE = _os\.environ\.get\("ARC_GK_GATE", "1"\)'),
        ("gk v3 coefficient 1/144", r"gate = gate \+ \(\(d3 \* k4\) \* \(1\.0 / 144\.0\)\) \* w8"),
        ("gk v3 replaces w1 before the O1 store", r"w1 = _gk_transport_gate\(w1, W_all, D3, _law_g4, a2, phi, i3, i4\)"),
        ("tmd-lite mode (2 = mean pinned at 2)", r'_C_TMD_MODE = _os\.environ\.get\("CI_TMD_LITE", "'),
        ("tmd-lite metric d = w * 2/mean(w)", r"_c_tmd_d = \(_cf_wwrs if _C_TMD_MODE == '1' else _cf_wwrs \* \(METRIC_C / fnp\.mean\(_cf_wwrs\)\)\)"),
        ("row norms sum(W*W, axis=1), W = w32.T", r"_cf_wwrs = fnp\.sum\(WW, axis=1\)"),
        ("mean metric constant", r"^METRIC_C = 2\.0"),
        ("per-layer gain table", r"^_CI_G = \{'fbb'"),
        ("CI_GK multipliers on the gain table", r'^_ci_gk_env = _os\.environ\.get\("CI_GK"'),
        ("k31 gains (old K31 slot, not K31F)", r"_t = tuple\(_x \* _CI_G\['k31'\]\[li\] for _x in K31_THETA\)"),
        ("mean terms of the Wick table", r"^TERM_SPECS = \{\(1,\): "),
    ]
    for label, pat in probes:
        h = find(pat)
        if not h:
            print(f"  {label:52s} NOT FOUND")
            continue
        ln, txt = h[0]
        print(f"  {label:52s} line {ln:5d}: {txt[:120]}")

    # decode the gain table and apply the CI_GK defaults exactly as lines 1892-1899 do
    src = next(l for l in lines if l.startswith("_CI_G = {'fbb'"))
    g = {}
    for m in re.finditer(r"'(\w+)': _hxl_n\('([0-9a-f]+)', 'd', \((\d+),?\)\)", src):
        g[m.group(1)] = memoryview(bytes.fromhex(m.group(2))).cast("d").tolist()
    env = re.search(r'"CI_GK", "([^"]*)"', next(l for l in lines if l.startswith("_ci_gk_env"))).group(1)
    for tok in env.split(","):
        nm, rg, mul = tok.strip().split(":")
        a, _, b = rg.partition("-")
        for l in range(int(a), int(b or a) + 1):
            g[nm][l] *= float(mul)
    fbb = [0.35 * g["fbb"][l] for l in range(16)]
    print("  CI_GK default:", env)
    print("  feedback-birth strength s_l = 0.35*g_fbb[l] (after CI_GK), l = 0..15:")
    print("   ", " ".join(f"{x:.4f}" for x in fbb))
    print("  s_l / (derived 1/2), layers 1..14:", f"{min(fbb[1:15]) / 0.5:.3f} .. {max(fbb[1:15]) / 0.5:.3f}")
    print("  s_l / (identity value 1/4), layers 1..14:", f"{min(fbb[1:15]) / 0.25:.3f} .. {max(fbb[1:15]) / 0.25:.3f}")
    print("  star gains g_s (used by K31F), l = 0..15:", " ".join(f"{g['star'][l]:.4f}" for l in range(16)))
    print("  K31F effective tau31*g_s at l = 3..14:", " ".join(f"{1.1 * g['star'][l]:.3f}" for l in range(3, 15)))
    print("  k31 gains (old seven-term K31 slot), l = 0..15:", " ".join(f"{g['k31'][l]:.4f}" for l in range(16)))
    return g


# ----------------------------------------------------------------------------------------------
# B. physical coefficients by exact first-order quadrature
# ----------------------------------------------------------------------------------------------
BETA = 2.0  # softplus sharpness; any smooth activation works, ReLU is the limit


def sp_derivs(x, kmax=5):
    """softplus_beta and its derivatives 0..kmax."""
    s = 1.0 / (1.0 + np.exp(-BETA * x))
    f0 = np.logaddexp(0.0, BETA * x) / BETA
    out = [f0, s, BETA * s * (1 - s), BETA ** 2 * s * (1 - s) * (1 - 2 * s),
           BETA ** 3 * s * (1 - s) * (1 - 6 * s + 6 * s * s),
           BETA ** 4 * s * (1 - s) * (1 - 14 * s + 36 * s * s - 24 * s ** 3)]
    return out[: kmax + 1]


GH_X, GH_W = np.polynomial.hermite_e.hermegauss(48)
GH_W = GH_W / math.sqrt(2.0 * math.pi)


def grid(dim):
    """tensor-product Gauss-Hermite nodes (first coordinate slowest) and their weights."""
    xs = np.stack(np.meshgrid(*([GH_X] * dim), indexing="ij"), -1).reshape(-1, dim)
    w = GH_W
    for _ in range(dim - 1):
        w = np.multiply.outer(w, GH_W)
    return xs, w.reshape(-1)


GRID3 = grid(3)
GRID2 = grid(2)


def gauss_points(mu, Sig):
    xs, ws = GRID3 if len(mu) == 3 else GRID2
    L = np.linalg.cholesky(Sig)
    return mu + xs @ L.T, ws


def marg(mu, v, k):
    x, w = GH_X, GH_W
    return float((w * sp_derivs(mu + math.sqrt(v) * x, 5)[k]).sum())


def cum3_first_order(mu, Sig, herm):
    """d/dkappa at kappa = 0 of kappa3(f(z0), f(z1), f(z2)) under the first-order Gram-Charlier density
    phi_Sig(z) (1 + kappa * herm(y, P)), y = P (z - mu), P = Sig^-1.  Exact up to quadrature error."""
    Z, w = gauss_points(mu, Sig)
    P = np.linalg.inv(Sig)
    Y = (Z - mu) @ P.T
    H = herm(Y, P)
    f = [sp_derivs(Z[:, i], 0)[0] for i in range(3)]
    def m0(*idx):
        return float((w * np.prod([f[i] for i in idx], axis=0)).sum())
    def m1(*idx):
        return float((w * H * np.prod([f[i] for i in idx], axis=0)).sum())
    x, y, z = 0, 1, 2
    d = (m1(x, y, z) - m1(x, y) * m0(z) - m0(x, y) * m1(z) - m1(x, z) * m0(y) - m0(x, z) * m1(y)
         - m1(y, z) * m0(x) - m0(y, z) * m1(x)
         + 2 * (m1(x) * m0(y) * m0(z) + m0(x) * m1(y) * m0(z) + m0(x) * m0(y) * m1(z)))
    return d


def cum3_gauss(mu, Sig):
    Z, w = gauss_points(mu, Sig)
    f = [sp_derivs(Z[:, i], 0)[0] for i in range(3)]
    def m(*idx):
        return float((w * np.prod([f[i] for i in idx], axis=0)).sum())
    return (m(0, 1, 2) - m(0, 1) * m(2) - m(0, 2) * m(1) - m(1, 2) * m(0) + 2 * m(0) * m(1) * m(2))


def part_b():
    hdr("B. physical coefficients, exact first-order quadrature (softplus, beta = 2)")
    mu = np.array([0.3, -0.2, 0.5])
    v = np.array([1.1, 0.8, 1.3])
    q = [marg(mu[i], v[i], 1) for i in range(3)]
    d = [marg(mu[i], v[i], 2) for i in range(3)]
    t = [marg(mu[i], v[i], 3) for i in range(3)]
    w4 = [marg(mu[i], v[i], 4) for i in range(3)]
    a, b, c = 0, 1, 2
    # kappa3_aab = kappa enters the density as (1/6) * 3 * kappa * H_aab = (kappa/2) H_aab
    def H_aab(Y, P):
        return 0.5 * (Y[:, a] ** 2 * Y[:, b] - P[a, a] * Y[:, b] - 2 * P[a, b] * Y[:, a])
    h = 1e-3
    def deriv_offdiag(i, j):
        vals = []
        for s in (h, -h):
            S = np.diag(v).astype(float)
            S[i, j] = S[j, i] = s
            vals.append(cum3_first_order(mu, S, H_aab))
        return (vals[0] - vals[1]) / (2 * h)
    t1b = deriv_offdiag(a, c)
    t1c = deriv_offdiag(b, c)
    tab = deriv_offdiag(a, b)
    print(f"  T1b: coefficient of kappa3_aab * C_ac   = {t1b:.8f};  (1/2) t_a q_b q_c = {0.5 * t[a] * q[b] * q[c]:.8f}"
          f"  ratio {t1b / (0.5 * t[a] * q[b] * q[c]):.6f}")
    print(f"  T1c: coefficient of kappa3_aab * C_bc   = {t1c:.8f};  (1/2) d_a d_b q_c = {0.5 * d[a] * d[b] * q[c]:.8f}"
          f"  ratio {t1c / (0.5 * d[a] * d[b] * q[c]):.6f}")
    print(f"  control: coefficient of kappa3_aab * C_ab = {tab:.2e} (no connected diagram reaches c: expected 0)")
    # Gaussian part: path birth (coefficient of C_ab C_bc) and doubled light edge (coefficient of C_ab^2 C_bc)
    hh = 0.03
    def g(s1, s2):
        S = np.diag(v).astype(float)
        S[a, b] = S[b, a] = s1
        S[b, c] = S[c, b] = s2
        return cum3_gauss(mu, S)
    G = {(i, j): g(i * hh, j * hh) for i in (-1, 0, 1) for j in (-1, 0, 1)}
    birth = (G[1, 1] - G[1, -1] - G[-1, 1] + G[-1, -1]) / (4 * hh * hh)
    d2d1 = ((G[1, 1] - 2 * G[0, 1] + G[-1, 1]) - (G[1, -1] - 2 * G[0, -1] + G[-1, -1])) / (2 * hh * hh * hh)
    c2edge = 0.5 * d2d1  # coefficient of s1^2 s2 is (1/2) d^3 g / ds1^2 ds2
    print(f"  birth: coefficient of C_ab C_bc        = {birth:.6f};  q_a d_b q_c = {q[a] * d[b] * q[c]:.6f}"
          f"  ratio {birth / (q[a] * d[b] * q[c]):.4f}")
    print(f"  C2edge: coefficient of C_ab^2 C_bc     = {c2edge:.6f};  (1/2) d_a t_b q_c = {0.5 * d[a] * t[b] * q[c]:.6f}"
          f"  ratio {c2edge / (0.5 * d[a] * t[b] * q[c]):.4f}  (finite differences, step {hh})")
    # vertex dressing (gk): kappa3_aaa = kappa on neuron a, C_ab = s; Cov(f_a, f_b) coefficient of kappa*s
    mu2, v2 = mu[:2], v[:2]
    def cov_first(S):
        Z, w = gauss_points(mu2, S)
        P = np.linalg.inv(S)
        Y = (Z - mu2) @ P.T
        Hh = (Y[:, 0] ** 3 - 3 * P[0, 0] * Y[:, 0]) / 6.0       # (1/6) H_aaa
        fa = sp_derivs(Z[:, 0], 0)[0]
        fb = sp_derivs(Z[:, 1], 0)[0]
        m1ab = float((w * Hh * fa * fb).sum())
        m1a = float((w * Hh * fa).sum())
        m0b = float((w * fb).sum())
        return m1ab - m1a * m0b                       # Cov's first-order change (H has zero mean in b alone)
    vals = []
    for s in (h, -h):
        S = np.diag(v2).astype(float)
        S[0, 1] = S[1, 0] = s
        vals.append(cov_first(S))
    dress = (vals[0] - vals[1]) / (2 * h)
    print(f"  gk dressing: coefficient of kappa3_aaa * C_ab in Cov = {dress:.8f};  (1/6) W4_a q_b = "
          f"{w4[a] * q[b] / 6:.8f}  ratio {dress / (w4[a] * q[b] / 6):.6f}")
    # doubled-index self-inheritance: kappa4(f_a, f_a, f_k, f_l) <- kappa4_aakl, independent neurons
    def H_aakl(Y, P):
        # (1/24) * 12 * H_aakl with diagonal P and distinct a, k, l: H = (y_a^2 - P_aa) y_k y_l
        return 0.5 * (Y[:, 0] ** 2 - P[0, 0]) * Y[:, 1] * Y[:, 2]
    S = np.diag(v).astype(float)
    Z, w = gauss_points(mu, S)
    P = np.linalg.inv(S)
    Y = (Z - mu) @ P.T
    Hk = H_aakl(Y, P)
    fa, fk, fl = (sp_derivs(Z[:, i], 0)[0] for i in range(3))
    # first-order change of kappa4(X, X, K, L): only moments containing X, X, K, L together move
    mXXKL = float((w * Hk * fa * fa * fk * fl).sum())
    mX = float((w * fa).sum())
    mXKL = float((w * Hk * fa * fk * fl).sum())
    # kappa4(X,X,K,L) first order = d[E XXKL] - 2 E[X] d[E XKL]  (other products of moments have no a-a-k-l block)
    k4c = mXXKL - 2 * mX * mXKL
    ex = GH_X
    za = mu[0] + math.sqrt(v[0]) * ex
    der = sp_derivs(za, 2)
    Ef1sq = float((GH_W * der[1] ** 2).sum())
    Eff2 = float((GH_W * der[0] * der[2]).sum())
    rr = Ef1sq + Eff2 - mX * d[0]
    print(f"  self-inheritance: coefficient of kappa4_aakl in kappa4(f_a,f_a,f_k,f_l) = {k4c:.8f};  "
          f"(E f'^2 + E f f'' - m d)_a q_k q_l = {rr * q[1] * q[2]:.8f}  ratio {k4c / (rr * q[1] * q[2]):.6f}")
    # ReLU limit of the same coefficient: 1/2 (E[(f^2)''] - 2 m E[f'']) = Phi - phi^2 - alpha Phi phi
    from math import erf, exp, pi, sqrt
    for al in (-1.0, 0.0, 0.7, 2.0):
        Phi = 0.5 * (1 + erf(al / sqrt(2)))
        phi = exp(-al * al / 2) / sqrt(2 * pi)
        sig = 1.3
        m = sig * (phi + al * Phi)
        dd = phi / sig
        print(f"    ReLU alpha={al:+.1f}: q - m d = {Phi - m * dd:.6f};  Phi - phi^2 - alpha Phi phi = "
              f"{Phi - phi * phi - al * Phi * phi:.6f};  e = 2m(1-q) = {2 * m * (1 - Phi):.6f}")


# ----------------------------------------------------------------------------------------------
# C. engine normalisation against ARC's factored path terms
# ----------------------------------------------------------------------------------------------
def sym3(T):
    return sum(np.transpose(T, p) for p in itertools.permutations(range(3))) / 6.0


def b1(X, Y):
    """engine birth source with P = I: Sym(X, I, Y), T_ijk = sum_r X_ir delta_jr Y_kr = X_ij Y_kj."""
    return sym3(np.einsum("ij,kj->ijk", X, Y))


def distinct_mask(n):
    i, j, k = np.indices((n, n, n))
    return (i != j) & (j != k) & (i != k)


def part_c():
    hdr("C. engine newborn-leg edits against ARC factor_k3.py:386-407 (n = 7, random inputs)")
    rng = np.random.default_rng(7)
    n = 7
    C = rng.normal(size=(n, n)) * 0.3
    C = (C + C.T) / 2
    np.fill_diagonal(C, 0)
    D = rng.normal(size=(n, n)) * 0.2        # D_ij = kappa3_iij (row index repeated), zero diagonal
    np.fill_diagonal(D, 0)
    q = rng.uniform(0.2, 0.9, n)
    d = rng.uniform(0.1, 0.5, n)
    r = rng.normal(size=n)                    # r = t/d (= -mu/v for ReLU)
    t = r * d
    w4 = rng.normal(size=n)
    mask = distinct_mask(n)
    Dq, Dd, Dr, Dt = map(np.diag, (q, d, r, t))
    A = Dq @ C                                # newborn A leg, a_b = d(w1) C_off

    def engine(a):
        return b1(3 * a, a @ Dd)

    def lin(delta):                           # part of the birth linear in the edit delta (birth is quadratic)
        return (engine(A + delta) - engine(A - delta)) / 2

    # physical tensors on all-distinct entries (Appendix B.5): sum over ordered (x, m, y)
    Tb = np.zeros((n, n, n))
    Tfb = np.zeros((n, n, n))
    Tc2 = np.zeros((n, n, n))
    for i, j, k in zip(*np.nonzero(mask)):
        sb = sfb = sc2 = 0.0
        for x, m, y in itertools.permutations((i, j, k)):
            sb += 0.5 * d[m] * q[x] * q[y] * C[x, m] * C[m, y]                 # each path counted twice (x<->y)
            sfb += 0.5 * t[m] * q[x] * q[y] * D[m, x] * C[m, y]                 # T1b
            sfb += 0.5 * d[x] * d[m] * q[y] * D[x, m] * C[m, y]                 # T1c
            sc2 += 0.5 * d[x] * t[m] * q[y] * C[x, m] ** 2 * C[m, y]            # doubled light edge
        Tb[i, j, k], Tfb[i, j, k], Tc2[i, j, k] = sb, sfb, sc2

    def err(X, Y):
        return float(np.abs((X - Y)[mask]).max() / np.abs(Y[mask]).max())

    print(f"  Gaussian birth Sym(3A, I, A d(w2)) vs physical path term: rel. max error {err(engine(A), Tb):.1e}")
    for cval in (0.25, 0.5):
        delta = cval * (Dd @ D + Dq @ D.T @ Dr)
        e1 = err(lin(delta), Tfb)
        print(f"  feedback-birth edit with coefficient {cval}: rel. max error vs physical T1b+T1c {e1:.3e}"
              f"   (ratio of norms {np.linalg.norm(lin(delta)[mask]) / np.linalg.norm(Tfb[mask]):.4f})")
    # inherited thin legs (V18): X = 3A + Xt, Y = A d(w2) + Yt, Xt = 1.5 d(w2) D, Yt = 0.5 d(w1) D^T d(w3)
    Xt = 1.5 * Dd @ D
    Yt = 0.5 * Dq @ D.T @ Dt
    v18 = b1(3 * A + Xt, A @ Dd + Yt)
    v18lin = b1(3 * A, Yt) + b1(Xt, A @ Dd)
    print(f"  inherited thin legs, linear part / physical T1b+T1c: ratio of norms "
          f"{np.linalg.norm(v18lin[mask]) / np.linalg.norm(Tfb[mask]):.4f}; rel. error vs 0.5*physical "
          f"{err(v18lin, 0.5 * Tfb):.1e}")
    # completing-square identity (exact): Sym(X,I,Y) = 3 Sym(B, I, B d) - 3 Sym(E, I, E d)
    F_ = A + 0.5 * Dd @ D
    G_ = A + 0.5 * Dq @ D.T @ Dr
    B = (F_ + G_) / 2
    E = (F_ - G_) / 2
    ident = 3 * b1(B, B @ Dd) - 3 * b1(E, E @ Dd)
    print(f"  completing-square identity, max |difference| = {np.abs(ident - v18).max():.1e} "
          f"(B = A + 1/4[d(w2) D + d(w1) D^T d(r)])")
    # ARC's factored augmented path terms (factor_k3.py:389-407), kappa4 (2,2) leg omitted
    w = {1: q, 2: d, 3: t, 4: w4}
    legs = [(1, 1, C, 1, "light"), (2, 1, D / 2, 2, "heavy21"), (1, 2, D.T / 2, 2, "heavy12"),
            (2, 2, C * C / 2, 2, "doubled")]
    budget = 3
    parts = {}
    for al, bl, sl, cl, nl in legs:
        for ar, br, sr, cr, nr in legs:
            if cl + cr > budget:
                continue
            fac1 = w[al][:, None] * sl
            fac3 = w[bl + ar][:, None] * w[br][None, :] * sr
            T = sym3(np.einsum("ij,jk->ijk", fac1, 3 * fac3))   # factors (fac1, 3 I, fac3^T)
            key = "+".join(sorted({nl, nr} - {"light"})) or "light"
            parts[key] = parts.get(key, 0) + T
    arc_fb = parts.get("heavy12", 0) + parts.get("heavy21", 0)
    print(f"  ARC light-light term vs engine Gaussian birth: rel. error {err(parts['light'], engine(A)):.1e}")
    print(f"  ARC one-heavy-leg terms vs physical T1b+T1c: rel. error {err(arc_fb, Tfb):.1e}")
    print(f"  ARC doubled-light-edge terms vs physical C2edge: rel. error {err(parts['doubled'], Tc2):.1e}")
    for tau in (0.5, 0.65, 1.0):
        delta = 0.5 * tau * (Dd @ (C * C) @ Dr)
        ratio = np.linalg.norm(lin(delta)[mask]) / np.linalg.norm(Tc2[mask])
        print(f"  C2E edit with tau_birth = {tau}: |linear part| / |physical C2edge| = {ratio:.4f}, "
              f"rel. error vs tau * physical {err(lin(delta), tau * Tc2):.1e}")


# ----------------------------------------------------------------------------------------------
# D. polarization rows
# ----------------------------------------------------------------------------------------------
def sym4(T):
    return sum(np.transpose(T, p) for p in itertools.permutations(range(4))) / 24.0


def part_d():
    hdr("D. polarization rows of a quartic source (K31F)")
    rng = np.random.default_rng(3)
    n, R = 5, 3
    A = rng.normal(size=(n, R))
    P = rng.normal(size=(n, R))
    cf = rng.normal(size=(5, R))     # cf[k, r]: coefficient of (a_r.x)^(4-k) (p_r.x)^k
    K = np.zeros((n,) * 4)
    for r in range(R):
        a, p = A[:, r], P[:, r]
        for k in range(5):
            vecs = [a] * (4 - k) + [p] * k
            K += cf[k, r] * sym4(np.einsum("i,j,k,l->ijkl", *vecs))
    K31 = np.array([[K[i, i, i, j] for j in range(n)] for i in range(n)])
    UA = cf[0] * A ** 3 + 0.75 * cf[1] * A ** 2 * P + 0.5 * cf[2] * A * P ** 2 + 0.25 * cf[3] * P ** 3
    UP = 0.25 * cf[1] * A ** 3 + 0.5 * cf[2] * A ** 2 * P + 0.75 * cf[3] * A * P ** 2 + cf[4] * P ** 3
    rows = UA @ A.T + UP @ P.T
    F = sum(cf[k] * A ** (4 - k) * P ** k for k in range(5)).sum(1)
    print(f"  K(x,x,x,y) slice vs U_A A^T + U_P P^T: max |diff| = {np.abs(K31 - rows).max():.1e}")
    print(f"  diagonal K(e_i,e_i,e_i,e_i) vs F_i: max |diff| = {np.abs(np.diag(K31) - F).max():.1e}")
    print("  weights of U_A on (A^3, A^2P, AP^2, P^3) = (1, 3/4, 1/2, 1/4) x (c40, c31, c22, c13); "
          "U_P = (1/4, 1/2, 3/4, 1) x (c31, c22, c13, c04)")


# ----------------------------------------------------------------------------------------------
# E. trace normalisation and gamma
# ----------------------------------------------------------------------------------------------
def part_e():
    hdr("E. r = 1 trace normalisation and the mean-metric gate")
    rng = np.random.default_rng(5)
    n = 7
    G = rng.normal(size=(n, n))
    G = (G + G.T) / 2
    np.fill_diagonal(G, 0)
    for omega in (1.0, 2.0):
        M = omega * np.eye(n)
        T = sym4(np.einsum("ij,kl->ijkl", G, M))
        tr = np.einsum("aakl->kl", T)
        off = ~np.eye(n, dtype=bool)
        print(f"  metric {omega:.0f} I: (6/(n+4)) sum_a T_aakl / G_kl on k != l = "
              f"{(6 / (n + 4) * tr[off] / G[off]).mean():.6f} (expected {omega:.0f})")
        aa = np.array([[T[a, a, 0, 1] for a in range(n)]])[0]
        print(f"    T_aa01 for a not in (0,1) = omega/6 * G_01: {np.allclose(aa[2:], omega / 6 * G[0, 1])}")
    # gamma: G' = (6/(n+4)) sum_a r_a q_k q_l T_aakl  ->  (omega/(n+4)) sum_{a not in {k,l}} r_a q_k G_kl q_l
    rvec = rng.uniform(0.3, 0.9, n)
    qv = rng.uniform(0.2, 0.9, n)
    omega = 2.0
    T = sym4(np.einsum("ij,kl->ijkl", G, omega * np.eye(n)))
    Gp = np.zeros((n, n))
    for k in range(n):
        for l in range(n):
            if k == l:
                continue
            s = sum(rvec[a] * T[a, a, k, l] for a in range(n) if a not in (k, l))
            Gp[k, l] = 6 / (n + 4) * s * qv[k] * qv[l]
    pred = np.zeros((n, n))
    for k in range(n):
        for l in range(n):
            if k != l:
                pred[k, l] = omega / (n + 4) * sum(rvec[a] for a in range(n) if a not in (k, l)) * qv[k] * G[k, l] * qv[l]
    print(f"  inherited core vs (omega/(n+4)) sum_a r_a D_q G D_q: max |diff| = {np.abs(Gp - pred).max():.1e}")
    from math import pi
    n0 = 1024
    print(f"  gamma at alpha = 0 (every r = 1/2 - 1/(2 pi)): 2n/(n+4) * (1/2 - 1/(2pi)) = "
          f"{2 * n0 / (n0 + 4) * (0.5 - 1 / (2 * pi)):.4f}")
    print(f"  O1 counterterm coefficient c_A = 6/(n+4) at n = 1024: {6 / (n0 + 4):.6e}")


# ----------------------------------------------------------------------------------------------
# F. power counting of the tabulated mean
# ----------------------------------------------------------------------------------------------
def block_cost(v):
    """ARC kprop_harmonic.py:42-48: cost(v) = sum(v) - 1 - [all entries even]."""
    return sum(v) - 1 - all(x % 2 == 0 for x in v)


def part_f():
    hdr("F. power counting of the tabulated Edgeworth mean")
    terms = [
        ("W0", [], True), ("kappa3 W3/6", [(3,)], True), ("kappa4 W4/24", [(4,)], True),
        ("kappa3^2 W6/72", [(3,), (3,)], True), ("kappa3 kappa4 W7/144", [(3,), (4,)], True),
        ("kappa4^2 W8/1152", [(4,), (4,)], False), ("kappa5 W5/120", [(5,)], False),
        ("kappa6 W6/720", [(6,)], False), ("kappa3^3 W9/1296", [(3,), (3,), (3,)], False),
        ("kappa3 kappa5 W8/720", [(3,), (5,)], False),
    ]
    print("  term                     ARC cost   per-entry size   naive size (kappa_k ~ n^-(k-2)/2)   in table")
    for name, blocks, inn in terms:
        cost = sum(block_cost(b) for b in blocks)
        naive = sum((sum(b) - 2) / 2 for b in blocks)
        print(f"  {name:24s} {cost:5d}      n^-{cost / 2:<4g}          n^-{naive:<4g}                          {inn}")
    print("  ARC's mean at k_max = 3 keeps cost <= 2 (simple) or <= 3 (augmented): W0, kappa3, kappa4 terms only.")
    rng = np.random.default_rng(11)
    rows = 400
    out = []
    for n in (256, 512, 1024, 2048, 4096):
        W = rng.normal(scale=math.sqrt(2.0 / n), size=(rows, n))
        s3 = np.sqrt(np.mean(np.sum(W ** 3, 1) ** 2))
        s4 = np.mean(np.sum(W ** 4, 1))
        s3abs = np.mean(np.sum(np.abs(W) ** 3, 1))
        out.append((n, s3, s4, s3abs))
        print(f"  n={n:5d}: rms sum_j W^3 = {s3:.3e}; mean sum_j W^4 = {s4:.3e}; mean sum_j |W|^3 = {s3abs:.3e}")
    ln = np.log([o[0] for o in out])
    for k, name in ((1, "sum W^3 (random sign)"), (2, "sum W^4"), (3, "sum |W|^3 (coherent sign)")):
        slope = np.polyfit(ln, np.log([o[k] for o in out]), 1)[0]
        print(f"  log-log slope in n, {name:26s}: {slope:+.3f}")


# ----------------------------------------------------------------------------------------------
# G. tmd-lite metric
# ----------------------------------------------------------------------------------------------
def part_g():
    hdr("G. tmd-lite: squared row norms of a He-initialised layer")
    rng = np.random.default_rng(13)
    n = 1024
    w = rng.normal(scale=math.sqrt(2.0 / n), size=(n, n))   # stored as (n_in, n_out): z = y @ w
    W = w.T                                                  # engine orientation (n_out, n_in): z = W y
    omega = (W * W).sum(axis=1)                              # = diag(W W^T) = diag(w^T w)
    print(f"  omega_a = sum_j W_aj^2: mean {omega.mean():.4f} (E = 2), sd {omega.std():.4f} "
          f"(sqrt(8/n) = {math.sqrt(8 / n):.4f}), range {omega.min():.3f} .. {omega.max():.3f}")
    print(f"  equals diag(W W^T): {np.allclose(omega, np.diag(W @ W.T))}; equals diag(w^T w): "
          f"{np.allclose(omega, np.diag(w.T @ w))}")
    dmet = omega * (2.0 / omega.mean())
    print(f"  tmd-lite metric d = 2 omega / mean(omega): mean {dmet.mean():.4f}, sd {dmet.std():.4f}")


# ----------------------------------------------------------------------------------------------
# H. cost of the dense O1 alternative
# ----------------------------------------------------------------------------------------------
def part_h():
    hdr("H. cost of forming W X_F W^T densely")
    n = 1024
    per_layer = 2 * (2 * n ** 3)
    print(f"  two n x n products per layer: {per_layer / 1e9:.3f} GF; over the 13 receiving layers 2..14: "
          f"{13 * per_layer / 1e9:.1f} GF")
    c0, dc = 0.1533, 0.00151
    print(f"  O1 official compute change +0.151 % of C = {c0} -> {dc * c0 * 2 ** 41 / 1e9:.2f} GF per predict")


# ----------------------------------------------------------------------------------------------
# J. source representation, D21 slabs and the O1 contraction
# ----------------------------------------------------------------------------------------------
def part_j():
    hdr("J. one source: kappa3_s = Sym(3A, P, A d(y)) + Sym(M, P, P); D21 slabs; O1 contraction and fold")
    rng = np.random.default_rng(17)
    n, qr = 6, 2
    A, P = rng.normal(size=(n, n)), rng.normal(size=(n, n))
    y, eta, eps = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    Z, L = rng.normal(size=(n, qr)), rng.normal(size=(n, qr))
    M = P * eta[None, :] + 3 * A * eps[None, :] + Z @ L.T

    def symf(X, Y, Zf):
        return sym3(np.einsum("ir,jr,kr->ijk", X, Y, Zf))

    K3 = symf(3 * A, P, A * y[None, :]) + symf(M, P, P)
    D21 = np.einsum("iij->ij", K3)
    D3 = np.einsum("iii->i", K3)
    LA = 2 * A * P * y[None, :] + P * P * eps[None, :]
    LP = A * A * y[None, :] + P * P * eta[None, :] / 3 + 2.0 / 3.0 * M * P
    thin = (P * P) @ L @ Z.T / 3
    D21s = LA @ A.T + LP @ P.T + thin
    D3s = 3 * (A * A * P * y[None, :]).sum(1) + (M * P * P).sum(1)
    print(f"  D21 = L_A A^T + L_P P^T + (1/3)(P*P) L Z^T: max |diff| = {np.abs(D21 - D21s).max():.1e}")
    print(f"  D3 = 3 sum_j y_j A_ij^2 P_ij + sum_j M_ij P_ij^2: max |diff| = {np.abs(D3 - D3s).max():.1e}")
    e = rng.normal(size=n)
    Te = np.einsum("i,ijk->jk", e, K3)
    al, pi_, ze = A.T @ e, P.T @ e, Z.T @ e
    ca = y * al + eps * pi_
    cb = y * pi_
    cd = eta * pi_ + eps * al + (L @ ze) / 3
    Tf = A @ np.diag(cb) @ A.T + A @ np.diag(ca) @ P.T + P @ np.diag(ca) @ A.T + P @ np.diag(cd) @ P.T
    Zthin = Z @ L.T @ np.diag(pi_) @ P.T
    Tf_full = Tf + (Zthin + Zthin.T) / 3
    print(f"  T_e = sum_i e_i kappa3_s[i,j,k] vs A D_cb A^T + A D_ca P^T + P D_ca A^T + P D_cd P^T + thin: "
          f"max |diff| = {np.abs(Te - Tf_full).max():.1e} (thin part norm share "
          f"{np.linalg.norm((Zthin + Zthin.T) / 3) / np.linalg.norm(Te):.2f} on these random inputs)")
    W = rng.normal(size=(n, n)) / math.sqrt(n)
    q = rng.uniform(0.2, 0.9, n)
    cA = 6.0 / (n + 4.0)
    XF = cA * np.diag(q) @ Tf @ np.diag(q)
    K31 = W @ XF @ W.T                      # (omega/2) (W X_F W^T) with omega = 2
    Ap, Pp = W @ (q[:, None] * A), W @ (q[:, None] * P)
    rho = rng.normal(size=n)                 # row weight t/(3d) = -mu/(3v)
    dLA = rho[:, None] * (Pp * ca[None, :] + Ap * cb[None, :])
    dLP = rho[:, None] * (Ap * ca[None, :] + Pp * cd[None, :])
    print(f"  fold: rho (W X_F W^T) vs dL_A A'^T + dL_P P'^T: max |diff| = "
          f"{np.abs(rho[:, None] * K31 - cA * (dLA @ Ap.T + dLP @ Pp.T)).max():.1e}")


# ----------------------------------------------------------------------------------------------
# I. official pairs
# ----------------------------------------------------------------------------------------------
def part_i():
    hdr("I. official paired changes (docs/writeup/calc/ledger_pairs_v1.py, saved evaluation JSONs)")
    pairs = ("334756:334775,334775:334780,334806:334933,335702:335956,336435:336471,336471:336603,"
             "336621:336682,336682:336706,336706:336731,336836:336885,337264:337303,337303:337350")
    res = subprocess.run([sys.executable, "-I", str(LEDGER), "--pairs", pairs], capture_output=True, text=True,
                         cwd=ROOT, timeout=600)
    print(res.stdout.strip() or res.stderr.strip()[-2000:])


def main():
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
    part_f()
    part_g()
    part_h()
    part_j()
    if "--no-ledger" not in sys.argv:
        part_i()


if __name__ == "__main__":
    main()
