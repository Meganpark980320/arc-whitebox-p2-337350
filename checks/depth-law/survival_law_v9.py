#!/usr/bin/env python3
"""Error-survival law for the v9 write-up (§2). Read-only on team data; no estimator run.

usage: OMP_NUM_THREADS=4 python3 survival_law_v9.py <budget_matrix_9A.json> <out.json> [nets] [samples]

A. Mean-field law vs the stored last-layer error ledger (64 validation networks, n = 1024).
   c_0 = 0, c_{k+1} = f(c_k) (arc-cosine map); chi_k = 1 - arccos(c_k)/pi; S(l) = prod_{k=l+1}^{15} chi_k.
B. Closed forms in c, checked by Gauss-Hermite quadrature (alpha ~ N(0, c/(1-c))).
C. Fresh He-initialised networks (width 1024, 16 layers, N(0, 2/n) weights, N(0, I) inputs):
   per-neuron alpha = mu/sigma from Monte Carlo, the replica cosine c = sum mu^2 / sum E[h^2],
   one-layer energy gains of J_k = diag(Phi(alpha_k)) W_k^T for random directions orthogonal to
   the incoming mean and for the mean direction, and survival to layer 15 from every layer.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

N_WIDTH, DEPTH = 1024, 16


def f(c: float) -> float:
    return (math.sqrt(max(0.0, 1 - c * c)) + (math.pi - math.acos(c)) * c) / math.pi


def chi(c: float) -> float:            # 2 E[Phi(alpha)^2] for alpha ~ N(0, c/(1-c))
    return 1 - math.acos(c) / math.pi


def chi_mean(c: float) -> float:       # gain along the incoming mean direction
    return chi(c) + (2 * c / math.pi) * math.sqrt((1 - c) / (1 + c))


def gamma_mf(c: float, n: int = N_WIDTH) -> float:
    return (n / (n + 4)) * (1 - math.sqrt(1 - c * c) / math.pi)


def depth_map(depth: int) -> list[float]:
    c = [0.0]
    for _ in range(depth - 1):
        c.append(f(c[-1]))
    return c


def part_a(ledger: dict) -> list[dict]:
    c = depth_map(DEPTH)
    rows = []
    for i, lab in enumerate(ledger["layers"]):
        if ledger["survive"][i] is None or lab in ("L0-5", "L15"):
            continue
        l = int(lab[1:])
        law = math.prod(chi(c[k]) for k in range(l + 1, DEPTH))
        lo, hi = ledger["survive_ci95"][i]
        rows.append(dict(layer=l, measured=ledger["survive"][i], ci95=[lo, hi], law=round(law, 4),
                         ratio=round(law / ledger["survive"][i], 4), inside=bool(lo <= round(law, 3) <= hi),
                         per_channel=ledger["survive_by_channel"][i]))
    return rows


def part_b() -> list[dict]:
    from math import erf
    x, w = hermegauss(200)
    w = w / w.sum()
    Phi = lambda a: 0.5 * (1 + np.vectorize(erf)(a / math.sqrt(2)))
    phi = lambda a: np.exp(-a * a / 2) / math.sqrt(2 * math.pi)
    out = []
    for c in (0.2, 0.5, 0.8, 0.923):
        a = math.sqrt(c / (1 - c)) * x
        E = lambda v: float((w * v).sum())
        pairs = {
            "E[phi]": (E(phi(a)), math.sqrt(1 - c) / math.sqrt(2 * math.pi)),
            "E[phi + alpha Phi]": (E(phi(a) + a * Phi(a)), 1 / math.sqrt(2 * math.pi * (1 - c))),
            "E[(alpha^2 - 1) phi]": (E((a * a - 1) * phi(a)), -(1 - c) ** 1.5 / math.sqrt(2 * math.pi)),
            "E[alpha phi]": (E(a * phi(a)), 0.0),
            "E[phi^2]": (E(phi(a) ** 2), math.sqrt((1 - c) / (1 + c)) / (2 * math.pi)),
            "2E[Phi^2]": (2 * E(Phi(a) ** 2), chi(c)),
            "2E[alpha^2 Phi^2]/E[alpha^2]": (2 * E(a * a * Phi(a) ** 2) / E(a * a), chi_mean(c)),
            "gamma": ((2 * N_WIDTH / (N_WIDTH + 4)) * E(Phi(a) - phi(a) ** 2 - a * Phi(a) * phi(a)), gamma_mf(c)),
        }
        out.append(dict(c=c, checks={k: [round(q, 7), round(v, 7)] for k, (q, v) in pairs.items()},
                        max_abs_diff=max(abs(q - v) for q, v in pairs.values())))
    return out


def part_c(nets: int, samples: int) -> dict:
    from math import erf
    n, L = N_WIDTH, DEPTH
    Phi = np.vectorize(lambda a: 0.5 * (1 + erf(a / math.sqrt(2))))
    c_mf = depth_map(L)
    res = dict(nets=nets, samples=samples, per_net=[])
    for net in range(nets):
        rng = np.random.default_rng(20261022 + net)
        W = [(rng.standard_normal((n, n), dtype=np.float32) * np.float32(math.sqrt(2 / n))) for _ in range(L)]
        y = rng.standard_normal((samples, n), dtype=np.float32)
        alpha, c_meas, ybar = [], [], []
        for k in range(L):
            h = y @ W[k]                                  # stored (input, output): h = y W
            mu = h.mean(0, dtype=np.float64)
            var = h.var(0, dtype=np.float64)
            alpha.append(mu / np.sqrt(var))
            c_meas.append(float((mu ** 2).sum() / (mu ** 2 + var).sum()))
            y = np.maximum(h, 0)
            ybar.append(y.mean(0, dtype=np.float64))
        G = [Phi(a) for a in alpha]                         # gate of J_k
        J = lambda k, d: (d @ W[k].astype(np.float64)) * G[k]   # J_k acting on rows of d
        gain_rand, gain_mean, gain_pernet = [None], [None], [None]
        for k in range(1, L):
            m = ybar[k - 1] / np.linalg.norm(ybar[k - 1])
            D = rng.standard_normal((64, n))
            D -= np.outer(D @ m, m)
            gain_rand.append(float(((J(k, D) ** 2).sum(1) / (D ** 2).sum(1)).mean()))
            gain_mean.append(float((J(k, m[None, :]) ** 2).sum()))
            gain_pernet.append(float((2 / n) * (G[k] ** 2).sum()))
        surv_rand, surv_mean, surv_pernet = [], [], []
        for l in range(L - 1):
            m = ybar[l] / np.linalg.norm(ybar[l])
            D = rng.standard_normal((64, n))
            D -= np.outer(D @ m, m)
            e0 = (D ** 2).sum(1)
            dm = m[None, :].copy()
            for k in range(l + 1, L):
                D = J(k, D)
                dm = J(k, dm)
            surv_rand.append(float(((D ** 2).sum(1) / e0).mean()))
            surv_mean.append(float((dm ** 2).sum()))
            surv_pernet.append(float(np.prod(gain_pernet[l + 1:])))
        res["per_net"].append(dict(net=net, c_meas=c_meas, gain_rand=gain_rand, gain_mean=gain_mean,
                                   gain_pernet=gain_pernet, surv_rand=surv_rand, surv_mean=surv_mean,
                                   surv_pernet=surv_pernet,
                                   alpha_sd=[float(a.std()) for a in alpha],
                                   alpha_sd_mf=[math.sqrt(c / (1 - c)) for c in c_mf]))
    res["c_mf"] = c_mf
    res["chi_mf"] = [chi(c) for c in c_mf]
    res["chi_mean_mf"] = [chi_mean(c) for c in c_mf]
    res["surv_law"] = [math.prod(chi(c_mf[k]) for k in range(l + 1, L)) for l in range(L - 1)]
    res["surv_mean_law"] = [math.prod(chi_mean(c_mf[k]) for k in range(l + 1, L)) for l in range(L - 1)]
    return res


def depth_scaling() -> list[dict]:
    rows = []
    for L in (8, 16, 32, 64, 128):
        c = depth_map(L)
        S = [math.prod(chi(c[k]) for k in range(l + 1, L)) for l in range(L)]
        rows.append(dict(depth=L, effective_layers=round(sum(S), 3), one_minus_chi_last=round(1 - chi(c[-1]), 5),
                         approx_3_over_last_plus_9=round(3 / (L - 1 + 9), 5), survival_from_middle=round(S[L // 2], 4)))
    return rows


def main() -> None:
    ledger = json.load(open(sys.argv[1]))
    nets = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    samples = int(sys.argv[4]) if len(sys.argv) > 4 else 8192
    out = dict(part_a_ledger=part_a(ledger), part_b_closed_forms=part_b(), depth_scaling=depth_scaling())
    out["part_c_simulation"] = part_c(nets, samples)
    with open(sys.argv[2], "x") as fh:
        json.dump(out, fh, indent=1)
    print("A. law vs ledger (layer, measured, CI, law, ratio, inside)")
    for r in out["part_a_ledger"]:
        print(f"  {r['layer']:2d}  {r['measured']:.3f}  [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]  {r['law']:.3f}  {r['ratio']:.3f}  {r['inside']}")
    print("B. closed forms: max |quadrature - formula| =", max(b["max_abs_diff"] for b in out["part_b_closed_forms"]))
    print("depth scaling:", out["depth_scaling"])
    C = out["part_c_simulation"]
    P = C["per_net"]
    mean = lambda key, i: float(np.mean([p[key][i] for p in P]))
    print("C. simulation, mean over nets (k: c_mf c_meas | chi_mf gain_rand gain_pernet | chi_mean_mf gain_mean)")
    for k in range(1, DEPTH):
        print(f"  {k:2d}: {C['c_mf'][k]:.3f} {mean('c_meas', k):.3f} | {C['chi_mf'][k]:.3f} {mean('gain_rand', k):.3f} "
              f"{mean('gain_pernet', k):.3f} | {C['chi_mean_mf'][k]:.3f} {mean('gain_mean', k):.3f}")
    print("   survival from layer l (l: law sim_rand pernet | mean_law sim_mean)")
    for l in range(DEPTH - 1):
        print(f"  {l:2d}: {C['surv_law'][l]:.3f} {mean('surv_rand', l):.3f} {mean('surv_pernet', l):.3f} | "
              f"{C['surv_mean_law'][l]:.3f} {mean('surv_mean', l):.3f}")


if __name__ == "__main__":
    main()
