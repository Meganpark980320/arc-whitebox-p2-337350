#!/usr/bin/env python3
"""Second simulation for §2: more fresh He networks; per-layer error gain (2/n) sum Phi(alpha)^2 and the
gate gamma = (2/(n+4)) sum r(alpha) from Monte Carlo alpha, against the mean-field closed forms; alpha kurtosis.
usage: OMP_NUM_THREADS=4 python3 survival_sim2_v9.py <out.json> [nets] [samples]"""
import json, math, sys
import numpy as np
from math import erf
n, L = 1024, 16
f = lambda c: (math.sqrt(max(0.0, 1 - c * c)) + (math.pi - math.acos(c)) * c) / math.pi
c_mf = [0.0]
for _ in range(L - 1):
    c_mf.append(f(c_mf[-1]))
chi_mf = [1 - math.acos(c) / math.pi for c in c_mf]
gam_mf = [(n / (n + 4)) * (1 - math.sqrt(1 - c * c) / math.pi) for c in c_mf]
Phi = np.vectorize(lambda a: 0.5 * (1 + erf(a / math.sqrt(2))))
phi = lambda a: np.exp(-a * a / 2) / math.sqrt(2 * math.pi)
nets = int(sys.argv[2]) if len(sys.argv) > 2 else 12
samples = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
G, GM, K, SD = [], [], [], []
for net in range(nets):
    rng = np.random.default_rng(7000 + net)
    y = rng.standard_normal((samples, n), dtype=np.float32)
    g, gm, ku, sd = [], [], [], []
    for k in range(L):
        W = rng.standard_normal((n, n), dtype=np.float32) * np.float32(math.sqrt(2 / n))
        h = y @ W
        mu = h.mean(0, dtype=np.float64); var = h.var(0, dtype=np.float64)
        a = mu / np.sqrt(var)
        P = Phi(a)
        g.append(float((2 / n) * (P ** 2).sum()))
        gm.append(float((2 / (n + 4)) * (P - phi(a) ** 2 - a * P * phi(a)).sum()))
        ku.append(float(((a - a.mean()) ** 4).mean() / a.var() ** 2) if k else float('nan'))
        sd.append(float(a.std()))
        y = np.maximum(h, 0)
    G.append(g); GM.append(gm); K.append(ku); SD.append(sd)
G, GM, K, SD = map(np.array, (G, GM, K, SD))
out = dict(nets=nets, samples=samples, chi_mf=chi_mf, gamma_mf=gam_mf,
           gain_mean=G.mean(0).tolist(), gain_sd=G.std(0, ddof=1).tolist(),
           gamma_mean=GM.mean(0).tolist(), gamma_sd=GM.std(0, ddof=1).tolist(),
           kurtosis_mean=K.mean(0).tolist(), alpha_sd_mean=SD.mean(0).tolist(),
           alpha_sd_mf=[math.sqrt(c / (1 - c)) for c in c_mf])
with open(sys.argv[1], 'x') as fh:
    json.dump(out, fh, indent=1)
se = lambda s: s / math.sqrt(nets)
print(' k  chi_mf  gain(sim)±se   | gamma_mf gamma(sim)±se | kurt  sd/sd_mf')
for k in range(1, L):
    print(f'{k:2d}  {chi_mf[k]:.3f}  {G[:,k].mean():.3f}±{se(G[:,k].std(ddof=1)):.3f} | {gam_mf[k]:.3f}  {GM[:,k].mean():.3f}±{se(GM[:,k].std(ddof=1)):.3f} | {K[:,k].mean():.2f}  {SD[:,k].mean()/out["alpha_sd_mf"][k]:.3f}')
for l in (6, 10, 14):
    s_sim = np.prod(G[:, l + 1:], axis=1)
    print(f'survival from {l}: law {np.prod(chi_mf[l+1:]):.3f}  sim per-net mean {s_sim.mean():.3f} ± {se(s_sim.std(ddof=1)):.3f}')
