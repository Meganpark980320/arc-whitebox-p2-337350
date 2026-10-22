# s2_sim_checks_v9.py (22 Oct 2026): copied unchanged from a second-round review check (check_s2_sim.py); usage: python3 s2_sim_checks_v9.py NETS SAMPLES
#!/usr/bin/env python3
"""Independent simulation checks of Section 2 on fresh He networks (n=1024, 16 layers).
Tests what the write-up's own simulation does not:
  (a) generic law with the EXACT gate Pr(h>0) (empirical) vs the Gaussian-closure gate Phi(alpha_hat);
  (b) one-layer gain for errors along the top principal directions of Cov(y) (orthogonal to the mean);
  (c) survival from layer 6 for random vs top-PC directions vs the mean direction;
  (d) spread of sigma_a^2 across neurons and its correlation with mu_a^2 (assumption sigma_a^2 ~ (2/n) tr Cov);
  (e) participation ratio of Cov(y).
usage: OMP_NUM_THREADS=4 python3 check_s2_sim.py [nets] [samples]
"""
import math
import sys
import numpy as np
from math import erf

n, L = 1024, 16
nets = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8192


def f(c):
    return (math.sqrt(max(0.0, 1 - c * c)) + (math.pi - math.acos(c)) * c) / math.pi


cmf = [0.0]
for _ in range(L - 1):
    cmf.append(f(cmf[-1]))
chi = [1 - math.acos(c) / math.pi for c in cmf]
Phi = np.vectorize(lambda a: 0.5 * (1 + erf(a / math.sqrt(2))))
PC_LAYERS = (4, 8, 12, 15)   # layer l: directions in y^(l-1) space
K_PC = 8
res = []
for net in range(nets):
    rng = np.random.default_rng(91000 + net)
    W = [rng.standard_normal((n, n), dtype=np.float32) * np.float32(math.sqrt(2 / n)) for _ in range(L)]
    y = rng.standard_normal((N, n), dtype=np.float32)
    g_emp, g_gau, ybar, mus, vars_, cmeas = [], [], [], [], [], []
    pcs = {}       # layer l -> (eigvecs of Cov(y^(l-1)) orthogonalised to mean, eigvals, PR)
    sig_stats = {}
    for k in range(L):
        if k in PC_LAYERS:
            yc = y.astype(np.float64)
            m_prev = yc.mean(0)
            Cy = (yc.T @ yc) / N - np.outer(m_prev, m_prev)
            ev, U = np.linalg.eigh(Cy)
            ev, U = ev[::-1], U[:, ::-1]
            pr = ev.sum() ** 2 / (ev ** 2).sum()
            mh = m_prev / np.linalg.norm(m_prev)
            V = U[:, :K_PC] - np.outer(mh, mh @ U[:, :K_PC])
            V /= np.linalg.norm(V, axis=0, keepdims=True)
            pcs[k] = (V.T.copy(), ev[:K_PC] / ev.sum(), pr, float((mh @ U[:, :3]) ** 2 @ np.ones(3)))
            del yc
        h = y @ W[k]
        mu = h.mean(0, dtype=np.float64)
        var = h.var(0, dtype=np.float64)
        a = mu / np.sqrt(var)
        g_gau.append(Phi(a))
        g_emp.append((h > 0).mean(0, dtype=np.float64))
        mus.append(mu); vars_.append(var)
        cmeas.append(float((mu ** 2).sum() / (mu ** 2 + var).sum()))
        if k in (8, 15):
            sig_stats[k] = dict(cv_var=float(var.std() / var.mean()),
                                corr_var_mu2=float(np.corrcoef(var, mu ** 2)[0, 1]),
                                alpha_sd=float(a.std()), alpha_sd_mf=math.sqrt(cmf[k] / (1 - cmf[k])))
        y = np.maximum(h, 0)
        ybar.append(y.mean(0, dtype=np.float64))
        del h
    W64 = None

    def J(k, D, gate):
        return (D @ W[k].astype(np.float64)) * gate[k]

    out = dict(net=net, cmeas=cmeas, sig=sig_stats)
    # (a) one-layer generic gains, both gates
    out["gain_rand_emp"], out["gain_rand_gau"], out["pernet_emp"] = {}, {}, {}
    for k in range(1, L):
        m = ybar[k - 1] / np.linalg.norm(ybar[k - 1])
        D = rng.standard_normal((64, n))
        D -= np.outer(D @ m, m)
        e0 = (D ** 2).sum(1)
        out["gain_rand_emp"][k] = float(((J(k, D, g_emp) ** 2).sum(1) / e0).mean())
        out["gain_rand_gau"][k] = float(((J(k, D, g_gau) ** 2).sum(1) / e0).mean())
        out["pernet_emp"][k] = float((2 / n) * (g_emp[k] ** 2).sum())
    # (b) top-PC one-layer gains
    out["pc"] = {}
    for k, (V, share, pr, mean_ov) in pcs.items():
        gains = (J(k, V, g_emp) ** 2).sum(1)   # V rows unit norm
        out["pc"][k] = dict(gains=gains.tolist(), var_share=share.tolist(), PR=pr, PR_over_n=pr / n,
                            mean_overlap_top3=mean_ov)
    # (c) survival from layer 6 (error in y^(6)) : random, top-PC (PCs of Cov(y^(6)) -> use layer-7 PCs? compute fresh)
    l0 = 6
    m = ybar[l0] / np.linalg.norm(ybar[l0])
    D = rng.standard_normal((64, n)); D -= np.outer(D @ m, m)
    e0 = (D ** 2).sum(1)
    Dg, De, dm = D.copy(), D.copy(), m[None, :].copy()
    for k in range(l0 + 1, L):
        Dg = J(k, Dg, g_gau); De = J(k, De, g_emp); dm = J(k, dm, g_emp)
    out["surv6_rand_gau"] = float(((Dg ** 2).sum(1) / e0).mean())
    out["surv6_rand_emp"] = float(((De ** 2).sum(1) / e0).mean())
    out["surv6_mean_emp"] = float((dm ** 2).sum())
    res.append(out)
    print(f"net {net} done", flush=True)

law = lambda l: math.prod(chi[k] for k in range(l + 1, L))
print("\n(a) one-layer generic gain: law chi | empirical-gate | Gaussian-gate | per-net (2/n)sum g_emp^2 ; c_mf vs c_meas")
for k in range(1, L):
    ge = np.mean([r["gain_rand_emp"][k] for r in res]); gg = np.mean([r["gain_rand_gau"][k] for r in res])
    pn = np.mean([r["pernet_emp"][k] for r in res]); cm = np.mean([r["cmeas"][k] for r in res])
    print(f" {k:2d}: {chi[k]:.4f} | {ge:.4f} | {gg:.4f} | {pn:.4f} ; {cmf[k]:.4f} {cm:.4f}")
print("\n(b) one-layer gain for top-8 PCs of Cov(y^(l-1)) (orthogonalised to the mean), empirical gate")
for k in PC_LAYERS:
    G = np.array([r["pc"][k]["gains"] for r in res])
    sh = np.array([r["pc"][k]["var_share"] for r in res])
    pr = np.mean([r["pc"][k]["PR_over_n"] for r in res])
    mo = np.mean([r["pc"][k]["mean_overlap_top3"] for r in res])
    print(f" layer {k}: chi={chi[k]:.4f}; PR/n={pr:.3f}; mean-dir overlap with top3 PCs={mo:.3f}")
    print("   PC var share:", " ".join(f"{x:.3f}" for x in sh.mean(0)))
    print("   PC gain     :", " ".join(f"{x:.3f}" for x in G.mean(0)))
print("\n(c) survival from layer 6: law", round(law(6), 4))
for key in ("surv6_rand_gau", "surv6_rand_emp", "surv6_mean_emp"):
    v = [r[key] for r in res]
    print(f"  {key}: mean {np.mean(v):.4f}  per-net {' '.join(f'{x:.3f}' for x in v)}")
print("\n(d) sigma^2 spread: layer: CV(var_a), corr(var_a, mu_a^2), alpha sd measured vs mean-field")
for k in (8, 15):
    print(f"  {k}: {np.mean([r['sig'][k]['cv_var'] for r in res]):.3f}  {np.mean([r['sig'][k]['corr_var_mu2'] for r in res]):.3f}  "
          f"{np.mean([r['sig'][k]['alpha_sd'] for r in res]):.3f} vs {res[0]['sig'][k]['alpha_sd_mf']:.3f}")
