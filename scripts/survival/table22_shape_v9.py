# table22_shape_v9.py (22 Oct 2026): copied unchanged from a second-round review check (check_table22_shape.py); usage: python3 table22_shape_v9.py NETS SAMPLES
"""Which Table 2.2 averages are fixed by the replica cosine (alpha-shape independent) and which need Gaussian alpha?
Fresh He nets; compare neuron averages over the measured alpha with the closed forms evaluated at the per-net measured c."""
import math, sys, numpy as np
from math import erf
n, L = 1024, 16
nets, N = int(sys.argv[1]), int(sys.argv[2])
Phi = np.vectorize(lambda a: 0.5*(1+erf(a/math.sqrt(2))))
phi = lambda a: np.exp(-a*a/2)/math.sqrt(2*math.pi)
rows = {8: [], 15: []}
for net in range(nets):
    rng = np.random.default_rng(4242 + net)
    y = rng.standard_normal((N, n), dtype=np.float32)
    for k in range(L):
        W = rng.standard_normal((n, n), dtype=np.float32)*np.float32(math.sqrt(2/n))
        h = y @ W
        if k in rows:
            mu = h.mean(0, dtype=np.float64); var = h.var(0, dtype=np.float64); a = mu/np.sqrt(var)
            c = float((mu**2).sum()/(mu**2+var).sum())
            P, p = Phi(a), phi(a)
            meas = dict(relu_mean=np.mean(p+a*P), chi=2*np.mean(P**2), curv=np.mean(p), k4inj=np.mean((a*a-1)*p),
                        gamma=(2*n/(n+4))*np.mean(P-p*p-a*P*p))
            cf = dict(relu_mean=1/math.sqrt(2*math.pi*(1-c)), chi=1-math.acos(c)/math.pi, curv=math.sqrt(1-c)/math.sqrt(2*math.pi),
                      k4inj=-(1-c)**1.5/math.sqrt(2*math.pi), gamma=(n/(n+4))*(1-math.sqrt(1-c*c)/math.pi))
            rows[k].append({key: meas[key]/cf[key] for key in meas})
        y = np.maximum(h, 0)
for k, r in rows.items():
    print(f"layer {k}: measured / closed form at the net's own c (mean over {nets} nets):",
          " ".join(f"{key} {np.mean([x[key] for x in r]):.3f}" for key in r[0]))
