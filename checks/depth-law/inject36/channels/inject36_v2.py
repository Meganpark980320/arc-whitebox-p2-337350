"""inject36 v2 (v1 unchanged): D3 noise only; decompose each layer's local mean error into the incoming-mean
direction and the rest, and propagate each part through the network's own first-order mean map
J_k = diag(Phi(alpha_k)) W_k (clean chain gates) to compare with the measured final error and with S(l)."""
import sys, math, json, torch
sys.path.insert(0, sys.argv[1])
import mlp_kprop.diagslice as ds
import mlp_kprop.factor_k3 as fk
from mlp_kprop.mlp import MLP
from mlp_kprop import kprop_harmonic as kh
torch.set_default_dtype(torch.float64); torch.set_grad_enabled(False)
n, L, sig, net = int(sys.argv[2]), 16, 1e-3, int(sys.argv[4])
st = dict(layer=-1, inj=None, noise=None)
orig_ds, orig_nl = ds.diagslice, fk.factored_nonlin_kprop_k3
def patched_ds(A, part, output_zero_repeated=False):
    out = orig_ds(A, part, output_zero_repeated=output_zero_repeated)
    if st['layer'] == st['inj'] and not isinstance(A, torch.Tensor) and A.ndim == 3 and tuple(part) == (3,):
        out = out + st['noise']
    return out
def patched_nl(*a, **k):
    st['layer'] += 1
    return orig_nl(*a, **k)
ds.diagslice, fk.factored_nonlin_kprop_k3 = patched_ds, patched_nl
def run(mlp, inj, seed):
    st.update(layer=-1, inj=inj, noise=sig * torch.randn(n, generator=torch.Generator().manual_seed(seed)))
    return kh.mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=3, output_all=True, kind=kh.Kind.SIMPLE, factor=True)
f = lambda c: (math.sqrt(max(0, 1 - c * c)) + (math.pi - math.acos(c)) * c) / math.pi
chi = lambda c: 1 - math.acos(c) / math.pi
chim = lambda c: chi(c) + (2 * c / math.pi) * math.sqrt((1 - c) / (1 + c))
c = [0.0]
for _ in range(L - 1): c.append(f(c[-1]))
S = [math.prod(chi(c[k]) for k in range(l + 1, L)) for l in range(L)]
Sm = [math.prod(chim(c[k]) for k in range(l + 1, L)) for l in range(L)]
torch.manual_seed(100 + net)
mlp = MLP(n, n, n, L + 1).double()
Kc = run(mlp, None, 0)
Phi = lambda a: 0.5 * (1 + torch.erf(a / math.sqrt(2)))
gate = []
for k in range(L):
    mu = Kc[f'pre{k}'][1].to_tensor(); var = Kc[f'pre{k}'][2].to_tensor().diagonal()
    gate.append(Phi(mu / var.sqrt()))
W = [mlp.Ws[k].weight for k in range(L)]
m = [Kc[f'act{k}'][1].to_tensor() for k in range(L)]
def prop(d, l):
    for k in range(l + 1, L):
        d = gate[k] * (W[k] @ d)
    return d
rows = []
for l in range(1, L - 1):
    K = run(mlp, l, 1000 * net + 10 * l)
    dl = K[f'act{l}'][1].to_tensor() - m[l]
    dF = K[f'act{L-1}'][1].to_tensor() - m[L - 1]
    u = m[l] / m[l].norm()
    par = (dl @ u) * u; perp = dl - par
    E = lambda v: float((v ** 2).sum())
    Snet = math.prod(float((2 / n) * (gate[k] ** 2).sum()) for k in range(l + 1, L))
    rows.append(dict(net=net, layer=l, measured=E(dF) / E(dl), linear=E(prop(dl, l)) / E(dl),
                     p_mean=E(par) / E(dl), perp_part=E(prop(perp, l)) / E(dl), par_part=E(prop(par, l)) / E(dl),
                     par_gain=(E(prop(par, l)) / E(par)) if E(par) > 0 else None,
                     S=S[l], S_mean=Sm[l], S_net=Snet))
json.dump(rows, open(sys.argv[3], 'w'))
print(net, 'done')
