"""inject36 v3 (v1, v2 unchanged): split the D3-noise effect into a mean channel and a variance channel.
Modes at layer l (applied to the nonlinear step's output, i.e. act_l): 'mean' adds eps to the mean, 'var' adds eps
to the diagonal of the covariance, 'd3' adds 1e-3 noise to the D3 read-out (as v1). For each run we record the local
mean and variance errors at act_l and the final-layer mean error."""
import sys, math, json, torch
sys.path.insert(0, sys.argv[1])
import mlp_kprop.diagslice as ds
import mlp_kprop.factor_k3 as fk
from mlp_kprop.mlp import MLP
from mlp_kprop import kprop_harmonic as kh
torch.set_default_dtype(torch.float64); torch.set_grad_enabled(False)
n, L, net = int(sys.argv[2]), 16, int(sys.argv[4])
st = dict(layer=-1, inj=None, mode=None, noise=None)
orig_ds, orig_nl = ds.diagslice, fk.factored_nonlin_kprop_k3
def patched_ds(A, part, output_zero_repeated=False):
    out = orig_ds(A, part, output_zero_repeated=output_zero_repeated)
    if st['mode'] == 'd3' and st['layer'] == st['inj'] and not isinstance(A, torch.Tensor) and A.ndim == 3 and tuple(part) == (3,):
        out = out + st['noise']
    return out
def patched_nl(*a, **k):
    st['layer'] += 1
    K = orig_nl(*a, **k)
    if st['layer'] == st['inj'] and st['mode'] == 'mean':
        K[1].core = K[1].core + st['noise']
    if st['layer'] == st['inj'] and st['mode'] == 'var':
        K[2].core = K[2].core + torch.diag(st['noise'])
    return K
ds.diagslice, fk.factored_nonlin_kprop_k3 = patched_ds, patched_nl
def run(mlp, inj, mode, seed, sig):
    st.update(layer=-1, inj=inj, mode=mode, noise=sig * torch.randn(n, generator=torch.Generator().manual_seed(seed)))
    K = kh.mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=3, output_all=True, kind=kh.Kind.SIMPLE, factor=True)
    return [(K[f'act{l}'][1].to_tensor().clone(), K[f'act{l}'][2].to_tensor().diagonal().clone()) for l in range(L)]
torch.manual_seed(100 + net)
mlp = MLP(n, n, n, L + 1).double()
clean = run(mlp, None, None, 0, 0.0)
E = lambda v: float((v ** 2).sum())
rows = []
for l in range(1, L - 1):
    for mode, sig in (('mean', 1e-4), ('var', 1e-4), ('d3', 1e-3)):
        o = run(mlp, l, mode, 1000 * net + 10 * l, sig)
        rows.append(dict(net=net, layer=l, mode=mode,
                         loc_mean=E(o[l][0] - clean[l][0]), loc_var=E(o[l][1] - clean[l][1]),
                         loc_cross=float(((o[l][0] - clean[l][0]) * (o[l][1] - clean[l][1])).sum()),
                         final=E(o[L - 1][0] - clean[L - 1][0])))
json.dump(rows, open(sys.argv[3], 'w'))
print(net, 'done')
