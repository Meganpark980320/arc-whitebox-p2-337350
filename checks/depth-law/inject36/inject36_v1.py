"""18264 잡음 실험 재현 + 현지/생존 분해.
ARC 기준 K=3 SIMPLE factored 사슬(n=256, 16층)에서 ℓ층 비선형 단계의 D3((3,) 조각)·D21((2,1) 조각) 읽기값에
iid N(0,σ²) 잡음을 넣고, (a) 넣은 층의 평균 오차, (b) 최종 층 평균 오차를 잰다."""
import sys, math, json, torch
sys.path.insert(0, sys.argv[1])
import mlp_kprop.diagslice as ds
import mlp_kprop.factor_k3 as fk
from mlp_kprop.mlp import MLP
from mlp_kprop import kprop_harmonic as kh
torch.set_default_dtype(torch.float64); torch.set_grad_enabled(False)
n, L, sig = int(sys.argv[2]), 16, 1e-3
nets = int(sys.argv[3])
st = dict(layer=-1, inj=None, kinds=(), noise={})
orig_ds, orig_nl = ds.diagslice, fk.factored_nonlin_kprop_k3
def patched_ds(A, part, output_zero_repeated=False):
    out = orig_ds(A, part, output_zero_repeated=output_zero_repeated)
    if st['layer'] == st['inj'] and not isinstance(A, torch.Tensor) and A.ndim == 3:
        p = tuple(part)
        if p == (3,) and 'D3' in st['kinds']:
            out = out + st['noise']['D3']
        elif p in ((2, 1), (1, 2)) and 'D21' in st['kinds']:
            N = st['noise']['D21']
            out = out + (N if p == (2, 1) else N.T)
    return out
def patched_nl(*a, **k):
    st['layer'] += 1
    return orig_nl(*a, **k)
ds.diagslice = patched_ds
fk.factored_nonlin_kprop_k3 = patched_nl
def run(mlp, inj, kinds, seed):
    st.update(layer=-1, inj=inj, kinds=kinds)
    g = torch.Generator().manual_seed(seed)
    st['noise'] = dict(D3=sig * torch.randn(n, generator=g), D21=sig * torch.randn(n, n, generator=g))
    K_in = {1: torch.zeros(n), 2: torch.eye(n)}
    Kb = kh.mlp_kprop(mlp, K_in, k_max=3, output_all=True, kind=kh.Kind.SIMPLE, factor=True)
    return [Kb[f'act{l}'][1].to_tensor().clone() for l in range(L)]
res = []
for net in [int(sys.argv[5])]:
    torch.manual_seed(100 + net)
    mlp = MLP(n, n, n, L + 1).double()
    clean = run(mlp, None, (), 0)
    for kinds in (('D3',), ('D21',), ('D3', 'D21')):
        for l in range(L):
            for s in range(1):
                m = run(mlp, l, kinds, 1000 * net + 10 * l + s)
                d = [float(((a - b) ** 2).mean()) for a, b in zip(m, clean)]
                res.append(dict(net=net, kinds='+'.join(kinds), layer=l, seed=s, dmse=d))
        print(net, kinds, 'done', flush=True)
json.dump(res, open(sys.argv[4], 'w'))
