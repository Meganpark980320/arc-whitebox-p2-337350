"""abl_an_v2.py (local, 22 Oct 2026): second ablation round.  (1) mini: each new arm against its round-1 base on the same nets
(base A for A1lam/A1fbb/A1mech, D for D0, D' for the three strength arms, F for Fphys); all 100 nets, and the 60 nets whose two
runs had the same CPU family (cross-box byte families differ in the last bits).  (2) full split, nets 144-191 and 960-975 (used once):
the main arms against A, as abl_an_v1 table.  Writes results/abl_table_v2.json."""
import glob, json
from pathlib import Path
import numpy as np
MC_N = 2 ** 41 / 33_637_376
def load(pat):
    R = {}
    for f in glob.glob(pat):
        r = json.load(open(f)); R.setdefault(Path(f).parent.name, {})[r['id']] = r
    return R
def boot(x, y, n=10000, seed=0):
    rng = np.random.default_rng(seed); k = len(x); idx = rng.integers(0, k, (n, k))
    return np.quantile(x[idx].sum(1) / y[idx].sum(1), [0.025, 0.975])
R1 = load('results/explore-abl[1-5]/full/*/[0-9]*.json')
R2 = load('results/explore-abl*/r2/*/[0-9]*.json')
out = {'mini': [], 'full': []}
print('(1) mini: new arm vs round-1 base (ratio of summed final-layer MSE; 95% bootstrap; wins = nets better than base)')
for arm, base in (('mini_A1lam', 'A'), ('mini_A1fbb', 'A'), ('mini_A1mech', 'A'), ('mini_D0', 'D'), ('mini_Dpfbbq', 'Dp'),
                  ('mini_Dpc2e', 'Dp'), ('mini_Dpo1', 'Dp'), ('mini_Fphys', 'F')):
    ids = sorted(i for i, r in R2[arm].items() if r['status'] == 'OK' and R1[base].get(i, {}).get('status') == 'OK')
    x = np.array([R2[arm][i]['raw'] for i in ids]); y = np.array([R1[base][i]['raw'] for i in ids])
    same = [k for k, i in enumerate(ids) if R2[arm][i]['cpu'][:20] == R1[base][i]['cpu'][:20]]
    lo, hi = boot(x, y); xs, ys = x[same], y[same]; los, his = boot(xs, ys)
    F = np.mean([R2[arm][i]['F'] for i in ids]) / 1e9; C = np.mean([max(0.1, R2[arm][i]['F'] / 2 ** 41) for i in ids])
    sig = np.array([R2[arm][i].get('avg_variance') or 0.0740 for i in ids]); vs = (sig.sum() / (MC_N * C)) / x.sum()
    print(f'  {arm:12s} vs {base:2s}: {x.sum() / y.sum():.4f} [{lo:.4f}, {hi:.4f}] wins {int((x < y).sum()):3d}/{len(ids)} | same-CPU {len(same)}: {xs.sum() / ys.sum():.4f} [{los:.4f}, {his:.4f}] | {F:.1f} GF, x MC {vs:.0f}')
    out['mini'].append(dict(arm=arm, base=base, n=len(ids), ratio=x.sum() / y.sum(), ci=[lo, hi], wins=int((x < y).sum()), n_same=len(same),
                            ratio_same=xs.sum() / ys.sum(), ci_same=[los, his], GF=F, C=C, vs_mc=vs, mse=x.mean()))
print('\n(2) full split, once-used nets 144-191, 960-975: ratio to A, x MC at equal compute')
arms = ['full_A', 'full_B', 'full_C', 'full_D', 'full_Dp', 'full_E', 'full_F', 'full_Fp']
paired = sorted(set.intersection(*[{i for i, r in R2[a].items() if r['status'] == 'OK'} for a in arms]))
mA = np.array([R2['full_A'][i]['raw'] for i in paired]); sig = np.array([R2['full_A'][i].get('avg_variance') or 0.0740 for i in paired])
print(f'  paired nets: {len(paired)}; avg_variance present: {sum(1 for i in paired if R2["full_A"][i].get("avg_variance"))}')
for a in arms:
    m = np.array([R2[a][i]['raw'] for i in paired]); C = np.mean([max(0.1, R2[a][i]['F'] / 2 ** 41) for i in paired])
    lo, hi = boot(m, mA); vs = (sig.sum() / (MC_N * C)) / m.sum(); h = np.array([i < 192 for i in paired])
    print(f'  {a:8s} MSE {m.mean():.4e} ratio {m.sum() / mA.sum():.4f} [{lo:.4f}, {hi:.4f}] wins {int((m < mA).sum()):2d}/{len(paired)} '
          f'shards 9-11 / 60: {m[h].sum() / mA[h].sum():.3f} / {m[~h].sum() / mA[~h].sum():.3f} | C {C:.3f} x MC {vs:.0f}')
    out['full'].append(dict(arm=a, mse=m.mean(), ratio=m.sum() / mA.sum(), ci=[lo, hi], wins=int((m < mA).sum()), C=C, vs_mc=vs))
st = {}
for a, rs in R2.items():
    for r in rs.values(): st[r['status']] = st.get(r['status'], 0) + 1
timed = sum(1 for rs in R2.values() for r in rs.values() if r.get('dl') and r['dl'].get('proj') != -1.0)
lost = sum(1 for rs in R2.values() for r in rs.values() if r.get('flips_after') and not all(r['flips_after'].values()))
print('\nstatus', st, '| timed guard events', timed, '| flips lost', lost)
Path('results/abl_table_v2.json').write_text(json.dumps(out, indent=1, default=float)); print('written results/abl_table_v2.json')
