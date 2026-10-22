"""abl_an_v1.py (local, 22 Oct 2026): gate and analysis of the #337350 ablation (spec docs/writeup/v9/ABLATION-SPEC-v9.md §§4, 7, 8).
usage: abl_an_v1.py gate RESDIR          smoke gate: A twice identical, Z == A, every other arm differs from A on both nets, no guard event
                                          (the designed light second predict, dl proj -1, is not one), VmHWM < 12 GB (in-process: VmPeak
                                          counts server arrays too, so the official 8 GiB client cap does not apply), flips held
       abl_an_v1.py table RESDIR [...]   full run: one or more result folders (results/<pod>/full); per-arm table vs A on the paired nets
Receipts: <RESDIR>/<ARM>/<id>.json from run_abl_v1.py."""
import json, sys
from pathlib import Path
import numpy as np
MC_N = 2 ** 41 / 33_637_376          # Monte Carlo samples one budget buys (whestbench mc_samples_at_budget) = 65,374.4
SIGMA2_OFFICIAL = 1.132e-6 * MC_N    # fallback sigma^2 when a receipt has no avg_variance (the 50 official public networks)
ORDER = ['A', 'B', 'C', 'D', 'Dp', 'E', 'Ep', 'F', 'Fp']
LABEL = {'A': 'A  as shipped', 'B': 'B  learned mean corrections off', 'C': 'C  B + learned kappa4 parts off',
         'D': 'D  C + hand-set strengths derived, gains 1', 'Dp': "D' D + lambda physical", 'E': 'E  B + our six mechanisms off',
         'Ep': "E' D' + our six mechanisms off", 'F': 'F  starting point #332675', 'Fp': "F' starting point, fitted lambda stack off"}


def load(dirs):
    R = {}
    for d in dirs:
        for f in Path(d).glob('*/[0-9]*.json'):
            r = json.loads(f.read_text()); R.setdefault(f.parent.name, {})[r['id']] = r
    return R


def gate(d):
    R = load([d]); bad = []
    def sha(a, i): return R.get(a, {}).get(i, {}).get('pred_sha')
    for i in (0, 5):
        if not sha('A', i) or sha('A', i) != sha('Arep', i): bad.append(f'A not deterministic on {i}')
        if sha('Z', i) != sha('A', i): bad.append(f'Z != A on {i}')
        for a in R:
            if a in ('A', 'Arep', 'Z'): continue
            if sha(a, i) == sha('A', i): bad.append(f'{a} equals A on {i} (switch had no effect?)')
    for a, rs in sorted(R.items()):
        for i, r in sorted(rs.items()):
            vmp = (r.get('vm') or {}).get('VmPeak', 0) / 1024; hwm = (r.get('vm') or {}).get('VmHWM', 0) / 1024 / 1024
            tag = f"{a:5s} {i:3d} {r.get('status')} raw {r.get('raw', float('nan')):.4e} F {r.get('F', 0) / 1e9:7.1f} GF {r.get('predict_s', 0):6.1f} s VmPeak {vmp:6.0f} MiB"
            timed_dl = r.get('dl') and r['dl'].get('proj') != -1.0
            if r.get('guard') or timed_dl or (r.get('rg') or {}).get('levers'): tag += ' GUARD ' + json.dumps([r.get('guard'), r.get('dl')])[:150]; bad.append(f'guard event {a} {i}')
            if r.get('status') != 'OK': bad.append(f'{a} {i} {r.get("status")} {str(r.get("error"))[:120]}')
            if hwm >= 12: bad.append(f'{a} {i} VmHWM {hwm:.1f} GB')
            if r.get('flips_after') and not all(r['flips_after'].values()): bad.append(f'{a} {i} flip lost')
            print(tag, r.get('flips_after', ''), (r.get('cpu') or '')[:30])
    print('GATE', 'PASS' if not bad else 'FAIL'); [print('  -', b) for b in bad]
    return not bad


def boot(x, y, n=10000, seed=0):
    rng = np.random.default_rng(seed); k = len(x); idx = rng.integers(0, k, (n, k))
    r = x[idx].sum(1) / y[idx].sum(1); return np.quantile(r, [0.025, 0.975])


def table(dirs):
    R = load(dirs)
    ok = lambda a: {i for i, r in R.get(a, {}).items() if r.get('status') == 'OK'}
    arms = [a for a in ORDER if a in R]
    paired = sorted(set.intersection(*[ok(a) for a in arms]))
    print(f'paired networks: {len(paired)} (arms {arms})')
    chk = [a for a in R if a.startswith('Achk')]
    for a in chk:
        for i in (0, 5):
            s = R[a].get(i, {}).get('pred_sha'); print(f'cross-pod check {a} net {i}:', 'same as A' if s and s == R['A'].get(i, {}).get('pred_sha') else 'DIFFERENT', R[a].get(i, {}).get('cpu', '')[:30])
    sig = np.array([R['A'][i].get('avg_variance') or SIGMA2_OFFICIAL for i in paired])
    noav = sum(1 for i in paired if not R['A'][i].get('avg_variance'))
    if noav: print(f'note: {noav} receipts without avg_variance; sigma^2 = {SIGMA2_OFFICIAL:.4f} used')
    mA = np.array([R['A'][i]['raw'] for i in paired])
    half1 = np.array([i < 50 for i in paired])
    rows = []
    for a in arms:
        m = np.array([R[a][i]['raw'] for i in paired]); F = np.array([R[a][i]['F'] for i in paired], float)
        C = np.maximum(0.1, F / 2 ** 41); lo, hi = boot(m, mA)
        guards = sum(1 for i in paired if R[a][i].get('guard') or (R[a][i].get('dl') and R[a][i]['dl'].get('proj') != -1.0))
        per = m / mA
        rows.append(dict(arm=a, label=LABEL[a], GF=F.mean() / 1e9, GFmax=F.max() / 1e9, C=C.mean(), mse=m.mean(), mse_nc=(m - sig / 1e9).mean(),
                         ratio=m.sum() / mA.sum(), lo=lo, hi=hi, wins=int((m < mA).sum()), h1=m[half1].sum() / mA[half1].sum(),
                         h2=m[~half1].sum() / mA[~half1].sum(), worst=per.max(), worst_id=paired[int(per.argmax())],
                         nmatch=sig.sum() / m.sum(), budgets=sig.sum() / m.sum() / MC_N, vs_mc=(sig.sum() / (MC_N * C.mean())) / m.sum(),
                         guards=guards, layers=[float(np.mean([R[a][i]['all_mse'][l] for i in paired])) for l in range(16)]))
    print(f"{'arm':48s} {'GF':>6s} {'C':>6s} {'MSE':>10s} {'MSE-noise':>10s} {'ratio to A [95% CI]':>24s} {'wins':>5s} {'halves':>13s} {'worst':>12s} {'x MC@C':>7s} {'budgets':>8s} {'guards':>6s}")
    for r in rows:
        print(f"{r['label']:48s} {r['GF']:6.1f} {r['C']:6.3f} {r['mse']:10.4e} {r['mse_nc']:10.4e} {r['ratio']:7.3f} [{r['lo']:.3f}, {r['hi']:.3f}] {r['wins']:5d} "
              f"{r['h1']:6.3f}/{r['h2']:6.3f} {r['worst']:6.2f}@{r['worst_id']:<4d} {r['vs_mc']:7.0f} {r['budgets']:8.0f} {r['guards']:6d}")
    print('\nper-layer MSE ratio to A (layers 0..15):')
    LA = rows[0]['layers']
    for r in rows: print(f"{r['arm']:3s}", ' '.join(f'{x / y:5.2f}' for x, y in zip(r['layers'], LA)))
    out = Path(dirs[0]).parent.parent / 'abl_table_v1.json'
    out.write_text(json.dumps(dict(paired=paired, rows=rows), indent=1)); print('\nwritten', out)


if __name__ == '__main__':
    if sys.argv[1] == 'gate': sys.exit(0 if gate(sys.argv[2]) else 1)
    table(sys.argv[2:])
