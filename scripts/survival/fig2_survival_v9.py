"""fig2_survival_v9.py (22 Oct 2026): Figure 2 of the v9 write-up (§2), written to docs/writeup/assets/v9-fig2-survival.svg (refuses to
overwrite).  (a) survival to the output S(l): mean-field law (line), stored ledger (64 validation networks, 95 % CI), fresh networks
(12, mean +- s.e.).  (b) share of the final error by the layer where it was made: from the law and the per-layer MSE of the shipped build
(100 public mini networks, ablation arm A) against the ledger.  Palette: reference categorical slots 1-3, validated (dataviz skill)."""
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = Path('<team-repo>')
OUT = R / 'docs/writeup/assets/v9-fig2-survival.svg'
assert not OUT.exists(), OUT
LAW, LEDGER, FRESH = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff'
plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': 0.8})
sim = json.load(open(R / 'docs/writeup/v9/calc/survival_law_v9_12nets.out.json'))['part_c_simulation']
law = np.array(sim['surv_law'])                                   # l = 0..14
fr = np.array([p['surv_rand'] for p in sim['per_net']]); fm, fse = fr.mean(0), fr.std(0, ddof=1) / math.sqrt(len(fr))
L = json.load(open(R / 'experiments/claude_explore_20261014/ledger9a/budget_matrix_9A.json'))
lay = [int(x[1:]) for x in L['layers'][1:-1]]                     # L6..L14
lv = np.array(L['survive'][1:-1]); lci = np.array(L['survive_ci95'][1:-1])
# (b) shares by creation layer
f = lambda c: (math.sqrt(max(0, 1 - c * c)) + (math.pi - math.acos(c)) * c) / math.pi
c = [0.0]
for _ in range(15): c.append(f(c[-1]))
chi = [1 - math.acos(x) / math.pi for x in c]
S = [math.prod(chi[k] for k in range(l + 1, 16)) for l in range(16)]
A = next(r for r in json.load(open(R / 'experiments/claude_abl_20261022/results/abl_table_v1.json'))['rows'] if r['arm'] == 'A')['layers']
new = [A[0]] + [A[l] - chi[l] * A[l - 1] for l in range(1, 16)]
share = [100 * new[l] * S[l] / A[15] for l in range(16)]
groups = [(0, 5), (6, 9), (10, 12), (13, 14), (15, 15)]
law_g = [sum(share[a:b + 1]) for a, b in groups]
rt = L['row_total']                                               # [L0-5, L6, ..., L15]
led_g = [rt[0], sum(rt[1:5]), sum(rt[5:8]), sum(rt[8:10]), rt[10]]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={'width_ratios': [1.15, 1]})
fig.patch.set_facecolor(SURF)
for a in (ax, bx):
    a.set_facecolor(SURF); a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.grid(axis='y', color=GRID, linewidth=0.6); a.set_axisbelow(True)
ls = np.arange(15)
ax.plot(ls, law, color=LAW, linewidth=2, label='mean-field law, no fit', zorder=3)
ax.errorbar(ls, fm, yerr=1.96 * fse, fmt='D', ms=4.5, color=FRESH, mec=SURF, mew=0.8, elinewidth=1, capsize=0, label='12 fresh networks', zorder=4)
ax.errorbar(lay, lv, yerr=[lv - lci[:, 0], lci[:, 1] - lv], fmt='o', ms=5.5, color=LEDGER, mec=SURF, mew=0.8, elinewidth=1.2, capsize=0,
            label='our error ledger, 64 networks', zorder=5)
ax.set_xlabel('layer ℓ where the error is made'); ax.set_ylabel('share of its energy reaching layer 15')
ax.set_xlim(-0.5, 14.6); ax.set_ylim(0, 1.0); ax.set_xticks(range(0, 15, 2))
ax.text(1.0, 0.12, 'law', color=INK, fontsize=8.5); ax.text(4.2, 0.30, 'fresh networks', color=INK, fontsize=8.5)
ax.text(9.8, 0.36, 'ledger', color=INK, fontsize=8.5)
ax.set_title('(a) How much of an error survives', loc='left', fontsize=9.5, color=INK)
ax.legend(frameon=False, fontsize=8, loc='upper left')
x = np.arange(len(groups)); w = 0.36
bx.bar(x - w / 2 - 0.01, law_g, w, color=LAW, label='law + per-layer MSE (no fit)', zorder=3)
bx.bar(x + w / 2 + 0.01, led_g, w, color=LEDGER, label='error ledger', zorder=3)
bx.set_xticks(x); bx.set_xticklabels([f'{a}–{b}' if a != b else f'{a}' for a, b in groups])
bx.set_xlabel('layers where the final error was made'); bx.set_ylabel('share of final-layer error (%)')
bx.set_ylim(0, 36); bx.set_title('(b) Where the final error was made', loc='left', fontsize=9.5, color=INK)
bx.legend(frameon=False, fontsize=8, loc='upper left')
fig.tight_layout(w_pad=2.5)
fig.savefig(OUT, format='svg', facecolor=SURF)
print('written', OUT, '| shares law', [round(v, 1) for v in law_g], 'ledger', [round(v, 1) for v in led_g])
