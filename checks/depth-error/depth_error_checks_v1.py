"""depth_error_checks_v1.py [out.txt]  (v9 write-up, section 3). Reads stored receipts only; runs no estimator.

Recomputes the numbers section 3 quotes from small JSON files:
  A. ledger groups (repo: experiments/claude_explore_20261014/ledger9a/budget_matrix_9A.json)
  B. one-step true-input test per layer (~/arcwb_data/onestep/544_n21.json, 624_n21v2.json):
     new error R (engine step without its kappa4->kappa3 feed), R after adding the true supply X, share of R removed by X
     (raw and after subtracting each energy's half-split noise), noise of the truth T, transport / birth shares
  C. growth curve of the relative three-index error (~/arcwb_data/oldtri_t2/res/n0/*.d21.json, 14 tuning networks)
     and the relaxation check e_{l+1} = r e_l + s_l with r = 0.8 (consistency check, not a fit)
  D. Monte Carlo noise of the 2^24 truth against the three-index error it is compared with
     (~/arcwb_data/oldtri_s1/res/b/*.d21.json, field noise_K21 = |half A - half B| / (2 |full|))
  E. kappa4 memory and per-layer split (~/arcwb_data/onestep/624_gh_L12.json, 624_lin.json)
Energies are fractions of |T|^2 on the receipts' row subset (one-step) or support (oracle logs)."""
import glob
import json
import os
import sys

import numpy as np

REPO = '<team-repo>'
DATA = '<local-path>'
out = open(sys.argv[1], 'w') if len(sys.argv) > 1 else None


def say(s=''):
    print(s)
    if out:
        out.write(s + '\n')


# ---------------------------------------------------------------- A. ledger
B = json.load(open(f'{REPO}/experiments/claude_explore_20261014/ledger9a/budget_matrix_9A.json'))
lay, sh, sv = B['layers'], B['share'], B['survive']
groups = {'L6-9': ['L6', 'L7', 'L8', 'L9'], 'L10-12': ['L10', 'L11', 'L12'], 'L13-14': ['L13', 'L14'], 'L15': ['L15']}
say('== A. ledger (64 validation networks, %% of last-layer error energy); channels: variance | k3+closure+heads | k4 after corrector')
for g, ls in groups.items():
    v = np.array([sh[lay.index(l)] for l in ls]).sum(0)
    say('  %-6s %6.2f %6.2f %6.2f  row %6.2f' % (g, v[0], v[1], v[2], v.sum()))
say('  L0-5 (unsplit) %.2f; column totals %s' % (B['front_L0_5_unsplit']['share'], B['col_total']))
say('  variance share / survival (new variance error per layer, as %% of last-layer error once undone by fading):')
say('   ' + ' '.join('%s %.1f' % (l, sh[i][0] / sv[i]) for i, l in enumerate(lay) if i > 0))
say('  per-layer survival ratio S(l)/S(l+1): ' + ' '.join('%s %.3f' % (lay[i], sv[i] / sv[i + 1]) for i in range(1, len(lay) - 1)))
say()

# ---------------------------------------------------------------- B. one-step
say('== B. one-step true-input test (energies in % of |T|^2; n = noise energy of the same quantity, half-split)')
say('  net layer | R new  (n)   | R after true X (n) | X share raw / noise-corr | X (n share) | T noise | npz diff | transport | birth | cos(birth,211) | cos(R,X)')
rows = []
for f, rk in (('544_n21.json', 'R_fbb'), ('624_n21v2.json', 'R_eng0')):
    o = json.load(open(f'{DATA}/onestep/{f}'))
    for l, r in o['res'].items():
        it = r['items']
        R, nR = it[rk]['rel'], it[rk]['noise']
        K, nK = it['R_k4']['rel'], it['R_k4']['noise']
        X, nX = it['k4sup']['rel'], it['k4sup']['noise']
        raw = 1 - K / R
        cor = 1 - (K - nK) / (R - nR)
        cos = r['cos'].get('R_eng0~k4sup', r['cos'].get('R_fbb~k4sup'))
        f68 = it['R_f68nom']['rel'] if 'R_f68nom' in it else float('nan')
        rows.append((o['id'], int(l), R, nR, K, nK, raw, cor, X, nX, f68))
        say('  %d L%-2s | %.3f (%.3f) | %.3f (%.3f) | %.3f / %.3f | %.3f (%.0f%%) | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f%s' % (
            o['id'], l, 100 * R, 100 * nR, 100 * K, 100 * nK, raw, cor, 100 * X, 100 * nX / X, 100 * it['T']['noise'],
            100 * r['npz_rel_diff'], it['111']['rel'], it['birth']['rel'], r['cos']['birth~211'], cos,
            '' if f68 != f68 else ' | with F68 form in base: R %.3f, share of it removed by X %.3f' % (100 * f68, 1 - K / f68)))
for net in (544, 624):
    rr = [x for x in rows if x[0] == net]
    say('  pooled %d: share 1 - sum(R after X) / sum(R) = %.3f ; noise-corrected %.3f' % (
        net, 1 - sum(x[4] for x in rr) / sum(x[2] for x in rr), 1 - sum(x[4] - x[5] for x in rr) / sum(x[2] - x[3] for x in rr)))
say('  per-layer ranges: R new %.2f-%.2f %%, after X %.2f-%.2f %%, share raw %.2f-%.2f, noise-corrected %.2f-%.2f, residual noise share %.0f-%.0f %% of R' % (
    100 * min(x[2] for x in rows), 100 * max(x[2] for x in rows), 100 * min(x[4] for x in rows), 100 * max(x[4] for x in rows),
    min(x[6] for x in rows), max(x[6] for x in rows), min(x[7] for x in rows), max(x[7] for x in rows),
    100 * min(x[3] / x[2] for x in rows), 100 * max(x[3] / x[2] for x in rows)))
say('  mean new error R, network 544: %.3f %% (standing chain error quoted for 544 in onestep README section 2: 1.36-1.70 %%, mean 1.56 %%)' % (
    100 * np.mean([x[2] for x in rows if x[0] == 544])))
m = np.mean([x[2] for x in rows if x[0] == 544])
say('  retention 1 - new/standing = 1 - %.3f/1.56 = %.2f ; allowing growth g = 1.09 of the standing error: g - new/standing = %.2f' % (
    100 * m, 1 - 100 * m / 1.56, 1.09 - 100 * m / 1.56))
say()

# ---------------------------------------------------------------- C. growth curve + relaxation check
logs = {}
for f in glob.glob(f'{DATA}/oldtri_t2/res/n0/*.d21.json'):
    logs[int(os.path.basename(f).split('.')[0])] = {r['li']: r for r in json.load(open(f))}
say('== C. growth curve, relative three-index error sqrt(sum e_Et / sum |K*-P*|^2), %d networks %s' % (len(logs), sorted(logs)))
rel = {}
for li in range(6, 15):
    rs = [d[li] for d in logs.values() if li in d]
    rel[li] = float(np.sqrt(sum(r['e_Et'] for r in rs) / sum(r['e_Kt'] - r['e_Pt'] for r in rs)))
say('  ' + ' '.join('L%d %.4f' % (li, rel[li]) for li in rel))
e = {li: 100 * rel[li] ** 2 for li in rel}
say('  energy e_l (%% of |T|^2): ' + ' '.join('L%d %.3f' % (li, e[li]) for li in e))
say('  per-step energy growth: ' + ' '.join('L%d->%d x%.3f' % (li, li + 1, e[li + 1] / e[li]) for li in range(6, 14)))
say('  mean growth L6->L14: x%.4f per layer' % ((e[14] / e[6]) ** (1 / 8)))
r = 0.8
s = {li: e[li] - r * e[li - 1] for li in range(10, 15)}
say('  implied source s_l = e_l - %.1f e_(l-1): ' % r + ' '.join('L%d %.3f' % (li, s[li]) for li in s) + '  (%% of |T|^2)')
say('  plateau s/(1-r) for s = 0.33 / 0.50 (one-step range): %.2f / %.2f %% -> relative error %.3f / %.3f' % (
    0.33 / (1 - r), 0.50 / (1 - r), np.sqrt(0.0033 / (1 - r)), np.sqrt(0.0050 / (1 - r))))
say()

# ---------------------------------------------------------------- D. truth noise vs three-index error
say('== D. MC noise of the 2^24 truth (noise_K21^2 |K*|^2) as a share of the measured three-index error energy e_Et, oracle base arm, 16 networks')
acc = {}
for f in glob.glob(f'{DATA}/oldtri_s1/res/b/*.d21.json'):
    for r_ in json.load(open(f)):
        a = acc.setdefault(r_['li'], [0.0, 0.0, []])
        a[0] += r_['noise_K21'] ** 2 * r_['e_Kt']; a[1] += r_['e_Et']; a[2].append(r_['noise_K21'])
say('  ' + ' '.join('L%d noise/err %.3f (rel noise %.4f)' % (li, a[0] / a[1], np.mean(a[2])) for li, a in sorted(acc.items())))
say()

# ---------------------------------------------------------------- E. kappa4 memory and split
gh = json.load(open(f'{DATA}/onestep/624_gh_L12.json'))
say('== E. GH counterfactual, network 624, slot layer %d: k, beta, rho' % gh['l'])
say('  ' + ' | '.join('k=%s beta %.3f rho %.3f' % (k, v['beta'], v['rho']) for k, v in gh['res'].items()))
lin = json.load(open(f'{DATA}/onestep/624_lin.json'))
say('== E. per-layer split of X*, network 624 (input layer p -> slot p+1)')
for p, v in lin['res'].items():
    say('  slot L%d: carried beta %.3f rho %.3f | new beta %.3f rho %.3f | beta sum %.3f | cos(carried,new) %.3f | birth share of new %.3f (cos %.3f)' % (
        int(p) + 1, v['beta_lin'], v['rho_lin'], v['beta_new'], v['rho_new'], v['beta_lin'] + v['beta_new'], v['cos_lin_new'],
        v['beta_birth_in_new'], v['cos_birth_new']))
if out:
    out.close()
