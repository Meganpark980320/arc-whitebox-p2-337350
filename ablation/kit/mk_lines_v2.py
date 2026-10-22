"""mk_lines_v2.py (22 Oct 2026): second ablation round.  Line = 'KIND ARM LINE SRC ENV POST IDS' (drive_abl_v2.py).  Box w (w = 0..4):
(1) the new arms on mini nets w, w+5, ..., w+95 (single switches on A; D' with shipped strengths; D with lambda = 0; the start with physical
lambda); (2) the main arms on the once-used public full nets 144-191 (shards 9-11) and 960-975 (shard 60), split as nets[w::5]."""
from pathlib import Path
R = '/workspace/abl'
L, S, V, P = f'{R}/src/linked/estimator.py', f'{R}/src/start/estimator.py', f'{R}/src/start_v25off/estimator.py', f'{R}/src/start_lamphys/estimator.py'
e = lambda n: f'{R}/envs/{n}.txt'
NEW = {'A1lam': (L, e('A1lam'), 'noguard'), 'A1fbb': (L, e('A1fbb'), 'noguard'), 'A1mech': (L, e('A1mech'), 'noguard'),
       'D0': (L, e('D0'), 'noguard,nocorr,a14off'), 'Dpfbbq': (L, e('Dp_fbbq'), 'noguard,nocorr,a14off'),
       'Dpc2e': (L, e('Dp_c2e065'), 'noguard,nocorr,a14off'), 'Dpo1': (L, e('Dp_o115'), 'noguard,nocorr,a14off'), 'Fphys': (P, e('Fphys'), '-')}
MAIN = {'A': (L, '-', 'noguard'), 'B': (L, '-', 'noguard,nocorr'), 'C': (L, e('C'), 'noguard,nocorr,a14off'),
        'D': (L, e('D'), 'noguard,nocorr,a14off'), 'Dp': (L, e('Dp'), 'noguard,nocorr,a14off'), 'E': (L, e('E'), 'noguard,nocorr'),
        'F': (S, '-', '-'), 'Fp': (V, '-', '-')}
once = list(range(144, 192)) + list(range(960, 976))
for w in range(5):
    ls = [f'mini {a} {w} {s} {en} {p} ' + ','.join(map(str, range(w, 100, 5))) for a, (s, en, p) in NEW.items()]
    ls += [f'full {a} {w} {s} {en} {p} ' + ','.join(map(str, once[w::5])) for a, (s, en, p) in MAIN.items()]
    Path(f'lines2_w{w}.txt').write_text('\n'.join(ls) + '\n')
    print('box', w, len(ls), 'lines', sum(len(l.split()[-1].split(',')) for l in ls), 'predicts')
