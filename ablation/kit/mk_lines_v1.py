"""mk_lines_v1.py (22 Oct 2026): write the line files of the #337350 ablation (pod paths under /workspace/abl).
Line = 'ARM LINE SRC ENV POST IDS'.  Official-worker layout: line w predicts nets w, w+5, ..., w+95 of the public mini split, in order.
smoke: every arm (and a repeat of A) on nets 0,5 (line 0).  full: one file per worker w (9 arms x 1 line), one box per worker."""
from pathlib import Path
R = '/workspace/abl'
LINKED, START, V25OFF = f'{R}/src/linked/estimator.py', f'{R}/src/start/estimator.py', f'{R}/src/start_v25off/estimator.py'
ARMS = {  # arm: (src, env, post); noguard on every arm of the linked build (time guards cannot fire; run_abl_v1 doc)
    'A': (LINKED, '-', 'noguard'), 'Z': (LINKED, f'{R}/envs/Z.txt', 'noguard'), 'B': (LINKED, '-', 'noguard,nocorr'),
    'C': (LINKED, f'{R}/envs/C.txt', 'noguard,nocorr,a14off'), 'D': (LINKED, f'{R}/envs/D.txt', 'noguard,nocorr,a14off'),
    'Dp': (LINKED, f'{R}/envs/Dp.txt', 'noguard,nocorr,a14off'), 'E': (LINKED, f'{R}/envs/E.txt', 'noguard,nocorr'),
    'Ep': (LINKED, f'{R}/envs/Ep.txt', 'noguard,nocorr,a14off'), 'F': (START, '-', '-'), 'Fp': (V25OFF, '-', '-')}
def line(arm, w, ids, name=None):
    s, e, p = ARMS[arm]; return f'{name or arm} {w} {s} {e} {p} {",".join(map(str, ids))}'
smoke = [line(a, 0, [0, 5]) for a in ARMS] + [line('A', 0, [0, 5], name='Arep')]
Path('lines_smoke.txt').write_text('\n'.join(smoke) + '\n')
full_arms = ['A', 'B', 'C', 'D', 'Dp', 'E', 'Ep', 'F', 'Fp']
for w in range(5):   # one box per official worker: every arm of a net runs on the same box (same-machine pairs, whatever its CPU family)
    ls = [line(a, w, list(range(w, 100, 5))) for a in full_arms]
    Path(f'lines_full_w{w}.txt').write_text('\n'.join(ls) + '\n'); print('worker', w, len(ls), 'lines', 20 * len(ls), 'predicts')
print('smoke', len(smoke), 'lines')
