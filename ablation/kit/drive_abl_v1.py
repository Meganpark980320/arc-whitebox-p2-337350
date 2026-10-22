"""drive_abl_v1.py LINES CONC [RES] (on the pod, 22 Oct 2026): run every line 'ARM LINE SRC ENV POST IDS' of LINES with run_abl_v1.py,
CONC lines at a time, in file order (lines of different arms interleaved by the line generator).  Results /workspace/abl/RES/<ARM>/ (RES default res),
stdout/stderr per line in /workspace/abl/logs_RES/.  Log /workspace/abl/drive_RES.log; at the end it writes /workspace/abl/DONE_RES (non-empty:
'ABL_DONE ok N / M receipts')."""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
W = Path('/workspace/abl'); LINES, CONC = sys.argv[1], int(sys.argv[2]); RES = sys.argv[3] if len(sys.argv) > 3 else 'res'
jobs = []
for l in open(LINES):
    l = l.split('#')[0].strip()
    if l: arm, line, src, env, post, ids = l.split(); jobs.append((arm, line, src, env, post, ids))
(W / f'logs_{RES}').mkdir(parents=True, exist_ok=True)


def log(s):
    with open(W / f'drive_{RES}.log', 'a') as f: f.write(f'{time.strftime("%T")} {s}\n')


def one(j):
    arm, line, src, env, post, ids = j; t0 = time.time()
    with open(W / f'logs_{RES}' / f'{arm}_{line}.log', 'w') as lg:
        r = subprocess.run(['/workspace/venv/bin/python', str(W / 'run_abl_v1.py'), '--src', src, '--arm', arm, '--line', line, '--env', env,
                            '--post', post, '--ids', ids, '--out', str(W / RES)], stdout=lg, stderr=subprocess.STDOUT)
    log(f'{arm} line {line} rc={r.returncode} secs={time.time() - t0:.0f}')


log(f'start {len(jobs)} lines, conc {CONC}')
with ThreadPoolExecutor(CONC) as ex: list(ex.map(one, jobs))
want = sum(len([x for x in j[5].split(',') if x]) for j in jobs)
ok = 0
for f in (W / RES).rglob('[0-9]*.json'):
    try: ok += json.loads(f.read_text()).get('status') == 'OK'
    except Exception: pass
log(f'ABL_DONE ok {ok} / {want}')
(W / f'DONE_{RES}').write_text(f'ABL_DONE ok {ok} / {want} receipts\n')
