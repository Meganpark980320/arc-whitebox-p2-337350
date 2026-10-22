"""run_abl_v2.py (22 Oct 2026) = run_abl_v1.py (unchanged, used for the first ablation) + --kind full: nets of the public full split
(full-%05d-of-00067.parquet in --hfdata, shard = id // 16).  run_abl_v1 doc: (write-up v9 ablation of #337350; derived from run_net5 / run_net12).
One LINE = one arm in one process, like an official worker: apply the arm's env file, import the build, setup once, apply the post-setup
flips, then predict the listed public mini nets in the given order (no warm-up).  In-process flopscope (pod venv), BLAS 1 thread.
Receipts: OUT/<ARM>/<id>.json per net (status, raw = layer-15 MSE vs the 1e9-sample truth, all_mse, pred_sha, F, calls, times, avg_variance,
guard/dl/rg logs, flip state after the predict, VmPeak/VmHWM), OUT/<ARM>/_setup_<LINE>.json, OUT/<ARM>/_end_<LINE>.json.
A line stops (rest NOT_RUN) when a post-setup flip no longer holds after a predict (sign of a safety-net re-setup).
POST token noguard (applied right after import, every arm of the linked build): the time-based guards cannot fire, so values do not depend
on machine speed: _CF_RG_ON = False (residual guard and its levers, incl. the head lever and the sticky rule, all act through _cf_rg_step),
_CF_DL_TRIGGER_S = _CF_DL_TRIGGER2_S = 1e9 (deadline guard).  The position-based light first predicts and the exception safety net stay.
usage: run_abl_v2.py --kind mini|full [--hfdata /workspace/hf/data] ...; v1: --src EST --arm ARM --line L --env FILE|- --post noguard,nocorr,a14off|- --ids 0,5,10 --out DIR [--data /workspace/hfmini/data]"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'): os.environ[_k] = '1'
import argparse, hashlib, importlib.util, json, sys, time, traceback
from pathlib import Path
import numpy as np
BUDGET = 2 ** 41
SHA = lambda b: hashlib.sha256(b).hexdigest()


def vm():
    out = {}
    for l in Path('/proc/self/status').read_text().splitlines():
        k = l.split(':')[0]
        if k in ('VmPeak', 'VmHWM'): out[k] = int(l.split()[1])   # kB
    return out


def arrow(table, key, row, shape):
    import pyarrow as pa
    v = table[key].slice(row, 1).combine_chunks(); v = getattr(v, 'storage', v)
    while pa.types.is_list(v.type) or pa.types.is_large_list(v.type) or pa.types.is_fixed_size_list(v.type): v = v.flatten()
    return np.asarray(v.to_numpy(zero_copy_only=False), np.float32).reshape(shape)


def load_mini(d, i, kind='mini'):
    import pyarrow.parquet as pq
    fp = f'{d}/mini-{i // 16:05d}-of-00007.parquet' if kind == 'mini' else f'{d}/full-{i // 16:05d}-of-00067.parquet'
    have = set(pq.read_schema(fp).names)
    cols = ['mlp_id', 'mlp_name', 'mlp_seed', 'weights', 'all_layer_means'] + (['avg_variance'] if 'avg_variance' in have else [])
    t = pq.read_table(fp, columns=cols, filters=[('mlp_id', '=', i)], use_threads=False)
    ids = t['mlp_id'].to_pylist()
    if ids.count(i) != 1: raise RuntimeError(f'mini {i}: {ids.count(i)} rows in {fp}')
    row = ids.index(i)
    av = float(t['avg_variance'][row].as_py()) if 'avg_variance' in cols else None
    return (arrow(t, 'weights', row, (16, 1024, 1024)), arrow(t, 'all_layer_means', row, (16, 1024)), str(t['mlp_name'][row].as_py()),
            int(t['mlp_seed'][row].as_py()), av, fp)


def main():
    p = argparse.ArgumentParser()
    for k in ('--src', '--arm', '--line', '--env', '--post', '--ids', '--out'): p.add_argument(k, required=True)
    p.add_argument('--data', default='/workspace/hfmini/data')
    p.add_argument('--kind', choices=['mini', 'full'], default='mini'); p.add_argument('--hfdata', default='/workspace/hf/data')
    a = p.parse_args()
    out = Path(a.out) / a.arm; out.mkdir(parents=True, exist_ok=True)
    ids = [int(x) for x in a.ids.split(',') if x]
    if any((out / f'{i}.json').exists() for i in ids): sys.exit(f'refuse: receipts for line {a.line} already exist in {out}')
    envs = {}
    if a.env != '-':
        for l in Path(a.env).read_text().splitlines():
            l = l.strip()
            if l and not l.startswith('#') and '=' in l: k, v = l.split('=', 1); envs[k.strip()] = v.strip()
    os.environ.update(envs)
    post = [x for x in a.post.split(',') if x and x != '-']
    assert set(post) <= {'nocorr', 'a14off', 'noguard'}, post
    cpu = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), '?')
    base = dict(kind=a.kind, arm=a.arm, line=int(a.line), src=a.src, src_sha=SHA(Path(a.src).read_bytes()), runner_sha=SHA(Path(__file__).read_bytes()),
                env=envs, post=post, cpu=cpu)
    def write(name, rec):
        tmp = out / f'.{name}.tmp'; tmp.write_text(json.dumps(rec, default=repr) + '\n'); os.replace(tmp, out / name)
    spec = importlib.util.spec_from_file_location(f'abl_{a.arm}_{a.line}_{os.getpid()}', a.src)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    import flopscope as fl, flopscope.numpy as xp
    from whestbench import MLP, SetupContext
    from threadpoolctl import threadpool_limits
    versions = dict(flopscope=getattr(fl, '__version__', '?'), numpy=np.__version__)
    try:
        import whestbench; versions['whestbench'] = getattr(whestbench, '__version__', '?')
    except Exception: pass
    base['versions'] = versions
    t0 = time.perf_counter()
    spec.loader.exec_module(m)
    NOGUARD = (('_CF_RG_ON', False), ('_CF_DL_TRIGGER_S', 1e9), ('_CF_DL_TRIGGER2_S', 1e9))
    if 'noguard' in post:
        base['noguard_applied'] = [n for n, v in NOGUARD if hasattr(m, n)]
        for n, v in NOGUARD:
            if hasattr(m, n): setattr(m, n, v)
    est = m.Estimator()

    def do_setup():
        with fl.BudgetContext(flop_budget=4 * BUDGET, wall_time_limit_s=1000, quiet=True) as sc:
            est.setup(SetupContext(width=1024, depth=16, flop_budget=BUDGET, api_version='1.0', submission_dir=str(Path(a.src).parent), seed=0))
        if 'nocorr' in post: m.NO_CORR = True
        if 'a14off' in post: est.core._law_arrays = None
        return int(sc.flops_used)

    def flips():
        st = {}
        if 'nocorr' in post: st['nocorr'] = bool(getattr(m, 'NO_CORR', None))
        if 'a14off' in post: st['a14off'] = getattr(getattr(est, 'core', None), '_law_arrays', 'missing') is None
        if 'noguard' in post: st['noguard'] = all(getattr(m, n) == v for n, v in NOGUARD if hasattr(m, n))
        return st

    with threadpool_limits(limits=1):
        Fs = do_setup()
        gl0 = list(getattr(est, '_cf_guard_log', []) or [])
        write(f'_setup_{a.line}.json', dict(base, event='setup', secs=time.perf_counter() - t0, F_setup=Fs, flips=flips(),
                                            setup_guard=[str(x)[:300] for x in gl0], vm=vm()))
        glen = len(gl0); stop = None
        for seq, i in enumerate(ids, 1):
            rec = dict(base, id=i, seq=seq, n_in_line=len(ids))
            if stop:
                rec.update(status='NOT_RUN', error=stop); write(f'{i}.json', rec); continue
            t1 = time.perf_counter()
            try:
                W, truth, name, seed, av, fp = load_mini(a.data if a.kind == 'mini' else a.hfdata, i, a.kind)
                rec.update(mlp_name=name, mlp_seed=seed, avg_variance=av, data_file=os.path.basename(fp),
                           weights_sha16=SHA(np.ascontiguousarray(W).view(np.uint8))[:16])
                mlp = MLP(width=1024, depth=16, weights=[xp.asarray(x, dtype=xp.float32) for x in W], seed=seed, name=name); del W
            except BaseException as e:
                rec.update(status='INFRA', error='load: ' + repr(e)[:600], tb=traceback.format_exc()[-3000:]); write(f'{i}.json', rec)
                stop = f'line stopped after a load failure on {i}'; continue
            try:
                s = time.perf_counter()
                with fl.BudgetContext(flop_budget=4 * BUDGET, wall_time_limit_s=1000, quiet=True) as ctx:
                    pred = est.predict(mlp, BUDGET)
                rec['predict_s'] = time.perf_counter() - s
                try: rec['residual_s'] = ctx.residual_wall_time_s
                except Exception: pass
                pr = np.asarray(pred, dtype=np.float32)
                if pr.shape != (16, 1024): raise RuntimeError(f'prediction shape {pr.shape}')
                e64 = pr.astype(np.float64) - truth.astype(np.float64); F = int(ctx.flops_used)
                rec.update(status='OK', raw=float(np.mean(e64[-1] ** 2)), all_mse=[float(np.mean(e64[l] ** 2)) for l in range(16)], F=F,
                           calls=len(getattr(ctx, 'op_log', None) or []), finite=bool(np.isfinite(pr).all()), pred_sha=SHA(np.ascontiguousarray(pr).tobytes()))
            except BaseException as e:
                rec.update(status='FAILED', error=repr(e)[:800], tb=traceback.format_exc()[-5000:], predict_s=time.perf_counter() - t1)
                try:
                    do_setup(); rec['resetup'] = 'ok'
                except BaseException as e2:
                    rec['resetup'] = 'failed: ' + repr(e2)[:300]; stop = f'line stopped: re-setup failed after {i}'
            gl = list(getattr(est, '_cf_guard_log', []) or [])
            if len(gl) > glen: rec['guard'] = [str(x)[:300] for x in gl[glen:]]
            glen = len(gl)
            dl = getattr(est, '_cf_dl_last', None)
            if dl and isinstance(dl, dict) and dl.get('level'): rec['dl'] = {k: (round(x, 1) if isinstance(x, float) else x) for k, x in dl.items()}
            rg = getattr(est, '_cf_rg_last', None)
            if rg is not None: rec['rg'] = json.loads(json.dumps(rg, default=repr))
            rec['flips_after'] = flips(); rec['vm'] = vm(); rec['secs'] = time.perf_counter() - t1
            if post and not all(rec['flips_after'].values()): stop = f'line stopped: a post-setup flip no longer holds after {i}'
            write(f'{i}.json', rec)
            print(json.dumps({k: rec.get(k) for k in ('arm', 'line', 'id', 'seq', 'status', 'raw', 'F', 'predict_s')}), flush=True)
            try: del mlp
            except NameError: pass
        write(f'_end_{a.line}.json', dict(base, event='end', secs=time.perf_counter() - t0, stop=stop))


if __name__ == '__main__':
    main()
