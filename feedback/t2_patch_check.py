"""T2. FLOP counts, output bytes and peak memory of six einsums; run stock and with PYTHONPATH=<patched copy>, then compare."""
import sys, tracemalloc, hashlib
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp

rng = np.random.default_rng(7)
def arr(*s): return fnp.asarray(rng.standard_normal(s, dtype=np.float32))
A = arr(8, 256, 256); B = arr(8, 256, 256); c = arr(8, 256)
M = arr(64, 64); S_ = fnp.asarray((lambda m: (m + m.T) / 2)(rng.standard_normal((64, 64)).astype(np.float32)))
X = arr(128, 64); Y = arr(64, 32); Z = arr(32, 16)
cases = {
    "kij,kij,kj->i (A,B,c)": ("kij,kij,kj->i", (A, B, c)),
    "kij,kij,kj->i (A,A,c) repeated": ("kij,kij,kj->i", (A, A, c)),
    "ij,jk,kl->il chain": ("ij,jk,kl->il", (X, Y, Z)),
    "ik,kl,jl->ij sandwich (X,M,X)": ("ik,kl,jl->ij", (X, M, X)),
    "ik,kl,jl->ij sandwich (X,S,X) sym": ("ik,kl,jl->ij", (X, fnp.as_symmetric(S_, symmetric_axes=(0, 1)) if hasattr(fnp, "as_symmetric") else S_, X)),
    "ab,bc,cd,de->ae 4 ops": ("ab,bc,cd,de->ae", (arr(40, 50), arr(50, 60), arr(60, 70), arr(70, 30))),
}
with fs.BudgetContext(flop_budget=10**15, quiet=True) as b:
    for label, (expr, ops) in cases.items():
        tracemalloc.start(); f0 = b.flops_used
        try:
            r = fnp.einsum(expr, *ops)
            out = hashlib.sha256(np.asarray(r).tobytes()).hexdigest()[:10]
        except Exception as e:
            out = f"{type(e).__name__}: {str(e)[:60]}"
        peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
        print(f"{sys.argv[1]:8s} | {label:36s} | flops {b.flops_used - f0:>12,d} | out {out} | peak {peak/2**20:5.1f} MiB")
