"""T2. Peak traced memory of a 3-operand float32 einsum: NumPy against flopscope (in-process).

flopscope's cost analysis builds a float64 np.empty buffer of every operand's shape on the first call
for each (expression, shapes) pair. Run with PYTHONPATH pointing at a patched copy to compare.
"""
import tracemalloc
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp

k, n = 8, 512
rng = np.random.default_rng(0)
a = rng.standard_normal((k, n, n), dtype=np.float32)
b = rng.standard_normal((k, n, n), dtype=np.float32)
c = rng.standard_normal((k, n), dtype=np.float32)
print(f"flopscope {fs.__version__} at {fs.__file__}")
print(f"operands: 2 x {a.nbytes / 2**20:.0f} MiB + {c.nbytes / 2**20:.2f} MiB (float32)")

def peak(label, f):
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    f()
    p = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    print(f"{label:46s} peak +{p / 2**20:7.1f} MiB")

peak("numpy.einsum('kij,kij,kj->i')", lambda: np.einsum("kij,kij,kj->i", a, b, c))
with fs.BudgetContext(flop_budget=10**13, quiet=True):
    A, B, C = fnp.asarray(a), fnp.asarray(b), fnp.asarray(c)
    peak("flopscope einsum('kij,kij,kj->i'), 1st call", lambda: fnp.einsum("kij,kij,kj->i", A, B, C))
    peak("flopscope einsum('kij,kij,kj->i'), 2nd call", lambda: fnp.einsum("kij,kij,kj->i", A, B, C))
    peak("flopscope einsum('kij,kij->i')  (2 operands)", lambda: fnp.einsum("kij,kij->i", A, B))
