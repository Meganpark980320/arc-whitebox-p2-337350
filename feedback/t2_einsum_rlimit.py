"""T2. The same einsum under an address-space limit with 24 MiB of headroom (Linux).

usage: python t2_einsum_rlimit.py numpy|flopscope
NumPy passes; stock flopscope raises MemoryError for a float64 buffer of an operand's shape.
"""
import resource
import sys
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp

k, n = 8, 512
a = np.ones((k, n, n), dtype=np.float32)
b = np.ones((k, n, n), dtype=np.float32)
c = np.ones((k, n), dtype=np.float32)
which = sys.argv[1]
with fs.BudgetContext(flop_budget=10**13, quiet=True):
    A, B, C = fnp.asarray(a), fnp.asarray(b), fnp.asarray(c)
    np.einsum("kij,kij,kj->i", a, b, c)        # warm up NumPy
    fnp.einsum("ij,ij->i", A[0], B[0])         # warm up flopscope (2 operands: no buffers)
    vm = int([l for l in open("/proc/self/status") if l.startswith("VmSize")][0].split()[1]) * 1024
    lim = vm + 24 * 2**20
    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    try:
        if which == "numpy":
            np.einsum("kij,kij,kj->i", a, b, c)
        else:
            fnp.einsum("kij,kij,kj->i", A, B, C)
        print(which, ": ok")
    except MemoryError as e:
        print(which, ":", type(e).__name__, str(e)[:120])
