"""T1. Reductions with a tuple `axis`: works in-process, raises TypeError in client-server mode.

Run once with flopscope (in-process) and once with flopscope-client against flopscope-server.
"""
import flopscope as fs
import flopscope.numpy as fnp

with fs.BudgetContext(flop_budget=10**9, quiet=True):
    x = fnp.ones((2, 3, 4))
    for name in ("sum", "mean", "max"):
        for ax in ((0, 1), 0):
            try:
                r = getattr(fnp, name)(x, axis=ax)
                print(f"{name}(axis={ax!r}): ok, shape {tuple(r.shape)}")
            except Exception as e:
                print(f"{name}(axis={ax!r}): {type(e).__name__}: {str(e)[:110]}")
