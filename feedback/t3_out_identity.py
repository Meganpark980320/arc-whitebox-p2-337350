"""T3. Is the object returned by an out= call the out array itself, as in NumPy?"""
import flopscope as fs
import flopscope.numpy as fnp

with fs.BudgetContext(flop_budget=10**9, quiet=True):
    a = fnp.ones((4, 4))
    c = fnp.zeros((4, 4))
    r = fnp.add(a, a, out=c)
    print("add(a, a, out=c) is c:", r is c)
    fnp.multiply(c, 0.0, out=c)
    print("a later write through c shows in r:", float(fnp.sum(r)) == 0.0)
