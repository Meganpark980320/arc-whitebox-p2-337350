# Tool and scoring feedback: reproductions

Write-up §6.3 and Appendix H. Each script prints its own result; `outputs/` holds ours (22 October 2026), with local paths replaced by placeholders.

| item | script | what you should see |
|---|---|---|
| T1: a tuple `axis` fails in client-server mode | `t1_tuple_axis.py` | in-process: ok; client-server: `TypeError: 'list' object cannot be interpreted as an integer` for `sum`, `mean` and `max` with `axis=(0, 1)` |
| T2: einsum's cost analysis allocates a float64 buffer of every operand's shape | `t2_einsum_memory.py`, `t2_einsum_rlimit.py`, `t2_patch_check.py`, `einsum-dummies.patch` | first 3-operand call: +32.3 MiB against NumPy's +0.1 MiB; under a 24 MiB address-space headroom NumPy passes and flopscope raises `MemoryError … (8, 512, 512) … float64`; with the patch: same FLOPs and outputs, no buffers |
| T3: `out=` returns a new object in client-server mode | `t3_out_identity.py` | `r is c` True in-process, False in client-server mode; both still name the same server array |
| T4: the wheel's `weights.csv` disagrees with billing | `t4_weights_csv.py` | "Active Weight" differs from the billed weight for 96 of 461 operations |

S1–S3 (failure cost, setup window, compute weight) are observations on the scoring rules from whestbench 0.16.2's source and our official records; Appendix H gives the lines and records.

## Environment

Releases: flopscope 0.12.3, flopscope-client and flopscope-server 0.12.3, whestbench 0.16.2, NumPy 2.2.6, Python 3.10, Linux. The client and server talk through a Unix socket, as in flopscope's docker setup, without the hardened image.

```bash
git clone --depth 1 --branch v0.12.3 https://github.com/AIcrowd/flopscope.git
uv venv -p 3.10 server && VIRTUAL_ENV=server uv pip install flopscope==0.12.3 numpy==2.2.6 whestbench==0.16.2 pyzmq msgpack
VIRTUAL_ENV=server uv pip install --no-deps ./flopscope/flopscope-server
uv venv -p 3.10 client && VIRTUAL_ENV=client uv pip install ./flopscope/flopscope-client
# optional, for the patched runs: a patched copy of the package
mkdir patched && cp -r server/lib/python3.10/site-packages/flopscope patched/
(cd patched && patch -p2 < ../einsum-dummies.patch)
SERVER_PY=server/bin/python CLIENT_PY=client/bin/python PATCHED=$PWD/patched ./run_all.sh
```

`einsum-dummies.patch` is against the v0.12.3 tag (`git apply` from the repository root). It replaces each `np.empty(shape)` dummy with `np.broadcast_to(np.empty(()), shape)` at four sites: same shape, distinct object, 8 bytes. With it, flopscope's einsum, contraction, accumulation and symmetry tests give the same results as without it (1,008 passed), and no test in the full suite fails only with the patch (Appendix H.6).
