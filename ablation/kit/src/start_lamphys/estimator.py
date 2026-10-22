"""Phase 2 estimator: factorized k_max=3 cumulant propagation (port of
ARC's published reference, arXiv 2605.05179, repo mlp_cumulant_propagation),
with the prior covprop/matched-MC blend retained as the fallback branch.

Primary branch (weak coupling, L/n <= 0.05, cost affordable):
  RUNG 6 (default, _V25_STACK = True): 504aldo's V29 estimator
  (whest-p2-cumulant-k3, MIT, full attribution; see the embedded section's
  banner for the provenance of every inherited fitted constant), with our
  join-routing improvement. Measured on our mini_p2: raw 2.1-2.2e-8 at
  C/B 0.243 steady state.
  COMPARISON ARM (_V25_STACK = False): our physics-validated chain below --
  a float32 port of the reference's exact scored code path
  (coerce_input -> per layer linear_kprop + nonlin_kprop with k_max=3,
  kind=SIMPLE, use_pK=True, factor=True), restructured as the "lean"
  chain of research/port_k3_lean.py (mechanisms from
  504aldo/whest-p2-cumulant-k3 (MIT), re-derived and parity-verified):
    - rung 0 (exact): right-factor fusion of the (2,1) consumption
      contractions + the newborn propagator transport as a copy
      (W^T @ I = W^T);
    - rung 1 (their F63, exact algebra + rank-16 residual): the Fa2/M
      leg is never transported; M = P d(s) + 3 Fa1 d(e) + Z L^T with
      Z/L a rank-16 thin leg from a deterministic randomized range
      finder (sketch = first 16 columns of the layer weight). Costs
      ~+1-2% raw MSE vs the exact chain (f64 gate in port_k3_lean.py);
    - rung 2 (exact, output-preserving): last layer computes only the
      mean, so only var = colsum(W * (C @ W)), D3 and the kappa4
      diagonal are formed; all (2,1) contractions are skipped.
    - rung 3 (their V17/F68, F105 pre-activation law): memoryless kappa4
      regeneration. G_pre = diag(dG) + lam_l C_pre_off with dG transported
      exactly (n^2) and lam_l a 16-scalar table REFIT on our own mini_p2
      MLPs 0..59 against the r=1 core of ARC's reference augmented chain
      (see _KP3_LAM provenance note). The K4->K3 feed enters the newborn
      source as column scalings of legs already transported plus two thin
      Z columns: zero n^3 cost. Suite shape (1024x16) only.
  Dense-matmul census 932 n^3 + O(n^2 r) thin work at 1024x16
  (~0.46 x 2^41 billed), vs 1744 n^3 (1.45-1.49 x B billed) for the
  previous exact block-shared chain. f64 parity/MSE gates:
  research/port_k3_lean.py __main__ report.

Fallback branch (strong coupling or unaffordable shapes): the Phase 2
blend validated OOS on the public mini split (order-3 Hermite covariance
propagation, 0.85, blended with damped layer-1-matched antithetic MC,
0.15; research/p2_blendcheck.py), with its own plain-sampling guards.
Routing is decided from (n, L, budget) only; no ground truth is consulted
at runtime and no constant is fitted at runtime.
"""

from __future__ import annotations

import gc as _gc
import math
import warnings
from functools import lru_cache
from itertools import product as _iproduct

import struct

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_W_COV = 0.85
_RIDGE_FRAC = 0.3
_TARGET_UTILISATION = 0.0975
_RESIDUAL_RESERVE = 2.0e9
_MIN_MATCH_RATIO = 2
_MIN_MC = 512
_INV_SQRT_2PI = 0.3989422804014327
_PI = 3.141592653589793

_KPROP_MAX_COUPLING = 0.05   # L/n above this: strong coupling, blend wins
_KPROP_MAX_COST_X_B = 0.98   # hard cap: C_m > B_m zeroes the run
_F32 = None                  # set to fnp.float32 lazily (module import safety)


def _f32():
    global _F32
    if _F32 is None:
        _F32 = fnp.float32
    return _F32


# ===========================================================================
# k_max=3 cumulant propagation port (research/port_k3.py, xp -> fnp, f32)
# ===========================================================================
def _norm_pdf(x):
    # stats helpers return float64; cast back or downstream ops bill 2x
    return fnp.asarray(flops.stats.norm.pdf(x), dtype=_f32())


def _norm_cdf(x):
    return fnp.asarray(flops.stats.norm.cdf(x), dtype=_f32())


class _Poly:
    """Minimal ascending-coefficient polynomial (stdlib only; the grader
    sandbox provides no third-party packages). Used solely to precompute
    the ReLU wick coefficient polynomials at setup time (degrees <= 8)."""

    __slots__ = ("coef",)

    def __init__(self, coef):
        c = [float(x) for x in coef] or [0.0]
        while len(c) > 1 and c[-1] == 0.0:
            c.pop()
        self.coef = c

    def __add__(self, o):
        if not isinstance(o, _Poly):
            o = _Poly([o])
        n = max(len(self.coef), len(o.coef))
        a = self.coef + [0.0] * (n - len(self.coef))
        b = o.coef + [0.0] * (n - len(o.coef))
        return _Poly([x + y for x, y in zip(a, b)])

    __radd__ = __add__

    def __mul__(self, o):
        if not isinstance(o, _Poly):
            return _Poly([o * x for x in self.coef])
        out = [0.0] * (len(self.coef) + len(o.coef) - 1)
        for i, x in enumerate(self.coef):
            for j, y in enumerate(o.coef):
                out[i + j] += x * y
        return _Poly(out)

    __rmul__ = __mul__

    def __pow__(self, k: int):
        out = _Poly([1.0])
        for _ in range(k):
            out = out * self
        return out


Polynomial = _Poly


@lru_cache(maxsize=None)
def _He_poly(n: int) -> _Poly:
    """Probabilists' Hermite He_n via the recurrence
    He_{k+1} = x He_k - k He_{k-1}."""
    hm1, h = _Poly([1.0]), _Poly([0.0, 1.0])
    if n == 0:
        return hm1
    for k in range(1, n):
        hm1, h = h, _Poly([0.0, 1.0]) * h + (-float(k)) * hm1
    return h


@lru_cache(maxsize=None)
def _relu_wick_poly(p: int, k: int):
    """(P1, P2) with E[d^k ReLU(Z)^p] = sigma^{p-k}(P1(a)phi(a)+P2(a)Phi(a))."""
    alpha = Polynomial([0, 1])
    binom = lambda inner: math.prod(range(p - k + 1, p + 1)) * sum(
        math.comb(p - k, j) * alpha ** (p - k - j) * inner(j) for j in range(p - k + 1)
    )
    inner_1 = lambda j: sum(
        math.comb(j, 2 * m)
        * math.prod(range(1, 2 * m, 2))
        * _He_poly(j - 2 * m - 1)
        * (-1) ** (j - 1)
        for m in range((j - 1) // 2 + 1)
    )
    inner_2 = lambda j: math.prod(range(1, j, 2)) if j % 2 == 0 else Polynomial([0])
    return binom(inner_1), binom(inner_2)


def _eval_poly(poly: Polynomial, x):
    y = fnp.zeros_like(x)
    for ck in poly.coef[::-1]:
        y = y * x + float(ck)
    return y


def _He_rec(n: int, x):
    if n == 0:
        return fnp.ones_like(x)
    hm1 = fnp.ones_like(x)
    h = x
    for i in range(1, n):
        hm1, h = h, x * h - i * hm1
    return h


def _relu_wick_coef(mean, var, k: int, p: int = 1):
    if (var < 1e-10).any():
        var = fnp.clip(var, 1e-10, None)
    sigma = fnp.sqrt(var)
    alpha = mean / sigma
    if k < p:
        P1, P2 = _relu_wick_poly(p, k)
        return sigma ** (p - k) * (
            _eval_poly(P1, alpha) * _norm_pdf(alpha)
            + _eval_poly(P2, alpha) * _norm_cdf(alpha)
        )
    elif p > 1:
        return math.factorial(p) * _relu_wick_coef(mean, var, k - p + 1, 1)
    else:
        if k == 0:
            return sigma * _norm_pdf(alpha) + mean * _norm_cdf(alpha)
        elif k == 1:
            return _norm_cdf(alpha)
        else:
            return ((-1) ** (k - 2) * sigma ** (-(k - 1))
                    * _He_rec(k - 2, alpha) * _norm_pdf(alpha))


@lru_cache(maxsize=None)
def _set_partitions(k: int):
    base = tuple(range(k))
    blocks: list = []
    parts: list = []

    def build(i: int) -> None:
        if i == len(base):
            parts.append(tuple(frozenset(b) for b in blocks))
            return
        for b in blocks:
            b.append(base[i])
            build(i + 1)
            b.pop()
        blocks.append([base[i]])
        build(i + 1)
        blocks.pop()

    build(0)
    return tuple(parts)


@lru_cache(maxsize=None)
def _vector_partitions(n, prev=None):
    if all(x == 0 for x in n):
        return ((),)
    parts = []
    for v in _iproduct(*[range(x + 1) for x in n]):
        if sum(v) == 0:
            continue
        if prev is not None and v < prev:
            continue
        resid = tuple(x - y for x, y in zip(n, v))
        for part in _vector_partitions(resid, v):
            parts.append((v,) + part)
    return tuple(parts)


def _vec_part_coef(part, divide_fac: bool = True):
    if len(part) == 0:
        return 1
    part_set = list(set(part))
    counts = {v: part.count(v) for v in part_set}
    n = [sum(v[i] for v in part) for i in range(len(part[0]))]
    ret = 1 / math.prod(
        math.factorial(c) * math.prod(math.factorial(v_i) for v_i in v) ** c
        for v, c in counts.items()
    )
    if not divide_fac:
        ret *= math.prod(math.factorial(ni) for ni in n)
        ret = int(round(ret))
    return ret


@lru_cache(maxsize=None)
def _pk_to_k_coef(vpart) -> float:
    d = len(vpart[0])

    def is_disconnected(tau) -> bool:
        for mblock in tau:
            for i in range(d):
                if len([j for j in mblock if vpart[j][i] > 0]) > 1:
                    return False
        return True

    ret = 0
    for tau in _set_partitions(len(vpart)):
        if is_disconnected(tau):
            ret += (-1) ** (len(tau) - 1) * math.factorial(len(tau) - 1)
    return ret


@lru_cache(maxsize=None)
def _pk_to_k_plan(int_part):
    plan = []
    for vpart in _vector_partitions(int_part):
        c = _vec_part_coef(vpart, divide_fac=False) * _pk_to_k_coef(vpart)
        if c != 0:
            plan.append((vpart, float(c)))
    return tuple(plan)


def _get_block(slices: dict, block):
    nz = [i for i, x in enumerate(block) if x > 0]
    key = tuple(block[i] for i in nz)
    skey = tuple(sorted(key, reverse=True))
    S = slices[skey]
    if len(key) == 2 and key != skey:
        S = S.T
    if len(nz) == len(block):
        return S
    return S[:, None] if nz == [0] else S[None, :]


def _zero_diag(M):
    return M - fnp.diag(fnp.diag(M))


def _sym2(M):
    return 0.5 * (M + M.T)


def _pk_to_k(pk_slices: dict) -> dict:
    out = {}
    for part, S in pk_slices.items():
        acc = fnp.zeros_like(S)
        for vpart, c in _pk_to_k_plan(part):
            prod = None
            for block in vpart:
                b = _get_block(pk_slices, block)
                prod = b if prod is None else prod * b
            acc = acc + prod * c
        if acc.ndim == 2:
            acc = _zero_diag(acc)
        out[part] = acc
    return out


@lru_cache(maxsize=None)
def _proj_coef_4_2(n: int) -> float:
    d, r, j = 4, 2, 2
    c = d - r + n / 2 - 1
    a = (
        (-1) ** r
        * (c - r)
        / 4**j
        / math.factorial(r)
        / math.factorial(j - r)
        / c
        / math.prod([1 - c + m for m in range(j)])
    )
    return struct.unpack("f", struct.pack("f", a))[0]


_KP3_R_RES = 16  # rank of the M-leg residual (F63 ladder; research/port_k3_lean.py)

# Rung 4 (their V22/V24, MIT; re-derived + gate-verified in
# research/port_k3_lean.py): sources older than _KP3_AGE1 transports leave
# the dense stacks for ONE shared transported basis (rank r1 = 3n/8) with
# static per-source factors; sources older than _KP3_AGE2 move into a
# nested rank-(7n/32) sub-basis. The join basis is a one-pass randomized
# range finder on the contraction-WEIGHTED leg Gram (dA = 9 + w2^2 + 9e^2,
# dP = 1 + s^2 -- the squared column scalings with which A and P enter the
# consumed legs; Frobenius weighting is the known 14-31x failure mode).
# Dense legs are re-formed from the basis every layer (Hadamards do not
# factor); the two dense contractions re-associate exactly through the
# factors with one trailing @ Qc^T per layer.
_KP3_AGE1 = 4
_KP3_AGE2 = 7

# Rung 5 (their V26/V29 pattern, MIT): recursive Strassen on the batched
# dense families (young transports, the fused right-factor contractions as
# a hub kernel, the covariance sandwich, the rung-4 formings), everything
# batched so the python dispatch count stays flat, with all recursion
# scratch in a persistent pool (their F53: fresh result buffers dominate
# residual time). _STRASSEN = False restores the exact prior op stream
# (fallback if the sponsor's fair-accounting ruling changes).
_STRASSEN = True
_STRASSEN_LEVELS = 4   # measured frontier: L4 C/B 0.238 resid ~0.29s;
                       # L5 C/B 0.226 resid ~0.32s (see rung-5 report)
_STRASSEN_MIN = 32     # smallest leaf block side (their V29 ships 32)


class _Strassen:
    """Recursive Strassen on batched operands (adapted from 504aldo's
    estimator_v26 (MIT)). Layouts (last two axes = the matrix):
      mm : X (bx, P, m, kd) @ Y (by, P, kd, w) -> out (by, P, m, w), bx in {1, by}
      hub: X (k, P, m, kd), Y (k, P, w, kd)    -> out (P, m, w) = sum_k X_k Y_k^T
    Seven products M1..M7 with the classic Strassen combos; every combo /
    product is ONE op over the whole batch, written with out= into pooled
    buffers shared by role."""

    def __init__(self, dtype, bmax):
        self.dtype = dtype
        self.bmax = int(bmax)
        self.pool = {}

    def _buf(self, key, shape):
        b = self.pool.get(key)
        if b is None or b.shape[0] < shape[0]:
            b = fnp.empty(shape, dtype=self.dtype)
            self.pool[key] = b
        return b

    @staticmethod
    def _ok(m, kd, w, lev):
        d = 2 ** lev
        return (lev > 0 and m % d == 0 and kd % d == 0 and w % d == 0
                and min(m, kd, w) // d >= _STRASSEN_MIN)

    @classmethod
    def lev_for(cls, m, kd, w, lev):
        """Deepest feasible recursion depth <= lev for these dims."""
        while lev > 0 and not cls._ok(m, kd, w, lev):
            lev -= 1
        return lev

    def _combos(self, X, Q, kind, key):
        """Flat (b, 7P, h, q) combo buffer; the 7 slots are index views
        (a reshape of the 5-D form is billed by flopscope)."""
        b, P, m, kd = X.shape
        h, q = m // 2, kd // 2
        buf = self._buf(key, (max(self.bmax, b), 7 * P, h, q))[:b]
        V = [buf[:, i * P:(i + 1) * P] for i in range(7)]
        X11, X12, X21, X22 = Q
        if kind == "L":
            fnp.add(X11, X22, out=V[0]); fnp.add(X21, X22, out=V[1])
            fnp.copyto(V[2], X11);       fnp.copyto(V[3], X22)
            fnp.add(X11, X12, out=V[4]); fnp.subtract(X21, X11, out=V[5])
            fnp.subtract(X12, X22, out=V[6])
        else:
            fnp.add(X11, X22, out=V[0]); fnp.copyto(V[1], X11)
            fnp.subtract(X12, X22, out=V[2]); fnp.subtract(X21, X11, out=V[3])
            fnp.copyto(V[4], X22);       fnp.add(X11, X12, out=V[5])
            fnp.add(X21, X22, out=V[6])
        return buf

    @staticmethod
    def _assemble(M, P, out):
        """M flat (b, 7P, h, w); the 7 products are index views."""
        h, w = M.shape[2], M.shape[3]
        V = [M[:, i * P:(i + 1) * P] for i in range(7)]
        C11, C12 = out[..., :h, :w], out[..., :h, w:]
        C21, C22 = out[..., h:, :w], out[..., h:, w:]
        fnp.add(V[0], V[3], out=C11); fnp.subtract(C11, V[4], out=C11)
        fnp.add(C11, V[6], out=C11)
        fnp.add(V[2], V[4], out=C12)
        fnp.add(V[1], V[3], out=C21)
        fnp.subtract(V[0], V[1], out=C22); fnp.add(C22, V[2], out=C22)
        fnp.add(C22, V[5], out=C22)

    # fused level-1 leaves (their V29: the deepest combo buffers dominate
    # pool memory; build each product's operands in two small temps instead)
    _LSPEC = (((0, 1.0), (3, 1.0)), ((2, 1.0), (3, 1.0)), ((0, 1.0),),
              ((3, 1.0),), ((0, 1.0), (1, 1.0)), ((2, 1.0), (0, -1.0)),
              ((1, 1.0), (3, -1.0)))
    _RSPEC = (((0, 1.0), (3, 1.0)), ((0, 1.0),), ((1, 1.0), (3, -1.0)),
              ((2, 1.0), (0, -1.0)), ((3, 1.0),), ((0, 1.0), (1, 1.0)),
              ((2, 1.0), (3, 1.0)))

    @staticmethod
    def _leaf_operand(Q, spec, tmp):
        if len(spec) == 1:
            return Q[spec[0][0]]
        (i, _si), (j, sj) = spec
        if sj > 0:
            fnp.add(Q[i], Q[j], out=tmp)
        else:
            fnp.subtract(Q[i], Q[j], out=tmp)
        return tmp

    def mm(self, X, Y, out, lev):
        bx, P, m, kd = X.shape
        by, w = Y.shape[0], Y.shape[3]
        lev = self.lev_for(m, kd, w, lev)
        if lev == 0:
            fnp.matmul(X, Y, out=out)
            return
        h, q, v = m // 2, kd // 2, w // 2
        QX = (X[..., :h, :q], X[..., :h, q:], X[..., h:, :q], X[..., h:, q:])
        QY = (Y[..., :q, :v], Y[..., :q, v:], Y[..., q:, :v], Y[..., q:, v:])
        Mb = self._buf(("M", lev, P, h, v),
                       (max(self.bmax, by), 7 * P, h, v))[:by]
        if lev == 1:
            t1 = self._buf(("f1", P, h, q), (max(self.bmax, bx), P, h, q))[:bx]
            t2 = self._buf(("f2", P, q, v), (max(self.bmax, by), P, q, v))[:by]
            for i in range(7):
                lft = self._leaf_operand(QX, self._LSPEC[i], t1)
                rgt = self._leaf_operand(QY, self._RSPEC[i], t2)
                fnp.matmul(lft, rgt, out=Mb[:, i * P:(i + 1) * P])
        else:
            Xc = self._combos(X, QX, "L", ("L", lev, P, h, q))
            Yc = self._combos(Y, QY, "R", ("R", lev, P, q, v))
            self.mm(Xc, Yc, Mb, lev - 1)
        self._assemble(Mb, P, out)

    def hub(self, X, Y, out, lev):
        k, P, m, kd = X.shape
        w = Y.shape[2]
        lev = self.lev_for(m, kd, w, lev)
        if lev == 0:
            T = self._buf(("K", P, m, w), (max(self.bmax, k), P, m, w))[:k]
            fnp.matmul(X, fnp.swapaxes(Y, -1, -2), out=T)
            fnp.sum(T, axis=0, out=out)
            return
        h, q, v = m // 2, kd // 2, w // 2
        QX = (X[..., :h, :q], X[..., :h, q:], X[..., h:, :q], X[..., h:, q:])
        # (Y^T) blocks on the un-transposed Y (rows c, cols j)
        QY = (Y[..., :v, :q], Y[..., v:, :q], Y[..., :v, q:], Y[..., v:, q:])
        Mb = self._buf(("HM", lev, P, h, v), (1, 7 * P, h, v))
        if lev == 1:
            t1 = self._buf(("f1", P, h, q), (max(self.bmax, k), P, h, q))[:k]
            t2h = self._buf(("fh", P, v, q), (max(self.bmax, k), P, v, q))[:k]
            T = self._buf(("K7", P, h, v), (max(self.bmax, k), P, h, v))[:k]
            for i in range(7):
                lft = self._leaf_operand(QX, self._LSPEC[i], t1)
                rgt = self._leaf_operand(QY, self._RSPEC[i], t2h)
                fnp.matmul(lft, fnp.swapaxes(rgt, -1, -2), out=T)
                fnp.sum(T, axis=0, out=Mb[0, i * P:(i + 1) * P])
        else:
            Xc = self._combos(X, QX, "L", ("L", lev, P, h, q))
            Yc = self._combos(Y, QY, "R", ("R", lev, P, v, q))
            self.hub(Xc, Yc, Mb[0], lev - 1)
        self._assemble(Mb, P, out[None])

    def mm2(self, A, B, lev, key):
        """Single (n, m) @ (m, w) product through the pool. `key` makes the
        output buffer call-site-private (pooled outputs alias otherwise).
        Operands enter as [None, None] views so transpose views never get
        reshape-copied."""
        out = self._buf(("O", key, A.shape[0], B.shape[1]),
                        (A.shape[0], B.shape[1]))
        self.mm(A[None, None], B[None, None], out[None, None], lev)
        return out

# Rung 3: memoryless kappa4 regeneration (504aldo/whest-p2-cumulant-k3
# (MIT) V17/F68, F105 pre-activation law; re-derived and gate-verified in
# research/port_k3_lean.py). The r=1 matrix core of the pre-activation
# kappa4 is G_pre = diag(dG) + lam_l * C_pre_off; dG transported exactly
# (n^2), lam_l from the 16-scalar table below.
#
# _KP3_LAM_FIT provenance: through-origin LS of G_pre_off on C_pre_off
# per layer, where G_pre is the r=1 matrix core of the dense augmented
# k_max=3 lean chain (research/aug_g_chain.py, an f64 numpy re-derivation
# of 504aldo's lean/lean_k3_aug.py (MIT), itself verified against that
# torch chain to ~1e-13 on means AND G), run on OUR public mini_p2 MLPs
# 0..35 (fit corpus within the assigned 0..59; holdout MLPs 60..99 never
# entered the fit). Extraction: research/fit_lambda_k4.py. Indexed by the
# CURRENT layer (pre-activation lambda at layer l); entry 0 never
# consumed. Pooled per-layer uncentered R^2 of the one-mode law on our
# corpus: 0.66 (l=1) declining to 0.16 (l=15).
#
# DORMANT: the rung-3 holdout gate (mean MSE < 3.0e-8 on MLPs 60..99)
# was NOT met on our data (the F68 "regen halves raw MSE" transfer does
# not hold on mini_p2: the dense aug chain itself is WORSE than the plain
# chain here; see the rung-3 report). The mechanism is fully wired and
# gate-verified; set _KP3_LAM = _KP3_LAM_FIT to enable (~ -13% raw).
_KP3_LAM_FIT = [0.0,
                9.5379e-03, 6.4840e-03, 7.1941e-03, 7.5251e-03, 7.8929e-03,
                8.0394e-03, 8.0541e-03, 7.9731e-03, 7.7874e-03, 7.5872e-03,
                7.3126e-03, 7.0438e-03, 6.6478e-03, 6.4832e-03, 6.0340e-03]
_KP3_LAM = None  # type: list | None -- None keeps the rung-3 regen dormant
_KP3_METRIC_C = 2.0


class _KP3State:
    __slots__ = ("mean", "cov", "var", "Fa1", "P", "Z", "Lst", "c", "s", "e",
                 "c1", "c2", "nb", "k4", "metric", "g4", "varp",
                 "ka", "kb", "LG", "FO", "F2", "Qc", "U",
                 "Sg", "S2", "dA", "dP")

    def __init__(self, mean, cov, Fa1, P, Z, Lst, c, s, e, c1, c2, nb, k4,
                 metric, var=None, g4=None, varp=None):
        # kappa_3 birth blocks stacked on a leading batch axis: Fa1/P are
        # (m, n, n), Z/Lst are (m, n, r+2), c/s/e/c1/c2 are (m, n);
        # m = transported birth layers. nb = newborn awaiting its first
        # transport (fac1, s_b, e_b, Z_b, L_b, w2_b, c1_b, c2_b). The
        # Fa2/M leg is never stored: M = P d(s) + 3 Fa1 d(e) + Z Lst^T
        # (lean rung 1). The last two Z columns carry the rung-3 feed
        # (M_t1 with L column w1^2; the transported Y3 vector with a zero
        # L column); c1/c2 are the static X3 column scalings (zeros when
        # the source was born without a feed). var replaces cov on the
        # trimmed last layer (rung 2). g4/varp: post-ReLU kappa4 diagonal
        # core and variances of the previous layer (rung 3 dG transport).
        self.mean, self.cov, self.var = mean, cov, var
        self.Fa1, self.P, self.Z, self.Lst = Fa1, P, Z, Lst
        self.c, self.s, self.e = c, s, e
        self.c1, self.c2 = c1, c2
        self.nb = nb
        self.k4, self.metric = k4, metric
        self.g4, self.varp = g4, varp
        # rung 4/5: stack slots [0:kb] = tier 2, [kb:ka] = tier 1, [ka:]
        # dense young; LG = interleaved (m, 2, n, n) A|P leg view into the
        # ping-pong pool (their V29 family layout)
        self.ka = self.kb = 0
        self.LG = None
        self.FO = None               # (ka-kb, 2, r1, n) tier-1 factors A|P
        self.F2 = None               # (kb, 2, r2, n) tier-2 factors
        self.Qc = self.U = None      # shared basis (n, r1); sub-basis (r1, r2)
        self.Sg = self.S2 = None     # weighted Gram cores
        self.dA = self.dP = None     # (m, n) Gram column weights (birth)


def _bT(X):
    return fnp.transpose(X, (0, 2, 1))


def _kp3_linear(state: _KP3State, w, set_metric: float,
                trim: bool, l: int = 0, conf: bool = False,
                sw=None, lev: int = 0, pool=None) -> _KP3State:
    """Linear step on the interleaved slot buffers (their V29 layout, MIT):
    the A and P legs of source k live in one ping-pong (L-1, 2, n, n)
    family, so every young transport / rung-4 forming is ONE batched op;
    all results go out= into persistent pooled buffers (their F53)."""
    W = w.T
    n = W.shape[0]
    f32 = _f32()
    mean = W @ state.mean
    if trim:
        cov = None
        if sw is not None and sw.lev_for(n, n, n, lev) >= 1:
            CW = sw.mm2(state.cov, w, lev, "cw")
            var = fnp.sum(w * CW, axis=0)
        else:
            var = fnp.sum(w * (state.cov @ w), axis=0)
    else:
        if sw is not None and sw.lev_for(n, n, n, lev) >= 2:
            T1 = sw.mm2(W, state.cov, lev, "wc")
            Cp = sw.mm2(T1, w, lev, "wcw")
            cov = flops.as_symmetric(_sym2(Cp), symmetry=(0, 1))
        else:
            cov = fnp.einsum("ij,ia,jb->ab", state.cov, w, w)
        var = None
    LGn = Zn2 = None
    c = se = e = c1 = c2 = dA = dP = None
    ka = kb = 0
    FO = F2 = Qc = U = Sg = S2 = None
    k = 0
    if state.LG is not None:
        k = state.LG.shape[0]
        ka, kb = state.ka, state.kb
        FO, F2 = state.FO, state.F2
        Qc, U, Sg, S2 = state.Qc, state.U, state.Sg, state.S2
        dA, dP = state.dA, state.dP
        r1 = max(1, (3 * n) // 8)
        r2 = max(1, (7 * n) // 32)
        ka_t = ka
        if conf and not trim:
            ka_t = max(ka, min(k, l - _KP3_AGE1))
        if ka_t > ka:
            # ---- rung 4 join (one source/layer; post-wick legs) ----
            Aj, Pj = state.LG[ka, 0], state.LG[ka, 1]
            dAj = fnp.reshape(dA[ka], (-1, 1))
            dPj = fnp.reshape(dP[ka], (-1, 1))
            Om = w[:, :r1]
            levj = min(lev, 2)  # joins: shallow recursion (wall-time trade)
            if sw is not None and levj >= 1:
                Yq = (sw.mm2(Aj, dAj * sw.mm2(Aj.T, Om, levj, "jA"),
                             levj, "jAY")
                      + sw.mm2(Pj, dPj * sw.mm2(Pj.T, Om, levj, "jP"),
                               levj, "jPY"))
                if ka > 0:
                    Qp = Qc
                    Sfull = Sg if kb == 0 else Sg + U @ (S2 @ U.T)
                    Yq = Yq + sw.mm2(
                        Qp, Sfull @ sw.mm2(Qp.T, Om, levj, "jQ"), levj, "jQY")
                Qn, _ = fnp.linalg.qr(Yq)
                FAj = sw.mm2(Qn.T, Aj, levj, "jFA") + 0.0
                FPj = sw.mm2(Qn.T, Pj, levj, "jFP") + 0.0
            else:
                Yq = Aj @ (dAj * (Aj.T @ Om)) + Pj @ (dPj * (Pj.T @ Om))
                if ka > 0:
                    Qp = Qc
                    Sfull = Sg if kb == 0 else Sg + U @ (S2 @ U.T)
                    Yq = Yq + Qp @ (Sfull @ (Qp.T @ Om))
                Qn, _ = fnp.linalg.qr(Yq)
                FAj = Qn.T @ Aj
                FPj = Qn.T @ Pj
            Fj = fnp.stack([FAj, FPj], axis=0)[None]         # (1, 2, r1, n)
            if ka > 0:
                Tq = (sw.mm2(Qn.T, Qp, levj, "jTq") + 0.0
                      if sw is not None and levj >= 1 else Qn.T @ Qp)
                if ka > kb:
                    if sw is not None and levj >= 1:
                        k1r = ka - kb
                        rot = fnp.empty(FO.shape, dtype=f32)
                        sw.mm(Tq[None, None],
                              fnp.reshape(FO, (2 * k1r, 1, r1, n)),
                              fnp.reshape(rot, (2 * k1r, 1, r1, n)), levj)
                    else:
                        rot = fnp.matmul(Tq[None, None], FO)
                    FO = fnp.concatenate([rot, Fj], axis=0)
                    Sg = Tq @ (Sg @ Tq.T)
                else:
                    FO, Sg = Fj, None
                if kb > 0:
                    U = Tq @ U
            else:
                FO = Fj
            Sj = FAj @ (dAj * FAj.T) + FPj @ (dPj * FPj.T)
            Sg = Sj if Sg is None else Sg + Sj
            ka = ka_t
            kb_t = kb
            if _KP3_AGE2 > 0:
                kb_t = max(kb, min(ka - 1, l - _KP3_AGE2))
            if kb_t > kb:
                # ---- nested move of the oldest tier-1 member ----
                FA1, FP1 = FO[0, 0], FO[0, 1]
                dAb = fnp.reshape(dA[kb], (-1, 1))
                dPb = fnp.reshape(dP[kb], (-1, 1))
                S_s = FA1 @ (dAb * FA1.T) + FP1 @ (dPb * FP1.T)
                G2 = S_s if kb == 0 else S_s + U @ (S2 @ U.T)
                Om2 = w[:r1, :r2]
                for _pass in range(2):        # QPASS2 = 2
                    Un, _ = fnp.linalg.qr(G2 @ Om2)
                    Om2 = Un
                F2j = fnp.stack([Un.T @ FA1, Un.T @ FP1], axis=0)[None]
                if kb > 0:
                    T2 = Un.T @ U
                    F2 = fnp.concatenate(
                        [fnp.matmul(T2[None, None], F2), F2j], axis=0)
                else:
                    F2 = F2j
                S2 = Un.T @ (G2 @ Un)
                U = Un
                FO = FO[1:]
                Sg = Sg - S_s
                kb = kb_t
            Qc = (sw.mm2(W, Qn, levj, "qc") + 0.0
                  if sw is not None and levj >= 1 else W @ Qn)
        elif ka > 0:
            if sw is not None and sw.lev_for(n, n, Qc.shape[1], lev) >= 1:
                Qc = sw.mm2(W, Qc, lev, "qc") + 0.0
            else:
                Qc = W @ Qc
        # ---- form confined legs + transport the young, out= into slots ---
        LGn = pool["LG1"][:k + 1]
        if kb > 0:
            QU = Qc @ U
            sw_ok = sw is not None and lev >= 1
            dst = fnp.reshape(LGn[:kb], (2 * kb, 1, n, n))
            if sw_ok:
                sw.mm(fnp.reshape(QU, (1, 1, n, r2)),
                      fnp.reshape(F2, (2 * kb, 1, r2, n)), dst, lev)
            else:
                fnp.matmul(fnp.reshape(QU, (1, 1, n, r2)),
                           fnp.reshape(F2, (2 * kb, 1, r2, n)), out=dst)
        if ka > kb:
            k1 = ka - kb
            dst = fnp.reshape(LGn[kb:ka], (2 * k1, 1, n, n))
            if sw is not None and lev >= 1:
                sw.mm(fnp.reshape(Qc, (1, 1, n, r1)),
                      fnp.reshape(FO, (2 * k1, 1, r1, n)), dst, lev)
            else:
                fnp.matmul(fnp.reshape(Qc, (1, 1, n, r1)),
                           fnp.reshape(FO, (2 * k1, 1, r1, n)), out=dst)
        if ka < k:
            ky = k - ka
            dst = fnp.reshape(LGn[ka:k], (2 * ky, 1, n, n))
            src = fnp.reshape(state.LG[ka:k], (2 * ky, 1, n, n))
            Wb4 = fnp.reshape(W, (1, 1, n, n))
            if sw is not None and lev >= 1:
                sw.mm(Wb4, src, dst, lev)
            else:
                fnp.matmul(Wb4, src, out=dst)
        rz = state.Z.shape[2]
        Zn2 = pool["Z1"][:k + 1, :, :rz]
        fnp.matmul(W[None], state.Z, out=Zn2[:k])
        c, se, e = state.c, state.s, state.e
        c1, c2 = state.c1, state.c2
    if state.nb is not None:
        fac1, s_b, e_b, Z_b, L_b, w2_b, c1_b, c2_b, dA_b, dP_b = state.nb
        if LGn is None:
            LGn = pool["LG1"][:1]
            Zn2 = pool["Z1"][:1, :, :Z_b.shape[1]]
        if sw is not None and sw.lev_for(n, n, n, lev) >= 1:
            sw.mm(fnp.reshape(W, (1, 1, n, n)),
                  fnp.reshape(fac1, (1, 1, n, n)),
                  fnp.reshape(LGn[k, 0], (1, 1, n, n)), lev)
        else:
            fnp.matmul(W, fac1, out=LGn[k, 0])
        fnp.copyto(LGn[k, 1], W)          # newborn P copy: W^T @ I
        fnp.matmul(W, Z_b, out=Zn2[k])
        Lst = (L_b[None] if state.Lst is None
               else fnp.concatenate([state.Lst, L_b[None]], axis=0))
        c = w2_b[None] if c is None else fnp.concatenate([c, w2_b[None]], axis=0)
        se = s_b[None] if se is None else fnp.concatenate([se, s_b[None]], axis=0)
        e = e_b[None] if e is None else fnp.concatenate([e, e_b[None]], axis=0)
        dA = dA_b[None] if dA is None else fnp.concatenate([dA, dA_b[None]], axis=0)
        dP = dP_b[None] if dP is None else fnp.concatenate([dP, dP_b[None]], axis=0)
        if c1_b is not None:
            c1 = (c1_b[None] if c1 is None
                  else fnp.concatenate([c1, c1_b[None]], axis=0))
            c2 = (c2_b[None] if c2 is None
                  else fnp.concatenate([c2, c2_b[None]], axis=0))
        k += 1
    else:
        Lst = state.Lst
    pool["LG0"], pool["LG1"] = pool["LG1"], pool["LG0"]
    pool["Z0"], pool["Z1"] = pool["Z1"], pool["Z0"]
    out = _KP3State(mean, cov, None, None, None, Lst, c, se, e, c1, c2,
                    None, state.k4, float(set_metric), var=var,
                    g4=state.g4, varp=state.varp)
    out.LG = LGn
    out.Z = Zn2
    out.ka, out.kb = ka, kb
    out.FO, out.F2 = FO, F2
    out.Qc, out.U, out.Sg, out.S2 = Qc, U, Sg, S2
    out.dA, out.dP = dA, dP
    return out


def _kp3_nonlin(state: _KP3State, w, trim: bool, lam=None,
                regen: bool = False, sw=None, lev: int = 0,
                pool=None) -> _KP3State:
    n = state.mean.shape[0]
    mean = state.mean
    var = state.var if trim else fnp.diag(state.cov).copy()
    cm = state.metric
    f32 = _f32()

    _wc: dict = {}

    def wk(k: int, p: int = 1):
        if (k, p) not in _wc:
            _wc[(k, p)] = _relu_wick_coef(mean, var, k, p)
        return _wc[(k, p)]

    # rung 3: exact transported kappa4 diagonal dG (n^2). NOT an einsum:
    # same-object repeated operands crash flopscope (their F45).
    if lam is not None and state.g4 is not None:
        W = w.T
        WW = W * W
        dG = WW @ (state.g4 - state.varp * lam) + var * lam
    else:
        dG = None

    # S3 / S21 from the lean interleaved stacks (rungs 0-4; mechanisms from
    # 504aldo/whest-p2-cumulant-k3 (MIT), re-derived and gate-verified in
    # research/port_k3_lean.py). All (m,n,n) intermediates out= into the
    # persistent scratch pool (their F53).
    if state.LG is not None:
        m = state.LG.shape[0]
        Fa1 = state.LG[:, 0]
        P = state.LG[:, 1]
        Z, Lst = state.Z, state.Lst
        cb = state.c[:, None, :]
        sb = state.s[:, None, :]
        eb = state.e[:, None, :]
        AP = fnp.multiply(Fa1, P, out=pool["ap"][:m])
        PP = fnp.multiply(P, P, out=pool["pp"][:m])
        T = pool["t"][:m]
        Ub = pool["u"][:m]
        MP = pool["mp"][:m]
        fnp.matmul(Z, _bT(Lst), out=MP)
        fnp.multiply(MP, P, out=MP)
        fnp.multiply(PP, sb, out=T)
        fnp.add(MP, T, out=MP)
        fnp.multiply(AP, eb * 3.0, out=T)
        fnp.add(MP, T, out=MP)
        fnp.multiply(AP, Fa1, out=T)
        S3 = 3.0 * fnp.matmul(T, state.c[:, :, None])[:, :, 0].sum(axis=0)
        fnp.multiply(MP, P, out=T)
        S3 = S3 + T.sum(axis=2).sum(axis=0)
        if state.c1 is not None:
            R = (fnp.matmul(AP, state.c1[:, :, None])[:, :, 0]
                 + fnp.matmul(PP, state.c2[:, :, None])[:, :, 0])
            Yk = Z[:, :, Z.shape[2] - 1]
            S3 = S3 + (R * Yk).sum(axis=0)
        if not trim:
            LL = pool["ll"][:m]           # interleaved (m, 2, n, n): LA | LP
            LA = LL[:, 0]
            LP = LL[:, 1]
            fnp.multiply(AP, cb * 2.0, out=LA)
            fnp.multiply(PP, eb, out=T)
            fnp.add(LA, T, out=LA)
            fnp.multiply(Fa1, Fa1, out=LP)
            fnp.multiply(LP, cb, out=LP)
            fnp.multiply(PP, sb * (1.0 / 3.0), out=T)
            fnp.add(LP, T, out=LP)
            fnp.multiply(MP, 2.0 / 3.0, out=T)
            fnp.add(LP, T, out=LP)
            if state.c1 is not None:
                fnp.multiply(Fa1, state.c1[:, None, :], out=T)
                fnp.multiply(P, state.c2[:, None, :], out=Ub)
                fnp.add(T, Ub, out=T)
                fnp.multiply(T, Yk[:, :, None] * (2.0 / 3.0), out=T)
                fnp.add(LP, T, out=LP)
            ka, kb = state.ka, state.kb
            D21 = None
            if ka > 0:
                # rung 4: confined contractions through the factors; one
                # trailing @ Qc^T per layer (tier 2 lifted through U first)
                inner = None
                if ka > kb:
                    k1 = ka - kb
                    r1c = state.FO.shape[2]
                    Xf = fnp.reshape(LL[kb:ka], (2 * k1, 1, n, n))
                    Yf = fnp.reshape(state.FO, (2 * k1, 1, r1c, n))
                    I1 = sw._buf(("I1", n, r1c), (1, n, r1c)) if sw is not None \
                        else fnp.empty((1, n, r1c), dtype=f32)
                    if sw is not None and lev >= 1:
                        sw.hub(Xf, Yf, I1, lev)
                    else:
                        fnp.sum(fnp.matmul(Xf, fnp.swapaxes(Yf, -1, -2)),
                                axis=0, out=I1[None][0])
                    inner = I1[0] + 0.0
                if kb > 0:
                    r2c = state.F2.shape[2]
                    Xf = fnp.reshape(LL[:kb], (2 * kb, 1, n, n))
                    Yf = fnp.reshape(state.F2, (2 * kb, 1, r2c, n))
                    I2 = sw._buf(("I2", n, r2c), (1, n, r2c)) if sw is not None \
                        else fnp.empty((1, n, r2c), dtype=f32)
                    if sw is not None and lev >= 1:
                        sw.hub(Xf, Yf, I2, lev)
                    else:
                        fnp.sum(fnp.matmul(Xf, fnp.swapaxes(Yf, -1, -2)),
                                axis=0, out=I2[None][0])
                    lift = I2[0] @ state.U.T
                    inner = lift if inner is None else inner + lift
                if sw is not None and sw.lev_for(n, inner.shape[1], n, lev) >= 1:
                    D21 = sw.mm2(inner, state.Qc.T, lev, "tr") + 0.0
                else:
                    D21 = inner @ state.Qc.T
            if ka < m:
                ky = m - ka
                Xh = fnp.reshape(LL[ka:m], (2 * ky, 1, n, n))
                Yh = fnp.reshape(state.LG[ka:m], (2 * ky, 1, n, n))
                if sw is not None and lev >= 1:
                    Dh = sw._buf(("HO", n), (1, n, n))
                    sw.hub(Xh, Yh, Dh, lev)
                    Dh = Dh[0]
                else:
                    Dh = fnp.matmul(Xh, fnp.swapaxes(Yh, -1, -2)).sum(axis=0)[0]
                D21 = Dh if D21 is None else D21 + Dh
            D21 = D21 + fnp.matmul(fnp.matmul(PP, Lst),
                                   _bT(Z)).sum(axis=0) * (1.0 / 3.0)
            if state.c1 is not None:
                D21 = D21 + (R.T @ Yk) * (1.0 / 3.0)
            S21 = _zero_diag(D21)
    else:
        S3 = fnp.zeros(n, dtype=f32)
        S21 = fnp.zeros((n, n), dtype=f32)

    if dG is not None:
        s4 = dG * _KP3_METRIC_C  # (4,)-slice of the r=1 core
    else:
        s4 = state.k4 * cm * cm * fnp.ones(n, dtype=f32)

    if trim:
        # rung 2: the mean consumes only var, S3 and the kappa4 diagonal
        mean_new = (wk(0, 1) + (1.0 / 6.0) * S3 * wk(3, 1)
                    + (1.0 / 24.0) * s4 * wk(4, 1))
        return _KP3State(mean_new, None, None, None, None, None, None, None,
                         None, None, None, None, 0.0, 1.0)

    S11 = _zero_diag(state.cov)
    if dG is not None:
        g22 = dG * (_KP3_METRIC_C / 6.0)
        S22 = _zero_diag(g22[:, None] + g22[None, :])
        S31 = S11 * lam  # wk431 = 0.5 m lam C_off (m = 2)
    else:
        S22 = _zero_diag(state.k4 * cm * cm / 3.0
                         * fnp.ones((n, n), dtype=f32))
        S31 = None
    Q = S11 * S11

    pk: dict = {}
    for p_ in (1, 2, 3, 4):
        pk[(p_,)] = (wk(0, p_) + (1.0 / 6.0) * S3 * wk(3, p_)
                     + (1.0 / 24.0) * s4 * wk(4, p_))

    def two_site(p1: int, p2: int):
        T2s = (wk(1, p1)[:, None] * wk(1, p2)[None, :]) * S11
        T2s = T2s + 0.5 * (wk(2, p1)[:, None] * wk(2, p2)[None, :]) * Q
        cnt = 2.0 if p1 == p2 else 1.0
        T2s = T2s + cnt * 0.5 * (wk(1, p1)[:, None] * wk(2, p2)[None, :]) * S21.T
        if (p1, p2) == (2, 1):
            T2s = T2s + 0.5 * (wk(2, p1)[:, None] * wk(1, p2)[None, :]) * S21
        T2s = T2s + 0.25 * (wk(2, p1)[:, None] * wk(2, p2)[None, :]) * S22
        if S31 is not None:
            # the four (3,1)-slice use-side terms (their wk431 append),
            # written symmetrically (pk11/pk22 are symmetrized after)
            T2s = T2s + (1.0 / 6.0) * (wk(1, p1)[:, None] * wk(3, p2)[None, :]
                                       + wk(3, p1)[:, None] * wk(1, p2)[None, :]) * S31
        return T2s

    pk[(1, 1)] = _zero_diag(_sym2(two_site(1, 1)))
    pk[(2, 1)] = _zero_diag(two_site(2, 1))
    pk[(2, 2)] = _zero_diag(_sym2(two_site(2, 2)))

    w1, w2 = wk(1, 1), wk(2, 1)
    fac1 = w1[:, None] * S11

    r3 = (w1 * w1 * w1) * S3
    r21 = ((w1 * w1)[:, None] * w1[None, :] * S21
           + (w1 * w1)[:, None] * (S11 * S11) * w2[None, :])

    # rung 3 feed: birth dslices of (X3, P=I, Y3) and M_t1
    if dG is not None:
        y_b = w2 * (0.25 * _KP3_METRIC_C)
        w1sq = w1 * w1
        dgw = w1sq * dG
        u_b = y_b * dG
        c1_b = w1 * lam
        r3 = r3 + dgw * y_b + u_b * w1sq
        r21 = r21 + ((1.0 / 3.0) * dgw[:, None] * y_b[None, :]
                     + (2.0 / 3.0) * y_b[:, None] * (fac1 * c1_b[None, :])
                     + (1.0 / 3.0) * w1sq[:, None] * u_b[None, :])
    elif regen:
        zn = fnp.zeros(n, dtype=f32)
        y_b = u_b = c1_b = dgw = zn
        w1sq = zn
    else:
        y_b = u_b = c1_b = dgw = w1sq = None

    K = _pk_to_k(pk)
    K3s3 = K[(3,)] - r3
    K3s21 = _zero_diag(K[(2, 1)] - r21)

    mean_new = K[(1,)]
    cov_new = flops.as_symmetric(_sym2(K[(1, 1)] + fnp.diag(K[(2,)])),
                                 symmetry=(0, 1))
    a2c = _proj_coef_4_2(n)
    k4_new = a2c * (24.0 * K[(4,)].sum()) + a2c * (24.0 * K[(2, 2)].sum())

    # birth (F63 M-leg split): K3s21 = S_sep + Rres with the exact leading
    # term S_sep = d(e) C_off d(w1) riding on Fa1 (C_off symmetric), and
    # Rres compressed to rank r by a deterministic randomized range finder
    # (sketch = first r columns of the layer weight, one power iteration)
    e_b = wk(1, 2) - 2.0 * mean_new * w1
    S_sep = _zero_diag(e_b[:, None] * S11 * w1[None, :])
    Rres = K3s21 - S_sep
    r = min(_KP3_R_RES, n)
    Om = w[:, :r]
    Y = Rres @ Om
    Qm, _ = fnp.linalg.qr(Y)
    Y = Rres @ (Rres.T @ Qm)
    Qm, _ = fnp.linalg.qr(Y)
    Bm = Qm.T @ Rres                       # Rres ~= Qm @ Bm
    if regen:
        # feed columns (zeros when this layer has no feed, so every source
        # keeps r+2 Z columns): M_t1 = u_b (w1^2)^T inside the M leg; the
        # Y3 vector y_b transported with a zero L column (out of the M leg)
        Z_b = fnp.concatenate([Bm.T * 3.0, fnp.reshape(u_b, (-1, 1)),
                               fnp.reshape(y_b, (-1, 1))], axis=1)
        L_b = fnp.concatenate([Qm, fnp.reshape(w1sq, (-1, 1)),
                               fnp.zeros((n, 1), dtype=f32)], axis=1)
    else:
        Z_b, L_b = Bm.T * 3.0, Qm
    dA_b = 9.0 + w2 * w2 + 9.0 * e_b * e_b   # rung 4 Gram column weights
    dP_b = 1.0 + K3s3 * K3s3
    nb = (fac1, K3s3, e_b, Z_b, L_b, w2, c1_b, dgw, dA_b, dP_b)

    # rung 3: post-ReLU kappa4 diagonal core (r=1 harmonic projection)
    if regen:
        k22row = K[(2, 2)] @ fnp.ones(n, dtype=f32)
        cA = 6.0 / (n + 4.0)
        cI = -3.0 / ((n + 2.0) * (n + 4.0))
        g4_new = (cA * (K[(4,)] + k22row)
                  + cI * (fnp.sum(K[(4,)]) + fnp.sum(k22row)))
        varp_new = K[(2,)]
    else:
        g4_new = varp_new = None

    out = _KP3State(mean_new, cov_new, None, None, None, state.Lst,
                    state.c, state.s, state.e, state.c1, state.c2, nb,
                    float(k4_new), 1.0, g4=g4_new, varp=varp_new)
    if state.LG is not None:
        # the wick rides on every transported mode, applied in place
        ws3 = w1[None, None, :, None]
        fnp.multiply(state.LG, ws3, out=state.LG)
        fnp.multiply(state.Z, w1[None, :, None], out=state.Z)
        out.LG, out.Z = state.LG, state.Z
    out.ka, out.kb = state.ka, state.kb
    out.FO, out.F2 = state.FO, state.F2
    out.Qc = state.Qc * w1[:, None] if state.Qc is not None else None
    out.U, out.Sg, out.S2 = state.U, state.Sg, state.S2
    out.dA, out.dP = state.dA, state.dP
    return out


def _kprop3_layer_means(weights, n: int, sw=None, pool=None):
    """All post-ReLU layer means via the interleaved lean block-shared k=3
    port (rungs 0-5 of research/port_k3_lean.py; rung 3 dormant). Rung 4
    (shared-subspace confinement) runs whenever the shape gives it room;
    rung 5 (Strassen) through `sw`."""
    f32 = _f32()
    L = len(weights)
    scales = [1.0] + [2.0] * (L - 1)
    regen = (_KP3_LAM is not None and n == 1024 and L == len(_KP3_LAM))
    conf = (n >= 64 and L > _KP3_AGE1 + 1)
    lev = _STRASSEN_LEVELS if (sw is not None and _STRASSEN) else 0
    if lev == 0:
        sw = None
    rz = min(_KP3_R_RES, n) + (2 if regen else 0)
    if pool is None:
        pool = {}
    key = (L, n, rz)
    if pool.get("key") != key:
        pool.clear()
        pool["key"] = key
        kmax = max(L - 1, 1)
        pool["LG0"] = fnp.empty((kmax, 2, n, n), dtype=f32)
        pool["LG1"] = fnp.empty((kmax, 2, n, n), dtype=f32)
        pool["Z0"] = fnp.empty((kmax, n, rz), dtype=f32)
        pool["Z1"] = fnp.empty((kmax, n, rz), dtype=f32)
        pool["ll"] = fnp.empty((kmax, 2, n, n), dtype=f32)
        for nm in ("ap", "pp", "t", "u", "mp"):
            pool[nm] = fnp.empty((kmax, n, n), dtype=f32)
    state = _KP3State(
        mean=fnp.zeros(n, dtype=f32),
        cov=flops.as_symmetric(fnp.eye(n, dtype=f32), symmetry=(0, 1)),
        Fa1=None, P=None, Z=None, Lst=None, c=None, s=None, e=None,
        c1=None, c2=None, nb=None, k4=0.0, metric=1.0,
    )
    rows = []
    for l, w in enumerate(weights):
        w = fnp.asarray(w, dtype=f32)
        trim = l == L - 1
        lam = float(_KP3_LAM[l]) if (regen and l > 0) else None
        state = _kp3_linear(state, w, scales[l], trim, l=l, conf=conf,
                            sw=sw, lev=lev, pool=pool)
        state = _kp3_nonlin(state, w, trim, lam=lam, regen=regen,
                            sw=sw, lev=lev, pool=pool)
        rows.append(state.mean)
    return rows


def _kprop3_cost(n: int, depth: int) -> float:
    """Lean-chain census (research/port_k3_lean.py): per layer l with k = l
    sources, transports 4k - 2 (newborn P is a copy) + contractions 4k
    (skipped at the trimmed last layer) + cov 4 (2 at the last layer), in
    n^3 units of flops, plus O(n^2 r) thin legs / range finders."""
    sum_k = depth * (depth - 1) / 2.0
    r = min(_KP3_R_RES, n) + 2  # +2 feed columns on the thin leg
    dense = 8.0 * sum_k - 6.0 * max(depth - 1, 0) + 4.0 * depth - 2.0
    thin = (8.0 * sum_k + 14.0 * max(depth - 1, 0)) * (float(r) / n)
    return 1.05 * ((dense + thin) * n**3 + 200.0 * depth * n * n)




# ===========================================================================
# ATTRIBUTED ACCURACY STACK (rung 6 merge): 504aldo's estimator V29
# (whest-p2-cumulant-k3, MIT license), embedded below after a complete
# first-hand source review, with three renames (Estimator -> _V25Core,
# _Strassen -> _StrassenV29, _zero_diag -> _zero_diag_ip) and ONE functional
# change, marked "MERGE": the shared-basis join products are routed through
# their own Strassen kernel at capped depth 3 (our rung-5 measurement:
# bills 0.686x, ~free in residual; their F86 left joins dense).
#
# PROVENANCE OF INHERITED FITTED CONSTANTS (shipped as-disclosed, NOT refit
# by us; all fitted by 504aldo on their public Phase-2 dump snapshot):
# LAM (16 pre-activation kappa4 lambdas) x scale 0.95, REF_R (15 reference
# ratios) and BETA 1.0 (adaptive per-MLP lambda), R_RES/R_FB 16,
# AGE_OLD/R_OLD 4/384, AGE_OLD2/R_OLD2 7/224, the pruned 62-term TERM_SPECS,
# CORR_BETA (embedded but OFF by default, V17_NO_CORR=1).
# Known structure (research/astra_k4_overshoot.md): the LAM table is ~3x
# the physically measured lambda of the true kappa4 core; the oversized
# C-shaped correction cancels the pruned base's C-shaped truncation bias,
# so part of this stack's accuracy lives in error cancellation. It is
# measured to transfer to our mini_p2 data unchanged (raw 2.14e-8, nets
# 60-63) and is promotion-gated on never-before-used nets 36-59.
#
# _V25_STACK = False routes suite shapes to our physics-validated rungs 0-5
# chain instead (raw ~4.1e-8 at ~0.238 B; the comparison arm for the
# write-up, including the dormant rung-3 refit/0.3x-physical lambda arms).
# ===========================================================================
_V25_STACK = True

"""K3-simple factored cumulant propagation + MEMORYLESS KAPPA4 REGENERATION (V17)
+ D21 FEEDBACK THIN LEGS (V18, 2026-09-02, F69)
+ FINAL-LAYER TRIM (V19, 2026-09-03)
+ RESIDUAL-TIME FUSION (V20, 2026-09-03)
+ AGE-GATED SHARED BASIS FOR OLD SOURCES (V21, 2026-09-03, F66 option C / F72)
+ PRE-ACTIVATION LAMBDA TABLE (V22, 2026-09-03, F68 prefit)
+ NESTED SECOND TIER FOR THE OLDEST SOURCES (V24, 2026-09-03, F72 tiers)
+ ADAPTIVE PER-MLP LAMBDA (V25, 2026-09-03)
+ POOLED BATCHED STRASSEN-WINOGRAD PRODUCTS (V26-V28, 2026-09-05, F79/F80)
+ NEWBORN + COVARIANCE RIDING THE TRANSPORT FAMILY, BLOCK-SYMMETRIC C_PRE (V29, 2026-09-06).

V29: the dense young legs are pre-scaled by the wick w1 at the wick stage, so the
transport family multiplies by the raw W and the newborn's A leg (born as w1 * C_off) and
the post-ReLU covariance C ride along as two extra slots; C_pre = (W C) W^T is assembled
from the three block products of its 2x2 partition (one Strassen family), layer 0 uses the
aliased Gram w32^T w32, and the trimmed last layer reads var from (W C) * W.

V25: the regenerated kappa4 off-diagonal coefficient is no longer a frozen per-layer
table but lam_l = LAM[l] * ((mean(dG)/mean(var)) / REF_R[l])^BETA, evaluated online at the
layer from the transported kappa4 diagonal dG and the pre-activation variance (both already
computed there). Fitted offline on public dumps 0-7: the per-MLP LS lambdas track this
ratio with log-log slope ~1 at every layer and the unexplained spread drops from 3-10% to
1-2%. Kill-switch: V25_BETA=0 -> V24 values.

V24: sources older than AGE_OLD2 transports are re-confined to a rank-R_OLD2 sub-basis U
(r1 x r2) INSIDE the shared tier-1 basis Qc:  A_s = Qc U FA2_s.  Legs are formed from
QU = Qc U (r2/n units per leg), the dslice contraction of tier-2 sources is lifted by U^T
into the tier-1 inner product (one trailing Qc^T per layer, unchanged), U rotates with the
tier-1 factors at each rebuild, and the oldest tier-1 member moves into tier 2 in factor
space right after each rebuild (range finder on the r1 x r1 core, no n-space work). Lean
reference (nested tiers 4:384 + 7:256, dumps 0/1): 2.155e-8 vs 2.164e-8 = free.
Kill-switch: V24_AGE_OLD2=0 -> V22 op stream.

V21: sources older than AGE_OLD transports are confined to one shared rank-R_OLD basis
Q (n x r): A_s = Q FA_s, P_s = Q FP_s with static factors. Per old source-layer the
dense legs are formed from the factors (2 n^2 r) and the two dslice contractions go
through the factors (2 n^2 r) instead of 4 dense n^3 units. One source joins the old
group per layer; at a join the basis is rebuilt by a one-pass randomized range finder on
the weighted leg Gram of old + joiner (sketch = a slice of the layer weight), the old
factors are rotated into the new basis and the Gram core S (r x r) is updated. The lean
reference (g_regen_probe.py "...+agerank:3:384") measures +8.4% raw on dumps 0/1 for a
~25% cost cut. Kill-switch: V21_NO_CONFINE=1 -> exact V20 op stream.

V20: exact, bit-identical values; only the client-side op stream changes. The nonlin
term products are written with out= into one persistent (T, n, n) buffer instead of
~30 fresh (n,n) multiplies + a ~50-matrix fnp.stack per layer (measured 4 ms + 3 ms of
residual per layer), and the range-finder sketches are one contiguous copy of the
weight slice per layer instead of strided views (strided qr/matmul input costs ~0.8 ms
each). Kill-switch: none needed (values identical); V19 stays the reference.

V19: exact cost cut, zero accuracy change. The final layer emits only the mean, whose
(1,) nonlin terms need var = diag(C_pre), D3 and the K4 diagonal -- never C_off, the
(2,1) dslice D21, nor the (1,1)/(2,1)/(2,2) term program. V18 computed all of them at
the last layer and discarded them: the two dense dslice contractions (2k n^3 units,
k = 15 live sources) plus the full C_pre sandwich (2 units, only its diagonal is used).
V19 transports only diag(C_pre) = colsum(w32 * (C @ w32)) (1 unit) and runs the source
machinery in D3-only mode at the last layer (transports + n^2 contractions, no D21).
Expected C: 0.509xB -> ~0.48xB. Kill-switch: V19_FULL_LAST=1 -> exact V18 op stream.
Below this line the V18 docstring follows unchanged.


V18: the V1.6 "K3-dslice feedback into the new-block factors" (dropped since V1.6 as
DS-shadowed) is worth -10% raw on the regen base (F69) and is rank <= 16 in D21.
Birth factors of the B1 hub pair become
  X1 = 3 a_b + Xt,   Xt = 1.5 d(w2) D21            ~ F1 R1,  F1 = d(w2) Q,   R1 = 1.5 Bm
  Y1 = a_b d(w2) + Yt, Yt = 0.5 d(w1) D21^T d(w3)  ~ F2 R2,  F2 = d(w1) Bm^T, R2 = 0.5 Q^T d(w3)
with D21 ~ Q Bm from a rank-R_FB randomized range finder (same recipe as Rres). The
identity-born thin legs F1/F2 travel in their own transported stack Zf (like Z, but never
entering the M-leg einsums); in the dslice contraction the new terms either add to the fused left factors
(LA += Xt*P*w2/3 + P*Yt; LP += A*Yt + Xt*A*w2/3 + Xt*Yt/3; D3 += rowsum(P*3*LPadd)) or
contract against the thin right factors ((AP + Xt*P/3) R2^T F2^T and
(AP*w2 + P*Yt)/3 R1^T F1^T). Extra n^3: ZERO; extra ~16 n^2 r per source-layer.
Lean reference: g_regen_probe.py "regen:c+no31+feedc+usec+fitfull+rres:16+sb2+nowk22
+d21r:16": 2.195e-8 / 2.091e-8 on P2 dumps 0/1 (V17 port: 2.51 / 2.25e-8).
Kill-switch: V18_NO_FB=1 -> exact V17 behaviour (zero thin legs).
Below this line the V17 docstring follows unchanged.


V17 (2026-09-02, F68): the augmented-K3 kappa4 channel (F51/F64) without any
per-source kappa4 memory. The kappa4 matrix core G of the aug chain is, to
R^2 ~0.9-0.97, diag(g) + lambda_l * C_off with ONE frozen scalar per layer
(LAM table, fitted offline on public MLPs). Everything the channel needs then
rides on objects V16b already carries:
  - transported diagonal dG_pre = (W*W) g + lambda (diag(C_pre) - (W*W) var)  [n^2]
  - use side: wk4_row = m dG, wk4_mat = (m/6)(dG_i + dG_j) (== V16b's vec+ww form),
    plus the (3,1) slice wk431 = (m/2) lambda C_pre_off feeding 4 extra n^2 terms
  - K4->K3 feed: hub pair X3 = d(w1) G_pre d(w1) = lambda A d(w1) + P d(w1^2 dG)
    (column scalings of the A/P legs), Y3 = (m/4) w2 1^T (rank-1, folds into the
    P-group + one rank-1 outer product), M_t1 = (m/4)(w2*dG)(w1^2)^T (rank-1 M-type
    -> one extra thin Z/L column).  Extra n^3 cost: ZERO.
Lean reference: scratch/lean_k3_aug.py regen chain (g_regen_probe.py
"regen:c+no31+feedc+usec"): 2.196e-8 / 2.161e-8 on P2 dumps 0/1 vs 4.1e-8 for V16b.
Below this line the V16b docstring follows unchanged.


V16b (2026-09-01, F63): exact cost restructuring of the source machinery.
  - The M leg (B3 = Sym(M x P x P)) is never transported: M_b = diag(S3c)
    + 3*S21^T and S21 = S_sep + Rres with S_sep = diag(e) C_off diag(w1) the
    exact leading (2,1) Wick term (a column-scaled copy of a_b) and Rres a
    D21-driven residual that is numerically low-rank (rank-64 keeps final MSE
    at parity, F63). Hence M = P diag(s) + 3 A diag(e) + 3 Z L^T with Z = P Rr
    a thin (n,r) transported leg and L static.
  - V27 (F80): every fresh (n,n)/(k,n,r) result of the layer body goes out= into a
    persistent named buffer (_Pool); thin stacks and shared-basis factors live in
    ping-pong slot buffers (no concatenates); slow einsum forms -> matmul(out=).
  - All dslice contractions collapse into TWO fused einsums per layer (right
    factors A and P) plus O(n^2 r) thin terms: per source-layer 2 dense
    transports + 2 dense contractions (was 3 + 4).
  - Rres range finder: randomized (Omega = a slice of the layer weight), one
    power iteration, flopscope-billed QR; ~4 n^2 r FLOPs per birth.

Port of the reference kprop k_max=3 SIMPLE factored algorithm (companion paper
arXiv:2605.05179) with cost engineering measured in F42-F45, plus two
accuracy riders measured in F47/F49 (combined 1.44x lower final MSE):
  - vec+ww (augmented-lite v0): the radial K4 channel keeps a per-neuron
    diagonal-content vector instead of a single scalar; born from K(4,) and
    the K(2,2) column sums, transported one layer by the true (W*W) row
    action in place of one average-metric cup. ~n^2/layer, cost-neutral.
  - online mean correction: per-layer ridge-fitted delta = feats @ beta
    (13 per-neuron features already computed in the loop; beta fitted
    offline on the public teacher-forced trajectory, embedded below).
    Applied inside the loop so it propagates.
Base cost engineering:
  - shared-P hub blocks per source layer: B1 = Sym(X1 x P x Y1) with X1 = 3*abar,
    B3 = Sym(M x P x P); identity-born legs all evolve by the same map
    leg <- W @ (w1 * leg), so one P per source serves every sub-block, and the
    wick scaling is folded into the next layer's weight (WD).
  - V1.6 ablations (each measured to cost <~1% final MSE): the B2 hub sub-block
    and the K3-dslice feedback into new-block factors are DS-shadowed at birth
    (their diagonal-slice content migrates into the exact M chain automatically);
    the WK22 struct term is dropped from Y1; nonlin terms pruned at relative
    contribution >= 1e-5 (62 of 90 kept).
  - residual-time engineering: per-source matrices held as batched (k,n,n)
    stacks; all wick vectors built at once in the unified form
    w(k,p) = const*sigma^e*(P1(a)phi + P2(a)Phi); nonlin terms evaluated as one
    stacked einsum per arity; dslice hub contractions fused into 3/4-operand
    einsums with no (k,n,n) intermediates; final layer computes only the mean.

Cost: ~1.83e12 flopscope FLOPs per MLP = 0.833x the 2^41 budget (deterministic,
data-independent). Local mini-dump accuracy: mse_final ~3.2e-8 test-set mean
(V1.6 base alone: ~6.2e-8; exact K3-simple reference: 4.8e-8 at an over-budget
2.24x; cov-prop baseline: 4.4e-6).
Prediction: per-layer post-ReLU means, shape (depth, width). No randomness.
"""

# (embedded module header: __future__ / flopscope / whestbench imports
#  already present at the top of this file)

# ---- embedded coefficient tables (generated by dump_k3_tables2.py) ----
# v2 K3-simple tables (dump_k3_tables2.py; pruned at rel>=1e-5)
WICK_PAIRS = [(0, 1),
 (0, 2),
 (0, 3),
 (0, 4),
 (1, 1),
 (1, 2),
 (2, 1),
 (2, 2),
 (3, 1),
 (3, 2),
 (3, 3),
 (3, 4),
 (4, 1),
 (4, 2),
 (4, 3),
 (4, 4),
 (5, 1),
 (5, 2),
 (6, 1),
 (6, 2),
 (7, 1)]

WICK_UNIFIED = [(1.0, 1, [1.0], [0.0, 1.0]),
 (1.0, 2, [0.0, 1.0], [1.0, 0.0, 1.0]),
 (1.0, 3, [2.0, 0.0, 1.0], [0.0, 3.0, 0.0, 1.0]),
 (1.0, 4, [0.0, 5.0, 0.0, 1.0], [3.0, 0.0, 6.0, 0.0, 1.0]),
 (1.0, 0, [], [1.0]),
 (1.0, 1, [2.0], [0.0, 2.0]),
 (1.0, -1, [1.0], []),
 (2.0, 0, [], [1.0]),
 (1.0, -2, [-0.0, -1.0], []),
 (2.0, -1, [1.0], []),
 (6.0, 0, [], [1.0]),
 (1.0, 1, [24.0], [0.0, 24.0]),
 (1.0, -3, [-1.0, 0.0, 1.0], []),
 (2.0, -2, [-0.0, -1.0], []),
 (6.0, -1, [1.0], []),
 (24.0, 0, [], [1.0]),
 (1.0, -4, [-0.0, 3.0, -0.0, -1.0], []),
 (2.0, -3, [-1.0, 0.0, 1.0], []),
 (1.0, -5, [3.0, 0.0, -6.0, 0.0, 1.0], []),
 (2.0, -4, [-0.0, 3.0, -0.0, -1.0], []),
 (1.0, -6, [-0.0, -15.0, -0.0, 10.0, -0.0, -1.0], [])]

TERM_SPECS = {(1,): [('ones1', 'ones1', 0, None, 1.0),
        ('d3', 'ones1', 8, None, 0.16666666666666666),
        ('g4', 'ones1', 12, None, 0.041666666666666664),
        ('d3', 'd3', 18, None, 0.013888888888888888),
        ('d3', 'g4', 20, None, 0.006944444444444444)],
 (1, 1): [('c_off', 'ones2', 4, 4, 1.0),
          ('d21T', 'ones2', 4, 6, 1.0),
          ('c_off', 'c_off', 6, 6, 0.5),
          ('wk4m', 'ones2', 6, 6, 0.25),
          ('d3row', 'c_off', 4, 12, 0.3333333333333333),
          ('c_off', 'd21T', 6, 8, 1.0),
          ('d3row', 'd21T', 4, 16, 0.16666666666666666),
          ('g4row', 'c_off', 4, 16, 0.08333333333333333),
          ('d3row', 'd21', 6, 12, 0.16666666666666666),
          ('d21T', 'd21T', 6, 12, 0.25),
          ('c_off', 'wk4m', 8, 8, 0.25),
          ('d21T', 'd21', 8, 8, 0.25),
          ('g4row', 'd21T', 4, 18, 0.041666666666666664),
          ('d3row', 'wk4m', 6, 16, 0.08333333333333333),
          ('g4row', 'd21', 6, 16, 0.041666666666666664),
          ('d21T', 'wk4m', 8, 12, 0.25),
          ('g4row', 'wk4m', 6, 18, 0.020833333333333332)],
 (2,): [('ones1', 'ones1', 1, None, 1.0),
        ('d3', 'ones1', 9, None, 0.16666666666666666),
        ('g4', 'ones1', 13, None, 0.041666666666666664)],
 (2, 1): [('c_off', 'ones2', 5, 4, 1.0),
          ('d21T', 'ones2', 5, 6, 0.5),
          ('d21', 'ones2', 7, 4, 0.5),
          ('c_off', 'c_off', 7, 6, 0.5),
          ('wk4m', 'ones2', 7, 6, 0.25),
          ('d3row', 'c_off', 5, 12, 0.16666666666666666),
          ('c_off', 'd21T', 7, 8, 0.5),
          ('c_off', 'd21', 9, 6, 0.5),
          ('c_off', 'd3col', 13, 4, 0.16666666666666666),
          ('d3row', 'd21T', 5, 16, 0.08333333333333333),
          ('g4row', 'c_off', 5, 16, 0.041666666666666664),
          ('d3row', 'd21', 7, 12, 0.08333333333333333),
          ('d21T', 'd21T', 7, 12, 0.125),
          ('c_off', 'wk4m', 9, 8, 0.25),
          ('d21T', 'd21', 9, 8, 0.25),
          ('c_off', 'g4col', 17, 4, 0.041666666666666664),
          ('d21', 'd3col', 17, 4, 0.08333333333333333),
          ('g4row', 'd21T', 5, 18, 0.020833333333333332),
          ('g4row', 'd21', 7, 16, 0.020833333333333332),
          ('d21', 'g4col', 19, 4, 0.020833333333333332)],
 (2, 2): [('c_off', 'ones2', 5, 5, 1.0),
          ('d21T', 'ones2', 5, 7, 1.0),
          ('c_off', 'c_off', 7, 7, 0.5),
          ('wk4m', 'ones2', 7, 7, 0.25),
          ('d3row', 'c_off', 5, 13, 0.3333333333333333),
          ('c_off', 'd21T', 7, 9, 1.0),
          ('d3row', 'd21T', 5, 17, 0.16666666666666666),
          ('g4row', 'c_off', 5, 17, 0.08333333333333333),
          ('d21T', 'd21T', 7, 13, 0.25),
          ('c_off', 'wk4m', 9, 9, 0.25),
          ('g4row', 'd21T', 5, 19, 0.041666666666666664)],
 (3,): [('ones1', 'ones1', 2, None, 1.0),
        ('d3', 'ones1', 10, None, 0.16666666666666666),
        ('g4', 'ones1', 14, None, 0.041666666666666664)],
 (4,): [('ones1', 'ones1', 3, None, 1.0),
        ('d3', 'ones1', 11, None, 0.16666666666666666),
        ('g4', 'ones1', 15, None, 0.041666666666666664)]}

PK2K_TABLE = {(1,): [(((1,),), 1.0)],
 (1, 1): [(((1, 1),), 1.0)],
 (2,): [(((1,), (1,)), -1.0), (((2,),), 1.0)],
 (2, 1): [(((1, 0), (1, 1)), -2.0), (((2, 1),), 1.0)],
 (2, 2): [(((0, 1), (1, 0), (1, 1)), 4.0),
          (((0, 1), (2, 1)), -2.0),
          (((1, 0), (1, 2)), -2.0),
          (((1, 1), (1, 1)), -2.0),
          (((2, 2),), 1.0)],
 (3,): [(((1,), (1,), (1,)), 2.0), (((1,), (2,)), -3.0), (((3,),), 1.0)],
 (4,): [(((1,), (1,), (1,), (1,)), -6.0),
        (((1,), (1,), (2,)), 12.0),
        (((1,), (3,)), -4.0),
        (((2,), (2,)), -3.0),
        (((4,),), 1.0)]}

# Online mean-correction coefficients, (depth, n_features) = (16, 13); ridge-fitted
# offline (lam=1e-3) on the teacher-forced trajectory of 5 public v2-phase2 MLPs
# (k3_aug_corr.py, F47/F49). Feature order:
# 1, mu_pre, var_pre, sigma, alpha, |alpha|, phi, Phi, pred, D3, D21n, K4, sigma*phi
CORR_BETA = [
    [0.0006411968414813632, 0.0, 0.00032431282848653057, -0.0003042144758593203, 0.0, 0.0, 0.0, 0.0, -0.0007625526069403323, 0.0, 0.0, 0.0, -0.0007625526055534857],
    [-0.0004483531031709554, -8.08970612775365e-08, 0.00024032689914639557, 0.0003234947612489975, 7.734051179371133e-06, -8.401877300643312e-06, 0.0026976072084027533, -2.9522256040202606e-06, -1.1604426266687203e-05, 9.457618761612473e-06, -3.3186036376400295e-05, -0.008465131129601192, -0.002371947086869632],
    [-0.00030645743691215543, 2.660655267636766e-06, 0.0006192175275623578, -0.0002468827461681828, -2.9900896046038434e-06, 6.841942903175705e-08, 0.0033128783967715193, 5.123025223409135e-06, -1.831086058750464e-06, -8.948552609048747e-06, -3.292520087183994e-05, 0.024826431355466128, -0.003290775627340336],
    [-0.00010464301263958924, 1.0735707741330386e-06, 0.0009154900192795534, -0.0006467314093895341, 3.6870994916648376e-06, 6.046342808060834e-06, 0.00341846204288833, -8.975061063301133e-06, -8.178979563111877e-06, 9.51453794000349e-05, -3.005588379050443e-05, 0.031163100958057806, -0.003816642326501922],
    [-0.0006681684602995246, -7.2789662985629386e-06, 2.591925967081249e-05, 0.0007797247634619618, 8.477664175402185e-06, 1.7025270836169033e-06, 0.003419282618548069, 7.535308544950828e-07, -5.733309150056555e-06, -2.9160257628201262e-05, -1.722583323732972e-05, 0.04076990884979579, -0.004248021529544134],
    [-0.0006734457317912116, 3.1290157938005224e-05, 0.00022889868689835653, 0.0007465462593369972, 6.851356914590185e-07, 2.1755554980020755e-05, 0.004058574189699476, -4.277653393515956e-06, -6.039384411257449e-05, -9.02015920342041e-05, -1.9819140691513135e-05, 0.041867576221848744, -0.005466361831539052],
    [-0.0006750844151648486, 6.180105939061905e-05, 0.0004389649313793868, 0.0006339652749835816, -6.694983507237225e-07, 4.690187390705702e-05, 0.003977857680309406, 4.383031188174939e-06, -0.00012887677523401884, 0.0002505563048388552, -2.5566781991513985e-05, 0.07694913516123503, -0.0057040529266279925],
    [-0.000972879718186106, 5.1156558236482596e-05, -0.0004913239875599442, 0.0017893987001835238, 1.90937415308986e-06, 3.802173926700858e-05, 0.0038578329372384434, 2.8522169318710464e-06, -0.00010738838919747425, -0.00015415811216305824, -2.5539604454967108e-05, 0.08311668913952446, -0.005918296132102073],
    [-0.0007850991486069435, 7.899688664131035e-05, -0.00022144314048615243, 0.001495955739849875, 1.8359978888302925e-07, 5.023393155879106e-05, 0.0034762600760059295, 3.215904175585605e-06, -0.00015314900768603073, -0.0004475649473906088, -3.028530737478287e-05, 0.06152086703424944, -0.00569825101925022],
    [-0.0005795122172445841, 0.00011465424961793733, 0.00039693472854898914, 0.0007970833957439235, 6.825630979361196e-06, 7.966232390898656e-05, 0.0032776762940374586, -1.387178929061937e-06, -0.0002567385150822406, 0.00028972790478315425, -2.7005665153789784e-05, 0.07590370226079919, -0.005546910408445549],
    [-0.0004949253254431177, 9.892837012135289e-05, 0.0005826341044064549, 0.0006128630824666212, 1.8258883411132417e-06, 6.024891049057027e-05, 0.003354817421260185, -6.690225809034046e-06, -0.00020554230085297872, 0.0002695214755196366, -2.5128600253275433e-05, 0.08226262500093764, -0.006092394095370619],
    [-0.0007584153570441029, 0.0001792621247156503, 0.0001931413288044634, 0.0014013628928244146, 5.054405581529621e-06, 0.00010157511263920278, 0.004187883074782575, -5.420295053799967e-06, -0.00037802160373858374, 0.00014440604770373433, -2.2764810929109396e-05, 0.07738752039037704, -0.008027695439553941],
    [-0.0008291520607940194, 0.0002115021197150293, -0.00012740149902796318, 0.0017326132691935436, -2.1624432383474912e-06, 0.0001087192480144056, 0.004533661385130805, 4.837349485145022e-07, -0.00041674041200178005, 0.00018467937018199515, -1.9126957306380207e-05, 0.07896437764069765, -0.008986219483970914],
    [-0.000658293372173689, 0.00018380103458278174, 0.00046374648579651084, 0.001216035726510531, 4.012828172272205e-06, 9.570320459811334e-05, 0.004325867834172417, -1.2894882645758776e-05, -0.00038451298928645256, 0.0005113367132840055, -2.241500653445156e-05, 0.07104629639605271, -0.00897436256704376],
    [-0.0009527557374376014, 0.00019049799075311402, -0.0009213478791293237, 0.0025553941675787975, 3.508060734949607e-07, 9.248186905581894e-05, 0.004641633979578725, -3.1682752945685096e-06, -0.00038201626600816054, 3.0155769164602614e-05, -2.3334690641924517e-05, 0.07182226701958777, -0.01001369215216409],
    [-0.0006908532933348147, 0.00018408588405074964, 0.0004249876754929348, 0.0013835702671030146, 5.170891351788878e-06, 9.144035619901057e-05, 0.0044972055980902715, -1.0115381018603137e-05, -0.00038927641644205714, 0.0002930281376816756, -2.8026123982155874e-05, 0.157561818094021, -0.010070547059124272],
]

# ---- end embedded tables ----

# Regenerated kappa4 off-diagonal coefficient per layer, fitted at the PRE-activation
# (transported G_pre_off = LAM[l] * C_pre_off after layer l's ReLU); LS fits on the dense
# aug chain, mean over public P2 dumps 0-7 (runs/lam_prefit_d0-7.log, V22). The older
# post-ReLU table (runs/g_regen_10.log) differs mainly at layer 0 (1.95e-3 -> 4.99e-3).
# Suite shape only (gated like CORR_BETA).
LAM = [2.2e-3 / 0.95] * 16   # ablation copy: physical lambda (2.2e-3 after the 0.95 scale)
METRIC_C = 2.0
# debug kill-switches (parity_v17.py): set from the environment before import
import os as _os
import gc as _gc
NO_FEED = _os.environ.get("V17_NO_FEED", "0") == "1"
NO_WK431 = _os.environ.get("V17_NO_WK431", "0") == "1"
NO_REGEN = _os.environ.get("V17_NO_REGEN", "0") == "1"
NO_FB = _os.environ.get("V18_NO_FB", "0") == "1"  # V18: D21 feedback thin legs off
FULL_LAST = _os.environ.get("V19_FULL_LAST", "0") == "1"  # V19: 1 -> V18 op stream
NO_CONFINE = _os.environ.get("V21_NO_CONFINE", "0") == "1"  # V21: 1 -> V20 op stream
QPASS = int(_os.environ.get("V21_QPASS", "1"))  # V21: subspace-iteration passes of the basis range finder
NO_SRC_LAST = _os.environ.get("V19_NO_SRC_LAST", "0") == "1"  # V19 probe: D3(last) := 0, skip source transports
# The V16b CORR_BETA mean rider was ridge-fitted on V16b's OWN trajectory and is
# anti-correlated with V16b's base error; on the regen base it HURTS (dump0: 3.57e-8
# with vs 2.51e-8 without). Off by default until refitted on the V17 trajectory.
NO_CORR = _os.environ.get("V17_NO_CORR", "1") == "1"
# V25: table scale 0.95 (8-dump scan 0.9/0.92/0.95/1.0 -> 2.2752/2.2727/2.2704/2.2782e-8)
LAM = [c * float(_os.environ.get("V17_LAM_SCALE", "0.95")) for c in LAM]
# V25: reference ratio mean(dG)/mean(var) at layer l+1 (8-dump mean, scratch/lam_obs_d0-7.npz)
# and the log-log exponent of the adaptive rule (pooled fit 1.08; 1.0 shipped).
REF_R = [6.58815e-03, 8.18414e-03, 8.53136e-03, 8.38859e-03, 8.10153e-03, 7.74287e-03,
         7.35951e-03, 6.91093e-03, 6.55752e-03, 6.17892e-03, 5.83953e-03, 5.56824e-03,
         5.32459e-03, 5.05165e-03, 4.77181e-03]
BETA = float(_os.environ.get("V25_BETA", "1.0"))
DEBUG = []  # parity_v17.py: per-layer dict of diagnostics when V17_DEBUG=1

# pruned V16b table + the 4 (3,1)-slice use-side terms
_I = WICK_PAIRS.index
TERM_SPECS[(1, 1)].append(('wk431', 'ones2', _I((1, 1)), _I((3, 1)), 1.0 / 3.0))
TERM_SPECS[(2, 1)].append(('wk431', 'ones2', _I((1, 2)), _I((3, 1)), 1.0 / 6.0))
TERM_SPECS[(2, 1)].append(('wk431', 'ones2', _I((3, 2)), _I((1, 1)), 1.0 / 6.0))
TERM_SPECS[(2, 2)].append(('wk431', 'ones2', _I((1, 2)), _I((3, 2)), 1.0 / 3.0))
ALPHA_DEG = 5
SIG_EMIN, SIG_EMAX = -6, 4
D2_IPS = [(1, 1), (2, 1), (2, 2)]
D1_IPS = [(1,), (2,), (3,), (4,)]
D2_OUT = [(1, 1), (2, 1), (2, 2)]
D1_OUT = [(2,), (3,), (4,)]
MODE0_MISSING = {"d21", "d21T", "d3col", "d3row", "d3", "wk4m", "g4col", "g4row", "g4",
                 "wk431"}


def _statics(n: int) -> dict:
    c = 4 - 2 + n / 2.0 - 1.0
    P2 = (c - 2.0) / (16.0 * 2.0 * c * (1.0 - c) * (2.0 - c))
    return dict(P2=P2, k4_c4=24.0, k4_c22=24.0, wk4_c4=1.0, wk4_c22=1.0 / 3.0,
                cA=6.0 / (n + 4.0), cI=-3.0 / ((n + 2.0) * (n + 4.0)))


def _zero_diag_ip(A):
    fnp.fill_diagonal(A, 0.0)
    return A


def _build_wick_consts():
    np_ = len(WICK_PAIRS)
    C1 = [[0.0] * np_ for _ in range(ALPHA_DEG + 1)]
    C2 = [[0.0] * np_ for _ in range(ALPHA_DEG + 1)]
    # sigma-power selection with the constant prefactor folded in
    SELC = [[0.0] * np_ for _ in range(SIG_EMAX - SIG_EMIN + 1)]
    for j, (const, e, p1, p2) in enumerate(WICK_UNIFIED):
        SELC[e - SIG_EMIN][j] = const
        for i, c in enumerate(p1):
            C1[i][j] = c
        for i, c in enumerate(p2):
            C2[i][j] = c
    return C1, C2, SELC


def _term_prog(mode: int):
    """Fused-einsum program for the nonlin terms of one availability mode."""
    missing = MODE0_MISSING if mode == 0 else ({"wk431"} if mode == 2 else set())
    np_ = len(WICK_PAIRS)
    d2 = []  # (group, a, b, wl, wr, coef)
    d1 = []
    for g, ip in enumerate(D2_IPS):
        for a, b, wl, wr, coef in TERM_SPECS.get(ip, []):
            if a in missing or b in missing:
                continue
            d2.append((g, a, b, wl, wr, coef))
    for g, ip in enumerate(D1_IPS):
        for a, b, wl, wr, coef in TERM_SPECS.get(ip, []):
            if a in missing or b in missing:
                continue
            d1.append((g, a, b, wl, coef))
    SL2 = [[0.0] * np_ for _ in d2]
    SR2 = [[0.0] * np_ for _ in d2]
    IND2 = [[0.0] * len(D2_IPS) for _ in d2]
    AB2 = []
    for t, (g, a, b, wl, wr, coef) in enumerate(d2):
        SL2[t][wl] = coef
        SR2[t][wr] = 1.0
        IND2[t][g] = 1.0
        AB2.append((a, b) if a <= b else (b, a))
    SL1 = [[0.0] * np_ for _ in d1]
    IND1 = [[0.0] * len(D1_IPS) for _ in d1]
    B1 = []
    for t, (g, a, b, wl, coef) in enumerate(d1):
        SL1[t][wl] = coef
        IND1[t][g] = 1.0
        B1.append((a, b) if a <= b else (b, a))
    return dict(SL2=SL2, SR2=SR2, IND2=IND2, AB2=AB2, SL1=SL1, IND1=IND1, B1=B1)

# ---- V26 (F79): Strassen-Winograd block products on pooled buffers ----------------
STRASSEN_LEVELS = int(_os.environ.get("V26_STRASSEN", "5"))   # V28 ladder (suite harness, C/B | residual): L3 0.3107 | 0.170 s, L4 0.2973 | 0.187, L5 0.2872 | 0.205, L5 + shared basis 0.2680 | 0.242   # dump-0 ladder: 1/2/3 -> C/B 0.344/0.327/0.3125; mini residual at L3 = 0.356 s mean with a 0.42 s tail (7/100 failed) -> ship L1 until the base residual is cut
STRASSEN_HUB = int(_os.environ.get("V26_STRASSEN_HUB", str(STRASSEN_LEVELS)))
STRASSEN_NEW = int(_os.environ.get("V26_STRASSEN_NEW", "0"))   # V29: unused (the newborn rides the family)
CPRE_LEV = int(_os.environ.get("V29_CPRE_LEV", _os.environ.get("V26_STRASSEN", "5")))  # V29: C_pre block family levels
STRASSEN_SB = int(_os.environ.get("V28_STRASSEN_SB", str(STRASSEN_LEVELS)))   # V28: shared-basis formings + contractions   # newborn transport levels (0: ~30 ms residual for 0.5% cost is a bad trade)
STRASSEN_MIN = int(_os.environ.get("V26_STRASSEN_MIN", "16"))  # MERGE-tuned (rung 7): 32 -> 16, C/B -0.0056, raw +0.06% (noise) on untouched nets  # smallest leaf block side (V28: 32; f32 error grows ~1.4x per level, MSE unchanged to 4 digits at L5)
STRASSEN_FIRST = int(_os.environ.get("V28_STRASSEN_FIRST", "5"))  # MERGE-tuned (rung 7): stage off; first-call C/B 0.2576 -> 0.2347 (residual re-verified on a quiet machine)
STRASSEN_FUSE_P = int(_os.environ.get("V28_STRASSEN_FUSE_P", "343"))  # V28: fused per-product leaf when the leaf batch has >= this many blocks (deep levels: 7x smaller pools, +4 ops per leaf)  # V28: level cap of the first predict() of a process (see _predict_core)
WARM = _os.environ.get("V26_WARM", "1") == "1"


class _StrassenV29:
    """Recursive Strassen on batched operands. Layouts (last two axes = the matrix):
      plain: X (bx, P, m, kd) @ Y (by, P, kd, w) -> out (by, P, m, w), bx in {1, by}
      hub:   X (k, P, m, kd), Y (k, P, w, kd)    -> out (P, m, w) = sum_k X_k @ Y_k^T
    Every combo / product is ONE flopscope op over the whole batch, written with out=
    into pooled buffers shared by role (F53: fresh result buffers dominate residual).
    The seven products: M1=(X11+X22)(Y11+Y22) M2=(X21+X22)Y11 M3=X11(Y12-Y22)
    M4=X22(Y21-Y11) M5=(X11+X12)Y22 M6=(X21-X11)(Y11+Y12) M7=(X12-X22)(Y21+Y22);
    C11=M1+M4-M5+M7 C12=M3+M5 C21=M2+M4 C22=M1-M2+M3+M6.  For the hub kernel the
    right operand is Y^T, whose blocks are (Y^T)11=Y[..,:h,:q], (Y^T)12=Y[..,h:,:q],
    (Y^T)21=Y[..,:h,q:], (Y^T)22=Y[..,h:,q:] (no transposes: the kernel contracts j)."""

    def __init__(self, dtype, bmax):
        self.dtype = dtype
        self.bmax = int(bmax)
        self.pool = {}

    def _buf(self, key, shape):
        """V27: pools grow to the largest batch actually seen (the suite's young-source
        batch is <= 2 (AGE_OLD + 1), a third of the 2 (L-1) capacity), so deeper levels
        stay within memory; a larger batch later simply reallocates once."""
        b = self.pool.get(key)
        if b is None or b.shape[0] < shape[0]:
            b = fnp.empty(shape, dtype=self.dtype)
            fnp.copyto(b, 0.0)   # metered page touch (first-MLP residual, F80)
            self.pool[key] = b
        return b

    def _buf2(self, key, shape5):
        """V27: a pooled (b, 7, P, h, q) buffer together with its (b, 7P, h, q) view,
        both made once at allocation (fnp.reshape is a logged op: 0.06-0.3 ms per call,
        462 calls per MLP at four levels)."""
        e = self.pool.get(key)
        b = int(shape5[0])
        if e is None or e[0].shape[0] < b:
            b5 = fnp.empty(shape5, dtype=self.dtype)
            fnp.copyto(b5, 0.0)   # metered page touch (first-MLP residual, F80)
            b4 = fnp.reshape(b5, (b, shape5[1] * shape5[2], shape5[3], shape5[4]))
            e = (b5, b4)
            self.pool[key] = e
        return e[0][:b], e[1][:b]

    def level(self, m, kd, w, lev):
        """V28: largest level <= lev at which every block side divides and stays
        >= STRASSEN_MIN (a family whose sides do not allow `lev` runs shallower
        instead of falling back to dense)."""
        while lev > 0 and not self._ok(m, kd, w, lev):
            lev -= 1
        return lev

    @staticmethod
    def _ok(m, kd, w, lev):
        d = 2 ** lev
        return (lev > 0 and m % d == 0 and kd % d == 0 and w % d == 0
                and min(m, kd, w) // d >= STRASSEN_MIN)

    def _combos(self, X, Q, kind, key):
        """Seven Strassen combos of the quadrant views Q=(X11,X12,X21,X22) of X
        (b, P, m, kd) into a pooled (bmax, 7, P, h, q) buffer; returns (b, 7P, h, q)."""
        b, P, m, kd = X.shape
        h, q = m // 2, kd // 2
        buf, buf4 = self._buf2(key, (b, 7, P, h, q))
        X11, X12, X21, X22 = Q
        if kind == "L":
            fnp.add(X11, X22, out=buf[:, 0]); fnp.add(X21, X22, out=buf[:, 1])
            fnp.copyto(buf[:, 2], X11);       fnp.copyto(buf[:, 3], X22)
            fnp.add(X11, X12, out=buf[:, 4]); fnp.subtract(X21, X11, out=buf[:, 5])
            fnp.subtract(X12, X22, out=buf[:, 6])
        else:
            fnp.add(X11, X22, out=buf[:, 0]); fnp.copyto(buf[:, 1], X11)
            fnp.subtract(X12, X22, out=buf[:, 2]); fnp.subtract(X21, X11, out=buf[:, 3])
            fnp.copyto(buf[:, 4], X22);       fnp.add(X11, X12, out=buf[:, 5])
            fnp.add(X21, X22, out=buf[:, 6])
        return buf4

    @staticmethod
    def _assemble(M, out):
        """M (b, 7, P, h, w) products -> out (b, P, m, w) quadrants (8 ops)."""
        h, w = M.shape[3], M.shape[4]
        C11, C12 = out[..., :h, :w], out[..., :h, w:]
        C21, C22 = out[..., h:, :w], out[..., h:, w:]
        fnp.add(M[:, 0], M[:, 3], out=C11); fnp.subtract(C11, M[:, 4], out=C11)
        fnp.add(C11, M[:, 6], out=C11)
        fnp.add(M[:, 2], M[:, 4], out=C12)
        fnp.add(M[:, 1], M[:, 3], out=C21)
        fnp.subtract(M[:, 0], M[:, 1], out=C22); fnp.add(C22, M[:, 2], out=C22)
        fnp.add(C22, M[:, 5], out=C22)

    def mm(self, X, Y, out, lev):
        """plain: X (bx, P, m, kd), Y (by, P, kd, w) -> out (by, P, m, w)."""
        bx, P, m, kd = X.shape
        by, w = Y.shape[0], Y.shape[3]
        if not self._ok(m, kd, w, lev):
            fnp.matmul(X, Y, out=out)
            return
        h, q, v = m // 2, kd // 2, w // 2
        XQ = (X[..., :h, :q], X[..., :h, q:], X[..., h:, :q], X[..., h:, q:])
        YQ = (Y[..., :q, :v], Y[..., :q, v:], Y[..., q:, :v], Y[..., q:, v:])
        if P >= STRASSEN_FUSE_P and not self._ok(h, q, v, lev - 1):
            # V28: fused leaf (deep levels only) -- one product at a time into the out quadrants
            self._leaf_mm(XQ, YQ, (out[..., :h, :v], out[..., :h, v:], out[..., h:, :v], out[..., h:, v:]),
                          bx, by, P, h, q, v)
            return
        Xc = self._combos(X, XQ, "L", ("L", P, h, q))
        Yc = self._combos(Y, YQ, "R", ("R", P, q, v))
        # V28: the products reuse this level's left-combo buffer when the callee recurses
        # (its combos consume Xc before anything is written into out); a fused-leaf callee
        # reads X while writing out, so it keeps a separate M. Square blocks only.
        callee_recurses = self._ok(h // 2, q // 2, v // 2, lev - 2)
        mkey = ("L", P, h, q) if (callee_recurses and q == v) else ("M", P, h, v)
        Mb, Mb4 = self._buf2(mkey, (by, 7, P, h, v))
        self.mm(Xc, Yc, Mb4, lev - 1)
        self._assemble(Mb, out)

    def _leaf_mm(self, XQ, YQ, OQ, bx, by, P, h, q, v):
        """Fused leaf of mm: M6->C22, M4->C11, M1 (+C11, +C22), M2->C21 (C22-=), M3->C12
        (C22+=), M5 (C11-=, C12+=), M7 (C11+=): 7 matmul writes + 8 adds, one combo
        pair alive at a time (copies for the raw-quadrant operands keep every matmul
        input contiguous)."""
        X11, X12, X21, X22 = XQ
        Y11, Y12, Y21, Y22 = YQ
        C11, C12, C21, C22 = OQ
        Lb = self._buf(("L1", P, h, q), (bx, P, h, q))[:bx]
        Rb = self._buf(("R1", P, q, v), (by, P, q, v))[:by]
        Mb = self._buf(("M1", P, h, v), (by, P, h, v))[:by]
        # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
        fnp.subtract(X21, X11, out=Lb); fnp.add(Y11, Y12, out=Rb)
        fnp.matmul(Lb, Rb, out=C22)                                   # M6 -> C22
        fnp.add(X21, X22, out=Lb)
        fnp.matmul(Lb, Y11, out=C21)                                  # M2 -> C21
        fnp.subtract(C22, C21, out=C22)                               # C22 -= M2
        fnp.subtract(Y21, Y11, out=Rb)
        fnp.matmul(X22, Rb, out=C11)                                  # M4 -> C11
        fnp.add(C21, C11, out=C21)                                    # C21 += M4
        fnp.add(X11, X22, out=Lb); fnp.add(Y11, Y22, out=Rb)
        fnp.matmul(Lb, Rb, out=Mb)                                    # M1
        fnp.add(C11, Mb, out=C11); fnp.add(C22, Mb, out=C22)
        fnp.subtract(Y12, Y22, out=Rb)
        fnp.matmul(X11, Rb, out=C12)                                  # M3 -> C12
        fnp.add(C22, C12, out=C22)                                    # C22 += M3
        fnp.add(X11, X12, out=Lb)
        fnp.matmul(Lb, Y22, out=Mb)                                   # M5
        fnp.subtract(C11, Mb, out=C11); fnp.add(C12, Mb, out=C12)
        fnp.subtract(X12, X22, out=Lb); fnp.add(Y21, Y22, out=Rb)
        fnp.matmul(Lb, Rb, out=Mb)                                    # M7
        fnp.add(C11, Mb, out=C11)

    def hub(self, X, Y, out, lev):
        """hub: X (k, P, m, kd), Y (k, P, w, kd) -> out (P, m, w) = sum_k X_k Y_k^T."""
        k, P, m, kd = X.shape
        w = Y.shape[2]
        if not self._ok(m, kd, w, lev):
            # dense: batched GEMM over (k, P) then a k-sum (the einsum form
            # 'kpij,kpcj->pic' takes a slow non-BLAS path: 0.5 s per call at 512^2)
            T = self._buf(("K", P, m, w), (k, P, m, w))[:k]
            fnp.matmul(X, fnp.swapaxes(Y, -1, -2), out=T)
            fnp.sum(T, axis=0, out=out)
            return
        h, q, v = m // 2, kd // 2, w // 2
        XQ = (X[..., :h, :q], X[..., :h, q:], X[..., h:, :q], X[..., h:, q:])
        # (Y^T) blocks expressed on the un-transposed Y (rows c, cols j)
        YQ = (Y[..., :v, :q], Y[..., v:, :q], Y[..., :v, q:], Y[..., v:, q:])
        if P >= STRASSEN_FUSE_P and not self._ok(h, q, v, lev - 1):
            self._leaf_hub(XQ, YQ, (out[..., :h, :v], out[..., :h, v:], out[..., h:, :v], out[..., h:, v:]),
                           k, P, h, q, v)
            return
        Xc = self._combos(X, XQ, "L", ("L", P, h, q))
        Yc = self._combos(Y, YQ, "R", ("R", P, v, q))
        Mb, Mb4 = self._buf2(("HM", P, h, v), (1, 7, P, h, v))
        self.hub(Xc, Yc, Mb4[0], lev - 1)
        self._assemble(Mb, out[None])

    def _leaf_hub(self, XQ, YQ, OQ, k, P, h, q, v):
        """Fused leaf of hub (same product order as _leaf_mm); each product is a batched
        GEMM over (k, P) into T then a k-sum into its quadrant / the (P, h, v) scratch."""
        X11, X12, X21, X22 = XQ
        Y11, Y12, Y21, Y22 = YQ
        C11, C12, C21, C22 = OQ
        Lb = self._buf(("L1", P, h, q), (k, P, h, q))[:k]
        Rb = self._buf(("R1", P, v, q), (k, P, v, q))[:k]
        T = self._buf(("K", P, h, v), (k, P, h, v))[:k]
        Mb = self._buf(("KS", P, h, v), (1, P, h, v))[0]
        def prod(L_, R_, dst):
            fnp.matmul(L_, fnp.swapaxes(R_, -1, -2), out=T)
            fnp.sum(T, axis=0, out=dst)

        fnp.subtract(X21, X11, out=Lb); fnp.add(Y11, Y12, out=Rb); prod(Lb, Rb, C22)   # M6 -> C22
        fnp.add(X21, X22, out=Lb); prod(Lb, Y11, C21)                                # M2 -> C21
        fnp.subtract(C22, C21, out=C22)                                              # C22 -= M2
        fnp.subtract(Y21, Y11, out=Rb); prod(X22, Rb, C11)                           # M4 -> C11
        fnp.add(C21, C11, out=C21)                                                   # C21 += M4
        fnp.add(X11, X22, out=Lb); fnp.add(Y11, Y22, out=Rb); prod(Lb, Rb, Mb)       # M1
        fnp.add(C11, Mb, out=C11); fnp.add(C22, Mb, out=C22)
        fnp.subtract(Y12, Y22, out=Rb); prod(X11, Rb, C12)                           # M3 -> C12
        fnp.add(C22, C12, out=C22)                                                   # C22 += M3
        fnp.add(X11, X12, out=Lb); prod(Lb, Y22, Mb)                                 # M5
        fnp.subtract(C11, Mb, out=C11); fnp.add(C12, Mb, out=C12)
        fnp.subtract(X12, X22, out=Lb); fnp.add(Y21, Y22, out=Rb); prod(Lb, Rb, Mb)  # M7
        fnp.add(C11, Mb, out=C11)


class _Pool:
    """V27 (F80): persistent named scratch buffers. A fresh (n,n) result costs ~0.10 ms
    of residual (allocation + wrapper), the same op written with out= into a pooled
    buffer ~0.013 ms; the pool lives on the estimator across predict() calls (per-MLP
    allocation made the residual creep over a suite, F79). Keys carry the shape, so a
    different width simply adds buffers."""

    def __init__(self, dtype):
        self.dtype = dtype
        self.bufs = {}

    def get(self, name, shape):
        key = (name, tuple(int(x) for x in shape))
        b = self.bufs.get(key)
        if b is None:
            b = fnp.empty(key[1], dtype=self.dtype)
            fnp.copyto(b, 0.0)   # metered page touch (see class note)
            self.bufs[key] = b
        return b

    def get_pair(self, name, shape):
        """V28: an (S, 2, a, b) slot buffer together with its one-time (2S, 1, a, b) view
        (fnp.reshape is a logged, per-element billed op; slabs of whole slots stay
        contiguous, so [2i:2j] of the view is slots i..j-1 with the pair axis folded
        into the Strassen batch)."""
        key = ("pair", name, tuple(int(x) for x in shape))
        e = self.bufs.get(key)
        if e is None:
            b5 = fnp.empty(key[2], dtype=self.dtype)
            fnp.copyto(b5, 0.0)
            b4 = fnp.reshape(b5, (key[2][0] * key[2][1], 1, key[2][2], key[2][3]))
            e = (b5, b4)
            self.bufs[key] = e
        return e


class _V25Core(BaseEstimator):  # 504aldo V29, renamed
    AGE_OLD = int(_os.environ.get("V21_AGE_OLD", "4"))   # V21: sources with age > AGE_OLD are confined
    R_OLD = int(_os.environ.get("V21_R_OLD", "352"))     # V21 lean ladder F72 ships 384; MERGE-tuned (rung 7): 352 measured raw +2.3% (hard subset) for C/B -0.0073; full untouched-set mean 2.2721e-8 <= 2.35e-8 gate
    AGE_OLD2 = int(_os.environ.get("V24_AGE_OLD2", "7"))  # V24: age gate of the nested tier (0 = off)
    R_OLD2 = int(_os.environ.get("V24_R_OLD2", "224"))    # V24: rank of the nested sub-basis U (8-dump frontier runs/v24_7_*_d0-7.log: 224 best, 128 cliff)
    QPASS2 = int(_os.environ.get("V24_QPASS2", "2"))      # V24: passes of the r1-space range finder
    R_FB = 16    # V18: rank of the D21 feedback thin legs (F69 lean ladder: 8/16/32 -> 2.17/2.14/2.14e-8)
    R_RES = 16   # rank of the S21 residual leg (V17 ladder on dumps 0/1: 16/32/64 -> 2.35/2.39/2.41e-8 at 0.492/0.502/0.518xB)

    def __init__(self) -> None:
        self._setup_rng = None
        self._wick_consts = _build_wick_consts()
        self._progs = {0: _term_prog(0), 1: _term_prog(1), 2: _term_prog(2)}
        self._i11 = WICK_PAIRS.index((1, 1))
        self._i21 = WICK_PAIRS.index((2, 1))
        self._i12 = WICK_PAIRS.index((1, 2))
        self._i31 = WICK_PAIRS.index((3, 1))

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)
        if WARM:
            # V26: first-call warm-ups (the residual audit showed one-off 5-30 ms gaps
            # before the first qr / norm.pdf / sandwich of every MLP: library init).
            try:
                f32 = fnp.float32
                z = fnp.zeros((64, 16), dtype=f32)
                fnp.linalg.qr(z + 1.0)
                v = fnp.zeros(64, dtype=f32)
                flops.stats.norm.pdf(v)
                flops.stats.norm.cdf(v)
                e = flops.as_symmetric(fnp.eye(64, dtype=f32), symmetry=(0, 1))
                fnp.einsum("ij,ia,jb->ab", e, z[:, :16] + 1.0, z[:, :16] + 1.0)
                # one tiny call per op signature the chain uses (matmul / einsum path
                # and symmetry machinery initialize lazily: 24 ms before the first
                # small matmul, 1-2 ms before each first einsum subscript)
                a = fnp.zeros((4, 21), dtype=f32) + 1.0
                b = fnp.zeros((21, 64), dtype=f32) + 1.0
                m = fnp.zeros((64, 64), dtype=f32) + 1.0
                t3 = fnp.zeros((3, 64, 64), dtype=f32) + 1.0
                g3 = fnp.zeros((3, 3), dtype=f32) + 1.0
                _ = a @ b
                _ = m @ m
                _ = m @ v
                fnp.einsum("tij,ti,tj,tg->gij", t3, t3[:, :, 0], t3[:, 0, :], g3)
                fnp.einsum("ti,ti,tg->gi", t3[:, :, 0], t3[:, :, 1], g3)
                fnp.einsum("iq,kqn->kin", m[:, :16], t3[:, :16, :])
                fnp.einsum("ki,kc->ic", t3[:, :, 0], t3[:, :, 1])
                fnp.einsum("kij,kij->i", t3, t3)
                fnp.einsum("kij,kj->ki", t3, t3[:, 0, :])
                fnp.einsum("ki,ki->i", t3[:, :, 0], t3[:, :, 1])
                fnp.einsum("kij,kqj->iq", t3, t3[:, :16, :])
                fnp.einsum("kiq,kjq->kij", t3, t3)
                fnp.einsum("kij,kjq->kiq", t3, t3)
                fnp.einsum("kiq,kcq->ic", t3, t3)
                fnp.einsum("pq,kqn->kpn", m[:16, :16], t3[:, :16, :])
                fnp.einsum("cij,cjk->cik", t3[:1], t3)
                fnp.sum(t3, axis=(0, 2))
                fnp.sum(t3, axis=0)
                fnp.diag(m)
                fnp.diagflat(v)
                fnp.fill_diagonal(m, 0.0)
                fnp.matmul(t3, fnp.swapaxes(t3, -1, -2))
                fnp.concatenate([m, m], axis=1)
                fnp.stack([v, v], axis=1)
                fnp.maximum(v, 1e-10)
                fnp.sqrt(v + 1.0)
                fnp.power(v + 1.0, 1.0)
                fnp.clip(v, 0.5, 2.0)
                fnp.mean(v)
                fnp.copy(m)
                fnp.abs(v)
            except Exception:  # noqa: BLE001  warm-up must never fail a submission
                pass

    # ------------------------------------------------------------------
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        was = _gc.isenabled()
        _gc.disable()
        try:
            return self._predict_core(mlp, budget)
        finally:
            if was:
                _gc.enable()

    def _predict_core(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        n = mlp.width
        f32 = fnp.float32
        st = _statics(n)
        metric2 = 2.0 ** 2
        L = len(mlp.weights)
        # vec+ww K4 transport and the fitted mean correction are calibrated for the
        # Phase-2 suite shape ONLY; any other shape (e.g. the grader smoke MLP)
        # runs the shape-generic V1.6 base path. The correction destabilizes small
        # widths (non-finite blowup observed at 16x64) -- never apply it off-suite.
        riders = (n == 1024 and L == len(CORR_BETA))

        C1l, C2l, SELCl = self._wick_consts
        C1 = fnp.asarray(C1l, dtype=f32)
        C2 = fnp.asarray(C2l, dtype=f32)
        SELC = fnp.asarray(SELCl, dtype=f32)
        FP = {}
        for mode, p in self._progs.items():
            FP[mode] = {k: fnp.asarray(v, dtype=f32)
                        for k, v in p.items() if k in ("SL2", "SR2", "IND2", "SL1", "IND1")}
        ones_n = fnp.ones(n, dtype=f32)
        zeros_n = fnp.zeros(n, dtype=f32)
        ones_col = (ones_n)[:, None]
        ones_row = ones_col.T
        ones2 = ones_col * ones_row
        beta_rows = [fnp.asarray(b, dtype=f32) for b in CORR_BETA]
        # V20: one persistent buffer for the d=2 term products (max terms over modes)
        t2max = max(len(p["AB2"]) for p in self._progs.values())
        # V27 (F80): persistent pooled buffers (see _Pool)
        pool = getattr(self, "_pool", None)
        if pool is None or pool.dtype != f32:
            pool = _Pool(f32)
            self._pool = pool

        def NN(name):
            return pool.get(name, (n, n))

        abbuf = pool.get("abbuf", (t2max, n, n))
        T1 = NN("t1")   # layer-local (n,n) scratch

        mu = fnp.zeros(n, dtype=f32)
        C = flops.as_symmetric(fnp.eye(n, dtype=f32), symmetry=(0, 1))
        A_st = P_st = Z_st = L_st = None
        newborn = None
        w2b_list = []
        s_list = []
        e_list = []
        c1_list = []   # per source: lambda_b * w1_b   (X3 = A*c1 + P*c2, column scalings)
        c2_list = []   # per source: w1_b^2 * dG_pre_b
        y_list = []    # per source: (m/4) w2_b       (Y3 = y 1^T)
        g_prev = None  # post-ReLU kappa4 diagonal core of the previous layer
        var_prev = None
        lam_prev = 0.0
        regen = riders and not NO_REGEN  # memoryless kappa4 channel: suite shape only
        bufs = None
        legs = None
        # V26: pooled Strassen products (F79); the pools persist across predict() calls
        # (allocating/freeing ~2.5 GB per MLP made the residual clock creep up over a
        # 100-MLP suite). Pool keys carry the block shapes, so a different width simply
        # adds new buffers; a different depth changes the batch capacity -> rebuild.
        smm = getattr(self, "_smm", None)
        if smm is None or smm.bmax != 2 * (L - 1) or smm.dtype != f32:
            smm = _StrassenV29(f32, 2 * (L - 1))
            self._smm = smm
        ncall = getattr(self, "_ncall", 0)
        self._ncall = ncall + 1
        s_lev = STRASSEN_LEVELS if ncall > 0 else min(STRASSEN_LEVELS, STRASSEN_FIRST)
        s_sb = STRASSEN_SB
        self._s_hub = s_lev
        r = min(int(self.R_RES), n)  # smoke shapes can be narrower than the rank
        rfb = 0 if NO_FB else min(int(self.R_FB), n)  # V18 feedback rank
        # V21: shared-basis state for old sources (suite-width only: r must be < n)
        r_old = int(self.R_OLD)
        confine = (not NO_CONFINE) and r_old < n
        age_old = int(self.AGE_OLD)
        ka = 0           # number of confined (old) sources = leading slots of the stacks
        Qc = None        # (n, r) current basis of the old legs (pre-wick of this layer)
        fa_side = f2_side = qc_side = z_side = 0   # V27: ping-pong buffer sides
        FAP = FAP2 = None    # V28: interleaved factor slabs (m, 2, r, n) [slot, {A, P}]
        fap4 = fap24 = None  # their one-time (2 (L-1), 1, r, n) views (current side)
        fa_off = 0           # slot offset of FAP inside its side buffer (1 after a nest)
        legs4 = None
        FAo = FPo = None   # (ka, r, n) static factors: A_s = Qc FA_s, P_s = Qc FP_s
        Sg = None        # (r, r) weighted Gram core of the TIER-1 legs in the factor basis
        # V24: nested tier 2 (slots [0:kb] of the stacks, the oldest sources)
        r2 = int(self.R_OLD2)
        age_old2 = int(self.AGE_OLD2)
        nest = confine and age_old2 > 0 and r2 < r_old
        kb = 0           # number of tier-2 sources (leading slots, subset of the ka confined)
        U = None         # (r1, r2) shared sub-basis of tier 2 inside Qc's factor space
        FA2 = FP2 = None   # (kb, r2, n) static factors: A_s = Qc U FA2_s
        S2 = None        # (r2, r2) weighted Gram core of tier 2 in the U basis
        QU = None        # (n, r2) = Qc U, formed once per layer
        dA_list = []     # per source: hub-column Gram weights of the A-type legs
        dP_list = []     # per source: hub-column Gram weights of the P-type legs
        R1T_st = R2T_st = None  # V18: static right factors of the thin legs, (k, n, rfb)
        Zf_st = None  # V18: transported thin legs [F1 | F2], (k, n, 2 rfb); separate from Z
                      # so the M-leg einsums (Z L^T, PPL Z^T) never see them
        K4_sigma = None
        K4_vec = None
        rows = []

        w1_prev = None  # wick w(1) of the previous layer, folded into WD

        for li, w in enumerate(mlp.weights):
            last = li == L - 1
            trim = last and not FULL_LAST  # V19: mean-only final layer
            skip_src = trim and NO_SRC_LAST
            w32 = w if w.dtype == fnp.float32 else w.astype(f32)
            W = w32.T
            # ---- linear ----
            mu = W @ mu
            C_pre = None
            if li == 0:
                # V29: C = I at the input, so the sandwich is the plain Gram w32^T w32
                # (aliased 2-operand einsum: exact, 0.5 u, output tagged symmetric; F77)
                if trim:
                    var = fnp.maximum(fnp.sum(w32 * w32, axis=0), 1e-10)
                else:
                    C_pre = fnp.einsum("ia,ib->ab", w32, w32)
            elif skip_src:
                # probe path only (V19_NO_SRC_LAST): no family at this layer
                if trim:
                    var = fnp.maximum(fnp.sum(w32 * (C @ w32), axis=0), 1e-10)
                else:
                    C_pre = fnp.einsum("ij,ia,jb->ab", C, w32, w32)
            # li >= 1: C_pre is formed after the transport family below (W C rides in it)
            # Source stacks evolve by W @ diag(w1_prev); the newborn (added after
            # last layer's wick) evolves by the raw W. Fold the wick into WD so
            # the stacks never need a separate (k,n,n) scaling pass.
            # Dense legs live in ping-pong slot buffers (F53: no fresh (k,n,n)
            # results, no concatenates): transport k slots into the spare buffer,
            # write the newborn into slot k, swap.
            if A_st is not None and not skip_src:
                k = A_st.shape[0]
                WD = fnp.multiply(W, (w1_prev)[None, :], out=NN("wd"))
                WDb = WD[None]
                # V21: one source per layer crosses the age gate; join it to the shared
                # basis (post-wick legs, like the lean chain), skip the join at the last
                # layer (its legs are used once more, densely, for D3 only).
                ka_t = ka
                if confine and not last:
                    ka_t = max(ka, min(k, li - age_old))
                if ka_t > ka:
                    j = ka  # the joiner (exactly one per layer)
                    w1c = (w1_prev)[:, None]
                    Aj = A_st[j]   # V29: dense legs are already w1-scaled (wick stage)
                    Pj = P_st[j]
                    dAj = (dA_list[j])[:, None]
                    dPj = (dP_list[j])[:, None]
                    Om = pool.get("om", (n, r_old))         # fixed sketch (n, r)
                    fnp.copyto(Om, w32[:, :r_old])
                    Qp = fnp.multiply(w1c, Qc, out=pool.get("qp", (n, r_old))) if ka > 0 else None
                    Yq = pool.get("yq", (n, r_old))
                    Yq1 = pool.get("yq1", (n, r_old))
                    Yq2 = pool.get("yq2", (n, r_old))
                    lev_j = smm.level(n, n, r_old, 3)  # MERGE: joins via Strassen
                    for _pass in range(QPASS):
                        # Yq = G Om with G = Aj dA Aj^T + Pj dP Pj^T + Qp Sg Qp^T
                        smm.mm(Aj.T[None, None], Om[None, None],
                               Yq1[None, None], lev_j)
                        fnp.multiply(dAj, Yq1, out=Yq1)
                        smm.mm(Aj[None, None], Yq1[None, None],
                               Yq[None, None], smm.level(n, r_old, r_old, 3))
                        smm.mm(Pj.T[None, None], Om[None, None],
                               Yq1[None, None], lev_j)
                        fnp.multiply(dPj, Yq1, out=Yq1)
                        smm.mm(Pj[None, None], Yq1[None, None],
                               Yq2[None, None], smm.level(n, r_old, r_old, 3))
                        fnp.add(Yq, Yq2, out=Yq)
                        if ka > 0:
                            # V24: full core of the confined legs = tier-1 core + lifted tier-2 core
                            Sfull = Sg if kb == 0 else Sg + U @ (S2 @ U.T)
                            Sq = pool.get("yq_rr", (r_old, r_old))
                            smm.mm(Qp.T[None, None], Om[None, None],
                                   Sq[None, None], smm.level(r_old, n, r_old, 3))
                            smm.mm(Qp[None, None], (Sfull @ Sq)[None, None],
                                   Yq2[None, None], smm.level(n, r_old, r_old, 3))
                            fnp.add(Yq, Yq2, out=Yq)
                        Qn, _ = fnp.linalg.qr(Yq)           # (n, r) orthonormal
                        Om = Qn
                    # V27: factors live in ping-pong slot buffers (rotation reads one side,
                    # writes the other; the joiner takes slot m1 = number of tier-1 members)
                    fa_side ^= 1
                    fap_new, fap4_new = pool.get_pair(("fap", fa_side), (L - 1, 2, r_old, n))
                    m1 = ka - kb
                    # MERGE: factor projections via Strassen (their 4-D slot views)
                    smm.mm(Qn.T[None, None], Aj[None, None],
                           fap4_new[2 * m1:2 * m1 + 1], smm.level(r_old, n, n, 3))
                    smm.mm(Qn.T[None, None], Pj[None, None],
                           fap4_new[2 * m1 + 1:2 * m1 + 2], smm.level(r_old, n, n, 3))
                    FAj = fap_new[m1, 0]
                    FPj = fap_new[m1, 1]
                    if ka > 0:
                        Tq = Qn.T @ Qp                      # (r, r) rotation of the old factors
                        if ka > kb:
                            # MERGE: factor rotation via Strassen on the paired views
                            smm.mm(Tq[None, None], fap4[2 * fa_off:2 * (fa_off + m1)],
                                   fap4_new[:2 * m1], smm.level(r_old, r_old, n, 3))
                            Sg = Tq @ (Sg @ Tq.T)
                        else:
                            Sg = None
                        if kb > 0:
                            U = Tq @ U                      # V24: the sub-basis rides along
                    FAP = fap_new[:m1 + 1]
                    fap4 = fap4_new
                    fa_off = 0
                    FAo = FAP[:, 0]
                    FPo = FAP[:, 1]
                    Sj = (FAj @ (dAj * FAj.T)) + (FPj @ (dPj * FPj.T))
                    Sg = Sj if Sg is None else Sg + Sj
                    # V24: nested move of the oldest tier-1 member into tier 2, in the
                    # orthonormal coordinates Qn of this rebuild (lean: youngest gate first,
                    # then the older gate on the already-confined legs).
                    kb_t = kb
                    if nest:
                        kb_t = max(kb, min(ka_t - 1, li - age_old2))
                    if kb_t > kb:
                        assert kb_t == kb + 1, (li, kb, kb_t)
                        jb = kb                             # stack slot of the mover
                        FA1 = FAo[0]
                        FP1 = FPo[0]
                        dAb = (dA_list[jb])[:, None]
                        dPb = (dP_list[jb])[:, None]
                        S_s = (FA1 @ (dAb * FA1.T)) + (FP1 @ (dPb * FP1.T))   # (r1, r1)
                        G2 = S_s if kb == 0 else S_s + U @ (S2 @ U.T)
                        Om2 = fnp.copy(w32[:r_old, :r2])                        # sketch (r1, r2)
                        for _pass in range(int(self.QPASS2)):
                            Un, _ = fnp.linalg.qr(G2 @ Om2)
                            Om2 = Un
                        f2_side ^= 1
                        fap2_new, fap24_new = pool.get_pair(("fap2", f2_side), (L - 1, 2, r2, n))
                        fnp.matmul(Un.T, FA1, out=fap2_new[kb, 0])
                        fnp.matmul(Un.T, FP1, out=fap2_new[kb, 1])
                        if kb > 0:
                            T2 = Un.T @ U                                       # (r2, r2)
                            fnp.matmul(T2[None, None], FAP2, out=fap2_new[:kb])
                        FAP2 = fap2_new[:kb + 1]
                        fap24 = fap24_new
                        FA2 = FAP2[:, 0]
                        FP2 = FAP2[:, 1]
                        S2 = Un.T @ (G2 @ Un)
                        U = Un
                        FAP = FAP[1:]
                        fa_off = 1
                        FAo = FAP[:, 0]
                        FPo = FAP[:, 1]
                        Sg = Sg - S_s
                        kb = kb_t
                    qc_side ^= 1
                    Qc = fnp.matmul(W, Qn, out=pool.get(("qc", qc_side), (n, r_old)))  # wick already inside Qn
                    ka = ka_t
                elif ka > 0:
                    qc_side ^= 1
                    Qc = fnp.matmul(WD, Qc, out=pool.get(("qc", qc_side), (n, r_old)))
                if ka > kb:
                    # formed dense legs of the tier-1 sources (pre-wick of this layer):
                    # V28 ONE Strassen family over the interleaved (A, P) factor slab
                    m1 = ka - kb
                    smm.mm(Qc[None, None], fap4[2 * fa_off:2 * (fa_off + m1)],
                           legs4["AP1"][2 * kb:2 * ka], smm.level(n, r_old, n, s_sb))
                if kb > 0:
                    # V24: tier-2 legs through the sub-basis
                    QU = fnp.matmul(Qc, U, out=pool.get("qu", (n, r2)))
                    smm.mm(QU[None, None], fap24[:2 * kb], legs4["AP1"][:2 * kb],
                           smm.level(n, r2, n, s_sb))
                # V29: the dense young legs were pre-scaled by w1_prev at the previous
                # wick, so the family's left operand is the raw W; the newborn's A leg
                # (slot k A position, born w1-scaled) and the covariance C (slot k P
                # position) ride along: out [ka:k] = transported legs, [k,0] = W a_b,
                # [k,1] = W C (consumed by the C_pre block below, then overwritten by W).
                # Level 0 falls back to the dense batched matmul inside mm().
                extra = 2 if newborn is not None else 0
                if 2 * k + extra > 2 * ka:
                    smm.mm(W[None, None], legs4["AP0"][2 * ka:2 * k + extra],
                           legs4["AP1"][2 * ka:2 * k + extra], smm.level(n, n, n, s_lev))
                z_side ^= 1
                Z_st = fnp.matmul(WDb, Z_st, out=pool.get(("z", z_side), (L - 1, n, r + 2))[:k])
                if Zf_st is not None:
                    Zf_st = fnp.matmul(WDb, Zf_st, out=pool.get(("zf", z_side), (L - 1, n, 2 * rfb))[:k])
                legs["AP0"], legs["AP1"] = legs["AP1"], legs["AP0"]
                legs4["AP0"], legs4["AP1"] = legs4["AP1"], legs4["AP0"]
            else:
                k = 0
            if newborn is not None and not skip_src:
                a_b, Rr, Lr, s_b, e_b, Ff = newborn
                if A_st is None:
                    # V29: layer 1 has no stack yet: the newborn + C family alone (slot 0
                    # written at birth on the AP0 side; output to AP1, then swap)
                    smm.mm(W[None, None], legs4["AP0"][0:2], legs4["AP1"][0:2],
                           smm.level(n, n, n, s_lev))
                    legs["AP0"], legs["AP1"] = legs["AP1"], legs["AP0"]
                    legs4["AP0"], legs4["AP1"] = legs4["AP1"], legs4["AP0"]
                # V29: C_pre from the transported covariance W C in the newborn's P slot
                WC = legs["AP0"][k, 1]
                if trim:
                    # var = diag(W C W^T) = rowsum((W C) * W)
                    var = fnp.maximum(fnp.sum(fnp.multiply(WC, W, out=T1), axis=1), 1e-10)
                else:
                    C_pre = self._sym_product(WC, w32, n, NN("cpre"), min(CPRE_LEV, s_lev))
                fnp.copyto(legs["AP0"][k, 1], W)
                A_st = legs["AP0"][:k + 1, 0]
                P_st = legs["AP0"][:k + 1, 1]
                # V27: the newborn's thin columns go into slot k of the current Z side
                # (the transport above wrote slots [:k] of that side); L was written into
                # its slot at birth.
                zb = pool.get(("z", z_side), (L - 1, n, r + 2))
                fnp.matmul(W, Rr, out=zb[k])
                Z_st = zb[:k + 1]
                L_st = pool.get("l", (L - 1, n, r + 2))[:k + 1]
                if Ff is not None:
                    zfb = pool.get(("zf", z_side), (L - 1, n, 2 * rfb))
                    fnp.matmul(W, Ff, out=zfb[k])
                    Zf_st = zfb[:k + 1]
                s_list.append(s_b)
                e_list.append(e_b)
                newborn = None

            # ---- WK slices ----
            if trim:
                C_off = None
            else:
                var = fnp.maximum(fnp.diag(C_pre), 1e-10)
                C_off = _zero_diag_ip(C_pre)
            mode = 0 if A_st is None else 1
            if mode == 1 and skip_src:
                D3, D21 = fnp.zeros(n, dtype=f32), None
                if regen:
                    WW = fnp.multiply(W, W, out=NN("ww"))
                    dG = WW @ (g_prev - var_prev * lam_prev) + var * lam_prev
                    g4row = dG * METRIC_C
                    wk4m = wk431 = None
                elif riders:
                    g4row = ((W * W) @ K4_vec) * 0.5 * float(st["wk4_c4"] * metric2)
                    wk4m = None
                else:
                    g4row = ones_n * (K4_sigma * float(st["wk4_c4"] * metric2))
                    wk4m = None
            elif mode == 1:
                if bufs is None:
                    bufs = {nm: pool.get(nm, (L - 1, n, n))
                            for nm in ("ap", "pp", "t", "mp", "xt", "yt", "u")}
                    bufs["lap"], bufs["lap4"] = pool.get_pair("lap", (L - 1, 2, n, n))   # V26: [LA | LP]
                    bufs["hub"] = pool.get("hub", (n, n))             # V26: hub result
                    bufs["ppl"] = pool.get("ppl", (L - 1, n, r + 2))  # V27: PP L stack
                    bufs["d21"] = NN("d21")
                    bufs["t1"] = T1
                    if rfb > 0:
                        bufs["gyr"] = pool.get("gyr", (L - 1, n, rfb))
                        bufs["gxr"] = pool.get("gxr", (L - 1, n, rfb))
                        bufs["d21fb"] = NN("d21fb")
                D3, D21 = self._dslices(A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list,
                                        c1_list, c2_list, y_list, n, bufs,
                                        r, rfb, Zf_st, R1T_st, R2T_st,
                                        need_d21=not trim,
                                        ka=ka, Qc=Qc, kb=kb, U2=U,
                                        apb=legs["AP0"], apb4=legs4["AP0"],
                                        sb1=(fap4[2 * fa_off:2 * (fa_off + ka - kb)] if ka > kb else None),
                                        sb2=(fap24[:2 * kb] if kb > 0 else None), s_sb=s_sb)
                if regen:
                    # F68: exact transported diagonal of the regenerated core
                    # G = diag(g) + lam C_off:  dG = (W*W) g + lam (diag(C_pre)
                    # - (W*W) var).  NOT an einsum: same-object repeated operands
                    # crash flopscope (F45).
                    WW = fnp.multiply(W, W, out=NN("ww"))
                    if BETA != 0.0:
                        # V25: adaptive lambda. dG is affine in lam: t_g + lam * t_v.
                        t_g = WW @ g_prev
                        t_v = var - WW @ var_prev
                        dG0 = t_g + t_v * lam_prev                # table value first
                        rr = fnp.mean(dG0) / fnp.mean(var)
                        ref = float(REF_R[min(li - 1, len(REF_R) - 1)])
                        # clamp: an odd MLP must never turn the rule into NaN/inf (zero fallback)
                        lam_prev = lam_prev * fnp.power(fnp.clip(rr / ref, 0.5, 2.0), BETA)
                        dG = t_g + t_v * lam_prev                 # var == diag(C_pre)
                    else:
                        dG = WW @ (g_prev - var_prev * lam_prev) + var * lam_prev  # var == diag(C_pre)
                    g4row = dG * METRIC_C
                    g22c = (dG * (METRIC_C / 6.0))[:, None]
                    wk4m = _zero_diag_ip(fnp.add(g22c, g22c.T, out=NN("wk4m")))
                    wk431 = None if trim else fnp.multiply(C_off, (0.5 * METRIC_C * lam_prev), out=NN("wk431"))
                    if _os.environ.get("V17_DEBUG", "0") == "1":
                        DEBUG.append(dict(layer=li, dG=dG, g_prev=g_prev, var_prev=var_prev,
                                          lam=lam_prev, var=var, D3=D3, D21=D21))
                elif riders:
                    # vec+ww: per-neuron K4 diagonal content transported by the
                    # true (W*W) row action; one avg-metric cup replaced
                    # (/metric_c = *0.5). NOT an einsum: same-object repeated
                    # operands crash flopscope (F45).
                    gv = ((W * W) @ K4_vec) * 0.5
                    g4row = gv * float(st["wk4_c4"] * metric2)
                    g22 = gv * float(0.5 * st["wk4_c22"] * metric2)
                    g22c = (g22)[:, None]
                    wk4m = _zero_diag_ip(fnp.add(g22c, g22c.T, out=NN("wk4m")))
                else:
                    g4v = K4_sigma * float(st["wk4_c4"] * metric2)
                    g22v = K4_sigma * float(st["wk4_c22"] * metric2)
                    g4row = ones_n * g4v
                    wk4m = _zero_diag_ip(ones2 * g22v)
            else:
                D3 = D21 = g4row = wk4m = None

            # ---- wick matrix ----
            sigma = fnp.sqrt(var)
            alpha = mu / sigma
            phi = flops.stats.norm.pdf(alpha).astype(f32)
            Phi = flops.stats.norm.cdf(alpha).astype(f32)
            a2 = alpha * alpha
            a3 = a2 * alpha
            a4 = a3 * alpha
            a5 = a4 * alpha
            APOW = fnp.stack([ones_n, alpha, a2, a3, a4, a5], axis=1)
            inv = 1.0 / sigma
            i2 = inv * inv
            i3 = i2 * inv
            i4 = i3 * inv
            i5 = i4 * inv
            i6 = i5 * inv
            s2 = sigma * sigma
            s3 = s2 * sigma
            s4 = s3 * sigma
            SB = fnp.stack([i6, i5, i4, i3, i2, inv, ones_n, sigma, s2, s3, s4], axis=1)
            phic = (phi)[:, None]
            Phic = (Phi)[:, None]
            W_all = ((APOW @ C1) * phic + (APOW @ C2) * Phic) * (SB @ SELC)
            WT = W_all.T

            # ---- nonlin terms (fused) ----
            pmode = mode if (mode == 0 or (regen and not NO_WK431)) else 2
            prog = self._progs[pmode]
            fpm = FP[pmode]
            b2 = {"ones2": ones2, "c_off": C_off}
            b1 = {"ones1": ones_n}
            if mode == 1:
                if not trim:
                    d3col = fnp.multiply((D3)[:, None], ones_row, out=NN("d3col"))
                    g4col = fnp.multiply((g4row)[:, None], ones_row, out=NN("g4col"))
                    b2.update({"d21": D21, "d21T": D21.T, "wk4m": wk4m,
                               "d3col": d3col, "d3row": d3col.T,
                               "g4col": g4col, "g4row": g4col.T})
                    if regen and not NO_WK431:
                        b2["wk431"] = wk431
                b1.update({"d3": D3, "g4": g4row})
            ab_cache = {}

            def ab(pair, tbl):
                if pair in ab_cache:
                    return ab_cache[pair]
                a, b = pair
                if a in ("ones2", "ones1"):
                    val = tbl[b]
                elif b in ("ones2", "ones1"):
                    val = tbl[a]
                else:
                    val = tbl[a] * tbl[b]
                ab_cache[pair] = val
                return val

            if not trim:
                ab2 = prog["AB2"]
                for t_, (a_, b_) in enumerate(ab2):
                    if a_ in ("ones2", "ones1"):
                        fnp.copyto(abbuf[t_], b2[b_])
                    elif b_ in ("ones2", "ones1"):
                        fnp.copyto(abbuf[t_], b2[a_])
                    else:
                        fnp.multiply(b2[a_], b2[b_], out=abbuf[t_])
                ABstack = abbuf[:len(ab2)]
                WL2 = fpm["SL2"] @ WT
                WR2 = fpm["SR2"] @ WT
                PK2 = fnp.einsum("tij,ti,tj,tg->gij", ABstack, WL2, WR2, fpm["IND2"])
            B1stack = fnp.stack([ab(p, b1) for p in prog["B1"]], axis=0)
            WL1 = fpm["SL1"] @ WT
            PK1 = fnp.einsum("ti,ti,tg->gi", B1stack, WL1, fpm["IND1"])

            if not trim:
                pk11 = fnp.add(PK2[0], PK2[0].T, out=NN("pk11"))
                pk11 = _zero_diag_ip(fnp.multiply(pk11, 0.5, out=pk11))
                pk21 = _zero_diag_ip(PK2[1])
                pk22 = fnp.add(PK2[2], PK2[2].T, out=NN("pk22"))
                pk22 = _zero_diag_ip(fnp.multiply(pk22, 0.5, out=pk22))
            pk1v, pk2v, pk3v, pk4v = PK1[0], PK1[1], PK1[2], PK1[3]

            # ---- online mean correction: delta = feats @ beta[l] (13 dots/neuron);
            # mu still holds the pre-nonlin mean here, pk1v == K(1,) is the pred.
            if riders and not NO_CORR:
                if mode == 1:
                    D3f = D3
                    D21n = fnp.abs(D21, out=T1) @ ones_n if D21 is not None else fnp.zeros(n, dtype=f32)
                    k4f = ones_n * K4_sigma
                else:
                    D3f = fnp.zeros(n, dtype=f32)
                    D21n = fnp.zeros(n, dtype=f32)
                    k4f = fnp.zeros(n, dtype=f32)
                feats = fnp.stack([ones_n, mu, var, sigma, alpha, fnp.abs(alpha),
                                   phi, Phi, pk1v, D3f, D21n, k4f, sigma * phi],
                                  axis=1)
                delta = feats @ beta_rows[li]
            else:
                delta = None

            if last:
                rows.append(pk1v if delta is None else pk1v + delta)
                break

            # ---- wick old blocks + dslice scalings ----
            w1 = W_all[:, self._i11]
            w2 = W_all[:, self._i21]
            w1col = (w1)[:, None]
            w1_prev = w1  # thin stacks / basis pick up this wick via WD at the next linear
            if mode == 1 and ka < A_st.shape[0]:
                # V29: dense young legs pre-scaled by w1 (rows) in place, so the next
                # transport family multiplies by the raw W (W (w1 A) == (W w1) A up to
                # rounding); the joiner reads them scaled too.
                kk = A_st.shape[0]
                fnp.multiply(legs4["AP0"][2 * ka:2 * kk], w1col[None, None],
                             out=legs4["AP0"][2 * ka:2 * kk])
            if mode == 1:
                D3_w = D3 * w1 ** 3
                D21_w = fnp.multiply(w1col * w1col, D21, out=NN("d21w"))
                fnp.multiply(D21_w, (w1)[None, :], out=D21_w)
            else:
                D3_w = None
                D21_w = None

            # ---- new source (V1.6, struct-free Y1 = a_b * D(w2)) ----
            if legs is None:
                # V26: A and P legs interleaved per slot: [:, 0] = A, [:, 1] = P
                # V28: with one-time (2 (L-1), 1, n, n) views for the Strassen batches
                pairs = {nm: pool.get_pair(nm, (L - 1, 2, n, n)) for nm in ("AP0", "AP1")}
                legs = {nm: pairs[nm][0] for nm in pairs}
                legs4 = {nm: pairs[nm][1] for nm in pairs}
            # V29: the newborn's A leg is born into its stack slot (slot = number of
            # sources present at this layer); the next layer's family transports it
            k_b = 0 if A_st is None else A_st.shape[0]
            a_b = fnp.multiply(w1col, C_off, out=legs["AP0"][k_b, 0])
            if rfb > 0 and mode == 1:
                # V18 (F69): D21 feedback thin legs. D21 ~ Qf Bf (rank rfb range finder,
                # one power iteration, sketch = a slice of the layer weight).
                w3 = W_all[:, self._i31]
                Omf = pool.get("omf", (n, rfb))  # V20: contiguous sketch
                fnp.copyto(Omf, w32[:, :rfb])
                Yf = D21 @ Omf
                Qf, _ = fnp.linalg.qr(Yf)
                Yf = D21 @ (D21.T @ Qf)
                Qf, _ = fnp.linalg.qr(Yf)
                Bf = Qf.T @ D21                       # D21 ~= Qf @ Bf
                F1_b = (w2)[:, None] * Qf            # Xt = F1 R1, R1 = 1.5 Bf
                R1T_b = Bf.T * 1.5                              # (n, rfb) = R1^T
                F2_b = w1col * Bf.T                             # Yt = F2 R2, R2 = 0.5 Qf^T d(w3)
                R2T_b = Qf * (w3 * 0.5)[:, None]     # (n, rfb) = R2^T
                Xt_b = fnp.matmul(F1_b, R1T_b.T, out=NN("xtb"))
                Yt_b = fnp.matmul(F2_b, R2T_b.T, out=NN("ytb"))
                X1_b = fnp.multiply(a_b, 3.0, out=NN("x1b"))
                fnp.add(X1_b, Xt_b, out=X1_b)
                Y1_b = fnp.multiply(a_b, (w2)[None, :], out=NN("y1b"))
                fnp.add(Y1_b, Yt_b, out=Y1_b)
                xd = fnp.diag(Xt_b)
                yd = fnp.diag(Yt_b)
                # birth dslices of B1 with P = I (general form; diag(a_b) = 0)
                # (same operation order as the V26 expression: bit-identical)
                D21_new = fnp.multiply((xd)[:, None], Y1_b.T, out=NN("d21new"))
                fnp.multiply(X1_b, Y1_b, out=T1)
                fnp.add(D21_new, T1, out=D21_new)
                fnp.multiply((yd)[:, None], X1_b.T, out=T1)
                fnp.add(D21_new, T1, out=D21_new)
                fnp.multiply(D21_new, 1.0 / 3.0, out=D21_new)
                D3_new = xd * yd
            else:
                # birth dslices of B1 with P = I: diag(a_b) = 0, so only the a*Y1 term
                # survives: D21_new = a_b^2 * D(w2)-row; D3_new = 0.
                D21_new = fnp.multiply(a_b, a_b, out=NN("d21new"))
                fnp.multiply(D21_new, (w2)[None, :], out=D21_new)
                D3_new = None
                if rfb > 0:
                    F1_b = fnp.zeros((n, rfb), dtype=f32)
                    F2_b = F1_b
                    R1T_b = F1_b
                    R2T_b = F1_b
            if regen and mode == 1 and not NO_FEED:
                # K4->K3 feed (F68): X3 = diag(w1^2 dG) + lam a_b d(w1), Y3 = y 1^T
                # with y = (m/4) w2; M_t1 = u v^T, u = (m/4) w2*dG, v = w1^2.
                # Birth (P = I) dslices: D21 += (1/3)[dgw y^T + 2 y (.) X3_off]
                # + (1/3) v u^T;  D3 += dgw*y + u*v.
                w1sq = w1 * w1
                y_b = w2 * (0.25 * METRIC_C)
                dgw = w1sq * dG
                u_b = y_b * dG
                c1_b = w1 * lam_prev
                # (V26 expression, same operation order, out= into the pooled buffers)
                fnp.multiply((dgw)[:, None], (y_b)[None, :], out=T1)
                fnp.multiply(T1, 1.0 / 3.0, out=T1)
                fnp.add(D21_new, T1, out=D21_new)
                fnp.multiply(a_b, (c1_b)[None, :], out=T1)
                fnp.multiply((y_b * (2.0 / 3.0))[:, None], T1, out=T1)
                fnp.add(D21_new, T1, out=D21_new)
                fnp.multiply((w1sq)[:, None], (u_b * (1.0 / 3.0))[None, :], out=T1)
                fnp.add(D21_new, T1, out=D21_new)
                D3_new = ((dgw * y_b + u_b * w1sq) if D3_new is None
                          else D3_new + dgw * y_b + u_b * w1sq)
            else:
                y_b = fnp.zeros(n, dtype=f32)
                u_b = y_b
                c1_b = y_b
                dgw = y_b
                w1sq = y_b
            rep21 = D21_new if D21_w is None else fnp.add(D21_w, D21_new, out=NN("rep21"))
            rep21 = _zero_diag_ip(rep21)

            # ---- pK -> K (per-entry; d=1 outputs are vector ops, d=2 few entries) ----
            pk = {(1,): pk1v, (2,): pk2v, (3,): pk3v, (4,): pk4v,
                  (1, 1): pk11, (2, 1): pk21, (2, 2): pk22}

            def pk_slice(vec):
                nz = tuple(sorted((x for x in vec if x > 0), reverse=True))
                base = pk[nz]
                if len(vec) == 1:
                    return base
                if len(nz) == 2:
                    return base if vec == tuple(sorted(vec, reverse=True)) else base.T
                return ((base)[:, None] if vec[0] > 0
                        else (base)[None, :])

            K = {}
            for out_part, entries in PK2K_TABLE.items():
                if len(out_part) == 2:
                    continue    # V27: matrix parts assembled below with out= (same order)
                acc2 = None
                for vpart, coef in entries:
                    prod = None
                    for blk in vpart:
                        fct = pk_slice(blk)
                        prod = fct if prod is None else prod * fct
                    contrib = prod * coef
                    acc2 = contrib if acc2 is None else acc2 + contrib
                K[out_part] = acc2
            # V27 (F80): the three matrix parts of PK2K_TABLE, unrolled in the generic
            # loop's operation order (bit-identical), written into pooled buffers:
            #   (1,1) = pk11 * 1.0 = pk11
            #   (2,1) = (pk1 (.) pk11) * (-2) + pk21
            #   (2,2) = ((pk1^T (.) pk1) (.) pk11) * 4 + (pk1^T (.) pk21) * (-2)
            #           + (pk1 (.) pk21^T) * (-2) + (pk11 * pk11) * (-2) + pk22
            pk1c = (pk1v)[:, None]
            pk1r = (pk1v)[None, :]
            K11 = pk11
            K21 = fnp.multiply(pk1c, pk11, out=NN("k21"))
            fnp.multiply(K21, -2.0, out=K21)
            fnp.add(K21, pk21, out=K21)
            K22 = fnp.multiply(pk1r, pk1c, out=NN("k22"))
            fnp.multiply(K22, pk11, out=K22)
            fnp.multiply(K22, 4.0, out=K22)
            fnp.multiply(pk1r, pk21, out=T1)
            fnp.multiply(T1, -2.0, out=T1)
            fnp.add(K22, T1, out=K22)
            fnp.multiply(pk1c, pk21.T, out=T1)
            fnp.multiply(T1, -2.0, out=T1)
            fnp.add(K22, T1, out=K22)
            fnp.multiply(pk11, pk11, out=T1)
            fnp.multiply(T1, -2.0, out=T1)
            fnp.add(K22, T1, out=K22)
            fnp.add(K22, pk22, out=K22)
            K21 = _zero_diag_ip(K21)
            K22 = _zero_diag_ip(K22)
            K1v = K[(1,)]
            K2v, K3v, K4v = K[(2,)], K[(3,)], K[(4,)]

            # ---- assemble ----
            mu = K1v if delta is None else K1v + delta
            # V27: diagflat + add -> pooled buffer, diagonal written in place (bit-identical:
            # K11 has a zero diagonal)
            C = fnp.add(K11, K11.T, out=NN("c"))
            fnp.multiply(C, 0.5, out=C)
            fnp.fill_diagonal(C, K2v)
            C = flops.as_symmetric(C, symmetry=(0, 1))
            # V29: C rides the next layer's transport family in the P position of the
            # newborn's slot (the family writes W C there; the newborn's P leg W is
            # copied over it once C_pre is formed)
            fnp.copyto(legs["AP0"][k_b, 1], C)
            S3c = K3v - D3_w if D3_w is not None else K3v
            if D3_new is not None:
                S3c = S3c - D3_new
            S21 = _zero_diag_ip(fnp.subtract(K21, rep21, out=NN("s21")))
            # S21 = S_sep + Rres: S_sep = exact leading (2,1) Wick term (the (c_off,
            # ones2) term of pk21 minus 2*pk1*[(c_off, ones2) term of pk11]); M_b^T
            # part 3*S_sep^T = 3*a_b*diag(e) rides on the A leg for free. Rres is
            # compressed to rank r (randomized range finder, one power iteration).
            e_b = W_all[:, self._i12] - 2.0 * pk1v * w1
            S_sep = fnp.multiply((e_b)[:, None], C_off, out=NN("ssep"))
            S_sep = _zero_diag_ip(fnp.multiply(S_sep, (w1)[None, :], out=S_sep))
            Rres = fnp.subtract(S21, S_sep, out=S21)
            Om = pool.get("om_r", (n, r))  # V20: contiguous sketch
            fnp.copyto(Om, w32[:, :r])
            Y = fnp.matmul(Rres, Om, out=pool.get("y", (n, r)))
            Q, _ = fnp.linalg.qr(Y)
            Y2 = fnp.matmul(Rres.T, Q, out=pool.get("y2", (n, r)))
            Y = fnp.matmul(Rres, Y2, out=Y)
            Q, _ = fnp.linalg.qr(Y)
            Bm = fnp.matmul(Q.T, Rres, out=pool.get("bm", (r, n)))   # Rres ~= Q @ Bm
            # M_b = diag(S3c) + 3 S21^T = diag(S3c) + 3 a_b diag(e_b) + 3 Rr Lr^T
            # with Rr = Bm^T (n,r) [transported as Z = P Rr] and Lr = Q (static).
            # extra thin columns: M_t1 = u v^T -> Z column P u, L column v; and the Y3
            # row vector y is a LEFT factor of the hub tensor, so it is transported
            # like a leg (y(l) = P(l) y): Z column P y with a ZERO L column (keeps it
            # out of the M leg). Zeros off-suite / at layer 0 so every source keeps
            # r+2 columns.
            # V27: Rr into a persistent (n, r+2) buffer (read by next layer's W @ Rr);
            # Lr written straight into its stack slot k (static, never transported).
            Rr_full = pool.get("rr", (n, r + 2))
            fnp.multiply(Bm.T, 3.0, out=Rr_full[:, :r])
            fnp.copyto(Rr_full[:, r], u_b)
            fnp.copyto(Rr_full[:, r + 1], y_b)
            # slot of the source born here = number of sources present at this layer
            # (the newborn added above included): it joins the stacks at the next layer
            k_b = 0 if A_st is None else A_st.shape[0]
            lb = pool.get("l", (L - 1, n, r + 2))
            fnp.copyto(lb[k_b, :, :r], Q)
            fnp.copyto(lb[k_b, :, r], w1sq)
            fnp.copyto(lb[k_b, :, r + 1], zeros_n)
            Lr_full = None
            # V18: thin feedback legs [F1 | F2] travel in their own stack Zf (transported
            # like Z, never entering the M-leg einsums); right factors R1T/R2T are static.
            Ff_b = fnp.concatenate([F1_b, F2_b], axis=1) if rfb > 0 else None
            newborn = (a_b, Rr_full, Lr_full, S3c, e_b, Ff_b)
            if rfb > 0:
                r1b = pool.get("r1t", (L - 1, n, rfb))
                r2b = pool.get("r2t", (L - 1, n, rfb))
                fnp.copyto(r1b[k_b], R1T_b)
                fnp.copyto(r2b[k_b], R2T_b)
                R1T_st = r1b[:k_b + 1]
                R2T_st = r2b[:k_b + 1]
            w2b_list.append(w2)
            # V21: hub-column Gram weights of this source's legs (X1 = 3A, Y1 ~ A d(w2),
            # M ~ P d(s) + 3 A d(e)): A-type 9 + w2^2 + 9 e^2, P-type 1 + s^2
            dA_list.append(9.0 + w2 * w2 + 9.0 * e_b * e_b)
            dP_list.append(1.0 + S3c * S3c)
            c1_list.append(c1_b)
            c2_list.append(dgw)
            y_list.append(y_b)
            if regen:
                # post-ReLU kappa4 diagonal core (r=1 matrix-core harmonic projection)
                k22row = K22 @ ones_n
                g_prev = ((K4v + k22row) * float(st["cA"])
                          + (fnp.sum(K4v) + fnp.sum(k22row)) * float(st["cI"]))
                var_prev = K2v
                lam_prev = float(LAM[min(li, len(LAM) - 1)])
            K4_sigma = (fnp.sum(K4v) * float(st["k4_c4"])
                        + fnp.sum(K22) * float(st["k4_c22"])) * float(st["P2"])
            # per-neuron K4 diagonal content (mean equals K4_sigma by construction;
            # K22 is symmetric so K22 @ ones == its column sums)
            if riders:
                K4_vec = (K4v * float(st["k4_c4"])
                          + (K22 @ ones_n) * float(st["k4_c22"])) * float(n * st["P2"])
            rows.append(mu)

        return fnp.stack(rows, axis=0)

    # ------------------------------------------------------------------
    def _sym_product(self, X, Y, n, out, lev):
        """V29: out = X Y for a SYMMETRIC result (X = W C, Y = w32 = W^T): the three block
        products (11, 12, 22) of the 2x2 partition as ONE Strassen family (0.75 of the
        dense count before the recursion); the 21 block is the copy of 12^T. Odd n (smoke
        shapes) falls back to the plain product."""
        h = n // 2
        if n % 2 or n < 4:
            return fnp.matmul(X, Y, out=out)
        pool = self._pool
        X3 = pool.get("sym_x", (3, 1, h, n))
        Y3 = pool.get("sym_y", (3, 1, n, h))
        O3 = pool.get("sym_o", (3, 1, h, h))
        fnp.copyto(X3[0, 0], X[:h]); fnp.copyto(X3[1, 0], X[:h]); fnp.copyto(X3[2, 0], X[h:])
        fnp.copyto(Y3[0, 0], Y[:, :h]); fnp.copyto(Y3[1, 0], Y[:, h:]); fnp.copyto(Y3[2, 0], Y[:, h:])
        self._smm.mm(X3, Y3, O3, self._smm.level(h, n, h, lev))
        fnp.copyto(out[:h, :h], O3[0, 0]); fnp.copyto(out[:h, h:], O3[1, 0])
        fnp.copyto(out[h:, h:], O3[2, 0]); fnp.copyto(out[h:, :h], O3[1, 0].T)
        return out

    def _hub2(self, bufs, apb4, k0, k, n):
        """V26: sum over slots k0..k-1 of LA_k A_k^T + LP_k P_k^T as ONE Strassen family
        (LA/LP live interleaved in bufs["lap"], A/P in the legs, so the pair axis folds
        into the contraction batch; V28: on the one-time 4-D views)."""
        out = bufs["hub"]
        self._smm.hub(bufs["lap4"][2 * k0:2 * k], apb4[2 * k0:2 * k], out[None],
                      self._smm.level(n, n, n, min(STRASSEN_HUB, self._s_hub)))
        return out

    # ------------------------------------------------------------------
    def _dslices(self, A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list,
                 c1_list, c2_list, y_list, n, bufs, rres, rfb, Zf_st, R1T_st, R2T_st,
                 need_d21=True, ka=0, FAo=None, FPo=None, Qc=None,
                 kb=0, FA2=None, FP2=None, U2=None, apb=None, apb4=None,
                 sb1=None, sb2=None, s_sb=0):
        """(3,) and (2,1) dslices with the M leg expressed through A, P and the thin
        residual leg: M = P*diag(s) + 3*A*diag(e) + Z L^T (the factor 3 of the
        residual is folded into Z at birth). Two dense contractions (right factors
        A and P) + O(n^2 r) thin terms.
          D21 = 2H + T2 + T3/3 + 2T4/3,  H=(A*P*w2)A^T, T2=(A*A*w2)P^T,
                T3=(P*P)M^T, T4=(M*P)P^T
          D3  = 3 sum_j w2_j A_ij^2 P_ij + sum_j M_ij P_ij^2
        Every (k,n,n) intermediate is written with out= into pooled buffers
        (F53: fresh result buffers dominate residual time).
        need_d21=False (V19, final layer): D3 only -- no left factors, no dense
        contractions, no thin right-factor terms; returns D21 = None.
        """
        k = len(w2b_list)
        W2B = fnp.stack(w2b_list, axis=0)[:, None, :]
        Sb = fnp.stack(s_list, axis=0)[:, None, :]
        Eb = fnp.stack(e_list, axis=0)[:, None, :]
        AP = fnp.multiply(A_st, P_st, out=bufs["ap"][:k])
        PP = fnp.multiply(P_st, P_st, out=bufs["pp"][:k])
        T = bufs["t"][:k]
        # M*P = PP*s + 3 AP*e + (Z L^T)*P
        MP = fnp.matmul(Z_st, fnp.swapaxes(L_st, -1, -2), out=bufs["mp"][:k])   # V27: BLAS path
        fnp.multiply(MP, P_st, out=MP)
        fnp.multiply(PP, Sb, out=T)
        fnp.add(MP, T, out=MP)
        fnp.multiply(AP, Eb * 3.0, out=T)
        fnp.add(MP, T, out=MP)
        if need_d21:
            # leftA = 2 AP*w2 + PP*e   (right factor A)
            LA = fnp.multiply(AP, W2B * 2.0, out=bufs["lap"][:k, 0])
            fnp.multiply(PP, Eb, out=T)
            fnp.add(LA, T, out=LA)
            # leftP = A*A*w2 + PP*(s/3) + (2/3) M*P   (right factor P)
            LP = fnp.multiply(A_st, A_st, out=bufs["lap"][:k, 1])
            fnp.multiply(LP, W2B, out=LP)
            fnp.multiply(PP, Sb * (1.0 / 3.0), out=T)
            fnp.add(LP, T, out=LP)
            fnp.multiply(MP, 2.0 / 3.0, out=T)
            fnp.add(LP, T, out=LP)
        fnp.multiply(P_st, W2B, out=T)
        D3 = (fnp.einsum("kij,kij,kij->i", A_st, A_st, T) * 3.0
              + fnp.einsum("kij,kij->i", MP, P_st))
        if rfb > 0:
            # V18 (F69): B1 pair with X1 = 3A + Xt, Y1 = A*w2 + Yt; Xt = F1 R1, Yt = F2 R2
            # (F1/F2 = transported thin Z columns, R1T/R2T static (k, n, rfb)).
            F1 = Zf_st[:, :, :rfb]
            F2 = Zf_st[:, :, rfb:]
            Xt = fnp.einsum("kiq,kjq->kij", F1, R1T_st, out=bufs["xt"][:k])
            Yt = fnp.einsum("kiq,kjq->kij", F2, R2T_st, out=bufs["yt"][:k])
            U = bufs["u"][:k]
            # LPadd = A*Yt + (1/3) Xt*A*w2 + (1/3) Xt*Yt  (in T);  D3 += 3 rowsum(P*LPadd)
            fnp.multiply(A_st, Yt, out=T)
            fnp.multiply(Xt, A_st, out=U)
            fnp.multiply(U, W2B * (1.0 / 3.0), out=U)
            fnp.add(T, U, out=T)
            fnp.multiply(Xt, Yt, out=U)
            fnp.multiply(U, 1.0 / 3.0, out=U)
            fnp.add(T, U, out=T)
            D3 = D3 + fnp.einsum("kij,kij->i", P_st, T) * 3.0
            if need_d21:
                fnp.add(LP, T, out=LP)
                # LA += (1/3) Xt*P*w2 + P*Yt
                fnp.multiply(Xt, P_st, out=T)
                fnp.multiply(T, W2B * (1.0 / 3.0), out=T)
                fnp.add(LA, T, out=LA)
                fnp.multiply(P_st, Yt, out=U)
                fnp.add(LA, U, out=LA)
                # thin right factors:  GY = AP + (1/3) Xt*P      -> D21 += (GY R2^T) F2^T
                #                      GX = (1/3)(AP*w2 + P*Yt)  -> D21 += (GX R1^T) F1^T
                fnp.multiply(Xt, P_st, out=T)
                fnp.multiply(T, 1.0 / 3.0, out=T)
                fnp.add(T, AP, out=T)                 # GY
                # V27: thin contractions as batched matmul into the free (k,n,n) scratch
                # (Xt is consumed above) + a k-sum written out= (no fresh (n,n) results)
                GYR = fnp.matmul(T, R2T_st, out=bufs["gyr"][:k])
                fnp.matmul(GYR, fnp.swapaxes(F2, -1, -2), out=Xt)
                D21_fb = fnp.sum(Xt, axis=0, out=bufs["d21fb"])
                fnp.multiply(AP, W2B, out=T)
                fnp.add(T, U, out=T)                  # AP*w2 + P*Yt   (U holds P*Yt)
                fnp.multiply(T, 1.0 / 3.0, out=T)     # GX
                GXR = fnp.matmul(T, R1T_st, out=bufs["gxr"][:k])
                fnp.matmul(GXR, fnp.swapaxes(F1, -1, -2), out=Xt)
                fnp.sum(Xt, axis=0, out=bufs["t1"])
                fnp.add(D21_fb, bufs["t1"], out=D21_fb)
            else:
                D21_fb = None
        else:
            D21_fb = None
        # F68 feed pair (X3 = A*c1 + P*c2 column scalings, Y3 = y 1^T):
        #   D21 += (1/3) R y^T + (2/3) (y (.) X3) P^T,  D3 += y * R,
        #   R = rowsum(X3*P) = (A*P) c1 + (P*P) c2; y is the TRANSPORTED Y3 row vector
        #   (last Z column).   MP/T are free scratch now.
        C1 = fnp.stack(c1_list, axis=0)                 # (k, n) static column scalings
        C2 = fnp.stack(c2_list, axis=0)
        Yk = Z_st[:, :, rres + 1]                        # (k, n) transported y = P(l) y
        R = fnp.einsum("kij,kj->ki", AP, C1) + fnp.einsum("kij,kj->ki", PP, C2)
        D3 = D3 + fnp.einsum("ki,ki->i", R, Yk)
        if not need_d21:
            return D3, None
        fnp.multiply(P_st, C2[:, None, :], out=MP)
        fnp.multiply(A_st, C1[:, None, :], out=T)
        fnp.add(T, MP, out=T)
        fnp.multiply(T, (Yk * (2.0 / 3.0))[:, :, None], out=T)
        fnp.add(LP, T, out=LP)
        if ka > 0:
            # V21: old sources through the shared basis: [sum LA FAo^T + LP FPo^T] Qc^T
            # V24: tier-2 sources through U: [sum LA FA2^T + LP FP2^T] U2^T lifted first (U2 = sub-basis; U is the scratch buffer)
            # V28: each tier's contraction sum_k LA_k FA_k^T + LP_k FP_k^T is ONE hub
            # family over the interleaved slabs (rectangular right operand (2m, 1, r, n));
            # level 0 = the dense batched-matmul leaf + k-sum (same billing as the einsum)
            inner = None
            smm = self._smm
            if ka > kb:
                r1 = sb1.shape[2]
                inner = self._pool.get("inner", (1, n, r1))
                smm.hub(bufs["lap4"][2 * kb:2 * ka], sb1, inner, smm.level(n, n, r1, s_sb))
                inner = inner[0]
            if kb > 0:
                r2_ = sb2.shape[2]
                inner2 = self._pool.get("inner2", (1, n, r2_))
                smm.hub(bufs["lap4"][:2 * kb], sb2, inner2, smm.level(n, n, r2_, s_sb))
                lift = fnp.matmul(inner2[0], U2.T, out=self._pool.get("lift", (n, U2.shape[0])))
                inner = lift if inner is None else fnp.add(inner, lift, out=inner)
            # V27: D21 accumulates in one pooled buffer (a + b == b + a exactly, so the
            # hub-first order is bit-identical to V26's D21 + hub)
            if ka < k and STRASSEN_HUB > 0:
                D21 = self._hub2(bufs, apb4, ka, k, n)
                fnp.add(D21, fnp.matmul(inner, Qc.T, out=bufs["t1"]), out=D21)
            else:
                D21 = fnp.matmul(inner, Qc.T, out=bufs["d21"])
                if ka < k:
                    D21 = (D21 + fnp.einsum("kij,kcj->ic", LA[ka:], A_st[ka:])
                           + fnp.einsum("kij,kcj->ic", LP[ka:], P_st[ka:]))
            fnp.matmul(R.T, Yk, out=bufs["t1"])
            fnp.multiply(bufs["t1"], 1.0 / 3.0, out=bufs["t1"])
            fnp.add(D21, bufs["t1"], out=D21)
        else:
            if STRASSEN_HUB > 0:
                D21 = self._hub2(bufs, apb4, 0, k, n)
                fnp.matmul(R.T, Yk, out=bufs["t1"])
                fnp.multiply(bufs["t1"], 1.0 / 3.0, out=bufs["t1"])
                fnp.add(D21, bufs["t1"], out=D21)
            else:
                D21 = (fnp.einsum("kij,kcj->ic", LA, A_st)
                       + fnp.einsum("kij,kcj->ic", LP, P_st)
                       + fnp.einsum("ki,kc->ic", R, Yk) * (1.0 / 3.0))
        PPL = fnp.matmul(PP, L_st, out=bufs["ppl"][:k])   # V27: BLAS path (1.2 s -> 8 ms)
        t1 = fnp.einsum("kiq,kcq->ic", PPL, Z_st)
        fnp.multiply(t1, 1.0 / 3.0, out=t1)
        D21 = fnp.add(D21, t1, out=bufs["d21"])   # in place, or hub/dense result -> d21
        if D21_fb is not None:
            fnp.add(D21, D21_fb, out=D21)
        return D3, _zero_diag_ip(D21)




# ===========================================================================
# Estimator: kprop3 primary, validated blend as fallback
# ===========================================================================
class Estimator(BaseEstimator):
    """Factorized k_max=3 cumulant propagation; covprop/matched-MC fallback."""

    def __init__(self) -> None:
        self._setup_rng = None
        self._sw = None       # rung-5 Strassen pool (persists across predicts)
        self._kp3_pool = {}   # rung-5 leg/scratch slot buffers (F53)
        self._v25 = _V25Core() if _V25_STACK else None

    def _get_sw(self, depth: int):
        if not _STRASSEN:
            return None
        if self._sw is None:
            self._sw = _Strassen(_f32(), 16)  # buffers grow on demand
        return self._sw

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)
        warnings.filterwarnings("ignore", category=flops.SymmetryLossWarning)
        warnings.filterwarnings("ignore", category=flops.FlopscopeWarning)
        if self._v25 is not None:
            self._v25.setup(ctx)   # V29 warm-ups (op-signature first calls)
        # warm pure-python caches (polynomials, conversion plans) and the
        # scipy machinery behind flops.stats so first predict stays under
        # the residual wall-time cap
        for p in (1, 2, 3, 4):
            for k in range(0, p):
                _relu_wick_poly(p, k)
        for part in ((1,), (2,), (3,), (4,), (1, 1), (2, 1), (2, 2)):
            _pk_to_k_plan(part)
        try:
            with flops.BudgetContext(flop_budget=2**32, quiet=True):
                z = fnp.zeros(1, dtype=fnp.float32)
                flops.stats.norm.pdf(z)
                flops.stats.norm.cdf(z)
        except Exception:
            pass
        # full warm predict at a reduced width: initializes BLAS thread
        # pools, batched-matmul dispatch, and every code path, so the first
        # billed predict does not pay the cold start against the 0.4 s
        # residual wall-time cap (a breach zeroes that MLP's predictions)
        try:
            n0, L0 = 512, 16
            with flops.BudgetContext(flop_budget=2**42, quiet=True):
                rng = fnp.random.default_rng(0)
                ws = [rng.standard_normal((n0, n0), dtype=fnp.float32)
                      * (1.0 / math.sqrt(n0))] + [
                    rng.standard_normal((n0, n0), dtype=fnp.float32)
                    * math.sqrt(2.0 / n0) for _ in range(L0 - 1)]
                _kprop3_layer_means(ws, n0)
        except Exception:
            pass

    # -- routing -----------------------------------------------------------
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        n = mlp.width
        depth = mlp.depth
        if (depth / n <= _KPROP_MAX_COUPLING
                and _kprop3_cost(n, depth) <= _KPROP_MAX_COST_X_B * budget):
            if self._v25 is not None:
                try:  # rung 6: the attributed V29 accuracy stack (gc inside)
                    return self._v25.predict(mlp, budget)
                except Exception:  # pragma: no cover -- numerical fallback
                    pass
            was = _gc.isenabled()
            _gc.disable()   # their F53/V26: retained op logs push gen-2 GC
            try:            # collections inside predict (35-110 ms)
                rows = _kprop3_layer_means(mlp.weights, n,
                                           sw=self._get_sw(depth),
                                           pool=self._kp3_pool)
                return fnp.stack(rows, axis=0)
            except Exception:  # pragma: no cover -- numerical fallback
                pass
            finally:
                if was:
                    _gc.enable()
        return self._predict_blend(mlp, budget)

    # -- fallback: Phase 2 blend (unchanged, validated OOS) ----------------
    @staticmethod
    def _cov_cost(n: int, depth: int) -> float:
        return depth * (3.3 * n * n * n + 14.0 * n * n) + 4.0 * n * n * n

    @staticmethod
    def _mc_count(n: int, depth: int, budget: int, cov_cost: float) -> int:
        matmul = n * (2 * n - 1)
        per = depth * (matmul + 2 * n) - matmul // 2
        per += 2 * matmul + 4 * n
        per += 8 * n + 2 * n
        avail = _TARGET_UTILISATION * budget - cov_cost - _RESIDUAL_RESERVE
        avail -= 0.7 * n * n * n
        k = int(avail / per)
        k = max(0, min(k, 200_000))
        return k - (k & 1)

    @staticmethod
    def _covprop(weights, n: int, depth: int):
        mu = fnp.zeros(n, dtype=fnp.float32)
        cov = flops.as_symmetric(fnp.eye(n, dtype=fnp.float32), symmetry=(0, 1))
        rows = []
        for w in weights:
            w = fnp.asarray(w, dtype=fnp.float32)
            m = w.T @ mu
            c = fnp.einsum("ij,ia,jb->ab", cov, w, w)
            v = fnp.maximum(fnp.diag(c), 1e-12)
            s = fnp.sqrt(v)
            a = m / s
            phi = fnp.asarray(flops.stats.norm.pdf(a), dtype=fnp.float32)
            Phi = fnp.asarray(flops.stats.norm.cdf(a), dtype=fnp.float32)
            mean = m * Phi + s * phi
            ez2 = (m * m + v) * Phi + m * s * phi
            var = fnp.maximum(ez2 - mean * mean, 1e-12)
            b1 = s * Phi
            b2 = s * phi
            b3 = b2 * (-a)
            inv_s = 1.0 / s
            rho = c * fnp.outer(inv_s, inv_s)
            rho = fnp.maximum(fnp.minimum(rho, 1.0), -1.0)
            cov = (fnp.outer(b1, b1) * rho
                   + fnp.outer(b2, b2) * (rho * rho) * 0.5
                   + fnp.outer(b3, b3) * (rho * rho * rho) * (1.0 / 6.0))
            cov = cov - fnp.diag(fnp.diag(cov)) + fnp.diag(var)
            cov = flops.as_symmetric(cov, symmetry=(0, 1))
            mu = mean
            rows.append(mu)
        return rows

    @staticmethod
    def _layer1_moments(w1):
        s_cov = w1.T @ w1
        var = fnp.maximum(fnp.diag(s_cov), 1e-20)
        sd = fnp.sqrt(var)
        mean = sd * _INV_SQRT_2PI
        inv_sd = 1.0 / sd
        rho = s_cov * fnp.outer(inv_sd, inv_sd)
        rho = fnp.maximum(fnp.minimum(rho, 1.0), -1.0)
        sin_t = fnp.sqrt(fnp.maximum(1.0 - rho * rho, 0.0))
        kernel = sin_t + rho * (_PI - fnp.arccos(rho))
        second = fnp.outer(sd, sd) * kernel * (1.0 / (2.0 * _PI))
        return mean, second - fnp.outer(mean, mean)

    def _matched_mc(self, weights, n: int, depth: int, k: int, rng):
        w1 = fnp.asarray(weights[0], dtype=fnp.float32)
        x0 = rng.standard_normal((k // 2, n), dtype=fnp.float32)
        z0 = x0 @ w1
        z = fnp.concatenate([z0, -z0], axis=0)
        y = fnp.maximum(z, 0.0)

        mean1, cov1 = self._layer1_moments(w1)
        emp_mean = fnp.mean(y, axis=0)
        yc = y - emp_mean
        emp_cov = (yc.T @ yc) * (1.0 / k)
        tr = float(fnp.mean(fnp.diag(emp_cov)))
        eye = fnp.eye(n, dtype=fnp.float32) * (_RIDGE_FRAC * tr)
        l_emp = fnp.linalg.cholesky(emp_cov + eye)
        l_tgt = fnp.linalg.cholesky(cov1 + eye)
        a = fnp.linalg.solve(l_emp.T, l_tgt.T)
        y = yc @ a + mean1

        rows = [fnp.mean(y, axis=0)]
        for w in weights[1:]:
            w = fnp.asarray(w, dtype=fnp.float32)
            y = fnp.maximum(y @ w, 0.0)
            rows.append(fnp.mean(y, axis=0))
        return rows

    def _predict_blend(self, mlp: MLP, budget: int) -> fnp.ndarray:
        rng = fnp.random.default_rng(mlp.seed)
        n = mlp.width
        depth = mlp.depth
        weights = mlp.weights

        cov_cost = self._cov_cost(n, depth)
        if cov_cost > 0.5 * _TARGET_UTILISATION * budget:
            matmul = n * (2 * n - 1)
            per = depth * (matmul + 2 * n) + 16 * n
            k = max(2, min(int(_TARGET_UTILISATION * budget / per), 200_000))
            k -= k & 1
            x = rng.standard_normal((k, n), dtype=fnp.float32)
            rows = []
            y = x
            for w in weights:
                y = fnp.maximum(y @ fnp.asarray(w, dtype=fnp.float32), 0.0)
                rows.append(fnp.mean(y, axis=0))
            return fnp.stack(rows, axis=0)

        rows_a = self._covprop(weights, n, depth)

        k = self._mc_count(n, depth, budget, cov_cost)
        if k < max(_MIN_MC, _MIN_MATCH_RATIO * n):
            return fnp.stack(rows_a, axis=0)

        try:
            rows_b = self._matched_mc(weights, n, depth, k, rng)
        except Exception:  # pragma: no cover -- numerical fallback
            return fnp.stack(rows_a, axis=0)

        rows = [_W_COV * ra + (1.0 - _W_COV) * rb
                for ra, rb in zip(rows_a, rows_b)]
        return fnp.stack(rows, axis=0)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whest-starterkit"))
    from local_engine import build_mlp

    B = 2**41
    mlp = build_mlp(width=1024, depth=16, seed=0)
    est = Estimator()
    est.setup(type("C", (), {"seed": 0})())
    with flops.BudgetContext(flop_budget=10 * B, quiet=True) as ctx:
        pred = est.predict(mlp, B)
    print(f"kprop3 cost model = {_kprop3_cost(1024, 16):.4e} "
          f"({_kprop3_cost(1024, 16) / B:.3f} B)")
    print(f"flops_used = {ctx.flops_used:.4e}  (C/B = {ctx.flops_used / B:.4f})")
    print(f"residual_wall_time_s = {ctx.residual_wall_time_s:.4f} "
          f"({'PASS' if ctx.residual_wall_time_s <= 0.4 else 'FAIL'} vs 0.4s)")
    print("pred shape", pred.shape)
