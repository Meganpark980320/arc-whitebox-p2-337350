"""cov_participation_rank_v2.py [out.txt]  (v9 write-up, section 3.6, label [simulation])

Question: does the effective rank of the activation covariance fall with depth in this network family?
(ARC forum t/18078: "As network depth increases, the effective rank of the activation covariance matrix
decreases (eigenvalues become more skewed).")

Network family as in the challenge, but small width: bias-free ReLU MLP, 16 layers, weights iid N(0, 2/n),
input x ~ N(0, I_n). Row-vector convention of the one-step scripts: z_0 = x W_0, z_l = relu(z_{l-1}) W_l,
l = 0..15 (0 = first hidden layer). For every layer we estimate the covariance over inputs of the
pre-activation z_l and of the post-activation h_l = relu(z_l) and report its participation ratio
    PR = (tr C)^2 / tr(C^2),   as a fraction of n.
tr(C^2) is estimated without sampling bias from two independent halves of the samples, tr(C_A C_B);
tr C is the mean of the two half traces. N = 32 n inputs per network (16 n per half); v2: 8 networks per width
and a sampling check at n = 256 with N = 128 n (env PR_CHECK=1).

Light compute: one process, BLAS limited to one thread, n <= 512, < 0.3 GB, about a minute in total.
Deterministic seeds (network seed s, input seed 10_000 + s)."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
import time
import numpy as np

WIDTHS = (128, 256, 512)
SEEDS = tuple(range(8))
DEPTH = 16
SHOW = (0, 1, 2, 4, 6, 8, 10, 12, 14, 15)


def half_cov(Y):
    """Unbiased sample covariance (float64) of the rows of Y."""
    Y = Y.astype(np.float64)
    Y = Y - Y.mean(0, keepdims=True)
    return (Y.T @ Y) / (Y.shape[0] - 1)


def pr_frac(Ya, Yb, n):
    Ca, Cb = half_cov(Ya), half_cov(Yb)
    tr = 0.5 * (np.trace(Ca) + np.trace(Cb))
    tr2 = float(np.sum(Ca * Cb))          # = tr(Ca Cb) for symmetric matrices, unbiased for tr(C^2)
    return tr * tr / tr2 / n


def one_net(n, seed, mult=32):
    gw = np.random.default_rng(seed)
    W = [(gw.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(DEPTH)]
    gx = np.random.default_rng(10_000 + seed)
    N = mult * n
    x = gx.standard_normal((N, n)).astype(np.float32)
    h = x
    pre, post = [], []
    for l in range(DEPTH):
        z = h @ W[l]
        h = np.maximum(z, 0.0)
        pre.append(pr_frac(z[: N // 2], z[N // 2:], n))
        post.append(pr_frac(h[: N // 2], h[N // 2:], n))
    return np.array(pre), np.array(post)


def main():
    out = open(sys.argv[1], 'w') if len(sys.argv) > 1 else None

    def say(s):
        print(s, flush=True)
        if out:
            out.write(s + '\n')

    t0 = time.time()
    say('participation ratio PR/n of the covariance over inputs, per layer l (0 = first hidden layer)')
    say('family: bias-free ReLU MLP, 16 layers, W ~ N(0, 2/n), x ~ N(0, I); N = 32n inputs, half-split tr(C^2)')
    say('networks per width: %d (seeds %s); values are means over networks, (min-max) in brackets' % (len(SEEDS), SEEDS))
    res = {}
    for n in WIDTHS:
        P, Q = [], []
        for s in SEEDS:
            p, q = one_net(n, s)
            P.append(p); Q.append(q)
        res[n] = (np.array(P), np.array(Q))
        say('  n=%d done (%.0f s)' % (n, time.time() - t0))
    say('')
    head = 'layer l   ' + '  '.join('%5d' % l for l in SHOW)
    for name, k in (('pre-activation z_l', 0), ('post-activation h_l', 1)):
        say('== ' + name + ': PR/n')
        say(head)
        for n in WIDTHS:
            A = res[n][k]
            say(('n=%-4d    ' % n) + '  '.join('%5.3f' % A[:, l].mean() for l in SHOW))
        say(('1/(l+2)   ') + '  '.join('%5.3f' % (1.0 / (l + 2)) for l in SHOW))
        say(('1/(l+1)   ') + '  '.join('%5.3f' % (1.0 / (l + 1)) for l in SHOW))
        for n in WIDTHS:
            A = res[n][k]
            say(('n=%-4d rng ' % n) + ' '.join('[%.3f-%.3f]' % (A[:, l].min(), A[:, l].max()) for l in (6, 10, 15)) + '  (layers 6, 10, 15)')
        say('')
    if os.environ.get('PR_CHECK') == '1':
        say('== sampling check, n=256, seeds 0-3: PR/n at layers 6, 10, 15 with N = 32n vs N = 128n (pre-activation)')
        for s in range(4):
            a, _ = one_net(256, s, 32)
            b, _ = one_net(256, s, 128)
            say('  seed %d  N=32n %s   N=128n %s' % (s, ' '.join('%.3f' % a[l] for l in (6, 10, 15)), ' '.join('%.3f' % b[l] for l in (6, 10, 15))))
        say('')
    say('elapsed %.0f s' % (time.time() - t0))
    if out:
        out.close()


if __name__ == '__main__':
    main()
