"""Shared utilities: alignment, statistics, balanced splits, thermometer coding."""
import numpy as np, json, os

def align(T, ref, maxl=20):
    """Integer-lag alignment of each trace to ref by FFT cross-correlation. Output length D-2*maxl."""
    T = np.asarray(T, np.float32); D = T.shape[1]
    n = 1 << int(np.ceil(np.log2(2 * D)))
    Rf = np.conj(np.fft.rfft(ref - ref.mean(), n))
    out = np.empty((len(T), D - 2 * maxl), np.float32); lags = np.empty(len(T), int)
    for s in range(0, len(T), 500):
        B = T[s:s + 500]; X = np.fft.rfft(B - B.mean(1, keepdims=True), n, axis=1)
        cc = np.fft.irfft(X * Rf, n, axis=1)
        c = np.concatenate([cc[:, -maxl:], cc[:, :maxl + 1]], 1)
        l = c.argmax(1) - maxl; lags[s:s + 500] = l
        for i, li in enumerate(l): out[s + i] = B[i, maxl + li:D - maxl + li]
    return out, lags

def welch(A, B):
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    return (A.mean(0) - B.mean(0)) / np.sqrt(A.var(0, ddof=1) / len(A) + B.var(0, ddof=1) / len(B))

def snr(T, lab):
    T = np.asarray(T, np.float32); u = np.unique(lab)
    m = np.stack([T[lab == k].mean(0) for k in u]); v = np.stack([T[lab == k].var(0) for k in u])
    return m.var(0) / v.mean(0)

def hw(v):
    v = np.asarray(v, np.int64) & 0xFFFF
    return np.array([bin(int(x)).count("1") for x in v])

def balanced_splits(y, seed=1, fr=(0.5, 0.25, 0.25)):
    rng = np.random.default_rng(seed)
    pos = rng.permutation(np.where(y)[0]); neg = rng.permutation(np.where(~y)[0]); m = min(len(pos), len(neg))
    pos, neg = pos[:m], neg[:m]
    c = np.cumsum((0,) + fr)
    cut = lambda a, i: a[int(c[i] * m):int(c[i + 1] * m)]
    return [rng.permutation(np.r_[cut(pos, i), cut(neg, i)]) for i in range(3)]

def thresholds(Xtr, K):
    qs = np.linspace(0, 1, K + 2)[1:-1]
    return np.quantile(Xtr, qs, axis=0).astype(np.float32)

def pack(Xs, Q):
    """Thermometer code, threshold-major bit layout (bit k*D+t), packed as Julia BitVector chunks."""
    K, D = Q.shape; nbits = K * D; nch = -(-nbits // 64)
    out = np.empty((len(Xs), nch), '<u8')
    for s in range(0, len(Xs), 1000):
        B = (Xs[s:s + 1000, None, :] > Q[None]).reshape(-1, nbits)
        B = np.concatenate([B, np.zeros((len(B), nch * 64 - nbits), bool)], 1)
        out[s:s + 1000] = np.packbits(B, axis=1, bitorder='little').view('<u8')
    return out

def write_fptm_inputs(W, X, y, tr, va, te, K, extra_labels=None):
    os.makedirs(W, exist_ok=True)
    Q = thresholds(X[tr], K)
    for name, idx in [("train", tr), ("val", va), ("test", te)]:
        pack(X[idx], Q).tofile(f"{W}/{name}_bits.bin"); y[idx].astype(np.uint8).tofile(f"{W}/{name}_y.bin")
        np.random.default_rng(7).permutation(y[idx]).astype(np.uint8).tofile(f"{W}/{name}_ynull.bin")
    D = X.shape[1]
    json.dump({"D": int(D), "K": K, "nbits": int(K * D), "nchunks": int(-(-K * D // 64)),
               "n": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))}}, open(f"{W}/meta.json", "w"))
    return Q
