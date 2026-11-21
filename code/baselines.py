"""Baseline leakage detectors. Every test returns log10 of its (family-wise) p-value; detection at p < 1e-5.
- welch_bonf, chi2_bonf: per-sample tests with Bonferroni correction (as in the Chowdhury-Oswald artifact).
- mv_dcov: multivariate distance covariance test with random projections, code from the
  Chowdhury-Oswald artifact (RP_dcor.py), n_projs = 100. The O(n^2) pairwise distance sum is replaced
  by the identical O(n log n) sorted-order formula. The p-value uses the same gamma approximation.
- dlla_mlp: DL-LA MLP (120-90-50-2, BN, ReLU, softmax, MSE, Adam 1e-3), binomial test on held-out accuracy.
- welch2_bonf: univariate second-order Welch test on centered squared samples, Bonferroni.
- biv_bonf: exhaustive bivariate centered-product Welch test over all sample pairs, Bonferroni.
"""
import numpy as np, math
from scipy import stats
from scipy.special import gammaln

LOG10 = np.log(10)

def _welch_t_df(A, B):
    ma, mb = A.mean(0), B.mean(0); va, vb = A.var(0, ddof=1) / len(A), B.var(0, ddof=1) / len(B)
    t = (ma - mb) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(A) - 1) + vb ** 2 / (len(B) - 1))
    return t, df

def _bonf_log10(logp_min, m):
    return min(0.0, logp_min + np.log10(m))

def welch_bonf(X, y):
    t, df = _welch_t_df(X[y].astype(np.float64), X[~y].astype(np.float64))
    lp = stats.t.logsf(np.abs(t), df) / LOG10 + np.log10(2)
    return _bonf_log10(np.nanmin(lp), X.shape[1]), np.abs(t)

def welch2_bonf(X, y, block=2000):
    """Second-order univariate Welch test on centered squared samples, column blocks to bound memory."""
    D = X.shape[1]; lpmin = 0.0; tabs = np.empty(D)
    for s in range(0, D, block):
        Xc = X[:, s:s + block].astype(np.float64); Xc = (Xc - Xc.mean(0)) ** 2
        t, df = _welch_t_df(Xc[y], Xc[~y]); lp = stats.t.logsf(np.abs(t), df) / LOG10 + np.log10(2)
        lpmin = min(lpmin, float(np.nanmin(lp))); tabs[s:s + block] = np.abs(t)
    return _bonf_log10(lpmin, D), tabs

def chi2_bonf(X, y):
    """Pearson chi-squared contingency test per sample on the raw integer sample values."""
    Xi = np.rint(X).astype(np.int64); D = Xi.shape[1]; lps = np.empty(D)
    n1 = y.sum(); n0 = len(y) - n1; N = len(y)
    for j in range(D):
        v = Xi[:, j]; v = v - v.min(); K = v.max() + 1
        c1 = np.bincount(v[y], minlength=K); c0 = np.bincount(v[~y], minlength=K)
        tot = c0 + c1; nz = tot > 0; c0, c1, tot = c0[nz], c1[nz], tot[nz]
        if len(tot) < 2: lps[j] = 0.0; continue
        e1 = tot * n1 / N; e0 = tot * n0 / N
        dof = len(tot) - 1
        if dof == 1:  # Yates correction, as in scipy.stats.chi2_contingency
            st = (np.maximum(np.abs(c1 - e1) - 0.5, 0) ** 2 / e1 + np.maximum(np.abs(c0 - e0) - 0.5, 0) ** 2 / e0).sum()
        else:
            st = ((c1 - e1) ** 2 / e1 + (c0 - e0) ** 2 / e0).sum()
        lps[j] = stats.chi2.logsf(st, dof) / LOG10
    return _bonf_log10(lps.min(), D), -lps

# ---- MV-dcov (Chowdhury-Oswald artifact, RP_dcor.py) ----
def _gamma_ratio(p): return np.exp(gammaln((p + 1) / 2) - gammaln(p / 2))
def _rproj(X, p, rng):
    u = rng.standard_normal(p); u /= np.linalg.norm(u)
    return u @ X.T if p > 1 else u * X
def _dist_sum(x):  # sum_{i<j} |x_i - x_j|, identical to the artifact's dist_sum
    s = np.sort(x); n = len(s); k = np.arange(1, n + 1)
    return float(np.sum(s * (2 * k - n - 1)))

def mv_dcov(X, y, n_projs=100, seed=0):
    from dcor import u_distance_covariance_sqr
    rng = np.random.default_rng(seed)
    X = X.astype(np.float64); Y = y.astype(np.float64); n, p = X.shape; q = 1
    Cp = math.sqrt(math.pi) * _gamma_ratio(p); Cq = math.sqrt(math.pi) * _gamma_ratio(q)
    o1, s11, s12, o2, o3 = [], [], [], [], []; S2n = S3n = 0.0
    for _ in range(n_projs):
        x1 = _rproj(X, p, rng); y1 = _rproj(Y, q, rng); x2 = _rproj(X, p, rng); y2 = _rproj(Y, q, rng)
        S2n += 2 * _dist_sum(x1); S3n += 2 * _dist_sum(y1)
        o1.append(u_distance_covariance_sqr(x1, y1, method='mergesort'))
        s11.append(u_distance_covariance_sqr(x1, x1, method='mergesort'))
        s12.append(u_distance_covariance_sqr(y1, y1, method='mergesort'))
        o2.append(u_distance_covariance_sqr(x1, x2, method='mergesort'))
        o3.append(u_distance_covariance_sqr(y1, y2, method='mergesort'))
    o1b = Cp * Cq * np.mean(o1); S1b = Cp * Cq * np.mean(np.array(s11) * np.array(s12))
    S2b = Cp * S2n / (n_projs * n * (n - 1)); S3b = Cq * S3n / (n_projs * n * (n - 1))
    o2b = Cp ** 2 * np.mean(o2); o3b = Cq ** 2 * np.mean(o3)
    den = (((n_projs - 1) * o2b * o3b) + S1b) / n_projs
    a = 0.5 * (S2b * S3b) ** 2 / den; b = 0.5 * S2b * S3b / den
    stat = n * o1b + S2b * S3b
    return float(stats.gamma.logsf(stat, a=a, scale=1 / b) / LOG10), (float(stat), float(stats.gamma.ppf(0.95, a=a, scale=1 / b)))

def biv_bonf(X, y, block=64):
    """Exhaustive bivariate centered-product Welch test over all pairs (i<j), Bonferroni."""
    Xc = X.astype(np.float64); Xc -= Xc.mean(0); D = Xc.shape[1]; m = D * (D - 1) // 2
    best = 0.0; arg = None; A1, A0 = Xc[y], Xc[~y]
    for s in range(0, D, block):
        P1 = A1[:, s:s + block, None] * A1[:, None, :]; P0 = A0[:, s:s + block, None] * A0[:, None, :]
        t, df = _welch_t_df(P1.reshape(len(A1), -1), P0.reshape(len(A0), -1))
        t = np.abs(t).reshape(-1, D)
        for i in range(t.shape[0]): t[i, :s + i + 1] = 0  # keep pairs j > i only
        k = np.nanargmax(t); tv = t.flat[k]
        if tv > best: best = tv; arg = (s + k // D, k % D); bdf = df.reshape(-1, D).flat[k]
    lp = stats.t.logsf(best, bdf) / LOG10 + np.log10(2)
    return _bonf_log10(lp, m), arg

# ---- DL-LA MLP ----
def binom_log10sf(k, n):
    return float(stats.binom.logsf(k - 1, n, 0.5) / LOG10)

def dlla_mlp(Xtr, ytr, Xte, yte, epochs=30, seed=0, threads=2):
    import torch, torch.nn as nn
    torch.manual_seed(seed); torch.set_num_threads(threads)
    allx = np.concatenate([Xtr, Xte]).astype(np.float32)
    mu, sd = allx.mean(0), allx.std(0) + 1e-8   # DL-LA: point-wise standardization over the whole set
    f = lambda A: torch.from_numpy(((A - mu) / sd).astype(np.float32))
    xt, xv = f(Xtr), f(Xte); yt = torch.nn.functional.one_hot(torch.from_numpy(ytr.astype(np.int64)), 2).float()
    D = xt.shape[1]
    net = nn.Sequential(nn.Linear(D, 120), nn.ReLU(), nn.BatchNorm1d(120), nn.Linear(120, 90), nn.ReLU(), nn.BatchNorm1d(90),
                        nn.Linear(90, 50), nn.ReLU(), nn.BatchNorm1d(50), nn.Linear(50, 2), nn.Softmax(dim=1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8); lossf = nn.MSELoss()
    bs = int(max(32, min(2000, len(xt) // 4)))
    for _ in range(epochs):
        net.train(); perm = torch.randperm(len(xt))
        for s in range(0, len(xt), bs):
            idx = perm[s:s + bs]
            if len(idx) < 2: continue
            opt.zero_grad(); loss = lossf(net(xt[idx]), yt[idx]); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): pred = net(xv).argmax(1).numpy().astype(bool)
    k = int((pred == yte).sum())
    return binom_log10sf(k, len(yte)), k / len(yte)
