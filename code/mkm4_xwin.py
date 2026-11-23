"""Label-free restricted bivariate test on mkm4 (review round 4, B1). The trace holds two windows, one per arithmetic
share, of D0 samples each. The test pairs sample i of the first window with sample D0 + i + delta of the second,
|delta| <= Delta. It uses only the trace layout, not the shares or the secret. Centered-product Welch test with
Bonferroni over the tested pairs. Two streaming passes over the traces (global means, then per-class sums).
Usage: python3 mkm4_xwin.py mkm4|mkm4big     Writes $ROOT/results/mkm4_xwin_<ds>.json"""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np, sys, json
from scipy import stats

LOG10 = np.log(10); DELTAS = [0, 1, 2, 5, 10, 20, 50]; DM = max(DELTAS)
ds = sys.argv[1]
X = np.load(f"{ROOT}/data/{ds}/X.npy", mmap_mode="r"); y = np.load(f"{ROOT}/data/{ds}/y.npy").astype(bool)
N, D = X.shape; D0 = D // 2; CH = 1000
mu = np.zeros(D)
for s in range(0, N, CH): mu += np.asarray(X[s:s + CH], np.float64).sum(0)
mu /= N
S1 = np.zeros((2, 2 * DM + 1, D0)); S2 = np.zeros((2, 2 * DM + 1, D0)); n = np.zeros(2)
for s in range(0, N, CH):
    B = np.asarray(X[s:s + CH], np.float32) - mu.astype(np.float32); yy = y[s:s + CH]
    A, C = B[:, :D0], B[:, D0:]
    for c, m in ((1, yy), (0, ~yy)):
        Am, Cm = A[m], C[m]; n[c] += m.sum()
        for k, d in enumerate(range(-DM, DM + 1)):
            i0, i1 = max(0, -d), min(D0, D0 - d)
            P = (Am[:, i0:i1] * Cm[:, i0 + d:i1 + d]).astype(np.float64)
            S1[c, k, i0:i1] += P.sum(0); S2[c, k, i0:i1] += (P * P).sum(0)
    print(s, flush=True)
m1, m0 = S1[1] / n[1], S1[0] / n[0]
v1 = (S2[1] - n[1] * m1 ** 2) / (n[1] - 1) / n[1]; v0 = (S2[0] - n[0] * m0 ** 2) / (n[0] - 1) / n[0]
t = np.abs(m1 - m0) / np.sqrt(v1 + v0); df = (v1 + v0) ** 2 / (v1 ** 2 / (n[1] - 1) + v0 ** 2 / (n[0] - 1))
valid = np.zeros((2 * DM + 1, D0), bool)
for k, d in enumerate(range(-DM, DM + 1)): valid[k, max(0, -d):min(D0, D0 - d)] = True
lp = np.where(valid, stats.t.logsf(np.where(valid, t, 0), np.where(valid, df, 1)) / LOG10 + np.log10(2), 0.0)
res = {"ds": ds, "N": int(N), "D0": int(D0)}
for Dl in DELTAS:
    rows = slice(DM - Dl, DM + Dl + 1); sub = lp[rows]; m = int(valid[rows].sum()); k = np.unravel_index(np.argmin(sub), sub.shape)
    res[f"delta{Dl}"] = {"log10p_bonf": float(min(0.0, sub.min() + np.log10(m))), "pairs": m,
                         "best_pair": [int(k[1]), int(D0 + k[1] + (k[0] - Dl))]}
print(json.dumps(res, indent=1))
json.dump(res, open(f"{ROOT}/results/mkm4_xwin_{ds}.json", "w"), indent=1)
