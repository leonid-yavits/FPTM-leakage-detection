"""mkm4: bivariate centered-product Welch test restricted to the 10x10 sample pairs with the highest share SNR
(share SNR from the known shares, a profiling-level assumption), Bonferroni over the 100 pairs, on the balanced pool."""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, sys, json
sys.path.insert(0, CODE)
from common import balanced_splits
from scipy import stats
D = f"{ROOT}/data/mkm4"; X = np.load(f"{D}/X.npy", mmap_mode='r'); y = np.load(f"{D}/y.npy"); g = np.load(f"{D}/gt.npz")
D0 = 9960; p0 = np.argsort(-g["snr_s0"][:D0])[:10]; p1 = D0 + np.argsort(-g["snr_s1"][D0:])[:10]
tr, va, te = balanced_splits(y, seed=1); idx = np.sort(np.r_[tr, va, te]); yy = y[idx]
Xp = np.asarray(X[:, np.r_[p0, p1]], np.float64)[idx]; Xc = Xp - Xp.mean(0)
best = (0, None, None)
for i in range(10):
    for j in range(10, 20):
        cp = Xc[:, i] * Xc[:, j]; A, B = cp[yy], cp[~yy]
        r = stats.ttest_ind(A, B, equal_var=False)
        if abs(r.statistic) > abs(best[0]): best = (r.statistic, r.pvalue, (int(p0[i]), int(p1[j - 10] - D0)))
res = {"N": int(len(idx)), "pairs": 100, "max_abs_t": float(abs(best[0])), "log10p_bonf": float(min(0, np.log10(best[1]) + 2)), "pair": best[2]}
print(json.dumps(res)); json.dump(res, open(f"{ROOT}/results/mkm4_poi_biv.json", "w"))
