"""Robustness to trace misalignment (review round 4, B3). Each trace of the pooled set is cropped to a window of
D - s samples that starts at a uniform random offset in [0, s], as in the desynchronized ASCAD variants. The
localization split of the set is reused. Usage:
  python3 desync.py prep DS S    write FPTM inputs to $ROOT/work/DS_ds<S> (K = 3, thresholds from training traces)
  python3 desync.py base DS S    first-order tests (training and test traces), exhaustive bivariate test (ASCAD only),
                                 epoch-selected DL-LA and random forest (seeds 1 to 3); writes $ROOT/results/desync_DS_s<S>.json
"""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, sys, json
sys.path.insert(0, CODE)
from common import write_fptm_inputs
from baselines import welch_bonf, chi2_bonf, biv_bonf
from learned import dlla_select, rf_detect

mode, ds, s = sys.argv[1], sys.argv[2], int(sys.argv[3])
X0 = np.load(f"{ROOT}/data/{ds}/X.npy", mmap_mode="r"); y = np.load(f"{ROOT}/data/{ds}/y.npy").astype(bool)
sp = np.load(f"{ROOT}/work/{ds}_K3/split.npz"); tr, va, te = sp["tr"], sp["va"], sp["te"]
N, D = X0.shape
off = np.random.default_rng(2027 + s).integers(0, s + 1, N)
X = np.empty((N, D - s), np.float32)
for i0 in range(0, N, 2000):
    B = np.asarray(X0[i0:i0 + 2000], np.float32)
    for i in range(len(B)): X[i0 + i] = B[i, off[i0 + i]:off[i0 + i] + D - s]
W = f"{ROOT}/work/{ds}_ds{s}"
if mode == "prep":
    write_fptm_inputs(W, X, y, tr, va, te, 3); print("prep", ds, s, X.shape)
else:
    idx = np.sort(np.r_[tr, te]); ya = y[idx]; Dn = X.shape[1]
    from scipy import stats
    from baselines import _welch_t_df
    def w_raw(B, yy):   # smallest two-sided log10 p of the per-sample Welch test
        t, df = _welch_t_df(B[yy].astype(np.float64), B[~yy].astype(np.float64))
        return float(np.nanmin(stats.t.logsf(np.abs(t), df) / np.log(10) + np.log10(2)))
    def c_raw(B, yy): return float(-np.max(chi2_bonf(B, yy)[1]))
    def blockwise(f):   # Bonferroni over all Dn samples from column blocks (bounds memory on long traces)
        lp = min(f(X[idx, a:a + 2000], ya) for a in range(0, Dn, 2000))
        return min(0.0, lp + np.log10(Dn))
    res = {"ds": ds, "s": s, "welch": blockwise(w_raw), "chi2": blockwise(c_raw)}
    Xa = X[idx] if ds == "ascad" else None
    if ds == "ascad": res["biv"] = biv_bonf(Xa, ya, block=8)[0]
    Xtr, Xva, Xte = X[tr], X[va], X[te]; del X, Xa
    for seed in (1, 2, 3):
        lp, acc, ep = dlla_select(Xtr, y[tr], Xva, y[va], Xte, y[te], seed=seed, epochs=50 if ds == "ascad" else 30)
        res.setdefault("dlla_sel", []).append({"log10p": lp, "acc": acc, "epoch": ep})
        lp, acc = rf_detect(Xtr, y[tr], Xte, y[te], seed=seed)
        res.setdefault("rf", []).append({"log10p": lp, "acc": acc})
        print(seed, res["dlla_sel"][-1], res["rf"][-1], flush=True)
    print(json.dumps({k: v for k, v in res.items()}, indent=1))
    json.dump(res, open(f"{ROOT}/results/desync_{ds}_s{s}.json", "w"), indent=1)
