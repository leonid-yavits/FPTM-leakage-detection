"""False-positive check of the classical tests (review round 1, F17): Welch and chi2 (Bonferroni) on the largest
sweep subset of each public set, with the labels permuted 10 times. Writes $ROOT/results/null_classic.json."""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, sys, json
sys.path.insert(0, CODE)
from baselines import welch_bonf, chi2_bonf
res = {}
for ds, N in (("ref", 12000), ("pqm4", 12000), ("mkm4", 12000), ("ascad", 40000)):
    X = np.load(f"{ROOT}/data/{ds}/X.npy", mmap_mode="r"); y = np.load(f"{ROOT}/data/{ds}/y.npy")
    rng = np.random.default_rng(10007 * N); P, Ng = np.where(y)[0], np.where(~y)[0]
    idx = np.sort(np.r_[rng.choice(P, N // 2, replace=False), rng.choice(Ng, N // 2, replace=False)])
    Xa = np.asarray(X[idx], np.float32); ya = y[idx]; r = {"N": N, "welch": [], "chi2": []}
    for k in range(10):
        yp = np.random.default_rng(100 + k).permutation(ya)
        r["welch"].append(welch_bonf(Xa, yp)[0]); r["chi2"].append(chi2_bonf(Xa, yp)[0])
    r["false_detections_welch"] = int(sum(v < -5 for v in r["welch"])); r["false_detections_chi2"] = int(sum(v < -5 for v in r["chi2"]))
    r["min_log10p_welch"] = min(r["welch"]); r["min_log10p_chi2"] = min(r["chi2"])
    res[ds] = r; print(ds, json.dumps({k: v for k, v in r.items() if k not in ("welch", "chi2")}), flush=True)
json.dump(res, open(f"{ROOT}/results/null_classic.json", "w"), indent=1)
