"""MV-dCov stability check (review round 1, F16): the same trace subset tested with 10 projection seeds.
Writes $ROOT/results/mvdcov_seeds.json."""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, sys, json
sys.path.insert(0, CODE)
from baselines import mv_dcov
res = {}
for ds, N in (("gainp_s2000", 1000), ("gainp_s2000", 4000), ("pqm4", 4000), ("ascad", 16000)):
    X = np.load(f"{ROOT}/data/{ds}/X.npy", mmap_mode="r"); y = np.load(f"{ROOT}/data/{ds}/y.npy")
    rng = np.random.default_rng(10007 * N); P, Ng = np.where(y)[0], np.where(~y)[0]
    idx = np.sort(np.r_[rng.choice(P, N // 2, replace=False), rng.choice(Ng, N // 2, replace=False)])
    Xa = np.asarray(X[idx], np.float32); ya = y[idx]
    lp = [mv_dcov(Xa, ya, seed=s)[0] for s in range(10)]
    res[f"{ds}_N{N}"] = {"log10p": lp, "detections": int(sum(v < -5 for v in lp))}
    print(ds, N, [round(v, 1) for v in lp], flush=True)
json.dump(res, open(f"{ROOT}/results/mvdcov_seeds.json", "w"), indent=1)
