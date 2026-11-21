"""Localization on the simulated GainP traces. The noiseless simulation gives the leaking cycles exactly:
a cycle leaks if the Welch |t| between the classes, computed on all 50,000 noiseless traces, exceeds the
Bonferroni threshold (family-wise error rate 1e-5). A top-k cycle counts as a hit only if it is a leaking cycle
(no tolerance window). A second reference set holds the 100 cycles with the largest noiseless |t|.
Usage: python3 loc_eval_gainp.py DATASET"""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np, sys, json

ds = sys.argv[1]; W = f"{ROOT}/work/{ds}_K3"
g = np.load(f"{ROOT}/data/{ds}/gt.npz"); leak, strong, kind = g["leak"], g["strong100"], g["kind"]; D = len(leak)
S = {s: np.fromfile(f"{W}/S_s{s}.bin", np.float64)[:D] for s in (1, 2, 3)}
SA = {s: np.load(f"{W}/SA_s{s}.npy") for s in (1, 2, 3)}
scores = {"fptm": np.mean(list(S.values()), 0), "dlla_sa": np.mean(list(SA.values()), 0)}
res = {"ds": ds, "cycles": int(D), "compare_cycles": int(kind.sum()), "leaking_cycles": int(leak.sum()),
       "chance_leak": float(leak.mean()), "chance_strong100": float(strong.mean())}
for name, sc in scores.items():
    order = np.argsort(-sc)
    r = {}
    for k in (10, 20, 50, 100):
        r[f"p@{k}"] = float(leak[order[:k]].mean()); r[f"strong_p@{k}"] = float(strong[order[:k]].mean())
        r[f"write_cycles_in_top{k}"] = int((~kind[order[:k]]).sum())
    per = S if name == "fptm" else SA
    r["p@20_seeds"] = [float(leak[np.argsort(-v)[:20]].mean()) for v in per.values()]
    r["strong_p@20_seeds"] = [float(strong[np.argsort(-v)[:20]].mean()) for v in per.values()]
    res[name] = r
if True:   # Welch on the noisy training traces, as a pointwise reference ranking
    sp = np.load(f"{W}/split.npz"); order = np.argsort(-np.abs(np.nan_to_num(sp["t_tr"])))
    res["welch_train"] = {f"p@{k}": float(leak[order[:k]].mean()) for k in (10, 20, 50, 100)}
    res["welch_train"].update({f"strong_p@{k}": float(strong[order[:k]].mean()) for k in (10, 20, 50, 100)})
print(json.dumps(res, indent=1)); json.dump(res, open(f"{ROOT}/results/loc_{ds}.json", "w"), indent=1)
