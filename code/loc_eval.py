"""Localization evaluation: precision@k of S(t) against a reference leakage set, with the chance level.
Reference set:
  unmasked: samples whose Welch |t| on the held-out test split exceeds the Bonferroni threshold (FWER 1e-5).
  masked (ASCAD): samples where the SNR of a share (mask byte or masked value) exceeds a threshold, from the known masks.
A top-k sample counts as a hit if it lies within +-w samples of the reference set."""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, sys, json
from scipy import stats
ds, W, w = sys.argv[1], sys.argv[2], int(sys.argv[3]); seeds = sys.argv[4].split(",")
g = np.load(f"{ROOT}/data/{ds}/gt.npz"); sp = np.load(f"{W}/split.npz"); D = len(sp["t_te"])
if ds == "ascad":
    share = {k: g[k] for k in ("snr_mask0", "snr_vmask0", "snr_mask15", "snr_vmask15")}
    ref = np.zeros(D, bool)
    for k, v in share.items(): ref |= v > 0.1
    refname = "share SNR > 0.1 (mask bytes 0 and 15, and masked values)"
else:
    n = len(sp["te"]); thr = stats.t.isf(1e-5 / 2 / D, n - 2); ref = np.abs(sp["t_te"]) > thr
    refname = f"Welch |t| on test split > {thr:.2f} (Bonferroni)"
dil = np.zeros(D, bool)
for i in np.where(ref)[0]: dil[max(0, i - w):i + w + 1] = True
chance = dil.mean()
S = np.mean([np.fromfile(f"{W}/S_s{k}.bin", np.float64)[:D] for k in seeds], axis=0)
res = {"ds": ds, "ref": refname, "ref_samples": int(ref.sum()), "window": w, "chance": float(chance)}
for k in (5, 10, 20, 50):
    top = np.argsort(-S)[:k]; res[f"p@{k}"] = float(dil[top].mean())
    per = [float(dil[np.argsort(-np.fromfile(f"{W}/S_s{s}.bin", np.float64)[:D])[:k]].mean()) for s in seeds]
    res[f"p@{k}_seeds"] = per
if ds == "ascad":
    top = np.argsort(-S)[:20]
    res["top20"] = sorted(int(i) for i in top)
    res["top20_share"] = {k: [int(i) for i in top if v[i] > 0.1] for k, v in share.items()}
print(json.dumps(res))
json.dump(res, open(f"{ROOT}/results/loc_{ds}.json", "w"), indent=1)
