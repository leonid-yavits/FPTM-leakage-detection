"""Random forest on the ASCAD region-ablation inputs (review round 2, R2-14). Same split and sample subsets as
ascad_ablation.py (work/ascad_abl_<v>/split.npz). 300 trees, sqrt features, fit on the training part, binomial test on
the test part, impurity importance mapped to original sample indices. Writes $ROOT/results/ascad_ablation_rf.json"""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, sys, json
sys.path.insert(0, CODE)
from baselines import binom_log10sf
from sklearn.ensemble import RandomForestClassifier
g = np.load(f"{ROOT}/data/ascad/gt.npz")
def dil(b, w=3):
    d = np.zeros(len(b), bool)
    for i in np.where(b)[0]: d[max(0, i - w):i + w + 1] = True
    return d
M = dil((g["snr_mask0"] > 0.1) | (g["snr_mask15"] > 0.1)); Vm = dil((g["snr_vmask0"] > 0.1) | (g["snr_vmask15"] > 0.1))
ref = M | Vm
X = np.load(f"{ROOT}/data/ascad/X.npy", mmap_mode="r"); y = np.load(f"{ROOT}/data/ascad/y.npy")
out = {}
for v in ["all", "no_overlap", "no_overlap_no_mask", "no_overlap_no_mv", "mask_only", "mv_only", "both_shares_only"]:
    sp = np.load(f"{ROOT}/work/ascad_abl_{v}/split.npz"); I = sp["cols"]; tr, te = sp["tr"], sp["te"]
    Xtr = np.asarray(X[np.sort(tr)])[:, I]; ytr = y[np.sort(tr)]; Xte = np.asarray(X[np.sort(te)])[:, I]; yte = y[np.sort(te)]
    r = {"log10p": [], "acc": [], "p@20": []}; imp = np.zeros(700)
    for s in (1, 2, 3):
        rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=2, random_state=s).fit(Xtr, ytr)
        k = int((rf.predict(Xte) == yte).sum()); r["log10p"].append(binom_log10sf(k, len(yte))); r["acc"].append(k / len(yte))
        a = np.full(700, -1.0); a[I] = rf.feature_importances_; imp[I] += rf.feature_importances_ / 3
        r["p@20"].append(float(ref[np.argsort(-a)[:20]].mean()))
    m = np.full(700, -1.0); m[I] = imp[I]; top = np.argsort(-m)[:20]
    r["mean_p@20"] = float(ref[top].mean()); r["top20_mask"] = int(M[top].sum()); r["top20_mv"] = int(Vm[top].sum())
    out[v] = r; print(v, json.dumps(r), flush=True)
json.dump(out, open(f"{ROOT}/results/ascad_ablation_rf.json", "w"), indent=1)
