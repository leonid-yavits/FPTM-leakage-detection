"""ASCAD localization by the classical tests that detect: per-sample score = max bivariate |t| over all partners
(exhaustive centered-product Welch), and -log10 p of the per-sample chi2 test. Same traces as the timing run
(training + test parts of the localization split). Same precision@k metric."""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, sys, json
sys.path.insert(0, CODE)
from baselines import _welch_t_df, chi2_bonf
X = np.load(ROOT + "/data/ascad/X.npy", mmap_mode='r'); y = np.load(ROOT + "/data/ascad/y.npy")
sp = np.load(ROOT + "/work/ascad_K3/split.npz"); idx = np.sort(np.r_[sp["tr"], sp["te"]])
Xa = np.asarray(X[idx], np.float64); ya = y[idx]; D = Xa.shape[1]
Xc = Xa - Xa.mean(0); A1, A0 = Xc[ya], Xc[~ya]; tmax = np.zeros(D)
for s in range(0, D, 4):
    P1 = A1[:, s:s + 4, None] * A1[:, None, :]; P0 = A0[:, s:s + 4, None] * A0[:, None, :]
    t, _ = _welch_t_df(P1.reshape(len(A1), -1), P0.reshape(len(A0), -1)); t = np.abs(t).reshape(-1, D)
    for i in range(t.shape[0]): t[i, s + i] = 0          # exclude the squared term (i, i)
    tmax[s:s + 4] = np.maximum(tmax[s:s + 4], t.max(1)); tmax = np.maximum(tmax, t.max(0))
_, chi = chi2_bonf(Xa, ya)
g = np.load(ROOT + "/data/ascad/gt.npz"); ref = np.zeros(D, bool)
for k in ("snr_mask0", "snr_vmask0", "snr_mask15", "snr_vmask15"): ref |= g[k] > 0.1
dil = np.zeros(D, bool)
for i in np.where(ref)[0]: dil[max(0, i - 3):i + 4] = True
M = np.maximum(g["snr_mask0"], g["snr_mask15"]) > 0.1; V = np.maximum(g["snr_vmask0"], g["snr_vmask15"]) > 0.1
def dl(r):
    d = np.zeros(D, bool)
    for i in np.where(r)[0]: d[max(0, i - 3):i + 4] = True
    return d
Md, Vd = dl(M), dl(V)
res = {"N": int(len(ya))}
for name, S in (("bivariate", tmax), ("chi2", chi)):
    top = np.argsort(-S)
    res[name] = {f"p@{k}": float(dil[top[:k]].mean()) for k in (5, 10, 20, 50)}
    res[name]["p0@20"] = float(ref[top[:20]].mean())
    res[name]["top20_mask_region"] = int(Md[top[:20]].sum()); res[name]["top20_value_region"] = int(Vd[top[:20]].sum())
O = np.zeros(D, bool); O[140:191] = True
np.save(ROOT + "/work/ascad_K3/bivariate_tmax.npy", tmax); np.save(ROOT + "/work/ascad_K3/chi2_score.npy", chi)
for name, S in (("bivariate", tmax), ("chi2", chi)):
    t20 = np.argsort(-S)[:20]
    res[name]["top20_overlap"] = int(O[t20].sum()); res[name]["top20_mask_outside_overlap"] = int((Md & ~O)[t20].sum())
    res[name]["top20_value_outside_overlap"] = int((Vd & ~O)[t20].sum())
S = np.mean([np.fromfile(f"{ROOT}/work/ascad_K3/S_s{k}.bin", np.float64)[:D] for k in (1, 2, 3)], axis=0); top = np.argsort(-S)
res["fptm_top20_overlap"] = int(O[top[:20]].sum()); res["fptm_top20_mask_outside_overlap"] = int((Md & ~O)[top[:20]].sum()); res["fptm_top20_value_outside_overlap"] = int((Vd & ~O)[top[:20]].sum())
res["fptm_top20_mask_region"] = int(Md[top[:20]].sum()); res["fptm_top20_value_region"] = int(Vd[top[:20]].sum())
print(json.dumps(res, indent=1)); json.dump(res, open(ROOT + "/results/ascad_classic_loc.json", "w"), indent=1)
