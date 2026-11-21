"""ASCAD region ablation (review round 1, F11). FPTM inputs restricted to sample subsets, same split as work/ascad_K3.
  prep VARIANT   write $ROOT/work/ascad_abl_<VARIANT>/ and run first-order chi2 and Welch on the subset
  eval VARIANT   after fptm_detect.jl seeds 1 to 3: localization of the subset model in original sample indices
Share regions are data-defined: samples where the SNR of the mask (mask bytes 0 and 15) or of the masked value exceeds 0.1,
dilated by 3 samples. The overlap 140..190 holds the samples where single samples depend on both shares.
Variants: all, no_overlap, no_overlap_no_mask, no_overlap_no_mv, mask_only, mv_only, both_shares_only."""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, sys, json
sys.path.insert(0, CODE)
from common import write_fptm_inputs, welch
from baselines import chi2_bonf, welch_bonf

_g = np.load(f"{ROOT}/data/ascad/gt.npz")
def _dil(b, w=3):
    d = np.zeros(len(b), bool)
    for i in np.where(b)[0]: d[max(0, i - w):i + w + 1] = True
    return d
_M = _dil((_g["snr_mask0"] > 0.1) | (_g["snr_mask15"] > 0.1))      # mask share samples (SNR > 0.1, +-3)
_Vm = _dil((_g["snr_vmask0"] > 0.1) | (_g["snr_vmask15"] > 0.1))   # masked value share samples (SNR > 0.1, +-3)
_O = np.zeros(700, bool); _O[140:191] = True                          # overlap: single samples depend on both shares
_all = np.ones(700, bool)
V = {"all": np.where(_all)[0],
     "no_overlap": np.where(~_O)[0],
     "no_overlap_no_mask": np.where(~_O & ~_M)[0],
     "no_overlap_no_mv": np.where(~_O & ~_Vm)[0],
     "mask_only": np.where(_M & ~_O & ~_Vm)[0],
     "mv_only": np.where(_Vm & ~_O & ~_M)[0],
     "both_shares_only": np.where((_M | _Vm) & ~_O)[0]}
cmd, var = sys.argv[1], sys.argv[2]; I = V[var]
W = f"{ROOT}/work/ascad_abl_{var}"
g = np.load(f"{ROOT}/data/ascad/gt.npz")
msk = (g["snr_mask0"] > 0.1) | (g["snr_mask15"] > 0.1); vm = (g["snr_vmask0"] > 0.1) | (g["snr_vmask15"] > 0.1)
ref = msk | vm
dil = np.zeros(700, bool)
for i in np.where(ref)[0]: dil[max(0, i - 3):i + 4] = True

if cmd == "prep":
    X = np.asarray(np.load(f"{ROOT}/data/ascad/X.npy", mmap_mode="r"))[:, I]; y = np.load(f"{ROOT}/data/ascad/y.npy")
    sp = np.load(f"{ROOT}/work/ascad_K3/split.npz"); tr, va, te = sp["tr"], sp["va"], sp["te"]
    write_fptm_inputs(W, X, y, tr, va, te, 3)
    np.savez(f"{W}/split.npz", tr=tr, va=va, te=te, t_te=welch(X[te][y[te]], X[te][~y[te]]), cols=I)
    allidx = np.r_[tr, va, te]
    res = {"variant": var, "samples": int(len(I)), "traces": int(len(allidx)),
           "chi2_log10p": chi2_bonf(X[allidx], y[allidx])[0], "welch_log10p": welch_bonf(X[allidx], y[allidx])[0]}
    print(json.dumps(res)); json.dump(res, open(f"{W}/first_order.json", "w"))
else:
    res = json.load(open(f"{W}/first_order.json"))
    S = {s: np.fromfile(f"{W}/S_s{s}.bin", np.float64)[:len(I)] for s in (1, 2, 3)}
    full = lambda v: (lambda a: (a.__setitem__(I, v), a)[1])(np.full(700, -1.0))
    Sm = full(np.mean(list(S.values()), 0))
    res["chance_in_subset"] = float(dil[I].mean())
    for k in (10, 20):
        top = np.argsort(-Sm)[:k]
        res[f"p@{k}"] = float(dil[top].mean())
        res[f"p@{k}_seeds"] = [float(dil[np.argsort(-full(v))[:k]].mean()) for v in S.values()]
    top = np.argsort(-Sm)[:20]
    res["top20_mask_share"] = int(_M[top].sum()); res["top20_mv_share"] = int(_Vm[top].sum()); res["top20_neither"] = int((~_M & ~_Vm)[top].sum())
    res["top20"] = sorted(int(t) for t in top)
    runs = [l for l in open(f"{W}/runs.log") if "RESULT" in l]
    res["fptm_runs"] = [l.strip() for l in runs]
    print(json.dumps(res, indent=1)); json.dump(res, open(f"{ROOT}/results/ascad_ablation_{var}.json", "w"), indent=1)
