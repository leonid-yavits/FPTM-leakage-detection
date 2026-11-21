"""Numbers quoted in the paper that had no result file (review round 1, F15). Writes $ROOT/results/extra_stats.json.
  * threshold ablation K = 1, 3, 5: precision at k = 20 (pqm4, ASCAD), from work/<ds>_K<k>/S_s*.bin
  * precision without tolerance (window 0) for ref, pqm4, ASCAD
  * overlap of the top 10 samples of the unweighted and the vote-weighted score, per seed and set
  * mkm4 share ranges and the correlation of the second share with the unmasked coefficient
The drift measurements (pqm4 fixed key campaign, 20 Nov) are copied from the feasibility run log, because the raw
chunks 1 and 19 were deleted after the measurement; they can be recomputed with drift.py after downloading them again."""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np, json
from scipy import stats

def ref_mask(ds, W):
    g = np.load(f"{ROOT}/data/{ds}/gt.npz"); sp = np.load(f"{W}/split.npz"); D = len(sp["t_te"])
    if ds == "ascad":
        r = np.zeros(D, bool)
        for k in ("snr_mask0", "snr_vmask0", "snr_mask15", "snr_vmask15"): r |= g[k] > 0.1
    else:
        r = np.abs(sp["t_te"]) > stats.t.isf(1e-5 / 2 / D, len(sp["te"]) - 2)
    return r, D

def dilate(r, w):
    d = np.zeros(len(r), bool)
    for i in np.where(r)[0]: d[max(0, i - w):i + w + 1] = True
    return d

def scores(W, D, s):
    a = np.fromfile(f"{W}/S_s{s}.bin", np.float64); return a[:D], a[D:2 * D]

out = {"K_ablation_p@20": {}, "no_tolerance": {}, "vote_weighted_top10_overlap": {}}
for ds in ("pqm4", "ascad"):
    for K in (1, 3, 5):
        W = f"{ROOT}/work/{ds}_K{K}"; r, D = ref_mask(ds, W); d = dilate(r, 3)
        S = np.mean([scores(W, D, s)[0] for s in (1, 2, 3)], 0)
        out["K_ablation_p@20"][f"{ds}_K{K}"] = float(d[np.argsort(-S)[:20]].mean())
for ds in ("ref", "pqm4", "ascad"):
    W = f"{ROOT}/work/{ds}_K3"; r, D = ref_mask(ds, W)
    S = np.mean([scores(W, D, s)[0] for s in (1, 2, 3)], 0); o = np.argsort(-S)
    out["no_tolerance"][ds] = {f"p@{k}": float(r[o[:k]].mean()) for k in (10, 20)} | {"chance": float(r.mean())}
    out["vote_weighted_top10_overlap"][ds] = [int(len(set(np.argsort(-a)[:10]) & set(np.argsort(-b)[:10])))
                                             for a, b in (scores(W, D, s) for s in (1, 2, 3))]
g = np.load(f"{ROOT}/data/mkm4/gt.npz")
out["mkm4_shares"] = {"s0_range": [int(g["s0"].min()), int(g["s0"].max())], "s1_range": [int(g["s1"].min()), int(g["s1"].max())],
                      "corr_s1_b0": float(np.corrcoef(g["s1"], g["b0"])[0, 1]), "corr_s0_b0": float(np.corrcoef(g["s0"], g["b0"])[0, 1])}
import sys; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import welch
X = np.load(f"{ROOT}/data/mkm4/X.npy", mmap_mode="r"); y = np.load(f"{ROOT}/data/mkm4/y.npy"); w = X.shape[1] // 2
Xs = np.asarray(X[:, w + 455:w + 475], np.float64); t = welch(Xs[y], Xs[~y]); i = int(np.abs(t).argmax())
yp = np.random.default_rng(7).permutation(y); tp = welch(Xs[yp], Xs[~yp]); col = Xs[:, i]; hw = g["hw"]
out["mkm4_peak"] = {"peak_abs_t": float(abs(t[i])), "peak_sample_in_window": 455 + i,
                    "mean_by_hw": {int(h): float(col[hw == h].mean()) for h in np.unique(hw)},
                    "perm_max_abs_t_462_467": float(np.abs(tp[7:13]).max())}
out["drift_pqm4_fixed_key_transcribed"] = {
    "source": "feasibility run of 20 Nov 2026 (drift.py), Welch t on 10,000 vs 10,000 traces, 20,000 samples, threshold 4.5",
    "chunk0_random_interleaved_halves": {"max_t": 3.4, "samples_over_4.5": 0},
    "chunk0_first_vs_second_half": {"max_t": 14.0, "samples_over_4.5": 13191},
    "chunk0_vs_chunk1": {"max_t": 15.4, "samples_over_4.5": 13604},
    "chunk0_vs_chunk19": {"max_t": 90.0, "samples_over_4.5": 18399},
    "fixed_chunk0_vs_variable_chunk0": {"max_t": 127.9, "samples_over_4.5": 18701}}
print(json.dumps(out, indent=1)); json.dump(out, open(f"{ROOT}/results/extra_stats.json", "w"), indent=1)
