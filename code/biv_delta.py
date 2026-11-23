"""Lag-restricted bivariate Welch test (review round 4, B1). Centered-product Welch test over all sample pairs (i, j)
with 0 < j - i <= Delta, Bonferroni over the number of tested pairs. Delta = D - 1 is the exhaustive pair test.
Runs on the same repeated-sweep subsets as sweep_rep.py (seed 10007*N + r, all N traces).
Usage: python3 biv_delta.py ascad|ascad_noov N1,N2,... R0:R1
  ascad_noov: ASCAD with samples 140 to 190 removed (pairs involving them are not tested; lags in original samples).
Writes $ROOT/results/biv_delta_<variant>.jsonl, one record per subset with log10 p for every Delta."""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np, sys, json
from scipy import stats

LOG10 = np.log(10)
DELTAS = [int(v) for v in _os.environ.get("BIV_DELTAS", "1,2,5,10,20,50,100,200,300,400,699").split(",")]
var = sys.argv[1]; Ns = [int(v) for v in sys.argv[2].split(",")]; R0, R1 = [int(v) for v in sys.argv[3].split(":")]
X = np.load(f"{ROOT}/data/ascad/X.npy", mmap_mode="r"); y = np.load(f"{ROOT}/data/ascad/y.npy"); D = X.shape[1]
keep = np.ones(D, bool)
if var == "ascad_noov": keep[140:191] = False
P, Ng = np.where(y)[0], np.where(~y)[0]
out = open(f"{ROOT}/results/biv_delta_{var}{_os.environ.get('BIV_TAG', '')}.jsonl", "a")


def lag_minlogp(Xc, yy, kmax):
    """Per lag k = 1..kmax: (smallest two-sided log10 p over valid pairs at that lag, number of valid pairs)."""
    A1, A0 = Xc[yy], Xc[~yy]; res = []
    for k in range(1, kmax + 1):
        v = keep[:D - k] & keep[k:]
        if not v.any(): res.append((0.0, 0)); continue
        P1 = A1[:, :D - k][:, v] * A1[:, k:][:, v]; P0 = A0[:, :D - k][:, v] * A0[:, k:][:, v]
        m1, m0 = P1.mean(0), P0.mean(0); s1, s0 = P1.var(0, ddof=1) / len(P1), P0.var(0, ddof=1) / len(P0)
        t = np.abs(m1 - m0) / np.sqrt(s1 + s0); df = (s1 + s0) ** 2 / (s1 ** 2 / (len(P1) - 1) + s0 ** 2 / (len(P0) - 1))
        lp = stats.t.logsf(t, df) / LOG10 + np.log10(2)
        res.append((float(np.nanmin(lp)), int(v.sum())))
    return res


for N in Ns:
    for r in range(R0, R1):
        rng = np.random.default_rng(10007 * N + r); h = N // 2
        p = rng.choice(P, h, replace=False); n = rng.choice(Ng, h, replace=False)
        idx = np.sort(np.r_[p, n])
        Xc = np.asarray(X[idx], np.float64); Xc -= Xc.mean(0); yy = y[idx]
        L = lag_minlogp(Xc, yy, max(DELTAS))
        rec = {"N": N, "r": r, "var": var}
        for d in DELTAS:
            lp = min(x[0] for x in L[:d]); m = sum(x[1] for x in L[:d])
            rec[f"d{d}"] = min(0.0, lp + np.log10(m)); rec[f"m{d}"] = m
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(N, r, {d: round(rec[f"d{d}"], 1) for d in DELTAS}, flush=True)
