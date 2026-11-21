"""Detection rates over 10 repeats per trace count (review round 1, F12), with Wilson 95% intervals.
TTD90: smallest N of the grid with at least 9 of 10 detections at N and at every larger N of the grid.
Usage: python3 agg_rep.py ref pqm4 ascad. Writes $ROOT/results/aggregate_rep.json"""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, sys, math, collections

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

out = {}
for ds in sys.argv[1:]:
    rows = collections.defaultdict(dict)
    for f in (f"{ROOT}/results/{ds}_rep_sweep.jsonl", f"{ROOT}/results/{ds}_rep_fptm.jsonl"):
        if _os.path.exists(f):
            for l in open(f):
                r = json.loads(l); rows[(r["N"], r["r"])].update(r)
    Ns = sorted({k[0] for k in rows}); res = {"grid": Ns}
    for m in ("welch", "chi2", "welch2", "mvdcov", "mlp", "fptm"):
        per = {N: [rows[(N, r)][m] for r in range(10) if m in rows.get((N, r), {})] for N in Ns}
        if not any(per.values()): continue
        rate = {N: (sum(v < -5 for v in per[N]), len(per[N])) for N in Ns}
        ttd = None
        for N in Ns:
            if all(rate[M][1] > 0 and rate[M][0] >= 0.9 * rate[M][1] for M in Ns if M >= N): ttd = N; break
        res[m] = {"rate": {str(N): f"{k}/{n}" for N, (k, n) in rate.items()},
                  "wilson95": {str(N): [round(x, 2) for x in wilson(k, n)] for N, (k, n) in rate.items()}, "ttd90": ttd}
        print(f"{ds:6s} {m:7s} ttd90={ttd}  " + "  ".join(f"{N}:{k}/{n}" for N, (k, n) in rate.items()))
    out[ds] = res
json.dump(out, open(f"{ROOT}/results/aggregate_rep.json", "w"), indent=1)
