"""Traces to detection on the simulated GainP sets, same rule as aggregate.py (all three repeats detect at N and
at every larger N of the grid). Writes results/aggregate_gainp.json."""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, collections
SIG = [250, 500, 1000, 2000]
METHODS = ["welch", "chi2", "mvdcov", "mlp", "fptm"]
out = {}
for s in SIG:
    ds = f"gainp_s{s}"; rows = collections.defaultdict(dict)
    for f in (f"{ROOT}/results/{ds}_sweep.jsonl", f"{ROOT}/results/{ds}_fptm.jsonl"):
        if _os.path.exists(f):
            for l in open(f):
                r = json.loads(l); rows[(r["N"], r["r"])].update(r)
    Ns = sorted({k[0] for k in rows}); res = {"grid": Ns}
    for m in METHODS:
        per = {N: [rows[(N, r)][m] for r in range(3) if m in rows.get((N, r), {})] for N in Ns}
        per = {N: v for N, v in per.items() if v}
        if not per: continue
        ttd = None
        for N in sorted(per):
            if all(len(per[M]) == 3 and all(x < -5 for x in per[M]) for M in sorted(per) if M >= N): ttd = N; break
        res[m] = {"per_N": {str(N): per[N] for N in per}, "detections": {str(N): sum(x < -5 for x in per[N]) for N in per}, "ttd": ttd}
    res["acc"] = {str(N): {k: [rows[(N, r)].get(k) for r in range(3) if (N, r) in rows] for k in ("mlp_acc", "fptm_acc")} for N in Ns}
    out[ds] = res
    print(f"== {ds} grid {Ns}")
    for m in METHODS:
        if m in res: print(f"  {m:7s} ttd={res[m]['ttd']}  detections={res[m]['detections']}")
json.dump(out, open(f"{ROOT}/results/aggregate_gainp.json", "w"), indent=1)
