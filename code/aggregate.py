"""Aggregate sweep results: per dataset and method, detections per N and traces to detection
(smallest N in the grid such that all repeats detect at that N and at every larger N)."""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import json, sys, collections
DS = sys.argv[1:] or ["ref", "pqm4", "ascad", "mkm4"]
METHODS = ["welch", "chi2", "mvdcov", "welch2", "biv", "mlp", "fptm"]
out = {}
for ds in DS:
    rows = collections.defaultdict(dict)
    for f in (f"{ROOT}/results/{ds}_sweep.jsonl", f"{ROOT}/results/{ds}_fptm.jsonl"):
        try:
            for l in open(f):
                r = json.loads(l); rows[(r["N"], r["r"])].update(r)
        except FileNotFoundError: pass
    Ns = sorted({k[0] for k in rows})
    res = {}
    for m in METHODS:
        per = {}
        for N in Ns:
            v = [rows[(N, r)][m] for r in range(3) if (N, r) in rows and m in rows[(N, r)]]
            if v: per[N] = v
        if not per: continue
        ok = [N for N in sorted(per) if len(per[N]) == 3 and all(x < -5 for x in per[N])]
        ttd = None
        for N in sorted(per):
            if all(len(per[M]) == 3 and all(x < -5 for x in per[M]) for M in sorted(per) if M >= N): ttd = N; break
        res[m] = {"per_N": {str(N): per[N] for N in per}, "detections": {str(N): sum(x < -5 for x in per[N]) for N in per}, "ttd": ttd}
    for N in Ns:
        accs = {k: [rows[(N, r)].get(k) for r in range(3) if (N, r) in rows] for k in ("mlp_acc", "fptm_acc")}
        res.setdefault("acc", {})[str(N)] = accs
    out[ds] = res
    print(f"== {ds}  N grid {Ns}")
    for m in METHODS:
        if m in res: print(f"  {m:7s} ttd={res[m]['ttd']}  detections={res[m]['detections']}")
json.dump(out, open(f"{ROOT}/results/aggregate.json", "w"), indent=1)
