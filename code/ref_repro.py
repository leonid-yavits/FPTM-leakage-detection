"""ref: test accuracy and pairwise prediction disagreement of models retrained with seeds 1 to 5 (single thread),
for review round 1 (F18, F23). Writes $ROOT/results/ref_repro.json"""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np, json, itertools
P = {1: f"{ROOT}/work/models/ref_pred.tsv", **{s: f"{ROOT}/work/models/ref_s{s}_pred.tsv" for s in (2, 3, 4, 5)}}
pred = {s: np.loadtxt(p, dtype=np.int64) for s, p in P.items() if _os.path.exists(p)}
res = {"correct": {s: int((v[:, 3] == v[:, 4]).sum()) for s, v in pred.items()}, "n": int(len(next(iter(pred.values()))))}
res["accuracy"] = {s: c / res["n"] for s, c in res["correct"].items()}
res["pairwise_disagreement"] = {f"{a}-{b}": int((pred[a][:, 3] != pred[b][:, 3]).sum()) for a, b in itertools.combinations(sorted(pred), 2)}
res["errors_common_to_all"] = int(np.all([v[:, 3] != v[:, 4] for v in pred.values()], axis=0).sum())
print(json.dumps(res, indent=1)); json.dump(res, open(f"{ROOT}/results/ref_repro.json", "w"), indent=1)
