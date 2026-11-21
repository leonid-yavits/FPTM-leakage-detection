"""Size of the random forest of the localization runs (review round 2, R2-14): same split, data and settings as
loc_baselines.py (300 trees, sqrt features, fit on the training part), seed 1. Writes $ROOT/results/rf_size.json"""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np, json
from sklearn.ensemble import RandomForestClassifier
out = {}
for ds in ("ascad", "pqm4", "ref"):
    X = np.load(f"{ROOT}/data/{ds}/X.npy", mmap_mode="r"); y = np.load(f"{ROOT}/data/{ds}/y.npy")
    tr = np.load(f"{ROOT}/work/{ds}_K3/split.npz")["tr"]
    rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=2, random_state=1).fit(np.asarray(X[tr], np.float32), y[tr])
    nodes = int(sum(e.tree_.node_count for e in rf.estimators_)); leaves = int(sum(e.tree_.n_leaves for e in rf.estimators_))
    out[ds] = {"train_traces": int(len(tr)), "nodes": nodes, "internal_nodes": nodes - leaves, "leaves": leaves,
               "max_depth": int(max(e.tree_.max_depth for e in rf.estimators_))}
    print(ds, out[ds], flush=True)
json.dump(out, open(f"{ROOT}/results/rf_size.json", "w"), indent=1)
