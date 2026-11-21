"""Balanced splits + thermometer inputs + classical ground truth for one dataset."""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, sys, json
sys.path.insert(0, CODE)
from common import balanced_splits, write_fptm_inputs, welch
ds, K = sys.argv[1], int(sys.argv[2]); tag = sys.argv[3] if len(sys.argv) > 3 else f"K{K}"
D = f"{ROOT}/data/{ds}"; W = f"{ROOT}/work/{ds}_{tag}"
X = np.load(f"{D}/X.npy", mmap_mode='r'); y = np.load(f"{D}/y.npy")
tr, va, te = balanced_splits(y, seed=1)
X = np.asarray(X)
write_fptm_inputs(W, X, y, tr, va, te, K)
t_te = welch(X[te][y[te]], X[te][~y[te]]); t_tr = welch(X[tr][y[tr]], X[tr][~y[tr]])
np.savez(f"{W}/split.npz", tr=tr, va=va, te=te, t_te=t_te, t_tr=t_tr)
print(ds, tag, "sizes", len(tr), len(va), len(te), "| 1st-order Welch on test: max|t|=%.1f at %d" % (np.abs(t_te).max(), np.abs(t_te).argmax()))
