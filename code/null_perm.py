"""Permuted-label controls for the FPTM detector (review round 4, D). Writes label files {train,val,test}_ynull<R>.bin
into $ROOT/work/<DS>_K3 for R = 1..NR, each split permuted with its own generator (seed 1000 + R).
Usage: python3 null_perm.py DS NR"""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np, sys
ds, NR = sys.argv[1], int(sys.argv[2]); W = f"{ROOT}/work/{ds}_K3"
y = np.load(f"{ROOT}/data/{ds}/y.npy"); sp = np.load(f"{W}/split.npz")
for R in range(1, NR + 1):
    for name in ("train", "val", "test"):
        idx = sp[{"train": "tr", "val": "va", "test": "te"}[name]]
        np.random.default_rng(1000 + R).permutation(y[idx]).astype(np.uint8).tofile(f"{W}/{name}_ynull{R}.bin")
print(ds, NR)
