"""Held-aside pqm4 set (review round 4, D): variable key chunks 2 and 3 of the Ledger pqm4 campaign, never used for
any configuration choice. Same alignment reference (mean of the first 2,000 traces of chunk 0), the same label
threshold (HW(a0 mod q) > 6) and the same SNR ground truth as the pqm4 set. Writes $ROOT/data/pqm4h/."""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, sys, json, os
sys.path.insert(0, CODE)
from common import align, hw, snr
L = f"{ROOT}/dl/ledger/d0nj0n_mlkem_dataset/pqm4-ref/pqm4/variable_key"; Q = 3329
ref = np.asarray(np.load(f"{L}/leakages/traces_0.npy", mmap_mode="r")[:2000], np.float32).mean(0)
X = np.concatenate([align(np.load(f"{L}/leakages/traces_{i}.npy"), ref)[0] for i in (2, 3)])
a0 = np.concatenate([np.load(f"{L}/metadata/mult_a_{i}.npy")[:, 0] for i in (2, 3)]).astype(np.int64)
h = hw(a0 % Q); y = h > 6
d = f"{ROOT}/data/pqm4h"; os.makedirs(d, exist_ok=True)
np.save(f"{d}/X.npy", X.astype(np.float32)); np.save(f"{d}/y.npy", y)
np.savez(f"{d}/gt.npz", hw=h, a0=a0, snr_hw=snr(X, h))
json.dump({"impl": "pqm4", "label": "HW(a0 mod q) > 6", "secret": "a0", "chunks": [2, 3]}, open(f"{d}/info.json", "w"), indent=1)
print("pqm4h", X.shape, "pos frac %.3f" % y.mean())
