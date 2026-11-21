"""Wall-clock time per method on the main split (train+test traces for classical tests; train on train, test on test for MLP)."""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, sys, time, json
sys.path.insert(0, CODE)
from baselines import welch_bonf, chi2_bonf, mv_dcov, dlla_mlp, biv_bonf
ds = sys.argv[1]; epochs = int(sys.argv[2])
X = np.load(f"{ROOT}/data/{ds}/X.npy", mmap_mode='r'); y = np.load(f"{ROOT}/data/{ds}/y.npy")
sp = np.load(f"{ROOT}/work/{ds}_K3/split.npz"); tr, te = sp["tr"], sp["te"]
Xtr = np.asarray(X[np.sort(tr)], np.float32); ytr = y[np.sort(tr)]; Xte = np.asarray(X[np.sort(te)], np.float32); yte = y[np.sort(te)]
Xa = np.concatenate([Xtr, Xte]); ya = np.concatenate([ytr, yte]); res = {"ds": ds, "N_all": int(len(ya)), "N_train": int(len(ytr))}
for name, f in [("welch", lambda: welch_bonf(Xa, ya)), ("chi2", lambda: chi2_bonf(Xa, ya)), ("mvdcov", lambda: mv_dcov(Xa, ya)),
                ("mlp", lambda: dlla_mlp(Xtr, ytr, Xte, yte, epochs=epochs))] + ([("biv", lambda: biv_bonf(Xa, ya, block=4))] if ds == "ascad" else []):
    t0 = time.time(); f(); res[name] = round(time.time() - t0, 1); print(name, res[name], flush=True)
json.dump(res, open(f"{ROOT}/results/timing_{ds}.json", "w")); print(json.dumps(res))
