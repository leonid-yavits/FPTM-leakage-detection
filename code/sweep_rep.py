"""Traces-to-detection sweep. For each (N, repeat): draw N/2 traces per class (balanced) from the pool,
split each class in half into train/test. Classical tests use all N traces. MLP and FPTM train on the train
half and are tested on the test half. Writes results/<ds>_sweep.jsonl and FPTM input files."""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, sys, json, os, time
sys.path.insert(0, CODE)
from baselines import welch_bonf, chi2_bonf, mv_dcov, dlla_mlp, welch2_bonf, biv_bonf
from common import thresholds, pack

ds = sys.argv[1]; Ns = [int(v) for v in sys.argv[2].split(",")]; R0, R1 = [int(v) for v in sys.argv[3].split(":")]
methods = sys.argv[4].split(","); mlp_epochs = int(sys.argv[5]); K = 3
X = np.load(f"{ROOT}/data/{ds}/X.npy", mmap_mode='r'); y = np.load(f"{ROOT}/data/{ds}/y.npy")
os.makedirs(ROOT + "/results", exist_ok=True); FW = f"{ROOT}/work/sweep_{ds}_rep"; os.makedirs(FW, exist_ok=True)
out = open(f"{ROOT}/results/{ds}_rep_sweep.jsonl", "a")
P, Ng = np.where(y)[0], np.where(~y)[0]
for N in Ns:
    for r in range(R0, R1):
        rng = np.random.default_rng(10007 * N + r); h = N // 2
        p = rng.choice(P, h, replace=False); n = rng.choice(Ng, h, replace=False)
        tr = np.r_[p[:h // 2], n[:h // 2]]; te = np.r_[p[h // 2:], n[h // 2:]]
        idx = np.r_[tr, te]; order = np.argsort(idx); sidx = idx[order]
        Xa = np.empty((len(idx), X.shape[1]), np.float32); Xa[order] = X[sidx]   # sorted read from memmap
        ya = y[idx]; ntr = len(tr)
        rec = {"ds": ds, "N": N, "r": r, "ntr": int(ntr), "nte": int(len(te))}
        t0 = time.time()
        if "welch" in methods: rec["welch"] = welch_bonf(Xa, ya)[0]
        if "chi2" in methods: rec["chi2"] = chi2_bonf(Xa, ya)[0]
        if "mvdcov" in methods: rec["mvdcov"] = mv_dcov(Xa, ya, seed=r)[0]
        if "welch2" in methods: rec["welch2"] = welch2_bonf(Xa, ya)[0]
        if "biv" in methods: rec["biv"], arg = biv_bonf(Xa, ya, block=4); rec["biv_pair"] = [int(arg[0]), int(arg[1])]
        if "mlp" in methods:
            lp, acc = dlla_mlp(Xa[:ntr], ya[:ntr], Xa[ntr:], ya[ntr:], epochs=mlp_epochs, seed=r); rec["mlp"] = lp; rec["mlp_acc"] = acc
        if "fptm" in methods:   # thresholds from the train half only
            Q = thresholds(Xa[:ntr], K)
            pack(Xa[:ntr], Q).tofile(f"{FW}/{N}_{r}_train_bits.bin"); ya[:ntr].astype(np.uint8).tofile(f"{FW}/{N}_{r}_train_y.bin")
            pack(Xa[ntr:], Q).tofile(f"{FW}/{N}_{r}_test_bits.bin"); ya[ntr:].astype(np.uint8).tofile(f"{FW}/{N}_{r}_test_y.bin")
            with open(f"{FW}/list.txt", "a") as fl: fl.write(f"{N} {r} {ntr} {len(te)}\n")
        rec["sec"] = round(time.time() - t0, 1)
        out.write(json.dumps(rec) + "\n"); out.flush(); print(json.dumps(rec), flush=True)
json.dump({"D": int(X.shape[1]), "K": K, "nbits": int(K * X.shape[1]), "nchunks": int(-(-K * X.shape[1] // 64))}, open(f"{FW}/meta.json", "w"))
