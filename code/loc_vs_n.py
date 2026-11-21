"""ASCAD: localization precision versus trace count. Same subsets as the trace count sweep (same RNG formula).
Writes FPTM input bits (thresholds from the train half) and trains DL-LA + sensitivity analysis on each subset."""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, sys, os, json
sys.path.insert(0, CODE)
from common import thresholds, pack
import torch, torch.nn as nn
torch.set_num_threads(2)
ds = "ascad"; Ns = [8000, 16000, 32000]; FW = f"{ROOT}/work/locn_ascad"; os.makedirs(FW, exist_ok=True)
X = np.load(f"{ROOT}/data/{ds}/X.npy", mmap_mode='r'); y = np.load(f"{ROOT}/data/{ds}/y.npy"); D = X.shape[1]; K = 3
P, Ng = np.where(y)[0], np.where(~y)[0]
json.dump({"D": D, "K": K, "nbits": K * D, "nchunks": -(-K * D // 64)}, open(f"{FW}/meta.json", "w"))
open(f"{FW}/list.txt", "w").close()
for N in Ns:
    for r in range(3):
        rng = np.random.default_rng(10007 * N + r); h = N // 2
        p = rng.choice(P, h, replace=False); n = rng.choice(Ng, h, replace=False)
        tr = np.r_[p[:h // 2], n[:h // 2]]; te = np.r_[p[h // 2:], n[h // 2:]]
        Xtr = np.asarray(X[np.sort(tr)], np.float32); ytr = y[np.sort(tr)]; Xte = np.asarray(X[np.sort(te)], np.float32); yte = y[np.sort(te)]
        Q = thresholds(Xtr, K)
        pack(Xtr, Q).tofile(f"{FW}/{N}_{r}_train_bits.bin"); ytr.astype(np.uint8).tofile(f"{FW}/{N}_{r}_train_y.bin")
        pack(Xte, Q).tofile(f"{FW}/{N}_{r}_test_bits.bin"); yte.astype(np.uint8).tofile(f"{FW}/{N}_{r}_test_y.bin")
        with open(f"{FW}/list.txt", "a") as f: f.write(f"{N} {r} {len(ytr)} {len(yte)}\n")
        allx = np.concatenate([Xtr, Xte]); mu, sd = allx.mean(0), allx.std(0) + 1e-8
        xt = torch.from_numpy((Xtr - mu) / sd); xv = torch.from_numpy((Xte - mu) / sd)
        yt = nn.functional.one_hot(torch.from_numpy(ytr.astype(np.int64)), 2).float()
        torch.manual_seed(r)
        net = nn.Sequential(nn.Linear(D, 120), nn.ReLU(), nn.BatchNorm1d(120), nn.Linear(120, 90), nn.ReLU(), nn.BatchNorm1d(90),
                            nn.Linear(90, 50), nn.ReLU(), nn.BatchNorm1d(50), nn.Linear(50, 2), nn.Softmax(dim=1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3); lossf = nn.MSELoss(); bs = int(max(32, min(2000, len(xt) // 4)))
        for _ in range(50):
            net.train(); perm = torch.randperm(len(xt))
            for s in range(0, len(xt), bs):
                idx = perm[s:s + bs]
                if len(idx) < 2: continue
                opt.zero_grad(); lossf(net(xt[idx]), yt[idx]).backward(); opt.step()
        net.eval(); xb = xv.clone().requires_grad_(True); net(xb)[:, 1].sum().backward()
        np.save(f"{FW}/SA_{N}_{r}.npy", xb.grad.abs().mean(0).numpy()); print("done", N, r, flush=True)
