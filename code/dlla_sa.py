"""DL-LA localization baseline: train the DL-LA MLP on the same training split as the FPTM localization runs,
then compute sensitivity analysis (SA) on the test split: mean absolute gradient of the class-1 output with respect to
each (standardized) input sample. Scored with the same precision@k metric as the FPTM score."""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, sys, json
from scipy import stats
import torch, torch.nn as nn
ds, epochs = sys.argv[1], int(sys.argv[2]); torch.set_num_threads(2)
X = np.load(f"{ROOT}/data/{ds}/X.npy", mmap_mode='r'); y = np.load(f"{ROOT}/data/{ds}/y.npy")
sp = np.load(f"{ROOT}/work/{ds}_K3/split.npz"); tr, te = np.sort(sp["tr"]), np.sort(sp["te"]); D = X.shape[1]
Xtr = np.asarray(X[tr], np.float32); Xte = np.asarray(X[te], np.float32); ytr, yte = y[tr], y[te]
allx = np.concatenate([Xtr, Xte]); mu, sd = allx.mean(0), allx.std(0) + 1e-8; del allx
xt = torch.from_numpy((Xtr - mu) / sd); xv = torch.from_numpy((Xte - mu) / sd)
yt = nn.functional.one_hot(torch.from_numpy(ytr.astype(np.int64)), 2).float()
g = np.load(f"{ROOT}/data/{ds}/gt.npz")
if ds == "ascad":
    ref = np.zeros(D, bool)
    for k in ("snr_mask0", "snr_vmask0", "snr_mask15", "snr_vmask15"): ref |= g[k] > 0.1
else:
    ref = np.abs(sp["t_te"]) > stats.t.isf(1e-5 / 2 / D, len(sp["te"]) - 2)
dil = np.zeros(D, bool)
for i in np.where(ref)[0]: dil[max(0, i - 3):i + 4] = True
res = {"ds": ds, "seeds": []}
for seed in (1, 2, 3):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(D, 120), nn.ReLU(), nn.BatchNorm1d(120), nn.Linear(120, 90), nn.ReLU(), nn.BatchNorm1d(90),
                        nn.Linear(90, 50), nn.ReLU(), nn.BatchNorm1d(50), nn.Linear(50, 2), nn.Softmax(dim=1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); lossf = nn.MSELoss(); bs = int(max(32, min(2000, len(xt) // 4)))
    for _ in range(epochs):
        net.train(); perm = torch.randperm(len(xt))
        for s in range(0, len(xt), bs):
            idx = perm[s:s + bs]
            if len(idx) < 2: continue
            opt.zero_grad(); lossf(net(xt[idx]), yt[idx]).backward(); opt.step()
    net.eval()
    with torch.no_grad(): acc = float((net(xv).argmax(1).numpy().astype(bool) == yte).mean())
    sa = np.zeros(D)
    for s in range(0, len(xv), 500):
        xb = xv[s:s + 500].clone().requires_grad_(True); net(xb)[:, 1].sum().backward(); sa += xb.grad.abs().sum(0).numpy()
    sa /= len(xv); np.save(f"{ROOT}/work/{ds}_K3/SA_s{seed}.npy", sa)
    pk = {f"p@{k}": float(dil[np.argsort(-sa)[:k]].mean()) for k in (5, 10, 20, 50)}
    pk0 = {f"p0@{k}": float(ref[np.argsort(-sa)[:k]].mean()) for k in (10, 20)}
    res["seeds"].append({"seed": seed, "acc": acc, **pk, **pk0}); print(json.dumps(res["seeds"][-1]), flush=True)
S = np.mean([np.load(f"{ROOT}/work/{ds}_K3/SA_s{k}.npy") for k in (1, 2, 3)], axis=0)
res["mean_score"] = {**{f"p@{k}": float(dil[np.argsort(-S)[:k]].mean()) for k in (5, 10, 20, 50)},
                     **{f"p0@{k}": float(ref[np.argsort(-S)[:k]].mean()) for k in (10, 20)}, "top20": sorted(int(i) for i in np.argsort(-S)[:20])}
print(json.dumps(res["mean_score"])); json.dump(res, open(f"{ROOT}/results/sa_{ds}.json", "w"), indent=1)
