"""Additional localization baselines (review round 1, F20), on the same split and metric as the FPTM localization runs.
  * DL-LA MLP with epoch selection on the validation part (same rule as FPTM), then three attributions on the test part:
    sensitivity analysis (mean |gradient|), integrated gradients (mean |IG|, baseline = mean training trace, 32 steps),
    and, for ASCAD only, occlusion (drop of the mean correct-class output when one sample is set to its training mean).
  * Random forest (300 trees, sqrt features) on the training part, impurity-based importance.
Usage: python3 loc_baselines.py DATASET EPOCHS. Writes $ROOT/results/locb_<ds>.json"""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np, sys, json, copy
from scipy import stats
import torch, torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

ds, epochs = sys.argv[1], int(sys.argv[2]); torch.set_num_threads(2)
X = np.load(f"{ROOT}/data/{ds}/X.npy", mmap_mode='r'); y = np.load(f"{ROOT}/data/{ds}/y.npy")
sp = np.load(f"{ROOT}/work/{ds}_K3/split.npz"); tr, va, te = np.sort(sp["tr"]), np.sort(sp["va"]), np.sort(sp["te"]); D = X.shape[1]
Xtr, Xva, Xte = (np.asarray(X[i], np.float32) for i in (tr, va, te)); ytr, yva, yte = y[tr], y[va], y[te]
allx = np.concatenate([Xtr, Xva, Xte]); mu, sd = allx.mean(0), allx.std(0) + 1e-8; del allx
T = lambda A: torch.from_numpy((A - mu) / sd)
xt, xva, xv = T(Xtr), T(Xva), T(Xte)
yt = nn.functional.one_hot(torch.from_numpy(ytr.astype(np.int64)), 2).float()
g = np.load(f"{ROOT}/data/{ds}/gt.npz")
if ds == "ascad":
    ref = np.zeros(D, bool)
    for k in ("snr_mask0", "snr_vmask0", "snr_mask15", "snr_vmask15"): ref |= g[k] > 0.1
else:
    ref = np.abs(sp["t_te"]) > stats.t.isf(1e-5 / 2 / D, len(sp["te"]) - 2)
dil = np.zeros(D, bool)
for i in np.where(ref)[0]: dil[max(0, i - 3):i + 4] = True
P = lambda s: {f"p@{k}": float(dil[np.argsort(-s)[:k]].mean()) for k in (10, 20, 50)}
res = {"ds": ds, "chance": float(dil.mean()), "mlp": [], "rf": []}
base = torch.zeros(1, D)                      # mean training trace in standardized units
base[0] = xt.mean(0)
for seed in (1, 2, 3):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(D, 120), nn.ReLU(), nn.BatchNorm1d(120), nn.Linear(120, 90), nn.ReLU(), nn.BatchNorm1d(90),
                        nn.Linear(90, 50), nn.ReLU(), nn.BatchNorm1d(50), nn.Linear(50, 2), nn.Softmax(dim=1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); lossf = nn.MSELoss(); bs = int(max(32, min(2000, len(xt) // 4)))
    best = (-1, None, 0)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(xt))
        for s in range(0, len(xt), bs):
            idx = perm[s:s + bs]
            if len(idx) < 2: continue
            opt.zero_grad(); lossf(net(xt[idx]), yt[idx]).backward(); opt.step()
        net.eval()
        with torch.no_grad(): a = float((net(xva).argmax(1).numpy().astype(bool) == yva).mean())
        if a > best[0]: best = (a, copy.deepcopy(net.state_dict()), ep + 1)
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad(): out = net(xv); acc = float((out.argmax(1).numpy().astype(bool) == yte).mean())
    yt_te = torch.from_numpy(yte.astype(np.int64))
    sa = np.zeros(D); ig = np.zeros(D)
    for s in range(0, len(xv), 500):
        xb = xv[s:s + 500].clone().requires_grad_(True); net(xb)[:, 1].sum().backward(); sa += xb.grad.abs().sum(0).numpy()
        xb0 = xv[s:s + 500]; tot = torch.zeros_like(xb0)
        for a in (np.arange(32) + 0.5) / 32:
            xi = (base + a * (xb0 - base)).clone().requires_grad_(True); net(xi)[:, 1].sum().backward(); tot += xi.grad
        ig += ((xb0 - base) * tot / 32).abs().sum(0).numpy()
    r = {"seed": seed, "best_epoch": best[2], "val_acc": best[0], "test_acc": acc, "sa": P(sa), "ig": P(ig)}
    np.save(f"{ROOT}/work/{ds}_K3/SAsel_s{seed}.npy", sa); np.save(f"{ROOT}/work/{ds}_K3/IG_s{seed}.npy", ig)
    if ds == "ascad":
        with torch.no_grad():
            p0 = out[torch.arange(len(yte)), yt_te].mean().item(); occ = np.zeros(D)
            for t in range(D):
                xo = xv.clone(); xo[:, t] = base[0, t]
                occ[t] = p0 - net(xo)[torch.arange(len(yte)), yt_te].mean().item()
        r["occ"] = P(occ); np.save(f"{ROOT}/work/{ds}_K3/OCC_s{seed}.npy", occ)
    res["mlp"].append(r); print(json.dumps(r), flush=True)
    rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=2, random_state=seed).fit(Xtr, ytr)
    imp = rf.feature_importances_; racc = float((rf.predict(Xte) == yte).mean())
    np.save(f"{ROOT}/work/{ds}_K3/RF_s{seed}.npy", imp)
    res["rf"].append({"seed": seed, "test_acc": racc, **P(imp)}); print(json.dumps(res["rf"][-1]), flush=True)
mean = lambda pat: np.mean([np.load(f"{ROOT}/work/{ds}_K3/{pat}_s{s}.npy") for s in (1, 2, 3)], 0)
res["mean_score"] = {"sa_sel": P(mean("SAsel")), "ig": P(mean("IG")), "rf": P(mean("RF"))}
if ds == "ascad": res["mean_score"]["occ"] = P(mean("OCC"))
print(json.dumps(res["mean_score"])); json.dump(res, open(f"{ROOT}/results/locb_{ds}.json", "w"), indent=1)
