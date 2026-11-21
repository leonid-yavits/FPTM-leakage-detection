"""Repeated sweep for two further learned detectors (review round 2): a random forest (200 trees, sqrt features), and,
with --mlp-es, the DL-LA network with the epoch selected on a validation part (20% of the training half, at most EPOCHS
epochs). Same subsets as sweep_rep.py (seed 10007 N + r, training half / test half). Binomial test on the test half.
Usage: python3 rf_sweep.py DS N1,N2,... R0:R1 [--mlp-es EPOCHS]. Writes $ROOT/results/<ds>_rep_rf.jsonl"""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, sys, json, time, copy
sys.path.insert(0, CODE)
from baselines import binom_log10sf
from sklearn.ensemble import RandomForestClassifier

ds = sys.argv[1]; Ns = [int(v) for v in sys.argv[2].split(",")]; R0, R1 = [int(v) for v in sys.argv[3].split(":")]
es = int(sys.argv[sys.argv.index("--mlp-es") + 1]) if "--mlp-es" in sys.argv else 0
X = np.load(f"{ROOT}/data/{ds}/X.npy", mmap_mode="r"); y = np.load(f"{ROOT}/data/{ds}/y.npy")
P, Ng = np.where(y)[0], np.where(~y)[0]
out = open(f"{ROOT}/results/{ds}_rep_rf.jsonl", "a")

def mlp_es(Xtr, ytr, Xte, yte, epochs, seed):
    import torch, torch.nn as nn
    torch.manual_seed(seed); torch.set_num_threads(1)
    allx = np.concatenate([Xtr, Xte]); mu, sd = allx.mean(0), allx.std(0) + 1e-8
    nv = len(Xtr) // 5; perm = np.random.default_rng(seed).permutation(len(Xtr)); iv, it = perm[:nv], perm[nv:]
    f = lambda A: torch.from_numpy(((A - mu) / sd).astype(np.float32))
    xt, xv, xe = f(Xtr[it]), f(Xtr[iv]), f(Xte); yt = nn.functional.one_hot(torch.from_numpy(ytr[it].astype(np.int64)), 2).float()
    D = xt.shape[1]
    net = nn.Sequential(nn.Linear(D, 120), nn.ReLU(), nn.BatchNorm1d(120), nn.Linear(120, 90), nn.ReLU(), nn.BatchNorm1d(90),
                        nn.Linear(90, 50), nn.ReLU(), nn.BatchNorm1d(50), nn.Linear(50, 2), nn.Softmax(dim=1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); lossf = nn.MSELoss(); bs = int(max(32, min(2000, len(xt) // 4)))
    best = (-1, None, 0)
    for ep in range(epochs):
        net.train(); pr = torch.randperm(len(xt))
        for s in range(0, len(xt), bs):
            idx = pr[s:s + bs]
            if len(idx) < 2: continue
            opt.zero_grad(); lossf(net(xt[idx]), yt[idx]).backward(); opt.step()
        net.eval()
        with torch.no_grad(): a = float((net(xv).argmax(1).numpy().astype(bool) == ytr[iv]).mean())
        if a > best[0]: best = (a, copy.deepcopy(net.state_dict()), ep + 1)
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad(): k = int((net(xe).argmax(1).numpy().astype(bool) == yte).sum())
    return binom_log10sf(k, len(yte)), k / len(yte), best[2]

for N in Ns:
    for r in range(R0, R1):
        rng = np.random.default_rng(10007 * N + r); h = N // 2
        p = rng.choice(P, h, replace=False); n = rng.choice(Ng, h, replace=False)
        tr = np.r_[p[:h // 2], n[:h // 2]]; te = np.r_[p[h // 2:], n[h // 2:]]
        idx = np.r_[tr, te]; order = np.argsort(idx); sidx = idx[order]
        Xa = np.empty((len(idx), X.shape[1]), np.float32); Xa[order] = X[sidx]; ya = y[idx]; ntr = len(tr)
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=200, max_features="sqrt", n_jobs=1, random_state=r).fit(Xa[:ntr], ya[:ntr])
        k = int((rf.predict(Xa[ntr:]) == ya[ntr:]).sum())
        rec = {"ds": ds, "N": N, "r": r, "rf": binom_log10sf(k, len(te)), "rf_acc": k / len(te),
               "rf_nodes": int(sum(e.tree_.node_count for e in rf.estimators_))}
        if es:
            lp, acc, ep = mlp_es(Xa[:ntr], ya[:ntr], Xa[ntr:], ya[ntr:], es, r); rec.update({"mlp_es": lp, "mlp_es_acc": acc, "mlp_es_epoch": ep})
        rec["sec"] = round(time.time() - t0, 1); out.write(json.dumps(rec) + "\n"); out.flush(); print(json.dumps(rec), flush=True)
