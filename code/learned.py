"""Shared learned baselines for review round 4: DL-LA MLP with epoch selection on a validation part, random forest,
and linear discriminant analysis. Each returns the binomial log10 p on the test part and the test accuracy."""
import numpy as np, copy
from scipy import stats

LOG10 = np.log(10)


def binom_log10sf(k, n):
    return float(stats.binom.logsf(k - 1, n, 0.5) / LOG10)


def dlla_select(Xtr, ytr, Xva, yva, Xte, yte, seed=1, epochs=50, threads=2):
    import torch, torch.nn as nn
    torch.manual_seed(seed); torch.set_num_threads(threads)
    n_all = len(Xtr) + len(Xva) + len(Xte)   # per-sample standardization over all traces, without a concatenated copy
    mu = sum(A.sum(0, dtype=np.float64) for A in (Xtr, Xva, Xte)) / n_all
    sd = np.sqrt(sum(((A - mu) ** 2).sum(0) for A in (Xtr, Xva, Xte)) / n_all) + 1e-8
    mu, sd = mu.astype(np.float32), sd.astype(np.float32)
    T = lambda A: torch.from_numpy(((A - mu) / sd).astype(np.float32))
    xt, xva, xv = T(Xtr), T(Xva), T(Xte); D = xt.shape[1]
    yt = nn.functional.one_hot(torch.from_numpy(ytr.astype(np.int64)), 2).float()
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
    with torch.no_grad(): k = int((net(xv).argmax(1).numpy().astype(bool) == yte).sum())
    return binom_log10sf(k, len(yte)), k / len(yte), best[2]


def rf_detect(Xtr, ytr, Xte, yte, seed=1, trees=300):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=trees, max_features="sqrt", n_jobs=2, random_state=seed).fit(Xtr, ytr)
    k = int((rf.predict(Xte) == yte).sum())
    return binom_log10sf(k, len(yte)), k / len(yte)


def lda_detect(Xtr, ytr, Xte, yte):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, ytr)
    k = int((m.predict(Xte) == yte).sum())
    return binom_log10sf(k, len(yte)), k / len(yte)
