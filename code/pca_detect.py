"""Detection on a reduced input dimension (review round 4, B3). PCA fitted on the training traces of the localization
split; the first NC components of every trace feed the FPTM (K = 3 threshold bits per component), the epoch-selected
DL-LA, the random forest and linear discriminant analysis (LDA). Usage:
  python3 pca_detect.py prep DS NC   write FPTM inputs to $ROOT/work/DS_pca<NC>
  python3 pca_detect.py base DS NC   baselines on the components (seeds 1 to 3); writes $ROOT/results/pca_DS_<NC>.json
  python3 pca_detect.py raw DS       LDA on the raw samples; writes $ROOT/results/lda_raw_DS.json"""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, sys, json
sys.path.insert(0, CODE)
from common import write_fptm_inputs
from learned import dlla_select, rf_detect, lda_detect
from sklearn.decomposition import PCA

mode, ds = sys.argv[1], sys.argv[2]
X = np.load(f"{ROOT}/data/{ds}/X.npy", mmap_mode="r"); y = np.load(f"{ROOT}/data/{ds}/y.npy").astype(bool)
sp = np.load(f"{ROOT}/work/{ds}_K3/split.npz"); tr, va, te = np.sort(sp["tr"]), np.sort(sp["va"]), np.sort(sp["te"])
Xtr, Xva, Xte = (np.asarray(X[i], np.float32) for i in (tr, va, te))
if mode == "raw":
    lp, acc = lda_detect(Xtr, y[tr], Xte, y[te]); r = {"ds": ds, "lda_raw": {"log10p": lp, "acc": acc}}
    print(r); json.dump(r, open(f"{ROOT}/results/lda_raw_{ds}.json", "w"), indent=1); sys.exit()
nc = int(sys.argv[3])
pca = PCA(n_components=nc, svd_solver="randomized", random_state=1).fit(Xtr)
Z = {k: pca.transform(v).astype(np.float32) for k, v in (("tr", Xtr), ("va", Xva), ("te", Xte))}
if mode == "prep":
    Zall = np.zeros((len(y), nc), np.float32); Zall[tr], Zall[va], Zall[te] = Z["tr"], Z["va"], Z["te"]
    write_fptm_inputs(f"{ROOT}/work/{ds}_pca{nc}", Zall, y, sp["tr"], sp["va"], sp["te"], 3); print("prep", ds, nc)
else:
    res = {"ds": ds, "nc": nc, "explained": float(pca.explained_variance_ratio_.sum())}
    lp, acc = lda_detect(Z["tr"], y[tr], Z["te"], y[te]); res["lda"] = {"log10p": lp, "acc": acc}
    for seed in (1, 2, 3):
        lp, acc, ep = dlla_select(Z["tr"], y[tr], Z["va"], y[va], Z["te"], y[te], seed=seed, epochs=50)
        res.setdefault("dlla_sel", []).append({"log10p": lp, "acc": acc, "epoch": ep})
        lp, acc = rf_detect(Z["tr"], y[tr], Z["te"], y[te], seed=seed)
        res.setdefault("rf", []).append({"log10p": lp, "acc": acc})
    print(json.dumps(res, indent=1)); json.dump(res, open(f"{ROOT}/results/pca_{ds}_{nc}.json", "w"), indent=1)
