import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, sys, time
sys.path.insert(0,CODE); sys.path.insert(1,f"{ROOT}/src/mvld/Code")
from baselines import *
from baselines import _dist_sum
from scipy.stats import gamma as GM
import RP_dcor
X=np.load(f"{ROOT}/data/pqm4/X.npy",mmap_mode='r'); y=np.load(f"{ROOT}/data/pqm4/y.npy")
rng=np.random.default_rng(3); pos=rng.choice(np.where(y)[0],1000,replace=False); neg=rng.choice(np.where(~y)[0],1000,replace=False)
idx=np.r_[pos,neg]; Xs=np.asarray(X[idx],np.float64); ys=y[idx]
# 1) artifact MV-dcov vs port on a small low-dim slice (statistic and cutoff are random-projection estimates; compare decisions and magnitudes)
sl=Xs[:, 140:180]
np.random.seed(0); st,cut=RP_dcor.u_dist_cov_sqr_mv_test(sl, ys.astype(float), n_projs=100)
print("artifact: stat %.4g cutoff(0.05) %.4g detect=%s"%(st,cut,st>cut))
lp,sc=mv_dcov(sl,ys); print("port: log10p %.2f stat %.4g cutoff(0.05) %.4g"%(lp,*sc))
x=np.random.default_rng(1).standard_normal(500); print("dist_sum check", np.isclose(RP_dcor.dist_sum(x), sum(abs(x[i]-x[j]) for i in range(500) for j in range(i+1,500))), np.isclose(_dist_sum(x), RP_dcor.dist_sum(x)))
# 2) all tests, real vs null labels, full width
yn=np.random.default_rng(5).permutation(ys)
for name,lab in [("real",ys),("null",yn)]:
    t0=time.time(); a=welch_bonf(Xs,lab)[0]; b=chi2_bonf(Xs,lab)[0]; c=mv_dcov(Xs,lab)[0]
    h=len(lab)//2; tr=np.r_[np.arange(0,500),np.arange(1000,1500)]; te=np.r_[np.arange(500,1000),np.arange(1500,2000)]
    d=dlla_mlp(Xs[tr],lab[tr],Xs[te],lab[te],epochs=30)
    print(f"{name}: welch {a:.1f} chi2 {b:.1f} mvdcov {c:.1f} mlp {d[0]:.1f} (acc {d[1]:.3f})  [{time.time()-t0:.0f}s]")
