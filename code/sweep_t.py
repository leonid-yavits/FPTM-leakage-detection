import numpy as np, sys
from scipy.stats import t as tdist
W=sys.argv[1]; g=np.load(f"{W}/ground.npz"); X=np.load(f"{W}/X_pool.npy",mmap_mode='r'); y=g["y"]; tr,te=g["tr"],g["te"]; D=X.shape[1]
for line in open(f"{W}/sweep.txt"):
    N,r,h,m=map(int,line.split()); idx=np.fromfile(f"{W}/sw_{N}_{r}.idx",np.int32)
    rows=np.r_[tr[idx[:h]],te[idx[h:]]]; A=np.asarray(X[rows],np.float64); lab=y[rows]
    P,Q=A[lab],A[~lab]; t=(P.mean(0)-Q.mean(0))/np.sqrt(P.var(0,ddof=1)/len(P)+Q.var(0,ddof=1)/len(Q))
    thr_b=tdist.isf(1e-5/2/D, len(A)-2)   # Bonferroni, family-wise 1e-5, two-sided
    a=np.abs(t); print(f"TT N={N} r={r} max|t|={a.max():.2f} at {a.argmax()} tvla={int(a.max()>4.5)} bonf(thr {thr_b:.2f})={int(a.max()>thr_b)}")
