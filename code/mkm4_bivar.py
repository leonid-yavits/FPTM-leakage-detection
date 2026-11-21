import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, sys
sys.path.insert(0,CODE); from common import welch, hw
D=f"{ROOT}/data/mkm4"; X=np.load(f"{D}/X.npy",mmap_mode='r'); y=np.load(f"{D}/y.npy"); g=np.load(f"{D}/gt.npz")
D0=9960; s0=g["snr_s0"]; s1=g["snr_s1"]
p0=np.argsort(-s0[:D0])[:10]; p1=D0+np.argsort(-s1[D0:])[:10]
print("top SNR s0 in win0:",p0, np.round(s0[p0],3)); print("top SNR s1 in win1:",p1-D0, np.round(s1[p1],3))
print("SNR(HW b0) max:", g["snr_b"].max().round(4))
Xp=np.asarray(X[:, np.r_[p0,p1]],np.float64); Xc=Xp-Xp.mean(0)
best=[]
for i in range(10):
    for j in range(10,20):
        cp=Xc[:,i]*Xc[:,j]; t=welch(cp[y][:,None],cp[~y][:,None])[0]; best.append((abs(t),p0[i],p1[j-10]-D0))
best.sort(reverse=True); print("bivariate centered-product |t| top5 (all 20000 traces):",[(round(a,2),b,c) for a,b,c in best[:5]])
# alternative label: b0 itself in halves
b0=g["b0"]; y2=b0>=1665
best2=[]
for i in range(10):
    for j in range(10,20):
        cp=Xc[:,i]*Xc[:,j]; t=welch(cp[y2][:,None],cp[~y2][:,None])[0]; best2.append((abs(t),p0[i],p1[j-10]-D0))
best2.sort(reverse=True); print("label b0>=q/2, top5:",[(round(a,2),b,c) for a,b,c in best2[:5]])
