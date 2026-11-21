import numpy as np, sys
R=sys.argv[1]
def M(p): return np.asarray(np.load(f"{R}/{p}", mmap_mode='r'),np.float32).mean(0)
def lag(a,b,maxl=200):
    a=a-a.mean(); b=b-b.mean(); best=max(range(-maxl,maxl+1),key=lambda l: np.dot(a[max(0,l):len(a)+min(0,l)], b[max(0,-l):len(b)-max(0,l)]))
    return best
ref=M("fixed_key/leakages/traces_0.npy")
for p in ["fixed_key/leakages/traces_1.npy","fixed_key/leakages/traces_19.npy","variable_key/leakages/traces_0.npy","variable_key/leakages/traces_1.npy"]:
    m=M(p); print(p, "lag vs F0 mean:", lag(ref,m), " corr=%.5f"%np.corrcoef(ref,m)[0,1], " rms diff=%.3f"%np.sqrt(((ref-m)**2).mean()))
# per-trace lag distribution within F0 and F19 vs F0 mean, using a window
T0=np.asarray(np.load(f"{R}/fixed_key/leakages/traces_0.npy",mmap_mode='r')[:300],np.float32)
T19=np.asarray(np.load(f"{R}/fixed_key/leakages/traces_19.npy",mmap_mode='r')[:300],np.float32)
for name,T in [("F0",T0),("F19",T19)]:
    ls=[lag(ref,t,50) for t in T]; print(name,"per-trace lags: unique",np.unique(ls,return_counts=True))
