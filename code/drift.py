import numpy as np, sys
R=sys.argv[1]
def L(p): return np.asarray(np.load(f"{R}/{p}", mmap_mode='r'),np.float32)
def welch(A,B):
    ma,mb=A.mean(0),B.mean(0); va,vb=A.var(0,ddof=1),B.var(0,ddof=1)
    return (ma-mb)/np.sqrt(va/len(A)+vb/len(B))
def z(T): return (T-T.mean(1,keepdims=True))/T.std(1,keepdims=True)
def m(T): return T-T.mean(1,keepdims=True)
F0=L("fixed_key/leakages/traces_0.npy"); F1=L("fixed_key/leakages/traces_1.npy"); F19=L("fixed_key/leakages/traces_19.npy")
V0=L("variable_key/leakages/traces_0.npy"); V1=L("variable_key/leakages/traces_1.npy")
print("per-trace mean F0 first/last 1000: %.2f %.2f | F19: %.2f | V0: %.2f %.2f"%(F0[:1000].mean(),F0[-1000:].mean(),F19.mean(),V0[:1000].mean(),V0[-1000:].mean()))
print("per-trace std  F0 %.2f F19 %.2f V0 %.2f"%(F0.std(1).mean(),F19.std(1).mean(),V0.std(1).mean()))
def rep(name,t): a=np.abs(t); print("%-28s max|t|=%6.1f at %5d  #|t|>4.5=%5d"%(name,a.max(),a.argmax(),(a>4.5).sum()))
for tag,f in [("raw",lambda x:x),("mean-removed",m),("z-normalized",z)]:
    print("==",tag)
    rep("F0 half vs half",welch(f(F0[:5000]),f(F0[5000:])))
    rep("F0 vs F1 (adjacent chunks)",welch(f(F0),f(F1)))
    rep("F0 vs F19 (distant chunks)",welch(f(F0),f(F19)))
    rep("V0 vs V1",welch(f(V0),f(V1)))
    rep("F0 vs V0 (FvR)",welch(f(F0),f(V0)))
