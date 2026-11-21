import numpy as np, sys, json
R=sys.argv[1]; out=sys.argv[2]
def L(p): return np.load(f"{R}/{p}", mmap_mode='r')
def welch(A,B):
    A=np.asarray(A,np.float64); B=np.asarray(B,np.float64)
    ma,mb=A.mean(0),B.mean(0); va,vb=A.var(0,ddof=1),B.var(0,ddof=1)
    return (ma-mb)/np.sqrt(va/len(A)+vb/len(B))
F0=L("fixed_key/leakages/traces_0.npy"); V0=L("variable_key/leakages/traces_0.npy")
res={}
t_fvr=welch(F0,V0)
rng=np.random.default_rng(0); idx=rng.permutation(len(F0))
t_null=welch(F0[np.sort(idx[:5000])],F0[np.sort(idx[5000:])])
t_half=welch(F0[:5000],F0[5000:])
# SNR on secret coefficient a0 (variable key), classes by a0 value mod q
a0=np.asarray(L("variable_key/metadata/mult_a_0.npy")[:,0]).astype(int)
b0=np.asarray(L("variable_key/metadata/mult_b_0.npy")[:,0]).astype(int)
def snr(T,lab):
    T=np.asarray(T,np.float32); u=np.unique(lab)
    means=np.stack([T[lab==k].mean(0) for k in u]); vars_=np.stack([T[lab==k].var(0) for k in u])
    return means.var(0)/vars_.mean(0)
hw=lambda v: np.array([bin(x & 0xFFFF).count("1") for x in v])
snr_a0=snr(V0,hw(a0)); snr_b0=snr(V0,hw(b0)); snr_prod=snr(V0,hw((a0*b0)))
np.savez(out, t_fvr=t_fvr,t_null=t_null,t_half=t_half,snr_a0=snr_a0,snr_b0=snr_b0,snr_prod=snr_prod,
         mean_f=np.asarray(F0,np.float32).mean(0), mean_v=np.asarray(V0,np.float32).mean(0))
for k,v in [("t_fvr",t_fvr),("t_null",t_null),("t_half",t_half)]:
    a=np.abs(v); print(k,"max|t|=%.1f at %d, #|t|>4.5=%d"%(a.max(),a.argmax(),(a>4.5).sum()))
for k,v in [("snr_a0",snr_a0),("snr_b0",snr_b0),("snr_prod",snr_prod)]:
    print(k,"max=%.3f at %d"%(v.max(),v.argmax()))
print("a0 range",a0.min(),a0.max(),"b0 range",b0.min(),b0.max())
