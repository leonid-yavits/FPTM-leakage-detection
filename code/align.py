import numpy as np, sys
R=sys.argv[1]; OUT=sys.argv[2]; MAXL=20
def load(p): return np.asarray(np.load(f"{R}/{p}", mmap_mode='r'),np.float32)
ref=load("fixed_key/leakages/traces_0.npy")[:2000].mean(0)
D=len(ref); W=slice(MAXL,D-MAXL)
def align(T):
    # FFT cross-correlation per trace, integer lag in [-MAXL,MAXL]
    n=1<<int(np.ceil(np.log2(2*D)))
    Rf=np.conj(np.fft.rfft(ref-ref.mean(),n))
    out=np.empty((len(T),D-2*MAXL),np.float32); lags=np.empty(len(T),int)
    for s in range(0,len(T),500):
        B=T[s:s+500]; X=np.fft.rfft(B-B.mean(1,keepdims=True),n,axis=1)
        cc=np.fft.irfft(X*Rf,n,axis=1)
        c=np.concatenate([cc[:,-MAXL:],cc[:,:MAXL+1]],1)  # lags -MAXL..MAXL
        l=c.argmax(1)-MAXL; lags[s:s+500]=l
        for i,li in enumerate(l): out[s+i]=B[i,MAXL+li:D-MAXL+li]
    return out,lags
names={"F0":"fixed_key/leakages/traces_0.npy","F1":"fixed_key/leakages/traces_1.npy","F19":"fixed_key/leakages/traces_19.npy",
       "V0":"variable_key/leakages/traces_0.npy","V1":"variable_key/leakages/traces_1.npy"}
A={}
for k,p in names.items():
    A[k],l=align(load(p)); print(k,"lag mean %.2f std %.2f, |lag|=MAXL: %d"%(l.mean(),l.std(),(np.abs(l)==MAXL).sum()))
    np.save(f"{OUT}/{k}_aligned.npy",A[k])
def welch(A,B):
    return (A.mean(0)-B.mean(0))/np.sqrt(A.var(0,ddof=1)/len(A)+B.var(0,ddof=1)/len(B))
def rep(name,t): a=np.abs(t); print("%-28s max|t|=%6.1f at %5d  #|t|>4.5=%5d"%(name,a.max(),a.argmax(),(a>4.5).sum()))
F0=A["F0"]
rep("F0 half vs half",welch(F0[:5000],F0[5000:]))
rep("F0 vs F1",welch(F0,A["F1"])); rep("F0 vs F19",welch(F0,A["F19"])); rep("V0 vs V1",welch(A["V0"],A["V1"]))
rep("F0 vs V0 (FvR)",welch(F0,A["V0"])); rep("F1 vs V1 (FvR)",welch(A["F1"],A["V1"]))
print("mean corr F0/F19 %.4f F0/V0 %.4f"%(np.corrcoef(F0.mean(0),A["F19"].mean(0))[0,1],np.corrcoef(F0.mean(0),A["V0"].mean(0))[0,1]))
