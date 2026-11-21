import numpy as np, sys, json
R=sys.argv[1]; W=sys.argv[2]; K=int(sys.argv[3]); MAXL=20
X=np.concatenate([np.load(f"{W}/V0_aligned.npy"),np.load(f"{W}/V1_aligned.npy")])
a0=np.concatenate([np.load(f"{R}/variable_key/metadata/mult_a_{i}.npy")[:,0] for i in (0,1)]).astype(np.int64)
hw=np.array([bin(int(v)).count("1") for v in a0]); med=np.median(hw)
y=(hw>med)  # specific partition, interleaved within one campaign
rng=np.random.default_rng(1)
pos=rng.permutation(np.where(y)[0]); neg=rng.permutation(np.where(~y)[0]); m=len(pos)
neg=neg[:m]  # balance classes: chance level is exactly 50%
def cut(a,f0,f1): n=len(a); return a[int(f0*n):int(f1*n)]
tr=rng.permutation(np.r_[cut(pos,0,.5),cut(neg,0,.5)]); va=rng.permutation(np.r_[cut(pos,.5,.75),cut(neg,.5,.75)]); te=rng.permutation(np.r_[cut(pos,.75,1),cut(neg,.75,1)])
def welch(A,B): return (A.mean(0)-B.mean(0))/np.sqrt(A.var(0,ddof=1)/len(A)+B.var(0,ddof=1)/len(B))
t_tr=welch(X[tr][y[tr]],X[tr][~y[tr]]); t_te=welch(X[te][y[te]],X[te][~y[te]])
def snr(T,lab):
    u=np.unique(lab); m=np.stack([T[lab==k].mean(0) for k in u]); v=np.stack([T[lab==k].var(0) for k in u]); return m.var(0)/v.mean(0)
s=snr(X[tr],hw[tr])
print("label balance %.3f, median HW %g; balanced split sizes"%(y.mean(),med),len(tr),len(va),len(te),"pos frac",y[tr].mean(),y[va].mean(),y[te].mean())
for n,t in [("t train",t_tr),("t test",t_te)]:
    a=np.abs(t); print(n,"max|t|=%.1f at %d, #>4.5=%d"%(a.max(),a.argmax(),(a>4.5).sum()))
top=np.argsort(-s)[:10]; print("SNR(HW a0) top samples",top, np.round(s[top],3))
# thermometer code: per-sample quantile thresholds from TRAIN only
qs=np.linspace(0,1,K+2)[1:-1]
Q=np.quantile(X[tr],qs,axis=0).astype(np.float32)   # (K,D)
D=X.shape[1]; nbits=K*D; nchunks=-(-nbits//64)
def pack(Xs):
    B=(Xs[:,None,:]>Q[None,:,:]).reshape(len(Xs),nbits)       # threshold-major, matches booleanize
    pad=np.zeros((len(Xs),nchunks*64-nbits),bool); B=np.concatenate([B,pad],1)
    return np.packbits(B,axis=1,bitorder='little').view('<u8')
for name,idx in [("train",tr),("val",va),("test",te)]:
    P=pack(X[idx]); P.tofile(f"{W}/{name}_bits.bin"); y[idx].astype(np.uint8).tofile(f"{W}/{name}_y.bin")
    # null labels: random permutation of labels within the split
    rng.permutation(y[idx]).astype(np.uint8).tofile(f"{W}/{name}_ynull.bin")
json.dump({"D":int(D),"K":K,"nbits":int(nbits),"nchunks":int(nchunks),"n":{"train":len(tr),"val":len(va),"test":len(te)}},open(f"{W}/meta.json","w"))
np.savez(f"{W}/ground.npz",t_tr=t_tr,t_te=t_te,snr=s,Q=Q,tr=tr,va=va,te=te,y=y)
np.save(f"{W}/X_pool.npy",X)
