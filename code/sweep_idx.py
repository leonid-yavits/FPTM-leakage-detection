import numpy as np, sys
W=sys.argv[1]; g=np.load(f"{W}/ground.npz"); ntr=len(g["tr"]); nte=len(g["te"])
Ns=[200,400,800,1600,3200,6292]; R=3
rows=[]
for N in Ns:
    for r in range(R):
        rng=np.random.default_rng(100*N+r); h=N//2
        a=np.sort(rng.choice(ntr,h,replace=False)); b=np.sort(rng.choice(nte,min(h,nte),replace=False))
        np.r_[a,b].astype(np.int32).tofile(f"{W}/sw_{N}_{r}.idx"); rows.append((N,r,h,len(b)))
open(f"{W}/sweep.txt","w").write("\n".join(f"{N} {r} {h} {m}" for N,r,h,m in rows))
