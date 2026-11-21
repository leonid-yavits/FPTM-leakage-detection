import numpy as np, sys
W=sys.argv[1]; tags=sys.argv[2:]
g=np.load(f"{W}/ground.npz"); t=np.abs(g["t_te"]); s=g["snr"]; D=len(t)
from scipy.stats import spearmanr
tt=np.argsort(-t)[:20]; print("t-test top20:",np.sort(tt))
for tag in tags:
    a=np.fromfile(f"{W}/S_{tag}.bin",np.float64); Si,Sv=a[:D],a[D:]
    for nm,S in [("S_inc",Si),("S_vote",Sv)]:
        top=np.argsort(-S)[:20]
        # peak-window hit: fraction of FPTM top-k within ±3 samples of any t-test sample with |t|>4.5
        lk=np.where(t>4.5)[0]; near=lambda i: np.any(np.abs(lk-i)<=3) if len(lk) else False
        hit=np.mean([near(i) for i in top[:10]])
        print(f"{tag:12s} {nm}: top10={np.sort(top[:10])} hit@10(±3 of |t|>4.5)={hit:.1f} "
              f"rank of t-peak {t.argmax()}={int((S>S[t.argmax()]).sum())+1} spearman(S,|t|)={spearmanr(S,t)[0]:.3f} max/median={S.max()/max(np.median(S[S>0]),1e-9):.1f}")
