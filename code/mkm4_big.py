"""mkm4 at larger trace counts (all 10 chunks of the variable-key capture), memory-lean.
Stage 'build': extract, align (same references as the 20k build), store aligned int16 memmap and labels.
Stage 'sweep': for N in Ns and 3 repeats: Welch and chi2 (Bonferroni) blockwise; FPTM inputs (thresholds from train half)."""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, sys, os, tarfile, json, time
sys.path.insert(0, CODE)
from common import align, hw
from baselines import _welch_t_df, _bonf_log10, LOG10
from scipy import stats
TAR = f"{ROOT}/dl/mkm4/var.tar.gz"; M = f"{ROOT}/dl/mkm4/100k_capture_all_2"; OUT = f"{ROOT}/data/mkm4big"; Q = 3329
os.makedirs(OUT, exist_ok=True)

def build():
    need = [f"100k_capture_all_2/traces/traces_s{s}_{i}.npy" for i in range(2, 10) for s in (0, 1)] + \
           [f"100k_capture_all_2/metadata/bp_s{s}_{i}.npy" for i in range(2, 10) for s in (0, 1)]
    with tarfile.open(TAR, "r:gz") as tf:
        for m in tf:
            if m.name in need: tf.extract(m, f"{ROOT}/dl/mkm4", filter="data"); print("extracted", m.name, flush=True)
    r0 = np.asarray(np.load(f"{M}/traces/traces_s0_0.npy", mmap_mode='r')[:2000], np.float32).mean(0)
    r1 = np.asarray(np.load(f"{M}/traces/traces_s1_0.npy", mmap_mode='r')[:2000], np.float32).mean(0)
    X = np.lib.format.open_memmap(f"{OUT}/X.npy", mode="w+", dtype=np.int16, shape=(100000, 19920)); h = []
    for i in range(10):
        a0, _ = align(np.load(f"{M}/traces/traces_s0_{i}.npy"), r0); a1, _ = align(np.load(f"{M}/traces/traces_s1_{i}.npy"), r1)
        X[i * 10000:(i + 1) * 10000] = np.rint(np.concatenate([a0, a1], 1)).astype(np.int16); del a0, a1
        b = (np.load(f"{M}/metadata/bp_s0_{i}.npy")[:, 0].astype(np.int64) + np.load(f"{M}/metadata/bp_s1_{i}.npy")[:, 0].astype(np.int64)) % Q
        h.append(hw(b)); print("aligned chunk", i, flush=True)
    X.flush(); h = np.concatenate(h); np.save(f"{OUT}/y.npy", h > 6); np.save(f"{OUT}/hw.npy", h)
    # remove extracted raw chunks 2..9 to save disk
    for n in need:
        if "traces/" in n: os.remove(f"{ROOT}/dl/mkm4/{n}")

def sweep(Ns, K=3):
    X = np.load(f"{OUT}/X.npy", mmap_mode='r'); y = np.load(f"{OUT}/y.npy"); D = X.shape[1]
    P, Ng = np.where(y)[0], np.where(~y)[0]; FW = f"{ROOT}/work/sweep_mkm4big"; os.makedirs(FW, exist_ok=True)
    nch = -(-K * D // 64); json.dump({"D": D, "K": K, "nbits": K * D, "nchunks": nch}, open(f"{FW}/meta.json", "w"))
    out = open(f"{ROOT}/results/mkm4big_sweep.jsonl", "a")
    for N in Ns:
        for r in range(3):
            t0 = time.time(); rng = np.random.default_rng(10007 * N + r); h = N // 2
            p = rng.choice(P, h, replace=False); n = rng.choice(Ng, h, replace=False)
            tr = np.sort(np.r_[p[:h // 2], n[:h // 2]]); te = np.sort(np.r_[p[h // 2:], n[h // 2:]]); al = np.sort(np.r_[tr, te]); ya = y[al]
            lw = lc = 0.0; Qt = np.empty((K, D), np.float32); qs = np.linspace(0, 1, K + 2)[1:-1]
            for s in range(0, D, 1000):
                B = np.asarray(X[al, s:s + 1000]); Bf = B.astype(np.float64)
                t, df = _welch_t_df(Bf[ya], Bf[~ya]); lw = min(lw, float(np.nanmin(stats.t.logsf(np.abs(t), df) / LOG10 + np.log10(2))))
                n1 = ya.sum(); n0 = len(ya) - n1
                for j in range(B.shape[1]):
                    v = B[:, j].astype(np.int64); v -= v.min(); Kc = v.max() + 1
                    c1 = np.bincount(v[ya], minlength=Kc); c0 = np.bincount(v[~ya], minlength=Kc); tot = c0 + c1; nz = tot > 0
                    c0, c1, tot = c0[nz], c1[nz], tot[nz]
                    if len(tot) < 2: continue
                    e1 = tot * n1 / len(ya); e0 = tot * n0 / len(ya); st = ((c1 - e1) ** 2 / e1 + (c0 - e0) ** 2 / e0).sum()
                    lc = min(lc, float(stats.chi2.logsf(st, len(tot) - 1) / LOG10))
                Qt[:, s:s + 1000] = np.quantile(np.asarray(X[tr, s:s + 1000], np.float32), qs, axis=0)
            rec = {"ds": "mkm4", "N": N, "r": r, "welch": _bonf_log10(lw, D), "chi2": _bonf_log10(lc, D)}
            for part, idx in (("train", tr), ("test", te)):
                with open(f"{FW}/{N}_{r}_{part}_bits.bin", "wb") as fb:
                    for s in range(0, len(idx), 1000):
                        Bt = np.asarray(X[idx[s:s + 1000]], np.float32)
                        bits = (Bt[:, None, :] > Qt[None]).reshape(len(Bt), -1)
                        bits = np.concatenate([bits, np.zeros((len(Bt), nch * 64 - K * D), bool)], 1)
                        fb.write(np.packbits(bits, axis=1, bitorder='little').view('<u8').tobytes())
                y[idx].astype(np.uint8).tofile(f"{FW}/{N}_{r}_{part}_y.bin")
            with open(f"{FW}/list.txt", "a") as fl: fl.write(f"{N} {r} {len(tr)} {len(te)}\n")
            rec["sec"] = round(time.time() - t0, 1); out.write(json.dumps(rec) + "\n"); out.flush(); print(json.dumps(rec), flush=True)

if __name__ == "__main__":
    if sys.argv[1] == "build": build()
    else: sweep([int(v) for v in sys.argv[2].split(",")])
