"""SNR of the simulated GainP traces per noise level (review round 4, E15). Class SNR per cycle on noiseless energies:
Var over the two classes of the class mean / (mean within-class variance + sigma^2). Quantization ignored.
Reports the maximum over cycles and the median over leaking cycles (ground truth). Writes $ROOT/results/gainp_snr.json"""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, sys, json
sys.path.insert(0, CODE)
from gainp_traces import energy_fJ, W
mis = np.load(f"{W}/mis.npy", mmap_mode="r"); kind = np.load(f"{W}/kind.npy"); y = np.load(f"{W}/y.npy")
lk = np.load(f"{ROOT}/data/gainp_s1000/gt.npz")["leak"].astype(bool)
n = 20000; E = energy_fJ(np.asarray(mis[:n]), kind).astype(np.float64); yy = y[:n]
m1, m0 = E[yy].mean(0), E[~yy].mean(0); v1, v0 = E[yy].var(0), E[~yy].var(0)
sig = ((m1 - m0) / 2) ** 2; within = (v1 + v0) / 2

print("leaking cycles in gt:", int(lk.sum()))
out = {}
for s in (250, 500, 1000, 2000):
    snr = sig / (within + s ** 2)
    out[str(s)] = {"max": float(snr.max()), "median_leaking": float(np.median(snr[lk]))}
    print(s, out[str(s)])
json.dump(out, open(f"{ROOT}/results/gainp_snr.json", "w"), indent=1)
