"""Simulated GainP power traces of the ML-KEM modular product kernel, fixed key versus random key.

One trace = one kernel run over 768 columns (the 3 x 256 key coefficients of ML-KEM-768), all in lockstep.
The trace sample of a cycle is the energy of that cycle, summed over the 768 columns.
Classes alternate trace by trace: even traces use one fixed key, odd traces a fresh random key.
The ciphertext coefficients are random in every trace.

  build N      simulate N noiseless traces, write $ROOT/work/gainp/{mis.npy, kind.npy, y.npy}
  make SIGMA   add Gaussian noise (SIGMA in fJ per cycle), quantize to 8 bits, write $ROOT/data/gainp_s<SIGMA>/
"""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, sys, json, os, time
sys.path.insert(0, CODE)
from gainp_mlkem import kernel, Q
from gainp_sim import E_CMP_MIS, E_CMP_MATCH, E_WRITE

NCOL = 768
W = f"{ROOT}/work/gainp"


def build(N, batch=500):
    os.makedirs(W, exist_ok=True)
    rng = np.random.default_rng(2026)
    a_fix = rng.integers(0, Q, NCOL)
    y = (np.arange(N) % 2).astype(bool)                       # 0: fixed key, 1: random key
    mis = None; t0 = time.time()
    for s in range(0, N, batch):
        n = min(batch, N - s)
        a = np.where(y[s:s + n, None], rng.integers(0, Q, (n, NCOL)), a_fix[None, :])
        b = rng.integers(0, Q, (n, NCOL))
        g = kernel(a.ravel(), b.ravel(), cols_per_group=NCOL)
        m = g.mismatch_matrix()
        if mis is None:
            mis = np.lib.format.open_memmap(f"{W}/mis.npy", "w+", np.int16, (N, m.shape[1]))
            np.save(f"{W}/kind.npy", np.array([k == 'C' for k in g.kind]))
        mis[s:s + n] = m
        print(f"{s + n}/{N} {time.time() - t0:.0f}s", flush=True)
    mis.flush(); np.save(f"{W}/y.npy", y); np.save(f"{W}/a_fix.npy", a_fix)


def energy_fJ(mis, kind):
    e = np.where(kind[None, :], mis * (E_CMP_MIS * 1e15) + (NCOL - mis) * (E_CMP_MATCH * 1e15), NCOL * E_WRITE * 1e15)
    return e.astype(np.float32)


def make(sigma, seed=7):
    mis = np.load(f"{W}/mis.npy", mmap_mode='r'); kind = np.load(f"{W}/kind.npy"); y = np.load(f"{W}/y.npy")
    N, D = mis.shape; out = f"{ROOT}/data/gainp_s{sigma:g}"; os.makedirs(out, exist_ok=True)
    X = np.lib.format.open_memmap(f"{out}/X.npy", "w+", np.uint8, (N, D))
    rng = np.random.default_rng(seed)
    e0 = energy_fJ(np.asarray(mis[:1000]), kind)
    lo = float((e0 - 4 * sigma).min()); hi = float((e0 + 4 * sigma).max())   # one 8-bit range, set on 1,000 traces
    span = hi - lo
    for s in range(0, N, 2000):
        e = energy_fJ(np.asarray(mis[s:s + 2000]), kind) + rng.normal(0, sigma, (min(2000, N - s), D)).astype(np.float32)
        X[s:s + 2000] = np.clip(np.round((e - lo) / span * 255), 0, 255).astype(np.uint8)
    X.flush(); np.save(f"{out}/y.npy", y)
    json.dump({"impl": "GainP model, ML-KEM modular product, 768 columns", "label": "random key (1) vs fixed key (0)",
               "sigma_fJ": sigma, "adc_bits": 8}, open(f"{out}/info.json", "w"), indent=1)
    print("wrote", out, X.shape)


if __name__ == "__main__":
    if sys.argv[1] == "build": build(int(sys.argv[2]))
    else: make(float(sys.argv[2]))
