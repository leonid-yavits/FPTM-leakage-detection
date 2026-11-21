"""Build pooled, aligned trace matrices and partition labels for each public trace set.
Writes $FPTM_ROOT/data/<ds>/{X.npy, y.npy, info.json, gt.npz}."""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, h5py, json, os, sys
sys.path.insert(0, CODE)
from common import align, hw, snr, welch

L = f"{ROOT}/dl/ledger/d0nj0n_mlkem_dataset/pqm4-ref"
M = f"{ROOT}/dl/mkm4/100k_capture_all_2"
A = f"{ROOT}/dl/ascad/ASCAD_data/ASCAD_databases/ASCAD.h5"
OUT = f"{ROOT}/data"; Q = 3329

def save(ds, X, y, info, gt):
    d = f"{OUT}/{ds}"; os.makedirs(d, exist_ok=True)
    np.save(f"{d}/X.npy", X.astype(np.float32)); np.save(f"{d}/y.npy", y)
    np.savez(f"{d}/gt.npz", **gt); json.dump(info, open(f"{d}/info.json", "w"), indent=1)
    print(ds, X.shape, "pos frac %.3f" % y.mean(), info)

def ledger_unmasked(impl, ds):
    T = [np.load(f"{L}/{impl}/variable_key/leakages/traces_{i}.npy") for i in (0, 1)]
    ref = np.asarray(T[0][:2000], np.float32).mean(0)
    X = np.concatenate([align(t, ref)[0] for t in T])
    if impl == "pqm4":
        a0 = np.concatenate([np.load(f"{L}/{impl}/variable_key/metadata/mult_a_{i}.npy")[:, 0] for i in (0, 1)]).astype(np.int64)
    else:  # ref metadata stored as little-endian bytes of int16 coefficients
        a0 = np.concatenate([np.load(f"{L}/{impl}/variable_key/metadata/mult_a_{i}.npy").view('<i2')[:, 0] for i in (0, 1)]).astype(np.int64)
    h = hw(a0 % Q); med = np.median(h); y = h > med
    save(ds, X, y, {"impl": impl, "label": f"HW(a0 mod q) > {med:g}", "secret": "a0", "chunks": [0, 1]},
         {"hw": h, "a0": a0, "snr_hw": snr(X, h)})

def mkm4():
    S0 = [np.load(f"{M}/traces/traces_s0_{i}.npy") for i in (0, 1)]
    S1 = [np.load(f"{M}/traces/traces_s1_{i}.npy") for i in (0, 1)]
    r0 = np.asarray(S0[0][:2000], np.float32).mean(0); r1 = np.asarray(S1[0][:2000], np.float32).mean(0)
    X0 = np.concatenate([align(t, r0)[0] for t in S0]); X1 = np.concatenate([align(t, r1)[0] for t in S1])
    X = np.concatenate([X0, X1], 1)
    s0 = np.concatenate([np.load(f"{M}/metadata/bp_s0_{i}.npy")[:, 0] for i in (0, 1)]).astype(np.int64)
    s1 = np.concatenate([np.load(f"{M}/metadata/bp_s1_{i}.npy")[:, 0] for i in (0, 1)]).astype(np.int64)
    b0 = (s0 + s1) % Q; h = hw(b0); med = np.median(h); y = h > med
    save("mkm4", X, y, {"impl": "mkm4", "label": f"HW(b0) > {med:g}, b0 = (s0+s1) mod q", "secret": "b0",
                        "D_share0": int(X0.shape[1]), "chunks": [0, 1]},
         {"hw": h, "b0": b0, "s0": s0, "s1": s1, "snr_s0": snr(X, hw(s0 % (1 << 16))), "snr_s1": snr(X, hw(s1 % (1 << 16))), "snr_b": snr(X, h)})

SBOX = np.array([0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16])

def ascad():
    with h5py.File(A) as h:
        X = np.concatenate([h["Profiling_traces/traces"][:], h["Attack_traces/traces"][:]]).astype(np.float32)
        md = np.concatenate([h["Profiling_traces/metadata"][:], h["Attack_traces/metadata"][:]])
        lab = np.concatenate([h["Profiling_traces/labels"][:], h["Attack_traces/labels"][:]])
    v = SBOX[md["plaintext"][:, 2] ^ md["key"][:, 2]]
    assert np.all(v == lab), "label mismatch with Sbox(p[2]^k[2])"
    h8 = hw(v); y = h8 > 4
    gt = {"hw": h8, "v": v}
    # locate share leakage: SNR on HW of each mask byte and of v xor each mask byte
    best = []
    for j in range(16):
        s_m = snr(X, hw(md["masks"][:, j])); s_vm = snr(X, hw(v ^ md["masks"][:, j]))
        gt[f"snr_mask{j}"] = s_m; gt[f"snr_vmask{j}"] = s_vm; best.append((j, s_m.max(), s_vm.max()))
    gt["snr_v"] = snr(X, h8)
    print("ASCAD per-mask-byte max SNR (mask, v^mask):", [(j, round(a, 3), round(b, 3)) for j, a, b in best])
    save("ascad", X, y, {"impl": "ASCAD v1 fixed key, profiling+attack sets", "label": "HW(Sbox(p[2] xor k[2])) > 4", "secret": "Sbox output byte 2"}, gt)

if __name__ == "__main__":
    for ds in sys.argv[1:]:
        {"pqm4": lambda: ledger_unmasked("pqm4", "pqm4"), "ref": lambda: ledger_unmasked("ref", "ref"), "mkm4": mkm4, "ascad": ascad}[ds]()
