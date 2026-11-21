"""Acquisition drift in the pqm4 recordings (Sec. 5.1). Welch t between chunks and halves of the same campaign and between
the fixed and variable key campaigns, on raw, mean-removed and z-normalized traces. Loads two chunks at a time.
Usage: python3 drift.py <root of pqm4-ref/pqm4>"""
import numpy as np, sys
R = sys.argv[1]
def L(p): return np.asarray(np.load(f"{R}/{p}", mmap_mode="r"), np.float32)
def welch(A, B):
    ma, mb = A.mean(0), B.mean(0); va, vb = A.var(0, ddof=1), B.var(0, ddof=1)
    return (ma - mb) / np.sqrt(va / len(A) + vb / len(B))
def z(T): return (T - T.mean(1, keepdims=True)) / T.std(1, keepdims=True)
def m(T): return T - T.mean(1, keepdims=True)
def rep(name, t): a = np.abs(t); print("%-30s max|t|=%6.1f at %5d  #|t|>4.5=%5d of %d" % (name, a.max(), a.argmax(), (a > 4.5).sum(), len(a)), flush=True)
F = lambda i: f"fixed_key/leakages/traces_{i}.npy"; V = lambda i: f"variable_key/leakages/traces_{i}.npy"
pairs = [("F0 half vs half", F(0), None), ("F0 vs F1 (adjacent chunks)", F(0), F(1)), ("F0 vs F19 (distant chunks)", F(0), F(19)),
         ("V0 vs V1", V(0), V(1)), ("F0 vs V0 (fixed vs variable)", F(0), V(0))]
rng = np.random.default_rng(0)
for name, a, b in pairs:
    A = L(a)
    if b is None:
        for tag, f in [("raw", lambda x: x), ("mean-removed", m), ("z-normalized", z)]:
            T = f(A); rep(f"{tag:13s} {name}", welch(T[:len(T) // 2], T[len(T) // 2:]))
        p = rng.permutation(len(A)); rep(f"{'raw':13s} F0 random split", welch(A[p[:len(A) // 2]], A[p[len(A) // 2:]]))
    else:
        B = L(b)
        for tag, f in [("raw", lambda x: x), ("mean-removed", m), ("z-normalized", z)]:
            rep(f"{tag:13s} {name}", welch(f(A), f(B)))
        del B
    del A
print("DONE", flush=True)
