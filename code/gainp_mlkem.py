"""ML-KEM modular product c = a*b mod q (q = 3329) on the GainP model, one coefficient pair per column.

a: decapsulation key coefficient (12 bits, resident in the array before the kernel starts).
b: ciphertext coefficient (12 bits, written at the start of the kernel).
Multiplication: 12 shift-and-add steps p += a_i * (b << i), bit serial, with a ripple carry.
Reduction: restoring division by q << k for k = 11 .. 0: t = p - (q << k), and p = t where no borrow.
All columns run the same instruction sequence, so the cycle count does not depend on the data.
"""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gainp_sim import GainP

Q = 3329
A, B, P, T = 0, 12, 24, 48          # first row of each field: a[12], b[12], p[24], t[24]
C0, C1 = 72, 73                     # carry rows (alternating), borrow uses C0
NROWS = 74


def bits(v, n):
    return ((v[None, :] >> np.arange(n)[:, None]) & 1).astype(bool)


def kernel(a, b, cols_per_group=1, log=True):
    a = np.asarray(a, np.int64); b = np.asarray(b, np.int64)
    g = GainP(NROWS, len(a), cols_per_group, log=log)
    g.S[A:A + 12] = bits(a, 12)                                   # resident key, not part of the trace
    for i in range(12):                                           # write ciphertext coefficient
        g.write_row(B + i, bits(b, 12)[i])
    zero = np.zeros(len(a), bool)
    for i in range(24):                                           # clear product field
        g.write_row(P + i, zero)
    # multiplication
    for i in range(12):
        c, nc = C0, C1
        g.write_row(c, zero)
        ai = A + i
        for j in range(12):
            bj, pr = B + j, P + i + j
            # carry out = maj(a_i & b_j, p, c), computed before p is overwritten
            g.func(nc, [[(pr, 1), (c, 1)], [(ai, 1), (bj, 1), (pr, 1)], [(ai, 1), (bj, 1), (c, 1)]])
            # sum = p xor c xor (a_i & b_j)
            g.func(pr, [[(ai, 0), (pr, 1), (c, 0)], [(ai, 0), (pr, 0), (c, 1)],
                        [(bj, 0), (pr, 1), (c, 0)], [(bj, 0), (pr, 0), (c, 1)],
                        [(ai, 1), (bj, 1), (pr, 0), (c, 0)], [(ai, 1), (bj, 1), (pr, 1), (c, 1)]])
            c, nc = nc, c
        pr = P + i + 12                                           # final carry into bit i+12 (no carry out)
        g.func(pr, [[(pr, 1), (c, 0)], [(pr, 0), (c, 1)]])
    # reduction
    bw = C0
    for k in range(11, -1, -1):
        qk = Q << k
        g.write_row(bw, zero)
        for i in range(k, 24):
            pi, ti = P + i, T + i
            if (qk >> i) & 1 == 0:
                g.func(ti, [[(pi, 1), (bw, 0)], [(pi, 0), (bw, 1)]])     # diff = p xor bw
                g.func(bw, [[(pi, 0), (bw, 1)]])                          # borrow = ~p & bw
            else:
                g.func(ti, [[(pi, 0), (bw, 0)], [(pi, 1), (bw, 1)]])     # diff = ~(p xor bw)
                g.func(bw, [[(pi, 0)], [(bw, 1)]])                        # borrow = ~p | bw
        for i in range(k, 24):                                    # p = borrow ? p : t
            g.func(P + i, [[(bw, 0), (T + i, 1)], [(bw, 1), (P + i, 1)]])
    return g


def result(g):
    return (g.S[P:P + 24].astype(np.int64) << np.arange(24)[:, None]).sum(0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.integers(0, Q, 20000); b = rng.integers(0, Q, 20000)
    a[:4] = [0, Q - 1, Q - 1, 1]; b[:4] = [Q - 1, Q - 1, 0, 1]
    g = kernel(a, b, log=True)
    r = result(g)
    ok = np.array_equal(r, (a * b) % Q)
    print("correct:", ok, "| cycles:", len(g.kind), "compares:", g.n_cmp, "writes:", g.n_wr,
          "| latency %.2f us" % (len(g.kind) * 2e-3))
