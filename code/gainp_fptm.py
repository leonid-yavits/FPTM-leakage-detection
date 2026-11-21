"""FPTM inference on the GainP model: one test trace per column, all traces in parallel.

Usage: python3 gainp_fptm.py WORKDIR PREFIX OUTJSON
  WORKDIR: FPTM input folder (meta.json, test_bits.bin), PREFIX: output prefix of dump_model.jl.

Row layout of a column: literal slots (one per literal of the current clause), a 4-bit failure counter,
two carry rows, two 11-bit accumulators (one per clause polarity) and a result row.
For each clause:
  1. The host writes the input bits of the clause's literals into the literal slots (one row write per literal).
  2. The counter is cleared. For each literal, a saturating increment adds one if the literal is false and the
     counter is below L_F = 8. The counter then holds min(f, 8), where f is the number of false literals.
  3. The counter is added to the accumulator of the clause's polarity (ripple carry, Fig. 8 of the GainP paper).
The positive vote sum is 8C - acc_pos and the negative vote sum is 8C - acc_neg, so the prediction
"positive votes exceed negative votes" equals acc_pos < acc_neg. A borrow chain computes it into the result row.
"""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, sys, json, re, time
sys.path.insert(0, CODE)
from gainp_sim import GainP, T_CYCLE

LF, C = 8, 200
ACCW = 11                                   # 8 * 200 = 1600 < 2^11


def bit(X, b):
    """Input bit b (0-based, Julia BitVector chunk layout) of every trace."""
    return ((X[:, b // 64] >> np.uint64(b % 64)) & np.uint64(1)).astype(bool)


def read_clauses(path):
    cl = []
    for line in open(path):
        pol, j, lits = line.rstrip("\n").split("\t")
        L = [(int(a), int(b)) for a, b in (s.split(":") for s in lits.split(",") if s)]
        cl.append((pol, int(j), L))
    return cl


def run(W, prefix):
    meta = json.load(open(f"{W}/meta.json")); nch = meta["nchunks"]; n = meta["n"]["test"]
    X = np.fromfile(f"{W}/test_bits.bin", "<u8").reshape(n, nch)
    cl = read_clauses(f"{prefix}_clauses.tsv")
    Lmax = max(len(L) for _, _, L in cl)
    CNT = Lmax; CR0, CR1 = CNT + 4, CNT + 5; ACC = {"pos": CNT + 6, "neg": CNT + 6 + ACCW}; RES = CNT + 6 + 2 * ACCW
    nrows = RES + 1
    assert nrows <= 128, nrows                      # one GainP column holds 128 CAM bits
    g = GainP(nrows, n, log=False)
    zero = np.zeros(n, bool)
    for r in range(ACC["pos"], ACC["pos"] + 2 * ACCW):
        g.write_row(r, zero)
    for pol, j, L in cl:
        for s, (b, neg) in enumerate(L):            # 1. load literal input bits
            g.write_row(s, bit(X, b))
        for k in range(4):                          # 2. clear counter
            g.write_row(CNT + k, zero)
        for s, (b, neg) in enumerate(L):            #    saturating increment if literal false
            fail = 1 if neg else 0                  # positive literal fails on 0, negated literal on 1
            cr, nc = CR0, CR1
            g.func(cr, [[(s, fail), (CNT + 3, 0)]])
            for k in range(4):
                ck = CNT + k
                if k < 3:
                    g.func(nc, [[(ck, 1), (cr, 1)]])
                g.func(ck, [[(ck, 1), (cr, 0)], [(ck, 0), (cr, 1)]])
                cr, nc = nc, cr
        A = ACC[pol]; cr, nc = CR0, CR1             # 3. acc += counter
        g.write_row(cr, zero)
        for k in range(ACCW):
            ak = A + k
            if k < 4:
                ck = CNT + k
                if k < ACCW - 1:
                    g.func(nc, [[(ck, 1), (cr, 1)], [(ak, 1), (cr, 1)], [(ak, 1), (ck, 1)]])
                g.func(ak, [[(ak, 0), (ck, 0), (cr, 1)], [(ak, 0), (ck, 1), (cr, 0)],
                            [(ak, 1), (ck, 0), (cr, 0)], [(ak, 1), (ck, 1), (cr, 1)]])
            else:
                if k < ACCW - 1:
                    g.func(nc, [[(ak, 1), (cr, 1)]])
                g.func(ak, [[(ak, 1), (cr, 0)], [(ak, 0), (cr, 1)]])
            cr, nc = nc, cr
    # 4. prediction = borrow of acc_pos - acc_neg
    bw, nb = CR0, CR1
    g.write_row(bw, zero)
    for k in range(ACCW):
        x, yv = ACC["pos"] + k, ACC["neg"] + k
        g.func(nb, [[(x, 0), (yv, 1)], [(x, 0), (bw, 1)], [(yv, 1), (bw, 1)]])
        bw, nb = nb, bw
    g.func(RES, [[(bw, 1)]])
    acc = lambda p: (g.S[ACC[p]:ACC[p] + ACCW].astype(np.int64) << np.arange(ACCW)[:, None]).sum(0)
    return g, g.S[RES].copy(), C * LF - acc("pos"), C * LF - acc("neg"), cl, nrows


if __name__ == "__main__":
    W, prefix, outj = sys.argv[1], sys.argv[2], sys.argv[3]
    t0 = time.time()
    g, pred, vp, vn, cl, nrows = run(W, prefix)
    ref = np.loadtxt(f"{prefix}_pred.tsv", dtype=np.int64)
    res = {"workdir": W, "test_traces": int(len(pred)), "clauses": len(cl),
           "literals": int(sum(len(L) for _, _, L in cl)), "max_literals_per_clause": int(max(len(L) for _, _, L in cl)),
           "rows_used": nrows, "cycles": len(g.kind) if g.log else g.n_cmp + g.n_wr,
           "compares": g.n_cmp, "writes": g.n_wr,
           "latency_us": (g.n_cmp + g.n_wr) * T_CYCLE * 1e6,
           "energy_nJ_per_trace_mean": float(g.E.mean() * 1e9), "energy_nJ_per_trace_min": float(g.E.min() * 1e9),
           "energy_nJ_per_trace_max": float(g.E.max() * 1e9),
           "match_votes_pos": bool(np.array_equal(vp, ref[:, 1])), "match_votes_neg": bool(np.array_equal(vn, ref[:, 2])),
           "match_pred": bool(np.array_equal(pred.astype(int), ref[:, 3])),
           "accuracy": float((pred.astype(int) == ref[:, 4]).mean()), "sim_seconds": round(time.time() - t0, 1)}
    print(json.dumps(res)); json.dump(res, open(outj, "w"), indent=1)
