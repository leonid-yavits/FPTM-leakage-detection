"""ASCAD clause analysis: fraction of clauses that include literals of both share cores (SNR > thr),
compared with two permutation nulls. Null 1 keeps every clause's literals and places the cores at random samples.
Null 2 keeps the cores and the multiset of literal samples (so the concentration of literals on leaking samples is kept)
and redistributes the literal samples at random over clauses of unchanged size. Null 2 tests whether clauses combine
both shares more often than the marginal literal frequencies imply.
Input: results/ascad_clauses_s{1,2,3}.tsv from dump_clauses.jl; data/ascad/gt.npz from build_datasets.py.
Reported in the paper with thr = 1.0 (16 mask samples, 23 masked-value samples)."""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, json, sys
GT = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/data/ascad/gt.npz"
RES = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/results"
g = np.load(GT)
ms = np.maximum(g["snr_mask0"], g["snr_mask15"]); vs = np.maximum(g["snr_vmask0"], g["snr_vmask15"])
rng = np.random.default_rng(0); res = {}
for thr in (0.5, 1.0):
    M = ms > thr; V = vs > thr
    for sd in (1, 2, 3):
        cl = []
        for line in open(f"{RES}/ascad_clauses_s{sd}.tsv"):
            pol, j, v, lits = line.rstrip("\n").split("\t")
            cl.append((float(v), np.array(sorted({int(x.split(':')[0]) for x in lits.split(',') if x}))))
        votes = np.array([c[0] for c in cl])
        def stat(Mm, Vv):
            both = np.array([Mm[s].any() and Vv[s].any() for _, s in cl]); return both.mean(), votes[both].sum() / votes.sum()
        obs = stat(M, V); null = []
        for _ in range(2000):
            p = rng.permutation(len(ms)); null.append(stat(M[p], V[p]))
        null = np.array(null); pval = (1 + (null[:, 0] >= obs[0]).sum()) / 2001
        lit = np.concatenate([s for _, s in cl])
        sizes = np.array([len(s) for _, s in cl]); cuts = np.cumsum(sizes)[:-1]; null2 = []
        for _ in range(2000):
            parts = np.split(rng.permutation(lit), cuts)
            null2.append(np.mean([M[q].any() and V[q].any() for q in parts]))
        null2 = np.array(null2); p2 = (1 + (null2 >= obs[0]).sum()) / 2001
        res[f"thr{thr}_s{sd}"] = dict(both=obs[0], vote_both=obs[1], null_mean=float(null[:, 0].mean()), p=pval,
                                      null2_mean=float(null2.mean()), null2_q99=float(np.quantile(null2, 0.99)), p2=p2,
                                      lit_mask=float(M[lit].mean()), lit_val=float(V[lit].mean()), frac_mask=float(M.mean()), frac_val=float(V.mean()))
        print(thr, sd, json.dumps(res[f"thr{thr}_s{sd}"]))
json.dump(res, open(f"{RES}/ascad_clause_null.json", "w"), indent=1)
