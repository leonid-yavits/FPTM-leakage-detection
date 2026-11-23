"""LaTeX rows for the misalignment and reduced-dimension tables (review round 4) from results/r4_summary.json."""
import os as _os, json
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
S = json.load(open(f"{ROOT}/results/r4_summary.json"))
def one(v): return "--" if v is None else "none" if v > -5 else f"${v:.0f}$" if v < -99.5 else f"${v:.1f}$"
def seeds(L):
    if not L: return "--"
    n = sum(lp < -5 for _, lp in L); lo, hi = min(lp for _, lp in L), max(lp for _, lp in L)
    f = lambda v: f"{v:.0f}" if v < -99.5 else f"{v:.1f}"
    return f"{n}/{len(L)} (${f(lo)}$ to ${f(hi)}$)"
print("% misalignment")
for ds, ss in (("ascad", (0, 10, 25, 50)), ("pqm4", (5, 10, 20))):
    for s in ss:
        r = S[f"desync_{ds}_s{s}"]
        print(f"{'ASCAD' if ds == 'ascad' else 'pqm4'} & {s} & {one(r['welch'])} & {one(r['chi2'])} & {one(r['biv']) if ds == 'ascad' else 'n/a'} & {seeds(r['dlla_sel'])} & {seeds(r['rf'])} & {seeds(r['fptm'])} \\\\")
print("% pca")
for ds in ("ascad", "pqm4"):
    for nc in (20, 100):
        r = S[f"pca_{ds}_{nc}"]
        lda = one(r["lda"][1]) if r["lda"] else "--"
        print(f"{'ASCAD' if ds == 'ascad' else 'pqm4'} & {nc} & {100 * r['explained']:.0f}\\% & {lda} & {seeds(r['dlla_sel'])} & {seeds(r['rf'])} & {seeds(r['fptm'])} \\\\" if r["explained"] else f"{ds} {nc} pending")
for ds in ("ascad", "ref", "pqm4"):
    L = S[f"null_{ds}"]
    if L: print(ds, "null", len(L), "min log10p %.2f" % min(lp for _, lp in L), "acc %.3f to %.3f" % (min(a for a, _ in L), max(a for a, _ in L)))
