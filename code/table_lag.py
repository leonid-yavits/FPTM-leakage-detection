"""LaTeX rows of the lag-restricted pair test table (review round 4) from results/biv_delta_ascad*.jsonl."""
import os as _os, json, collections
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
def load(*files):
    m = {}
    for f in files:
        p = f"{ROOT}/results/{f}"
        if _os.path.exists(p):
            for l in open(p): r = json.loads(l); m.setdefault((r["N"], r["r"]), {}).update(r)
    d = collections.defaultdict(list)
    for (N, _), r in sorted(m.items()): d[N].append(r)
    return d
def rows(d, Ns, deltas, label):
    out = []
    for x in deltas:
        cells = []
        for N in Ns:
            rs = [r for r in d.get(N, []) if f"d{x}" in r]
            cells.append(str(sum(r[f"d{x}"] < -5 for r in rs)) if len(rs) == 10 else "$\\cdot$")
        m = next(r[f"m{x}"] for N in Ns for r in d.get(N, []) if f"m{x}" in r)
        out.append(f"{label} & {x} & {m:,}".replace(",", "{,}") + " & " + " & ".join(cells) + " \\\\")
        label = ""
    return out
full = load("biv_delta_ascad.jsonl", "biv_delta_ascad_small.jsonl")
noov = load("biv_delta_ascad_noov_d.jsonl")
Ns = [1000, 2000, 2828, 4000, 5657, 8000, 11314, 16000, 22627]
print("\n".join(rows(full, Ns, [1, 10, 100, 200, 300, 400, 699], "All")))
print("\\midrule")
print("\n".join(rows(noov, Ns, [1, 10, 100, 200, 300, 400, 699], "All $-$ O")))
