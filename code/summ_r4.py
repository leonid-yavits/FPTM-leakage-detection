"""Summary of the review round 4 results (desync, PCA, permuted-label controls) for the manuscript tables."""
import os as _os, json, glob, re
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
R = f"{ROOT}/results"
def fptm(log):
    if not _os.path.exists(log): return None
    return [(float(m.group(1)), float(m.group(2))) for m in re.finditer(r"test=([0-9.]+) .*log10p=(-?[0-9.]+)", open(log).read())]
out = {}
for ds, ss in (("ascad", (0, 10, 25, 50)), ("pqm4", (5, 10, 20))):
    for s in ss:
        b = f"{R}/desync_{ds}_s{s}.json"; r = json.load(open(b)) if _os.path.exists(b) else {}
        out[f"desync_{ds}_s{s}"] = {"fptm": fptm(f"{R}/r4_desync_fptm_{ds}_s{s}.log"), "welch": r.get("welch"), "chi2": r.get("chi2"),
                                    "biv": r.get("biv"), "dlla_sel": [(x["acc"], x["log10p"]) for x in r.get("dlla_sel", [])],
                                    "rf": [(x["acc"], x["log10p"]) for x in r.get("rf", [])]}
for ds in ("ascad", "pqm4"):
    for nc in (20, 100):
        b = f"{R}/pca_{ds}_{nc}.json"; r = json.load(open(b)) if _os.path.exists(b) else {}
        out[f"pca_{ds}_{nc}"] = {"fptm": fptm(f"{R}/r4_pca_fptm_{ds}_{nc}.log"), "explained": r.get("explained"),
                                 "lda": (r["lda"]["acc"], r["lda"]["log10p"]) if "lda" in r else None,
                                 "dlla_sel": [(x["acc"], x["log10p"]) for x in r.get("dlla_sel", [])],
                                 "rf": [(x["acc"], x["log10p"]) for x in r.get("rf", [])]}
for ds, f in (("ascad", "r4_null_ascad.log"), ("ref", "r4_nullB_ref.log"), ("pqm4", "r4_nullB_pqm4.log")):
    out[f"null_{ds}"] = fptm(f"{R}/{f}")
json.dump(out, open(f"{R}/r4_summary.json", "w"), indent=1)
for k, v in out.items():
    print(k, json.dumps(v, default=lambda x: x)[:400].replace("], [", "],[") if v else None)
