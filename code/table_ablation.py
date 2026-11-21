"""Print the LaTeX rows of the ASCAD region-ablation table from results/ascad_ablation_<v>.json and ascad_ablation_rf.json."""
import os as _os, json
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
rf = json.load(open(f"{ROOT}/results/ascad_ablation_rf.json"))
names = {"all": "All", "no_overlap": "Without overlap", "no_overlap_no_mask": "Without overlap and mask",
         "no_overlap_no_mv": "Without overlap and masked value", "mask_only": "Mask only", "mv_only": "Masked value only",
         "both_shares_only": "Both shares only"}
f = lambda v: "none" if v > -5 else f"${v:.1f}$"
L = lambda xs: ", ".join(f"${x:.1f}$".replace("$-0.0$", "$0.0$") for x in xs)
for v, n in names.items():
    d = json.load(open(f"{ROOT}/results/ascad_ablation_{v}.json"))
    fp = [float(r.split("log10p=")[1]) for r in d["fptm_runs"]]
    ch = d["chance_in_subset"]; pr = lambda p: "n/a" if ch == 1.0 else f"{p:.2f}"
    print(f"{n} & {d['samples']} & {ch:.2f} & {f(d['welch_log10p'])} & {f(d['chi2_log10p'])} & {L(fp)} & {pr(d['p@20'])} & {L(rf[v]['log10p'])} & {pr(rf[v]['mean_p@20'])} \\\\")
