"""Figure of the GainP case study: (a) traces to detection versus noise level, (b) noiseless leakage and FPTM score per cycle.
Usage: python3 fig_gainp.py OUTDIR"""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np, json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = sys.argv[1]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
BLUE, AQUA, YELLOW, GRAY, ORANGE = "#2a78b5", "#2fa39a", "#c99a1a", "#6e6d68", "#e0662b"
agg = json.load(open(f"{ROOT}/results/aggregate_gainp.json"))
SIG = [250, 500, 1000, 2000]
M = [("welch", "Welch", BLUE, "s"), ("chi2", r"$\chi^2$", AQUA, "D"), ("mvdcov", "MV-dCov", YELLOW, "v"),
     ("mlp", "DL-LA MLP", GRAY, "X"), ("fptm", "FPTM", ORANGE, "o")]

fig = plt.figure(figsize=(12, 4.8))
gs = GridSpec(3, 2, figure=fig, height_ratios=[0.13, 0.5, 0.5], width_ratios=[1, 1.55], wspace=0.22, hspace=0.18)
# (a) traces to detection
axl = fig.add_subplot(gs[0, 0]); axl.axis("off")
ax = fig.add_subplot(gs[1:, 0])
NONE = 1.0e5
offs = {"welch": 0.955, "chi2": 0.985, "mvdcov": 1.015, "mlp": 1.045, "fptm": 1.0}
for key, lab, c, mk in M:
    xs, ys, nd = [], [], []
    for s in SIG:
        ds = f"gainp_s{s}"
        if ds not in agg or key not in agg[ds]: continue
        v = agg[ds][key]["ttd"]
        (xs.append(s * offs[key]), ys.append(v)) if v else nd.append(s * offs[key])
    ax.plot(xs, ys, color=c, marker=mk, ms=8 if key != "fptm" else 9, lw=2, label=lab, mec="white", mew=0.8,
            zorder=5 if key == "fptm" else 3)
    if nd:
        ax.plot(nd, [NONE] * len(nd), ls="none", marker=mk, ms=8, mfc="white", mec=c, mew=1.8, zorder=4)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(SIG); ax.set_xticklabels([str(s) for s in SIG])
ax.set_xlabel(r"Noise standard deviation $\sigma$ (fJ)"); ax.set_ylabel("Traces to detection")
ax.set_ylim(300, 2.2e5); ax.set_title("(a) Detection", fontsize=11, loc="left")
h = [Line2D([], [], color=c, marker=mk, ms=8, lw=2, mec="white", label=lab) for _, lab, c, mk in M]
axl.legend(handles=h, loc="center", ncol=3, frameon=False, fontsize=9.5)

# (b) per-cycle leakage and FPTM score, sigma = 1000 fJ
g = np.load(f"{ROOT}/data/gainp_s1000/gt.npz"); D = len(g["leak"]); x = np.arange(D)
W = f"{ROOT}/work/gainp_s1000_K3"
S = np.mean([np.fromfile(f"{W}/S_s{s}.bin", np.float64)[:D] for s in (1, 2, 3)], 0)
t = np.abs(g["t_noiseless"]); top = np.argsort(-S)[:20]
b0 = int(np.argmax(g["phase"] == 1)); b1 = int(np.argmax(g["phase"] == 2))
axl2 = fig.add_subplot(gs[0, 1]); axl2.axis("off")
a1 = fig.add_subplot(gs[1, 1]); a2 = fig.add_subplot(gs[2, 1], sharex=a1)
a1.plot(x, t, color=GRAY, lw=0.6); a1.set_ylabel(r"Noiseless $|t|$")
a1.set_title(r"(b) Localization, $\sigma=1{,}000$ fJ", fontsize=11, loc="left")
a2.plot(x, S, color=ORANGE, lw=0.6); a2.plot(top, S[top], ls="none", marker="o", ms=5, color=ORANGE, mec="black", mew=0.6)
a2.set_ylabel(r"FPTM $S(t)$"); a2.set_xlabel("Cycle")
for a in (a1, a2):
    a.axvspan(b0, b1, color="#2a78b5", alpha=0.07, lw=0); a.axvspan(b1, D, color="#2fa39a", alpha=0.07, lw=0)
    a.set_xlim(0, D)
plt.setp(a1.get_xticklabels(), visible=False)
hl = [Patch(color="#2a78b5", alpha=0.25, label="Multiplication"), Patch(color="#2fa39a", alpha=0.25, label="Reduction"),
      Line2D([], [], ls="none", marker="o", ms=6, color=ORANGE, mec="black", mew=0.6, label="20 highest $S(t)$")]
axl2.legend(handles=hl, loc="center", ncol=3, frameon=False, fontsize=9.5)
fig.savefig(f"{OUT}/fig_gainp.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_gainp.png", dpi=200, bbox_inches="tight")
print("ok")
