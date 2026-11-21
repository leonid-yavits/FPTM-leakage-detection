import numpy as np, sys, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

W = sys.argv[1]; OUT = sys.argv[2]
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#52514e"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
g = np.load(f"{W}/ground.npz"); t = np.abs(g["t_te"]); D = len(t); x = np.arange(D)
S = np.mean([np.fromfile(f"{W}/S_s{k}.bin", np.float64)[:D] for k in (1, 2, 3)], axis=0)

# Figure 1: localization, full trace (left) and zoom on the leakage cluster (right)
fig = plt.figure(figsize=(10, 5.6))
gs = GridSpec(3, 2, figure=fig, height_ratios=[0.12, 1, 1], width_ratios=[2.2, 1], hspace=0.35, wspace=0.18)
axl = fig.add_subplot(gs[0, :]); axl.axis("off")
Z = (120, 200)
for col, (lo, hi) in enumerate([(0, D), Z]):
    a1 = fig.add_subplot(gs[1, col]); a2 = fig.add_subplot(gs[2, col], sharex=a1)
    a1.plot(x[lo:hi], t[lo:hi], color=BLUE, lw=1.0 if col == 0 else 2)
    a1.axhline(4.5, color=GRAY, lw=1, ls="--")
    a2.plot(x[lo:hi], S[lo:hi], color=ORANGE, lw=1.0 if col == 0 else 2)
    a1.set_xlim(lo, hi - 1); a1.set_ylim(0, t.max() * 1.15); a2.set_ylim(0, S.max() * 1.15)
    plt.setp(a1.get_xticklabels(), visible=False)
    a2.set_xlabel("Trace sample index" + (" (zoom)" if col else ""))
    if col == 0:
        a1.set_ylabel("Welch |t|"); a2.set_ylabel("FPTM S(t)")
        for a in (a1, a2): a.axvspan(*Z, color="#e6e5e0", lw=0)
axl.legend(handles=[Line2D([], [], color=BLUE, lw=2, label="Welch |t|, test split"),
                    Line2D([], [], color=GRAY, lw=1, ls="--", label="TVLA threshold |t| = 4.5"),
                    Line2D([], [], color=ORANGE, lw=2, label="FPTM S(t), included literals, mean of 3 seeds")],
           loc="center", ncol=3, frameon=False)
fig.savefig(f"{OUT}/fig_localization_pqm4.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{OUT}/fig_localization_pqm4.pdf", bbox_inches="tight")

# Figure 2: traces to detection
def parse(path, pat):
    d = {}
    for ln in open(path):
        m = re.search(pat, ln)
        if m: d.setdefault(int(m["N"]), []).append(int(m["v"]))
    return d
fp = parse(f"{W}/sweep_fptm.txt", r"N=(?P<N>\d+) r=\d+ .*detect=(?P<v>\d)")
tb = parse(f"{W}/sweep_t.txt", r"N=(?P<N>\d+) r=\d+ .*bonf\(thr [\d.]+\)=(?P<v>\d)")
tv = parse(f"{W}/sweep_t.txt", r"N=(?P<N>\d+) r=\d+ .*tvla=(?P<v>\d)")
Ns = sorted(fp)
fig = plt.figure(figsize=(6.4, 4.4))
gs = GridSpec(2, 1, figure=fig, height_ratios=[0.12, 1], hspace=0.1)
axl = fig.add_subplot(gs[0]); axl.axis("off"); ax = fig.add_subplot(gs[1])
series = [(fp, ORANGE, "o", "-", "FPTM, binomial p < 1e-5", 8, 3),
          (tb, BLUE, "s", "-", "Welch t-test, Bonferroni (FWER 1e-5)", 12, 2),
          (tv, GRAY, "^", "--", "Welch t-test, |t| > 4.5 (uncorrected)", 9, 1)]
for d, c, mk, ls, lab, ms, z in series:
    ax.plot(Ns, [np.mean(d[n]) for n in Ns], color=c, marker=mk, ms=ms, lw=2, ls=ls, label=lab, zorder=z, mec="white", mew=1)
ax.set_xscale("log", base=2); ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
ax.set_ylim(-0.08, 1.08); ax.set_xlabel("Number of traces N"); ax.set_ylabel("Fraction of 3 repeats with detection")
axl.legend(*ax.get_legend_handles_labels(), loc="center", ncol=1, frameon=False, fontsize=10)
fig.savefig(f"{OUT}/fig_traces_to_detection_pqm4.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{OUT}/fig_traces_to_detection_pqm4.pdf", bbox_inches="tight")
print("ok")
