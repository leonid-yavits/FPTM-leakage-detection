import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

OUT = sys.argv[1]; which = sys.argv[2].split(",")
BLUE, ORANGE, GRAY, AQUA, VIOLET, YELLOW, RED = "#2a78d6", "#eb6834", "#52514e", "#1baf7a", "#6250d6", "#eda100", "#e34948"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "pdf.fonttype": 42})
W = ROOT + "/work"

def S_mean(ds, D):
    return np.mean([np.fromfile(f"{W}/{ds}_K3/S_s{k}.bin", np.float64)[:D] for k in (1, 2, 3)], axis=0)

if "loc_unmasked" in which:
    fig = plt.figure(figsize=(10, 7.6))
    gs = GridSpec(5, 2, figure=fig, height_ratios=[0.12, 1, 1, 1, 1], width_ratios=[2.2, 1], hspace=0.55, wspace=0.18)
    axl = fig.add_subplot(gs[0, :]); axl.axis("off")
    for row, (ds, Z) in enumerate([("ref", None), ("pqm4", None)]):
        sp = np.load(f"{W}/{ds}_K3/split.npz"); t = np.abs(sp["t_te"]); D = len(t); S = S_mean(ds, D); x = np.arange(D)
        c = int(np.argmax(t)); Z = (max(0, c - 60), min(D, c + 60))
        for col, (lo, hi) in enumerate([(0, D), Z]):
            a1 = fig.add_subplot(gs[1 + 2 * row, col]); a2 = fig.add_subplot(gs[2 + 2 * row, col], sharex=a1)
            a1.plot(x[lo:hi], t[lo:hi], color=BLUE, lw=0.8 if col == 0 else 2)
            a2.plot(x[lo:hi], S[lo:hi], color=ORANGE, lw=0.8 if col == 0 else 2)
            a1.set_xlim(lo, hi - 1); a1.set_ylim(0, t.max() * 1.15); a2.set_ylim(0, S.max() * 1.15)
            plt.setp(a1.get_xticklabels(), visible=False)
            if col == 0:
                a1.set_ylabel(f"{ds}\nWelch |t|"); a2.set_ylabel(f"{ds}\nFPTM S(t)")
                for a in (a1, a2): a.axvspan(*Z, color="#e6e5e0", lw=0)
            if row == 1: a2.set_xlabel("Trace sample index" + (" (zoom)" if col else ""))
    axl.legend(handles=[Line2D([], [], color=BLUE, lw=2, label="Welch |t| on held-out test traces"),
                        Line2D([], [], color=ORANGE, lw=2, label="FPTM localization score S(t), mean of 3 seeds")],
               loc="center", ncol=2, frameon=False)
    fig.savefig(f"{OUT}/fig_loc_unmasked.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_loc_unmasked.png", dpi=200, bbox_inches="tight")

if "loc_ascad" in which:
    g = np.load(ROOT + "/data/ascad/gt.npz"); D = 700; x = np.arange(D); S = S_mean("ascad", D)
    SA = np.mean([np.load(f"{W}/ascad_K3/SAsel_s{k}.npy") for k in (1, 2, 3)], axis=0)
    mask = np.maximum(g["snr_mask0"], g["snr_mask15"]); mval = np.maximum(g["snr_vmask0"], g["snr_vmask15"])
    fig = plt.figure(figsize=(9, 7.8))
    gs = GridSpec(5, 1, figure=fig, height_ratios=[0.12, 1, 1, 1, 1], hspace=0.35)
    axl = fig.add_subplot(gs[0]); axl.axis("off")
    a0 = fig.add_subplot(gs[1]); a1 = fig.add_subplot(gs[2], sharex=a0); a2 = fig.add_subplot(gs[3], sharex=a0); a3 = fig.add_subplot(gs[4], sharex=a0)
    a0.plot(x, mask, color=AQUA, lw=1.5); a0.set_ylabel("Mask SNR")
    a1.plot(x, mval, color=VIOLET, lw=1.5); a1.plot(x, g["snr_v"], color=GRAY, lw=1.5, ls="--"); a1.set_ylabel("Value SNR")
    a2.plot(x, S, color=ORANGE, lw=1.5); a2.set_ylabel("FPTM S(t)")
    a3.plot(x, SA, color=BLUE, lw=1.5); a3.set_ylabel("DL-LA SA"); a3.set_xlabel("Trace sample index")
    a0.set_ylim(0, mask.max() * 1.15); a1.set_ylim(0, mval.max() * 1.15); a2.set_ylim(0, S.max() * 1.15); a3.set_ylim(0, SA.max() * 1.15); a0.set_xlim(0, D - 1)
    for a in (a0, a1, a2): plt.setp(a.get_xticklabels(), visible=False)
    axl.legend(handles=[Line2D([], [], color=AQUA, lw=2, label="Mask share"), Line2D([], [], color=VIOLET, lw=2, label="Masked value share"),
                        Line2D([], [], color=GRAY, lw=2, ls="--", label="Unmasked value"), Line2D([], [], color=ORANGE, lw=2, label="FPTM S(t)"),
                        Line2D([], [], color=BLUE, lw=2, label="DL-LA sensitivity, epoch selected")],
               loc="center", ncol=5, frameon=False, fontsize=9.5)
    fig.savefig(f"{OUT}/fig_loc_ascad.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_loc_ascad.png", dpi=200, bbox_inches="tight")

if "ttd" in which:
    agg = json.load(open(ROOT + "/results/aggregate.json"))
    M = [("welch", "Welch", BLUE, "s", "-"), ("chi2", r"$\chi^2$", AQUA, "D", "-"),
         ("welch2", "Welch, 2nd order", VIOLET, "^", "--"), ("biv", "Bivariate Welch", RED, "P", "--"), ("mlp", "DL-LA", GRAY, "X", "-"),
         ("fptm", "FPTM", ORANGE, "o", "-")]
    sets = [d for d in ("ref", "pqm4", "ascad") if d in agg]; TOP = 2000.0
    TITLE = {"ref": "ref", "pqm4": "pqm4", "ascad": "ASCAD"}
    fig = plt.figure(figsize=(12, 4.6))
    gs = GridSpec(2, len(sets), figure=fig, height_ratios=[0.14, 1], wspace=0.28, hspace=0.25)
    axl = fig.add_subplot(gs[0, :]); axl.axis("off"); handles = {}
    for i, ds in enumerate(sets):
        ax = fig.add_subplot(gs[1, i])
        for key, lab, c, mk, ls in M:
            if key not in agg[ds]: continue
            per = agg[ds][key]["per_N"]; Ns = sorted(int(n) for n in per)
            y = [max(0.1, min(TOP, -np.median(per[str(n)]))) for n in Ns]   # p below double precision (-inf) drawn at TOP
            h, = ax.plot(Ns, y, color=c, marker=mk, ms=8 if key != "fptm" else 9, lw=2, ls=ls, label=lab, zorder=5 if key == "fptm" else 3, mec="white", mew=0.8)
            handles[lab] = h
        ax.axhline(5, color="#8a8984", lw=1, ls=":"); ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(Ns); ax.set_xticklabels([f"{n//1000}k" if n >= 1000 else str(n) for n in Ns], fontsize=9, rotation=45 if ds == "ascad" else 0)
        ax.set_ylim(0.07, TOP * 1.6); ax.set_title(TITLE[ds], fontsize=11); ax.set_xlabel("Traces N")
        if i == 0: ax.set_ylabel(r"$-\log_{10} p$ (median of 3)")
    hl = [handles[l] for _, l, *_ in M if l in handles] + [Line2D([], [], color="#8a8984", lw=1, ls=":")]
    axl.legend(hl, [h.get_label() for h in hl[:-1]] + [r"Threshold $p=10^{-5}$"], loc="center", ncol=4, frameon=False, fontsize=10)
    fig.savefig(f"{OUT}/fig_ttd.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_ttd.png", dpi=200, bbox_inches="tight")
print("ok")
