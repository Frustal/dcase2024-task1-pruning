"""Figure for the prior practical work, redrawn in the thesis's own style.

This is NOT a result of this thesis. It is the author's earlier practical work (IMP and SNIP on
ResNet-50, Oxford Flowers-102), reproduced here only so Section 6.2 can make the cross-scale
contrast with real numbers instead of a recollection.

Values are transcribed from that project's own logs, one row per pruning level:

    output/logs/imp_r50/metrics.csv          (11 levels, 5 % - 85 % sparsity)
    output/logs/snip_r50/metrics.csv         (11 levels, same targets)
    output/logs/default_r18|r34|r50/metrics.csv   (dense ResNet baselines)
    output/logs/default_effnet_b0/metrics.csv     (EfficientNet-B0 reference)

from that project's own run logs. They are inlined rather than read from it, so this repository
builds standalone. Two further runs in those logs,
``default_r18_custom`` (0.4510) and ``default_r18_custom_v2`` (0.1586), were failed architecture
experiments and are excluded here exactly as they were excluded from the original plot.

Run as ``uv run python -m experiments.plot_prior_work [--outdir DIR]``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parent.parent

RC = {
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# (params, test_acc) per sparsity level, ascending sparsity
IMP = [(22480618, 0.9047), (21297428, 0.9058), (20114237, 0.9096), (17747856, 0.9086),
       (15381475, 0.9083), (13015095, 0.9132), (10648714, 0.9112), (8282333, 0.9070),
       (7099142, 0.9062), (4732762, 0.8974), (3549571, 0.8925)]
SNIP = [(22480618, 0.9005), (21297428, 0.9080), (20114237, 0.9014), (17747856, 0.9136),
        (15381475, 0.9096), (13015095, 0.9036), (10648713, 0.8932), (8282331, 0.8907),
        (7099143, 0.8866), (4732762, 0.8279), (3549572, 0.7408)]
DENSE = [(11228838, 0.9093, "ResNet-18"), (21336998, 0.9185, "ResNet-34"),
         (23717030, 0.9065, "ResNet-50")]
EFFNET = (4138210, 0.9228)

BLACK, ORANGE, TEAL = "#222222", "#c1662f", "#1b7f6f"


def make_figure():
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    m = 1e6

    ax.plot([p / m for p, _, _ in DENSE], [a for _, a, _ in DENSE],
            "o--", color=BLACK, lw=1.2, ms=5, label="dense ResNet baselines")
    ax.plot(EFFNET[0] / m, EFFNET[1], "X", color=BLACK, ms=7, mfc="white", mew=1.2,
            label="EfficientNet-B0")
    ax.plot([p / m for p, _ in IMP], [a for _, a in IMP],
            "o-", color=ORANGE, lw=1.4, ms=4, label="IMP (ResNet-50)")
    ax.plot([p / m for p, _ in SNIP], [a for _, a in SNIP],
            "o-", color=TEAL, lw=1.4, ms=4, label="SNIP (ResNet-50)")

    # the collapse is the whole point of the figure, so name it
    ax.annotate("SNIP collapses\nat 85 % sparsity",
                xy=(SNIP[-1][0] / m, SNIP[-1][1]), xytext=(14, 6),
                textcoords="offset points", fontsize=7.5, color=TEAL,
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.7))

    ax.set_xlabel("parameters (millions)")
    ax.set_ylabel("test accuracy, Flowers-102")
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, borderpad=0.5)
    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=REPO / "reports" / "figures")
    args = ap.parse_args()
    with plt.rc_context(RC):
        fig = make_figure()
    args.outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = args.outdir / f"fig7_prior_work.{ext}"
        fig.savefig(path, metadata={"CreationDate": None} if ext == "pdf" else None)
        print(f"wrote {path}")
    plt.close(fig)

    gap = (IMP[-1][1] - SNIP[-1][1]) * 100
    print(f"\nIMP - SNIP at 85 % sparsity: {gap:.2f} pp")
    print(f"IMP at 85 % vs dense ResNet-50: {(IMP[-1][1] - DENSE[2][1]) * 100:+.2f} pp")
    print(f"SNIP at 85 % vs dense ResNet-50: {(SNIP[-1][1] - DENSE[2][1]) * 100:+.2f} pp")


if __name__ == "__main__":
    main()
