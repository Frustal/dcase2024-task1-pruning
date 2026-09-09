"""Dataset figure for the thesis: recording devices in the 25 % train split vs the test split.

The point the figure has to make is the one the prose kept spending a paragraph on: training
is dominated by the reference device A and never sees S4--S6, while the test split is flat
across nine devices. Everything is counted from the real files -- ``meta.csv`` joined against
the official split lists in ``baseline/dataset/splits/`` -- so the numbers in the caption and
the bars cannot drift apart.

Run as ``uv run python -m experiments.plot_dataset [--outdir DIR]``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parent.parent
META = REPO / "data" / "tau_scenes_dataset" / "meta.csv"
SPLITS = REPO / "baseline" / "dataset" / "splits"

# shared with experiments/plot_results.py -- Computer Modern, to match the thesis body text
RC = {
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
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

SEEN = "#3b6ea5"      # devices present in training
UNSEEN = "#c1662f"    # devices the model only meets at test time


def device_counts() -> tuple[pd.Series, pd.Series]:
    meta = pd.read_csv(META, sep="\t").set_index("filename")
    out = []
    for name in ("split25.csv", "test.csv"):
        files = pd.read_csv(SPLITS / name, sep="\t")["filename"].values
        rows = meta.loc[meta.index.intersection(files)]
        out.append(rows["source_label"].value_counts().sort_index())
    return out[0], out[1]


def make_figure(train: pd.Series, test: pd.Series):
    devices = sorted(set(train.index) | set(test.index))
    # Separate y-scales on purpose: device A alone carries 25,520 of the 34,900 training
    # segments, so a shared axis would flatten the test panel into a stripe.
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6))

    for ax, counts, title in (
        (axes[0], train, f"train, 25% split ({int(train.sum()):,} segments)"),
        (axes[1], test, f"development test ({int(test.sum()):,} segments)"),
    ):
        vals = [int(counts.get(d, 0)) for d in devices]
        colours = [SEEN if train.get(d, 0) > 0 else UNSEEN for d in devices]
        ax.bar(range(len(devices)), vals, color=colours, width=0.72)
        ax.set_xticks(range(len(devices)))
        ax.set_xticklabels([d.upper() for d in devices])
        ax.set_title(title, fontsize=8.5, pad=6)
        ax.set_ylim(0, max(vals) * 1.30)
        ax.set_ylabel("segments")
        ax.set_xlabel("recording device")

    # Exact counts live in the caption; on the bars they only collide.
    a_share = train.get("a", 0) / train.sum()
    axes[0].annotate(f"A alone: {a_share:.0%}\nof the split",
                     xy=(0.96, 0.92), xycoords="axes fraction", ha="right", va="top",
                     fontsize=7.4, color=SEEN)
    axes[1].annotate("unseen in\ntraining", xy=(7.0, 3300), xytext=(0, 8),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=7.4, color=UNSEEN)
    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=REPO / "reports" / "figures")
    args = ap.parse_args()

    train, test = device_counts()
    with plt.rc_context(RC):
        fig = make_figure(train, test)
    args.outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = args.outdir / f"fig6_dataset_devices.{ext}"
        fig.savefig(path, metadata={"CreationDate": None} if ext == "pdf" else None)
        print(f"wrote {path}")
    plt.close(fig)

    print("\ntrain:", dict(train), f"total={int(train.sum())}")
    print("test :", dict(test), f"total={int(test.sum())}")


if __name__ == "__main__":
    main()
