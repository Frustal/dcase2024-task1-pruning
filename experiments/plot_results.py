"""Publication figures for the pruning results, from ``reports/results.json``.

Five figures, each written as both PNG (200 dpi, for drafts and slides) and PDF (vector, for
LaTeX ``\\includegraphics``):

  fig1_accuracy_vs_params   THE main figure. The supervisor asked for a plot rather than
                            interpolated values, and this is why: below 24 k parameters no dense
                            model exists at a pruned size, and only a plot makes the difference
                            between a measured point and a line segment visible. Read the shape,
                            not any single point.
  fig2_accuracy_vs_macs     the compute view. IMP and SNIP collapse onto a single vertical line
                            because unstructured masking changes no tensor shape.
  fig3_macs_vs_params       why a parameter-matched comparison is not a compute-matched one: the
                            dense references reorder between the two axes.
  fig4_imp_retrain_budget   IMP at 20 / 35 / 150 retrain epochs, isolating how much of IMP's
                            damage is under-training rather than pruning.
  fig5_seed_scatter         every individual SNIP and DSP run, so the reader can see the spread
                            the means in figures 1 and 2 are hiding -- and see that DSP's is the
                            wider of the two.

SEED INFORMATION IS PART OF THE DESIGN, not an annotation added afterwards. Two encodings carry
it, and they are used identically in every figure:

  * a FILLED marker with an error bar means a mean over three or more seeds, the bar being
    ±1 sample standard deviation;
  * a HOLLOW marker with no bar means a SINGLE run. A reader must be able to tell at a glance
    which numbers are one draw, because six of the twelve models here are.

The previous version of these figures drew one global ±1.67 pp band (the cm=1.3 seed spread)
along the whole dense curve, because that was the only spread the project had measured. Ten more
runs later there are real per-point spreads and they differ by a factor of five between points,
so the band has been replaced by per-point error bars. Do not reinstate it.

Other design constraints, all deliberate:
  * Okabe-Ito colourblind-safe palette, and every series is additionally distinguished by marker
    shape and line style, so the figures survive greyscale printing.
  * Font sizes are set for a figure reproduced at roughly half a text column.
  * No seaborn and no style sheet that would need a network fetch; matplotlib's Agg backend only.
  * Deterministic: no jitter beyond fixed offsets, no randomness, axis limits derived from data.

Not a standalone entry point in normal use; ``experiments.collect_results`` calls
``make_figures``. It can be run directly against an existing reports/results.json for a quick
re-render:  ``uv run python -m experiments.plot_results``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
FIGDIR = REPO / "reports" / "figures"

# Bare mode drops the in-image titles and footers. The report figures keep both because
# reports/results.md is read standalone; a thesis puts that text in a LaTeX caption instead.
BARE = False


def _title(ax, text: str) -> None:
    if not BARE:
        ax.set_title(text, pad=8)


def _suptitle(fig, text: str, **kw) -> None:
    if not BARE:
        fig.suptitle(text, **kw)


def _footer(fig, y: float, text: str) -> None:
    if not BARE:
        fig.text(0.5, y, text, ha="center", va="top", fontsize=6.2, color="#444444", wrap=True)

# Okabe-Ito. Distinguishable under deuteranopia, protanopia and tritanopia.
BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#999999"

# colour, marker, linestyle, z-order. Marker shape carries the same information as colour so the
# figures do not depend on colour at all.
STYLE = {
    "dense":   {"color": BLACK,      "marker": "o", "ls": "-",  "zorder": 3},
    "IMP-150": {"color": BLUE,       "marker": "s", "ls": "--", "zorder": 4},
    "IMP-35":  {"color": SKY,        "marker": "v", "ls": ":",  "zorder": 2},
    "IMP-20":  {"color": PURPLE,     "marker": "^", "ls": ":",  "zorder": 2},
    "SNIP":    {"color": VERMILLION, "marker": "D", "ls": "-.", "zorder": 4},
    "DSP":     {"color": GREEN,      "marker": "P", "ls": "--", "zorder": 5},
}

# Point labels in figure 1 are hand-placed: the pruned curves cross the dense curve, so one
# uniform offset always collides with some series. Entries are (dx, dy, ha, anchor), with anchor
# "below" or "above" offsetting from the end of the error bar rather than the marker, so a point
# with a wide seed spread pushes its own label clear. A new dense reference needs an entry here.
FIG1_LABEL_OFFSETS = {
    "base=8":  (0, -11, "center", "below"),
    "base=16": (-4, 9, "right", "above"),
    "cm=0.5":  (0, -10, "center", "below"),
    "cm=1.0":  (0, -10, "center", "below"),
    "base=24": (0, -10, "center", "below"),
    "cm=1.3":  (10, -6, "left", "below"),
}

# Same rationale as FIG1_LABEL_OFFSETS, for the MACs axis of figure 2, where the two dense
# families cross and a uniform offset puts base=24 on top of cm=0.5.
FIG2_LABEL_OFFSETS = {
    "base=8":  (0, -11, "center", "below"),
    "base=16": (6, 9, "left", "above"),
    "base=24": (-4, 10, "right", "above"),
    "cm=0.5":  (-6, -10, "right", "below"),
    "cm=1.0":  (0, 10, "center", "above"),
    "cm=1.3":  (0, -10, "center", "below"),
}

RC = {
    # Computer Modern, to match the thesis body text. The thesis is typeset under the JKU
    # template's `nofancyfonts' branch (Latin Modern / CM), and DejaVu Sans axis labels beside
    # CM body text read as a foreign object on the page. "cmr10" ships with matplotlib, so no
    # system font install is needed; DejaVu Serif is the per-glyph fallback for anything cmr10
    # lacks. cmr10 has no U+2212, hence unicode_minus off.
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.4,
    "lines.markersize": 5,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,   # embed TrueType, not Type 3 -- some thesis templates reject Type 3
    "ps.fonttype": 42,
}


def save(fig, name: str) -> list[Path]:
    """Write one figure as PNG and PDF, byte-for-byte reproducibly.

    matplotlib stamps a CreationDate into every PDF, which makes a re-run look like a content
    change in git even when nothing moved. Suppressing it keeps `git status` honest about whether
    the data actually changed.
    """
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        path = FIGDIR / f"{name}.{ext}"
        fig.savefig(path, metadata={"CreationDate": None} if ext == "pdf" else None)
        out.append(path)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------------
# the seed-aware point renderer -- used by every figure, so the encoding never drifts between them
# --------------------------------------------------------------------------------------------

def seed_series(ax, xs, ys, ns, sds, st, label=None, *, line=True, hollow_all=False,
                ls=None, alpha=1.0):
    """Draw one series with seeds encoded in the marker.

    ``ns`` is the seed count per point and ``sds`` the sample s.d. per point (None where n=1).
    Points with n>=2 get a filled marker and a ±1 s.d. error bar; points with n=1 get a hollow
    marker and no bar. The connecting line is drawn separately from the markers so the two
    marker styles can coexist on one curve.
    """
    colour, marker = st["color"], st["marker"]
    z = st["zorder"]
    if line:
        ax.plot(xs, ys, color=colour, linestyle=(ls if ls is not None else st["ls"]),
                marker="none", zorder=z, label=label, alpha=alpha)
        label = None  # the line already carries the legend entry

    multi = [i for i, n in enumerate(ns) if n >= 2 and not hollow_all]
    single = [i for i, n in enumerate(ns) if n < 2 or hollow_all]
    if multi:
        ax.errorbar([xs[i] for i in multi], [ys[i] for i in multi],
                    yerr=[sds[i] or 0.0 for i in multi], fmt="none", ecolor=colour,
                    elinewidth=1.0, capsize=2.8, zorder=z + 1, alpha=alpha)
        ax.plot([xs[i] for i in multi], [ys[i] for i in multi], color=colour, marker=marker,
                linestyle="none", markersize=5, zorder=z + 1, label=label, alpha=alpha)
        label = None
    if single:
        ax.plot([xs[i] for i in single], [ys[i] for i in single], color=colour, marker=marker,
                linestyle="none", markersize=5.5, markerfacecolor="white", markeredgewidth=1.1,
                zorder=z + 1, label=label, alpha=alpha)


def seed_legend_handles() -> list:
    """The two proxy entries that explain the filled/hollow encoding."""
    return [
        Line2D([], [], color=GREY, marker="o", linestyle="none", markersize=5,
               label="mean of $\\geq$3 seeds, bar = $\\pm$1 s.d."),
        Line2D([], [], color=GREY, marker="o", linestyle="none", markersize=5.5,
               markerfacecolor="white", markeredgewidth=1.1, label="single seed (no bar)"),
    ]


def label_dense(ax, ann_x, row, offsets, colour):
    """Annotate a dense point, anchored past the END of its error bar.

    Anchoring on the marker instead puts the label inside the bar for any point with a real seed
    spread, which is now most of them -- cm=1.3's bar alone is 3.3 pp tall.
    """
    dx, dy, ha, anchor = offsets[row["label"]]
    sd = row["accuracy_std"] or 0.0
    y = row["accuracy"] + (sd if anchor == "above" else -sd)
    ax.annotate(row["label"], (ann_x, y),
                textcoords="offset points", xytext=(dx, dy), ha=ha,
                va="bottom" if anchor == "above" else "top",
                fontsize=6, color=colour, zorder=8)


def curve_series(res: dict, method: str, xkey: str):
    levels = res["curves"][method]["levels"]
    return ([l[xkey] for l in levels], [l["accuracy"] for l in levels],
            [l["n_seeds"] for l in levels], [l["accuracy_std"] for l in levels])


def _dense_family(res: dict, family: str) -> list[dict]:
    """One width sweep, ordered by size.

    cm=1.8 is `base_channels=32, cm=1.8` -- the top end of BOTH sweeps, not a member of one of
    them. It therefore terminates whichever family is being drawn, which is what makes the two
    lines share an endpoint and the crossing between them legible.
    """
    rows = [d for d in res["dense"] if d["family"] == family or d["label"] == "cm=1.8 (source)"]
    return sorted(rows, key=lambda d: d["params"])


def dense_series(res: dict, xkey: str, rows=None):
    rows = sorted(rows if rows is not None else res["dense"], key=lambda d: d[xkey])
    return ([d[xkey] for d in rows], [d["accuracy"] for d in rows],
            [d["n_seeds"] for d in rows], [d["accuracy_std"] for d in rows], rows)


# --------------------------------------------------------------------------------------------
# figure 1 -- accuracy vs parameters
# --------------------------------------------------------------------------------------------

def fig_accuracy_vs_params(res: dict) -> list[Path]:
    src = res["constants"]["source_model"]
    pub = res["constants"]["dcase_published_baseline"]
    stats = res["constants"]["statistics"]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.2, 4.2))

        xs, ys, ns, sds, rows = dense_series(res, "params")
        # Below the smallest exactly-matched level there is no dense model at a pruned size, so
        # the dense line is a segment between base=8 and base=16 rather than a locus of measured
        # models. Shading the region keeps that visible instead of leaving it to the caption.
        smallest_exact = min(m["params_nonzero"] for m in res["margins"]
                             if m["reference_kind"] == "exact")
        ax.axvspan(0, smallest_exact, color=GREY, alpha=0.07, linewidth=0, zorder=0)

        seed_series(ax, xs, ys, ns, sds, STYLE["dense"], label="dense, trained from scratch")
        for r in rows:
            if r["label"] == "cm=1.8 (source)":
                continue
            label_dense(ax, r["params"], r, FIG1_LABEL_OFFSETS, "#333333")

        for method in ("IMP-150", "SNIP", "DSP"):
            x, y, n, sd = curve_series(res, method, "params_nonzero")
            seed_series(ax, x, y, n, sd, STYLE[method],
                        label=f"{method} (n={res['curves'][method]['n_seeds']})")

        ax.plot([src["params"]], [src["accuracy"]], marker="*", markersize=13, color=BLACK,
                markerfacecolor="white", markeredgewidth=1.1, linestyle="none", zorder=9,
                label=f"cm=1.8 source model, {src['accuracy']:.4f}, n=1 (pruned FROM this)")
        ax.axhline(pub["accuracy"], color=GREY, linestyle=(0, (1, 3)), linewidth=1.0, zorder=1)
        ax.annotate(f"DCASE published baseline {pub['accuracy']:.4f} $\\pm$ {pub['std']:.4f} "
                    "\n(organisers' 5 runs -- NOT our source model)",
                    xy=(0.985, pub["accuracy"]), xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points", ha="right", va="bottom",
                    fontsize=6.2, color="#777777", zorder=8)
        ax.annotate("no dense model at\na pruned model's size", xy=(smallest_exact, 0.925),
                    xycoords=("data", "axes fraction"), xytext=(-5, 0),
                    textcoords="offset points", ha="right", va="top", fontsize=6.2,
                    color="#777777", zorder=8)

        _finish(ax, "trainable parameters (non-zero)",
                "macro-average accuracy, dev-test set")
        _title(ax, "Pruned subnetworks versus same-size dense models")
        _param_axis(ax, res)
        ax.set_xlim(3000, 65500)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + seed_legend_handles(), labels + [h.get_label() for h in
                                                            seed_legend_handles()],
                  loc="lower right", frameon=True, framealpha=0.95, borderpad=0.5, fontsize=6.8)

        _footer(fig, -0.06,
                 "Filled markers are means over three or more seeds and the bars are $\\pm$1 "
                 "sample standard deviation; hollow markers are a SINGLE run and carry no bar. "
                 "IMP-150 and the cm=1.8 source are single runs, so their positions are one draw "
                 f"from a distribution whose pooled spread is {stats['pooled_seed_sd_pp']:.2f} pp. "
                 "Dotted line: the organisers' published baseline over THEIR five runs -- "
                 "reproduction evidence, not a comparison point; the star is the source model "
                 "this project actually pruned.\nShaded region: no dense model exists at a pruned "
                 "size there, so the two deepest pruned points are compared against base=16, "
                 "which is larger. Neither the SNIP nor the DSP mean curve is monotonic; read the "
                 "shape, not a point.")
        return save(fig, "fig1_accuracy_vs_params")


# --------------------------------------------------------------------------------------------
# figure 2 -- accuracy vs MACs
# --------------------------------------------------------------------------------------------

def fig_accuracy_vs_macs(res: dict) -> list[Path]:
    src = res["constants"]["source_model"]
    budget = res["constants"]["dcase_max_macs"]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.2, 4.2))

        # The dense references are drawn as TWO families, not one line: on the MACs axis the two
        # width knobs disagree about the ordering, and a single connected line would hide that.
        # cm=1.8 is base_channels=32 at cm=1.8, i.e. the top end of BOTH sweeps, so it terminates
        # both lines -- that shared endpoint is what makes the crossing legible.
        for family, label, colour in (("cm", "dense, channels_multiplier sweep", BLACK),
                                      ("base", "dense, base_channels sweep", ORANGE)):
            rows = _dense_family(res, family)
            st = {"color": colour, "marker": "o", "ls": "-", "zorder": 3}
            x, y, n, sd, rows = dense_series(res, "macs", rows)
            seed_series(ax, x, y, n, sd, st, label=label)
            for r in rows:
                if r["label"] == "cm=1.8 (source)":
                    continue
                label_dense(ax, r["macs"], r, FIG2_LABEL_OFFSETS, colour)

        # DSP: calculated packed-shape MACs; the packing itself is not implemented. Its x
        # position is itself a mean -- a DSP seed re-runs the
        # filter grouping, so different seeds land at different compute costs.
        x, y, n, sd = curve_series(res, "DSP", "macs_executed")
        seed_series(ax, x, y, n, sd, STYLE["DSP"], label="DSP, packed-shape MACs (n=3)")

        # IMP and SNIP execute a constant MAC count, since masking changes no tensor shape,
        # so each is a vertical stack at the dense baseline. The hollow dotted series beside
        # them is the hypothetical sparse-kernel figure, unreachable without sparse kernels.
        for method in ("IMP-150", "SNIP"):
            st = STYLE[method]
            nseeds = res["curves"][method]["n_seeds"]
            x, y, n, sd = curve_series(res, method, "macs_executed")
            seed_series(ax, x, y, n, sd, st, line=False,
                        label=f"{method}, MACs executed (constant, n={nseeds})")
            xh, yh, nh, sdh = curve_series(res, method, "macs_nonzero")
            ax.plot(xh, yh, color=st["color"], marker=st["marker"], linestyle=(0, (1, 2)),
                    markersize=6, markerfacecolor="none", markeredgewidth=1.1, alpha=0.85,
                    zorder=st["zorder"] - 1,
                    label=f"{method}, hypothetical -- unreachable (needs sparse kernels)")

        ax.axvline(budget, color=GREY, linestyle="--", linewidth=1.0, zorder=1)
        ax.annotate(f"DCASE MACs limit {budget / 1e6:.0f}M", xy=(budget, 0.985),
                    xycoords=("data", "axes fraction"), xytext=(-4, 0),
                    textcoords="offset points", ha="right", va="top",
                    fontsize=6.3, color="#777777", zorder=8)
        ax.annotate("IMP and SNIP: all five levels\nsit on this vertical line",
                    xy=(src["macs"], 0.26), xycoords=("data", "axes fraction"), xytext=(-38, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=6.3, color="#555555", zorder=8,
                    arrowprops=dict(arrowstyle="->", color="#888888", linewidth=0.8))
        ax.plot([src["macs"]], [src["accuracy"]], marker="*", markersize=13, color=BLACK,
                markerfacecolor="white", markeredgewidth=1.1, linestyle="none", zorder=9,
                label="cm=1.8 source model, n=1 (both sweeps' top end)")

        _finish(ax, "multiply-accumulate operations per 1 s clip",
                "macro-average accuracy, dev-test set")
        _title(ax, "The compute axis: only structured pruning moves left")
        ax.set_xlim(0, budget * 1.07)
        ax.set_ylim(0.432, 0.522)
        ax.xaxis.set_major_formatter(lambda v, _p: f"{v / 1e6:.0f}M")
        handles, labels = ax.get_legend_handles_labels()
        extra = seed_legend_handles()
        ax.legend(handles + extra, labels + [h.get_label() for h in extra],
                  loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=6.3,
                  frameon=False, columnspacing=1.2, handlelength=2.2, labelspacing=0.3)

        _footer(fig, -0.19,
                 "Filled markers with bars are what the tensor shapes cost: dense execution "
                 "for IMP, SNIP and the dense references, calculated packed shapes for DSP, "
                 "whose packing is not implemented. Averaged over three seeds; IMP's and "
                 "SNIP's sit in a single vertical stack "
                 "because masking changes no tensor shape, and IMP's are hollow because IMP is a "
                 "single run. Hollow, dotted markers are what their surviving weights WOULD cost "
                 "if every zero could be skipped -- unreachable without sparse kernels, which are "
                 "out of scope here, and never to be quoted unlabelled.\nThe two dense families "
                 "cross: base=24 has more parameters than cm=1.0 but far fewer MACs, because cm "
                 "only widens the deep narrow stages where almost no compute lives.")
        return save(fig, "fig2_accuracy_vs_macs")


# --------------------------------------------------------------------------------------------
# figure 3 -- MACs vs parameters
# --------------------------------------------------------------------------------------------

def fig_macs_vs_params(res: dict) -> list[Path]:
    src = res["constants"]["source_model"]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.0, 3.8))

        for family, colour, label, dxy, ha in (
                ("cm", BLACK, "dense, channels_multiplier sweep", (-5, 6), "right"),
                ("base", ORANGE, "dense, base_channels sweep", (6, -11), "left")):
            rows = sorted(_dense_family(res, family), key=lambda d: d["params"])
            ax.plot([r["params"] for r in rows], [r["macs"] for r in rows], color=colour,
                    marker="o", linestyle="-", label=label, zorder=3)
            for r in rows:
                # cm=1.8 terminates both lines; labelling it twice just prints it twice.
                if r["label"] == "cm=1.8 (source)" and family == "base":
                    continue
                ax.annotate(r["label"], (r["params"], r["macs"]), textcoords="offset points",
                            xytext=dxy, ha=ha, fontsize=6, color=colour, zorder=6)

        # No seed encoding on this figure: both axes are architecture, not accuracy. The one place
        # the seed shows is DSP, whose packed-shape MACs differ between seeds because its seed re-runs
        # the filter grouping -- so DSP gets a vertical min/max bar and the others do not.
        for method in ("IMP-150", "SNIP", "DSP"):
            st = STYLE[method]
            levels = res["curves"][method]["levels"]
            xs = [l["params_nonzero"] for l in levels]
            ax.plot(xs, [l["macs_executed"] for l in levels],
                    color=st["color"], marker=st["marker"], linestyle=st["ls"],
                    label=("DSP, packed-shape MACs" if method == "DSP" else f"{method}, executed"), zorder=st["zorder"])
            if res["curves"][method]["structured"]:
                lo = [l["macs_executed"] - min(l["macs_executed_values"]) for l in levels]
                hi = [max(l["macs_executed_values"]) - l["macs_executed"] for l in levels]
                ax.errorbar(xs, [l["macs_executed"] for l in levels], yerr=[lo, hi], fmt="none",
                            ecolor=st["color"], elinewidth=1.0, capsize=2.8,
                            zorder=st["zorder"] + 1)
            else:
                ax.plot(xs, [l["macs_nonzero"] for l in levels],
                        color=st["color"], marker=st["marker"], linestyle=(0, (1, 2)),
                        markerfacecolor="none", markeredgewidth=1.0, alpha=0.8,
                        label=f"{method}, hypothetical sparse", zorder=st["zorder"] - 1)

        lim = max(src["params"], 62000)
        ax.plot([0, lim], [0, src["macs"]], color=GREY, linewidth=0.8,
                linestyle=(0, (4, 3)), zorder=1,
                label="proportional scaling (MACs $\\propto$ params)")

        _finish(ax, "trainable parameters (non-zero)",
                "multiply-accumulate operations per 1 s clip")
        _title(ax, "Parameter count does not determine compute")
        ax.yaxis.set_major_formatter(lambda v, _p: f"{v / 1e6:.0f}M")
        _param_axis(ax, res)
        ax.set_xlim(0, 65500)
        ax.set_ylim(0, res["constants"]["dcase_max_macs"] * 1.02)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=6.4,
                  frameon=False, columnspacing=1.4, handlelength=2.2, labelspacing=0.3)
        dsp_spread = max(l["macs_star_spread_pct"] for l in res["curves"]["DSP"]["levels"])
        _footer(fig, -0.15,
                 f"IMP and SNIP execute a flat {src['macs'] / 1e6:.2f}M MACs at every level -- "
                 "the horizontal line at the top. DSP's packed-shape MACs track the base_channels "
                 f"dense family to within {_dsp_track_dev(res):.1f}% at every level, which is the "
                 "sense in which structured pruning recovers a real architecture rather than a "
                 f"mask; the bars on DSP are its min-max range over three seeds, up to "
                 f"{dsp_spread:.1f}% wide, because a DSP seed re-runs the filter grouping and "
                 "therefore arrives at a different architecture.\nThe diagonal marks what "
                 "proportional scaling would give; every dense point above it spends more compute "
                 "per parameter than the baseline does, every point below it spends less.")
        return save(fig, "fig3_macs_vs_params")


# --------------------------------------------------------------------------------------------
# figure 4 -- IMP retrain budget
# --------------------------------------------------------------------------------------------

def fig_imp_retrain(res: dict) -> list[Path]:
    stats = res["constants"]["statistics"]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.0, 3.9))

        xs, ys, ns, sds, _rows = dense_series(res, "params")
        seed_series(ax, xs, ys, ns, sds, STYLE["dense"], label="dense, trained from scratch")

        for method in ("IMP-20", "IMP-35", "IMP-150"):
            x, y, n, sd = curve_series(res, method, "params_nonzero")
            seed_series(ax, x, y, n, sd, STYLE[method],
                        label=res["curves"][method]["description"] + " (n=1)")

        gaps = next(f for f in res["findings"]
                    if f["id"] == "imp_retrain_need_is_sparsity_dependent")["numbers"]["gaps"]
        deepest = gaps[-1]
        lvl20 = res["curves"]["IMP-20"]["levels"][-1]
        lvl150 = res["curves"]["IMP-150"]["levels"][-1]
        ax.annotate("", xy=(lvl150["params_nonzero"], lvl150["accuracy"]),
                    xytext=(lvl20["params_nonzero"], lvl20["accuracy"]),
                    arrowprops=dict(arrowstyle="<->", color=GREY, linewidth=0.9))
        ax.annotate(f"{deepest['gap_pp']:+.2f} pp", fontsize=7, color="#444444", ha="left",
                    xy=(lvl20["params_nonzero"], (lvl20["accuracy"] + lvl150["accuracy"]) / 2),
                    xytext=(7, 0), textcoords="offset points", va="center")

        _finish(ax, "trainable parameters (non-zero)",
                "macro-average accuracy, dev-test set")
        _title(ax, "IMP's retraining requirement grows with sparsity")
        _param_axis(ax, res)
        ax.set_xlim(3000, 65500)
        handles, labels = ax.get_legend_handles_labels()
        extra = seed_legend_handles()
        ax.legend(handles + extra, labels + [h.get_label() for h in extra],
                  loc="lower right", frameon=True, framealpha=0.95, borderpad=0.5, fontsize=6.6)
        _footer(fig, -0.06,
                 "The three IMP curves are the same pruning criterion at three retraining "
                 "budgets, so the vertical spread between them is compute, not method. It is "
                 f"negligible above {gaps[0]['size_kb']:.0f} KB and worth {deepest['gap_pp']:.2f} "
                 f"pp at {deepest['size_kb']:.2f} KB -- more than twice the "
                 f"{stats['pooled_seed_sd_pp']:.2f} pp pooled seed scatter, which is why this "
                 "comparison survives at one seed.\nAll three IMP curves are hollow because each "
                 "is a SINGLE run; only the dense reference here has seed replication, and its "
                 "bars are $\\pm$1 s.d. where it does. IMP-150 matches SNIP's per-level training "
                 "budget exactly, which is what removes the compute confound from Figure 1.")
        return save(fig, "fig4_imp_retrain_budget")


# --------------------------------------------------------------------------------------------
# figure 5 -- per-seed scatter
# --------------------------------------------------------------------------------------------

def fig_seed_scatter(res: dict) -> list[Path]:
    """Every SNIP and DSP run as its own point, next to the mean the other figures show.

    This is the figure that makes "DSP varies more than SNIP" visible rather than asserted, and
    it is also the honest counterweight to figures 1 and 2: those show five means per method and
    this shows the fifteen numbers each mean is built from.
    """
    # Deterministic horizontal offsets so three runs at the same size do not overplot. In points,
    # applied in display space, so they do not distort the size axis.
    OFFSETS = (-6.0, 0.0, 6.0)

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.6), sharey=True)
        b16 = next(d for d in res["dense"] if d["label"] == "base=16")

        for ax, method in zip(axes, ("SNIP", "DSP")):
            st = STYLE[method]
            levels = res["curves"][method]["levels"]
            xs = [l["size_kb"] for l in levels]

            for i, l in enumerate(levels):
                for k, v in enumerate(l["accuracy_values"]):
                    ax.plot([l["size_kb"]], [v], marker="o", markersize=3.6, color=st["color"],
                            markerfacecolor="white", markeredgewidth=0.9, linestyle="none",
                            zorder=4,
                            transform=ax.transData + matplotlib.transforms.ScaledTranslation(
                                OFFSETS[k % len(OFFSETS)] / 72.0, 0, fig.dpi_scale_trans))
                # the min-max span, so the reader sees the range and not only the s.d.
                ax.plot([l["size_kb"], l["size_kb"]],
                        [min(l["accuracy_values"]), max(l["accuracy_values"])],
                        color=st["color"], linewidth=0.8, alpha=0.45, zorder=3)

            ax.plot(xs, [l["accuracy"] for l in levels], color=st["color"], marker=st["marker"],
                    linestyle=st["ls"], markersize=5.5, zorder=6)
            ax.errorbar(xs, [l["accuracy"] for l in levels],
                        yerr=[l["accuracy_std"] for l in levels], fmt="none", ecolor=st["color"],
                        elinewidth=1.1, capsize=3.0, zorder=6)

            ax.axhline(b16["accuracy"], color=ORANGE, linestyle=(0, (4, 3)), linewidth=1.0,
                       zorder=1)
            # the per-level s.d. printed under each level, so the spread is readable as a
            # number and not only as a bar length
            for l in levels:
                ax.annotate(f"{l['accuracy_spread_pp']:.2f}", (l["size_kb"], 0.025),
                            xycoords=("data", "axes fraction"), ha="center", va="bottom",
                            fontsize=6.2, color=st["color"], zorder=8,
                            # DSP's widest bar reaches the bottom of the axes and would otherwise
                            # be drawn straight through this number
                            bbox=dict(boxstyle="square,pad=0.12", facecolor="white",
                                      edgecolor="none"))

            ax.invert_xaxis()
            ax.set_xlabel("fp16 payload (KiB)")
            ax.set_title(f"{method}, {res['curves'][method]['n_seeds']} seeds -- mean per-level "
                         f"s.d. "
                         f"{sum(l['accuracy_spread_pp'] for l in levels) / len(levels):.2f} pp",
                         pad=6, fontsize=9)

        axes[0].set_ylabel("macro-average accuracy, dev-test set")
        axes[0].annotate("base=16 dense (n=3)", xy=(0.98, b16["accuracy"]),
                         xycoords=("axes fraction", "data"), xytext=(0, 3),
                         textcoords="offset points", ha="right", va="bottom", fontsize=6.0,
                         color=ORANGE, zorder=7)


        _suptitle(fig, "Every individual run behind the means in Figures 1 and 2", y=1.0,
                  fontsize=10)
        fig.tight_layout()
        _footer(fig, -0.02,
                 "The small number under each level is that level's sample s.d. in "
                 "percentage points. Hollow circles are the three individual seeds at each "
                 "level, spread "
                 "horizontally only to stop them overplotting; the thin vertical line is their "
                 "min-max range and the filled marker is the mean with $\\pm$1 sample s.d. Both "
                 "panels share the accuracy axis, so the two spreads are directly comparable. "
                 "DSP's are wider on average and its worst level is far wider than SNIP's worst, "
                 "which is expected: a SNIP seed changes only which weights the mask keeps, while "
                 "a DSP seed re-runs phase A and therefore trains a different architecture.\nNote "
                 "what is NOT visible here: a level at which either method reliably fails. Each "
                 "single-seed curve has one isolated low point and they land at different sizes.")
        return save(fig, "fig5_seed_scatter")


# --------------------------------------------------------------------------------------------
# shared axis furniture
# --------------------------------------------------------------------------------------------

def _dsp_track_dev(res: dict) -> float:
    """Max relative deviation of DSP's packed-shape MACs from the base_channels dense family."""
    f = next(x for x in res["findings"] if x["id"] == "dsp_macs_track_base_channels_family")
    return f["numbers"]["max_abs_rel_dev_pct"]


def _finish(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _param_axis(ax, res: dict) -> None:
    """A secondary top axis in KB, since the challenge budget is stated in bytes, not parameters."""
    ax.xaxis.set_major_formatter(lambda v, _p: f"{v / 1000:.0f}k")
    bytes_per = res["constants"]["fp16_bytes_per_param"]
    sec = ax.secondary_xaxis("top", functions=(lambda p: p * bytes_per / 1024,
                                              lambda kb: kb * 1024 / bytes_per))
    sec.set_xlabel("fp16 payload (KiB)", fontsize=8)
    sec.tick_params(labelsize=7.5)


def make_figures(res: dict) -> list[Path]:
    out = []
    out += fig_accuracy_vs_params(res)
    out += fig_accuracy_vs_macs(res)
    out += fig_macs_vs_params(res)
    out += fig_imp_retrain(res)
    out += fig_seed_scatter(res)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bare", action="store_true",
                    help="omit in-image titles and footers (for figures that get a LaTeX caption)")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="write here instead of reports/figures")
    args = ap.parse_args()

    global BARE, FIGDIR
    BARE = args.bare
    if args.outdir is not None:
        FIGDIR = args.outdir

    path = REPO / "reports" / "results.json"
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)
    for f in make_figures(res):
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
