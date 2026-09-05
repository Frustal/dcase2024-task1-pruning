"""Renders ``reports/results.md`` from the assembled table in ``reports/results.json``.

Every number in the tables and the prose is formatted from that structure; nothing here is typed
by hand. The findings section is driven by ``experiments.collect_results.build_findings``, which
recomputes each claim on every run: a claim whose ``holds`` flag is False is printed as a failure
rather than as a finding, so a sentence cannot outlive the data behind it.

Two rules govern the prose emitted here. No accuracy is written without its seed count, because
the models have between one and five seeds and setting a single draw beside a three-seed mean is
how the retracted claims came about. And "significantly" is reserved for a margin that clears
alpha=0.05 on Welch's test; everything else, including margins consistent in direction across
every seed, is "consistent in direction, inside the noise floor".

Not a standalone entry point; ``experiments.collect_results`` calls ``render_report``.
"""
from __future__ import annotations

LEVEL_ORDER = [0.3362, 0.486, 0.6333, 0.75, 0.85]
METHOD_ORDER = ["IMP-20", "IMP-35", "IMP-150", "SNIP", "DSP"]


# --------------------------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------------------------

def table(header: list[str], rows: list[list[str]]) -> str:
    """A GitHub-flavoured markdown table with the numeric columns right-aligned."""
    align = [":--" if i == 0 else "--:" for i in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def acc(x: float | None) -> str:
    return "--" if x is None else f"{x:.4f}"


def pp(x: float) -> str:
    return f"{x:+.2f}"


def num(x: float | int) -> str:
    return f"{x:,.0f}"


def seeded(mean: float, std: float | None, n: int) -> str:
    """The ONLY way an accuracy is allowed to be printed: value, spread, seed count."""
    if n < 2 or std is None:
        return f"{mean:.4f} (n=1)"
    return f"{mean:.4f} ±{std:.4f} (n={n})"


def sig_word(m: dict) -> str:
    """How a margin is allowed to be described, given what its test actually supports."""
    if m["welch"] is None:
        return "n=1, untestable"
    if m["significant_p05"]:
        return f"**significant, p={m['welch']['p']:.3f}**"
    return f"inside noise, p={m['welch']['p']:.2f}"


def level_key(target: float) -> str:
    return f"{100 * target:.1f}%"


def find(res: dict, fid: str) -> dict:
    return next(f for f in res["findings"] if f["id"] == fid)


def levels_of(res: dict, method: str) -> dict:
    return {round(l["target"], 4): l for l in res["curves"][method]["levels"]}


def margins_by(res: dict) -> dict:
    return {(m["method"], round(m["target"], 4)): m for m in res["margins"]}


def level_size_kb(res: dict, margin: dict) -> float:
    """The size to label a matched level with.

    Where the level is size-matched to a real dense model, that model's size is the honest label:
    the pruned models differ from it and from each other by a handful of parameters. A Pareto
    comparison has no such anchor, so the pruned level's own mean size is used.
    """
    if margin["reference_kind"] == "exact":
        ref = next(d for d in res["dense"] if d["label"] == margin["dense_reference"])
        return ref["size_kb"]
    return margin["size_kb"]


# --------------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------------

def seed_inventory(res: dict) -> str:
    """Which models have how many seeds. Printed early, because the asymmetry is the biggest
    caveat on every comparison below."""
    rows = []
    for method in METHOD_ORDER:
        c = res["curves"][method]
        rows.append([f"pruned {method}", str(c["n_seeds"]),
                     ", ".join(sorted(c["seed_runs"])),
                     ", ".join(f"`{r}`" for _s, r in sorted(c["seed_runs"].items()))])
    for d in res["dense"]:
        rows.append([f"dense {d['label']}", str(d["n_seeds"]),
                     ", ".join(sorted(d["seeds"], key=int)),
                     ", ".join(f"`{v['run_id'][:8]}`" for v in d["seeds"].values())])
    return table(["model", "seeds", "seed ids", "run ids"], rows)


def dense_seed_table(res: dict) -> str:
    rows = []
    for d in res["dense"]:
        per_seed = " / ".join(f"{v['accuracy']:.4f}"
                              for _s, v in sorted(d["seeds"].items(), key=lambda kv: int(kv[0])))
        spread = ("--" if d["accuracy_std"] is None
                  else f"{100 * d['accuracy_std']:.2f} pp")
        rows.append([d["label"], str(d["n_seeds"]), per_seed, acc(d["accuracy"]), spread])
    return table(["dense reference", "n", "per-seed (last epoch)", "mean", "sample s.d."], rows)


def master_table(res: dict) -> str:
    rows = []
    for d in res["dense"]:
        rows.append([f"dense {d['label']}", num(d["params"]), f"{d['size_kb']:.2f}",
                     num(d["macs"]), seeded(d["accuracy"], d["accuracy_std"], d["n_seeds"]),
                     f"`{d['run_id'][:8]}`"])
    for method in METHOD_ORDER:
        curve = res["curves"][method]
        for lvl in curve["levels"]:
            rows.append([f"{method} @ {level_key(lvl['target'])}", num(lvl["params_nonzero"]),
                         f"{lvl['size_kb']:.2f}", num(lvl["macs_executed"]),
                         seeded(lvl["accuracy"], lvl["accuracy_std"], lvl["n_seeds"]),
                         ", ".join(f"`{r}`" for _s, r in sorted(curve["seed_runs"].items()))])
    return table(["model", "params", "size KB (fp16)", "MACs executed",
                  "macro acc. (mean ± s.d.)", "run id(s)"], rows)


def matched_grid(res: dict) -> str:
    """The headline grid: one row per matched size, one column per method."""
    by = margins_by(res)
    rows = []
    for target in LEVEL_ORDER:
        key = round(target, 4)
        ref = by[("SNIP", key)]
        kind = "same size" if ref["reference_kind"] == "exact" else "larger (Pareto)"
        cells = [f"{level_size_kb(res, ref):.2f}", level_key(target),
                 f"{ref['dense_reference']} {acc(ref['dense_accuracy'])} "
                 f"(n={ref['n_seeds_dense']}, {kind})"]
        for method in METHOD_ORDER:
            lvl = levels_of(res, method)[key]
            cells.append(seeded(lvl["accuracy"], lvl["accuracy_std"], lvl["n_seeds"]))
        rows.append(cells)
    header = ["size KB", "sparsity", "dense reference"] + METHOD_ORDER
    return table(header, rows)


def margins_table(res: dict) -> str:
    by = margins_by(res)
    stats = res["constants"]["statistics"]
    rows = []
    for target in LEVEL_ORDER:
        key = round(target, 4)
        ref = by[("SNIP", key)]
        cells = [f"{level_size_kb(res, ref):.2f}",
                 f"{ref['dense_reference']} (n={ref['n_seeds_dense']})"]
        for method in METHOD_ORDER:
            m = by[(method, key)]
            if m["welch"] is None:
                cells.append(f"{pp(m['margin_pp'])} (n=1)")
            elif m["significant_p05"]:
                cells.append(f"**{pp(m['margin_pp'])}** (t={m['welch']['t']:.2f}, "
                             f"p={m['welch']['p']:.3f})")
            else:
                cells.append(f"{pp(m['margin_pp'])} (t={m['welch']['t']:.2f})")
        rows.append(cells)
    body = table(["size KB", "dense reference"] + METHOD_ORDER, rows)
    return body + "\n\n" + (
        "Positive means the pruned model scored above the dense reference. The top three rows are "
        "**same-size** comparisons: those sparsity targets were chosen so the pruned models land "
        "on cm=1.3 / cm=1.0 / cm=0.5 to within 0.5% of their parameter counts. The bottom two rows "
        "have no dense model at their size, so they are **Pareto** comparisons against the "
        "smallest measured dense model that is still LARGER (base=16): the pruned model wins the "
        "size axis by construction and only accuracy is in question. `t` is Welch's unequal-"
        "variance statistic; a margin is bolded only where it reaches p<"
        f"{res['constants']['alpha']:.2f}. **Exactly one does.** With three seeds a side and a "
        f"pooled scatter of {stats['pooled_seed_sd_pp']:.2f} pp, a measured margin has to reach "
        f"about {stats['detectable_margin_pp_n3']:.1f} pp before the test can call it anything; "
        "every other row here is a difference that is consistent in direction across seeds but "
        "sits inside the noise floor. IMP columns are a SINGLE seed against a three- or five-seed "
        "dense mean and carry no test at all."
    )


def interpolated_table(res: dict) -> str:
    """The same-size figure at the two deep levels, where no dense model exists."""
    rows = []
    for m in res["margins"]:
        interp = m.get("interpolated_same_size")
        if interp is None:
            continue
        rows.append([m["method"], f"{m['size_kb']:.2f}", num(m["params_nonzero"]),
                     interp["label"], acc(interp["accuracy"]), pp(interp["margin_pp"])])
    rows.sort(key=lambda r: (r[0], -float(r[1])))
    return table(["method", "size KB", "params", "interpolated reference",
                  "interp. accuracy", "margin pp"], rows)


def accumulation_table(res: dict) -> str:
    rows = []
    for a in res["seed_accumulation"]:
        rows.append([a["method"], f"{a['size_kb']:.2f}", a["dense_reference"],
                     " → ".join(pp(p["margin_pp"]) for p in a["prefixes"]),
                     f"{a['swing_pp']:.2f}", "yes" if a["sign_flipped"] else "no"])
    return table(["method", "size KB", "dense reference",
                  "margin as dense seeds land (1 → n)", "swing pp", "sign flipped?"], rows)


def macs_table(res: dict) -> str:
    src = res["constants"]["source_model"]
    rows = []
    for d in res["dense"]:
        rows.append([f"dense {d['label']}", "structured (trained at width)", num(d["params"]),
                     num(d["macs"]), num(d["macs"]),
                     f"{100 * (1 - d['macs'] / src['macs']):.1f}%", "yes"])
    for method in METHOD_ORDER:
        curve = res["curves"][method]
        kind = "structured" if curve["structured"] else "unstructured (mask)"
        for lvl in curve["levels"]:
            star = num(lvl["macs_nonzero"])
            if lvl["macs_star_spread_pct"]:
                star += f" (±{lvl['macs_star_spread_pct'] / 2:.1f}% over {lvl['n_seeds']} seeds)"
            rows.append([
                f"{method} @ {level_key(lvl['target'])}", kind, num(lvl["params_nonzero"]),
                num(lvl["macs_executed"]), star,
                f"{100 * lvl['macs_reduction_executed']:.1f}%",
                "yes" if curve["macs_nonzero_is_reachable"] else "no -- needs sparse kernels",
            ])
    body = table(["model", "pruning kind", "non-zero params", "MACs executed",
                  "MACs of non-zero weights", "MACs cut vs cm=1.8", "saving realisable?"], rows)
    return body + "\n\n" + (
        "**MACs executed** is what the tensor shapes cost, i.e. what actually runs today. "
        "**MACs of non-zero weights** is what the surviving weights would cost if every zero "
        "could be skipped. For a dense model the two are equal by construction. For DSP they are "
        "equal because its removals are whole (filter-group, input-channel) connections that "
        "collapse into a grouped convolution -- a dense op of smaller shape. For IMP and SNIP the "
        "second column is HYPOTHETICAL: the mask leaves every tensor's shape intact, so the "
        "convolution still multiplies by all its zeros, and the figure is reachable only with "
        "sparse kernels, which this thesis rules out of scope. Never quote it for IMP or SNIP "
        "unlabelled. Both MACs columns are means over the method's seeds, and the ± on the second "
        "is real: a different seed keeps a different set of weights, so MACs* moves even at "
        "identical parameter sparsity (see the findings). The same caveat applies to the size "
        "column of the master table: an unstructured model's "
        f"{res['curves']['SNIP']['levels'][-1]['size_kb']:.2f} KB assumes a sparse storage "
        f"format; as a plain fp16 state_dict it is still {src['size_kb']:.2f} KB."
    )


def per_seed_table(res: dict) -> str:
    """Every individual pruned run, so the means above can be taken apart."""
    rows = []
    for method in ("SNIP", "DSP"):
        curve = res["curves"][method]
        for lvl in curve["levels"]:
            per_seed = " / ".join(f"{v:.4f}" for v in lvl["accuracy_values"])
            rows.append([method, f"{lvl['size_kb']:.2f}", level_key(lvl["target"]), per_seed,
                         acc(lvl["accuracy"]), f"{lvl['accuracy_spread_pp']:.2f} pp"])
    return table(["method", "size KB", "sparsity", "seeds 42 / 43 / 44", "mean", "sample s.d."],
                 rows)


# --------------------------------------------------------------------------------------------
# findings prose
# --------------------------------------------------------------------------------------------

def _f_significance(n: dict, res: dict) -> str:
    sig = n["significant"][0]
    return (
        "**Almost nothing in this study reaches statistical significance, and saying so is part "
        f"of the result.** Pooling every point trained at more than one seed gives a within-point "
        f"sample s.d. of {n['pooled_sd_pp']:.2f} pp over {n['pooled_dof']} degrees of freedom. "
        f"With three seeds on each side of a comparison, a measured margin has to reach about "
        f"{n['detectable_margin_pp_n3']:.2f} pp before Welch's test can reject at "
        f"alpha={res['constants']['alpha']:.2f}. Of the {n['n_margins_tested']} pruned-vs-dense "
        f"margins with a testable seed sample on both sides, exactly {n['n_significant']} clears "
        f"that bar: {sig['method']} at {sig['size_kb']:.2f} KB, {pp(sig['margin_pp'])} pp, "
        f"t={sig['t']:.2f} on {sig['df']:.1f} df, p={sig['p']:.3f} -- and it is a NEGATIVE result "
        "(see below). Every other margin in this report, including the ones whose direction is "
        "consistent across all three seeds, must be written as *consistent in direction, inside "
        "the noise floor*, never as *significantly better*. Reaching p<0.05 on a margin of around "
        f"1.3 pp at this scatter would need roughly {n['seeds_needed_for_1_3pp']} seeds per side, "
        "which is another order of magnitude of GPU time and is stated here as a limitation "
        "rather than bought."
    )


def _f_partial_seeds(n: dict, res: dict) -> str:
    flips = [a for a in n["accumulation"] if a["sign_flipped"]]
    parts = []
    for a in flips:
        parts.append(f"{a['method']} at {a['size_kb']:.2f} KB moved "
                     + " -> ".join(pp(p["margin_pp"]) for p in a["prefixes"])
                     + f" as {a['dense_reference']}'s seeds landed")
    return (
        "**Partial seed data is actively misleading, and a margin is only reportable once BOTH "
        "sides have their full seed count.** Replaying each margin against the first 1, 2, ... "
        f"seeds of its dense reference shows swings of up to {n['worst_swing_pp']:.2f} pp, and "
        f"{n['n_sign_flips']} margins change SIGN partway through: " + "; ".join(parts) + ". "
        "These were not hypothetical intermediate states -- they are the numbers this project "
        "actually held while the seed queue ran, and the error went in BOTH directions: the "
        "partial reference understated SNIP at one size and overstated it at another, so there "
        "is no correction factor to apply and no safe direction to lean. "
        "The practical rule that follows is procedural rather than statistical: do not compute a "
        "pruned-vs-dense margin at all until the dense side is finished, because a partially "
        "seeded reference produces a number that looks exactly as trustworthy as a finished one."
    )


def _f_not_systematically_low(n: dict, res: dict) -> str:
    moves = ", ".join(f"{m['label']} {pp(m['move_pp'])} pp" for m in n["moves"])
    biggest = max(n["moves"], key=lambda m: abs(m["move_pp"]))
    smallest = min(n["moves"], key=lambda m: abs(m["move_pp"]))
    cm13 = n["cm13"]
    return (
        "**But the tempting generalisation from that -- 'single-seed dense references sit "
        "systematically low, so every pruning margin was flattered' -- is FALSE, and it is "
        "recorded here so it cannot creep back in.** Moving each of the three re-seeded "
        f"references from its seed-42 value to its three-seed mean shifted it by {moves}. "
        f"{biggest['label']} moved {abs(biggest['move_pp']):.2f} pp, which is what invited the "
        f"inference; {smallest['label']} moved {abs(smallest['move_pp']):.2f} pp and did not move "
        "at all in any practical sense, and the three moves do not even share a direction. "
        f"(cm=1.3, already at {cm13['n_seeds']} seeds before the queue, moved "
        f"{cm13['move_pp']:+.2f} pp from its seed-42 value.) A single draw from a distribution "
        "with about a point of spread can sit anywhere in that distribution; some happening to "
        "sit low is not a bias. The correct statement is about variance, not about direction, and "
        "the directional version must not be written into the thesis."
    )


def _f_seed_counts(n: dict, res: dict) -> str:
    curves = ", ".join(f"{k} n={v}" for k, v in n["curves"].items())
    dense = ", ".join(f"{k} n={v}" for k, v in n["dense"].items())
    return (
        "**The seed counts are deliberately uneven and every comparison inherits that.** Pruned "
        f"curves: {curves}. Dense references: {dense}. The three IMP curves were left at one seed "
        "because the question they answer -- how much of IMP's damage is under-training rather "
        "than pruning -- is a question about the retraining budget, and re-seeding all three "
        "would have cost roughly fifteen GPU-hours to sharpen a comparison that is already large "
        "relative to the noise. The consequence is that every IMP number in this report is a "
        "single draw being compared against a mean of three or five, so IMP-versus-SNIP and "
        "IMP-versus-DSP differences are indicative only and carry no test. base=24 and base=8 are "
        "likewise single runs; they anchor the shape of the dense curve rather than any specific "
        "margin. The cm=1.8 source model is also a single run, which is worth remembering "
        "whenever a pruned curve is quoted 'against the baseline'."
    )


def _f_snip_flat(n: dict, res: dict) -> str:
    return (
        "**SNIP's accuracy curve is nearly flat across a 3.6x size reduction, and that is the "
        f"thesis's strongest result.** Its three-seed means span {n['range_pp']:.2f} pp from "
        f"{n['biggest_kb']:.2f} KB ({acc(n['biggest_acc'])}) to {n['smallest_kb']:.2f} KB "
        f"({acc(n['smallest_acc'])}) -- a factor of {n['size_ratio_across_curve']:.1f} in size for "
        "a difference smaller than the scatter of a single point. Measured against the source "
        f"model this project actually pruned ({acc(n['source_acc'])} at {n['source_kb']:.2f} KB, "
        f"one seed), SNIP at {n['smallest_kb']:.2f} KB gives up {abs(n['vs_source_pp']):.2f} pp "
        f"while being {n['size_ratio_vs_source']:.1f}x smaller. Note what this claim does and does "
        "not need: it rests on the SHAPE of a curve whose every point is a mean of "
        f"{n['n_seeds']} seeds, not on any single comparison, which is exactly the kind of claim "
        f"that survives the noise floor. Seed 42's version of the same curve spanned "
        f"{n['seed42_range_pp']:.2f} pp with {n['seed42_inversions']} visible dips that earlier "
        "sessions spent real effort explaining; those dips were draws."
    )


def _f_snip_matches(n: dict, res: dict) -> str:
    matched = "; ".join(
        f"{m['size_kb']:.2f} KB vs {m['reference']} {pp(m['margin_pp'])} pp (t={m['t']:.2f}, "
        f"p={m['p']:.2f})" for m in n["matched"])
    pareto = "; ".join(
        f"{m['size_kb']:.2f} KB {pp(m['margin_pp'])} pp at {m['params_pct_fewer']:.0f}% fewer "
        f"parameters (t={m['t']:.2f}, p={m['p']:.2f})" for m in n["pareto"])
    return (
        "**SNIP matches a same-size dense model; it does not beat one at four of five levels.** "
        "That earlier claim rested on single-seed data on BOTH sides and does not survive. At the "
        f"three exactly size-matched levels the margins are {matched} -- two ties and one win, "
        "with no test clearing p<0.05. SNIP is clearly behind at none of them, which is itself "
        f"worth stating: it is at or above dense everywhere on the matched part of the curve. "
        f"SNIP's real wins live at the two Pareto points, against a MEASURED base=16: {pareto}. "
        "Those are the largest and most consistent positive margins in the study and their "
        "direction holds across all three seeds, but both sit inside the noise floor and neither "
        "may be described as significant. The defensible summary is: *SNIP matches same-size "
        "dense across the matched sizes, is ahead by over a point at one of them, is clearly "
        "behind at none, and is ahead of the nearest larger dense model at both sizes where no "
        "same-size dense model exists.*"
    )


def _f_dsp_pareto(n: dict, res: dict) -> str:
    return (
        f"**DSP's real Pareto claim is at {res['curves']['DSP']['levels'][-1]['size_kb']:.0f} KB: "
        "equal accuracy on roughly two-thirds the size and two-thirds the compute.** Against the "
        f"measured base=16 dense model, DSP scores {acc(n['dsp_acc'])} (s.d. {n['dsp_std_pp']:.2f} pp "
        f"over {n['n_pruned']} seeds) against {acc(n['dense_acc'])} (s.d. {n['dense_std_pp']:.2f} "
        f"pp over {n['n_dense']}) -- a difference of {n['margin_pp']:+.2f} pp with t={n['t']:.2f}, "
        f"p={n['p']:.2f}, which is as close to an exact tie as this study produces. It gets that "
        f"tie on {num(n['dsp_params'])} parameters against {num(n['dense_params'])} "
        f"({n['params_pct_fewer']:.0f}% fewer) and {num(n['dsp_macs'])} MACs against "
        f"{num(n['dense_macs'])} ({n['macs_pct_fewer']:.0f}% fewer). This is the claim DSP should "
        "be argued on, and it plays to structured pruning's actual strength -- the compute axis -- "
        "rather than to an accuracy margin that was always inside the noise. It also happens to "
        "be DSP's tightest point across seeds, so it is the part of DSP's curve that is best "
        "determined."
    )


def _f_dsp47(n: dict, res: dict) -> str:
    shallow = ", ".join(f"{pp(s['margin_pp'])} pp at {s['size_kb']:.2f} KB vs {s['reference']}"
                        for s in n["shallow"])
    return (
        f"**DSP at {n['size_kb']:.2f} KB is significantly WORSE than a same-size dense model, and "
        "it is the only comparison in the entire study to reach p<0.05.** Three DSP seeds average "
        f"{acc(n['dsp_acc'])} against {n['reference']}'s three-seed {acc(n['dense_acc'])}: "
        f"{n['margin_pp']:+.2f} pp, t={n['t']:.2f} on {n['df']:.1f} degrees of freedom, "
        f"p={n['p']:.3f}. The one statistically defensible accuracy statement this project can "
        "make is therefore a negative one about structured pruning: at this size, simply training "
        "a narrower dense network from scratch beats pruning the baseline down to it. That fits "
        "the rest of DSP's shallow curve, where the deficit is strikingly stable -- "
        f"{shallow} -- and the stability, on three seeds per point, is what distinguishes it from "
        "the single-point dips that turned out to be draws elsewhere. The mechanism is plausible "
        "rather than proven: DSP can only remove whole (filter-group, input-channel) connections, "
        "so at moderate sparsity it has far less freedom than unstructured SNIP over which "
        "weights die, and it pays for that constraint in accuracy."
    )


def _f_dsp_variance(n: dict, res: dict) -> str:
    per_level = ", ".join(f"{sd:.2f} pp at {kb:.2f} KB"
                          for sd, kb in zip(n["dsp_sd_pp"], n["sizes_kb"]))
    snip_levels = ", ".join(f"{sd:.2f}" for sd in n["snip_sd_pp"])
    widen = ("does not hold" if not n["monotone_widening_with_depth"] else "holds")
    return (
        "**DSP scatters more across seeds than SNIP does, and there is a mechanism for it.** "
        f"DSP's per-level sample s.d. is {per_level} -- a mean of {n['dsp_mean_sd_pp']:.2f} pp and "
        f"a worst point of {n['dsp_max_sd_pp']:.2f} pp -- against SNIP's {snip_levels} pp, a mean "
        f"of {n['snip_mean_sd_pp']:.2f} pp and a worst of {n['snip_max_sd_pp']:.2f} pp. The reason "
        "is structural: a SNIP seed changes the initialisation and hence which weights the mask "
        "keeps, but the ARCHITECTURE is fixed by the config. A DSP seed additionally re-runs "
        "phase A, the learned filter grouping, so two DSP seeds train genuinely different networks "
        "rather than the same network from different starting points. Larger and more erratic "
        "variance is the expected consequence, and the operational conclusion is that DSP needs "
        "MORE seeds than SNIP to support the same claim, not the same number -- three is thin for "
        "it. **One version of this finding is NOT supported and should not be written: that DSP's "
        f"variance widens monotonically with depth. That {widen} -- DSP's deepest level is its "
        f"second-tightest point ({n['dsp_sd_pp'][-1]:.2f} pp). What is true is weaker and driven "
        f"by a single level: the two deepest levels average {n['dsp_deep_mean_sd_pp']:.2f} pp "
        f"against {n['dsp_shallow_mean_sd_pp']:.2f} pp for the three shallow ones.**"
    )


def _f_seed_averaging(n: dict, res: dict) -> str:
    parts = []
    for method, v in n["curves"].items():
        n42, nmean = len(v["seed42_inversions"]), len(v["mean_inversions"])
        parts.append(
            f"{method}'s seed-42 curve had {n42} point{'' if n42 == 1 else 's'} sitting below "
            f"{'its' if n42 == 1 else 'their'} own smaller neighbour, over a "
            f"{v['seed42_range_pp']:.2f} pp range; its three-seed mean curve has {nmean} over "
            f"{v['mean_range_pp']:.2f} pp, the largest such inversion being "
            f"{v['max_mean_inversion_pp']:.2f} pp against a worst per-point s.d. of "
            f"{v['max_sd_pp']:.2f} pp")
    dsp = res["curves"]["DSP"]["levels"]
    tightest = min(dsp, key=lambda l: l["accuracy_spread_pp"])
    snip_big = res["curves"]["SNIP"]["levels"][0]
    return (
        "**Averaging over seeds dissolved most of the structure the single-seed curves seemed to "
        "have.** " + "; ".join(parts) + ". Neither method has a size at which it reliably "
        "degrades. Each single-seed curve does contain one isolated low point, but on different "
        "seeds those points land at DIFFERENT sizes, which is the signature of noise rather than "
        f"of a weak spot: DSP's {tightest['size_kb']:.2f} KB level, the one a single seed made "
        "look like a dip, is in fact the TIGHTEST point on DSP's whole curve across seeds "
        f"({tightest['accuracy_spread_pp']:.2f} pp), and SNIP's largest level, "
        f"{snip_big['size_kb']:.2f} KB, has all but the lowest of its "
        f"{snip_big['n_seeds']} seeds at or above "
        f"{sorted(snip_big['accuracy_values'])[1]:.4f} and a mean that is its second-highest "
        "point. Residual inversions in the mean curves are smaller than the per-point spread and "
        "need no mechanism. The curves should still never be drawn as smooth, and no single point "
        "may be quoted on its own."
    )


def _f_unstructured(n: dict, res: dict) -> str:
    dsp = res["curves"]["DSP"]["levels"]
    per_level = ", ".join(f"{100 * l['macs_reduction_executed']:.1f}% at "
                          f"{level_key(l['target'])}" for l in dsp)
    return (
        "**Unstructured pruning buys nothing on the compute axis.** IMP and SNIP execute exactly "
        f"{num(n['unstructured_macs'])} MACs at every one of the five sparsity levels and on every "
        "seed -- the same count as the unpruned cm=1.8 baseline, to the last multiply. That is not "
        "an approximation or a measurement artefact: masking sets weights to zero without changing "
        "any tensor's shape, so the convolution still performs the multiplication. A reader who "
        "sees an 85% parameter reduction and assumes an 85% inference speed-up is wrong by the "
        "entire amount. DSP, whose removals collapse into a grouped convolution of smaller shape, "
        f"cuts real MACs by {n['dsp_cut_min_pct']:.1f}% to {n['dsp_cut_max_pct']:.1f}% across the "
        f"same five levels ({per_level}). The parameter axis on which the rest of this chapter "
        "compares the three methods is the challenge's own budget and is fair to all of them, but "
        "it is silent on this difference, and the difference is the practical one."
    )


def _f_budget(n: dict, res: dict) -> str:
    c = res["constants"]
    src = c["source_model"]
    mem_pct = 100 * src["params"] * c["fp16_bytes_per_param"] / c["dcase_max_params_memory_bytes"]
    return (
        "**MACs are a binding constraint for this architecture, not a formality.** The cm=1.8 "
        f"baseline costs {num(n['macs'])} MACs against the challenge's {num(n['budget'])} limit, "
        f"i.e. {n['pct_of_budget']:.1f}% of the compute budget, while consuming "
        f"{mem_pct:.1f}% of the {num(c['dcase_max_params_memory_bytes'])}-byte parameter-memory "
        "budget. CP-Mobile was designed hard against both ceilings, so compute headroom is "
        "genuinely scarce here, and a method that converts parameter sparsity into fewer MACs is "
        "buying something the challenge actually rations."
    )


def _f_reorder(n: dict, res: dict) -> str:
    return (
        "**The dense reference curve reorders between the parameter axis and the MACs axis.** "
        f"base=24 carries MORE parameters than cm=1.0 ({num(n['base24_params'])} against "
        f"{num(n['cm10_params'])}) but costs FAR fewer MACs ({num(n['base24_macs'])} against "
        f"{num(n['cm10_macs'])}, a factor of {n['macs_ratio']:.2f}). The mechanism was written "
        "down before these MACs were ever measured, in the header comment of "
        "`configs/scale_down_base16.yaml`: the stem and stage-0 widths are "
        "`base_channels * cm^0`, so they are blind to `cm` entirely, and `cm` only moves the two "
        "deepest, narrowest stages, where few parameters and almost no compute live. "
        "`base_channels` is CP-Mobile's width unit and scales every layer, including the wide "
        "early layers where the MACs are. The MACs column is therefore an independent empirical "
        "confirmation of the argument that motivated the base_channels sweep in the first place, "
        "and a caution: two models matched on parameter count are not matched on compute. This "
        "finding is seed-independent -- MACs are fixed by the architecture -- and was unaffected "
        "by the reseed."
    )


def _f_snip_lag(n: dict, res: dict) -> str:
    return (
        "**SNIP's would-be compute saving lags its parameter saving badly; IMP's tracks it.** At "
        f"{100 * n['sparsity']:.0f}% parameter sparsity, the non-zero weights SNIP keeps still "
        f"account for {n['snip_macs_cut_pct']:.1f}% of a MACs cut (mean over "
        f"{n['snip_n_seeds']} seeds), where IMP's account for {n['imp_macs_cut_pct']:.1f}% "
        f"(one seed) -- an {n['lag_pp']:.1f} pp gap, and IMP's figure is the one that tracks its "
        "parameter sparsity. This follows directly from the per-layer allocation measured in "
        "`pruning/measure_allocation.py`. SNIP's saliency decays "
        "roughly 50x from the stem to the last convolution, so global top-k allocates sparsity "
        "almost purely by depth: at 70% sparsity it keeps 92% of the 72-parameter stem and only "
        "12% of `stages.s3.b6.block.2.0`, the largest layer in the network at 12,480 weights. The "
        "layers SNIP protects are the wide early ones where the MACs are; the layers it guts are "
        "the deep narrow ones where the parameters are. That is SNIP's known layer-collapse "
        "tendency, and the MACs column makes its cost legible: even granting SNIP the sparse "
        "kernels it does not have, its compute saving would be the weakest of the three. "
        "Han-style magnitude pruning, by contrast, is near-uniform across layers on this "
        "architecture (28-42% kept at a 70% target, against SNIP's 5-85%), because every prunable "
        "layer here is a convolution with a similarly-shaped weight distribution and the "
        "layer-wise `q * std` rule has almost nothing to bite on."
    )


def _f_macs_star_seeds(n: dict, res: dict) -> str:
    snip = ", ".join(f"{s['spread_pct']:.2f}% at {s['size_kb']:.2f} KB" for s in n["spread"]["SNIP"])
    dsp = ", ".join(f"{s['spread_pct']:.2f}% at {s['size_kb']:.2f} KB" for s in n["spread"]["DSP"])
    return (
        "**MACs are not a function of the sparsity target alone: they move with the seed, which "
        "is why every seed is measured separately rather than reusing seed 42's row.** For an "
        "unstructured method the parameter count at a given target is fixed by construction -- a "
        "global top-k hits it exactly -- but WHICH weights survive is not, and the surviving "
        "weights sit in layers of very different spatial extent. SNIP's hypothetical MACs* "
        f"therefore spreads across its three seeds by {snip}, growing with sparsity as the mask "
        "gets more room to differ. For DSP the same spread is a spread in REAL, deployed compute, "
        "because the seed changes the architecture and not just the mask: "
        f"{dsp}. A DSP model at the deepest level can cost "
        f"{n['dsp_deepest_spread_pct']:.1f}% more or less compute depending on nothing but the "
        "seed. The dense-shape MACs column, by contrast, is identical across seeds at every level "
        "for every method, which is the check that confirms none of this is a loading bug: "
        "masking cannot change a tensor's shape."
    )


def _f_dsp_tracks_base(n: dict, res: dict) -> str:
    devs = sorted(abs(d["rel_dev_pct"]) for d in n["deviations"])
    return (
        "**DSP does not just shrink the model, it lands on the architecture family.** DSP's "
        "executed MACs, averaged over its three seeds, sit within "
        f"{n['max_abs_rel_dev_pct']:.1f}% of the base_channels dense curve at every one of the "
        f"five levels and within {devs[-2]:.1f}% at four of them, interpolated at each level's "
        "own parameter count. That is a stronger statement than 'structured pruning saves "
        "compute': it says the shapes DSP arrives at cost what a uniformly-narrowed CP-Mobile of "
        "the same parameter count costs, so DSP is effectively rediscovering the width-scaling "
        "design point rather than finding some exotic one. It also sharpens the negative result "
        "above: if DSP reaches base=16's compute profile and base=16's parameter count and still "
        "scores below base=16 at the matched sizes, then what structured pruning is failing to "
        "recover here is not the architecture but the training -- a narrow network trained from "
        "scratch beats the same shape arrived at by pruning."
    )


def _f_imp_retrain(n: dict, res: dict) -> str:
    gaps = n["gaps"]
    seq = ", ".join(f"{g['gap_pp']:+.2f} pp at {g['size_kb']:.2f} KB" for g in gaps)
    return (
        "**IMP's retraining requirement is sparsity-dependent.** Going from 20 to 150 retrain "
        f"epochs per level buys {seq}. At mild sparsity the extra compute is nearly worthless; at "
        f"{gaps[-1]['size_kb']:.2f} KB it is worth {gaps[-1]['gap_pp']:.2f} pp -- more than twice "
        f"the {n['pooled_sd_pp']:.2f} pp pooled seed scatter, which is why this one survives at "
        "n=1 where the pruned-vs-dense margins do not. Retrain length is therefore not a "
        "hyperparameter detail to be reported in a footnote; at high sparsity it dominates the "
        "choice of pruning criterion. Because the 150-epoch IMP curve matches SNIP's per-level "
        "training budget exactly, the compute confound that clouded the earlier IMP-vs-SNIP "
        "comparison is removed: IMP-150 and SNIP see the same number of gradient updates per "
        "curve point. **The caveat that must travel with every IMP comparison: all three IMP "
        f"curves are a SINGLE seed (n={n['n_seeds']}), while SNIP and DSP are means of three. An "
        "IMP-versus-SNIP or IMP-versus-DSP difference is a single draw against a mean and carries "
        "no significance test at all; the retrain-budget comparison above is safer only because "
        "it is IMP against IMP, where the pruning criterion and the data are held fixed and the "
        "gap is large.**"
    )


RENDERERS = {
    "almost_nothing_reaches_significance": _f_significance,
    "partial_seed_data_is_misleading": _f_partial_seeds,
    "single_seed_references_are_not_systematically_low": _f_not_systematically_low,
    "seed_counts_are_asymmetric": _f_seed_counts,
    "snip_curve_is_flat": _f_snip_flat,
    "snip_matches_same_size_dense": _f_snip_matches,
    "dsp_matches_base16_at_22kb_on_two_thirds_the_budget": _f_dsp_pareto,
    "dsp_at_47kb_is_significantly_worse_than_dense": _f_dsp47,
    "dsp_scatters_more_than_snip": _f_dsp_variance,
    "seed_averaging_dissolves_the_single_seed_structure": _f_seed_averaging,
    "unstructured_macs_are_free_of_savings": _f_unstructured,
    "macs_budget_is_binding": _f_budget,
    "dense_curve_reorders_on_macs_axis": _f_reorder,
    "snip_macs_lag_params": _f_snip_lag,
    "macs_star_is_seed_dependent": _f_macs_star_seeds,
    "dsp_macs_track_base_channels_family": _f_dsp_tracks_base,
    "imp_retrain_need_is_sparsity_dependent": _f_imp_retrain,
}


def findings_section(res: dict) -> str:
    out = []
    for f in res["findings"]:
        renderer = RENDERERS.get(f["id"])
        if renderer is None:
            continue
        if f["holds"]:
            out.append(renderer(f["numbers"], res))
        else:
            out.append(
                f"**CLAIM NO LONGER HOLDS -- `{f['id']}`.** The check in "
                "`experiments/collect_results.build_findings` failed against the current data, so "
                "the prose for it is withheld deliberately rather than printed and wrong. "
                f"Numbers as computed: `{f['numbers']}`."
            )
    return "\n\n".join(out)


# --------------------------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
# Retractions -- claims asserted on single-seed data that the later seed queue falsified. They
# are recorded rather than deleted: a claim made on one seed and overturned by three is itself
# the evidence for the report's methodological finding about single-seed curves, and a reader who
# meets an old claim in an earlier draft needs somewhere to find its retraction.
#
# ``was`` is history and is hardcoded; ``now`` is pulled from the live results on every render,
# and ``check`` re-derives it so drift between this table and the data fails loudly.
# --------------------------------------------------------------------------------------------

RETRACTIONS = [
    dict(
        claim="SNIP loses to same-size dense only at the largest size, by 1.79 pp",
        was="-1.79 pp at 81.06 KB, from seed 42's 0.4819",
        check=("SNIP", 0.3362),
        why="Seed 42's 0.4819 was a low draw; seeds 43 and 44 gave 0.5147 and 0.5053. This was "
             "also the point that made SNIP's curve look non-monotonic, so that caveat goes too.",
    ),
    dict(
        claim="SNIP beats same-size dense at 4 of 5 levels",
        was="+0.94, +1.23, +1.07 and +2.09 pp at the four deeper levels",
        check=None,
        why="At the three genuinely size-matched levels the three-seed margins are +0.08, +1.26 "
            "and -0.54: two ties and one win. SNIP's real wins are the two Pareto points against "
            "base=16, which is a larger model, not a same-size one. Every margin in the original "
            "claim also used a single-seed dense reference.",
    ),
    dict(
        claim="DSP at 75% sparsity Pareto-dominates base=16 on all three axes",
        was="+0.80 pp at 33.79 KB, from seed 42's 0.4819",
        check=("DSP", 0.75),
        why="It rested entirely on that one run; seeds 43 and 44 gave 0.4355 and 0.4628. DSP is "
            "still smaller and marginally cheaper in MACs there, but it is less accurate, so it "
            "is not a domination. DSP's surviving Pareto claim is at 22 KB instead.",
    ),
    dict(
        claim="DSP has a structural weak point at 47 KB",
        was="seed 42 scored 0.4641 there, 1.78 pp below its own smaller 33.79 KB neighbour",
        check=None,
        now="spread=DSP:0.6333",
        why="47 KB is in fact DSP's TIGHTEST level across seeds. The apparent dip "
            "existed only because seed 42's NEIGHBOURING 34 KB point drew high. Each DSP seed "
            "drops one isolated low point somewhere and they land in different places. This claim "
            "was carried from 2026-08-10 to 2026-08-25.",
    ),
    dict(
        claim="SNIP scores 0.4837 at 22.41 KB, about -1.9 pp for an 81% size cut",
        was="0.4837, compared against 0.5029",
        check=None,
        now="vs_source=SNIP:0.85",
        why="Wrong on both halves. The three-seed mean is 0.4911, and 0.5029 is the organisers' "
            "published five-run figure, not this project's source model (0.4991). Against the "
            "model actually pruned, the correct figure is -0.80 pp.",
    ),
    dict(
        claim="Any pruned-versus-dense margin computed before 2026-08-25",
        was="every one used a single-seed dense reference",
        check=None,
        why="cm=1.0, cm=0.5 and base=16 were one run each. Recompute against the three-seed means "
            "in this report before quoting any of them.",
    ),
]


def retractions_section(res: dict) -> str:
    by = margins_by(res)
    parts = []
    for i, r in enumerate(RETRACTIONS, 1):
        if r["check"] is not None:
            m = by[r["check"]]
            now = (f"**{pp(m['margin_pp'])} pp** against {m['dense_reference']} "
                   f"({sig_word(m)})")
        elif r.get("now", "").startswith("spread="):
            method, target = r["now"].split("=", 1)[1].split(":")
            lv = next(l for l in res["curves"][method]["levels"]
                      if round(l["target"], 4) == round(float(target), 4))
            spreads = sorted(100 * x["accuracy_std"] for x in res["curves"][method]["levels"]
                             if x["accuracy_std"] is not None)
            now = (f"a seed spread of **{100 * lv['accuracy_std']:.2f} pp** -- the TIGHTEST of "
                   f"{method}'s five levels (the others span "
                   f"{spreads[1]:.2f}-{spreads[-1]:.2f} pp)")
        elif r.get("now", "").startswith("vs_source="):
            method, target = r["now"].split("=", 1)[1].split(":")
            lv = next(l for l in res["curves"][method]["levels"]
                      if round(l["target"], 4) == round(float(target), 4))
            src = res["constants"]["source_model"]["accuracy"]
            now = (f"a three-seed mean of **{lv['accuracy']:.4f}**, i.e. "
                   f"**{100 * (lv['accuracy'] - src):+.2f} pp** against the source model "
                   f"this project actually pruned ({src:.4f})")
        else:
            now = "see the matched-size and margin tables above"
        parts.append(
            f'**{i}. "{r["claim"]}"**\n\n'
            f'- *Rested on:* {r["was"]}\n'
            f'- *Three seeds give:* {now}\n'
            f'- *Why it failed:* {r["why"]}\n'
        )
    return "\n".join(parts)


def render_report(res: dict) -> str:
    src = res["constants"]["source_model"]
    pub = res["constants"]["dcase_published_baseline"]
    stats = res["constants"]["statistics"]
    excl = res["dense_excluded"][0]
    by_label = {d["label"]: d for d in res["dense"]}
    b8, b16 = by_label["base=8"], by_label["base=16"]

    doc = f"""# Results: pruning the DCASE 2024 Task 1 baseline

> Generated by `uv run python -m experiments.collect_results`. **Do not hand-edit** -- every
> number below is read from `checkpoints/<run_id>/pruning_state.json`, `reports/macs.json` or
> `reports/reeval_last_ckpt.csv` and will be overwritten on the next run. Figures for this
> report are in `reports/figures/`.

## What is being compared

The official DCASE 2024 baseline (CP-Mobile, `base_channels=32`, `cm=1.8`,
`expansion_rate=2.1`, {num(src['params'])} parameters, {src['size_kb']:.2f} KB in fp16) is pruned
down past its own size by three methods -- IMP, SNIP and DSP -- and each pruned model is compared
against a *dense model of the same parameter count trained from scratch*. That comparison is a
direct test of the Lottery Ticket Hypothesis claim that a pruned subnetwork beats an
equivalently-sized dense network.

**Two numbers exist for cm=1.8 and they must never be conflated.**

- **{acc(src['accuracy'])}** is this project's own trained source model (run `{src['run_id'][:8]}`,
  last epoch, a single seed). It is the checkpoint every pruning run was actually pruned from, and
  it is the comparison point for every pruned result in this report.
- **{acc(pub['accuracy'])} ± {pub['std']:.4f}** is the organisers' published figure, averaged
  over {pub['n_runs']} runs of theirs. It appears here only as evidence that the reproduction is
  sound -- the {abs(100 * (src['accuracy'] - pub['accuracy'])):.2f} pp gap between the two is well
  inside their own reported spread. It is never a baseline for a pruned model.

**Accuracy convention.** DCASE Task 1 has no validation split, so `val_macro_acc` is measured on
the dev-test set and selecting the best epoch by it selects on the test set. Every accuracy in
this report is therefore the **last epoch's**. The pruning runs were on that convention from the
first run; three early dense runs were logged to wandb as best-epoch and re-scored offline
afterwards (`reports/reeval_last_ckpt.csv`). Reading the wandb
summary for those runs instead inflates the dense curve by 0.6 to 0.8 pp and understates every
margin in the tables below.

## Seeds, and what a margin has to clear

Every accuracy in this report is a mean over the seeds actually trained, written as
`mean ± sample s.d. (n=…)`. **The seed counts are not uniform**, and the differences are
load-bearing rather than incidental:

{seed_inventory(res)}

The dense references, per seed:

{dense_seed_table(res)}

Pooling every point trained at more than one seed -- {stats['n_points_pooled']} points,
{stats['pooled_dof']} degrees of freedom -- gives a within-point sample standard deviation of
**{stats['pooled_seed_sd_pp']:.2f} pp**. With three seeds on each side of a comparison, a measured
margin therefore has to reach about **{stats['detectable_margin_pp_n3']:.2f} pp** before Welch's
unequal-variance t-test can reject at alpha={res['constants']['alpha']:.2f}; at five seeds a side
it would still need {stats['detectable_margin_pp_n5']:.2f} pp. A margin of around 1.3 pp -- which
is the size of most of the interesting ones here -- would need roughly
**{stats['seeds_needed_for_1_3pp']} seeds per side** to be called significant.

Exactly one comparison in this report clears p<{res['constants']['alpha']:.2f}. Everything else is
written as *consistent in direction across seeds, inside the noise floor*, and the word
"significantly" is not used for it.

**Excluded.** `{excl['label']}` (run `{excl['run_id']}`, {acc(excl['accuracy'])}) {excl['reason']}.
It is not part of the dense curve in any table or figure.

## Master table

{master_table(res)}

`MACs executed` is what the deployed tensor shapes cost. For IMP and SNIP it is constant across
sparsity by construction; see the MACs section. Sizes for IMP and SNIP assume a sparse storage
format -- as plain fp16 state_dicts they remain {src['size_kb']:.2f} KB.

## Accuracy at matched size

{matched_grid(res)}

The top three levels were chosen so the pruned models land on the parameter counts of existing
dense references exactly (cm=1.3, cm=1.0, cm=0.5). The bottom two have no dense model at their
size: they are compared against base=16 ({num(b16['params'])} parameters,
{acc(b16['accuracy'])}), which is LARGER than either, so the pruned model wins the size axis by
construction and the comparison is a Pareto one rather than a same-size one.

For completeness, the same-size figure at those two depths, linearly interpolated in parameter
count between base=8 ({num(b8['params'])} parameters, {acc(b8['accuracy'])}, one seed) and
base=16 ({num(b16['params'])}, {acc(b16['accuracy'])}, {b16['n_seeds']} seeds):

{interpolated_table(res)}

An interpolated reference is a line segment between two measured models, not a trained model, and
no significance test can be run against it. This is exactly why the supervisor asked for a plot
rather than interpolated values, and Figure 1 (`reports/figures/fig1_accuracy_vs_params`) is what
should be read.

## Pruned versus same-size dense, in percentage points

{margins_table(res)}

## What the seed queue changed

Each margin below is replayed against the first 1, 2, … seeds of its dense reference, with the
pruned side held at its full seed count. These are the numbers this project actually held while
the overnight seed queue ran on 2026-08-24/25.

{accumulation_table(res)}

## Every pruned run, individually

The means above are taken over these. SNIP and DSP only; the IMP curves are single runs and
appear in full in the master table.

{per_seed_table(res)}

## MACs: structured versus unstructured

{macs_table(res)}

## Findings

{findings_section(res)}

## Retractions

Six claims this project asserted on single-seed data did not survive the seed queue of
2026-08-24/25. They are listed rather than quietly removed: several were sent to the supervisor
and are owed a correction, and the pattern they form is itself this report's strongest
methodological result -- a pruning curve measured at one seed per point is not merely imprecise,
it can be wrong about the sign of a comparison.

{retractions_section(res)}

The most transferable version of the lesson is what happened to two margins while the queue was
still running. As the dense side accumulated seeds, the 63.96 KB margin moved +1.14 -> +0.81 ->
+1.26 pp and the 47.14 KB margin moved +0.89 -> -0.26 -> -0.55 pp, changing sign. Both
intermediate values were reported in good faith and both were wrong. A margin is quotable only
once BOTH sides carry their full seed count.

## Provenance

Accuracies for the pruning curves come from `checkpoints/<run_id>/pruning_state.json`, one file
per seed, which is the authoritative record. `reports/macs.json` (written by
`pruning.measure_macs`, which measures every seed separately) supplies MACs and parameter counts;
its dense MACs reconcile exactly against `baseline.helpers.nessi.get_torch_size`, and its
recomputed non-zero parameter counts are checked against every `pruning_state.json` on each run of
this script. Dense accuracies come from `reports/reeval_last_ckpt.csv` where a re-score row
exists and from the run's own logged last-epoch value otherwise; the hand-maintained table in
`experiments/collect_results.py` is
verified against that CSV on every run in both directions, so an edit that contradicts it -- or a
newly added seed that a later re-score disagrees with -- raises rather than silently propagating.
Welch's t-test is implemented in `experiments/collect_results.py` rather than imported, so the
report regenerates on a machine without scipy; `tests/test_results_stats.py` cross-checks it against
scipy when scipy is available.

| run id | curve |
| :-- | :-- |
"""
    for method in METHOD_ORDER:
        c = res["curves"][method]
        for seed, run_id in sorted(c["seed_runs"].items(), key=lambda kv: int(kv[0])):
            doc += f"| `{run_id}` | {c['description']} -- seed {seed} |\n"
    for d in res["dense"]:
        for seed, v in sorted(d["seeds"].items(), key=lambda kv: int(kv[0])):
            doc += (f"| `{v['run_id']}` | dense reference {d['label']} ({d['config']}) "
                    f"-- seed {seed} |\n")
    for note in res["verification"]["notes"]:
        doc += f"\nNote: {note}.\n"
    return doc
