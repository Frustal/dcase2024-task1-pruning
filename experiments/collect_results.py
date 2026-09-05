"""Joins the pruning and dense-baseline results into one table and writes the report.

Sources: ``checkpoints/<run_id>/pruning_state.json`` (authoritative per-seed accuracy, sparsity,
non-zero parameter count and size per pruned level), ``reports/macs.json`` (the same rows plus
measured MACs, and the dense references' MACs and parameter counts), ``reports/reeval_last_ckpt.csv``
(last-epoch re-scores of the early dense runs) and ``DENSE_RUNS`` below (dense accuracies no file
holds in full). They are cross-checked against each other, then written out as
``reports/results.json``, ``reports/results.md`` (via ``experiments.results_text``) and
``reports/figures/`` (via ``experiments.plot_results``). Downstream code reads the JSON.

Seed counts are uneven -- SNIP and DSP have three, cm=1.3 five, the IMP curves and three of the
dense widths one -- so every accuracy carries ``n_seeds``, ``accuracy_std`` and its per-seed
values, and a single draw is never presented as a three-seed mean.

Accuracy is reported at the LAST training epoch. DCASE Task 1 has no validation split, so
``val_macro_acc`` is measured on the dev-test set and picking the best epoch by it selects on the
test set. The pruning runs used that convention from the first run; three early dense runs were
logged as best-epoch and re-scored offline, and their wandb summaries would inflate the dense
curve by 0.6 to 0.8 pp.

Welch's t-test is implemented here rather than imported, so the report regenerates without scipy;
``tests/test_results_stats.py`` compares the two when scipy is installed.

Run:
    uv run python -m experiments.collect_results
    uv run python -m experiments.collect_results --no-figures
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHECKPOINTS = REPO / "checkpoints"
REPORTS = REPO / "reports"
MACS_JSON = REPORTS / "macs.json"
REEVAL_CSV = REPORTS / "reeval_last_ckpt.csv"

FP16_BYTES = 2
MAX_MACS = 30_000_000        # baseline/helpers/nessi.py
MAX_PARAMS_MEMORY = 128_000  # baseline/helpers/nessi.py, bytes

ALPHA = 0.05                 # two-sided, for every significance statement in the report

# The organisers' published baseline figure, averaged over THEIR five runs. It is not this
# project's source model and must never be the comparison point for a pruned model: the pruned
# models came from the cm=1.8 checkpoint trained here, which scored 0.4991. Both are kept because
# the 0.38 pp gap between them is the evidence that the reproduction is sound.
DCASE_PUBLISHED = {"accuracy": 0.5029, "std": 0.0087, "n_runs": 5}

# --------------------------------------------------------------------------------------------
# Dense reference accuracies per seed, as
# (label, family, [(seed, run_id, last-epoch accuracy, has_csv_row, best-epoch accuracy)]).
# The one hand-maintained table in the pipeline, because no single file holds all of it.
# ``verify_dense_against_csv`` re-reads reports/reeval_last_ckpt.csv on every run and raises if
# this drifts from it, in either direction.
# --------------------------------------------------------------------------------------------
DENSE_RUNS = [
    ("cm=1.8 (source)", "cm", [
        (42, "762f0b4fde744636b143bf56646046b7", 0.4991, True, None),
    ]),
    ("cm=1.3", "cm", [
        (42, "jd152x6s", 0.5192, True,  0.5251),
        (43, "zuwc7sld", 0.5018, False, None),
        (44, "9me6xu2z", 0.5106, False, None),
        (45, "0083zgp0", 0.4766, False, None),
        (46, "qxbrd2ei", 0.4908, False, None),
    ]),
    ("base=24", "base", [
        (42, "rzkykzg3", 0.4981, False, None),
    ]),
    ("cm=1.0", "cm", [
        (42, "lgzlk5nh", 0.4967, True,  0.5048),
        (43, "vyrnych6", 0.5032, False, None),
        (44, "5nfzxoj2", 0.4867, False, None),
    ]),
    ("cm=0.5", "cm", [
        (42, "46nzd7gd", 0.4894, True,  0.4963),
        (43, "k7en8200", 0.5125, False, None),
        (44, "ot9ixa39", 0.5094, False, None),
    ]),
    ("base=16", "base", [
        (42, "5feac1qb", 0.4739, False, None),
        (43, "ki7pe3zf", 0.4792, False, None),
        (44, "qv5ym32m", 0.4681, False, None),
    ]),
    ("base=8", "base", [
        (42, "e7suq2am", 0.4538, False, None),
    ]),
]

# base=8 / cm=3.0 (run 1gfhagnf, 0.4129) failed as a dense reference: it scored far below every
# other dense model and belongs to neither width sweep. Excluded from the dense curve, and
# recorded here so the exclusion is explicit rather than an omission somebody later "fixes".
EXCLUDED_DENSE = [("base=8 / cm=3.0", "1gfhagnf", 0.4129,
                   "failed as a dense reference; excluded from the dense curve")]

# The pruning curves and the seeds actually trained. IMP appears three times because retrain
# length is itself a variable: the 20/35/150-epoch curves separate under-training from pruning
# damage. structured=True means the removals collapse into a smaller dense op, so the non-zero
# MAC count is the one that runs.
CURVES = [
    ("IMP-20",  False, "IMP, 20 retrain epochs per level",
     [(42, "d3jo4g1u")]),
    ("IMP-35",  False, "IMP, 35 retrain epochs per level",
     [(42, "86guyp3j")]),
    ("IMP-150", False, "IMP, 150 retrain epochs per level",
     [(42, "66d7cow0")]),
    ("SNIP",    False, "SNIP, single-shot at init, full 150-epoch recipe per level",
     [(42, "iel3qnvt"), (43, "pjqyh53l"), (44, "cul3sq06")]),
    ("DSP",     True,  "DSP, structured, beta bisection, lambda=2e-3",
     [(42, "pv6k1wze"), (43, "quuhqh9f"), (44, "wd05nhmr")]),
]

# A pruned level counts as size-matched to a dense reference when the parameter counts agree to
# this relative tolerance. The closest pair of dense references (base=24 and cm=1.0) differ by 7%,
# so a match at 0.5% is unambiguous.
MATCH_RTOL = 0.005


class VerificationError(RuntimeError):
    """Raised when two sources that must agree do not."""


# --------------------------------------------------------------------------------------------
# statistics -- written out rather than imported, because scipy is not a declared dependency and
# reports/results.json has to regenerate identically without it. tests/test_results_stats.py
# compares these against scipy wherever scipy is importable.
# --------------------------------------------------------------------------------------------

def mean_std(values: list[float]) -> tuple[float, float | None]:
    """Mean and SAMPLE standard deviation (n-1). std is None for a single observation."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, var ** 0.5


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction expansion for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def betainc_reg(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value of Student's t. p = I_{df/(df+t^2)}(df/2, 1/2)."""
    if df <= 0:
        return float("nan")
    return betainc_reg(df / 2.0, 0.5, df / (df + t * t))


def t_critical(df: float, alpha: float = ALPHA) -> float:
    """Two-sided critical |t| at ``alpha``. Bisection on the monotone p(t) above."""
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def welch(a: list[float], b: list[float]) -> dict | None:
    """Welch's unequal-variance t-test for ``a`` minus ``b``.

    None when either side has fewer than two observations: a single draw carries no variance
    estimate, so the report says "n=1" rather than printing something that looks like evidence.
    """
    if len(a) < 2 or len(b) < 2:
        return None
    ma, sa = mean_std(a)
    mb, sb = mean_std(b)
    va, vb = sa ** 2 / len(a), sb ** 2 / len(b)
    se = math.sqrt(va + vb)
    if se == 0.0:
        return None
    t = (ma - mb) / se
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    p = t_two_sided_p(t, df)
    return {
        "diff_pp": 100 * (ma - mb),
        "t": t,
        "df": df,
        "se_pp": 100 * se,
        "p": p,
        "significant_p05": p < ALPHA,
        "n_a": len(a),
        "n_b": len(b),
    }


def pooled_sd(samples: list[list[float]]) -> tuple[float, int]:
    """Pooled within-point sample s.d. over every point that has at least two seeds."""
    num = 0.0
    dof = 0
    for values in samples:
        if len(values) < 2:
            continue
        _m, s = mean_std(values)
        num += (len(values) - 1) * s ** 2
        dof += len(values) - 1
    return math.sqrt(num / dof), dof


def detectable_margin_pp(sd: float, n_per_side: int, alpha: float = ALPHA) -> float:
    """Smallest true difference a two-sample t-test could call significant, in pp.

    Not a power calculation: this is the observed-margin threshold, how large a measured
    difference has to be before |t| clears the critical value at the pooled s.d.
    """
    df = 2 * n_per_side - 2
    return 100 * t_critical(df, alpha) * sd * math.sqrt(2.0 / n_per_side)


def seeds_needed_for(margin_pp: float, sd: float, alpha: float = ALPHA, cap: int = 200) -> int:
    """Seeds PER SIDE at which a margin of ``margin_pp`` would clear ``alpha``."""
    for n in range(2, cap + 1):
        if detectable_margin_pp(sd, n, alpha) <= margin_pp:
            return n
    return cap


# --------------------------------------------------------------------------------------------
# loading + verification
# --------------------------------------------------------------------------------------------

def load_macs() -> list[dict]:
    if not MACS_JSON.exists():
        raise FileNotFoundError(
            f"{MACS_JSON} is missing. Regenerate it with:\n"
            "    uv run python -m pruning.measure_macs --json reports/macs.json"
        )
    with open(MACS_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def load_reeval() -> dict:
    with open(REEVAL_CSV, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {(r["run_id"], r["checkpoint"]): r for r in rows}


def load_state(run_id: str) -> list[dict]:
    """``pruning_state.json``'s completed levels, ascending in sparsity. AUTHORITATIVE."""
    path = CHECKPOINTS / run_id / "pruning_state.json"
    with open(path, encoding="utf-8") as fh:
        state = json.load(fh)
    return sorted(state["completed"], key=lambda e: e["sparsity"])


def verify_dense_against_csv(reeval: dict) -> list[str]:
    """Fail loudly if DENSE_RUNS has drifted from reports/reeval_last_ckpt.csv.

    Checked for every seed, not only the ones the CSV covers: a seed marked has_csv_row=False
    must have no row, so a re-score added later cannot disagree in silence. The CSV stores four
    decimals, hence the 5e-5 tolerance. The diagnostic best-epoch value is checked too, since its
    purpose is to show how much test-set peak-picking inflated the dense curve.
    """
    notes = []
    for label, _family, seeds in DENSE_RUNS:
        for seed, run_id, last_acc, has_csv_row, best_acc in seeds:
            who = f"{label} seed {seed} ({run_id})"
            row = reeval.get((run_id, "last"))
            if not has_csv_row:
                if row is not None:
                    raise VerificationError(
                        f"{who} is marked as having no CSV row, but {REEVAL_CSV.name} contains "
                        f"one ({row['test_macro_accuracy']}). Set has_csv_row=True and reconcile."
                    )
                notes.append(f"{who}: no re-score row -- trained after the last-epoch switch, so "
                             "its logged number was already last-epoch")
                continue
            if row is None:
                raise VerificationError(
                    f"{who} is marked has_csv_row=True but has no 'last' row in "
                    f"{REEVAL_CSV.name}."
                )
            if abs(float(row["test_macro_accuracy"]) - last_acc) > 5e-5:
                raise VerificationError(
                    f"{who}: DENSE_RUNS says last-epoch {last_acc}, but {REEVAL_CSV.name} says "
                    f"{row['test_macro_accuracy']}."
                )
            if best_acc is not None:
                brow = reeval.get((run_id, "best"))
                if brow is None or abs(float(brow["test_macro_accuracy"]) - best_acc) > 5e-5:
                    got = brow["test_macro_accuracy"] if brow else "no row"
                    raise VerificationError(
                        f"{who}: DENSE_RUNS says best-epoch {best_acc}, but {REEVAL_CSV.name} "
                        f"says {got}."
                    )
    return notes


def verify_curve_against_macs(name: str, seed: int, run_id: str, state: list[dict],
                              macs_rows: list[dict]) -> None:
    """The accuracy/params in reports/macs.json were copied from pruning_state.json. Prove it."""
    by_target = {round(r["target"], 4): r for r in macs_rows}
    for entry in state:
        row = by_target.get(round(entry["target"], 4))
        if row is None:
            raise VerificationError(
                f"{name} seed {seed} ({run_id}): reports/macs.json has no row for target "
                f"{entry['target']}. macs.json is stale -- rerun "
                "`uv run python -m pruning.measure_macs --json reports/macs.json`."
            )
        if abs(row["accuracy"] - entry["test_macro_accuracy"]) > 1e-9:
            raise VerificationError(
                f"{name} seed {seed} target {entry['target']}: macs.json accuracy "
                f"{row['accuracy']} != pruning_state.json {entry['test_macro_accuracy']}. "
                "macs.json is stale -- rerun `uv run python -m pruning.measure_macs "
                "--json reports/macs.json`."
            )
        if row["params_nonzero"] != entry["nonzero_params"]:
            raise VerificationError(
                f"{name} seed {seed} target {entry['target']}: macs.json counts "
                f"{row['params_nonzero']} non-zero params, pruning_state.json says "
                f"{entry['nonzero_params']}."
            )


# --------------------------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------------------------

def build_dense(macs_rows: list[dict]) -> list[dict]:
    by_label = {r["label"]: r for r in macs_rows if r["group"] == "dense"}
    out = []
    for label, family, seeds in DENSE_RUNS:
        m = by_label[label]
        values = [acc for _s, _r, acc, _c, _b in seeds]
        mean, std = mean_std(values)
        primary = seeds[0]
        row = {
            "label": label,
            # Seed 42, kept as "the" run id for provenance tables and figure code that need
            # only one identifier. Every seed is in ``seeds``.
            "run_id": primary[1],
            "family": family,          # which width knob was moved: cm, or base_channels
            "config": m["config"],
            # Architecture-determined and therefore identical across seeds, so not aggregated.
            "params": m["params_total"],
            "size_kb": m["params_total"] * FP16_BYTES / 1024,
            "macs": m["macs_dense"],   # a dense model executes exactly its non-zero MACs
            "accuracy": mean,
            "accuracy_std": std,
            "accuracy_values": [acc for _s, _r, acc, _c, _b in seeds],
            "n_seeds": len(seeds),
            "accuracy_convention": "last epoch",
            "seeds": {
                str(seed): {
                    "run_id": run_id,
                    "accuracy": acc,
                    "accuracy_best_epoch": best,
                    "source": (f"reports/reeval_last_ckpt.csv ({run_id},last)" if has_csv else
                               f"wandb run {run_id} (trained after the last-epoch switch; no "
                               "re-score needed)"),
                }
                for seed, run_id, acc, has_csv, best in seeds
            },
        }
        row["accuracy_source"] = (
            f"mean of {len(seeds)} seed(s): " +
            ", ".join(f"{s}={a:.4f} ({r})" for s, r, a, _c, _b in seeds)
        )
        out.append(row)
    out.sort(key=lambda r: r["params"], reverse=True)
    return out


def build_curves(macs_rows: list[dict]) -> dict:
    curves = {}
    for name, structured, description, seed_runs in CURVES:
        # target -> {seed: {...}}; every seed of a method runs the same five targets.
        per_target: dict[float, dict[int, dict]] = {}
        for seed, run_id in seed_runs:
            state = load_state(run_id)
            rows_for_run = [r for r in macs_rows if r.get("run_id") == run_id]
            verify_curve_against_macs(name, seed, run_id, state, rows_for_run)
            by_target = {round(r["target"], 4): r for r in rows_for_run}
            for entry in state:
                key = round(entry["target"], 4)
                m = by_target[key]
                # macs_executed is what the model costs today. Masking never changes a tensor
                # shape, so IMP and SNIP still multiply by every zero and sit on the dense count
                # at every level. DSP's removals collapse into a grouped conv, so its saving is
                # real, and it varies by seed because a DSP seed re-runs phase A.
                per_target.setdefault(key, {})[seed] = {
                    "run_id": run_id,
                    "target": entry["target"],
                    "sparsity": entry["sparsity"],
                    "params_nonzero": entry["nonzero_params"],
                    "size_kb": entry["size_kb"],
                    "accuracy": entry["test_macro_accuracy"],
                    "macs_executed": m["macs_structural"] if structured else m["macs_dense"],
                    "macs_nonzero": m["macs_structural"],
                    "macs_dense_shape": m["macs_dense"],
                }

        levels = []
        for key in sorted(per_target):
            per_seed = per_target[key]
            seeds_sorted = sorted(per_seed)
            if len(per_seed) != len(seed_runs):
                raise VerificationError(
                    f"{name} target {key}: {len(per_seed)} seed(s) present but {len(seed_runs)} "
                    "declared. A curve with a missing level cannot be averaged."
                )
            accs = [per_seed[s]["accuracy"] for s in seeds_sorted]
            mean_acc, std_acc = mean_std(accs)
            params = [per_seed[s]["params_nonzero"] for s in seeds_sorted]
            sizes = [per_seed[s]["size_kb"] for s in seeds_sorted]
            executed = [per_seed[s]["macs_executed"] for s in seeds_sorted]
            nonzero = [per_seed[s]["macs_nonzero"] for s in seeds_sorted]
            shapes = {per_seed[s]["macs_dense_shape"] for s in seeds_sorted}
            if len(shapes) != 1:
                raise VerificationError(
                    f"{name} target {key}: dense-shape MACs differ across seeds ({shapes}). "
                    "Masking cannot change a tensor shape; the checkpoints do not match the "
                    "config."
                )
            dense_shape = shapes.pop()
            mean_exec = sum(executed) / len(executed)
            mean_nonzero = sum(nonzero) / len(nonzero)
            levels.append({
                "target": per_seed[seeds_sorted[0]]["target"],
                "sparsity": sum(per_seed[s]["sparsity"] for s in seeds_sorted) / len(seeds_sorted),
                # Near-identical across seeds for SNIP, whose global top-k hits the target
                # exactly, and off by a few dozen for DSP, whose bisection lands on whole channel
                # groups. Tables use the mean; the per-seed values stay so it can be audited.
                "params_nonzero": round(sum(params) / len(params)),
                "params_nonzero_values": params,
                "size_kb": sum(sizes) / len(sizes),
                "accuracy": mean_acc,
                "accuracy_std": std_acc,
                "accuracy_values": accs,
                "accuracy_spread_pp": 100 * std_acc if std_acc is not None else None,
                "n_seeds": len(accs),
                "accuracy_convention": "last epoch",
                "accuracy_source": "; ".join(
                    f"seed {s}: checkpoints/{per_seed[s]['run_id']}/pruning_state.json"
                    for s in seeds_sorted),
                "macs_executed": round(mean_exec),
                "macs_executed_values": executed,
                "macs_nonzero": round(mean_nonzero),
                "macs_nonzero_values": nonzero,
                "macs_star_spread_pct": (100 * (max(nonzero) - min(nonzero)) / mean_nonzero
                                         if len(nonzero) > 1 else None),
                "macs_dense_shape": dense_shape,
                "macs_reduction_executed": 1.0 - mean_exec / dense_shape,
                "macs_reduction_nonzero": 1.0 - mean_nonzero / dense_shape,
                "macs_source": "reports/macs.json",
                "seeds": {str(s): per_seed[s] for s in seeds_sorted},
            })

        curves[name] = {
            "method": name,
            "run_id": seed_runs[0][1],
            "seed_runs": {str(s): r for s, r in seed_runs},
            "n_seeds": len(seed_runs),
            "structured": structured,
            "description": description,
            # An unstructured model's 22 KB needs a sparse storage format; as a plain fp16
            # state_dict it is still the source model's 119.43 KB. DSP's size is deployable.
            "size_is_deployable": structured,
            "macs_nonzero_is_reachable": structured,
            "levels": levels,
        }
    return curves


def match_dense(params: int, dense: list[dict]) -> dict:
    """The dense reference a pruned model of ``params`` parameters is judged against.

    "exact"        a dense model of that width exists. The top three sparsity targets were chosen
                   to land on cm=1.3 / cm=1.0 / cm=0.5, so a Welch test between two measured seed
                   samples is meaningful.
    "pareto"       below 24 k parameters no dense model lands on a pruned size, so the comparison
                   runs against the smallest measured dense model that is still larger, base=16.
                   The pruned model wins the size axis by construction; only accuracy is open.
    "interpolated" the same-size figure at those depths, linear in parameter count between base=8
                   and base=16. A line segment rather than a trained model, so no test runs
                   against it; figure 1 is what should be read.
    """
    asc = sorted(dense, key=lambda r: r["params"])
    nearest = min(asc, key=lambda r: abs(r["params"] - params))
    if abs(nearest["params"] - params) <= MATCH_RTOL * params:
        return {"kind": "exact", "reference": nearest, "interpolated": None}

    lo = max((r for r in asc if r["params"] < params), key=lambda r: r["params"], default=None)
    hi = min((r for r in asc if r["params"] > params), key=lambda r: r["params"], default=None)
    if hi is None:
        return {"kind": "none", "reference": None, "interpolated": None}
    interpolated = None
    if lo is not None:
        frac = (params - lo["params"]) / (hi["params"] - lo["params"])
        interpolated = {
            "label": f"interp({lo['label']}, {hi['label']})",
            "accuracy": lo["accuracy"] + frac * (hi["accuracy"] - lo["accuracy"]),
            "between": [lo["label"], hi["label"]],
        }
    return {"kind": "pareto", "reference": hi, "interpolated": interpolated}


def build_margins(dense: list[dict], curves: dict) -> list[dict]:
    out = []
    for name, curve in curves.items():
        for lvl in curve["levels"]:
            match = match_dense(lvl["params_nonzero"], dense)
            ref = match["reference"]
            if ref is None:
                continue
            margin = lvl["accuracy"] - ref["accuracy"]
            test = welch(lvl["accuracy_values"], ref["accuracy_values"])
            row = {
                "method": name,
                "target": lvl["target"],
                "size_kb": lvl["size_kb"],
                "params_nonzero": lvl["params_nonzero"],
                "accuracy": lvl["accuracy"],
                "accuracy_std": lvl["accuracy_std"],
                "n_seeds_pruned": lvl["n_seeds"],
                "reference_kind": match["kind"],
                "dense_reference": ref["label"],
                "dense_accuracy": ref["accuracy"],
                "dense_accuracy_std": ref["accuracy_std"],
                "n_seeds_dense": ref["n_seeds"],
                "dense_params": ref["params"],
                "dense_macs": ref["macs"],
                "params_vs_reference_pct": 100 * (1 - lvl["params_nonzero"] / ref["params"]),
                "macs_vs_reference_pct": 100 * (1 - lvl["macs_executed"] / ref["macs"]),
                "margin_pp": 100 * margin,
                "welch": test,
                "significant_p05": bool(test and test["significant_p05"]),
                # True when either side is a single run: no test, and the margin is one draw.
                "single_seed_side": lvl["n_seeds"] < 2 or ref["n_seeds"] < 2,
                "interpolated_same_size": None,
            }
            if match["interpolated"] is not None:
                interp = match["interpolated"]
                row["interpolated_same_size"] = {
                    "label": interp["label"],
                    "accuracy": interp["accuracy"],
                    "between": interp["between"],
                    "margin_pp": 100 * (lvl["accuracy"] - interp["accuracy"]),
                    "note": "line segment between two measured models, not a trained model; "
                            "no significance test is possible against it",
                }
            out.append(row)
    return out


def build_seed_accumulation(dense: list[dict], curves: dict, margins: list[dict]) -> list[dict]:
    """How each margin moved as the dense reference accumulated seeds.

    Each entry replays the margin against the first 1, 2, ... seeds of the dense reference with
    the pruned side at full count, the state this project was in while the seeds were landing.
    Margins swung by more than a point, in both directions, before settling.
    """
    by_label = {d["label"]: d for d in dense}
    out = []
    for m in margins:
        ref = by_label[m["dense_reference"]]
        if ref["n_seeds"] < 2 or m["n_seeds_pruned"] < 2:
            continue
        prefixes = []
        for k in range(1, ref["n_seeds"] + 1):
            partial = ref["accuracy_values"][:k]
            prefixes.append({
                "dense_seeds": k,
                "dense_accuracy": sum(partial) / k,
                "margin_pp": 100 * (m["accuracy"] - sum(partial) / k),
            })
        swing = max(p["margin_pp"] for p in prefixes) - min(p["margin_pp"] for p in prefixes)
        out.append({
            "method": m["method"],
            "size_kb": m["size_kb"],
            "dense_reference": m["dense_reference"],
            "prefixes": prefixes,
            "final_margin_pp": prefixes[-1]["margin_pp"],
            "swing_pp": swing,
            "sign_flipped": (min(p["margin_pp"] for p in prefixes) < 0
                             < max(p["margin_pp"] for p in prefixes)),
        })
    out.sort(key=lambda r: -r["swing_pp"])
    return out


# --------------------------------------------------------------------------------------------
# findings -- every claim the write-up makes is recomputed from the assembled table and carries a
# ``holds`` flag. When a claim stops holding, after a re-score or an added seed,
# ``experiments.results_text`` prints the failure instead of the prose, so a sentence cannot
# outlive its data. Two claims carried in prose for three weeks were falsified this way.
# --------------------------------------------------------------------------------------------

def build_findings(res: dict) -> list[dict]:
    dense = res["dense"]
    curves = res["curves"]
    by_label = {d["label"]: d for d in dense}
    source = res["constants"]["source_model"]
    stats = res["constants"]["statistics"]
    findings = []

    def lvl(method: str, target: float) -> dict:
        return next(l for l in curves[method]["levels"] if abs(l["target"] - target) < 1e-6)

    def margin(method: str, target: float) -> dict:
        return next(m for m in res["margins"]
                    if m["method"] == method and abs(m["target"] - target) < 1e-6)

    # ---------------------------------------------------------------- statistical framing
    # With three seeds a side almost nothing here reaches p<0.05, and every accuracy
    # finding below is read through that, so it comes first.
    sig = [m for m in res["margins"] if m["significant_p05"]]
    findings.append({
        "id": "almost_nothing_reaches_significance",
        "holds": len(sig) == 1 and sig[0]["method"] == "DSP",
        "numbers": {
            "pooled_sd_pp": stats["pooled_seed_sd_pp"],
            "pooled_dof": stats["pooled_dof"],
            "detectable_margin_pp_n3": stats["detectable_margin_pp_n3"],
            "n_margins_tested": sum(1 for m in res["margins"] if m["welch"]),
            "n_significant": len(sig),
            "significant": [{"method": m["method"], "size_kb": m["size_kb"],
                             "margin_pp": m["margin_pp"], "t": m["welch"]["t"],
                             "df": m["welch"]["df"], "p": m["welch"]["p"]} for m in sig],
            "seeds_needed_for_1_3pp": stats["seeds_needed_for_1_3pp"],
        },
    })

    # Partial seed data is misleading; measured by replaying the reference's seed prefixes.
    accum = res["seed_accumulation"]
    worst = max(accum, key=lambda a: a["swing_pp"]) if accum else None
    flips = [a for a in accum if a["sign_flipped"]]
    findings.append({
        "id": "partial_seed_data_is_misleading",
        "holds": bool(worst and worst["swing_pp"] > 0.5 and flips),
        "numbers": {"accumulation": accum,
                    "worst_swing_pp": worst["swing_pp"] if worst else None,
                    "n_sign_flips": len(flips)},
    })

    # The tempting generalisation ("single-seed dense references sit low") is false:
    # cm=1.0 and cm=0.5 rose when re-seeded, base=16 did not move.
    def move(label: str) -> dict:
        d = by_label[label]
        return {"label": label, "seed42": d["accuracy_values"][0], "mean": d["accuracy"],
                "move_pp": 100 * (d["accuracy"] - d["accuracy_values"][0]),
                "n_seeds": d["n_seeds"]}

    # cm=1.3 was already at five seeds and is reported for context, not as part of the claim.
    queue_moves = [move(l) for l in ("cm=1.0", "cm=0.5", "base=16")]
    findings.append({
        "id": "single_seed_references_are_not_systematically_low",
        "holds": (not all(m["move_pp"] > 0 for m in queue_moves)
                  and max(abs(m["move_pp"]) for m in queue_moves) > 1.0
                  and min(abs(m["move_pp"]) for m in queue_moves) < 0.1),
        "numbers": {"moves": queue_moves, "cm13": move("cm=1.3")},
    })

    findings.append({
        "id": "seed_counts_are_asymmetric",
        "holds": (curves["IMP-150"]["n_seeds"] == 1 and curves["SNIP"]["n_seeds"] == 3
                  and curves["DSP"]["n_seeds"] == 3),
        "numbers": {
            "curves": {name: c["n_seeds"] for name, c in curves.items()},
            "dense": {d["label"]: d["n_seeds"] for d in dense},
        },
    })

    # ------------------------------------------------------------------ accuracy findings
    snip = curves["SNIP"]["levels"]
    accs = [l["accuracy"] for l in snip]
    biggest, smallest = snip[0], snip[-1]
    # seed 42 alone, for the contrast with the much more eventful single-seed curve
    s42 = [l["accuracy_values"][0] for l in snip]
    s42_dips = sum(1 for a, b in zip(s42, s42[1:]) if b > a)
    findings.append({
        "id": "snip_curve_is_flat",
        "holds": 100 * (max(accs) - min(accs)) < 2.5 and smallest["n_seeds"] >= 3,
        "numbers": {
            "range_pp": 100 * (max(accs) - min(accs)),
            "biggest_kb": biggest["size_kb"], "biggest_acc": biggest["accuracy"],
            "smallest_kb": smallest["size_kb"], "smallest_acc": smallest["accuracy"],
            "size_ratio_across_curve": biggest["size_kb"] / smallest["size_kb"],
            "vs_source_pp": 100 * (smallest["accuracy"] - source["accuracy"]),
            "source_acc": source["accuracy"], "source_kb": source["size_kb"],
            "size_ratio_vs_source": source["size_kb"] / smallest["size_kb"],
            "n_seeds": smallest["n_seeds"],
            "seed42_range_pp": 100 * (max(s42) - min(s42)),
            "seed42_inversions": s42_dips,
        },
    })

    # SNIP matches same-size dense rather than beating it at four of five levels; the
    # earlier claim rested on single-seed data on both sides.
    exact = [margin("SNIP", t) for t in (0.3362, 0.486, 0.6333)]
    pareto = [margin("SNIP", t) for t in (0.75, 0.85)]
    findings.append({
        "id": "snip_matches_same_size_dense",
        "holds": (all(not m["significant_p05"] for m in exact)
                  and all(m["margin_pp"] > 0 for m in pareto)
                  and all(m["reference_kind"] == "exact" for m in exact)),
        "numbers": {
            "matched": [{"size_kb": m["size_kb"], "reference": m["dense_reference"],
                         "margin_pp": m["margin_pp"], "t": m["welch"]["t"],
                         "p": m["welch"]["p"], "n_pruned": m["n_seeds_pruned"],
                         "n_dense": m["n_seeds_dense"]} for m in exact],
            "pareto": [{"size_kb": m["size_kb"], "reference": m["dense_reference"],
                        "margin_pp": m["margin_pp"], "t": m["welch"]["t"], "p": m["welch"]["p"],
                        "params_pct_fewer": m["params_vs_reference_pct"]} for m in pareto],
            "n_clear_losses": sum(1 for m in exact + pareto
                                  if m["margin_pp"] < 0 and m["significant_p05"]),
        },
    })

    # DSP's Pareto claim lives at 22 KB, not at the 34 KB point a single seed suggested.
    dsp85 = lvl("DSP", 0.85)
    m85 = margin("DSP", 0.85)
    b16 = by_label["base=16"]
    findings.append({
        "id": "dsp_matches_base16_at_22kb_on_two_thirds_the_budget",
        "holds": (dsp85["params_nonzero"] < b16["params"]
                  and dsp85["macs_executed"] < b16["macs"]
                  and not m85["significant_p05"]
                  and abs(m85["margin_pp"]) < 0.5),
        "numbers": {
            "dsp_params": dsp85["params_nonzero"], "dense_params": b16["params"],
            "dsp_macs": dsp85["macs_executed"], "dense_macs": b16["macs"],
            "dsp_acc": dsp85["accuracy"], "dsp_std_pp": dsp85["accuracy_spread_pp"],
            "dense_acc": b16["accuracy"], "dense_std_pp": 100 * b16["accuracy_std"],
            "margin_pp": m85["margin_pp"], "t": m85["welch"]["t"], "p": m85["welch"]["p"],
            "params_pct_fewer": m85["params_vs_reference_pct"],
            "macs_pct_fewer": m85["macs_vs_reference_pct"],
            "n_pruned": dsp85["n_seeds"], "n_dense": b16["n_seeds"],
        },
    })

    # The only significant comparison in the study, and it is a negative result about DSP.
    m47 = margin("DSP", 0.6333)
    findings.append({
        "id": "dsp_at_47kb_is_significantly_worse_than_dense",
        "holds": bool(m47["significant_p05"] and m47["margin_pp"] < 0),
        "numbers": {
            "size_kb": m47["size_kb"], "reference": m47["dense_reference"],
            "dsp_acc": m47["accuracy"], "dense_acc": m47["dense_accuracy"],
            "margin_pp": m47["margin_pp"], "t": m47["welch"]["t"], "df": m47["welch"]["df"],
            "p": m47["welch"]["p"],
            "shallow": [{"size_kb": margin("DSP", t)["size_kb"],
                         "reference": margin("DSP", t)["dense_reference"],
                         "margin_pp": margin("DSP", t)["margin_pp"]}
                        for t in (0.3362, 0.486, 0.6333)],
        },
    })

    # DSP scatters more than SNIP, with a mechanism: a DSP seed re-runs phase A, so two DSP
    # seeds train different architectures where two SNIP seeds differ only in which weights the
    # mask keeps. The spread does not grow monotonically with depth.
    dsp_sd = [l["accuracy_spread_pp"] for l in curves["DSP"]["levels"]]
    snip_sd = [l["accuracy_spread_pp"] for l in curves["SNIP"]["levels"]]
    findings.append({
        "id": "dsp_scatters_more_than_snip",
        "holds": (sum(dsp_sd) / len(dsp_sd) > sum(snip_sd) / len(snip_sd)
                  and max(dsp_sd) > max(snip_sd)),
        "numbers": {
            "dsp_sd_pp": dsp_sd, "snip_sd_pp": snip_sd,
            "dsp_mean_sd_pp": sum(dsp_sd) / len(dsp_sd),
            "snip_mean_sd_pp": sum(snip_sd) / len(snip_sd),
            "dsp_max_sd_pp": max(dsp_sd), "snip_max_sd_pp": max(snip_sd),
            "sizes_kb": [l["size_kb"] for l in curves["DSP"]["levels"]],
            # Reported because the intuitive version, "DSP's variance widens with depth", is not
            # what the data says.
            "monotone_widening_with_depth": all(a < b for a, b in zip(dsp_sd, dsp_sd[1:])),
            "deepest_is_second_tightest": sorted(dsp_sd).index(dsp_sd[-1]) == 1,
            "dsp_deep_mean_sd_pp": sum(dsp_sd[3:]) / 2,
            "dsp_shallow_mean_sd_pp": sum(dsp_sd[:3]) / 3,
        },
    })

    # Averaging over seeds removed most of the structure the single-seed curves seemed to
    # have. Both mean curves keep one small inversion, inside the per-point spread.
    def inversions(levels: list[dict], key: str) -> list[dict]:
        return [{"bigger_kb": a["size_kb"], "bigger_acc": a[key],
                 "smaller_kb": b["size_kb"], "smaller_acc": b[key],
                 "gap_pp": 100 * (b[key] - a[key])}
                for a, b in zip(levels, levels[1:]) if b[key] > a[key]]

    seed42_curves = {}
    for name in ("SNIP", "DSP"):
        levels = curves[name]["levels"]
        s42 = [{"size_kb": l["size_kb"], "accuracy": l["accuracy_values"][0]} for l in levels]
        rng42 = 100 * (max(l["accuracy"] for l in s42) - min(l["accuracy"] for l in s42))
        rng = 100 * (max(l["accuracy"] for l in levels) - min(l["accuracy"] for l in levels))
        seed42_curves[name] = {
            "seed42_range_pp": rng42, "mean_range_pp": rng,
            "seed42_inversions": inversions(s42, "accuracy"),
            "mean_inversions": inversions(levels, "accuracy"),
            "max_mean_inversion_pp": max([i["gap_pp"] for i in inversions(levels, "accuracy")],
                                         default=0.0),
            "max_sd_pp": max(l["accuracy_spread_pp"] for l in levels),
        }
    findings.append({
        "id": "seed_averaging_dissolves_the_single_seed_structure",
        "holds": all(
            len(v["mean_inversions"]) < len(v["seed42_inversions"])
            or v["max_mean_inversion_pp"] < v["max_sd_pp"]
            for v in seed42_curves.values()),
        "numbers": {"curves": seed42_curves},
    })

    # ------------------------------------------------------------------------ MACs findings
    # MACs are architecture rather than luck, so these are seed-independent in substance.

    # Unstructured pruning buys nothing on the compute axis.
    unstructured = [c for c in curves.values() if not c["structured"]]
    flat = all(l["macs_executed"] == source["macs"] for c in unstructured for l in c["levels"])
    dsp_cuts = [l["macs_reduction_executed"] for l in curves["DSP"]["levels"]]
    findings.append({
        "id": "unstructured_macs_are_free_of_savings",
        "holds": flat and min(dsp_cuts) > 0.30,
        "numbers": {"unstructured_macs": source["macs"],
                    "dsp_cut_min_pct": 100 * min(dsp_cuts),
                    "dsp_cut_max_pct": 100 * max(dsp_cuts)},
    })

    findings.append({
        "id": "macs_budget_is_binding",
        "holds": source["macs_pct_of_budget"] > 85.0,
        "numbers": {"macs": source["macs"], "budget": res["constants"]["dcase_max_macs"],
                    "pct_of_budget": source["macs_pct_of_budget"]},
    })

    b24, cm10 = by_label["base=24"], by_label["cm=1.0"]
    findings.append({
        "id": "dense_curve_reorders_on_macs_axis",
        "holds": b24["params"] > cm10["params"] and b24["macs"] < cm10["macs"],
        "numbers": {"base24_params": b24["params"], "cm10_params": cm10["params"],
                    "base24_macs": b24["macs"], "cm10_macs": cm10["macs"],
                    "macs_ratio": cm10["macs"] / b24["macs"]},
    })

    # SNIP's MACs reduction lags its parameter reduction, IMP's tracks it.
    snip85, imp85 = lvl("SNIP", 0.85), lvl("IMP-150", 0.85)
    findings.append({
        "id": "snip_macs_lag_params",
        "holds": snip85["macs_reduction_nonzero"] < imp85["macs_reduction_nonzero"] - 0.05,
        "numbers": {"sparsity": snip85["sparsity"],
                    "snip_macs_cut_pct": 100 * snip85["macs_reduction_nonzero"],
                    "imp_macs_cut_pct": 100 * imp85["macs_reduction_nonzero"],
                    "lag_pp": 100 * (imp85["macs_reduction_nonzero"]
                                     - snip85["macs_reduction_nonzero"]),
                    "imp_n_seeds": imp85["n_seeds"], "snip_n_seeds": snip85["n_seeds"]},
    })

    # MACs* is seed-dependent even for unstructured pruning, which is why MACs are
    # measured per seed rather than reused from seed 42. The mask moves; the shapes do not.
    spread = {name: [{"size_kb": l["size_kb"], "spread_pct": l["macs_star_spread_pct"],
                      "values": l["macs_nonzero_values"]}
                     for l in curves[name]["levels"]]
              for name in ("SNIP", "DSP")}
    snip_spreads = [s["spread_pct"] for s in spread["SNIP"]]
    dsp_spreads = [s["spread_pct"] for s in spread["DSP"]]
    findings.append({
        "id": "macs_star_is_seed_dependent",
        "holds": max(snip_spreads) > 1.0 and max(dsp_spreads) > 1.0,
        "numbers": {"spread": spread,
                    "snip_max_spread_pct": max(snip_spreads),
                    "dsp_max_spread_pct": max(dsp_spreads),
                    "snip_deepest_spread_pct": snip_spreads[-1],
                    "dsp_deepest_spread_pct": dsp_spreads[-1]},
    })

    # DSP's executed MACs land on the base_channels dense family rather than merely near
    # it, measured as relative deviation from that curve, interpolated in parameter count.
    base_family = sorted([d for d in dense if d["family"] == "base"]
                         + [by_label["cm=1.8 (source)"]], key=lambda d: d["params"])

    def base_family_macs(params: int) -> float:
        lo = max((d for d in base_family if d["params"] <= params), key=lambda d: d["params"])
        hi = min((d for d in base_family if d["params"] >= params), key=lambda d: d["params"])
        if lo["params"] == hi["params"]:
            return float(lo["macs"])
        frac = (params - lo["params"]) / (hi["params"] - lo["params"])
        return lo["macs"] + frac * (hi["macs"] - lo["macs"])

    devs = [{"target": l["target"], "dsp_macs": l["macs_executed"],
             "base_family_macs": base_family_macs(l["params_nonzero"]),
             "rel_dev_pct": 100 * (l["macs_executed"] / base_family_macs(l["params_nonzero"]) - 1)}
            for l in curves["DSP"]["levels"]]
    findings.append({
        "id": "dsp_macs_track_base_channels_family",
        "holds": max(abs(d["rel_dev_pct"]) for d in devs) < 10.0,
        "numbers": {"deviations": devs,
                    "max_abs_rel_dev_pct": max(abs(d["rel_dev_pct"]) for d in devs)},
    })

    # IMP's retrain-length requirement is sparsity-dependent (the compute confound). One
    # seed per curve, so the gap is a difference of two single draws, not of two means.
    gaps = [{"target": a["target"], "size_kb": a["size_kb"],
             "gap_pp": 100 * (b["accuracy"] - a["accuracy"])}
            for a, b in zip(curves["IMP-20"]["levels"], curves["IMP-150"]["levels"])]
    findings.append({
        "id": "imp_retrain_need_is_sparsity_dependent",
        "holds": gaps[-1]["gap_pp"] > gaps[0]["gap_pp"] + 3.0,
        "numbers": {"gaps": gaps,
                    "n_seeds": curves["IMP-20"]["n_seeds"],
                    "pooled_sd_pp": stats["pooled_seed_sd_pp"]},
    })
    return findings


def assemble() -> dict:
    macs_rows = load_macs()
    reeval = load_reeval()
    notes = verify_dense_against_csv(reeval)

    dense = build_dense(macs_rows)
    curves = build_curves(macs_rows)

    # Pooled within-point scatter over every point with more than one seed: ten pruned levels and
    # four dense widths. It replaces the older single noise-floor figure, which was cm=1.3's seed
    # spread carried across the whole study for want of anything else.
    samples = [d["accuracy_values"] for d in dense if d["n_seeds"] > 1]
    samples += [l["accuracy_values"] for c in curves.values() for l in c["levels"]
                if l["n_seeds"] > 1]
    sd, dof = pooled_sd(samples)
    cm13 = next(r for r in dense if r["label"] == "cm=1.3")

    source = next(r for r in dense if r["label"] == "cm=1.8 (source)")
    res = {
        "about": (
            "Authoritative joined results for the DCASE 2024 Task 1 pruning thesis. Generated by "
            "experiments/collect_results.py -- do not hand-edit. Every accuracy carries its "
            "convention, its seed count and its source file."
        ),
        "constants": {
            "fp16_bytes_per_param": FP16_BYTES,
            "dcase_max_macs": MAX_MACS,
            "dcase_max_params_memory_bytes": MAX_PARAMS_MEMORY,
            "alpha": ALPHA,
            "source_model": {
                "label": source["label"], "run_id": source["run_id"],
                "accuracy": source["accuracy"], "n_seeds": source["n_seeds"],
                "params": source["params"],
                "size_kb": source["size_kb"], "macs": source["macs"],
                "macs_pct_of_budget": 100 * source["macs"] / MAX_MACS,
                "note": ("this project's own trained cm=1.8 checkpoint -- the model every pruning "
                         "run was actually pruned from. All pruned-vs-baseline comparisons use "
                         "this number, not the organisers' published one. Single seed."),
            },
            "dcase_published_baseline": {
                **DCASE_PUBLISHED,
                "note": ("the organisers' figure over THEIR five runs. Reproduction evidence only "
                         "-- never a comparison point for a pruned model."),
            },
            "statistics": {
                "pooled_seed_sd_pp": 100 * sd,
                "pooled_dof": dof,
                "n_points_pooled": sum(1 for s in samples if len(s) > 1),
                "detectable_margin_pp_n3": detectable_margin_pp(sd, 3),
                "detectable_margin_pp_n5": detectable_margin_pp(sd, 5),
                "seeds_needed_for_1_3pp": seeds_needed_for(1.3, sd),
                "cm13_sd_pp": 100 * cm13["accuracy_std"],
                "note": (
                    "pooled within-point sample s.d. over every point trained at more than one "
                    "seed. detectable_margin_pp_n3 is how large a measured margin has to be, with "
                    "three seeds a side, before a two-sample t-test at alpha=0.05 would call it "
                    "significant. Margins below it are 'consistent in direction, inside the noise "
                    "floor' -- never 'significantly better'."
                ),
            },
        },
        "dense": dense,
        "dense_excluded": [{"label": l, "run_id": r, "accuracy": a, "reason": why}
                           for l, r, a, why in EXCLUDED_DENSE],
        "curves": curves,
        "margins": build_margins(dense, curves),
        "verification": {
            "dense_vs_reeval_csv": "passed",
            "curves_vs_macs_json": "passed",
            "notes": notes,
        },
    }
    res["seed_accumulation"] = build_seed_accumulation(dense, curves, res["margins"])
    res["findings"] = build_findings(res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", default=str(REPORTS / "results.json"))
    ap.add_argument("--md", default=str(REPORTS / "results.md"))
    ap.add_argument("--no-figures", action="store_true", help="skip regenerating reports/figures/")
    args = ap.parse_args()

    res = assemble()

    out_json = Path(args.json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")

    from experiments.results_text import render_report
    out_md = Path(args.md)
    out_md.write_text(render_report(res), encoding="utf-8")
    print(f"wrote {out_md}")

    if not args.no_figures:
        from experiments.plot_results import make_figures
        for f in make_figures(res):
            print(f"wrote {f}")

    failed = [f["id"] for f in res["findings"] if not f["holds"]]
    print("\nverification: dense table vs reports/reeval_last_ckpt.csv -- passed")
    print("verification: pruning_state.json vs reports/macs.json      -- passed")
    if failed:
        print("findings that DO NOT hold against the current data: " + ", ".join(failed))
    else:
        print(f"findings: all {len(res['findings'])} hold against the current data")


if __name__ == "__main__":
    main()
