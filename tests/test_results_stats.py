"""Invariants of the results pipeline: statistics, verification, report wiring.

The load-bearing test is ``test_welch_matches_scipy``. ``experiments.collect_results`` implements
Welch's t-test and the Student-t tail by hand so ``reports/results.md`` regenerates without
scipy installed, which is only safe if the hand-rolled version is right; it is checked against
scipy where available and skipped where not. The rest guard the two ways a wrong number could
reach the thesis quietly: a seed contradicting reports/reeval_last_ckpt.csv, and a finding whose
prose has no renderer and would vanish from the report instead of raising.

    uv run pytest tests/test_results_stats.py -v
"""
import json
import random

import pytest

from experiments import collect_results as cr
from experiments import results_text


# --------------------------------------------------------------------------------------------
# the statistics
# --------------------------------------------------------------------------------------------

def test_mean_std_is_the_sample_sd_and_none_at_n1():
    mean, sd = cr.mean_std([0.5, 0.6, 0.7])
    assert abs(mean - 0.6) < 1e-12
    assert abs(sd - 0.1) < 1e-12          # n-1 denominator, not n
    assert cr.mean_std([0.5]) == (0.5, None)


def test_welch_is_none_when_either_side_has_one_observation():
    assert cr.welch([0.1], [0.2, 0.3, 0.4]) is None
    assert cr.welch([0.1, 0.2, 0.3], [0.2]) is None
    assert cr.welch([0.1], [0.2]) is None


def test_welch_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = random.Random(0)
    for _ in range(200):
        a = [rng.gauss(0.50, 0.010) for _ in range(rng.randint(2, 8))]
        b = [rng.gauss(0.49, 0.020) for _ in range(rng.randint(2, 8))]
        ours = cr.welch(a, b)
        theirs = scipy_stats.ttest_ind(a, b, equal_var=False)
        assert abs(ours["t"] - theirs.statistic) < 1e-9
        assert abs(ours["p"] - theirs.pvalue) < 1e-11
        assert abs(ours["df"] - theirs.df) < 1e-9


def test_t_critical_is_the_inverse_of_the_two_sided_p():
    for df in (2.0, 3.0, 4.0, 10.0, 100.0):
        t = cr.t_critical(df, 0.05)
        assert abs(cr.t_two_sided_p(t, df) - 0.05) < 1e-8


def test_detectable_margin_shrinks_with_more_seeds():
    sd = 0.013
    margins = [cr.detectable_margin_pp(sd, n) for n in (3, 4, 5, 10)]
    assert margins == sorted(margins, reverse=True)
    # the threshold the report quotes: ~3 pp at three seeds a side with this project's scatter
    assert 2.0 < margins[0] < 4.0
    assert cr.seeds_needed_for(margins[0], sd) == 3


# --------------------------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------------------------

def test_dense_table_agrees_with_the_reeval_csv():
    cr.verify_dense_against_csv(cr.load_reeval())   # raises on drift


def test_a_seed_that_contradicts_the_csv_fails_loudly(monkeypatch):
    """An edited accuracy must not be able to reach the report."""
    label, family, seeds = cr.DENSE_RUNS[1]         # cm=1.3, whose seed 42 has a CSV row
    seed, run_id, acc, has_csv, best = seeds[0]
    monkeypatch.setattr(cr, "DENSE_RUNS",
                        [(label, family, [(seed, run_id, acc + 0.01, has_csv, best)])])
    with pytest.raises(cr.VerificationError):
        cr.verify_dense_against_csv(cr.load_reeval())


def test_a_new_seed_that_secretly_has_a_csv_row_fails_loudly(monkeypatch):
    """The other direction: marking a run as having no re-score when the CSV holds one."""
    label, family, seeds = cr.DENSE_RUNS[1]
    seed, run_id, acc, _has_csv, _best = seeds[0]
    monkeypatch.setattr(cr, "DENSE_RUNS",
                        [(label, family, [(seed, run_id, acc, False, None)])])
    with pytest.raises(cr.VerificationError):
        cr.verify_dense_against_csv(cr.load_reeval())


# --------------------------------------------------------------------------------------------
# the assembled table and the report
# --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def res():
    # assemble() reads every run's pruning_state.json. checkpoints/ is gitignored and does not
    # travel with a clone, so skip rather than error when the raw results are not present.
    try:
        return cr.assemble()
    except FileNotFoundError as exc:
        pytest.skip("checkpoints/ not present, cannot assemble results: %s" % exc)


def test_every_accuracy_carries_its_seed_count(res):
    for d in res["dense"]:
        assert d["n_seeds"] == len(d["accuracy_values"]) == len(d["seeds"])
        assert (d["accuracy_std"] is None) == (d["n_seeds"] < 2)
    for curve in res["curves"].values():
        for lvl in curve["levels"]:
            assert lvl["n_seeds"] == curve["n_seeds"] == len(lvl["accuracy_values"])
            assert (lvl["accuracy_std"] is None) == (lvl["n_seeds"] < 2)


def test_single_seed_margins_carry_no_test(res):
    for m in res["margins"]:
        one_sided = m["n_seeds_pruned"] < 2 or m["n_seeds_dense"] < 2
        assert one_sided == (m["welch"] is None) == m["single_seed_side"]
        assert not (m["welch"] is None and m["significant_p05"])


def test_dense_shape_macs_are_identical_across_seeds(res):
    """Masking cannot change a tensor's shape, so a difference here means a wrong config
    or a mismatched checkpoint. DSP included: it collapses into a smaller op only at
    deployment."""
    src = res["constants"]["source_model"]["macs"]
    for curve in res["curves"].values():
        for lvl in curve["levels"]:
            assert lvl["macs_dense_shape"] == src


def test_every_finding_has_a_renderer(res):
    """A finding with no renderer would vanish from the report instead of raising."""
    missing = [f["id"] for f in res["findings"] if f["id"] not in results_text.RENDERERS]
    assert not missing, f"findings with no prose renderer: {missing}"


def test_report_renders_and_is_deterministic(res):
    first = results_text.render_report(res)
    second = results_text.render_report(cr.assemble())
    assert first == second
    assert json.dumps(res, indent=2) == json.dumps(cr.assemble(), indent=2)


def _split_off_retractions(text: str) -> tuple[str, str]:
    """Everything outside the Retractions section, and the section itself."""
    start = text.index("\n## Retractions\n")
    end = text.index("\n## Provenance\n", start)
    return text[:start] + text[end:], text[start:end]


def test_retracted_claims_are_absent_from_the_report(res):
    """Claims the multi-seed runs falsified: quotable inside the Retractions section,
    which exists to record what was withdrawn, but asserted nowhere else in the report."""
    outside, section = _split_off_retractions(results_text.render_report(res))
    for dead in ("Pareto-dominates", "beats same-size dense at 4 of 5",
                 "structural weak point at 47"):
        assert dead not in outside, f"retracted claim resurfaced outside Retractions: {dead!r}"
        assert dead in section, (
            f"{dead!r} is no longer recorded in the Retractions section -- a withdrawn claim "
            "was deleted rather than retracted"
        )


def test_every_retraction_that_names_a_level_matches_the_live_margin(res):
    """The hardcoded history must not drift away from the recomputed present."""
    by = {(m["method"], round(m["target"], 4)): m for m in res["margins"]}
    for r in results_text.RETRACTIONS:
        if r["check"] is None:
            continue
        assert r["check"] in by, f"retraction points at a missing level: {r['check']}"
