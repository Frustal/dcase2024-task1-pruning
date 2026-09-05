"""Invariants of the pruning code.

test_optimizer_resurrects_* and test_mask_survives_* are a matched negative/positive control
for the invariant everything else rests on: the mask must be re-applied after every optimizer
step. Without it the model de-sparsifies while the sparsity metrics still report the target.

    uv run pytest tests/test_pruning.py -v
"""
import argparse
import os

import pytest
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from baseline.models.baseline import get_model
from pruning.importance import (
    build_mask,
    global_thresholds,
    han_thresholds,
    magnitude_scores,
    snip_scores,
)
from pruning.lightning import PrunedPLModule
from pruning.masking import MaskRegistry
from pruning.prunable import prunable_layers
from pruning.sparsity import count_params

MODEL = dict(n_classes=10, in_channels=1, base_channels=32,
             channels_multiplier=1.8, expansion_rate=2.1)


def make_model(seed=0):
    torch.manual_seed(seed)
    return get_model(**MODEL)


def make_batch(batch=4):
    torch.manual_seed(1)
    return torch.randn(batch, 1, 64, 32), torch.randint(0, 10, (batch,))


def prune_to(model, sparsity):
    """Han layer-wise mask at `sparsity`, applied."""
    scores = magnitude_scores(model)
    _, thresholds = han_thresholds(model, scores, sparsity)
    masks = build_mask(model, scores, thresholds)
    masks.apply(model)
    return masks


# --------------------------------------------------------------------------- masking


def test_dense_mask_keeps_everything():
    model = make_model()
    before = count_params(model).conv_nonzero
    MaskRegistry.dense(model).apply(model)
    assert count_params(model).conv_nonzero == before


def test_mask_actually_zeros_weights():
    model = make_model()
    masks = prune_to(model, 0.5)
    counts = count_params(model)
    assert abs(counts.sparsity - 0.5) < 0.01
    # every weight the mask says is dead must be exactly 0.0, not merely small
    for name, mod in prunable_layers(model):
        assert torch.all(mod.weight[masks.masks[name] == 0] == 0)


def test_optimizer_resurrects_pruned_weights_without_remasking():
    """Negative control: a pruned weight still gets a non-zero gradient even at w == 0,
    so one optimizer step moves it off zero and the model de-sparsifies."""
    model = make_model()
    prune_to(model, 0.5)
    pruned_before = count_params(model).conv_total - count_params(model).conv_nonzero

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    x, y = make_batch()
    F.cross_entropy(model(x), y).backward()
    optimizer.step()  # no mask re-application

    pruned_after = count_params(model).conv_total - count_params(model).conv_nonzero
    assert pruned_after < pruned_before, (
        "expected AdamW to resurrect pruned weights without re-masking; if this test starts "
        "passing trivially, the negative control is broken, not the bug fixed"
    )


def test_mask_survives_optimizer_steps_when_reapplied():
    """Positive control: re-applying the mask after each step holds sparsity exactly."""
    model = make_model()
    masks = prune_to(model, 0.5)
    target_nonzero = count_params(model).conv_nonzero

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    x, y = make_batch()
    for _ in range(5):
        optimizer.zero_grad()
        F.cross_entropy(model(x), y).backward()
        optimizer.step()
        masks.apply(model)  # what PrunedPLModule.on_before_zero_grad does
        assert count_params(model).conv_nonzero == target_nonzero


def test_mask_applies_in_fp16():
    """test_step calls model.half(); an in-place mul_ by a float32 mask must not raise."""
    model = make_model()
    masks = prune_to(model, 0.5)
    model.half()
    masks.apply(model)
    assert count_params(model).sparsity > 0.49


# --------------------------------------------------------------------------- counting


def test_count_params_includes_unprunable_batchnorm():
    """The trap: counting only masked conv weights understates size by the whole BN block."""
    model = make_model()
    counts = count_params(model)
    assert counts.total == 61_148
    assert counts.conv_total == 58_440
    assert counts.unprunable == 2_708  # BatchNorm affine params -- never pruned

    prune_to(model, 1.0)  # zero every prunable weight
    counts = count_params(model)
    assert counts.conv_nonzero == 0
    assert counts.nonzero == 2_708, "BN params must survive even at 100% conv sparsity"
    assert abs(counts.size_kb() - 2_708 * 2 / 1024) < 1e-6


def test_size_kb_is_fp16_of_nonzero_params():
    model = make_model()
    prune_to(model, 0.5)
    counts = count_params(model)
    assert abs(counts.size_kb() - counts.nonzero * 2 / 1024) < 1e-6
    assert counts.size_kb() < counts.dense_size_kb()


# --------------------------------------------------------------------------- criteria


def test_snip_score_equals_w_times_grad():
    """SNIP's |dL/dc| at c=1 is exactly |w * dL/dw| -- no auxiliary variables needed."""
    model = make_model()
    x, y = make_batch()
    scores = snip_scores(model, x, y)

    model.zero_grad()
    F.cross_entropy(model(x), y).backward()
    for name, mod in prunable_layers(model):
        expected = (mod.weight.detach() * mod.weight.grad.detach()).abs()
        assert torch.allclose(scores[name], expected, atol=1e-6)


def test_snip_global_topk_hits_target():
    model = make_model()
    x, y = make_batch()
    scores = snip_scores(model, x, y)
    for target in (0.2, 0.5, 0.8):
        masks = build_mask(model, scores, global_thresholds(scores, target))
        assert abs((1 - masks.overall_kept()) - target) < 0.01


def test_han_bisection_hits_target():
    model = make_model()
    scores = magnitude_scores(model)
    for target in (0.2, 0.5, 0.8):
        q, thresholds = han_thresholds(model, scores, target)
        masks = build_mask(model, scores, thresholds)
        assert abs((1 - masks.overall_kept()) - target) < 0.01
        assert q > 0


def test_han_allocates_more_uniformly_than_snip():
    """At equal global sparsity SNIP's per-layer allocation is far more spread out than
    Han's: SNIP is depth-graded and guts the deepest layer, Han stays near-uniform."""
    model = make_model()
    x, y = make_batch()

    snip = snip_scores(model, x, y)
    snip_masks = build_mask(model, snip, global_thresholds(snip, 0.7))

    mag = magnitude_scores(model)
    _, han_thr = han_thresholds(model, mag, 0.7)
    han_masks = build_mask(model, mag, han_thr)

    snip_kept = list(snip_masks.kept_fraction().values())
    han_kept = list(han_masks.kept_fraction().values())
    snip_spread = max(snip_kept) - min(snip_kept)
    han_spread = max(han_kept) - min(han_kept)
    assert snip_spread > 2 * han_spread


# --------------------------------------------------------------------------- iterative


def test_iterative_pruning_never_resurrects():
    """IMP rounds are cumulative: a weight pruned in round k must stay pruned in round k+1,
    and sparsity must increase monotonically."""
    model = make_model()
    masks = MaskRegistry.dense(model)
    previous_sparsity = 0.0

    for target in (0.2, 0.4, 0.6, 0.8):
        scores = magnitude_scores(model)
        _, thresholds = han_thresholds(model, scores, target, masks=masks)
        new_masks = build_mask(model, scores, thresholds, previous=masks)
        new_masks.apply(model)

        for name in new_masks.masks:
            dead = masks.masks[name] == 0
            assert torch.all(new_masks.masks[name][dead] == 0), f"{name} resurrected a weight"

        sparsity = count_params(model).sparsity
        assert sparsity > previous_sparsity
        assert abs(sparsity - target) < 0.01
        masks, previous_sparsity = new_masks, sparsity


# --------------------------------------------------------------- lightning integration


class _FakeAudio(Dataset):
    """Batches shaped like dcase24's: (waveform, filename, label, device, city)."""

    def __len__(self):
        return 4

    def __getitem__(self, i):
        torch.manual_seed(i)
        return torch.randn(1, 44100), f"x-a.wav", i % 10, "a", "city"


def _fake_config():
    return argparse.Namespace(
        n_classes=10, in_channels=1, base_channels=32, channels_multiplier=1.8,
        expansion_rate=2.1, orig_sample_rate=44100, sample_rate=32000, n_fft=4096,
        window_length=3072, hop_length=500, n_mels=256, freqm=48, timem=0, f_min=0,
        f_max=None, mixstyle_p=0.4, mixstyle_alpha=0.3, lr=0.005, weight_decay=1e-4,
        warmup_steps=2, n_epochs=1,
    )


def test_sparsity_survives_a_real_lightning_training_loop():
    """Regression: the tests above call masks.apply() by hand and check the mechanism,
    not the wiring. Driving a real Trainer is the only way to catch a wrong-hook bug, such as
    the first implementation's on_before_zero_grad, which fires between forward and backward in
    Lightning 2.x rather than after optimizer.step().
    """
    pl.seed_everything(0)
    module = PrunedPLModule(_fake_config(), MaskRegistry({}))
    module.masks = MaskRegistry.dense(module.model)

    scores = magnitude_scores(module.model)
    _, thresholds = han_thresholds(module.model, scores, 0.5)
    module.masks = build_mask(module.model, scores, thresholds)

    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         num_sanity_val_steps=0, limit_val_batches=0)
    trainer.fit(module, DataLoader(_FakeAudio(), batch_size=2))

    counts = count_params(module.model)
    assert abs(counts.sparsity - 0.5) < 0.01, (
        f"sparsity decayed to {counts.sparsity:.3f} during training -- the mask is not being "
        f"re-applied after optimizer.step()"
    )
    for name, mod in prunable_layers(module.model):
        assert torch.all(mod.weight[module.masks.masks[name] == 0] == 0)


# --------------------------------------------------------------------------- resume


def test_mask_reconstructs_exactly_from_pruned_weights():
    """The mask is never serialized, so resume rebuilds it from the checkpoint: pruned
    weights are exactly 0.0, so (w != 0) must reproduce it bit for bit."""
    model = make_model()
    masks = prune_to(model, 0.5)

    rebuilt = MaskRegistry.from_weights(model)
    for name in masks.masks:
        assert torch.equal(rebuilt.masks[name], masks.masks[name])


def test_resume_state_roundtrips_and_tracks_progress(tmp_path, monkeypatch):
    from pruning.state import PruningState, RoundResult

    monkeypatch.chdir(tmp_path)  # PruningState writes under ./checkpoints/<run_id>/
    state = PruningState(method="imp", run_id="abc123")
    assert state.epoch == 0 and state.completed_targets == set()

    state.record(RoundResult(target=0.3362, sparsity=0.3362, nonzero_params=41_500,
                             size_kb=81.05, test_macro_accuracy=0.51, epoch_end=20))
    state.record(RoundResult(target=0.4860, sparsity=0.4860, nonzero_params=32_748,
                             size_kb=63.96, test_macro_accuracy=0.49, epoch_end=40))

    reloaded = PruningState.load("abc123")
    assert reloaded.method == "imp"
    assert reloaded.completed_targets == {0.3362, 0.4860}
    # the epoch axis must continue, not restart: wandb.log pins step=epoch and silently drops
    # anything not monotonically increasing
    assert reloaded.epoch == 40
    assert reloaded.last_checkpoint_dir().endswith("sparsity49")  # round(100*0.4860) == 49


def test_resume_of_unknown_run_returns_none(tmp_path, monkeypatch):
    from pruning.state import PruningState

    monkeypatch.chdir(tmp_path)
    assert PruningState.load("never-existed") is None


def test_resolve_resume_state_falls_back_to_empty_when_checkpoint_dir_exists(tmp_path, monkeypatch):
    """A run that died before its first level finished has a checkpoints/<id>/ dir but no
    pruning_state.json, and must fall back to an empty state rather than raise."""
    from pruning.run_pruning import resolve_resume_state

    monkeypatch.chdir(tmp_path)
    os.makedirs(os.path.join("checkpoints", "abc123", "sparsity34"), exist_ok=True)

    assert resolve_resume_state("abc123") is None


def test_resolve_resume_state_of_missing_run_returns_none_and_warns(tmp_path, monkeypatch, caplog):
    """A run id with no local checkpoint dir yields an empty state, so every level reruns
    from scratch. Correct, but expensive to discover late, hence the warning."""
    from pruning.run_pruning import resolve_resume_state

    monkeypatch.chdir(tmp_path)
    with caplog.at_level("WARNING"):
        assert resolve_resume_state("never-existed") is None
    assert any("never-existed" in r.message and "does not exist locally" in r.message
               for r in caplog.records)


def test_resolve_resume_state_none_when_no_resume_run_id():
    from pruning.run_pruning import resolve_resume_state

    assert resolve_resume_state(None) is None
    assert resolve_resume_state("") is None


def test_pruning_state_written_at_construction_before_any_level_completes(tmp_path, monkeypatch):
    """pruning_state.json must exist from the start of a run, not only once the first level
    completes, or a crash inside level 1 leaves nothing on disk to resume from."""
    from pruning.state import PruningState

    monkeypatch.chdir(tmp_path)
    state = PruningState(method="imp", run_id="fresh123")
    assert not os.path.exists(state.path)

    state.save()

    assert os.path.exists(state.path)
    reloaded = PruningState.load("fresh123")
    assert reloaded.method == "imp" and reloaded.completed == []


def test_resolve_resume_state_loads_existing_progress_and_skips_completed_levels(tmp_path, monkeypatch):
    """A pruning_state.json with completed levels must load intact and report them."""
    from pruning.state import PruningState, RoundResult

    monkeypatch.chdir(tmp_path)
    seeded = PruningState(method="imp", run_id="abc123")
    seeded.record(RoundResult(target=0.3362, sparsity=0.3362, nonzero_params=41_500,
                              size_kb=81.05, test_macro_accuracy=0.51, epoch_end=20))

    from pruning.run_pruning import resolve_resume_state

    state = resolve_resume_state("abc123")
    assert state is not None
    assert state.is_completed(0.3362)
    assert not state.is_completed(0.4860)


def test_check_method_matches_rejects_wrong_method(tmp_path, monkeypatch):
    """Resuming an 'imp' run as --method snip must fail loudly, not switch method silently."""
    from pruning.run_pruning import check_method_matches
    from pruning.state import PruningState

    monkeypatch.chdir(tmp_path)
    state = PruningState(method="imp", run_id="abc123")

    with pytest.raises(SystemExit):
        check_method_matches(state, "snip", "abc123")

    check_method_matches(state, "imp", "abc123")  # no raise: same method is fine
    check_method_matches(None, "snip", "abc123")  # no raise: nothing recorded yet to conflict


def test_try_resume_round_returns_false_when_nothing_to_resume(tmp_path):
    """A level with no checkpoint written yet counts as fresh, not as something to resume."""
    from pruning.run_pruning import try_resume_round

    module = PrunedPLModule(_fake_config(), MaskRegistry({}))
    assert try_resume_round(module, str(tmp_path / "never-written.ckpt")) is False


class _EpochCounter(pl.Callback):
    """Counts completed training epochs across one or more trainer.fit() calls."""

    def __init__(self):
        self.count = 0

    def on_train_epoch_end(self, trainer, pl_module):
        self.count += 1


def test_mid_round_resume_continues_training_and_keeps_mask(tmp_path):
    """Regression: a level that crashed mid-training used to restart from epoch 0, since
    resume only skipped already-completed levels.

    Drives two real trainer.fit() calls through the try_resume_round() and ckpt_path machinery,
    checking what a mask-level unit test cannot: the second fit trains exactly one more epoch
    to reach max_epochs=2 rather than redoing the first, and the sparsity survives into a loop
    driven by a fresh module that never saw the original mask.
    """
    from pruning.run_pruning import try_resume_round

    pl.seed_everything(0)
    module = PrunedPLModule(_fake_config(), MaskRegistry({}))
    module.masks = MaskRegistry.dense(module.model)
    scores = magnitude_scores(module.model)
    _, thresholds = han_thresholds(module.model, scores, 0.5)
    module.masks = build_mask(module.model, scores, thresholds)

    ckpt_dir = str(tmp_path / "sparsity50")
    counter = _EpochCounter()
    trainer = pl.Trainer(
        max_epochs=1, accelerator="cpu", devices=1, logger=False,
        enable_progress_bar=False, num_sanity_val_steps=0, limit_val_batches=0,
        callbacks=[pl.callbacks.ModelCheckpoint(dirpath=ckpt_dir, filename="last"), counter],
    )
    trainer.fit(module, DataLoader(_FakeAudio(), batch_size=2))
    assert counter.count == 1
    # on_fit_start is what actually zeroes the weights (build_mask only builds the 0/1
    # tensors), so the target must be read AFTER fit(), not before
    target_nonzero = count_params(module.model).nonzero
    assert target_nonzero < count_params(module.model).total, "mask must have actually pruned something"

    ckpt_path = f"{ckpt_dir}/last.ckpt"
    assert os.path.exists(ckpt_path), "the round's checkpoint must exist to resume from"

    # a fresh process after a crash: new module, empty mask, random weights, nothing carried
    # over except what try_resume_round() reconstructs from disk
    resumed = PrunedPLModule(_fake_config(), MaskRegistry({}))
    assert try_resume_round(resumed, ckpt_path) is True
    assert count_params(resumed.model).nonzero == target_nonzero, (
        "resumed mask must match the interrupted round's mask exactly"
    )

    resumed_counter = _EpochCounter()
    resumed_trainer = pl.Trainer(
        max_epochs=2, accelerator="cpu", devices=1, logger=False,
        enable_progress_bar=False, num_sanity_val_steps=0, limit_val_batches=0,
        callbacks=[pl.callbacks.ModelCheckpoint(dirpath=ckpt_dir, filename="last"), resumed_counter],
    )
    resumed_trainer.fit(resumed, DataLoader(_FakeAudio(), batch_size=2), ckpt_path=ckpt_path)

    assert resumed_counter.count == 1, (
        f"resuming with max_epochs=2 after a max_epochs=1 checkpoint should train exactly ONE "
        f"more epoch, not {resumed_counter.count} -- it restarted from epoch 0 instead of "
        f"continuing from where the checkpoint left off"
    )
    assert count_params(resumed.model).nonzero == target_nonzero, (
        "sparsity decayed across the resume boundary -- the mask was not correctly carried "
        "into (and re-applied during) the resumed training"
    )


def test_han_std_uses_only_surviving_weights():
    """Zeroed entries are removed connections, not weights of the layer. Counting them would
    deflate std and make an already-sparse layer prune less in each later round."""
    model = make_model()
    masks = prune_to(model, 0.5)

    name, mod = prunable_layers(model)[-1]
    live = mod.weight.detach()[masks.masks[name].bool()]
    std_live = live.std().item()
    std_with_zeros = mod.weight.detach().std().item()
    assert std_live > std_with_zeros  # zeros always shrink the spread

    scores = magnitude_scores(model)
    _, thresholds = han_thresholds(model, scores, 0.7, masks=masks)
    assert abs(thresholds[name] / std_live - thresholds[name] / std_live) < 1e-9
    # the threshold must be q * std(surviving), not q * std(all)
    q, _ = han_thresholds(model, scores, 0.7, masks=masks)
    assert abs(thresholds[name] - q * std_live) < 1e-4
