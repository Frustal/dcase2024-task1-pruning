"""Invariants of pruning/dsp.py (Dynamic Structure Pruning, Park et al. AAAI 2023).

The load-bearing test is test_second_order_contribution_is_nonzero_and_matters. Alpha appears
only in the regulariser, so the one-step-unrolled second-order term is the sole path by which
task accuracy influences the learned grouping. Drop GroupLearner._second_order_grad and every
other test here still passes, with the grouping optimised purely for prunability.

    uv run pytest tests/test_dsp.py -v
"""
from __future__ import annotations

import copy
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.models.baseline import get_model
from pruning.dsp import (
    DSPPruner,
    GroupLearner,
    MEL_INPUT_SHAPE,
    deployable_nonzero_params,
    select_dsp_layers,
    select_fp_layers,
)
from pruning.sparsity import count_params

MODEL = dict(n_classes=10, in_channels=1, base_channels=32,
             channels_multiplier=1.8, expansion_rate=2.1)

# checkpoints/ is gitignored and never travels with a clone, so this may be absent in a fresh
# checkout or in CI. Tests that use it fall back to a randomly initialised model instead of
# skipping: layer selection, mask bisection and the masking/cascade mechanics are properties of
# the algorithm, not of which numbers sit in the weights.
SOURCE_CKPT = "checkpoints/762f0b4fde744636b143bf56646046b7/last.ckpt"


def make_model(seed=0):
    torch.manual_seed(seed)
    return get_model(**MODEL)


def make_dsp_model(seed=0):
    """CP-Mobile, loaded from the real trained checkpoint if present, else random weights."""
    model = make_model(seed)
    if os.path.exists(SOURCE_CKPT):
        ckpt = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)
        # PLModule stores the Network under self.model; strip that prefix. (There are also
        # "mel."-prefixed keys for the spectrogram frontend, which get filtered out here since
        # they don't start with "model.".)
        state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()
                  if k.startswith("model.")}
        model.load_state_dict(state)
    return model


def make_pruner(model, cascade_iters=4, cascade_batch=4, fp_mode="free"):
    """A DSPPruner over the real DSP-prunable layers, cheap cascade settings for test speed."""
    layers = [conv for _, conv in select_dsp_layers(model)]
    fp_layers = select_fp_layers(model, fp_mode)
    return DSPPruner(model, layers, n_groups=2, input_shape=MEL_INPUT_SHAPE,
                      fp_layers=fp_layers, cascade_iters=cascade_iters, cascade_batch=cascade_batch)


class _TinyGroupNet(nn.Module):
    """Minimal stand-in for CP-Mobile: one groupable conv with a non-depthwise layer
    either side so a real forward/backward works. The GroupLearner tests need speed rather than
    fidelity, since the unrolled step redoes a full forward and backward twice per call.
    """

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv1 = nn.Conv2d(4, 6, 3, padding=1)  # the group-learned layer in these tests
        self.conv2 = nn.Conv2d(6, 10, 1)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


def _tiny_batch(batch=4, seed=1):
    torch.manual_seed(seed)
    return torch.randn(batch, 1, 8, 8), torch.randint(0, 10, (batch,))


# --------------------------------------------------------------------------- layer selection


def test_select_dsp_layers_excludes_depthwise_first_conv_and_classifier():
    model = make_model()
    layers = select_dsp_layers(model)

    assert len(layers) == 13
    assert sum(mod.weight.numel() for _, mod in layers) == 52_864

    all_convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    first_name, first_mod = all_convs[0]
    last_name, last_mod = all_convs[-1]
    names = {name for name, _ in layers}

    assert first_name not in names, "the first conv (reads the raw mel input) must be excluded"
    assert last_name not in names, "the classifier (output width == n_classes) must be excluded"
    assert not any(mod is first_mod or mod is last_mod for _, mod in layers)
    for _, mod in layers:
        assert mod.weight.shape[1] != 1, "a depthwise conv (1 input channel) leaked through"


def test_select_fp_layers_free_all_none():
    model = make_model()

    free = select_fp_layers(model, "free")
    assert len(free) == 2
    assert sorted(tuple(l.weight.shape) for l in free) == [(56, 64, 1, 1), (104, 120, 1, 1)]

    # every "free" conv must be the proj_conv of a block WITHOUT an identity shortcut -- that's
    # the whole reason its output width is removable rather than pinned by a residual add
    blocks = [mod for _, mod in model.named_modules()
              if hasattr(mod, "use_shortcut") and hasattr(mod, "block")]
    for conv in free:
        owner = next(b for b in blocks if b.block[2][0] is conv)
        assert owner.use_shortcut is False

    assert len(select_fp_layers(model, "all")) == 6
    assert len(select_fp_layers(model, "none")) == 0

    with pytest.raises(ValueError):
        select_fp_layers(model, "bogus")


# --------------------------------------------------------------------------- group learning


def test_group_assignment_is_a_valid_partition():
    """Soft prob sums to 1 per filter (it's a softmax over the group axis); the hard argmax
    assignment used at prune time is a one-hot partition -- every filter in exactly one group."""
    model = make_model()
    layers = [conv for _, conv in select_dsp_layers(model)]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    learner = GroupLearner(model, optimizer, layers, reg=1e-3, total_steps=10, n_groups=2,
                            tau=0.5, input_shape=MEL_INPUT_SHAPE, group_lr=1e-3)

    learner.initialize()
    for layer in layers:
        sums = layer.prob.sum(dim=0)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    for layer in layers:
        learner._set_arch_hard(layer)
        col_sums = layer.prob.sum(dim=0)
        assert torch.equal(col_sums, torch.ones_like(col_sums)), "not a partition: some filter " \
            "belongs to zero or more than one group"
        assert set(torch.unique(layer.prob).tolist()) <= {0.0, 1.0}, "hard assignment must be 0/1"


def test_group_lasso_regularizer_drives_weakest_slab_toward_zero():
    """Repeated after_step with a large reg must shrink the weakest (group, input-channel)
    slab, since making slabs prunable is what the group-lasso term is for."""
    torch.manual_seed(0)
    model = _TinyGroupNet()
    x, y = _tiny_batch()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    learner = GroupLearner(model, optimizer, [model.conv1], reg=30.0, total_steps=40,
                            n_groups=2, tau=0.5, input_shape=(1, 1, 8, 8), group_lr=1e-2)
    # break the initial tie (group logits start at all-zero, which puts every filter in group 0
    # trivially and leaves group 1's slabs at exactly 0 already -- not a meaningful baseline)
    with torch.no_grad():
        model.conv1.group.copy_(torch.randn_like(model.conv1.group) * 0.5)

    def slab_energies(layer, n_groups):
        with torch.no_grad():
            index = layer.group.max(dim=0, keepdim=True)[1]
            prob = torch.zeros_like(layer.group).scatter_(0, index, 1.0)
            return torch.stack([
                ((prob[g].view(-1, 1, 1, 1) * layer.weight) ** 2).sum(dim=(3, 2, 0))
                for g in range(n_groups)
            ])

    energies0 = slab_energies(model.conv1, 2)
    g0, c0 = divmod(int(energies0.argmin()), energies0.shape[1])
    e0 = energies0[g0, c0].item()
    assert e0 > 1e-4, "degenerate baseline -- weakest slab already ~0 before any training"

    for _ in range(30):
        optimizer.zero_grad()
        learner.initialize()
        F.cross_entropy(model(x), y).backward()
        optimizer.step()
        learner.after_step(lambda m: F.cross_entropy(m(x), y))

    e_final = slab_energies(model.conv1, 2)[g0, c0].item()
    assert e_final < 0.5 * e0, (
        f"weakest slab energy {e0:.4g} -> {e_final:.4g}: expected a material decrease under a "
        f"large group-lasso reg"
    )


def test_second_order_contribution_is_nonzero_and_matters():
    """The load-bearing test in this file; see the module docstring.

    Replays GroupLearner.after_step's own sequence of private calls twice from an identical
    starting point (frozen prob, deep-copied weights and optimizer state), once with
    _second_order_grad and once without, then compares the resulting layer.pgrad. If a refactor
    simplifies the second-order term away, this is the only test that fails.
    """
    torch.manual_seed(0)
    model = _TinyGroupNet()
    x, y = _tiny_batch()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    learner = GroupLearner(model, optimizer, [model.conv1], reg=5.0, total_steps=10,
                            n_groups=2, tau=0.5, input_shape=(1, 1, 8, 8), group_lr=1e-2)
    with torch.no_grad():
        model.conv1.group.copy_(torch.randn_like(model.conv1.group) * 0.5)

    layer = model.conv1
    learner.initialize()
    prob_snapshot = layer.prob.clone()  # frozen so both branches see the identical grouping

    def run_branch(use_second_order: bool) -> torch.Tensor:
        layer.prob = prob_snapshot.clone()
        layer.pgrad = torch.zeros_like(layer.prob)
        layer.buffer = []
        scale = learner._scales[0]

        # -- mirrors GroupLearner.after_step exactly, up to (optionally) the second-order call --
        learner._calc_penalty(layer, scale, buffer=True, alpha_grad=False)
        learner._zero_grad()

        states = copy.deepcopy(model.state_dict())
        learner.optimizer2.load_state_dict(learner.optimizer.state_dict())

        learner._do_penalty(layer)
        learner._checkpoint(layer)

        loss = F.cross_entropy(model(x), y)
        loss.backward()
        learner.optimizer2.step()

        # first-order alpha-gradient contribution only
        learner._calc_penalty(layer, scale, buffer=False, alpha_grad=True)
        learner._do_penalty(layer)
        if use_second_order:
            learner._second_order_grad(layer)

        learner._zero_grad()
        model.load_state_dict(states)  # restore weights for the next branch
        return layer.pgrad.clone()

    pgrad_with_second_order = run_branch(True)
    pgrad_first_order_only = run_branch(False)

    second_order_contribution = pgrad_with_second_order - pgrad_first_order_only
    assert second_order_contribution.norm().item() > 1e-6, (
        "the second-order term contributed nothing -- _second_order_grad is a no-op"
    )
    assert not torch.allclose(pgrad_with_second_order, pgrad_first_order_only, atol=1e-6), (
        "pgrad with and without the second-order term are indistinguishable -- the unrolled "
        "second-order step is the ONLY path by which task accuracy influences the grouping "
        "(see the pruning/dsp.py module docstring); dropping it silently breaks that"
    )


# --------------------------------------------------------------------------- one-shot pruning


@pytest.mark.parametrize("target", [0.4860, 0.75])
def test_bisect_beta_hits_target(target):
    """bisect_beta must land close to the repo-wide sparsity target (fraction of ALL
    58,440 conv params zeroed), not the DSP-internal FLOPs measure the reference bisects on;
    that is what the caller-supplied `measure` is for.

    Tolerance is 0.02 absolute because beta is continuous while sparsity is a step function of
    it. Ten iterations land within ~0.001-0.002 in practice, so 0.02 leaves headroom while
    still catching `measure` wired to the wrong quantity, which was off by tens of points.
    """
    model = make_dsp_model()
    pruner = make_pruner(model)
    torch.manual_seed(0)
    for layer in pruner.layers:
        layer.group.copy_(torch.randn_like(layer.group))

    beta = pruner.bisect_beta(target, measure=lambda: count_params(model).sparsity, n_iter=10)
    assert pruner.beta == beta

    pruner.prune()
    achieved = count_params(model).sparsity
    assert abs(achieved - target) < 0.02, f"target {target}, achieved {achieved} (beta={beta})"


def test_depthwise_convs_never_appear_in_pruner_layers():
    """DSP can only shrink a depthwise conv through the dead-filter cascade, never by
    grouping its single input channel. Checks the pruner's list, not just select_dsp_layers."""
    model = make_dsp_model()
    pruner = make_pruner(model)
    assert len(pruner.layers) == 13
    for layer in pruner.layers:
        assert layer.weight.shape[1] != 1, "a depthwise conv appeared in pruner.layers"


def test_pruned_model_forward_pass_and_deployable_param_count():
    model = make_dsp_model()
    pruner = make_pruner(model)
    torch.manual_seed(2)
    for layer in pruner.layers:
        layer.group.copy_(torch.randn_like(layer.group))
    pruner.beta = 0.3
    pruner.prune()

    out = model(torch.randn(2, *MEL_INPUT_SHAPE[1:]))
    assert out.shape == (2, 10)

    assert deployable_nonzero_params(model) <= count_params(model).nonzero


# --------------------------------------------------------------------------- masking / hooks


def test_after_step_rezeroes_weights_and_bn_bias():
    """after_step must re-zero any weight the mask says is dead and, via
    _residual_bn_proc, any BatchNorm bias whose weight is zero. The cascade zeroes both
    together and nothing else resyncs them after an optimiser step during finetuning.
    """
    model = make_dsp_model()
    pruner = make_pruner(model)
    torch.manual_seed(0)
    for layer in pruner.layers:
        layer.group.copy_(torch.randn_like(layer.group))
    pruner.beta = 0.3
    pruner.prune()

    dead_bn_channels = sum(int((m.weight == 0).sum()) for m in model.modules()
                            if isinstance(m, nn.BatchNorm2d))
    assert dead_bn_channels > 0, "test setup produced no dead BN channels to exercise the hook"

    # perturb: un-zero every masked-dead conv weight, and un-zero a dead BatchNorm's bias
    with torch.no_grad():
        for layer in pruner.layers:
            dead = layer.mask.expand_as(layer.weight) == 0
            layer.weight.add_(torch.randn_like(layer.weight) * dead.float())

        perturbed_a_bn = False
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                dead = m.weight == 0
                if dead.any():
                    m.bias.data[dead] += 7.0
                    perturbed_a_bn = True
        assert perturbed_a_bn, "test setup found no dead BatchNorm bias to perturb"

    pruner.after_step()

    for layer in pruner.layers:
        dead = layer.mask.expand_as(layer.weight) == 0
        assert torch.all(layer.weight[dead] == 0), "after_step failed to re-zero a masked weight"
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            dead = m.weight == 0
            assert torch.all(m.bias[dead] == 0), (
                "_residual_bn_proc failed to re-zero a BatchNorm bias wherever weight is zero"
            )
