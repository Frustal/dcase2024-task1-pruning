"""Dynamic Structure Pruning (DSP), Park et al., AAAI 2023 (arXiv:2303.09736).

Ported from the authors' released implementation (https://github.com/irishev/DSP). Group learning
(:class:`GroupLearner`) trains a per-conv logit matrix jointly with the weights under a group-lasso
regulariser; :class:`DSPPruner` then discretises that grouping and drops each group's lowest-energy
input channels in one shot, cheaply enough that its ``beta`` budget can be bisected to hit a target
size; fine-tuning re-applies the resulting mask after every optimiser step.

Deviations from the reference, each marked ``DEVIATION`` at its site:

* :meth:`DSPPruner.bisect_beta` solves for ``beta`` against a caller-supplied parameter-sparsity
  measure rather than the reference's internal FLOPs percentage.
* :func:`_penalty_scale` defaults to ``mode="params"``, dropping the ``sqrt(act_size)`` factor from
  the per-layer regulariser scale.
* :func:`select_fp_layers` defaults to ``mode="free"``, restricting whole-filter pruning to blocks
  whose output width is not pinned by an identity shortcut.

The one-step-unrolled second-order term (:meth:`GroupLearner._second_order_grad`) must not be
replaced by a first-order DARTS approximation. In DARTS the architecture variable appears in the
forward pass; in DSP it does not, because the network runs dense during group learning and ``alpha``
enters only through the regulariser. That term is the only path from task loss to the grouping, so
without it the grouping is optimised for prunability alone.
"""

from __future__ import annotations

import copy
from typing import Callable, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.helpers.mixstyle import mixstyle
from pruning.lightning import PrunedPLModule

__all__ = [
    "select_dsp_layers",
    "select_fp_layers",
    "deployable_nonzero_params",
    "GroupLearner",
    "DSPPruner",
    "GroupLearningPLModule",
    "MEL_INPUT_SHAPE",
]

# Tensor entering the CNN: (batch, 1, n_mels, frames); 65 frames from 1 s clips resampled
# 44100 -> 32000 at hop_length=500. Used by the dry forward pass and the cascade's random batches.
MEL_INPUT_SHAPE = (1, 1, 256, 65)


# ---- Layer selection ------------------------------------------------------------------------


def select_dsp_layers(
    model: nn.Module,
    skip_first_conv: bool = True,
    skip_classifier: bool = True,
    exclude_substrings: Sequence[str] = (),
) -> list[tuple[str, nn.Conv2d]]:
    """Return the ``(name, module)`` pairs DSP can group-prune, in forward order.

    Depthwise convs are excluded: a depthwise filter has one input channel, so there is nothing to
    group on, and CP-Mobile's six shrink only via the cascade in :meth:`DSPPruner.prune`. The first
    conv (single-channel mel input) and the classifier (fixed output width) are excluded too.
    """
    selected: list[tuple[str, nn.Conv2d]] = []
    convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("model contains no Conv2d layers")

    first_name = convs[0][0]
    last_name = convs[-1][0]
    for name, conv in convs:
        if skip_first_conv and name == first_name:
            continue
        if skip_classifier and name == last_name:
            continue
        if any(s in name for s in exclude_substrings):
            continue
        if conv.weight.shape[1] == 1:  # depthwise: nothing to group on the input axis
            continue
        selected.append((name, conv))
    return selected


def select_fp_layers(model: nn.Module, mode: str = "free") -> list[nn.Conv2d]:
    """Return the convs whose output filters DSP may prune, i.e. whole-filter pruning.

    The reference applies this to the last conv of every residual block (``fp_every_nth_conv``),
    which on CP-Mobile is each block's ``proj_conv``. Blocks b1, b2, b3 and b5 use an identity
    shortcut (no skip branch here carries a 1x1 projection), so the elementwise add pins their
    ``proj_conv`` output width to the block's input width: zeroing such a filter still computes
    correctly, but the channel cannot be removed. Blocks b4 and b6 have no shortcut and are free.

    DEVIATION: the default ``"free"`` takes only the unconstrained blocks, so reported structured
    compression matches what is removable without structural packing; ``"all"`` reproduces the
    reference, kept so a range probe can measure both from one group-learned checkpoint at no extra
    training cost; ``"none"`` disables filter pruning.

    ``exp_conv`` filters are not candidates but still die in the cascade, which is DSP's only route
    to the depthwise convs each of them feeds.
    """
    if mode == "none":
        return []
    if mode not in ("free", "all"):
        raise ValueError(f"unknown fp mode {mode!r}, expected 'free', 'all' or 'none'")

    fp: list[nn.Conv2d] = []
    for _, module in model.named_modules():
        if not (hasattr(module, "use_shortcut") and hasattr(module, "block")):
            continue
        proj = module.block[2][0]
        if not isinstance(proj, nn.Conv2d):
            continue
        if mode == "all" or not module.use_shortcut:
            fp.append(proj)
    return fp


@torch.no_grad()
def attach_dsp_buffers(model: nn.Module, n_groups: int, with_mask: bool = True) -> None:
    """Register the ``group`` (and optionally ``mask``) buffers on every DSP-prunable conv.

    Must run on a freshly built model before a DSP checkpoint is loaded into it, since
    ``load_state_dict`` is strict in both directions and a phase-A checkpoint carries ``group``
    alone while a phase-C one carries both. Pass ``with_mask=False`` when forking from phase A.

    Buffers, never ``nn.Parameter``: ``pruning.sparsity.count_params`` counts everything in
    ``model.parameters()`` that is not a prunable conv weight in full, so parameters here would add
    1,680 to every reported ``nonzero_params`` and 3.3 KB to every ``size_kb``. :class:`GroupLearner`
    does optimise ``group`` as a parameter, but only in phase A, which reports no curve point.
    """
    for _, conv in select_dsp_layers(model):
        if not hasattr(conv, "group"):
            conv.register_buffer(
                "group", torch.zeros(n_groups, conv.weight.shape[0], device=conv.weight.device)
            )
        if with_mask and not hasattr(conv, "mask"):
            conv.register_buffer(
                "mask",
                torch.ones(conv.weight.shape[0], conv.weight.shape[1], 1, 1, device=conv.weight.device),
            )


@torch.no_grad()
def deployable_nonzero_params(model: nn.Module) -> int:
    """Non-zero count across every parameter, BatchNorm included.

    Reported alongside, never instead of, ``pruning.sparsity.count_params``, which counts conv
    non-zeros plus BatchNorm in full. That convention fits IMP and SNIP, whose masks never touch BN
    and whose conv zeros need sparse kernels to become real savings. DSP differs on both counts, so
    this is the number it actually delivers.
    """
    return int(sum(int((p != 0).sum().item()) for p in model.parameters()))


@torch.no_grad()
def _record_layer_stats(model: nn.Module, layers: Iterable[nn.Conv2d], input_shape: Sequence[int]) -> None:
    """Attach ``.flops`` and ``.act_size`` to each layer via a single dry forward pass.

    ``act_size`` is the layer's input activation element count. The reference divides it by a fixed
    ``16*32*32`` (CIFAR-shaped); this port normalises by the network's own input element count, so
    the quantity is dimensionless and architecture-independent. Only relative values across layers
    matter, any global factor being absorbed into ``reg``.
    """
    handles = []
    norm = 1
    for s in input_shape:
        norm *= s

    def make_hook(module: nn.Conv2d):
        def hook(mod, inp, out):
            flops = mod.weight.numel() * out.shape[2] * out.shape[3]
            act = 1
            for s in inp[0].shape:
                act *= s
            mod.flops = float(flops)
            mod.act_size = float(act) / float(norm)

        return module.register_forward_hook(hook)

    for layer in layers:
        handles.append(make_hook(layer))

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    model(torch.randn(*input_shape, device=device))
    if was_training:
        model.train()
    for h in handles:
        h.remove()


def _penalty_scale(layer: nn.Conv2d, n_groups: int, mode: str) -> float:
    """Per-layer weighting of the group-lasso penalty.

    ``mode="flops"`` reproduces the released implementation:
    ``sqrt(act_size) * kernel_h * (n_groups / out_channels) ** 1.5``.

    DEVIATION: the default ``mode="params"`` drops the ``sqrt(act_size)`` term, which weights each
    layer by its activation footprint and is what makes the released code prune FLOPs preferentially
    (its README: "we slightly changed the implementation of regularization scaling to obtain better
    speedup ... usually more pruned FLOPS and fewer pruned parameters"). These experiments match
    methods on parameter count, and on CP-Mobile the objectives disagree: early layers are spatially
    large but narrow, late layers small but wide and holding most of the weights
    (``stages.s3.b6.block.2.0`` alone has 12,480 of 61,148), so the activation term would steer
    pruning away from exactly the parameters that have to go. ``(n_groups / out_channels) ** 1.5``
    is retained: it normalises for how many filters share a group and is not FLOPs-specific.
    """
    k = float(layer.weight.shape[2])
    group_norm = (float(n_groups) / float(layer.weight.shape[0])) ** 1.5
    if mode == "flops":
        return (layer.act_size ** 0.5) * k * group_norm
    if mode == "params":
        return k * group_norm
    raise ValueError(f"unknown penalty_scale mode {mode!r}, expected 'flops' or 'params'")


# ---- Stage 1: differentiable group learning --------------------------------------------------


class GroupLearner:
    """Learns the filter grouping jointly with the weights (reference: ``GroupWrapper``).

    Not an ``nn.Module``: it decorates a model in place and is driven by a Lightning module that
    owns the training loop. Call :meth:`initialize` before each forward pass, :meth:`after_step`
    after the optimiser step.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        layers: Sequence[nn.Conv2d],
        reg: float,
        total_steps: int,
        n_groups: int,
        tau: float,
        input_shape: Sequence[int],
        group_lr: float = 1e-3,
        penalty_scale: str = "params",
        order: float = 0.5,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.layers = list(layers)
        self.reg = reg
        self.total_steps = max(int(total_steps), 1)
        self.n_groups = n_groups
        self.tau = tau
        self.order = order
        self.penalty_scale = penalty_scale
        self.steps = 0

        group_parameters = []
        for layer in self.layers:
            if not hasattr(layer, "group"):
                layer.register_parameter(
                    "group",
                    nn.Parameter(torch.zeros(n_groups, layer.weight.shape[0], device=layer.weight.device)),
                )
            group_parameters.append(layer.group)

        # A second optimizer of the task optimizer's type, used only for the unrolled trial step so
        # the real optimizer's state is never advanced. Its param_groups are mirrored rather than
        # rebuilt from model.parameters(), because the `group` parameters registered above belong to
        # the model but are trained by `group_optimizer`, which would break the load_state_dict.
        self.optimizer2 = type(optimizer)(
            [{"params": list(g["params"])} for g in optimizer.param_groups],
            lr=optimizer.defaults["lr"],
        )
        self.group_optimizer = torch.optim.Adam(group_parameters, lr=group_lr, eps=1e-12)

        _record_layer_stats(model, self.layers, input_shape)
        self._scales = [_penalty_scale(l, n_groups, penalty_scale) for l in self.layers]

    # -- assignment ---------------------------------------------------------------------------

    def _set_arch(self, layer: nn.Conv2d) -> None:
        """Sample a soft group assignment. ``prob`` keeps its graph back to ``layer.group``."""
        layer.prob = F.gumbel_softmax(layer.group / self.tau, dim=0)
        layer.pgrad = torch.zeros_like(layer.prob)
        layer.buffer = []

    @torch.no_grad()
    def _set_arch_hard(self, layer: nn.Conv2d) -> None:
        index = layer.group.max(dim=0, keepdim=True)[1]
        layer.prob = torch.zeros_like(layer.group).scatter_(0, index, 1.0)

    def initialize(self) -> None:
        """Resample the relaxed grouping. Must run before every forward pass."""
        for layer in self.layers:
            self._set_arch(layer)

    # -- regulariser --------------------------------------------------------------------------

    @torch.no_grad()
    def _calc_penalty(self, layer: nn.Conv2d, scale: float, *, buffer: bool, alpha_grad: bool) -> None:
        """Accumulate ``dR/dW`` into ``layer.penalty``.

        ``R = s(alpha) * lasso(W, alpha)``, with ``lasso`` the group-lasso norm over each
        (group, input-channel) slab and ``s(alpha)`` the order-``p`` group norm. The reference's
        three entry points differ only in whether they stash reusable terms for the second-order
        step and whether they also accumulate ``dR/dalpha``; the two flags unify them.
        """
        layer.penalty = torch.zeros_like(layer.weight)
        for p, pg in zip(layer.prob, layer.pgrad):
            group_weights = p.view(-1, 1, 1, 1) * layer.weight
            lasso = (group_weights ** 2).sum(dim=(3, 2, 0), keepdim=True) ** 0.5
            normalized_weights = group_weights / (1e-8 + lasso)
            dlasso_dw = p.view(-1, 1, 1, 1) * normalized_weights

            g_order = p ** self.order
            g_order_sum = g_order.sum()
            gnorm = g_order_sum ** (1 / self.order)

            dR_dw = gnorm * dlasso_dw
            layer.penalty.add_(scale * dR_dw)

            if not (buffer or alpha_grad):
                continue

            dlasso_da = (layer.weight * normalized_weights).sum(dim=(3, 2, 1))
            dgnorm_da = g_order * gnorm / (p * g_order_sum + 1e-8)

            if buffer:
                layer.buffer.append((dgnorm_da, dlasso_dw, gnorm, normalized_weights, dlasso_da, lasso))
            if alpha_grad:
                pg.add_(gnorm * dlasso_da + lasso.sum() * dgnorm_da)

    @torch.no_grad()
    def _second_order_grad(self, layer: nn.Conv2d) -> None:
        """The one-step-unrolled second-order term ``d2R/dalpha dW``.

        ``grad`` is how far the weights moved during the trial step, the unrolled
        ``-xi * d(L + reg*R)/dW``. Contracting it with ``d2R/dalpha dW`` is what tells the grouping
        which assignments the task loss prefers; without it ``alpha`` gets no accuracy signal.
        """
        grad = layer.weight - layer.checkpoint
        for pg, buf in zip(layer.pgrad, layer.buffer):
            dgnorm_da, dlasso_dw, gnorm, normalized_weights, dlasso_da, lasso = buf
            d2R_dadw = dgnorm_da * (dlasso_dw * grad).sum(dim=(3, 2, 1)) + gnorm * (
                (2 * normalized_weights * grad).sum(dim=(3, 2, 1))
                - dlasso_da * (dlasso_dw * grad / lasso).sum(dim=(3, 2, 1))
            )
            pg.add_(d2R_dadw)

    @torch.no_grad()
    def _do_penalty(self, layer: nn.Conv2d) -> None:
        """Proximal-style penalty step, ramped linearly over training as in the reference."""
        lr = self.optimizer.param_groups[0]["lr"]
        alpha = -self.reg * lr * (self.steps + 1) / self.total_steps
        layer.weight.add_(layer.penalty, alpha=alpha)

    @torch.no_grad()
    def _checkpoint(self, layer: nn.Conv2d) -> None:
        layer.checkpoint = layer.weight.clone().detach()

    def _zero_grad(self) -> None:
        self.optimizer.zero_grad(True)
        self.group_optimizer.zero_grad(True)

    # -- the unrolled step --------------------------------------------------------------------

    def after_step(self, loss_closure: Callable[[nn.Module], torch.Tensor]) -> None:
        """One bilevel update of the grouping. Call after the task optimiser has stepped.

        ``loss_closure(model)`` must run a forward pass and return a scalar loss. A closure rather
        than ``(x, y)`` lets the caller reproduce its own training loss exactly, here including the
        Frequency-MixStyle and mixup applied inside ``training_step``. Statement order follows the
        reference: the second ``_do_penalty`` runs before ``_second_order_grad``, so the measured
        weight delta is the full unrolled step, task gradient plus regulariser.
        """
        for layer, scale in zip(self.layers, self._scales):
            self._calc_penalty(layer, scale, buffer=True, alpha_grad=False)
        self._zero_grad()

        states = copy.deepcopy(self.model.state_dict())
        self.optimizer2.load_state_dict(self.optimizer.state_dict())

        for layer in self.layers:
            self._do_penalty(layer)
            self._checkpoint(layer)

        loss = loss_closure(self.model)
        loss.backward()
        self.optimizer2.step()

        for layer, scale in zip(self.layers, self._scales):
            self._calc_penalty(layer, scale, buffer=False, alpha_grad=True)
        for layer in self.layers:
            self._do_penalty(layer)
        for layer in self.layers:
            self._second_order_grad(layer)

        self._zero_grad()
        self.model.load_state_dict(states)

        # Backprop the accumulated dR/dalpha through the Gumbel-Softmax into `group`.
        for layer in self.layers:
            (layer.prob * layer.pgrad).sum().backward()
        self.group_optimizer.step()

        # Final penalty application under the hard (argmax) assignment.
        for layer in self.layers:
            self._set_arch_hard(layer)
        for layer, scale in zip(self.layers, self._scales):
            self._calc_penalty(layer, scale, buffer=False, alpha_grad=False)
        for layer in self.layers:
            self._do_penalty(layer)

        self.steps += 1

    @torch.no_grad()
    def group_logit_std(self) -> float:
        """Mean std of the group logits; near zero means the grouping has not committed yet."""
        stds = [layer.group.std().item() for layer in self.layers]
        return sum(stds) / len(stds)


# ---- Stage 2: one-shot group-channel pruning -------------------------------------------------


class DSPPruner:
    """Applies group-channel pruning to a group-learned model (reference: ``PruneWrapper``)."""

    def __init__(
        self,
        model: nn.Module,
        layers: Sequence[nn.Conv2d],
        n_groups: int,
        input_shape: Sequence[int],
        fp_layers: Sequence[nn.Conv2d] = (),
        cascade_iters: int = 64,
        cascade_batch: int = 16,
    ) -> None:
        self.model = model
        self.layers = list(layers)
        self.fp_layers = list(fp_layers)
        self.n_groups = n_groups
        self.input_shape = list(input_shape)
        self.cascade_iters = cascade_iters
        self.cascade_batch = cascade_batch
        self.beta = 0.0

        for layer in self.layers:
            if not hasattr(layer, "group"):
                layer.register_buffer(
                    "group", torch.zeros(n_groups, layer.weight.shape[0], device=layer.weight.device)
                )
            if not hasattr(layer, "mask"):
                layer.register_buffer(
                    "mask",
                    torch.ones(layer.weight.shape[0], layer.weight.shape[1], 1, 1, device=layer.weight.device),
                )

        _record_layer_stats(model, self.layers, input_shape)

    @torch.no_grad()
    def _set_arch_hard(self, layer: nn.Conv2d) -> None:
        index = layer.group.max(dim=0, keepdim=True)[1]
        layer.prob = torch.zeros_like(layer.group).scatter_(0, index, 1.0)

    @torch.no_grad()
    def _find_mask(self, layer: nn.Conv2d) -> None:
        """Per group, drop the lowest-energy input channels while staying under ``beta``."""
        layer.mask.fill_(1)
        importance = layer.weight.data ** 2
        imp = torch.stack([((p.view(-1, 1, 1, 1) ** 2) * importance).sum(dim=(3, 2, 0)) for p in layer.prob], dim=0)
        imp = imp / (imp.sum(dim=1, keepdim=True) + 1e-12)
        rank = imp.sort(dim=1)[0]
        csoi = rank.cumsum(dim=1)
        count = (csoi < self.beta).long().sum(dim=1)
        # clamp: at beta near 0 no channel qualifies, count is 0, and `count - 1` indexes -1, which
        # picks the LARGEST channel as the threshold and wipes the group instead of sparing it.
        # Inherited from the reference, whose bisection never reaches beta = 0.
        count = count.clamp(min=1)
        th = rank[torch.arange(rank.size(0)), count - 1].unsqueeze(1)
        mask = (layer.prob.unsqueeze(2) * (imp > th).float().unsqueeze(1)).sum(0)
        layer.mask.copy_(mask.view(mask.size(0), mask.size(1), 1, 1))

    @torch.no_grad()
    def _find_mask_fp(self, layer: nn.Conv2d) -> None:
        """Whole-filter (output channel) pruning, applied on top of the group-channel mask."""
        importance = layer.weight.data ** 2
        imp = importance.sum(dim=(3, 2, 1))
        imp = imp / (imp.sum() + 1e-12)
        rank = imp.sort(dim=0)[0]
        csoi = rank.cumsum(dim=0)
        count = (csoi < self.beta).long().sum(dim=0).clamp(min=1)  # see _find_mask on the clamp
        th = rank[count - 1]
        mask = (imp > th).float().unsqueeze(1)
        layer.mask.mul_(mask.view(mask.size(0), mask.size(1), 1, 1))

    @torch.no_grad()
    def _apply_mask(self, layer: nn.Conv2d) -> None:
        layer.weight.mul_(layer.mask)

    @torch.no_grad()
    def _residual_bn_proc(self) -> None:
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.bias.mul_((m.weight.abs() > 0).float())

    def _cascade(self) -> None:
        """Remove structurally dead units by finding parameters with identically zero gradient.

        Masking input channels can leave a filter with no live input, or a downstream conv with no
        live source; such units are dead but still counted. The reference detects them by pushing
        random batches through and deleting anything whose gradient never becomes non-zero, which
        propagates structural removal into the depthwise convs and BatchNorm DSP cannot reach.
        """
        device = next(self.model.parameters()).device
        shape = [self.cascade_batch] + list(self.input_shape[1:])
        for _ in range(self.cascade_iters):
            out = self.model(torch.randn(*shape, device=device))
            target = torch.randint(0, out.shape[1], (out.shape[0],), device=device)
            F.cross_entropy(out.flatten(1) if out.dim() > 2 else out, target).backward()

        with torch.no_grad():
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    if m.weight.grad is None:
                        continue
                    live_out = (m.weight.grad.abs().sum(dim=(3, 2, 1), keepdim=True) > 0).float()
                    live_in = (m.weight.grad.abs().sum(dim=(3, 2, 0), keepdim=True) > 0).float()
                    m.weight.mul_(live_out).mul_(live_in)
                    if hasattr(m, "mask"):
                        m.mask.mul_(live_out).mul_(live_in)
                elif isinstance(m, nn.BatchNorm2d):
                    if m.weight.grad is None:
                        continue
                    live = (m.weight.grad.abs() > 0).float()
                    m.weight.mul_(live)
                    m.bias.mul_(live)
        self.model.zero_grad(True)

    def prune(self) -> tuple[float, float]:
        """Mask, cascade, and report ``(flops_pruned_pct, params_pruned_pct)``.

        Discretises the grouping first: idempotent and cheap, so it happens here rather than as an
        unstated precondition, forgetting which fails with a bare ``AttributeError``.
        """
        for layer in self.layers:
            self._set_arch_hard(layer)
        for layer in self.layers:
            self._find_mask(layer)
        for layer in self.fp_layers:
            self._find_mask_fp(layer)
        for layer in self.layers:
            self._apply_mask(layer)
        self._cascade()
        return self.summary()

    @torch.no_grad()
    def summary(self, verbose: bool = False) -> tuple[float, float]:
        remaining_flops = remaining_params = total_flops = total_params = 0.0
        for n, layer in enumerate(self.layers):
            kernels = (layer.weight.abs().sum(dim=(3, 2)) > 0).float()
            live_frac = kernels.sum().item() / kernels.numel()
            remaining_flops += layer.flops * live_frac
            remaining_params += layer.weight.numel() * live_frac
            total_flops += layer.flops
            total_params += layer.weight.numel()
            if verbose:
                remaining = torch.mm(layer.prob, kernels)
                r_ch = (remaining > 0).float().sum(dim=1)
                r_f = (remaining.sum(1) / (r_ch + 1e-8)).round()
                print(
                    f"[{n}] live {100 * live_frac:5.1f}%  structure "
                    f"{list(zip(r_f.long().tolist(), r_ch.long().tolist()))}"
                )
        return 100 * (1 - remaining_flops / total_flops), 100 * (1 - remaining_params / total_params)

    def bisect_beta(
        self,
        target: float,
        measure: Callable[[], float],
        n_iter: int = 14,
        verbose: bool = False,
    ) -> float:
        """Solve for the ``beta`` whose pruned model hits ``target`` under ``measure``.

        DEVIATION: the reference bisects on its own internal FLOPs percentage
        (``if pflops > rate*100``); this port bisects on a caller-supplied ``measure()``. Every
        curve here is matched on parameter count rather than FLOPs, and the percentages this class
        computes internally cover the DSP-prunable layers only (52,864 params on CP-Mobile) whereas
        IMP and SNIP report sparsity over all conv parameters (58,440). Bisecting against the
        project-wide measure is what lands DSP on exactly the same table rows as the other methods.

        ``measure`` must be increasing in ``beta`` and is called on a freshly pruned model each
        iteration; each probe costs only a mask computation plus the cascade passes, pruning being
        one-shot. Returns the solved ``beta`` and leaves ``self.beta`` set to it, with the weights
        restored to their unpruned state, so the caller must call :meth:`prune` once more to commit.
        """
        checkpoint = copy.deepcopy(self.model.state_dict())
        self.beta = 0.15
        lower, upper = 0.0, 1.0
        for _ in range(n_iter):
            self.prune()
            achieved = measure()
            self.model.load_state_dict(checkpoint)
            if verbose:
                print(f"  beta={self.beta:.6f} -> {achieved:.6f} (target {target:.6f})")
            if achieved > target:
                self.beta, upper = (self.beta + lower) / 2, self.beta
            else:
                self.beta, lower = (self.beta + upper) / 2, self.beta
        return self.beta

    def after_step(self) -> None:
        """Re-apply the mask. Must be called from ``on_train_batch_end``.

        Not from ``on_before_zero_grad``: in Lightning 2.x that hook lands between forward and
        backward, so masking there mutates tensors autograd has already saved.
        """
        for layer in self.layers:
            self._apply_mask(layer)
        self._residual_bn_proc()


# ---- Lightning integration -------------------------------------------------------------------


class GroupLearningPLModule(PrunedPLModule):
    """Phase A: trains the weights and the filter grouping jointly.

    The inherited mask machinery is inert here, group learning running on a fully dense network; the
    class is reused so the phase logs the same epoch metrics as every other run.
    """

    def __init__(self, config, masks, epoch_offset: int = 0) -> None:
        super().__init__(config, masks, epoch_offset)
        self.learner: GroupLearner | None = None
        self._dsp_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def _build_learner(self) -> None:
        """Build the learner on the first batch: it needs the optimiser Lightning actually
        configured (unwrapped from ``LightningOptimizer``) and a model already on its final device,
        neither of which exists at ``__init__`` time.
        """
        cfg = self.config
        layers = [conv for _, conv in select_dsp_layers(self.model)]
        steps_per_epoch = max(len(self.trainer.train_dataloader), 1)
        self.learner = GroupLearner(
            model=self.model,
            optimizer=self.optimizers().optimizer,
            layers=layers,
            reg=cfg.dsp_reg,
            total_steps=steps_per_epoch * cfg.dsp_group_epochs,
            n_groups=cfg.dsp_groups,
            tau=cfg.dsp_tau,
            input_shape=MEL_INPUT_SHAPE,
            group_lr=cfg.dsp_group_lr,
            penalty_scale=cfg.dsp_penalty_scale,
        )

    def training_step(self, train_batch, batch_idx):
        """Replicate ``PLModule.training_step`` so the augmented features can be cached.

        Deliberate duplication rather than ``super()``: the unrolled step needs the exact tensor the
        outer forward pass saw, after mel extraction and Frequency-MixStyle. Re-deriving it would
        redraw MixStyle's random mixing and unroll against a different input.
        """
        x, files, labels, devices, cities = train_batch
        x = self.mel_forward(x)
        if self.config.mixstyle_p > 0:
            x = mixstyle(x, self.config.mixstyle_p, self.config.mixstyle_alpha)
        self._dsp_cache = (x.detach(), labels)

        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, labels, reduction="none").mean()
        self.training_step_outputs.append(loss.detach().cpu())
        self.log("train_loss", loss, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        if self.learner is None:
            self._build_learner()
        self.learner.initialize()  # resample the relaxed grouping before the forward pass

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.learner is not None and self._dsp_cache is not None:
            x, y = self._dsp_cache
            self.learner.after_step(lambda m: F.cross_entropy(m(x), y))
        super().on_train_batch_end(outputs, batch, batch_idx)

    def epoch_metrics_extra(self) -> dict[str, float]:
        extra = super().epoch_metrics_extra()
        if self.learner is not None:
            # near zero means the grouping has not committed to anything yet
            extra["dsp/group_logit_std"] = self.learner.group_logit_std()
        return extra


class DSPFinetunePLModule(PrunedPLModule):
    """Phase C: finetunes the pruned model, pinning both the conv mask and the dead BN channels.

    :class:`PrunedPLModule` alone is not enough: its ``on_train_batch_end`` re-applies the conv
    weight mask, all IMP and SNIP need since neither touches BatchNorm. DSP does touch it, the
    cascade zeroing the BatchNorm affine parameters of structurally dead channels. Those zeros
    otherwise hold only through an unstated invariant spanning three mechanisms: a dead channel's
    downstream conv input is masked, so its BatchNorm gradient is exactly zero, and AdamW's
    decoupled decay leaves an exact zero alone. The reported parameter counts depend on it.
    """

    def __init__(self, config, masks, epoch_offset: int = 0) -> None:
        super().__init__(config, masks, epoch_offset)
        self._bn_mask: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def capture_bn_mask(self) -> int:
        """Snapshot which BatchNorm channels are currently dead. Call after pruning/loading."""
        self._bn_mask = {
            name: (mod.weight != 0).float()
            for name, mod in self.model.named_modules()
            if isinstance(mod, nn.BatchNorm2d)
        }
        return int(sum(int((m == 0).sum()) for m in self._bn_mask.values()))

    @torch.no_grad()
    def _apply_bn_mask(self) -> None:
        for name, mod in self.model.named_modules():
            if not (isinstance(mod, nn.BatchNorm2d) and name in self._bn_mask):
                continue
            mask = self._bn_mask[name]
            if mask.device != mod.weight.device:
                # captured while the model was still on CPU; migrate once, in place
                mask = mask.to(mod.weight.device)
                self._bn_mask[name] = mask
            mod.weight.mul_(mask)
            mod.bias.mul_(mask)

    def on_fit_start(self):
        super().on_fit_start()
        self._apply_bn_mask()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        self._apply_bn_mask()

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self._apply_bn_mask()
