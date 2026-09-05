"""The two pruning criteria, implemented as described in their papers.

Han et al. 2015 (arXiv:1506.02626) scores |w| on trained weights and prunes where
|w| <= q * std(W_layer), a layer-wise threshold whose "quality parameter" q is hand-tuned. Lee
et al. 2019 (arXiv:1810.02340), SNIP, scores |w * dL/dw| at a variance-scaling init and keeps the
global top-k. One deviation from Han: his q yields emergent sparsity rather than a target, which
cannot share an x-axis with SNIP's top-k, so it is bisected here to hit a requested sparsity.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from pruning.masking import MaskRegistry
from pruning.prunable import prunable_layers

logger = logging.getLogger(__name__)

Scores = dict[str, torch.Tensor]


@torch.no_grad()
def magnitude_scores(model: nn.Module) -> Scores:
    """Han's criterion: absolute weight value."""
    return {name: mod.weight.detach().abs() for name, mod in prunable_layers(model)}


def snip_scores(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Scores:
    """SNIP connection sensitivity |w * dL/dw|, from a single mini-batch.

    Equals |dL/dc_j| at c=1 for an auxiliary connectivity mask c (their eq. 5), so the chain rule
    supplies it without the auxiliary variables. The paper's normalization by the sum of all |g|
    (eq. 6) divides by one positive constant and cannot reorder the scores, so it is skipped. BN
    runs in train mode to normalize with batch statistics, since at init the running stats are
    still (0, 1) and say nothing about the data.
    """
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), y).backward()

    scores = {}
    for name, mod in prunable_layers(model):
        if mod.weight.grad is None:
            raise RuntimeError(f"no gradient reached {name}; cannot compute SNIP saliency")
        scores[name] = (mod.weight.detach() * mod.weight.grad.detach()).abs()

    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return scores


def global_thresholds(scores: Scores, sparsity: float) -> dict[str, float]:
    """SNIP: one threshold shared by every layer, cutting the global bottom `sparsity`."""
    flat = torch.cat([s.reshape(-1) for s in scores.values()])
    k = int(len(flat) * sparsity)
    if k < 1:
        return {name: -float("inf") for name in scores}  # prune nothing
    threshold = torch.kthvalue(flat.float(), k).values.item()
    return {name: threshold for name in scores}


@torch.no_grad()
def _layer_stds(model: nn.Module, masks: MaskRegistry | None) -> dict[str, float]:
    """std of each layer's surviving weights. On an already-pruned model the zeroed entries are
    removed connections, not weights; counting them would deflate the std and make an
    already-sparse layer prune progressively less each round."""
    stds = {}
    for name, mod in prunable_layers(model):
        w = mod.weight.detach()
        if masks is not None:
            live = w[masks.masks[name].to(w.device).bool()]
        else:
            live = w.reshape(-1)
        # 0 or 1 surviving weights have no meaningful spread
        stds[name] = live.std().item() if live.numel() > 1 else 0.0
    return stds


def han_thresholds(
    model: nn.Module,
    scores: Scores,
    sparsity: float,
    masks: MaskRegistry | None = None,
    iterations: int = 50,
    q_max: float = 20.0,
) -> tuple[float, dict[str, float]]:
    """Han's layer-wise q*std(W_layer) threshold, with q solved for a target global sparsity.
    Returns (q, {layer: threshold}). `sparsity` counts all prunable weights, including ones
    already pruned in a previous round, so on an iterative schedule it is a cumulative target."""
    stds = _layer_stds(model, masks)
    n_total = sum(s.numel() for s in scores.values())

    def sparsity_at(q: float) -> float:
        pruned = sum(int((scores[name] <= q * stds[name]).sum()) for name in scores)
        return pruned / n_total

    # sparsity_at is monotone non-decreasing in q, so bisection converges. `hi` is returned under
    # the invariant sparsity_at(lo) < target <= sparsity_at(hi), i.e. the smallest q known to meet
    # the target; the midpoint could land below it, leaving weights alive at a 100% target.
    lo, hi = 0.0, q_max
    if sparsity_at(hi) < sparsity:
        logger.warning(
            "q_max=%.1f only reaches %.1f%% sparsity, short of the %.1f%% target; "
            "the layer weight distributions are wider than expected",
            q_max, 100 * sparsity_at(hi), 100 * sparsity,
        )
        return hi, {name: hi * stds[name] for name in scores}

    for _ in range(iterations):
        mid = (lo + hi) / 2
        if sparsity_at(mid) < sparsity:
            lo = mid
        else:
            hi = mid
    return hi, {name: hi * stds[name] for name in scores}


def build_mask(
    model: nn.Module,
    scores: Scores,
    thresholds: dict[str, float],
    previous: MaskRegistry | None = None,
) -> MaskRegistry:
    """Mask from thresholds, intersected with any previous mask so that a pruned weight stays
    pruned across iterative rounds. Magnitude scores give that for free, since a zeroed weight
    never exceeds a positive threshold, but stating it makes resurrection impossible."""
    mask = MaskRegistry.from_thresholds(model, scores, thresholds)
    if previous is not None:
        for name in mask.masks:
            mask.masks[name] = mask.masks[name] * previous.masks[name].to(mask.masks[name].device)
    return mask
