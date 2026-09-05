"""Binary weight masks and their enforcement during training.

What matters here is when the mask is re-applied. AdamW carries momentum and applies decoupled
weight decay, so zeroing a weight once is not enough: it drifts off zero on the very next step
while the mask-derived sparsity metric still reports the target. PrunedPLModule therefore
re-applies the mask after every optimizer step, in on_train_batch_end.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from pruning.prunable import prunable_layers


class MaskRegistry:
    """Holds one 0/1 mask per prunable weight tensor, and enforces it on a model."""

    def __init__(self, masks: dict[str, torch.Tensor] | None = None):
        self.masks: dict[str, torch.Tensor] = masks or {}

    @classmethod
    def dense(cls, model: nn.Module) -> "MaskRegistry":
        """An all-ones registry: nothing pruned, the identity element."""
        return cls({name: torch.ones_like(mod.weight) for name, mod in prunable_layers(model)})

    @classmethod
    def from_weights(cls, model: nn.Module) -> "MaskRegistry":
        """Reconstruct the mask from an already-pruned model: dead weights are exactly zero. Used
        on crash-resume, so the mask is never serialized beside the checkpoint and cannot fall out
        of sync with it; lossless because pruning writes exact 0.0."""
        return cls({name: (mod.weight.detach() != 0).to(mod.weight.dtype)
                    for name, mod in prunable_layers(model)})

    @classmethod
    def from_thresholds(
        cls,
        model: nn.Module,
        scores: dict[str, torch.Tensor],
        thresholds: dict[str, float],
    ) -> "MaskRegistry":
        """Keep weight j iff score_j > threshold(layer). Both papers' rules reduce to this;
        only the score and the threshold differ (see importance.py)."""
        return cls({
            name: (scores[name] > thresholds[name]).to(mod.weight.dtype)
            for name, mod in prunable_layers(model)
        })

    @torch.no_grad()
    def apply(self, model: nn.Module) -> None:
        """Zero every pruned weight. Call after every optimizer.step()."""
        for name, mod in prunable_layers(model):
            mask = self.masks[name]
            if mask.device != mod.weight.device:
                # first call after Lightning moves the model to its device
                mask = mask.to(mod.weight.device)
                self.masks[name] = mask
            # test_step calls model.half(), and an in-place mul_ of a half tensor by a float32
            # mask raises. Cast here so the stored mask keeps full precision.
            mod.weight.mul_(mask.to(mod.weight.dtype))

    def kept_fraction(self) -> dict[str, float]:
        return {name: mask.mean().item() for name, mask in self.masks.items()}

    def overall_kept(self) -> float:
        kept = sum(int(m.sum()) for m in self.masks.values())
        total = sum(m.numel() for m in self.masks.values())
        return kept / total

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: mask.cpu() for name, mask in self.masks.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.masks = {name: mask.clone() for name, mask in state.items()}
