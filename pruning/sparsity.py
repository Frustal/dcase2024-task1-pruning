"""Size accounting for an unstructured, masked model.

Han and SNIP zero weights but leave tensor shapes intact, so nessi.get_torch_size() keeps
reporting the dense 119.43 KB at any sparsity. Size is therefore reported as non-zero parameters,
with BatchNorm's 2,708 unprunable params always counted in full: omitting them understates size
by 5.29 KB.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from pruning.prunable import prunable_layers

FP16_BYTES = 2


@dataclass(frozen=True)
class ParamCounts:
    total: int             # every parameter in the model, dense
    nonzero: int           # every parameter, counting only non-zeros in prunable weights
    conv_total: int        # prunable conv weights, dense
    conv_nonzero: int      # prunable conv weights, non-zero
    unprunable: int        # BatchNorm affine params etc., always counted in full

    @property
    def sparsity(self) -> float:
        """Fraction of prunable weights that are zero. The x-axis of the pruning curves."""
        return 1.0 - self.conv_nonzero / self.conv_total

    @property
    def overall_sparsity(self) -> float:
        """Fraction of all params that are zero; always lower, since BN cannot be pruned."""
        return 1.0 - self.nonzero / self.total

    def size_kb(self, bytes_per_param: int = FP16_BYTES) -> float:
        """Non-zero size in KB, fp16 by default: the challenge rule test_step enforces."""
        return self.nonzero * bytes_per_param / 1024

    def dense_size_kb(self, bytes_per_param: int = FP16_BYTES) -> float:
        return self.total * bytes_per_param / 1024


@torch.no_grad()
def count_params(model: nn.Module) -> ParamCounts:
    prunable = dict(prunable_layers(model))
    prunable_weights = {id(mod.weight) for mod in prunable.values()}

    conv_total = sum(mod.weight.numel() for mod in prunable.values())
    conv_nonzero = sum(int(mod.weight.count_nonzero()) for mod in prunable.values())
    # anything that is not a prunable conv weight stays dense and is counted in full
    unprunable = sum(p.numel() for p in model.parameters() if id(p) not in prunable_weights)

    return ParamCounts(
        total=conv_total + unprunable,
        nonzero=conv_nonzero + unprunable,
        conv_total=conv_total,
        conv_nonzero=conv_nonzero,
        unprunable=unprunable,
    )


@torch.no_grad()
def per_layer_sparsity(model: nn.Module) -> dict[str, float]:
    """Fraction of each prunable layer's weights that are zero. Logged every round: it exposes
    SNIP's depth-graded layer collapse against Han's near-uniform allocation."""
    return {
        name: 1.0 - int(mod.weight.count_nonzero()) / mod.weight.numel()
        for name, mod in prunable_layers(model)
    }
