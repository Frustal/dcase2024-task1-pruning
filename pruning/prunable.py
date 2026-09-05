"""Which parameters of a CP-Mobile model are subject to pruning.

Han (2015) and Lee (2019) both prune conv and fully-connected weights. In CP-Mobile every
learnable layer is an nn.Conv2d (the classifier head is a 1x1 conv) and none carries a bias, so
"prunable" means every conv weight tensor and nothing else, stem and classifier included, as in
both papers. BatchNorm affine params are never pruned; at 2,708 params they set a 5.29 KB fp16
floor that pruning cannot go below.
"""
import torch.nn as nn

PrunableLayers = list[tuple[str, nn.Conv2d]]


def prunable_layers(model: nn.Module) -> PrunableLayers:
    """All (name, module) pairs whose .weight is subject to pruning, in model order."""
    return [(name, mod) for name, mod in model.named_modules() if isinstance(mod, nn.Conv2d)]


def layer_kind(mod: nn.Conv2d) -> str:
    """depthwise / pointwise / 3x3, used only for reporting and per-layer breakdowns."""
    if mod.groups > 1:
        return "depthwise"
    return "pointwise" if mod.weight.shape[2] == 1 else "3x3"
