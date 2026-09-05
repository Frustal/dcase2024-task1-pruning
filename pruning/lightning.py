"""PLModule with a weight mask enforced throughout training.

Subclassing the training module keeps the mel pipeline, MixStyle, the validation aggregation and
the fp16 test_step identical to the dense runs, which is what makes the two comparable.
"""
from __future__ import annotations

import logging

from pruning.masking import MaskRegistry
from pruning.sparsity import count_params
from training.run_training import PLModule

logger = logging.getLogger(__name__)


class PrunedPLModule(PLModule):
    def __init__(self, config, masks: MaskRegistry, epoch_offset: int = 0):
        super().__init__(config)
        self.masks = masks
        self.epoch_offset = epoch_offset

    def on_fit_start(self):
        # first point at which the model is on its final device, and checkpoint weights, which
        # arrive dense, have been loaded
        self.masks.apply(self.model)
        counts = count_params(self.model)
        logger.info(
            "training with %.1f%% of conv weights pruned | %d non-zero params | %.2f KB fp16",
            100 * counts.sparsity, counts.nonzero, counts.size_kb(),
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Re-apply the mask after the optimizer has stepped. The load-bearing hook.

        dL/dw is generally non-zero even where w == 0, so the step moves a pruned weight straight
        back off zero and the model silently de-sparsifies while the mask-derived metric keeps
        reporting the target. Zeroing gradients is weaker: AdamW's decoupled weight decay and
        buffered momentum move the weight anyway. It has to be this hook, not on_before_zero_grad,
        which Lightning 2.x fires between forward and backward, where masking mutates tensors
        autograd has saved and the next backward raises.
        """
        self.masks.apply(self.model)

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        # test_step calls model.half(); re-assert so the accuracy reported is the sparse model's
        self.masks.apply(self.model)

    def epoch_metrics_extra(self) -> dict[str, float]:
        counts = count_params(self.model)
        return {
            "pruning/sparsity": counts.sparsity,
            "pruning/nonzero_params": counts.nonzero,
            "pruning/size_kb": counts.size_kb(),
        }
