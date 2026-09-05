import contextlib
import io
import logging
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import wandb

from baseline.helpers import nessi

logger = logging.getLogger(__name__)

# challenge rule: parameters are stored as fp16 for the size budget
BYTES_PER_PARAM = 2


def log_epoch_metrics(
    metrics: dict[str, float],
    epoch: int,
    extra: dict[str, float] | None = None,
) -> None:
    """`extra` is merged into the payload at the same wandb step -- used by pruning runs to
    attach the current sparsity to each epoch, since wandb.log pins an explicit step here and
    a second wandb.log call would land on the wrong one."""
    payload = {
        "epoch": epoch,
        "train/loss": metrics["train_loss"],
        "val/loss": metrics["val_loss"],
        "val/macro_accuracy": metrics["val_macro_accuracy"],
        "system/cpu_memory_mb": psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2),
    }
    if "epoch_duration_sec" in metrics:
        payload["time/epoch_duration_sec"] = metrics["epoch_duration_sec"]
    if torch.cuda.is_available():
        payload["system/gpu_memory_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
    if extra:
        payload.update(extra)
    wandb.log(payload, step=epoch)


def log_final_metrics(
    model: torch.nn.Module,
    test_accuracy_per_class: dict[str, float],
    input_size: tuple[int, ...] | None = None,
) -> None:
    macro_accuracy = sum(test_accuracy_per_class.values()) / len(test_accuracy_per_class)
    metrics = {"test/macro_accuracy": macro_accuracy}
    metrics.update({f"test/accuracy_{name}": acc for name, acc in test_accuracy_per_class.items()})

    param_count = sum(p.numel() for p in model.parameters())
    metrics["model/parameter_count"] = param_count

    if input_size is not None:
        # torchinfo.summary() (called inside get_torch_size) prints a results table
        # to stdout; wandb's console-capture wrapper mirrors stdout to the run log
        # using the Windows console codepage (cp1252), which can't encode some of
        # torchinfo's table characters and crashes the whole run. Only the returned
        # numbers are needed here, not the printed table, so redirect stdout away from
        # wandb's capture for this call.
        # torchinfo can also fail outright (shape mismatches, tracing errors) on
        # pruned or modified architectures, which must not kill an otherwise-complete
        # run before the model checkpoint and wandb.finish() get a chance to run.
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                macs, params = nessi.get_torch_size(model, input_size=input_size)
            size_kb = params * BYTES_PER_PARAM / 1024
            metrics["model/macs"] = macs
            metrics["model/size_kb"] = size_kb

            if macs > nessi.MAX_MACS or params > nessi.MAX_PARAMS_MEMORY:
                logger.warning(
                    "model EXCEEDS nessi budget: macs=%d (limit %d, %+d over) | "
                    "size=%.1f KB (limit %.1f KB, %+.1f KB over)",
                    macs, nessi.MAX_MACS, macs - nessi.MAX_MACS,
                    size_kb, nessi.MAX_PARAMS_MEMORY / 1024,
                    size_kb - nessi.MAX_PARAMS_MEMORY / 1024,
                )
            else:
                logger.info(
                    "model within nessi budget: macs=%d/%d | size=%.1f/%.1f KB",
                    macs, nessi.MAX_MACS, size_kb, nessi.MAX_PARAMS_MEMORY / 1024,
                )
        except Exception:
            logger.exception("nessi model complexity profiling failed, skipping")
            metrics["model/complexity_profiling_failed"] = True

    if torch.cuda.is_available():
        metrics["model/peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

    wandb.log(metrics)


def log_pruning_metrics(
    sparsity_target: float,
    sparsity_actual: float,
    pruning_iterations: int,
    accuracy_delta: float,
) -> None:
    wandb.log(
        {
            "pruning/sparsity_target": sparsity_target,
            "pruning/sparsity_actual": sparsity_actual,
            "pruning/iterations": pruning_iterations,
            "pruning/accuracy_delta": accuracy_delta,
        }
    )


def log_confusion_matrix(cm_array: np.ndarray, class_names: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm_array, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close(fig)


def log_model_checkpoint(model: torch.nn.Module) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "model.pt"
        torch.save(model, model_path)
        artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="model")
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)
