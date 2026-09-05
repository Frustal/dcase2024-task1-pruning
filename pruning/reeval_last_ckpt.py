"""Re-score already-trained checkpoints on the last epoch instead of the best one.

    DATASET_PATH=data/tau_scenes_dataset uv run python -m pruning.reeval_last_ckpt \
        --config configs/scale_down_cm1.3.yaml --run_id jd152x6s

DCASE Task 1 has no validation split, so what the code calls "val" is the dev-test set and
selecting the epoch with the highest val_macro_acc selects on the test set. As in the official
baseline, the last epoch is reported instead. Only the dense runs are affected, since
run_pruning.py already tests the in-memory weights left by fit(); best.ckpt is scored too, so the
gap between the two conventions is measured rather than assumed.
"""
import csv
import logging
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from pruning.lightning import PrunedPLModule
from pruning.masking import MaskRegistry
from pruning.run_pruning import load_source_weights, macro_accuracy
from pruning.sparsity import count_params
from training.dataset_loader import load_dcase24
from training.run_training import (
    DEFAULT_DATASET_PATH,
    apply_config_file,
    build_parser,
    setup_logging,
)
from training.worker_init import worker_init_fn

logger = logging.getLogger(__name__)


def make_test_dataloader(config, dcase24):
    """Only the test set; re-scoring never touches the training data."""
    return DataLoader(dataset=dcase24.get_test_set(),
                      worker_init_fn=worker_init_fn, num_workers=config.num_workers,
                      persistent_workers=config.num_workers > 0, pin_memory=True,
                      batch_size=config.batch_size)


def find_checkpoints(run_id: str, pruned: bool) -> list[tuple[str, str]]:
    """Return (label, path) for every checkpoint to score, in curve order. A pruned run keeps one
    sparsity<NN>/ subdirectory per level, a dense run both checkpoints in its own root."""
    root = os.path.join("checkpoints", run_id)
    if not os.path.isdir(root):
        raise SystemExit(f"no such checkpoint directory: {root}")

    dirs = [("", root)]
    if pruned:
        levels = sorted(d for d in os.listdir(root) if d.startswith("sparsity"))
        if not levels:
            raise SystemExit(f"--pruned given but no sparsity*/ subdirectories under {root}")
        # 2-digit names, so lexicographic order is ascending sparsity order
        dirs = [(d, os.path.join(root, d)) for d in levels]

    found = []
    for level, directory in dirs:
        for which in ("last", "best"):
            path = os.path.join(directory, f"{which}.ckpt")
            if os.path.exists(path):
                found.append((f"{level}/{which}" if level else which, path))
            else:
                logger.warning("missing %s -- skipping", path)
    return found


def score(config, ckpt_path: str, test_dl) -> tuple[float, object]:
    """One test pass over a checkpoint's weights. Returns (macro accuracy, ParamCounts).

    A fresh module per checkpoint is deliberate: test_step calls model.half() and never converts
    back, so a reused module would score everything after the first as a half-precision model
    loaded with fp32 weights. The mask is rebuilt from the weights (w != 0) rather than
    deserialized; on a dense checkpoint that is an all-ones no-op, which lets both kinds of run
    share one code path.
    """
    pl_module = PrunedPLModule(config, MaskRegistry({}))
    load_source_weights(pl_module, ckpt_path)
    pl_module.masks = MaskRegistry.from_weights(pl_module.model)

    trainer = pl.Trainer(logger=False, accelerator="auto", devices=1,
                         precision=config.precision, enable_progress_bar=False)
    trainer.test(pl_module, dataloaders=test_dl, ckpt_path=None)

    pl_module.model.float()  # undo test_step's half(), so count_params reads clean values
    return macro_accuracy(pl_module), count_params(pl_module.model)


def add_reeval_args(parser):
    parser.add_argument('--pruned', action='store_true',
                        help="walk sparsity*/ subdirectories (an IMP or SNIP curve) instead of "
                             "scoring a single dense run's checkpoints")
    parser.add_argument('--out', type=str, default="reports/reeval_last_ckpt.csv",
                        help="CSV to append results to; created with a header if absent")
    return parser


if __name__ == '__main__':
    setup_logging()

    args = apply_config_file(add_reeval_args(build_parser())).parse_args()
    if args.run_id is None:
        raise SystemExit("--run_id is required: it names the checkpoints/<run_id>/ directory to "
                         "re-score (it is NOT used to open a wandb run here -- nothing is logged)")

    pl.seed_everything(args.seed, workers=True)
    dcase24 = load_dcase24(os.environ.get('DATASET_PATH', DEFAULT_DATASET_PATH))
    test_dl = make_test_dataloader(args, dcase24)

    checkpoints = find_checkpoints(args.run_id, args.pruned)
    logger.info("re-scoring %d checkpoints from run %s", len(checkpoints), args.run_id)

    rows = []
    for label, path in checkpoints:
        accuracy, counts = score(args, path, test_dl)
        rows.append({
            "run_id": args.run_id,
            "checkpoint": label,
            "sparsity": round(counts.sparsity, 6),
            "nonzero_params": counts.nonzero,
            "size_kb": round(counts.size_kb(), 2),
            "test_macro_accuracy": round(accuracy, 4),
        })
        logger.info("%s | %.2f%% sparsity | %d non-zero | %.2f KB fp16 | macro acc=%.4f",
                    label, 100 * counts.sparsity, counts.nonzero, counts.size_kb(), accuracy)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %d rows to %s", len(rows), args.out)

    print(f"\n{'checkpoint':<24}{'sparsity':>10}{'non-zero':>10}{'size KB':>10}{'macro acc':>11}")
    for row in rows:
        print(f"{row['checkpoint']:<24}{100 * row['sparsity']:>9.2f}%{row['nonzero_params']:>10}"
              f"{row['size_kb']:>10.2f}{row['test_macro_accuracy']:>11.4f}")
