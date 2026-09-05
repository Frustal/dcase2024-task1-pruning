"""Entrypoint for the pruning experiments.

    DATASET_PATH=data/tau_scenes_dataset uv run python -m pruning.run_pruning \
        --config configs/imp.yaml

Each method sweeps a list of target sparsities and logs one accuracy-vs-sparsity curve to a
single wandb run. They differ in where the sparsity comes from and what a curve costs:

  IMP (Han et al. 2015) starts from a trained checkpoint and alternates prune and retrain.
  Each target is one round and one curve point, so a whole curve is one job. Surviving weights
  carry forward between rounds and are never re-initialized.

  SNIP (Lee et al. 2019) prunes once at initialization from a single mini-batch, then trains
  the sparse network normally. Nothing is shared between levels, so each target needs its own
  full training run and an n-point curve costs n trainings.
"""
import copy
import logging
import os

import pytorch_lightning as pl
import torch
import wandb
from torch.utils.data import DataLoader

from experiments.logging_utils import log_confusion_matrix
from experiments.wandb_setup import setup_wandb
from pruning.importance import (
    build_mask,
    global_thresholds,
    han_thresholds,
    magnitude_scores,
    snip_scores,
)
from pruning.dsp import (
    MEL_INPUT_SHAPE,
    DSPPruner,
    DSPFinetunePLModule,
    GroupLearningPLModule,
    attach_dsp_buffers,
    deployable_nonzero_params,
    select_dsp_layers,
    select_fp_layers,
)
from pruning.lightning import PrunedPLModule
from pruning.masking import MaskRegistry
from pruning.sparsity import count_params, per_layer_sparsity
from pruning.state import PruningState, RoundResult, round_checkpoint_dir
from training.dataset_loader import load_dcase24
from training.waveform_cache import build_datasets
from training.run_training import (
    DEFAULT_DATASET_PATH,
    apply_config_file,
    build_parser,
    setup_logging,
)
from training.worker_init import worker_init_fn

logger = logging.getLogger(__name__)


def make_dataloaders(config, dcase24):
    train_ds, test_ds = build_datasets(config, dcase24)
    train_dl = DataLoader(dataset=train_ds,
                          worker_init_fn=worker_init_fn, num_workers=config.num_workers,
                          persistent_workers=config.num_workers > 0, pin_memory=True,
                          batch_size=config.batch_size, shuffle=True)
    test_dl = DataLoader(dataset=test_ds,
                         worker_init_fn=worker_init_fn, num_workers=config.num_workers,
                         persistent_workers=config.num_workers > 0, pin_memory=True,
                         batch_size=config.batch_size)
    return train_dl, test_dl


def load_source_weights(pl_module, ckpt_path):
    """Load a trained baseline's weights into the module's model (weights only, no optimizer)."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    model_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    pl_module.model.load_state_dict(model_state)
    logger.info("loaded source weights from %s", ckpt_path)


def macro_accuracy(pl_module) -> float:
    per_class = pl_module.last_test_per_class_acc
    return sum(per_class.values()) / len(per_class)


def make_trainer(config, max_epochs, checkpoint_dir):
    return pl.Trainer(
        max_epochs=max_epochs,
        logger=False,
        accelerator="auto",
        devices=1,
        precision=config.precision,
        callbacks=[
            pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, filename="best",
                                         monitor="val_macro_acc", mode="max", save_top_k=1),
            pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, filename="last"),
            pl.callbacks.TQDMProgressBar(refresh_rate=10),
        ],
    )


def restore_from(pl_module, state):
    """Reload the last completed round's weights and rebuild its mask, for --resume_run_id.

    The mask is derived from the weights rather than deserialized: pruned weights are exactly
    zero in the checkpoint, so it cannot fall out of sync with what it describes.
    """
    ckpt = os.path.join(state.last_checkpoint_dir(), "last.ckpt")
    load_source_weights(pl_module, ckpt)
    pl_module.masks = MaskRegistry.from_weights(pl_module.model)
    counts = count_params(pl_module.model)
    logger.info("resumed at %.2f%% sparsity (%d non-zero params) from %s",
                100 * counts.sparsity, counts.nonzero, ckpt)


def try_resume_round(pl_module, ckpt_path: str) -> bool:
    """Pick up a round/level that crashed mid-training, instead of redoing it from scratch.

    round_checkpoint_dir() is keyed by (run id, target), so if `ckpt_path` exists it can only
    be a "last.ckpt" this exact round already wrote before dying -- nothing else could have put
    it there. Loads its weights and rebuilds the mask with the same lossless (w != 0) technique
    restore_from() uses for IMP's round-to-round resume, so the caller can skip re-pruning /
    re-scoring entirely and pass ckpt_path straight to trainer.fit() for a full Lightning resume
    (optimizer state and epoch count, not just a weight reload).

    Returns False (nothing to resume, proceed as if this round is fresh) if it never started.
    """
    if not os.path.exists(ckpt_path):
        return False
    load_source_weights(pl_module, ckpt_path)
    pl_module.masks = MaskRegistry.from_weights(pl_module.model)
    counts = count_params(pl_module.model)
    logger.info(
        "found mid-round checkpoint %s -- resuming at %.2f%% sparsity (%d non-zero params)",
        ckpt_path, 100 * counts.sparsity, counts.nonzero,
    )
    return True


def evaluate_and_log(pl_module, trainer, test_dl, target, wandb_run, epoch):
    """Test the current (masked) model and record one point of the accuracy-vs-sparsity curve.

    ckpt_path=None so the in-memory weights are tested, NOT a reloaded 'best' checkpoint --
    Lightning's ModelCheckpoint saves a dense state_dict, and reloading it would restore the
    pruned weights to non-zero and silently evaluate a denser model than the one we claim.
    """
    trainer.test(pl_module, dataloaders=test_dl, ckpt_path=None)
    # test_step calls model.half() for the challenge's fp16 rule and never restores it. A plain
    # training run tests once at the end and never notices, but here another round of
    # trainer.fit() follows and fp16 weights against fp32 activations raise on the first conv.
    pl_module.model.float()

    counts = count_params(pl_module.model)
    accuracy = macro_accuracy(pl_module)

    logger.info(
        "sparsity target=%.0f%% actual=%.2f%% | %d non-zero params | %.2f KB fp16 | "
        "test macro acc=%.4f",
        100 * target, 100 * counts.sparsity, counts.nonzero, counts.size_kb(), accuracy,
    )

    wandb_run.log({
        "curve/sparsity_target": target,
        "curve/sparsity": counts.sparsity,
        "curve/nonzero_params": counts.nonzero,
        "curve/size_kb": counts.size_kb(),
        "curve/test_macro_accuracy": accuracy,
        **{f"layer_sparsity/{name}": s for name, s in per_layer_sparsity(pl_module.model).items()},
    }, step=epoch)
    return accuracy, counts


def run_imp(config, dcase24, wandb_run, state):
    """Han et al. 2015: train -> [prune -> retrain] x N, iteratively, on trained weights."""
    train_dl, test_dl = make_dataloaders(config, dcase24)

    # every IMP round is a retrain, so the module uses the retrain recipe from the start: a
    # reduced learning rate, and a short warmup because the dense recipe's 2000 steps would not
    # finish inside a short round
    retrain_config = copy.deepcopy(config)
    retrain_config.lr = config.retrain_lr
    retrain_config.warmup_steps = config.retrain_warmup_steps
    retrain_config.n_epochs = config.retrain_epochs

    pl_module = PrunedPLModule(retrain_config, MaskRegistry({}))
    epoch = state.epoch

    if state.completed:
        # resume: pick the weights (and therefore the mask) up from the last finished round,
        # rather than re-pruning the dense source from scratch
        restore_from(pl_module, state)
    else:
        load_source_weights(pl_module, config.source_ckpt)
        pl_module.masks = MaskRegistry.dense(pl_module.model)

        # sparsity=0 reference point: the source model through the same test path as every pruned
        # point, so the curve starts from a measured number rather than a quoted one
        dense_trainer = pl.Trainer(logger=False, accelerator="auto", devices=1,
                                   precision=retrain_config.precision)
        evaluate_and_log(pl_module, dense_trainer, test_dl, 0.0, wandb_run, epoch)

    for target in config.sparsities:
        if state.is_completed(target):
            logger.info("skipping already-completed sparsity %.4f", target)
            continue

        ckpt_dir = round_checkpoint_dir(wandb_run.id, target)
        resume_ckpt = os.path.join(ckpt_dir, "last.ckpt")
        q = None  # only computed on a fresh round; None on resume means "don't log curve/han_q"
        if try_resume_round(pl_module, resume_ckpt):
            logger.info("=== IMP round: target sparsity %.2f%% -- resumed mid-round ===",
                        100 * target)
        else:
            scores = magnitude_scores(pl_module.model)
            q, thresholds = han_thresholds(pl_module.model, scores, target, masks=pl_module.masks)
            pl_module.masks = build_mask(pl_module.model, scores, thresholds, previous=pl_module.masks)
            logger.info("=== IMP round: target sparsity %.2f%% (Han q=%.4f) ===", 100 * target, q)
            resume_ckpt = None

        pl_module.epoch_offset = epoch
        trainer = make_trainer(retrain_config, max_epochs=config.retrain_epochs,
                               checkpoint_dir=ckpt_dir)
        trainer.fit(pl_module, train_dl, test_dl, ckpt_path=resume_ckpt)
        epoch += config.retrain_epochs

        if q is not None:
            wandb_run.log({"curve/han_q": q}, step=epoch)
        accuracy, counts = evaluate_and_log(pl_module, trainer, test_dl, target, wandb_run, epoch)
        state.record(RoundResult(target=target, sparsity=counts.sparsity,
                                 nonzero_params=counts.nonzero, size_kb=counts.size_kb(),
                                 test_macro_accuracy=accuracy, epoch_end=epoch))

    log_confusion_matrix(pl_module.last_test_confusion_matrix, pl_module.label_ids)


def snip_batch(pl_module, train_dl, device):
    """One mini-batch of log-mel features, with augmentation OFF.

    SNIP specifies only "a mini-batch of training data". MixStyle and spec-augment would
    inject noise into the saliency, so mel_forward's augmentation path is bypassed by calling
    the mel transform directly (mel_forward applies mel_augment whenever self.training, which
    must stay True here so BatchNorm uses batch statistics -- at init the running statistics
    are untrained and carry no information about the data).
    """
    waveforms, _, labels, _, _ = next(iter(train_dl))
    with torch.no_grad():
        x = (pl_module.mel(waveforms.to(device)) + 1e-5).log()
    return x, labels.to(device)


def run_snip(config, dcase24, wandb_run, state):
    """Lee et al. 2019: prune once at init from one mini-batch, then train normally.

    Each sparsity level is fully independent (a fresh init, a fresh mask, a fresh training
    run) -- unlike IMP there is no weight state to carry FORWARD from one level to the next.
    Resuming a completed level therefore just means skipping it (state.is_completed). But a
    level that crashed mid-training (see try_resume_round) does have state worth carrying: its
    own checkpoint, picked back up rather than redone from scratch.
    """
    train_dl, test_dl = make_dataloaders(config, dcase24)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epoch = state.epoch

    for target in config.sparsities:
        if state.is_completed(target):
            logger.info("skipping already-completed sparsity %.4f", target)
            continue

        # identical random init at every sparsity level, so the mask is the only thing that varies
        # along the curve. Redundant when resuming, since the checkpoint overwrites it, but kept
        # unconditional to keep the two branches alike.
        pl.seed_everything(config.seed, workers=True)
        pl_module = PrunedPLModule(config, MaskRegistry({}))

        ckpt_dir = round_checkpoint_dir(wandb_run.id, target)
        resume_ckpt = os.path.join(ckpt_dir, "last.ckpt")
        if try_resume_round(pl_module, resume_ckpt):
            logger.info("=== SNIP: target sparsity %.2f%% -- resumed mid-level ===", 100 * target)
        else:
            # SNIP prunes ONCE at this fresh init, before any training -- unlike IMP there is no
            # previous round's mask to intersect with
            pl_module.masks = MaskRegistry.dense(pl_module.model)
            pl_module.to(device)
            x, y = snip_batch(pl_module, train_dl, device)
            scores = snip_scores(pl_module.model, x, y)
            pl_module.masks = build_mask(pl_module.model, scores,
                                         global_thresholds(scores, target))
            pl_module.cpu()  # let the Trainer own device placement from here
            resume_ckpt = None
            logger.info("=== SNIP: target sparsity %.2f%%, training %d epochs ===",
                        100 * target, config.n_epochs)

        pl_module.epoch_offset = epoch
        trainer = make_trainer(config, max_epochs=config.n_epochs, checkpoint_dir=ckpt_dir)
        trainer.fit(pl_module, train_dl, test_dl, ckpt_path=resume_ckpt)
        epoch += config.n_epochs

        accuracy, counts = evaluate_and_log(pl_module, trainer, test_dl, target, wandb_run, epoch)
        state.record(RoundResult(target=target, sparsity=counts.sparsity,
                                 nonzero_params=counts.nonzero, size_kb=counts.size_kb(),
                                 test_macro_accuracy=accuracy, epoch_end=epoch))

    log_confusion_matrix(pl_module.last_test_confusion_matrix, pl_module.label_ids)


def resolve_resume_state(resume_run_id: str | None) -> PruningState | None:
    """Load `pruning_state.json` for --resume_run_id, or None when there's nothing to load yet.

    "Nothing yet" covers both a fresh run (no --resume_run_id) and a run that died before its
    very first sparsity level finished -- PruningState.record() only writes the file after a
    level completes, so a crash inside level 1 leaves no file to load. Both cases are handled
    identically downstream: the caller builds a fresh empty PruningState. That's the correct
    fallback for the second case too (see run_pruning.py's __main__), just worth flagging loudly
    when it means real progress is being discarded -- i.e. checkpoints/<resume_run_id>/ exists
    locally (weights were written) but pruning_state.json doesn't, or the directory is missing
    entirely (redoing everything from scratch on what should have been a resume).
    """
    if not resume_run_id:
        return None
    state = PruningState.load(resume_run_id)
    if state is not None:
        return state
    ckpt_root = os.path.join("checkpoints", resume_run_id)
    if not os.path.isdir(ckpt_root):
        logger.warning(
            "--resume_run_id %s given but %s does not exist locally -- starting from an "
            "empty state, every sparsity level will be redone from scratch",
            resume_run_id, ckpt_root,
        )
    else:
        logger.warning(
            "--resume_run_id %s given but no %s/pruning_state.json (run died before its "
            "first sparsity level finished) -- starting from an empty state; any mid-round "
            "checkpoint under %s/sparsity*/ will still be picked up level by level",
            resume_run_id, ckpt_root, ckpt_root,
        )
    return None


def check_method_matches(state: PruningState | None, method: str, resume_run_id: str | None) -> None:
    """Refuse to resume a run under the wrong pruning method (e.g. an 'imp' run as 'snip').

    Only meaningful when `state` actually carries a recorded method -- an empty state (fresh
    run, or a run that died before its first level ever completed) has nothing to conflict with.
    """
    if state and state.method != method:
        raise SystemExit(f"run {resume_run_id} is a '{state.method}' run, "
                         f"cannot resume it as '{method}'")


def group_learning_dir(run_id):
    """Phase A lives outside the per-sparsity layout: one run shared by every target."""
    return os.path.join("checkpoints", run_id, "group_learning")


def run_dsp(config, dcase24, wandb_run, state):
    """Park et al. 2023: learn a filter grouping, prune group-channels once, then finetune.

    Structurally unlike the other two methods. IMP carries weights forward from level to level and
    SNIP restarts from a fresh init at each level; DSP does neither. It trains ONE group-learned
    model (phase A), and every target size forks from that same checkpoint (phase B/C). Pruning
    itself is one-shot and needs no training, so beta is bisected per target -- which is why the
    whole curve costs one long run plus five finetunes rather than five of everything.
    """
    train_dl, test_dl = make_dataloaders(config, dcase24)
    epoch = state.epoch

    # ---- phase A: differentiable group learning -------------------------------------------
    group_dir = group_learning_dir(wandb_run.id)
    group_ckpt = os.path.join(group_dir, "last.ckpt")
    group_done = os.path.join(group_dir, "phase_a_complete")

    if os.path.exists(group_done):
        logger.info("phase A already complete, reusing %s", group_ckpt)
    else:
        pl.seed_everything(config.seed, workers=True)
        gl_module = GroupLearningPLModule(config, MaskRegistry({}))
        resume_ckpt = os.path.join(group_dir, "last.ckpt")
        if try_resume_round(gl_module, resume_ckpt):
            logger.info("=== DSP phase A: group learning -- resumed mid-phase ===")
        else:
            load_source_weights(gl_module, config.source_ckpt)
            gl_module.masks = MaskRegistry.dense(gl_module.model)
            resume_ckpt = None
            logger.info("=== DSP phase A: group learning, %d epochs (groups=%d, lambda=%g) ===",
                        config.dsp_group_epochs, config.dsp_groups, config.dsp_reg)

        gl_module.epoch_offset = epoch
        trainer = make_trainer(config, max_epochs=config.dsp_group_epochs, checkpoint_dir=group_dir)
        trainer.fit(gl_module, train_dl, test_dl, ckpt_path=resume_ckpt)
        epoch += config.dsp_group_epochs
        # the group logits live on the conv modules, so they ride along in the module state_dict
        torch.save({"state_dict": gl_module.state_dict()}, group_ckpt)
        open(group_done, "w").close()
        logger.info("phase A done, group logit std %.4f", gl_module.learner.group_logit_std())

    # ---- phases B and C: one-shot prune + finetune, per target ------------------------------
    for target in config.sparsities:
        if state.is_completed(target):
            logger.info("skipping already-completed sparsity %.4f", target)
            continue

        ckpt_dir = round_checkpoint_dir(wandb_run.id, target)
        resume_ckpt = os.path.join(ckpt_dir, "last.ckpt")

        pl.seed_everything(config.seed, workers=True)
        pl_module = DSPFinetunePLModule(config, MaskRegistry({}))
        # load_state_dict is strict both ways and the two checkpoints differ: a mid-level checkpoint
        # carries `group` and `mask`, the phase-A one only `group`. Attach to match whichever is
        # about to be loaded. These are buffers, not parameters; see attach_dsp_buffers.
        resuming = os.path.exists(resume_ckpt)
        attach_dsp_buffers(pl_module.model, config.dsp_groups, with_mask=resuming)

        if resuming and try_resume_round(pl_module, resume_ckpt):
            logger.info("=== DSP: target sparsity %.2f%% -- resumed mid-level ===", 100 * target)
        else:
            # every target forks from the SAME phase-A checkpoint -- nothing carries over from the
            # previous target, unlike IMP
            load_source_weights(pl_module, group_ckpt)
            model = pl_module.model
            layers = [conv for _, conv in select_dsp_layers(model)]
            pruner = DSPPruner(
                model, layers, n_groups=config.dsp_groups, input_shape=MEL_INPUT_SHAPE,
                fp_layers=select_fp_layers(model, config.dsp_fp_mode),
            )
            beta = pruner.bisect_beta(target, measure=lambda: count_params(model).sparsity)
            pruner.prune()
            counts = count_params(model)
            logger.info("=== DSP: target %.2f%% -> beta %.4f, achieved %.2f%% "
                        "(%d params comparable / %d deployable) ===",
                        100 * target, beta, 100 * counts.sparsity,
                        counts.nonzero, deployable_nonzero_params(model))
            wandb_run.log({"curve/dsp_beta": beta,
                           "curve/dsp_deployable_params": deployable_nonzero_params(model)},
                          step=epoch)
            # the non-zero pattern after pruning IS the mask: it already folds in both the
            # group-channel mask and everything the dead-filter cascade removed
            pl_module.masks = MaskRegistry.from_weights(model)
            resume_ckpt = None
            logger.info("finetuning %d epochs", config.dsp_finetune_epochs)

        # snapshot the BatchNorm channels the cascade killed so finetuning cannot revive them;
        # works on both paths, since a resumed checkpoint already carries those zeros
        dead_bn = pl_module.capture_bn_mask()
        logger.info("pinning %d dead BatchNorm affine params through finetuning", dead_bn)

        pl_module.epoch_offset = epoch
        trainer = make_trainer(config, max_epochs=config.dsp_finetune_epochs,
                               checkpoint_dir=ckpt_dir)
        trainer.fit(pl_module, train_dl, test_dl, ckpt_path=resume_ckpt)
        epoch += config.dsp_finetune_epochs

        accuracy, counts = evaluate_and_log(pl_module, trainer, test_dl, target, wandb_run, epoch)
        # re-measure AFTER finetuning rather than trusting the count taken at prune time: this is
        # what catches a mask or BN pin that silently stopped holding during those 150 epochs
        deployable = deployable_nonzero_params(pl_module.model)
        logger.info("post-finetune: %d comparable / %d deployable non-zero params",
                    counts.nonzero, deployable)
        wandb_run.log({"curve/dsp_deployable_params_final": deployable}, step=epoch)
        state.record(RoundResult(target=target, sparsity=counts.sparsity,
                                 nonzero_params=counts.nonzero, size_kb=counts.size_kb(),
                                 test_macro_accuracy=accuracy, epoch_end=epoch))

    log_confusion_matrix(pl_module.last_test_confusion_matrix, pl_module.label_ids)


def add_pruning_args(parser):
    # not required=True: argparse would enforce it on the command line even when a --config file
    # supplies the value through set_defaults, making `method:` in a config useless. Validated
    # after parsing instead.
    parser.add_argument('--method', type=str, choices=['imp', 'snip', 'dsp'], default=None)
    # Cumulative targets, ascending. The first three land the pruned model on exactly the
    # parameter count of an existing dense reference, so the comparison is head-to-head at
    # identical size rather than interpolated:
    #   0.3362 -> 41,500 params / 81.05 KB  == cm=1.3
    #   0.4860 -> 32,748 params / 63.96 KB  == cm=1.0
    #   0.6333 -> 24,140 params / 47.15 KB  == cm=0.5
    # The last two go below every dense reference in the family (33.8 KB, 22.4 KB).
    parser.add_argument('--sparsities', type=float, nargs='+',
                        default=[0.3362, 0.4860, 0.6333, 0.75, 0.85])
    # IMP only: the trained model to prune. Defaults to the cm=1.8 baseline (epoch 149/150;
    # it predates the best-checkpoint callback, so last.ckpt is its only trained snapshot).
    parser.add_argument('--source_ckpt', type=str,
                        default="checkpoints/762f0b4fde744636b143bf56646046b7/last.ckpt")
    parser.add_argument('--retrain_epochs', type=int, default=20)
    # Han retrains at a fraction of the original LR (1/10 LeNet, 1/100 AlexNet) to avoid
    # destroying the co-adapted features that survived pruning
    parser.add_argument('--retrain_lr', type=float, default=0.0005)
    # the dense recipe's 2000-step warmup is ~20 epochs at ~100 steps/epoch, so it would
    # never finish inside a retrain round
    parser.add_argument('--retrain_warmup_steps', type=int, default=100)

    # DSP only. Phase A (group learning) is one long run shared by every target; phase C
    # (finetune) is per target. beta is not a flag -- it is bisected per target at prune time.
    parser.add_argument('--dsp_groups', type=int, default=2)
    parser.add_argument('--dsp_reg', type=float, default=2e-3)      # lambda
    parser.add_argument('--dsp_tau', type=float, default=0.5)       # Gumbel-Softmax temperature
    parser.add_argument('--dsp_group_lr', type=float, default=1e-3)
    parser.add_argument('--dsp_group_epochs', type=int, default=60)
    parser.add_argument('--dsp_finetune_epochs', type=int, default=150)
    parser.add_argument('--dsp_finetune_lr', type=float, default=0.0005)
    parser.add_argument('--dsp_finetune_warmup_steps', type=int, default=100)
    parser.add_argument('--dsp_fp_mode', type=str, choices=['free', 'all', 'none'],
                        default='free')
    parser.add_argument('--dsp_penalty_scale', type=str, choices=['params', 'flops'],
                        default='params')
    return parser


if __name__ == '__main__':
    setup_logging()

    parser = apply_config_file(add_pruning_args(build_parser()))
    args = parser.parse_args()
    if args.method is None:
        raise SystemExit("--method must be given, either on the CLI or in the --config YAML")
    if not all(a < b for a, b in zip(args.sparsities, args.sparsities[1:])):
        raise SystemExit("--sparsities must be strictly ascending: IMP prunes cumulatively, "
                         "so a later target below an earlier one can never be reached")
    if args.run_name is None:
        args.run_name = f"{args.method}-cm{args.channels_multiplier}-{args.subset}pct"

    pl.seed_everything(args.seed, workers=True)
    dcase24 = load_dcase24(os.environ.get('DATASET_PATH', DEFAULT_DATASET_PATH))

    # --resume_run_id continues a crashed run: same wandb run, same checkpoint dir, completed
    # levels skipped. resume="must" so a mistyped id fails loudly instead of silently starting a
    # fresh run. A missing pruning_state.json is not a typo, it means the run died before its
    # first level completed.
    state = resolve_resume_state(args.resume_run_id)
    check_method_matches(state, args.method, args.resume_run_id)

    run = setup_wandb(
        project_name=args.wandb_project, run_name=args.run_name, config=vars(args),
        run_id=args.resume_run_id or args.run_id,
        resume="must" if args.resume_run_id else ("allow" if args.run_id else None),
    )
    wandb.define_metric("curve/*", step_metric="epoch")
    wandb.define_metric("pruning/*", step_metric="epoch")
    wandb.define_metric("layer_sparsity/*", step_metric="epoch")

    if state is None:
        state = PruningState(method=args.method, run_id=run.id)
        # write pruning_state.json now, not after the first level completes -- otherwise a
        # crash inside level 1 leaves nothing on disk for a future --resume_run_id to find
        state.save()
        logger.info("started wandb run %s (method=%s, sparsities=%s)",
                    run.id, args.method, args.sparsities)
    else:
        logger.info("resuming wandb run %s (method=%s) -- %d/%d levels already done: %s",
                    run.id, args.method, len(state.completed), len(args.sparsities),
                    sorted(state.completed_targets))

    try:
        if args.method == 'imp':
            run_imp(args, dcase24, run, state)
        elif args.method == 'dsp':
            run_dsp(args, dcase24, run, state)
        else:
            run_snip(args, dcase24, run, state)
    finally:
        wandb.finish()
