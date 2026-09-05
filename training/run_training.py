"""Train and evaluate the DCASE 2024 Task 1 acoustic scene classification baseline.

Wraps the CP-Mobile baseline in a Lightning module that turns waveforms into log mel
spectrograms, trains on the TAU Urban Acoustic Scenes development split, and reports
macro-average accuracy on the development-test split.
"""
import argparse
import json
import logging
import os
import time

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio
import transformers
import wandb
import yaml
from torch.utils.data import DataLoader

from baseline.helpers import nessi
from baseline.helpers.mixstyle import mixstyle
from training.worker_init import worker_init_fn
from baseline.models.baseline import get_model
from experiments.logging_utils import (
    log_confusion_matrix,
    log_epoch_metrics,
    log_final_metrics,
    log_model_checkpoint,
)
from experiments.wandb_setup import setup_wandb
from training.dataset_loader import load_dcase24
from training.waveform_cache import DEFAULT_CACHE_DIR, build_datasets

DEFAULT_DATASET_PATH = "data/tau_scenes_dataset"

logger = logging.getLogger(__name__)


class PLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        resample = torchaudio.transforms.Resample(
            orig_freq=self.config.orig_sample_rate,
            new_freq=self.config.sample_rate
        )

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.window_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max
        )

        freqm = torchaudio.transforms.FrequencyMasking(config.freqm, iid_masks=True)
        timem = torchaudio.transforms.TimeMasking(config.timem, iid_masks=True)

        self.mel = torch.nn.Sequential(
            resample,
            mel
        )

        self.mel_augment = torch.nn.Sequential(
            freqm,
            timem
        )

        self.model = get_model(n_classes=config.n_classes,
                               in_channels=config.in_channels,
                               base_channels=config.base_channels,
                               channels_multiplier=config.channels_multiplier,
                               expansion_rate=config.expansion_rate
                               )

        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.label_ids = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
                          'street_pedestrian', 'street_traffic', 'tram']
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}

        # PL 2 dropped the epoch-end output arguments, so steps accumulate here
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.last_train_loss = None
        self.last_val_loss = None
        self.last_val_macro_acc = None
        self.last_test_per_class_acc = None
        self.last_test_confusion_matrix = None
        self.test_confusion_matrix = None
        self._epoch_start_time = None

        # wandb.log pins step=epoch, which must increase monotonically across a run. A pruning
        # run drives several trainer.fit() calls inside one wandb run and each fit restarts
        # current_epoch at 0, so subclasses bump this to keep the epoch axis continuous.
        self.epoch_offset = 0

    def epoch_metrics_extra(self) -> dict[str, float]:
        """Extra key/values for each epoch's wandb payload; pruning subclasses add sparsity."""
        return {}

    def mel_forward(self, x):
        """Log mel spectrogram of a batch of waveforms, augmented while training."""
        x = self.mel(x)
        if self.training:
            x = self.mel_augment(x)
        x = (x + 1e-5).log()
        return x

    def forward(self, x):
        """Class logits for a batch of raw waveforms."""
        x = self.mel_forward(x)
        x = self.model(x)
        return x

    def configure_optimizers(self):
        """AdamW with a warmup + cosine schedule, stepped once per batch."""

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]

    def training_step(self, train_batch, batch_idx):
        x, files, labels, devices, cities = train_batch
        x = self.mel_forward(x)

        if self.config.mixstyle_p > 0:
            # frequency mixstyle
            x = mixstyle(x, self.config.mixstyle_p, self.config.mixstyle_alpha)
        y_hat = self.model(x)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        loss = samples_loss.mean()

        self.training_step_outputs.append(loss.detach().cpu())
        self.log("train_loss", loss, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        self.last_train_loss = torch.stack(self.training_step_outputs).mean().item()
        self.training_step_outputs.clear()

        if self.last_val_loss is None:
            # validation hasn't run for this epoch yet (e.g. check_val_every_n_epoch > 1)
            return

        epoch_duration_sec = time.time() - self._epoch_start_time
        log_epoch_metrics(
            {
                "train_loss": self.last_train_loss,
                "val_loss": self.last_val_loss,
                "val_macro_accuracy": self.last_val_macro_acc,
                "epoch_duration_sec": epoch_duration_sec,
            },
            epoch=self.current_epoch + self.epoch_offset,
            extra=self.epoch_metrics_extra(),
        )
        logger.info(
            "epoch %d/%d - train_loss=%.4f val_loss=%.4f val_macro_acc=%.4f (%.1fs)",
            self.current_epoch + 1, self.trainer.max_epochs,
            self.last_train_loss, self.last_val_loss, self.last_val_macro_acc, epoch_duration_sec,
        )

    def validation_step(self, val_batch, batch_idx):
        x, files, labels, devices, cities = val_batch

        y_hat = self.forward(x)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {'loss': samples_loss.mean(), "n_correct": n_correct,
                   "n_pred": torch.as_tensor(len(labels), device=self.device)}

        for d in self.device_ids:
            results["devloss." + d] = torch.as_tensor(0., device=self.device)
            results["devcnt." + d] = torch.as_tensor(0., device=self.device)
            results["devn_correct." + d] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(dev_names):
            results["devloss." + d] = results["devloss." + d] + samples_loss[i]
            results["devn_correct." + d] = results["devn_correct." + d] + n_correct_per_sample[i]
            results["devcnt." + d] = results["devcnt." + d] + 1

        for l in self.label_ids:
            results["lblloss." + l] = torch.as_tensor(0., device=self.device)
            results["lblcnt." + l] = torch.as_tensor(0., device=self.device)
            results["lbln_correct." + l] = torch.as_tensor(0., device=self.device)
        for i, l in enumerate(labels):
            results["lblloss." + self.label_ids[l]] = results["lblloss." + self.label_ids[l]] + samples_loss[i]
            results["lbln_correct." + self.label_ids[l]] = \
                results["lbln_correct." + self.label_ids[l]] + n_correct_per_sample[i]
            results["lblcnt." + self.label_ids[l]] = results["lblcnt." + self.label_ids[l]] + 1
        results = {k: v.cpu() for k, v in results.items()}
        self.validation_step_outputs.append(results)

    def on_validation_epoch_end(self):
        logs = self._aggregate_epoch_outputs(self.validation_step_outputs)
        self.validation_step_outputs.clear()

        if self.trainer.sanity_checking:
            # a couple of batches only, so this is not a real epoch and per-class accuracy is
            # often nan when a class does not appear
            return

        self.last_val_loss = logs["loss"].item()
        self.last_val_macro_acc = logs["macro_avg_acc"].item()
        self.log_dict({"val_loss": logs["loss"], "val_macro_acc": logs["macro_avg_acc"]},
                      prog_bar=True, logger=False)

    def on_test_epoch_start(self):
        self.test_confusion_matrix = torch.zeros(
            len(self.label_ids), len(self.label_ids), dtype=torch.long
        )

    def test_step(self, test_batch, batch_idx):
        x, files, labels, devices, cities = test_batch

        # challenge rule: 128 KB of parameter memory. At 61,148 parameters that allows 16-bit
        # precision (~122 kB), so reported scores must come from an fp16 forward pass.
        self.model.half()
        x = self.mel_forward(x)
        x = x.half()
        y_hat = self.model(x)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()
        self.log("test_acc", n_correct.float() / len(labels), prog_bar=True, logger=False,
                 on_step=True, on_epoch=False)

        num_classes = len(self.label_ids)
        flat_index = labels.cpu() * num_classes + preds.cpu()
        self.test_confusion_matrix += torch.bincount(
            flat_index, minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)

        dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {'loss': samples_loss.mean(), "n_correct": n_correct,
                   "n_pred": torch.as_tensor(len(labels), device=self.device)}

        for d in self.device_ids:
            results["devloss." + d] = torch.as_tensor(0., device=self.device)
            results["devcnt." + d] = torch.as_tensor(0., device=self.device)
            results["devn_correct." + d] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(dev_names):
            results["devloss." + d] = results["devloss." + d] + samples_loss[i]
            results["devn_correct." + d] = results["devn_correct." + d] + n_correct_per_sample[i]
            results["devcnt." + d] = results["devcnt." + d] + 1

        for l in self.label_ids:
            results["lblloss." + l] = torch.as_tensor(0., device=self.device)
            results["lblcnt." + l] = torch.as_tensor(0., device=self.device)
            results["lbln_correct." + l] = torch.as_tensor(0., device=self.device)
        for i, l in enumerate(labels):
            results["lblloss." + self.label_ids[l]] = results["lblloss." + self.label_ids[l]] + samples_loss[i]
            results["lbln_correct." + self.label_ids[l]] = \
                results["lbln_correct." + self.label_ids[l]] + n_correct_per_sample[i]
            results["lblcnt." + self.label_ids[l]] = results["lblcnt." + self.label_ids[l]] + 1
        self.test_step_outputs.append(results)

    def on_test_epoch_end(self):
        logs = self._aggregate_epoch_outputs(self.test_step_outputs)
        self.test_step_outputs.clear()
        self.last_test_per_class_acc = {l: logs["acc." + l].item() for l in self.label_ids}
        self.last_test_confusion_matrix = self.test_confusion_matrix.numpy()
        logger.info("test macro accuracy: %.4f", logs["macro_avg_acc"].item())

    def predict_step(self, eval_batch, batch_idx, dataloader_idx=0):
        x, files = eval_batch

        # fp16 at inference, as in test_step
        self.model.half()

        x = self.mel_forward(x)
        x = x.half()
        y_hat = self.model(x)

        return files, y_hat

    def _aggregate_epoch_outputs(self, step_outputs):
        """Reduce per-step outputs to per-device, per-class and macro-average metrics."""
        outputs = {k: [] for k in step_outputs[0]}
        for step_output in step_outputs:
            for k in step_output:
                outputs[k].append(step_output[k])
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs['loss'].mean()
        acc = sum(outputs['n_correct']) * 1.0 / sum(outputs['n_pred'])

        logs = {'acc': acc, 'loss': avg_loss}

        for d in self.device_ids:
            dev_loss = outputs["devloss." + d].sum()
            dev_cnt = outputs["devcnt." + d].sum()
            dev_corrct = outputs["devn_correct." + d].sum()
            logs["loss." + d] = dev_loss / dev_cnt
            logs["acc." + d] = dev_corrct / dev_cnt
            logs["cnt." + d] = dev_cnt
            logs["acc." + self.device_groups[d]] = logs.get("acc." + self.device_groups[d], 0.) + dev_corrct
            logs["count." + self.device_groups[d]] = logs.get("count." + self.device_groups[d], 0.) + dev_cnt
            logs["lloss." + self.device_groups[d]] = logs.get("lloss." + self.device_groups[d], 0.) + dev_loss

        for d in set(self.device_groups.values()):
            logs["acc." + d] = logs["acc." + d] / logs["count." + d]
            logs["lloss." + d] = logs["lloss." + d] / logs["count." + d]

        for l in self.label_ids:
            lbl_loss = outputs["lblloss." + l].sum()
            lbl_cnt = outputs["lblcnt." + l].sum()
            lbl_corrct = outputs["lbln_correct." + l].sum()
            logs["loss." + l] = lbl_loss / lbl_cnt
            logs["acc." + l] = lbl_corrct / lbl_cnt
            logs["cnt." + l] = lbl_cnt

        logs["macro_avg_acc"] = torch.mean(torch.stack([logs["acc." + l] for l in self.label_ids]))
        return logs


def train(config, dcase24):
    """Train a model for the configured number of epochs and log results to wandb."""
    run_id = config.resume_run_id or config.run_id
    if config.resume_run_id:
        resume = "must"  # a typo should fail loudly rather than start a fresh run
    elif config.run_id:
        resume = "allow"  # pre-assigned id; resumes only if it already exists
    else:
        resume = None
    run = setup_wandb(
        project_name=config.wandb_project,
        run_name=config.run_name,
        config=vars(config),
        run_id=run_id,
        resume=resume,
    )
    logger.info("started wandb run %s (project=%s, run_name=%s)",
               run.id, config.wandb_project, config.run_name)

    # resuming reuses run.id, so this points at the crashed run's own checkpoint dir
    resume_ckpt_path = os.path.join("checkpoints", run.id, "last.ckpt")
    if config.resume_run_id and os.path.exists(resume_ckpt_path):
        logger.info("resuming training from %s", resume_ckpt_path)
    else:
        if config.resume_run_id:
            logger.warning(
                "resume_run_id=%s given but no checkpoint found at %s -- starting fresh in that run",
                config.resume_run_id, resume_ckpt_path,
            )
        resume_ckpt_path = None

    assert config.subset in {100, 50, 25, 10, 5}, "Specify an integer value in: {100, 50, 25, 10, 5} to use one of " \
                                                  "the given subsets."
    train_ds, test_ds = build_datasets(config, dcase24)
    train_dl = DataLoader(dataset=train_ds,
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          persistent_workers=config.num_workers > 0,
                          pin_memory=True,
                          batch_size=config.batch_size,
                          shuffle=True)

    test_dl = DataLoader(dataset=test_ds,
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         persistent_workers=config.num_workers > 0,
                         pin_memory=True,
                         batch_size=config.batch_size)

    logger.info("train set: %d samples (subset=%d%%) | test set: %d samples",
               len(train_dl.dataset), config.subset, len(test_dl.dataset))

    pl_module = PLModule(config)

    checkpoint_dir = os.path.join("checkpoints", run.id)
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         logger=False,
                         accelerator='auto',
                         devices=1,
                         precision=config.precision,
                         callbacks=[
                             # best epoch by val_macro_acc. Diagnostic only, since nothing is
                             # reported off it (see trainer.test below); the best-vs-last gap
                             # shows how much test-set peak-picking would inflate a score.
                             pl.callbacks.ModelCheckpoint(
                                 dirpath=checkpoint_dir,
                                 filename="best",
                                 monitor="val_macro_acc",
                                 mode="max",
                                 save_top_k=1,
                             ),
                             # separate unmonitored callback so last.ckpt is overwritten every
                             # epoch: save_last=True on a monitored callback (PL 2.6) rewrites
                             # "last" only alongside a new top-k save, so once val_macro_acc
                             # plateaus a resume would rewind to the best epoch, not the crash.
                             pl.callbacks.ModelCheckpoint(
                                 dirpath=checkpoint_dir,
                                 filename="last",
                             ),
                             pl.callbacks.TQDMProgressBar(refresh_rate=10),
                         ])
    trainer.fit(pl_module, train_dl, test_dl, ckpt_path=resume_ckpt_path)

    # Test the LAST epoch, not the best one: DCASE Task 1 has no validation split, so what this
    # code calls "val" is the development-test set and picking the highest val_macro_acc epoch
    # selects on the test set. pruning/run_pruning.py evaluates the same way, keeping dense and
    # pruned scores comparable. ckpt_path=None tests the in-memory weights, which after fit() are
    # the last epoch's; ckpt_path="last" raises, as the unmonitored callback above sets
    # best_model_path rather than last_model_path.
    logger.info("training complete, running final test on last epoch's weights...")
    trainer.test(pl_module, dataloaders=test_dl, ckpt_path=None)

    log_confusion_matrix(pl_module.last_test_confusion_matrix, pl_module.label_ids)

    # test_step left the model in fp16 and torchinfo needs a matching-dtype input, so profile
    # complexity with the model temporarily back in fp32
    logger.info("profiling model complexity via nessi...")
    pl_module.model.float()
    sample = next(iter(test_dl))[0][0].unsqueeze(0)
    input_size = pl_module.mel_forward(sample).size()
    log_final_metrics(pl_module.model, pl_module.last_test_per_class_acc, input_size=input_size)
    pl_module.model.half()

    log_model_checkpoint(pl_module.model)
    logger.info("logged final metrics, confusion matrix, and model checkpoint to run %s", run.id)
    wandb.finish()


def evaluate(config, dcase24):
    """Score a checkpoint on the development-test split and write evaluation-set predictions."""
    from sklearn import preprocessing
    import pandas as pd

    assert config.ckpt_id is not None, "A value for argument 'ckpt_id' must be provided."
    ckpt_dir = os.path.join("checkpoints", config.ckpt_id)
    assert os.path.exists(ckpt_dir), f"No such folder: {ckpt_dir}"
    ckpt_file = os.path.join(ckpt_dir, "last.ckpt")
    assert os.path.exists(ckpt_file), f"No such file: {ckpt_file}. Implement your own mechanism to select" \
                                      f"the desired checkpoint."

    os.makedirs("predictions", exist_ok=True)
    out_dir = os.path.join("predictions", config.ckpt_id)
    os.makedirs(out_dir, exist_ok=True)

    pl_module = PLModule.load_from_checkpoint(ckpt_file, config=config)
    trainer = pl.Trainer(logger=False,
                         accelerator='auto',
                         devices=1,
                         precision=config.precision)

    test_dl = DataLoader(dataset=dcase24.get_test_set(),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         persistent_workers=config.num_workers > 0,
                         pin_memory=True,
                         batch_size=config.batch_size)

    sample = next(iter(test_dl))[0][0].unsqueeze(0).to(pl_module.device)
    shape = pl_module.mel_forward(sample).size()
    macs, params = nessi.get_torch_size(pl_module.model, input_size=shape)

    logger.info("model complexity: MACs=%d, params=%d", macs, params)
    assert macs <= nessi.MAX_MACS, "The model exceeds the MACs limit and must not be submitted to the challenge!"
    assert params <= nessi.MAX_PARAMS_MEMORY, \
        "The model exceeds the parameter limit and must not be submitted to the challenge!"

    allowed_precision = int(nessi.MAX_PARAMS_MEMORY / params * 8)
    logger.info("max allowed parameter precision given the %d-byte budget: %d bit",
               nessi.MAX_PARAMS_MEMORY, allowed_precision)

    # model details for the technical report
    info = {}
    info['MACs'] = macs
    info['Params'] = params
    res = trainer.test(pl_module, test_dl)
    info['test'] = res

    # generate predictions on evaluation set
    eval_dl = DataLoader(dataset=dcase24.get_eval_set(),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         pin_memory=True,
                         batch_size=config.batch_size)

    predictions = trainer.predict(pl_module, dataloaders=eval_dl)
    all_files = [item[len("audio/"):] for files, _ in predictions for item in files]
    all_predictions = torch.cat([torch.as_tensor(p) for _, p in predictions], 0)
    all_predictions = F.softmax(all_predictions, dim=1)

    # recover the class order the baseline's LabelEncoder produces from the meta csv
    df = pd.read_csv(dcase24.dataset_config['meta_csv'], sep="\t")
    le = preprocessing.LabelEncoder()
    le.fit_transform(df[['scene_label']].values.reshape(-1))
    class_names = le.classes_
    df = {'filename': all_files}
    scene_labels = [class_names[i] for i in torch.argmax(all_predictions, dim=1)]
    df['scene_label'] = scene_labels
    for i, label in enumerate(class_names):
        df[label] = all_predictions[:, i]
    df = pd.DataFrame(df)

    df.to_csv(os.path.join(out_dir, 'output.csv'), sep='\t', index=False)
    torch.save(pl_module.model.state_dict(), os.path.join(out_dir, "model_state_dict.pt"))
    with open(os.path.join(out_dir, "info.json"), "w") as json_file:
        json.dump(info, json_file)
    logger.info("predictions, model state dict, and info.json written to %s", out_dir)


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S", force=True)
    # that logger also carries useful messages (e.g. "GPU available"), so drop only the ad
    logging.getLogger("pytorch_lightning.utilities.rank_zero").addFilter(
        lambda record: "litlogger" not in record.getMessage().lower()
    )

    if torch.cuda.is_available():
        # trade a little precision for throughput on Tensor Core GPUs
        torch.set_float32_matmul_precision('high')


def build_parser():
    """The full training argument parser, exposed so pruning/ can extend it with
    parser.add_argument() instead of duplicating every flag."""
    parser = argparse.ArgumentParser(description='DCASE 24 argument parser')

    # YAML config (see configs/*.yaml); CLI flags passed alongside --config still win
    parser.add_argument('--config', type=str, default=None)

    # general
    # one shared W&B project so every run stays comparable in the UI
    parser.add_argument('--wandb_project', type=str, default="dcase24-pruning-thesis")
    # defaults to an auto-generated cm/subset name; pruning runs pass their own
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--precision', type=str, default="32")
    # fixes the python/numpy/torch RNG state and DataLoader worker seeding
    parser.add_argument('--seed', type=int, default=42)
    # wandb id of an interrupted run: reuses its history and last.ckpt to continue
    parser.add_argument('--resume_run_id', type=str, default=None)
    # pre-assign a wandb id to a FRESH run so it can be polled externally before it starts
    # logging; ignored when --resume_run_id is also given
    parser.add_argument('--run_id', type=str, default=None)
    # decode every WAV once into a memmap instead of once per epoch (training/waveform_cache.py).
    # Value-preserving, and the largest speedup available, at ~11.4 GB of disk for the 25% train
    # split plus the test set. Opt-in.
    parser.add_argument('--waveform_cache', action='store_true')
    parser.add_argument('--cache_dir', type=str, default=DEFAULT_CACHE_DIR)

    # evaluation
    parser.add_argument('--evaluate', action='store_true')  # predictions on eval set
    parser.add_argument('--ckpt_id', type=str, default=None)  # wandb run id of the checkpoint to load

    # dataset
    parser.add_argument('--orig_sample_rate', type=int, default=44100)
    parser.add_argument('--subset', type=int, default=25)  # {100, 50, 25, 10, 5}

    # model
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--in_channels', type=int, default=1)
    # the three axes for scaling the baseline width
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--channels_multiplier', type=float, default=1.8)
    parser.add_argument('--expansion_rate', type=float, default=2.1)

    # training
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--mixstyle_p', type=float, default=0.4)  # frequency mixstyle
    parser.add_argument('--mixstyle_alpha', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--roll_sec', type=int, default=0.1)  # random time shift, in seconds

    # peak learning rate of the cosine schedule
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--warmup_steps', type=int, default=2000)

    # preprocessing
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_length', type=int, default=3072)  # in samples (corresponds to 96 ms)
    parser.add_argument('--hop_length', type=int, default=500)  # in samples (corresponds to ~16 ms)
    parser.add_argument('--n_fft', type=int, default=4096)
    parser.add_argument('--n_mels', type=int, default=256)
    parser.add_argument('--freqm', type=int, default=48)  # mask up to 'freqm' spectrogram bins
    parser.add_argument('--timem', type=int, default=0)  # mask up to 'timem' spectrogram frames
    parser.add_argument('--f_min', type=int, default=0)  # mel bins are created for freqs. between 'f_min' and 'f_max'
    parser.add_argument('--f_max', type=int, default=None)

    return parser


def apply_config_file(parser):
    """Fold a --config YAML file's values into `parser` as defaults, so a flag passed on the
    CLI alongside --config still overrides it."""
    config_peek_parser = argparse.ArgumentParser(add_help=False)
    config_peek_parser.add_argument('--config', type=str, default=None)
    config_args, _ = config_peek_parser.parse_known_args()
    if not config_args.config:
        return parser

    with open(config_args.config) as f:
        yaml_config = yaml.safe_load(f)
    # configs/*.yaml uses a couple of names that don't match the CLI flag names
    key_map = {'project_name': 'wandb_project', 'experiment_name': 'run_name'}
    yaml_config = {key_map.get(k, k): v for k, v in yaml_config.items()}
    known_dests = {action.dest for action in parser._actions}
    # dataset_path comes from the DATASET_PATH env var, not a flag; every config file sets it
    unknown_keys = set(yaml_config) - known_dests - {'dataset_path'}
    if unknown_keys:
        logger.warning("config keys not recognized as CLI flags, ignoring: %s", sorted(unknown_keys))
    parser.set_defaults(**{k: v for k, v in yaml_config.items() if k in known_dests})
    return parser


if __name__ == '__main__':
    setup_logging()

    parser = apply_config_file(build_parser())
    args = parser.parse_args()
    if args.run_name is None:
        args.run_name = f"cm{args.channels_multiplier}-{args.subset}pct"

    pl.seed_everything(args.seed, workers=True)

    dataset_path = os.environ.get('DATASET_PATH', DEFAULT_DATASET_PATH)
    dcase24 = load_dcase24(dataset_path)

    if args.evaluate:
        evaluate(args, dcase24)
    else:
        train(args, dcase24)
