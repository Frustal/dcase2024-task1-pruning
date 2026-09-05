"""Measures how SNIP and Han/IMP allocate sparsity across CP-Mobile's layers.

Prints three tables: (A) the parameter breakdown, including the BatchNorm size floor, (B) the
per-layer SNIP saliency |w * dL/dw| at init on a real log-mel batch, and (C) the per-layer
percentage of weights kept, SNIP's global top-k against Han's layer-wise q*std. Both criteria are
the papers' (arXiv:1810.02340 and arXiv:1506.02626), with Han's q bisected to a target sparsity
so the two share an axis. Reads baseline/ and checkpoints/, writes nothing back.

Run: DATASET_PATH=data/tau_scenes_dataset uv run python -m pruning.measure_allocation
"""
import os

import torch
import torchaudio
from torch.utils.data import DataLoader

from baseline.models.baseline import get_model
from pruning.importance import global_thresholds, han_thresholds, magnitude_scores, snip_scores
from pruning.masking import MaskRegistry
from pruning.prunable import layer_kind, prunable_layers
from pruning.sparsity import count_params
from training.dataset_loader import load_dcase24
from training.worker_init import worker_init_fn

# trained cm=1.8 baseline, epoch 149/150. Predates the best-checkpoint callback, so last.ckpt is
# this run's only trained-weight snapshot.
CKPT = "checkpoints/762f0b4fde744636b143bf56646046b7/last.ckpt"
DEFAULT_DATASET_PATH = "data/tau_scenes_dataset"
SEED = 42
SPARSITIES = (0.3, 0.5, 0.7)

# model + mel settings, verbatim from configs/baseline.yaml
MODEL = dict(n_classes=10, in_channels=1, base_channels=32,
             channels_multiplier=1.8, expansion_rate=2.1)
MEL = dict(sample_rate=32000, n_fft=4096, win_length=3072, hop_length=500,
           n_mels=256, f_min=0, f_max=None)
ORIG_SAMPLE_RATE, BATCH_SIZE, SUBSET, ROLL_SEC = 44100, 256, 25, 0.1


def load_trained(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    model = get_model(**MODEL)
    model.load_state_dict({k[len("model."):]: v for k, v in state.items() if k.startswith("model.")})
    return model


def real_batch():
    """One mini-batch of log-mel features from the real 25% training split."""
    dcase24 = load_dcase24(os.environ.get("DATASET_PATH", DEFAULT_DATASET_PATH))
    loader = DataLoader(
        dataset=dcase24.get_training_set(SUBSET, roll=int(ORIG_SAMPLE_RATE * ROLL_SEC)),
        worker_init_fn=worker_init_fn, num_workers=0, batch_size=BATCH_SIZE, shuffle=True,
    )
    waveforms, _, labels, _, _ = next(iter(loader))
    # PLModule.mel_forward's pipeline, minus spec-augment and MixStyle: SNIP wants the clean task
    # gradient, and augmentation would only add noise to the saliency scores.
    mel = torch.nn.Sequential(
        torchaudio.transforms.Resample(orig_freq=ORIG_SAMPLE_RATE, new_freq=MEL["sample_rate"]),
        torchaudio.transforms.MelSpectrogram(**MEL),
    )
    return (mel(waveforms) + 1e-5).log(), labels


def main():
    torch.manual_seed(SEED)
    x, y = real_batch()
    print(f"real log-mel batch: {tuple(x.shape)}  mean={x.mean():.3f}  std={x.std():.3f}\n")

    # ---- A. parameter breakdown ----
    init_model = get_model(**MODEL)
    counts = count_params(init_model)
    print(f"=== A. parameter breakdown (cm={MODEL['channels_multiplier']}, "
          f"base={MODEL['base_channels']}) ===")
    print(f"  total                : {counts.total:,}  ({counts.dense_size_kb():.2f} KB fp16)")
    print(f"  conv weights         : {counts.conv_total:,}  "
          f"({counts.conv_total / counts.total:.1%})  <- prunable")
    print(f"  batchnorm params     : {counts.unprunable:,}  "
          f"({counts.unprunable / counts.total:.1%})  <- never pruned")
    print(f"  size floor (BN only) : {counts.unprunable * 2 / 1024:.2f} KB fp16\n")

    # ---- B. SNIP saliency at init ----
    snip = snip_scores(init_model, x, y)
    print("=== B. SNIP saliency |w * dL/dw| at init (real batch) ===")
    print(f'{"layer":<34}{"kind":<11}{"n":>8}{"mean|w|":>10}{"mean|w*g|":>12}')
    for name, mod in prunable_layers(init_model):
        print(f"{name:<34}{layer_kind(mod):<11}{mod.weight.numel():>8,}"
              f"{mod.weight.data.abs().mean():>10.4f}{snip[name].mean():>12.2e}")
    print()

    # ---- C. per-layer allocation, SNIP vs Han ----
    trained = load_trained(CKPT)
    magnitudes = magnitude_scores(trained)

    snip_kept, han_kept, qs = {}, {}, {}
    for sparsity in SPARSITIES:
        snip_mask = MaskRegistry.from_thresholds(
            init_model, snip, global_thresholds(snip, sparsity))
        snip_kept[sparsity] = snip_mask.kept_fraction()

        qs[sparsity], han_thr = han_thresholds(trained, magnitudes, sparsity)
        han_kept[sparsity] = MaskRegistry.from_thresholds(
            trained, magnitudes, han_thr).kept_fraction()

    print("=== C. % of each layer's weights KEPT ===")
    print("SNIP: global top-k on |w*g|, at random init (Lee et al. 2019)")
    print(f"Han:  layer-wise q*std(W_layer), on trained weights from {CKPT} (Han et al. 2015)\n")
    sp_hdr = "".join(f"{s:>7.0%}" for s in SPARSITIES)
    print(f'{"":<34}{"":<11}{"":>8}   SNIP{"":<10}   Han')
    print(f'{"layer":<34}{"kind":<11}{"n":>8}{sp_hdr}   {sp_hdr}')
    for name, mod in prunable_layers(init_model):
        s_cells = "".join(f"{snip_kept[s][name]:>7.0%}" for s in SPARSITIES)
        h_cells = "".join(f"{han_kept[s][name]:>7.0%}" for s in SPARSITIES)
        print(f"{name:<34}{layer_kind(mod):<11}{mod.weight.numel():>8,}{s_cells}   {h_cells}")
    print(f'\n  Han solved q: ' + ", ".join(f"{s:.0%} -> q={qs[s]:.4f}" for s in SPARSITIES))


if __name__ == "__main__":
    main()
