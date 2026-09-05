# Third-party code and attribution

This repository contains code from other projects. What follows is what was taken, from where,
and under which licence. Everything not listed here was written for this thesis and is covered
by `LICENSE` (MIT).

## `baseline/` — DCASE 2024 Task 1 baseline system

Vendored verbatim from the challenge organisers' repository:

- Source: https://github.com/CPJKU/dcase2024_task1_baseline
- Authors: Florian Schmid, Paul Primus, Toni Heittola, Annamaria Mesaros,
  Irene Martín-Morató, Khaled Koutini, Gerhard Widmer (CP-JKU)
- Licence: MIT

The directory holds the CP-Mobile model definition, the mel front-end, the dataset wrapper,
the Frequency-MixStyle augmentation, the seeding helper and the NeSsi complexity checker.

It is treated as read-only. Two mechanical changes were needed to vendor it as a package, and
they are the only differences from upstream:

- `models/baseline.py`: the import `from models.helpers.utils import make_divisible` became
  `from baseline.models.helpers.utils import ...`, because the code now sits under `baseline/`
  rather than at the repository root;
- `dataset/dcase24.py`: one trailing blank line removed.

No logic, hyperparameter or architecture was touched. Everything else extends the baseline from
the outside: `training/dataset_loader.py`, for instance, works around the dataset module's
hardcoded path by generating a patched copy at import time instead of editing the original.

The split CSVs under `baseline/dataset/splits/` are the organisers' official subsets, vendored
so that a run does not depend on fetching them from GitHub at runtime.

Papers describing the baseline:

- F. Schmid et al., *Data-Efficient Low-Complexity Acoustic Scene Classification in the DCASE
  2024 Challenge*, arXiv:2405.10018, 2024.
- F. Schmid, T. Morocutti, S. Masoudian, K. Koutini, G. Widmer, *Distilling the Knowledge of
  Transformers and CNNs with CP-Mobile*, DCASE Workshop 2023, pp. 161–165.

## `pruning/dsp.py` — port of Dynamic Structure Pruning

Ported to CP-Mobile from the authors' released implementation:

- Source: https://github.com/irishev/DSP
- Authors: Jun-Hyung Park, Yeachan Kim, Junho Kim, Joon-Young Choi, SangKeun Lee
- Licence: Apache 2.0
- Paper: *Dynamic Structure Pruning for Compressing CNNs*, AAAI 2023, 37(8), pp. 9408–9416.

This is an **adaptation**, not a faithful reimplementation. Three deviations from the reference
are marked `DEVIATION` in `pruning/dsp.py` and summarised in its module docstring:

1. the pruning threshold is solved against this project's parameter-sparsity measure over all
   convolution weights, not against the reference's internal FLOPs percentage;
2. the regulariser's per-layer scale drops the reference's activation-size term, which biases
   pruning toward FLOPs rather than parameters;
3. whole-filter pruning is restricted to blocks whose output width is not pinned by an identity
   shortcut, so that every parameter reported as removed is genuinely removable.

## Methods implemented from published descriptions

`pruning/importance.py` implements the saliency criteria of:

- S. Han, J. Pool, J. Tran, W. J. Dally, *Learning both Weights and Connections for Efficient
  Neural Networks*, NIPS 2015, pp. 1135–1143.
- N. Lee, T. Ajanthan, P. H. S. Torr, *SNIP: Single-shot Network Pruning based on Connection
  Sensitivity*, ICLR 2019.

No code was taken from either; both are implemented from the papers, and the places where this
project departs from them are documented in the source.

## Dataset

TAU Urban Acoustic Scenes 2022 Mobile, development set. Not included in this repository. See
the README for where to obtain it.
