# Pruning the DCASE 2024 Task 1 baseline

Bachelor's thesis code. Johannes Kepler University Linz, Institute of Computational Perception.

Three pruning methods are applied to the official DCASE 2024 Task 1 baseline (CP-Mobile,
61,148 parameters) and each pruned model is compared against a **dense network of the same
parameter count trained from scratch** under an identical recipe. That comparison is a direct
test of the practical Lottery Ticket claim: does a subnetwork found by pruning beat a smaller
network you could simply have trained instead?

The setting makes the question harder than usual. CP-Mobile is not a large network waiting to
be compressed. It already spends 95.5 % of the challenge's 128 kB parameter budget and 89.8 %
of its 30 M multiply-accumulate budget, and it is trained here on the challenge's 25 % data
split. There is very little slack left to remove.

| | |
| --- | --- |
| Task | Acoustic scene classification, 10 classes |
| Dataset | TAU Urban Acoustic Scenes 2022 Mobile, 25 % training split |
| Metric | Macro-average accuracy, development-test set, last epoch |
| Baseline | CP-Mobile, 61,148 params, 119.43 KiB fp16, 26,954,388 MACs |
| Methods | IMP (magnitude, iterative), SNIP (single-shot at init), DSP (structured) |
| Replication | 3 seeds per point for SNIP, DSP and the contested dense references |

## Results

![Accuracy against parameter count](reports/figures/fig1_accuracy_vs_params.png)

Pruned minus same-size dense, in percentage points. Positive means the pruned model is ahead.
The top three rows are exact size matches; the bottom two sit in a gap in the architecture's
width grid, so they are compared against `base=16`, which is **larger** than the pruned model.

| size (KiB) | dense reference | IMP-150 | SNIP (n=3) | DSP (n=3) |
| ---: | :--- | ---: | ---: | ---: |
| 81.05 | cm=1.3 (n=5) | −0.29 | **+0.08** | −1.86 |
| 63.96 | cm=1.0 (n=3) | −0.53 | **+1.26** | −1.55 |
| 47.15 | cm=0.5 (n=3) | −2.34 | **−0.54** | **−3.23** *(p = 0.029)* |
| 33.82 | base=16 (n=3) | −0.86 | **+2.28** | −1.37 |
| 22.41 | base=16 (n=3) | −3.31 | **+1.74** | −0.04 |

**The answer is no: pruning matches same-size dense training here, it does not beat it.** SNIP
produces two ties and one win across the three exact matches. DSP sits below the dense curve
throughout, and at 47 KiB it is significantly worse (Welch t = −3.97, p = 0.029), the only
comparison in the study that reaches p < 0.05 and a negative one.

Four things are worth pulling out.

**SNIP's curve is nearly flat.** Its five three-seed means span 1.70 pp across a 3.6× reduction
in parameters. Against the source model this project actually pruned (0.4991), SNIP at 22.41 KiB
gives up 0.80 pp while being 5.3× smaller.

**Only structured pruning buys compute.** IMP and SNIP execute 26,954,388 MACs at every sparsity
level, identical to the dense baseline, because masking a weight does not change any tensor's
shape. DSP cuts executed MACs by 31.6 % to 82.4 %. Its best result is at 22 KiB, where it ties
`base=16` on accuracy using 38 % fewer parameters and 39 % fewer MACs.

**IMP's retraining budget matters more than the criterion at high sparsity.** Going from 20 to
150 retrain epochs per round is worth +0.16 pp at 81 KiB and +7.05 pp at 22 KiB. Han's paper
never specifies an epoch budget, so any IMP result at high sparsity is under-determined until
that number is stated.

**Almost nothing here is statistically significant, and that is a result.** The pooled
seed-to-seed standard deviation is 1.29 pp over 30 degrees of freedom. With three seeds a side,
a margin has to reach 2.93 pp before Welch's test can reject at α = 0.05. Exactly one does.
Several conclusions drawn from single-seed data earlier in the project reversed sign once more
seeds landed, including two changes of sign; `reports/results.md` lists them under
**Retractions** rather than quietly dropping them.

Full tables, per-seed values, significance tests and run identifiers are in
[`reports/results.md`](reports/results.md), which is generated, never hand-edited.

## Layout

```
baseline/      DCASE 2024 baseline, vendored read-only (see NOTICE.md)
training/      dense training: recipe, dataset wiring, optional waveform cache
pruning/       IMP, SNIP and DSP; masking, sparsity accounting, MACs measurement
experiments/   results collection, statistics and figure generation
configs/       one YAML per experiment; no hyperparameters live in code
tests/         56 tests, including regressions on the two highest-stakes invariants
reports/       generated results, figures and the MACs measurements
```

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

The dataset is not included. Download the TAU Urban Acoustic Scenes 2022 Mobile development
set from [Zenodo](https://zenodo.org/record/6337421), unpack it so that `audio/`,
`evaluation_setup/` and `meta.csv` sit inside one directory, and point `DATASET_PATH` at it:

```bash
export DATASET_PATH=data/tau_scenes_dataset      # PowerShell: $env:DATASET_PATH = "data/..."
```

The official split CSVs are vendored under `baseline/dataset/splits/`, so no file is fetched at
runtime.

## Running things

Train a dense reference:

```bash
uv run python -m training.run_training --config configs/scale_down_base16.yaml --seed 42
```

Run a pruning curve. Each config carries its own sparsity list, chosen so the pruned models land
on the parameter counts of the dense references:

```bash
uv run python -m pruning.run_pruning --config configs/snip.yaml --seed 42
uv run python -m pruning.run_pruning --config configs/imp.yaml --seed 42
uv run python -m pruning.run_pruning --config configs/imp_retrain150.yaml --seed 42
uv run python -m pruning.run_pruning --config configs/dsp.yaml --seed 42
```

A run that dies partway can be resumed without repeating finished levels:

```bash
uv run python -m pruning.run_pruning --config configs/snip.yaml --resume_run_id <wandb_run_id>
```

Measure how the two unstructured criteria distribute sparsity across layers:

```bash
uv run python -m pruning.measure_allocation
```

Regenerate `reports/results.md`, `reports/results.json` and every figure from the checkpoints:

```bash
uv run python -m experiments.collect_results
uv run python -m experiments.plot_results --bare --outdir <dir>   # figures without in-image captions
```

Run the tests:

```bash
uv run pytest
```

## Notes on method

A few decisions are easy to get wrong and are worth stating up front.

**Accuracy is always the last epoch's.** DCASE Task 1 has no validation split, so selecting the
best epoch by test accuracy is selecting on the test set. On the dense runs here that inflates
the number by roughly 0.5 to 0.8 pp, which is the size of most of the margins being measured.

**Two different numbers exist for the cm=1.8 baseline.** This project's own trained checkpoint
scores 0.4991 and is what every pruning run started from; the organisers report 0.5029 ± 0.0087
over five runs. The second is reproduction evidence, not a comparison point, and the two are
never mixed.

**Masked models are reported by non-zero parameter count.** As plain fp16 tensor files, IMP and
SNIP models still occupy 119.43 KiB whatever their sparsity. Their quoted sizes assume a sparse
storage format, and their MAC counts do not improve at all. DSP's removals are structural, so
its shapes really do shrink.

**The DSP implementation is an adaptation.** Three deviations from the reference are marked
`DEVIATION` in `pruning/dsp.py` and listed in `NOTICE.md`.

## Licence

MIT, see [`LICENSE`](LICENSE). Third-party code and its licences are listed in
[`NOTICE.md`](NOTICE.md).
