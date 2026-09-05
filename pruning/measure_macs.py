"""Measures MACs (multiply-accumulates) for every model in the comparison, dense and pruned.

The rest of the thesis compares methods on parameter count, which hides the largest practical
difference between them. IMP and SNIP are unstructured: a zeroed weight keeps its place in the
tensor, the convolution still multiplies by it, and MACs stay at the dense figure at every
sparsity level. DSP is structured: it removes whole (filter-group, input-channel) connections
that collapse into a grouped convolution, so its zeros really do leave the computation. Hence two
numbers per model:

  macs_dense       what the tensor shapes cost, i.e. what actually runs today.
  macs_structural  what the non-zero weights cost: equal to macs_dense for a dense model, the
                   deployable figure for DSP, and for IMP and SNIP a hypothetical reachable only
                   with sparse kernels, which are out of scope here. Label it as such wherever
                   it is reported.

The convention matches nessi/torchinfo, so the numbers compare with what run_training.py logs:
``weight.numel() * out_H * out_W`` for a Conv2d, ``weight.numel()`` for a Linear, its affine
parameter count for a BatchNorm (only so the columns reconcile exactly), and nothing for
activations or pooling. The dense column is asserted against ``nessi.get_torch_size`` at startup
so a drift fails loudly. Parameter counts come from ``pruning.sparsity.count_params``, since a
naive non-zero count over ``model.parameters()`` would call a freshly built dense model 2.2%
pruned: CP-Mobile zero-initialises 1,354 BatchNorm biases.

Run: uv run python -m pruning.measure_macs --json reports/macs.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from baseline.helpers import nessi
from baseline.models.baseline import get_model
from baseline.models.mel import AugmentMelSTFT
from pruning.sparsity import count_params

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"
CHECKPOINTS = REPO / "checkpoints"

FP16_BYTES = 2
CLIP_SECONDS = 1.0

MODEL_KEYS = ("n_classes", "in_channels", "base_channels", "channels_multiplier", "expansion_rate")


# ---- input shape ----

def mel_input_shape(cfg: dict) -> tuple[int, ...]:
    """Shape of the log-mel tensor the CP-Mobile model actually receives.

    Derived from the config's mel settings and a one-second clip rather than hardcoded, so it
    tracks configs/*.yaml. freqm/timem are 0: SpecAugment changes values, never shape.
    """
    mel = AugmentMelSTFT(
        n_mels=cfg["n_mels"], sr=cfg["sample_rate"], win_length=cfg["window_length"],
        hopsize=cfg["hop_length"], n_fft=cfg["n_fft"], freqm=0, timem=0,
        fmin=cfg["f_min"], fmax=cfg["f_max"],
    ).eval()
    with torch.no_grad():
        out = mel(torch.zeros(1, int(cfg["sample_rate"] * CLIP_SECONDS)))
    return (1, 1, out.shape[1], out.shape[2])


# ---- MACs ----

@dataclass
class MacCounts:
    macs_dense: int
    macs_structural: int
    params_total: int
    params_nonzero: int

    @property
    def macs_reduction(self) -> float:
        return 1.0 - self.macs_structural / self.macs_dense

    @property
    def size_kb(self) -> float:
        return self.params_nonzero * FP16_BYTES / 1024


@torch.no_grad()
def count_macs(model: nn.Module, input_shape: tuple[int, ...]) -> MacCounts:
    dense = structural = 0
    handles = []

    def conv_hook(mod, inp, out):
        nonlocal dense, structural
        spatial = out.shape[2] * out.shape[3]
        dense += mod.weight.numel() * spatial
        structural += int((mod.weight != 0).sum().item()) * spatial

    def linear_hook(mod, inp, out):
        nonlocal dense, structural
        dense += mod.weight.numel()
        structural += int((mod.weight != 0).sum().item())

    def norm_hook(mod, inp, out):
        # torchinfo attributes mult-adds to BatchNorm equal to its affine parameter count. Tiny,
        # but required, or the reconciliation against nessi fails by exactly 2,708.
        nonlocal dense, structural
        n = sum(p.numel() for p in mod.parameters())
        dense += n
        structural += n

    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(linear_hook))
        elif isinstance(mod, nn.modules.batchnorm._BatchNorm):
            handles.append(mod.register_forward_hook(norm_hook))

    was_training = model.training
    model.eval()
    model(torch.zeros(*input_shape))
    if was_training:
        model.train()
    for h in handles:
        h.remove()

    counts = count_params(model)
    return MacCounts(dense, structural, counts.total, counts.nonzero)


def assert_matches_nessi(model: nn.Module, input_shape: tuple[int, ...], counts: MacCounts) -> None:
    """Fail loudly if the hook convention has drifted from the one the challenge budget uses."""
    with contextlib.redirect_stdout(io.StringIO()):
        macs, params = nessi.get_torch_size(model, input_size=input_shape)
    if macs != counts.macs_dense or params != counts.params_total:
        raise AssertionError(
            "MAC/param convention drifted from nessi: "
            f"ours ({counts.macs_dense}, {counts.params_total}) vs "
            f"nessi ({macs}, {params})"
        )


# ---- loading ----

def load_config(name: str) -> dict:
    with open(CONFIGS / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build(cfg: dict) -> nn.Module:
    return get_model(**{k: cfg[k] for k in MODEL_KEYS})


def load_weights(model: nn.Module, ckpt: Path, *, dsp_groups: int | None = None) -> nn.Module:
    """Load a training or pruning checkpoint's model weights into ``model``. DSP checkpoints
    carry extra ``group``/``mask`` buffers on the prunable convs, and load_state_dict is strict
    both ways, so those buffers have to exist on the model first."""
    if dsp_groups is not None:
        from pruning.dsp import attach_dsp_buffers
        attach_dsp_buffers(model, dsp_groups, with_mask=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k[len("model."):]: v for k, v in state.items() if k.startswith("model.")})
    return model


# ---- the model inventory ----

# Dense references, one trained model per width: the scale-down curve pruning is judged against.
DENSE = [
    ("cm=1.8 (source)", "baseline.yaml", "762f0b4fde744636b143bf56646046b7"),
    ("cm=1.3",          "scale_down_cm1.3.yaml",  None),
    ("base=24",         "scale_down_base24.yaml", None),
    ("cm=1.0",          "scale_down_cm1.0.yaml",  None),
    ("cm=0.5",          "scale_down_cm0.5.yaml",  None),
    ("base=16",         "scale_down_base16.yaml", None),
    ("base=8",          "scale_down_base8.yaml",  None),
]

# Pruning curves, one entry per (method, seed); all share the same five sparsity targets, so the
# table rows line up. Every seed is measured rather than reusing seed 42's: a different seed keeps
# a different mask, so MACs* differs at identical parameter sparsity, and a DSP seed also re-runs
# phase A, the learned grouping, changing the executed shapes too. IMP is single-seed (20/35/150
# retrain epochs, seed 42) while SNIP and DSP have three, an asymmetry that must stay visible.
CURVES = [
    ("IMP-20",  42, "d3jo4g1u", "imp.yaml",            False),
    ("IMP-35",  42, "86guyp3j", "imp_retrain35.yaml",  False),
    ("IMP-150", 42, "66d7cow0", "imp_retrain150.yaml", False),
    ("SNIP",    42, "iel3qnvt", "snip.yaml",           False),
    ("SNIP",    43, "pjqyh53l", "snip.yaml",           False),
    ("SNIP",    44, "cul3sq06", "snip.yaml",           False),
    ("DSP",     42, "pv6k1wze", "dsp.yaml",            True),
    ("DSP",     43, "quuhqh9f", "dsp.yaml",            True),
    ("DSP",     44, "wd05nhmr", "dsp.yaml",            True),
]


def level_dirs(run_id: str) -> list[tuple[str, Path]]:
    """The per-sparsity subdirectories of a curve, in ascending sparsity order."""
    root = CHECKPOINTS / run_id
    return sorted(((d.name, d) for d in root.glob("sparsity*") if d.is_dir()),
                  key=lambda nd: int(nd[0][len("sparsity"):]))


def completed_levels(run_id: str) -> dict[str, dict]:
    """``pruning_state.json``'s completed list, keyed by the sparsity-dir name it corresponds to.
    That file is the authoritative record of accuracy and size; this script only adds MACs."""
    with open(CHECKPOINTS / run_id / "pruning_state.json", encoding="utf-8") as fh:
        state = json.load(fh)
    out = {}
    for entry in state["completed"]:
        out[f"sparsity{int(round(entry['target'] * 100))}"] = entry
    return out


def measure_dense(verify: bool) -> list[dict]:
    rows = []
    for label, cfg_name, run_id in DENSE:
        cfg = load_config(cfg_name)
        shape = mel_input_shape(cfg)
        model = build(cfg)
        if run_id is not None:
            ckpt = CHECKPOINTS / run_id / "last.ckpt"
            if ckpt.exists():
                load_weights(model, ckpt)
        counts = count_macs(model, shape)
        if verify:
            assert_matches_nessi(model, shape, counts)
        rows.append({"group": "dense", "method": "dense", "label": label,
                     "config": cfg_name, **asdict(counts)})
    return rows


def measure_curves(verify: bool) -> list[dict]:
    rows = []
    for method, seed, run_id, cfg_name, is_dsp in CURVES:
        cfg = load_config(cfg_name)
        shape = mel_input_shape(cfg)
        state = completed_levels(run_id)
        for dir_name, path in level_dirs(run_id):
            ckpt = path / "last.ckpt"
            if not ckpt.exists():
                continue
            model = build(cfg)
            load_weights(model, ckpt, dsp_groups=cfg["dsp_groups"] if is_dsp else None)
            counts = count_macs(model, shape)
            if verify:
                # dense MACs must still equal the unpruned model's, since pruning never changes a
                # shape here; a failure means the checkpoint's architecture is not the config's
                assert_matches_nessi(model, shape, counts)
            entry = state.get(dir_name, {})
            rows.append({
                "group": "pruned", "method": method, "seed": seed, "run_id": run_id,
                "level": dir_name,
                "label": f"{method} s{seed} {dir_name}",
                "target": entry.get("target"),
                "sparsity": entry.get("sparsity"),
                "accuracy": entry.get("test_macro_accuracy"),
                "state_nonzero_params": entry.get("nonzero_params"),
                "state_size_kb": entry.get("size_kb"),
                **asdict(counts),
            })
    return rows


def macs_star_spread(rows: list[dict]) -> list[dict]:
    """Per-(method, level) spread of MACs* across seeds. For DSP it also covers executed MACs,
    because a DSP seed changes the architecture itself."""
    buckets: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row["group"] != "pruned":
            continue
        buckets.setdefault((row["method"], row["level"]), []).append(row)
    out = []
    for (method, level), group in buckets.items():
        if len(group) < 2:
            continue
        star = [r["macs_structural"] for r in group]
        dense_shape = [r["macs_dense"] for r in group]
        out.append({
            "method": method, "level": level, "n_seeds": len(group),
            "seeds": sorted(r["seed"] for r in group),
            "macs_star_min": min(star), "macs_star_max": max(star),
            "macs_star_mean": sum(star) / len(star),
            "macs_star_spread_pct": 100 * (max(star) - min(star)) / (sum(star) / len(star)),
            # masking never changes a shape, so shape MACs must be identical across seeds for
            # every method, DSP included: its removals collapse into a smaller op only at deploy
            "macs_dense_min": min(dense_shape), "macs_dense_max": max(dense_shape),
            "macs_dense_identical": min(dense_shape) == max(dense_shape),
        })
    out.sort(key=lambda d: (d["method"], d["level"]))
    return out


def check_against_state(rows: list[dict]) -> list[str]:
    """Cross-check the recomputed parameter counts against pruning_state.json. Both come from the
    same ``count_params`` on the same checkpoints, so they must agree exactly; a mismatch means the
    checkpoint on disk is not the one the state file describes."""
    problems = []
    for row in rows:
        expected = row.get("state_nonzero_params")
        if expected is not None and expected != row["params_nonzero"]:
            problems.append(
                f"{row['label']}: state.json says {expected} non-zero params, "
                f"checkpoint has {row['params_nonzero']}"
            )
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", type=str, default=None,
                        help="write the full table to this path as JSON")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the nessi reconciliation (faster, but unguarded)")
    args = parser.parse_args()
    verify = not args.no_verify

    rows = measure_dense(verify) + measure_curves(verify)

    problems = check_against_state(rows)

    dense_macs = rows[0]["macs_dense"]
    print(f"\nCP-Mobile cm=1.8 dense reference: {dense_macs:,} MACs, "
          f"limit {nessi.MAX_MACS:,} ({100 * dense_macs / nessi.MAX_MACS:.1f}% of budget)\n")

    header = f"{'model':<24} {'params':>8} {'size KB':>8} {'acc':>7} {'MACs':>12} {'MACs*':>12} {'cut':>7}"
    print(header)
    print("-" * len(header))
    last_group = None
    for row in rows:
        group_key = (row["method"], row.get("seed"))
        if group_key != last_group:
            print()
            last_group = group_key
        acc = row.get("accuracy")
        counts = MacCounts(row["macs_dense"], row["macs_structural"],
                           row["params_total"], row["params_nonzero"])
        print(f"{row['label']:<24} {row['params_nonzero']:>8,} {counts.size_kb:>8.2f} "
              f"{(f'{acc:.4f}' if acc is not None else '-'):>7} "
              f"{row['macs_dense']:>12,} {row['macs_structural']:>12,} "
              f"{100 * counts.macs_reduction:>6.1f}%")

    print("\nMACs  = what the tensor SHAPES cost -- what actually runs today.")
    print("MACs* = what the NON-ZERO weights cost. For dense models the two are equal. For DSP")
    print("        it is the deployable figure (its removals collapse into a grouped conv). For")
    print("        IMP and SNIP it is HYPOTHETICAL -- reachable only with sparse kernels, which")
    print("        this thesis rules out of scope. Never report MACs* for IMP/SNIP unlabelled.")

    spreads = macs_star_spread(rows)
    if spreads:
        print("\nPER-SEED MACs* SPREAD (why MACs are re-measured per seed, not reused from 42)")
        hdr = (f"{'method':<8} {'level':<11} {'n':>2} {'MACs* min':>12} {'MACs* max':>12} "
               f"{'spread':>7}  shape MACs equal?")
        print(hdr)
        print("-" * len(hdr))
        for sp in spreads:
            print(f"{sp['method']:<8} {sp['level']:<11} {sp['n_seeds']:>2} "
                  f"{sp['macs_star_min']:>12,} {sp['macs_star_max']:>12,} "
                  f"{sp['macs_star_spread_pct']:>6.2f}%  "
                  f"{'yes' if sp['macs_dense_identical'] else 'NO -- INVESTIGATE'}")

    if problems:
        print("\nMISMATCHES against pruning_state.json:")
        for p in problems:
            print(f"  ! {p}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
