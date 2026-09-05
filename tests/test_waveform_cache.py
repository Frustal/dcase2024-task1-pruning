"""Invariants of the waveform cache.

Caching is justified only by the decode being deterministic, which makes
test_cached_waveforms_are_bit_identical_to_fresh_decode load-bearing: a cached tensor differing
from a fresh torchaudio.load() means the cache is changing the training data.
test_roll_still_varies_per_access is the matched control for the other failure mode, caching
above RollDataset, which freezes the augmentation to one shift per clip and shows up in no metric.

Uses the real dataset at subset=5 (the smallest split) and is skipped when it is not present.

    uv run pytest tests/test_waveform_cache.py -v
"""
import os

import numpy as np
import pytest
import torch

from training.dataset_loader import load_dcase24
from training.waveform_cache import cached, cached_training_set

DATASET_PATH = os.environ.get("DATASET_PATH", "data/tau_scenes_dataset")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(DATASET_PATH),
    reason=f"real dataset not present at {DATASET_PATH}; set DATASET_PATH",
)

SUBSET = 5  # smallest split, ~7k clips -- enough to exercise everything, quick to build


@pytest.fixture(scope="module")
def dcase24():
    return load_dcase24(DATASET_PATH)


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("wavecache"))


@pytest.fixture(scope="module")
def uncached(dcase24):
    return dcase24.get_training_set(SUBSET, roll=False)


@pytest.fixture(scope="module")
def cached_ds(dcase24, uncached, cache_dir):
    return cached(uncached, f"train{SUBSET}", cache_dir, num_workers=4, dataset_dir=DATASET_PATH)


def test_cached_waveforms_are_bit_identical_to_fresh_decode(uncached, cached_ds):
    assert len(cached_ds) == len(uncached)
    # spread the probes across the whole split rather than clustering at the start, so a
    # row-offset bug in the memmap writer cannot slip through
    for index in range(0, len(uncached), max(1, len(uncached) // 50)):
        fresh = uncached[index][0]
        from_cache = cached_ds[index][0]
        assert torch.equal(fresh, from_cache), f"waveform mismatch at index {index}"


def test_metadata_matches_uncached_exactly(uncached, cached_ds):
    """Labels, devices and cities must keep their original types, not just their values:
    the collate function and the per-device/per-class aggregation depend on a 0-dim int64 tensor
    and numpy integers respectively.
    """
    for index in (0, len(uncached) // 3, len(uncached) - 1):
        _, f_fresh, l_fresh, d_fresh, c_fresh = uncached[index]
        _, f_cache, l_cache, d_cache, c_cache = cached_ds[index]
        assert f_fresh == f_cache
        assert torch.equal(torch.as_tensor(l_fresh), torch.as_tensor(l_cache))
        assert l_fresh.dtype == l_cache.dtype
        assert (d_fresh, c_fresh) == (d_cache, c_cache)
        assert type(d_fresh) is type(d_cache)


def test_waveform_dtype_and_shape_preserved(uncached, cached_ds):
    fresh, from_cache = uncached[0][0], cached_ds[0][0]
    assert fresh.dtype == from_cache.dtype == torch.float32
    assert fresh.shape == from_cache.shape


def test_roll_still_varies_per_access(dcase24, cache_dir):
    """RollDataset must sit OUTSIDE the cache, so the shift is redrawn on every access."""
    rolled = cached_training_set(dcase24, SUBSET, roll=4410, cache_dir=cache_dir,
                                 num_workers=4, dataset_dir=DATASET_PATH)
    np.random.seed(0)
    draws = {rolled[7][0].numpy().tobytes() for _ in range(12)}
    assert len(draws) > 1, "roll augmentation is frozen -- the cache is wrapping RollDataset"


def test_rolled_output_is_a_shift_of_the_cached_clip(dcase24, cached_ds, cache_dir):
    """The roll must be a pure circular shift of the cached waveform, not a different clip."""
    rolled = cached_training_set(dcase24, SUBSET, roll=4410, cache_dir=cache_dir,
                                 num_workers=4, dataset_dir=DATASET_PATH)
    base = cached_ds[7][0]
    shifted = rolled[7][0]
    assert sorted(base.flatten().tolist()) == sorted(shifted.flatten().tolist())


class Truncated(torch.utils.data.Dataset):
    """Half of another dataset. Module-level so it survives pickling to a 'spawn' worker."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset) // 2


def test_stale_cache_is_rebuilt_when_length_changes(dcase24, uncached, cache_dir):
    """A cache built for one subset must never be served for another."""
    name = "collide"
    cached(uncached, name, cache_dir, num_workers=4, dataset_dir=DATASET_PATH)

    rebuilt = cached(Truncated(uncached), name, cache_dir, num_workers=4,
                     dataset_dir=DATASET_PATH)
    assert len(rebuilt) == len(uncached) // 2
    assert torch.equal(rebuilt[3][0], uncached[3][0])
