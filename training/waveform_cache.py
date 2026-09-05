"""Decode every WAV once and serve waveforms from a memory-mapped file.

Training is dataloader-bound: torchaudio.load() in BasicDCASE24Dataset.__getitem__ costs
~7.6 ms/sample, nearly all of the ~0.275 s a 256-sample batch spends in the dataloader against
0.142 s in training_step. That decode is deterministic and repeats every epoch, so caching it
removes the cost without changing a sample value.

The cache must sit strictly below the first source of randomness:

    RollDataset               <- random time shift, different every access, must stay live
      CachedWaveformDataset   <- inserted here
        SimpleSelectionDataset
          BasicDCASE24Dataset <- torchaudio.load(), the part being cached

Caching at or above RollDataset would freeze the roll to one shift per clip for the whole run;
the same reasoning rules out caching mel spectrograms, since SpecAugment redraws masks per
step. Clips are 24-bit PCM and float32's 24-bit mantissa holds every sample exactly, so the
cache is bit-identical to a fresh decode; int16 would not be.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = "cache"


def _identity_collate(batch):
    """Pass the raw list of tuples through unbatched. Must be module level, not a lambda:
    DataLoader workers use the 'spawn' start method on Windows, which pickles collate_fn, and
    an unpicklable local surfaces as an EOFError inside the worker."""
    return batch


class CachedWaveformDataset(TorchDataset):
    """Serves (waveform, file, label, device, city) with the waveform read from a memmap.

    Types match the wrapped dataset exactly: float32 (1, 44100) tensor, numpy str_ filename,
    0-dim int64 label tensor, numpy int64 device and city ids."""

    def __init__(self, waveform_path: str, meta: dict):
        self.waveform_path = waveform_path
        self.files = meta["files"]
        self.labels = meta["labels"]
        self.devices = meta["devices"]
        self.cities = meta["cities"]
        # opened lazily: a np.memmap does not survive pickling to a 'spawn' worker, so every
        # worker opens the file itself. The shared OS page cache keeps that from costing N copies.
        self._waveforms = None

    def __getitem__(self, index):
        if self._waveforms is None:
            self._waveforms = np.load(self.waveform_path, mmap_mode="r")
        # np.array() copies out of the memmap; handing a view to a worker would keep the
        # mapping alive across the queue and can fault after the parent rebinds the file
        waveform = torch.from_numpy(np.array(self._waveforms[index]))
        return waveform, self.files[index], self.labels[index], self.devices[index], \
            self.cities[index]

    def __len__(self):
        return len(self.files)


def _build(dataset, waveform_path: Path, meta_path: Path, num_workers: int, dataset_dir: str):
    """Decode the whole dataset once, streaming it into a memmap on disk. The DataLoader only
    parallelises the decode (~8 min single-threaded for both splits, ~1 min over 8 workers);
    shuffle is off so row i is dataset index i, and the identity collate keeps tuples unbatched."""
    logger.info("building waveform cache for %d samples -> %s", len(dataset), waveform_path)
    waveform_path.parent.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=num_workers,
                        collate_fn=_identity_collate)

    memmap = None
    files, labels, devices, cities = [], [], [], []
    written = 0
    for batch in loader:
        for waveform, file, label, device, city in batch:
            if memmap is None:
                # allocate on the first sample, once its true shape is known
                memmap = np.lib.format.open_memmap(
                    waveform_path, mode="w+", dtype=np.float32,
                    shape=(len(dataset), *waveform.shape),
                )
            memmap[written] = waveform.numpy()
            files.append(file)
            labels.append(label)
            devices.append(device)
            cities.append(city)
            written += 1
        if written % 6400 < 32:
            logger.info("  cached %d/%d", written, len(dataset))

    if written != len(dataset):
        raise RuntimeError(f"cache build wrote {written} rows but dataset has {len(dataset)}")
    memmap.flush()
    del memmap

    meta = {
        "files": np.array(files),
        "labels": torch.stack([torch.as_tensor(v) for v in labels]),
        "devices": np.array(devices),
        "cities": np.array(cities),
        "dataset_dir": dataset_dir,
        "length": len(dataset),
    }
    torch.save(meta, meta_path)
    logger.info("waveform cache ready: %d samples, %.2f GB",
                len(dataset), waveform_path.stat().st_size / 1e9)
    return meta


def cached(dataset, name: str, cache_dir: str, num_workers: int, dataset_dir: str):
    """Wrap `dataset` so its waveforms come from a memmap, building the cache if needed. A
    cache is reused only when it was built from the same dataset_dir with the same length,
    which catches the realistic mistakes (a changed --subset or DATASET_PATH) cheaply."""
    cache_root = Path(cache_dir)
    waveform_path = cache_root / f"{name}_waveforms.npy"
    meta_path = cache_root / f"{name}_meta.pt"

    if waveform_path.exists() and meta_path.exists():
        meta = torch.load(meta_path, weights_only=False)
        if meta.get("length") == len(dataset) and meta.get("dataset_dir") == dataset_dir:
            logger.info("reusing waveform cache %s (%d samples)", waveform_path, len(dataset))
            return CachedWaveformDataset(str(waveform_path), meta)
        logger.warning("waveform cache %s is stale (length/dataset_dir mismatch) -- rebuilding",
                       waveform_path)

    meta = _build(dataset, waveform_path, meta_path, num_workers, dataset_dir)
    return CachedWaveformDataset(str(waveform_path), meta)


def cached_training_set(dcase24, subset: int, roll, cache_dir: str, num_workers: int,
                        dataset_dir: str):
    """get_training_set() with the decode cached and the roll augmentation still live. Asks
    dcase24 for the unrolled set and re-applies RollDataset outside the cache, so the
    composition matches the uncached path exactly."""
    base = dcase24.get_training_set(subset, roll=False)
    ds = cached(base, f"train{subset}", cache_dir, num_workers, dataset_dir)
    if roll:
        ds = dcase24.RollDataset(ds, shift_range=roll)
    return ds


def cached_test_set(dcase24, cache_dir: str, num_workers: int, dataset_dir: str):
    """get_test_set() with the decode cached. No augmentation is applied to the test set."""
    return cached(dcase24.get_test_set(), "test", cache_dir, num_workers, dataset_dir)


def dataset_dir_of(dcase24) -> str:
    """The directory the patched dcase24 module was generated against."""
    return str(dcase24.dataset_dir)


def build_datasets(config, dcase24):
    """The (train, test) datasets for a run, cached or not depending on --waveform_cache.
    Shared with pruning/run_pruning.py so the two entrypoints cannot drift into building their
    data differently, which every pruned-vs-dense comparison depends on."""
    roll_samples = config.orig_sample_rate * config.roll_sec
    if not getattr(config, "waveform_cache", False):
        return dcase24.get_training_set(config.subset, roll=roll_samples), dcase24.get_test_set()

    cache_dir = getattr(config, "cache_dir", None) or DEFAULT_CACHE_DIR
    dataset_dir = dataset_dir_of(dcase24)
    train = cached_training_set(dcase24, config.subset, roll_samples, cache_dir,
                                config.num_workers, dataset_dir)
    test = cached_test_set(dcase24, cache_dir, config.num_workers, dataset_dir)
    return train, test
