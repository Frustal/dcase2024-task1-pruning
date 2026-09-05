"""Windows-safe replacement for baseline.helpers.seed.worker_init_fn.

Upstream's spawn_get() computes `int(2 ** (32 * shift) * s)` with `s` a numpy.uint32; numpy
converts the Python int operand via the platform C `long`, which is 32-bit on Windows, so
`2 ** 32` overflows. baseline/ is vendored unmodified, so the same seeding is reproduced here
with `s` cast to a plain Python int, keeping the arithmetic arbitrary-precision everywhere.
"""
import random

import numpy as np
import torch


def worker_init_fn(wid):
    seed_sequence = np.random.SeedSequence([torch.initial_seed(), wid])

    to_seed = _spawn_get_int(seed_sequence, 2)
    torch.random.manual_seed(to_seed)

    np_seed = _spawn_get_ndarray(seed_sequence, 2)
    np.random.seed(np_seed)

    py_seed = _spawn_get_int(seed_sequence, 2)
    random.seed(py_seed)


def _spawn_get_ndarray(seedseq, n_entropy):
    child = seedseq.spawn(1)[0]
    return child.generate_state(n_entropy, dtype=np.uint32)


def _spawn_get_int(seedseq, n_entropy):
    state = _spawn_get_ndarray(seedseq, n_entropy)
    state_as_int = 0
    for shift, s in enumerate(state):
        state_as_int = state_as_int + (2 ** (32 * shift)) * int(s)
    return state_as_int
