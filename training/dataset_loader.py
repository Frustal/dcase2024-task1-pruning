"""Import baseline.dataset.dcase24 with a configurable dataset directory.

dcase24.py hardcodes `dataset_dir = None` and asserts on it at import time, and baseline/ is
vendored unmodified, so this writes a patched copy of its source to
training/_generated_dcase24.py and imports that. The copy needs a real file on disk rather
than a sys.modules entry: Windows DataLoader workers use the 'spawn' start method, so each
worker is a fresh interpreter that imports the module by name to unpickle the dataset objects,
and a dotted name with no backing file raises ModuleNotFoundError there.

split_path is also pointed at the vendored split CSVs. dcase24.py downloads a split only when
it is missing locally, so lookup becomes local-first with the original URL as fallback.
"""
import importlib
import sys
from pathlib import Path
from types import ModuleType

_BASELINE_DIR = Path(__file__).resolve().parents[1] / "baseline"
_DCASE24_PATH = _BASELINE_DIR / "dataset" / "dcase24.py"
_VENDORED_SPLITS_DIR = _BASELINE_DIR / "dataset" / "splits"
_PLACEHOLDER = "dataset_dir = None"

_GENERATED_MODULE_NAME = "training._generated_dcase24"
_GENERATED_PATH = Path(__file__).resolve().parent / "_generated_dcase24.py"


def load_dcase24(dataset_dir: str, splits_dir: Path = _VENDORED_SPLITS_DIR) -> ModuleType:
    """Write the patched dcase24 module for `dataset_dir` and import it."""
    source = _DCASE24_PATH.read_text()
    if _PLACEHOLDER not in source:
        raise RuntimeError(f"Expected '{_PLACEHOLDER}' in {_DCASE24_PATH}; baseline file may have changed")
    patched_source = source.replace(_PLACEHOLDER, f"dataset_dir = {dataset_dir!r}", 1)
    _GENERATED_PATH.write_text(patched_source)

    sys.modules.pop(_GENERATED_MODULE_NAME, None)
    module = importlib.import_module(_GENERATED_MODULE_NAME)

    module.dataset_config['split_path'] = str(splits_dir)
    return module
