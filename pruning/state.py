"""Crash-resume state for a pruning run.

A pruning job is long: IMP takes ~8 h in one process, a SNIP curve a full 150-epoch training per
sparsity level. Each round is written to disk as it completes, so this JSON file is the
authoritative record of every completed level and what makes an interrupted run resumable.
Rounds are tracked explicitly rather than inferred from the sparsityNN/ directories, since a
round that crashed midway leaves one of those too. The mask is not stored: pruned weights are
exactly 0.0, so it is rebuilt losslessly on resume (MaskRegistry.from_weights).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class RoundResult:
    target: float          # requested sparsity for this round
    sparsity: float        # sparsity actually achieved
    nonzero_params: int
    size_kb: float
    test_macro_accuracy: float
    epoch_end: int         # the run's global epoch counter after this round finished


@dataclass
class PruningState:
    method: str
    run_id: str
    completed: list[RoundResult] = field(default_factory=list)

    @property
    def path(self) -> str:
        return os.path.join("checkpoints", self.run_id, "pruning_state.json")

    @property
    def completed_targets(self) -> set[float]:
        return {r.target for r in self.completed}

    def is_completed(self, target: float, tol: float = 1e-6) -> bool:
        """Whether `target` has already run, compared with a tolerance rather than by membership
        in `completed_targets`: targets are floats typed by hand into configs, and exact equality
        would re-run a finished level over a last-digit difference."""
        return any(abs(r.target - target) < tol for r in self.completed)

    @property
    def epoch(self) -> int:
        """Global epoch counter to resume from. wandb.log pins step=epoch and drops anything not
        monotonically increasing, so a resumed run continues the axis rather than restart at 0."""
        return self.completed[-1].epoch_end if self.completed else 0

    def last_checkpoint_dir(self) -> str | None:
        """Checkpoint dir of the most recent completed round, where IMP picks the weights up."""
        if not self.completed:
            return None
        return round_checkpoint_dir(self.run_id, self.completed[-1].target)

    def record(self, result: RoundResult) -> None:
        self.completed.append(result)
        self.save()

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # write-then-rename, so a crash mid-write cannot leave a truncated, unparseable file
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"method": self.method, "run_id": self.run_id,
                       "completed": [asdict(r) for r in self.completed]}, f, indent=2)
        os.replace(tmp, self.path)

    @classmethod
    def load(cls, run_id: str) -> "PruningState | None":
        path = os.path.join("checkpoints", run_id, "pruning_state.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            raw = json.load(f)
        return cls(method=raw["method"], run_id=raw["run_id"],
                   completed=[RoundResult(**r) for r in raw["completed"]])


def round_checkpoint_dir(run_id: str, target: float) -> str:
    return os.path.join("checkpoints", run_id, f"sparsity{round(100 * target):02d}")
