from typing import Any

import wandb


def setup_wandb(
    project_name: str,
    run_name: str,
    config: dict[str, Any],
    run_id: str | None = None,
    resume: str | None = None,
) -> wandb.sdk.wandb_run.Run:
    # disable W&B's automatic system-resource monitoring (CPU/GPU/disk/network,
    # ~20+ auto-generated charts). The handful of metrics that matter are logged
    # explicitly in log_epoch_metrics instead.
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        id=run_id,
        resume=resume,
        settings=wandb.Settings(x_disable_stats=True),
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("system/*", step_metric="epoch")
    wandb.define_metric("time/*", step_metric="epoch")
    return run
