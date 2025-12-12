"""Lightweight, optional Weights & Biases helper."""

from __future__ import annotations

from typing import Any, Dict, Optional


def init_wandb(enable: bool, project: str, run_name: str | None, config: Dict[str, Any]) -> Optional[Any]:
    """Initialize a wandb run if enabled and the package is available."""
    if not enable:
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        print("wandb is not installed; disable logging or install wandb to enable it.")
        return None

    return wandb.init(project=project, name=run_name, config=config)
