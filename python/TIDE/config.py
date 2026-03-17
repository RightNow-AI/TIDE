from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


@dataclass
class TIDEConfig:
    checkpoint_interval: int = 4
    exit_threshold: float = 0.85
    min_layers: int = 8
    compaction_threshold: float = 0.25
    exit_strategy: str = "identity"
    calibration_samples: int = 2000
    calibration_dataset: str = "wikitext"
    router_bottleneck_dim: int = 128
    convergence_threshold: float = 0.98
    kv_cache_strategy: str = "zero_pad"  # "zero_pad" or future "propagate"
    profile: bool = False

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def load(cls, path: str | Path) -> TIDEConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
