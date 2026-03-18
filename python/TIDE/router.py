from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn


class TokenRouter(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck_dim: int = 128):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, 1, bias=False)
        self.act = nn.SiLU()

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.up(self.act(self.down(hidden_state)))).squeeze(-1)


@dataclass
class RouterCheckpoint:
    routers: Dict[int, TokenRouter]
    hidden_dim: int
    bottleneck_dim: int

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "hidden_dim": self.hidden_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "router_layers": {},
        }
        for layer_idx, router in self.routers.items():
            state["router_layers"][layer_idx] = router.state_dict()
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> RouterCheckpoint:
        state = torch.load(path, map_location=device, weights_only=True)
        hidden_dim = state["hidden_dim"]
        bottleneck_dim = state["bottleneck_dim"]
        routers = {}
        for layer_idx_str, router_state in state["router_layers"].items():
            layer_idx = int(layer_idx_str) if isinstance(layer_idx_str, str) else layer_idx_str
            router = TokenRouter(hidden_dim, bottleneck_dim)
            router.load_state_dict(router_state)
            router.to(device)
            routers[layer_idx] = router
        return cls(routers=routers, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)
