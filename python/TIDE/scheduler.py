from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from TIDE.config import TIDEConfig


@dataclass
class ExitStats:
    """Tracks per-layer exit statistics."""
    total_tokens: int = 0
    exits_per_layer: Dict[int, int] = field(default_factory=dict)
    remaining_tokens: int = 0

    @property
    def total_exited(self) -> int:
        return sum(self.exits_per_layer.values())

    @property
    def exit_rate(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.total_exited / self.total_tokens

    def summary(self) -> str:
        lines = [f"Total tokens: {self.total_tokens}, Exited: {self.total_exited} ({self.exit_rate:.1%})"]
        for layer_idx in sorted(self.exits_per_layer):
            count = self.exits_per_layer[layer_idx]
            pct = count / self.total_tokens * 100 if self.total_tokens else 0
            lines.append(f"  Layer {layer_idx}: {count} exits ({pct:.1f}%)")
        lines.append(f"  Ran all layers: {self.remaining_tokens}")
        return "\n".join(lines)


class SkipScheduler:
    """Tracks active/exited tokens and manages batch compaction."""

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        config: TIDEConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.total_tokens = batch_size * seq_len
        self.hidden_dim = hidden_dim
        self.config = config
        self.device = device
        self.dtype = dtype

        self.output_buffer = torch.zeros(
            self.total_tokens, hidden_dim, device=device, dtype=dtype
        )
        self.active_positions = torch.arange(self.total_tokens, device=device)
        self.stats = ExitStats(total_tokens=self.total_tokens)

    def process_exits(
        self,
        hidden_states: torch.Tensor,
        exit_mask: torch.Tensor,
        layer_idx: int,
        final_norm: nn.Module,
    ) -> torch.Tensor:
        """Process token exits: scatter exited tokens to output, compact remaining.

        Args:
            hidden_states: [N_active, D] current hidden states
            exit_mask: [N_active] boolean mask, True = exit
            layer_idx: current layer index (for stats)
            final_norm: final normalization to apply to exited tokens

        Returns:
            Compacted hidden states for remaining tokens [N_remaining, D]
        """
        n_exit = exit_mask.sum().item()

        if n_exit == 0:
            return hidden_states

        self.stats.exits_per_layer[layer_idx] = n_exit

        # Extract and process exited tokens
        exited_hidden = hidden_states[exit_mask]
        exited_positions = self.active_positions[exit_mask]

        # Apply final norm and scatter to output buffer
        with torch.no_grad():
            normed = final_norm(exited_hidden)
        self.output_buffer[exited_positions] = normed.to(self.dtype)

        # Compact remaining
        keep_mask = ~exit_mask
        hidden_states = hidden_states[keep_mask]
        self.active_positions = self.active_positions[keep_mask]

        return hidden_states

    def should_compact(self) -> bool:
        """Check if remaining batch is above compaction_threshold."""
        if self.total_tokens <= 1:
            return False
        remaining_frac = self.active_positions.shape[0] / self.total_tokens
        return remaining_frac >= self.config.compaction_threshold

    def finalize(self, hidden_states: torch.Tensor, final_norm: nn.Module) -> None:
        """Process tokens that survived all layers."""
        if hidden_states.shape[0] > 0:
            self.stats.remaining_tokens = hidden_states.shape[0]
            normed = final_norm(hidden_states)
            self.output_buffer[self.active_positions] = normed.to(self.dtype)

    @property
    def n_active(self) -> int:
        return self.active_positions.shape[0]
