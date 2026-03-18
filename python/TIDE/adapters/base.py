from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseAdapter(ABC):
    """Abstract adapter for extracting components from different transformer architectures."""

    @abstractmethod
    def get_layers(self, model: nn.Module) -> list[nn.Module]:
        """Return the list of transformer decoder layers."""
        ...

    @abstractmethod
    def get_hidden_state(self, layer_output) -> torch.Tensor:
        """Extract hidden state tensor from a layer's output tuple."""
        ...

    @abstractmethod
    def get_final_norm(self, model: nn.Module) -> nn.Module:
        """Return the final LayerNorm/RMSNorm before the LM head."""
        ...

    @abstractmethod
    def get_lm_head(self, model: nn.Module) -> nn.Module:
        """Return the language model head (linear projection to vocab)."""
        ...

    @abstractmethod
    def get_router_input_dim(self, model: nn.Module) -> int:
        """Return the hidden dimension used as router input."""
        ...

    @abstractmethod
    def get_embedding(self, model: nn.Module) -> nn.Module:
        """Return the token embedding module."""
        ...

    def get_router_extra_features(self, layer_output) -> Optional[torch.Tensor]:
        """Optional extra features to concat with hidden state for routing."""
        return None

    def get_position_embeddings(
        self, model: nn.Module, position_ids: torch.Tensor, hidden: torch.Tensor
    ) -> Optional[tuple]:
        """Compute (cos, sin) position embeddings if needed by the architecture."""
        return None
