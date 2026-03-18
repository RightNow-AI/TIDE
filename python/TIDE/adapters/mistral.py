from __future__ import annotations

import torch
import torch.nn as nn

from TIDE.adapters.base import BaseAdapter


class MistralAdapter(BaseAdapter):
    """Adapter for MistralForCausalLM."""

    def get_layers(self, model: nn.Module) -> list[nn.Module]:
        return list(model.model.layers)

    def get_hidden_state(self, layer_output) -> torch.Tensor:
        return layer_output[0]

    def get_final_norm(self, model: nn.Module) -> nn.Module:
        return model.model.norm

    def get_lm_head(self, model: nn.Module) -> nn.Module:
        return model.lm_head

    def get_router_input_dim(self, model: nn.Module) -> int:
        return model.config.hidden_size

    def get_embedding(self, model: nn.Module) -> nn.Module:
        return model.model.embed_tokens

    def get_position_embeddings(
        self, model: nn.Module, position_ids: torch.Tensor, hidden: torch.Tensor
    ):
        rotary_emb = getattr(model.model, "rotary_emb", None)
        if rotary_emb is None:
            return None
        return rotary_emb(hidden, position_ids)
