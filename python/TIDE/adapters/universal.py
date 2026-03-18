from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from TIDE.adapters.base import BaseAdapter

logger = logging.getLogger("TIDE")


class UniversalAdapter(BaseAdapter):
    """Adapter that probes any HuggingFace model to extract transformer components.

    Discovers layers, norms, embeddings, and LM head by traversing known attribute
    paths across architectures (LLaMA, GPT-2, GPT-NeoX, Phi, Falcon, etc.).
    Falls back to heuristic search (largest ModuleList, Linear matching vocab_size).
    """

    # Ordered probe paths for each component
    _LAYER_PATHS = [
        "model.layers",           # LLaMA, Mistral, Qwen, Gemma
        "transformer.h",          # GPT-2, GPT-J
        "transformer.layers",     # Falcon, some custom
        "gpt_neox.layers",        # GPT-NeoX, Pythia
        "model.decoder.layers",   # OPT
    ]

    _NORM_PATHS = [
        "model.norm",             # LLaMA, Mistral, Qwen, Gemma
        "transformer.ln_f",       # GPT-2, GPT-J
        "transformer.final_layernorm",  # GPT-NeoX
        "model.decoder.final_layer_norm",  # OPT
        "model.final_layernorm",  # Falcon
    ]

    _LM_HEAD_PATHS = [
        "lm_head",                # Most models
        "output",                 # Some custom
    ]

    _EMBEDDING_PATHS = [
        "model.embed_tokens",     # LLaMA, Mistral, Qwen, Gemma
        "transformer.wte",        # GPT-2, GPT-J
        "transformer.word_embeddings",  # Falcon
        "gpt_neox.embed_in",      # GPT-NeoX, Pythia
        "model.decoder.embed_tokens",  # OPT
    ]

    def __init__(self):
        super().__init__()
        # Cached probe results (set on first call per model)
        self._cached_layers: Optional[list[nn.Module]] = None
        self._cached_norm: Optional[nn.Module] = None
        self._cached_lm_head: Optional[nn.Module] = None
        self._cached_embedding: Optional[nn.Module] = None
        self._cached_hidden_dim: Optional[int] = None
        self._probed_model_id: Optional[int] = None

    def _ensure_probed(self, model: nn.Module):
        """Run probing once per model instance."""
        model_id = id(model)
        if self._probed_model_id == model_id:
            return
        self._probe(model)
        self._probed_model_id = model_id

    def _probe(self, model: nn.Module):
        """Probe model structure and cache all component references."""
        diag = []  # Diagnostic info for error reporting

        # --- Layers ---
        self._cached_layers = self._probe_layers(model, diag)

        # --- Final norm ---
        self._cached_norm = self._probe_by_paths(model, self._NORM_PATHS, "final_norm")
        if self._cached_norm is None:
            self._cached_norm = self._find_sibling_norm(model, diag)
        if self._cached_norm is None:
            diag.append("final_norm: could not find via paths or sibling search")

        # --- LM head ---
        self._cached_lm_head = self._probe_by_paths(model, self._LM_HEAD_PATHS, "lm_head")
        if self._cached_lm_head is None:
            self._cached_lm_head = self._find_lm_head_by_shape(model)
        if self._cached_lm_head is None:
            diag.append("lm_head: could not find via paths or shape search")

        # --- Embedding ---
        self._cached_embedding = self._probe_by_paths(
            model, self._EMBEDDING_PATHS, "embedding"
        )
        if self._cached_embedding is None:
            self._cached_embedding = self._find_embedding_by_shape(model)
        if self._cached_embedding is None:
            diag.append("embedding: could not find via paths or shape search")

        # --- Hidden dim ---
        self._cached_hidden_dim = getattr(model.config, "hidden_size", None)
        if self._cached_hidden_dim is None:
            diag.append("hidden_dim: model.config.hidden_size not found")

        # Validate critical components
        missing = []
        if self._cached_layers is None:
            missing.append("layers")
        if self._cached_norm is None:
            missing.append("final_norm")
        if self._cached_lm_head is None:
            missing.append("lm_head")
        if self._cached_hidden_dim is None:
            missing.append("hidden_dim")

        if missing:
            raise RuntimeError(
                f"UniversalAdapter failed to probe model {model.__class__.__name__}. "
                f"Missing components: {missing}. "
                f"Diagnostics:\n" + "\n".join(f"  - {d}" for d in diag)
            )

        logger.info(
            f"UniversalAdapter probed {model.__class__.__name__}: "
            f"{len(self._cached_layers)} layers, hidden_dim={self._cached_hidden_dim}"
        )

    def _probe_layers(self, model: nn.Module, diag: list) -> Optional[list[nn.Module]]:
        """Find transformer decoder layers."""
        for path in self._LAYER_PATHS:
            obj = _getattr_path(model, path)
            if obj is not None and isinstance(obj, nn.ModuleList) and len(obj) > 0:
                diag.append(f"layers: found at '{path}' ({len(obj)} layers)")
                return list(obj)

        # Fallback: find the largest ModuleList
        best = None
        best_path = None
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and (best is None or len(module) > len(best)):
                best = module
                best_path = name

        if best is not None and len(best) >= 2:
            diag.append(f"layers: fallback to largest ModuleList at '{best_path}' ({len(best)})")
            return list(best)

        return None

    def _probe_by_paths(
        self, model: nn.Module, paths: list[str], label: str
    ) -> Optional[nn.Module]:
        """Try each attribute path, return first hit."""
        for path in paths:
            obj = _getattr_path(model, path)
            if obj is not None:
                return obj
        return None

    def _find_sibling_norm(self, model: nn.Module, diag: list) -> Optional[nn.Module]:
        """Find a norm layer that is a sibling of the layers container."""
        if self._cached_layers is None or len(self._cached_layers) == 0:
            return None

        # Find the parent module that contains the layers ModuleList
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                if list(module)[0] is self._cached_layers[0]:
                    # Found the container — look for sibling norm modules
                    parent_path = name.rsplit(".", 1)[0] if "." in name else ""
                    parent = _getattr_path(model, parent_path) if parent_path else model
                    if parent is None:
                        continue
                    for child_name, child in parent.named_children():
                        if _is_norm_layer(child):
                            diag.append(
                                f"final_norm: found sibling norm '{child_name}' "
                                f"of layers container '{name}'"
                            )
                            return child
        return None

    def _find_lm_head_by_shape(self, model: nn.Module) -> Optional[nn.Module]:
        """Find a Linear layer whose output dim matches vocab_size."""
        vocab_size = getattr(model.config, "vocab_size", None)
        if vocab_size is None:
            return None
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.out_features == vocab_size:
                return module
        return None

    def _find_embedding_by_shape(self, model: nn.Module) -> Optional[nn.Module]:
        """Find an Embedding layer whose num_embeddings matches vocab_size."""
        vocab_size = getattr(model.config, "vocab_size", None)
        if vocab_size is None:
            return None
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding) and module.num_embeddings == vocab_size:
                return module
        return None

    # --- BaseAdapter interface ---

    def get_layers(self, model: nn.Module) -> list[nn.Module]:
        self._ensure_probed(model)
        return self._cached_layers

    def get_hidden_state(self, layer_output) -> torch.Tensor:
        # Universal for all HuggingFace models: first element of the tuple
        return layer_output[0]

    def get_final_norm(self, model: nn.Module) -> nn.Module:
        self._ensure_probed(model)
        return self._cached_norm

    def get_lm_head(self, model: nn.Module) -> nn.Module:
        self._ensure_probed(model)
        return self._cached_lm_head

    def get_router_input_dim(self, model: nn.Module) -> int:
        self._ensure_probed(model)
        return self._cached_hidden_dim

    def get_embedding(self, model: nn.Module) -> nn.Module:
        self._ensure_probed(model)
        return self._cached_embedding

    def get_position_embeddings(
        self, model: nn.Module, position_ids: torch.Tensor, hidden: torch.Tensor
    ) -> Optional[tuple]:
        """Try to find rotary_emb on the model's inner container."""
        for attr in ("model", "transformer", "gpt_neox"):
            container = getattr(model, attr, None)
            if container is not None:
                rotary = getattr(container, "rotary_emb", None)
                if rotary is not None:
                    return rotary(hidden, position_ids)
        return None

    @classmethod
    def probe(cls, model: nn.Module) -> UniversalAdapter:
        """Convenience: create adapter and eagerly probe."""
        adapter = cls()
        adapter._probe(model)
        adapter._probed_model_id = id(model)
        return adapter


# --- Helpers ---

def _getattr_path(obj, path: str):
    """Resolve a dotted attribute path, returning None if any step fails."""
    if not path:
        return obj
    for attr in path.split("."):
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


def _is_norm_layer(module: nn.Module) -> bool:
    """Check if a module is a normalization layer."""
    norm_types = (nn.LayerNorm, nn.RMSNorm) if hasattr(nn, "RMSNorm") else (nn.LayerNorm,)
    if isinstance(module, norm_types):
        return True
    # Check class name for custom RMSNorm implementations
    cls_name = module.__class__.__name__.lower()
    return "rmsnorm" in cls_name or "layernorm" in cls_name
