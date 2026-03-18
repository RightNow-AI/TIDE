from __future__ import annotations

import logging

import torch.nn as nn

from TIDE.adapters.base import BaseAdapter
from TIDE.adapters.llama import LlamaAdapter
from TIDE.adapters.mistral import MistralAdapter
from TIDE.adapters.qwen import QwenAdapter

logger = logging.getLogger("TIDE")

ADAPTER_REGISTRY: dict[str, type[BaseAdapter]] = {
    "LlamaForCausalLM": LlamaAdapter,
    "MistralForCausalLM": MistralAdapter,
    "Qwen2ForCausalLM": QwenAdapter,
}


def register_adapter(class_name: str, adapter_cls: type[BaseAdapter]) -> None:
    """Register a user-defined adapter for a model class name."""
    ADAPTER_REGISTRY[class_name] = adapter_cls


def get_adapter(model: nn.Module) -> BaseAdapter:
    """Two-tier adapter resolution.

    1. Check ADAPTER_REGISTRY for known model class names (fast, exact match)
    2. Fall back to UniversalAdapter which probes the model structure
    """
    class_name = model.__class__.__name__

    # Tier 1: hardcoded registry
    if class_name in ADAPTER_REGISTRY:
        return ADAPTER_REGISTRY[class_name]()

    # Tier 2: universal probing
    logger.info(
        f"No registered adapter for '{class_name}', trying UniversalAdapter"
    )
    from TIDE.adapters.universal import UniversalAdapter
    return UniversalAdapter.probe(model)
