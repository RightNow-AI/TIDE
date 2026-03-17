import pytest
import torch
import torch.nn as nn

from TIDE.adapters.auto import get_adapter, register_adapter, ADAPTER_REGISTRY
from TIDE.adapters.base import BaseAdapter
from TIDE.adapters.universal import UniversalAdapter


# Reuse TinyModel from test_calibrate
from tests.test_calibrate import TinyModel, TinyAdapter


class TestAdapterInterface:
    def test_get_adapter_returns_correct_type(self):
        model = TinyModel()
        adapter = get_adapter(model)
        assert isinstance(adapter, BaseAdapter)
        assert isinstance(adapter, TinyAdapter)

    def test_get_layers(self):
        model = TinyModel(num_layers=8)
        adapter = get_adapter(model)
        layers = adapter.get_layers(model)
        assert len(layers) == 8
        assert all(isinstance(l, nn.Module) for l in layers)

    def test_get_hidden_state(self):
        model = TinyModel()
        adapter = get_adapter(model)
        x = torch.randn(2, 16, 64)
        layer = adapter.get_layers(model)[0]
        out = layer(x)
        hidden = adapter.get_hidden_state(out)
        assert hidden.shape == (2, 16, 64)

    def test_get_final_norm(self):
        model = TinyModel()
        adapter = get_adapter(model)
        norm = adapter.get_final_norm(model)
        assert isinstance(norm, nn.Module)
        x = torch.randn(2, 64)
        out = norm(x)
        assert out.shape == (2, 64)

    def test_get_lm_head(self):
        model = TinyModel(vocab_size=256)
        adapter = get_adapter(model)
        head = adapter.get_lm_head(model)
        assert isinstance(head, nn.Module)
        x = torch.randn(2, 64)
        out = head(x)
        assert out.shape == (2, 256)

    def test_get_embedding(self):
        model = TinyModel(vocab_size=256)
        adapter = get_adapter(model)
        emb = adapter.get_embedding(model)
        ids = torch.tensor([[1, 2, 3]])
        out = emb(ids)
        assert out.shape == (1, 3, 64)

    def test_get_router_input_dim(self):
        model = TinyModel()
        adapter = get_adapter(model)
        assert adapter.get_router_input_dim(model) == 64

    def test_unprobeable_model_raises(self):
        """Model with no recognizable structure raises RuntimeError from UniversalAdapter."""
        class UnknownModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("C", (), {"hidden_size": None, "vocab_size": None})()
        with pytest.raises(RuntimeError, match="UniversalAdapter failed"):
            get_adapter(UnknownModel())


class TestUniversalAdapter:
    def test_probes_tinymodel_structure(self):
        """UniversalAdapter should probe TinyModel correctly (LLaMA-like paths)."""
        # Remove TinyModel from registry temporarily to force universal fallback
        saved = ADAPTER_REGISTRY.pop("TinyModel", None)
        try:
            model = TinyModel(num_layers=8, vocab_size=256)
            adapter = get_adapter(model)
            assert isinstance(adapter, UniversalAdapter)

            layers = adapter.get_layers(model)
            assert len(layers) == 8

            norm = adapter.get_final_norm(model)
            assert norm is model.model.norm

            head = adapter.get_lm_head(model)
            assert head is model.lm_head

            emb = adapter.get_embedding(model)
            assert emb is model.model.embed_tokens

            assert adapter.get_router_input_dim(model) == 64
        finally:
            if saved is not None:
                ADAPTER_REGISTRY["TinyModel"] = saved

    def test_register_adapter(self):
        """User-defined adapters take priority over UniversalAdapter."""
        class CustomModel(nn.Module):
            pass

        class CustomAdapter(BaseAdapter):
            def get_layers(self, model): return []
            def get_hidden_state(self, layer_output): return layer_output[0]
            def get_final_norm(self, model): return nn.LayerNorm(1)
            def get_lm_head(self, model): return nn.Linear(1, 1)
            def get_router_input_dim(self, model): return 1
            def get_embedding(self, model): return nn.Embedding(1, 1)

        register_adapter("CustomModel", CustomAdapter)
        try:
            adapter = get_adapter(CustomModel())
            assert isinstance(adapter, CustomAdapter)
        finally:
            ADAPTER_REGISTRY.pop("CustomModel", None)
