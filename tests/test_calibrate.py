import pytest
import torch
import torch.nn as nn

from TIDE.calibrate import (
    collect_hidden_states,
    compute_convergence_labels,
    train_routers,
    calibrate,
)
from TIDE.config import TIDEConfig
from TIDE.adapters.auto import ADAPTER_REGISTRY
from TIDE.adapters.base import BaseAdapter


# --- Tiny model for testing ---

class TinyRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x


class TinyDecoderLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = TinyRMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x, attention_mask=None):
        return (x + self.mlp(self.norm(x)),)


class TinyConfig:
    hidden_size = 64
    vocab_size = 256


class _TinyCache:
    """Minimal KV cache stub for TinyModel generation tests."""
    def __init__(self, n_layers):
        self.key_cache = [torch.empty(0) for _ in range(n_layers)]
        self.value_cache = [torch.empty(0) for _ in range(n_layers)]


class _TinyOutput:
    """Mimics HF model output with hidden_states and past_key_values."""
    def __init__(self, logits, hidden_states=None, past_key_values=None):
        self.logits = logits
        self.hidden_states = hidden_states
        self.past_key_values = past_key_values


class TinyModel(nn.Module):
    def __init__(self, num_layers=8, dim=64, vocab_size=256):
        super().__init__()
        self.config = TinyConfig()
        self.config.vocab_size = vocab_size
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, dim)
        self.model.layers = nn.ModuleList([TinyDecoderLayer(dim) for _ in range(num_layers)])
        self.model.norm = TinyRMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self._n_layers = num_layers

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False,
                return_dict=False, use_cache=False, past_key_values=None, **kwargs):
        x = self.model.embed_tokens(input_ids)
        hidden_states = [x] if output_hidden_states else None

        for layer in self.model.layers:
            x = layer(x)[0]
            if output_hidden_states:
                hidden_states.append(x)

        logits = self.lm_head(self.model.norm(x))

        cache = past_key_values if past_key_values is not None else _TinyCache(self._n_layers)

        return _TinyOutput(
            logits=logits,
            hidden_states=tuple(hidden_states) if hidden_states else None,
            past_key_values=cache if use_cache else None,
        )

    def __class_getitem__(cls, item):
        return cls


class TinyAdapter(BaseAdapter):
    def get_layers(self, model):
        return list(model.model.layers)

    def get_hidden_state(self, layer_output):
        return layer_output[0]

    def get_final_norm(self, model):
        return model.model.norm

    def get_lm_head(self, model):
        return model.lm_head

    def get_router_input_dim(self, model):
        return model.config.hidden_size

    def get_embedding(self, model):
        return model.model.embed_tokens


# Register tiny model adapter
ADAPTER_REGISTRY["TinyModel"] = TinyAdapter


class TinyEncoding(dict):
    """Dict-like encoding that supports .to(device), .input_ids, .attention_mask."""
    def __init__(self, input_ids, attention_mask):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, device):
        return self


class TinyTokenizer:
    """Minimal tokenizer for testing."""
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size

    def __call__(self, texts, return_tensors="pt", padding=True,
                 truncation=True, max_length=512):
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        seq_len = min(32, max_length)
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        return TinyEncoding(input_ids, attention_mask)


@pytest.fixture
def tiny_setup():
    model = TinyModel(num_layers=8, dim=64, vocab_size=256)
    tokenizer = TinyTokenizer(vocab_size=256)
    config = TIDEConfig(
        checkpoint_interval=4,
        calibration_samples=16,
        convergence_threshold=0.95,
        router_bottleneck_dim=32,
        min_layers=0,
    )
    return model, tokenizer, config


class TestCollectHiddenStates:
    def test_shapes(self, tiny_setup):
        model, tokenizer, config = tiny_setup
        hs = collect_hidden_states(model, tokenizer, "dummy", config)
        assert "final" in hs
        # Should have checkpoints at layers 3, 7
        assert 3 in hs
        assert 7 in hs
        # All should have same number of tokens
        n_tokens = hs["final"].shape[0]
        assert hs[3].shape == (n_tokens, 64)
        assert hs[7].shape == (n_tokens, 64)


class TestConvergenceLabels:
    def test_binary_labels(self, tiny_setup):
        model, tokenizer, config = tiny_setup
        hs = collect_hidden_states(model, tokenizer, "dummy", config)
        labels = compute_convergence_labels(hs, config)
        for layer_idx, lab in labels.items():
            assert set(lab.unique().tolist()).issubset({0.0, 1.0})


class TestTrainRouters:
    def test_loss_decreases(self, tiny_setup):
        model, tokenizer, config = tiny_setup
        hs = collect_hidden_states(model, tokenizer, "dummy", config)
        labels = compute_convergence_labels(hs, config)
        routers = train_routers(hs, labels, config, epochs=50, device="cpu")
        assert len(routers) == len(labels)
        for layer_idx, router in routers.items():
            out = router(torch.randn(4, 64))
            assert out.shape == (4,)
            assert (out >= 0).all() and (out <= 1).all()


class TestFullPipeline:
    def test_calibrate_save_load(self, tiny_setup, tmp_path):
        model, tokenizer, config = tiny_setup
        save_path = str(tmp_path / "router.pt")
        checkpoint = calibrate(model, tokenizer, config=config, save_path=save_path)
        assert len(checkpoint.routers) > 0
        assert checkpoint.hidden_dim == 64

        from TIDE.router import RouterCheckpoint
        loaded = RouterCheckpoint.load(save_path)
        assert loaded.hidden_dim == checkpoint.hidden_dim
        assert set(loaded.routers.keys()) == set(checkpoint.routers.keys())
