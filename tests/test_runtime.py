import pytest
import torch
import torch.nn as nn

from TIDE.config import TIDEConfig
from TIDE.router import RouterCheckpoint, TokenRouter
from TIDE.runtime import TIDERuntime
from TIDE.scheduler import ExitStats

# Reuse TinyModel
from tests.test_calibrate import TinyModel, TinyAdapter, ADAPTER_REGISTRY


@pytest.fixture
def tiny_runtime(tmp_path):
    """Create a TIDERuntime with a tiny model and random routers."""
    model = TinyModel(num_layers=8, dim=64, vocab_size=256)
    hidden_dim = 64
    bottleneck_dim = 32

    # Create routers at checkpoint layers (3, 7)
    routers = {
        3: TokenRouter(hidden_dim, bottleneck_dim),
        7: TokenRouter(hidden_dim, bottleneck_dim),
    }

    checkpoint = RouterCheckpoint(
        routers=routers, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim
    )
    router_path = str(tmp_path / "router.pt")
    checkpoint.save(router_path)

    config = TIDEConfig(
        checkpoint_interval=4,
        exit_threshold=0.5,
        min_layers=0,
        router_bottleneck_dim=bottleneck_dim,
    )

    runtime = TIDERuntime(model, router_path, config=config, use_cuda_kernels=False)
    return runtime, model


class TestTIDERuntimeOutput:
    def test_output_shape(self, tiny_runtime):
        runtime, model = tiny_runtime
        input_ids = torch.randint(0, 256, (2, 16))
        logits = runtime(input_ids)
        assert logits.shape == (2, 16, 256)

    def test_batch_size_1(self, tiny_runtime):
        runtime, model = tiny_runtime
        input_ids = torch.randint(0, 256, (1, 8))
        logits = runtime(input_ids)
        assert logits.shape == (1, 8, 256)


class TestTIDERuntimeThresholds:
    def test_threshold_1_no_exits(self, tmp_path):
        """threshold=1.0 means no tokens exit early."""
        model = TinyModel(num_layers=8, dim=64, vocab_size=256)
        routers = {3: TokenRouter(64, 32), 7: TokenRouter(64, 32)}
        checkpoint = RouterCheckpoint(routers=routers, hidden_dim=64, bottleneck_dim=32)
        router_path = str(tmp_path / "router.pt")
        checkpoint.save(router_path)

        config = TIDEConfig(
            checkpoint_interval=4, exit_threshold=1.0, min_layers=0
        )
        runtime = TIDERuntime(model, router_path, config=config, use_cuda_kernels=False)

        input_ids = torch.randint(0, 256, (2, 16))
        logits = runtime(input_ids)
        assert logits.shape == (2, 16, 256)
        # With threshold=1.0, no token should exit
        assert runtime.last_stats.total_exited == 0

    def test_threshold_0_all_exit(self, tmp_path):
        """threshold=0.0 means all tokens exit at first checkpoint."""
        model = TinyModel(num_layers=8, dim=64, vocab_size=256)
        routers = {3: TokenRouter(64, 32), 7: TokenRouter(64, 32)}
        checkpoint = RouterCheckpoint(routers=routers, hidden_dim=64, bottleneck_dim=32)
        router_path = str(tmp_path / "router.pt")
        checkpoint.save(router_path)

        config = TIDEConfig(
            checkpoint_interval=4, exit_threshold=0.0, min_layers=0
        )
        runtime = TIDERuntime(model, router_path, config=config, use_cuda_kernels=False)

        input_ids = torch.randint(0, 256, (2, 16))
        logits = runtime(input_ids)
        assert logits.shape == (2, 16, 256)
        # With threshold=0.0, all tokens should exit at layer 3
        assert runtime.last_stats.total_exited > 0


class TestTIDERuntimeStats:
    def test_stats_tracking(self, tiny_runtime):
        runtime, model = tiny_runtime
        input_ids = torch.randint(0, 256, (2, 16))
        runtime(input_ids)

        stats = runtime.last_stats
        assert stats is not None
        assert stats.total_tokens == 32  # 2 * 16
        assert stats.total_exited + stats.remaining_tokens == stats.total_tokens


class TestTIDERuntimeGenerate:
    def test_generate_produces_tokens(self, tiny_runtime):
        runtime, model = tiny_runtime
        input_ids = torch.randint(0, 256, (1, 4))
        output = runtime.generate(input_ids, max_new_tokens=5, temperature=0)
        assert output.shape[0] == 1
        assert output.shape[1] >= 5  # At least input + some generated
