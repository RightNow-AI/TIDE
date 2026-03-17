import pytest
import torch

from TIDE.router import RouterCheckpoint, TokenRouter


class TestTokenRouter:
    def test_output_shape(self):
        router = TokenRouter(hidden_dim=256, bottleneck_dim=64)
        x = torch.randn(8, 256)
        out = router(x)
        assert out.shape == (8,)

    def test_output_range(self):
        router = TokenRouter(hidden_dim=256, bottleneck_dim=64)
        x = torch.randn(32, 256)
        out = router(x)
        assert (out >= 0).all() and (out <= 1).all()

    def test_single_token(self):
        router = TokenRouter(hidden_dim=128, bottleneck_dim=32)
        x = torch.randn(1, 128)
        out = router(x)
        assert out.shape == (1,)

    def test_gradient_flow(self):
        router = TokenRouter(hidden_dim=128, bottleneck_dim=32)
        x = torch.randn(4, 128, requires_grad=True)
        out = router(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 128)

    def test_different_bottleneck_dims(self):
        for bd in [32, 64, 128, 256]:
            router = TokenRouter(hidden_dim=512, bottleneck_dim=bd)
            x = torch.randn(4, 512)
            out = router(x)
            assert out.shape == (4,)


class TestRouterCheckpoint:
    def test_save_load_roundtrip(self, tmp_path):
        hidden_dim = 128
        bottleneck_dim = 32
        routers = {
            3: TokenRouter(hidden_dim, bottleneck_dim),
            7: TokenRouter(hidden_dim, bottleneck_dim),
            11: TokenRouter(hidden_dim, bottleneck_dim),
        }
        checkpoint = RouterCheckpoint(
            routers=routers,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        )

        path = tmp_path / "router.pt"
        checkpoint.save(path)

        loaded = RouterCheckpoint.load(path)
        assert loaded.hidden_dim == hidden_dim
        assert loaded.bottleneck_dim == bottleneck_dim
        assert set(loaded.routers.keys()) == {3, 7, 11}

        # Check weights match
        x = torch.randn(4, hidden_dim)
        for layer_idx in routers:
            orig_out = routers[layer_idx](x)
            loaded_out = loaded.routers[layer_idx](x)
            assert torch.allclose(orig_out, loaded_out, atol=1e-6)
