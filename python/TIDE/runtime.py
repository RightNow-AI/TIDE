from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from TIDE.adapters.auto import get_adapter
from TIDE.config import TIDEConfig
from TIDE.router import RouterCheckpoint, TokenRouter
from TIDE.scheduler import ExitStats
from TIDE.utils import setup_logging

logger = setup_logging()

# CUDA kernel loading with graceful fallback
_cuda_available = False
_C = None


def _load_cuda_kernels():
    global _cuda_available, _C
    try:
        # Try standard import first (works with editable install + build_ext --inplace)
        import TIDE._C as _C_module
        _C = _C_module
        _cuda_available = True
        logger.info("CUDA kernels loaded successfully")
    except ImportError:
        # Fallback: load the .so via torch.ops.load_library (for TORCH_LIBRARY extensions)
        try:
            import importlib.util
            import glob
            import os
            spec = importlib.util.find_spec("TIDE")
            if spec and spec.submodule_search_locations:
                for search_path in spec.submodule_search_locations:
                    so_files = glob.glob(os.path.join(search_path, "_C*.so"))
                    if so_files:
                        torch.ops.load_library(so_files[0])
                        _cuda_available = True
                        logger.info(f"CUDA kernels loaded via torch.ops.load_library")
                        return
            _cuda_available = False
            logger.info("CUDA kernels not available, using Python fallback")
        except Exception:
            _cuda_available = False
            logger.info("CUDA kernels not available, using Python fallback")


_load_cuda_kernels()


class TIDERuntime(nn.Module):
    """Main TIDE runtime: wraps a HuggingFace model with early-exit inference.

    Uses a hook-based approach: registers forward hooks on each checkpoint layer
    to evaluate routers and freeze converged tokens. Frozen tokens keep their
    hidden state from the exit point while remaining tokens continue through
    deeper layers. This preserves attention correctness.
    """

    def __init__(
        self,
        model: nn.Module,
        router_path: str | Path,
        config: Optional[TIDEConfig] = None,
        use_cuda_kernels: bool = True,
    ):
        super().__init__()
        self.model = model
        self.config = config or TIDEConfig()
        self.adapter = get_adapter(model)
        model_on_cuda = self._device.type == "cuda"
        self.use_cuda = use_cuda_kernels and _cuda_available and model_on_cuda

        # Load routers
        checkpoint = RouterCheckpoint.load(router_path, device=self._device_str)
        self.routers: Dict[int, TokenRouter] = {}
        for layer_idx, router in checkpoint.routers.items():
            router = router.to(self._device)
            router.requires_grad_(False)
            if self.use_cuda:
                router.down.weight.data = router.down.weight.data.t().contiguous()
                router.up.weight.data = router.up.weight.data.t().contiguous()
            self.routers[layer_idx] = router

        self._layers = self.adapter.get_layers(model)
        self._final_norm = self.adapter.get_final_norm(model)
        self._lm_head = self.adapter.get_lm_head(model)

        self.last_stats: Optional[ExitStats] = None
        logger.info(
            f"TIDERuntime initialized: {len(self._layers)} layers, "
            f"{len(self.routers)} routers, CUDA={'on' if self.use_cuda else 'off'}"
        )

    @property
    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def _device_str(self) -> str:
        return str(self._device)

    def _score_router(self, hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Score tokens for exit at a checkpoint layer.

        Args:
            hidden: [B, S, D] or [N, D] hidden states
        Returns:
            scores reshaped to match hidden's leading dimensions
        """
        router = self.routers[layer_idx]
        orig_shape = hidden.shape[:-1]  # [B, S] or [N]
        flat = hidden.reshape(-1, hidden.shape[-1])

        if self.use_cuda and flat.is_cuda:
            # CUDA path: fused RMSNorm + router in one kernel
            # Weights are pre-transposed for the kernel
            norm_weight = self._final_norm.weight
            scores = torch.ops.tide.fused_layernorm_route(
                flat, norm_weight, router.down.weight, router.up.weight, 1e-6
            )
        else:
            # Python fallback
            normed = self._final_norm(flat.float())
            scores = router(normed)

        return scores.view(orig_shape)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with per-token early exit via frozen-token approach.

        Tokens that converge at a checkpoint get their normed hidden state saved.
        They continue through the model (for attention) but their contribution to
        the final output comes from the exit point, not the last layer.
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Run full model forward, collecting all layer hidden states
        model_output = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # hidden_states: [embedding, layer_0, layer_1, ..., layer_N]
        all_hidden = model_output.hidden_states
        D = all_hidden[-1].shape[-1]
        dtype = all_hidden[-1].dtype

        output_buffer = torch.zeros(B, S, D, device=device, dtype=dtype)
        exited_mask = torch.zeros(B, S, dtype=torch.bool, device=device)
        stats = ExitStats(total_tokens=B * S)

        # Check routers at each checkpoint layer (post-hoc, using saved hidden states)
        for layer_idx in sorted(self.routers.keys()):
            if layer_idx < self.config.min_layers:
                continue

            hidden = all_hidden[layer_idx + 1]  # +1 because index 0 is embeddings
            scores = self._score_router(hidden, layer_idx)  # [B, S]
            new_exits = (scores > self.config.exit_threshold) & (~exited_mask)

            if new_exits.any():
                normed = self._final_norm(hidden.float()).to(dtype)
                output_buffer[new_exits] = normed[new_exits]
                exited_mask = exited_mask | new_exits
                stats.exits_per_layer[layer_idx] = new_exits.sum().item()

        # Remaining tokens use the final hidden state
        remaining_mask = ~exited_mask
        if remaining_mask.any():
            final_hidden = all_hidden[-1]
            normed = self._final_norm(final_hidden.float()).to(dtype)
            output_buffer[remaining_mask] = normed[remaining_mask]
        stats.remaining_tokens = remaining_mask.sum().item()

        self.last_stats = stats
        logits = self._lm_head(output_buffer)
        return logits

    @staticmethod
    def _pad_kv_cache(cache, exit_layer: int, total_layers: int):
        """Pad KV cache for skipped layers with zeros to maintain sequence length consistency.

        When early exit skips layers exit_layer+1..total_layers-1, those layers never
        ran, so their KV cache entries are one position short. This pads them with zeros
        so the next forward pass sees consistent sequence lengths across all layers.
        """
        if not hasattr(cache, "key_cache"):
            return
        if exit_layer >= len(cache.key_cache):
            return
        ref_k = cache.key_cache[exit_layer]
        ref_v = cache.value_cache[exit_layer]
        if ref_k.dim() < 3:
            return  # Cache not populated yet (e.g. stub or first step)
        seq_len = ref_k.shape[2]

        for idx in range(exit_layer + 1, min(total_layers, len(cache.key_cache))):
            if cache.key_cache[idx].shape[2] < seq_len:
                pad_k = torch.zeros_like(ref_k[:, :, -1:, :])
                pad_v = torch.zeros_like(ref_v[:, :, -1:, :])
                cache.key_cache[idx] = torch.cat(
                    [cache.key_cache[idx], pad_k], dim=2
                )
                cache.value_cache[idx] = torch.cat(
                    [cache.value_cache[idx], pad_v], dim=2
                )

    def _generate_with_skipping(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Generate tokens with post-hoc early exit evaluation.

        Runs the full model forward with output_hidden_states=True, then evaluates
        routers post-hoc to decide which layer's hidden state to use for logits.
        All layers run every step (preserving KV cache correctness), but the output
        comes from the earliest layer whose router fires.

        Compatible with all transformers versions (no hooks or exceptions).
        """
        device = input_ids.device
        B = input_ids.shape[0]
        n_layers = len(self._layers)
        generated = input_ids.clone()
        gen_stats = ExitStats(total_tokens=0)

        # Prefill: run full model
        prefill_out = self.model(
            generated, use_cache=True, return_dict=True,
        )
        past_key_values = prefill_out.past_key_values
        next_logits = prefill_out.logits[:, -1, :]

        for step in range(max_new_tokens):
            next_token = self._sample_next_token(next_logits, temperature, top_k, top_p)
            generated = torch.cat([generated, next_token], dim=-1)

            if (next_token == 2).all():
                break

            gen_stats.total_tokens += B

            # Run full forward with hidden states for post-hoc router evaluation
            out = self.model(
                next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
            past_key_values = out.past_key_values
            all_hidden = out.hidden_states  # [emb, layer_0, ..., layer_N]

            # Evaluate routers post-hoc to find earliest exit
            exit_hidden = None
            exit_layer = None
            for layer_idx in sorted(self.routers.keys()):
                if layer_idx < self.config.min_layers:
                    continue
                hidden = all_hidden[layer_idx + 1]  # +1 for embedding offset
                scores = self._score_router(hidden, layer_idx)
                if (scores > self.config.exit_threshold).all():
                    exit_hidden = hidden
                    exit_layer = layer_idx
                    break

            if exit_hidden is not None:
                # Use hidden state from exit layer
                normed = self._final_norm(exit_hidden.float()).to(exit_hidden.dtype)
                lm_out = self._lm_head(normed)
                next_logits = lm_out[:, -1, :] if lm_out.dim() == 3 else lm_out
                gen_stats.exits_per_layer[exit_layer] = (
                    gen_stats.exits_per_layer.get(exit_layer, 0) + B
                )
            else:
                # No exit — use final logits
                next_logits = out.logits[:, -1, :]
                gen_stats.remaining_tokens += B

        self.last_stats = gen_stats

        if gen_stats.total_tokens > 0:
            total_layer_runs = 0
            for lidx, count in gen_stats.exits_per_layer.items():
                total_layer_runs += (lidx + 1) * count
            total_layer_runs += gen_stats.remaining_tokens * n_layers
            full_cost = gen_stats.total_tokens * n_layers
            estimated_speedup = full_cost / total_layer_runs if total_layer_runs > 0 else 1.0
            logger.info(
                f"Generation: {gen_stats.total_tokens} tokens, "
                f"{gen_stats.total_exited} exits ({gen_stats.exit_rate:.1%}), "
                f"estimated {estimated_speedup:.2f}x equivalent speedup"
            )
        return generated

    @staticmethod
    def _sample_next_token(
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Sample a single next token from logits [B, vocab]."""
        next_logits = logits.clone()

        if temperature > 0:
            next_logits = next_logits / temperature

            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < topk_vals[:, -1:]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_logits, descending=True
                )
                cumulative_probs = (
                    torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                )
                remove_mask = cumulative_probs > top_p
                remove_mask[:, 1:] = remove_mask[:, :-1].clone()
                remove_mask[:, 0] = False
                sorted_logits[remove_mask] = float("-inf")
                next_logits = sorted_logits.scatter(
                    1, sorted_indices, sorted_logits
                )

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        return next_token

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive generation with post-hoc early exit.

        Each decode step runs all layers (preserving KV cache correctness),
        then evaluates routers to select which layer's output to use for
        the next token. Compatible with all transformers versions.
        """
        return self._generate_with_skipping(
            input_ids, max_new_tokens, temperature, top_k, top_p,
        )

    @staticmethod
    def calibrate(
        model: nn.Module,
        tokenizer,
        dataset: str = "wikitext",
        num_samples: int = 2000,
        save_path: str = "./router.pt",
        config: Optional[TIDEConfig] = None,
    ) -> RouterCheckpoint:
        """Convenience static method wrapping calibrate.py."""
        from TIDE.calibrate import calibrate as _calibrate

        cfg = config or TIDEConfig()
        cfg.calibration_samples = num_samples
        cfg.calibration_dataset = dataset
        return _calibrate(model, tokenizer, config=cfg, save_path=save_path)
