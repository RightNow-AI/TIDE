from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from TIDE.config import TIDEConfig
from TIDE.router import RouterCheckpoint, TokenRouter
from TIDE.utils import cosine_similarity_batch, setup_logging

logger = setup_logging()


@torch.no_grad()
def collect_hidden_states(
    model: nn.Module,
    tokenizer,
    dataset_name: str,
    config: TIDEConfig,
    max_length: int = 512,
) -> Dict[int, torch.Tensor]:
    """Forward pass using output_hidden_states=True to collect all layer outputs.

    Returns dict mapping layer_idx -> tensor of shape [num_tokens, hidden_dim].
    """
    from TIDE.adapters.auto import get_adapter

    adapter = get_adapter(model)
    layers = adapter.get_layers(model)
    device = next(model.parameters()).device
    n_layers = len(layers)
    checkpoint_layers = list(range(config.checkpoint_interval - 1, n_layers, config.checkpoint_interval))

    texts = _load_calibration_texts(dataset_name, config.calibration_samples)
    logger.info(f"Collected {len(texts)} calibration texts")

    hidden_states = {idx: [] for idx in checkpoint_layers}
    hidden_states["final"] = []

    for i in range(0, len(texts), 8):
        batch_texts = texts[i : i + 8]
        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        outputs = model(
            **encodings,
            output_hidden_states=True,
            return_dict=True,
        )

        # output_hidden_states gives [embedding_output, layer_0_output, ..., layer_N_output]
        # So layer i output is at index i+1
        all_hidden = outputs.hidden_states
        attention_mask = encodings.get("attention_mask", None)

        for layer_idx in checkpoint_layers:
            h = all_hidden[layer_idx + 1]  # +1 because index 0 is embeddings
            flat = h.reshape(-1, h.shape[-1])
            if attention_mask is not None:
                mask_flat = attention_mask.reshape(-1).bool()
                flat = flat[mask_flat]
            hidden_states[layer_idx].append(flat.cpu())

        # Final hidden state (after last layer, before norm)
        h_final = all_hidden[-1]
        flat_final = h_final.reshape(-1, h_final.shape[-1])
        if attention_mask is not None:
            mask_flat = attention_mask.reshape(-1).bool()
            flat_final = flat_final[mask_flat]
        hidden_states["final"].append(flat_final.cpu())

    result = {}
    for key in hidden_states:
        if hidden_states[key]:
            result[key] = torch.cat(hidden_states[key], dim=0)
    logger.info(f"Collected hidden states at {len(checkpoint_layers)} checkpoints, "
                f"{result.get('final', torch.empty(0)).shape[0]} total tokens")
    return result


def compute_convergence_labels(
    hidden_states: Dict, config: TIDEConfig
) -> Dict[int, torch.Tensor]:
    """Compute binary convergence labels: 1 if cosine_sim(h_l, h_final) > threshold."""
    final = hidden_states["final"]
    labels = {}
    for layer_idx, h_l in hidden_states.items():
        if layer_idx == "final":
            continue
        n = min(h_l.shape[0], final.shape[0])
        sim = cosine_similarity_batch(h_l[:n], final[:n])
        labels[layer_idx] = (sim > config.convergence_threshold).float()
        converged_pct = labels[layer_idx].mean().item() * 100
        logger.info(f"  Layer {layer_idx}: {converged_pct:.1f}% tokens converged (cosine > {config.convergence_threshold})")
    return labels


def train_routers(
    hidden_states: Dict,
    labels: Dict[int, torch.Tensor],
    config: TIDEConfig,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Dict[int, TokenRouter]:
    """Train one TokenRouter per checkpoint layer via binary cross-entropy."""
    routers = {}
    hidden_dim = hidden_states["final"].shape[-1]

    for layer_idx in sorted(labels.keys()):
        h = hidden_states[layer_idx].to(device).float()
        y = labels[layer_idx].to(device).float()
        n = min(h.shape[0], y.shape[0])
        h, y = h[:n], y[:n]

        router = TokenRouter(hidden_dim, config.router_bottleneck_dim).to(device)
        optimizer = torch.optim.Adam(router.parameters(), lr=lr)
        criterion = nn.BCELoss()

        best_loss = float("inf")
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = router(h)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()

            if (epoch + 1) % 25 == 0:
                acc = ((pred > 0.5) == (y > 0.5)).float().mean().item()
                logger.info(f"  Layer {layer_idx} epoch {epoch+1}: loss={loss.item():.4f} acc={acc:.3f}")

        routers[layer_idx] = router.cpu()
        logger.info(f"  Layer {layer_idx} final loss: {best_loss:.4f}")

    return routers


def calibrate(
    model: nn.Module,
    tokenizer,
    config: Optional[TIDEConfig] = None,
    save_path: str = "./router.pt",
    device: Optional[str] = None,
) -> RouterCheckpoint:
    """Full calibration pipeline: collect -> label -> train -> save."""
    config = config or TIDEConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=== TIDE Calibration ===")
    logger.info(f"Config: interval={config.checkpoint_interval}, threshold={config.convergence_threshold}")

    logger.info("Step 1/3: Collecting hidden states...")
    hidden_states = collect_hidden_states(model, tokenizer, config.calibration_dataset, config)

    logger.info("Step 2/3: Computing convergence labels...")
    labels = compute_convergence_labels(hidden_states, config)

    logger.info("Step 3/3: Training routers...")
    routers = train_routers(hidden_states, labels, config, device=device)

    hidden_dim = hidden_states["final"].shape[-1]
    checkpoint = RouterCheckpoint(
        routers=routers,
        hidden_dim=hidden_dim,
        bottleneck_dim=config.router_bottleneck_dim,
    )
    checkpoint.save(save_path)
    logger.info(f"Saved router checkpoint to {save_path}")
    return checkpoint


def _load_calibration_texts(dataset_name: str, num_samples: int) -> List[str]:
    """Load calibration texts from a dataset."""
    try:
        from datasets import load_dataset

        if "wikitext" in dataset_name:
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        else:
            ds = load_dataset(dataset_name, split="train", streaming=True)

        texts = []
        for sample in ds:
            text = sample.get("text", "")
            if len(text.strip()) > 100:
                texts.append(text.strip())
            if len(texts) >= num_samples:
                break
        return texts
    except Exception as e:
        logger.warning(f"Failed to load dataset '{dataset_name}': {e}. Using dummy data.")
        return [f"The quick brown fox jumps over the lazy dog. Sentence {i}." for i in range(num_samples)]
