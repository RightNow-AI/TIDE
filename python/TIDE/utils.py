from __future__ import annotations

import logging

import torch


def cosine_similarity_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = torch.nn.functional.normalize(a, dim=-1)
    b_norm = torch.nn.functional.normalize(b, dim=-1)
    return (a_norm * b_norm).sum(dim=-1)


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("TIDE")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[TIDE %(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper()))
    return logger


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
