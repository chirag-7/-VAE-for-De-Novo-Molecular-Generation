"""Training loop for autoregressive SMILES language models (CharRNN, MolGPT).

Supports mixed-precision (AMP) on CUDA, gradient clipping, AdamW, and
validation. Models are trained with teacher forcing: the input is
``tokens[:, :-1]`` and the target is ``tokens[:, 1:]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from molgen.utils import get_device, get_logger


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    amp: bool = True
    log_every_epoch: bool = True


def lm_loss(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Next-token cross-entropy, ignoring padding targets."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_idx
    )


def _forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Call a model and return logits, tolerating models that also return state."""
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def evaluate_language_model(
    model: nn.Module, loader: DataLoader, device: torch.device, pad_idx: int
) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        loss = lm_loss(_forward_logits(model, inputs), targets, pad_idx)
        total += loss.item() * batch.size(0)
        count += batch.size(0)
    return total / max(count, 1)


def train_language_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    config: TrainConfig | None = None,
    device: torch.device | None = None,
    pad_idx: int = 0,
) -> list[dict]:
    """Train ``model`` and return a per-epoch history of train/val losses."""
    config = config or TrainConfig()
    device = device or get_device()
    logger = get_logger()
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    history: list[dict] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = lm_loss(_forward_logits(model, inputs), targets, pad_idx)
            scaler.scale(loss).backward()
            if config.grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * batch.size(0)
            count += batch.size(0)

        train_loss = total / max(count, 1)
        val_loss = (
            evaluate_language_model(model, val_loader, device, pad_idx)
            if val_loader is not None
            else None
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if config.log_every_epoch:
            msg = f"epoch {epoch}/{config.epochs} train_loss={train_loss:.4f}"
            if val_loss is not None:
                msg += f" val_loss={val_loss:.4f}"
            logger.info(msg)

    return history
