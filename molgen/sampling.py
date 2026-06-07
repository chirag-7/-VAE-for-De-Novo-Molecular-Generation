"""Autoregressive sampling from SMILES language models (CharRNN, MolGPT).

Generates token sequences left-to-right starting from BOS, with temperature
plus optional top-k / top-p (nucleus) filtering, and decodes them to SMILES
with the model's tokenizer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from molgen.utils import get_device


def _filter_logits(logits: torch.Tensor, top_k: int | None, top_p: float | None) -> torch.Tensor:
    """Apply top-k and/or top-p (nucleus) filtering to a ``(batch, vocab)`` logit tensor."""
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        # Always keep at least the top token.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        to_remove = remove.scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(to_remove, float("-inf"))
    return logits


def _forward_logits(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def sample(
    model,
    tokenizer,
    num_samples: int = 100,
    max_len: int = 120,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    device: torch.device | None = None,
    batch_size: int = 256,
) -> list[str]:
    """Generate ``num_samples`` SMILES strings from ``model``.

    Sampling stops a sequence at EOS; sequences are decoded with the tokenizer
    (special tokens stripped). Works with any model whose forward returns
    next-token logits (optionally alongside hidden state).
    """
    device = device or get_device()
    model.to(device)
    model.eval()
    results: list[str] = []
    remaining = num_samples
    while remaining > 0:
        batch = min(batch_size, remaining)
        seqs = torch.full((batch, 1), tokenizer.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        for _ in range(max_len):
            logits = _forward_logits(model, seqs)[:, -1, :] / max(temperature, 1e-6)
            logits = _filter_logits(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            seqs = torch.cat([seqs, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == tokenizer.eos_id)
            if bool(finished.all()):
                break
        for row in seqs.tolist():
            results.append(tokenizer.decode(row))
        remaining -= batch
    return results
