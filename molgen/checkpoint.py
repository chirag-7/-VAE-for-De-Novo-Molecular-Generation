"""Save and load model + tokenizer checkpoints in a single file.

A checkpoint stores the model kind and constructor kwargs, its state dict, and
the tokenizer kind and vocabulary, so a trained generator can be fully
reconstructed for inference without re-specifying the architecture.
"""

from __future__ import annotations

import torch

from molgen.char_rnn import CharRNN
from molgen.molgpt import MolGPT
from molgen.tokenizers import SmilesTokenizer
from molgen.vae import BetaTCVAE

_MODELS = {"charrnn": CharRNN, "molgpt": MolGPT, "vae": BetaTCVAE}


def _tokenizer_kind(tokenizer) -> str:
    return "selfies" if type(tokenizer).__name__ == "SelfiesTokenizer" else "smiles"


def save_checkpoint(path, model, model_kind: str, model_kwargs: dict, tokenizer) -> None:
    """Save model architecture/state and tokenizer vocabulary to ``path``."""
    if model_kind not in _MODELS:
        raise ValueError(f"unknown model_kind {model_kind!r}; expected one of {list(_MODELS)}")
    torch.save(
        {
            "model_kind": model_kind,
            "model_kwargs": model_kwargs,
            "model_state": model.state_dict(),
            "tokenizer_kind": _tokenizer_kind(tokenizer),
            "vocab": list(tokenizer.itos),
        },
        path,
    )


def load_checkpoint(path, device=None):
    """Reconstruct ``(model, tokenizer)`` from a checkpoint saved by save_checkpoint."""
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    if ckpt["tokenizer_kind"] == "selfies":
        from molgen.selfies_tokenizer import SelfiesTokenizer

        tokenizer = SelfiesTokenizer(ckpt["vocab"])
    else:
        tokenizer = SmilesTokenizer(ckpt["vocab"])
    model = _MODELS[ckpt["model_kind"]](**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state"])
    return model, tokenizer
