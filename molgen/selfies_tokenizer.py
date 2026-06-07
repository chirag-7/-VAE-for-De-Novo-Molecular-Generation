"""SELFIES-based tokenization.

SELFIES (Krenn et al., 2020) is a molecular string representation in which
*every* symbol sequence decodes to a valid molecule. This is attractive for
generation: a model can emit arbitrary SELFIES symbols and the decoder still
yields a syntactically valid SMILES.

Requires the optional ``selfies`` dependency (``pip install molgen[selfies]``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import selfies as sf
import torch

from molgen.tokenizers import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, SPECIAL_TOKENS, UNK_TOKEN


class SelfiesTokenizer:
    """Maps SMILES to/from integer ids via the SELFIES representation."""

    def __init__(self, vocab: Sequence[str]):
        self.itos = list(vocab)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        for tok in SPECIAL_TOKENS:
            if tok not in self.stoi:
                raise ValueError(f"vocab is missing required special token {tok!r}")
        self.pad_id = self.stoi[PAD_TOKEN]
        self.bos_id = self.stoi[BOS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @staticmethod
    def smiles_to_selfies(smiles: str) -> str | None:
        """Return the SELFIES encoding of ``smiles``, or None if it cannot be encoded."""
        try:
            return sf.encoder(smiles)
        except Exception:
            return None

    @classmethod
    def from_smiles(cls, smiles_iter: Iterable[str], min_freq: int = 1) -> SelfiesTokenizer:
        """Build a tokenizer by collecting SELFIES symbols from a SMILES corpus."""
        counter: Counter[str] = Counter()
        for smi in smiles_iter:
            encoded = cls.smiles_to_selfies(smi)
            if encoded is None:
                continue
            counter.update(sf.split_selfies(encoded))
        symbols = [
            sym
            for sym, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
            if count >= min_freq
        ]
        return cls(SPECIAL_TOKENS + symbols)

    def tokenize(self, smiles: str) -> list[str]:
        encoded = self.smiles_to_selfies(smiles)
        return list(sf.split_selfies(encoded)) if encoded is not None else []

    def encode(
        self, smiles: str, add_bos_eos: bool = True, max_len: int | None = None
    ) -> list[int]:
        ids = [self.stoi.get(sym, self.unk_id) for sym in self.tokenize(smiles)]
        if add_bos_eos:
            ids = [self.bos_id, *ids, self.eos_id]
        if max_len is not None:
            ids = ids[:max_len] + [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Decode ids back to a SMILES string (always syntactically valid).

        Stops at the first EOS and skips pad/bos/unk; the remaining SELFIES
        symbols are joined and decoded via the SELFIES decoder.
        """
        skip = {self.pad_id, self.bos_id, self.unk_id}
        symbols: list[str] = []
        for raw in ids:
            i = int(raw)
            if i == self.eos_id:
                break
            if i in skip:
                continue
            if 0 <= i < len(self.itos):
                symbols.append(self.itos[i])
        return sf.decoder("".join(symbols))

    def encode_batch(self, smiles_list: Sequence[str], max_len: int | None = None) -> torch.Tensor:
        encoded = [self.encode(s, add_bos_eos=True, max_len=max_len) for s in smiles_list]
        if max_len is None:
            width = max(len(e) for e in encoded)
            encoded = [e + [self.pad_id] * (width - len(e)) for e in encoded]
        return torch.tensor(encoded, dtype=torch.long)
