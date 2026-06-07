"""SMILES tokenization.

Provides an atom-aware, regex-based SMILES tokenizer with explicit special
tokens and a corpus-built vocabulary. This replaces the toy character map
that only handled ``C``, ``O``, ``(`` and ``)``.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

import torch

# Atom-level SMILES regex from Schwaller et al., "Molecular Transformer" (2019).
# Matches bracket atoms, two-letter atoms (Br, Cl), the organic subset,
# bonds, ring-closure digits, and stereochemistry markers.
SMILES_REGEX = (
    r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|"
    r"\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)
_TOKEN_RE = re.compile(SMILES_REGEX)

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


def smiles_tokens(smiles: str) -> list[str]:
    """Split a SMILES string into atom/bond tokens using the atom-level regex."""
    return _TOKEN_RE.findall(smiles)


class SmilesTokenizer:
    """Maps SMILES strings to/from integer id sequences.

    The vocabulary is an ordered list of tokens whose first entries are the
    special tokens (pad/bos/eos/unk). Build one from a corpus with
    :meth:`from_smiles`.
    """

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

    @classmethod
    def from_smiles(cls, smiles_iter: Iterable[str], min_freq: int = 1) -> SmilesTokenizer:
        """Build a tokenizer from a corpus, keeping tokens with count >= min_freq."""
        counter: Counter[str] = Counter()
        for smi in smiles_iter:
            counter.update(smiles_tokens(smi))
        # Deterministic order: by descending frequency, then lexicographically.
        tokens = [
            tok
            for tok, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
            if count >= min_freq
        ]
        return cls(SPECIAL_TOKENS + tokens)

    def tokenize(self, smiles: str) -> list[str]:
        return smiles_tokens(smiles)

    def encode(
        self, smiles: str, add_bos_eos: bool = True, max_len: int | None = None
    ) -> list[int]:
        """Encode a SMILES string into token ids, optionally padded/truncated."""
        ids = [self.stoi.get(tok, self.unk_id) for tok in smiles_tokens(smiles)]
        if add_bos_eos:
            ids = [self.bos_id, *ids, self.eos_id]
        if max_len is not None:
            ids = ids[:max_len] + [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: Sequence[int], strip_special: bool = True) -> str:
        """Decode token ids back into a SMILES string.

        When ``strip_special`` is set, decoding stops at the first EOS and
        pad/bos/eos tokens are dropped.
        """
        specials = {self.pad_id, self.bos_id, self.eos_id}
        out: list[str] = []
        for raw in ids:
            i = int(raw)
            if strip_special:
                if i == self.eos_id:
                    break
                if i in specials:
                    continue
            out.append(self.itos[i] if 0 <= i < len(self.itos) else UNK_TOKEN)
        return "".join(out)

    def encode_batch(self, smiles_list: Sequence[str], max_len: int | None = None) -> torch.Tensor:
        """Encode a list of SMILES into a padded ``(batch, seq)`` LongTensor."""
        encoded = [self.encode(s, add_bos_eos=True, max_len=max_len) for s in smiles_list]
        if max_len is None:
            width = max(len(e) for e in encoded)
            encoded = [e + [self.pad_id] * (width - len(e)) for e in encoded]
        return torch.tensor(encoded, dtype=torch.long)
