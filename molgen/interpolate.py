"""Interpolate between two molecules in the VAE latent space.

Encode both endpoints to fixed length, walk the latent mean linearly from one
to the other, and decode each step. With a
:class:`~molgen.selfies_tokenizer.SelfiesTokenizer` every decoded point is a
syntactically valid molecule, so the returned path is the ordered sequence of
distinct canonical molecules seen along the way.
"""

from __future__ import annotations

import torch

from molgen.chem import canonicalize_smiles

__all__ = ["interpolate_smiles"]


def interpolate_smiles(
    model, tokenizer, smiles_1, smiles_2, max_len, device, num_steps=10, temperature=1.0
):
    """Decode latent points on the line between two molecules.

    Args:
        model: A trained :class:`~molgen.vae.BetaTCVAE` (already on ``device``).
        tokenizer: A toolkit tokenizer with ``encode``/``decode`` and ``pad_id``.
        smiles_1: Start SMILES.
        smiles_2: End SMILES.
        max_len: Sequence length the model was trained at.
        device: Torch device.
        num_steps: Number of interpolation steps (the path has ``num_steps + 1``
            points, including both endpoints' latents).
        temperature: Softmax temperature for sampling decoded tokens.

    Returns:
        Ordered list of distinct canonical SMILES along the interpolation path.
    """
    model.eval()
    ids_1 = torch.tensor([tokenizer.encode(smiles_1, max_len=max_len)], device=device)
    ids_2 = torch.tensor([tokenizer.encode(smiles_2, max_len=max_len)], device=device)

    with torch.no_grad():
        mu_1, _ = model.encode(ids_1)  # single latent vector per endpoint
        mu_2, _ = model.encode(ids_2)

        path = []
        seen = set()
        for i in range(num_steps + 1):
            alpha = i / num_steps
            z = mu_1 * (1 - alpha) + mu_2 * alpha
            logits = model.decode(z, max_len)
            probs = torch.softmax(logits.squeeze(0) / temperature, dim=-1)
            ids = torch.multinomial(probs, 1).flatten().tolist()
            canon = canonicalize_smiles(tokenizer.decode(ids))
            if canon and canon not in seen:
                seen.add(canon)
                path.append(canon)

    return path


def main():
    """Train a tiny VAE on the bundled sample set and interpolate two molecules."""
    from torch.utils.data import DataLoader, TensorDataset

    from molgen.data import load_sample_smiles
    from molgen.selfies_tokenizer import SelfiesTokenizer
    from molgen.vae import BetaTCVAE, train

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smiles = load_sample_smiles()
    tokenizer = SelfiesTokenizer.from_smiles(smiles)
    data = tokenizer.encode_batch(smiles, max_len=None)
    max_len = data.shape[1]

    model = BetaTCVAE(tokenizer.vocab_size, 64, 256, 64, 4, 2, tokenizer.pad_id, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(data), batch_size=64, shuffle=True)
    for epoch in range(10):
        train(model, loader, optimizer, device, beta=0.5, gamma=0.1)

    a, b = smiles[0], smiles[1]
    path = interpolate_smiles(model, tokenizer, a, b, max_len, device, num_steps=10)
    print(f"From: {a}")
    print(f"To:   {b}")
    print(f"Path ({len(path)} distinct molecules):")
    for s in path:
        print(f"  {s}")


if __name__ == "__main__":
    main()
