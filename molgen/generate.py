"""Generate molecules near a seed by perturbing its VAE latent.

Encode a seed molecule, jitter its latent mean with random directions of
increasing magnitude, decode each perturbed latent, and keep the valid,
canonical, distinct results. With a :class:`~molgen.selfies_tokenizer.SelfiesTokenizer`
every decode is syntactically valid, so the only filtering is RDKit
canonicalization and de-duplication against the seed.
"""

from __future__ import annotations

import torch

from molgen.chem import canonicalize_smiles

__all__ = ["generate_nearby_smiles"]


def generate_nearby_smiles(
    model,
    tokenizer,
    smiles,
    max_len,
    num_samples,
    device,
    temperature=1.0,
    distance_multiplier=0.5,
):
    """Sample canonical molecules near ``smiles`` in the VAE latent space.

    Args:
        model: A trained :class:`~molgen.vae.BetaTCVAE` (already on ``device``).
        tokenizer: A toolkit tokenizer with ``encode``/``decode`` and ``pad_id``
            (use :class:`~molgen.selfies_tokenizer.SelfiesTokenizer` for
            always-valid decoding).
        smiles: Seed SMILES to explore around.
        max_len: Sequence length the model was trained at.
        num_samples: Number of perturbed latents to decode.
        device: Torch device.
        temperature: Softmax temperature for sampling decoded tokens.
        distance_multiplier: Scales the per-step latent perturbation magnitude.

    Returns:
        Sorted list of distinct canonical SMILES (excluding the seed itself).
    """
    model.eval()
    seed_ids = torch.tensor([tokenizer.encode(smiles, max_len=max_len)], device=device)
    pad_mask = seed_ids == tokenizer.pad_id

    with torch.no_grad():
        memory, mu = model.encode(seed_ids)
        seed_canon = canonicalize_smiles(smiles)
        results = set()
        for i in range(num_samples):
            direction = torch.randn_like(mu)
            direction = direction / direction.norm()
            z = mu + distance_multiplier * (i + 1) * direction
            logits = model.decode_logits(z, memory, memory_pad_mask=pad_mask)
            probs = torch.softmax(logits.squeeze(0) / temperature, dim=-1)
            ids = torch.multinomial(probs, 1).flatten().tolist()
            canon = canonicalize_smiles(tokenizer.decode(ids))
            if canon and canon != seed_canon:
                results.add(canon)

    return sorted(results)


def main():
    """Train a tiny VAE on the bundled sample set and generate near a seed."""
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

    seed = smiles[0]
    generated = generate_nearby_smiles(
        model, tokenizer, seed, max_len, 200, device, temperature=1.0
    )
    print(f"Seed: {seed}")
    print(f"Generated {len(generated)} distinct nearby molecules:")
    for s in generated[:20]:
        print(f"  {s}")


if __name__ == "__main__":
    main()
