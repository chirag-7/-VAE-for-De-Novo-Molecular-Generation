import math

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class SMILESDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_len):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        smiles = self.smiles_list[idx]
        tokenized_smiles = self.tokenizer(smiles, self.max_len)

        return tokenized_smiles


def smiles_data_loader(
    csv_file, tokenizer, batch_size, max_len=None, test_split=0.2, shuffle=True, num_workers=0
):
    """
    Load SMILES data from a CSV file, tokenize it, and split into train and test sets.

    Args:
        csv_file (str): Path to the CSV file containing SMILES strings.
        tokenizer (callable): Function to tokenize the SMILES strings.
        batch_size (int): Number of samples per batch.
        max_len (int): Maximum length of the tokenized sequences.
        test_split (float): Proportion of data to be used as test set.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the test set.
    """
    data = pd.read_csv(csv_file)
    smiles_list = data["SMILES"].tolist()

    if max_len is None:
        max_len = max(len(smiles) for smiles in smiles_list)

    train_smiles, test_smiles = train_test_split(smiles_list, test_size=test_split, random_state=42)
    train_dataset = SMILESDataset(train_smiles, tokenizer, max_len)
    test_dataset = SMILESDataset(test_smiles, tokenizer, max_len)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return train_loader, test_loader


class SimpleTokenizer:
    def __init__(self):
        self.char_to_idx = {"C": 0, "O": 1, "(": 2, ")": 3, "<pad>": 4}
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.pad_idx = self.char_to_idx["<pad>"]
        self.vocab_size = len(self.char_to_idx)

    def tokenize(self, smiles, max_len):
        """
        Tokenize the SMILES string and pad to the maximum length.

        Args:
            smiles (str): SMILES string to tokenize.
            max_len (int): Maximum length of the tokenized sequence.

        Returns:
            torch.Tensor: Tokenized and padded SMILES string.
        """
        tokenized_smiles = [self.char_to_idx[char] for char in smiles]
        tokenized_smiles += [self.pad_idx] * (max_len - len(tokenized_smiles))
        return torch.tensor(tokenized_smiles, dtype=torch.long)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for batch-first (batch, seq, dim) inputs."""

    def __init__(self, embedding_dim, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: embedding_dim // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embedding_dim)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class BetaTCVAE(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, latent_dim, nhead, num_layers, pad_idx, device
    ):
        super(BetaTCVAE, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, nhead, hidden_dim, batch_first=True),
            num_layers=num_layers,
        )
        self.mu = nn.Linear(embedding_dim, latent_dim)
        self.log_var = nn.Linear(embedding_dim, latent_dim)

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embedding_dim, nhead, hidden_dim, batch_first=True),
            num_layers=num_layers,
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

        self.device = device
        self.pad_idx = pad_idx

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std

    def forward(self, x):
        # Mask padding positions so attention never attends to <pad> tokens.
        pad_mask = x == self.pad_idx  # (batch, seq), True where padded
        embedded = self.pos_encoder(self.embedding(x))
        encoded = self.encoder(embedded, src_key_padding_mask=pad_mask)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)

        decoded = self.decoder(
            z,
            encoded,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=pad_mask,
        )
        out = self.fc_out(decoded)

        return out, mu, log_var


def loss_function(recon_x, x, mu, log_var, beta, gamma, pad_idx=-100):
    BCE = F.cross_entropy(
        recon_x.view(-1, recon_x.size(-1)), x.view(-1), ignore_index=pad_idx, reduction="mean"
    )
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    TC = (log_var.exp() - 1 - log_var).mean()

    return BCE + beta * KLD + gamma * TC


def masked_token_accuracy(preds, targets, pad_idx):
    """Token-level accuracy over non-padding positions.

    Returns ``(num_correct, num_tokens)`` so callers can aggregate across batches.
    """
    mask = targets != pad_idx
    correct = int(((preds == targets) & mask).sum().item())
    total = int(mask.sum().item())
    return correct, total


def train(model, train_loader, optimizer, device, beta, gamma):
    model.train()
    total_loss = 0

    for x in train_loader:
        x = x.to(device)
        optimizer.zero_grad()

        recon_x, mu, log_var = model(x)
        loss = loss_function(recon_x, x, mu, log_var, beta, gamma, pad_idx=model.pad_idx)
        loss.backward()
        total_loss += loss.item()

        optimizer.step()

    return total_loss / len(train_loader)


def test(model, test_loader, device, beta, gamma):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            recon_x, mu, log_var = model(x)
            loss = loss_function(recon_x, x, mu, log_var, beta, gamma, pad_idx=model.pad_idx)
            total_loss += loss.item()

            preds = torch.argmax(recon_x, dim=-1)
            batch_correct, batch_total = masked_token_accuracy(preds, x, model.pad_idx)
            correct += batch_correct
            total += batch_total

    avg_loss = total_loss / len(test_loader)
    avg_accuracy = correct / total if total > 0 else 0.0

    return avg_loss, avg_accuracy


def main():
    simple_tokenizer = SimpleTokenizer()
    csv_file = "molecules.csv"
    batch_size = 64
    max_len = None  # Can be set to a specific value, or let the function calculate

    train_loader, test_loader = smiles_data_loader(
        csv_file, simple_tokenizer.tokenize, batch_size, max_len
    )

    # Hyperparameters
    vocab_size = simple_tokenizer.vocab_size
    embedding_dim = 16
    hidden_dim = 64
    latent_dim = 16
    nhead = 4
    num_layers = 2
    pad_idx = simple_tokenizer.pad_idx
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model
    model = BetaTCVAE(
        vocab_size, embedding_dim, hidden_dim, latent_dim, nhead, num_layers, pad_idx, device
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training settings
    epochs = 10
    beta = 1.0
    gamma = 0.1

    # Training loop
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, device, beta, gamma)
        print(f"Epoch: {epoch + 1}, Loss: {train_loss:.4f}")
        test_loss, test_accuracy = test(model, test_loader, device, beta, gamma)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Save the model
    torch.save(model.state_dict(), "beta_tc_vae_model.pth")


if __name__ == "__main__":
    main()
