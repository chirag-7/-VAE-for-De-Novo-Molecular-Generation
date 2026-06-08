"""Tests for the molgen command-line interface."""

import json

from molgen.cli import main
from molgen.data import load_sample_smiles


def test_eval_command_prints_report(tmp_path, capsys):
    generated = tmp_path / "gen.smi"
    generated.write_text("CCO\nc1ccccc1\ninvalid$$\n", encoding="utf-8")
    main(["eval", "--generated", str(generated)])
    report = json.loads(capsys.readouterr().out)
    assert report["n_generated"] == 3
    assert "validity" in report


def test_train_then_sample_roundtrip(tmp_path):
    data = tmp_path / "data.smi"
    data.write_text("\n".join(load_sample_smiles()[:40]) + "\n", encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    main(
        [
            "train",
            "--data", str(data),
            "--model", "charrnn",
            "--epochs", "1",
            "--batch-size", "8",
            "--embedding-dim", "16",
            "--hidden-dim", "32",
            "--num-layers", "1",
            "--out", str(checkpoint),
        ]
    )
    assert checkpoint.exists()

    generated = tmp_path / "gen.smi"
    main(
        [
            "sample",
            "--checkpoint", str(checkpoint),
            "--num", "5",
            "--max-len", "20",
            "--out", str(generated),
        ]
    )
    assert len(generated.read_text(encoding="utf-8").splitlines()) == 5
