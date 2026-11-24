"""Quels modeles peuvent reellement tourner hors ligne ?

Ces tests ne chargent AUCUN poids : ils regardent seulement les fichiers
du snapshot present dans le cache Hugging Face. C'est utile parce que
``Generator.model_post_init`` demande ``vocab.json`` via
``get_path_to_vocab_file()`` : un modele qui n'a que ``tokenizer.json``
ne peut pas demarrer, meme avec --cache_model.
"""
import os
from pathlib import Path

import pytest

HUB = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"


def snapshot(repo: str) -> Path | None:
    """Dernier snapshot local d'un repo, ou None s'il n'est pas telecharge."""
    folder = HUB / f"models--{repo.replace('/', '--')}" / "snapshots"
    if not folder.is_dir():
        return None
    snaps = sorted(p for p in folder.iterdir() if p.is_dir())
    return snaps[-1] if snaps else None


def test_qwen3_0_6b_has_the_files_the_sdk_requests() -> None:
    """Le modele par defaut est complet : vocab.json + merges.txt."""
    snap = snapshot("Qwen/Qwen3-0.6B")
    if snap is None:
        pytest.skip("Qwen/Qwen3-0.6B n'est pas dans le cache HF local")

    assert (snap / "vocab.json").is_file()
    assert (snap / "merges.txt").is_file()
    assert (snap / "tokenizer.json").is_file()
    assert (snap / "config.json").is_file()


@pytest.mark.xfail(
    strict=False,
    reason="MiniCPM5 n'a que tokenizer.json : get_path_to_vocab_file() "
           "demande vocab.json et leve LocalEntryNotFoundError en offline",
)
def test_minicpm5_1b_has_the_files_the_sdk_requests() -> None:
    """Modele cite dans le README, mais incompatible avec le chemin actuel."""
    snap = snapshot("openbmb/MiniCPM5-1B")
    assert snap is not None, "MiniCPM5 n'est pas dans le cache HF local"
    assert (snap / "vocab.json").is_file()
    assert (snap / "merges.txt").is_file()


@pytest.mark.xfail(
    strict=False,
    reason="Llama-3.2-1B est gated : le dossier du cache n'a pas de snapshot",
)
def test_llama_3_2_1b_snapshot_is_usable_offline() -> None:
    """Deuxieme modele cite dans le README : absent du cache local."""
    snap = snapshot("meta-llama/Llama-3.2-1B")
    assert snap is not None
    assert (snap / "config.json").is_file()
