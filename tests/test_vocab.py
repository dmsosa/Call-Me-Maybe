"""Contrat du vocabulaire : ids, masques, et ids « spéciaux ».

Ces tests utilisent un ``vocab.json`` minuscule écrit dans ``tmp_path`` :
aucun modèle, aucun téléchargement. Le dernier test, lui, tourne sur le
vrai ``vocab.json`` de Qwen3-0.6B s'il est présent dans le cache HF.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.core.vocab import Vocab

VOCAB = {
    "Hello": 0,
    "Ġworld": 1,
    "1": 2,
    "-": 3,
    ".": 4,
    ",": 5,
    "}": 6,
    '"': 7,
    "true": 8,
    "false": 9,
    "a\\": 10,
    "\\": 11,
}


def write_vocab(tmp_path: Path, payload: dict[str, int]) -> str:
    """Écrit *payload* comme vocab.json et retourne son chemin."""
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def fake_vocab(tmp_path: Path) -> Vocab:
    """Vocab de 11 tokens couvrant chiffre, signe, point, quote, etc."""
    return Vocab(vocab_path=write_vocab(tmp_path, VOCAB), n_logits=12)


# --------------------------------------------------------------------------
# Décodage des tokens : vocab.json est en byte-level (Ġ = espace).
# --------------------------------------------------------------------------


def test_ids_to_tokens_undoes_the_byte_level_encoding(tmp_path: Path) -> None:
    vocab = fake_vocab(tmp_path)

    assert vocab.ids_to_tokens[0] == "Hello"
    assert vocab.ids_to_tokens[1] == " world"


# --------------------------------------------------------------------------
# Masques : chaque id doit tomber dans la bonne catégorie, et une seule.
# --------------------------------------------------------------------------


def test_number_masks_point_to_the_right_ids(tmp_path: Path) -> None:
    vocab = fake_vocab(tmp_path)

    assert vocab.digit_mask is not None
    assert vocab.minus_mask is not None
    assert vocab.dot_mask is not None
    assert vocab.end_number_mask is not None
    assert list(np.flatnonzero(vocab.digit_mask)) == [2]
    assert list(np.flatnonzero(vocab.minus_mask)) == [3]
    assert list(np.flatnonzero(vocab.dot_mask)) == [4]
    assert list(np.flatnonzero(vocab.end_number_mask)) == [5, 6]


def test_string_tables_classify_each_token(tmp_path: Path) -> None:
    vocab = fake_vocab(tmp_path)
    assert vocab.str_valid is not None
    assert vocab.str_closed is not None
    assert vocab.str_escaped is not None

    # '"' ferme la chaîne à l'index 0 et reste un token valide
    assert bool(vocab.str_valid[0][7]) is True
    assert int(vocab.str_closed[0][7]) == 0
    # 'a\' laisse une séquence d'échappement ouverte
    assert bool(vocab.str_valid[0][10]) is True
    assert bool(vocab.str_escaped[0][10]) is True


# --------------------------------------------------------------------------
# Ids « spéciaux » : quote, true, false doivent venir du fichier.
# --------------------------------------------------------------------------


def test_quote_true_and_false_ids_come_from_the_vocab_file(
    tmp_path: Path,
) -> None:
    vocab = fake_vocab(tmp_path)

    assert vocab.quote_id == 7
    assert vocab.true_id == 8
    assert vocab.false_id == 9


# --------------------------------------------------------------------------
# Erreurs : un mauvais chemin doit produire un message lisible.
# --------------------------------------------------------------------------


def test_missing_vocab_file_gives_a_readable_error(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"

    with pytest.raises(Exception) as excinfo:
        Vocab(vocab_path=str(missing), n_logits=4)

    assert "nope.json" in str(excinfo.value)


# --------------------------------------------------------------------------
# Le vrai vocabulaire (intégration), seulement s'il est en cache.
# --------------------------------------------------------------------------


def find_real_vocab() -> Path | None:
    """Cherche le vocab.json de Qwen3-0.6B dans le cache HuggingFace."""
    roots = [os.environ.get("HF_HOME"), str(Path.home() / ".cache")]
    for root in roots:
        if not root:
            continue
        base = Path(root) / "huggingface" / "hub"
        pattern = "models--Qwen--Qwen3-0.6B/snapshots/*/vocab.json"
        for path in sorted(base.glob(pattern)):
            return path
    return None


REAL_VOCAB = find_real_vocab()


@pytest.mark.skipif(REAL_VOCAB is None, reason="vocab.json absent du cache")
def test_real_vocab_ids_are_contiguous_and_ascending() -> None:
    """Hypothèse de conception : l'id d'un token est son rang.

    ``_build_masks`` parcourt ``ids_to_tokens.values()`` avec
    ``enumerate`` : cela ne reste vrai que si les ids du fichier sont
    exactement 0, 1, 2, ... dans l'ordre du fichier.
    """
    assert REAL_VOCAB is not None
    raw = json.loads(REAL_VOCAB.read_text(encoding="utf-8"))
    ids = list(raw.values())

    assert ids == list(range(len(ids)))


@pytest.mark.skipif(REAL_VOCAB is None, reason="vocab.json absent du cache")
def test_real_vocab_special_ids_and_digit_mask() -> None:
    assert REAL_VOCAB is not None
    vocab = Vocab(vocab_path=str(REAL_VOCAB), n_logits=151936)

    assert vocab.ids_to_tokens[vocab.quote_id] == '"'
    assert vocab.ids_to_tokens[vocab.true_id] == "true"
    assert vocab.ids_to_tokens[vocab.false_id] == "false"
    assert vocab.digit_mask is not None
    for token_id in np.flatnonzero(vocab.digit_mask):
        assert vocab.ids_to_tokens[int(token_id)].isdigit()


# --------------------------------------------------------------------------
# Portabilité : un autre modèle doit échouer clairement, pas mentir.
# --------------------------------------------------------------------------

REQUIRED_SINGLE_TOKENS = ['"', "true", "false"]


@pytest.mark.parametrize("missing", REQUIRED_SINGLE_TOKENS)
def test_missing_special_token_is_a_clear_error(
    tmp_path: Path, missing: str
) -> None:
    """Sans ``true`` en un seul token, le décodeur booléen est faux.

    ``tokens_to_ids.get("true", 1866)`` renvoie silencieusement l'id de
    Qwen3 : avec un autre modèle, on masquerait un token au hasard. Il
    faut échouer au démarrage.
    """
    payload = {k: v for k, v in VOCAB.items() if k != missing}

    with pytest.raises(ValueError):
        Vocab(vocab_path=write_vocab(tmp_path, payload), n_logits=11)
