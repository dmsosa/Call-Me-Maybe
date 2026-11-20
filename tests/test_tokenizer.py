"""Contrat du tokenizer maison : ``encode``/``decode`` sans le SDK.

Deux niveaux :

* des tests **synthétiques** (un ``vocab.json`` + ``merges.txt`` de
  quelques tokens écrits dans ``tmp_path``) qui fixent l'algorithme :
  pré-tokenisation, application des merges par rang, aller-retour ;
* des tests **d'intégration** qui comparent ton tokenizer au tokenizer
  de référence sur le vrai vocabulaire de Qwen3-0.6B (présent dans le
  cache HF), sur les 11 prompts du projet et sur les morceaux
  structurels que le décodeur encode.

Tant que ``src/core/tokenizer.py`` ne s'importe pas, seul
``test_the_module_imports`` échoue et les autres sont *skipped* avec la
raison exacte : pas besoin de lire le code pour savoir où on en est.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

# --------------------------------------------------------------------------
# Import tolérant : le module est encore un brouillon.
# --------------------------------------------------------------------------

IMPORT_ERROR: BaseException | None = None
try:
    from src.core.tokenizer import PersonalTokenizer
except BaseException as exc:  # noqa: BLE001 - on veut le message, pas le type
    PersonalTokenizer = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc

SKIP = pytest.mark.skipif(
    PersonalTokenizer is None,
    reason=f"src/core/tokenizer.py ne s'importe pas : {IMPORT_ERROR!r}",
)


# --------------------------------------------------------------------------
# Le vrai tokenizer de référence, si le modèle est en cache.
# --------------------------------------------------------------------------


def find_snapshot() -> Path | None:
    """Dossier snapshot Qwen3-0.6B qui contient vocab.json + merges.txt."""
    roots = [os.environ.get("HF_HOME"), str(Path.home() / ".cache")]
    for root in roots:
        if not root:
            continue
        base = Path(root) / "huggingface" / "hub"
        pattern = "models--Qwen--Qwen3-0.6B/snapshots/*/vocab.json"
        for path in sorted(base.glob(pattern)):
            snapshot = path.parent
            if (snapshot / "merges.txt").is_file():
                return snapshot
    return None


SNAPSHOT = find_snapshot()
NEEDS_REAL_FILES = pytest.mark.skipif(
    SNAPSHOT is None, reason="snapshot Qwen3-0.6B absent du cache HF"
)


def snapshot_dir() -> Path:
    """``SNAPSHOT`` non optionnel (mypy ne voit pas le narrowing)."""
    assert SNAPSHOT is not None
    return SNAPSHOT


def _encode_is_ready() -> bool:
    """Vrai quand ``encode`` répond au moins sur un caractère."""
    if PersonalTokenizer is None or SNAPSHOT is None:
        return False
    try:
        ids = PersonalTokenizer(
            vocab_path=str(snapshot_dir() / "vocab.json"),
            merges_path=str(snapshot_dir() / "merges.txt"),
        ).encode("a")
    except Exception:  # noqa: BLE001 - tout échec = pas prêt
        return False
    return isinstance(ids, list) and all(isinstance(i, int) for i in ids)


NEEDS_ENCODE = pytest.mark.skipif(
    not _encode_is_ready(),
    reason="encode pas encore fonctionnel (J2 à J5)",
)


@pytest.fixture(scope="module")
def reference() -> Any:
    """Tokenizer officiel (transformers), chargé depuis le disque."""
    if SNAPSHOT is None:
        pytest.skip("snapshot absent")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(str(snapshot_dir()))


@pytest.fixture(scope="module")
def personal() -> Any:
    """Ton tokenizer, sur les vrais fichiers."""
    if SNAPSHOT is None:
        pytest.skip("snapshot absent")
    assert PersonalTokenizer is not None
    return PersonalTokenizer(
        vocab_path=str(snapshot_dir() / "vocab.json"),
        merges_path=str(snapshot_dir() / "merges.txt"),
    )


def project_prompts() -> list[str]:
    data = json.loads(
        Path("data/input/function_calling_tests.json").read_text(
            encoding="utf-8"
        )
    )
    return [item["prompt"] for item in data]


# --------------------------------------------------------------------------
# 0. Le module existe et expose les deux méthodes demandées.
# --------------------------------------------------------------------------


def test_the_module_imports_and_exposes_encode_decode() -> None:
    assert PersonalTokenizer is not None, IMPORT_ERROR
    assert callable(PersonalTokenizer.encode)
    assert callable(PersonalTokenizer.decode)


# --------------------------------------------------------------------------
# 1. Tests synthétiques : l'algorithme, sans le vrai vocabulaire.
# --------------------------------------------------------------------------


def gpt2_byte_to_char() -> dict[int, str]:
    """Mapping octet -> caractère « sûr » de GPT-2 (Ġ = espace, Ċ = \\n)."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    chars = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(printable, map(chr, chars)))


def tiny_tokenizer(tmp_path: Path) -> object:
    """Vocabulaire de 256 jetons + 2 merges (« a b », puis « ab c »)."""
    byte_to_char = gpt2_byte_to_char()
    vocab = {char: index for index, char in enumerate(byte_to_char.values())}
    vocab["ab"] = len(vocab)
    vocab["abc"] = len(vocab)
    vocab["Ġb"] = len(vocab)

    (tmp_path / "vocab.json").write_text(
        json.dumps(vocab), encoding="utf-8"
    )
    (tmp_path / "merges.txt").write_text(
        "#version: 0.2\na b\nab c\nĠ b\n", encoding="utf-8"
    )
    assert PersonalTokenizer is not None
    return PersonalTokenizer(
        vocab_path=str(tmp_path / "vocab.json"),
        merges_path=str(tmp_path / "merges.txt"),
    )


def test_synthetic_merges_are_applied_in_rank_order(tmp_path: Path) -> None:
    """« abc » = (((a,b),c) : il faut deux passes, dans l'ordre du fichier."""
    tokenizer = tiny_tokenizer(tmp_path)
    vocab = json.loads((tmp_path / "vocab.json").read_text(encoding="utf-8"))

    encoded = tokenizer.encode("abc")  # type: ignore[attr-defined]

    assert encoded == [vocab["abc"]]


def test_synthetic_unmerged_characters_stay_separate(tmp_path: Path) -> None:
    tokenizer = tiny_tokenizer(tmp_path)
    vocab = json.loads((tmp_path / "vocab.json").read_text(encoding="utf-8"))

    assert tokenizer.encode("ac") == [  # type: ignore[attr-defined]
        vocab["a"],
        vocab["c"],
    ]


def test_synthetic_space_goes_through_the_byte_mapping(tmp_path: Path) -> None:
    """Un espace ne s'encode pas « espace » mais « Ġ » (byte-level)."""
    tokenizer = tiny_tokenizer(tmp_path)
    vocab = json.loads((tmp_path / "vocab.json").read_text(encoding="utf-8"))

    assert tokenizer.encode("a b") == [  # type: ignore[attr-defined]
        vocab["a"],
        vocab["Ġb"],
    ]


def test_synthetic_round_trip(tmp_path: Path) -> None:
    tokenizer = tiny_tokenizer(tmp_path)

    for text in ("", "abc", "a b", "cab", "ac\n", "🙂"):
        ids = tokenizer.encode(text)  # type: ignore[attr-defined]

        if text.isascii():
            assert tokenizer.decode(ids) == text  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# 2. Intégration : mêmes ids que le tokenizer officiel.
# --------------------------------------------------------------------------


@NEEDS_REAL_FILES
@NEEDS_ENCODE
def test_encode_matches_the_reference_on_the_project_prompts(
    personal: Any, reference: Any
) -> None:
    for prompt in project_prompts():
        assert personal.encode(prompt) == reference.encode(
            prompt, add_special_tokens=False
        ), prompt


@NEEDS_REAL_FILES
@NEEDS_ENCODE
def test_encode_matches_the_reference_on_structural_chunks(
    personal: Any, reference: Any
) -> None:
    """Les morceaux que le décodeur encode lui-même, un par un."""
    chunks = [
        '{"prompt": "',
        '", "name": "',
        '", "parameters": {',
        "}",
        '"',
        "\\",
        "true",
        "false",
        "fn_add_numbers",
        "0",
        "-12",
        "3.5",
        " ",
        "\n",
    ]

    for chunk in chunks:
        assert personal.encode(chunk) == reference.encode(
            chunk, add_special_tokens=False
        ), repr(chunk)


@NEEDS_REAL_FILES
@NEEDS_ENCODE
def test_encode_does_not_add_special_tokens(personal: Any) -> None:
    """``encode`` doit être « brut » : pas de <|im_start|> ajouté."""
    ids = personal.encode("What is the sum of 2 and 3?")

    assert 151644 not in ids  # <|im_start|>
    assert 151645 not in ids  # <|im_end|>


@NEEDS_REAL_FILES
@NEEDS_ENCODE
def test_decode_inverts_encode_on_the_project_prompts(
    personal: Any, reference: Any
) -> None:
    for prompt in project_prompts():
        assert personal.decode(personal.encode(prompt)) == prompt


@NEEDS_REAL_FILES
@NEEDS_ENCODE
def test_decode_matches_the_reference(
    personal: Any, reference: Any
) -> None:
    for text in ("café", "中文", "a\tb", "🙂 emoji", ""):
        ids = reference.encode(text, add_special_tokens=False)

        assert personal.decode(list(ids)) == reference.decode(
            ids, skip_special_tokens=True
        ), text


@NEEDS_REAL_FILES
@NEEDS_ENCODE
def test_every_id_is_inside_the_vocab(personal: Any) -> None:
    raw = (snapshot_dir() / "vocab.json").read_text(encoding="utf-8")
    vocab = json.loads(raw)
    size = len(vocab)

    for prompt in project_prompts():
        ids = personal.encode(prompt)
        assert all(0 <= token_id < size for token_id in ids)


@NEEDS_REAL_FILES
@NEEDS_ENCODE
def test_missing_merges_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(Exception) as excinfo:
        PersonalTokenizer(
            vocab_path=str(snapshot_dir() / "vocab.json"),
            merges_path=str(tmp_path / "nope.txt"),
        )

    assert "nope.txt" in str(excinfo.value)


@NEEDS_REAL_FILES
@NEEDS_ENCODE
@pytest.mark.xfail(
    strict=False,
    reason="tokens ajoutés (<|im_start|>...) : ils ne viennent pas de BPE "
    "mais de tokenizer.json (26 ids 151643+)",
)
def test_added_tokens_are_out_of_scope(
    personal: Any, reference: Any
) -> None:
    """``encode`` pur BPE ne peut pas produire ``<|im_start|>`` (id 151644).

    Le contexte du décodeur en contient 7 occurrences : sans support des
    tokens ajoutés, le contexte fait 121 ids au lieu de 98.
    """
    assert personal.encode("<|im_start|>") == reference.encode(
        "<|im_start|>", add_special_tokens=False
    )
