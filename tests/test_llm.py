"""Contrat du wrapper LLM : listes plates, logits mémoïsés.

``LargeLanguageModel`` ne dépend du SDK que par quatre méthodes : on lui
donne donc un faux modèle, sans torch ni poids.
"""
import numpy as np

from src.core.llm import LargeLanguageModel


class FakeTensor(list):
    """Imite le tensor 1-D renvoyé par le SDK (``.tolist()``)."""

    def tolist(self) -> list[int]:
        return list(self)


class FakeModel:
    """Imite ``Small_LLM_Model`` : ``encode`` renvoie une pseudo-batch."""

    def __init__(self, size: int = 5) -> None:
        self.size = size
        self.calls = 0

    def encode(self, text: str) -> list[FakeTensor]:
        return [FakeTensor(ord(char) for char in text)]

    def get_logits_from_input_ids(self, ids: list[int]) -> list[float]:
        self.calls += 1
        return [float(len(ids))] * self.size

    def get_path_to_vocab_file(self) -> str:
        return "vocab.json"

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in ids)


def build_llm() -> tuple[LargeLanguageModel, FakeModel]:
    """Construit le wrapper autour du faux modèle."""
    fake = FakeModel()
    return LargeLanguageModel.model_construct(model=fake), fake


def test_encode_returns_a_flat_list_of_ids() -> None:
    llm, _fake = build_llm()

    assert llm.encode("ab") == [97, 98]


def test_decode_is_the_inverse_of_encode() -> None:
    llm, _fake = build_llm()

    assert llm.decode(llm.encode("hello")) == "hello"


def test_logits_are_memoized_on_the_sequence() -> None:
    llm, fake = build_llm()

    first = llm.logits([1, 2, 3])
    second = llm.logits([1, 2, 3])

    assert fake.calls == 1
    assert llm.calls == 1
    assert np.array_equal(first, second)


def test_two_different_sequences_are_two_model_calls() -> None:
    llm, fake = build_llm()

    llm.logits([1, 2])
    llm.logits([2, 1])

    assert fake.calls == 2
    assert llm.calls == 2


def test_logits_are_float32_arrays() -> None:
    llm, _fake = build_llm()

    logits = llm.logits([1])

    assert isinstance(logits, np.ndarray)
    assert logits.dtype == np.float32
    assert logits.shape == (5,)


def test_vocab_path_is_a_string() -> None:
    llm, _fake = build_llm()

    assert llm.vocab_path() == "vocab.json"
