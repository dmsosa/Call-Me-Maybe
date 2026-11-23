"""Le contrat des evenements, sans modele ni terminal.

``Change`` et ``Part`` sont des dataclasses : un defaut mutable
(``ids: list[int] = []``) fait echouer l'import du module des la collection.
"""
from src.core.events import Change, Origin, Part


def test_change_can_be_constructed_without_arguments() -> None:
    change = Change()

    assert change.event_type == "append"
    assert change.origin is Origin.SCAFFOLD
    assert not change.ids
    assert not change.top


def test_two_changes_do_not_share_a_mutable_default() -> None:
    first, second = Change(), Change()

    if isinstance(first.ids, list):
        first.ids.append(1)
        assert not second.ids, "defaut mutable partage entre deux Change"


def test_part_remembers_who_wrote_it_and_how_many_tokens() -> None:
    part = Part(3, Origin.MODEL, "greet", (124, 567))

    assert part.index == 3
    assert part.origin is Origin.MODEL
    assert part.text == "greet"
    assert part.n_tokens == 2
    assert Part(0, Origin.SCAFFOLD, "{").n_tokens == 0
