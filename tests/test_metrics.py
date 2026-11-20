"""Contrat des métriques : agrégats calculés, fichier écrit sans lever.

Les agrégats (``total_calls``, ``cache_hits``, ``successes``, ``errors``)
sont des champs **calculés** : ils ne doivent jamais pouvoir diverger de
la liste ``prompts``.
"""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.model.metrics import CmmMetrics, PromptMetrics


def metrics(tmp_path: Path) -> CmmMetrics:
    """CmmMetrics pointant sur un fichier jetable.

    Le champ est un ``Path`` (``write`` appelle ``.with_name`` et
    ``.parent.mkdir``, qui n'existent que sur ``Path``) : on le
    construit donc avec un ``Path``, jamais avec une ``str``.
    """
    return CmmMetrics(metrics_path=tmp_path / "metrics.json")


def entry(
    index: int = 0,
    *,
    calls: int = 3,
    error: str | None = None,
    cache: str = "miss",
) -> PromptMetrics:
    return PromptMetrics(
        index=index,
        prompt="p",
        name="fn_add",
        parameters={"a": 1.0},
        calls=calls,
        duration_s=0.5,
        cache=cache,  # type: ignore[arg-type]
        error=error,
    )


def test_empty_metrics_have_zero_aggregates(tmp_path: Path) -> None:
    m = metrics(tmp_path)

    assert m.prompts == []
    assert (m.total_calls, m.cache_hits, m.successes, m.errors) == (0, 0, 0, 0)


def test_aggregates_are_computed_from_the_prompts(tmp_path: Path) -> None:
    m = metrics(tmp_path)
    m.add_prompt(entry(0, calls=3, cache="hit"))
    m.add_prompt(entry(1, calls=4, error="boom"))
    m.add_prompt(entry(2, calls=5))

    assert m.total_calls == 12
    assert m.cache_hits == 1
    assert m.successes == 2
    assert m.errors == 1


def test_successes_plus_errors_always_equals_the_prompt_count(
    tmp_path: Path,
) -> None:
    m = metrics(tmp_path)

    for index in range(5):
        m.add_prompt(entry(index, error=None if index % 2 else "boom"))
        assert m.successes + m.errors == len(m.prompts)


def test_dump_has_exactly_the_expected_keys(tmp_path: Path) -> None:
    m = metrics(tmp_path)
    m.add_prompt(entry())

    assert list(m.model_dump()) == [
        "metrics_path",
        "started_at",
        "model_load_s",
        "total_s",
        "prompts",
        "total_calls",
        "cache_hits",
        "successes",
        "errors",
    ]


def test_cache_field_rejects_unknown_values(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        metrics(tmp_path).add_prompt(entry(cache="HIT"))


def test_computed_fields_cannot_be_passed_as_input() -> None:
    with pytest.raises(ValidationError):
        CmmMetrics.model_validate({"total_calls": 3})


# ------------------------------------------------------- écriture du fichier


def test_metrics_path_accepts_a_path_object(tmp_path: Path) -> None:
    """``write`` utilise ``Path.with_name`` : le champ doit être un Path."""
    m = CmmMetrics(metrics_path=tmp_path / "metrics.json")

    assert isinstance(m.metrics_path, Path)


def test_write_creates_a_reloadable_utf8_file(tmp_path: Path) -> None:
    m = metrics(tmp_path)
    m.add_prompt(
        PromptMetrics(index=0, prompt="résume l'été", name="fn_x", calls=7)
    )
    m.total_s = 1.5

    m.write()

    text = (tmp_path / "metrics.json").read_text(encoding="utf-8")
    raw = json.loads(text)
    assert raw["total_calls"] == 7
    assert raw["total_s"] == 1.5
    assert "résume l'été" in text
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_write_never_raises_on_an_impossible_path() -> None:
    m = CmmMetrics(
        metrics_path=Path("/proc/definitely/not/writable/metrics.json")
    )
    m.add_prompt(entry())

    m.write()          # doit seulement avertir sur stderr


def test_metrics_can_be_built_without_arguments() -> None:
    """``Generator.metrics`` a ``default_factory=CmmMetrics``.

    La fabrique appelle ``CmmMetrics()`` : si ``metrics_path`` est requis,
    aucun ``Generator`` ne peut être construit sans le passer à la main.
    """
    m = CmmMetrics()

    assert m.prompts == []
