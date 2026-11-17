"""Contrat du fichier de sortie (chapitre V.4 du sujet).

Le contrat actuel de ``JsonWriter`` est :

* ``JsonWriter(output_path=...)`` ;
* ``export_json(function_call_list)`` écrit la liste et ne retourne rien ;
* le fichier contient une liste d'objets à exactement trois clés :
  ``prompt``, ``name``, ``parameters``.
"""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.exception.exception import OutputError
from src.model.schema import FunctionCall
from src.storage.json_writer import JsonWriter


def make_results() -> list[FunctionCall]:
    return [
        FunctionCall(
            prompt="Greet shrek",
            name="fn_greet",
            parameters={"name": "shrek"},
        ),
        FunctionCall(
            prompt="What is the square root of 16?",
            name="fn_get_square_root",
            parameters={"a": 16.0},
        ),
    ]


def test_creates_missing_parent_directories(tmp_path: Path) -> None:
    out = tmp_path / "data" / "output" / "results.json"
    JsonWriter(output_path=out).export_json(make_results())
    assert out.is_file()


def test_export_writes_the_file(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    JsonWriter(output_path=out).export_json(make_results())
    assert out.is_file()
    assert json.loads(out.read_text(encoding="utf-8")) != []


def test_output_is_valid_json_with_exactly_three_keys(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    JsonWriter(output_path=out).export_json(make_results())
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 2
    for item in data:
        assert set(item) == {"prompt", "name", "parameters"}


def test_values_and_types_are_preserved(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    JsonWriter(output_path=out).export_json(make_results())
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data[0]["parameters"]["name"] == "shrek"
    assert data[1]["parameters"]["a"] == 16.0
    assert isinstance(data[1]["parameters"]["a"], float)


def test_unicode_is_not_escaped(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    results = [
        FunctionCall(
            prompt="Résumé à l'écran",
            name="fn_reverse_string",
            parameters={"s": "héllo"},
        )
    ]
    JsonWriter(output_path=out).export_json(results)
    raw = out.read_text(encoding="utf-8")
    assert "é" in raw
    assert "\\u" not in raw


def test_export_overwrites_an_existing_file(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    out.write_text("contenu précédent", encoding="utf-8")
    JsonWriter(output_path=out).export_json(make_results()[:1])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["name"] == "fn_greet"


def test_extra_key_in_a_call_is_rejected() -> None:
    with pytest.raises(ValidationError):
        FunctionCall(
            prompt="p",
            name="fn_x",
            parameters={},
            inutile=1,  # type: ignore[call-arg]
        )


def test_unwritable_destination_raises_output_error(tmp_path: Path) -> None:
    blocker = tmp_path / "fichier"
    blocker.write_text("", encoding="utf-8")
    out = blocker / "results.json"  # parent = un fichier, pas un dossier
    writer = JsonWriter(output_path=out)
    with pytest.raises(OutputError, match="results.json"):
        writer.export_json(make_results())


# --------------------------------------------------------------------------
# Le schéma de sortie : trois champs obligatoires, jamais null.
# --------------------------------------------------------------------------


def test_function_call_requires_all_three_fields() -> None:
    with pytest.raises(ValidationError):
        FunctionCall.model_validate({"prompt": "p", "name": "fn_x"})

    with pytest.raises(ValidationError):
        FunctionCall.model_validate(
            {"prompt": "p", "name": "fn_x", "parameters": None}
        )


def test_export_has_no_null_values(tmp_path: Path) -> None:
    out = tmp_path / "results.json"

    JsonWriter(output_path=out).export_json(make_results())

    raw = out.read_text(encoding="utf-8")
    assert "null" not in raw
