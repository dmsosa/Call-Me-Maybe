"""Contrat du fichier de sortie (chapitre V.4 du sujet)."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.exception.exception import OutputError
from src.storage.json_writer import JsonWriter
from src.storage.schema import FunctionCall, OutputList


def make_results() -> OutputList:
    return OutputList(
        [
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
    )


def test_creates_missing_parent_directories(tmp_path: Path) -> None:
    out = tmp_path / "data" / "output" / "results.json"
    writer = JsonWriter(output_path=out, output_payload=make_results())
    writer.export_json()
    assert out.is_file()


def test_export_returns_the_written_path(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    written = JsonWriter(
        output_path=out, output_payload=make_results()
    ).export_json()
    assert written == out


def test_output_is_valid_json_with_exactly_three_keys(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    JsonWriter(output_path=out, output_payload=make_results()).export_json()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 2
    for item in data:
        assert set(item) == {"prompt", "name", "parameters"}


def test_values_and_types_are_preserved(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    JsonWriter(output_path=out, output_payload=make_results()).export_json()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data[0]["parameters"]["name"] == "shrek"
    assert data[1]["parameters"]["a"] == 16.0
    assert isinstance(data[1]["parameters"]["a"], float)


def test_unicode_is_not_escaped(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    results = OutputList(
        [
            FunctionCall(
                prompt="Résumé à l'écran",
                name="fn_reverse_string",
                parameters={"s": "héllo"},
            )
        ]
    )
    JsonWriter(output_path=out, output_payload=results).export_json()
    raw = out.read_text(encoding="utf-8")
    assert "é" in raw
    assert "\\u" not in raw


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
    writer = JsonWriter(output_path=out, output_payload=make_results())
    with pytest.raises(OutputError, match="impossible d'écrire"):
        writer.export_json()


def test_set_output_payload_replaces_the_content(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    writer = JsonWriter(output_path=out, output_payload=make_results())
    writer.set_output_payload(
        [
            {
                "prompt": "p",
                "name": "fn_greet",
                "parameters": {"name": "x"},
            }
        ]
    )
    writer.export_json()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["name"] == "fn_greet"


def test_set_output_payload_rejects_an_invalid_payload(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    writer = JsonWriter(output_path=out, output_payload=make_results())
    with pytest.raises(OutputError, match="schema"):
        writer.set_output_payload([{"prompt": "p", "name": "fn_x"}])
    # le payload précédent doit rester intact
    assert len(writer.output_payload.root) == 2
