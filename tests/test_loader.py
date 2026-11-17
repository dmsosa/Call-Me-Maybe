"""Contrat du chargement des fichiers d'entrée.

Chaque test décrit UN comportement du loader. Les cas d'erreur vérifient
qu'une ``InputError`` est levée AVEC un message exploitable : jamais un
traceback brut.

L'API testée est celle de ``src.storage.json_loader`` :
``load_function_list`` et ``load_prompts``.
"""
import json
from pathlib import Path
from typing import Any

import pytest

from src.exception.exception import InputError
from src.model.schema import FnDefinition
from src.storage.json_loader import (
    FunctionList,
    load_function_list,
    load_prompts,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "input"
FUNCTIONS = DATA / "functions_definition.json"
PROMPTS = DATA / "function_calling_tests.json"


def write_json(tmp_path: Path, name: str, payload: Any) -> Path:
    """Écrit *payload* en JSON dans tmp_path et retourne le chemin."""
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def by_name(functions: FunctionList) -> dict[str, FnDefinition]:
    """Index nom -> définition (le loader ne fournit que la liste)."""
    return {fn.name: fn for fn in functions.root}


# --------------------------------------------------------------------------
# Cas nominaux : le fichier fourni doit se charger sans erreur.
# --------------------------------------------------------------------------


def test_load_functions_nominal() -> None:
    functions = load_function_list(str(FUNCTIONS))
    assert len(functions.root) == 5
    assert "fn_greet" in by_name(functions)


def test_load_functions_keeps_parameter_names_and_types() -> None:
    functions = by_name(load_function_list(str(FUNCTIONS)))
    params = functions["fn_substitute_string_with_regex"].parameters
    assert list(params) == ["source_string", "regex", "replacement"]
    assert params["regex"].type == "string"
    assert functions["fn_add_numbers"].parameters["a"].type == "number"


def test_load_prompts_nominal() -> None:
    prompts = load_prompts(str(PROMPTS))
    assert len(prompts.root) == 11
    assert prompts.root[0].prompt == "What is the sum of 2 and 3?"


def test_loading_twice_gives_the_same_result() -> None:
    assert len(load_prompts(str(PROMPTS)).root) == 11
    assert len(load_prompts(str(PROMPTS)).root) == 11


# --------------------------------------------------------------------------
# Cas d'erreur : le message doit dire OU est le problème.
# --------------------------------------------------------------------------


def test_missing_file_names_the_path() -> None:
    with pytest.raises(InputError, match="nope.json"):
        load_function_list("nope.json")


def test_invalid_json_reports_line_and_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"name": }', encoding="utf-8")
    with pytest.raises(InputError, match="line 1"):
        load_function_list(str(path))


def test_root_must_be_a_list(tmp_path: Path) -> None:
    path = write_json(tmp_path, "objet.json", {"a": 1})
    with pytest.raises(InputError, match="schema"):
        load_function_list(str(path))


def test_duplicate_function_names(tmp_path: Path) -> None:
    fn = {
        "name": "fn_x",
        "description": "",
        "parameters": {},
        "returns": {"type": "number"},
    }
    path = write_json(tmp_path, "dupes.json", [fn, fn])
    with pytest.raises(InputError, match="fn_x"):
        load_function_list(str(path))


def test_unknown_parameter_type_names_the_field(tmp_path: Path) -> None:
    fn = {"name": "fn_x", "parameters": {"x": {"type": "array"}}}
    path = write_json(tmp_path, "type.json", [fn])
    with pytest.raises(InputError, match=r"parameters\.x\.type"):
        load_function_list(str(path))


def test_function_name_must_be_snake_case(tmp_path: Path) -> None:
    path = write_json(tmp_path, "nom.json", [{"name": "Fn Greet"}])
    with pytest.raises(InputError, match=r"0\.name"):
        load_function_list(str(path))


def test_prompt_must_be_a_string(tmp_path: Path) -> None:
    path = write_json(tmp_path, "prompt.json", [{"prompt": 42}])
    with pytest.raises(InputError, match=r"0\.prompt"):
        load_prompts(str(path))


def test_empty_prompt_is_rejected(tmp_path: Path) -> None:
    path = write_json(tmp_path, "vide.json", [{"prompt": "   "}])
    with pytest.raises(InputError, match=r"0\.prompt"):
        load_prompts(str(path))


def test_prompts_loader_rejects_the_functions_file() -> None:
    with pytest.raises(InputError, match="schema"):
        load_prompts(str(FUNCTIONS))


def test_directory_instead_of_file() -> None:
    with pytest.raises(InputError, match="data/input"):
        load_function_list(str(DATA))


def test_invalid_utf8_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "binaire.json"
    path.write_bytes(b"\xff\xfe\x00\x00")

    with pytest.raises(InputError, match="[Uu]nicode"):
        load_function_list(str(path))


def test_loader_strips_the_prompt(tmp_path: Path) -> None:
    """Caractéristique assumée : les espaces autour sont retirés."""
    path = write_json(tmp_path, "p.json", [{"prompt": "  hello  "}])

    loaded = load_prompts(str(path))

    assert loaded.root[0].prompt == "hello"
