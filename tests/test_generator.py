"""Contrat de l'orchestration : une sortie par prompt, jamais de crash.

``Generator`` construit normalement un vrai décodeur (modèle + vocabulaire).
Pour tester son contrat sans modèle, on utilise une sous-classe dont
``model_post_init`` ne charge rien, et on injecte un faux décodeur.
"""
import json
from pathlib import Path
from typing import Any

import pytest

from src.core.generator import Generator
from src.model.schema import (
    FnDefinition,
    FunctionCall,
    ParamSpec,
    PromptDefinition,
)
from src.storage.json_writer import JsonWriter

GREET = (
    '{"prompt": "Greet shrek", "name": "fn_greet", '
    '"parameters": {"name": "shrek"}}'
)
SUM = (
    '{"prompt": "Sum", "name": "fn_add_numbers", '
    '"parameters": {"a": 2, "b": 3}}'
)


class FakeCalls:
    """Compteur d'appels au modèle (affiché par ``run``)."""

    def __init__(self) -> None:
        self.calls = 0


class FakeDecoder:
    """Décodeur scripté : une réponse (ou une exception) par prompt."""

    def __init__(self, answers: dict[str, object]) -> None:
        self.answers = answers
        self.seen: list[str] = []

    def generate_json(self, prompt: str) -> str:
        self.seen.append(prompt)
        answer = self.answers[prompt]
        if isinstance(answer, Exception):
            raise answer
        assert isinstance(answer, str)
        return answer


class OfflineGenerator(Generator):
    """Generator sans modèle : le décodeur est fourni de l'extérieur."""

    def model_post_init(self, context: Any) -> None:
        self._store_fn_names()


def build_generator(
    decoder: FakeDecoder,
    prompts: list[str],
    tmp_path: Path,
    functions: list[FnDefinition] | None = None,
) -> OfflineGenerator:
    """Construit un generateur hors-ligne sur *prompts*."""
    if functions is None:
        functions = [
            FnDefinition(
                name="fn_greet",
                parameters={"name": ParamSpec(type="string")},
            ),
            FnDefinition(
                name="fn_add_numbers",
                parameters={
                    "a": ParamSpec(type="number"),
                    "b": ParamSpec(type="number"),
                },
            ),
        ]
    return OfflineGenerator.model_construct(
        llm=FakeCalls(),
        fn_list=functions,
        prompts=[PromptDefinition(prompt=p) for p in prompts],
        writer=JsonWriter(output_path=tmp_path / "results.json"),
        decoder=decoder,
    )


def params(call: FunctionCall) -> dict[str, object]:
    """Paramètres de *call*, jamais None."""
    return call.parameters or {}


def test_one_entry_per_prompt_in_order(tmp_path: Path) -> None:
    prompts = ["Greet shrek", "Sum", "boom"]
    decoder = FakeDecoder(
        {
            "Greet shrek": GREET,
            "Sum": SUM,
            "boom": RuntimeError("model exploded"),
        }
    )
    generator = build_generator(decoder, prompts, tmp_path)

    generator.run()

    assert [call.prompt for call in generator.results] == prompts
    assert decoder.seen == prompts
    assert generator.errors == 1
    assert generator.results[0].name == "fn_greet"
    assert params(generator.results[1]) == {"a": 2.0, "b": 3.0}
    assert isinstance(params(generator.results[1])["a"], float)
    assert params(generator.results[2]) == {}


def test_invalid_json_falls_back(tmp_path: Path) -> None:
    decoder = FakeDecoder({"p": "not json"})
    generator = build_generator(decoder, ["p"], tmp_path)

    generator.run()

    assert generator.errors == 1
    assert generator.results[0].name == "fn_greet"
    assert params(generator.results[0]) == {}


def test_unknown_function_name_falls_back(tmp_path: Path) -> None:
    answer = '{"prompt": "p", "name": "fn_nope", "parameters": {}}'
    generator = build_generator(FakeDecoder({"p": answer}), ["p"], tmp_path)

    generator.run()

    assert generator.errors == 1
    assert params(generator.results[0]) == {}


def test_wrong_parameter_type_falls_back(tmp_path: Path) -> None:
    answer = (
        '{"prompt": "p", "name": "fn_add_numbers", '
        '"parameters": {"a": "two", "b": 3}}'
    )
    generator = build_generator(FakeDecoder({"p": answer}), ["p"], tmp_path)

    generator.run()

    assert generator.errors == 1


def test_extra_parameter_falls_back(tmp_path: Path) -> None:
    answer = (
        '{"prompt": "p", "name": "fn_greet", '
        '"parameters": {"name": "x", "extra": 1}}'
    )
    generator = build_generator(FakeDecoder({"p": answer}), ["p"], tmp_path)

    generator.run()

    assert generator.errors == 1


def test_only_the_provided_parameters_are_checked(tmp_path: Path) -> None:
    """Choix du projet : seuls les paramètres fournis sont validés.

    Le décodeur ne génère que ce que le modèle écrit, donc un paramètre
    déclaré mais absent n'est pas une erreur (caractéristique assumée).
    """
    answer = (
        '{"prompt": "p", "name": "fn_add_numbers", "parameters": {"a": 2}}'
    )
    generator = build_generator(FakeDecoder({"p": answer}), ["p"], tmp_path)

    generator.run()

    assert generator.errors == 0
    assert params(generator.results[0]) == {"a": 2.0}


def test_output_file_always_matches_the_prompts(tmp_path: Path) -> None:
    prompts = ["Greet shrek", "boom"]
    decoder = FakeDecoder(
        {"Greet shrek": GREET, "boom": RuntimeError("model exploded")}
    )
    generator = build_generator(decoder, prompts, tmp_path)

    generator.run()

    raw = (tmp_path / "results.json").read_text(encoding="utf-8")
    written = json.loads(raw)
    assert [item["prompt"] for item in written] == prompts
    assert set(written[0]) == {"prompt", "name", "parameters"}
    assert written[0]["parameters"] == {"name": "shrek"}


def test_fallback_needs_at_least_one_function(tmp_path: Path) -> None:
    """Caractéristique assumée : sans fonction, le fallback n'a rien."""
    decoder = FakeDecoder({"p": "not json"})
    generator = build_generator(decoder, ["p"], tmp_path, functions=[])

    with pytest.raises(StopIteration):
        generator.run()


def test_unwritable_output_does_not_crash_the_run(tmp_path: Path) -> None:
    """Un dossier de sortie impossible ne doit pas interrompre le run."""
    blocker = tmp_path / "fichier"
    blocker.write_text("", encoding="utf-8")
    generator = build_generator(FakeDecoder({"p": GREET}), ["p"], tmp_path)
    generator.writer = JsonWriter(output_path=blocker / "out.json")

    generator.run()

    assert generator.errors == 0
