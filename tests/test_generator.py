"""Contrat de l'orchestration : une sortie par prompt, jamais de crash.

``Generator`` construit normalement un vrai décodeur (modèle + vocabulaire).
Pour tester son contrat sans modèle, on utilise une sous-classe dont
``model_post_init`` ne charge rien, et on injecte un faux décodeur.
"""
import json
import time
from pathlib import Path
from typing import Any

import pytest

from src.core.events import Change, Origin
from src.core.generator import Generator
from src.model.schema import (
    FnDefinition,
    FunctionCall,
    ParamSpec,
    PromptDefinition,
)
from src.model.metrics import CmmMetrics
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
        metrics=CmmMetrics(metrics_path=tmp_path / "metrics.json"),
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
    assert generator.errors_count == 1
    assert generator.results[0].name == "fn_greet"
    assert params(generator.results[1]) == {"a": 2.0, "b": 3.0}
    assert isinstance(params(generator.results[1])["a"], float)
    assert params(generator.results[2]) == {}


def test_invalid_json_falls_back(tmp_path: Path) -> None:
    decoder = FakeDecoder({"p": "not json"})
    generator = build_generator(decoder, ["p"], tmp_path)

    generator.run()

    assert generator.errors_count == 1
    assert generator.results[0].name == "fn_greet"
    assert params(generator.results[0]) == {}


def test_unknown_function_name_falls_back(tmp_path: Path) -> None:
    answer = '{"prompt": "p", "name": "fn_nope", "parameters": {}}'
    generator = build_generator(FakeDecoder({"p": answer}), ["p"], tmp_path)

    generator.run()

    assert generator.errors_count == 1
    assert params(generator.results[0]) == {}


def test_wrong_parameter_type_falls_back(tmp_path: Path) -> None:
    answer = (
        '{"prompt": "p", "name": "fn_add_numbers", '
        '"parameters": {"a": "two", "b": 3}}'
    )
    generator = build_generator(FakeDecoder({"p": answer}), ["p"], tmp_path)

    generator.run()

    assert generator.errors_count == 1


def test_extra_parameter_falls_back(tmp_path: Path) -> None:
    answer = (
        '{"prompt": "p", "name": "fn_greet", '
        '"parameters": {"name": "x", "extra": 1}}'
    )
    generator = build_generator(FakeDecoder({"p": answer}), ["p"], tmp_path)

    generator.run()

    assert generator.errors_count == 1


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

    assert generator.errors_count == 0
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

    assert generator.errors_count == 0


def _run_ready(tmp_path: Path) -> OfflineGenerator:
    """Generateur hors-ligne pret a emettre, avec le contexte du 2e prompt."""
    generator = build_generator(FakeDecoder({}), ["a", "b"], tmp_path)
    generator.prompt_count = len(generator.prompts)
    generator.run_index, generator.run_prompt = 1, "b"
    generator.run_cache_state = "miss"
    generator.run_started, generator.run_calls = time.time(), 0
    return generator


def test_forward_adds_the_run_context_to_the_change(tmp_path: Path) -> None:
    """Ce que le decodeur ignore (prompt, index, cache, s/token) vient d'ici.

    Sans ces champs, l'en-tete de la vue affiche ``0/0``, ``cache: off``
    et ``0.00 s/token`` : la generation marche, mais l'ecran ment.
    """
    generator = _run_ready(tmp_path)
    seen: list[Change] = []
    generator.on_change = seen.append

    generator._forward(
        Change(origin=Origin.MODEL, text="x", event_type="append")
    )

    change = seen[0]
    assert change.event_type == "append"
    assert change.origin is Origin.MODEL
    assert change.text == "x", "le contenu du decodeur ne doit pas etre perdu"
    assert change.prompt == "b"
    assert change.prompt_index == 2
    assert change.total_prompts == 2
    assert change.cache == "miss"
    assert change.spt >= 0.0


def test_emit_sends_a_prompt_start_frame(tmp_path: Path) -> None:
    """``prompt_start`` / ``prompt_end`` partent avec le meme contexte."""
    generator = _run_ready(tmp_path)
    seen: list[Change] = []
    generator.on_change = seen.append

    generator._emit("prompt_start")
    generator._emit("prompt_end")

    assert [c.event_type for c in seen] == ["prompt_start", "prompt_end"]
    assert all(c.prompt == "b" and c.total_prompts == 2 for c in seen)


def test_no_visualizer_means_no_emission(tmp_path: Path) -> None:
    """Sans ``on_change``, emettre ne doit rien faire, ni planter."""
    generator = _run_ready(tmp_path)
    generator.on_change = None

    generator._emit("prompt_start")
    generator._forward(Change(origin=Origin.MODEL, text="x"))
