"""Contrat de la ligne de commande (sans charger le modèle)."""
import sys
import time
from pathlib import Path

import pytest

from src.__main__ import init_argparse, main


def test_argparse_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog"])

    args = init_argparse()

    assert args.functions_definition.endswith("functions_definition.json")
    assert args.input.endswith("function_calling_tests.json")
    assert args.output.endswith("function_calling_results.json")


def test_argparse_reads_the_command_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "--output", "custom.json"])

    assert init_argparse().output == "custom.json"


def test_main_refuses_an_empty_function_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Zéro fonction = rien à appeler : le CLI s'arrête avant le modèle."""
    functions = tmp_path / "functions.json"
    functions.write_text("[]", encoding="utf-8")
    prompts = tmp_path / "prompts.json"
    prompts.write_text('[{"prompt": "hello"}]', encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--functions_definition",
            str(functions),
            "--input",
            str(prompts),
        ],
    )

    code = main()

    assert code == 1
    assert "Zero (0) functions" in capsys.readouterr().err


def test_argparse_exposes_the_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["prog"])

    assert init_argparse().model == "Qwen/Qwen3-0.6B"


def test_argparse_reads_an_alternative_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "--model", "Qwen/Qwen3-1.7B"])

    assert init_argparse().model == "Qwen/Qwen3-1.7B"


def test_main_prints_the_summary_without_a_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Le resume final s'appuie sur errors_count : il ne doit pas planter."""
    import types

    functions = tmp_path / "functions.json"
    functions.write_text('[{"name": "fn_ping"}]', encoding="utf-8")
    prompts = tmp_path / "prompts.json"
    prompts.write_text('[{"prompt": "ping"}]', encoding="utf-8")

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            pass

    fake_sdk = types.ModuleType("llm_sdk")
    fake_sdk.Small_LLM_Model = FakeModel          # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llm_sdk", fake_sdk)
    monkeypatch.setattr(
        "src.core.llm.LargeLanguageModel", lambda model: object()
    )

    class FakeGenerator:
        def __init__(self, **kwargs: object) -> None:
            self.results: list[object] = []
            self.errors_count = 0
            self.metrics = type(
                "M", (), {"started_at": time.time(), "finished_at": time.time()}
            )()

        def run(self) -> None:
            pass

    monkeypatch.setattr("src.core.generator.Generator", FakeGenerator)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--functions_definition", str(functions),
            "--input", str(prompts),
            "--output", str(tmp_path / "out.json"),
            "--metrics_path", str(tmp_path / "metrics.json"),
        ],
    )

    code = main()

    assert code == 0
    assert "Completed:" in capsys.readouterr().out
