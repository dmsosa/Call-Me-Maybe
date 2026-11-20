"""Contrat de la ligne de commande (sans charger le modèle)."""
import sys
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
