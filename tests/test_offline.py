"""Le mode hors ligne doit être posé AVANT le premier import Hugging Face.

``huggingface_hub`` et ``transformers`` lisent ``HF_HUB_OFFLINE`` /
``TRANSFORMERS_OFFLINE`` au moment de leur import : les poser plus tard
n'a aucun effet. On le vérifie dans le vrai chemin de code (``main``) en
remplaçant ``llm_sdk`` par un espion qui affiche l'environnement au moment
de son import.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Modeles cites dans le README (Bonus part). Aucun n'est charge : on
# verifie seulement que le CLI transmet le nom au SDK.
MODEL_NAMES = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3.5-0.8B",
    "meta-llama/Llama-3.2-1B",
    "openbmb/MiniCPM5-1B",
    "smollm-135m",
    "smollm-360m",
    "tiiuae/Falcon-H1-Tiny-Multilingual-100M",
]

SPY = '''
import os

print("SPY", os.environ.get("HF_HUB_OFFLINE"),
      os.environ.get("TRANSFORMERS_OFFLINE"))
raise RuntimeError("spy: on arrete avant de charger le modele")
'''

# Ce spy regarde ce que Hugging Face a *lu* dans l'environnement, pas
# seulement la variable : c'est ce drapeau qui decide de telecharger ou non.
HUB_SPY = '''
import os
import huggingface_hub as hub
import huggingface_hub.constants as constants

print("SPYHUB", os.environ.get("HF_HUB_OFFLINE"),
      constants.HF_HUB_OFFLINE, hub.is_offline_mode())
raise RuntimeError("spy: on arrete avant de charger le modele")
'''

# Ce spy laisse ``main`` construire le modele, mais arrete juste apres,
# pour comparer les arguments avec et sans --cache_model.
MODEL_SPY = '''
class Small_LLM_Model:
    def __init__(self, *args, **kwargs):
        print("MODELEARGS", args, sorted(kwargs.items()))
        raise RuntimeError("spy: arret apres la construction")
'''

FUNCTIONS = """[
  {
    "name": "fn_add",
    "description": "Add two numbers.",
    "parameters": {"a": {"type": "number"}, "b": {"type": "number"}}
  }
]
"""

PROMPTS = """[{"prompt": "What is the sum of 2 and 3?"}]
"""


def run_cli(
    tmp_path: Path,
    extra_args: list[str],
    *,
    spy: str = SPY,
    env_overrides: dict[str, str] | None = None,
) -> str:
    """Lance ``python -m src`` avec un ``llm_sdk`` espion, rend la sortie."""
    spy_dir = tmp_path / "spy"
    package = spy_dir / "llm_sdk"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text(spy, encoding="utf-8")

    functions = tmp_path / "functions.json"
    functions.write_text(FUNCTIONS, encoding="utf-8")
    prompts = tmp_path / "prompts.json"
    prompts.write_text(PROMPTS, encoding="utf-8")

    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    if env_overrides:
        env.update(env_overrides)
    env["PYTHONPATH"] = os.pathsep.join([str(spy_dir), str(ROOT)])

    done = subprocess.run(
        [
            sys.executable, "-m", "src",
            "--functions_definition", str(functions),
            "--input", str(prompts),
            "--output", str(tmp_path / "out.json"),
            *extra_args,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    return done.stdout + done.stderr


def test_cache_model_flag_sets_offline_before_the_import(
    tmp_path: Path,
) -> None:
    output = run_cli(tmp_path, ["--cache_model"])

    assert "SPY 1 1" in output, (
        "avec --cache_model, HF_HUB_OFFLINE et TRANSFORMERS_OFFLINE "
        f"doivent valoir 1 des l'import de llm_sdk (sortie : {output!r})"
    )


def test_without_the_flag_offline_mode_is_not_forced(tmp_path: Path) -> None:
    output = run_cli(tmp_path, [])

    assert "SPY None None" in output, (
        "sans --cache_model, le programme doit laisser le Hub en ligne "
        f"(sortie : {output!r})"
    )


def test_cache_model_is_seen_by_huggingface_itself(tmp_path: Path) -> None:
    """Preuve que le telechargement est coupe, pas juste une variable posee.

    ``huggingface_hub`` fige ``HF_HUB_OFFLINE`` a l'import. Le spy importe
    la bibliotheque au moment ou ``llm_sdk`` est importe et affiche la
    constante reellement utilisee par ``hf_hub_download``.
    """
    output = run_cli(tmp_path, ["--cache_model"], spy=HUB_SPY)

    assert "SPYHUB 1 True True" in output, (
        "huggingface_hub doit etre en mode hors ligne avant de charger "
        f"le modele (sortie : {output!r})"
    )


def test_cache_model_overrides_a_preset_offline_zero(tmp_path: Path) -> None:
    """Un HF_HUB_OFFLINE=0 deja present ne doit pas annuler le flag.

    ``setdefault`` laisse la valeur existante : dans ce cas le flag ne fait
    rien et le programme peut retenter un telechargement.
    """
    output = run_cli(
        tmp_path,
        ["--cache_model"],
        env_overrides={
            "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0",
        },
    )

    assert "SPY 1 1" in output, (
        "--cache_model doit forcer '1' meme si l'environnement dit '0' "
        f"(sortie : {output!r})"
    )


def test_short_flag_c_also_switches_to_offline(tmp_path: Path) -> None:
    assert "SPY 1 1" in run_cli(tmp_path, ["-c"])


def test_cache_flag_does_not_change_model_arguments(tmp_path: Path) -> None:
    """Le flag ne doit changer que l'acces au Hub, jamais le modele.

    Meme appel a ``Small_LLM_Model`` avec et sans le flag : sinon les deux
    modes n'ont pas les memes poids / le meme device / la meme precision.
    """
    normal = run_cli(tmp_path, [], spy=MODEL_SPY)
    cached = run_cli(tmp_path, ["--cache_model"], spy=MODEL_SPY)

    def model_args(output: str) -> list[str]:
        return [line for line in output.splitlines() if "MODELEARGS" in line]

    assert model_args(normal), normal
    assert model_args(normal) == model_args(cached), (
        f"arguments differents : {model_args(normal)} vs {model_args(cached)}"
    )


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_model_flag_is_parsed_unchanged(
        monkeypatch: pytest.MonkeyPatch, model_name: str) -> None:
    """Le parser n'altere aucun des noms de modele du README.

    Test sans sous-processus : le vrai chemin jusqu'au SDK est couvert
    une fois par le test suivant.
    """
    from src.__main__ import init_argparse

    monkeypatch.setattr(sys, "argv", ["src", "--model", model_name])

    assert init_argparse().model == model_name


def test_a_non_default_model_is_forwarded_to_the_sdk(tmp_path: Path) -> None:
    """Un --model non par defaut arrive au SDK meme avec --cache_model."""
    model_name = "openbmb/MiniCPM5-1B"
    normal = run_cli(tmp_path, ["--model", model_name], spy=MODEL_SPY)
    cached = run_cli(
        tmp_path, ["--model", model_name, "--cache_model"], spy=MODEL_SPY
    )

    def model_args(output: str) -> list[str]:
        return [line for line in output.splitlines() if "MODELEARGS" in line]

    expected = f"('model_name', '{model_name}')"
    assert expected in "\n".join(model_args(normal)), normal
    assert model_args(normal) == model_args(cached), (
        f"{model_name}: {model_args(normal)} vs {model_args(cached)}"
    )
