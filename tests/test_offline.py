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

ROOT = Path(__file__).resolve().parents[1]

SPY = '''
import os

print("SPY", os.environ.get("HF_HUB_OFFLINE"),
      os.environ.get("TRANSFORMERS_OFFLINE"))
raise RuntimeError("spy: on arrete avant de charger le modele")
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


def run_cli(tmp_path: Path, extra_args: list[str]) -> str:
    """Lance ``python -m src`` avec un ``llm_sdk`` espion, rend la sortie."""
    spy_dir = tmp_path / "spy"
    package = spy_dir / "llm_sdk"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(SPY, encoding="utf-8")

    functions = tmp_path / "functions.json"
    functions.write_text(FUNCTIONS, encoding="utf-8")
    prompts = tmp_path / "prompts.json"
    prompts.write_text(PROMPTS, encoding="utf-8")

    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
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
