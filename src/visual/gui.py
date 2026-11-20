from dataclasses import dataclass
from ssl import CHANNEL_BINDING_TYPES
import time

from pydantic import BaseModel, ConfigDict, Field
from blessed import Terminal
from src.exception.exception import VisualizerQuit


@dataclass
class Change:
    prompt_index: int      # 7
    prompt: str            # "Sum 12 and 30"
    text: str              # le JSON deja genere, decode
    step: int              # numero de token
    choices: int           # combien de tokens autorises par le masque
    chosen: str            # le token retenu
    top: list[tuple[str, float]]   # (token, logit) avant contrainte


class CmmGui(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    term: Terminal = Field(default_factory=Terminal, exclude=True)
    model_name: str
    index: int = 0
    total: int = 0
    cache: str = "off"          # ou bool + libelle
    spt: float = 0.0            # secondes / token (moyenne glissante)
    prompt_text: str = ""
    data_str: str = ""
    top5: list[tuple[str, int, float]] = Field(default_factory=list)
    history: list[Change] = Field(default_factory=list)

    def model_post_init(self, context):
        pass

    def quit_visuals(self) -> None:
        tick = 0
        keys = ""
        with self.term.cbreak(), self.term.hidden_cursor(), self.term.fullscreen():
            while True:
                self._handle_key_press()
                lines = [line.ljust(self.term.width) for line in self.frame()]
                print(self.term.home + self.term.clear + "\n".join(lines),
                      end="", flush=True)
                key = self.term.inkey(timeout=0.2)
                if key == "q":
                    break
                if key:
                    keys += key
                tick += 1
        print(f"Finished after {time.time() - self.start} frames")  

    def on_change(self) -> None:
        self.last = step
        self.frame()
        if self.step_mode:
            while True:
                key = self.term.inkey(timeout=None)     # bloquant : c'est VOULU
                if key == " ":
                    return                          # pas suivant
                if key == "a":
                    self.step_mode = False
                    return
                if key == "q":
                    raise VisualizerQuit
                
    def frame(self) -> list[str]:
        """Construit l'image ENTIERE, en texte brut, ligne par ligne."""
        p = (self.index / self.total) * 100
        bar = "#" * ((p // self.total) % 30)
        return [
            term.bold("visualiseur d'exemple") + term.normal,
            "",
            f"tick    : {tick}",
            f"[{bar:<30}]",
            "",
            f"touches : {keys!r}",
            "",
            " [q] quitter",
        ]