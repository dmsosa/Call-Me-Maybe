from dataclasses import dataclass
from enum import Enum

from pydantic import Field


class Origin(str, Enum):          # (str, Enum) : StrEnum est 3.11+
    SCAFFOLD = "scaffold"         # texte ecrit par nous
    MODEL    = "model"            # token echantillonne sous contrainte
    INPUT    = "input"            # prompt recopie


@dataclass(frozen=True, slots=True)
class Part:
    index: int
    origin: Origin
    text: str
    ids: tuple[int, ...] = ()

    @property
    def n_tokens(self) -> int:
        return len(self.ids)


@dataclass(frozen=True)
class Change:
    # Filled by generator
    model_name: str = ""
    prompt_index: int = 0
    total_prompts: int = 0
    cache: str = "off"
    spt: float = 0.0
    prompt: str = ""
    # Filled by decoder
    event_type: str = "append"
    step: int = 0
    origin: Origin = Origin.SCAFFOLD
    text: str = ""
    ids: tuple[int, ...] = ()
    data_str: str = ""
    choices: int = 0
    top: tuple[tuple[str, int, float], ...] = ()
    state: str = ""
