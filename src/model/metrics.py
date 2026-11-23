from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


class PromptMetrics(BaseModel):
    """Metrics for one prompt."""
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    prompt: str
    name: str | None = None
    parameters: dict[str, object] | None = None
    calls: int = Field(ge=0, default=0)
    duration_s: float = Field(ge=0.0, default=0.0)
    cache: Literal["hit", "miss", "off"] = "off"
    error: str | None = None


class CmmMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics_path: Path = Path("data/output/metrics.json")
    started_at: float = Field(ge=0.0, default=0.0)
    finished_at: float = Field(ge=0.0, default=0.0)
    model_load_s: float = Field(ge=0.0, default=0.0)
    total_s: float = Field(ge=0.0, default=0.0)
    prompts: list[PromptMetrics] = Field(default_factory=list)

    @computed_field                       # ← inclus dans model_dump()
    @property
    def total_calls(self) -> int:
        total = 0
        for p in self.prompts:
            total += p.calls
        return total

    @computed_field
    @property
    def cache_hits(self) -> int:
        total = 0
        for p in self.prompts:
            if p.cache == "hit":
                total += 1
        return total

    @computed_field
    @property
    def successes(self) -> int:
        total = 0
        for p in self.prompts:
            if not p.error:
                total += 1
        return total

    @computed_field
    @property
    def errors(self) -> int:
        total = 0
        for p in self.prompts:
            if p.error:
                total += 1
        return total

    def add_prompt(self, entry: PromptMetrics) -> None:
        self.prompts.append(entry)

    def write(self) -> None:
        """
        Writes a temporary metrics.json.tmp file creating parent dir
        if necessary. Replace with formatted JSON after successful writing
        """
        tmp = self.metrics_path.with_name(self.metrics_path.name + ".tmp")
        try:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(self.model_dump_json(indent=2) + "\n",
                           encoding="utf-8")
            os.replace(tmp, self.metrics_path)
        except OSError as exc:
            print(f"warning: cannot write metrics {self.metrics_path}: {exc}",
                  file=sys.stderr)
            tmp.unlink(missing_ok=True)

    def summary(self) -> str:
        """Une ligne de bilan pour la fin du run."""
        done = len(self.prompts)
        return (f"{self.successes}/{done} ok, {self.errors} erreurs, "
                f"{self.total_calls} appels ({self.cache_hits} cache), "
                f"{self.total_s:.1f}s")
    