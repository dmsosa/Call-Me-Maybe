import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from src.visual.constants import RESET, YELLOW
from src.model.schema import FnDefinition, FunctionCall, value_matches


CACHE_VERSION = 1


class CacheEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, object]


class CmmCache(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    version: int = CACHE_VERSION
    enabled: bool = False
    hits: int = Field(default=0, ge=0)
    entries: dict[str, CacheEntry] = Field(default_factory=dict)
    
    def model_post_init(self, _context: Any) -> None:
        if self.enabled:
            self._load()

    def _load(self) -> None:
        try:
            with self.path.open('r', encoding="utf-8") as f:
                raw = f.read()
                data = json.loads(raw)
        except FileNotFoundError as e:
            print(
                f"{YELLOW}Warning: Cache is not created yet, "
                "if it is enabled, then new entries are going to be "
                f"saved.{RESET}",
                file=sys.stderr
                )
            return
        except (OSError, ValueError) as exc:
            print(
                f"{YELLOW}Warning: It was not possible to read the cache, "
                f"{self.path}: {exc} {RESET}",
                file=sys.stderr
                )
            return

        if not isinstance(data, dict):
            return
        if data.get("version") != self.version:
            print(
                f"{YELLOW}Warning: cache ignore (version "
                f"{data.get('version')}) {RESET}",
                file=sys.stderr
                )
            return

        entries = data.get("entries")
        if not isinstance(entries, dict):
            return
        for key, value in entries.items():
            try:
                self.entries[key] = CacheEntry.model_validate(value)
            except ValidationError:
                print(
                    f"{YELLOW}Warning: invalid entry "
                    f"ignored in cache{RESET}",
                    file=sys.stderr
                    )

    def _save(self) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        payload = {
            "version": self.version,
            "entries": {
                key: entry.model_dump() for key, entry in self.entries.items()
            },
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp, self.path)
        except OSError as exc:
            print(f"warning: cache non ecrit ({self.path}): {exc}",
                  file=sys.stderr)
            tmp.unlink(missing_ok=True)

    def cache_key(self, prompt: str, fns: list[FnDefinition]) -> str:
        canon = json.dumps(
            [fn.model_dump() for fn in fns],
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        raw = f"{self.version}\x00{prompt}\x00{canon}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def lookup(
            self,
            prompt: str,
            fn_list: list[FnDefinition],
            fn_by_name: dict[str, FnDefinition]
            ) -> FunctionCall | None:
        if not self.enabled:
            return None
        entry = self.entries.get(self.cache_key(prompt, fn_list))
        if entry is None or entry.prompt != prompt:
            return None
        selected = fn_by_name.get(entry.name)
        if selected is None:
            return None
        missing_params = set(selected.parameters) - set(entry.parameters)
        if missing_params:
            return None
        for param_name, value in entry.parameters.items():
            spec = selected.parameters.get(param_name)
            if spec is None or not value_matches(value, spec.type):
                return None                     # fichier d'entree modifie
        self.hits += 1
        return FunctionCall(
            prompt=prompt, name=entry.name,
            parameters=dict(entry.parameters),
        )

    def store(
        self,
        prompt: str,
        fn_list: list[FnDefinition],
        call: FunctionCall
    ) -> None:
        if not self.enabled:
            return
        self.entries[self.cache_key(prompt, fn_list)] = CacheEntry(
            prompt=prompt, name=call.name, parameters=call.parameters,
        )
        self._save()

    def clear(self) -> None:
        self.entries.clear()
        self._save()
