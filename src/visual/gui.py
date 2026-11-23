import time

from pydantic import BaseModel, ConfigDict, Field
from blessed import Terminal
from src.core.events import Change, Origin, Part
from src.exception.exception import VisualizerQuit
from src.visual.constants import HOR
from .gui_utils import blank_grid, \
    fill, \
    put, \
    box, \
    progress_bar, \
    merge_cells, spinner, \
    tail_cells, \
    wrap_cells, \
    make_printable, \
    Cell


class CmmGui(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    term: Terminal = Field(default_factory=Terminal, exclude=True)
    model_name: str = Field(min_length=0, default="Qwen/Qwen3-0.6B")
    step_mode: bool = False
    started_at: float = Field(default_factory=time.time)
    last: Change | None = Field(default=None, exclude=True)
    parts: list[Part] = Field(default_factory=list, exclude=True)
    total_parts: int = Field(default=0, ge=0, exclude=True)

    def on_change(self, change: Change) -> None:
        """
        Called by the decoder (appending changes)
        or generator (only for start or end)
        """
        if change.event_type == "prompt_start":
            self.parts.clear()
        elif change.event_type == "append":
            self.parts.append(Part(change.step, change.origin,
                                   change.text, ()))
            self.total_parts += 1
        elif change.event_type == "end_visualization":
            input("Press ENTER to continue...")
            return
        self.last = change
        self.draw()
        if self.step_mode and change.event_type == "append":
            self._wait_key()

    def _wait_key(self) -> None:
        while True:
            key = self.term.inkey(timeout=None)
            if key == " ":
                return
            if key == "a":
                self.step_mode = False
                return
            if key == "q":
                raise VisualizerQuit

    def draw(self) -> None:
        if self.last is None:
            return
        w = min(self.term.width, 120)
        h = min(self.term.height, 40)
        if w <= 120 or h <= 40:
            print("Terminal is too small, can not print the grid.")
            return
        rows = self.frame(self.last, w, h)
        body = "\n".join([self._paint_line(row, w) for row in rows])
        print(self.term.home + self.term.clear + body, end="", flush=True)

    def _paint_line(self, row: list[Cell], width: int) -> str:
        plain = "".join([t for t, _ in row])
        out = []
        for text, origin in merge_cells(row):
            if origin is Origin.MODEL:
                out.append(self.term.green(text))
            elif origin is Origin.INPUT:
                out.append(self.term.cyan(text))
            elif origin is None:
                out.append(text)
            else:
                out.append(self.term.dim(text))
        pad = " " * max(0, width - len(plain))
        return "".join(out) + pad
  
    def frame(self, change: Change, w: int, h: int) -> list[list[Cell]]:
        """
        Prepares a 2D grid of characters, to be printed line by line
        """
        bar = progress_bar(change.prompt_index, change.total_prompts, w)
        rows = blank_grid(w, h)
        fill(rows, 0, 0, w - 1, 0)
        put(rows, 0, 0, bar)
        glyph = spinner(change.step) if change.event_type != "prompt_end" else "✓"
        put(rows, 0, 1, f"Call Me Maybe {glyph}".center(w))
        put(rows, 0, 2, HOR * w)
        
        info = [
            f"  model  : {change.model_name}",
            f"prompt : {change.prompt_index}/{change.total_prompts}   "
            f"cache: {change.cache}   {change.spt:.2f} s/token",
            f"> {change.prompt}",
            "",
            f"- white: harcoded text",
            f"- {self.term.cyan}blue: text from the user as input{self.term.normal}",
            f"- {self.term.green}green: text from the model{self.term.normal}",
        ]
        for k, line in enumerate(info):
            put(rows, 1, 2 + k, line[: w - 2])
        put(rows, 0, 2 + len(info), HOR * w)

        top, bottom = 2 + len(info) + 1, h - 3

        right_w = min(38, max(28, w // 3))
        split = w - right_w - 1
        models = sum(1 for p in self.parts if p.origin is Origin.MODEL)
        box(rows, 0, top, split, bottom,
            f"buffer: {len(change.data_str)} chars, {models} model parts")
        box(rows, split + 1, top, w - 1, bottom,
            f"top-5 ({change.choices} masked allowed)")

        inner, inner_h = split - 1, bottom - top - 1
        cells: list[Cell] = []
        for p in self.parts:
            for ch in p.text:
                cells.append((ch, p.origin))
        lines = wrap_cells(tail_cells(cells, inner * inner_h), inner)[-inner_h:]
        
        for k, line in enumerate(lines):
            put(rows, 1, top + 1 + k, "".join(ch for ch, _ in line))
            for j, (ch, origin) in enumerate(line):
                rows[top + 1 + k][1 + j] = (ch, origin)

        for k, (tok, tid, logit) in enumerate(change.top[:inner_h]):
            put(rows, split + 2, top + 1 + k,
                f"{make_printable(tok):<12}{tid:<8}{logit:6.2f}"[: right_w - 2])

        put(rows, 0, h - 2, HOR * w)
        keys = "[q] quit   [space] pas   [a] auto   [o] output"
        put(rows, 1, h - 1, keys[: w - 2])
        put(rows, 1, h, HOR * w)

        for row in rows:
            assert len(row) == w, (len(row), w)
        return rows

    def close(self) -> None:
        print(self.term.normal + self.term.clear, end="", flush=True)
