import unicodedata

from blessed import Terminal

from src.core.events import Origin
from src.visual.constants import CORNER_BL, \
            CORNER_BR, \
            CORNER_TL, \
            CORNER_TR, \
            HOR, \
            VER, \
            SPINNER_FRAMES



Cell = tuple[str, Origin | None]


# ---------------------------------------------------------------- primitives
def blank_grid(w: int, h: int) -> list[list[Cell]]:
    row: list[Cell] = [(" ", None) for _ in range(w)]
    return [list(row) for _ in range(h)]


def one_column(ch: str) -> str:
    if unicodedata.combining(ch):
        return ""
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return "?"
    return ch


def put(
        rows: list[list[Cell]],
        x: int,
        y: int,
        text: str,
        origin: Origin | None = None
        ) -> None:
    if not (0 <= y < len(rows)) or x >= len(rows[y]):
        return
    row = rows[y]
    if x <= 0:
        text, x = text[-x:], 0
    text = text[: len(row) - x]
    rows[y] = row[:x] + [(one_column(c), origin) for c in text] + row[x + len(text):]


def fit(text: str, width: int, term: Terminal) -> str:
    if term.length(text) <= width:
        return text + " " * (width - term.length(text))
    out = ""
    for ch in text:
        if term.length(out + ch) > width:
            break
        out += ch
    return out + " " * (width - term.length(out))


def fill(
        rows: list[list[Cell]],
        x0: int,
        y0: int,
        x1: int,
        y1: int
        ) -> None:
    for y in range(y0, y1 + 1):
        if 0 <= y < len(rows):
            put(rows, x0, y, " " * (x1 - x0 + 1))


def box(
        rows: list[list[Cell]],
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        title: str = ""
        ) -> None:
    w = x1 - x0 + 1
    fill(rows, x0, y0, x1, y1)
    put(rows, x0, y0, CORNER_TL + HOR * (w - 2) + CORNER_TR)
    put(rows, x0, y1, CORNER_BL + HOR * (w - 2) + CORNER_BR)
    for y in range(y0 + 1, y1):
        put(rows, x0, y, VER)
        put(rows, x1, y, VER)
    if title:
        put(rows, x0 + 2, y0, f" {title} "[: max(0, w - 4)])


def progress_bar(done: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "?" * width
    ratio = max(0.0, min(1.0, done / total))   # clamp : jamais <0 ni >1
    filled = round(ratio * width)              # int() avant "*" sur une str
    return "#" * filled + "-" * (width - filled)


def tail_cells(cells: list[Cell], limit: int) -> list[Cell]:
    return cells[-limit:] if limit > 0 else []


def wrap_cells(cells: list[Cell], width: int) -> list[list[Cell]]:
    if width <= 0:
        return [[]]
    return [cells[i:i + width] for i in range(0, len(cells), width)] or [[]]


def merge_cells(cells: list[Cell]) -> list[Cell]:
    out: list[Cell] = []
    for ch, origin in cells:
        if out and out[-1][1] == origin:
            out[-1] = (out[-1][0] + ch, origin)
        else:
            out.append((ch, origin))
    return out


def make_printable(token: str) -> str:
    return (token.replace("\u0120", "\u2423").replace("\u010a", "\u23ce")
                 .replace("\u0121", "?"))


def spinner(step: int) -> str:
    return "[" + SPINNER_FRAMES[step % len(SPINNER_FRAMES)] + "]"