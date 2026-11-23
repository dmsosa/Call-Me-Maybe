"""La vue sans vrai terminal : blessed se degrade proprement en texte brut."""
import io

from blessed import Terminal

from src.core.events import Change, Origin
from src.visual.gui import CmmGui

W, H = 80, 25          # valeurs de repli de blessed hors TTY (verifie)


def terminal() -> Terminal:
    """Un Terminal sans TTY : largeur 80, hauteur 25, aucune sequence ANSI."""
    return Terminal(stream=io.StringIO())


def prompt_start() -> Change:
    return Change(
        event_type="prompt_start",
        model_name="Qwen/Qwen3-0.6B",
        prompt_index=2,
        total_prompts=3,
        cache="miss",
        spt=0.21,
        prompt="Salut",
    )


def append(origin: Origin, text: str) -> Change:
    return Change(
        event_type="append",
        model_name="Qwen/Qwen3-0.6B",
        prompt_index=2,
        total_prompts=3,
        cache="miss",
        spt=0.21,
        prompt="Salut",
        step=0,
        origin=origin,
        text=text,
        data_str=text,
    )


def test_prompt_start_header_is_drawn(capsys: object) -> None:
    """L'en-tete affiche le modele, l'index, le cache et le prompt."""
    gui = CmmGui(term=terminal())
    gui.on_change(prompt_start())

    out = capsys.readouterr().out       # type: ignore[attr-defined]
    assert "Call Me Maybe" in out
    assert "model  : Qwen/Qwen3-0.6B" in out
    assert "prompt : 2/3" in out
    assert "cache: miss" in out
    assert "0.21 s/token" in out
    assert "Salut" in out


def test_appends_are_kept_with_their_origin(capsys: object) -> None:
    """Chaque morceau est memorise avec la bonne origine, puis dessine."""
    gui = CmmGui(term=terminal())
    gui.on_change(prompt_start())
    gui.on_change(append(Origin.SCAFFOLD, '{"name": "'))
    gui.on_change(append(Origin.MODEL, "greet"))

    assert [p.text for p in gui.parts] == ['{"name": "', "greet"]
    assert [p.origin for p in gui.parts] == [Origin.SCAFFOLD, Origin.MODEL]
    assert "greet" in capsys.readouterr().out     # type: ignore[attr-defined]


def test_a_new_prompt_clears_the_previous_buffer() -> None:
    """Sans ce clear, le prompt 2 afficherait la fin du prompt 1."""
    gui = CmmGui(term=terminal())
    gui.on_change(append(Origin.MODEL, "ancien"))
    gui.on_change(prompt_start())

    assert gui.parts == []


def test_frame_has_exactly_one_row_per_line_and_the_right_width() -> None:
    """Le grille est rectangle : c'est ce que _paint_line suppose."""
    gui = CmmGui(term=terminal())

    rows = gui.frame(prompt_start(), W, H)

    assert len(rows) == H
    assert all(len(row) == W for row in rows)
    assert all(isinstance(cell, tuple) for row in rows for cell in row)


def test_wait_key_is_called_only_in_step_mode(monkeypatch: object) -> None:
    """En auto, aucun inkey bloquant ne doit figer la generation."""
    waits: list[int] = []
    monkeypatch.setattr(          # type: ignore[attr-defined]
        CmmGui, "_wait_key", lambda self: waits.append(1)
    )
    gui = CmmGui(term=terminal())

    gui.on_change(append(Origin.MODEL, "a"))
    assert waits == []

    gui.step_mode = True
    gui.on_change(append(Origin.MODEL, "b"))
    assert len(waits) == 1


def test_close_works_without_a_tty(capsys: object) -> None:
    """close() est appele dans le finally de main(), evenement ou pas."""
    CmmGui(term=terminal()).close()
    capsys.readouterr()       # type: ignore[attr-defined]


def test_spinner_changes_with_the_step(capsys: object) -> None:
    """Pendant un prompt, un caractere tournant indique l'activite."""
    gui = CmmGui(term=terminal())

    gui.on_change(append(Origin.MODEL, "a"))
    first = capsys.readouterr().out      # type: ignore[attr-defined]
    gui.on_change(append(Origin.MODEL, "b"))
    second = capsys.readouterr().out     # type: ignore[attr-defined]

    # Contrat : le spinner est entoure de crochets, pour ne pas le confondre
    # avec le "/" de "s/token" ou les tirets de l'en-tete.
    glyphs = ["[" + c + "]" for c in "|/-\\"]
    assert any(g in first for g in glyphs), "aucun spinner dans la frame"
    assert any(g in second for g in glyphs), "aucun spinner dans la frame"


def test_spinner_is_replaced_by_a_fixed_mark_on_prompt_end(
    capsys: object,
) -> None:
    """Un spinner qui continue apres la fin donnerait l'illusion du calcul."""
    gui = CmmGui(term=terminal())

    gui.on_change(Change(event_type="prompt_end", prompt="Salut"))

    out = capsys.readouterr().out        # type: ignore[attr-defined]
    assert "Call Me Maybe" in out
    assert not any("[" + c + "]" in out for c in "|/-\\")


def test_a_very_long_wide_token_does_not_overflow_a_row(capsys: object) -> None:
    """Un token CJK occupe 2 colonnes : tronquer par caracteres ne suffit pas."""
    change = Change(
        event_type="append",
        model_name="Qwen/Qwen3-0.6B",
        prompt_index=1,
        total_prompts=2,
        prompt=" Salut",
        data_str="x",
        top=(("中文" * 20, 123456, 12345.67),),
    )
    gui = CmmGui(term=terminal())

    gui.on_change(change)

    out = capsys.readouterr().out     # type: ignore[attr-defined]
    lines = out.splitlines()
    assert len(lines) == H
    assert all(gui.term.length(line) <= W for line in lines), [
        (i, gui.term.length(line)) for i, line in enumerate(lines)
        if gui.term.length(line) > W
    ]
