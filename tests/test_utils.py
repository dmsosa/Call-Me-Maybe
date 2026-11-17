"""Contrat de formatage des durées affichées à l'écran."""
from src.core.utils import fmt_time


def test_fmt_time_seconds_only() -> None:
    assert fmt_time(5.3) == "0:05.30"


def test_fmt_time_minutes() -> None:
    assert fmt_time(105.3) == "1:45.30"


def test_fmt_time_rounds_without_showing_60_seconds() -> None:
    """59.999 s vaut 1 minute : jamais d'affichage "0:60.00"."""
    assert fmt_time(59.999) == "1:00.00"
