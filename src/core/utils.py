import sys
import time


def fmt_time(seconds: float) -> str:
    """Format a duration as minutes:seconds, e.g. 105.3 -> "1:45.30".

    divmod(105.3, 60) gives (1.0, 45.3): whole minutes and the rest.
    {s:05.2f} = 2 decimals, padded with zeros to 5 characters, so
    5.3 seconds shows as "05.30" (like a clock).

    The rounding happens BEFORE divmod on purpose: 59.999 s must show
    "1:00.00", not "0:60.00".
    """
    m, s = divmod(round(seconds, 2), 60)
    return f"{int(m)}:{s:05.2f}"


def print_line(s: str, delay: float = 0.05) -> None:
    for ch in s:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)


def erase_line(s: str) -> None:
    sys.stdout.write("\r" + " " * len(s) + "\r")
    sys.stdout.flush()
