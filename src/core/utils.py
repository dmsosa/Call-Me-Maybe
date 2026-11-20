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


# Reset
RESET = "\033[0m"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"


# Regular colors (bright)
BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"


# Bold / bright variants
BOLD_BLACK = "\033[90m"
BOLD_RED = "\033[91m"
BOLD_GREEN = "\033[92m"
BOLD_YELLOW = "\033[93m"
BOLD_BLUE = "\033[94m"
BOLD_MAGENTA = "\033[95m"
BOLD_CYAN = "\033[96m"
BOLD_WHITE = "\033[97m"


# Text styles
BOLD = "\033[1m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"
BLINK = "\033[5m"


HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"


def print_line(s: str, delay: float = 0.05) -> None:
    for ch in s:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)


def erase_line(s: str) -> None:
    sys.stdout.write("\r" + " " * len(s) + "\r")
    sys.stdout.flush()
