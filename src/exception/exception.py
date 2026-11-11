from pathlib import Path

from pydantic import ValidationError


class InputError(Exception):
    """
    Errors related to the input files, malformed JSON,
    file errors, etc
    """


class OutputError(Exception):
    """
    Errors related to the output files, malformed JSON,
    file errors, etc
    """


def format_schema_error(exc: ValidationError, path: Path | str) -> str:
    """
    Transforms a Validation Error in a readable error for
    InputError or OutputError, or other.
    """
    details = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        msg = err["msg"].removeprefix("Value error, ")
        details.append(f"{loc} : {msg}")
    return f"{path} does not respect the schema -> " + " | ".join(details)
