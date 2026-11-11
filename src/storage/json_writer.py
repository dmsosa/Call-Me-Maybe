import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from ..exception.exception import OutputError, format_schema_error
from .schema import OutputList


class JsonWriter(BaseModel):
    """
    Writes a list of function calls as output
    it has methods for replacing the current output
    and writing the output itself.
    """

    model_config = ConfigDict(extra="forbid")

    output_path: Path
    output_payload: OutputList

    def set_output_payload(self, payload: Any) -> None:
        """
        Replace the payload and pass schema validations
        raising error if not valid schema
        """
        try:
            self.output_payload = OutputList(payload)
        except ValidationError as exc:
            raise OutputError(
                format_schema_error(exc, self.output_path)
            )

    def export_json(self) -> Path:
        """
        Writes the output file, raises error in case it is not
        possible to write the file because any of the following
        errors:
        - FileNotFoundError,
        - IsADirectoryError,
        - PermissionError,
        - NotADirectoryError
        """
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                self.output_payload.model_dump(),
                indent=2,
                ensure_ascii=False,
            )
            self.output_path.write_text(payload + "\n", encoding="utf-8")
        except OSError as exc:
            raise OutputError(
                f"Is not possible to write output json file {self.output_path} : {exc.strerror}"
            )
        return self.output_path
