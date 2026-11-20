from enum import Enum, auto
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from ..exception.exception import OutputError, format_schema_error
from ..model.schema import FunctionCall, OutputList


class WritingState(Enum):
    PROMPT = auto(),
    FUNCTION_NAME = auto(),
    FUNCTION_PARAMS = auto(),
    FINISH = auto()

class JsonWriter(BaseModel):
    """
    Writes a list of function calls as output
    it has methods for replacing the current output
    and writing the output itself.
    """

    model_config = ConfigDict(extra="forbid")

    output_path: Path

    def export_json(self, function_call_list: list[FunctionCall]) -> None:
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
            output = OutputList(function_call_list)

            payload = json.dumps(
                [call.model_dump() for call in output.root],
                indent=2,
                ensure_ascii=False,
            )
            self.output_path.write_text(payload + "\n", encoding="utf-8")
        except ValidationError as exc:
            raise OutputError(
                            "Error while writing output json file"
                            f"{self.output_path} : {exc}"
                            )
        except OSError as exc:
            raise OutputError(
                f"Is not possible to write output json file {self.output_path} : {exc.strerror}"
            )
