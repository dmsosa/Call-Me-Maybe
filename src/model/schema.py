import re
from typing import Literal, Optional, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator

SNAKE_CASE_REGEXP = r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$"

ParamType: TypeAlias = Literal["string", "number", "boolean", "integer"]


def value_matches(value: object, declared: ParamType) -> bool:
    """Check that a parsed value matches its declared type.

    ``bool`` is a subclass of ``int`` in Python, so it is explicitly
    excluded from the numeric types: a boolean must never pass as an
    integer or a number.

    Args:
        value: The value parsed from the generated JSON.
        declared: The type declared for the parameter.

    Returns:
        True if the value is valid for the declared type.
    """
    if declared == "string":
        return isinstance(value, str)
    if declared == "boolean":
        return isinstance(value, bool)
    if declared == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class ParamSpec(BaseModel):
    """
    A parameter, it only contains a type
    """
    model_config = ConfigDict(extra="forbid")
    type: ParamType

class FnDefinition(BaseModel):
    """
    This is what the JSON for "Function" field must contain
    """
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, pattern=SNAKE_CASE_REGEXP)
    description: str = ""
    parameters: dict[str, ParamSpec] = Field(default_factory=dict)
    returns: ParamSpec | None = None

class PromptDefinition(BaseModel):
    """
    A prompt JSON, it only contains a 'prompt' field
    """
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)

    @field_validator("prompt", mode="before")
    @classmethod
    def strip_prompt(cls, v: str) -> str:
        return v.strip()

class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: Optional[str]
    name: Optional[str]
    parameters: Optional[dict[str, object]]

class OutputList(RootModel[list[FunctionCall]]):
    pass