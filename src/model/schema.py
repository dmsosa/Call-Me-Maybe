import re
from typing import Literal, Optional, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator

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

    @model_validator(mode="after")
    def valid_param_names(self) -> "FnDefinition":
        for name in self.parameters:
            if not re.fullmatch(SNAKE_CASE_REGEXP, name):
                raise ValueError("" \
                            "Parameter name must be a " \
                            f"valid snake_Case string, received: '{name}'"
                            )
            
        return self

class PromptDefinition(BaseModel):
    """
    A prompt JSON, it only contains a 'prompt' field
    """
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)

    @field_validator("prompt", mode="after")
    @classmethod
    def strip_prompt(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Prompt must not be empty")
        return v

class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: dict[str, object]

class OutputList(RootModel[list[FunctionCall]]):
    pass