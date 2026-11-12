import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator

SNAKE_CASE_REGEXP = r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$"

ParamType = Literal["string", "number", "boolean"]

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
    A prompt, it only contains a type
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