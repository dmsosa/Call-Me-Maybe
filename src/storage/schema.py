from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel


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
	name: str = Field(min_length=1)
	description: str = ""
	parameters: dict[str, ParamSpec] = Field(default_factory=dict)
	returns: ParamSpec | None = None

class PromptDefinition(BaseModel):
	"""
	A prompt, it only contains a type
	"""
	model_config = ConfigDict(extra="forbid")
	prompt: str = Field(min_length=1)

class FunctionCall(BaseModel):
	model_config = ConfigDict(extra="forbid")
	prompt: str
	name: str
	parameters: dict[str, object]

class OutputList(RootModel[list[FunctionCall]]):
    pass