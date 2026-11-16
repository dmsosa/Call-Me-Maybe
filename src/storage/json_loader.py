import json
from pathlib import Path
from typing import Any
from pydantic import RootModel, ValidationError, model_validator
from src.exception.exception import InputError, format_schema_error
from src.model.schema import FnDefinition, PromptDefinition


class FunctionList(RootModel[list[FnDefinition]]):
	"""
	Will read the JSON input files (a list of Function Definitions)
	this will also store prompts within the same class.
	"""

	@model_validator(mode="after")
	def _duplicated_names(self) -> "FunctionList":
		seen: set[str] = set()
		for fn in self.root:
			if fn.name in seen:
				raise ValueError(f"duplicated name of function : {fn.name}")
			seen.add(fn.name)
		return self
	

class PromptList(RootModel[list[PromptDefinition]]):
	@model_validator(mode="after")
	def _non_empty(self) -> "PromptList":
		for i, prompt in enumerate(self.root):
			if not prompt.prompt.strip():
				raise ValueError(f"prompt number {i + 1} is empty")
		return self


def _read_json_from_path(path_str: str) -> Any:
	"""
	Reads and returns the content of a JSON File, handling errors
	"""
	try:
		path = Path(path_str)
		with path.open("r", encoding="utf-8") as f:
			return json.load(f)
	except FileNotFoundError as e:
		raise InputError(f"File not found {path_str}")
	except PermissionError as e:
		raise InputError(f"Not permissions to open the file {path_str}")
	except UnicodeDecodeError as e:
		raise InputError(f"Unicode decoding error for file {path_str}")
	except json.JSONDecodeError as e:
		raise InputError(
			f"Invalid JSON in file {path}," \
		    f" line {e.lineno}, col {e.colno}: {e.msg}"
			)
	except OSError as e:                      # 1
		raise InputError(f"Impossible to read file {path} : {e.strerror}")
	except Exception as e:         # 2
		raise InputError(
			f"Unexpected exception raised while attempting to read json, "
			f"{e}"
		) from None


def load_function_list(path_str: str) -> FunctionList:
	json_content = _read_json_from_path(path_str)
	try:
		list = FunctionList(json_content)
		return list
	except ValidationError as e:
		raise InputError(format_schema_error(e, path_str)) from None
	except Exception as e:
		raise InputError(f"Unexpected exception raised while creating Function Storage: \n{e}\n")


def load_prompts(path_str: str) -> PromptList:
	json_content = _read_json_from_path(path_str)
	try:
		list = PromptList(json_content)
		return list
	except ValidationError as e:
		raise InputError(format_schema_error(e, path_str)) from None
	except Exception as e:
		raise InputError(f"Unexpected exception raised while creating Function Storage: \n{e}\n")
