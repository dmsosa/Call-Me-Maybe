from dataclasses import dataclass
from enum import Enum, auto
import json
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, RootModel, ValidationError, model_validator

from src.exception.exception import InputError, format_schema_error
from src.model.schema import FnDefinition, PromptDefinition

class FunctionStorage(RootModel[list[FnDefinition]]):
	"""
	Will read the JSON input files (a list of Function Definitions)
	this will also store prompts within the same class.
	"""

	@model_validator(mode="after")
	def _duplicated_names(self) -> "FunctionStorage":
		seen: set[str] = set()
		for fn in self.root:
			if fn.name in seen:
				raise ValueError(f"duplicated name of function : {fn.name}")
			seen.add(fn.name)
		return self

	@property
	def by_name(self) -> dict[str, FnDefinition]:
		return {fn.name: fn for fn in self.root}

class PromptStorage(RootModel[list[PromptDefinition]]):
	@model_validator(mode="after")
	def _non_empty(self) -> "PromptStorage":
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





def load_fn_storage(path_str: str) -> FunctionStorage:
	json_content = _read_json_from_path(path_str)
	try:
		storage = FunctionStorage(json_content)
		return storage
	except ValidationError as e:
		raise InputError(format_schema_error(e, path_str)) from None
	except Exception as e:
		raise InputError(f"Unexpected exception raised while creating Function Storage: \n{e}\n")


def load_prompts(path_str: str) -> PromptStorage:
	json_content = _read_json_from_path(path_str)
	try:
		storage = PromptStorage(json_content)
		return storage
	except ValidationError as e:
		raise InputError(format_schema_error(e, path_str)) from None
	except Exception as e:
		raise InputError(f"Unexpected exception raised while creating Function Storage: \n{e}\n")


# def read_functions(self, path: str) -> None:
# 	try:
# 		with open(path, 'r') as f:
# 			json_content = json.load(f)
# 	except (FileNotFoundError, PermissionError, IOError) as e:
# 		raise e
# 	except json.JSONDecodeError as e:
# 		raise e
# 	except Exception as e:
# 		raise e

# 	for fn in json_content:
# 		fn_def = FnDefinition(**fn)
# 		self.fn_to_def[fn_def.name] = fn_def

# 	self._check_dup_names()

# def read_prompts(self, content: str) -> None:
# 	try:
# 		json_content = json.load(content)
# 	except json.JSONDecodeError as e:
# 		raise e
# 	except json.JSONDecodeError as e:
# 		raise e

# 	for line, json_object in enumerate(json_content):
# 		prompt = json_object.get("prompt", "")
# 		if (len(prompt) < 1):
# 			raise Exception(f"Invalid prompt file, empty prompt in line {line + 1}")
# 		self.prompts.append(prompt)

# 	self.prompts = result

# def _find_duplicates(self, json_content) -> list[str]:
# 	"""Return names that appear more than once."""
# 	seen: set[str] = set()
# 	dupes: list[str] = []
# 	for fn_name in self.fn_to_def.keys():
# 		if fn_name in seen:
# 			dupes.append(fn_name)
# 		else:
# 			seen.add(fn_name)
# 	return dupes
