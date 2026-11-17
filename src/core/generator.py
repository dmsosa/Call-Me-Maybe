import json
import sys
import time
from typing import Any, Optional
from pydantic import BaseModel, Field

from src.core.constrained_decoder import ConstrainedDecoder
from src.core.llm import LargeLanguageModel
from src.core.utils import fmt_time
from src.core.vocab import Vocab
from src.exception.exception import OutputError
from src.model.schema import FnDefinition, FunctionCall, PromptDefinition, value_matches
from src.storage.json_loader import FunctionList, PromptList
from src.storage.json_writer import JsonWriter


class Tokenizer(BaseModel):
    fn_by_name: FunctionList
    prompts: PromptList

class Generator(BaseModel):
    llm: LargeLanguageModel
    fn_list: list[FnDefinition]
    prompts: list[PromptDefinition]
    writer: JsonWriter
    results: list[FunctionCall] = Field(default_factory=list)
    errors: int = Field(ge=0, default=0)
    fn_by_name: dict[str, FnDefinition] = Field(default_factory=dict, exclude=True)
    decoder: Optional[ConstrainedDecoder] = Field(default=None, exclude=True)
    
    def model_post_init(self, _context: Any):
        self._store_fn_names()
        n_logits = len(self.llm.logits(self.llm.encode("a")))
        self.decoder = ConstrainedDecoder(
            llm=self.llm,
            vocab=Vocab(
                vocab_path=self.llm.model.get_path_to_vocab_file(),
                n_logits=n_logits
                ),
            fn_by_name=self.fn_by_name,
            )

    def _store_fn_names(self) -> None:
        for fn in self.fn_list:
            self.fn_by_name[fn.name] = fn

    def run(self) -> None:
        """
        Prepares the output results for this generator,
        which is going to be a list of Function Calls
        if there is an error generating one function call,
        then a fallback value is used
        """
        results = []
        for i, prompt in enumerate(self.prompts):
            t = time.time()
            before = self.llm.calls
            try:
                result = self.generate_function_call(prompt.prompt)
            except Exception as e:
                # never crash on one prompt: write a fallback entry instead
                # so the output stays aligned with the input
                print(f"  warning: {e}", file=sys.stderr)
                result = self.fallback_result(prompt.prompt)
                self.errors += 1

            results.append(result)
            # Write the function calls after each iteration
            self._write_results(results)
            # progress line: time, number of model runs, result.
            # {n:3d} = whole number padded to 3 characters (aligns columns)
            print(f"[{i}/{len(self.prompts)}] {fmt_time(time.time() - t)} "
                  f"{self.llm.calls - before:3d} calls  {result.name} "
                  f"{json.dumps(result.parameters)}", file=sys.stderr)

        self.errors
        self.results = results

    def generate_function_call(self, prompt: str) -> FunctionCall:
        """
        Generates one FunctionCall object for a given prompt
        it raises an Exception if there is an error during the
        generation process.
        """
        if not self.decoder:
            raise ValueError(f"Tried to generate JSON without decoder")
             
        generated_json = self.decoder.generate_json(prompt)
        try:
            data = json.loads(generated_json)
        except json.JSONDecodeError as e:
            raise Exception(f"generated JSON string is not a valid JSON: {generated_json}")
        except Exception:
            raise Exception(f"generated JSON data is not a valid object: {data}")
        
        name = data.get("name")
        parameters = data.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, dict):
            raise ValueError(
                f"The generated JSON has an unexpected shape: {generated_json!r}"
            )

        selected = self.fn_by_name.get(name)
        if selected is None:
            raise ValueError(f"Generated function is not defined: {name!r}")

        resolved: dict[str, object] = {}
        
        for param_name, value in parameters.items():
            definition = selected.parameters.get(param_name)
            if definition is None:
                raise ValueError(
                    f"Unexpected parameter {param_name!r} in {name!r}"
                )
            if not value_matches(value, definition.type):
                raise ValueError(
                    f"The parameter {param_name!r} of {name!r} is not a "
                    f"valid {definition.type}: {value!r}"
                )
            if definition.type == "number":
                value = float(value)
            resolved[param_name] = value

        return FunctionCall(
            prompt=prompt, name=name, parameters=resolved
            )
    
    # DO I know how to generate dummy text using IA? 
    # I want to be able to pass the list of prompts, and
    # store the outputs in a json with the format:
    # response: string
    def fallback_result(self, prompt: str) -> FunctionCall:
        first_fn_key = next(iter(self.fn_by_name))
        first_fn = self.fn_by_name.get(first_fn_key)
        return FunctionCall(
            prompt=prompt,
            name=first_fn.name,
            parameters={}
        )

    def _write_results(self, function_calls: list[FunctionCall]) -> None:
        try:
            print(
                f"\nWriting results to \
                {self.writer.output_path.as_posix()}..."
                )
            self.writer.export_json(function_calls)
        except OutputError as e:
            print(
                f"error: cannot write {self.writer.output_path}: \
                {e}",
                file=sys.stderr
                )
