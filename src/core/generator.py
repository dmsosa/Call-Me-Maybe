from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Optional
from pydantic import BaseModel, Field
from src.core.cache import CmmCache
from src.core.constrained_decoder import ConstrainedDecoder
from src.core.events import Origin
from src.core.llm import LargeLanguageModel
from src.core.utils import fmt_time
from src.core.vocab import Vocab
from src.exception.exception import OutputError, VisualizerQuit
from src.model.metrics import CmmMetrics, PromptMetrics
from src.model.schema import FnDefinition, FunctionCall, PromptDefinition, value_matches
from src.storage.json_writer import JsonWriter
from src.visual.gui import Change


class Generator(BaseModel):
    llm: LargeLanguageModel
    fn_list: list[FnDefinition]
    prompts: list[PromptDefinition]
    writer: JsonWriter
    metrics: CmmMetrics = Field(default_factory=CmmMetrics)
    results: list[FunctionCall] = Field(default_factory=list)
    cache_enabled: bool = Field(default=False)
    cache_path: str = Field(default="data/output/cache_enabled.json")
    cache: Optional[CmmCache] = Field(default=None)
    on_change: Optional[Callable[[Change], None]] = Field(default=None)
    fn_by_name: dict[str, FnDefinition] = Field(default_factory=dict, exclude=True)
    decoder: Optional[ConstrainedDecoder] = Field(default=None, exclude=True)
    errors_count: int = Field(ge=0, default=0, exclude=True)
    run_index: int = Field(default=0, exclude=True)
    run_prompt: str = Field(default="", exclude=True)
    prompt_count: int = Field(default=0, gt=0, exclude=True)
    run_cache_state: str = Field(default="off", exclude=True)
    run_started: float = Field(default=0.0, exclude=True)
    run_calls: int = Field(default=0, exclude=True)

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
            on_change=self._forward
            )

        self.cache = CmmCache(
            path=Path(self.cache_path),
            enabled=self.cache_enabled
            )
        self.prompt_count = len(self.prompts)
        if self.prompt_count > 10000:
            raise ValueError("Exceeded prompt limit: 10000")

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
            self.run_index, self.run_prompt = i, prompt.prompt
            self.run_started = time.time()
            self.run_calls = self.llm.calls
            error = None
            cached_result = None

            if self.cache_enabled:                        
                cached_result = self.cache.lookup(
                    prompt=prompt.prompt,
                    fn_list=self.fn_list,
                    fn_by_name=self.fn_by_name
                )
                self.run_cache_state = "hit" if cached_result \
                                    is not None else "miss"
            self._emit("prompt_start")
            if cached_result is not None:
                result = cached_result
                self._forward(Change(
                    event_type="append",
                    step=i + 1,
                    spt=0,
                    origin=Origin.SCAFFOLD,
                    cache="hit",
                    text=f"{json.dumps(result.model_dump())} [CACHED]",
                ))
            else:
                try:
                    result = self.generate_function_call(prompt.prompt)
                    self.run_cache_state = "miss"
                    if self.cache is not None and self.cache.enabled:
                        self.cache.store(
                            prompt=prompt.prompt,
                            fn_list=self.fn_list,
                            call=result
                            )
                except VisualizerQuit:
                    break
                except Exception as e:
                    # never crash on one prompt: write a fallback entry instead
                    # so the output stays aligned with the input
                    print(f"  warning: {e}", file=sys.stderr)
                    result = self.fallback_result(prompt.prompt)
                    self.errors_count += 1
                    error = e.__str__()
                    self.run_cache_state = "miss"

            results.append(result)
            # Write the function calls after each iteration
            self._write_results(results)

            self.metrics.add_prompt(PromptMetrics(
                index=i, prompt=prompt.prompt, name=result.name,
                parameters=result.parameters or None,
                calls=self.llm.calls - self.run_calls,
                duration_s=time.time() - self.run_started,
                error=error,
                cache=self.run_cache_state
            ))
            self.metrics.write()
            
            if self.on_change is None:
                # progress line: time, number of model runs, result.
                # {n:3d} = whole number padded to 3 characters (aligns columns)
                print(f"[{i}/{len(self.prompts)}] {fmt_time(time.time() - self.run_started)} "
                      f"{self.llm.calls - self.run_calls:3d} calls  {result.name} "
                      f"{json.dumps(result.parameters)}", file=sys.stderr)
            self._emit("prompt_end")

        self.metrics.finished_at = time.time()
        self._emit("end_visualization")
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
                "\nWriting results to "
                f"{self.writer.output_path.as_posix()}..."
                )
            self.writer.export_json(function_calls)
        except OutputError as e:
            print(
                f"error: cannot write {self.writer.output_path}: \
                {e}",
                file=sys.stderr
                )

    def _get_context(self) -> dict[str, Any]:
        calls = max(1, self.llm.calls - self.run_calls)
        return dict(
            prompt_index=self.run_index + 1,
            total_prompts=self.prompt_count,
            cache=self.run_cache_state,
            spt=(time.time() - self.run_started) / calls,
            prompt=self.run_prompt,
        )

    def _forward(self, change: Change) -> None:
        if self.on_change is not None:
            self.on_change(replace(change, **self._get_context()))

    def _emit(self, event_type: str) -> None:
        if self.on_change is not None:
            self.on_change(Change(event_type=event_type, **self._get_context()))
