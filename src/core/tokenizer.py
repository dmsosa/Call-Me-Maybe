from cmd import PROMPT
from typing import Any, Optional
import numpy as np
from idna import decode
from llm_sdk import Small_LLM_Model
import json
from pydantic import BaseModel, Field

from src.model.schema import FunctionCall
from src.storage.json_loader import FunctionStorage, PromptStorage
from src.storage.json_writer import JsonWriter, WritingState

MAX_GENERATION_LEN = 40

CONTEXT = "You are an assistant that answers in JSON"

class Tokenizer(BaseModel):
    fn_storage: FunctionStorage
    prompts: PromptStorage

class Generator(BaseModel):
    fn_storage: FunctionStorage
    prompts: PromptStorage
    writer: JsonWriter
    max_len: int = Field(gt=0, lt=5000, default=MAX_GENERATION_LEN)
    fn_names: list[str] = []

    def model_post_init(self, context):
        self._store_fn_names()
        return super().model_post_init(context)

    def _store_fn_names(self) -> None:
        for fn in self.fn_storage.root:
            self.fn_names.append(fn.name)

    # DO I know how to generate dummy text using IA? 
    # I want to be able to pass the list of prompts, and
    # store the outputs in a json with the format:
    # response: string
    def generate_json(self, model: Small_LLM_Model, prompt: str) -> str:
        # Need to prepare the build prompt for each Function Call to be built
        prompt_ids = model.encode(prompt).tolist()[0]
        end_of_prompt = len(prompt_ids)
        while True:
            logits = model.get_logits_from_input_ids(prompt_ids)
            max_logit = np.argmax(logits)
            prompt_ids.append(max_logit)
            if model.decode(max_logit) == '"':
                break
        json_result: dict[str, Any] = {
            "prompt": prompt
        }
        json_result["name"] = model.decode(prompt_ids[end_of_prompt:])
        selected_fn = self.fn_storage.by_name.get(json_result["name"])
        if not selected_fn:
            return ""
        for _ in range(len(selected_fn.parameters)):
            json_result["parameters"]: []
        return json.dumps(json_result)                   # <- decode la liste ENTIÈRE
    # def generate(self, model: Small_LLM_Model) -> list[FunctionCall]:
    #     self._store_fn_names()
    #     generated_count = 0
    #     for prompt in self.prompts.root:
    #         self.writer.state = WritingState.PROMPT            
    #         while not self.writer.state == WritingState.FINISH: # Run Until EOS or the generation limit
    #             if self.writer.state == WritingState.PROMPT:
    #                 self.writer.function_call["prompt"] = prompt
    #                 self.writer.state = WritingState.FUNCTION_NAME
    #             elif self.writer.state == WritingState.FUNCTION_NAME:
    #                 encoded = model.encode(prompt.prompt)
    #                 input_ids = encoded.to_list()[0]
    #                 logits = model.get_logits_from_input_ids(input_ids)
    #                 next_token_id = max(range(len(logits)), key=lambda idx: logits[idx])
    #                 found_name = ""
    #                 next_token = model.decode([next_token_id])
    #                 if next_token in self.fn_names:
    #                     found_name = next_token
    #                 else:
    #                     found_name = "p"
    #                 self.writer.function_call["name"] = found_name
    #                 self.writer.state = WritingState.FINISH
    #         generated_count += 1
    #         if generated_count == len(self.prompts.root):
    #             self.writer.state = WritingState.FINISH
    #         new_payload = self.writer.output_payload + [self.writer.function_call_payload]
    #         self.writer.output_payload = new_payload
    #         if generated_count == 1:
    #             break
    #     self.writer.export_json()			