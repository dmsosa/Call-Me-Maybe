from typing import Optional

from llm_sdk import Small_LLM_Model
from pydantic import BaseModel

from src.storage.json_loader import FunctionStorage, PromptStorage


class Tokenizer(BaseModel):
	fn_storage: FunctionStorage
	prompts: PromptStorage
	model: Optional[Small_LLM_Model] = None

	def model_post_init(self, context):
		self.model = Small_LLM_Model()

	def pass_to_model(self) -> None:
		if not self.model:
			return 
		for prompt in self.prompts:
			self.model.encode(prompt)
			print(prompt)

	def tokenize(self, string: str) -> list[str]:
		pass