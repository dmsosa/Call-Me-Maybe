"""Thin wrapper around llm_sdk: cached logits as numpy arrays."""

import numpy as np
import numpy.typing as npt

from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ConfigDict, Field

class LargeLanguageModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Small_LLM_Model
    cache: dict[tuple[int, ...], npt.NDArray[np.float32]] = Field(default_factory=dict)
    calls: int = Field(ge=0, default=0)
    """Wraps Small_LLM_Model using only its public methods.

    The SDK has no KV cache: every call re-reads the WHOLE sequence
    (~0.5-1.5 s per call on CPU, ~0.08 s on GPU). So:
      * keep the prompt short,
      * never call the model when only one token is valid,
      * never compute the same sequence twice (cache below).
    """

    def encode(self, text: str) -> list[int]:
        """Text -> list of token ids."""
        # the SDK returns a 2D tensor of shape [1, n] (a "batch" of one
        # sentence): take row 0 and turn it into a plain Python list
        ids: list[int] = self.model.encode(text)[0].tolist()
        return ids

    def logits(self, ids: list[int]) -> npt.NDArray[np.float32]:
        """Next-token logits for `ids`, memoized on tuple(ids).

        Same idea as functools.lrucache, but a dict on the instance
        (lrucache on a method keeps `self` alive and needs hashable args).
        """
        # lists can't be dict keys (they can change), tuples can
        key = tuple(ids)
        # .get(key) returns the stored value, or None if not there
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        self.calls += 1
        # the SDK gives a Python list of ~151k floats; a numpy array
        # lets the decoder mask and argmax it in one fast operation
        out = np.array(self.model.get_logits_from_input_ids(ids),
                       dtype=np.float32)
        self.cache[key] = out
        return out

    def vocab_path(self) -> str:
        """Path to vocab.json (a map of type: token string -> id)."""
        # str(...) only reassures mypy that the SDK returns a string
        return str(self.model.get_path_to_vocab_file())

    def decode(self, token_ids: list[int]) -> str:
        return self.model.decode(token_ids)