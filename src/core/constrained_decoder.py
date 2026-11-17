from enum import Enum, auto
import json
from typing import Any, Optional
from pydantic import BaseModel, Field
import numpy as np
import numpy.typing as npt
from src.core.llm import LargeLanguageModel
from src.core.vocab import Vocab
from src.model.schema import FnDefinition, FunctionCall


# Safety limit: a value can never be longer than this many tokens,
# so a confused model can't loop forever.
TOKEN_LIMIT = 48


# Qwen3 chat format. <|im_start|>/<|im_end|> are special tokens that
# mark who is talking (system / user / assistant).
# {functions} is a placeholder, filled in later with .format().
# "\\\\d+" in Python source = the text \\d+ in the prompt, which is
# how the regex \d+ looks inside JSON (backslashes are doubled).
SYSTEM = (
    "<|im_start|>system\n"
    "You are a JSON writer agent, pick the function matching the request and extract its arguments "
    "exactly from the request. Answer in JSON. Regex arguments are "
    "Python regular expressions (e.g. \\\\d+, [abc]); replacements are "
    "the literal characters to insert, never their names.\n"
    "Functions:\n{functions}<|im_end|>\n"
    "<|im_start|>user\n"
)
# The empty <think></think> tells Qwen3 to skip its "reasoning" text
# and answer directly (saves a lot of tokens = time).
USER = (
    "{prompt}\n"
    "Function to call: <|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


class DecoderState(Enum):
    """
    Decoder Finite State Machine state
    """
    FN_PROMPT = auto()
    FN_NAME = auto()
    FN_PARAMS = auto()
    FINISH = auto()


class ConstrainedDecoder(BaseModel):
    llm: LargeLanguageModel
    vocab: Vocab
    fn_by_name: dict[str, FnDefinition]
    token_limit: int = Field(gt=0, lt=5000, default=TOKEN_LIMIT)
    fn_name_ids: dict[str, list[int]] = Field(default_factory=dict, exclude=True)
    base_context_ids: list[int] = Field(default_factory=list, exclude=True)
    state: DecoderState = DecoderState.FN_PROMPT
    selected_fn: Optional[FnDefinition] = Field(default=None, exclude=True)

    def model_post_init(self, context) -> None:
        # name -> its token ids, e.g. "fn_greet" -> [id, id, id]
        self.fn_name_ids = {fn_name: self.llm.encode(fn_name) for fn_name in self.fn_by_name}
        # One short line per function, e.g.
        #   fn_add_numbers(a: number, b: number) - Add two numbers...
        lines = []
        for f in self.fn_by_name.values():
            # f"{k}: {v.type}" builds "a: number" for each parameter,
            # ", ".join(...) glues them: "a: number, b: number"
            args = ", ".join(f"{k}: {v.type}" for k, v in f.parameters.items())
            lines.append(f"{f.name}({args}) - {f.description}")
        # The system part is the same for every prompt: encode it once.
        # "\n".join(lines) puts one function per line.
        base_context = SYSTEM.format(functions="\n".join(lines))
        self.base_context_ids = self.llm.encode(base_context)

        # It is more efficient to encode all function names once
        # Rather than encode them all for each prompt generation
        self.fn_name_ids = {fn_name: self.llm.encode(fn_name) for fn_name in self.fn_by_name}

    # DO I know how to generate dummy text using IA? 
    # I want to be able to pass the list of prompts, and
    # store the outputs in a json with the format:
    # response: string
    def generate_json(self, prompt: str) -> str:
        """
        Iterates over the state machine which does not stop
        building our string until it reaches the state "FINISH"
        returns a data string which is expected to have valid
        JSON format.

        For each prompt, the same initial context is encoded
        to help the model make more accurate answers.

        We need to return only the guesses that have been done
        by the model, but taking into account that in order to
        get the next "guess" (next_token) it is necessary to
        keep passing along the token_ids encoded so far, which
        initially are our base_context.

        To separate the decode answer from the entire token_ids
        that were encoded during the generation, we store the 
        length of the base_ids.
        """
        self.state = DecoderState.FN_PROMPT
        token_ids = self.base_context_ids + self.llm.encode(USER.format(prompt=prompt))
        base_ids_len = len(token_ids)
        data = ""
        while self.state != DecoderState.FINISH:
            self._handle_state(prompt, token_ids, data)

        data += "}"
        token_ids += self.llm.encode("}")
        return self.llm.decode(token_ids[base_ids_len:])

    def _handle_state(self, prompt: str, token_ids: list[int], data: str) -> None:
            if self.state == DecoderState.FN_PROMPT:
                chunk = '{"prompt": ' + json.dumps(prompt) + f', "name": "'
                data += chunk
                token_ids += self.llm.encode(chunk)
                self.state = DecoderState.FN_NAME
            elif self.state == DecoderState.FN_NAME:
                self._handle_fn_name(token_ids, data)
            elif self.state == DecoderState.FN_PARAMS:
                self._handle_params(prompt, token_ids, data)

    def _handle_fn_name(self, token_ids: list[int], data: str) -> None:
        name = self.select_best_fn_name(token_ids)
        chunk = f'{name}",' + '"parameters": {'
        self.selected_fn = self.fn_by_name.get(name)
        data += chunk
        token_ids += self.llm.encode(chunk)
        self.state = DecoderState.FN_PARAMS

    def _handle_params(self, prompt: str, token_ids: list[int], data: str) -> None:
        # now, for each param, I need to know its type.
        # I am going to repeat this process the amount of
        # times as the params lenght
        # params is a dict, so I can iterate over its items
        # and get its name and type.
        if not self.selected_fn:
            raise ValueError(f"did not found function name matching this prompt: {prompt}")
        param_count = len(self.selected_fn.parameters)
        if param_count == 0:
            chunk = '}'
            data += chunk
            token_ids += self.llm.encode(chunk)
            self.state = DecoderState.FINISH
            return
        for i, (param_name, param) in enumerate(self.selected_fn.parameters.items()):
            if i == 0:
                chunk = f'"{param_name}": '
            else:
                chunk = f', "{param_name}": '
            data += chunk
            token_ids += self.llm.encode(chunk)
            value: Any = None
            if param.type == "number":
                value = self.generate_number(token_ids)
            elif param.type == "integer":
                value = self.generate_number(token_ids, integer=True)
            elif param.type == "boolean":
                value = self.generate_boolean(token_ids)
            elif param.type == "string":
                value = self.generate_string(token_ids)
            if i == param_count - 1:
                chunk = value + '}'
            else:
                chunk = value
            data += chunk
            token_ids += self.llm.encode(chunk)
        self.state = DecoderState.FINISH

    def apply_mask(self, mask: npt.NDArray[np.bool_], logits: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Apply a mask to logits, setting invalid tokens to -inf.

        Args:
            logits: The original logits array.
            valid_ids: List of token IDs that are allowed.

        Returns:
            A masked logits array where invalid tokens have -inf.
        """
        return np.where(mask, logits, -np.inf)

    def select_best_token(self, masked: npt.NDArray[np.float32]) -> int:
        """Select the token with the highest logit.

        Args:
            masked: The masked logits array.

        Returns:
            The ID of the token with the highest logit.

        Raises:
            ValueError: If the mask does not allow any token.
        """
        if not np.isfinite(masked).any():
            raise ValueError("No valid token available for the current state")

        best_token = np.argmax(masked)
        return int(best_token)

    def generate_string(self, token_ids: list[int]) -> str:
        """
        Challenge: avoid to generate uncompleted escaped sequences
        if one token ends with backslash (\\), it's necessary that
        the following token starts with a valid escape sequence char
        (/bfnrt or another backslash). Otherwise, break the loop to
        keep JSON valid

        After generating the output, if it ends         the following token starts with a valid escape sequence char
        (/bfnrt or another backslash). Otherwise, break the loop to
        keep JSON valid

        After generating the output, if it with backslash,
        remove it with output = output[:-1]
        and then remove any whitespace with rstrip(" ")
        """
        output = '"' # starts output with an opening quote
        ids_copy = list(token_ids) # local copy for this function
        ids_copy.append(self.llm.encode('"')[0])
        escaped = 0
        for _ in range(self.token_limit):
            logits = self.llm.logits(ids_copy)
            masked = self.apply_mask(self.vocab.str_valid[escaped], logits)
            best_token = self.select_best_token(masked)
            close_quote_idx = int(self.vocab.str_closed[escaped][best_token])
            if close_quote_idx >= 0:
                output += self.vocab.ids_to_tokens[best_token][:close_quote_idx + 1]
                ids_copy.append(best_token)
                break
            output += self.vocab.ids_to_tokens[best_token]
            escaped = int(self.vocab.str_escaped[escaped][best_token])
            ids_copy.append(best_token)
 
        if output[-1] == "\\" or escaped:
            output = output[:-1]
            ids_copy.append(self.vocab.tokens_to_ids.get("\\", 59))
        
        output = output.rstrip(" ")

        # If output does not have the closing quote due to token_limit
        # Then add it manually to not break the JSON
        if output[-1] != '"':
            output += '"'

        return output


    def generate_boolean(self, token_ids: list[int]) -> str:
        """
        Pour generer une numbre, on doit tout simplement recevoir les logits
        qui sont nummeres valides, s'il n'y a pas
        """
        logits = self.llm.logits(token_ids)
        masked = self.apply_mask(self.only(self.vocab.true_id, self.vocab.false_id), logits)
        best_token = np.argmax(masked)
        value = "true" if best_token == self.vocab.true_id else "false"
        return value

    def generate_number(self, token_ids: list[int], integer: bool = False) -> str:
        """
        To generate a number I receive the token_ids that the model
        suggest, limiting the tokens received by the variable token_limit

        - Minus sign is only allowed for the first character
        - a dot '.' is only allowed if type is not integer, and
          if and only if it does not contains a dot already
        - if the number ends by dot, erase it.
        - if the model ask me to terminate the number, only allow it 
          if the last character is not a dot.
        """
        output = ""
        ids_copy = list(token_ids) # local copy for this function
        for _ in range(self.token_limit):
            # any(...) is True if at least one character is a digit
            has_digit = any(c.isdigit() for c in output)
            # Build the mask for this step. Digits are always allowed.
            # .copy() is needed: without it we would modify the shared
            # digit_mask itself and break it for the next numbers.
            combined_mask = self.vocab.digit_mask.copy()
            # `a |= b` is "a = a OR b" position by position: it adds
            # the True slots of b to a (= allows those tokens too).
            if not output:
                # "-" only as the very first character
                combined_mask |= self.vocab.minus_mask
            if has_digit and not integer and "." not in output:
                # one "." max, after at least one digit, never in ints
                combined_mask |= self.vocab.dot_mask
            if has_digit and not output.endswith("."):
                # may stop only once the number is complete
                # ("", "-" and "12." are not valid numbers)
                combined_mask |= self.vocab.end_number_mask
            logits = self.llm.logits(ids_copy)
            masked = self.apply_mask(combined_mask, logits)
            best_token = self.select_best_token(masked)
            if self.vocab.end_number_mask[best_token]:
                break           # model says the number is finished
            output += self.vocab.ids_to_tokens[best_token]
            ids_copy.append(best_token)
        # rstrip(".") removes "." characters from the END only.
        output = output.rstrip(".")  # in case we hit max tokens on ex: "12."
        return output

    def only(self, *token_ids: int) -> npt.NDArray[np.bool_]:
        """Mask allowing exactly `token_ids`.

        The `*` means "any number of arguments": only(5, 9, 12) gives
        token_ids == (5, 9, 12).
        """
        mask = np.zeros(self.vocab.n_logits, dtype=bool)  # all False
        # numpy lets you set many positions at once:
        # mask[[5, 9, 12]] = True sets those three slots to True
        mask[list(token_ids)] = True
        return mask
   
    def select_best_fn_name(self, previous_ids: list[int]) -> str:
        """
        Uses the current input ids to build a function name,
        it stops requesting logits to the model when
        there is only one or no candidates.
        """
        picked: list[int] = []
        candidates = list(self.fn_by_name)
        step = 0
        while len(candidates) > 1:
            next_candidates: dict[int, list[str]] = {}
            for name in candidates:
                name_ids = self.fn_name_ids[name]
                key = name_ids[step] if step < len(name_ids) else self.vocab.quote_id
                next_candidates.setdefault(key, []).append(name)
            if len(next_candidates) == 1:
                # every candidate continues the same way: forced token,
                # no need to ask the model.
                chosen_token_id = list(next_candidates)[0] # list(dict) -> its keys
            else:
                # real choice: the model picks, but only among these.
                # ids + picked = the prompt followed by the name so far;
                # *next_candidates passes each key of next_candidates as a separate argument
                logits = self.llm.logits(previous_ids + picked)
                masked = self.apply_mask(self.only(*next_candidates), logits)
                chosen_token_id = masked.argmax()
            
            candidates = next_candidates[chosen_token_id]
            picked.append(chosen_token_id)
            step += 1
            if chosen_token_id == self.vocab.quote_id:
                break
        if len(candidates) == 0:
            raise ValueError("No candidate function found for this prompt")
        return candidates[0]
