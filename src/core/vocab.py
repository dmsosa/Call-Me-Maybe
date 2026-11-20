import json
from typing import Any
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, ConfigDict

from src.core.byte_level import build_char_to_byte


# Characters that may follow a backslash inside a JSON string
# (\" \\ \/ \b \f \n \r \t). \u is left out on purpose: it would
# need 4 hex digits after it and we don't need it for our inputs.
# Note: in Python source '\\' is ONE backslash, so this string is
# the 8 characters  "  \  /  b  f  n  r  t
JSON_ESCAPES = '"\\/bfnrt'


REQUIRED_TOKENS = ('"', "true", "false", "\\")


class Vocab(BaseModel):
    """
    Token id -> text table and precomputed numpy masks.

    The model does not know about letters, it only knows ~151k "tokens"
    (pieces of text, each with an id). To constrain generation we need to
    know, for every token id, what text it stands for, and then which
    tokens are allowed in each situation (inside a number, inside a
    string...). A "mask" is a numpy array of True/False with one slot per
    token id: True = allowed.

    Every mask is built ONCE at startup. Per step we then only do
    `np.where(mask, logits, -np.inf).argmax()`, never a Python loop
    over the ~151k tokens.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vocab_path: str
    n_logits: int = Field(gt=0)
    tokens_to_ids: dict[str, int] = Field(default_factory=dict, exclude=True)
    ids_to_tokens: dict[int, str] = Field(default_factory=dict, exclude=True)
    digit_mask: npt.NDArray[np.bool_] | None = Field(default=None, exclude=True)
    minus_mask: npt.NDArray[np.bool_] | None = Field(default=None, exclude=True)
    dot_mask: npt.NDArray[np.bool_] | None = Field(default=None, exclude=True)
    end_number_mask: npt.NDArray[np.bool_] | None = Field(default=None, exclude=True)
    str_valid: list[npt.NDArray[np.bool_]] | None = Field(default=None, exclude=True)
    str_closed: list[npt.NDArray[np.int32]] | None = Field(default=None, exclude=True)
    str_escaped: list[npt.NDArray[np.bool_]] | None = Field(default=None, exclude=True)
    quote_id: int = Field(default=0, exclude=True)
    true_id: int = Field(default=0, exclude=True)
    false_id: int = Field(default=0, exclude=True)
    backslash_id: int = Field(default=0, exclude=True)

    def model_post_init(self, context) -> None:
        try:
            with open(self.vocab_path, encoding="utf-8") as f:
                # vocab.json is a map from "token" to id, with lot
                # of tokens
                raw: dict[str, int] = json.load(f)
        except (FileNotFoundError, IOError, PermissionError, json.JSONDecodeError) as e:
            raise ValueError(f"Error creating the Vocab, bad path_to_vocab str: \n{e}")
        # one throwaway call just to learn the logits size (151936)
        char_to_byte: dict[str, int] = build_char_to_byte()
        self._build_mappings(raw, char_to_byte)
        self._build_masks()

        # Token ids we need to refer to directly. encode returns a
        # list, [0] takes its first (here: only) id.
        missing = [t for t in REQUIRED_TOKENS if t not in self.tokens_to_ids]
        if missing:
            raise ValueError(
                f"model vocabulary unusable for constrained decoding: "
                f"missing single token(s) {missing}"
            )
        self.quote_id = self.tokens_to_ids.get('"', 0) # 1 is the real id of double-quote (see vocab.json xd)
        self.true_id = self.tokens_to_ids.get('true', 0) # 1866 is the real id of true (see vocab.json)
        self.false_id = self.tokens_to_ids.get('false', 0) # 3576 is the real id of false (see vocab.json)
        self.backslash_id = self.tokens_to_ids.get('\\', 0) # 3576 is the real id of false (see vocab.json)

    def _build_mappings(self, raw_logits: dict[str, Any], char_to_byte: dict[str, int]) -> None:
        for token, id in raw_logits.items():
            if id < self.n_logits:
                char_bytes_values = []
                for char in token:
                    char_bytes_values.append(char_to_byte.get(char, 0))
                
                # # From string
                # b1 = bytes("Hello", "utf-8")  # b'Hello'

                # # From integers
                # b2 = bytes([32, 72, 101, 108, 108, 111])  # b' Hello'

                # # From size (null bytes)
                # b3 = bytes(3)  # b'\x00\x00\x00'
                word_in_bytes = bytes(char_bytes_values)
                decoded_bytes =  word_in_bytes.decode("utf-8", errors="replace")
                
                # this is map ids to token
                self.ids_to_tokens[id] = decoded_bytes
                # this is map tokens to ids.
                self.tokens_to_ids[decoded_bytes] = id

    def _build_masks(self) -> None:

        # np.zeros(size, dtype=bool) = an array of `size` False values.
        # We then flip to True the ids that belong to each group.
        # - number masks
        self.digit_mask = np.zeros(self.n_logits, dtype=bool)       # "0".."9"...
        self.minus_mask = np.zeros(self.n_logits, dtype=bool)       # "-"
        self.dot_mask = np.zeros(self.n_logits, dtype=bool)         # "."
        self.end_number_mask = np.zeros(self.n_logits, dtype=bool)  # ",", "}"...

        # - string tables
        # A valid string is everything that is not an unesca
        # Indexed by state: [0] = normal, [1] = right after a '\'.
        # str_valid[s][id]   -> may token `id` be used in state s?
        # str_closed[s][id]   -> index of the closing '"' in it, or -1
        # str_escaped[s][id] -> does it end with a pending '\'?
        # [... for _ in range(2)] builds a list of 2 separate arrays.
        self.str_valid = [np.zeros(self.n_logits, dtype=bool) for _ in range(2)]

        # np.full(self.n_logits, -1) = an array of `self.n_logits` values, all -1
        self.str_closed = [np.full(self.n_logits, -1, dtype=np.int32)
                          for _ in range(2)]
        self.str_escaped = [np.zeros(self.n_logits, dtype=bool) for _ in range(2)]

        # One pass over the whole vocabulary, done once at startup.
        for idx, token_text in enumerate(self.ids_to_tokens.values()):
            if not token_text:
                # empty string = special token: leave it all False
                continue

            # .isascii() rules out digits of other alphabets,
            # .isdigit() = only digit characters
            if token_text.isascii() and token_text.isdigit():
                self.digit_mask[idx] = True
            elif token_text == "-":
                self.minus_mask[idx] = True
            elif token_text == ".":
                self.dot_mask[idx] = True

            # text[0] in ",}" = first character is ',' or '}'
            # text.strip() = text without spaces around it
            elif token_text[0] in ",}" or token_text.strip() in (",", "}"):
                # what the model would write after a number in JSON;
                # we use it as the "number is finished" signal
                self.end_number_mask[idx] = True

            for state in (0, 1):
                # bool(0) is False, bool(1) is True
                ok, close, esc = self._scan_string_token(token_text, bool(state))
                self.str_valid[state][idx] = ok
                self.str_closed[state][idx] = close
                self.str_escaped[state][idx] = esc

    def _scan_string_token(self, text: str, escaped: bool) -> tuple[bool, int, bool]:
        """Simulate appending `text` inside a JSON string.

        Used at startup to pre-classify every token for string generation.

        Args:
            text: decoded token text.
            escaped: True if the previous char was an unfinished backslash.

        Returns:
            (valid, close_index, escaped_after). close_index is the index of
            the unescaped '"' that ends the string, or -1 if still open.

        Examples (escaped=False):_build_ids_to_text
            'hello'  -> (True, -1, False)   plain text, string stays open
            'lo",'   -> (True, 2, False)    the '"' at index 2 closes it
            'a\\'    -> (True, -1, True)    ends with a lone backslash
            '\\x'    -> (False, -1, False)  \\x is not a valid JSON escape
        """
        # enumerate gives (position, character) pairs: (0,'l'), (1,'o')...
        for i, ch in enumerate(text):
            if escaped:
                # previous char was '\': this char must be a legal escape
                if ch not in JSON_ESCAPES:
                    return False, -1, False
                escaped = False
            elif ch == "\\":
                # start of an escape sequence, the next char decides
                escaped = True
            elif ch == '"':
                # an unescaped quote = end of the string value
                return True, i, False
            # ord(ch) < 0x20: character codes 0-31 are invisible control
            # characters (0x20 is hexadecimal for 32, the space)
            elif ord(ch) < 0x20 or ch == "\ufffd":
                # raw control chars (newline, tab...) are illegal in JSON
                # strings; '\ufffd' or '�' means "broken utf-8 piece"
                return False, -1, False
        # got to the end of the token without closing the string
        return True, -1, escaped
