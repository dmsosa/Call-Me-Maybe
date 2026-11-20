import json
from typing import Any
import unicodedata
import regex
from pydantic import BaseModel, Field

from src.core.byte_level import build_byte_to_char, build_char_to_byte
from src.exception.exception import InputError


PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


class PersonalTokenizer(BaseModel):
    """
    Qu'est ce que le tokenisateur faire? Il y a deux methods:
    encode: le plus important
    decode: sa parait trivial.
    """
    vocab_path: str
    merges_path: str
    merges: list[tuple[str, str]] = []
    tokens_to_ids: dict[str, int] = Field(default_factory=dict)
    ids_to_tokens: dict[int, str] = Field(default_factory=dict)
    byte_to_char: dict[int, str] = Field(default_factory=dict)
    char_to_byte: dict[int, str] = Field(default_factory=dict)
    merges_rank: dict[tuple[str, str], int] = Field(default_factory=dict)

    def model_post_init(self, _ctx: Any) -> None:
        try:
            with open(self.vocab_path, 'r') as f:
                raw = f.read()
                data_json = json.loads(raw)
        except (FileNotFoundError, PermissionError, Exception) as e:
            raise InputError(
                "Could not read the required "
                f"vocab file while constructing Tokenizer {e}"
                )
        for key, value in data_json.items():
            self.tokens_to_ids[key] = value
        self.ids_to_tokens = { i: t for t, i in self.tokens_to_ids.items()}
        self.byte_to_char = build_byte_to_char()
        self.char_to_byte = build_char_to_byte()
        self._load_merges()
 
    def _load_merges(self) -> None:
        try:
            with open(self.merges_path, 'r') as f:
                lines = f.read().splitlines()
                # skip first line, it is a comment
        except (FileNotFoundError, PermissionError, Exception) as e:
            raise InputError(
                "Could not read the required "
                f"merge file while constructing Tokenizer {e}"
                )
        for idx, line in enumerate(lines[1:]):
            if not line.strip():
                continue
            octet_a, octet_b = line.split()
            self.merges_rank[(octet_a, octet_b)] = idx

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        return text


    def _pretokenize(self, text: str) -> list[str]:
        matches: list[str] = regex.findall(PRETOKENIZE_REGEX, text)
        return matches

    def _bpe(self, chunk: str) -> list[str]:
        # return the list of token_ids that represent
        # a single chunk 
        # from " I like cafe"
        # cafe > [5, 78, 32454]
        """Apply BPE merges to a single pre-tokenized chunk."""
        byte_chunk = "".join(
            self.byte_to_char[ch] for ch in chunk.encode("utf-8")
        ) # from " Shrek" to ""ĠShrek"
        
        parts = list(byte_chunk) # ['a', 'b', 'c'] : un caractère par part
        
        while len(parts) > 1:
            cand = [(self.merges_rank[p], i)                     # (priorite, position)
                    for i, p in enumerate(zip(parts, parts[1:]))
                    if p in self.merges_rank]                    # seulement les paires connues
            if not cand:                              # plus aucune regle -> on s'arrete
                break
            rank, i = min(cand)                       # plus petit rang, puis le plus a gauche
            parts[i:i+2] = [parts[i] + parts[i+1]]    # on FUSIONNE cette paire
        return parts

    def encode(self, text: str) -> list[int]:
        """
        Normalize text,
        Divide it into chunks with the pretokenizer regex,
        for each chunk, find which merge pairs build it up using BPE

        One you have the tokens that build a given chunk, append them
        to the results 
        """
        normalized = self._normalize(text)
        chunks = self._pretokenize(normalized)

        results = []
        for chunk in chunks:
            tokens = self._bpe(chunk)
            results.extend(self.tokens_to_ids[t] for t in tokens)

        return results
        
    def decode(self, ids: list[int]) -> str:

        text_bytes = []
        tokens = []
        for value in ids:
            tokens.append(self.ids_to_tokens[value])
        for token in tokens:
            for ch in token:
                text_bytes.append(self.char_to_byte.get(ch, 0))

        raw = bytes(text_bytes)
        return raw.decode("utf-8", errors="replace")

    # def _tokenizer_algorithm(self, pre_tokenized_corpus: list[str]) -> None:
    #     base_vocab = set()
    #     for ch in pre_tokenized_corpus:
    #         if ch not in base_vocab:
    #             base_vocab.add(ch)
    #     token_pairs = {ch: 1 for ch in base_vocab}
    #     for i in range(self.vocab_size):
    #         print(f"Training at token {i}")
    #         for text in self.corpus:
    #             pair = token_pairs[i]
    #             new_pair = pair + text
    #             token_pairs.setdefault((pair, new_pair), 1) += 1
    #         max_pair = max(token_pairs.values())
    #         base_vocab.add(max_pair)