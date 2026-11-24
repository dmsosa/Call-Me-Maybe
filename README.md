*This project has been created as part of the 42 curriculum by durisosa.*

# Call Me Maybe — constrained function calling with a small LLM

Turn free-form prompts into **valid function calls** using `Qwen/Qwen3-0.6B`,
without fine-tuning and without prompt engineering: at every generation step the
model's logits are masked so that only tokens that keep the output valid remain
possible.

## Description

The program reads two JSON files:

* a list of **function definitions** (`name`, `description`, `parameters`,
  optional `returns`);
* a list of **prompts** (`{"prompt": "..."}`).

For each prompt it writes one function call to a JSON file, with **exactly three
keys** and always valid JSON:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers",
 "parameters": {"a": 2, "b": 3}}
```

The goal is *not* to make the model do something clever with text. The goal is to
make a small model produce JSON that is **correct by construction**: the decoder
never has the opportunity to emit a token that would break the format, use an
undeclared function, or put a string where a number was declared.

No crash on bad input, no crash on a prompt the model cannot answer: a single
prompt may fail, but the program always finishes and always writes one entry per
prompt so that the output stays aligned with the input.

## Instructions

### Requirements

* Linux or macOS, Python ≥ 3.10 (uses `X | None`, `list[int]`).
* [`uv`](https://docs.astral.sh/uv/) (installed by `make install` if missing).
* ~2.5 GB of disk/RAM for the model, and network access **the first time** to
  download `Qwen/Qwen3-0.6B` into the Hugging Face cache (afterwards it runs
  fully offline, see `--cache_model`).

### Install

```bash
make install        # installs uv if needed, links the HF cache, runs uv sync
# or, without make:
uv sync
```

### Run

```bash
make run                              # uv run python -m src
uv run python -m src                  # same thing
uv run python -m src --help           # all options
```

| Option | Default | Meaning |
|---|---|---|
| `--functions_definition` | `data/input/functions_definition.json` | function catalogue |
| `--input` | `data/input/function_calling_tests.json` | prompts to answer |
| `--output` | `data/output/function_calling_results.json` | produced function calls |
| `--metrics_path` | `data/output/metrics.json` | per-prompt metrics |
| `--cache_results` | off | reuse/store answers in `data/output/cache_results.json` |
| `--cache_model`, `-c` | off | force Hugging Face **offline** mode (model must already be cached) |

Other useful targets:

```bash
make lint           # flake8 + mypy
make lint-strict    # flake8 + mypy --strict
make debug          # same as run, under pdb
make cpu-time       # same as run, timing it with the GPU hidden
make clean          # __pycache__, .mypy_cache, data/output
make run ARGS="--cache_results --cache_model"
```

### Offline mode

`huggingface_hub` and `transformers` read `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`
**at import time**, so setting them after the import has no effect. `-c` sets both
before the lazy model import:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # equivalent, from the shell
unshare -rn uv run python -m src -c              # proof: no network at all
```

## Example usage

```bash
$ make run ARGS="--cache_results --cache_model"
Loading input files...
    5 functions loaded successfully...
    11 prompts loaded successfully...

Loading LLM Model from cache...
    model loaded in 0:05.60
Writing results to data/output/function_calling_results.json...
[0/11] 0:02.31  13 calls  fn_add_numbers {"a": 2, "b": 3}
[1/11] 0:03.02   9 calls  fn_add_numbers {"a": 265, "b": 345}
...
Completed: 5 successes, 6 errors
Results in: data/output/function_calling_results.json
```

Second run with `--cache_results` (same prompts, unchanged function catalogue):

```
[0/11] 0:00.00   0 calls  fn_add_numbers {"a": 2, "b": 3}
```

`metrics.json` explains where the time and the failures went:

```json
{
  "started_at": "2026-11-17T16:11:32.126945Z",
  "model_load_s": 5.6,
  "total_s": 41.2,
  "prompts": [
    {"index": 0, "prompt": "What is the sum of 2 and 3?",
     "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0},
     "calls": 13, "duration_s": 2.31, "cache": "hit", "error": null},
    {"index": 1, "prompt": "...", "name": "fn_add_numbers",
     "parameters": null, "calls": 48, "duration_s": 11.2,
     "cache": "miss", "error": "No valid token available for the current state"}
  ],
  "total_calls": 61, "cache_hits": 1, "successes": 1, "errors": 1
}
```

## Algorithm explanation

### 1. From prompt to base context

The prompt is embedded in the same shape the model expects for a tool call, and
**escaped with `json.dumps`** (`"`, `\`, newlines, control characters). Escaping
is better than masking here: the prompt comes from the outside, so the decoder
must not be given a chance to corrupt it.

The whole context is encoded once (`llm.encode`), then extended incrementally
with the tokens the decoder *chooses*: every step appends the emitted chunk to
`token_ids`, so the model always sees the real prefix it produced.

### 2. The state machine

```
FN_PROMPT ──► FN_NAME ──► FN_PARAMS ──► FINISH
               │            │
               │            └── per parameter, according to its declared type
               └── one of the declared function names
```

Each state owns its own **mask**: a `numpy` boolean array of size `n_logits`
(= 151 936 for Qwen3-0.6B, the model's logit vector size, *not* the vocabulary
size: the last ids are unused).

```python
masked = np.where(mask, logits, -np.inf)
best_token = int(np.argmax(masked))
```

`argmax` on the masked vector = "the best token **among the allowed ones**".
Greedy decoding: no temperature, no sampling, therefore the output is
deterministic for a given model and input.

### 3. Function name (`FN_NAME`)

All candidate names are compared token by token. As long as every remaining
candidate continues with the same token, the token is **forced** and the model is
not called at all. Only when candidates diverge does the decoder ask for logits
and mask them to the handful of possible continuation ids. When one candidate
remains, its remaining ids are copied and the state moves on.

This keeps the (many) shared prefixes free: with `fn_add_numbers` and
`fn_greet`, only the few tokens after `fn_` cost a forward pass.

### 4. Parameters (`FN_PARAMS`)

For each declared parameter, in the order of the definition file, the decoder
writes `"name": ` (with the JSON escaping driven by the *declared* name, which is
validated to be `snake_case` at load time), then generates the value according to
its type.

**number / integer** — digits are always allowed; `-` only as the first
character; `.` only once, only after a digit, never for `integer`; the "end of
number" token only once at least one digit has been emitted and the value does
not end with `.`. A trailing `.` is stripped. The value is emitted as digits, so
`2` stays `2` and never becomes `2.0` in the JSON text.

**boolean** — the mask allows exactly two token ids (`true` → 1866, `false` →
3849 in the Qwen3 vocabulary), and the higher logit wins. One forward pass.

**string** — a three-table automaton is precomputed over the vocabulary:
* `str_valid[state]` — token can appear while inside a string;
* `str_closed[state]` — position of the first unescaped `"` inside the token
  (`-1` if none): if present, the string ends **inside** that token and the rest
  of the token is dropped;
* `str_escaped[state]` — the state after emitting the token (0 = normal,
  1 = inside an escape sequence such as `\"`).

A token that would leave a dangling backslash is never emitted, and an
unterminated string is closed before being handed to the parser.

### 5. Closing and validation

After the last parameter the decoder appends `}` and returns a **string**. The
generator then does what a normal program would do with untrusted input:

1. `json.loads` must succeed;
2. `name` must be a string and `parameters` an object;
3. `name` must exist in the catalogue;
4. every emitted parameter must be **declared** for that function;
5. every value must match its declared type (`value_matches`), with `bool`
   explicitly rejected for `number`/`integer` (`bool` is a subclass of `int` in
   Python, a classic trap);
6. `number` values are normalised to `float`.

Any failure → one fallback entry for that prompt, error message on `stderr`, run
continues, error counted in `metrics.json`.

## Design decisions

| Decision | Alternative | Why |
|---|---|---|
| Mask logits at every step | generate free text, then parse/repair | correctness by construction, no repair loop, no retry |
| `argmax` (greedy) | sampling with temperature | deterministic output, reproducible runs, testable |
| Escape the prompt with `json.dumps` | mask the prompt characters too | the prompt is external data; escaping is simpler and total |
| Masks precomputed as `numpy` arrays in `Vocab` | per-step regex/string checks | one `np.where` + one `argmax` per step; O(1) per token, no per-token string work |
| Forced tokens without a model call | always ask the model | saves whole forward passes on shared prefixes |
| Fallback per prompt | crash / skip the prompt | output must stay aligned with the input (one entry per prompt) |
| Validate the generated JSON again | trust the decoder | the decoder is code, the JSON is the contract; belt and braces |
| `pydantic` models for input/output/cache/metrics | plain dicts | schema errors are reported with a field path instead of a `KeyError` |
| L1 answer cache (`--cache_results`) | no cache | a re-run of the same prompts costs 0 model calls; the key includes the function catalogue and a version, so a modified input can never be served from an old answer |
| Metrics in a **separate** file | extra keys in the output | the output contract is *exactly* `prompt`/`name`/`parameters` |
| Atomic writes (`.tmp` + `os.replace`) | direct `write_text` | a killed process never leaves a truncated JSON file |

## Performance analysis

**Speed** (measured on the school's CPU-only machine, model already in cache):

| What | Measure |
|---|---|
| Model load (offline, warm cache) | 5.6 s – 9.0 s |
| One forward pass (19-token context) | ~0.23 s |
| Cost model | **1 forward pass per generated token** — the SDK exposes no KV cache, so every step re-reads the whole sequence |
| Cost per prompt | 5–15 calls for a number/boolean call, up to 48 for a string (the `token_limit`) |
| Second run with `--cache_results` | **0 calls** for every cached prompt |

Consequences, in order of impact: keep the prompt short (the cost is linear in
context length × steps), avoid useless calls (forced tokens), and never compute
the same sequence twice (in-memory logits cache keyed by the token tuple). The
`calls` column printed per prompt and `total_calls` in `metrics.json` make the
cost visible instead of guessed.

**Accuracy.** Two different notions, deliberately separated:

* *Schema accuracy* — 100 % by construction, and enforced again after decoding
  (name known, parameters declared, types matching). A malformed output entry is
  not possible: either the decoder produced something that validates, or the
  entry is a fallback.
* *Semantic accuracy* — did the model pick the right function with the right
  values? That is the model's job, not the decoder's. It is measured, not
  claimed: `successes / (successes + errors)`, plus the per-prompt failure
  reasons, are in `metrics.json`. The constrained decoder guarantees that a
  *failure* is always visible (fallback + error), never silent.

**Reliability.**

* one entry per prompt, even when the model fails (alignment with the input);
* no exception escapes `run()`: every prompt is isolated in its own `try`;
* output, cache and metrics are written atomically;
* a corrupted or outdated cache is ignored, never fatal;
* the whole program can run with the network completely disabled (`unshare -rn`).

## Challenges faced

1. **The SDK has no KV cache.** Every generated token costs a full forward pass
   over the whole sequence. Solution: minimise steps (forced tokens, short
   contexts) and memoise logits by token tuple; document the limit instead of
   pretending it can be fixed.
2. **Escaping the prompt.** The prompt is injected into JSON; `json.dumps` was
   the right tool, but the decoder initially double-escaped the quotes it added
   itself. Fixed by separating "external text" (escaped) from "structural JSON"
   (written by the decoder).
3. **A function with zero parameters.** The closing `}` was written inside the
   parameter loop, so a zero-parameter function produced invalid JSON. Fixed by
   handling `param_count == 0` before the loop.
4. **Tokens are not characters.** A single token can contain `",` — i.e. the end
   of a string *and* the beginning of a separator. The string automaton stores
   the position of the closing quote inside the token and truncates there.
5. **Numbers are not floats.** "number" must accept `2`, `-3`, `2.5`, but never
   `2.` or `--3` or `.5`; the mask encodes the grammar, and the generator must
   not reformat the value (`2` must stay `2` in the JSON text).
6. **`bool` is an `int` in Python.** `value_matches` excludes `bool` from the
   numeric types explicitly, otherwise `true` would silently be accepted as a
   number.
7. **Hugging Face offline mode.** The environment variables are read at import
   time, so they must be set before the lazy `llm_sdk` import — hence the
   `--cache_model` flag sets them inside `main()`, just before importing. Without
   the flag, a network that *hangs* costs a timeout per hub file (~13 s each,
   measured) even though the model is already cached.
8. **Cache invalidation.** A cache keyed on the prompt only would return stale
   answers after a change in the function catalogue. The key is
   `sha256(version | prompt | canonical(function definitions))`, the file carries
   its own `version`, and every hit is re-validated before being used.
9. **A cache must not hide errors.** A failing prompt is never stored: otherwise
   one transient failure would become permanent. A stale entry is a *miss*, never
   an exception.

## Testing strategy

The central idea: **the decoder, the generator, the cache and the loader are all
testable without the model**, so the suite runs in seconds and can be run on
every change.

* `FakeLLM` — char-level `encode`/`decode` plus a scripted list of logits, so a
  test can decide exactly which token wins at each step. It asserts that the
  script is non-empty: any unexpected model call fails the test loudly.
* `FakeVocab` — a duck-typed vocabulary with 200 logits, known ids for `"`,
  `true`, `false`, and hand-built masks.
* `FakeTensor` / `FakeModel` — exercise the `LargeLanguageModel` wrapper (logits
  memoisation, float32 conversion, call counting) without torch.
* `OfflineGenerator` — a `Generator` subclass whose `model_post_init` loads
  nothing, so orchestration (one entry per prompt, fallback, file writing,
  metrics, cache) is tested in milliseconds.
* `model_construct` — builds pydantic models without validation when a test needs
  to inject a fake object.

Rules the suite follows:

* **one failing test = one bug**, with the file and line in the message;
* green **characterisation tests** for behaviours that are intentional (prompt
  trimming, empty function list rejected by the CLI, `enumerate` masks relying
  on contiguous vocab ids — the last one is guarded by a test that checks the
  real `vocab.json` is 0..N-1);
* every bug fixed gets a regression test that would have caught it;
* the suite is split by module (`test_loader`, `test_writer`, `test_decoder`,
  `test_generator`, `test_vocab`, `test_llm`, `test_schema`, `test_metrics`,
  `test_cache`, `test_offline`, `test_main`, `test_utils`) and complemented by
  tests against the **real** `vocab.json` when it is present in the cache
  (skipped otherwise).

Run it with:

```bash
uv run python -m pytest -q            # 119 tests, no model, a few seconds
uv run flake8 tests/ && uv run python -m mypy tests/
```

## Bonus part


As a bonus, the following features were added to the project:

Support for other models:
model_name	Params	Context	Notes
Qwen/Qwen3-0.6B	0.6B	32K	Your current default
meta-llama/Llama-3.2-1B	1.2B	128K	Community licence
openbmb/MiniCPM5-1B	1.1B	131K	LlamaForCausalLM arch, on-device SOTA
huggingface.co/smollm-135m	135M	—	Tiny, edge
huggingface.co/smollm-360m	360M	—	Edge
tiiuae/Falcon-H1-Tiny-Multilingual-100M	90M	—	Hybrid Mamba

Pretty straigthforward, it was only needed to pass a new argument called --model_name, and pass it to the constructor of the Small_LLM_Model, it only causes more difficulties if you plan to use your own Tokenizer as well, since it is not enough just to replicate the results of the Qwen model, but to actually use the vocab.json and merges.txt files from the model to be able to get the same tokens and ids to use. (For that, it's necessary to use the methods (get_path_to_vocab_file and get_path_to_merges_file))

 a visualizer was added using the blessed library. This was the most difficult part since it meant make some changes within the generation process, again, the event emitter pattern was used here to be able to make the Gui "listen" for events that the generator is going to emit and then allow it to run the functions in charge of printing to the terminal.

## Resources

**Model & inference**

* Qwen3-0.6B model card — <https://huggingface.co/Qwen/Qwen3-0.6B>
* Hugging Face `huggingface_hub` cache management and offline mode —
  <https://huggingface.co/docs/huggingface_hub/guides/manage-cache>
* Hugging Face environment variables —
  <https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables>
* Transformers documentation — <https://huggingface.co/docs/transformers>

**JSON, schema and validation**

* RFC 8259 (JSON) — <https://www.rfc-editor.org/rfc/rfc8259>
* JSON Schema — <https://json-schema.org/>
* Pydantic v2 documentation — <https://docs.pydantic.dev/latest/>

**Numerics**

* `numpy.argmax` — <https://numpy.org/doc/stable/reference/generated/numpy.argmax.html>
* `numpy.where` (masking) — <https://numpy.org/doc/stable/reference/generated/numpy.where.html>

**Tokenisation** (used to build the mask tables)

* GPT-2 byte-level BPE reference implementation —
  <https://github.com/openai/gpt-2/blob/master/src/encoder.py>
* Hugging Face tokenizers documentation — <https://huggingface.co/docs/tokenizers>

### Use of AI

An AI assistant (OpenCode) was used during this project, mainly as a reviewer
and a sparring partner:

* **Understanding** — explaining constrained decoding, token masks, byte-level
  BPE, Hugging Face caching and pydantic v2 idioms.
* **Tests** — designing and writing the `tests/` suite: fakes (`FakeLLM`,
  `FakeVocab`, `OfflineGenerator`), the fake-based decoder tests, the
  cache/metrics contracts, and the bug reports (file + line + failing
  assertion). This is the part with the largest AI contribution.
* **Review** — reading the code and pointing at concrete defects (unescaped
  parameter names, `OutputError.strerror`, `enumerate` over `values()`,
  `metrics.write()` signature, cache key mismatch), each reported as a failing
  test or a precise reference.
* **Design discussion** — options for the L1 cache (key content, invalidation,
  atomicity), metrics, and offline mode.
* **Documentation** — drafting this README from the code and from measured
  numbers; the measurements were run locally.

All the production code in `src/` was written by the student. The only exception
is a one-line rounding fix in `fmt_time` (`src/core/utils.py`), explicitly
requested by the student after a failing test showed `0:60.00` instead of
`1:00.00`.

## Known limitations

* No KV cache in the SDK: one forward pass per generated token.
* `string` parameters support JSON escapes but not `\uXXXX` sequences.
* The cache is invalidated by a manual `CACHE_VERSION` bump when the decoding
  algorithm changes (by design: a hash cannot guess that).
* Semantic accuracy depends entirely on the 0.6B model; the decoder only
  guarantees the format.
