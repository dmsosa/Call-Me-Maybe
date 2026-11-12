from llm_sdk import Small_LLM_Model


def fmt_time(seconds: float) -> str:
    """Format a duration as minutes:seconds, e.g. 105.3 -> "1:45.30".

    divmod(105.3, 60) gives (1.0, 45.3): whole minutes and the rest.
    {s:05.2f} = 2 decimals, padded with zeros to 5 characters, so
    5.3 seconds shows as "05.30" (like a clock).
    """
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:05.2f}"


def to_ids(model: Small_LLM_Model, text: str) -> list[int]:
    """Encodes a text into token ids, ready to append to the stream.
    The structure text (keys, quotes, braces) is known beforehand, so unlike
    the free zones we don't have to guess it token by token — encode() does
    the splitting for us.
    """
    ids: list[int] = model.encode(text).tolist()[0]
    return ids


def build_prompt(functions) -> list[int]:
    pass