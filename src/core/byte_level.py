def build_char_to_byte() -> dict[str, int]:
    """
    Maps unicode that GPT-2 use back to their original integer value (in bytes)
    so that only values from 0 to 256 values exist.

    Inverse of GPT-2 bytes_to_unicode ('Ġ' -> space, 'Ċ' -> newline).

    vocab.json keys are byte-level strings: printable bytes map to
    themselves, the other bytes were shifted to chr(256 + n).

    Example: the token for " hello" is stored as "Ġhello" in vocab.json,
    because a space (byte 32) is not printable, so it was shifted to
    chr(256 + 32) = 'Ġ'. This table undoes that shift.
    """
    # Bytes that were kept as themselves: 
    # the printable ASCII range
    # '!'..'~' 
    # and two printable Latin-1 ranges.
    # ord("!") gives the number of a character (33).
    # this contains integers 33-126, 161-172, 174-255
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))

    # printable char -> its own byte value.
    # This is a "dict comprehension": {key: value for item in list}.
    # chr(33) is the reverse of ord: number -> character ("!").
    table = {chr(b): b for b in bs}

    # Every other byte (space, newline, control chars...) was given a
    # replacement character chr(256), chr(257), ... in increasing order.
    n = 0
    for b in range(256):
        if b not in bs:
            table[chr(256 + n)] = b
            n += 1

    # Result: a table of the form 
    return table


def build_byte_to_char() -> dict[int, str]:
    # the byte or number 32, is going to be mapped
    # to a special char, Ġ
    return {value: key for key, value in build_char_to_byte().items()}
