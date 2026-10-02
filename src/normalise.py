"""
normalise.py
BanglaIndoorVQA: THE one text normalisation function.

Rules are fixed by the Main Paper Guide, section 3.5:
  Unicode NFC, strip ZWNJ/ZWJ that are not doing conjunct work after hasant,
  collapse whitespace, strip terminal danda and punctuation, lowercase Latin.

Import this file. Do not write a second copy of normalise() anywhere,
in any script, for any reason. If the rule changes it changes here, once,
and every asset that depends on it stays consistent.

Usage:
    from normalise import normalise
    normalise("  চেয়ার।  ")   ->  "চেয়ার"
"""

import re
import unicodedata

ZWNJ = "\u200c"          # zero width non joiner
ZWJ = "\u200d"           # zero width joiner
HASANT = "\u09cd"        # Bangla hasant / virama
DANDA = "\u0964"         # Bangla full stop
DOUBLE_DANDA = "\u0965"

# Characters stripped from BOTH ends of a string. Internal punctuation is left
# alone, so a name like  শিল-পাটা  keeps its hyphen.
_EDGE_CHARS = (
    DANDA + DOUBLE_DANDA
    + ".,;:!?"
    + "\"'`"
    + "()[]{}"
    + " \t\n\r"
)


def _strip_zero_width(text):
    """Remove ZWNJ and ZWJ unless the previous character is hasant.

    After a hasant these characters control whether a conjunct forms, so
    removing them would change the word. Everywhere else they are invisible
    noise from copy paste and keyboards, and they break string equality.
    """
    out = []
    for i, ch in enumerate(text):
        if ch in (ZWNJ, ZWJ):
            prev = text[i - 1] if i > 0 else ""
            if prev == HASANT:
                out.append(ch)
            # otherwise: drop it
        else:
            out.append(ch)
    return "".join(out)


def normalise(text):
    """Return the canonical comparable form of a Bangla or English string."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    s = _strip_zero_width(s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip(_EDGE_CHARS)
    s = s.lower()
    return s


if __name__ == "__main__":
    # Self check. Run  python normalise.py  and every line must print OK.
    cases = [
        ("চেয়ার।", "চেয়ার"),
        ("  চেয়ার  ", "চেয়ার"),
        ('"ফ্রিজ"।', "ফ্রিজ"),
        ("Fridge.", "fridge"),
        ("শিল-পাটা।", "শিল-পাটা"),
        ("", ""),
        (None, ""),
    ]
    failures = 0
    for raw, want in cases:
        got = normalise(raw)
        status = "OK  " if got == want else "FAIL"
        if got != want:
            failures += 1
        print(f"{status} normalise({raw!r}) -> {got!r}   expected {want!r}")
    print()
    print(f"{len(cases) - failures}/{len(cases)} self checks passed")
