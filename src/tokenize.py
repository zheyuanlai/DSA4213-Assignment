r"""Cleaning and deterministic PTB/GloVe-style word tokenisation (TASK §2).

Cleaning, in this exact order:
  1. <br />, <br/>, <br> → space
  2. html.unescape
  3. character normalisation to ASCII punctuation:
       - C1 control characters U+0080–U+009F (Windows-1252 bytes mis-decoded as Latin-1) are
         re-interpreted as cp1252 (\x91 \x92 → ‘ ’, \x93 \x94 → “ ”, \x96 → –, \x97 → —, \x85 → …,
         \x84 → „, \x82 → ‚, \x8b \x9b → ‹ ›, …); undefined cp1252 bytes are removed;
       - curly/angle single quotes, prime, back-tick and the standalone acute accent (U+00B4,
         used as an apostrophe, e.g. "don´t") → ' ; curly/low double quotes and the standalone
         diaeresis U+00A8 (used as a double quote, e.g. ¨cabaret¨) → " ;
       - hyphens/en dash (U+2010–U+2013, U+2212) → - ; em dash/horizontal bar (U+2014, U+2015) → -- ;
         ellipsis U+2026 → ... ;
       - remaining control (Cc) and format/zero-width (Cf) characters are removed.
  4. lowercase
  5. collapse whitespace (Unicode \s, so NBSP is included)

Tokenisation of the cleaned text: clitics are split first (don't → do n't, it's → it 's, likewise
're 've 'll 'd 'm); then one regex emits, in priority order: clitic tokens; dotted acronyms
((?:letter.){2,}: h.p., u.s., e.g., i.e.); the titles mr. mrs. ms. dr. jr. sr. st. vs.; numbers
with internal separators (3.5, 10,000, 8/10); Unicode-aware words with internal hyphens or
apostrophes (well-made, o'clock, café); "..."; "--"; any other non-space character on its own.
"""
from __future__ import annotations

import html
import re
import unicodedata

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")

# --- step 3a: C1 range → cp1252 meaning (undefined bytes 0x81 0x8D 0x8F 0x90 0x9D → removed)
_C1_MAP: dict[int, str] = {}
for _b in range(0x80, 0xA0):
    try:
        _C1_MAP[_b] = bytes([_b]).decode("cp1252")
    except UnicodeDecodeError:
        _C1_MAP[_b] = ""

# --- step 3b: quotes, apostrophes, dashes, ellipsis → ASCII
_PUNCT_MAP = {
    # single quotes / apostrophes
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "ʼ": "'",
    "‹": "'", "›": "'", "`": "'", "´": "'",
    # double quotes (U+00A8 diaeresis is used as a double quote in this corpus, e.g. ¨cabaret¨)
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"', "¨": '"',
    # dashes
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "−": "-",
    "—": "--", "―": "--",
    # ellipsis
    "…": "...",
}
_PUNCT_TABLE = str.maketrans({**{chr(k): v for k, v in _C1_MAP.items()}, **_PUNCT_MAP})


def _strip_controls(text: str) -> str:
    """Remove remaining control (Cc) and format/zero-width (Cf) characters, keeping whitespace."""
    return "".join(ch for ch in text if ch in " \t\n\r" or unicodedata.category(ch) not in ("Cc", "Cf"))


_CLITIC_NT = re.compile(r"(\w)n't\b")
_CLITIC_OTHER = re.compile(r"(\w)'(s|re|ve|ll|d|m)\b")

TITLES = ("mr", "mrs", "ms", "dr", "jr", "sr", "st", "vs")

TOKEN_RE = re.compile(
    r"n't"                                   # negation clitic (after splitting)
    r"|'(?:s|re|ve|ll|d|m)\b"                # other clitics (after splitting)
    r"|(?:[^\W\d_]\.){2,}"                   # dotted acronyms / initials: h.p. u.s. e.g. i.e. a.k.a.
    r"|\b(?:" + "|".join(TITLES) + r")\."     # titles: mr. mrs. ms. dr. jr. sr. st. vs.
    r"|\d+(?:[.,:/]\d+)+"                    # numbers with internal separators: 3.5  10,000  8/10
    r"|[^\W_]+(?:[-'][^\W_]+)*"              # words / plain numbers (Unicode letters), internal - and '
    r"|\.\.\."                               # ellipsis
    r"|--"                                   # dash
    r"|\S",                                  # any other single non-space character
    re.UNICODE,
)


def normalise_chars(text: str) -> str:
    text = text.translate(_PUNCT_TABLE)      # C1 → cp1252 and curly quotes → ASCII in one pass …
    text = text.translate(_PUNCT_TABLE)      # … then normalise the quotes/dashes that the C1 step produced
    return _strip_controls(text)


def clean(text: str) -> str:
    text = _BR_RE.sub(" ", text)
    text = html.unescape(text)
    text = normalise_chars(text)
    text = text.lower()
    text = _WS_RE.sub(" ", text).strip()
    return text


def split_clitics(text: str) -> str:
    text = _CLITIC_NT.sub(r"\1 n't", text)
    text = _CLITIC_OTHER.sub(r"\1 '\2", text)
    return text


def tokenize(clean_text: str) -> list[str]:
    """Tokenise an already-cleaned (lower-cased) string."""
    return TOKEN_RE.findall(split_clitics(clean_text))


def clean_and_tokenize(raw: str) -> tuple[str, list[str]]:
    c = clean(raw)
    return c, tokenize(c)


if __name__ == "__main__":  # quick manual check: tokenise stdin line by line
    import sys

    for line in sys.stdin:
        print(" ".join(clean_and_tokenize(line)[1]))
