"""Edge-case checks for cleaning and tokenisation (TASK §2)."""
import pytest

from src.tokenize import clean, clean_and_tokenize, tokenize

CASES = [
    # clitics
    ("I don't think it's good.", ["i", "do", "n't", "think", "it", "'s", "good", "."]),
    ("can't won't they're I've you'll he'd I'm", ["ca", "n't", "wo", "n't", "they", "'re", "i", "'ve", "you", "'ll", "he", "'d", "i", "'m"]),
    # hyphens, numbers
    ("well-made 10,000 3.5/10 1980s 8/10 in 1995.", ["well-made", "10,000", "3.5/10", "1980s", "8/10", "in", "1995", "."]),
    # internal apostrophes, ellipsis, dashes, repeated punctuation
    ("o'clock rock'n'roll -- yes!! wait... ok", ["o'clock", "rock'n'roll", "--", "yes", "!", "!", "wait", "...", "ok"]),
    # accented letters stay whole
    ("café amélie naïve São Paulo", ["café", "amélie", "naïve", "são", "paulo"]),
    # dotted acronyms / initials and titles
    ("H.P. Lovecraft lives in the U.S. e.g. i.e. a.k.a. Mr. Mrs. Ms. Dr. Jr. Sr. St. vs.",
     ["h.p.", "lovecraft", "lives", "in", "the", "u.s.", "e.g.", "i.e.", "a.k.a.", "mr.", "mrs.", "ms.", "dr.", "jr.", "sr.", "st.", "vs."]),
    # titles must not fire inside words
    ("the 1st. the cast. films. first.", ["the", "1st", ".", "the", "cast", ".", "films", ".", "first", "."]),
    # C1 controls (Windows-1252): \x96 en dash, \x97 em dash, \x85 ellipsis, \x91/\x92 quotes, \x93/\x94 quotes
    ("a movie \x96 only gifted \x97 out \x85 yet \x91quoted\x92 \x93double\x94 don\x92t",
     ["a", "movie", "-", "only", "gifted", "--", "out", "...", "yet", "'", "quoted", "'", '"', "double", '"', "do", "n't"]),
    # Unicode dashes, ellipsis, acute accent as apostrophe, back-tick
    ("good – bad — ugly… don´t it`s ¨cabaret¨", ["good", "-", "bad", "--", "ugly", "...", "do", "n't", "it", "'s", '"', "cabaret", '"']),
    # zero-width / control characters removed, NBSP collapsed
    ("zero​width﻿ and nbsp \x07bell", ["zerowidth", "and", "nbsp", "bell"]),
    # <br> variants and html entities
    ("Hello<br />World<br/>again<br>and &amp; &quot;x&quot; &#150; y", ["hello", "world", "again", "and", "&", '"', "x", '"', "-", "y"]),
    # curly quotes
    ("“quoted” ‘single’ actors'", ['"', "quoted", '"', "'", "single", "'", "actors", "'"]),
]


@pytest.mark.parametrize("raw,expected", CASES)
def test_tokenize_cases(raw, expected):
    _, toks = clean_and_tokenize(raw)
    assert toks == expected


def test_clean_is_ascii_punctuation_only():
    c = clean("\x96\x97\x85\x91\x92\x93\x94–—…´`​")
    assert c == "---...''\"\"---...''"


def test_tokenize_deterministic_and_idempotent_clean():
    raw = "It's a well-made film... Mr. Smith's U.S. tour — 10/10!"
    c1, t1 = clean_and_tokenize(raw)
    c2, t2 = clean_and_tokenize(raw)
    assert t1 == t2 and c1 == c2 and clean(c1) == c1
    assert tokenize(c1) == t1
