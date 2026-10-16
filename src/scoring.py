"""
scoring.py  (ASSET-9)
BanglaIndoorVQA main paper: THE deterministic scorer. Design v2.1, section 8.

No LLM judge, no human judgement, no randomness. The same answer always gets
the same label. Import this file. Never copy a function out of it.

What it implements
    8.1  normalise() (imported, never copied) and lead-in stripping
    8.2  answer language and fidelity
    8.3  naming items: Tier 1, Tier 2, sibling, ambiguous, English copy, other
    8.4  count, room, material, colour
    8.5  forced choice
    8.6  strict and lenient scores

Typical use
    from scoring import load_lexicon, load_closed_vocab, score_item
    lex = load_lexicon("lexicon.csv")
    voc = load_closed_vocab("closed_vocab.csv")
    rec = score_item(item, "L-BB", model_answer, lex, voc)

    item is one probe record in the P-CORE.json record format. Only qa_id,
    qa_type and answer_bn are read. The gold is always answer_bn: English
    answers are scored through the lexicon's English names and the vocabulary
    (Design 4.6), so no English gold answer is ever needed.

FILE CONTRACTS (the ASSET-4 v2 script must write exactly these)
    lexicon.csv
        The 16 columns of Design 4.2, UTF-8. Several values in one cell are
        separated by "|" (";" is read too). The scorer reads name_id,
        name_bn, variants_bn, canonical_en, variants_en, super_bn, super_en,
        cluster_id. Siblings are the other objects with the same cluster_id;
        the siblings column is for people and is not read. Objects outside
        every cluster have an empty cluster_id.
    closed_vocab.csv
        Columns: kind, class_id, forms_bn, forms_en, note. One row per class.
        kind is room, material, colour or number. For numbers, class_id is
        the number in Latin digits. forms_bn holds every accepted form in
        Bangla letters (English words in Bangla letters included); forms_en
        holds every accepted form in Latin letters. Same separator.

RULES BUILT IN HERE THAT THE DESIGN DOES NOT STATE
(letters as in the ASSET-9 chat; D numbers are given at the chat close)
    A. File contracts: the two formats above.
    B. Spacing rule: a name matches whether or not its words are separated
       by a space or a hyphen.              MatchSettings.spacing_insensitive
    C. Ending rule: a closed list of Bangla noun endings and the English
       plural are accepted on the last word of a name, only when what is
       left is a listed term.               MatchSettings.suffix_tolerance
    D. Strict rule: the correct name, class or number must itself be written
       in the requested language. Fidelity (8.2) is scored separately, on
       the whole answer. Digit-only counts fit any language.
    E. Refusal rule: an empty answer, or a refusal phrase in an answer that
       holds no scoreable content.
    F. Label rule: in forced choice only the labels that were shown count.
       A lone "a" is a label only when it stands alone, is marked like one,
       or ends the answer.
    G. Exact-gold rule: a naming or room gold must equal a listed Bangla
       name exactly; otherwise the item is unscoreable and reported.
    H. English-copy scope: the last one to three Bangla words of the answer
       are romanised and compared (Design 8.3 says "its romanised form").
    (A to H are D102 to D109, addendum B3.)

ADDED IN STAGE 2 (0.2.0-stage2, after Gate A). No scoring rule changed: every
label, strict and lenient value of 0.1.0-stage1 is unchanged.
    I. Excluded items (D110, addendum C5): score_item(..., excluded=<reason>)
       returns gold_status "excluded", scoreable False and every score field
       empty (label, strict, lenient, fidelity, refusal, content_lang_ok,
       content_found, fc_*). load_excluded() reads probes/excluded_items.csv.
       summarise() counts excluded records apart and keeps them out of every
       rate.
    J. content_found (new key, in every record): open conditions only. True
       when the answer holds scoreable content in the sense of rule E: a term
       of any object (not only another object's generic name), an English
       copy, a number, or a class of the gold's kind. None for forced choice,
       unscoreable and excluded items.
    K. Checks on the frozen files, used by the audit and the tests:
       check_5c_fields (addendum A2), parallel_pairs and genitive_only_forms
       (D117 with the ending rule D104).

CHANGED IN 0.2.1-stage2 (D194, made after reading pilot answers)
    L. The refusal list gains 21 phrases, taken from the words a native
       speaker marked as refusals in the no-image pilot answers (L-BB0,
       L-EE0). The ending list is unchanged. No label, strict or lenient
       value changes: a refusal phrase only matters when no content is found.
    M. refusal_phrase (new key, in every record): True when the answer holds a
       listed refusal phrase, whatever else it holds; None for excluded items.
       It lets later chats count answers scored on content that also refuse
       (a count answer whose একটি is an article, D194).
"""

import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    from normalise import normalise
except ImportError:  # normalise.py sits next to this file in src/
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from normalise import normalise

SCORER_VERSION = "0.2.1-stage2"

# --------------------------------------------------------------------------
# 0. Labels, settings, fixed lists
# --------------------------------------------------------------------------

TIER1 = "tier1"
TIER2 = "tier2"
SIBLING = "sibling"
AMBIGUOUS = "ambiguous"
ENGLISH_COPY = "english_copy"
OTHER_WRONG = "other_wrong"
CORRECT = "correct"
WRONG = "wrong"
INVALID = "invalid"

NAMING_TYPES = ("obj_name", "object")
CLOSED_TYPES = ("obj_name", "object", "count", "scene_id", "obj_property")

ZWNJ = "\u200c"
ZWJ = "\u200d"
HASANT = "\u09cd"
NUKTA = "\u09bc"


@dataclass(frozen=True)
class MatchSettings:
    """The two matching rules the design does not state (rules B and C)."""
    spacing_insensitive: bool = True
    suffix_tolerance: bool = True

    def describe(self):
        return ("spacing-insensitive %s, suffix tolerance %s"
                % ("ON" if self.spacing_insensitive else "OFF",
                   "ON" if self.suffix_tolerance else "OFF"))


DEFAULT_SETTINGS = MatchSettings()

# Design 8.1: the fixed list of lead-in phrases. Longest first.
LEAD_INS = tuple(sorted(
    (normalise(x) for x in
     ("এটি একটি", "এটা", "এটি", "it is a", "this is a", "it's a")),
    key=len, reverse=True))

# Rule C: Bangla endings accepted on the last word of a name. A stripped
# form only counts when what is left is itself a listed term, so a wrong
# strip can never create a match on its own.
BN_SUFFIXES = tuple(sorted(
    {normalise(x) for x in
     ("টি", "টা", "টো", "টে", "খানা", "খানি",      # classifiers
      "গুলো", "গুলি", "গুলা",                      # plural
      "ের", "র", "য়ের",                           # genitive
      "তে", "ে", "য়", "কে")},                      # locative, objective
    key=len, reverse=True))
BN_SUFFIX_DEPTH = 2          # "বালতিটির" = বালতি + টি + র
MIN_STEM_LEN = 2
# What each ending marks, as in the comments above. Read only by the
# proofreading sheet (B5); the tests check that it holds the same endings.
BN_SUFFIX_ROLES = {normalise(x): role for role, xs in (
    ("classifier", ("টি", "টা", "টো", "টে", "খানা", "খানি")),
    ("plural", ("গুলো", "গুলি", "গুলা")),
    ("genitive", ("ের", "র", "য়ের")),
    ("locative", ("তে", "ে", "য়")),
    ("objective", ("কে",))) for x in xs}
# Genitive endings that cannot be the natural end of a word, for the D117
# check genitive_only_forms() (the bare "র" is left out: many words end in it).
GENITIVE_ENDINGS = tuple(sorted({normalise("য়ের"), normalise("ের")},
                                key=len, reverse=True))

# Design 8.3: English copy threshold, normalised edit similarity 0.6 or more.
# Kept as a fraction so that exactly 0.6 passes without float rounding.
EN_COPY_THRESHOLD = (3, 5)
EN_COPY_MAX_WINDOW = 3

# Rule E: refusal phrases. Matched as whole words, in this order of words.
REFUSAL_PHRASES = tuple(normalise(x) for x in (
    # English
    "cannot", "can't", "can not", "unable", "not able", "don't know",
    "do not know", "not sure", "no image", "not visible", "not provided",
    "not clear", "unclear", "unknown", "sorry",
    # Bangla
    "জানি না", "জানিনা", "জানা নেই", "বলতে পারছি না", "বলতে পারি না",
    "বলা সম্ভব নয়", "বলা যাচ্ছে না", "বোঝা যাচ্ছে না", "দেখা যাচ্ছে না",
    "দেখতে পাচ্ছি না", "দেখতে পারছি না", "ছবি নেই", "কোনো ছবি নেই",
    "কোন ছবি নেই", "ছবি দেওয়া হয়নি", "ছবি প্রদান", "ছবিটি দিন",
    "দুঃখিত", "নিশ্চিত নই", "অজানা", "সম্ভব নয়",
    # added in 0.2.1-stage2 (D194), from the proofreading of the pilot
    "missing", "unspecified", "nothing", "please provide", "need to see",
    "didn't include",
    "প্রদান করা হয়নি", "দেওয়া হয়নি", "দেখতে পাইনি", "অনুপস্থিত",
    "নির্দিষ্ট নয়", "নির্দেশিত নয়", "উল্লেখ করা হয়নি", "উল্লেখ নেই",
    "উপাদান নেই", "স্পষ্ট নয়", "অসম্পূর্ণ", "প্রদান করুন", "প্রদান করেননি",
    "সম্ভব হচ্ছে না", "চিত্র নেই",
))

BN_LABELS = tuple(normalise(x) for x in ("ক", "খ", "গ", "ঘ"))
EN_LABELS = ("a", "b", "c", "d")
_LABEL_CUE_WORDS = ("option", "choice", "letter")

LEXICON_COLUMNS = (
    "name_id", "name_bn", "variants_bn", "canonical_en", "variants_en",
    "super_bn", "super_en", "cluster_id", "siblings", "room_scope",
    "cultural_specificity", "origin3", "origin_note", "en_route",
    "en_unique", "freq_obj_name")
VOCAB_COLUMNS = ("kind", "class_id", "forms_bn", "forms_en", "note")
VOCAB_KINDS = ("room", "material", "colour", "number")
_MULTI_SEP = re.compile(r"[|;\n]")


class LexiconError(ValueError):
    """The lexicon or vocabulary file breaks a rule the scorer depends on."""


# --------------------------------------------------------------------------
# 1. Conditions (Design section 7: L, N, R; B or E; trailing 0 = no image)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    cid: str
    experiment: str          # L, N or R
    question_lang: str       # bn or en
    answer_lang: str         # bn or en; "" for forced choice
    image: bool
    task: str                # open or fc
    specificity: bool = False


def _cond(cid, q, a, image=True, task="open", specificity=False):
    return Condition(cid, cid[0], q, a, image, task, specificity)


CONDITIONS = {c.cid: c for c in (
    _cond("L-BB", "bn", "bn"), _cond("L-BE", "bn", "en"),
    _cond("L-EB", "en", "bn"), _cond("L-EE", "en", "en"),
    _cond("L-BB0", "bn", "bn", image=False),
    _cond("L-EE0", "en", "en", image=False),
    _cond("N-BB", "bn", "bn"), _cond("N-BE", "bn", "en"),
    _cond("N-BBs", "bn", "bn", specificity=True),
    _cond("N-FCB", "bn", "", task="fc"),
    _cond("N-FCB0", "bn", "", image=False, task="fc"),
    _cond("N-FCE", "en", "", task="fc"),
    _cond("N-FCE0", "en", "", image=False, task="fc"),
    _cond("R-BB", "bn", "bn"), _cond("R-EE", "en", "en"),
    _cond("R-BB0", "bn", "bn", image=False),
)}


def condition_spec(condition_id):
    """Return the Condition for an ID from Design section 7, or fail loudly."""
    try:
        return CONDITIONS[condition_id]
    except KeyError:
        raise KeyError("Unknown condition_id %r. Known: %s"
                       % (condition_id, ", ".join(sorted(CONDITIONS))))


# --------------------------------------------------------------------------
# 2. Text: preparation, tokens, language
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Tok:
    text: str        # zero-width characters removed
    kind: str        # bn, lat, dig or oth
    start: int
    end: int


_BN_CLASS = "\u0980-\u09e3\u09f0\u09f1\u09fc\u09fe"
_TOKEN_RE = re.compile(
    "(?P<bn>[" + _BN_CLASS + ZWNJ + ZWJ + "]+)"
    "|(?P<lat>[a-z]+(?:'[a-z]+)*)"
    "|(?P<dig>[0-9\u09e6-\u09ef]+)"
    "|(?P<oth>[^\\W\\d_]+)")
_QUOTES = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}
_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate("০১২৩৪৫৬৭৮৯")}


def tokenise(text):
    """Split normalised text into whole words. Punctuation, spaces and
    hyphens separate words. Zero-width characters are dropped from each word,
    so the two ways of typing র‍্য can never split a match."""
    out = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        t = m.group()
        if kind == "bn":
            t = t.replace(ZWNJ, "").replace(ZWJ, "")
            if not t:
                continue
        out.append(Tok(t, kind, m.start(), m.end()))
    return out


def strip_lead_in(text):
    """Design 8.1: remove one lead-in phrase from the start of the answer."""
    for lead in LEAD_INS:
        if text == lead:
            return ""
        if text.startswith(lead) and not _is_word_char(text[len(lead)]):
            return normalise(text[len(lead):])
    return text


def _is_word_char(ch):
    return bool(_TOKEN_RE.fullmatch(ch)) or ch in (HASANT, NUKTA)


def detect_language(tokens):
    """Design 8.2. Bangla if Bangla letters are at least half of all letters;
    English if Latin letters are; otherwise mixed. Bangla vowel signs count as
    letters, as Latin vowels do. Digits only: 'digits'. Nothing: 'none'."""
    bn = sum(len(t.text) for t in tokens if t.kind == "bn")
    lat = sum(len(t.text.replace("'", "")) for t in tokens if t.kind == "lat")
    oth = sum(len(t.text) for t in tokens if t.kind == "oth")
    total = bn + lat + oth
    if total == 0:
        lang = "digits" if any(t.kind == "dig" for t in tokens) else "none"
    elif 2 * bn >= total:
        lang = "bn"
    elif 2 * lat >= total:
        lang = "en"
    else:
        lang = "mixed"
    return {"lang": lang, "bn_letters": bn, "lat_letters": lat,
            "other_letters": oth}


def is_fidelity(lang, requested):
    """Design 8.2: digits only match any requested language."""
    return lang == requested or lang == "digits"


@dataclass
class Prepared:
    raw: str
    norm: str            # normalise() only
    core: str            # lead-in removed
    tokens: list
    norm_tokens: list
    lang: dict


def prepare_answer(raw):
    """normalise(), strip the lead-in, tokenise, detect the language."""
    raw = "" if raw is None else str(raw)
    for k, v in _QUOTES.items():
        raw = raw.replace(k, v)
    norm = normalise(raw)
    core = strip_lead_in(norm)
    tokens = tokenise(core)
    return Prepared(raw, norm, core, tokens, tokenise(norm),
                    detect_language(tokens))


# --------------------------------------------------------------------------
# 3. Romanisation and edit similarity (English copies; also the en_route
#    cross-check of Design 4.4, which must import these, not rewrite them)
# --------------------------------------------------------------------------

_RRA, _RHA, _YYA = "\u09dc", "\u09dd", "\u09df"      # internal symbols only
_NUKTA_MERGE = {"ড": _RRA, "ঢ": _RHA, "য": _YYA}
_BN_CONS = {
    "ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh", "ঙ": "ng", "চ": "ch",
    "ছ": "ch", "জ": "j", "ঝ": "jh", "ঞ": "n", "ট": "t", "ঠ": "th",
    "ড": "d", "ঢ": "dh", "ণ": "n", "ত": "t", "থ": "th", "দ": "d",
    "ধ": "dh", "ন": "n", "প": "p", "ফ": "f", "ব": "b", "ভ": "v",
    "ম": "m", "য": "j", "র": "r", "ল": "l", "শ": "sh", "ষ": "sh",
    "স": "s", "হ": "h", "ৎ": "t", _RRA: "r", _RHA: "rh", _YYA: "y",
}
_BN_VSIGN = {"া": "a", "ি": "i", "ী": "i", "ু": "u", "ূ": "u", "ৃ": "ri",
             "ে": "e", "ৈ": "oi", "ো": "o", "ৌ": "ou"}
_BN_VIND = {"অ": "o", "আ": "a", "ই": "i", "ঈ": "i", "উ": "u", "ঊ": "u",
            "ঋ": "ri", "এ": "e", "ঐ": "oi", "ও": "o", "ঔ": "ou"}
_BN_SILENT_END = ("ঁ", "ঃ")


def _romanise_word(word):
    """One Bangla word to Latin letters. The table follows how English words
    are written in Bangla letters (ফ = f, ভ = v, ্যা = a, ওয় = w), because its
    only job is to compare a Bangla-script answer with an English name."""
    syms = []
    i = 0
    while i < len(word):
        ch = word[i]
        nxt = word[i + 1] if i + 1 < len(word) else ""
        if nxt == NUKTA and ch in _NUKTA_MERGE:
            syms.append(_NUKTA_MERGE[ch])
            i += 2
        elif ch in (NUKTA, ZWJ, ZWNJ):
            i += 1
        else:
            syms.append(ch)
            i += 1

    n = len(syms)

    def at(k):
        return syms[k] if k < n else ""

    out = []
    vowel_seen = False
    p = 0
    while p < n:
        c = syms[p]
        if c in _BN_CONS:
            base = _BN_CONS[c]
            if c == "ঙ" and at(p + 1) == HASANT and at(p + 2) in "কখগঘ":
                base = "n"
            if at(p + 1) == HASANT:
                if at(p + 2) == "য":                    # ya-phala
                    out.append(base)
                    if at(p + 3) == "া":
                        out.append("a")
                        p += 4
                    else:
                        out.append("y")
                        p += 3
                    vowel_seen = True
                else:
                    out.append(base)
                    p += 2
                continue
            if at(p + 1) in _BN_VSIGN:
                out.append(base + _BN_VSIGN[at(p + 1)])
                vowel_seen = True
                p += 2
                continue
            out.append(base)
            is_final = all(s in _BN_SILENT_END for s in syms[p + 1:])
            if c != _YYA and not vowel_seen and not is_final:
                out.append("o")                         # inherent vowel
                vowel_seen = True
            p += 1
            continue
        if c in _BN_VIND:
            if c in ("অ", "এ") and at(p + 1) == HASANT and at(p + 2) == "য":
                out.append("a")
                p += 4 if at(p + 3) == "া" else 3
                vowel_seen = True
                continue
            if c == "ও" and at(p + 1) == _YYA:
                out.append("w")
                p += 2
                continue
            if c == "উ" and at(p + 1) == "ই":
                out.append("wi")
                p += 2
                vowel_seen = True
                continue
            out.append(_BN_VIND[c])
            vowel_seen = True
            p += 1
            continue
        if c in _BN_VSIGN:
            out.append(_BN_VSIGN[c])
            vowel_seen = True
        elif c == "ং":
            out.append("ng")
        p += 1
    return "".join(out)


def romanise_bn(text):
    """Romanise every Bangla word of a string; other words pass through."""
    parts = []
    for t in tokenise(normalise(text)):
        parts.append(_romanise_word(t.text) if t.kind == "bn" else t.text)
    return " ".join(parts)


def edit_distance(a, b):
    """Levenshtein distance."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def edit_similarity(a, b):
    """Normalised edit similarity: 1 - distance / length of the longer one."""
    m = max(len(a), len(b))
    return 1.0 if m == 0 else 1.0 - edit_distance(a, b) / m


def is_similar(a, b, threshold=EN_COPY_THRESHOLD):
    """True if edit_similarity(a, b) >= threshold, in exact arithmetic."""
    m = max(len(a), len(b))
    if m == 0:
        return False
    num, den = threshold
    return (m - edit_distance(a, b)) * den >= num * m


def _letters_only(term):
    return re.sub(r"[^a-z]", "", term)


def find_english_copy(tokens, english_terms):
    """Design 8.3, English copy. Bangla puts the noun last, so in a short
    answer the name sits at the end: take the last one, two or three Bangla
    words, romanise them, and compare with every English Tier-1 name of the
    gold object. Word endings are not removed here: tried on real answers,
    that only added false matches. Returns (similarity, English term, Bangla
    words) for the best pass, or None."""
    targets = [(t, _letters_only(t)) for t in english_terms]
    targets = [(t, s) for t, s in targets if s]
    run = []
    for t in reversed(tokens):
        if t.kind == "bn":
            run.insert(0, t)
        elif run:
            break
    best = None
    for k in range(1, min(len(run), EN_COPY_MAX_WINDOW) + 1):
        window = run[-k:]
        rom = "".join(_romanise_word(t.text) for t in window)
        if not rom:
            continue
        for term, target in targets:
            if is_similar(rom, target):
                cand = (edit_similarity(rom, target), term,
                        " ".join(t.text for t in window))
                if best is None or cand[0] > best[0]:
                    best = cand
    return best


# --------------------------------------------------------------------------
# 4. Word endings (rule C) and English name forms (Design 4.4)
# --------------------------------------------------------------------------

def stems_bn(word, depth=BN_SUFFIX_DEPTH):
    """Forms of a Bangla word with up to `depth` listed endings removed,
    longest first."""
    found = []
    frontier = [word]
    for _ in range(depth):
        nxt = []
        for w in frontier:
            for suf in BN_SUFFIXES:
                if w.endswith(suf) and len(w) - len(suf) >= MIN_STEM_LEN:
                    stem = w[:-len(suf)]
                    if stem not in found and stem != word:
                        found.append(stem)
                        nxt.append(stem)
        frontier = nxt
    return sorted(found, key=len, reverse=True)


def stems_en(word):
    """Singular candidates of an English word, plus the form without 's."""
    out = []
    base = word
    if base.endswith("'s") and len(base) > 3:
        base = base[:-2]
        out.append(base)
    if base.endswith("ies") and len(base) > 4:
        out.append(base[:-3] + "y")
    if base.endswith("ves") and len(base) > 4:
        out.append(base[:-3] + "f")
        out.append(base[:-3] + "fe")
    if base.endswith("es") and len(base) > 3:
        out.append(base[:-2])
    if base.endswith("s") and not base.endswith("ss") and len(base) > 2:
        out.append(base[:-1])
    seen = []
    for s in out:
        if len(s) >= MIN_STEM_LEN and s not in seen and s != word:
            seen.append(s)
    return seen


_ARTICLES = ("a", "an", "the")


def english_forms(term):
    """Design 4.4: lower-cased, articles removed, singular. Returns the set of
    comparable forms of one English name (spaces and hyphens ignored)."""
    words = [t.text for t in tokenise(normalise(term))]
    while words and words[0] in _ARTICLES:
        words = words[1:]
    if not words:
        return set()
    head = "".join(words[:-1])
    return {head + w for w in [words[-1]] + stems_en(words[-1])}


def compute_en_unique(objects):
    """Design 4.4: an object is English-unique if its English Tier-1 set
    shares no term with any other object's. Returns {name_id: bool or None}.
    None means the object has no English name yet."""
    owners = {}
    forms = {}
    for oid, obj in objects.items():
        fs = set()
        for term in obj.t1_en:
            fs |= english_forms(term)
        forms[oid] = fs
        for f in fs:
            owners.setdefault(f, set()).add(oid)
    out = {}
    for oid, fs in forms.items():
        if not fs:
            out[oid] = None
        else:
            out[oid] = all(owners[f] == {oid} for f in fs)
    return out


# --------------------------------------------------------------------------
# 5. Term index: whole-word search, longest match first (Design 8.3 step 1)
# --------------------------------------------------------------------------

@dataclass
class Match:
    start: int               # token index, inclusive
    end: int                 # token index, exclusive
    key: str
    senses: list
    exact: bool
    surface: str


class TermIndex:
    def __init__(self, settings=DEFAULT_SETTINGS):
        self.settings = settings
        self.joiner = "" if settings.spacing_insensitive else " "
        self.entries = {}
        self.max_tokens = 1

    def key_of(self, text):
        return self.joiner.join(t.text for t in tokenise(normalise(text)))

    def add(self, term, sense):
        toks = tokenise(normalise(term))
        key = self.joiner.join(t.text for t in toks)
        if not key:
            return ""
        self.max_tokens = max(self.max_tokens, len(toks))
        senses = self.entries.setdefault(key, [])
        if sense not in senses:
            senses.append(sense)
        return key

    def find_all(self, tokens):
        """Every listed term found as whole words. Overlaps are settled by
        taking the longest match, so পড়ার টেবিল wins over টেবিল."""
        n = len(tokens)
        extra = 2 if self.settings.spacing_insensitive else 0
        max_w = min(n, self.max_tokens + extra)
        cands = []
        for i in range(n):
            for j in range(i + 1, min(n, i + max_w) + 1):
                texts = [t.text for t in tokens[i:j]]
                key = self.joiner.join(texts)
                surface = " ".join(texts)
                if key in self.entries:
                    cands.append(Match(i, j, key, self.entries[key], True,
                                       surface))
                if not self.settings.suffix_tolerance:
                    continue
                last = tokens[j - 1]
                if last.kind == "bn":
                    stems = stems_bn(last.text)
                elif last.kind == "lat":
                    stems = stems_en(last.text)
                else:
                    stems = []
                for stem in stems:
                    key2 = self.joiner.join(texts[:-1] + [stem])
                    if key2 in self.entries:
                        cands.append(Match(i, j, key2, self.entries[key2],
                                           False, surface))
        cands.sort(key=lambda m: (-len(m.key.replace(" ", "")),
                                  0 if m.exact else 1, m.start))
        taken = []
        used = set()
        for m in cands:
            span = set(range(m.start, m.end))
            if span & used:
                continue
            used |= span
            taken.append(m)
        taken.sort(key=lambda m: m.start)
        return taken


# --------------------------------------------------------------------------
# 6. Lexicon (Design 4.2) and closed-answer vocabulary (Design 4.5)
# --------------------------------------------------------------------------

def _split_cell(cell):
    out = []
    for part in _MULTI_SEP.split(cell or ""):
        term = normalise(part)
        if term and term not in out:
            out.append(term)
    return out


def _read_csv(path, required):
    path = Path(path)
    try:
        path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise LexiconError(
            "%s is not saved as UTF-8. In Google Sheets use File > Download"
            " > Comma-separated values. In Excel choose 'CSV UTF-8'."
            % path.name)
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        header = [(h or "").strip().lower() for h in (reader.fieldnames or [])]
        missing = [c for c in required if c not in header]
        if missing:
            raise LexiconError("%s: missing column(s): %s. Found: %s"
                               % (path.name, ", ".join(missing),
                                  ", ".join(header)))
        rows = []
        for raw in reader:
            row = {}
            for k, v in raw.items():
                if isinstance(v, list):          # cells beyond the header
                    v = "|".join(x or "" for x in v)
                row[(k or "").strip().lower()] = (v or "").strip()
            if any(row.values()):
                rows.append(row)
    return rows


def _yes_no(value, where, errors):
    v = value.strip().lower()
    if v == "":
        return None
    if v in ("yes", "no"):
        return v == "yes"
    errors.append("%s: must be yes or no, found %r" % (where, value))
    return None


def _script_of(term):
    return detect_language(tokenise(term))["lang"]


@dataclass
class LexObject:
    name_id: str
    name_bn: str
    t1_bn: list
    t1_en: list
    t2_bn: list
    t2_en: list
    cluster_id: str
    en_route: object
    en_unique: object
    row: dict
    keys_t1: set = field(default_factory=set)
    keys_t2: set = field(default_factory=set)


@dataclass(frozen=True)
class Sense:
    name_id: str
    tier: int
    lang: str


class Lexicon:
    def __init__(self, settings):
        self.settings = settings
        self.objects = {}
        self.index = TermIndex(settings)
        self.bn_t1_key_to_id = {}
        self.clusters = {}
        self.warnings = []
        self.path = ""

    def siblings_of(self, name_id):
        cid = self.objects[name_id].cluster_id
        if not cid:
            return set()
        return self.clusters[cid] - {name_id}

    def describe(self):
        return ("%d objects, %d in %d clusters, %d terms, matching: %s"
                % (len(self.objects),
                   sum(1 for o in self.objects.values() if o.cluster_id),
                   len(self.clusters), len(self.index.entries),
                   self.settings.describe()))


def build_lexicon(rows, settings=DEFAULT_SETTINGS, source="lexicon"):
    """Build and validate a Lexicon from row dicts (Design 4.2 columns)."""
    lex = Lexicon(settings)
    lex.path = source
    errors = []
    for n, row in enumerate(rows, 2):                    # row 1 is the header
        where = "%s row %d" % (source, n)
        name_id = row.get("name_id", "")
        name_bn = normalise(row.get("name_bn", ""))
        if not name_id:
            errors.append("%s: empty name_id" % where)
            continue
        if name_id in lex.objects:
            errors.append("%s: duplicate name_id %s" % (where, name_id))
            continue
        if not name_bn:
            errors.append("%s (%s): empty name_bn" % (where, name_id))
            continue
        t1_bn = [name_bn] + [t for t in _split_cell(row.get("variants_bn"))
                             if t != name_bn]
        t1_en = _split_cell(row.get("canonical_en"))
        t1_en += [t for t in _split_cell(row.get("variants_en"))
                  if t not in t1_en]
        obj = LexObject(
            name_id=name_id, name_bn=name_bn, t1_bn=t1_bn, t1_en=t1_en,
            t2_bn=_split_cell(row.get("super_bn")),
            t2_en=_split_cell(row.get("super_en")),
            cluster_id=row.get("cluster_id", ""),
            en_route=_yes_no(row.get("en_route", ""),
                             "%s en_route" % where, errors),
            en_unique=_yes_no(row.get("en_unique", ""),
                              "%s en_unique" % where, errors),
            row=dict(row))
        lex.objects[name_id] = obj
        if obj.cluster_id:
            lex.clusters.setdefault(obj.cluster_id, set()).add(name_id)

        for tier, lang, terms in ((1, "bn", obj.t1_bn), (1, "en", obj.t1_en),
                                  (2, "bn", obj.t2_bn), (2, "en", obj.t2_en)):
            for term in terms:
                key = lex.index.add(term, Sense(name_id, tier, lang))
                if not key:
                    errors.append("%s (%s): term %r has no letters"
                                  % (where, name_id, term))
                    continue
                (obj.keys_t1 if tier == 1 else obj.keys_t2).add(key)
                if tier == 1 and lang == "bn":
                    other = lex.bn_t1_key_to_id.get(key)
                    if other and other != name_id:
                        errors.append(
                            "%s: Bangla Tier-1 name %r belongs to both %s and"
                            " %s. One Bangla name, one object."
                            % (where, term, other, name_id))
                    lex.bn_t1_key_to_id[key] = name_id
                script = _script_of(term)
                if script not in (lang, "digits", "none"):
                    lex.warnings.append(
                        "%s (%s): %r sits in a %s column but is written in"
                        " %s letters" % (where, name_id, term, lang, script))
        both = obj.keys_t1 & obj.keys_t2
        if both:
            errors.append("%s (%s): term(s) %s are both Tier 1 and Tier 2 of"
                          " the same object"
                          % (where, name_id, ", ".join(sorted(both))))
        if not obj.t1_en:
            lex.warnings.append("%s (%s, %s): no English name yet; this object"
                                " cannot be scored in English"
                                % (where, name_id, name_bn))

    if errors:
        raise LexiconError("%d problem(s) in %s:\n  - %s"
                           % (len(errors), source, "\n  - ".join(errors)))

    recomputed = compute_en_unique(lex.objects)
    for oid, obj in lex.objects.items():
        if obj.en_unique is not None and recomputed[oid] is not None \
                and obj.en_unique != recomputed[oid]:
            lex.warnings.append(
                "%s (%s): en_unique says %s, but the English names give %s"
                % (oid, obj.name_bn, "yes" if obj.en_unique else "no",
                   "yes" if recomputed[oid] else "no"))
    for cid, members in lex.clusters.items():
        if len(members) < 2:
            lex.warnings.append("cluster %s has one member only" % cid)
    return lex


def load_lexicon(path, settings=DEFAULT_SETTINGS):
    """Read lexicon.csv (Design 4.2). Raises LexiconError on any rule the
    scorer depends on. Softer findings are kept in .warnings."""
    rows = _read_csv(path, LEXICON_COLUMNS)
    return build_lexicon(rows, settings, Path(path).name)


@dataclass(frozen=True)
class ClassSense:
    class_id: str
    lang: str


class ClosedVocab:
    def __init__(self, settings):
        self.settings = settings
        self.indexes = {k: TermIndex(settings) for k in VOCAB_KINDS}
        self.classes = {k: set() for k in VOCAB_KINDS}
        # forms as written in the file (normalised), per kind and class:
        # {kind: {class_id: {"bn": [...], "en": [...]}}}. Read by the checks
        # of section 10 only; scoring uses the indexes.
        self.forms = {k: {} for k in VOCAB_KINDS}
        self.warnings = []
        self.path = ""

    def describe(self):
        return ", ".join("%s %d" % (k, len(self.classes[k]))
                         for k in VOCAB_KINDS)


def build_closed_vocab(rows, settings=DEFAULT_SETTINGS, source="closed_vocab"):
    voc = ClosedVocab(settings)
    voc.path = source
    errors = []
    owner = {k: {} for k in VOCAB_KINDS}
    for n, row in enumerate(rows, 2):
        where = "%s row %d" % (source, n)
        kind = row.get("kind", "").lower()
        class_id = row.get("class_id", "")
        if kind not in VOCAB_KINDS:
            errors.append("%s: kind must be one of %s, found %r"
                          % (where, ", ".join(VOCAB_KINDS), kind))
            continue
        if not class_id:
            errors.append("%s: empty class_id" % where)
            continue
        if kind == "number":
            if not re.fullmatch(r"[0-9]+", class_id):
                errors.append("%s: a number class_id must be Latin digits,"
                              " found %r" % (where, class_id))
                continue
            class_id = str(int(class_id))
        if class_id in voc.classes[kind]:
            errors.append("%s: duplicate %s class %s"
                          % (where, kind, class_id))
            continue
        voc.classes[kind].add(class_id)
        forms = [("bn", t) for t in _split_cell(row.get("forms_bn"))]
        forms += [("en", t) for t in _split_cell(row.get("forms_en"))]
        if not forms:
            errors.append("%s (%s %s): no forms" % (where, kind, class_id))
            continue
        for lang, term in forms:
            voc.forms[kind].setdefault(class_id, {"bn": [], "en": []})[lang] \
                .append(term)
            key = voc.indexes[kind].add(term, ClassSense(class_id, lang))
            if not key:
                errors.append("%s (%s %s): form %r has no letters"
                              % (where, kind, class_id, term))
                continue
            prev = owner[kind].get(key)
            if prev and prev != class_id:
                errors.append("%s: form %r is in two %s classes, %s and %s"
                              % (where, term, kind, prev, class_id))
            owner[kind][key] = class_id
            script = _script_of(term)
            if script not in (lang, "digits", "none"):
                voc.warnings.append(
                    "%s (%s %s): %r sits in forms_%s but is written in %s"
                    " letters" % (where, kind, class_id, term, lang, script))
    if errors:
        raise LexiconError("%d problem(s) in %s:\n  - %s"
                           % (len(errors), source, "\n  - ".join(errors)))
    if len(voc.classes["room"]) != 4:
        voc.warnings.append("Design 4.5 expects 4 room classes, found %d"
                            % len(voc.classes["room"]))
    shared = set(owner["material"]) & set(owner["colour"])
    for key in sorted(shared):
        voc.warnings.append("form %r is both a material and a colour; a gold"
                            " answer with it cannot be typed" % key)
    return voc


def load_closed_vocab(path, settings=DEFAULT_SETTINGS):
    """Read closed_vocab.csv (ASSET-10). See the file contract at the top."""
    rows = _read_csv(path, VOCAB_COLUMNS)
    return build_closed_vocab(rows, settings, Path(path).name)


def load_probe(path):
    """Read a probe file. Accepts the P-CORE.json wrapper ({"items": [...]})
    or a bare list of records."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    items = data["items"] if isinstance(data, dict) and "items" in data \
        else data
    if not isinstance(items, list):
        raise ValueError("%s: expected a list of records or {'items': [...]}"
                         % path)
    for need in ("qa_id", "qa_type", "answer_bn"):
        bad = [i for i, it in enumerate(items) if need not in it]
        if bad:
            raise ValueError("%s: %d record(s) lack %r (first at index %d)"
                             % (path, len(bad), need, bad[0]))
    return items


# --------------------------------------------------------------------------
# 6b. The excluded-items list (D110, addendum C4)
# --------------------------------------------------------------------------

EXCLUDED_STATUS = "excluded"
EXCLUDED_COLUMNS = ("qa_id", "probes", "qa_type", "reason", "source",
                    "added_at")
EXCLUDED_REASONS = ("gold_not_one_object", "gold_wrong_kind", "gold_wrong",
                    "other")
_QA_ID_RE = re.compile(r"[A-Za-z]+\d+_\d+_q\d+")


class ExcludedListError(ValueError):
    """probes/excluded_items.csv breaks its format (addendum C4)."""


def load_excluded(path):
    """Read probes/excluded_items.csv. Returns {qa_id: reason code}.

    Stops on a missing column, a value that is not a qa_id, an unknown reason
    code, or a qa_id listed twice. The list only grows, one row per item, and
    holds no question or answer text (D110)."""
    try:
        rows = _read_csv(path, EXCLUDED_COLUMNS)
    except LexiconError as exc:
        raise ExcludedListError(str(exc))
    out = {}
    errors = []
    for n, row in enumerate(rows, 2):
        qa_id = row.get("qa_id", "")
        reason = row.get("reason", "")
        if not _QA_ID_RE.fullmatch(qa_id):
            errors.append("row %d: %r is not a qa_id" % (n, qa_id))
        elif qa_id in out:
            errors.append("row %d: %s is listed twice" % (n, qa_id))
        elif reason not in EXCLUDED_REASONS:
            errors.append("row %d: %s has reason %r; known reasons: %s"
                          % (n, qa_id, reason, ", ".join(EXCLUDED_REASONS)))
        else:
            out[qa_id] = reason
    if errors:
        raise ExcludedListError("%d problem(s) in %s:\n  - %s"
                                % (len(errors), Path(path).name,
                                   "\n  - ".join(errors)))
    return out


# --------------------------------------------------------------------------
# 7. Gold answers
# --------------------------------------------------------------------------

@dataclass
class Gold:
    status: str = "ok"       # ok, not_closed_type, gold_unmapped, gold_ambiguous
    kind: str = ""           # naming, count, room, material, colour
    norm: str = ""
    name_id: str = ""
    number: object = None
    classes: frozenset = frozenset()
    hint: str = ""
    residual: tuple = ()


@dataclass
class _Number:
    value: int
    start: int
    end: int
    source: str              # digit or word
    lang: str                # bn, en, or "" for digits


def _first_number(tokens, vocab):
    """Design 8.4: the first number in the text. Digits in either script, or
    a number form from the vocabulary (classifiers handled by rule C)."""
    best = None
    for i, t in enumerate(tokens):
        if t.kind == "dig":
            best = _Number(int(t.text.translate(_DIGIT_MAP)), i, i + 1,
                           "digit", "")
            break
    for m in vocab.indexes["number"].find_all(tokens):
        if best is None or m.start < best.start:
            sense = m.senses[0]
            langs = {s.lang for s in m.senses}
            best = _Number(int(sense.class_id), m.start, m.end, "word",
                           "bn" if langs == {"bn"} else
                           "en" if langs == {"en"} else "bn+en")
        break                                   # find_all is sorted by start
    return best


def _residual(tokens, spans):
    covered = set()
    for a, b in spans:
        covered |= set(range(a, b))
    return tuple(t.text for i, t in enumerate(tokens) if i not in covered)


def resolve_gold(item, lexicon, vocab):
    """Turn answer_bn into what the scorer compares against: an object, a
    number, a room class, or a set of material or colour classes."""
    qa_type = item["qa_type"]
    norm = strip_lead_in(normalise(item["answer_bn"]))
    tokens = tokenise(norm)
    gold = Gold(norm=norm)

    if qa_type in NAMING_TYPES:
        gold.kind = "naming"
        key = lexicon.index.key_of(norm)
        name_id = lexicon.bn_t1_key_to_id.get(key)
        if name_id:
            gold.name_id = name_id
            return gold
        gold.status = "gold_unmapped"
        ids = sorted({s.name_id for m in lexicon.index.find_all(tokens)
                      for s in m.senses if s.tier == 1 and s.lang == "bn"})
        gold.hint = "contains: " + "|".join(ids) if ids else ""
        return gold

    if qa_type == "count":
        gold.kind = "count"
        num = _first_number(tokens, vocab)
        if num is None:
            gold.status = "gold_unmapped"
            return gold
        gold.number = num.value
        gold.residual = _residual(tokens, [(num.start, num.end)])
        return gold

    if qa_type == "scene_id":
        gold.kind = "room"
        matches = vocab.indexes["room"].find_all(tokens)
        classes = {s.class_id for m in matches for s in m.senses}
        whole = len(matches) == 1 and matches[0].start == 0 \
            and matches[0].end == len(tokens) and matches[0].exact
        if whole:
            gold.classes = frozenset(classes)
            return gold
        gold.status = "gold_unmapped"
        gold.hint = "contains: " + "|".join(sorted(classes)) if classes else ""
        return gold

    if qa_type == "obj_property":
        found = {}
        for kind in ("material", "colour"):
            matches = vocab.indexes[kind].find_all(tokens)
            if matches:
                found[kind] = matches
        if len(found) == 1:
            kind, matches = next(iter(found.items()))
            gold.kind = kind
            gold.classes = frozenset(s.class_id for m in matches
                                     for s in m.senses)
            gold.residual = _residual(tokens, [(m.start, m.end)
                                               for m in matches])
            return gold
        gold.status = "gold_ambiguous" if found else "gold_unmapped"
        gold.hint = "material and colour terms" if found else ""
        return gold

    gold.status = "not_closed_type"
    return gold


# --------------------------------------------------------------------------
# 8. Scoring
# --------------------------------------------------------------------------

def _has_refusal_phrase(tokens):
    texts = [t.text for t in tokens]
    for want in _REFUSAL_TOKENS:
        k = len(want)
        for i in range(len(texts) - k + 1):
            if texts[i:i + k] == want:
                return True
    return False


_REFUSAL_TOKENS = tuple([t.text for t in tokenise(p)] for p in REFUSAL_PHRASES)


def _langs(senses):
    return "+".join(sorted({s.lang for s in senses}))


def score_naming(prep, gold, lexicon, requested):
    """Design 8.3. Returns the label, strict and lenient, with the evidence."""
    obj = lexicon.objects[gold.name_id]
    siblings = lexicon.siblings_of(obj.name_id)
    matches = lexicon.index.find_all(prep.tokens)

    gold_t1, gold_t2, others, shown = [], [], [], []
    for m in matches:
        g1 = [s for s in m.senses if s.name_id == obj.name_id and s.tier == 1]
        g2 = [s for s in m.senses if s.name_id == obj.name_id and s.tier == 2]
        o1 = [s for s in m.senses if s.name_id != obj.name_id and s.tier == 1]
        if g1:
            gold_t1.append(g1)
            role = "T1"
        elif g2:                         # generic-name precedence, 8.3 rule 4
            gold_t2.append(g2)
            role = "T2"
        elif o1:
            others.append(o1)
            role = "SIB" if {s.name_id for s in o1} & siblings else "OTHER"
        else:
            role = "other-generic"
        shown.append("%s:%s" % (role, m.surface))

    out = {"matched": shown, "named_ids": [], "label_lang": "",
           "copy_term": None, "copy_sim": None, "content_lang_ok": False}
    no_content = False

    if gold_t1 and not others:
        label = TIER1
        flat = [s for g in gold_t1 for s in g]
        out["label_lang"] = _langs(flat)
        out["content_lang_ok"] = any(s.lang == requested for s in flat)
    elif gold_t1 and others:
        label = AMBIGUOUS
    elif others:
        id_sets = [{s.name_id for s in o} for o in others]
        common = set.intersection(*id_sets)
        if len(others) > 1 and not common:
            label = AMBIGUOUS
        else:
            ids = common or id_sets[0]
            label = SIBLING if ids & siblings else OTHER_WRONG
        out["label_lang"] = _langs([s for o in others for s in o])
    elif gold_t2:
        label = TIER2
        flat = [s for g in gold_t2 for s in g]
        out["label_lang"] = _langs(flat)
        out["content_lang_ok"] = any(s.lang == requested for s in flat)
    else:
        label = OTHER_WRONG
        no_content = True
        if prep.lang["lang"] == "bn" and not matches and obj.t1_en:
            hit = find_english_copy(prep.tokens, obj.t1_en)
            if hit:
                label = ENGLISH_COPY
                no_content = False
                out["copy_sim"] = round(hit[0], 3)
                out["copy_term"] = hit[1]
                out["matched"] = ["COPY:%s" % hit[2]]
    if others:
        out["named_ids"] = sorted({s.name_id for o in others for s in o})

    out["label"] = label
    out["strict"] = int(label == TIER1 and out["content_lang_ok"])
    out["lenient"] = int(label in (TIER1, ENGLISH_COPY))
    out["_no_content"] = no_content
    return out


def score_count(prep, gold, vocab, requested):
    """Design 8.4, count: the first number must equal the gold number."""
    num = _first_number(prep.tokens, vocab)
    lang = prep.lang["lang"]
    correct = num is not None and num.value == gold.number
    if num is None:
        lang_ok = False
    elif num.source == "word":
        lang_ok = requested in num.lang.split("+")
    else:                       # a digit: digit-only answers fit any language
        lang_ok = lang == "digits" or is_fidelity(lang, requested)
    return {"label": CORRECT if correct else WRONG,
            "strict": int(correct and lang_ok), "lenient": int(correct),
            "content_lang_ok": bool(num is not None and lang_ok),
            "found_value": None if num is None else num.value,
            "_no_content": num is None}


def score_classes(prep, gold, vocab, requested):
    """Design 8.4, room, material and colour.
    Room: exactly the gold room and no other.
    Material and colour: at least one class found, all of them in the gold."""
    matches = vocab.indexes[gold.kind].find_all(prep.tokens)
    found = {s.class_id for m in matches for s in m.senses}
    if gold.kind == "room":
        correct = found == set(gold.classes)
    else:
        correct = bool(found) and found <= set(gold.classes)
    langs = {s.lang for m in matches for s in m.senses
             if s.class_id in gold.classes}
    lang_ok = requested in langs
    return {"label": CORRECT if correct else WRONG,
            "strict": int(correct and lang_ok), "lenient": int(correct),
            "content_lang_ok": bool(found) and lang_ok,
            "found_value": sorted(found), "_no_content": not found}


def _label_tokens(tokens, text, label_lang):
    """Indices of the tokens that are option labels, in order (rule F)."""
    hits = []
    n = len(tokens)
    for i, t in enumerate(tokens):
        if label_lang == "bn":
            if t.kind == "bn" and t.text in BN_LABELS:
                hits.append((i, BN_LABELS.index(t.text)))
            continue
        if t.kind != "lat" or t.text not in EN_LABELS:
            continue
        if t.text != "a":
            hits.append((i, EN_LABELS.index(t.text)))
            continue
        # "a" is also the English article. It is a label only when it stands
        # alone, is marked like a label, or ends the answer.
        after = text[t.end:t.end + 1]
        before = text[t.start - 1:t.start] if t.start > 0 else ""
        prev_word = tokens[i - 1].text if i > 0 else ""
        if n == 1 or after in (")", "]", ".", ":") or before == "(" \
                or i == n - 1 or prev_word in _LABEL_CUE_WORDS:
            hits.append((i, 0))
    return hits


def score_forced_choice(answer, options, correct_index, label_lang,
                        settings=DEFAULT_SETTINGS):
    """Design 8.5. Take the first option label. If there is none, match the
    option text. Otherwise the answer is invalid and scored wrong.

    options        the four option texts in the order shown to the model
    correct_index  position of the gold option, 0 to 3
    label_lang     bn for ক খ গ ঘ, en for A B C D
    """
    if len(options) != 4:
        raise ValueError("Forced choice needs exactly 4 options (Design 4.7)")
    if correct_index not in (0, 1, 2, 3):
        raise ValueError("correct_index must be 0, 1, 2 or 3")
    prep = answer if isinstance(answer, Prepared) else prepare_answer(answer)
    tokens = prep.norm_tokens            # lead-ins are not stripped here
    choice, method = None, None

    hits = _label_tokens(tokens, prep.norm, label_lang)
    if hits:
        choice, method = hits[0][1], "label"
    else:
        index = TermIndex(settings)
        for k, opt in enumerate(options):
            index.add(opt, k)
        found = {k for m in index.find_all(tokens) for k in m.senses}
        if len(found) == 1:
            choice, method = found.pop(), "text"

    other = "en" if label_lang == "bn" else "bn"
    cross = choice is None and (
        bool(_label_tokens(tokens, prep.norm, other))
        or (len(tokens) == 1 and tokens[0].kind == "dig"))
    valid = choice is not None
    correct = valid and choice == correct_index
    return {"label": CORRECT if correct else (WRONG if valid else INVALID),
            "strict": int(correct), "lenient": int(correct),
            "fc_valid": valid, "fc_choice": choice, "fc_method": method,
            "fc_correct_index": correct_index, "fc_cross_label": cross}


_RECORD_KEYS = (
    "scorer_version", "qa_id", "qa_type", "condition_id", "task", "kind",
    "requested_lang", "answer_raw", "answer_norm", "n_words", "is_empty",
    "lang", "bn_letters", "lat_letters", "other_letters", "mixed_script",
    "fidelity", "scoreable", "gold_status", "gold_norm", "gold_hint",
    "label", "strict", "lenient", "content_lang_ok", "label_lang", "refusal",
    "gold_name_id", "cluster_id", "in_cluster", "en_route", "en_unique",
    "origin3", "matched", "named_ids", "copy_term", "copy_sim",
    "gold_value", "found_value",
    "fc_valid", "fc_choice", "fc_method", "fc_correct_index",
    "fc_cross_label",
    "content_found",                     # added in 0.2.0-stage2 (rule J)
    "refusal_phrase")                    # added in 0.2.1-stage2 (rule M)


def score_item(item, condition_id, answer, lexicon, vocab, options=None,
               correct_index=None, excluded=None):
    """Score one model answer to one probe item under one condition.

    Returns a flat dict with the same keys every time (None where a field
    does not apply), so the output loads straight into a table.
    strict and lenient are 1 or 0, or None when the gold cannot be resolved
    (scoreable is False and gold_status says why).

    excluded   None, or the reason code of an item on
               probes/excluded_items.csv (load_excluded). Such an item is
               never scored (D110): gold_status is "excluded", scoreable is
               False, gold_hint holds the reason code, and every score field
               stays empty. The answer is still described (language, words,
               empty), and forced choice needs no option set for it.
    """
    cond = condition_spec(condition_id)
    prep = prepare_answer(answer)
    rec = dict.fromkeys(_RECORD_KEYS)
    lang = prep.lang["lang"]
    rec.update(
        scorer_version=SCORER_VERSION, qa_id=item["qa_id"],
        qa_type=item["qa_type"], condition_id=condition_id, task=cond.task,
        requested_lang=cond.answer_lang or None, answer_raw=prep.raw,
        answer_norm=prep.norm, n_words=len(prep.norm.split()),
        is_empty=not prep.norm_tokens, lang=lang,
        bn_letters=prep.lang["bn_letters"],
        lat_letters=prep.lang["lat_letters"],
        other_letters=prep.lang["other_letters"],
        mixed_script=prep.lang["bn_letters"] > 0
        and prep.lang["lat_letters"] > 0,
        gold_norm=normalise(item["answer_bn"]))
    if excluded:
        rec.update(scoreable=False, gold_status=EXCLUDED_STATUS,
                   gold_hint=excluded if isinstance(excluded, str) else None)
        return rec
    phrase = _has_refusal_phrase(prep.norm_tokens)
    rec["refusal_phrase"] = phrase

    if cond.task == "fc":
        if options is None or correct_index is None:
            raise ValueError("%s is forced choice: pass options and"
                             " correct_index" % condition_id)
        res = score_forced_choice(prep, options, correct_index,
                                  cond.question_lang, lexicon.settings)
        rec.update(res)
        rec.update(kind="forced_choice", scoreable=True, gold_status="ok",
                   refusal=rec["is_empty"] or (not res["fc_valid"] and phrase))
        return rec

    rec["fidelity"] = is_fidelity(lang, cond.answer_lang)
    gold = resolve_gold(item, lexicon, vocab)
    rec.update(kind=gold.kind or None, gold_status=gold.status,
               gold_hint=gold.hint or None, scoreable=gold.status == "ok")
    if gold.status != "ok":
        rec["refusal"] = rec["is_empty"] or phrase
        return rec

    if gold.kind == "naming":
        obj = lexicon.objects[gold.name_id]
        res = score_naming(prep, gold, lexicon, cond.answer_lang)
        rec.update(gold_name_id=obj.name_id, cluster_id=obj.cluster_id or None,
                   in_cluster=bool(obj.cluster_id),
                   en_route=obj.row.get("en_route") or None,
                   en_unique=obj.row.get("en_unique") or None,
                   origin3=obj.row.get("origin3") or None)
    elif gold.kind == "count":
        res = score_count(prep, gold, vocab, cond.answer_lang)
        rec["gold_value"] = gold.number
    else:
        res = score_classes(prep, gold, vocab, cond.answer_lang)
        rec["gold_value"] = sorted(gold.classes)
    no_content = res.pop("_no_content")
    rec.update(res)
    rec["refusal"] = rec["is_empty"] or (no_content and phrase)
    rec["content_found"] = not no_content
    return rec


# --------------------------------------------------------------------------
# 9. A small summary for scripts (not the statistics of Design section 9)
# --------------------------------------------------------------------------

def summarise(records, by=("condition_id",)):
    """Counts and simple rates per group. For eyeballing a run only.

    Excluded records (gold_status "excluded") are counted in `excluded` and
    are in no other number: not in n, not in any rate (D110, D163).
    invalid: share of scoreable forced-choice records labelled invalid.
    no_content: share of scoreable open records with content_found False."""
    groups = {}
    for r in records:
        groups.setdefault(tuple(r.get(k) for k in by), []).append(r)
    rows = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        every = groups[key]
        rs = [r for r in every if r.get("gold_status") != EXCLUDED_STATUS]
        sc = [r for r in rs if r["scoreable"]]
        fid = [r for r in rs if r["fidelity"] is not None]
        fc = [r for r in sc if r.get("task") == "fc"]
        op = [r for r in sc if r.get("task") == "open"]

        def rate(xs, f):
            return round(100.0 * sum(1 for x in xs if f(x)) / len(xs), 1) \
                if xs else None

        row = dict(zip(by, key))
        row.update(
            n=len(rs), excluded=len(every) - len(rs), scoreable=len(sc),
            strict=rate(sc, lambda r: r["strict"] == 1),
            lenient=rate(sc, lambda r: r["lenient"] == 1),
            fidelity=rate(fid, lambda r: r["fidelity"]),
            refusal=rate(rs, lambda r: r["refusal"]),
            empty=rate(rs, lambda r: r["is_empty"]),
            le8_words=rate(rs, lambda r: r["n_words"] <= 8),
            mixed_script=rate(rs, lambda r: r["mixed_script"]),
            invalid=rate(fc, lambda r: r["label"] == INVALID),
            no_content=rate(op, lambda r: r.get("content_found") is False))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# 10. Checks on the frozen files (rule K). Used by asset9_scorer.py audit and
#     by tests/test_real_files.py. They read files; they never change them.
# --------------------------------------------------------------------------

_KIND_5C = {"naming": "name", "count": "count", "room": "room",
            "material": "material", "colour": "colour"}


def check_5c_fields(item, gold, vocab):
    """Addendum A2: the scorer must derive the same closed_kind, gold_count and
    gold_room as ASSET-5C. gold is resolve_gold(item, ...).

    Returns None when the item carries no closed_kind (P-NAME has none), else
    the list of disagreements, empty when everything agrees. The messages hold
    kinds, numbers and class ids only, never the gold text."""
    want = item.get("closed_kind")
    if want is None:
        return None
    if gold.status != "ok":
        return ["gold not scoreable (%s)" % gold.status]
    bad = []
    got = _KIND_5C.get(gold.kind, gold.kind)
    if got != want:
        bad.append("kind: scorer %s, 5C %s" % (got, want))
    if want == "count" or "gold_count" in item:
        if gold.number != item.get("gold_count"):
            bad.append("count: scorer %s, 5C %s"
                       % (gold.number, item.get("gold_count")))
    if want == "room" or "gold_room" in item:
        index = vocab.indexes["room"]
        senses = index.entries.get(index.key_of(item.get("gold_room") or ""), [])
        room = {s.class_id for s in senses}
        if not room:
            bad.append("room: gold_room is not a listed room form")
        elif room != set(gold.classes):
            bad.append("room: scorer %s, 5C %s"
                       % ("|".join(sorted(gold.classes)), "|".join(sorted(room))))
    return bad


# English letter names in Bangla letters, used only to find acronym pairs
# such as এসি / ac for the D117 check. A missing spelling only leaves a pair
# out of the check; it never changes a score.
_LETTER_NAMES = {
    "a": "এ", "b": "বি", "c": "সি", "d": "ডি", "e": "ই", "f": "এফ", "g": "জি",
    "h": "এইচ", "i": "আই", "j": "জে", "k": "কে", "l": "এল", "m": "এম",
    "n": "এন", "o": "ও", "p": "পি", "q": "কিউ", "r": "আর", "s": "এস",
    "t": "টি", "u": "ইউ", "v": "ভি", "w": "ডাব্লিউ", "x": "এক্স",
    "y": "ওয়াই", "z": "জেড"}
_LETTER_BN = {normalise(v): k for k, v in _LETTER_NAMES.items()}


def spelled_letters(term):
    """The Latin letters that a Bangla term spells by English letter names
    (এসি gives "ac", টিভি gives "tv"), or None if it does not."""
    toks = tokenise(normalise(term))
    if not toks or any(t.kind != "bn" for t in toks):
        return None
    s = "".join(t.text for t in toks)
    names = sorted(_LETTER_BN, key=len, reverse=True)
    memo = {}

    def walk(i):
        if i == len(s):
            return ""
        if i not in memo:
            memo[i] = None
            for n in names:
                if s.startswith(n, i):
                    rest = walk(i + len(n))
                    if rest is not None:
                        memo[i] = _LETTER_BN[n] + rest
                        break
        return memo[i]

    return walk(0)


def parallel_pairs(lexicon=None, vocab=None):
    """D117: pairs (Bangla form, English form) of one object or one class that
    are the same word in two scripts. Found with the scorer's own measures:
    the Bangla form spells the English letters ("acronym"), or its
    romanisation reaches the English-copy threshold of Design 8.3
    ("romanised"). Returns tuples (source, owner_id, bn, en, how), where
    source is "lexicon" or a vocabulary kind. Pairs that the measures miss
    (a native word and its English name) are simply not returned."""
    def how(bn, en):
        target = _letters_only(en)
        if not target:
            return None
        if spelled_letters(bn) == target:
            return "acronym"
        if is_similar(_letters_only(romanise_bn(bn)), target):
            return "romanised"
        return None

    pairs = []
    if lexicon is not None:
        for oid in sorted(lexicon.objects):
            obj = lexicon.objects[oid]
            for bn in obj.t1_bn:
                for en in obj.t1_en:
                    h = how(bn, en)
                    if h:
                        pairs.append(("lexicon", oid, bn, en, h))
    if vocab is not None:
        for kind in VOCAB_KINDS:
            for cid in sorted(vocab.forms[kind]):
                forms = vocab.forms[kind][cid]
                for bn in forms["bn"]:
                    for en in forms["en"]:
                        h = how(bn, en)
                        if h:
                            pairs.append((kind, cid, bn, en, h))
    return pairs


def genitive_only_forms(vocab):
    """D117 with the ending rule (D104). The ending rule accepts an ending on
    an answer when the bare form is listed, never the other way round. A
    Bangla form listed only with a genitive ending (ের, য়ের) therefore leaves
    its bare form unaccepted, while the English form of the same class is
    accepted. Material and colour only: their golds are written in the
    genitive (B2), while room and number words can end in the same letters
    by nature (সতের is 17). Returns tuples (kind, class_id, listed form, bare
    form)."""
    out = []
    for kind in ("material", "colour"):
        index = vocab.indexes[kind]
        for cid in sorted(vocab.forms[kind]):
            for form in vocab.forms[kind][cid]["bn"]:
                for end in GENITIVE_ENDINGS:
                    if form.endswith(end) and len(form) - len(end) >= MIN_STEM_LEN:
                        bare = form[:-len(end)]
                        owners = {s.class_id for s in
                                  index.entries.get(index.key_of(bare), [])}
                        if cid not in owners:
                            out.append((kind, cid, form, bare))
                        break
    return out
