#!/usr/bin/env python3
"""
asset2_stage23.py
BanglaIndoorVQA, ASSET-2 stages 2, 3 and 3b: human post-editing of the English
questions, blind back-translation, and the faithfulness result
(Design 4.6, human check HT-2, DESIGN_ADDENDUM D4 and D5, decisions D130 to D138).

No API call, no GPU, no cost. The script makes sheets for people, checks the
sheets they fill, writes the final English questions file and computes the
stage 3b result. It never translates and never judges anything itself.

Commands, in the order of the work
    make-postedit    stage 2. One post-edit sheet for each of the 2 post-editors
                     (PE1, PE2), and one workbook for both (agree_first) with
                     the object names and wordings of the count questions.
                     Scope pilot (the 32 PILOT strings) first, then scope rest
                     (the other 621).
    make-crosscheck  stage 2. Needs the filled post-edit sheets. Each
                     post-editor re-checks a random 10% of the other's half
                     and, in addition, every string that 10 or more dataset
                     items use.
    merge            stage 2. Checks the filled sheets and writes the final
                     English questions file. It keeps string_id and
                     question_bn_norm (D4); the English is in question_en.
    make-backtrans   stage 3. Draws 150 strings (seeded rank, seed 42) from the
                     final file and makes one sheet of 75 for each of the 2
                     back-translators (BT1, BT2). They see the English only.
    make-compare     stage 3b. Sheets in which the post-editors compare each
                     back-translation with the Bangla original. A pair goes to
                     the post-editor who did not post-edit that string.
    result           stage 3b. Percent faithful, Wilson 95% CI, 95% threshold.
    status           what exists and what the next step is.
    dryrun           the whole chain with made-up people, in its own folder.

Two kinds of findings in filled sheets
    ERRORS  (an empty decision, Bangla letters in the English, a changed grey
            cell ...): the person corrects the filled file. Nothing is written
            until no error is left.
    REVIEW ROWS  (one Bangla wording with two English wordings, one Bangla
            object name with two English names, a fixed word that is missing
            ...): the two post-editors read each row in review_*.csv. They
            correct what is wrong. What is right as it is, the runner accepts
            with --accept-review N, where N is the number of rows left.

People (Design 10, HT-2; D125)
    2 post-editors and 2 back-translators: 4 different people. A back-translator
    never post-edited. Until gate-A is tagged, none of the four and nobody who
    runs this script is in the HT-1 pair. A back-translator never runs this
    script and never opens a file in Drive english/.

What it never does
    It never prints a question, a translation or a back-translation. The screen
    and the *_report.txt files hold ids and counts only. Everything with text
    is written to Drive english/, which is closed to the HT-1 pair until Gate A.

Where things live
    Code (this file, src/normalise.py): GitHub.
    Everything this script reads and writes: Google Drive english/ only.
        english/english_questions_final_PILOT.csv   32 rows, for the pilot
        english/english_questions_final.csv         653 rows
        english/stage2/to_send/   sheets for the post-editors (the script writes)
        english/stage2/filled/    the same files, filled (people put them here)
        english/stage3/round1/to_send/ and filled/  the same for stages 3 and 3b
        english/amendments/       late corrections (amendments_*.csv with the
                                  columns string_id, question_en_new, reason);
                                  merge applies them last

Usage (from the repo root, in Colab; needs openpyxl)
    python scripts/asset2_stage23.py dryrun --synthetic
    python scripts/asset2_stage23.py dryrun
    python scripts/asset2_stage23.py make-postedit --scope pilot --people-rules-confirmed
    python scripts/asset2_stage23.py make-crosscheck --scope pilot
    python scripts/asset2_stage23.py merge --scope pilot
    python scripts/asset2_stage23.py make-postedit --scope rest --people-rules-confirmed
    python scripts/asset2_stage23.py make-crosscheck --scope rest
    python scripts/asset2_stage23.py merge --scope all
    python scripts/asset2_stage23.py make-backtrans --people-rules-confirmed
    python scripts/asset2_stage23.py make-compare
    python scripts/asset2_stage23.py result
    python scripts/asset2_stage23.py status

Other flags
    --accept-review N     make-crosscheck, merge: N review rows were read and are right
    --overwrite-final     merge: replace a final file that exists (the old one is kept)
    --force               make-*: replace a sheet that was never given to a person
    --round 2             make-backtrans, make-compare, result: a fresh 150 after a fail
    --english-dir PATH    every command: the Drive folder english/
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# Import from src/ (one normalise(), imported everywhere)
# --------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
for _cand in (_HERE.parent.parent / "src", _HERE.parent / "src", Path.cwd() / "src"):
    if (_cand / "normalise.py").exists():
        sys.path.insert(0, str(_cand))
        break
try:
    from normalise import normalise
except ImportError as _exc:  # pragma: no cover
    sys.exit(f"STOP: cannot import src/normalise.py ({_exc}). Run this script from "
             "the repo root, with normalise.py in src/.")

# --------------------------------------------------------------------------
# Fixed settings. Changing any of these is a decision, not a tweak.
# --------------------------------------------------------------------------
SCRIPT_NAME = "asset2_stage23"
SCRIPT_VERSION = "2026-10-09"

SEED = 42
EXPECTED_STRINGS = 653             # DESIGN_ADDENDUM D1
EXPECTED_PILOT = 32                # DESIGN_ADDENDUM A1
POST_EDITORS = ("PE1", "PE2")      # role codes; the team lead knows who is who
TRANSLATORS = ("BT1", "BT2")       # never the same people as PE1 and PE2 (HT-2)
CROSSCHECK_NUM, CROSSCHECK_DEN = 1, 10   # 10% of the other's half (Design 4.6)
HIGH_USE_MIN_ITEMS = 10            # a string used by this many items is cross-checked too (D132)
SAMPLE_SIZE = 150                  # Design 4.6, stage 3 (75 for each translator)
THRESHOLD_PCT = 95                 # Design 4.6: at least 95% faithful
Z95 = 1.959963984540054            # two-sided 95%
ENGLISH_COLUMN = "question_en"     # the column the harness reads (D149)
SPELLING = "american"              # one English spelling for all questions (D133)

DEFAULT_ROOT = "/content/drive/MyDrive/BanglaIndoorVQA"

MT_HEADER_NEEDED = ("seq", "string_id", "in_pilot", "in_pclust", "qa_types", "probes",
                    "n_items", "question_bn", "mt_en", "mt_status", "qc_flags",
                    "n_surface_forms", "question_bn_norm")
FINAL_HEADER = ["seq", "string_id", "in_pilot", "in_pclust", "qa_types", "probes",
                "n_items", "question_bn", ENGLISH_COLUMN, "question_bn_norm",
                "n_surface_forms", "first_pass_en", "edit_status", "post_editor",
                "cross_check", "amended", "final_version"]

GROUP_ORDER = ("room", "property", "count", "naming", "riddle", "other")
TYPE_LABEL = {"room": "room", "property": "color or material", "count": "count",
              "naming": "name", "riddle": "riddle", "other": "other"}


def nfc(text):
    return unicodedata.normalize("NFC", text)


# Bangla words the checks look for. These are grammar words and the wordings
# fixed by the stage 1 prompt (rules 5 and 9). No dataset text is in this file.
IMAGE_WORD_BN = nfc("ছবি")
COMPOUND_THINGS_BN = nfc("জিনিসপত্র")
FIXED_WORDINGS = (     # (Bangla wording, English word, pattern for the English, not this compound)
    (nfc("দেখা যাচ্ছে"), "visible", r"\bvisible\b", None),
    (nfc("জিনিস"), "thing", r"\bthings?\b", COMPOUND_THINGS_BN),
    (nfc("বস্তু"), "object", r"\bobjects?\b", None),
)
# A wording that is not fixed, but that should get one English word in all strings.
WATCHED_WORDINGS = ((COMPOUND_THINGS_BN, "things", r"\bthings\b"),)
HOW_MANY_PREFIXES = tuple(nfc(w) for w in ("কয়", "কত", "কউ", "কটি", "কটা"))
COUNT_VERB_TOKENS = frozenset(nfc(w) for w in (
    "দেখা", "যাচ্ছে", "যাছে", "যায়", "আছে", "রয়েছে", "রাখা", "বসানো"))

# Notes from the stage 1 reading (DESIGN_ADDENDUM A5 and D5). Ids only, no dataset text.
_OWN_ANSWER = ("Known: the Bangla question holds a word that is also its answer. Translate it "
               "as it is written. The team decides about it later (stage 4).")
KNOWN_NOTES = {
    "q_5b17277c23b1": "Stage 1 reading: a detail of the Bangla may be missing in the "
                      "English. Compare word by word.",
    "q_68a96e66e28c": "Stage 1 reading: the English may hold a word that the Bangla "
                      "does not have. Compare word by word.",
    "q_9ec4ec4e1dd4": "Known: a word is missing in the Bangla. Translate the question that "
                      "was meant.",
    "q_f6f91cb847ba": _OWN_ANSWER,
    "q_f7e9d599bbb8": _OWN_ANSWER,
}

# One spelling for all questions. British form -> American form. Whole words only.
_BRITISH_TO_AMERICAN = {
    "colour": "color", "colours": "colors", "coloured": "colored", "colourful": "colorful",
    "colouring": "coloring", "grey": "gray", "aluminium": "aluminum", "centre": "center",
    "centres": "centers", "metre": "meter", "metres": "meters", "litre": "liter",
    "litres": "liters", "fibre": "fiber", "travelling": "traveling", "jewellery": "jewelry",
    "mould": "mold", "tyre": "tire", "tyres": "tires", "cosy": "cozy", "pyjamas": "pajamas",
    "programme": "program", "programmes": "programs", "favourite": "favorite",
    "neighbour": "neighbor", "neighbours": "neighbors", "odour": "odor", "vapour": "vapor",
    "flavour": "flavor", "theatre": "theater", "organise": "organize",
    "organising": "organizing", "organised": "organized", "organiser": "organizer",
    "sanitise": "sanitize", "sanitiser": "sanitizer", "moustache": "mustache",
    "plough": "plow", "humour": "humor", "labour": "labor", "honour": "honor",
}
# American forms that are also correct British words are not flagged in British mode.
_AMBIGUOUS_AMERICAN = {"meter", "meters", "tire", "tires", "program", "programs", "mold"}
if SPELLING == "american":
    WRONG_SPELLING = dict(_BRITISH_TO_AMERICAN)
else:
    WRONG_SPELLING = {a: b for b, a in _BRITISH_TO_AMERICAN.items()
                      if a not in _AMBIGUOUS_AMERICAN}
SPELLING_EXAMPLE = "color" if SPELLING == "american" else "colour"

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
_PREFIX = ""          # the dry run indents the output of the commands it calls
_CAPTURE = None       # a list while the dry run runs a command without showing it


def say(msg=""):
    for line in str(msg).split("\n"):
        text = _PREFIX + line if line else ""
        if _CAPTURE is not None:
            _CAPTURE.append(text)
        else:
            print(text, flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_hex(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rank_key(purpose, item_id):
    """Seeded rank (seed 42): items are ordered by this value. It does not
    depend on the order of any file or on Python's random module."""
    return sha256_hex(f"{SEED}|{purpose}|{item_id}")


def ceil_share(n):
    return (n * CROSSCHECK_NUM + CROSSCHECK_DEN - 1) // CROSSCHECK_DEN


def other_editor(pe):
    return POST_EDITORS[1] if pe == POST_EDITORS[0] else POST_EDITORS[0]


class Stop(Exception):
    """A condition that must stop the run. The message tells the user why."""


class SheetErrors(Stop):
    """Filled sheets hold errors that people must fix."""

    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors


class ReviewPending(Stop):
    """Review rows exist that people have not looked at yet."""

    def __init__(self, message, n):
        super().__init__(message)
        self.n = n


def write_csv(path, header, rows):
    """Write a CSV in UTF-8 with BOM (opens cleanly in Excel and Sheets)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)
    os.replace(tmp, path)


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_text(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def ids_line(ids, limit=12):
    ids = list(ids)
    return ", ".join(ids[:limit]) + (f" and {len(ids) - limit} more" if len(ids) > limit else "")


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
_EN_CHAR_FIXES = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2026": "...", "\uff1f": "?", "\u00a0": " ", "\u2009": " ", "\u200a": " ",
    "\u202f": " ", "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",
}


def clean_en(text):
    """Typed English as plain text: straight quotes, single spaces, no edges.
    Spreadsheet programs turn ' into a curly quote on their own."""
    s = "".join(_EN_CHAR_FIXES.get(ch, ch) for ch in (text or ""))
    return " ".join(s.split())


def clean_bn(text):
    """Typed Bangla: single spaces, no edges. Joiners are left alone."""
    return " ".join((text or "").replace("\u00a0", " ").replace("\ufeff", "").split())


def has_bangla(text):
    return any("\u0980" <= ch <= "\u09ff" for ch in text)


def english_problems(text):
    """Reasons why a text cannot be a final English question (empty list = fine)."""
    if not text:
        return ["is empty"]
    probs = []
    if has_bangla(text):
        probs.append("holds Bangla letters")
    elif any(ord(ch) > 126 or ord(ch) < 32 for ch in text):
        bad = sorted({ch for ch in text if ord(ch) > 126 or ord(ch) < 32})
        probs.append("holds a character that is not plain English ("
                     + " ".join(f"U+{ord(ch):04X}" for ch in bad[:4]) + ")")
    if not text.endswith("?"):
        probs.append("does not end with a question mark")
    if len(text.split()) < 2:
        probs.append("is one word")
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in WRONG_SPELLING:
            probs.append(f"has the spelling '{word}': write '{WRONG_SPELLING[word]}' "
                         f"({SPELLING} spelling in all questions)")
            break
    return probs


def bangla_problems(text):
    """Reasons why a text cannot be a Bangla back-translation."""
    if not text:
        return ["is empty"]
    n_bn = sum(1 for ch in text if "\u0980" <= ch <= "\u09ff")
    n_la = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if n_bn == 0:
        return ["holds no Bangla letters"]
    if n_la > n_bn:
        return ["is mostly in Latin letters"]
    return []


def type_group(qa_types):
    types = set(qa_types.split("|"))
    if "scene_id" in types:
        return "room"
    if "obj_property" in types:
        return "property"
    if "count" in types:
        return "count"
    if "obj_name" in types:
        return "naming"
    if "object" in types:
        return "riddle"
    return "other"


def count_parts_bn(norm):
    """(object term, frame) of a Bangla count question, or None.
    The term is the words between the how-many word and the verb."""
    toks = nfc(norm).split()
    i = next((k for k, w in enumerate(toks) if w.startswith(HOW_MANY_PREFIXES)), None)
    if i is None:
        return None
    j = next((k for k in range(i + 1, len(toks)) if toks[k] in COUNT_VERB_TOKENS), len(toks))
    term = " ".join(toks[i + 1:j])
    if not term:
        return None
    return term, " ".join(toks[:i] + ["<N>", "<X>"] + toks[j:])


_EN_COUNT = re.compile(
    r"^(.*?\bhow many )(.+?)( (?:are|is|were|was|can|do|does|did|have|has)\b.*)$", re.I)


def count_parts_en(en):
    """(object term, frame) of an English count question, or None."""
    m = _EN_COUNT.match((en or "").strip())
    if not m:
        return None
    return m.group(2).lower(), (m.group(1) + "<X>" + m.group(3)).lower()


def wilson(k, n, z=Z95):
    """Wilson score interval for k successes in n trials."""
    if n <= 0 or k < 0 or k > n:
        raise ValueError("wilson: need 0 <= k <= n and n > 0")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def passes_threshold(k, n):
    """At least 95% faithful, in whole-number arithmetic."""
    return k * 100 >= THRESHOLD_PCT * n


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
class Ctx:
    """Where the files are, and the sizes this run expects."""

    def __init__(self, english_dir, expect_counts=True, sample_size=SAMPLE_SIZE):
        self.root = Path(english_dir)
        self.expect_counts = expect_counts
        self.sample_size = sample_size
        r = self.root
        self.mt_all = r / "english_questions_mt.csv"
        self.mt_pilot = r / "english_questions_mt_PILOT.csv"
        self.registry = r / "asset2_strings.csv"
        self.final_all = r / "english_questions_final.csv"
        self.final_pilot = r / "english_questions_final_PILOT.csv"
        self.amend_dir = r / "amendments"
        self.s2 = r / "stage2"
        self.s2_send = self.s2 / "to_send"
        self.s2_filled = self.s2 / "filled"
        self.assignment = self.s2 / "assignment.csv"
        self.history = self.s2 / "final_history"
        self.manifest = r / "stage23_manifest.json"

    def s3(self, rnd):
        return self.root / "stage3" / f"round{rnd}"

    def record(self, command, details):
        data = {"asset": "ASSET-2 stages 2, 3 and 3b", "runs": []}
        if self.manifest.exists():
            with open(self.manifest, "r", encoding="utf-8") as f:
                data = json.load(f)
        data["runs"].append(dict({"command": command, "utc": utc_now(),
                                  "script": SCRIPT_NAME, "script_version": SCRIPT_VERSION,
                                  "seed": SEED}, **details))
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(self.manifest) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.manifest)


SCOPE_LABEL = {"pilot": "PILOT", "rest": "REST", "all": "ALL"}


def in_scope(s, scope):
    if scope == "pilot":
        return s["in_pilot"]
    if scope == "rest":
        return not s["in_pilot"]
    return True


# --------------------------------------------------------------------------
# The stage 1 files
# --------------------------------------------------------------------------
def load_strings(ctx):
    """Read english_questions_mt.csv, check it against DESIGN_ADDENDUM D4, and
    return the strings in running order."""
    for p in (ctx.mt_all, ctx.mt_pilot, ctx.registry):
        if not p.exists():
            raise Stop(f"Missing stage 1 file: {p}")
    rows = read_csv(ctx.mt_all)
    if not rows:
        raise Stop(f"{ctx.mt_all} has no rows.")
    missing = [c for c in MT_HEADER_NEEDED if c not in rows[0]]
    if missing:
        raise Stop(f"{ctx.mt_all} lacks the columns {missing}. Is this the stage 1 output?")
    out, seen = [], set()
    for r in rows:
        sid, norm = r["string_id"], r["question_bn_norm"]
        if sid in seen:
            raise Stop(f"{ctx.mt_all}: string_id {sid} appears twice.")
        seen.add(sid)
        if sid != "q_" + sha256_hex(norm)[:12]:
            raise Stop(f"{ctx.mt_all}: {sid} is not the id of its question_bn_norm (D4).")
        if normalise(r["question_bn"]) != norm:
            raise Stop(f"{ctx.mt_all}: {sid}: normalise(question_bn) differs from "
                       "question_bn_norm. The file was changed after stage 1.")
        if r["mt_status"] != "ok" or not r["mt_en"].strip():
            raise Stop(f"{ctx.mt_all}: {sid} has no first-pass translation. Finish stage 1.")
        s = {"seq": int(r["seq"]), "string_id": sid, "in_pilot": r["in_pilot"] == "yes",
             "in_pclust": r["in_pclust"], "qa_types": r["qa_types"], "probes": r["probes"],
             "n_items": int(r["n_items"]), "question_bn": r["question_bn"],
             "mt_en": clean_en(r["mt_en"]), "qc_flags": r["qc_flags"],
             "n_surface_forms": r["n_surface_forms"], "norm": norm,
             "group": type_group(r["qa_types"])}
        s["count_bn"] = count_parts_bn(norm) if s["group"] == "count" else None
        out.append(s)
    out.sort(key=lambda s: s["seq"])
    n_pilot = sum(1 for s in out if s["in_pilot"])
    if ctx.expect_counts and (len(out) != EXPECTED_STRINGS or n_pilot != EXPECTED_PILOT):
        raise Stop(f"{ctx.mt_all} has {len(out)} strings, {n_pilot} in PILOT. Expected "
                   f"{EXPECTED_STRINGS} and {EXPECTED_PILOT} (DESIGN_ADDENDUM D1). Use the "
                   "stage 1 files of 5 Oct 2026 from Drive english/.")
    reg = {r["string_id"]: r for r in read_csv(ctx.registry)}
    if set(reg) != seen or any(reg[s["string_id"]]["question_bn"] != s["question_bn"] for s in out):
        raise Stop(f"{ctx.registry} and {ctx.mt_all} do not hold the same strings.")
    pil = {r["string_id"]: clean_en(r["mt_en"]) for r in read_csv(ctx.mt_pilot)}
    if pil != {s["string_id"]: s["mt_en"] for s in out if s["in_pilot"]}:
        raise Stop(f"{ctx.mt_pilot} is not the PILOT part of {ctx.mt_all}.")
    return out


# --------------------------------------------------------------------------
# The split into two halves, and the 10% cross-check
# --------------------------------------------------------------------------
def build_units(rows):
    """Strings that must stay with one post-editor: the same first-pass English
    (spelling variants of one question) or the same Bangla object name in a
    count question. Returns the units, each a list of strings in running order."""
    parent = {s["string_id"]: s["string_id"] for s in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    groups = defaultdict(list)
    for s in rows:
        groups[("en", s["mt_en"].lower())].append(s["string_id"])
        if s["count_bn"]:
            groups[("term", s["count_bn"][0])].append(s["string_id"])
    for ids in groups.values():
        for sid in ids[1:]:
            a, b = find(ids[0]), find(sid)
            if a != b:
                parent[b] = a
    units = defaultdict(list)
    for s in sorted(rows, key=lambda s: s["seq"]):
        units[find(s["string_id"])].append(s)
    return list(units.values())


def split_halves(rows, scope):
    """Give every string of the scope to PE1 or PE2. Units stay together. Each
    question type is shared evenly, and so are the strings that many items
    use. The order comes from the seeded rank."""
    by_group = defaultdict(list)
    for unit in build_units(rows):
        high = any(s["n_items"] >= HIGH_USE_MIN_ITEMS for s in unit)
        by_group[(unit[0]["group"], high)].append(unit)
    assign, unit_of, total = {}, {}, Counter()
    a, b = POST_EDITORS
    for stratum in [(g, high) for g in GROUP_ORDER for high in (True, False)]:
        units = sorted(by_group.get(stratum, []),
                       key=lambda u: (-len(u), rank_key(f"split:{scope}", u[0]["string_id"])))
        here = Counter()
        for unit in units:
            ka, kb = (here[a], total[a]), (here[b], total[b])
            if ka < kb:
                pe = a
            elif kb < ka:
                pe = b
            else:
                pe = a if int(rank_key(f"tie:{scope}", unit[0]["string_id"]), 16) % 2 == 0 else b
            for s in unit:
                assign[s["string_id"]] = pe
                unit_of[s["string_id"]] = unit[0]["string_id"]
            here[pe] += len(unit)
            total[pe] += len(unit)
    return assign, unit_of


def crosscheck_selection(half, scope):
    """What the other post-editor re-checks in one half: a random 10% (Design
    4.6) and, in addition, every string that many dataset items use.
    half: the strings of one post-editor. Returns {string_id: basis}."""
    ranked = sorted((s["string_id"] for s in half),
                    key=lambda sid: rank_key(f"crosscheck:{scope}", sid))
    chosen = {sid: "random" for sid in ranked[:ceil_share(len(ranked))]}
    for s in half:
        if s["n_items"] >= HIGH_USE_MIN_ITEMS and s["string_id"] not in chosen:
            chosen[s["string_id"]] = "high_use"
    return chosen


ASSIGN_HEADER = ["string_id", "scope", "post_editor", "unit", "cross_checker", "cross_basis"]


def load_assignment(ctx):
    if not ctx.assignment.exists():
        return {}
    return {r["string_id"]: r for r in read_csv(ctx.assignment)}


def save_assignment(ctx, records):
    rows = sorted(records.values(), key=lambda r: (r["scope"], r["post_editor"], r["string_id"]))
    write_csv(ctx.assignment, ASSIGN_HEADER, [[r[c] for c in ASSIGN_HEADER] for r in rows])


def scope_assignment(ctx, strings, scope, must_exist=True):
    """The stored split of one scope. Once the sheets are made, the stored
    split is the truth: it is read, never computed again."""
    part = [s for s in strings if in_scope(s, scope)]
    have = {sid: r for sid, r in load_assignment(ctx).items() if r["scope"] == scope}
    if not have and must_exist:
        raise Stop(f"No split is stored for scope {scope}. Run make-postedit --scope {scope} first.")
    if have and set(have) != {s["string_id"] for s in part}:
        raise Stop(f"The split of scope {scope} stored in {ctx.assignment} holds other strings "
                   "than the stage 1 file holds now. The stage 1 files changed after the sheets "
                   "were made. Do not go on: ask in the ASSET-2 chat.")
    return part, have


# --------------------------------------------------------------------------
# Workbooks (xlsx). One work tab and one READ ME tab per file.
# --------------------------------------------------------------------------
def _openpyxl():
    try:
        import openpyxl  # in Colab by default
        return openpyxl
    except ImportError:
        raise Stop("The package openpyxl is missing. In the notebook run: "
                   "pip install -q openpyxl") from None


def _sheet_text(value):
    """Text that is safe in a cell: no control characters."""
    return "".join(ch for ch in str(value) if ch >= " " or ch == "\n")


def write_workbook(path, tabs, readme_lines):
    """tabs: list of dict(name, header, rows, widths, inputs, choices). The first
    tab is the work tab, so that a CSV export from Google Sheets gives it. The
    last tab is READ ME; a line that starts with '# ' is a heading."""
    openpyxl = _openpyxl()
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    fill_head = PatternFill("solid", fgColor="1F3864")
    fill_head_in = PatternFill("solid", fgColor="7F6000")
    fill_ro = PatternFill("solid", fgColor="EDEDED")
    fill_in = PatternFill("solid", fgColor="FFF8DC")
    for tab in tabs:
        ws = wb.create_sheet(tab["name"])
        header, rows = tab["header"], tab["rows"]
        inputs = set(tab.get("inputs", ()))
        for j, name in enumerate(header, 1):
            c = ws.cell(row=1, column=j, value=name)
            c.font = Font(bold=True, color="FFFFFF", size=11)
            c.fill = fill_head_in if name in inputs else fill_head
            c.alignment = Alignment(wrap_text=True, vertical="center")
            c.border = border
            ws.column_dimensions[get_column_letter(j)].width = tab["widths"].get(name, 16)
        for i, row in enumerate(rows, 2):
            for j, (name, value) in enumerate(zip(header, row), 1):
                if isinstance(value, int):
                    c = ws.cell(row=i, column=j, value=value)
                else:
                    text = _sheet_text(value)
                    c = ws.cell(row=i, column=j, value=text)
                    if text.startswith("="):
                        c.data_type = "s"          # text, never a formula
                c.font = Font(size=12)
                c.alignment = Alignment(wrap_text=True, vertical="top")
                c.border = border
                c.fill = fill_in if name in inputs else fill_ro
                if name in inputs:
                    c.number_format = "@"          # typed text stays text
        last = max(len(rows) + 1, 2)
        for name, values in tab.get("choices", {}).items():
            dv = DataValidation(type="list", formula1='"' + ",".join(values) + '"',
                                allow_blank=True)
            dv.error = "Choose one of: " + ", ".join(values)
            dv.errorTitle = "Not a valid choice"
            dv.showErrorMessage = True
            ws.add_data_validation(dv)
            col = get_column_letter(header.index(name) + 1)
            dv.add(f"{col}2:{col}{last}")
        ws.freeze_panes = "A2"
        ws.row_dimensions[1].height = 30
        ws.page_setup.orientation = "landscape"       # if somebody prints it
        ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
        ws.sheet_properties.pageSetUpPr.fitToPage = True

    info = wb.create_sheet("READ ME")
    info.column_dimensions["A"].width = 118
    for i, line in enumerate(readme_lines, 1):
        heading = line.startswith("# ")
        c = info.cell(row=i, column=1, value=_sheet_text(line[2:] if heading else line))
        c.font = Font(bold=heading, size=14 if i == 1 else 12)
        c.alignment = Alignment(wrap_text=True, vertical="top")
    info.page_setup.fitToWidth, info.page_setup.fitToHeight = 1, 0
    info.sheet_properties.pageSetUpPr.fitToPage = True
    wb.active = 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.xlsx")
    wb.save(tmp)
    os.replace(tmp, path)


def _cell_text(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def read_table(path, tab=None, need=()):
    """Read a filled sheet (.xlsx or .csv). Returns a list of dicts keyed by the
    lower-case column names. Empty rows are dropped; row order does not matter."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            raw = [list(r) for r in csv.reader(f)]
    else:
        openpyxl = _openpyxl()
        try:
            wb = openpyxl.load_workbook(path, data_only=True)
        except Exception as exc:
            raise Stop(f"Cannot open {path} as an Excel file ({type(exc).__name__}). Save it "
                       "again as .xlsx (in Google Sheets: File, Download, Microsoft Excel)."
                       ) from None
        ws = wb[tab] if tab and tab in wb.sheetnames else None
        if ws is None:
            for cand in wb.worksheets:
                first = next(cand.iter_rows(min_row=1, max_row=1, values_only=True), ())
                if all(n in [_cell_text(c).strip().lower() for c in first] for n in need):
                    ws = cand
                    break
        if ws is None:
            raise Stop(f"{path}: no tab holds the columns {list(need)}. Use the file that "
                       "the script made; do not rename its columns.")
        raw = [list(r) for r in ws.iter_rows(values_only=True)]
    if not raw:
        raise Stop(f"{path} is empty.")
    header = [_cell_text(c).strip().lower() for c in raw[0]]
    missing = [n for n in need if n not in header]
    if missing:
        raise Stop(f"{path}: the columns {missing} are missing. Use the file that the "
                   "script made; do not rename its columns.")
    out = []
    for line in raw[1:]:
        vals = [_cell_text(c) for c in line]
        if not any(v.strip() for v in vals):
            continue
        out.append({h: (vals[i] if i < len(vals) else "") for i, h in enumerate(header) if h})
    return out


def find_filled(folder, stem):
    """The filled copy of one sheet. The name must start with the name of the
    file that was sent."""
    folder = Path(folder)
    cands = []
    if folder.is_dir():
        for p in sorted(folder.iterdir()):
            if p.name.startswith(("~$", ".")) or not p.stem.lower().startswith(stem.lower()):
                continue
            if p.stem[len(stem):][:1].isalnum():
                continue
            if p.suffix.lower() == ".gsheet":
                raise Stop(f"{p} is a Google Sheets file, which this script cannot read. Open "
                           "it in Google Sheets, choose File, Download, Microsoft Excel (.xlsx), "
                           f"and put the .xlsx into {folder}.")
            if p.suffix.lower() in (".xlsx", ".csv"):
                cands.append(p)
    if not cands:
        raise Stop(f"The filled sheet {stem}.xlsx is not in {folder}. Put the filled file "
                   "there under the same name. (If the person filled the copy in to_send/, "
                   "copy that file into filled/.)")
    if len(cands) > 1:
        raise Stop(f"More than one file in {folder} starts with {stem}: "
                   + ", ".join(p.name for p in cands) + ". Keep one.")
    return cands[0]


def send_sheet(path, tab, readme_lines, force):
    """Write a sheet for a person. A sheet that exists is never replaced by a
    different one unless --force is given: the person may already have it."""
    path = Path(path)
    if path.exists():
        old = read_table(path, tab["name"], tab["header"])
        new = [{h: _cell_text(v) for h, v in zip(tab["header"], row)} for row in tab["rows"]]
        old = [{h: r.get(h, "") for h in tab["header"]} for r in old]
        if old == new:
            return "unchanged"
        if not force:
            raise Stop(f"{path} exists and differs from the sheet the script would write "
                       "now. If a person already has the old file, keep it: do not make a "
                       "second version of a sheet that is being filled. If the old file "
                       "was never sent, run again with --force.")
    write_workbook(path, [tab], readme_lines)
    return "written"


# --------------------------------------------------------------------------
# The four sheets and their READ ME tabs
# --------------------------------------------------------------------------
RULES_SHORT = [
    "# The rules in one place (the guide for post-editors explains each one with examples)",
    "A. Same meaning, same detail and same difficulty as the Bangla. Add nothing, remove "
    "nothing, explain nothing.",
    "B. Never add the name of an object that the Bangla only describes. Keep general words "
    "general.",
    "C. Do not make the question clearer than the Bangla is.",
    "D. If the Bangla names an object, use the ordinary everyday English word. If there is "
    "none, write the Bangla name in English letters and do not explain it. One Bangla name "
    "gets one English name in all rows.",
    "E. Fixed words: দেখা যাচ্ছে = visible. জিনিস = thing. বস্তু = object. ছবি = image, when "
    "it means the image the question is asked about. A fixed word may change only if both "
    "post-editors change it in every row.",
    f"F. Plain household English. One question. It ends with a question mark. "
    f"{SPELLING.capitalize()} spelling ({SPELLING_EXAMPLE}).",
    "G. If the Bangla has a spelling mistake or a missing word, translate the question that "
    "was meant.",
    "H. The same Bangla wording gets the same English wording in every row.",
]
PEOPLE_RULES_PE = [
    "# Do not",
    "- change the grey columns, or add or delete rows;",
    "- use a translation tool or an AI tool. Never paste a question into one: the questions "
    "are test items. A dictionary for a single word is fine;",
    "- show this file to the two people who write the lexicon (HT-1), or to the two "
    "back-translators.",
]

README_POSTEDIT = [
    "# POST-EDIT SHEET (ASSET-2 stage 2)",
    "",
    "# What this is",
    "Each row is one Bangla question and its English translation. A machine wrote the "
    "English. You read every row and make the English right.",
    "",
    "# Before you start",
    "Go through the file agree_first with the other post-editor. It shows the object names "
    "and wordings that many rows share, so that both of you write them in one way.",
    "",
    "# What to do, row by row, in the tab postedit",
    "1. Read the Bangla in question_bn.",
    "2. Read the English in first_pass_en.",
    "3. If the column attention holds a note, read it. The column n_items shows how many "
    "dataset items use this question: a large number means that a mistake costs more.",
    "4. In decision choose keep (the English is right as it is) or edit (something must "
    "change).",
    "5. If you chose edit, write the whole corrected English question in edited_en. Write "
    "the full question, not only the changed word.",
    "6. Use note for anything the team lead should know. Never write a note inside the "
    "English.",
    "",
] + RULES_SHORT + [""] + PEOPLE_RULES_PE + [
    "",
    "# When you finish",
    "Every row has keep or edit. Save the file as .xlsx under the same name and give it "
    "back to the team lead.",
]

README_CROSSCHECK = [
    "# CROSS-CHECK SHEET (ASSET-2 stage 2)",
    "",
    "# What this is",
    "These rows were post-edited by the other post-editor. They are a random 10% of that "
    f"person's half and, in addition, every question that {HIGH_USE_MIN_ITEMS} or more dataset "
    "items use (column n_items). You check them with the same rules.",
    "",
    "# What to do, row by row, in the tab crosscheck",
    "1. Read the Bangla in question_bn, then the English in english_to_check.",
    "2. In verdict choose ok (you would not change anything) or change.",
    "3. If you chose change, write the whole English question as you would write it in "
    "proposed_en, and say why in reason.",
    "4. When you have finished all rows, sit with the other post-editor. For every row "
    "marked change, agree on one English question and write it in agreed_en. If you both "
    "decide that the first text was right, write that first text in agreed_en.",
    "",
] + RULES_SHORT + [""] + PEOPLE_RULES_PE + [
    "",
    "# When you finish",
    "Every row has ok or change, and every change row has agreed_en. Save the file as "
    ".xlsx under the same name and give it back to the team lead.",
]

README_BACKTRANS = [
    "# BACK-TRANSLATION SHEET (ASSET-2 stage 3)",
    "",
    "# What this is",
    "Each row is one English question about a photo taken inside a home in Bangladesh. You "
    "write the same question in Bangla. You do not see the photo and you do not need it.",
    "",
    "# What to do, row by row, in the tab backtranslation",
    "1. Read the English in question_en.",
    "2. Write the same question in Bangla in back_translation_bn. Do not answer it.",
    "3. Use note only if something in the English is unclear to you. Translate it anyway.",
    "",
    "# Rules (the guide for back-translators explains each one with examples)",
    "A. Translate what the English says: the same meaning, the same detail. Add nothing, "
    "remove nothing, explain nothing.",
    "B. If the English describes a thing without naming it, describe it in the same way in "
    "Bangla. Do not write the name of the thing.",
    "C. If the English names a thing, write the Bangla name you would use at home.",
    "D. Write natural everyday Bangla, in Bangla letters, as one question.",
    "E. If the English sounds odd, do not repair it. Translate what it says.",
    "F. Translate each row on its own. The same sentence can appear twice.",
    "",
    "# Do not",
    "- use a translation tool or an AI tool (Google Translate, ChatGPT, Gemini or any "
    "other). Never paste a question into one: the questions are test items. A dictionary "
    "for a single word is fine; say so in note;",
    "- open any dataset file, question list or English question file of the project;",
    "- ask the post-editors, the other back-translator or anyone else about a row;",
    "- change the column question_en, or add or delete rows.",
    "",
    "# When you finish",
    "Every row has a Bangla question. Save the file as .xlsx under the same name and give "
    "it back to the team lead.",
]

README_COMPARE = [
    "# COMPARISON SHEET (ASSET-2 stage 3b)",
    "",
    "# What this is",
    "Each row is a pair: the Bangla original of a question, and a Bangla back-translation "
    "that another person wrote from the English question alone. You decide whether the two "
    "ask the same question.",
    "",
    "# What to do, row by row, in the tab compare",
    "1. Read original_bn and back_translation_bn. Do not look at question_en yet.",
    "2. In verdict choose faithful or not faithful.",
    "   faithful: the two ask for the same thing, with the same detail, and one is not "
    "easier to answer than the other. Different words, word order or spelling do not "
    "matter. If the original has a spelling mistake or a missing word, judge against the "
    "question that was meant.",
    "   not faithful: something is added, missing or changed; or one names an object that "
    "the other only describes; or one names a different object; or one is clearer or "
    "easier than the other.",
    "3. Only if you chose not faithful: read question_en (and translator_note, if the "
    "back-translator left a note), write in what_differs what is different, and choose the "
    "cause.",
    "   English question: the English says something else than the original, or is "
    "unclear.",
    "   back-translation: the English is right and the back-translator made a mistake.",
    "   no closer English word: English has no word that keeps the difference.",
    "   not sure: you cannot tell.",
    "4. Later, with the other post-editor: for each row whose cause is English question, "
    "agree on a corrected English question and write it in fix_en.",
    "",
    "# Do not",
    "- change the grey columns, or add or delete rows;",
    "- ask the back-translators what they meant;",
    "- mark a pair faithful because the English is right. You judge the two Bangla "
    "sentences.",
    "",
    "# When you finish",
    "Every row has a verdict, and every not faithful row has a cause. Save the file as "
    ".xlsx under the same name and give it back to the team lead.",
]

POSTEDIT_COLS = ["row", "type", "n_items", "attention", "question_bn", "first_pass_en",
                 "decision", "edited_en", "note", "string_id"]
CROSS_COLS = ["row", "type", "n_items", "question_bn", "english_to_check", "verdict",
              "proposed_en", "reason", "agreed_en", "string_id"]
BT_COLS = ["item", "question_en", "back_translation_bn", "note"]
COMPARE_COLS = ["pair", "type", "original_bn", "back_translation_bn", "same_text", "verdict",
                "cause", "what_differs", "fix_en", "question_en", "translator_note", "string_id"]
WIDTHS = {"row": 5, "type": 10, "n_items": 9, "attention": 28, "question_bn": 42, "first_pass_en": 42,
          "decision": 10, "edited_en": 42, "note": 22, "string_id": 16,
          "english_to_check": 42, "verdict": 12, "proposed_en": 42, "reason": 22,
          "agreed_en": 42, "item": 13, "question_en": 58, "back_translation_bn": 58,
          "pair": 9, "original_bn": 44, "same_text": 10, "cause": 22, "what_differs": 28,
          "fix_en": 44, "translator_note": 24, "bn_name": 28, "bn_wording": 44, "first_pass_english": 44,
          "n_rows": 8, "post_editors": 12, "flag": 30, "agreed_english": 44,
          "string_ids": 30}
DECISIONS = ["keep", "edit"]
CHECK_VERDICTS = ["ok", "change"]
FAITH_VERDICTS = ["faithful", "not faithful"]
CAUSES = ["English question", "back-translation", "no closer English word", "not sure"]
CAUSE_LABEL = {"english": "English question", "back_translation": "back-translation",
               "language": "no closer English word", "unsure": "not sure"}
CAUSE_KEY = {"english question": "english", "back translation": "back_translation",
             "no closer english word": "language", "not sure": "unsure"}


def _key(text):
    return " ".join(text.strip().lower().replace("_", " ").replace("-", " ").split())


# --------------------------------------------------------------------------
# Reading the filled stage 2 sheets
# --------------------------------------------------------------------------
def read_postedit(ctx, scope, part, assign):
    """Read both filled post-edit sheets of a scope. Returns (results, errors,
    info). results[string_id] holds the English after the first pass."""
    by_id = {s["string_id"]: s for s in part}
    results, errors, info = {}, [], Counter()
    for pe in POST_EDITORS:
        stem = f"postedit_{SCOPE_LABEL[scope]}_{pe}"
        path = find_filled(ctx.s2_filled, stem)
        rows = read_table(path, "postedit", POSTEDIT_COLS)
        expected = {sid for sid, r in assign.items() if r["post_editor"] == pe}
        seen = Counter(r["string_id"].strip() for r in rows)
        for sid in sorted(expected - set(seen)):
            errors.append((stem, "", sid, "E01 the row of this string is missing"))
        for sid in sorted(set(seen) - expected):
            errors.append((stem, "", sid, "E01 this row does not belong to this sheet"))
        for r in rows:
            sid, row_no = r["string_id"].strip(), r["row"].strip()
            if sid not in expected:
                continue
            if seen[sid] > 1:
                errors.append((stem, row_no, sid, "E01 this string is in the sheet more than once"))
                continue
            s = by_id[sid]
            if normalise(r["question_bn"]) != s["norm"] or clean_en(r["first_pass_en"]) != s["mt_en"]:
                errors.append((stem, row_no, sid, "E02 a grey cell (question_bn or "
                               "first_pass_en) was changed. Put the edit into edited_en"))
                continue
            decision, edited = r["decision"].strip().lower(), clean_en(r["edited_en"])
            info["cells_cleaned"] += int(edited != r["edited_en"].strip() and bool(edited))
            if decision not in DECISIONS:
                errors.append((stem, row_no, sid, "E03 decision is empty or is not keep or edit"))
                continue
            if decision == "edit":
                if not edited:
                    errors.append((stem, row_no, sid, "E04 decision is edit but edited_en is empty"))
                    continue
                if edited == s["mt_en"]:
                    errors.append((stem, row_no, sid, "E06 decision is edit but edited_en equals "
                                   "the first pass. Choose keep, or write the corrected text"))
                    continue
                text = edited
            else:
                if edited and edited != s["mt_en"]:
                    errors.append((stem, row_no, sid, "E05 decision is keep but edited_en holds "
                                   "another text. Choose edit, or empty edited_en"))
                    continue
                text = s["mt_en"]
            for prob in english_problems(text):
                errors.append((stem, row_no, sid, f"E07 the English {prob}"))
            results[sid] = {"en": text, "decision": decision, "pe": pe, "sheet": stem,
                            "row": row_no, "note": " ".join(r["note"].split())}
            info[f"{pe}_{decision}"] += 1
            info["notes"] += int(bool(r["note"].strip()))
    return results, errors, info


def read_crosscheck(ctx, scope, part, assign, first):
    """Read both filled cross-check sheets of a scope. Returns (cross, errors).
    cross[string_id] = dict(outcome, en, checker, ...)."""
    by_id = {s["string_id"]: s for s in part}
    cross, errors = {}, []
    for checker in POST_EDITORS:
        stem = f"crosscheck_{SCOPE_LABEL[scope]}_{checker}"
        expected = {sid for sid, r in assign.items() if r["cross_checker"] == checker}
        if not expected:
            raise Stop(f"No cross-check is stored for scope {scope}. Run make-crosscheck "
                       f"--scope {scope} first.")
        path = find_filled(ctx.s2_filled, stem)
        rows = read_table(path, "crosscheck", CROSS_COLS)
        sent = {}
        if (ctx.s2_send / f"{stem}.xlsx").exists():
            sent = {r["string_id"].strip(): clean_en(r["english_to_check"])
                    for r in read_table(ctx.s2_send / f"{stem}.xlsx", "crosscheck", CROSS_COLS)}
        seen = Counter(r["string_id"].strip() for r in rows)
        for sid in sorted(expected - set(seen)):
            errors.append((stem, "", sid, "E01 the row of this string is missing"))
        for sid in sorted(set(seen) - expected):
            errors.append((stem, "", sid, "E01 this row does not belong to this sheet"))
        for r in rows:
            sid, row_no = r["string_id"].strip(), r["row"].strip()
            if sid not in expected or sid not in first:
                continue
            if seen[sid] > 1:
                errors.append((stem, row_no, sid, "E01 this string is in the sheet more than once"))
                continue
            shown = clean_en(r["english_to_check"])
            if normalise(r["question_bn"]) != by_id[sid]["norm"] or (sid in sent and shown != sent[sid]):
                errors.append((stem, row_no, sid, "E02 a grey cell (question_bn or "
                               "english_to_check) was changed. Put your text into proposed_en"))
                continue
            if shown != first[sid]["en"]:
                errors.append((stem, row_no, sid, "E11 this string was changed in the post-edit "
                               f"sheet {first[sid]['sheet']} (row {first[sid]['row']}) after it "
                               "was cross-checked. Undo that change there. Put the new text into "
                               "agreed_en of this sheet, with verdict change, agreed by both"))
                continue
            verdict = r["verdict"].strip().lower()
            proposed, agreed = clean_en(r["proposed_en"]), clean_en(r["agreed_en"])
            if verdict not in CHECK_VERDICTS:
                errors.append((stem, row_no, sid, "E12 verdict is empty or is not ok or change"))
                continue
            if verdict == "ok":
                if proposed or agreed:
                    errors.append((stem, row_no, sid, "E15 verdict is ok but proposed_en or "
                                   "agreed_en holds a text. Choose change, or empty those cells"))
                    continue
                cross[sid] = {"outcome": "ok", "en": shown, "checker": checker,
                              "proposed": "", "reason": "", "sheet": stem, "row": row_no}
                continue
            if not proposed and not agreed:
                errors.append((stem, row_no, sid, "E13 verdict is change but proposed_en is empty"))
                continue
            if not agreed:
                errors.append((stem, row_no, sid, "E14 verdict is change but agreed_en is empty. "
                               "The two post-editors agree on one text and write it there"))
                continue
            for prob in english_problems(agreed):
                errors.append((stem, row_no, sid, f"E07 the English in agreed_en {prob}"))
            cross[sid] = {"outcome": "changed" if agreed != shown else "kept_after_discussion",
                          "en": agreed, "checker": checker, "proposed": proposed,
                          "reason": " ".join(r["reason"].split()), "sheet": stem, "row": row_no}
    return cross, errors


def read_amendments(ctx, valid_ids):
    """Late corrections: files amendments_*.csv or .xlsx in english/amendments/
    with the columns string_id, question_en_new, reason. Returns (amend, errors)."""
    amend, errors = {}, []
    if not ctx.amend_dir.is_dir():
        return amend, errors
    files = sorted(p for p in ctx.amend_dir.iterdir()
                   if p.name.lower().startswith("amendments_")
                   and p.suffix.lower() in (".csv", ".xlsx") and not p.name.startswith("~$"))
    for path in files:
        for n, r in enumerate(read_table(path, None, ("string_id", "question_en_new", "reason")), 2):
            sid, text = r["string_id"].strip(), clean_en(r["question_en_new"])
            if sid not in valid_ids:
                errors.append((path.name, str(n), sid, "E21 unknown string_id"))
                continue
            if not r["reason"].strip():
                errors.append((path.name, str(n), sid, "E22 the reason is empty"))
            for prob in english_problems(text):
                errors.append((path.name, str(n), sid, f"E07 the English in question_en_new {prob}"))
            if sid in amend and amend[sid]["en"] != text:
                errors.append((path.name, str(n), sid, "E23 two amendments give this string "
                               f"different texts (also in {amend[sid]['file']})"))
                continue
            amend[sid] = {"en": text, "file": path.name, "reason": " ".join(r["reason"].split())}
    return amend, errors


def stop_on_errors(ctx, label, errors):
    path = ctx.s2 / f"errors_{label}.txt"
    if not errors:
        if path.exists():
            path.unlink()               # the list of an earlier run is out of date
        return
    errors = sorted(set(errors), key=lambda e: (e[0], int(e[1]) if e[1].isdigit() else 0, e[2:]))
    lines = [f"{sheet}" + (f" row {row}" if row else "") + (f" ({sid})" if sid else "") + f": {msg}"
             for sheet, row, sid, msg in errors]
    write_text(path, lines)
    say(f"\n  ERRORS: {len(errors)}. 'row' is the number in the column named row.")
    for line in lines[:25]:
        say("    " + line)
    if len(lines) > 25:
        say(f"    ... and {len(lines) - 25} more")
    raise SheetErrors(f"{len(errors)} errors in the filled sheets. The full list is in {path}. "
                      "The person who filled the sheet corrects it in the filled file; then "
                      "run the same command again. Nothing was written.", errors)


# --------------------------------------------------------------------------
# Consistency: rows that a person must look at before the file is final
# --------------------------------------------------------------------------
def review_rows(entries):
    """entries: dicts with string_id, norm, group, mt_en, en. Returns a list of
    (class, group key, string_id, message). Nothing here is a judgement: each
    row asks a person to look."""
    out = []

    def add(cls, key, sid, msg):
        out.append((cls, key, sid, msg))

    # R1: the wordings fixed by rule 9 of the stage 1 prompt (D129, D5)
    for bn, word, pattern, not_bn in FIXED_WORDINGS:
        users = [e for e in entries
                 if bn in (nfc(e["norm"]).replace(not_bn, " ") if not_bn else nfc(e["norm"]))]
        without = [e for e in users if not re.search(pattern, e["en"].lower())]
        if users and len(without) == len(users) and len(users) > 1:
            add("fixed_wording", word, without[0]["string_id"],
                f"'{word}' was replaced in all {len(users)} strings that have the Bangla "
                "wording. D5 allows this, with a decision row")
        else:
            for e in without:
                add("fixed_wording", word, e["string_id"],
                    f"the Bangla has the wording that is fixed as '{word}', the English has "
                    f"no '{word}'. It may change only if it changes in every string (D5)")
    # R1b: a wording that is not fixed but should get one English word everywhere
    for bn, word, pattern in WATCHED_WORDINGS:
        users = [e for e in entries if bn in nfc(e["norm"])]
        with_word = {e["string_id"] for e in users if re.search(pattern, e["en"].lower())}
        if 0 < len(with_word) < len(users):
            few_with = len(with_word) * 2 < len(users)
            for e in users:
                if (e["string_id"] in with_word) == few_with:
                    add("watched_wording", word, e["string_id"],
                        f"{len(users)} strings have the same Bangla wording; {len(with_word)} "
                        f"write '{word}' for it and {len(users) - len(with_word)} do not. One "
                        "Bangla wording should get one English wording")
    # R2: the image word (rule 5 of the stage 1 prompt)
    for e in entries:
        if IMAGE_WORD_BN in nfc(e["norm"]) and not re.search(r"\bimages?\b", e["en"].lower()):
            add("image_word", "", e["string_id"],
                "the Bangla has the word for image, the English has no 'image'. Right only "
                "if the Bangla word means something else here, such as a picture in the room")
    # R3: spelling variants of one question
    same = defaultdict(list)
    for e in entries:
        same[e["mt_en"].lower()].append(e)
    for group in same.values():
        if len({e["en"].lower() for e in group}) > 1:
            for e in group:
                add("same_first_pass", group[0]["string_id"], e["string_id"],
                    f"{len(group)} strings had the same first-pass English and now differ. "
                    "Right only if the Bangla strings differ in meaning")
    # R4, R5: count questions, one Bangla frame and one Bangla object name
    frames, terms = defaultdict(list), defaultdict(list)
    for e in entries:
        if e["group"] != "count":
            continue
        bn, en = count_parts_bn(e["norm"]), count_parts_en(e["en"])
        if en is None:
            add("count_form", "", e["string_id"],
                "a count question without the form 'How many ... are ...'")
            continue
        if bn is not None:
            terms[bn[0]].append((e, en[0]))
            frames[bn[1]].append((e, en[1]))
    for cls, groups, what in (("count_frame", frames, "wording"), ("count_term", terms, "object name")):
        for key, members in groups.items():
            counts = Counter(v for _e, v in members).most_common()
            if len(counts) < 2:
                continue
            tie = counts[0][1] == counts[1][1]
            for e, value in members:
                if tie or value != counts[0][0]:
                    add(cls, key, e["string_id"],
                        f"one Bangla {what}, {len(counts)} English versions in {len(members)} "
                        f"count questions; this row has '{value}'"
                        + ("" if tie else f", {counts[0][1]} rows have '{counts[0][0]}'"))
    # R6 to R9: form
    for e in entries:
        nb, ne = len(e["norm"].split()), len(e["en"].split())
        if ne > 2.5 * nb + 4 or ne < 0.6 * nb:
            add("length", "", e["string_id"], f"{ne} English words for {nb} Bangla words")
        if any(ch in e["en"] for ch in "()[]{}/"):
            add("brackets_or_slash", "", e["string_id"],
                "brackets or a slash: no notes and no alternatives inside a question")
        if e["en"].count("?") > 1:
            add("several_questions", "", e["string_id"], "more than one question mark")
        if e["en"][:1] and not e["en"][:1].isupper():
            add("capital", "", e["string_id"], "the question does not start with a capital letter")
    return out


REVIEW_HEADER = ["class", "group", "string_id", "post_editor", "where_to_correct",
                 "question_bn", ENGLISH_COLUMN, "message", "note"]


def handle_reviews(ctx, label, reviews, entries, accept):
    """Write the review file. Stop unless the runner states how many rows the
    post-editors have read and accepted."""
    by_id = {e["string_id"]: e for e in entries}
    path = ctx.s2 / f"review_{label}.csv"
    rows = []
    for cls, key, sid, msg in sorted(reviews):
        e = by_id[sid]
        rows.append([cls, key, sid, e.get("pe", ""), e.get("where", ""),
                     e["question_bn"], e["en"], msg, e.get("note", "")])
    write_csv(path, REVIEW_HEADER, rows)
    counts = Counter(r[0] for r in rows)
    if not rows:
        say("  review rows: 0")
        return counts
    say(f"  review rows: {len(rows)}  (" + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + ")")
    say(f"  the rows, with their text: {path}")
    if accept == len(rows):
        say(f"  accepted by the runner: --accept-review {accept}")
        return counts
    raise ReviewPending(
        f"{len(rows)} review rows are waiting in {path}. The two post-editors read every "
        "row. What is wrong they correct in the filled file that the column "
        "where_to_correct names. Then run the same command again. When every row that is "
        "left is right as it is, run "
        f"again with --accept-review {len(rows)}. Nothing final was written."
        + (f" (You gave --accept-review {accept}; the number is now {len(rows)}.)"
           if accept else ""), len(rows))


def term_rows(entries):
    """A table of the object names in the count questions: Bangla, English."""
    pairs = defaultdict(list)
    for e in entries:
        if e["group"] != "count":
            continue
        bn, en = count_parts_bn(e["norm"]), count_parts_en(e["en"])
        if bn and en:
            pairs[(bn[0], en[0])].append(e)
    n_en, n_bn = Counter(bn for bn, _ in pairs), Counter(en for _, en in pairs)
    rows = []
    for (bn, en), members in sorted(pairs.items()):
        flag = []
        if n_en[bn] > 1:
            flag.append("one Bangla name, several English names")
        if n_bn[en] > 1:
            flag.append("one English name, several Bangla names")
        rows.append([bn, en, len(members), "|".join(sorted({m.get("pe", "") for m in members})),
                     "; ".join(flag), "|".join(m["string_id"] for m in members)])
    return rows


TERM_HEADER = ["bn_term", "en_term", "n_strings", "post_editors", "flag", "string_ids"]

README_AGREE = [
    "# AGREE FIRST (ASSET-2 stage 2)",
    "",
    "# What this is",
    "A help for the two post-editors. It lists what the count questions share: the object "
    "names (tab object names) and the wordings around them (tab count wordings), as the "
    "machine translated them.",
    "One Bangla name should get one English name in all questions, and one Bangla wording "
    "one English wording. Rows that the machine translated in more than one way come first "
    "and carry a flag.",
    "",
    "# What to do",
    "1. Go through both tabs together, before you start your own sheets.",
    "2. Where the first-pass English is right, do nothing.",
    "3. Where the flag shows several versions, or where you want another English name or "
    "wording, agree on one and write it in agreed_english.",
    "4. Each of you then uses the agreed English in the own sheet. The column post_editors "
    "shows whose sheet holds the rows.",
    "",
    "# Good to know",
    "- In the tab count wordings, <N> stands for the how-many word and <X> for the object "
    "name.",
    "- No script reads this file. It is your shared note. The merge command checks the "
    "result in your sheets.",
    "- Do not show this file to the two people who write the lexicon (HT-1), or to the "
    "back-translators.",
]


def agreement_tabs(entries):
    """Two tabs for the post-editors' talk before they start: the object names
    and the wordings of the count questions, as the first pass wrote them."""
    terms, frames = defaultdict(list), defaultdict(list)
    for e in entries:
        if e["group"] != "count":
            continue
        bn, en = count_parts_bn(e["norm"]), count_parts_en(e["en"])
        if bn and en:
            terms[bn[0]].append((e, en[0]))
            frames[bn[1]].append((e, en[1]))
    used_by = defaultdict(set)
    for bn, members in terms.items():
        for _e, en in members:
            used_by[en].add(bn)

    def rows_of(groups, names):
        rows = []
        for bn, members in groups.items():
            versions = Counter(v for _e, v in members).most_common()
            flag = [f"{len(versions)} English versions"] if len(versions) > 1 else []
            if names:
                others = sorted({o for v, _n in versions for o in used_by[v]} - {bn})
                if others:
                    flag.append("same English as: " + ", ".join(others))
            row = [bn, " | ".join(f"{v} ({n})" for v, n in versions), len(members),
                   "|".join(sorted({e["pe"] for e, _v in members})), "; ".join(flag), ""]
            if names:
                row.append(" ".join(e["string_id"] for e, _v in members))
            rows.append((len(versions) < 2, bn, row))
        return [row for _k, _bn, row in sorted(rows)]

    common = ["first_pass_english", "n_rows", "post_editors", "flag", "agreed_english"]
    return [{"name": "object names", "header": ["bn_name"] + common + ["string_ids"],
             "rows": rows_of(terms, True), "widths": WIDTHS, "inputs": ("agreed_english",)},
            {"name": "count wordings", "header": ["bn_wording"] + common,
             "rows": rows_of(frames, False), "widths": WIDTHS, "inputs": ("agreed_english",)}]


# --------------------------------------------------------------------------
# Command: make-postedit
# --------------------------------------------------------------------------
def pilot_hints(ctx):
    """How the PILOT round wrote an object name or a question, as a hint for
    the rest. Empty until the PILOT final file exists."""
    by_term, by_mt = {}, {}
    if not ctx.final_pilot.exists():
        return by_term, by_mt
    for r in read_csv(ctx.final_pilot):
        by_mt[r["first_pass_en"].lower()] = r[ENGLISH_COLUMN]
        if type_group(r["qa_types"]) == "count":
            bn, en = count_parts_bn(r["question_bn_norm"]), count_parts_en(r[ENGLISH_COLUMN])
            if bn and en:
                by_term[bn[0]] = en[0]
    return by_term, by_mt


def attention_notes(rows, hints):
    """Marks for the post-editor. rows: the strings of one sheet, in sheet order."""
    by_term, by_mt = hints
    same_en, same_term = defaultdict(list), defaultdict(list)
    for n, s in enumerate(rows, 1):
        same_en[s["mt_en"].lower()].append(n)
        if s["count_bn"]:
            same_term[s["count_bn"][0]].append(n)
    notes = []
    for n, s in enumerate(rows, 1):
        parts = []
        if s["n_items"] >= HIGH_USE_MIN_ITEMS:
            parts.append(f"{s['n_items']} dataset items use this question: take extra care. The "
                         "other post-editor checks this row too.")
        flags = s["qc_flags"].split("|") if s["qc_flags"] else []
        if "no_image_word" in flags:
            parts.append("Machine flag: the Bangla has the word ছবি but the English has no "
                         "'image'. Check what ছবি means here.")
        if "picture_or_photo" in flags:
            parts.append("Machine flag: the English has 'picture' or 'photo'. Right only if "
                         "ছবি here does not mean the image the question is asked about.")
        for flag in flags:
            if flag not in ("no_image_word", "picture_or_photo"):
                parts.append(f"Machine flag: {flag}.")
        if s["string_id"] in KNOWN_NOTES:
            parts.append(KNOWN_NOTES[s["string_id"]])
        if COMPOUND_THINGS_BN in nfc(s["norm"]):
            parts.append("The Bangla has জিনিসপত্র. The fixed word 'thing' is for জিনিস only. "
                         "Use one English word for জিনিসপত্র in all rows.")
        others = [m for m in same_en[s["mt_en"].lower()] if m != n]
        if others:
            parts.append("Same first-pass English as row " + ", ".join(map(str, others))
                         + ": keep them the same, unless the Bangla differs in meaning.")
        if s["count_bn"]:
            others = [m for m in same_term[s["count_bn"][0]] if m != n]
            if others:
                parts.append("Same object name as row " + ", ".join(map(str, others))
                             + ": use one English name.")
            if s["count_bn"][0] in by_term:
                parts.append(f"PILOT wrote '{by_term[s['count_bn'][0]]}' for this object name.")
        if s["mt_en"].lower() in by_mt and by_mt[s["mt_en"].lower()] != s["mt_en"]:
            parts.append("A PILOT string with the same first-pass English became: "
                         + by_mt[s["mt_en"].lower()])
        notes.append(" ".join(parts))
    return notes


def cmd_make_postedit(ctx, scope, confirmed, force=False):
    say(f"[make-postedit] scope {scope}: post-edit sheets for {len(POST_EDITORS)} post-editors")
    if not confirmed:
        raise Stop("Refused: add --people-rules-confirmed after you have checked that the two "
                   "post-editors, and you, are not in the HT-1 pair (D125: until gate-A is "
                   "tagged nobody in the HT-1 pair opens a file in Drive english/ other than "
                   "asset2_strings.csv), and that neither of them will be a back-translator.")
    strings = load_strings(ctx)
    part, have = scope_assignment(ctx, strings, scope, must_exist=False)
    say(f"  read {len(strings)} strings; scope {scope}: {len(part)}")
    stored = load_assignment(ctx)
    if have:
        assign = {sid: r["post_editor"] for sid, r in have.items()}
        unit_of = {sid: r["unit"] for sid, r in have.items()}
        say(f"  the split of this scope is already stored in {ctx.assignment.name}; it is kept")
    else:
        assign, unit_of = split_halves(part, scope)
        for sid, pe in assign.items():
            stored[sid] = {"string_id": sid, "scope": scope, "post_editor": pe,
                           "unit": unit_of[sid], "cross_checker": "", "cross_basis": ""}
    hints = pilot_hints(ctx) if scope == "rest" else ({}, {})
    if scope == "rest" and not ctx.final_pilot.exists():
        say("  note: the PILOT final file does not exist yet, so the sheets carry no hints "
            "from the PILOT round. Better: finish the PILOT round first.")
    files = []
    for pe in POST_EDITORS:
        # sheet order: by type; count questions by object name, so that the
        # rows of one object stand together; then the running order of stage 1
        rows = sorted((s for s in part if assign[s["string_id"]] == pe),
                      key=lambda s: (GROUP_ORDER.index(s["group"]),
                                     s["count_bn"][0] if s["count_bn"] else "", s["seq"]))
        notes = attention_notes(rows, hints)
        tab = {"name": "postedit", "header": POSTEDIT_COLS, "widths": WIDTHS,
               "inputs": ("decision", "edited_en", "note"), "choices": {"decision": DECISIONS},
               "rows": [[n, TYPE_LABEL[s["group"]], s["n_items"], note, s["question_bn"],
                         s["mt_en"], "", "", "", s["string_id"]]
                        for n, (s, note) in enumerate(zip(rows, notes), 1)]}
        path = ctx.s2_send / f"postedit_{SCOPE_LABEL[scope]}_{pe}.xlsx"
        state = send_sheet(path, tab, README_POSTEDIT, force)
        files.append((pe, len(rows), sum(1 for n in notes if n), state, path))
    save_assignment(ctx, stored)
    entries = [dict(s, en=s["mt_en"], pe=assign[s["string_id"]]) for s in part]
    agree_path = ctx.s2_send / f"agree_first_{SCOPE_LABEL[scope]}.xlsx"
    agree_state = "kept: it exists and may hold your notes"
    if force or not agree_path.exists():
        write_workbook(agree_path, agreement_tabs(entries), README_AGREE)
        agree_state = "written"
    n_units = len(set(unit_of.values()))
    say(f"  split: {n_units} units; {len(part) - n_units} strings stay with a spelling variant "
        "of themselves or with a question that has the same object name")
    for pe in POST_EDITORS:
        c = Counter(s["group"] for s in part if assign[s["string_id"]] == pe)
        say(f"  {pe}: {sum(c.values())} strings  ("
            + ", ".join(f"{TYPE_LABEL[g]} {c[g]}" for g in GROUP_ORDER if c[g]) + ")")
    say("  wrote:")
    for _pe, n, n_att, state, path in files:
        say(f"    {path}   {n} rows, {n_att} with an attention note   [{state}]")
    say(f"    {agree_path}   object names and wordings of the count questions, for both "
        f"post-editors together   [{agree_state}]")
    say(f"    {ctx.assignment}")
    ctx.record("make-postedit", {"scope": scope, "n_strings": len(part),
                                 "per_editor": {pe: n for pe, n, _a, _s, _p in files},
                                 "people_rules_confirmed_by_runner": True})
    say(f"\n  NEXT: give each postedit file to its post-editor with the guide, and "
        f"{agree_path.name} to both. Put the filled postedit files into {ctx.s2_filled}. "
        f"Then run: make-crosscheck --scope {scope}")
    return 0


# --------------------------------------------------------------------------
# Command: make-crosscheck
# --------------------------------------------------------------------------
def scope_entries(part, results, cross=None, amend=None):
    """One entry per string with its current English, for checks and outputs."""
    entries = []
    for s in part:
        r = results[s["string_id"]]
        e = dict(s, en=r["en"], pe=r["pe"], note=r["note"], first_en=r["en"],
                 decision=r["decision"], cross="", amended="",
                 where=f"{r['sheet']} row {r['row']}: decision edit, new text in edited_en")
        if cross and s["string_id"] in cross:
            c = cross[s["string_id"]]
            e["en"], e["cross"] = c["en"], c["outcome"]
            e["where"] = f"{c['sheet']} row {c['row']}: verdict change, new text in agreed_en"
        if amend and s["string_id"] in amend:
            a = amend[s["string_id"]]
            if a["en"] != e["en"]:
                e["en"], e["amended"] = a["en"], a["file"]
            e["where"] = f"amendments/{a['file']}"
        entries.append(e)
    return entries


def pilot_context(ctx, strings):
    """The final PILOT strings as fixed context for checks on the rest."""
    if not ctx.final_pilot.exists():
        return []
    final = {r["string_id"]: r for r in read_csv(ctx.final_pilot)}
    return [dict(s, en=final[s["string_id"]][ENGLISH_COLUMN], pe=final[s["string_id"]]["post_editor"],
                 note="", where="PILOT final file: change it only with a file in amendments/")
            for s in strings if s["in_pilot"] and s["string_id"] in final]


def cmd_make_crosscheck(ctx, scope, accept_review=0, force=False):
    say(f"[make-crosscheck] scope {scope}: each post-editor re-checks "
        f"{100 * CROSSCHECK_NUM // CROSSCHECK_DEN}% of the other's half")
    strings = load_strings(ctx)
    part, assign = scope_assignment(ctx, strings, scope)
    results, errors, info = read_postedit(ctx, scope, part, assign)
    stop_on_errors(ctx, f"postedit_{SCOPE_LABEL[scope]}", errors)
    say(f"  read the filled post-edit sheets: {len(results)} of {len(part)} strings, 0 errors")
    for pe in POST_EDITORS:
        say(f"  {pe}: keep {info[pe + '_keep']}, edit {info[pe + '_edit']}")
    entries = scope_entries(part, results)
    context = pilot_context(ctx, strings) if scope == "rest" else []
    handle_reviews(ctx, f"first_pass_{SCOPE_LABEL[scope]}", review_rows(entries + context),
                   entries + context, accept_review)
    stored = load_assignment(ctx)
    files = []
    for pe in POST_EDITORS:
        checker = other_editor(pe)
        half = [s for s in part if assign[s["string_id"]]["post_editor"] == pe]
        chosen = {s["string_id"]: assign[s["string_id"]]["cross_basis"] for s in half
                  if assign[s["string_id"]]["cross_checker"]}          # stored: kept
        if not chosen:
            chosen = crosscheck_selection(half, scope)
        for s in half:
            sid = s["string_id"]
            stored[sid]["cross_checker"] = checker if sid in chosen else ""
            stored[sid]["cross_basis"] = chosen.get(sid, "")
        rows = [e for e in entries if e["string_id"] in chosen]
        tab = {"name": "crosscheck", "header": CROSS_COLS, "widths": WIDTHS,
               "inputs": ("verdict", "proposed_en", "reason", "agreed_en"),
               "choices": {"verdict": CHECK_VERDICTS},
               "rows": [[n, TYPE_LABEL[e["group"]], e["n_items"], e["question_bn"], e["en"], "",
                         "", "", "", e["string_id"]] for n, e in enumerate(rows, 1)]}
        path = ctx.s2_send / f"crosscheck_{SCOPE_LABEL[scope]}_{checker}.xlsx"
        state = send_sheet(path, tab, README_CROSSCHECK, force)
        n_random = sum(1 for b in chosen.values() if b == "random")
        files.append((checker, pe, n_random, len(rows) - n_random, len(half), state, path))
    save_assignment(ctx, stored)
    say("  wrote:")
    for checker, pe, n, n_high, n_half, state, path in files:
        say(f"    {path}   for {checker}: {n + n_high} of the {n_half} strings of {pe} "
            f"(random 10%: {n}; used by {HIGH_USE_MIN_ITEMS} or more items: {n_high})   [{state}]")
    ctx.record("make-crosscheck", {"scope": scope,
                                   "per_checker": {c: {"random": n, "high_use": h}
                                                   for c, _p, n, h, _h, _s, _f in files},
                                   "first_pass": dict(info), "accept_review": accept_review})
    say(f"\n  NEXT: give each crosscheck file to its post-editor. The two then agree on every "
        f"row marked change (column agreed_en). Put the filled files into {ctx.s2_filled}. "
        f"Then run: merge --scope {'pilot' if scope == 'pilot' else 'all'}")
    return 0


# --------------------------------------------------------------------------
# Command: merge
# --------------------------------------------------------------------------
def final_rows(entries, version):
    rows = []
    for e in sorted(entries, key=lambda e: e["seq"]):
        rows.append([e["seq"], e["string_id"], "yes" if e["in_pilot"] else "no", e["in_pclust"],
                     e["qa_types"], e["probes"], e["n_items"], e["question_bn"], e["en"],
                     e["norm"], e["n_surface_forms"], e["mt_en"],
                     "kept" if e["en"] == e["mt_en"] else "edited", e["pe"], e["cross"],
                     e["amended"], version])
    return rows


def write_final(ctx, path, entries, overwrite):
    """Write a final file. A final file that exists is replaced only with
    --overwrite-final, and the old version is kept in final_history/."""
    version, state, changed = 1, "written", []
    if path.exists():
        old = read_csv(path)
        old_version = int(old[0]["final_version"]) if old else 0
        core = [r[:-1] for r in final_rows(entries, 0)]
        old_core = [[str(r.get(c, "")) for c in FINAL_HEADER[:-1]] for r in old]
        if [[str(v) for v in r] for r in core] == old_core:
            return old_version, "unchanged", []
        old_en = {r["string_id"]: r[ENGLISH_COLUMN] for r in old}
        changed = sorted(e["string_id"] for e in entries if old_en.get(e["string_id"]) != e["en"])
        if not overwrite:
            raise Stop(f"{path} exists (version {old_version}) and the sheets now give another "
                       f"content: the English of {len(changed)} strings differs"
                       + (f" ({ids_line(changed)})" if changed else " (only other columns)")
                       + ". If a run has already used this file (the pilot, or the "
                       "back-translation draw), think before you replace it. To replace it, "
                       "run again with --overwrite-final. The old version is kept in "
                       f"{ctx.history}.")
        ctx.history.mkdir(parents=True, exist_ok=True)
        keep = ctx.history / f"{path.stem}_v{old_version}.csv"
        if not keep.exists():
            shutil.copyfile(path, keep)
        version, state = old_version + 1, "replaced"
    write_csv(path, FINAL_HEADER, final_rows(entries, version))
    ctx.history.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, ctx.history / f"{path.stem}_v{version}.csv")
    return version, state, changed


LOG_HEADER = ["string_id", "scope", "n_items", "post_editor", "decision", "first_pass_en",
              "after_post_edit", "cross_checker", "cross_basis", "cross_check", "proposed_en",
              "cross_check_reason", "amended", "amend_reason", ENGLISH_COLUMN, "note"]


def cmd_merge(ctx, scope, accept_review=0, overwrite=False):
    label = SCOPE_LABEL[scope]
    say(f"[merge] scope {scope}: check the filled sheets and write the final English file")
    strings = load_strings(ctx)
    entries, errors, info, cross_all = [], [], Counter(), {}
    amend, amend_errors = read_amendments(ctx, {s["string_id"] for s in strings})
    errors += amend_errors
    for sc in (("pilot",) if scope == "pilot" else ("pilot", "rest")):
        part, assign = scope_assignment(ctx, strings, sc)
        results, e1, i1 = read_postedit(ctx, sc, part, assign)
        cross, e2 = read_crosscheck(ctx, sc, part, assign, results)
        errors += e1 + e2
        info.update(i1)
        cross_all.update(cross)
        if not e1 and not e2:
            want = {sid for sid, r in assign.items() if r["cross_checker"]}
            if set(cross) != want:
                raise Stop(f"Scope {sc}: the cross-check sheets do not cover the stored "
                           "selection. Ask in the ASSET-2 chat.")
            part_entries = scope_entries(part, results, cross, amend)
            for e in part_entries:
                e["scope"], e["checker"] = sc, assign[e["string_id"]]["cross_checker"]
                e["basis"] = assign[e["string_id"]]["cross_basis"]
            entries += part_entries
    stop_on_errors(ctx, f"merge_{label}", errors)
    n_scope = len([s for s in strings if in_scope(s, scope)])
    if len(entries) != n_scope:
        raise Stop(f"Internal check failed: {len(entries)} strings merged, {n_scope} expected.")
    for e in entries:
        for prob in english_problems(e["en"]):
            raise Stop(f"Internal check failed: the final English of {e['string_id']} {prob}.")
    say(f"  read the filled sheets: {len(entries)} strings, 0 errors")
    n_cross = Counter(e["cross"] for e in entries if e["cross"])
    n_amend = sum(1 for e in entries if e["amended"])
    reviews = review_rows(entries)
    counts = handle_reviews(ctx, label, reviews, entries, accept_review)

    # outputs
    target = ctx.final_pilot if scope == "pilot" else ctx.final_all
    version, state, changed = write_final(ctx, target, entries, overwrite)
    write_csv(ctx.s2 / f"postedit_log_{label}.csv", LOG_HEADER, [
        [e["string_id"], e["scope"], e["n_items"], e["pe"], e["decision"], e["mt_en"],
         e["first_en"], e["checker"], e["basis"], e["cross"],
         cross_all.get(e["string_id"], {}).get("proposed", ""),
         cross_all.get(e["string_id"], {}).get("reason", ""), e["amended"],
         amend[e["string_id"]]["reason"] if e["amended"] else "", e["en"], e["note"]]
        for e in sorted(entries, key=lambda e: e["seq"])])
    write_csv(ctx.s2 / f"term_table_{label}.csv", TERM_HEADER, term_rows(entries))
    write_csv(ctx.s2 / f"notes_{label}.csv",
              ["string_id", "post_editor", "n_items", "probes", "question_bn", ENGLISH_COLUMN, "note"],
              [[e["string_id"], e["pe"], e["n_items"], e["probes"], e["question_bn"], e["en"],
                e["note"]] for e in sorted(entries, key=lambda e: e["seq"]) if e["note"]])
    pilot_note = ""
    if scope == "all":
        pil = [e for e in entries if e["in_pilot"]]
        if not ctx.final_pilot.exists():
            write_final(ctx, ctx.final_pilot, pil, False)
            pilot_note = f"wrote {ctx.final_pilot} (it did not exist)"
        else:
            old = {r["string_id"]: r[ENGLISH_COLUMN] for r in read_csv(ctx.final_pilot)}
            diff = sorted(e["string_id"] for e in pil if old.get(e["string_id"]) != e["en"])
            pilot_note = (f"PILOT strings whose English differs from {ctx.final_pilot.name}: "
                          f"{len(diff)}" + (f" ({ids_line(diff)}). That file was left as it is: "
                                            "the pilot is tied to it. The full runs read "
                                            f"{ctx.final_all.name}." if diff else ""))
    n_edit = sum(1 for e in entries if e["en"] != e["mt_en"])
    shared = sum(1 for _t, c in Counter(e["en"].lower() for e in entries).items() if c > 1)
    report = [
        f"ASSET-2 stage 2 merge   scope {scope}   utc {utc_now()}   script {SCRIPT_NAME} {SCRIPT_VERSION}",
        f"strings in the final file: {len(entries)}",
        f"English kept as the first pass wrote it: {len(entries) - n_edit}",
        f"English changed by people: {n_edit}",
    ] + [f"  {pe}: {info[pe + '_keep'] + info[pe + '_edit']} strings, keep {info[pe + '_keep']}, "
         f"edit {info[pe + '_edit']}" for pe in POST_EDITORS] + [
        f"cross-checked by the other post-editor: {sum(n_cross.values())} strings",
    ] + [f"  {name}: {sum(1 for e in entries if e['basis'] == basis)}  (" + ", ".join(
        f"{label} {sum(1 for e in entries if e['basis'] == basis and e['cross'] == k)}"
        for k, label in (("ok", "ok"), ("changed", "changed"),
                         ("kept_after_discussion", "kept after discussion"))) + ")"
         for basis, name in (("random", "random 10% of each half"),
                             ("high_use", f"used by {HIGH_USE_MIN_ITEMS} or more items"))] + [
        "  checked by " + ", ".join(
            f"{pe}: {sum(1 for e in entries if e['checker'] == pe)}" for pe in POST_EDITORS),
        f"amended by a file in amendments/: {n_amend}"
        + (f"  ({ids_line(sorted(e['string_id'] for e in entries if e['amended']))})" if n_amend else ""),
        f"typed cells cleaned (curly quotes, double spaces): {info['cells_cleaned']}",
        f"rows with a note from a post-editor: {info['notes']}  (read them: notes_{label}.csv)",
        f"English strings shared by two or more Bangla strings: {shared}",
        f"review rows: {sum(counts.values())}"
        + (" (" + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + f"), accepted by the "
           f"runner with --accept-review {accept_review}" if counts else ""),
        f"spelling: {SPELLING}   English column: {ENGLISH_COLUMN}",
        f"final file: {target.name}   version {version}   [{state}]"
        + (f"   English changed against the version before: {len(changed)}" if state == "replaced" else ""),
        f"sha256 of the final file: {file_sha256(target)}",
    ] + ([pilot_note] if pilot_note else []) + [
        "",
        f"RESULT: COMPLETE for scope {scope}",
    ]
    if counts:
        report += ["", "review rows accepted (ids only; the text is in review_" + label + ".csv):"]
        report += [f"  {sid}: {cls}" for cls, _k, sid, _m in sorted(reviews)]
    write_text(ctx.s2 / f"merge_report_{label}.txt", report)
    for line in report[1:report.index(f"RESULT: COMPLETE for scope {scope}") + 1]:
        say("  " + line if line else "")
    say("  wrote:")
    for p, what in ((target, f"{len(entries)} rows; the English is in the column {ENGLISH_COLUMN}"),
                    (ctx.s2 / f"merge_report_{label}.txt", "this report; ids and counts only"),
                    (ctx.s2 / f"postedit_log_{label}.csv", "every string, step by step"),
                    (ctx.s2 / f"notes_{label}.csv", "the rows with a note from a post-editor"),
                    (ctx.s2 / f"term_table_{label}.csv", "object names in the count questions"),
                    (ctx.s2 / f"review_{label}.csv", "the review rows")):
        say(f"    {p}   ({what})")
    ctx.record("merge", {"scope": scope, "n_strings": len(entries), "n_changed_by_people": n_edit,
                         "cross_check": {b: dict(Counter(e["cross"] for e in entries
                                                         if e["basis"] == b))
                                         for b in ("random", "high_use")},
                         "n_amended": n_amend,
                         "review_rows": dict(counts), "accept_review": accept_review,
                         "final_file": target.name, "final_version": version, "state": state,
                         "sha256": file_sha256(target), "spelling": SPELLING,
                         "english_column": ENGLISH_COLUMN})
    if scope == "pilot":
        say(f"\n  NEXT: the pilot reads {target.name} with --english-column {ENGLISH_COLUMN}. "
            "Do not replace that file once the pilot has started. Then run: make-postedit "
            "--scope rest")
    elif (ctx.s3(1) / "backtrans_key_R1.csv").exists():
        say("\n  NEXT: a back-translation draw exists already. Run status to see the next step.")
    else:
        say("\n  NEXT: run make-backtrans --people-rules-confirmed (stage 3).")
    return 0


# --------------------------------------------------------------------------
# Stage 3: the draw and the back-translation sheets
# --------------------------------------------------------------------------
def load_final(ctx, strings):
    if not ctx.final_all.exists():
        raise Stop(f"{ctx.final_all} does not exist. Run merge --scope all first: the "
                   "back-translation is drawn from the final file of all strings.")
    final = read_csv(ctx.final_all)
    by_id = {s["string_id"]: s for s in strings}
    if {r["string_id"] for r in final} != set(by_id) or len(final) != len(strings):
        raise Stop(f"{ctx.final_all} does not hold exactly the {len(strings)} strings of stage 1.")
    for r in final:
        if r["question_bn_norm"] != by_id[r["string_id"]]["norm"]:
            raise Stop(f"{ctx.final_all}: {r['string_id']} has another question_bn_norm than stage 1.")
        probs = english_problems(r[ENGLISH_COLUMN])
        if probs:
            raise Stop(f"{ctx.final_all}: the English of {r['string_id']} {probs[0]}.")
    return {r["string_id"]: r for r in final}


BT_KEY_HEADER = ["item", "translator", "string_id", "draw_rank", ENGLISH_COLUMN, "final_version"]


def cmd_make_backtrans(ctx, rnd, confirmed, force=False):
    n, half = ctx.sample_size, ctx.sample_size // 2
    say(f"[make-backtrans] round {rnd}: draw {n} strings (seeded rank, seed {SEED}), "
        f"{half} for each of {len(TRANSLATORS)} back-translators")
    if not confirmed:
        raise Stop("Refused: add --people-rules-confirmed after you have checked that the two "
                   "back-translators did not post-edit (HT-2), are not in the HT-1 pair (D125), "
                   "have not opened the files in Drive english/, and will each get only their "
                   "own sheet. You, the runner, are not a back-translator.")
    strings = load_strings(ctx)
    final = load_final(ctx, strings)
    by_id = {s["string_id"]: s for s in strings}
    ranked = sorted(final, key=lambda sid: rank_key("backtrans", sid))
    sample = ranked[n * (rnd - 1):n * rnd]
    if len(sample) < n:
        raise Stop(f"Round {rnd} needs strings {n * (rnd - 1) + 1} to {n * rnd} of the ranking, "
                   f"but there are only {len(ranked)} strings.")
    rank_of = {sid: i + 1 for i, sid in enumerate(ranked)}
    order = sorted(sample, key=lambda sid: rank_key(f"bt-assign:R{rnd}", sid))
    who = {sid: TRANSLATORS[0] if i < half else TRANSLATORS[1] for i, sid in enumerate(order)}
    out = ctx.s3(rnd)
    key_rows, files = [], []
    for bt in TRANSLATORS:
        mine = sorted((sid for sid in sample if who[sid] == bt),
                      key=lambda sid: rank_key(f"bt-order:R{rnd}", sid))
        rows = []
        for i, sid in enumerate(mine, 1):
            item = f"R{rnd}-{bt}-{i:03d}"
            rows.append([item, final[sid][ENGLISH_COLUMN], "", ""])
            key_rows.append([item, bt, sid, rank_of[sid], final[sid][ENGLISH_COLUMN],
                             final[sid]["final_version"]])
        files.append((bt, rows))
    key_path = out / f"backtrans_key_R{rnd}.csv"
    if key_path.exists():
        old = [[r[c] for c in BT_KEY_HEADER] for r in read_csv(key_path)]
        if old != [[str(v) for v in r] for r in key_rows] and not force:
            raise Stop(f"{key_path} exists and differs from the draw made now: the final "
                       "English file changed after the draw. If the back-translators already "
                       "have their sheets, keep the old sheets and go on with make-compare; "
                       "the result is then about the English they were given. To throw the "
                       "old draw away and send new sheets, run again with --force.")
    written = []
    for bt, rows in files:
        tab = {"name": "backtranslation", "header": BT_COLS, "widths": WIDTHS,
               "inputs": ("back_translation_bn", "note"), "rows": rows}
        path = out / "to_send" / f"backtrans_R{rnd}_{bt}.xlsx"
        written.append((bt, len(rows), send_sheet(path, tab, README_BACKTRANS, force), path))
    write_csv(key_path, BT_KEY_HEADER, key_rows)
    groups = Counter(by_id[sid]["group"] for sid in sample)
    editors = Counter(final[sid]["post_editor"] for sid in sample)
    texts = Counter(final[sid][ENGLISH_COLUMN].lower() for sid in sample)
    report = [
        f"ASSET-2 stage 3 draw   round {rnd}   utc {utc_now()}   script {SCRIPT_NAME} {SCRIPT_VERSION}",
        f"pool: {len(ranked)} strings of {ctx.final_all.name}, version "
        f"{final[sample[0]]['final_version']}, sha256 {file_sha256(ctx.final_all)}",
        f"drawn: {len(sample)} strings, ranks {n * (rnd - 1) + 1} to {n * rnd} of the seeded "
        f"ranking (seed {SEED})",
        "by type: " + ", ".join(f"{TYPE_LABEL[g]} {groups[g]}" for g in GROUP_ORDER if groups[g]),
        "by post-editor: " + ", ".join(f"{pe} {editors[pe]}" for pe in POST_EDITORS),
        f"PILOT strings in the draw: {sum(1 for sid in sample if by_id[sid]['in_pilot'])}",
        f"changed by people in stage 2: {sum(1 for sid in sample if final[sid]['edit_status'] == 'edited')}",
        f"English sentences that occur twice or more in the draw: {sum(1 for c in texts.values() if c > 1)}",
        "per back-translator: " + ", ".join(f"{bt} {k}" for bt, k, _s, _p in written),
        "",
        "drawn strings (ids only):",
    ] + [f"  {sid}" for sid in sample]
    write_text(out / f"draw_report_R{rnd}.txt", report)
    for line in report[1:9]:
        say("  " + line)
    say("  wrote:")
    for _bt, k, state, path in written:
        say(f"    {path}   {k} rows, English only   [{state}]")
    say(f"    {key_path}   (links items to strings; never give it to a back-translator)")
    say(f"    {out / f'draw_report_R{rnd}.txt'}")
    ctx.record("make-backtrans", {"round": rnd, "n_drawn": len(sample), "by_type": dict(groups),
                                  "final_sha256": file_sha256(ctx.final_all),
                                  "people_rules_confirmed_by_runner": True})
    say(f"\n  NEXT: give each back-translator only their own file, with the guide for "
        f"back-translators. Put the filled files into {out / 'filled'}. Then run: "
        f"make-compare" + (f" --round {rnd}" if rnd != 1 else ""))
    return 0


def stop_on_sheet_errors(path, errors, unit, kind, who):
    """Stage 3 and 3b: list the errors of the filled sheets and stop."""
    if not errors:
        if path.exists():
            path.unlink()               # the list of an earlier run is out of date
        return
    errors = sorted(set(errors))
    lines = [f"{sheet} {unit} {code}: {msg}" for sheet, code, msg in errors]
    write_text(path, lines)
    say(f"\n  ERRORS: {len(errors)}")
    for line in lines[:25]:
        say("    " + line)
    if len(lines) > 25:
        say(f"    ... and {len(lines) - 25} more")
    raise SheetErrors(f"{len(errors)} errors in the filled {kind} sheets. The full list is in "
                      f"{path}. The {who} corrects the filled file; then run the same command "
                      "again. Nothing was written.", errors)


# --------------------------------------------------------------------------
# Stage 3b: the comparison sheets
# --------------------------------------------------------------------------
def read_bt_key(ctx, rnd):
    path = ctx.s3(rnd) / f"backtrans_key_R{rnd}.csv"
    if not path.exists():
        raise Stop(f"{path} does not exist. Run make-backtrans"
                   + (f" --round {rnd}" if rnd != 1 else "") + " first.")
    return read_csv(path)


CMP_KEY_HEADER = ["pair", "judge", "string_id", "item", "translator", "post_editor", "type",
                  "same_text", ENGLISH_COLUMN, "original_bn", "back_translation_bn",
                  "translator_note"]


def cmd_make_compare(ctx, rnd, force=False):
    say(f"[make-compare] round {rnd}: comparison sheets for the post-editors (stage 3b)")
    strings = load_strings(ctx)
    by_id = {s["string_id"]: s for s in strings}
    key = read_bt_key(ctx, rnd)
    editor = {sid: r["post_editor"] for sid, r in load_assignment(ctx).items()}
    out = ctx.s3(rnd)
    bt_text, bt_note, errors = {}, {}, []
    for bt in TRANSLATORS:
        stem = f"backtrans_R{rnd}_{bt}"
        rows = read_table(find_filled(out / "filled", stem), "backtranslation", BT_COLS)
        mine = {r["item"]: r for r in key if r["translator"] == bt}
        seen = Counter(r["item"].strip() for r in rows)
        for item in sorted(set(mine) - set(seen)):
            errors.append((stem, item, "E31 this row is missing"))
        for r in rows:
            item = r["item"].strip()
            if item not in mine:
                errors.append((stem, item, "E31 this row does not belong to this sheet"))
                continue
            if seen[item] > 1:
                errors.append((stem, item, "E31 this item is in the sheet more than once"))
                continue
            if clean_en(r["question_en"]) != clean_en(mine[item][ENGLISH_COLUMN]):
                errors.append((stem, item, "E32 the grey cell question_en was changed"))
                continue
            text = clean_bn(r["back_translation_bn"])
            probs = bangla_problems(text)
            if probs:
                errors.append((stem, item, f"E33 the back-translation {probs[0]}"))
                continue
            bt_text[mine[item]["string_id"]] = text
            bt_note[mine[item]["string_id"]] = " ".join(r["note"].split())
    stop_on_sheet_errors(out / f"errors_backtrans_R{rnd}.txt", errors, "item",
                         "back-translation", "back-translator")
    if len(bt_text) != len(key):
        raise Stop(f"Internal check failed: {len(bt_text)} back-translations for {len(key)} items.")
    say(f"  read the filled back-translation sheets: {len(bt_text)} of {len(key)}, 0 errors")
    order = sorted(key, key=lambda r: rank_key(f"compare-order:R{rnd}", r["string_id"]))
    cmp_rows, sheets = [], {pe: [] for pe in POST_EDITORS}
    for i, r in enumerate(order, 1):
        sid = r["string_id"]
        s = by_id[sid]
        judge = other_editor(editor[sid])        # never the post-editor of the string
        same = "identical" if normalise(bt_text[sid]) == s["norm"] else ""
        pair = f"R{rnd}-C{i:03d}"
        cmp_rows.append([pair, judge, sid, r["item"], r["translator"], editor[sid],
                         TYPE_LABEL[s["group"]], same, r[ENGLISH_COLUMN], s["question_bn"],
                         bt_text[sid], bt_note[sid]])
        sheets[judge].append([pair, TYPE_LABEL[s["group"]], s["question_bn"], bt_text[sid], same,
                              "", "", "", "", r[ENGLISH_COLUMN], bt_note[sid], sid])
    written = []
    for pe in POST_EDITORS:
        tab = {"name": "compare", "header": COMPARE_COLS, "widths": WIDTHS,
               "inputs": ("verdict", "cause", "what_differs", "fix_en"),
               "choices": {"verdict": FAITH_VERDICTS, "cause": CAUSES}, "rows": sheets[pe]}
        path = out / "to_send" / f"compare_R{rnd}_{pe}.xlsx"
        written.append((pe, len(sheets[pe]), send_sheet(path, tab, README_COMPARE, force), path))
    key_path = out / f"compare_key_R{rnd}.csv"
    write_csv(key_path, CMP_KEY_HEADER, cmp_rows)
    n_same = sum(1 for r in cmp_rows if r[7])
    n_notes = sum(1 for note in bt_note.values() if note)
    say(f"  pairs: {len(cmp_rows)}; back-translation identical to the original after "
        f"normalise(): {n_same}; rows with a note from a back-translator: {n_notes}")
    say("  each pair goes to the post-editor who did not post-edit that string")
    say("  wrote:")
    for _pe, k, state, path in written:
        say(f"    {path}   {k} pairs   [{state}]")
    say(f"    {key_path}")
    ctx.record("make-compare", {"round": rnd, "n_pairs": len(cmp_rows), "n_identical": n_same,
                                "per_judge": {pe: k for pe, k, _s, _p in written},
                                "back_translator_notes": n_notes})
    say(f"\n  NEXT: give each compare file to its post-editor. Put the filled files into "
        f"{out / 'filled'}. Then run: result" + (f" --round {rnd}" if rnd != 1 else ""))
    return 0


# --------------------------------------------------------------------------
# Stage 3b: the result
# --------------------------------------------------------------------------
def cmd_result(ctx, rnd):
    say(f"[result] round {rnd}: percent faithful with the Wilson 95% CI (stage 3b)")
    out = ctx.s3(rnd)
    key_path = out / f"compare_key_R{rnd}.csv"
    if not key_path.exists():
        raise Stop(f"{key_path} does not exist. Run make-compare"
                   + (f" --round {rnd}" if rnd != 1 else "") + " first.")
    key = {r["pair"]: r for r in read_csv(key_path)}
    judged, errors, fixes = {}, [], {}
    for pe in POST_EDITORS:
        stem = f"compare_R{rnd}_{pe}"
        mine = {p for p, r in key.items() if r["judge"] == pe}
        if not mine:
            continue
        rows = read_table(find_filled(out / "filled", stem), "compare", COMPARE_COLS)
        seen = Counter(r["pair"].strip() for r in rows)
        for pair in sorted(mine - set(seen)):
            errors.append((stem, pair, "E41 this row is missing"))
        for r in rows:
            pair = r["pair"].strip()
            if pair not in mine:
                errors.append((stem, pair, "E41 this row does not belong to this sheet"))
                continue
            if seen[pair] > 1:
                errors.append((stem, pair, "E41 this pair is in the sheet more than once"))
                continue
            k = key[pair]
            if (clean_bn(r["original_bn"]) != clean_bn(k["original_bn"])
                    or clean_bn(r["back_translation_bn"]) != clean_bn(k["back_translation_bn"])):
                errors.append((stem, pair, "E47 a grey cell (original_bn or back_translation_bn) "
                               "was changed"))
                continue
            verdict, cause = _key(r["verdict"]), CAUSE_KEY.get(_key(r["cause"]), "")
            fix = clean_en(r["fix_en"])
            if verdict not in ("faithful", "not faithful"):
                errors.append((stem, pair, "E42 verdict is empty or is not faithful or not faithful"))
                continue
            if verdict == "faithful":
                if r["cause"].strip() or fix:
                    errors.append((stem, pair, "E43 verdict is faithful but a cause or a fix is "
                                   "given. Empty those cells, or choose not faithful"))
                    continue
            else:
                if k["same_text"]:
                    errors.append((stem, pair, "E44 the two Bangla texts are identical but the "
                                   "verdict is not faithful"))
                    continue
                if not cause:
                    errors.append((stem, pair, "E45 verdict is not faithful but cause is empty "
                                   "or is not one of the four choices"))
                    continue
                if fix:
                    if cause != "english":
                        errors.append((stem, pair, "E46 fix_en is given but the cause is not "
                                       "English question"))
                        continue
                    probs = english_problems(fix)
                    if probs:
                        errors.append((stem, pair, f"E07 the English in fix_en {probs[0]}"))
                        continue
                    if fix == clean_en(k[ENGLISH_COLUMN]):
                        errors.append((stem, pair, "E48 fix_en equals the English that was "
                                       "back-translated. Write the corrected text, or empty it"))
                        continue
                    fixes[k["string_id"]] = (fix, pair)
            judged[pair] = {"faithful": verdict == "faithful", "cause": cause,
                            "what": " ".join(r["what_differs"].split())}
    stop_on_sheet_errors(out / f"errors_compare_R{rnd}.txt", errors, "pair", "comparison",
                         "post-editor")
    n = len(key)
    if len(judged) != n or n != ctx.sample_size or len({r["string_id"] for r in key.values()}) != n:
        raise Stop(f"Internal check failed: {len(judged)} verdicts for {n} pairs "
                   f"(expected {ctx.sample_size} different strings).")
    k = sum(1 for v in judged.values() if v["faithful"])
    lo, hi = wilson(k, n)
    passed = passes_threshold(k, n)
    need = -(-THRESHOLD_PCT * n // 100)
    bad = sorted(p for p, v in judged.items() if not v["faithful"])
    causes = Counter(judged[p]["cause"] for p in bad)

    def split(field):
        tot, good = Counter(), Counter()
        for p, v in judged.items():
            tot[key[p][field]] += 1
            good[key[p][field]] += int(v["faithful"])
        return ", ".join(f"{name} {good[name]} of {tot[name]}" for name in sorted(tot))

    no_fix = [p for p in bad if judged[p]["cause"] == "english" and key[p]["string_id"] not in fixes]
    amend_path = ctx.amend_dir / f"amendments_stage3b_R{rnd}.csv"
    write_csv(amend_path, ["string_id", "question_en_new", "reason"],
              [[sid, fix, f"stage 3b round {rnd}, pair {pair}: not faithful, cause English "
                "question; text agreed by the two post-editors"]
               for sid, (fix, pair) in sorted(fixes.items())])
    write_csv(out / f"stage3b_not_faithful_R{rnd}.csv",
              ["pair", "string_id", "judge", "translator", "post_editor", "type", "cause",
               "what_differs", "original_bn", "back_translation_bn", ENGLISH_COLUMN,
               "translator_note", "fix_en"],
              [[p, key[p]["string_id"], key[p]["judge"], key[p]["translator"],
                key[p]["post_editor"], key[p]["type"], CAUSE_LABEL[judged[p]["cause"]],
                judged[p]["what"], key[p]["original_bn"], key[p]["back_translation_bn"],
                key[p][ENGLISH_COLUMN], key[p]["translator_note"],
                fixes.get(key[p]["string_id"], ("", ""))[0]] for p in bad])
    report = [
        f"ASSET-2 stage 3b result   round {rnd}   utc {utc_now()}   script {SCRIPT_NAME} {SCRIPT_VERSION}",
        f"pairs judged: {n}",
        f"faithful: {k} of {n} = {100 * k / n:.2f}%",
        f"Wilson 95% CI: {100 * lo:.2f}% to {100 * hi:.2f}%",
        f"threshold: at least {THRESHOLD_PCT}% faithful, which is {need} of {n} (Design 4.6)",
        f"not faithful: {n - k}"
        + ("  (" + ", ".join(f"{CAUSE_LABEL[c]} {causes[c]}" for c in CAUSE_LABEL if causes[c])
           + ")" if bad else ""),
        "faithful by type: " + split("type"),
        "faithful by judge: " + split("judge"),
        "faithful by back-translator: " + split("translator"),
        "faithful by post-editor of the string: " + split("post_editor"),
        f"pairs with identical Bangla text: {sum(1 for r in key.values() if r['same_text'])}",
        f"corrected English agreed (fix_en): {len(fixes)}; cause English question without a "
        f"fix yet: {len(no_fix)}",
        "",
        f"RESULT: stage 3b {'PASS' if passed else 'FAIL'}"
        + (". Gate B also needs zero leaks in the naming questions (stage 4, after Gate A)."
           if passed else ". Design 4.6: fix, then repeat stage 3 on a fresh "
                          f"{n} (make-backtrans --round {rnd + 1})."),
        "",
        "not faithful pairs (ids only; the text is in stage3b_not_faithful_R" + f"{rnd}.csv):",
    ] + [f"  {p}  {key[p]['string_id']}  cause: {CAUSE_LABEL[judged[p]['cause']]}" for p in bad]
    write_text(out / f"stage3b_result_R{rnd}.txt", report)
    with open(out / f"stage3b_result_R{rnd}.json", "w", encoding="utf-8") as f:
        json.dump({"round": rnd, "utc": utc_now(), "n": n, "faithful": k,
                   "percent_faithful": round(100 * k / n, 4),
                   "wilson95_low": round(100 * lo, 4), "wilson95_high": round(100 * hi, 4),
                   "threshold_percent": THRESHOLD_PCT, "passed": passed,
                   "not_faithful_by_cause": dict(causes), "n_fixes": len(fixes),
                   "seed": SEED, "script_version": SCRIPT_VERSION}, f, indent=1)
    for line in report[1:14]:
        say("  " + line if line else "")
    say("  wrote:")
    say(f"    {out / f'stage3b_result_R{rnd}.txt'}   (ids and counts only)")
    say(f"    {out / f'stage3b_result_R{rnd}.json'}")
    say(f"    {out / f'stage3b_not_faithful_R{rnd}.csv'}   ({n - k} rows, with text)")
    say(f"    {amend_path}   (corrections: {len(fixes)})")
    ctx.record("result", {"round": rnd, "n": n, "faithful": k, "passed": passed,
                          "wilson95": [round(100 * lo, 4), round(100 * hi, 4)],
                          "not_faithful_by_cause": dict(causes), "n_fixes": len(fixes)})
    if no_fix:
        say(f"\n  NEXT: pairs with the cause English question and no fix_en yet: {len(no_fix)}. "
            "The two post-editors agree on the corrected English, write it in fix_en of the "
            "filled comparison sheet, and you run result again.")
    elif fixes:
        say(f"\n  NEXT: corrections agreed: {len(fixes)}. To put them into the final file, run: merge "
            "--scope all --overwrite-final (add --accept-review N if it asks).")
    return 0 if passed else 2


# --------------------------------------------------------------------------
# Command: status
# --------------------------------------------------------------------------
def cmd_status(ctx):
    say(f"[status] {ctx.root}")

    def have(path):
        return Path(path).exists()

    def filled(folder, stem):
        try:
            find_filled(folder, stem)
            return True
        except Stop:
            return False

    steps = []
    steps.append(("stage 1 files", all(have(p) for p in (ctx.mt_all, ctx.mt_pilot, ctx.registry)),
                  "put the stage 1 files into Drive english/"))
    for scope in ("pilot", "rest"):
        lab = SCOPE_LABEL[scope]
        steps.append((f"{lab}: post-edit sheets made",
                      all(have(ctx.s2_send / f"postedit_{lab}_{pe}.xlsx") for pe in POST_EDITORS),
                      f"make-postedit --scope {scope} --people-rules-confirmed"))
        steps.append((f"{lab}: post-edit sheets filled",
                      all(filled(ctx.s2_filled, f"postedit_{lab}_{pe}") for pe in POST_EDITORS),
                      f"the post-editors fill their sheets; put them into {ctx.s2_filled}"))
        steps.append((f"{lab}: cross-check sheets made",
                      all(have(ctx.s2_send / f"crosscheck_{lab}_{pe}.xlsx") for pe in POST_EDITORS),
                      f"make-crosscheck --scope {scope}"))
        steps.append((f"{lab}: cross-check sheets filled",
                      all(filled(ctx.s2_filled, f"crosscheck_{lab}_{pe}") for pe in POST_EDITORS),
                      f"the post-editors fill the cross-check sheets; put them into {ctx.s2_filled}"))
        if scope == "pilot":
            steps.append(("PILOT: final file", have(ctx.final_pilot), "merge --scope pilot"))
    steps.append(("ALL: final file", have(ctx.final_all), "merge --scope all"))
    rnd = 1
    while have(ctx.s3(rnd + 1)):
        rnd += 1
    out = ctx.s3(rnd)
    opt = f" --round {rnd}" if rnd != 1 else ""
    steps.append((f"round {rnd}: back-translation sheets made",
                  have(out / f"backtrans_key_R{rnd}.csv"),
                  f"make-backtrans{opt} --people-rules-confirmed"))
    steps.append((f"round {rnd}: back-translation sheets filled",
                  all(filled(out / "filled", f"backtrans_R{rnd}_{bt}") for bt in TRANSLATORS),
                  f"the back-translators fill their sheets; put them into {out / 'filled'}"))
    steps.append((f"round {rnd}: comparison sheets made", have(out / f"compare_key_R{rnd}.csv"),
                  f"make-compare{opt}"))
    steps.append((f"round {rnd}: comparison sheets filled",
                  all(filled(out / "filled", f"compare_R{rnd}_{pe}") for pe in POST_EDITORS),
                  f"the post-editors fill the comparison sheets; put them into {out / 'filled'}"))
    steps.append((f"round {rnd}: result", have(out / f"stage3b_result_R{rnd}.json"),
                  f"result{opt}"))
    nxt = None
    for name, done, todo in steps:
        say(f"  {'done' if done else '....'}  {name}")
        if not done and nxt is None:
            nxt = todo
    if nxt is None:
        with open(out / f"stage3b_result_R{rnd}.json", "r", encoding="utf-8") as f:
            res = json.load(f)
        say(f"  round {rnd}: faithful {res['faithful']} of {res['n']} = "
            f"{res['percent_faithful']:.2f}%, Wilson 95% CI {res['wilson95_low']:.2f}% to "
            f"{res['wilson95_high']:.2f}%, {'PASS' if res['passed'] else 'FAIL'}")
        final = {r["string_id"]: r[ENGLISH_COLUMN] for r in read_csv(ctx.final_all)}
        amend_path = ctx.amend_dir / f"amendments_stage3b_R{rnd}.csv"
        waiting = [r for r in (read_csv(amend_path) if amend_path.exists() else [])
                   if final.get(r["string_id"]) != clean_en(r["question_en_new"])]
        if waiting:
            nxt = (f"corrections of stage 3b that are not in the final file yet: {len(waiting)}. Run "
                   "merge --scope all --overwrite-final")
        elif not res["passed"]:
            nxt = ("stage 3b did not pass. Design 4.6: fix the English (the post-editors, "
                   "with files in amendments/), run merge --scope all --overwrite-final, then "
                   f"make-backtrans --round {rnd + 1} --people-rules-confirmed")
        else:
            nxt = ("nothing: stages 2, 3 and 3b are complete. Gate B also needs the leak "
                   "scan of stage 4, after Gate A")
    say(f"\n  NEXT: {nxt}")
    return 0


# --------------------------------------------------------------------------
# Command: dryrun (made-up people; its own folder; nothing real is touched)
# --------------------------------------------------------------------------
def _fill(src, dst, tab, fn):
    """Copy a sheet from to_send/ to filled/ and let fn fill each row. fn gets
    the row as a dict and returns the cells to set."""
    openpyxl = _openpyxl()
    wb = openpyxl.load_workbook(src)
    ws = wb[tab]
    header = [_cell_text(c.value) for c in ws[1]]
    for i in range(2, ws.max_row + 1):
        row = {h: _cell_text(ws.cell(row=i, column=j).value) for j, h in enumerate(header, 1)}
        if not any(row.values()):
            continue
        for name, value in (fn(i - 1, row) or {}).items():
            ws.cell(row=i, column=header.index(name) + 1, value=value)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    wb.save(dst)


def _pick(text, mod):
    return int(sha256_hex("dryrun|" + text)[:8], 16) % mod == 0


def _synthetic_inputs(ctx, n=100, n_pilot=20):
    """Made-up strings: single Bangla grammar words around invented, numbered
    nouns. They are not sentences and hold no dataset text, but they reach
    every check."""
    how_many, visible, exists = nfc("কয়টি"), nfc("দেখা যাচ্ছে"), nfc("আছে")
    thing, image = nfc("জিনিসটি"), nfc("ছবি")
    rows = []

    def add(qa_type, bn, en):
        rows.append((qa_type, bn, en))

    for i in range(1, 41):
        noun = f"নমুনা{(i + 1) // 2 if i <= 8 else i}"      # the first names occur twice
        en_noun = f"sample{(i + 1) // 2 if i <= 8 else i} items"
        if i <= 8 and i % 2 == 0:
            add("count", f"পরীক্ষায় {how_many} {noun} {exists}?",
                f"How many {en_noun} are in the test?")
        else:
            add("count", f"পরীক্ষায় {how_many} {noun} {visible}?",
                f"How many {en_noun} are visible in the test?")
    for i in range(1, 46):
        add("object", f"পরীক্ষা{i} {thing}?", f"What is the thing of test {i}?")
    for i in range(1, 6):
        add("obj_name", f"পরীক্ষা{i} {image} কোনটি?", f"Which one is test image {i}?")
    for i in range(1, 7):
        add("obj_property", f"পরীক্ষা{i} রং?", f"What is the color of test {i}?")
    add("scene_id", "পরীক্ষা ঘর কোনটি?", "Which one is the test room?")
    add("scene_id", "পরীক্ষা ঘর কোনটী?", "Which one is the test room?")       # a spelling variant
    add("scene_id", "পরীক্ষা ঘর এটি?", "Is this the test room?")
    add("scene_id", "পরীক্ষা ঘর ওটি?", "Is that the test room?")
    rows = rows[:n]
    order = {"scene_id": 0, "obj_property": 1, "count": 2, "obj_name": 3, "object": 4}
    per_type = Counter()
    recs = []
    for qa_type, bn, en in rows:
        per_type[qa_type] += 1
        recs.append({"qa_type": qa_type, "bn": bn, "en": en, "k": per_type[qa_type]})
    quota = {"scene_id": 2, "obj_property": 2, "count": 6, "obj_name": 2, "object": 8}
    for r in recs:
        r["pilot"] = r["k"] <= quota[r["qa_type"]]
    assert sum(r["pilot"] for r in recs) == n_pilot
    recs.sort(key=lambda r: (not r["pilot"], order[r["qa_type"]], normalise(r["bn"])))
    mt, reg = [], []
    for seq, r in enumerate(recs, 1):
        norm = normalise(r["bn"])
        sid = "q_" + sha256_hex(norm)[:12]
        pilot = "yes" if r["pilot"] else "no"
        # a few strings are used by many items, as in the real data
        n_items = 40 if r["k"] == 1 else 12 if r["k"] == quota[r["qa_type"]] + 1 else 1
        common = [seq, sid, pilot, "no", r["qa_type"], "DRYRUN", n_items]
        mt.append(common + [r["bn"], r["en"], "ok", "", 1, norm, "made-up", "dry", "", ""])
        reg.append(common + [1, "", r["bn"], norm])
    mt_header = ["seq", "string_id", "in_pilot", "in_pclust", "qa_types", "probes", "n_items",
                 "question_bn", "mt_en", "mt_status", "qc_flags", "n_surface_forms",
                 "question_bn_norm", "model_id", "prompt_version", "call_id", "translated_utc"]
    write_csv(ctx.mt_all, mt_header, mt)
    write_csv(ctx.mt_pilot, mt_header, [r for r in mt if r[2] == "yes"])
    write_csv(ctx.registry, ["seq", "string_id", "in_pilot", "in_pclust", "qa_types", "probes",
                             "n_items", "n_surface_forms", "example_qa_id", "question_bn",
                             "question_bn_norm"], reg)


def cmd_dryrun(english_dir, synthetic, verbose=False, keep=False):
    """Run every command with made-up people on a copy of the inputs."""
    last = []                 # the output of the command that ran last
    try:
        return _dryrun(english_dir, synthetic, verbose, keep, last)
    except Exception:
        say("\n  The dry run stopped. Output of the last command:")
        for line in last:
            say(line)
        raise


def _dryrun(english_dir, synthetic, verbose, keep, last):
    if synthetic:           # needs no Drive: a temporary folder on the machine
        base = Path(tempfile.mkdtemp(prefix="asset2_stage23_dryrun_"))
    else:
        base = Path(english_dir) / "stage23_dryrun"
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True)
    say("=" * 74)
    say(f"ASSET-2 stages 2, 3, 3b: DRY RUN with made-up people   script {SCRIPT_VERSION}")
    say(f"inputs: {'100 made-up strings' if synthetic else 'the real stage 1 files (copied)'}")
    say(f"folder: {base}  (its own folder; no real file is changed)")
    say("=" * 74)
    ctx = Ctx(base, expect_counts=not synthetic, sample_size=30 if synthetic else SAMPLE_SIZE)
    if synthetic:
        _synthetic_inputs(ctx)
    else:
        real = Ctx(english_dir)
        load_strings(real)                      # the real files must be sound
        for p in (real.mt_all, real.mt_pilot, real.registry):
            shutil.copyfile(p, base / p.name)
    checks = []

    def check(label, ok, detail=""):
        checks.append((label, bool(ok), detail))
        say(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f": {detail}" if detail else ""))
        if not ok and not verbose:
            say("        output of the command that ran last:")
            for line in last:
                say(line)

    def expect_stop(kind, fn, code=None, needle=None):
        """fn must stop with the given kind (and error code, or words in the message)."""
        try:
            run(fn)
        except kind as exc:
            if code is not None and not any(code in e[-1] for e in getattr(exc, "errors", [])):
                raise
            if needle is not None and needle not in str(exc):
                raise
            return exc
        return None

    depth = [0]

    def run(fn):
        """Run one command. Its output is shown only with --verbose or on a failure."""
        global _PREFIX, _CAPTURE
        if depth[0] == 0:
            _PREFIX = "      | "
            _CAPTURE = None if verbose else []
        depth[0] += 1
        try:
            return fn()
        finally:
            depth[0] -= 1
            if depth[0] == 0:
                if _CAPTURE is not None:
                    last[:] = _CAPTURE
                _PREFIX, _CAPTURE = "", None

    strings = run(lambda: load_strings(ctx))
    by_id = {s["string_id"]: s for s in strings}
    british = next(iter(WRONG_SPELLING))

    def sim_edit(text):
        return text[:-1] + " DRYRUN?"

    def fill_postedit(scope, faults):
        for pe in POST_EDITORS:
            name = f"postedit_{SCOPE_LABEL[scope]}_{pe}.xlsx"

            def fn(n, row, pe=pe):
                mt = row["first_pass_en"]
                if "no_decision" in faults and pe == POST_EDITORS[0] and n == 1:
                    return {}
                if "spelling" in faults and pe == POST_EDITORS[1] and n == 1:
                    return {"decision": "edit", "edited_en": mt[:-1] + f" {british}?"}
                if _pick(mt.lower(), 5):
                    return {"decision": "edit", "edited_en": sim_edit(mt), "note": "dry run"}
                return {"decision": "keep"}
            _fill(ctx.s2_send / name, ctx.s2_filled / name, "postedit", fn)

    def fill_crosscheck(scope, faults):
        for pe in POST_EDITORS:
            name = f"crosscheck_{SCOPE_LABEL[scope]}_{pe}.xlsx"

            def fn(n, row, pe=pe):
                text = row["english_to_check"]
                if n == 1:
                    new = text[:-1] + " CHECKED?"
                    return {"verdict": "change", "proposed_en": new, "reason": "dry run",
                            "agreed_en": "" if "no_agreed" in faults else new}
                if n == 2:
                    return {"verdict": "change", "proposed_en": text[:-1] + " MAYBE?",
                            "agreed_en": text}
                return {"verdict": "ok"}
            _fill(ctx.s2_send / name, ctx.s2_filled / name, "crosscheck", fn)

    def accept_and_run(make):
        """Run a command; if review rows wait, run it again with their number."""
        try:
            run(lambda: make(0))
            return 0
        except ReviewPending as exc:
            run(lambda: make(exc.n))
            return exc.n

    say("\n[1] PILOT round")
    run(lambda: cmd_make_postedit(ctx, "pilot", True))
    check("post-edit sheets refused without --people-rules-confirmed",
          expect_stop(Stop, lambda: cmd_make_postedit(ctx, "pilot", False),
                      needle="--people-rules-confirmed") is not None)
    fill_postedit("pilot", {"no_decision", "spelling"})
    exc = expect_stop(SheetErrors, lambda: cmd_make_crosscheck(ctx, "pilot"))
    codes = " ".join(e[-1][:3] for e in exc.errors) if exc else ""
    check("an empty decision is caught (E03)", "E03" in codes)
    check("a spelling of the other variety is caught (E07)", "E07" in codes)
    fill_postedit("pilot", set())
    n_rev_p1 = accept_and_run(lambda n: cmd_make_crosscheck(ctx, "pilot", n))
    fill_crosscheck("pilot", {"no_agreed"})
    exc = expect_stop(SheetErrors, lambda: cmd_merge(ctx, "pilot"), "E14")
    check("a change without an agreed text is caught (E14)", exc is not None)
    fill_crosscheck("pilot", set())
    n_rev_p2 = accept_and_run(lambda n: cmd_merge(ctx, "pilot", n))
    pil = read_csv(ctx.final_pilot)
    n_pilot = sum(1 for s in strings if s["in_pilot"])
    check("PILOT final file", len(pil) == n_pilot and all(c in pil[0] for c in
          ("string_id", "question_bn_norm", ENGLISH_COLUMN)), f"{len(pil)} rows")
    check("a second merge changes nothing", run(lambda: cmd_merge(ctx, "pilot", n_rev_p2)) == 0
          and read_csv(ctx.final_pilot) == pil)

    say("\n[2] REST round and the final file of all strings")
    run(lambda: cmd_make_postedit(ctx, "rest", True))
    fill_postedit("rest", set())
    n_rev_r1 = accept_and_run(lambda n: cmd_make_crosscheck(ctx, "rest", n))
    fill_crosscheck("rest", set())
    n_rev_all = accept_and_run(lambda n: cmd_merge(ctx, "all", n))
    final = read_csv(ctx.final_all)
    check("final file of all strings", len(final) == len(strings), f"{len(final)} rows")
    check("review rows were refused until accepted",
          max(n_rev_p1, n_rev_p2, n_rev_r1, n_rev_all) > 0,
          f"pilot {n_rev_p1} and {n_rev_p2}, rest {n_rev_r1}, all {n_rev_all}")
    join = {r["question_bn_norm"]: r for r in final}
    check("join rule (D4): normalise(question_bn) finds one English question for every string",
          len(join) == len(strings) and all(
              normalise(s["question_bn"]) in join and not english_problems(
                  join[normalise(s["question_bn"])][ENGLISH_COLUMN]) for s in strings))
    assign = load_assignment(ctx)
    halves = Counter(r["post_editor"] for r in assign.values())
    n_cross = sum(1 for r in assign.values() if r["cross_checker"])
    n_random = sum(1 for r in assign.values() if r["cross_basis"] == "random")
    want_random = sum(ceil_share(sum(1 for r in assign.values()
                                     if r["scope"] == sc and r["post_editor"] == pe))
                      for sc in ("pilot", "rest") for pe in POST_EDITORS)
    high = {s["string_id"] for s in strings if s["n_items"] >= HIGH_USE_MIN_ITEMS}
    check("two halves", sum(halves.values()) == len(strings)
          and abs(halves[POST_EDITORS[0]] - halves[POST_EDITORS[1]]) <= 4,
          ", ".join(f"{pe} {halves[pe]}" for pe in POST_EDITORS))
    check("cross-check by the other post-editor: a random 10% of each half, and every string "
          f"used by {HIGH_USE_MIN_ITEMS} or more items",
          n_random == want_random and all(assign[sid]["cross_checker"] for sid in high)
          and n_cross == n_random + sum(1 for sid in high if assign[sid]["cross_basis"] == "high_use")
          and all(r["cross_checker"] in ("", other_editor(r["post_editor"]))
                  for r in assign.values()),
          f"{n_cross} strings: random {n_random}, high use {n_cross - n_random}")
    sheet_sizes = Counter((r["scope"], r["cross_checker"]) for r in assign.values()
                          if r["cross_checker"])
    got = Counter(r["cross_check"] for r in final if r["cross_check"])
    # the made-up checker changes row 1 and withdraws the change of row 2
    check("cross-check outcomes are in the final file",
          got["changed"] == sum(1 for n in sheet_sizes.values() if n >= 1)
          and got["kept_after_discussion"] == sum(1 for n in sheet_sizes.values() if n >= 2)
          and sum(got.values()) == n_cross,
          f"ok {got['ok']}, changed {got['changed']}, kept after discussion "
          f"{got['kept_after_discussion']}")

    say("\n[3] Stage 3: draw and back-translation")
    check("back-translation sheets refused without --people-rules-confirmed",
          expect_stop(Stop, lambda: cmd_make_backtrans(ctx, 1, False),
                      needle="--people-rules-confirmed") is not None)
    run(lambda: cmd_make_backtrans(ctx, 1, True))
    out = ctx.s3(1)
    key = read_csv(out / "backtrans_key_R1.csv")
    per_bt = Counter(r["translator"] for r in key)
    check("draw", len(key) == ctx.sample_size and len({r["string_id"] for r in key}) == len(key)
          and all(per_bt[bt] == ctx.sample_size // 2 for bt in TRANSLATORS),
          f"{len(key)} strings, " + ", ".join(f"{bt} {per_bt[bt]}" for bt in TRANSLATORS))
    openpyxl = _openpyxl()
    leak = False
    for bt in TRANSLATORS:
        wb = openpyxl.load_workbook(out / "to_send" / f"backtrans_R1_{bt}.xlsx")
        for ws in wb.worksheets:
            if ws.title == "READ ME":
                continue
            for line in ws.iter_rows(values_only=True):
                leak = leak or any(has_bangla(_cell_text(c)) or _cell_text(c).startswith("q_")
                                   for c in line)
    check("the back-translators' work tab holds no Bangla and no string_id", not leak)
    item_sid = {r["item"]: r["string_id"] for r in key}
    # the made-up translators change the first row of each sheet and about 1 row in 40
    wrong = {r["item"] for r in key if r["item"].endswith("-001") or _pick(r["item"], 40)}

    def fill_bt(faults):
        for bt in TRANSLATORS:
            name = f"backtrans_R1_{bt}.xlsx"

            def fn(n, row, bt=bt):
                if "empty" in faults and bt == TRANSLATORS[0] and n == 1:
                    return {}
                text = by_id[item_sid[row["item"]]]["question_bn"]
                return {"back_translation_bn": (nfc("অন্য ") + text) if row["item"] in wrong else text}
            _fill(out / "to_send" / name, out / "filled" / name, "backtranslation", fn)

    fill_bt({"empty"})
    check("an empty back-translation is caught (E33)",
          expect_stop(SheetErrors, lambda: cmd_make_compare(ctx, 1), "E33") is not None)
    fill_bt(set())
    run(lambda: cmd_make_compare(ctx, 1))
    ckey = read_csv(out / "compare_key_R1.csv")
    check("each pair is judged by the post-editor who did not post-edit the string",
          len(ckey) == ctx.sample_size and all(
              r["judge"] == other_editor(assign[r["string_id"]]["post_editor"]) for r in ckey))
    check("identical pairs are marked", sum(1 for r in ckey if r["same_text"]) == len(key) - len(wrong),
          f"{len(key) - len(wrong)} identical, {len(wrong)} changed by the made-up translators")

    say("\n[4] Stage 3b: comparison and result")
    first_bad = {}
    for r in ckey:
        if not r["same_text"]:
            first_bad.setdefault(r["judge"], r["pair"])

    def fill_compare(faults):
        for pe in POST_EDITORS:
            name = f"compare_R1_{pe}.xlsx"
            if not (out / "to_send" / name).exists():
                continue

            def fn(n, row, pe=pe):
                if "no_verdict" in faults and n == 1:
                    return {}
                if row["same_text"]:
                    return {"verdict": "faithful"}
                if row["pair"] == first_bad.get(pe):
                    return {"verdict": "not faithful", "cause": "English question",
                            "what_differs": "dry run",
                            "fix_en": row["question_en"][:-1] + " FIXED?"}
                return {"verdict": "not faithful", "cause": "back-translation",
                        "what_differs": "dry run"}
            _fill(out / "to_send" / name, out / "filled" / name, "compare", fn)

    fill_compare({"no_verdict"})
    check("a missing verdict is caught (E42)",
          expect_stop(SheetErrors, lambda: cmd_result(ctx, 1), "E42") is not None)
    fill_compare(set())
    code = run(lambda: cmd_result(ctx, 1))
    with open(out / "stage3b_result_R1.json", "r", encoding="utf-8") as f:
        res = json.load(f)
    k_want = ctx.sample_size - len(wrong)
    lo, hi = wilson(k_want, ctx.sample_size)
    check("result", res["faithful"] == k_want and res["n"] == ctx.sample_size
          and abs(res["wilson95_low"] - 100 * lo) < 1e-3 and abs(res["wilson95_high"] - 100 * hi) < 1e-3
          and res["passed"] == passes_threshold(k_want, ctx.sample_size)
          and code == (0 if res["passed"] else 2),
          f"the made-up verdicts give {res['faithful']} of {res['n']} = "
          f"{res['percent_faithful']:.2f}%, CI {res['wilson95_low']:.2f}% to "
          f"{res['wilson95_high']:.2f}%, which the script reads as "
          f"{'PASS' if res['passed'] else 'FAIL'}")
    n_fix = len(first_bad)

    def merge_all(overwrite):
        return accept_and_run(lambda n: cmd_merge(ctx, "all", n, overwrite))

    check("a changed final file is refused without --overwrite-final",
          n_fix > 0 and expect_stop(Stop, lambda: merge_all(False),
                                    needle="--overwrite-final") is not None)
    merge_all(True)
    final2 = read_csv(ctx.final_all)
    check("stage 3b corrections reach the final file as version 2",
          sum(1 for r in final2 if r["amended"]) == n_fix and final2[0]["final_version"] == "2"
          and (ctx.history / "english_questions_final_v1.csv").exists(),
          f"corrections: {n_fix}")

    say("\n[5] Fixed numbers")
    # Reference values from an independent solution of the score equation.
    ref = {(143, 150): (90.6814, 97.7213), (142, 150): (89.8296, 97.2730),
           (150, 150): (97.5030, 100.0), (0, 150): (0.0, 2.4970), (75, 150): (42.0990, 57.9010)}
    check("Wilson interval against reference values", all(
        abs(100 * wilson(k, n)[0] - a) < 0.0002 and abs(100 * wilson(k, n)[1] - b) < 0.0002
        for (k, n), (a, b) in ref.items()))
    check("threshold: 143 of 150 passes, 142 of 150 fails",
          passes_threshold(143, 150) and not passes_threshold(142, 150))
    a1, _u = split_halves([s for s in strings if s["in_pilot"]], "pilot")
    a2, _u = split_halves(list(reversed([s for s in strings if s["in_pilot"]])), "pilot")
    check("the split does not depend on the order of the file", a1 == a2)
    run(lambda: cmd_status(ctx))

    bad = [label for label, ok, _d in checks if not ok]
    say("")
    if bad:
        say(f"DRY RUN FAILED: {len(bad)} of {len(checks)} checks. Do not start the real work. "
            f"Send this output to the ASSET-2 chat. The files of the run are in {base}.")
        return 1
    if keep:
        say(f"DRY RUN OK: {len(checks)} of {len(checks)} checks passed. Every file in {base} is "
            "made up: look at it, never use it, and delete the folder afterwards.")
    else:
        shutil.rmtree(base, ignore_errors=True)
        say(f"DRY RUN OK: {len(checks)} of {len(checks)} checks passed. The made-up files were "
            "removed again (use --keep to look at them).")
    return 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--english-dir", default=f"{DEFAULT_ROOT}/english",
                        help="Drive folder english/ (stage 1 files in, everything out)")
    common.add_argument("--allow-count-mismatch", action="store_true",
                        help="run although the file does not hold 653 strings, 32 in PILOT. "
                             "Needs a decision row.")
    ap = argparse.ArgumentParser(description="ASSET-2 stages 2, 3 and 3b: post-edit sheets, "
                                 "merge, back-translation, faithfulness (Design 4.6).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    people = ("you checked the people rules that the command prints when the flag is "
              "missing (HT-2, D125)")
    accept = "the number of review rows that the post-editors read and found right as they are"
    force = "replace a sheet that exists and was never given to a person"
    rnd = "1 for the first draw; 2 for a fresh 150 after stage 3b failed"
    p = sub.add_parser("make-postedit", parents=[common], help="stage 2: post-edit sheets")
    p.add_argument("--scope", choices=("pilot", "rest"), required=True)
    p.add_argument("--people-rules-confirmed", action="store_true", help=people)
    p.add_argument("--force", action="store_true", help=force)
    p = sub.add_parser("make-crosscheck", parents=[common],
                       help="stage 2: cross-check sheets (needs the filled post-edit sheets)")
    p.add_argument("--scope", choices=("pilot", "rest"), required=True)
    p.add_argument("--accept-review", type=int, default=0, metavar="N", help=accept)
    p.add_argument("--force", action="store_true", help=force)
    p = sub.add_parser("merge", parents=[common],
                       help="stage 2: check the filled sheets, write the final English file")
    p.add_argument("--scope", choices=("pilot", "all"), required=True)
    p.add_argument("--accept-review", type=int, default=0, metavar="N", help=accept)
    p.add_argument("--overwrite-final", action="store_true",
                   help="replace a final file that exists; the old version is kept in "
                        "stage2/final_history")
    p = sub.add_parser("make-backtrans", parents=[common],
                       help="stage 3: draw 150 strings, back-translation sheets")
    p.add_argument("--round", type=int, default=1, dest="rnd", help=rnd)
    p.add_argument("--people-rules-confirmed", action="store_true", help=people)
    p.add_argument("--force", action="store_true", help=force)
    p = sub.add_parser("make-compare", parents=[common], help="stage 3b: comparison sheets")
    p.add_argument("--round", type=int, default=1, dest="rnd", help=rnd)
    p.add_argument("--force", action="store_true", help=force)
    p = sub.add_parser("result", parents=[common],
                       help="stage 3b: percent faithful, Wilson 95 percent CI")
    p.add_argument("--round", type=int, default=1, dest="rnd", help=rnd)
    sub.add_parser("status", parents=[common], help="what exists and what the next step is")
    p = sub.add_parser("dryrun", parents=[common], help="the whole chain with made-up people")
    p.add_argument("--synthetic", action="store_true",
                   help="100 made-up strings instead of the real stage 1 files")
    p.add_argument("--verbose", action="store_true", help="show the output of every command")
    p.add_argument("--keep", action="store_true",
                   help="keep the made-up files of the dry run, to look at them")
    args = ap.parse_args(argv)

    if args.cmd == "dryrun":
        return cmd_dryrun(args.english_dir, args.synthetic, args.verbose, args.keep)
    say("=" * 74)
    say(f"ASSET-2 stages 2, 3, 3b   script {SCRIPT_VERSION}   command: {args.cmd}")
    say(f"folder: {args.english_dir}")
    say("This folder is closed to the HT-1 pair until gate-A is tagged (D125).")
    say("=" * 74)
    ctx = Ctx(args.english_dir, expect_counts=not args.allow_count_mismatch)
    if getattr(args, "rnd", 1) < 1:
        raise Stop("--round starts at 1.")
    if args.cmd == "make-postedit":
        return cmd_make_postedit(ctx, args.scope, args.people_rules_confirmed, args.force)
    if args.cmd == "make-crosscheck":
        return cmd_make_crosscheck(ctx, args.scope, args.accept_review, args.force)
    if args.cmd == "merge":
        return cmd_merge(ctx, args.scope, args.accept_review, args.overwrite_final)
    if args.cmd == "make-backtrans":
        return cmd_make_backtrans(ctx, args.rnd, args.people_rules_confirmed, args.force)
    if args.cmd == "make-compare":
        return cmd_make_compare(ctx, args.rnd, args.force)
    if args.cmd == "result":
        return cmd_result(ctx, args.rnd)
    return cmd_status(ctx)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Stop as exc:
        _PREFIX, _CAPTURE = "", None
        say(f"\nSTOP: {exc}")
        sys.exit(1)
