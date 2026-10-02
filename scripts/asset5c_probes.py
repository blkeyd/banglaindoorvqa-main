"""
asset5c_probes.py
BanglaIndoorVQA, ASSET-5C: build P-CORE-CLOSED, R-TEST and PILOT
(Design v2.1, sections 3.3 and 3.4; decision D90).

What it builds
  P-CORE-CLOSED  every P-CORE item that passes the closed-answer rule. No sampling.
  R-TEST         every item of the 15 test houses that passes the same rule.
  PILOT          100 different items (D97): 12 per closed type from P-CORE-CLOSED,
                 then 20 P-CLUST items and 20 other P-NAME items. Each stratum takes
                 its lowest SHA-256("42|PILOT|<qa_id>") ranks, the same seeded rank
                 that 5A and 5B use. The naming part skips ids already in the closed
                 part and P-NAME items whose question fails the rule.

The closed-answer rule (Design 3.3). Every string is normalise()d first. An item is
kept only if its QUESTION asks for the answer kind of its type and its GOLD is a
valid answer of that kind:
  obj_name, object  the question does not ask the room, the function, the location,
                    a count or a risk, and the gold is not a bare number
  count             the question has a how-many word, and the gold is one exact
                    number (approximate forms such as "প্রায় ..." and "অনেকগুলো" are out)
  scene_id          the question asks which room it is (not which part of the room),
                    and the gold is one of the four rooms; a spelling within one
                    character of a room name counts as that room
  obj_property      the question asks the colour or the material (risk and other
                    questions are out)
  spatial, obj_function, assistive are never closed-answer.
Anything the rule does not recognise is left out and listed in asset5c_review.csv,
so a person can read every decision. The rule never guesses in favour of keeping.

Nothing is written unless all of these hold:
  the split file matches Design 3.1 (houses, images and QA per split);
  the question and answer text inside P-CORE.json, P-NAME.json and P-CLUST.json is
  the text of the merged dataset (otherwise re-run the 5A and 5B draws first);
  P-CORE-CLOSED has exactly the per-type counts verified in the ASSET-5 chat (D96).
  Seven P-CORE items failed the rule on 2026-10-02 through their own text; for each
  one that passes after a dataset correction, the expected count rises by one;
  PILOT has 100 different items in the planned strata;
  no existing output would change membership (the 5A guard; --force overrides it).

Subcommands
  selftest            rule cases, then an end-to-end run on SYNTHETIC data
  draw --check-only   every check and every table, but writes only the review file
                      and the report. Use it first, and read the tables.
  draw                the real run; writes the three probe files and six id lists

  python asset5c_probes.py draw --data <merged.json> --splits <splits.csv> \
      --probes <Drive probes folder> --ids-out <repo>/probes [--check-only]

This file goes to GitHub, so it holds no dataset item: only short rule patterns,
qa_ids, and test sentences that were invented for the self-test (D93).

Needs normalise.py and asset5a_probes.py (same folder, or ../src for normalise.py).
Never copy normalise() into this file.
"""

import argparse
import contextlib
import datetime
import io
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(os.path.dirname(_HERE), "src"), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from normalise import normalise          # noqa: E402
import asset5a_probes as a5              # noqa: E402

say = a5.say
hkey = a5.hkey
SEED = 42
VERSION = "2026-10-03"   # printed on the first line of every report

NAMING_TYPES = ("obj_name", "object")
CLOSED_TYPES = ["obj_name", "object", "count", "scene_id", "obj_property"]
PILOT_PER_TYPE = 12
PILOT_CLUST = 20
PILOT_OTHER = 20

# ----------------------------------------------------------------------------
# Numbers the run must reproduce. EXPECT is a dict so that selftest can swap it.
# ----------------------------------------------------------------------------
EXPECT = {
    # Design 3.1
    "houses": {"train": 43, "val": 10, "test": 15},
    "images": {"train": 3296, "val": 449, "test": 661},
    "qa": {"train": 13668, "val": 1863, "test": 2745},
    # Design 3.4
    "pcore_n": 2040, "pcore_per_type": 255, "pname_n": 1000, "pclust_n": 492,
    # P-CORE-CLOSED per type, verified on the P-CORE.json of 2026-10-02 in the
    # ASSET-5 chat (decision D96).
    "pcore_closed": {"obj_name": 254, "object": 252, "count": 245,
                     "scene_id": 147, "obj_property": 213},
    # The seven P-CORE items that failed the rule on 2026-10-02 because their own
    # question or gold text does not fit their type. If the dataset text of one of
    # them is corrected so that it passes, the expected count of its type rises by
    # one. Nothing else may move the counts.
    "fixable": {"H34_0042_q1": "obj_name", "H23_0051_q4": "count",
                "H36_0025_q4": "count", "H01_0002_q1": "scene_id",
                "H01_0005_q2": "object", "H01_0010_q2": "object",
                "H13_0101_q2": "object"},
}
# Design v2.1, table in 3.3. Printed beside the result; not used as a check (D96).
DESIGN_TABLE = {"obj_name": 254, "object": 255, "count": 246, "scene_id": 147,
                "obj_property": 213}
RTEST_PLANNING_ESTIMATE = 1480


# ----------------------------------------------------------------------------
# The rule. Every Bangla literal goes through normalise(), because NFC rewrites
# letters such as য়, and a raw literal would then never match.
# ----------------------------------------------------------------------------
def _n(*words):
    return tuple(normalise(w) for w in words)


Q_WHICH_ROOM = _n("কোন ঘর", "কোন ধরনের ঘর")
Q_PART_WORDS = _n("অংশ", "জায়গা", "স্থান", "দিক")
# "which part / place / side", and "what kind of place": the other scene questions.
_Q_PLACE = re.compile(
    normalise("কোন") + r"\s*(?:" + "|".join(Q_PART_WORDS) + ")|"
    + normalise("ধরনের") + r"\s*(?:" + "|".join(_n("জায়গা", "স্থান")) + ")")
Q_FUNCTION = _n("কী কাজে", "কি কাজে")
Q_LOCATION = _n("কোথায়")
# How-many words as whole tokens. The last two are misspellings found in the data.
Q_HOW_MANY = frozenset(_n("কয়টি", "কয়টা", "কয়", "কতটি", "কতটা", "কত", "কতগুলো",
                          "কতগুলি", "কটি", "কটা", "কয়তা", "কউটি"))
STEM_COLOUR = _n("রং", "রঙ")
STEM_MATERIAL = _n("তৈর")
STEM_RISK = _n("ঝুঁকি", "ঝুকি")

ROOMS = _n("শোবার ঘর", "বসার ঘর", "রান্নাঘর", "বাথরুম")

APPROX_PREFIX = _n("প্রায়", "অনেক")
_NUMBER_WORDS = {
    1: ["এক"], 2: ["দুই", "দু"], 3: ["তিন"], 4: ["চার"], 5: ["পাঁচ"], 6: ["ছয়"],
    7: ["সাত"], 8: ["আট"], 9: ["নয়"], 10: ["দশ"], 11: ["এগারো", "এগার"],
    12: ["বারো", "বার"], 13: ["তেরো", "তের"], 14: ["চৌদ্দ", "চোদ্দ"],
    15: ["পনেরো", "পনের"], 16: ["ষোলো", "ষোল"], 17: ["সতেরো", "সতের"],
    18: ["আঠারো", "আঠার"], 19: ["উনিশ", "ঊনিশ"], 20: ["বিশ", "কুড়ি"],
}
_CLASSIFIERS = _n("টি", "টা", "টো", "টে", "খানা", "খানি")
NUMBER_FORMS = {}
for _value, _words in _NUMBER_WORDS.items():
    for _w in _words:
        for _c in ("",) + _CLASSIFIERS:
            NUMBER_FORMS[normalise(_w) + _c] = _value
_DIGITS = re.compile("^([0-9\u09e6-\u09ef]+)(?:" + "|".join(_CLASSIFIERS) + ")?$")
_BN_DIGIT = {chr(0x09e6 + i): str(i) for i in range(10)}
_TOKEN = re.compile(r"[^\s,;:!?\u0964\-()\[\]\"'/]+")

PURPOSE_END = normalise("জন্য")
YES_NO = frozenset(_n("হ্যাঁ", "না", "জি", "জ্বি"))
JOINERS = frozenset(_n("ও", "এবং", "বা", "অথবা"))


def tokens(text):
    return _TOKEN.findall(text)


def asks_which_room(q):
    return any(p in q for p in Q_WHICH_ROOM) and not any(p in q for p in Q_PART_WORDS)


def asks_room_or_place(q):
    """Any scene question: which room, which part of the room, what kind of place."""
    return any(p in q for p in Q_WHICH_ROOM) or bool(_Q_PLACE.search(q))


def asks_function(q):
    return any(p in q for p in Q_FUNCTION)


def asks_location(q):
    return any(p in q for p in Q_LOCATION)


def asks_how_many(q):
    return any(t in Q_HOW_MANY for t in tokens(q))


def _has_stem(q, stems):
    return any(t.startswith(s) for t in tokens(q) for s in stems)


def asks_colour(q):
    return _has_stem(q, STEM_COLOUR)


def asks_material(q):
    return _has_stem(q, STEM_MATERIAL)


def asks_risk(q):
    return _has_stem(q, STEM_RISK)


def parse_count(g):
    """(number, '') for one exact number, else (None, why)."""
    toks = g.split()
    if not toks:
        return None, "empty"
    if any(t.startswith(p) for t in toks for p in APPROX_PREFIX):
        return None, "approximate"
    s = "".join(toks)
    m = _DIGITS.match(s)
    if m:
        return int("".join(_BN_DIGIT.get(ch, ch) for ch in m.group(1))), ""
    if s in NUMBER_FORMS:
        return NUMBER_FORMS[s], ""
    return None, "not a bare exact number"


def _within_one_edit(a, b):
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    i = j = skipped = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            skipped += 1
            j += 1
            if skipped > 1:
                return False
    return True


def room_of(g):
    """(room, is_variant). Spaces are ignored; one character of slack is allowed."""
    s = g.replace(" ", "")
    for r in ROOMS:
        if s == r.replace(" ", ""):
            return r, g != r
    near = [r for r in ROOMS if _within_one_edit(s, r.replace(" ", ""))]
    if len(near) == 1:
        return near[0], True
    return None, False


def decide(qa_type, question, answer):
    """Apply the closed-answer rule to one item.

    Returns a dict: keep, kind, reason, and where they apply gold_count,
    gold_room, variant. kind is name, count, room, colour or material.
    """
    q, g = normalise(question), normalise(answer)
    out = {"keep": False, "kind": None, "reason": "", "q": q, "g": g}
    if qa_type in NAMING_TYPES:
        if asks_room_or_place(q):
            out["reason"] = "question asks the room or a place"
        elif asks_function(q):
            out["reason"] = "question asks the function"
        elif asks_location(q):
            out["reason"] = "question asks the location"
        elif asks_how_many(q):
            out["reason"] = "question asks a count"
        elif asks_risk(q):
            out["reason"] = "question asks about risk"
        elif not g:
            out["reason"] = "gold is empty"
        elif parse_count(g)[0] is not None:
            out["reason"] = "gold is a number"
        else:
            out.update(keep=True, kind="name")
    elif qa_type == "count":
        number, why = parse_count(g)
        if not asks_how_many(q):
            out["reason"] = "question has no how-many word"
        elif number is None:
            out["reason"] = f"gold is {why}"
        else:
            out.update(keep=True, kind="count", gold_count=number)
    elif qa_type == "scene_id":
        room, variant = room_of(g)
        if not asks_which_room(q):
            out["reason"] = "question does not ask which room"
        elif room is None:
            out["reason"] = "gold is not one of the four rooms"
        else:
            out.update(keep=True, kind="room", gold_room=room, variant=variant)
    elif qa_type == "obj_property":
        colour, material = asks_colour(q), asks_material(q)
        if asks_risk(q):
            out["reason"] = "risk question"
        elif colour and material:
            out["reason"] = "other question (colour and material together)"
        elif colour:
            out.update(keep=True, kind="colour")
        elif material:
            out.update(keep=True, kind="material")
        else:
            out["reason"] = "other question"
    else:
        out["reason"] = "type is not closed-answer"
    return out


def flags_for(qa_type, d):
    """Things a person should look at on a KEPT item. They never change membership."""
    q, g, out = d["q"], d["g"], []
    words = g.split()
    if d["kind"] == "name":
        if len(words) >= 4:
            out.append("gold has 4 or more words")
        if words and words[-1] == PURPOSE_END:
            out.append("gold ends with জন্য (a purpose, not a name?)")
        if room_of(g)[0]:
            out.append("gold is a room name")
        if words and words[0] in YES_NO:
            out.append("gold starts with yes or no")
        if g == q:
            out.append("gold copies the question")
        if "," in g or any(w in JOINERS for w in words):
            out.append("gold may name more than one object")
        if qa_type == "obj_name" and (asks_colour(q) or asks_material(q)):
            out.append("obj_name question mentions colour or material")
    elif d["kind"] in ("colour", "material"):
        if words and words[0] in YES_NO:
            out.append("gold starts with yes or no")
        if parse_count(g)[0] is not None:
            out.append("gold is a number")
        if len(words) >= 5:
            out.append("gold has 5 or more words")
    elif d["kind"] == "room" and d.get("variant"):
        out.append(f"spelling accepted as {d['gold_room']}")
    return out


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def stop(msg):
    raise SystemExit(msg)


def load_probe(path, name):
    if not os.path.exists(path):
        stop(f"[load] ERROR: {path} not found. {name}.json must be in the --probes folder.")
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    items = doc["items"]
    ids = [it["qa_id"] for it in items]
    if len(set(ids)) != len(ids):
        stop(f"[load] ERROR: duplicate qa_id inside {path}")
    say(f"[load] {name}: {len(items)} items from {path}")
    return items


def check_probe_text(name, probe_items, by_id):
    """The harness reads question and answer text from the probe files, so that
    text must be the text of the merged dataset."""
    missing = [it["qa_id"] for it in probe_items if it["qa_id"] not in by_id]
    if missing:
        stop(f"[check] ERROR: {len(missing)} {name} items are not in the dataset, e.g. "
             f"{missing[:5]}. Wrong --data file?")
    stale = [it["qa_id"] for it in probe_items
             if it["question_bn"] != by_id[it["qa_id"]]["question_bn"]
             or it["answer_bn"] != by_id[it["qa_id"]]["answer_bn"]]
    if stale:
        stop(f"[check] ERROR: {len(stale)} {name} items carry question or answer text that "
             f"differs from the dataset, e.g. {stale[:5]}. The dataset was edited after the "
             f"probe was drawn. Re-run the 5A and 5B draws (membership will not move, the "
             f"text is refreshed), then run this script again.")
    say(f"[check] {name}: all {len(probe_items)} items carry the dataset's current text")


def load_splits(path, items):
    rows = a5.read_csv(path)
    if not rows or not {"house_id", "split"} <= set(rows[0]):
        stop(f"[split] ERROR: {path} needs columns house_id and split; it has "
             f"{list(rows[0]) if rows else 'no rows'}")
    split = {}
    for r in rows:
        h, s = r["house_id"].strip(), r["split"].strip().lower()
        if h in split:
            stop(f"[split] ERROR: house {h} is listed twice in {path}")
        if s not in EXPECT["houses"]:
            stop(f"[split] ERROR: house {h} has split '{s}'; expected train, val or test")
        split[h] = s
    houses = {it["house_id"] for it in items}
    if set(split) != houses:
        stop(f"[split] ERROR: houses differ between {path} and the dataset: only in the "
             f"split file {sorted(set(split) - houses)[:5]}, only in the data "
             f"{sorted(houses - set(split))[:5]}")
    n_houses = Counter(split.values())
    images = defaultdict(set)
    qa = Counter()
    per_house_images = defaultdict(set)
    for it in items:
        s = split[it["house_id"]]
        images[s].add(it["image_id"])
        per_house_images[it["house_id"]].add(it["image_id"])
        qa[s] += 1
    got = {"houses": dict(n_houses), "images": {s: len(v) for s, v in images.items()},
           "qa": dict(qa)}
    for key in ("houses", "images", "qa"):
        if got[key] != EXPECT[key]:
            stop(f"[split] ERROR: {key} per split are {got[key]}; Design 3.1 says "
                 f"{EXPECT[key]}. Stop and ask.")
    if "n_images" in rows[0]:
        bad = [(r["house_id"], int(r["n_images"]), len(per_house_images[r["house_id"].strip()]))
               for r in rows if int(r["n_images"]) != len(per_house_images[r["house_id"].strip()])]
        if bad:
            stop(f"[split] ERROR: n_images differs from the dataset for {len(bad)} houses, "
                 f"e.g. (house, split file, data) {bad[:5]}")
    say(f"[split] {path}: houses {got['houses']}, images {got['images']}, QA {got['qa']}; "
        f"all match Design 3.1")
    return split


# ----------------------------------------------------------------------------
# Building the three sets
# ----------------------------------------------------------------------------
BASE_KEYS = ["qa_id", "image_id", "house_id", "capture_type", "qa_type", "question_bn",
             "answer_bn", "image_file"]
EXTRA_KEYS = ["split", "closed_kind", "gold_count", "gold_room", "in_pcore_closed",
              "pilot_part", "in_pclust", "name_bn", "cluster_id"]


def closed_item(it, d, split):
    out = {k: it.get(k) for k in BASE_KEYS}
    out["split"] = split[it["house_id"]]
    out["closed_kind"] = d["kind"]
    if "gold_count" in d:
        out["gold_count"] = d["gold_count"]
    if "gold_room" in d:
        out["gold_room"] = d["gold_room"]
    return out


def apply_rule(pool, split):
    """(kept items, list of (item, decision)) for the closed types in pool."""
    kept, decided = [], []
    for it in pool:
        if it["qa_type"] not in CLOSED_TYPES:
            continue
        d = decide(it["qa_type"], it["question_bn"], it["answer_bn"])
        decided.append((it, d))
        if d["keep"]:
            kept.append(closed_item(it, d, split))
    kept.sort(key=lambda r: (r["house_id"], r["qa_id"]))
    return kept, decided


def check_pcore_closed(pcc, pcore):
    per_type = Counter(it["qa_type"] for it in pcore)
    if len(pcore) != EXPECT["pcore_n"] or (EXPECT["pcore_per_type"] is not None and any(
            per_type[t] != EXPECT["pcore_per_type"] for t in a5.TYPES)):
        stop(f"[check] ERROR: P-CORE has {len(pcore)} items, per type {dict(per_type)}; "
             f"expected {EXPECT['pcore_n']} with {EXPECT['pcore_per_type']} per type.")
    got = Counter(it["qa_type"] for it in pcc)
    kept_ids = {it["qa_id"] for it in pcc}
    expected = dict(EXPECT["pcore_closed"])
    now_passing = [q for q in EXPECT["fixable"] if q in kept_ids]
    for q in now_passing:
        expected[EXPECT["fixable"][q]] += 1
    say()
    say("P-CORE-CLOSED by type        Design 3.3   expected now   this run")
    for t in CLOSED_TYPES:
        say(f"  {t:24s} {DESIGN_TABLE.get(t, 0):10d} {expected[t]:14d} {got[t]:10d}")
    say(f"  {'total':24s} {sum(DESIGN_TABLE.values()):10d} {sum(expected.values()):14d} "
        f"{len(pcc):10d}")
    if now_passing:
        say(f"  corrected in the dataset since 2026-10-02 and now kept: {sorted(now_passing)}")
    if any(got[t] != expected[t] for t in CLOSED_TYPES):
        stop(f"[check] STOP: P-CORE-CLOSED per type is {dict(got)}, expected {expected}. "
             f"Either the dataset text changed in a way this script does not know, or the "
             f"wrong files were given. Nothing was written. Bring this printout to the "
             f"ASSET-5 chat.")
    say(f"[check] P-CORE-CLOSED: {len(pcc)} items, per-type counts as verified (D96)")


def draw_pilot(pcc, pname, pclust_ids, split):
    rank = lambda it: hkey(SEED, "PILOT", it["qa_id"])
    closed = []
    for t in CLOSED_TYPES:
        pool = sorted((it for it in pcc if it["qa_type"] == t), key=rank)
        if len(pool) < PILOT_PER_TYPE:
            stop(f"[pilot] ERROR: only {len(pool)} {t} items in P-CORE-CLOSED")
        closed += [dict(it, pilot_part="closed") for it in pool[:PILOT_PER_TYPE]]
    taken = {it["qa_id"] for it in closed}
    naming, skipped = [], []
    for want_clust, n in ((True, PILOT_CLUST), (False, PILOT_OTHER)):
        pool = sorted((it for it in pname if (it["qa_id"] in pclust_ids) == want_clust), key=rank)
        got = []
        for it in pool:
            if len(got) == n:
                break
            if it["qa_id"] in taken:
                skipped.append((it["qa_id"], "already in the closed part"))
                continue
            d = decide(it["qa_type"], it["question_bn"], it["answer_bn"])
            if not d["keep"]:
                skipped.append((it["qa_id"], d["reason"]))
                continue
            row = {k: it.get(k) for k in BASE_KEYS}
            row.update(split=split[it["house_id"]], closed_kind="name", pilot_part="naming",
                       in_pclust=want_clust, name_bn=it.get("name_bn"),
                       cluster_id=it.get("cluster_id") if want_clust else None)
            got.append(row)
        if len(got) < n:
            stop(f"[pilot] ERROR: only {len(got)} P-NAME items for a stratum of {n}")
        naming += got
    pilot = sorted(closed + naming, key=lambda r: (r["pilot_part"], r["house_id"], r["qa_id"]))
    ids = [it["qa_id"] for it in pilot]
    assert len(ids) == len(set(ids)) == len(CLOSED_TYPES) * PILOT_PER_TYPE + PILOT_CLUST + PILOT_OTHER
    assert Counter(it["qa_type"] for it in closed) == {t: PILOT_PER_TYPE for t in CLOSED_TYPES}
    assert sum(1 for it in naming if it["in_pclust"]) == PILOT_CLUST
    return pilot, skipped


# ----------------------------------------------------------------------------
# Review rows and reporting
# ----------------------------------------------------------------------------
REVIEW_COLS = ["block", "scope", "qa_type", "decision", "note", "n_pcore", "n_test",
               "question", "gold", "qa_id"]


def review_rows(dec_pcore, dec_test, pname):
    """Every decision a person should be able to read, as rows for one CSV."""
    rows = []

    def table(block, qa_type, field, key, label, only=None):
        cells = defaultdict(lambda: [0, 0, "", ""])
        for col, decided in ((0, dec_pcore), (1, dec_test)):
            for it, d in decided:
                if it["qa_type"] != qa_type or (only and not only(d)):
                    continue
                c = cells[key(d)]
                c[col] += 1
                c[2], c[3] = label(d), c[3] or it["qa_id"]
        for k, (n1, n2, lab, qid) in sorted(cells.items(), key=lambda kv: (-kv[1][0] - kv[1][1], kv[0])):
            rows.append({"block": block, "scope": "P-CORE and test split", "qa_type": qa_type,
                         "decision": lab, "n_pcore": n1, "n_test": n2,
                         "question": k if field == "question" else "",
                         "gold": k if field == "gold" else "", "qa_id": qid})

    def scene_label(d):
        return "which room" if asks_which_room(d["q"]) else "not a which-room question: out"

    def prop_label(d):
        return d["kind"] if d["keep"] else d["reason"] + ": out"

    def count_label(d):
        number, why = parse_count(d["g"])
        return f"read as {number}" if number is not None else f"{why}: out"

    def room_label(d):
        room, variant = room_of(d["g"])
        if room is None:
            return "not one of the four rooms: out"
        return f"{room} (spelling variant)" if variant else room

    table("scene_id question", "scene_id", "question", lambda d: d["q"], scene_label)
    table("obj_property question", "obj_property", "question", lambda d: d["q"], prop_label)
    table("count gold", "count", "gold", lambda d: d["g"], count_label)
    table("room gold under a which-room question", "scene_id", "gold", lambda d: d["g"],
          room_label, only=lambda d: asks_which_room(d["q"]))

    for scope, decided in (("P-CORE", dec_pcore), ("test split", dec_test)):
        for it, d in decided:
            wrong_question = (it["qa_type"] in NAMING_TYPES and not d["keep"]) or \
                             (it["qa_type"] == "count" and not asks_how_many(d["q"]))
            if wrong_question:
                rows.append({"block": "dropped: question or gold does not fit the type",
                             "scope": scope, "qa_type": it["qa_type"], "decision": "out",
                             "note": d["reason"], "question": it["question_bn"],
                             "gold": it["answer_bn"], "qa_id": it["qa_id"]})
            if d["keep"]:
                for fl in flags_for(it["qa_type"], d):
                    rows.append({"block": "kept, please look", "scope": scope,
                                 "qa_type": it["qa_type"], "decision": "kept", "note": fl,
                                 "question": it["question_bn"], "gold": it["answer_bn"],
                                 "qa_id": it["qa_id"]})
    for it in pname:
        d = decide(it["qa_type"], it["question_bn"], it["answer_bn"])
        if not d["keep"]:
            rows.append({"block": "P-NAME item whose question is not a naming question",
                         "scope": "P-NAME" + (" and P-CLUST" if it.get("in_qualifying_cluster") else ""),
                         "qa_type": it["qa_type"], "decision": "still in P-NAME",
                         "note": d["reason"], "question": it["question_bn"],
                         "gold": it["answer_bn"], "qa_id": it["qa_id"]})
    return rows


def print_review(rows, limit=30):
    blocks = defaultdict(list)
    for r in rows:
        blocks[r["block"]].append(r)
    for block, rs in blocks.items():
        say()
        say(f"--- {block}: {len(rs)} rows ---")
        for r in rs[:limit]:
            if r.get("qa_id") and r["block"].startswith(("dropped", "kept", "P-NAME")):
                say(f"  {r['scope']:22s} {r['qa_id']:12s} {r['qa_type']:12s} {r['note']} | "
                    f"{r['question']} => {r['gold']}")
            else:
                text = r["question"] or r["gold"]
                say(f"  P-CORE {r['n_pcore']:4d}  test {r['n_test']:4d}  {r['decision']:38s} {text}")
        if len(rs) > limit:
            say(f"  ... and {len(rs) - limit} more rows in asset5c_review.csv")


def unique_strings(items, types=None):
    its = [it for it in items if types is None or it["qa_type"] in types]
    return {it["question_bn"] for it in its}, {normalise(it["question_bn"]) for it in its}


def print_string_counts(pcc, rtest, pname, pclust_ids, pilot):
    say()
    say("--- unique question strings (the scope of ASSET-2) ---")
    say("  set                              as written   normalise()d")
    for t in CLOSED_TYPES:
        raw, norm = unique_strings(pcc, [t])
        say(f"  P-CORE-CLOSED {t:18s} {len(raw):10d} {len(norm):13d}")
    raw, norm = unique_strings(pcc)
    say(f"  P-CORE-CLOSED all types          {len(raw):10d} {len(norm):13d}")
    by_type = defaultdict(set)
    for it in pcc:
        by_type[normalise(it["question_bn"])].add(it["qa_type"])
    shared = {q: sorted(ts) for q, ts in by_type.items() if len(ts) > 1}
    say(f"  strings used by two types: {len(shared)}"
        + "".join(f"\n      {q}  {ts}" for q, ts in sorted(shared.items())))
    rraw, rnorm = unique_strings(rtest)
    say(f"  R-TEST all types                 {len(rraw):10d} {len(rnorm):13d}")
    say(f"  R-TEST strings not in P-CORE-CLOSED          {len(rnorm - norm):13d}")
    union = norm | rnorm
    say(f"  P-CORE-CLOSED and R-TEST together            {len(union):13d}")
    _, nnorm = unique_strings(pname)
    _, cnorm = unique_strings([it for it in pname if it["qa_id"] in pclust_ids])
    say(f"  P-NAME strings not in that union             {len(nnorm - union):13d}"
        f"   (of {len(nnorm)}; P-CLUST alone adds {len(cnorm - union)})")
    _, pnorm = unique_strings(pilot)
    say(f"  PILOT strings (translate these first)        {len(pnorm):13d}")


def write_probe(path, probe, items, rule, source):
    doc = {"probe": probe, "n": len(items), "seed": SEED, "rule": rule, "source": source,
           "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "items": [{k: it[k] for k in BASE_KEYS + EXTRA_KEYS if it.get(k) is not None}
                     for it in items]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    say(f"[write] {path} ({len(items)} items)")


def write_ids(folder, name, items):
    path = os.path.join(folder, f"{name}.ids.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(sorted(it["qa_id"] for it in items)) + "\n")
    say(f"[write] {path} ({len(items)} qa_ids, no text)")


class _Tee(io.TextIOBase):
    def __init__(self, stream):
        self.stream, self.buf = stream, io.StringIO()

    def write(self, s):
        self.stream.write(s)
        self.buf.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()


# ----------------------------------------------------------------------------
# draw
# ----------------------------------------------------------------------------
RULE_TEXT = ("closed-answer rule of Design 3.3 as applied by asset5c_probes.py (D96): the "
             "question must ask for the answer kind of its type and the gold must be a valid "
             "answer of that kind")


def run_draw(args):
    say(f"ASSET-5C draw, script version {VERSION}, run "
        f"{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')} UTC")
    say(f"[args] data={args.data} | splits={args.splits} | probes={args.probes} | "
        f"out={args.out or args.probes} | ids-out={args.ids_out}")
    items = a5.load_items(args.data)
    a5.validate_items(items, args.counts)
    by_id = {it["qa_id"]: it for it in items}
    split = load_splits(args.splits, items)

    pcore = load_probe(os.path.join(args.probes, "P-CORE.json"), "P-CORE")
    pname = load_probe(os.path.join(args.probes, "P-NAME.json"), "P-NAME")
    pclust = load_probe(os.path.join(args.probes, "P-CLUST.json"), "P-CLUST")
    for name, probe in (("P-CORE", pcore), ("P-NAME", pname), ("P-CLUST", pclust)):
        check_probe_text(name, probe, by_id)
    pclust_ids = {it["qa_id"] for it in pclust}
    if len(pname) != EXPECT["pname_n"] or len(pclust) != EXPECT["pclust_n"] \
            or not pclust_ids <= {it["qa_id"] for it in pname}:
        stop(f"[check] ERROR: P-NAME has {len(pname)} items and P-CLUST {len(pclust)}; expected "
             f"{EXPECT['pname_n']} and {EXPECT['pclust_n']}, with P-CLUST inside P-NAME.")

    # P-CORE-CLOSED: the rule on the P-CORE members, text taken from the dataset.
    pcc, dec_pcore = apply_rule([by_id[it["qa_id"]] for it in pcore], split)
    check_pcore_closed(pcc, pcore)

    # R-TEST: the same rule on every item of the test houses.
    test_pool = [it for it in items if split[it["house_id"]] == "test"]
    rtest, dec_test = apply_rule(test_pool, split)
    pcc_ids = {it["qa_id"] for it in pcc}
    for it in rtest:
        it["in_pcore_closed"] = it["qa_id"] in pcc_ids
    expected_overlap = {it["qa_id"] for it in pcc if it["split"] == "test"}
    assert {it["qa_id"] for it in rtest if it["in_pcore_closed"]} == expected_overlap, \
        "a P-CORE-CLOSED test-house item is missing from R-TEST"
    if len({it["house_id"] for it in rtest}) != EXPECT["houses"]["test"]:
        stop("[check] ERROR: R-TEST does not cover every test house. Stop and ask.")

    pilot, skipped = draw_pilot(pcc, pname, pclust_ids, split)

    rows = review_rows(dec_pcore, dec_test, pname)
    print_review(rows)

    out = args.out or args.probes
    os.makedirs(out, exist_ok=True)
    targets = [("P-CORE-CLOSED", pcc, "P-CORE filtered by the rule; no sampling"),
               ("R-TEST", rtest, "every item of the 15 test houses filtered by the rule"),
               ("PILOT", pilot, f"{PILOT_PER_TYPE} per closed type from P-CORE-CLOSED, then "
                                f"{PILOT_CLUST} P-CLUST and {PILOT_OTHER} other P-NAME items; "
                                f"lowest SHA-256('{SEED}|PILOT|<qa_id>') in each stratum; "
                                f"100 different items")]
    check_only = getattr(args, "check_only", False)
    if not check_only:
        for name, its, _ in targets:
            a5.guard_frozen(os.path.join(out, f"{name}.json"), its, args.force)

    say()
    review_path = os.path.join(out, "asset5c_review.csv")
    a5.write_csv(review_path, rows, REVIEW_COLS)
    say(f"[write] {review_path} ({len(rows)} rows)")
    if check_only:
        say("[check-only] no probe file and no id list was written")
    else:
        for name, its, source in targets:
            write_probe(os.path.join(out, f"{name}.json"), name, its, RULE_TEXT, source)
        if args.ids_out:
            os.makedirs(args.ids_out, exist_ok=True)
            for name, its in (("P-CORE", pcore), ("P-CORE-CLOSED", pcc), ("R-TEST", rtest),
                              ("PILOT", pilot), ("P-NAME", pname), ("P-CLUST", pclust)):
                write_ids(args.ids_out, name, its)
        else:
            say("[write] --ids-out not given: the qa_id lists for the repo were NOT written")

    print_string_counts(pcc, rtest, pname, pclust_ids, pilot)

    bad_pname = [r for r in rows if r["block"].startswith("P-NAME item")]
    n_flags = sum(1 for r in rows if r["block"] == "kept, please look")
    n_wrong = Counter(r["scope"] for r in rows if r["block"].startswith("dropped"))
    cap = lambda its: dict(sorted(Counter(it["capture_type"] for it in its).items()))
    kinds = lambda its: dict(sorted(Counter(it["closed_kind"] for it in its).items()))
    say()
    say("=" * 78)
    say(f"P-CORE-CLOSED  {len(pcc)} items | {len({i['house_id'] for i in pcc})} houses | "
        f"{len({i['image_id'] for i in pcc})} images | items by capture {cap(pcc)}")
    say(f"               by type {dict(Counter(i['qa_type'] for i in pcc))}")
    say(f"               by kind {kinds(pcc)}")
    say(f"R-TEST         {len(rtest)} items | {len({i['house_id'] for i in rtest})} houses | "
        f"{len({i['image_id'] for i in rtest})} images | planning estimate about "
        f"{RTEST_PLANNING_ESTIMATE}")
    say(f"               by type {dict(Counter(i['qa_type'] for i in rtest))}")
    say(f"               by kind {kinds(rtest)}")
    say(f"               already in P-CORE-CLOSED: {sum(i['in_pcore_closed'] for i in rtest)}")
    say(f"PILOT          {len(pilot)} different items | closed "
        f"{dict(Counter(i['qa_type'] for i in pilot if i['pilot_part'] == 'closed'))}")
    say(f"               naming: {PILOT_CLUST} in P-CLUST, {PILOT_OTHER} not | "
        f"{len({i['house_id'] for i in pilot})} houses | skipped while drawing: {len(skipped)}")
    say(f"               first id {min(i['qa_id'] for i in pilot)}, last id "
        f"{max(i['qa_id'] for i in pilot)}")
    for qid, why in skipped:
        say(f"               skipped {qid}: {why}")
    say(f"To read        dropped for a wrong question or gold: {dict(n_wrong)}; kept items "
        f"flagged: {n_flags}")
    if bad_pname:
        say(f"WARNING        P-NAME holds {len(bad_pname)} items whose question is not a naming "
            f"question: {[r['qa_id'] for r in bad_pname]}.")
        say("               They stay in P-NAME until their question text is corrected in the")
        say("               dataset and 5A, 5B and 5C are run again. PILOT never draws them.")
    if check_only:
        say("CHECK ONLY     the sets above were computed but NOT written. Read the tables, then")
        say("               run draw again without --check-only.")
    say("=" * 78)
    return out


def cmd_draw(args):
    tee = _Tee(sys.stdout)
    with contextlib.redirect_stdout(tee):
        out = run_draw(args)
    path = os.path.join(out, "asset5c_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(tee.buf.getvalue())
    say(f"[write] {path} (this printout; it holds question text, so it stays in Drive)")


# ----------------------------------------------------------------------------
# selftest: rule cases, then an end-to-end run on SYNTHETIC data
# ----------------------------------------------------------------------------
# Every sentence below is INVENTED for the test. None is a dataset item (D93).
Q_NAME = "এই ছবির প্রধান বস্তুটির নাম কী?"
Q_ROOM = "এখানে কোন ঘর দেখানো হয়েছে?"
Q_COUNT = "এখানে কয়টি বস্তু আছে?"
Q_WHERE = "বস্তুটি কোথায় আছে?"
Q_USE = "বস্তুটি কী কাজে লাগে?"
RULE_CASES = [
    # (type, question, gold, keep, kind if kept or a fragment of the reason if not)
    ("obj_name", Q_NAME, "আলমারি।", True, "name"),
    ("obj_name", Q_ROOM, "চেয়ার।", False, "room"),
    ("obj_name", "এখানে কোন ধরনের ঘর আছে?", "গ্লাস।", False, "room"),
    ("obj_name", "ছবিতে ঘরের কোন অংশ আছে?", "টেবিল।", False, "place"),
    ("obj_name", Q_USE, "বেসিন।", False, "function"),
    ("object", "যে আসবাবে শোয়া হয় তার নাম কী?", "বিছানা।", True, "name"),
    ("object", "লাল রঙের কাঠের তৈরি বসার আসবাবটির নাম কী?", "চেয়ার।", True, "name"),
    ("object", "ঘরের কোণায় রাখা আসবাবটির নাম কী?", "আলনা।", True, "name"),
    ("object", "বস্তুটি কি কাজে ব্যবহার হয়?", "হাত মোছার জন্য।", False, "function"),
    ("object", Q_WHERE, "জানালার পাশে।", False, "location"),
    ("object", Q_COUNT, "দুইটি।", False, "count"),
    ("object", "তাকের ওপরের বস্তুটির নাম কী?", "একটি।", False, "number"),
    ("object", "বস্তুটি ব্যবহারে কি ঝুঁকি আছে?", "না।", False, "risk"),
    ("count", Q_COUNT, "একটি।", True, "count"),
    ("count", "এখানে কতটি বস্তু আছে?", "দুটো", True, "count"),
    ("count", "এখানে কয়তা বস্তু আছে?", "পনেরোটি।", True, "count"),
    ("count", "এখানে কউটি বস্তু আছে?", "২টি", True, "count"),
    ("count", "এখানে কয়টা বস্তু আছে?", "3", True, "count"),
    ("count", Q_COUNT, "প্রায় তিনটি", False, "approximate"),
    ("count", Q_COUNT, "অনেকগুলো।", False, "approximate"),
    ("count", Q_COUNT, "দরজা।", False, "not a bare exact number"),
    ("count", Q_COUNT, "দুইটি বালিশ", False, "not a bare exact number"),
    ("count", Q_WHERE, "একটি", False, "how-many"),
    ("count", "এখানে বস্তু আছে?", "পাঁচটি।", False, "how-many"),
    ("scene_id", Q_ROOM, "শোবার ঘর।", True, "room"),
    ("scene_id", "এখানে কোন ধরনের ঘর আছে?", "শোভার ঘর", True, "room"),
    ("scene_id", "এটি বাসার কোন ঘরের ছবি?", "রান্না ঘর।", True, "room"),
    ("scene_id", Q_ROOM, "শোবার", False, "four rooms"),
    ("scene_id", Q_ROOM, "শোবার ঘরের এক কোণ।", False, "four rooms"),
    ("scene_id", Q_ROOM, "বারান্দা।", False, "four rooms"),
    ("scene_id", "এখানে ঘরের কোন অংশ দেখানো হয়েছে?", "রান্নাঘর।", False, "which room"),
    ("scene_id", "এখানে কোন ঘরের অংশ দেখানো হয়েছে?", "শোবার ঘর।", False, "which room"),
    ("scene_id", "এখানে ঘরের কোনঅংশ আছে?", "বাথরুম।", False, "which room"),
    ("scene_id", "যে আসবাবে শোয়া হয় তার নাম কী?", "বিছানা।", False, "which room"),
    ("obj_property", "বস্তুটির রং কেমন?", "সাদা।", True, "colour"),
    ("obj_property", "বস্তুটি কোন রঙের?", "সবুজ-হলুদ", True, "colour"),
    ("obj_property", "বস্তুটি কী দিয়ে তৈরি?", "কাঠ ও কাঁচের।", True, "material"),
    ("obj_property", "বস্তুটি ব্যবহারে কি ঝুকি আছে?", "না।", False, "risk"),
    ("obj_property", "বস্তুটির রং কী এবং এটি কী দিয়ে তৈরি?", "সাদা, কাঠের।", False, "other"),
    ("obj_property", Q_COUNT, "একটি।", False, "other"),
    ("obj_property", Q_WHERE, "দেয়ালের পাশে।", False, "other"),
    ("spatial", Q_WHERE, "টেবিলের পাশে।", False, "not closed-answer"),
]
COUNT_CASES = [("একটি", 1), ("দুটি", 2), ("দুইটা", 2), ("তিনটে", 3), ("ছয়টি", 6), ("১২", 12),
               ("3", 3), ("৪ টি", 4), ("ষোলটি", 16), ("বিশ", 20), ("প্রায় বিশটি", None),
               ("অনেক", None), ("কয়েকটি", None), ("একটি চেয়ার", None), ("", None)]

SYN_NAMES = ["আলমারি", "কেবিনেট", "শোকেস", "চেয়ার", "বালতি", "পর্দা"]
SYN_CLUSTER = {"আলমারি", "কেবিনেট", "শোকেস"}
# (question, gold, kept by the rule). The truth column is written by hand here and
# never computed from the rule, so the end-to-end run really tests the rule.
SYN = {
    "scene_id": [(Q_ROOM, "শোবার ঘর।", True),
                 ("এখানে কোন ধরনের ঘর আছে?", "শোভার ঘর।", True),
                 ("এখানে ঘরের কোন অংশ দেখানো হয়েছে?", "ঘরের এক পাশ।", False),
                 (Q_ROOM, "বারান্দা।", False)],
    "object": [("যে আসবাবে শোয়া হয় তার নাম কী?", "বিছানা।", True),
               ("যে আসবাবে বসা হয় তার নাম কী?", "সোফা।", True),
               (Q_USE, "হাত মোছার জন্য।", False),
               (Q_WHERE, "জানালার পাশে।", False)],
    "count": [("এখানে কয়টি চেয়ার আছে?", "দুইটি।", True),
              ("এখানে কয়টি বালিশ আছে?", "৩টি", True),
              ("এখানে কয়টি পর্দা আছে?", "প্রায় তিনটি", False),
              ("এখানে বাক্স আছে?", "পাঁচটি।", False)],
    "obj_property": [("বস্তুটির রং কেমন?", "সাদা।", True), ("বস্তুটি কী দিয়ে তৈরি?", "কাঠের।", True),
                     ("বস্তুটি ব্যবহারে কি ঝুঁকি আছে?", "না।", False),
                     ("বস্তুটি কোথায় লাগানো?", "দেয়ালে।", False)],
    "spatial": [("চেয়ারটি কোথায় আছে?", "টেবিলের পাশে।", False)],
    "assistive": [("এখানে হাঁটার সময় কোনো সমস্যা হবে কি?", "না, পথ খোলা।", False)],
    "obj_function": [(Q_USE, "বসার জন্য।", False)],
}


def synthetic_dataset():
    """68 SYNTHETIC houses, 5 images each. Returns (records, truth by qa_id)."""
    records, truth = [], {}
    k = 0
    for h in range(1, a5.N_HOUSES + 1):
        house = f"H{h:02d}"
        for n, cap in enumerate(["overview", "group", "closeup", "closeup", "closeup"], start=1):
            img = f"{house}_{n:04d}"
            types = a5.CLOSEUP_TYPES if cap == "closeup" else \
                ["scene_id", "object", "spatial", "count", "assistive"]
            qas = []
            for t in types:
                k += 1
                if t == "obj_name":
                    name = SYN_NAMES[k % len(SYN_NAMES)]
                    bad = k % 41 == 0  # a function question pasted into the obj_name slot
                    q, a, keep = (Q_USE if bad else Q_NAME), name + "।", not bad
                else:
                    q, a, keep = SYN[t][k % len(SYN[t])]
                slot = {"scene_id": 1, "obj_name": 1, "object": 2, "obj_function": 2,
                        "spatial": 3, "obj_property": 3, "count": 4, "assistive": 5}[t]
                qid = f"{img}_q{slot}"
                truth[qid] = keep
                qas.append({"qa_id": qid, "type": t, "question": q, "answer": a})
            records.append({"image_id": img, "house_id": house, "capture_type": cap,
                            "image_file": f"{house}/{img}.jpg", "qa_pairs": qas})
    return records, truth


def cmd_selftest(args):
    global EXPECT
    say("#" * 70)
    say("SELF-TEST. Part 1 checks the rule on fixed strings. Part 2 runs the whole")
    say("script on SYNTHETIC data. No number printed here is a dataset figure.")
    say("#" * 70)
    for t, q, g, keep, frag in RULE_CASES:
        d = decide(t, q, g)
        assert d["keep"] == keep and frag in (d["kind"] if keep else d["reason"]), \
            f"rule case failed: {t} | {q} | {g} -> {d}"
    for g, want in COUNT_CASES:
        assert parse_count(normalise(g))[0] == want, f"count case failed: {g!r}"
    assert room_of(normalise("শোভার ঘর")) == (normalise("শোবার ঘর"), True)
    assert room_of(normalise("শোবার"))[0] is None and room_of(normalise("বারান্দা"))[0] is None
    say(f"[selftest] OK  {len(RULE_CASES)} rule cases, {len(COUNT_CASES)} number cases, room spellings")

    tmp = tempfile.mkdtemp(prefix="asset5c_selftest_")
    saved = EXPECT
    try:
        records, truth = synthetic_dataset()
        data = os.path.join(tmp, "synthetic_merged.json")
        with open(data, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
        with contextlib.redirect_stdout(io.StringIO()):
            items = a5.load_items(data)
        houses = sorted({it["house_id"] for it in items})
        split = {h: "train" if i < 43 else "val" if i < 53 else "test" for i, h in enumerate(houses)}
        sp = os.path.join(tmp, "splits.csv")
        a5.write_csv(sp, [{"house_id": h, "annotator_id": "A0", "split": s, "n_images": 5}
                          for h, s in split.items()],
                     ["house_id", "annotator_id", "split", "n_images"])
        probes = os.path.join(tmp, "probes")
        os.makedirs(probes)
        obj = [it for it in items if it["qa_type"] == "obj_name"]
        pname = [dict(it, name_bn=normalise(it["answer_bn"]),
                      cluster_id="SYN" if normalise(it["answer_bn"]) in SYN_CLUSTER else "",
                      in_qualifying_cluster=normalise(it["answer_bn"]) in SYN_CLUSTER) for it in obj]
        pclust = [it for it in pname if it["in_qualifying_cluster"]]
        for name, its in (("P-CORE", items), ("P-NAME", pname), ("P-CLUST", pclust)):
            with open(os.path.join(probes, f"{name}.json"), "w", encoding="utf-8") as f:
                json.dump({"probe": name, "items": its}, f, ensure_ascii=False)
        EXPECT = dict(saved,
                      images={s: 5 * n for s, n in saved["houses"].items()},
                      qa=dict(Counter(split[it["house_id"]] for it in items)),
                      pcore_n=len(items), pcore_per_type=None,
                      pname_n=len(pname), pclust_n=len(pclust),
                      pcore_closed={t: sum(1 for it in items if it["qa_type"] == t and truth[it["qa_id"]])
                                    for t in CLOSED_TYPES},
                      fixable={})
        ids = os.path.join(tmp, "ids")
        dargs = argparse.Namespace(data=data, splits=sp, probes=probes, out=None, ids_out=ids,
                                   counts=None, force=False, check_only=False)

        def run(a=dargs):
            with contextlib.redirect_stdout(io.StringIO()):
                return run_draw(a)

        say("[selftest] --- check-only, then draw, on synthetic data ---")
        run(argparse.Namespace(**dict(vars(dargs), check_only=True)))
        assert os.path.exists(os.path.join(probes, "asset5c_review.csv"))
        assert not os.path.exists(os.path.join(probes, "PILOT.json")) and not os.path.exists(ids), \
            "check-only wrote a probe file or an id list"
        run()
        load = lambda n: json.load(open(os.path.join(probes, f"{n}.json"), encoding="utf-8"))["items"]
        pcc, rtest, pilot = load("P-CORE-CLOSED"), load("R-TEST"), load("PILOT")
        want = {q for q, keep in truth.items() if keep}
        assert {i["qa_id"] for i in pcc} == want, "P-CORE-CLOSED differs from the hand-written truth"
        assert {i["qa_id"] for i in rtest} == {q for q in want if split[q[:3]] == "test"}, "R-TEST"
        assert all(i["split"] == "test" and i["in_pcore_closed"] for i in rtest)
        assert len(pilot) == 100 == len({i["qa_id"] for i in pilot})
        closed = [i for i in pilot if i["pilot_part"] == "closed"]
        naming = [i for i in pilot if i["pilot_part"] == "naming"]
        assert Counter(i["qa_type"] for i in closed) == {t: 12 for t in CLOSED_TYPES}
        assert sum(i["in_pclust"] for i in naming) == 20 and len(naming) == 40
        assert all(truth[i["qa_id"]] for i in pilot), "PILOT drew an item the rule rejects"
        assert all(i["gold_count"] in (2, 3) for i in pcc if i["closed_kind"] == "count")
        assert {i["gold_room"] for i in pcc if i["closed_kind"] == "room"} == {normalise("শোবার ঘর")}
        for name in ("P-CORE", "P-CORE-CLOSED", "R-TEST", "PILOT", "P-NAME", "P-CLUST"):
            lines = open(os.path.join(ids, f"{name}.ids.txt"), encoding="utf-8").read().split()
            assert all(re.fullmatch(r"H\d\d_\d{4}_q\d", x) for x in lines), "id list holds text"
        review = a5.read_csv(os.path.join(probes, "asset5c_review.csv"))
        assert any(r["block"].startswith("P-NAME item") for r in review), "P-NAME check silent"
        say("[selftest] OK  --check-only writes the review file and nothing else")
        say("[selftest] OK  P-CORE-CLOSED and R-TEST equal the hand-written truth; PILOT is")
        say("               100 different items in the planned strata; id lists hold no text")

        say("[selftest] --- draw again: identical ---")
        run()
        assert [i["qa_id"] for i in load("PILOT")] == [i["qa_id"] for i in pilot]
        say("[selftest] OK  a repeat draw reproduces all three sets")

        def must_stop(label, fn):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    fn()
            except SystemExit as e:
                say(f"[selftest] OK  {label:38s} -> {str(e)[:52]}...")
                return
            raise AssertionError(f"did not stop: {label}")

        good = EXPECT
        EXPECT = dict(good, pcore_closed=dict(good["pcore_closed"], count=good["pcore_closed"]["count"] + 1))
        must_stop("P-CORE-CLOSED count differs", run)
        EXPECT = good
        bad_sp = os.path.join(tmp, "bad_splits.csv")
        rows = a5.read_csv(sp)
        rows[0]["split"] = "test"
        a5.write_csv(bad_sp, rows, ["house_id", "annotator_id", "split", "n_images"])
        must_stop("split file differs from Design 3.1",
                  lambda: run(argparse.Namespace(**dict(vars(dargs), splits=bad_sp))))
        stale = json.load(open(os.path.join(probes, "P-NAME.json"), encoding="utf-8"))
        keep_text = stale["items"][0]["question_bn"]
        stale["items"][0]["question_bn"] = keep_text + " (পুরোনো)"
        with open(os.path.join(probes, "P-NAME.json"), "w", encoding="utf-8") as f:
            json.dump(stale, f, ensure_ascii=False)
        must_stop("probe text older than the dataset", run)
        stale["items"][0]["question_bn"] = keep_text
        with open(os.path.join(probes, "P-NAME.json"), "w", encoding="utf-8") as f:
            json.dump(stale, f, ensure_ascii=False)
        doc = json.load(open(os.path.join(probes, "PILOT.json"), encoding="utf-8"))
        doc["items"] = doc["items"][1:]
        with open(os.path.join(probes, "PILOT.json"), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        must_stop("an existing PILOT would change", run)
        EXPECT = dict(good, pcore_n=good["pcore_n"] + 1)
        must_stop("P-CORE is not the expected size", run)
    finally:
        EXPECT = saved
        shutil.rmtree(tmp, ignore_errors=True)
    say()
    say("SELF-TEST PASSED: the rule decides every fixed case as written, the three sets")
    say("come out as planned on synthetic data, and every check stops the run when it should.")


def main():
    ap = argparse.ArgumentParser(description="ASSET-5C: P-CORE-CLOSED, R-TEST and PILOT")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    s = sub.add_parser("draw")
    s.add_argument("--data", required=True, help="the gold-fixed merged dataset JSON")
    s.add_argument("--splits", required=True, help="splits.csv (house_id, split, n_images)")
    s.add_argument("--probes", required=True,
                   help="Drive folder that holds P-CORE.json, P-NAME.json and P-CLUST.json")
    s.add_argument("--out", default=None, help="where to write; default is the --probes folder")
    s.add_argument("--ids-out", dest="ids_out", default=None,
                   help="the repo's probes/ folder, for the qa_id lists (no text)")
    s.add_argument("--counts", default=None,
                   help="optional house_type_counts.csv from ASSET-1, for the per-cell cross-check")
    s.add_argument("--check-only", dest="check_only", action="store_true",
                   help="run every check and print every table, but write only the review "
                        "file and the report")
    s.add_argument("--force", action="store_true",
                   help="replace an existing output whose membership would change (needs a decision row)")
    args = ap.parse_args()
    {"selftest": cmd_selftest, "draw": cmd_draw}[args.cmd](args)


if __name__ == "__main__":
    main()
