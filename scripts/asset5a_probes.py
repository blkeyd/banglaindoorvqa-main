"""
asset5a_probes.py
BanglaIndoorVQA, ASSET-5A: draw P-CORE (2,040 QA) and P-HAZ (600 items),
plus a manifest that records exactly how each was drawn.

It does NOT draw P-NAME or P-CLUST (ASSET-5B, waits on ASSET-4A) or P-JUDGE
(drawn after E1 to E3).

Four subcommands:

  selftest  Needs only house_type_counts.csv. Runs the P-CORE allocation on the
            real counts, then audit, merge and draw end to end on SYNTHETIC
            items. Writes only to a temporary folder, which it deletes.
  audit     Lists every unique assistive and obj_property question string for
            the P-HAZ classification. Writes phaz_question_audit.csv.
  merge     Merges the two raters' copies of the audit file. Writes a draft
            class file with disagreements left blank for the adjudicator.
  draw      Draws P-CORE. Also draws P-HAZ if phaz_question_classes.csv is
            complete. Writes the probe JSONs, allocation table and manifest.

Run order on real data:
  1. selftest
  2. draw        P-CORE now; P-HAZ is skipped until the class file exists
  3. audit       then two raters fill their own copies, blind to each other
  4. merge       then a third person adjudicates and saves the class file
  5. draw again  P-CORE comes out identical; P-HAZ is drawn

Rules implemented (decision rows D47 to D49, proposed 2026-09-18):
  P-CORE   raking to 30 per house and 255 per type; two-level rounding
           (each house's close-up and scene totals first, then the QA types
           inside each track), each level followed by single-unit repair;
           overview/group split per house.
  Sampling every item gets rank SHA-256("42|<probe>|<qa_id>"); each stratum
           takes its lowest ranks. Text never affects which items are drawn.
  P-HAZ    eligible = assistive or obj_property items whose question passes
           the test in phaz_question_classes.csv; min(300, pool) hazard
           obj_property items, the rest assistive; proportional by house.

Imports the ONE normalise() from normalise.py, which must sit in the same
folder. Never copy normalise() into this file.
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import sys
import tempfile
from collections import Counter, defaultdict

from normalise import normalise

# ----------------------------------------------------------------------------
# Fixed design constants. Changing any of these changes the pre-registration.
# ----------------------------------------------------------------------------
SEED = 42
TYPES = ["assistive", "count", "obj_function", "obj_name", "obj_property",
         "object", "scene_id", "spatial"]
SCENE_TYPES = ["assistive", "count", "object", "scene_id", "spatial"]
CLOSEUP_TYPES = ["obj_function", "obj_name", "obj_property"]
CAPTURES = ["overview", "group", "closeup"]
N_HOUSES = 68
PCORE_PER_HOUSE = 30
PCORE_PER_TYPE = 255
PHAZ_SIZE = 600
PHAZ_OBJPROP_CAP = 300
PHAZ_TYPES = ["assistive", "obj_property"]
POSTEDIT_RATE_PER_HOUR = 60  # Guide, ASSET-2 stage 2

# Hint words shown in the audit file. They are NOT the rule. The rule is the
# yes/no test the raters apply (see RATER_TEST). Stems are normalise()d before
# matching, so nukta letters and joiners cannot break the match.
PREFLAG_STEMS_RAW = [
    "ঝুঁকি", "বিপদ", "বিপজ্জনক", "নিরাপ", "সতর্ক", "সাবধান", "আঘাত",
    "দুর্ঘটনা", "ক্ষতি", "বিদ্যুৎস্পৃষ্ট", "পুড়", "পোড়", "ধারালো",
    "পিছল", "পিচ্ছিল", "হোঁচট", "খেয়াল",
]
PREFLAG_STEMS = [normalise(s) for s in PREFLAG_STEMS_RAW]

RATER_TEST = ("If the image showed a danger, would a correct answer to this "
              "question be expected to mention it? Judge from the question "
              "text alone. Write yes or no.")

# ----------------------------------------------------------------------------
# JSON field names. The loader tries each candidate in order. If your merged
# JSON uses a different name, add it to the front of the right tuple.
# ----------------------------------------------------------------------------
RECORD_LIST_KEYS = ("records", "images", "data", "items", "annotations")
IMAGE_ID_KEYS = ("image_id", "id")
HOUSE_KEYS = ("house_id", "house")
CAPTURE_KEYS = ("capture_type", "capture")
QA_LIST_KEYS = ("qa", "qa_pairs", "qas", "questions", "qa_list")
QA_ID_KEYS = ("qa_id", "id")
QA_TYPE_KEYS = ("qa_type", "type", "question_type")
QUESTION_KEYS = ("question", "q", "question_bn")
ANSWER_KEYS = ("answer", "a", "answer_bn")
IMAGE_FILE_KEYS = ("image_file", "image_path", "file_name", "file", "path")

HOUSE_RE = re.compile(r"^(H\d{2})")


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def say(msg=""):
    print(msg, flush=True)


def hkey(*parts):
    """Deterministic rank. Same on every machine and Python version."""
    text = "|".join(str(p) for p in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def template_id(qa_type, question_norm):
    return "T_" + hkey(qa_type, question_norm)[:12]


def preflag_terms(question_norm):
    return [s for s in PREFLAG_STEMS if s in question_norm]


def bounded_lr(targets, total, lo, hi, tiebreak):
    """Largest-remainder rounding with per-cell bounds.

    targets  cell -> real-valued target
    total    the integer the result must sum to
    lo, hi   cell -> smallest and largest allowed integer
    tiebreak cell -> string; equal remainders are broken by the smaller string
    Starts from the floors, then adds (or removes) one unit at a time to the
    cell furthest below (or above) its target. With no bounds binding this is
    exactly the classical largest-remainder method.
    """
    cells = list(targets)
    if sum(lo[c] for c in cells) > total or sum(hi[c] for c in cells) < total:
        raise ValueError(f"bounded_lr: total {total} is outside the bounds "
                         f"[{sum(lo[c] for c in cells)}, {sum(hi[c] for c in cells)}]")
    k = {c: min(max(int(math.floor(targets[c] + 1e-9)), lo[c]), hi[c]) for c in cells}
    diff = total - sum(k.values())
    while diff > 0:
        room = [c for c in cells if k[c] < hi[c]]
        c = min(room, key=lambda c: (-round(targets[c] - k[c], 9), tiebreak[c]))
        k[c] += 1
        diff -= 1
    while diff < 0:
        room = [c for c in cells if k[c] > lo[c]]
        c = min(room, key=lambda c: (-round(k[c] - targets[c], 9), tiebreak[c]))
        k[c] -= 1
        diff += 1
    return k


# ----------------------------------------------------------------------------
# Loading and validation
# ----------------------------------------------------------------------------
def _pick(d, keys):
    for k in keys:
        if k in d:
            return k
    return None


def load_items(path):
    """Flatten the merged dataset JSON into one dict per QA item."""
    say(f"[load] reading {path}")
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)

    records = data
    if isinstance(data, dict):
        key = _pick(data, RECORD_LIST_KEYS)
        if key is None:
            for k, v in data.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    key = k
                    break
        if key is None and data and all(isinstance(v, dict) for v in data.values()):
            records = [dict(v, image_id=v.get("image_id", k)) for k, v in data.items()]
            say("[load] top level is a dict keyed by image id; using its values")
            key = ""
        if key is None:
            raise SystemExit(f"[load] ERROR: top level is a dict with keys "
                             f"{list(data)[:15]} and none holds the record list. "
                             f"Add the right key to RECORD_LIST_KEYS.")
        if key:
            records = data[key]
            say(f"[load] record list found under key '{key}'")
    if not isinstance(records, list) or not records:
        raise SystemExit("[load] ERROR: no records found.")

    first = records[0]
    k_img = _pick(first, IMAGE_ID_KEYS)
    k_house = _pick(first, HOUSE_KEYS)
    k_cap = _pick(first, CAPTURE_KEYS)
    k_qas = _pick(first, QA_LIST_KEYS)
    k_file = _pick(first, IMAGE_FILE_KEYS)
    missing = [n for n, k in (("image id", k_img), ("capture type", k_cap),
                              ("QA list", k_qas)) if k is None]
    if missing:
        raise SystemExit(f"[load] ERROR: could not find {missing} in the first "
                         f"record. Its keys are: {sorted(first)}. Add the right "
                         f"names to the *_KEYS tuples at the top of this script.")

    first_qas = first[k_qas]
    first_qa = (list(first_qas.values()) if isinstance(first_qas, dict) else first_qas)[0]
    k_qid = _pick(first_qa, QA_ID_KEYS)
    k_qtype = _pick(first_qa, QA_TYPE_KEYS)
    k_q = _pick(first_qa, QUESTION_KEYS)
    k_a = _pick(first_qa, ANSWER_KEYS)
    missing = [n for n, k in (("qa id", k_qid), ("qa type", k_qtype),
                              ("question", k_q), ("answer", k_a)) if k is None]
    if missing:
        raise SystemExit(f"[load] ERROR: could not find {missing} in the first "
                         f"QA entry. Its keys are: {sorted(first_qa)}. Add the "
                         f"right names to the *_KEYS tuples at the top.")
    say(f"[load] fields: image='{k_img}' house='{k_house or 'from image_id'}' "
        f"capture='{k_cap}' qa_list='{k_qas}' qa_id='{k_qid}' type='{k_qtype}' "
        f"question='{k_q}' answer='{k_a}'")

    items = []
    for rec in records:
        img = str(rec[k_img])
        if k_house and rec.get(k_house):
            house = str(rec[k_house])
        else:
            m = HOUSE_RE.match(img)
            if not m:
                raise SystemExit(f"[load] ERROR: no house id for image {img}")
            house = m.group(1)
        qas = rec[k_qas]
        qas = list(qas.values()) if isinstance(qas, dict) else qas
        for qa in qas:
            items.append({
                "qa_id": str(qa[k_qid]),
                "image_id": img,
                "house_id": house,
                "capture_type": str(rec[k_cap]),
                "qa_type": str(qa[k_qtype]),
                "question_bn": qa[k_q],
                "answer_bn": qa[k_a],
                "image_file": rec.get(k_file) if k_file else None,
            })
    say(f"[load] {len(records)} records, {len(items)} QA items")
    return items


def validate_items(items, counts_csv=None):
    """Hard checks. Any failure stops the script before anything is drawn."""
    say("[check] validating items")
    ids = Counter(it["qa_id"] for it in items)
    dup = [q for q, n in ids.items() if n > 1]
    if dup:
        raise SystemExit(f"[check] ERROR: {len(dup)} duplicate qa_id, e.g. {dup[:5]}. "
                         f"qa_id must be unique across the whole dataset, because the "
                         f"inference cache is keyed by (model, condition, qa_id). "
                         f"Schema v2 ids look like H01_0001_q1. Stop and ask before "
                         f"building any composite id.")
    bad_type = Counter(it["qa_type"] for it in items if it["qa_type"] not in TYPES)
    if bad_type:
        raise SystemExit(f"[check] ERROR: unknown qa types {dict(bad_type)}. "
                         f"If you see obj_hazard, you are reading a pre-v2 file.")
    bad_cap = Counter(it["capture_type"] for it in items if it["capture_type"] not in CAPTURES)
    if bad_cap:
        raise SystemExit(f"[check] ERROR: unknown capture types {dict(bad_cap)}")
    wrong_track = [it["qa_id"] for it in items
                   if (it["qa_type"] in CLOSEUP_TYPES) != (it["capture_type"] == "closeup")]
    if wrong_track:
        raise SystemExit(f"[check] ERROR: {len(wrong_track)} items on the wrong "
                         f"track, e.g. {wrong_track[:5]}")
    houses = sorted({it["house_id"] for it in items})
    if len(houses) != N_HOUSES:
        raise SystemExit(f"[check] ERROR: {len(houses)} houses, expected {N_HOUSES}")
    say(f"[check] {len(items)} items, {len(houses)} houses, qa_id unique, "
        f"types and capture tracks consistent")

    if counts_csv and not os.path.exists(counts_csv):
        say(f"[check] WARNING: {counts_csv} not found, so the per-cell cross-check "
            f"against ASSET-1 is SKIPPED. Pass --counts with the right path.")
        counts_csv = None
    if counts_csv:
        avail, capinfo = availability(items)
        with open(counts_csv, encoding="utf-8-sig", newline="") as f:
            rows = {r["house_id"]: r for r in csv.DictReader(f)}
        if sorted(rows) != houses:
            raise SystemExit("[check] ERROR: houses in counts CSV differ from data")
        diffs = [(h, t, int(rows[h][t]), avail[h][t]) for h in houses for t in TYPES
                 if int(rows[h][t]) != avail[h][t]]
        diffs += [(h, f"img_{c}", int(rows[h][f"img_{c}"]), capinfo[h][c])
                  for h in houses for c in CAPTURES
                  if f"img_{c}" in rows[h] and int(rows[h][f"img_{c}"]) != capinfo[h][c]]
        if diffs:
            raise SystemExit(f"[check] ERROR: {len(diffs)} cells differ from "
                             f"{counts_csv}, e.g. (house, type, csv, data) {diffs[:5]}. "
                             f"Either the loader is reading the wrong field or the "
                             f"data changed since ASSET-1.")
        say(f"[check] per-house QA counts by type and image counts by capture type "
            f"match {counts_csv} in all {len(houses) * (len(TYPES) + len(CAPTURES))} cells")
    return bool(counts_csv)


def availability(items):
    avail = {}
    capinfo = {}
    for it in items:
        h = it["house_id"]
        avail.setdefault(h, {t: 0 for t in TYPES})
        avail[h][it["qa_type"]] += 1
    imgs = defaultdict(set)
    for it in items:
        imgs[(it["house_id"], it["capture_type"])].add(it["image_id"])
    for h in avail:
        capinfo[h] = {c: len(imgs[(h, c)]) for c in CAPTURES}
    return avail, capinfo


def availability_from_counts_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    avail = {r["house_id"]: {t: int(r[t]) for t in TYPES} for r in rows}
    capinfo = {r["house_id"]: {"overview": int(r["img_overview"]),
                               "group": int(r["img_group"]),
                               "closeup": int(r["img_closeup"])} for r in rows}
    return avail, capinfo


# ----------------------------------------------------------------------------
# P-CORE allocation
# ----------------------------------------------------------------------------
def rake(avail, row_target, col_target, max_iter=10000, tol=1e-10):
    """Iterative proportional fitting of the availability table.

    Result x[h][t] = a_h * b_t * avail[h][t], with every row summing to
    row_target and every column to col_target. Each house keeps its own mix of
    types, tilted by one factor per type that is the same for every house.
    """
    houses = sorted(avail)
    x = {h: {t: float(avail[h][t]) for t in TYPES} for h in houses}
    for it in range(1, max_iter + 1):
        for h in houses:
            s = sum(x[h][t] for t in TYPES)
            f = row_target / s
            for t in TYPES:
                x[h][t] *= f
        for t in TYPES:
            s = sum(x[h][t] for h in houses)
            f = col_target / s
            for h in houses:
                x[h][t] *= f
        err = max(abs(sum(x[h][t] for t in TYPES) - row_target) for h in houses)
        if err < tol:
            return x, it
    raise SystemExit("[alloc] ERROR: raking did not converge")


TRACKS = {"closeup": CLOSEUP_TYPES, "scene": SCENE_TYPES}


def repair_columns(q, target, hi, col_totals, cols, level):
    """Single-unit column repair inside the table q[house][column].

    Moves one unit at a time, inside one house, from a column whose total is
    above its required total to one whose total is below it. Row sums never
    change. Each move is the one that adds the least squared distance to the
    raked target; ties are broken by hash. Returns the list of moves.
    """
    houses = sorted(q)
    moves = []
    while True:
        tot = {c: sum(q[h][c] for h in houses) for c in cols}
        surplus = [c for c in cols if tot[c] > col_totals[c]]
        deficit = [c for c in cols if tot[c] < col_totals[c]]
        if not surplus and not deficit:
            return moves
        if not surplus or not deficit:
            raise SystemExit(f"[alloc] ERROR: {level} totals {tot} cannot be repaired "
                             f"to {col_totals}. Stop and ask.")
        best = None
        for s in surplus:
            for d in deficit:
                for h in houses:
                    if q[h][s] >= 1 and q[h][d] < hi[h][d]:
                        cost = round((q[h][d] - target[h][d]) - (q[h][s] - target[h][s]), 9)
                        key = (cost, hkey(SEED, "P-CORE", "repair", level, h, s, d))
                        if best is None or key < best[0]:
                            best = (key, h, s, d)
        if best is None:
            raise SystemExit(f"[alloc] ERROR: {level} repair is stuck; no single-unit "
                             f"move exists. Stop and ask before changing anything.")
        _, h, s, d = best
        q[h][s] -= 1
        q[h][d] += 1
        moves.append({"level": level, "house": h, "from": s, "to": d})


def allocate_pcore(avail):
    """Integer quotas per (house, QA type) for P-CORE.

    Step 1 rakes the availability table to 30 per house and 255 per type.
    Steps 2 to 5 round it in two levels: first each house's close-up and
    scene totals, then the QA types inside each track. Rounding the track
    totals first keeps every house's close-up count within one item of its
    raked value, which is the within-house capture stratification D45 asks
    for. One-level rounding over all 8 types left four houses that do have
    close-up images with no close-up items at all.
    """
    houses = sorted(avail)
    if len(houses) != N_HOUSES:
        raise SystemExit(f"[alloc] ERROR: {len(houses)} houses, expected {N_HOUSES}")
    short_rows = [h for h in houses if sum(avail[h].values()) < PCORE_PER_HOUSE]
    short_cols = [t for t in TYPES if sum(avail[h][t] for h in houses) < PCORE_PER_TYPE]
    if short_rows or short_cols:
        raise SystemExit(f"[alloc] ERROR: infeasible. houses below 30: {short_rows}, "
                         f"types below 255: {short_cols}")

    say("[alloc] step 1: raking availability to 30 per house and 255 per type")
    x, iters = rake(avail, PCORE_PER_HOUSE, PCORE_PER_TYPE)
    over = [(h, t) for h in houses for t in TYPES if x[h][t] > avail[h][t] + 1e-9]
    if over:
        raise SystemExit(f"[alloc] ERROR: raked target exceeds availability in "
                         f"{over[:5]}. A capped method is needed; stop and ask.")
    say(f"[alloc]   converged in {iters} iterations, no target exceeds availability")

    X = {h: {g: sum(x[h][t] for t in ts) for g, ts in TRACKS.items()} for h in houses}
    A = {h: {g: sum(avail[h][t] for t in ts) for g, ts in TRACKS.items()} for h in houses}
    track_totals = {g: PCORE_PER_TYPE * len(ts) for g, ts in TRACKS.items()}

    say("[alloc] step 2: each house's close-up and scene totals by largest remainder")
    T = {h: bounded_lr(X[h], PCORE_PER_HOUSE, lo={g: 0 for g in TRACKS}, hi=A[h],
                       tiebreak={g: hkey(SEED, "P-CORE", "track", h, g) for g in TRACKS})
         for h in houses}
    say(f"[alloc]   close-up items before repair: {sum(T[h]['closeup'] for h in houses)} "
        f"(needs {track_totals['closeup']})")

    say(f"[alloc] step 3: track repair so close-up items total exactly "
        f"{track_totals['closeup']}")
    track_moves = repair_columns(T, X, A, track_totals, list(TRACKS), "track")
    say(f"[alloc]   {len(track_moves)} track moves")
    track_dev = max(abs(T[h]["closeup"] - X[h]["closeup"]) for h in houses)
    if track_dev >= 1:
        raise SystemExit(f"[alloc] ERROR: a house's close-up total moved {track_dev:.3f} "
                         f"from its raked value. Stop and ask.")

    say("[alloc] step 4: spread each track total over its QA types by largest remainder")
    quota = {h: {} for h in houses}
    for h in houses:
        for g, ts in TRACKS.items():
            quota[h].update(bounded_lr(
                {t: x[h][t] for t in ts}, T[h][g],
                lo={t: 0 for t in ts}, hi={t: avail[h][t] for t in ts},
                tiebreak={t: hkey(SEED, "P-CORE", "alloc", h, t) for t in ts}))
    cols = {t: sum(quota[h][t] for h in houses) for t in TYPES}
    say(f"[alloc]   column sums before repair: {cols}")

    say("[alloc] step 5: column repair to exactly 255, moving units only between "
        "types of the same track")
    type_moves = []
    for g, ts in TRACKS.items():
        type_moves += repair_columns(quota, x, avail, {t: PCORE_PER_TYPE for t in ts},
                                     ts, g)
    say(f"[alloc]   {len(type_moves)} type moves")
    moves = track_moves + type_moves

    # The four required assertions, plus availability and the track bound.
    assert all(sum(quota[h].values()) == PCORE_PER_HOUSE for h in houses), "row != 30"
    assert all(sum(quota[h][t] for h in houses) == PCORE_PER_TYPE for t in TYPES), "col != 255"
    assert all(quota[h][t] <= avail[h][t] for h in houses for t in TYPES), "over availability"
    assert len(houses) == N_HOUSES, "houses missing"
    assert all(sum(quota[h][t] for t in CLOSEUP_TYPES) == T[h]["closeup"] for h in houses), \
        "type repair changed a house's close-up total"
    dev = max(abs(quota[h][t] - x[h][t]) for h in houses for t in TYPES)
    say(f"[alloc]   every cell within {dev:.3f} of its raked target; every house's "
        f"close-up total within {track_dev:.3f}")
    return quota, x, iters, moves, dev, track_dev


def split_overview_group(house, k_by_type, ov, gr, probe):
    """Split each scene-track type's quota between overview and group images.

    The house's overview total is fixed first, in proportion to its image
    counts, then spread over the types by largest remainder.
    """
    K = sum(k_by_type.values())
    if K == 0:
        return {t: (0, 0) for t in k_by_type}
    if ov == 0 or gr == 0:
        return {t: (k, 0) if gr == 0 else (0, k) for t, k in k_by_type.items()}
    share = ov / (ov + gr)
    lo_t = {t: max(0, k - gr) for t, k in k_by_type.items()}
    hi_t = {t: min(k, ov) for t, k in k_by_type.items()}
    O = bounded_lr(
        {"ov": K * share, "gr": K * (1 - share)}, K,
        lo={"ov": sum(lo_t.values()), "gr": K - sum(hi_t.values())},
        hi={"ov": sum(hi_t.values()), "gr": K - sum(lo_t.values())},
        tiebreak={"ov": hkey(SEED, probe, "capsplit", house, "ov"),
                  "gr": hkey(SEED, probe, "capsplit", house, "gr")})["ov"]
    o = bounded_lr({t: k * share for t, k in k_by_type.items()}, O, lo_t, hi_t,
                   {t: hkey(SEED, probe, "capsplit", house, t) for t in k_by_type})
    return {t: (o[t], k_by_type[t] - o[t]) for t in k_by_type}


def cell_quotas_pcore(quota, capinfo):
    """(house, type, capture) -> number of items to draw."""
    cells = {}
    for h in sorted(quota):
        scene_k = {t: quota[h][t] for t in SCENE_TYPES}
        split = split_overview_group(h, scene_k, capinfo[h]["overview"],
                                     capinfo[h]["group"], "P-CORE")
        for t in SCENE_TYPES:
            cells[(h, t, "overview")] = split[t][0]
            cells[(h, t, "group")] = split[t][1]
        for t in CLOSEUP_TYPES:
            cells[(h, t, "closeup")] = quota[h][t]
    return cells


def sample_cells(items, cells, probe):
    """Take the lowest-ranked items in each cell."""
    by_cell = defaultdict(list)
    for it in items:
        by_cell[(it["house_id"], it["qa_type"], it["capture_type"])].append(it)
    out = []
    for cell in sorted(cells):
        n = cells[cell]
        if n == 0:
            continue
        pool = sorted(by_cell[cell], key=lambda it: hkey(SEED, probe, it["qa_id"]))
        if len(pool) < n:
            raise SystemExit(f"[draw] ERROR: cell {cell} needs {n}, has {len(pool)}")
        out.extend(pool[:n])
    return out


def draw_pcore(items, avail, capinfo):
    quota, x, iters, moves, dev, track_dev = allocate_pcore(avail)
    cells = cell_quotas_pcore(quota, capinfo)
    say("[draw] sampling P-CORE cells by SHA-256 rank under seed 42")
    drawn = sample_cells(items, cells, "P-CORE")
    drawn.sort(key=lambda it: (it["house_id"], it["qa_type"], it["qa_id"]))

    # Required assertions on the drawn items.
    per_house = Counter(it["house_id"] for it in drawn)
    per_type = Counter(it["qa_type"] for it in drawn)
    assert len(drawn) == N_HOUSES * PCORE_PER_HOUSE == 2040, "P-CORE size"
    assert all(per_house[h] == PCORE_PER_HOUSE for h in avail), "a row is not 30"
    assert len(per_house) == N_HOUSES, "a house is missing"
    assert all(per_type[t] == PCORE_PER_TYPE for t in TYPES), "a column is not 255"
    assert len({it["qa_id"] for it in drawn}) == len(drawn), "duplicate qa_id"
    say(f"[draw] P-CORE: {len(drawn)} items; every house 30, every type 255, "
        f"qa_id unique, {len(per_house)} houses")

    # Descriptives for the manifest.
    ratio = None
    for h in sorted(avail):
        s, c = avail[h]["assistive"], avail[h]["obj_name"]
        if s and c:
            ratio = (x[h]["obj_name"] / c) / (x[h]["assistive"] / s)
            break
    shift = []
    for h in sorted(avail):
        tot = sum(avail[h].values())
        nat = sum(avail[h][t] for t in CLOSEUP_TYPES) / tot
        got = sum(quota[h][t] for t in CLOSEUP_TYPES) / PCORE_PER_HOUSE
        shift.append(100 * (got - nat))
    cap_tot = Counter(it["capture_type"] for it in drawn)
    stats = {
        "raking_iterations": iters,
        "repair_moves": len(moves),
        "repair_move_list": moves,
        "max_abs_cell_deviation_from_raked_target": round(dev, 4),
        "max_abs_house_closeup_total_deviation_from_raked": round(track_dev, 4),
        "closeup_vs_scene_inclusion_ratio_within_house": round(ratio, 4) if ratio else None,
        "closeup_share_shift_pp_per_house": {
            "min": round(min(shift), 1), "median": round(statistics.median(shift), 1),
            "max": round(max(shift), 1)},
        "capture_totals": {c: cap_tot.get(c, 0) for c in CAPTURES},
        "distinct_images": len({it["image_id"] for it in drawn}),
        "houses_with_no_closeup_images": sorted(h for h in avail
                                                if avail[h]["obj_name"] == 0),
        "houses_with_no_closeup_items": sorted(h for h in avail
                                               if sum(quota[h][t] for t in CLOSEUP_TYPES) == 0),
    }
    return drawn, quota, x, cells, stats


def write_allocation_csv(path, avail, capinfo, quota, x, cells):
    cols = (["house_id", "img_overview", "img_group", "img_closeup"]
            + [f"{t}_avail" for t in TYPES] + [f"{t}_target" for t in TYPES]
            + [f"{t}_quota" for t in TYPES]
            + ["scene_items_overview", "scene_items_group", "total"])
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for h in sorted(quota):
            ov = sum(cells[(h, t, "overview")] for t in SCENE_TYPES)
            gr = sum(cells[(h, t, "group")] for t in SCENE_TYPES)
            w.writerow([h, capinfo[h]["overview"], capinfo[h]["group"], capinfo[h]["closeup"]]
                       + [avail[h][t] for t in TYPES]
                       + [f"{x[h][t]:.4f}" for t in TYPES]
                       + [quota[h][t] for t in TYPES]
                       + [ov, gr, sum(quota[h].values())])


# ----------------------------------------------------------------------------
# P-HAZ: audit, merge, draw
# ----------------------------------------------------------------------------
def build_audit_rows(items):
    groups = defaultdict(list)
    for it in items:
        if it["qa_type"] in PHAZ_TYPES:
            qn = normalise(it["question_bn"])
            groups[(it["qa_type"], qn)].append(it)
    rows = []
    for (qt, qn), its in groups.items():
        rows.append({
            "template_id": template_id(qt, qn),
            "qa_type": qt,
            "n_items": len(its),
            "n_houses": len({i["house_id"] for i in its}),
            "preflag_hint": " ".join(preflag_terms(qn)),
            "question": qn,
            "invites_hazard_report": "",
        })
    rows.sort(key=lambda r: (r["qa_type"], -r["n_items"], r["template_id"]))
    return rows


AUDIT_COLS = ["template_id", "qa_type", "n_items", "n_houses", "preflag_hint",
              "question", "invites_hazard_report"]


def write_csv(path, rows, cols):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def clean_yes_no(v):
    v = (v or "").strip().lower()
    return v if v in ("yes", "no") else None


def load_classes(path, items):
    """Return (classes, missing_ids). classes: template_id -> 'yes'/'no'."""
    rows = read_csv(path)
    classes = {}
    bad = []
    for r in rows:
        v = clean_yes_no(r.get("invites_hazard_report"))
        if v is None:
            bad.append(r.get("template_id"))
        else:
            classes[r["template_id"]] = v
    needed = {template_id(it["qa_type"], normalise(it["question_bn"]))
              for it in items if it["qa_type"] in PHAZ_TYPES}
    missing = sorted(needed - set(classes))
    unknown = sorted(set(classes) - needed)
    if bad:
        say(f"[phaz] {len(bad)} rows in {path} have no yes/no value")
    if unknown:
        say(f"[phaz] WARNING: {len(unknown)} template ids in the class file do not "
            f"occur in the data (question text changed since the audit?)")
    return classes, missing


def draw_phaz(items, classes):
    def tid(it):
        return template_id(it["qa_type"], normalise(it["question_bn"]))

    pool_obj = [it for it in items if it["qa_type"] == "obj_property" and classes[tid(it)] == "yes"]
    pool_ast = [it for it in items if it["qa_type"] == "assistive" and classes[tid(it)] == "yes"]
    n_obj = min(len(pool_obj), PHAZ_OBJPROP_CAP)
    n_ast = PHAZ_SIZE - n_obj
    say(f"[phaz] eligible pools: obj_property (hazard) {len(pool_obj)}, "
        f"assistive {len(pool_ast)}; taking {n_obj} + {n_ast}")
    if len(pool_ast) < n_ast:
        raise SystemExit(f"[phaz] ERROR: only {len(pool_ast)} eligible assistive items, "
                         f"need {n_ast}. Stop and ask; do not loosen the rule.")

    def per_house(pool):
        c = Counter(it["house_id"] for it in pool)
        return {h: c[h] for h in sorted(c)}

    cells = {}
    # Stratum 1: hazard-flavoured obj_property.
    ph = per_house(pool_obj)
    if n_obj == len(pool_obj):
        for h, n in ph.items():
            cells[(h, "obj_property", "closeup")] = n
    elif n_obj > 0:
        tot = sum(ph.values())
        alloc = bounded_lr({h: n_obj * n / tot for h, n in ph.items()}, n_obj,
                           {h: 0 for h in ph}, ph,
                           {h: hkey(SEED, "P-HAZ", "alloc", "obj", h) for h in ph})
        for h, n in alloc.items():
            cells[(h, "obj_property", "closeup")] = n
    # Stratum 2: assistive, by house, then overview/group within house.
    ph = per_house(pool_ast)
    tot = sum(ph.values())
    alloc = bounded_lr({h: n_ast * n / tot for h, n in ph.items()}, n_ast,
                       {h: 0 for h in ph}, ph,
                       {h: hkey(SEED, "P-HAZ", "alloc", "ast", h) for h in ph})
    capc = Counter((it["house_id"], it["capture_type"]) for it in pool_ast)
    for h, n in alloc.items():
        ov, gr = capc[(h, "overview")], capc[(h, "group")]
        if n == 0:
            continue
        if ov == 0 or gr == 0:
            cells[(h, "assistive", "overview")] = n if gr == 0 else 0
            cells[(h, "assistive", "group")] = n if ov == 0 else 0
            continue
        k = bounded_lr({"overview": n * ov / (ov + gr), "group": n * gr / (ov + gr)}, n,
                       {"overview": max(0, n - gr), "group": max(0, n - ov)},
                       {"overview": min(n, ov), "group": min(n, gr)},
                       {c: hkey(SEED, "P-HAZ", "capsplit", h, c) for c in ("overview", "group")})
        cells[(h, "assistive", "overview")] = k["overview"]
        cells[(h, "assistive", "group")] = k["group"]

    eligible = pool_obj + pool_ast
    drawn = sample_cells(eligible, cells, "P-HAZ")
    drawn = [dict(it, phaz_stratum=("obj_property_hazard" if it["qa_type"] == "obj_property"
                                    else "assistive"), template_id=tid(it)) for it in drawn]
    drawn.sort(key=lambda it: (it["house_id"], it["qa_type"], it["qa_id"]))

    assert len(drawn) == PHAZ_SIZE, "P-HAZ size"
    assert len({it["qa_id"] for it in drawn}) == PHAZ_SIZE, "P-HAZ duplicate qa_id"
    assert sum(1 for it in drawn if it["qa_type"] == "obj_property") == n_obj, "stratum 1 size"
    assert all(classes[it["template_id"]] == "yes" for it in drawn), "ineligible item drawn"
    say(f"[phaz] P-HAZ: {len(drawn)} items, qa_id unique, all eligible")

    stats = {
        "pool_obj_property_hazard": len(pool_obj),
        "pool_assistive_eligible": len(pool_ast),
        "assistive_total_in_corpus": sum(1 for it in items if it["qa_type"] == "assistive"),
        "n_obj_property_hazard": n_obj,
        "n_assistive": n_ast,
        "houses_represented": len({it["house_id"] for it in drawn}),
        "capture_totals": dict(Counter(it["capture_type"] for it in drawn)),
    }
    return drawn, stats


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
PROBE_ITEM_KEYS = ["qa_id", "image_id", "house_id", "capture_type", "qa_type",
                   "question_bn", "answer_bn", "image_file", "phaz_stratum", "template_id"]


def qa_id_list_hash(items):
    return hkey(*sorted(it["qa_id"] for it in items))


def guard_frozen(path, new_items, force):
    """Refuse to silently replace a probe whose membership would change."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        old = json.load(f)
    old_ids = sorted(it["qa_id"] for it in old["items"])
    new_ids = sorted(it["qa_id"] for it in new_items)
    if old_ids == new_ids:
        say(f"[guard] {os.path.basename(path)} exists; membership identical, overwriting")
        return
    changed = len(set(old_ids) ^ set(new_ids))
    if not force:
        raise SystemExit(f"[guard] STOP: {path} exists and the new draw differs in "
                         f"{changed} qa_ids. A probe must not change after it is "
                         f"frozen. If this is intended and logged as a decision, "
                         f"rerun with --force.")
    say(f"[guard] --force: replacing {path} ({changed} qa_ids differ)")


def write_probe(path, probe, items, dataset_sha, extra=None):
    doc = {
        "probe": probe,
        "n": len(items),
        "seed": SEED,
        "sampling": f'lowest SHA-256("{SEED}|{probe}|<qa_id>") ranks within each stratum',
        "dataset_sha256": dataset_sha,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "qa_id_list_sha256": qa_id_list_hash(items),
        "items": [{k: it[k] for k in PROBE_ITEM_KEYS if k in it and it[k] is not None}
                  for it in items],
    }
    if extra:
        doc.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    say(f"[write] {path} ({len(items)} items)")


def unique_string_counts(pcore, phaz):
    """ASSET-2 stage 0 (answers Q03 for wave 1): unique normalise()d strings."""
    def uq(items, field, types=None):
        return {normalise(it[field]) for it in items if types is None or it["qa_type"] in types}
    out = {"P-CORE": {"questions": len(uq(pcore, "question_bn")),
                      "answers": len(uq(pcore, "answer_bn")),
                      "by_type": {t: {"questions": len(uq(pcore, "question_bn", [t])),
                                      "answers": len(uq(pcore, "answer_bn", [t]))}
                                  for t in TYPES}}}
    total = out["P-CORE"]["questions"] + out["P-CORE"]["answers"]
    out["P-CORE"]["postedit_hours_at_60_per_hour_all_strings"] = round(total / POSTEDIT_RATE_PER_HOUR, 1)
    if phaz:
        q_new = uq(phaz, "question_bn") - uq(pcore, "question_bn")
        out["P-HAZ"] = {"questions": len(uq(phaz, "question_bn")),
                        "questions_not_already_in_P-CORE": len(q_new),
                        "answers": len(uq(phaz, "answer_bn")),
                        "note": "E7 scores against hazard_gold.csv (ASSET-6); whether "
                                "P-HAZ answers need English is an open E7 scoring question"}
    return out


def gold_checks(pcore, phaz):
    """Gold-answer problems a human must fix before ASSET-2 translates them.

    Only reports; never changes membership. Membership depends on qa_id alone,
    so fixing these answers in the dataset and re-running draw leaves both
    probes identical (the guard confirms it).
    """
    def same(items):
        return sorted(it["qa_id"] for it in items
                      if normalise(it["answer_bn"]) == normalise(it["question_bn"]))
    forms = Counter(normalise(it["answer_bn"]) for it in pcore if it["qa_type"] == "count")
    out = {
        "answer_equals_question": {"P-CORE": same(pcore), "P-HAZ": same(phaz) if phaz else []},
        "P-CORE_count_answer_forms": dict(forms.most_common()),
        "P-CORE_count_items_with_a_form_used_once": sorted(
            it["qa_id"] for it in pcore
            if it["qa_type"] == "count" and forms[normalise(it["answer_bn"])] == 1),
    }
    return out


# ----------------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------------
def cmd_audit(args):
    items = load_items(args.data)
    validate_items(items, args.counts)
    rows = build_audit_rows(items)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "phaz_question_audit.csv")
    write_csv(path, rows, AUDIT_COLS)
    by = Counter(r["qa_type"] for r in rows)
    flagged = Counter(r["qa_type"] for r in rows if r["preflag_hint"])
    say()
    say(f"[audit] unique normalised question strings: assistive {by['assistive']}, "
        f"obj_property {by['obj_property']}")
    say(f"[audit] with a hint word: assistive {flagged['assistive']}, "
        f"obj_property {flagged['obj_property']} (hints only, not the rule)")
    say(f"[audit] wrote {path}")
    say(f"[audit] rater test: {RATER_TEST}")
    say("[audit] each rater fills invites_hazard_report in their OWN copy.")


def cmd_merge(args):
    ref = {r["template_id"]: r for r in read_csv(args.audit)}
    r1 = {r["template_id"]: r for r in read_csv(args.rater1)}
    r2 = {r["template_id"]: r for r in read_csv(args.rater2)}
    for name, rr in (("rater1", r1), ("rater2", r2)):
        miss = sorted(set(ref) - set(rr))
        if miss:
            raise SystemExit(f"[merge] ERROR: {name} file lacks {len(miss)} templates, e.g. {miss[:3]}")
        blank = [t for t in ref if clean_yes_no(rr[t].get("invites_hazard_report")) is None]
        if blank:
            raise SystemExit(f"[merge] ERROR: {name} has {len(blank)} rows without yes/no, "
                             f"e.g. {blank[:3]}")
        damaged = [t for t in ref if normalise(rr[t].get("question", "")) != ref[t]["question"]]
        if damaged:
            say(f"[merge] WARNING: {len(damaged)} question texts in {name} differ from the "
                f"audit file. Usually Excel saved the file in the wrong encoding. The "
                f"yes/no values are still matched by template_id.")
    rows, agree, items_agree, items_all = [], 0, 0, 0
    per_type = defaultdict(lambda: [0, 0])
    for t, r in ref.items():
        a = clean_yes_no(r1[t]["invites_hazard_report"])
        b = clean_yes_no(r2[t]["invites_hazard_report"])
        n = int(r["n_items"])
        items_all += n
        per_type[r["qa_type"]][1] += 1
        if a == b:
            agree += 1
            items_agree += n
            per_type[r["qa_type"]][0] += 1
        rows.append(dict(r, rater1=a, rater2=b, needs_adjudication="no" if a == b else "yes",
                         invites_hazard_report=a if a == b else ""))
    rows.sort(key=lambda r: (r["needs_adjudication"] != "yes", r["qa_type"], -int(r["n_items"])))
    draft = os.path.join(os.path.dirname(args.audit), "phaz_question_classes_DRAFT.csv")
    write_csv(draft, rows, AUDIT_COLS[:-1] + ["rater1", "rater2", "needs_adjudication",
                                              "invites_hazard_report"])
    n_t = len(ref)
    y1 = sum(1 for t in ref if clean_yes_no(r1[t]["invites_hazard_report"]) == "yes") / n_t
    y2 = sum(1 for t in ref if clean_yes_no(r2[t]["invites_hazard_report"]) == "yes") / n_t
    p_o, p_e = agree / n_t, y1 * y2 + (1 - y1) * (1 - y2)
    kappa = round((p_o - p_e) / (1 - p_e), 4) if p_e < 1 else None
    rep = {"templates": n_t, "templates_agreed": agree,
           "raw_agreement_templates": round(p_o, 4),
           "cohen_kappa_templates": kappa,
           "yes_rate_rater1": round(y1, 4), "yes_rate_rater2": round(y2, 4),
           "items_under_agreed_templates": items_agree, "items_total": items_all,
           "by_type": {k: {"agreed": v[0], "templates": v[1]} for k, v in per_type.items()},
           "disagreements": [r["template_id"] for r in rows if r["needs_adjudication"] == "yes"]}
    rep_path = os.path.join(os.path.dirname(args.audit), "phaz_classification_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    say(f"[merge] {agree}/{n_t} templates agreed ({100 * p_o:.1f}%), Cohen's kappa "
        f"{kappa if kappa is not None else 'undefined (both raters gave one answer only)'}, "
        f"covering {items_agree}/{items_all} items")
    say(f"[merge] {len(rep['disagreements'])} need the adjudicator")
    say(f"[merge] wrote {draft}")
    say(f"[merge] wrote {rep_path}")
    say("[merge] adjudicator: fill the blank invites_hazard_report cells, then save the")
    say("        file as phaz_question_classes.csv in the same folder.")


def cmd_draw(args):
    t0 = datetime.datetime.now(datetime.timezone.utc)
    items = load_items(args.data)
    counts_checked = validate_items(items, args.counts)
    os.makedirs(args.out, exist_ok=True)
    data_sha = sha256_file(args.data)
    avail, capinfo = availability(items)

    say()
    pcore, quota, x, cells, pstats = draw_pcore(items, avail, capinfo)
    p_path = os.path.join(args.out, "P-CORE.json")
    guard_frozen(p_path, pcore, args.force)

    phaz, zstats, cls_path = None, None, args.classes or os.path.join(args.out, "phaz_question_classes.csv")
    say()
    if os.path.exists(cls_path):
        classes, missing = load_classes(cls_path, items)
        if missing:
            say(f"[phaz] NOT DRAWN: {len(missing)} question templates have no final yes/no "
                f"in {cls_path}. Finish adjudication, then rerun draw.")
        else:
            phaz, zstats = draw_phaz(items, classes)
            overlap = len({i["qa_id"] for i in phaz} & {i["qa_id"] for i in pcore})
            zstats["overlap_with_P-CORE"] = overlap
            zstats["class_file_sha256"] = sha256_file(cls_path)
            rep = os.path.join(os.path.dirname(cls_path), "phaz_classification_report.json")
            if os.path.exists(rep):
                with open(rep, encoding="utf-8") as f:
                    zstats["classification_agreement"] = json.load(f)
            say(f"[phaz] overlap with P-CORE: {overlap} items (allowed; independent draws)")
            guard_frozen(os.path.join(args.out, "P-HAZ.json"), phaz, args.force)
    else:
        say(f"[phaz] NOT DRAWN: {cls_path} does not exist yet. P-CORE does not need it.")

    say()
    write_probe(p_path, "P-CORE", pcore, data_sha)
    alloc_path = os.path.join(args.out, "pcore_allocation.csv")
    write_allocation_csv(alloc_path, avail, capinfo, quota, x, cells)
    say(f"[write] {alloc_path}")
    outputs = {"P-CORE.json": sha256_file(p_path), "pcore_allocation.csv": sha256_file(alloc_path)}
    if phaz:
        z_path = os.path.join(args.out, "P-HAZ.json")
        write_probe(z_path, "P-HAZ", phaz, data_sha)
        outputs["P-HAZ.json"] = sha256_file(z_path)

    here = os.path.dirname(os.path.abspath(__file__))
    manifest = {
        "asset": "ASSET-5A",
        "run_utc": t0.isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "normalise_py_sha256": sha256_file(os.path.join(here, "normalise.py")),
        "seed": SEED,
        "sampling_rule": "Within each stratum, items are ranked by SHA-256 of "
                         "'42|<probe>|<qa_id>' and the lowest ranks are taken. "
                         "Tie-breaks in rounding use the same hash. Text never "
                         "affects membership.",
        "dataset": {"path": os.path.basename(args.data), "sha256": data_sha,
                    "qa_items": len(items), "houses": len(avail),
                    "counts_checked_against": os.path.basename(args.counts) if counts_checked else None},
        "P-CORE": {
            "rule": "30 per house x 68 houses, 255 per QA type. Quotas: raking "
                    "(iterative proportional fitting) of the house x type availability "
                    "table to those margins; then two-level rounding: each house's "
                    "close-up and scene totals by largest remainder, single-unit repair "
                    "so close-up items total 765, then the QA types inside each track "
                    "by largest remainder, single-unit repair within track so every "
                    "type totals 255. Repairs minimise squared distance to the raked "
                    "target. Scene-track quotas split between overview and group "
                    "images in proportion to the house's image counts (largest "
                    "remainder, house total first). D45, D47, D48.",
            "n": len(pcore), **pstats},
        "P-HAZ": ({"rule": "Eligible: assistive and obj_property items whose question "
                           "passes the rater test in the frozen class file. Take "
                           "min(300, pool) hazard obj_property items and fill to 600 "
                           "with eligible assistive items; proportional by house; "
                           "assistive split overview/group within house; SHA-256 rank "
                           "sampling; independent of P-CORE. D49.",
                   "rater_test": RATER_TEST, "n": len(phaz), **zstats}
                  if phaz else {"status": "not drawn in this run"}),
        "asset2_stage0_unique_strings": unique_string_counts(pcore, phaz),
        "gold_checks": gold_checks(pcore, phaz),
        "outputs_sha256": outputs,
    }
    m_path = os.path.join(args.out, "probe_manifest_5A.json")
    with open(m_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    say(f"[write] {m_path}")

    s0 = manifest["asset2_stage0_unique_strings"]
    g = manifest["gold_checks"]
    say()
    say(f"[gold] answer identical to question: P-CORE {len(g['answer_equals_question']['P-CORE'])}, "
        f"P-HAZ {len(g['answer_equals_question']['P-HAZ'])}  (ids in the manifest)")
    say(f"[gold] P-CORE count answers use {len(g['P-CORE_count_answer_forms'])} distinct forms; "
        f"{len(g['P-CORE_count_items_with_a_form_used_once'])} item(s) use a form seen only once")
    say("[gold] fix these in the dataset before ASSET-2 translates them; membership will not move")
    say("=" * 70)
    say(f"P-CORE  {len(pcore)} items | repair moves {pstats['repair_moves']} | "
        f"distinct images {pstats['distinct_images']}")
    say(f"        capture totals {pstats['capture_totals']}")
    say(f"        close-up vs scene inclusion ratio {pstats['closeup_vs_scene_inclusion_ratio_within_house']}")
    say(f"P-HAZ   {'%d items' % len(phaz) if phaz else 'not drawn yet'}")
    say(f"Q03     P-CORE unique questions {s0['P-CORE']['questions']}, "
        f"unique answers {s0['P-CORE']['answers']}")
    if phaz:
        say(f"        P-HAZ questions not already in P-CORE "
            f"{s0['P-HAZ']['questions_not_already_in_P-CORE']}")
    say(f"Outputs in {os.path.abspath(args.out)}")
    say("=" * 70)


# ----------------------------------------------------------------------------
# Self-test: real counts for the allocation, synthetic items for the rest.
# ----------------------------------------------------------------------------
SYN_ATTR_Q = ["এটি কী রঙের?", "এটি কিসের তৈরি?", "এটির আকার কেমন?"]
SYN_HAZ_Q = ["এটি ব্যবহারে কী ঝুঁকি আছে?", "এটি কি বিপজ্জনক?"]
SYN_AST_Q = ["দৃষ্টি প্রতিবন্ধী ব্যক্তির এখানে কী খেয়াল রাখা উচিত?",
             "এখানে চলাফেরায় কোনো ঝুঁকি আছে কি?", "দরজাটি কোন দিকে?"]


def synthetic_items(avail, capinfo):
    """SYNTHETIC test items shaped like the real per-house counts."""
    items = []
    for h in sorted(avail):
        n = 0
        for cap in CAPTURES:
            for _ in range(capinfo[h][cap]):
                n += 1
                img = f"{h}_{n:04d}"
                types = CLOSEUP_TYPES if cap == "closeup" else SCENE_TYPES
                for slot, t in enumerate(types, start=1):
                    r = int(hkey("syn", img, t)[:8], 16)
                    if t == "obj_property":
                        q = (SYN_HAZ_Q if r % 4 == 0 else SYN_ATTR_Q)[r % 2]
                    elif t == "assistive":
                        q = SYN_AST_Q[r % 3]
                    else:
                        q = f"SYN {t} প্রশ্ন {r % 7}?"
                    items.append({"qa_id": f"{img}_q{slot}", "image_id": img, "house_id": h,
                                  "capture_type": cap, "qa_type": t, "question_bn": q,
                                  "answer_bn": f"SYN উত্তর {r % 11}।", "image_file": None})
    return items


def write_synthetic_json(items, path):
    """Group SYNTHETIC items back into one record per image, schema-v2 style."""
    recs = {}
    for it in items:
        r = recs.setdefault(it["image_id"], {"image_id": it["image_id"], "house_id": it["house_id"],
                                             "capture_type": it["capture_type"], "qa": []})
        r["qa"].append({"qa_id": it["qa_id"], "qa_type": it["qa_type"],
                        "question": it["question_bn"], "answer": it["answer_bn"]})
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"images": list(recs.values())}, f, ensure_ascii=False)


def cmd_selftest(args):
    say("#" * 70)
    say("SELF-TEST. The quotas (items per house, per type, per capture) come from")
    say("the REAL counts in house_type_counts.csv. The items themselves are SYNTHETIC,")
    say("so P-HAZ pools, overlap, unique-string counts, distinct images and gold")
    say("checks printed below are NOT dataset figures.")
    say("#" * 70)
    avail, capinfo = availability_from_counts_csv(args.counts)
    say(f"[selftest] read {args.counts}: {len(avail)} houses, "
        f"{sum(sum(v.values()) for v in avail.values())} QA")

    items = synthetic_items(avail, capinfo)
    validate_items(items)
    a2, c2 = availability(items)
    assert a2 == avail and c2 == capinfo, "synthetic items do not match the counts"

    pcore, quota, x, cells, pstats = draw_pcore(items, avail, capinfo)
    say(f"[selftest] repair moves {pstats['repair_moves']}, max cell deviation "
        f"{pstats['max_abs_cell_deviation_from_raked_target']}, capture totals "
        f"{pstats['capture_totals']}")
    say(f"[selftest] houses with no close-up images {pstats['houses_with_no_closeup_images']}")
    say(f"[selftest] houses with no close-up items  {pstats['houses_with_no_closeup_items']}")
    assert pstats["houses_with_no_closeup_items"] == pstats["houses_with_no_closeup_images"], \
        "a house with close-up images got no close-up items"
    say("[selftest] OK  every house that has close-up images has close-up items")

    # Determinism: same draw twice.
    pcore2 = draw_pcore(items, avail, capinfo)[0]
    assert [i["qa_id"] for i in pcore] == [i["qa_id"] for i in pcore2], "not deterministic"
    say("[selftest] OK  P-CORE is identical on a second run")

    # Text independence: change every answer and question, membership must not move.
    edited = [dict(it, answer_bn=it["answer_bn"] + " সংশোধিত",
                   question_bn=it["question_bn"] + " ") for it in items]
    pcore3 = draw_pcore(edited, avail, capinfo)[0]
    assert {i["qa_id"] for i in pcore} == {i["qa_id"] for i in pcore3}, "text changed membership"
    say("[selftest] OK  editing question and answer text does not change P-CORE")

    tmp = tempfile.mkdtemp(prefix="asset5a_selftest_")
    try:
        # End to end through the real subcommands, on a synthetic merged JSON.
        # Plant two gold problems in items that are in P-CORE, so the gold
        # check has something to find.
        plant_same = next(i["qa_id"] for i in pcore if i["qa_type"] == "obj_name")
        plant_count = next(i["qa_id"] for i in pcore if i["qa_type"] == "count")
        for it in items:
            if it["qa_id"] == plant_same:
                it["answer_bn"] = it["question_bn"]
            elif it["qa_id"] == plant_count:
                it["answer_bn"] = "দুটটি।"
        data = os.path.join(tmp, "synthetic_merged.json")
        write_synthetic_json(items, data)
        counts = os.path.join(tmp, "house_type_counts.csv")
        shutil.copy(args.counts, counts)
        say()
        say("[selftest] --- audit ---")
        cmd_audit(argparse.Namespace(data=data, counts=counts, out=tmp))
        audit = os.path.join(tmp, "phaz_question_audit.csv")
        rows = read_csv(audit)
        # Simulated raters: assistive yes unless it asks where the door is (a
        # landmark question); obj_property yes only for the hazard wording.
        truth = {}
        for r in rows:
            q = r["question"]
            truth[r["template_id"]] = "yes" if (r["qa_type"] == "assistive" and "দরজা" not in q) \
                or (r["qa_type"] == "obj_property" and r["preflag_hint"]) else "no"
        r1 = [dict(r, invites_hazard_report=truth[r["template_id"]]) for r in rows]
        r2 = [dict(r) for r in r1]
        r2[0]["invites_hazard_report"] = "no" if r2[0]["invites_hazard_report"] == "yes" else "yes"
        p1, p2 = os.path.join(tmp, "r1.csv"), os.path.join(tmp, "r2.csv")
        write_csv(p1, r1, AUDIT_COLS)
        write_csv(p2, r2, AUDIT_COLS)
        say("[selftest] --- merge ---")
        cmd_merge(argparse.Namespace(audit=audit, rater1=p1, rater2=p2))
        draft = read_csv(os.path.join(tmp, "phaz_question_classes_DRAFT.csv"))
        assert sum(1 for r in draft if r["needs_adjudication"] == "yes") == 1
        final = [dict(r, invites_hazard_report=r["invites_hazard_report"] or truth[r["template_id"]])
                 for r in draft]
        write_csv(os.path.join(tmp, "phaz_question_classes.csv"), final, list(final[0].keys()))

        say("[selftest] --- draw ---")
        draw_args = argparse.Namespace(data=data, counts=counts, out=tmp, classes=None, force=False)
        cmd_draw(draw_args)
        with open(os.path.join(tmp, "P-CORE.json"), encoding="utf-8") as f:
            pc_doc = json.load(f)
        with open(os.path.join(tmp, "P-HAZ.json"), encoding="utf-8") as f:
            pz_doc = json.load(f)
        with open(os.path.join(tmp, "probe_manifest_5A.json"), encoding="utf-8") as f:
            man = json.load(f)
        assert {i["qa_id"] for i in pc_doc["items"]} == {i["qa_id"] for i in pcore}, \
            "draw from file differs from the in-memory draw"
        assert len(pz_doc["items"]) == PHAZ_SIZE
        assert all("দরজা" not in normalise(i["question_bn"]) for i in pz_doc["items"]), \
            "a landmark question entered P-HAZ"
        assert man["dataset"]["counts_checked_against"], "ASSET-1 cross-check did not run"
        assert plant_same in man["gold_checks"]["answer_equals_question"]["P-CORE"]
        assert plant_count in man["gold_checks"]["P-CORE_count_items_with_a_form_used_once"]
        say("[selftest] OK  audit, merge and draw run end to end from a JSON file")
        say("[selftest] OK  P-HAZ holds no landmark question; the gold check finds both planted problems")

        say("[selftest] --- draw again, unchanged data: must succeed ---")
        cmd_draw(draw_args)
        say("[selftest] OK  a repeat draw reproduces both probes exactly")

        say("[selftest] --- draw against a tampered frozen P-CORE: must stop ---")
        pc_doc["items"] = pc_doc["items"][1:]
        with open(os.path.join(tmp, "P-CORE.json"), "w", encoding="utf-8") as f:
            json.dump(pc_doc, f, ensure_ascii=False)
        try:
            cmd_draw(draw_args)
            raise AssertionError("guard did not stop a changed probe")
        except SystemExit as e:
            say(f"[selftest] OK  the frozen-probe guard stopped the run: {str(e)[:60]}...")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    say()
    say("SELF-TEST PASSED: every house 30, every type 255, qa_id unique, 68 houses,")
    say("deterministic, text-independent; audit, merge, draw, gold check and the")
    say("frozen-probe guard all work end to end on synthetic data.")


def main():
    ap = argparse.ArgumentParser(description="ASSET-5A: draw P-CORE and P-HAZ")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest")
    s.add_argument("--counts", default="house_type_counts.csv")

    s = sub.add_parser("audit")
    s.add_argument("--data", required=True)
    s.add_argument("--counts", default="results/assets/house_type_counts.csv")
    s.add_argument("--out", default="results/assets")

    s = sub.add_parser("merge")
    s.add_argument("--audit", required=True)
    s.add_argument("--rater1", required=True)
    s.add_argument("--rater2", required=True)

    s = sub.add_parser("draw")
    s.add_argument("--data", required=True)
    s.add_argument("--counts", default="results/assets/house_type_counts.csv")
    s.add_argument("--out", default="results/assets")
    s.add_argument("--classes", default=None)
    s.add_argument("--force", action="store_true")

    args = ap.parse_args()
    {"selftest": cmd_selftest, "audit": cmd_audit,
     "merge": cmd_merge, "draw": cmd_draw}[args.cmd](args)


if __name__ == "__main__":
    main()
