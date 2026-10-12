#!/usr/bin/env python3
"""
asset12_nameref.py  (ASSET-12 and HT-3)
BanglaIndoorVQA main paper: the human naming reference.

Design v2.1 sections 4.8, 9.2, 9.5, 10 and 11 (Gate H); DESIGN_ADDENDUM C1, C4
and C5; decisions D205 to D214 (ASSET-12 chat).

What it does
    draw       NAME-REF: 200 P-NAME items, 50 per cell of (in P-CLUST) x
               (en_route), by seeded rank, after removing the namers' houses.
               Cells come from resolve_gold on the frozen lexicon, never from
               name_id or gold text (C5, D207). Also fixes each namer's order
               and item codes (D209) and writes the key.
    build      the two namer packages: one zip per namer with an Excel sheet
               and byte-identical copies of the prepared images (D210).
    check      format check of a returned sheet; never reads the lexicon or
               any scorer label (D211).
    receive    freezes a sheet that passes the check (Gate H evidence, D211).
    score      scores both sheets with the scorer: percent agreement and
               Cohen's kappa, house-clustered bootstrap CIs (D208, D212).
               --provisional before Gate F, --final only with the gate-F
               scorer and the gate-A lexicon.
    dryrun     the whole chain on the real inputs in a temporary folder, with
               stand-in names; nothing is kept.
    selftest   checks on invented data; needs no Drive.
    check-inputs  one PASS or FAIL line per input.
    status     what exists so far.

Rules kept here
    - Seed 42. One normalise(), imported from src/normalise.py. The scorer is
      imported from src/scoring.py, never copied.
    - Printouts show ids and counts only: never a question, a gold answer or
      a namer's name (D213). Files with text stay in Drive human/HT-3/.
    - The script reads only the house_id and annotator_id columns of
      house_assignment.csv (D206).
"""

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_VERSION = "asset12-2026-10-13"
SEED = 42
PER_CELL = 50
TOTAL = 200
N_BOOT = 10000
CONDITION = "N-BB"
ROLES = ("N1", "N2")
NA = "NA"

CELLS = ("PY-EY", "PY-EN", "PN-EY", "PN-EN")
CELL_TEXT = {
    "PY-EY": "in P-CLUST, en_route yes",
    "PY-EN": "in P-CLUST, en_route no",
    "PN-EY": "not in P-CLUST, en_route yes",
    "PN-EN": "not in P-CLUST, en_route no",
}
C1_CELLS = {"PY-EY": 316, "PY-EN": 176, "PN-EY": 297, "PN-EN": 211}
LABELS = ("tier1", "tier2", "sibling", "ambiguous", "english_copy",
          "other_wrong")

# Inputs as checked in the ASSET-12 chat (2026-10-12). A different file
# stops the run: ask before going on.
EXPECTED_SHA256 = {
    "P-NAME.json":
        "fc4bfb1dbda1f6f3a554e196dc049c90a9a72af4376170420f50b5decd7e9035",
    "P-CLUST.json":
        "7759dcce19e4c36596d6b39a39be8feefa467cf2088d5c03bdcace99a70f296f",
    "house_assignment.csv":
        "1dde22c04bddadae85f262e97c8429eb1b404ec22a4fe12050716287d478941c",
    "images_manifest.csv":
        "e726f6f3b10f2b57f75a2001bde98938321010b146e1502765d5076fa199738d",
    "lexicon.csv":
        "73b6bc48f1bda27863fbad69e26830c5f484d50702543b66297679bc07bc7448",
    "closed_vocab.csv":
        "de36f2905e43784704f8765c9886da1e9a9524d295ff10fac6c75b21f50e4b90",
}
EXPECTED_COUNTS = {"P-NAME": 1000, "P-CLUST": 492, "houses": 68,
                   "manifest_rows": 4406, "objects": 231}

REPO_DEFAULT = Path(__file__).resolve().parents[1]
DRIVE_DEFAULT = Path("/content/drive/MyDrive/BanglaIndoorVQA")

# --------------------------------------------------------------------------
# Texts the namers see (Bangla). The instruction is Design 4.8, as written.
# check-inputs and build confirm that no lexicon term appears in any of them.
# --------------------------------------------------------------------------

SHEET_INSTR = "নির্দেশনা"
SHEET_NAMES = "নাম"
SHEET_DECL = "ঘোষণা"
INSTRUCTION_LINES = (
    "ছবিতে মূল যে জিনিসটি দেখা যাচ্ছে, বাসায় আপনি সেটিকে যে নামে ডাকেন সেই নামটি লিখুন (এক বা দুই শব্দে)।",
    "নিশ্চিত না হলে যা মনে হয় তাই লিখুন। কোনো অভিধান, ইন্টারনেট বা অন্য কারও সাহায্য নেবেন না।",
)
TYPING_NOTE = ("লেখার নিয়ম: নামটি বাংলা অক্ষরে লিখুন, ইংরেজি বা রোমান অক্ষরে নয়। "
               "প্রতি সারিতে একটি নাম লিখুন। বাড়তি কিছু বলতে চাইলে পাশের মন্তব্যের ঘরে লিখুন।")
STEPS_TITLE = "কাজের ধাপ:"
STEPS = (
    "১. ছবির ফাইলের নাম আর সারির কোড মিলিয়ে নিন (যেমন {role}-001.jpg এবং {role}-001)।",
    "২. ছবিগুলো কোড অনুযায়ী ক্রমে, একটি একটি করে দেখুন।",
    "৩. সব নাম লেখা হলে \"ঘোষণা\" শিটে তিনটি উত্তর লিখুন, তারপর ফাইলটি সংরক্ষণ করে দলনেতাকে পাঠান।",
)
NAMES_HEADER = ("কোড", "নাম", "মন্তব্য (ঐচ্ছিক)")
DECL_HEADER = ("ক্রম", "ঘোষণা", "উত্তর (হ্যাঁ / না)")
DECLARATIONS = (
    ("D1", "এই কাজের আগে আমি কোনো এআই মডেলের দেওয়া উত্তর বা স্কোরারের ফলাফল দেখিনি।"),
    ("D2", "আমি কোনো অভিধান, ইন্টারনেট, এআই অ্যাপ, প্রকল্পের অন্য কোনো ফাইল বা অন্য কারও সাহায্য নিইনি।"),
    ("D3", "প্রতিটি ছবির নাম আমি নিজে, একা লিখেছি।"),
)

# Messages of the format check (D211). The same words are in the namers'
# guide, section "If you get a fix list".
E_EMPTY = "নাম লেখা হয়নি"
E_SCRIPT = "বাংলা অক্ষরে লিখুন"
E_DIGIT = "সংখ্যা নয়, শুধু নাম লিখুন"
E_TWO = "একটি নামই লিখুন; বাড়তি কথা মন্তব্যের ঘরে লিখুন"
E_GUESS = "নিশ্চিত না হলে যা মনে হয় তাই লিখুন"
E_MISSING = "এই কোডের সারি নেই"
E_DUP = "একই কোড দুবার আছে"
E_FOREIGN = "এই কোড আপনার শিটের নয়"
E_DECL = "ঘোষণার ঘরে হ্যাঁ বা না লিখুন"
EN_MESSAGE = {
    E_EMPTY: "empty", E_SCRIPT: "not in Bangla letters", E_DIGIT: "digits",
    E_TWO: "two names or extra words", E_GUESS: "a don't-know answer",
    E_MISSING: "row missing", E_DUP: "code twice",
    E_FOREIGN: "code not on this sheet", E_DECL: "declaration not filled",
}
ALL_PACKAGE_TEXTS = (INSTRUCTION_LINES + (TYPING_NOTE, STEPS_TITLE) + STEPS
                     + NAMES_HEADER + DECL_HEADER
                     + tuple(t for _, t in DECLARATIONS)
                     + (SHEET_INSTR, SHEET_NAMES, SHEET_DECL)
                     + tuple(EN_MESSAGE))

CODE_RE = re.compile(r"^(N[12])-(\d{3})$")
ID_LEAK_RE = re.compile(r"H\d{2}_\d{4}")
HOUSE_LEAK_RE = re.compile(r"(?<![A-Za-z0-9])H\d{2}(?![0-9])")
HOUSE_ID_RE = re.compile(r"^H\d{2}$")
DIGIT_RE = re.compile(r"[0-9০-৯]")
SEP_RE = re.compile(r"[/,;|\\()\[\]{}،]")
OR_WORDS = ("বা", "অথবা")
YES_ANSWERS = ("হ্যাঁ", "হ্যা", "হাঁ", "হা", "yes", "y")
NO_ANSWERS = ("না", "no", "n")
ZIP_TIME = (2026, 1, 1, 0, 0, 0)


class Ht3Error(Exception):
    """A rule of this asset is broken; the message says what to do."""


def say(msg=""):
    print(msg, flush=True)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_csv(path, header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow(r)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(buf.getvalue(), encoding="utf-8-sig")
    tmp.replace(path)


def read_csv_dicts(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def pct(x):
    return "-" if x is None else "%.1f%%" % (100.0 * x)


def num(x, digits=3):
    return "undefined" if x is None else ("%." + str(digits) + "f") % x


# --------------------------------------------------------------------------
# 1. The scorer and normalise(), imported from the repo
# --------------------------------------------------------------------------

def load_scorer(repo):
    src = str(Path(repo) / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import scoring  # noqa: E402  (the one scorer, never copied)
    from normalise import normalise  # noqa: E402  (the one normalise())
    return scoring, normalise


# --------------------------------------------------------------------------
# 2. Paths and inputs
# --------------------------------------------------------------------------

@dataclass
class Paths:
    repo: Path
    drive: Path
    lexicon: Path
    vocab: Path
    excluded: Path
    pname: Path
    pclust: Path
    houses: Path
    images: Path
    manifest: Path
    out: Path
    allow_other_inputs: bool = False

    @classmethod
    def from_args(cls, args):
        repo = Path(args.repo).resolve()
        drive = Path(args.drive)
        images = Path(args.images) if args.images else drive / "data" / "images_1024"
        return cls(
            repo=repo, drive=drive,
            lexicon=repo / "lexicon" / "lexicon.csv",
            vocab=repo / "lexicon" / "closed_vocab.csv",
            excluded=repo / "probes" / "excluded_items.csv",
            pname=Path(args.pname) if args.pname else drive / "probes" / "P-NAME.json",
            pclust=Path(args.pclust) if args.pclust else drive / "probes" / "P-CLUST.json",
            houses=Path(args.houses) if args.houses else drive / "data" / "house_assignment.csv",
            images=images,
            manifest=Path(args.manifest) if args.manifest else images / "images_manifest.csv",
            out=Path(args.out) if args.out else drive / "human" / "HT-3",
            allow_other_inputs=bool(getattr(args, "allow_other_inputs", False)))


def out_layout(out):
    out = Path(out)
    return {
        "manifest": out / "ht3_manifest.json",
        "draw_dir": out / "draw",
        "draw_csv": out / "draw" / "nameref_draw.csv",
        "draw_report": out / "draw" / "draw_report.txt",
        "key": out / "key" / "nameref_key.csv",
        "to_send": out / "to_send",
        "checks": out / "checks",
        "received": out / "received",
        "ids": out / "NAME-REF.ids.txt",
        "archive": out / "archive",
    }


def read_house_assignment(path):
    """house_id -> annotator_id. Only these two columns are taken; the name
    column is never stored or printed (D206)."""
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = [h.strip().lower() for h in next(reader)]
        try:
            ih, ia = header.index("house_id"), header.index("annotator_id")
        except ValueError:
            raise Ht3Error("house_assignment.csv needs the columns house_id"
                           " and annotator_id")
        for row in reader:
            if not row or not any(c.strip() for c in row):
                continue
            out[row[ih].strip()] = row[ia].strip()
    return out


def read_excluded(path):
    rows = read_csv_dicts(path)
    return {r["qa_id"].strip() for r in rows if r.get("qa_id", "").strip()}, len(rows)


def read_images_manifest(path):
    return {r["image_file"]: r for r in read_csv_dicts(path)}


@dataclass
class Item:
    qa_id: str
    image_id: str
    house_id: str
    image_file: str
    in_pclust: bool
    en_route: str
    gold_name_id: str
    generic_q: bool

    @property
    def cell(self):
        return (("PY" if self.in_pclust else "PN") + "-"
                + ("EY" if self.en_route == "yes" else "EN"))


@dataclass
class Inputs:
    paths: Paths
    sc: object
    normalise: object
    lex: object
    voc: object
    pname: list
    pname_by_id: dict
    pclust_ids: set
    excluded: set
    excluded_rows: int
    houses: dict
    items: list
    problems: list
    hashes: dict


def place_items(sc, normalise, lex, voc, pname, pclust_ids):
    """One Item per P-NAME record. The cell comes from resolve_gold on the
    frozen lexicon and from P-CLUST membership (C5, D207)."""
    qcount = Counter(normalise(it["question_bn"]) for it in pname)
    generic = qcount.most_common(1)[0][0] if qcount else ""
    items, problems = [], []
    for it in pname:
        gold = sc.resolve_gold(it, lex, voc)
        if gold.status != "ok":
            problems.append((it["qa_id"], "gold " + gold.status))
            continue
        obj = lex.objects[gold.name_id]
        if obj.en_route is None:
            problems.append((it["qa_id"], "en_route empty"))
            continue
        items.append(Item(
            qa_id=it["qa_id"], image_id=it["image_id"],
            house_id=it["house_id"], image_file=it["image_file"],
            in_pclust=it["qa_id"] in pclust_ids,
            en_route="yes" if obj.en_route else "no",
            gold_name_id=gold.name_id,
            generic_q=normalise(it["question_bn"]) == generic))
    return items, problems


def input_hash_problems(paths, hashes):
    """Compare input files with the values checked in the ASSET-12 chat."""
    bad = []
    for name, want in EXPECTED_SHA256.items():
        got = hashes.get(name)
        if got is not None and got != want:
            bad.append("%s: sha256 %s..., expected %s..." % (name, got[:12], want[:12]))
    return bad


def load_inputs(paths, need_manifest=False):
    sc, normalise = load_scorer(paths.repo)
    lex = sc.load_lexicon(paths.lexicon)
    voc = sc.load_closed_vocab(paths.vocab)
    pname = sc.load_probe(paths.pname)
    pclust = sc.load_probe(paths.pclust)
    excluded, n_excl = read_excluded(paths.excluded)
    houses = read_house_assignment(paths.houses)
    hashes = {
        "P-NAME.json": sha256_file(paths.pname),
        "P-CLUST.json": sha256_file(paths.pclust),
        "house_assignment.csv": sha256_file(paths.houses),
        "lexicon.csv": sha256_file(paths.lexicon),
        "closed_vocab.csv": sha256_file(paths.vocab),
        "excluded_items.csv": sha256_file(paths.excluded),
        "scoring.py": sha256_file(paths.repo / "src" / "scoring.py"),
        "normalise.py": sha256_file(paths.repo / "src" / "normalise.py"),
    }
    if need_manifest:
        hashes["images_manifest.csv"] = sha256_file(paths.manifest)
    bad = input_hash_problems(paths, hashes)
    if bad:
        msg = "Input files differ from the ones checked in the ASSET-12 chat:\n  " + "\n  ".join(bad)
        if not paths.allow_other_inputs:
            raise Ht3Error(msg + "\nStop and ask in the ASSET-12 chat before going on.")
        say("WARN " + msg + "\nWARN --allow-other-inputs is set: test run only, never the real run.")
    pclust_ids = {it["qa_id"] for it in pclust}
    items, problems = place_items(sc, normalise, lex, voc, pname, pclust_ids)
    return Inputs(paths=paths, sc=sc, normalise=normalise, lex=lex, voc=voc,
                  pname=pname, pname_by_id={it["qa_id"]: it for it in pname},
                  pclust_ids=pclust_ids, excluded=excluded,
                  excluded_rows=n_excl, houses=houses, items=items,
                  problems=problems, hashes=hashes)


# --------------------------------------------------------------------------
# 3. The draw (D206, D207)
# --------------------------------------------------------------------------

def rank_key(qa_id):
    return hashlib.sha256(("%d|NAME-REF|%s" % (SEED, qa_id)).encode("utf-8")).hexdigest()


def order_key(role, qa_id):
    return hashlib.sha256(("%d|NAME-REF|order|%s|%s" % (SEED, role, qa_id)).encode("utf-8")).hexdigest()


def parse_house_list(text, houses):
    out = []
    for part in (text or "").replace(";", ",").split(","):
        h = part.strip().upper()
        if not h:
            continue
        if not HOUSE_ID_RE.match(h) or h not in houses:
            raise Ht3Error("house %r is not in house_assignment.csv" % part.strip())
        if h not in out:
            out.append(h)
    return out


def namer_houses(namer1, namer2, also1, also2, houses):
    """The houses to exclude: both namers' houses plus any house a namer
    photographed, visited or checked (D206)."""
    ids = sorted(set(houses.values()))
    for n in (namer1, namer2):
        if n != NA and n not in ids:
            raise Ht3Error("namer id %r is not an annotator_id in"
                           " house_assignment.csv (known: %s) and is not NA"
                           % (n, ", ".join(ids)))
    if namer1 == namer2 and namer1 != NA:
        raise Ht3Error("the two namers must be different people (both %s)" % namer1)
    per_role = {}
    for role, n, also in (("N1", namer1, also1), ("N2", namer2, also2)):
        own = sorted(h for h, a in houses.items() if n != NA and a == n)
        extra = parse_house_list(also, houses)
        per_role[role] = {"annotator_id": n, "houses": own, "also_exclude": extra}
    excluded = sorted(set(per_role["N1"]["houses"]) | set(per_role["N1"]["also_exclude"])
                      | set(per_role["N2"]["houses"]) | set(per_role["N2"]["also_exclude"]))
    return per_role, excluded


@dataclass
class Drawn:
    item: Item
    rank: str
    drawn_as: str          # regular or fill
    fill_for: str          # the short cell, for a filled item


@dataclass
class DrawResult:
    drawn: list
    stats: dict
    skipped_excluded: list
    skipped_house: int
    skipped_same_image: list


def draw_nameref(items, excluded_qa, excluded_houses, per_cell=PER_CELL,
                 total=TOTAL):
    """Seeded rank within each cell; a short cell takes all it has and the
    rest comes from the cell with the most available items (D207)."""
    if per_cell * len(CELLS) != total:
        raise Ht3Error("per_cell x 4 must equal total")
    before = Counter(it.cell for it in items)
    avail = {c: [] for c in CELLS}
    skipped_excluded, skipped_house = [], 0
    for it in items:
        if it.qa_id in excluded_qa:
            skipped_excluded.append(it.qa_id)
            continue
        if it.house_id in excluded_houses:
            skipped_house += 1
            continue
        avail[it.cell].append(it)
    for c in CELLS:
        avail[c].sort(key=lambda it: rank_key(it.qa_id))
    n_after = {c: len(avail[c]) for c in CELLS}
    pos = {c: 0 for c in CELLS}
    used_images, same_image = set(), []
    drawn = []

    def take_next(c):
        while pos[c] < len(avail[c]):
            it = avail[c][pos[c]]
            pos[c] += 1
            if it.image_id in used_images:     # D209: one item per image
                same_image.append(it.qa_id)
                continue
            used_images.add(it.image_id)
            return it
        return None

    taken = Counter()
    for c in CELLS:
        while taken[c] < per_cell:
            it = take_next(c)
            if it is None:
                break
            drawn.append(Drawn(it, rank_key(it.qa_id), "regular", ""))
            taken[c] += 1
    short = {c: per_cell - taken[c] for c in CELLS if taken[c] < per_cell}
    filled_from = defaultdict(Counter)
    for c_short in CELLS:
        need = short.get(c_short, 0)
        while need > 0:
            cands = [c for c in CELLS if pos[c] < len(avail[c])]
            if not cands:
                raise Ht3Error("fewer than %d items are available after the"
                               " house exclusion; stop and decide with a"
                               " decision row" % total)
            src = sorted(cands, key=lambda c: (-n_after[c], CELLS.index(c)))[0]
            it = take_next(src)
            if it is None:
                continue
            drawn.append(Drawn(it, rank_key(it.qa_id), "fill", c_short))
            filled_from[c_short][src] += 1
            need -= 1
    if len(drawn) != total:
        raise Ht3Error("draw produced %d items, expected %d" % (len(drawn), total))
    stats = {}
    for c in CELLS:
        in_cell = [d for d in drawn if d.item.cell == c]
        objs = Counter(d.item.gold_name_id for d in in_cell)
        stats[c] = {
            "before_exclusion": before.get(c, 0), "after_exclusion": n_after[c],
            "drawn_regular": taken[c], "short": short.get(c, 0),
            "filled_from": dict(filled_from.get(c, {})),
            "in_cell_final": len(in_cell),
            "distinct_objects": len(objs),
            "largest_object_count": max(objs.values()) if objs else 0,
        }
    return DrawResult(drawn, stats, skipped_excluded, skipped_house, same_image)


def assign_codes(qa_ids, role):
    ordered = sorted(qa_ids, key=lambda q: order_key(role, q))
    return [("%s-%03d" % (role, i), q) for i, q in enumerate(ordered, 1)]


KEY_HEADER = ("role", "code", "qa_id", "image_id", "image_file", "house_id",
              "cell", "in_pclust", "en_route", "gold_name_id", "rank16",
              "order16", "drawn_as", "fill_for")


def key_rows(drawn):
    by_qa = {d.item.qa_id: d for d in drawn}
    rows = []
    for role in ROLES:
        for code, qa in assign_codes(list(by_qa), role):
            d = by_qa[qa]
            it = d.item
            rows.append((role, code, qa, it.image_id, it.image_file,
                         it.house_id, it.cell,
                         "yes" if it.in_pclust else "no", it.en_route,
                         it.gold_name_id, d.rank[:16],
                         order_key(role, qa)[:16], d.drawn_as, d.fill_for))
    return rows


def read_key(path):
    rows = read_csv_dicts(path)
    if not rows:
        raise Ht3Error("the key %s is empty" % path)
    return rows


# --------------------------------------------------------------------------
# 4. Images: hashes and metadata (D210)
# --------------------------------------------------------------------------

def jpeg_segments(data):
    """(marker, first bytes of the segment) for every segment before the
    image data."""
    if data[:2] != b"\xff\xd8":
        raise Ht3Error("not a JPEG file")
    i, out = 2, []
    n = len(data)
    while i < n:
        if data[i] != 0xFF:
            raise Ht3Error("JPEG structure not readable")
        while i < n and data[i] == 0xFF:
            i += 1
        if i >= n:
            break
        marker = data[i]
        i += 1
        if marker in (0xD9, 0xDA):           # end of image, start of scan
            break
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:
            continue
        seglen = int.from_bytes(data[i:i + 2], "big")
        out.append((marker, bytes(data[i + 2:i + 2 + min(seglen - 2, 8)])))
        i += seglen
    return out


def jpeg_metadata_blocks(data):
    """Metadata blocks other than the plain JFIF header: EXIF, XMP, ICC,
    IPTC, thumbnails, comments."""
    found = []
    for marker, head in jpeg_segments(data):
        if marker == 0xE0 and head.startswith(b"JFIF\x00"):
            continue
        if 0xE0 <= marker <= 0xEF:
            found.append("APP%d" % (marker - 0xE0))
        elif marker == 0xFE:
            found.append("COM")
    return found


# --------------------------------------------------------------------------
# 5. The namer workbook (D210)
# --------------------------------------------------------------------------

def make_workbook_bytes(role, codes):
    import openpyxl
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = SHEET_INSTR
    ws.column_dimensions["A"].width = 110
    lines = [(SHEET_INSTR, True, 16), ("", False, 12)]
    lines += [(t, False, 14) for t in INSTRUCTION_LINES]
    lines += [("", False, 12), (TYPING_NOTE, False, 13), ("", False, 12),
              (STEPS_TITLE, True, 13)]
    lines += [(s.format(role=role), False, 13) for s in STEPS]
    for r, (text, bold, size) in enumerate(lines, 1):
        cell = ws.cell(row=r, column=1, value=text if text else None)
        cell.font = Font(bold=bold, size=size)
        cell.alignment = Alignment(wrap_text=True, vertical="top")

    ws2 = wb.create_sheet(SHEET_NAMES)
    ws2.append(list(NAMES_HEADER))
    for c in ws2[1]:
        c.font = Font(bold=True, size=13)
    for code, _ in codes:
        ws2.append([code, None, None])
    ws2.column_dimensions["A"].width = 12
    ws2.column_dimensions["B"].width = 40
    ws2.column_dimensions["C"].width = 40
    for row in ws2.iter_rows(min_row=2, max_row=len(codes) + 1):
        row[0].font = Font(size=13)
        for c in row[1:]:
            c.font = Font(size=14)
            c.number_format = "@"
    ws2.freeze_panes = "A2"

    ws3 = wb.create_sheet(SHEET_DECL)
    ws3.append(list(DECL_HEADER))
    for c in ws3[1]:
        c.font = Font(bold=True, size=13)
    for did, text in DECLARATIONS:
        ws3.append([did, text, None])
    ws3.column_dimensions["A"].width = 8
    ws3.column_dimensions["B"].width = 90
    ws3.column_dimensions["C"].width = 20
    for row in ws3.iter_rows(min_row=2, max_row=len(DECLARATIONS) + 1):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
        for c in row:
            c.font = Font(size=13)
    dv = DataValidation(type="list", formula1='"হ্যাঁ,না"', allow_blank=True)
    ws3.add_data_validation(dv)
    dv.add("C2:C%d" % (len(DECLARATIONS) + 1))

    wb.properties.creator = "HT-3"
    wb.properties.lastModifiedBy = "HT-3"
    wb.properties.title = "HT-3 %s" % role
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def workbook_texts(data):
    """(where, text) for every sheet name, cell and document property."""
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data))
    out = []
    for ws in wb.worksheets:
        out.append(("sheet name", ws.title))
        for row in ws.iter_rows():
            for c in row:
                if c.value is not None:
                    out.append(("%s!%s" % (ws.title, c.coordinate), str(c.value)))
    p = wb.properties
    for k in ("title", "subject", "creator", "keywords", "description",
              "lastModifiedBy", "category"):
        v = getattr(p, k, None)
        if v:
            out.append(("property " + k, str(v)))
    return out


def leak_scan(texts, entry_names, sc, normalise, lex, forbidden_texts):
    """Nothing a namer receives may hold a qa_id, a house or image id, any
    question or gold text, or any term of the lexicon (D210)."""
    problems = []
    for where, t in list(texts) + [("zip entry", e) for e in entry_names]:
        if ID_LEAK_RE.search(t):
            problems.append(("id pattern", where))
        if HOUSE_LEAK_RE.search(t):
            problems.append(("house id pattern", where))
    norm_texts = [(w, normalise(t)) for w, t in texts]
    tok_texts = [(w, [k.text for k in sc.tokenise(t)]) for w, t in norm_texts]
    for f in forbidden_texts:
        needle = [k.text for k in sc.tokenise(normalise(f))] if f else []
        if not needle:
            continue
        for w, hay in tok_texts:
            if contains_words(hay, needle):
                problems.append(("question or gold text", w))
    for w, t in norm_texts:
        if lex.index.find_all(sc.tokenise(t)):
            problems.append(("lexicon term", w))
    return problems


def contains_words(hay, needle):
    """True if the words of needle appear side by side, as whole words, in hay."""
    k = len(needle)
    return any(hay[i:i + k] == needle for i in range(len(hay) - k + 1))


def package_text_problems(sc, normalise, lex):
    """Lexicon terms or id patterns in the fixed texts (check-inputs)."""
    probs = []
    for t in ALL_PACKAGE_TEXTS:
        if lex.index.find_all(sc.tokenise(normalise(t.format(role="N1")))):
            probs.append(t[:20])
        if ID_LEAK_RE.search(t) or HOUSE_LEAK_RE.search(t):
            probs.append(t[:20])
    return probs


def write_zip(path, role, xlsx_bytes, images):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    names = []
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
        def add(name, data):
            zi = zipfile.ZipInfo(name, date_time=ZIP_TIME)
            zi.external_attr = 0o644 << 16
            z.writestr(zi, data)
            names.append(name)
        add("HT3_%s/HT3_%s.xlsx" % (role, role), xlsx_bytes)
        for code, data in images:
            add("HT3_%s/images/%s.jpg" % (role, code), data)
    tmp.replace(path)
    return names


# --------------------------------------------------------------------------
# 6. Reading and checking a returned sheet (D211)
# --------------------------------------------------------------------------

def _clean_answer(value, normalise):
    return normalise("" if value is None else str(value)).replace("‌", "").replace("‍", "")


def read_returned(path):
    """Rows of the names sheet and the declaration sheet, found by content."""
    import openpyxl
    try:
        wb = openpyxl.load_workbook(path, data_only=True)
    except Exception as exc:            # not an xlsx file, or damaged
        raise Ht3Error("cannot open %s as an Excel .xlsx file (%s). Ask the"
                       " namer to save it as .xlsx." % (Path(path).name, type(exc).__name__))
    names_ws = decl_ws = None
    for ws in wb.worksheets:
        col_a = [ws.cell(row=r, column=1).value for r in range(1, ws.max_row + 1)]
        texts = [str(v).strip() for v in col_a if v is not None]
        if names_ws is None and any(CODE_RE.match(t) for t in texts):
            names_ws = ws
        if decl_ws is None and any(t in ("D1", "D2", "D3") for t in texts):
            decl_ws = ws
    name_col, note_col = 2, 3
    rows = []
    if names_ws is not None:
        for r in range(1, names_ws.max_row + 1):
            a = names_ws.cell(row=r, column=1).value
            if a is not None and str(a).strip() == NAMES_HEADER[0]:
                for c in range(2, names_ws.max_column + 1):
                    h = names_ws.cell(row=r, column=c).value
                    if h is not None and str(h).strip() == NAMES_HEADER[1]:
                        name_col = c
                    elif h is not None and str(h).strip().startswith("মন্তব্য"):
                        note_col = c
                break
        for r in range(1, names_ws.max_row + 1):
            a = names_ws.cell(row=r, column=1).value
            if a is None or not CODE_RE.match(str(a).strip()):
                continue
            rows.append((str(a).strip(), names_ws.cell(row=r, column=name_col).value,
                         names_ws.cell(row=r, column=note_col).value))
    decl = {}
    if decl_ws is not None:
        for r in range(1, decl_ws.max_row + 1):
            a = decl_ws.cell(row=r, column=1).value
            if a is not None and str(a).strip() in ("D1", "D2", "D3"):
                decl[str(a).strip()] = decl_ws.cell(row=r, column=3).value
    return names_ws is not None, rows, decl_ws is not None, decl


def name_problems(raw, sc, normalise):
    """Format only: never compares a name with the lexicon or the gold."""
    text = "" if raw is None else str(raw)
    n = normalise(text)
    if not n:
        return [E_EMPTY]
    lang = sc.detect_language(sc.tokenise(n))
    probs = []
    if lang["lat_letters"] > 0 or lang["other_letters"] > 0:
        probs.append(E_SCRIPT)
    if DIGIT_RE.search(n):
        probs.append(E_DIGIT)
    if SEP_RE.search(n) or any(w in OR_WORDS for w in n.split()):
        probs.append(E_TWO)
    prep = sc.prepare_answer(text)
    if sc._has_refusal_phrase(prep.norm_tokens):
        probs.append(E_GUESS)
    if not probs and lang["bn_letters"] == 0:
        probs.append(E_EMPTY)
    return probs


@dataclass
class CheckResult:
    role: str
    passed: bool
    errors: list = field(default_factory=list)          # (code, Bangla message)
    stop_errors: list = field(default_factory=list)     # for the runner only
    names: dict = field(default_factory=dict)            # code -> raw name
    notes: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


def check_returned(path, role, expected_codes, sc, normalise):
    res = CheckResult(role=role, passed=False)
    found_names, rows, found_decl, decl = read_returned(path)
    if not found_names:
        res.stop_errors.append("no sheet with the codes %s-001 to %s-%03d was found"
                               % (role, role, len(expected_codes)))
        return res
    expected = set(expected_codes)
    seen = Counter(code for code, _, _ in rows)
    other_role = [code for code in seen if not code.startswith(role + "-")]
    if other_role and not any(code.startswith(role + "-") for code in seen):
        res.stop_errors.append("this file holds the codes of the other namer;"
                               " check that --role is right")
        return res
    for code, name, note in rows:
        if code not in expected:
            res.errors.append((code, E_FOREIGN))
            continue
        if seen[code] > 1:
            continue
        res.names[code] = "" if name is None else str(name)
        res.notes[code] = "" if note is None else str(note)
        for p in name_problems(name, sc, normalise):
            res.errors.append((code, p))
    for code in sorted(c for c, k in seen.items() if k > 1 and c in expected):
        res.errors.append((code, E_DUP))
    for code in sorted(expected - set(seen)):
        res.errors.append((code, E_MISSING))
    if not found_decl:
        res.errors.append(("D1-D3", E_DECL))
    else:
        for did, _ in DECLARATIONS:
            a = _clean_answer(decl.get(did), normalise)
            if a in NO_ANSWERS:
                res.stop_errors.append(
                    "declaration %s is answered না: do not receive this sheet;"
                    " talk to the namer and decide with a decision row" % did)
            elif a not in YES_ANSWERS:
                res.errors.append((did, E_DECL))
    res.errors.sort(key=lambda e: (e[0].startswith("D"), e[0]))
    filled = [c for c in res.names if normalise(res.names[c])]
    res.info = {
        "rows_with_codes": len(rows), "names_filled": len(filled),
        "notes_filled": sum(1 for c in res.notes if normalise(res.notes[c])),
        "names_over_two_words": sum(1 for c in filled
                                    if len(normalise(res.names[c]).split()) > 2),
    }
    res.passed = not res.errors and not res.stop_errors
    return res


def report_check(res, path_name):
    lines = ["HT-3 format check (D211)  script %s  UTC %s" % (SCRIPT_VERSION, utc_now()),
             "Role %s, file %s" % (res.role, path_name),
             "Result: %s" % ("PASS" if res.passed else "FAIL"),
             "Rows with codes %(rows_with_codes)d, names filled %(names_filled)d,"
             " notes filled %(notes_filled)d, names over two words"
             " %(names_over_two_words)d (information only, never sent back)" % res.info
             if res.info else ""]
    for e in res.stop_errors:
        lines.append("STOP  " + e)
    by_kind = Counter(m for _, m in res.errors)
    for m, k in by_kind.items():
        lines.append("  %-28s %d" % (EN_MESSAGE.get(m, m), k))
    lines.append("")
    lines.append("Rows to fix (code: message in English):")
    for code, m in res.errors:
        lines.append("  %s: %s" % (code, EN_MESSAGE.get(m, m)))
    return "\n".join(lines) + "\n"


def fix_list_text(res):
    return "\n".join("%s: %s" % (code, m) for code, m in res.errors) + "\n"


# --------------------------------------------------------------------------
# 7. Agreement statistics (Design 9.2, 9.5; D212)
# --------------------------------------------------------------------------

def cohen_kappa(a, b):
    """(observed agreement, kappa). kappa is None when chance agreement is 1
    (both raters used one and the same category throughout)."""
    n = len(a)
    if n == 0 or n != len(b):
        return None, None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if abs(1.0 - pe) < 1e-12:
        return po, None
    return po, (po - pe) / (1.0 - pe)


def percentile(sorted_vals, q):
    """Linear interpolation between ranks (numpy's default)."""
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * q / 100.0
    f = math.floor(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def agreement_stats(rows):
    """rows: (house, label_N1, label_N2, strict_N1, strict_N2)."""
    n = len(rows)
    if n == 0:
        return {}
    a, b = [r[1] for r in rows], [r[2] for r in rows]
    sa, sb = [r[3] for r in rows], [r[4] for r in rows]
    po6, k6 = cohen_kappa(a, b)
    po2, k2 = cohen_kappa(sa, sb)
    return {"agree6": po6, "kappa6": k6, "agree_strict": po2,
            "kappa_strict": k2, "strict_N1": sum(sa) / n,
            "strict_N2": sum(sb) / n, "strict_pooled": (sum(sa) + sum(sb)) / (2.0 * n)}


def bootstrap(rows, n_boot=N_BOOT, seed=SEED):
    """House-clustered bootstrap (Design 9.2): resample houses with
    replacement, recompute every quantity, 2.5th and 97.5th percentiles."""
    by_house = defaultdict(list)
    for r in rows:
        by_house[r[0]].append(r)
    houses = sorted(by_house)
    rng = random.Random(seed)
    vals, skipped = defaultdict(list), Counter()
    for _ in range(n_boot):
        sample = []
        for _ in range(len(houses)):
            sample.extend(by_house[houses[rng.randrange(len(houses))]])
        for k, v in agreement_stats(sample).items():
            if v is None:
                skipped[k] += 1
            else:
                vals[k].append(v)
    ci = {}
    for k, v in vals.items():
        v.sort()
        ci[k] = (percentile(v, 2.5), percentile(v, 97.5))
    return ci, dict(skipped), len(houses)


# --------------------------------------------------------------------------
# 8. Git checks for the final scores (D208)
# --------------------------------------------------------------------------

def git_bytes(repo, rev_path):
    r = subprocess.run(["git", "-C", str(repo), "show", rev_path],
                       capture_output=True)
    return r.stdout if r.returncode == 0 else None


def git_out(repo, *args):
    r = subprocess.run(["git", "-C", str(repo)] + list(args),
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def frozen_identity(repo, paths):
    """True for each file that equals its tagged version."""
    checks = (("gate-F", "src/scoring.py", Path(repo) / "src" / "scoring.py"),
              ("gate-F", "src/normalise.py", Path(repo) / "src" / "normalise.py"),
              ("gate-A", "lexicon/lexicon.csv", paths.lexicon),
              ("gate-A", "lexicon/closed_vocab.csv", paths.vocab))
    out = []
    for tag, rel, local in checks:
        tagged = git_bytes(repo, "%s:%s" % (tag, rel))
        ok = tagged is not None and Path(local).read_bytes() == tagged
        out.append((tag, rel, ok, "tag missing" if tagged is None else
                    ("identical" if ok else "differs from the tagged file")))
    return out


# --------------------------------------------------------------------------
# 9. Manifest (ids, counts, hashes, times; no text)
# --------------------------------------------------------------------------

def load_manifest(out):
    p = out_layout(out)["manifest"]
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"asset": "ASSET-12", "script_version": SCRIPT_VERSION,
            "entries": [], "state": {}}


def save_manifest(out, man):
    man["script_version_last"] = SCRIPT_VERSION
    write_text(out_layout(out)["manifest"], json.dumps(man, ensure_ascii=False, indent=1))


def add_entry(man, command, **info):
    e = {"command": command, "utc": utc_now(), "script_version": SCRIPT_VERSION}
    e.update(info)
    man["entries"].append(e)


def require_key(out, man):
    lay = out_layout(out)
    if "draw" not in man.get("state", {}) or not lay["key"].exists():
        raise Ht3Error("no draw yet: run draw first")
    got = sha256_file(lay["key"])
    want = man["state"]["draw"]["key_sha256"]
    if got != want:
        raise Ht3Error("the key file changed since the draw (sha256 %s..., expected"
                       " %s...). Never edit it." % (got[:12], want[:12]))
    return read_key(lay["key"])


def received_roles(man):
    return sorted(man.get("state", {}).get("received", {}))


# --------------------------------------------------------------------------
# 10. Commands
# --------------------------------------------------------------------------

def do_draw(inp, out, namer1, namer2, also1, also2, redraw_reason=None,
            lexicon_writer="none"):
    lay = out_layout(out)
    man = load_manifest(out)
    if inp.problems:
        raise Ht3Error("%d P-NAME golds do not resolve or lack en_route (first:"
                       " %s). Stop and ask." % (len(inp.problems), inp.problems[0][0]))
    if "draw" in man["state"]:
        if not redraw_reason:
            raise Ht3Error("a draw exists already (%s). It is redone only before any"
                           " sheet is received, with --redraw \"<reason>\" (D207)."
                           % man["state"]["draw"]["utc"])
        if received_roles(man):
            raise Ht3Error("a sheet is already received; the draw never changes now (D207)")
        dest = lay["archive"] / utc_stamp()
        dest.mkdir(parents=True, exist_ok=True)
        for k in ("draw_dir", "to_send"):
            if lay[k].exists():
                shutil.move(str(lay[k]), str(dest / lay[k].name))
        if lay["key"].parent.exists():
            shutil.move(str(lay["key"].parent), str(dest / "key"))
        write_text(dest / "reason.txt", redraw_reason.strip() + "\n")
        man["state"].pop("build", None)
        say("Old draw moved to %s" % dest)
    if lexicon_writer not in ("none",) + ROLES:
        raise Ht3Error("--lexicon-writer must be N1, N2 or none")
    per_role, excl_houses = namer_houses(namer1, namer2, also1, also2, inp.houses)
    res = draw_nameref(inp.items, inp.excluded, set(excl_houses))
    drawn = sorted(res.drawn, key=lambda d: d.rank)
    draw_rows = [(d.item.qa_id, d.item.image_id, d.item.image_file,
                  d.item.house_id, d.item.cell,
                  "yes" if d.item.in_pclust else "no", d.item.en_route,
                  d.item.gold_name_id, d.rank[:16], d.drawn_as, d.fill_for,
                  "yes" if d.item.generic_q else "no") for d in drawn]
    write_csv(lay["draw_csv"], ("qa_id", "image_id", "image_file", "house_id",
                                "cell", "in_pclust", "en_route", "gold_name_id",
                                "rank16", "drawn_as", "fill_for",
                                "generic_question"), draw_rows)
    write_csv(lay["key"], KEY_HEADER, key_rows(drawn))
    houses_drawn = Counter(d.item.house_id for d in drawn)
    not_generic = sorted(d.item.qa_id for d in drawn if not d.item.generic_q)
    rep = ["ASSET-12 NAME-REF draw (D206, D207)",
           "script %s, seed %d, UTC %s" % (SCRIPT_VERSION, SEED, utc_now()),
           "Rank: SHA-256(\"%d|NAME-REF|<qa_id>\"), lowest first; %d per cell" % (SEED, PER_CELL),
           ""]
    for role in ROLES:
        pr = per_role[role]
        rep.append("%s = %s: own houses %d, also excluded %s" % (
            role, pr["annotator_id"], len(pr["houses"]),
            ",".join(pr["also_exclude"]) or "none"))
    rep.append("Namer who helped write the lexicon (D214): %s" % lexicon_writer)
    rep.append("Excluded houses (%d): %s" % (len(excl_houses), ",".join(excl_houses) or "none"))
    rep.append("P-NAME items %d; golds resolved %d; on the excluded list %d; from"
               " excluded houses %d; same-image skips %d"
               % (len(inp.pname), len(inp.items), len(res.skipped_excluded),
                  res.skipped_house, len(res.skipped_same_image)))
    rep.append("")
    rep.append("%-6s %-30s %7s %7s %7s %6s  %s" % ("cell", "", "before", "after",
                                                   "drawn", "short", "filled from"))
    for c in CELLS:
        s = res.stats[c]
        ff = ",".join("%s %d" % (k, v) for k, v in s["filled_from"].items()) or "-"
        rep.append("%-6s %-30s %7d %7d %7d %6d  %s" % (
            c, CELL_TEXT[c], s["before_exclusion"], s["after_exclusion"],
            s["drawn_regular"], s["short"], ff))
    rep.append("")
    rep.append("Final count by true cell: " + ", ".join(
        "%s %d" % (c, res.stats[c]["in_cell_final"]) for c in CELLS))
    rep.append("Distinct objects (largest object count): " + ", ".join(
        "%s %d (%d)" % (c, res.stats[c]["distinct_objects"],
                        res.stats[c]["largest_object_count"]) for c in CELLS))
    rep.append("Houses in NAME-REF: %d; items per house: max %d, min %d"
               % (len(houses_drawn), max(houses_drawn.values()), min(houses_drawn.values())))
    rep.append("Items whose question is not the generic P-NAME wording: %d%s"
               % (len(not_generic), (" (" + ", ".join(not_generic) + ")") if not_generic else ""))
    rep.append("")
    rep.append("Inputs (sha256): " + ", ".join("%s %s" % (k, v[:12]) for k, v in sorted(inp.hashes.items())))
    rep.append("Files: %s (sha256 %s), %s (sha256 %s)" % (
        lay["draw_csv"].name, sha256_file(lay["draw_csv"])[:16],
        lay["key"].name, sha256_file(lay["key"])[:16]))
    write_text(lay["draw_report"], "\n".join(rep) + "\n")
    man["state"]["draw"] = {
        "utc": utc_now(), "namers": per_role, "excluded_houses": excl_houses,
        "n": len(drawn), "cells": res.stats, "houses_in_nameref": len(houses_drawn),
        "not_generic_question": len(not_generic),
        "draw_sha256": sha256_file(lay["draw_csv"]),
        "key_sha256": sha256_file(lay["key"]), "inputs": inp.hashes,
        "redraw_reason": redraw_reason or "", "lexicon_writer": lexicon_writer}
    add_entry(man, "draw", namers={r: per_role[r]["annotator_id"] for r in ROLES},
              also_exclude={r: per_role[r]["also_exclude"] for r in ROLES},
              excluded_houses=len(excl_houses), n=len(drawn),
              redraw_reason=redraw_reason or "")
    save_manifest(out, man)
    return res, rep


def do_build(inp, out, manifest_rows):
    lay = out_layout(out)
    man = load_manifest(out)
    key = require_key(out, man)
    if received_roles(man):
        raise Ht3Error("a sheet is already received; the packages are not rebuilt")
    sc, normalise, lex = inp.sc, inp.normalise, inp.lex
    by_role = defaultdict(list)
    for r in key:
        by_role[r["role"]].append(r)
    image_cache, checked = {}, Counter()
    for r in key:
        f = r["image_file"]
        if f in image_cache:
            continue
        m = manifest_rows.get(f)
        if m is None:
            raise Ht3Error("an image of the draw is not in images_manifest.csv (%s)" % r["qa_id"])
        p = inp.paths.images / f
        if not p.exists():
            raise Ht3Error("an image of the draw is missing in %s (%s)" % (inp.paths.images, r["qa_id"]))
        data = p.read_bytes()
        if sha256_bytes(data) != m["sha256"]:
            raise Ht3Error("an image differs from images_manifest.csv (%s); the"
                           " namers must see the prepared copy (D148, D210)" % r["qa_id"])
        meta = jpeg_metadata_blocks(data)
        if meta:
            raise Ht3Error("an image holds metadata %s (%s); stop and ask" % (meta, r["qa_id"]))
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            size = im.size
        if str(size[0]) != str(m.get("width", size[0])) or str(size[1]) != str(m.get("height", size[1])):
            raise Ht3Error("an image size differs from the manifest (%s)" % r["qa_id"])
        image_cache[f] = data
        checked["images"] += 1
    forbidden = set()
    for r in key:
        it = inp.pname_by_id[r["qa_id"]]
        forbidden.add(normalise(it["question_bn"]))
        forbidden.add(normalise(it["answer_bn"]))
    staging = lay["to_send"].with_name("to_send.tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    report = ["ASSET-12 build (D209, D210)  script %s  UTC %s" % (SCRIPT_VERSION, utc_now()),
              "Images checked against images_manifest.csv: %d distinct, sha256 equal,"
              " no metadata block" % checked["images"]]
    built = {}
    for role in ROLES:
        rows = sorted(by_role[role], key=lambda r: r["code"])
        codes = [(r["code"], r["qa_id"]) for r in rows]
        if len(codes) != TOTAL:
            raise Ht3Error("the key holds %d codes for %s, expected %d" % (len(codes), role, TOTAL))
        xlsx = make_workbook_bytes(role, codes)
        images = [(r["code"], image_cache[r["image_file"]]) for r in rows]
        entries = ["HT3_%s/HT3_%s.xlsx" % (role, role)] + [
            "HT3_%s/images/%s.jpg" % (role, c) for c, _ in images]
        probs = leak_scan(workbook_texts(xlsx), entries, sc, normalise, lex, forbidden)
        if probs:
            kinds = Counter(k for k, _ in probs)
            raise Ht3Error("leak scan FAILED for %s: %s (first at %s). Nothing was"
                           " written for sending." % (role, dict(kinds), probs[0][1]))
        zpath = staging / ("HT3_%s.zip" % role)
        write_zip(zpath, role, xlsx, images)
        (staging / ("HT3_%s_blank.xlsx" % role)).write_bytes(xlsx)
        built[role] = {"zip_sha256": sha256_file(zpath), "xlsx_sha256": sha256_bytes(xlsx),
                       "images": len(images), "zip_bytes": zpath.stat().st_size}
        report.append("%s: %d images, %d codes, leak scan PASS, zip %s (%.1f MB, sha256 %s...)"
                      % (role, len(images), len(codes), zpath.name,
                         zpath.stat().st_size / 1e6, built[role]["zip_sha256"][:12]))
    if lay["to_send"].exists():
        shutil.rmtree(lay["to_send"])
    staging.rename(lay["to_send"])
    write_text(lay["to_send"] / "build_report.txt", "\n".join(report) + "\n")
    man["state"]["build"] = {"utc": utc_now(), "roles": built,
                             "manifest_sha256": sha256_file(inp.paths.manifest)}
    add_entry(man, "build", roles={r: built[r]["zip_sha256"][:12] for r in built})
    save_manifest(out, man)
    return report


def do_check(inp, out, role, file_path, write_reports=True):
    man = load_manifest(out)
    key = require_key(out, man)
    expected = [r["code"] for r in key if r["role"] == role]
    res = check_returned(file_path, role, expected, inp.sc, inp.normalise)
    if write_reports:
        lay = out_layout(out)
        stamp, k = utc_stamp(), 1
        while (lay["checks"] / ("check_%s_%s.txt" % (role, stamp))).exists():
            k += 1
            stamp = "%s_%d" % (utc_stamp(), k)
        write_text(lay["checks"] / ("check_%s_%s.txt" % (role, stamp)),
                   report_check(res, Path(file_path).name))
        if res.errors:
            write_text(lay["checks"] / ("fixlist_%s_%s.txt" % (role, stamp)), fix_list_text(res))
        add_entry(man, "check", role=role, file=Path(file_path).name,
                  file_sha256=sha256_file(file_path), passed=res.passed,
                  errors=len(res.errors), stop_errors=len(res.stop_errors))
        save_manifest(out, man)
    return res


def do_receive(inp, out, role, file_path):
    lay = out_layout(out)
    man = load_manifest(out)
    if role in received_roles(man):
        raise Ht3Error("%s is received already (%s); a received sheet is never replaced"
                       % (role, man["state"]["received"][role]["utc"]))
    res = do_check(inp, out, role, file_path)
    if not res.passed:
        raise Ht3Error("the sheet does not pass the check; send the fix list to the"
                       " namer first (see %s)" % lay["checks"])
    man = load_manifest(out)
    dest = lay["received"] / ("HT3_%s_received.xlsx" % role)
    if dest.exists():
        raise Ht3Error("%s exists; it is never overwritten" % dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(file_path, dest)
    rec = {"utc": utc_now(), "sha256": sha256_file(dest), "source_name": Path(file_path).name,
           "names_over_two_words": res.info["names_over_two_words"],
           "notes_filled": res.info["notes_filled"]}
    man.setdefault("state", {}).setdefault("received", {})[role] = rec
    add_entry(man, "receive", role=role, sha256=rec["sha256"])
    both = len(received_roles(man)) == len(ROLES)
    if both:
        key = read_key(lay["key"])
        ids = sorted({r["qa_id"] for r in key})
        write_text(lay["ids"], "\n".join(ids) + "\n")
        man["state"]["ids_file_sha256"] = sha256_file(lay["ids"])
    save_manifest(out, man)
    return rec, both


def score_rows(inp, key, names_by_role):
    """Score every name with the scorer, condition N-BB (D208)."""
    sc, normalise = inp.sc, inp.normalise
    out = []
    for r in key:
        role, code, qa = r["role"], r["code"], r["qa_id"]
        name = names_by_role[role].get(code, "")
        excluded = qa in inp.excluded
        base = {"role": role, "code": code, "qa_id": qa, "house_id": r["house_id"],
                "cell": r["cell"], "in_pclust": r["in_pclust"], "en_route": r["en_route"],
                "drawn_as": r["drawn_as"], "excluded": "yes" if excluded else "no",
                "name": name, "name_norm": normalise(name)}
        if excluded:
            base.update(label="", strict="", lenient="", fidelity="", lang="",
                        n_words="", named_ids="", copy_term="", copy_sim="",
                        gold_status="excluded", scorer_version=sc.SCORER_VERSION)
        else:
            rec = sc.score_item(inp.pname_by_id[qa], CONDITION, name, inp.lex, inp.voc)
            base.update(label=rec["label"] or "", strict=rec["strict"],
                        lenient=rec["lenient"], fidelity=rec["fidelity"],
                        lang=rec["lang"], n_words=rec["n_words"],
                        named_ids="|".join(rec["named_ids"] or []),
                        copy_term=rec["copy_term"] or "",
                        copy_sim="" if rec["copy_sim"] is None else rec["copy_sim"],
                        gold_status=rec["gold_status"], scorer_version=rec["scorer_version"])
        out.append(base)
    return out


SCORED_HEADER = ("role", "code", "qa_id", "house_id", "cell", "in_pclust",
                 "en_route", "drawn_as", "excluded", "name", "name_norm",
                 "label", "strict", "lenient", "fidelity", "lang", "n_words",
                 "named_ids", "copy_term", "copy_sim", "gold_status",
                 "scorer_version")


def summarise_scores(scored, n_boot=N_BOOT):
    by_qa = defaultdict(dict)
    for s in scored:
        by_qa[s["qa_id"]][s["role"]] = s
    counted = [q for q, d in by_qa.items()
               if all(d[r]["gold_status"] == "ok" for r in ROLES)]
    excluded = [q for q, d in by_qa.items() if d["N1"]["gold_status"] == "excluded"]
    unscoreable = [q for q, d in by_qa.items()
                   if q not in counted and q not in excluded]
    rows = [(by_qa[q]["N1"]["house_id"], by_qa[q]["N1"]["label"], by_qa[q]["N2"]["label"],
             int(by_qa[q]["N1"]["strict"]), int(by_qa[q]["N2"]["strict"])) for q in sorted(counted)]
    point = agreement_stats(rows)
    ci, skipped, n_houses = bootstrap(rows, n_boot=n_boot) if rows else ({}, {}, 0)
    comp = {}
    for role in ROLES:
        labs = Counter(by_qa[q][role]["label"] for q in counted)
        n = len(counted)
        comp[role] = {"counts": {l: labs.get(l, 0) for l in LABELS},
                      "shares": {l: (labs.get(l, 0) / n if n else None) for l in LABELS},
                      "strict": (sum(int(by_qa[q][role]["strict"]) for q in counted) / n) if n else None,
                      "lenient": (sum(int(by_qa[q][role]["lenient"]) for q in counted) / n) if n else None,
                      "fidelity": (sum(1 for q in counted if by_qa[q][role]["fidelity"]) / n) if n else None}
    pooled = Counter()
    for role in ROLES:
        pooled.update({l: comp[role]["counts"][l] for l in LABELS})
    n2 = 2 * len(counted)
    comp["pooled"] = {"counts": {l: pooled.get(l, 0) for l in LABELS},
                      "shares": {l: (pooled.get(l, 0) / n2 if n2 else None) for l in LABELS}}
    confusion = {l1: {l2: 0 for l2 in LABELS} for l1 in LABELS}
    for _, l1, l2, _, _ in rows:
        confusion[l1][l2] += 1
    per_cell = {}
    for c in CELLS:
        qs = [q for q in counted if by_qa[q]["N1"]["cell"] == c]
        per_cell[c] = {"n": len(qs)}
        for role in ROLES:
            labs = Counter(by_qa[q][role]["label"] for q in qs)
            per_cell[c][role] = {l: labs.get(l, 0) for l in LABELS}
    exact = sum(1 for q in counted
                if by_qa[q]["N1"]["name_norm"] == by_qa[q]["N2"]["name_norm"])
    same_other = []
    for q in sorted(counted):
        a, b = by_qa[q]["N1"], by_qa[q]["N2"]
        if a["label"] in ("sibling", "other_wrong", "ambiguous") and \
                b["label"] in ("sibling", "other_wrong", "ambiguous") and \
                a["named_ids"] and a["named_ids"] == b["named_ids"]:
            same_other.append((q, a["label"], b["label"], a["named_ids"]))
    return {"n_drawn": len(by_qa), "n_counted": len(counted), "n_excluded": len(excluded),
            "excluded_qa_ids": sorted(excluded), "unscoreable_qa_ids": sorted(unscoreable),
            "n_houses": n_houses, "point": point, "ci95": ci,
            "bootstrap": {"resamples": n_boot, "seed": SEED, "unit": "house",
                          "skipped_undefined": skipped},
            "composition": comp, "confusion_N1_rows_N2_cols": confusion,
            "per_cell": per_cell, "exact_same_name": exact,
            "same_other_object": same_other}


def roles_note(lexicon_writer):
    """D214: the namer who helped write the lexicon is reported separately and
    is never the main human row."""
    if lexicon_writer in ROLES:
        primary = [r for r in ROLES if r != lexicon_writer][0]
        return primary, ("Primary human row (F5): %s, independent of the lexicon. %s helped write"
                         " the lexicon (HT-1): reported separately as a secondary row; state it in"
                         " Methods and Limitations (D214)." % (primary, lexicon_writer))
    return None, "Neither namer helped write the lexicon."


def score_report(summary, mode, meta):
    banner = ("PROVISIONAL, not for reporting (Gate F not passed; D208)"
              if mode == "provisional" else "FINAL (gate-F scorer, gate-A lexicon)")
    p, ci = summary["point"], summary["ci95"]

    def with_ci(k, fmt):
        lo, hi = ci.get(k, (None, None))
        return "%s [%s, %s]" % (fmt(p.get(k)), fmt(lo), fmt(hi))

    L = ["HT-3 scores of the human names  [%s]" % banner,
         "script %s | scorer %s (sha256 %s...) | condition %s | UTC %s"
         % (SCRIPT_VERSION, meta["scorer_version"], meta["scorer_sha256"][:12], CONDITION, meta["utc"]),
         "git HEAD %s | lexicon %s... | vocab %s... | excluded list %s... (%d rows)"
         % (meta["git_head"], meta["lexicon_sha256"][:12], meta["vocab_sha256"][:12],
            meta["excluded_sha256"][:12], meta["excluded_rows"]),
         "Items: %d drawn, %d excluded at scoring, %d counted, %d unscoreable, %d houses"
         % (summary["n_drawn"], summary["n_excluded"], summary["n_counted"],
            len(summary["unscoreable_qa_ids"]), summary["n_houses"]),
         roles_note(meta.get("lexicon_writer", "none"))[1],
         "",
         "Labels (count and share)    " + "  ".join("%12s" % l for l in LABELS)]
    for who in ("N1", "N2", "pooled"):
        c = summary["composition"][who]
        L.append("  %-25s " % who + "  ".join(
            "%5d %6s" % (c["counts"][l], pct(c["shares"][l])) for l in LABELS))
    L.append("")
    L.append("Strict, Bangla: N1 %s, N2 %s, pooled %s" % (
        with_ci("strict_N1", pct), with_ci("strict_N2", pct), with_ci("strict_pooled", pct)))
    L.append("Fidelity (Bangla answer): N1 %s, N2 %s" % (
        pct(summary["composition"]["N1"]["fidelity"]), pct(summary["composition"]["N2"]["fidelity"])))
    L.append("Agreement, six labels (primary): %s; Cohen's kappa %s"
             % (with_ci("agree6", pct), with_ci("kappa6", num)))
    L.append("Agreement, strict (secondary): %s; Cohen's kappa %s"
             % (with_ci("agree_strict", pct), with_ci("kappa_strict", num)))
    L.append("Bootstrap: %(resamples)d resamples of houses, seed %(seed)d" % summary["bootstrap"]
             + "; resamples with undefined kappa skipped: %s" % (summary["bootstrap"]["skipped_undefined"] or "none"))
    L.append("")
    L.append("Confusion, rows N1, columns N2: " + " ".join(LABELS))
    for l1 in LABELS:
        L.append("  %-13s %s" % (l1, " ".join("%4d" % summary["confusion_N1_rows_N2_cols"][l1][l2] for l2 in LABELS)))
    L.append("")
    L.append("Per true cell (information): label counts N1 | N2")
    for c in CELLS:
        pc = summary["per_cell"][c]
        L.append("  %s n=%d  N1 %s | N2 %s" % (c, pc["n"],
                 " ".join(str(pc["N1"][l]) for l in LABELS), " ".join(str(pc["N2"][l]) for l in LABELS)))
    L.append("")
    L.append("Exact same name after normalise() (information): %d of %d"
             % (summary["exact_same_name"], summary["n_counted"]))
    L.append("Both namers named the same other object: %d items (ht3_same_other_object.csv);"
             " for HT-5 only, never an exclusion (D212)" % len(summary["same_other_object"]))
    if summary["excluded_qa_ids"]:
        L.append("Excluded at scoring: " + ", ".join(summary["excluded_qa_ids"]))
    return "\n".join(L) + "\n"


def do_score(inp, out, mode, n_boot=N_BOOT):
    lay = out_layout(out)
    man = load_manifest(out)
    key = require_key(out, man)
    if len(received_roles(man)) != len(ROLES):
        raise Ht3Error("both sheets must be received before scoring (received: %s)"
                       % (", ".join(received_roles(man)) or "none"))
    repo = inp.paths.repo
    identity = []
    if mode == "final":
        identity = frozen_identity(repo, inp.paths)
        bad = [x for x in identity if not x[2]]
        if bad:
            raise Ht3Error("final scores need the gate-F scorer and the gate-A lexicon"
                           " (D208): " + "; ".join("%s %s: %s" % (t, r, d) for t, r, _, d in bad)
                           + ". Use --provisional before Gate F.")
    names_by_role = {}
    for role in ROLES:
        rec = man["state"]["received"][role]
        p = lay["received"] / ("HT3_%s_received.xlsx" % role)
        if sha256_file(p) != rec["sha256"]:
            raise Ht3Error("the received sheet of %s changed since receipt; never edit it" % role)
        expected = [r["code"] for r in key if r["role"] == role]
        res = check_returned(p, role, expected, inp.sc, inp.normalise)
        if not res.passed:
            raise Ht3Error("the received sheet of %s no longer passes the check" % role)
        names_by_role[role] = res.names
    scored = score_rows(inp, key, names_by_role)
    summary = summarise_scores(scored, n_boot=n_boot)
    meta = {"utc": utc_now(), "mode": mode, "reportable": mode == "final",
            "scorer_version": inp.sc.SCORER_VERSION,
            "scorer_sha256": inp.hashes["scoring.py"], "normalise_sha256": inp.hashes["normalise.py"],
            "lexicon_sha256": inp.hashes["lexicon.csv"], "vocab_sha256": inp.hashes["closed_vocab.csv"],
            "excluded_sha256": inp.hashes["excluded_items.csv"], "excluded_rows": inp.excluded_rows,
            "git_head": git_out(repo, "rev-parse", "--short=12", "HEAD") or "unknown",
            "gate_f_commit": git_out(repo, "rev-list", "-n", "1", "gate-F") if mode == "final" else None,
            "identity_checks": [{"tag": t, "file": r, "ok": ok, "detail": d} for t, r, ok, d in identity],
            "key_sha256": man["state"]["draw"]["key_sha256"],
            "received_sha256": {r: man["state"]["received"][r]["sha256"] for r in ROLES},
            "condition": CONDITION, "script_version": SCRIPT_VERSION,
            "lexicon_writer": man["state"]["draw"].get("lexicon_writer", "none"),
            "primary_role": roles_note(man["state"]["draw"].get("lexicon_writer", "none"))[0]}
    folder = out / (("score_final_%s" % utc_stamp()) if mode == "final" else
                    ("score_provisional_%s_%s" % (inp.sc.SCORER_VERSION, utc_stamp())))
    folder.mkdir(parents=True, exist_ok=False)
    write_csv(folder / "ht3_scored_names.csv", SCORED_HEADER,
              [[s[k] for k in SCORED_HEADER] for s in scored])
    write_csv(folder / "ht3_same_other_object.csv", ("qa_id", "label_N1", "label_N2", "named_ids"),
              summary["same_other_object"])
    js = dict(meta)
    js["summary"] = {k: v for k, v in summary.items() if k != "same_other_object"}
    js["summary"]["same_other_object_n"] = len(summary["same_other_object"])
    write_text(folder / "ht3_agreement.json", json.dumps(js, ensure_ascii=False, indent=1, default=str))
    report = score_report(summary, mode, meta)
    write_text(folder / "ht3_report.txt", report)
    man.setdefault("state", {}).setdefault("scores", []).append(
        {"utc": meta["utc"], "mode": mode, "folder": folder.name,
         "scorer_version": meta["scorer_version"], "scorer_sha256": meta["scorer_sha256"]})
    add_entry(man, "score", mode=mode, folder=folder.name, scorer_version=meta["scorer_version"],
              n_counted=summary["n_counted"])
    save_manifest(out, man)
    return summary, report, folder


# --------------------------------------------------------------------------
# 11. check-inputs
# --------------------------------------------------------------------------

def cmd_check_inputs(args):
    paths = Paths.from_args(args)
    lines, n_fail = [], 0

    def line(status, what, detail=""):
        nonlocal n_fail
        if status == "FAIL":
            n_fail += 1
        lines.append("%-4s  %-42s %s" % (status, what, detail))
        say(lines[-1])

    say("ASSET-12 check-inputs  (script %s, UTC %s)" % (SCRIPT_VERSION, utc_now()))
    repo_files = {"src/scoring.py": paths.repo / "src" / "scoring.py",
                  "src/normalise.py": paths.repo / "src" / "normalise.py",
                  "lexicon/lexicon.csv": paths.lexicon,
                  "lexicon/closed_vocab.csv": paths.vocab,
                  "probes/excluded_items.csv": paths.excluded}
    drive_files = {"probes/P-NAME.json": paths.pname, "probes/P-CLUST.json": paths.pclust,
                   "data/house_assignment.csv": paths.houses,
                   "data/images_1024/images_manifest.csv": paths.manifest}
    missing = False
    for what, p in list(repo_files.items()) + list(drive_files.items()):
        if not p.exists():
            line("FAIL", what, "missing at %s" % p)
            missing = True
    if missing:
        say("check-inputs: stop, files are missing")
        return 1
    try:
        sc, normalise = load_scorer(paths.repo)
        line("PASS", "src/scoring.py", "version %s, sha256 %s..." % (
            sc.SCORER_VERSION, sha256_file(repo_files["src/scoring.py"])[:12]))
        line("PASS", "src/normalise.py", "sha256 %s..." % sha256_file(repo_files["src/normalise.py"])[:12])
    except Exception as exc:
        line("FAIL", "src/scoring.py import", repr(exc))
        return 1
    for name, p in (("lexicon.csv", paths.lexicon), ("closed_vocab.csv", paths.vocab)):
        got = sha256_file(p)
        ok = got == EXPECTED_SHA256[name]
        line("PASS" if ok else "FAIL", "lexicon/" + name, "sha256 %s... %s" % (
            got[:12], "as frozen" if ok else "expected %s..." % EXPECTED_SHA256[name][:12]))
        tagged = git_bytes(paths.repo, "gate-A:lexicon/" + name)
        if tagged is None:
            line("FAIL", "lexicon/%s at tag gate-A" % name, "tag gate-A not found (run git fetch --tags)")
        else:
            same = tagged == p.read_bytes()
            line("PASS" if same else "FAIL", "lexicon/%s at tag gate-A" % name,
                 "identical" if same else "differs from the tagged file")
    lex = sc.load_lexicon(paths.lexicon)
    voc = sc.load_closed_vocab(paths.vocab)
    ok = len(lex.objects) == EXPECTED_COUNTS["objects"] and not lex.warnings
    line("PASS" if ok else "FAIL", "lexicon loads in the scorer", lex.describe() + ", warnings %d" % len(lex.warnings))
    line("PASS", "closed_vocab loads in the scorer", voc.describe())
    excluded, n_excl = read_excluded(paths.excluded)
    pname = sc.load_probe(paths.pname)
    pname_ids = {it["qa_id"] for it in pname}
    line("PASS", "probes/excluded_items.csv", "%d rows, sha256 %s..., P-NAME items listed %d" % (
        n_excl, sha256_file(paths.excluded)[:12], len(excluded & pname_ids)))
    for name, p in (("P-NAME.json", paths.pname), ("P-CLUST.json", paths.pclust),
                    ("house_assignment.csv", paths.houses), ("images_manifest.csv", paths.manifest)):
        got = sha256_file(p)
        ok = got == EXPECTED_SHA256[name]
        line("PASS" if ok else "FAIL", name + " sha256", "%s... %s" % (
            got[:12], "as checked in the chat" if ok else "expected %s..." % EXPECTED_SHA256[name][:12]))
    houses_pn = {it["house_id"] for it in pname}
    caps = Counter(it.get("capture_type") for it in pname)
    ok = len(pname) == EXPECTED_COUNTS["P-NAME"] and set(caps) == {"closeup"}
    line("PASS" if ok else "FAIL", "P-NAME.json content", "%d items, capture %s, %d houses" % (
        len(pname), dict(caps), len(houses_pn)))
    pclust = sc.load_probe(paths.pclust)
    pclust_ids = {it["qa_id"] for it in pclust}
    ok = len(pclust) == EXPECTED_COUNTS["P-CLUST"] and pclust_ids <= pname_ids
    line("PASS" if ok else "FAIL", "P-CLUST.json content", "%d items, all in P-NAME: %s" % (
        len(pclust), pclust_ids <= pname_ids))
    for name, ids in (("P-NAME", pname_ids), ("P-CLUST", pclust_ids)):
        p = paths.repo / "probes" / ("%s.ids.txt" % name)
        if p.exists():
            listed = set(re.findall(r"H\d{2}_\d{4}_q\d+", p.read_text(encoding="utf-8-sig")))
            line("PASS" if listed == ids else "FAIL", "probes/%s.ids.txt" % name,
                 "same %d qa_ids as the json" % len(ids) if listed == ids else "differs from the json")
        else:
            line("INFO", "probes/%s.ids.txt" % name, "not in the repo; skipped")
    houses = read_house_assignment(paths.houses)
    per_ann = Counter(houses.values())
    ok = len(houses) == EXPECTED_COUNTS["houses"]
    line("PASS" if ok else "FAIL", "house_assignment.csv", "%d houses; annotator ids %s (names not read)" % (
        len(houses), ", ".join("%s %d" % (a, n) for a, n in sorted(per_ann.items()))))
    man_rows = read_images_manifest(paths.manifest)
    ok = len(man_rows) == EXPECTED_COUNTS["manifest_rows"]
    line("PASS" if ok else "FAIL", "images_manifest.csv rows", "%d" % len(man_rows))
    listed = sum(1 for it in pname if it["image_file"] in man_rows)
    present = sum(1 for it in pname if (paths.images / it["image_file"]).exists())
    ok = listed == len(pname) and present == len(pname)
    line("PASS" if ok else "FAIL", "P-NAME images", "%d of %d in the manifest, %d of %d in %s" % (
        listed, len(pname), present, len(pname), paths.images))
    items, problems = place_items(sc, normalise, lex, voc, pname, pclust_ids)
    cells = Counter(it.cell for it in items)
    ok = not problems and all(cells[c] == C1_CELLS[c] for c in CELLS)
    line("PASS" if ok else "FAIL", "golds resolve and NAME-REF cells (C1)", "%d of %d resolve; cells %s" % (
        len(items), len(pname), ", ".join("%s %d" % (c, cells[c]) for c in CELLS)))
    img = Counter(it["image_id"] for it in pname)
    ok = max(img.values()) == 1
    line("PASS" if ok else "FAIL", "one P-NAME item per image", "largest per image: %d" % max(img.values()))
    probs = package_text_problems(sc, normalise, lex)
    line("PASS" if not probs else "FAIL", "namer texts hold no lexicon term or id",
         "%d texts checked" % len(ALL_PACKAGE_TEXTS) if not probs else "%d texts with a problem" % len(probs))
    try:
        import openpyxl
        import PIL
        line("PASS", "openpyxl and Pillow", "openpyxl %s, Pillow %s" % (openpyxl.__version__, PIL.__version__))
    except Exception as exc:
        line("FAIL", "openpyxl and Pillow", repr(exc))
    gf = git_out(paths.repo, "rev-list", "-n", "1", "gate-F")
    line("INFO", "tag gate-F", "present (%s); score --final possible" % gf[:12] if gf else
         "not yet; only score --provisional is possible (D208)")
    say("check-inputs: %d FAIL" % n_fail if n_fail else "check-inputs: ALL PASS")
    return 1 if n_fail else 0


# --------------------------------------------------------------------------
# 12. Command wrappers
# --------------------------------------------------------------------------

def print_rep(lines):
    for l in lines:
        say(l)


def cmd_draw(args):
    paths = Paths.from_args(args)
    inp = load_inputs(paths)
    say("ASSET-12 draw: P-NAME %d items, golds resolved %d, excluded list %d rows" % (
        len(inp.pname), len(inp.items), inp.excluded_rows))
    res, rep = do_draw(inp, paths.out, args.namer1, args.namer2, args.also_exclude1,
                       args.also_exclude2, args.redraw, args.lexicon_writer)
    print_rep(rep)
    say("Wrote %s and %s" % (out_layout(paths.out)["draw_csv"], out_layout(paths.out)["key"]))
    say("NEXT: build --namers-cleared, once both namers have confirmed the rules of D205 and D214.")
    return 0


def cmd_build(args):
    if not args.namers_cleared:
        raise Ht3Error("build writes the packages that go to the namers. Give"
                       " --namers-cleared only after the conditions of D205 and D214"
                       " are met: both namers confirm they have seen no model answer and"
                       " no scorer result, and they have agreed not to open Drive outputs/,"
                       " scored/, models/, probes/, human/, the dataset or the lexicon,"
                       " and not to run or read the pilot, until both sheets are received.")
    paths = Paths.from_args(args)
    inp = load_inputs(paths, need_manifest=True)
    man_rows = read_images_manifest(paths.manifest)
    rep = do_build(inp, paths.out, man_rows)
    man = load_manifest(paths.out)
    man["state"]["build"]["namers_cleared_utc"] = utc_now()
    save_manifest(paths.out, man)
    print_rep(rep)
    say("Wrote %s (HT3_N1.zip, HT3_N2.zip, blank sheets for a look)" % out_layout(paths.out)["to_send"])
    say("Send each namer only their own zip and the guide. Never send the key or the draw.")
    return 0


def cmd_check(args):
    paths = Paths.from_args(args)
    inp = load_inputs(paths)
    res = do_check(inp, paths.out, args.role, args.file)
    say("HT-3 check of %s (%s): %s" % (args.role, Path(args.file).name, "PASS" if res.passed else "FAIL"))
    say("  rows with codes %(rows_with_codes)d, names filled %(names_filled)d, notes %(notes_filled)d,"
        " names over two words %(names_over_two_words)d (information only)" % res.info if res.info else "")
    for e in res.stop_errors:
        say("  STOP: " + e)
    if res.errors:
        say("  %d rows to fix. Send the namer this list (also saved in checks/):" % len(res.errors))
        for code, m in res.errors:
            say("    %s: %s" % (code, m))
    if res.passed:
        say("NEXT: receive --role %s --file <the same file>" % args.role)
    return 0 if res.passed else 1


def cmd_receive(args):
    paths = Paths.from_args(args)
    inp = load_inputs(paths)
    rec, both = do_receive(inp, paths.out, args.role, args.file)
    say("Received %s at %s (sha256 %s...). It is never replaced." % (args.role, rec["utc"], rec["sha256"][:12]))
    man = load_manifest(paths.out)
    for r in ROLES:
        got = man["state"].get("received", {}).get(r)
        say("  %s: %s" % (r, ("received " + got["utc"]) if got else "not yet"))
    if both:
        say("Both sheets received: Gate H evidence is complete once you confirm that neither"
            " namer saw model or scorer output before these times (D211).")
        say("Upload %s to the repo as probes/NAME-REF.ids.txt (D213)." % out_layout(paths.out)["ids"])
    return 0


def cmd_score(args):
    if args.provisional == args.final:
        raise Ht3Error("give exactly one of --provisional or --final")
    mode = "final" if args.final else "provisional"
    paths = Paths.from_args(args)
    inp = load_inputs(paths)
    summary, report, folder = do_score(inp, paths.out, mode, n_boot=args.boot)
    print_rep(report.splitlines())
    say("Wrote %s" % folder)
    return 0


def cmd_status(args):
    paths = Paths.from_args(args)
    man = load_manifest(paths.out)
    st = man.get("state", {})
    say("ASSET-12 status (%s)" % paths.out)
    d = st.get("draw")
    if not d:
        say("  draw: not yet")
        return 0
    say("  draw: %s, %d items, houses excluded %d, NAME-REF houses %d" % (
        d["utc"], d["n"], len(d["excluded_houses"]), d["houses_in_nameref"]))
    say("        namers " + ", ".join("%s=%s" % (r, d["namers"][r]["annotator_id"]) for r in ROLES))
    say("        final count by true cell " + ", ".join("%s %d" % (c, d["cells"][c]["in_cell_final"]) for c in CELLS))
    b = st.get("build")
    say("  build: " + ("%s, cleared %s" % (b["utc"], b.get("namers_cleared_utc", "?")) if b else "not yet"))
    checks = [e for e in man["entries"] if e["command"] == "check"]
    say("  checks run: %d (%d passed)" % (len(checks), sum(1 for e in checks if e.get("passed"))))
    for r in ROLES:
        got = st.get("received", {}).get(r)
        say("  %s: %s" % (r, ("received " + got["utc"]) if got else "not received"))
    for s in st.get("scores", []):
        say("  score: %s %s, scorer %s" % (s["utc"], s["mode"], s["scorer_version"]))
    return 0


# --------------------------------------------------------------------------
# 13. dryrun: the whole chain on the real inputs, in a temporary folder
# --------------------------------------------------------------------------

def _fill_workbook(blank_bytes, names, decl, dup_code=None, drop_code=None):
    """A stand-in for a namer: fills a blank sheet. Used by dryrun and selftest."""
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(blank_bytes))
    ws = wb[SHEET_NAMES]
    drop_row = None
    for r in range(2, ws.max_row + 1):
        code = ws.cell(row=r, column=1).value
        if code in names:
            ws.cell(row=r, column=2, value=names[code])
        if drop_code and code == drop_code:
            drop_row = r
    if dup_code:
        ws.append([dup_code, names.get(dup_code, ""), None])
    if drop_row:
        ws.delete_rows(drop_row)
    wd = wb[SHEET_DECL]
    for r in range(2, wd.max_row + 1):
        did = wd.cell(row=r, column=1).value
        if did in decl:
            wd.cell(row=r, column=3, value=decl[did])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _blank_from_zip(zpath, role):
    with zipfile.ZipFile(zpath) as z:
        return z.read("HT3_%s/HT3_%s.xlsx" % (role, role))


def cmd_dryrun(args):
    paths = Paths.from_args(args)
    tmp = Path(args.tmp) if args.tmp else Path(tempfile.mkdtemp(prefix="ht3_dryrun_"))
    if tmp.exists() and any(tmp.iterdir()):
        raise Ht3Error("%s is not empty; give an empty --tmp" % tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    out = tmp / "out"
    results = []

    def ok(what, cond, detail=""):
        results.append(cond)
        say("%s  %s %s" % ("PASS" if cond else "FAIL", what, detail))

    try:
        say("ASSET-12 dryrun in %s (stand-in names; nothing is kept)" % tmp)
        inp = load_inputs(paths, need_manifest=True)
        res, _ = do_draw(inp, out, args.namer1, args.namer2, args.also_exclude1, args.also_exclude2,
                         lexicon_writer=args.lexicon_writer)
        ok("draw", len(res.drawn) == TOTAL, "%d items" % len(res.drawn))
        do_build(inp, out, read_images_manifest(paths.manifest))
        lay = out_layout(out)
        ok("build", all((lay["to_send"] / ("HT3_%s.zip" % r)).exists() for r in ROLES), "two zips")
        key = read_key(lay["key"])
        gold = {r["code"]: inp.pname_by_id[r["qa_id"]]["answer_bn"] for r in key}
        codes = {role: [r["code"] for r in key if r["role"] == role] for role in ROLES}
        yes = {d: "হ্যাঁ" for d, _ in DECLARATIONS}
        ret = tmp / "returned"
        ret.mkdir()
        n1 = {c: gold[c] for c in codes["N1"]}
        (ret / "HT3_N1.xlsx").write_bytes(_fill_workbook(_blank_from_zip(lay["to_send"] / "HT3_N1.zip", "N1"), n1, yes))
        c2 = codes["N2"]
        bad = {c: gold[c] for c in c2}
        bad[c2[0]] = ""
        bad[c2[1]] = "abc"
        bad[c2[2]] = gold[c2[2]] + " ২"
        bad[c2[3]] = gold[c2[3]] + " / " + gold[c2[3]]
        bad[c2[4]] = "জানি না"
        decl_bad = dict(yes)
        decl_bad["D2"] = None
        blank2 = _blank_from_zip(lay["to_send"] / "HT3_N2.zip", "N2")
        (ret / "HT3_N2_bad.xlsx").write_bytes(_fill_workbook(blank2, bad, decl_bad, dup_code=c2[5], drop_code=c2[6]))
        r1 = do_check(inp, out, "N1", ret / "HT3_N1.xlsx")
        ok("check N1 (stand-in names, no faults)", r1.passed)
        r2 = do_check(inp, out, "N2", ret / "HT3_N2_bad.xlsx")
        want = {(c2[0], E_EMPTY), (c2[1], E_SCRIPT), (c2[2], E_DIGIT), (c2[3], E_TWO),
                (c2[4], E_GUESS), (c2[5], E_DUP), (c2[6], E_MISSING), ("D2", E_DECL)}
        ok("check N2 finds exactly the 8 planted faults", (not r2.passed) and set(r2.errors) == want,
           "found %d" % len(r2.errors))
        swaps = 20
        gid = {r["code"]: r["gold_name_id"] for r in key}
        fixed = {c: gold[c] for c in c2}
        donors = [c for c in c2[swaps:]]
        for i, c in enumerate(c2[:swaps]):
            donor = next(d for d in donors[i:] + donors[:i] if gid[d] != gid[c])
            fixed[c] = gold[donor]
        (ret / "HT3_N2.xlsx").write_bytes(_fill_workbook(blank2, fixed, yes))
        r3 = do_check(inp, out, "N2", ret / "HT3_N2.xlsx")
        ok("check N2 after the fixes", r3.passed)
        do_receive(inp, out, "N1", ret / "HT3_N1.xlsx")
        _, both = do_receive(inp, out, "N2", ret / "HT3_N2.xlsx")
        ok("receive both", both)
        try:
            do_receive(inp, out, "N1", ret / "HT3_N1.xlsx")
            ok("a received sheet is never replaced", False)
        except Ht3Error:
            ok("a received sheet is never replaced", True)
        summary, _, folder = do_score(inp, out, "provisional", n_boot=args.boot)
        n = summary["n_counted"]
        c1 = summary["composition"]["N1"]["counts"]
        c2c = summary["composition"]["N2"]["counts"]
        ok("score: gold names give Tier 1 on every N1 item", c1["tier1"] == n and n == TOTAL - summary["n_excluded"],
           "%d of %d" % (c1["tier1"], n))
        ok("score: N2 has exactly %d items not at Tier 1" % swaps, c2c["tier1"] == n - swaps,
           "%d of %d at Tier 1" % (c2c["tier1"], n))
        po, k = summary["point"]["agree6"], summary["point"]["kappa6"]
        ok("score: agreement and kappa as constructed", abs(po - (n - swaps) / n) < 1e-12 and k is not None and abs(k) < 1e-12,
           "agreement %s, kappa %s" % (pct(po), num(k)))
        ok("score: strict equals Tier 1 for Bangla names", abs(summary["point"]["strict_N1"] - 1.0) < 1e-12)
        ok("score: files written", (folder / "ht3_report.txt").exists() and (folder / "ht3_scored_names.csv").exists())
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)
    n_fail = sum(1 for r in results if not r)
    say("dryrun: %d checks, %s" % (len(results), "ALL PASS" if not n_fail else "%d FAIL" % n_fail))
    if not args.keep:
        say("Temporary folder deleted.")
    return 1 if n_fail else 0


# --------------------------------------------------------------------------
# 14. selftest (invented data only; no dataset text)
# --------------------------------------------------------------------------

def _st_lexicon(sc):
    def row(nid, bn, en, cluster="", route="no", super_bn="", super_en="", variants_bn=""):
        return {"name_id": nid, "name_bn": bn, "variants_bn": variants_bn,
                "canonical_en": en, "variants_en": "", "super_bn": super_bn,
                "super_en": super_en, "cluster_id": cluster, "siblings": "",
                "room_scope": "any", "cultural_specificity": "not_rated",
                "origin3": "other_loan", "origin_note": "", "en_route": route,
                "en_unique": "yes", "freq_obj_name": "1"}
    rows = [row("X1", "ঝুমকাল", "zumkal", "K1", "yes", "পাত্রক", "vessa"),
            row("X2", "টপারি", "topari", "K1", "no", "পাত্রক", "vessa"),
            row("X3", "মিলনো", "milno", "", "yes"),
            row("X4", "কাবলু", "kablu", "", "no"),
            row("X5", "রিথান", "rithan", "K2", "yes"),
            row("X6", "দোপলা", "dopla", "K2", "no")]
    vrows = [{"kind": "room", "class_id": c, "forms_bn": b, "forms_en": c, "note": ""}
             for c, b in (("bathroom", "স্নানঘরক"), ("bedroom", "শোয়ারঘরক"),
                          ("kitchen", "রাঁধাঘরক"), ("living room", "বসাঘরক"))]
    vrows.append({"kind": "number", "class_id": "1", "forms_bn": "একটি", "forms_en": "one", "note": ""})
    return sc.build_lexicon(rows, source="selftest"), sc.build_closed_vocab(vrows, source="selftest")


def _st_items(objs_by_cell, houses):
    """Invented probe records: (cell, object bn name, en_route) spread over houses."""
    pname, pclust_ids = [], set()
    k = 0
    for cell, n, bn in objs_by_cell:
        for i in range(n):
            k += 1
            h = houses[k % len(houses)]
            qa = "%s_%04d_q1" % (h, k)
            pname.append({"qa_id": qa, "image_id": "%s_%04d" % (h, k), "house_id": h,
                          "capture_type": "closeup", "qa_type": "obj_name",
                          "question_bn": "প্রশ্নক %d" % (k % 2), "answer_bn": bn,
                          "image_file": "%s/%s_%04d.jpg" % (h, h, k)})
            if cell.startswith("PY"):
                pclust_ids.add(qa)
    return pname, pclust_ids


def cmd_selftest(args):
    sc, normalise = load_scorer(Path(args.repo).resolve())
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))
        say("%s  %s" % ("OK  " if cond else "FAIL", name))

    lex, voc = _st_lexicon(sc)
    houses = {"H91": "A1", "H92": "A1", "H93": "A2", "H94": "A2", "H95": "A3", "H96": "A3"}
    hl = sorted(houses)
    # Cells: X1 PY-EY, X2 PY-EN, X3 PN-EY, X4 PN-EN (cell decided by P-CLUST and en_route)
    pname, pclust_ids = _st_items([("PY-EY", 12, "ঝুমকাল"), ("PY-EN", 4, "টপারি"),
                                   ("PN-EY", 9, "মিলনো"), ("PN-EN", 10, "কাবলু")], hl)
    items, probs = place_items(sc, normalise, lex, voc, pname, pclust_ids)
    check("place_items: every invented gold resolves", len(items) == len(pname) and not probs)
    cells = Counter(it.cell for it in items)
    check("place_items: cells come from P-CLUST membership and en_route",
          cells == Counter({"PY-EY": 12, "PY-EN": 4, "PN-EY": 9, "PN-EN": 10}))
    check("rank_key: seed 42 string, fixed value",
          rank_key("H91_0001_q1") == hashlib.sha256(b"42|NAME-REF|H91_0001_q1").hexdigest())
    r = draw_nameref(items, set(), set(), per_cell=3, total=12)
    check("draw: 12 items, 3 per cell when no cell is short",
          len(r.drawn) == 12 and all(r.stats[c]["in_cell_final"] == 3 for c in CELLS))
    r2 = draw_nameref(items, set(), set(), per_cell=3, total=12)
    check("draw: same result on a second run", [d.item.qa_id for d in r.drawn] == [d.item.qa_id for d in r2.drawn])
    expect = sorted([it for it in items if it.cell == "PN-EY"], key=lambda it: rank_key(it.qa_id))[:3]
    check("draw: lowest ranks are taken", {d.item.qa_id for d in r.drawn if d.item.cell == "PN-EY"} == {it.qa_id for it in expect})
    r3 = draw_nameref(items, set(), set(), per_cell=5, total=20)
    s = r3.stats
    check("fill: a short cell takes all it has (PY-EN 4 of 5)", s["PY-EN"]["drawn_regular"] == 4 and s["PY-EN"]["short"] == 1)
    check("fill: the missing item comes from the largest cell (PY-EY)", s["PY-EN"]["filled_from"] == {"PY-EY": 1})
    fills = [d for d in r3.drawn if d.drawn_as == "fill"]
    check("fill: a filled item keeps its true cell and names the short cell",
          len(fills) == 1 and fills[0].item.cell == "PY-EY" and fills[0].fill_for == "PY-EN")
    check("fill: 20 items in total", len(r3.drawn) == 20 and s["PY-EY"]["in_cell_final"] == 6)
    try:
        draw_nameref(items, set(), set(hl), per_cell=3, total=12)
        check("draw stops when fewer than the total are available", False)
    except Ht3Error:
        check("draw stops when fewer than the total are available", True)
    excl_house = {"H91", "H92"}
    r4 = draw_nameref(items, set(), excl_house, per_cell=2, total=8)
    check("exclusion: no item from an excluded house", all(d.item.house_id not in excl_house for d in r4.drawn))
    ex_qa = {expect[0].qa_id}
    r5 = draw_nameref(items, ex_qa, set(), per_cell=3, total=12)
    check("exclusion: a listed item is skipped", expect[0].qa_id not in {d.item.qa_id for d in r5.drawn}
          and r5.skipped_excluded == [expect[0].qa_id])
    mini = []
    for cell, (pc, er, gid) in (("PY-EY", (True, "yes", "X1")), ("PY-EN", (True, "no", "X2")),
                                ("PN-EN", (False, "no", "X4"))):
        for j in range(4):
            mini.append(Item("H9%d_%s%d_q1" % (j + 1, cell, j), "IMG-%s-%d" % (cell, j), "H9%d" % (j + 1),
                             "f", pc, er, gid, True))
    mini += [Item("H91_A_q1", "IMG-SAME", "H91", "f", False, "yes", "X3", True),
             Item("H92_B_q1", "IMG-SAME", "H92", "f", False, "yes", "X3", True),
             Item("H93_C_q1", "IMG-OTHER", "H93", "f", False, "yes", "X3", True)]
    r6 = draw_nameref(mini, set(), set(), per_cell=3, total=12)
    imgs = Counter(d.item.image_id for d in r6.drawn)
    check("one item per image: the second item on an image is skipped and the cell is filled",
          max(imgs.values()) == 1 and len(r6.skipped_same_image) == 1
          and r6.stats["PN-EY"]["short"] == 1 and r6.stats["PN-EY"]["filled_from"] == {"PY-EY": 1})
    per_role, ex = namer_houses("A1", "NA", "", "H95", houses)
    check("namers: houses of A1 plus an extra house of N2", ex == ["H91", "H92", "H95"])
    _, ex0 = namer_houses("NA", "NA", "", "", houses)
    check("namers: NA and NA exclude no house", ex0 == [])
    for bad_args, what in ((("A1", "A1", "", ""), "same annotator twice"),
                           (("A9", "NA", "", ""), "unknown annotator id"),
                           (("NA", "NA", "H77", ""), "unknown extra house")):
        try:
            namer_houses(*bad_args, houses)
            check("namers: %s is refused" % what, False)
        except Ht3Error:
            check("namers: %s is refused" % what, True)
    qas = [d.item.qa_id for d in r.drawn]
    c1, c2 = assign_codes(qas, "N1"), assign_codes(qas, "N2")
    check("codes: N1-001 to N1-012 and N2-001 to N2-012", [c for c, _ in c1] == ["N1-%03d" % i for i in range(1, 13)]
          and [c for c, _ in c2] == ["N2-%03d" % i for i in range(1, 13)])
    check("codes: the two namers get different orders", [q for _, q in c1] != [q for _, q in c2])
    check("codes: same items for both namers", sorted(q for _, q in c1) == sorted(q for _, q in c2))
    # images
    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (8, 6), (200, 10, 10)).save(buf, "JPEG", quality=90)
        clean = buf.getvalue()
        check("jpeg: a plain Pillow JPEG has no metadata block", jpeg_metadata_blocks(clean) == [])
        buf2 = io.BytesIO()
        exif = Image.Exif()
        exif[0x010F] = "maker"
        Image.new("RGB", (8, 6)).save(buf2, "JPEG", quality=90, exif=exif.tobytes())
        check("jpeg: an EXIF block is found", "APP1" in jpeg_metadata_blocks(buf2.getvalue()))
        com = clean[:2] + b"\xff\xfe\x00\x04hi" + clean[2:]
        check("jpeg: a comment block is found", "COM" in jpeg_metadata_blocks(com))
    except ImportError:
        check("Pillow available", False)
    # workbook, leak scan, check, receive, score
    try:
        import openpyxl  # noqa: F401
        codes = c1
        xlsx = make_workbook_bytes("N1", codes)
        texts = workbook_texts(xlsx)
        entries = ["HT3_N1/HT3_N1.xlsx"] + ["HT3_N1/images/%s.jpg" % c for c, _ in codes]
        check("leak scan: a clean package passes", leak_scan(texts, entries, sc, normalise, lex, {"প্রশ্নক ১"}) == [])
        check("leak scan: an image id is caught", leak_scan(texts + [("x", "see H91_0004")], entries, sc, normalise, lex, set()) != [])
        check("leak scan: a house id is caught", leak_scan(texts, entries + ["H93/x.jpg"], sc, normalise, lex, set()) != [])
        check("leak scan: a lexicon term is caught", leak_scan(texts + [("x", "এটি টপারি")], entries, sc, normalise, lex, set()) != [])
        check("leak scan: a question text is caught", leak_scan(texts + [("x", "প্রশ্নক 7 আছে")], entries, sc, normalise, lex, {normalise("প্রশ্নক 7")}) != [])
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            names = {c: "ঝুমকাল" for c, _ in codes}
            yes = {d: "হ্যাঁ" for d, _ in DECLARATIONS}
            good = td / "good.xlsx"
            good.write_bytes(_fill_workbook(xlsx, names, yes))
            exp = [c for c, _ in codes]
            res = check_returned(good, "N1", exp, sc, normalise)
            check("check: a correct sheet passes", res.passed and len(res.names) == 12)
            bad = dict(names)
            bad[exp[0]] = ""
            bad[exp[1]] = "fan"
            bad[exp[2]] = "ঝুমকাল ২"
            bad[exp[3]] = "ঝুমকাল বা টপারি"
            bad[exp[4]] = "জানিনা"
            bad[exp[5]] = "ঝুমকাল (বড়)"
            d2 = dict(yes)
            d2["D3"] = ""
            badp = td / "bad.xlsx"
            badp.write_bytes(_fill_workbook(xlsx, bad, d2, dup_code=exp[6], drop_code=exp[7]))
            res = check_returned(badp, "N1", exp, sc, normalise)
            want = {(exp[0], E_EMPTY), (exp[1], E_SCRIPT), (exp[2], E_DIGIT), (exp[3], E_TWO),
                    (exp[4], E_GUESS), (exp[5], E_TWO), (exp[6], E_DUP), (exp[7], E_MISSING), ("D3", E_DECL)}
            check("check: each planted fault is found, nothing else", (not res.passed) and set(res.errors) == want)
            nop = td / "no.xlsx"
            d3 = dict(yes)
            d3["D2"] = "না"
            nop.write_bytes(_fill_workbook(xlsx, names, d3))
            res = check_returned(nop, "N1", exp, sc, normalise)
            check("check: a declaration answered না stops the receipt", (not res.passed) and res.stop_errors)
            res = check_returned(good, "N2", [c for c, _ in c2], sc, normalise)
            check("check: the other namer's file is refused", (not res.passed) and res.stop_errors)
            three = dict(names)
            three[exp[0]] = "ঝুমকাল টপারি মিলনো"
            tp = td / "three.xlsx"
            tp.write_bytes(_fill_workbook(xlsx, three, yes))
            res = check_returned(tp, "N1", exp, sc, normalise)
            check("check: a three-word name is information only", res.passed and res.info["names_over_two_words"] == 1)
            txt = td / "x.txt"
            txt.write_text("not a workbook")
            try:
                check_returned(txt, "N1", exp, sc, normalise)
                check("check: a file that is not .xlsx is refused", False)
            except Ht3Error:
                check("check: a file that is not .xlsx is refused", True)
    except ImportError:
        check("openpyxl available", False)
    # kappa and bootstrap
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    po, k = cohen_kappa(a, b)
    check("kappa: textbook table gives agreement 0.70 and kappa 0.40", abs(po - 0.7) < 1e-12 and abs(k - 0.4) < 1e-12)
    po, k = cohen_kappa(["t"] * 5, ["t"] * 5)
    check("kappa: undefined when both raters use one category", po == 1.0 and k is None)
    check("percentile: linear interpolation", percentile([1, 2, 3, 4], 50) == 2.5 and percentile([5], 97.5) == 5)
    rows = [("H9%d" % (i % 5), "tier1" if i % 3 else "tier2", "tier1" if i % 4 else "sibling", 1 if i % 3 else 0,
             1 if i % 4 else 0) for i in range(40)]
    ci1, sk1, nh = bootstrap(rows, n_boot=300)
    ci2, _, _ = bootstrap(rows, n_boot=300)
    check("bootstrap: same CIs with the same seed", ci1 == ci2 and nh == 5)
    check("bootstrap: CI holds the point estimate", ci1["agree6"][0] <= agreement_stats(rows)["agree6"] <= ci1["agree6"][1])
    _, sk3, _ = bootstrap([("H91", "tier1", "tier1", 1, 1)] * 6, n_boot=50)
    check("bootstrap: undefined kappa resamples are counted", sk3.get("kappa6") == 50)
    # scoring of names through the scorer (N-BB)
    item = {"qa_id": "H91_0001_q1", "qa_type": "obj_name", "answer_bn": "ঝুমকাল"}
    lab = {n: sc.score_item(item, CONDITION, n, lex, voc)["label"] for n in ("ঝুমকাল", "পাত্রক", "টপারি", "কাবলু")}
    check("scorer: Tier 1, Tier 2, sibling, other wrong", lab == {"ঝুমকাল": "tier1", "পাত্রক": "tier2",
                                                                "টপারি": "sibling", "কাবলু": "other_wrong"})
    rec = sc.score_item(item, CONDITION, "zumkal", lex, voc)
    check("scorer: a Latin-letter name is strict wrong under N-BB", rec["label"] == "tier1" and rec["strict"] == 0)
    # git identity for the final scores
    if shutil.which("git"):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for rel in ("src/scoring.py", "src/normalise.py", "lexicon/lexicon.csv", "lexicon/closed_vocab.csv"):
                (td / rel).parent.mkdir(parents=True, exist_ok=True)
                (td / rel).write_text("v1 " + rel)

            def g(*a):
                return subprocess.run(["git", "-C", str(td)] + list(a), capture_output=True, text=True)
            g("init", "-q")
            g("-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
            g("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "x")
            g("tag", "gate-A")
            fake = Paths(repo=td, drive=td, lexicon=td / "lexicon/lexicon.csv", vocab=td / "lexicon/closed_vocab.csv",
                         excluded=td, pname=td, pclust=td, houses=td, images=td, manifest=td, out=td)
            idt = frozen_identity(td, fake)
            check("final mode: refused without the tag gate-F", not all(x[2] for x in idt))
            g("tag", "gate-F")
            check("final mode: allowed when every file equals its tag", all(x[2] for x in frozen_identity(td, fake)))
            (td / "src/scoring.py").write_text("v2")
            check("final mode: refused when the scorer changed after gate-F", not all(x[2] for x in frozen_identity(td, fake)))
    else:
        check("git available", False)
    check("texts: the instruction is Design 4.8 as written",
          INSTRUCTION_LINES[0].endswith("(এক বা দুই শব্দে)।") and INSTRUCTION_LINES[1].startswith("নিশ্চিত না হলে"))
    n_ok = sum(1 for _, okk in results if okk)
    say("")
    say("selftest: %d of %d checks OK" % (n_ok, len(results)))
    return 0 if n_ok == len(results) else 1


# --------------------------------------------------------------------------
# 15. Command line
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="ASSET-12: NAME-REF draw, naming sheets and HT-3 scores")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--repo", default=str(REPO_DEFAULT), help="repo root (default: this script's repo)")
        p.add_argument("--drive", default=str(DRIVE_DEFAULT), help="Drive project folder")
        p.add_argument("--pname")
        p.add_argument("--pclust")
        p.add_argument("--houses")
        p.add_argument("--images")
        p.add_argument("--manifest")
        p.add_argument("--out", help="default: <drive>/human/HT-3")
        p.add_argument("--allow-other-inputs", action="store_true",
                       help="tests only: go on when an input differs from the checked one")

    p = sub.add_parser("selftest")
    common(p)                 # paths are accepted and not used: invented data only
    p = sub.add_parser("check-inputs")
    common(p)
    for name in ("draw", "dryrun"):
        p = sub.add_parser(name)
        common(p)
        p.add_argument("--namer1", default=NA, help="annotator_id or NA (D206)")
        p.add_argument("--namer2", default=NA)
        p.add_argument("--also-exclude1", default="", help="house ids, comma separated")
        p.add_argument("--also-exclude2", default="")
        p.add_argument("--lexicon-writer", default="none", choices=("none",) + ROLES,
                       help="the role of a namer who helped write the lexicon (D214)")
        if name == "draw":
            p.add_argument("--redraw", help="reason; only before any sheet is received (D207)")
        else:
            p.add_argument("--tmp", help="an empty folder (default: a new temporary folder)")
            p.add_argument("--keep", action="store_true", help="keep the temporary folder")
            p.add_argument("--boot", type=int, default=N_BOOT)
    p = sub.add_parser("build")
    common(p)
    p.add_argument("--namers-cleared", action="store_true",
                   help="both namers confirmed the rules of D205 and D214")
    for name in ("check", "receive"):
        p = sub.add_parser(name)
        common(p)
        p.add_argument("--role", required=True, choices=ROLES)
        p.add_argument("--file", required=True)
    p = sub.add_parser("score")
    common(p)
    p.add_argument("--provisional", action="store_true")
    p.add_argument("--final", action="store_true")
    p.add_argument("--boot", type=int, default=N_BOOT)
    p = sub.add_parser("status")
    common(p)
    args = ap.parse_args(argv)
    if getattr(args, "redraw", None) is not None and not args.redraw.strip():
        ap.error("--redraw needs a reason")
    fn = {"selftest": cmd_selftest, "check-inputs": cmd_check_inputs, "draw": cmd_draw,
          "dryrun": cmd_dryrun, "build": cmd_build, "check": cmd_check, "receive": cmd_receive,
          "score": cmd_score, "status": cmd_status}[args.cmd]
    try:
        return fn(args)
    except Ht3Error as exc:
        say("STOP: %s" % exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
