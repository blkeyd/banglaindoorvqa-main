"""
asset4_lexicon.py  (ASSET-4 v2, with ASSET-10)
BanglaIndoorVQA main paper: lexicon.csv and closed_vocab.csv for Gate A.
Design v2.1, sections 4.2 to 4.5, read with DESIGN_ADDENDUM.md. Replaces the
v1 script asset4b_lexicon.py.

This script never writes a name, a variant or a class. People do that (HT-1).
It prepares the working sheets, checks what people wrote with the scorer's own
functions, and writes the two frozen files only when nothing is left to fix.

HARD RULE (Design 4.2): no model output is read here, and none may be looked
at while the sheets are filled.

Subcommands
  selftest  Runs build, check and freeze on made-up data in a temporary folder
            and proves that every rule fires. Needs no project file.
  build     Writes the HT-1 working files. With --update it refreshes an
            existing sheet after the dataset or the probe files changed, and
            keeps everything people have written.
  check     Reads the filled sheets and reports ERROR, TODO, WARNING and INFO
            lines. Writes reports only. Run it as often as you like.
  freeze    The same checks in final mode. With zero ERROR and zero TODO it
            writes lexicon.csv, closed_vocab.csv and excluded_items.csv, loads
            them back with the scorer and re-runs the gold audit on them.

Rows of the lexicon sheet (decisions of the ASSET-4 v2 chat; see
DESIGN_ADDENDUM.md, section C)
  One row per Bangla spelling that is (a) a naming gold of P-NAME,
  P-CORE-CLOSED or R-TEST, (b) a surface form of a cluster member in
  clusters_parsed.csv, (c) a colloquial form the lexicon file lists for such
  a name, or (d) the lexicon file's own spelling of such a name. A row is its
  own object, or a spelling of another row (same_object_as), or not a name
  (is_name = no; its items go on the exclusion list, D110).

Exclusion list (D110)
  A gold that cannot be scored is not corrected in the dataset. Its item is
  listed in excluded_items.csv, is still run, and is never scored. A reason
  is always about the gold or the question, never about a model's answer.

Files, all in the HT-1 folder (Drive human/HT-1/, never in Git)
  lexicon_sheet.csv        people fill it          closed_vocab_sheet.csv
  excluded_items.csv       items never scored      (qa_id, reason, note)
  closed_gold_forms.csv    every count, room, material and colour gold
  corpus_other_names.csv   other short answers in the dataset (for variants)
  reports/                 written by check and freeze
  frozen/                  copies of the three frozen files
Frozen files, in the repo
  lexicon/lexicon.csv, lexicon/closed_vocab.csv   formats of D102
  probes/excluded_items.csv                       qa_ids and reasons, no text

What the script fills in, always from the team's own files
  cluster_id        clusters_parsed.csv with D43 applied; tested against every
                    P-NAME item. People cannot change it.
  same_object_as    pre-filled where two rows are the same spelling under the
                    scorer's spacing rule (D103), where ASSET-1 lists them as
                    one cluster member, or where the lexicon file calls one a
                    colloquial form of the other.
  room_scope        the room section of the lexicon file, where there is one.
  head_noun         the last word of the name. People correct compounds.
  tag_4a            the frozen ASSET-4A tag of that spelling.

Version 2026-10-10 (after the first real check of the HT-1 sheets)
  New ERROR  two different members of clusters_parsed.csv merged into one
             object; a comma, slash or danda used instead of | inside a cell;
             a typed line that has no name_bn.
  New WARNING  a general word made a spelling of one specific object; two
             names that the lexicon file keeps apart inside one object.
  Clearer    each problem is reported once, with the names of the rows; the
             en_route comparison runs even while the sheet still has errors,
             and the report says which checks could not run yet.

Imports, never copies
  normalise()                                   src/normalise.py
  load and build functions, resolve_gold,
  romanise_bn, is_similar, compute_en_unique    src/scoring.py
  load_items, validate_items                    scripts/asset5a_probes.py
  name_id                                       scripts/asset4a_specificity.py
"""

import argparse
import csv
import datetime
import itertools
import json
import re
import shutil
import sys
import tempfile
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "src", _HERE.parent / "scripts"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from normalise import normalise            # noqa: E402
import scoring as S                        # noqa: E402
import asset5a_probes as a5                # noqa: E402
import asset4a_specificity as a4           # noqa: E402

say = a5.say
hkey = a5.hkey

SCRIPT_VERSION = "asset4-v2 2026-10-10"
SEED = 42

# ----------------------------------------------------------------------------
# 0. Fixed values
# ----------------------------------------------------------------------------
ROOM_SCOPES = ("kitchen", "bedroom", "living room", "bathroom", "any")
ORIGINS = ("english", "other_loan", "inherited", "uncertain")
TAGS_4A = ("culturally_specific", "globally_common", "ambiguous")
NOT_RATED = "not_rated"

PROBES = (("P-NAME", "P-NAME.json"), ("P-CORE-CLOSED", "P-CORE-CLOSED.json"),
          ("R-TEST", "R-TEST.json"))
PCLUST_FILE = "P-CLUST.json"
COUNT_COL = {"P-NAME": "n_pname", "P-CORE-CLOSED": "n_pcore_closed",
             "R-TEST": "n_rtest"}

# Room sections of BanglaIndoorVQA_Lexicon.txt. A section that is not listed
# here (the study and electronics section, the new words at the end) gives no
# room: people choose it.
TXT_ROOMS = {normalise(k): v for k, v in {
    "রান্নাঘর": "kitchen", "শোবার ঘর": "bedroom", "বসার ঘর": "living room",
    "বাথরুম": "bathroom", "সাধারণ / সব ঘর": "any"}.items()}
# The four room classes of ASSET-10 use the same words as room_scope.
ROOM_CLASS = {k: v for k, v in TXT_ROOMS.items() if v != "any"}

# D43, as ASSET-1 counted and ASSET-5B applied it. build tests this table
# against the cluster fields of every P-NAME item and stops on any difference.
D43_NOT_A_CLUSTER = normalise("অন্যান্য জোড়া")
D43_PAIR = {normalise("টব"): "টব / ফুলদানি", normalise("ফুলদানি"): "টব / ফুলদানি"}
D43_REMOVED = {(normalise("পানি ও পরিবেশনের পাত্র"), normalise("বদনা"))}

# Words too vague for Tier 2: they fit every object (kept from the v1 script).
VACUOUS = {normalise(x) for x in (
    "জিনিস", "বস্তু", "জিনিসপত্র", "কিছু", "thing", "things", "object",
    "objects", "item", "items", "stuff")}
ARTICLES = ("a", "an", "the")

EXCLUSION_REASONS = ("gold_not_one_object", "gold_wrong_kind", "gold_wrong",
                     "other")

# How English letter names are written in Bangla letters. Used only by the
# en_route cross-check (Design 4.4, "or an acronym match"). One native speaker
# should read this table once.
LETTER_NAMES = {
    "a": ("এ",), "b": ("বি",), "c": ("সি",), "d": ("ডি",), "e": ("ই",),
    "f": ("এফ",), "g": ("জি",), "h": ("এইচ",), "i": ("আই",), "j": ("জে",),
    "k": ("কে",), "l": ("এল",), "m": ("এম",), "n": ("এন",), "o": ("ও",),
    "p": ("পি",), "q": ("কিউ",), "r": ("আর",), "s": ("এস",), "t": ("টি",),
    "u": ("ইউ",), "v": ("ভি",), "w": ("ডব্লিউ", "ডাব্লিউ", "ডব্লু"),
    "x": ("এক্স",), "y": ("ওয়াই",), "z": ("জেড", "জেট")}

# ---- the lexicon sheet -----------------------------------------------------
HUMAN_COLS = ["is_name", "same_object_as", "canonical_en", "variants_en",
              "variants_bn", "super_bn", "super_en", "room_scope", "head_noun",
              "origin3", "origin_note", "en_route", "en_route_note", "writer",
              "checker", "notes"]
INFO_COLS = ["in_scope", "priority", "cluster_id", "cluster_member", "tag_4a",
             "n_pname", "n_pcore_closed", "n_rtest", "freq_obj_name",
             "n_object_answers", "in_lexicon_txt", "room_txt", "example_images",
             "auto_note"]
SHEET_COLS = (["row_id", "name_bn", "priority", "cluster_id"] + HUMAN_COLS
              + [c for c in INFO_COLS if c not in ("priority", "cluster_id")])
CONTENT_COLS = ["canonical_en", "variants_en", "variants_bn", "super_bn",
                "super_en", "room_scope", "origin3", "origin_note", "en_route"]

VOCAB_SHEET_COLS = list(S.VOCAB_COLUMNS) + ["writer", "checker"]
EXCLUSION_COLS = ["qa_id", "reason", "note"]
FROZEN_EXCLUSION_COLS = ["qa_id", "probes", "qa_type", "reason", "source",
                         "added_at"]

SHEET_FILE = "lexicon_sheet.csv"
VOCAB_FILE = "closed_vocab_sheet.csv"
EXCLUSION_FILE = "excluded_items.csv"
STATE_FILE = "asset4_state.json"
GOLD_FORMS_FILE = "closed_gold_forms.csv"
OTHER_NAMES_FILE = "corpus_other_names.csv"

KEY = S.TermIndex()          # the scorer's own matching key (spacing rule on)


# ----------------------------------------------------------------------------
# 1. Small helpers
# ----------------------------------------------------------------------------
def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def key_of(text):
    return KEY.key_of(text)


def split_cell(cell):
    """Split a cell exactly as the scorer's loader does ('|', ';' or newline)."""
    return S._split_cell(cell or "")     # one definition: the scorer's


def script_of(term):
    """bn, en, mixed, digits or none: the scorer's own language rule."""
    return S.detect_language(S.tokenise(normalise(term)))["lang"]


def words_of(term):
    return [t.text for t in S.tokenise(normalise(term))]


_WRONG_SEPARATOR = re.compile(r"[\u0964\u0965,/\\\\]")


def wrong_separators(cell):
    """Characters inside one value that people use by mistake instead of '|'
    (a danda, a comma, a slash). Returns [(value, character)]."""
    out = []
    for term in split_cell(cell):
        m = _WRONG_SEPARATOR.search(term)
        if m:
            out.append((term, m.group()))
    return out


def gold_string(item):
    """The gold as the scorer reads it (Design 8.1)."""
    return S.strip_lead_in(normalise(item["answer_bn"]))


def read_table(path, required=(), what="file"):
    """Read a CSV saved from Sheets, Excel or Python. Header names are
    lower-cased and stripped; every cell is stripped."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"[read] ERROR: {what} not found: {path}")
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            header = [(h or "").strip().lower() for h in (reader.fieldnames or [])]
            rows = []
            for line, raw in enumerate(reader, 2):     # line 1 is the header
                row = {}
                for k, v in raw.items():
                    if k is None:
                        continue
                    if isinstance(v, list):
                        v = "|".join(x or "" for x in v)
                    row[(k or "").strip().lower()] = (v or "").strip()
                if any(row.values()):
                    row["_line"] = str(line)           # the row number in the sheet
                    rows.append(row)
    except UnicodeDecodeError:
        raise SystemExit(
            f"[read] ERROR: {path.name} is not saved as UTF-8. In Google Sheets"
            f" use File > Download > Comma-separated values. In Excel choose"
            f" 'CSV UTF-8'.")
    missing = [c for c in required if c not in header]
    if missing:
        raise SystemExit(f"[read] ERROR: {path.name} lacks column(s) {missing}."
                         f" Found: {header}")
    return rows


def write_table(path, rows, cols, bom=True):
    """bom=True for sheets people open; bom=False for the frozen files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig" if bom else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


class Report:
    """ERROR: a rule is broken. TODO: something is not filled yet.
    WARNING: allowed, but read it. INFO: for the record."""

    def __init__(self):
        self.lines = {"ERROR": [], "TODO": [], "WARNING": [], "INFO": []}
        self.open = OrderedDict()        # what is not filled yet -> [where]
        self.n_open = 0

    def missing(self, what, where):
        """Something is not filled yet. Collected, then printed as one line
        per kind by close(): a TODO in check, an ERROR in freeze."""
        self.open.setdefault(what, []).append(str(where))
        self.n_open += 1

    def close(self, final):
        for what, places in self.open.items():
            line = f"{what}: {len(places)}: " + "; ".join(places)
            (self.error if final else self.todo)(line)
        self.open = OrderedDict()

    def error(self, msg):
        self.lines["ERROR"].append(msg)

    def todo(self, msg):
        self.lines["TODO"].append(msg)

    def warn(self, msg):
        self.lines["WARNING"].append(msg)

    def info(self, msg):
        self.lines["INFO"].append(msg)

    def n(self, kind):
        return len(self.lines[kind])

    def text(self, limit=None):
        out = []
        for kind in ("ERROR", "TODO", "WARNING", "INFO"):
            ls = self.lines[kind]
            shown = ls if limit is None else ls[:limit]
            if limit is not None:
                shown = [m if len(m) <= 600 else m[:600] + " ... (full list in"
                         " the report file)" for m in shown]
            out += [f"{kind:8s} {m}" for m in shown]
            if limit is not None and len(ls) > limit:
                out.append(f"{kind:8s} ... {len(ls) - limit} more in the report file")
        return out


# ----------------------------------------------------------------------------
# 2. Inputs
# ----------------------------------------------------------------------------
_RULE = re.compile(r"^=+$")
_ONE_LINE_HEAD = re.compile(r"^=+\s*([^=].*?)\s*=+$")
_SUB = re.compile(r"^-{3,}\s*(.+?)\s*-{3,}$")
_COLLOQ_PAREN = re.compile(
    r"([^|()\[\]]+?)\s*\(\s*কথ্য(?:\s+সমার্থক)?\s*:\s*([^()\[\]|]+?)\s*\)")
_COLLOQ_BRACKET = re.compile(r"\[\s*কথ্য(?:\s+সমার্থক)?\s*:\s*([^\[\]|]+?)\s*\]")
_ENTRY_DASH = re.compile(r"\s+[—–-]\s+")


def parse_lexicon_txt(path):
    """Read BanglaIndoorVQA_Lexicon.txt, the source of truth for names.

    Returns a dict:
      rooms       {key: set of room words}   from the room sections
      sections    {key: [section titles]}    every section a name stands in
      colloquial  {colloquial form: main name}
      keys        every key written in the file (names and colloquial forms)
      forms       {key: the spelling as written, normalised}
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"[lexicon.txt] ERROR: not found: {path}. Pass"
                         f" --lexicon-txt with its place in Drive.")
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    blocks, title, cur, i = [], "", [], 0
    while i < len(lines):
        s = lines[i].strip()
        if _RULE.match(s):
            boxed = (i + 2 < len(lines) and lines[i + 1].strip()
                     and not _RULE.match(lines[i + 1].strip())
                     and _RULE.match(lines[i + 2].strip()))
            if boxed:
                blocks.append((title, cur))
                title, cur = lines[i + 1].strip(), []
                i += 3
                continue
            i += 1
            continue
        m = _ONE_LINE_HEAD.match(s)
        if m:
            blocks.append((title, cur))
            title, cur = m.group(1).strip(), []
            i += 1
            continue
        cur.append(lines[i])
        i += 1
    blocks.append((title, cur))

    out = {"rooms": defaultdict(set), "sections": defaultdict(list),
           "colloquial": {}, "keys": set(), "forms": {}}

    def add_form(text, section=None):
        n = normalise(text)
        k = key_of(n)
        if not k:
            return None
        out["keys"].add(k)
        out["forms"].setdefault(k, n)
        if section is not None:
            label = section if len(section) <= 40 else "(new words at the end)"
            out["sections"][k].append(label)
            room = TXT_ROOMS.get(normalise(section))
            if room:
                out["rooms"][k].add(room)
        return n

    def add_colloquial(main, coll_text):
        main_n = normalise(main)
        for part in re.split(r"[,/]", coll_text):
            c = normalise(part)
            if c and main_n and key_of(c) != key_of(main_n):
                out["colloquial"][c] = main_n
                add_form(c)
                add_form(main_n)

    def looks_like_name(text):
        t = text.strip().lstrip("◆").strip().rstrip("।").strip()
        return bool(t) and len(t.split()) <= 4 \
            and not re.search(r"[:()\[\]|=—–→•,;?!]", t)

    for title, body in blocks:
        tnorm = normalise(title)
        lines_ = [x.strip() for x in body if x.strip()]
        for s_ in lines_:
            for m in _COLLOQ_PAREN.finditer(s_):
                add_colloquial(m.group(1), m.group(2))
        has_sub = any(_SUB.match(x) for x in lines_)
        entry_lines = [x for x in lines_ if _ENTRY_DASH.search(x)]
        plain = [x for x in lines_ if not _SUB.match(x)]
        if has_sub and len(entry_lines) >= 0.5 * max(1, len(plain)):
            # the "confusable words" block: name, a dash, a description
            seen_sub = False
            for s_ in lines_:
                if _SUB.match(s_):
                    seen_sub = True
                    continue
                if not seen_sub or not _ENTRY_DASH.search(s_):
                    continue
                head = _ENTRY_DASH.split(s_, maxsplit=1)[0]
                parts = [x for x in re.split(r"[/=]", head) if normalise(x)]
                for x in parts:
                    add_form(x)
                for m in _COLLOQ_BRACKET.finditer(s_):
                    add_colloquial(parts[0], m.group(1))
            continue
        if tnorm.startswith(normalise("ঘরের নাম")):
            continue                  # the four room names are not objects
        names = [x for x in plain if looks_like_name(x)]
        if plain and len(names) >= 0.9 * len(plain):
            # a list of names: a room section, the study section, new words
            for x in names:
                add_form(x.lstrip("◆").strip(), section=title)
        # anything else is prose (the header): only its colloquial notes count
    say(f"[lexicon.txt] {len(out['keys'])} names and colloquial forms,"
        f" {len(out['rooms'])} with a room section,"
        f" {len(out['colloquial'])} colloquial notes, from {path}")
    return out


def load_clusters(parsed_csv, counts_csv=None):
    """clusters_parsed.csv with D43 applied.

    Returns (members, form_to_member, counts). A member is a dict with
    cluster_id ("" if the member is in no cluster), member and forms."""
    rows = read_table(parsed_csv, ("cluster_id", "member", "surface_forms"),
                      "clusters_parsed.csv")

    def d43(cid, mem):
        c, m = normalise(cid), normalise(mem)
        if (c, m) in D43_REMOVED:
            return None
        if c == D43_NOT_A_CLUSTER:
            return D43_PAIR.get(m, "")
        return cid.strip()

    members, form_to = [], {}
    for r in rows:
        cid = d43(r["cluster_id"], r["member"])
        if cid is None:
            continue
        forms = []
        for part in r["surface_forms"].split("|"):
            f = normalise(part)
            if f and f not in forms:
                forms.append(f)
        if not forms:
            raise SystemExit(f"[clusters] ERROR: member {r['member']!r} has no"
                             f" surface form in {parsed_csv}")
        mem = {"cluster_id": cid, "member": r["member"].strip(), "forms": forms}
        members.append(mem)
        for f in forms:
            if f in form_to:
                raise SystemExit(f"[clusters] ERROR: the form {f!r} belongs to"
                                 f" two members after D43: {form_to[f]['member']!r}"
                                 f" and {mem['member']!r}. Stop and ask.")
            form_to[f] = mem
    counts = {}
    if counts_csv and Path(counts_csv).exists():
        for r in read_table(counts_csv, ("cluster_id", "member", "obj_name_count"),
                            "cluster_counts.csv"):
            if d43(r["cluster_id"], r["member"]) is None:
                continue
            counts[normalise(r["member"])] = int(r["obj_name_count"])
    n_cl = len({m["cluster_id"] for m in members if m["cluster_id"]})
    say(f"[clusters] {len(members)} members, {sum(1 for m in members if m['cluster_id'])}"
        f" of them in {n_cl} clusters, {len(form_to)} surface forms (D43 applied)")
    return members, form_to, counts


def load_tags(path):
    rows = read_table(path, ("name_id", "name_bn", "cultural_specificity"),
                      "cultural_specificity.csv")
    tags = {}
    for r in rows:
        if a4.name_id(normalise(r["name_bn"])) != r["name_id"]:
            raise SystemExit(f"[tags] ERROR: name_id of {r['name_bn']!r} in {path}"
                             f" is not the one asset4a_specificity.name_id gives."
                             f" Wrong file?")
        if r["cultural_specificity"] not in TAGS_4A:
            raise SystemExit(f"[tags] ERROR: unknown tag {r['cultural_specificity']!r}")
        tags[r["name_id"]] = {"tag": r["cultural_specificity"],
                              "name_bn": normalise(r["name_bn"]),
                              "n_obj_name": r.get("n_obj_name", "")}
    c = Counter(t["tag"] for t in tags.values())
    say(f"[tags] {len(tags)} frozen ASSET-4A tags: "
        + ", ".join(f"{c[k]} {k}" for k in TAGS_4A))
    return tags


def load_probes(probe_dir):
    probe_dir = Path(probe_dir)
    probes = OrderedDict()
    for name, fname in PROBES:
        probes[name] = S.load_probe(probe_dir / fname)
        c = Counter(it["qa_type"] for it in probes[name])
        say(f"[probes] {name}: {len(probes[name])} items ("
            + ", ".join(f"{k} {c[k]}" for k in sorted(c)) + ")")
    pclust = S.load_probe(probe_dir / PCLUST_FILE)
    ids = {it["qa_id"] for it in probes["P-NAME"]}
    if not all(it["qa_id"] in ids for it in pclust):
        raise SystemExit("[probes] ERROR: P-CLUST is not a subset of P-NAME")
    say(f"[probes] P-CLUST: {len(pclust)} items in"
        f" {len({it.get('cluster_id') for it in pclust})} clusters")
    return probes, pclust


def unique_items(probes):
    """Every item once, with the probe sets it sits in."""
    out = OrderedDict()
    for name, items in probes.items():
        for it in items:
            rec = out.setdefault(it["qa_id"], {"item": it, "probes": []})
            rec["probes"].append(name)
    return out


def corpus_counters(items):
    obj = Counter(gold_string(it) for it in items if it["qa_type"] == "obj_name")
    objt = Counter(gold_string(it) for it in items if it["qa_type"] == "object")
    return obj, objt


def load_corpus(data_path, counts_csv, validate=True):
    items = a5.load_items(str(data_path))
    if validate:
        a5.validate_items(items, str(counts_csv) if counts_csv else None)
    return items


def read_5c_flags(path):
    """Golds that ASSET-5C asked people to look at (optional input)."""
    flags = defaultdict(list)
    if not path or not Path(path).exists():
        return flags
    for r in read_table(path, (), "asset5c_review.csv"):
        if r.get("block", "").startswith("kept") and r.get("gold"):
            flags[gold_string({"answer_bn": r["gold"]})].append(
                f"5C: {r.get('note', '')} ({r.get('qa_id', '')})")
    return flags


# ----------------------------------------------------------------------------
# 3. The rows of the sheet
# ----------------------------------------------------------------------------
def compute_scope(probes, pclust, members, form_to, tags, txt, obj_counter,
                  objt_counter, flags=None, corpus_items=None):
    """Everything the script knows about each row, from the input files only."""
    flags = flags or {}
    rows = OrderedDict()

    def row(n):
        if n not in rows:
            if not key_of(n):
                raise SystemExit(f"[scope] ERROR: the gold {n!r} has no letters")
            if "|" in n or ";" in n:
                raise SystemExit(f"[scope] ERROR: the gold {n!r} contains | or ;,"
                                 f" which separate values in a cell. Stop and ask.")
            rows[n] = {"name_bn": n, "row_id": a4.name_id(n), "n": Counter(),
                       "images": OrderedDict(), "sources": set()}
        return rows[n]

    for probe, items in probes.items():
        for it in items:
            if it["qa_type"] in S.NAMING_TYPES:
                r = row(gold_string(it))
                r["n"][probe] += 1
                r["sources"].add("gold")
                r["images"].setdefault(it["image_id"],
                                       it.get("image_file") or it["image_id"])
    for m in members:
        for f in m["forms"]:
            row(f)["sources"].add("cluster")
    for coll, main in txt["colloquial"].items():
        if main in rows or coll in rows:
            row(coll)["sources"].add("colloquial")
            row(main)["sources"].add("colloquial")
    for n in list(rows):
        # the lexicon file is the source of truth for name_bn (Design 4.2):
        # if it writes this name with another spacing, that spelling gets a row
        written = txt["forms"].get(key_of(n))
        if written and written != n and key_of(written) == key_of(n):
            row(written)["sources"].add("lexicon file spelling")

    ids = Counter(r["row_id"] for r in rows.values())
    clash = [i for i, c in ids.items() if c > 1]
    if clash:
        raise SystemExit(f"[scope] ERROR: row_id clash {clash}. Stop and ask.")

    if corpus_items:
        for it in corpus_items:
            if it["qa_type"] in S.NAMING_TYPES:
                n = gold_string(it)
                if n in rows and len(rows[n]["images"]) < 50:
                    rows[n]["images"].setdefault(
                        it["image_id"], it.get("image_file") or it["image_id"])

    qualifying = sorted({it["cluster_id"] for it in pclust if it.get("cluster_id")})

    # groups that the files force to be one object
    by_key = defaultdict(list)
    for n in rows:
        by_key[key_of(n)].append(n)
    key_groups = [sorted(v) for v in by_key.values() if len(v) > 1]
    member_groups = [list(m["forms"]) for m in members if len(m["forms"]) > 1]

    parent = {n: n for n in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    why, kinds = defaultdict(list), defaultdict(list)

    def others(g, x):
        return ", ".join(y for y in g if y != x)

    for g in key_groups:
        for x in g[1:]:
            union(g[0], x)
        for x in g:
            why[x].append(f"same spelling as {others(g, x)} under the spacing"
                          f" rule (D103)")
            kinds[x].append("spacing rule (D103)")
    for g in member_groups:
        for x in g[1:]:
            union(g[0], x)
        for x in g:
            why[x].append(f"one cluster member with {others(g, x)} in"
                          f" clusters_parsed.csv")
            kinds[x].append("one cluster member (ASSET-1)")
    for coll, main in txt["colloquial"].items():
        if coll in rows and main in rows:
            union(coll, main)
            why[coll].append(f"colloquial form of {main} in the lexicon file")
            kinds[coll].append("colloquial form (lexicon file)")

    first_form = {m["forms"][0] for m in members}
    mains = set(txt["colloquial"].values())
    colls = set(txt["colloquial"])

    def rank(n):
        # 1 the name the lexicon file marks as the main one of a colloquial
        # pair; 2 a spelling exactly as the file writes it; 3 the spelling the
        # annotators used most; 4 the first form ASSET-1 lists.
        r = rows[n]
        return (n in mains,
                txt["forms"].get(key_of(n)) == n and n not in colls,
                key_of(n) in txt["keys"] and n not in colls,
                sum(r["n"].values()), obj_counter.get(n, 0), n in first_form)

    sets = defaultdict(list)
    for n in rows:
        sets[find(n)].append(n)
    suggested = {}
    for group in sets.values():
        if len(group) < 2:
            continue
        best = sorted(group, key=lambda n: (tuple(-int(x) for x in rank(n)), n))[0]
        for n in group:
            if n != best:
                suggested[n] = best

    for n, r in rows.items():
        mem = form_to.get(n)
        rooms = txt["rooms"].get(key_of(n), set())
        r["cluster_id"] = mem["cluster_id"] if mem else ""
        r["cluster_member"] = mem["member"] if mem else ""
        r["tag_4a"] = tags.get(r["row_id"], {}).get("tag", "")
        r["freq_obj_name"] = obj_counter.get(n, 0)
        r["n_object_answers"] = objt_counter.get(n, 0)
        r["in_lexicon_txt"] = "yes" if key_of(n) in txt["keys"] else "no"
        r["room_txt"] = " | ".join(txt["sections"].get(key_of(n), []))
        r["room_prefill"] = (next(iter(rooms)) if len(rooms) == 1
                             else "any" if len(rooms) > 1 else "")
        r["head_prefill"] = re.split(r"[\s\-]+", n)[-1]
        r["suggest_same_as"] = suggested.get(n, "")
        r["merge_kind"] = kinds.get(n, [""])[0]
        notes = list(why.get(n, [])) + flags.get(n, [])
        r["auto_note"] = "; ".join(dict.fromkeys(notes))
        n1 = r["n"]["P-NAME"] + r["n"]["P-CORE-CLOSED"]
        if n1 or (mem and mem["cluster_id"] in qualifying):
            r["priority"] = 1
        elif r["n"]["R-TEST"]:
            r["priority"] = 2
        else:
            r["priority"] = 3
        ex = sorted(r["images"], key=lambda img: hkey(SEED, "4-example", img))[:3]
        r["example_images"] = " ".join(str(r["images"][i]) for i in ex)

    return {"rows": rows, "qualifying": qualifying, "key_groups": key_groups,
            "member_groups": member_groups}


def info_cells(r):
    return {"in_scope": "yes", "priority": r["priority"],
            "cluster_id": r["cluster_id"], "cluster_member": r["cluster_member"],
            "tag_4a": r["tag_4a"], "n_pname": r["n"]["P-NAME"],
            "n_pcore_closed": r["n"]["P-CORE-CLOSED"], "n_rtest": r["n"]["R-TEST"],
            "freq_obj_name": r["freq_obj_name"],
            "n_object_answers": r["n_object_answers"],
            "in_lexicon_txt": r["in_lexicon_txt"], "room_txt": r["room_txt"],
            "example_images": r["example_images"], "auto_note": r["auto_note"]}


def new_sheet_row(r):
    row = {c: "" for c in SHEET_COLS}
    row.update(row_id=r["row_id"], name_bn=r["name_bn"], is_name="yes",
               same_object_as=r["suggest_same_as"])
    if not r["suggest_same_as"]:
        row.update(room_scope=r["room_prefill"], head_noun=r["head_prefill"])
    row.update(info_cells(r))
    return row


def sort_sheet(rows):
    def k(r):
        n = sum(int(r.get(c) or 0) for c in COUNT_COL.values())
        return (r.get("in_scope") != "yes", int(r.get("priority") or 9), -n,
                r["name_bn"])
    return sorted(rows, key=k)


# ----------------------------------------------------------------------------
# 4. The en_route cross-check (Design 4.4)
# ----------------------------------------------------------------------------
_LETTER_OF = {}
for _letter, _names in LETTER_NAMES.items():
    for _nm in _names:
        _LETTER_OF["".join(words_of(_nm))] = _letter
_MAX_LETTER_NAME = max(len(k) for k in _LETTER_OF)


def acronym_readings(word):
    """Latin letter strings a Bangla word spells when read as letter names:
    এসি -> {'ac'}. Empty if the word is not made of letter names only."""
    out = set()

    def walk(pos, acc):
        if pos == len(word):
            if len(acc) >= 2:
                out.add("".join(acc))
            return
        for size in range(1, _MAX_LETTER_NAME + 1):
            piece = word[pos:pos + size]
            if len(piece) == size and piece in _LETTER_OF:
                walk(pos + size, acc + [_LETTER_OF[piece]])

    if len(word) <= 24:
        walk(0, [])
    return out


def _letters(text):
    return re.sub(r"[^a-z]", "", text)


def en_route_script(t1_bn, t1_en):
    """Design 4.4: is a romanised Bangla Tier-1 name close to an English
    Tier-1 name of the same object (similarity 0.6 or more, the scorer's own
    functions), or is it that name spelt as letters?
    Returns (flag, evidence text)."""
    best = None
    for b in t1_bn:
        toks = [t for t in S.tokenise(normalise(b))]
        options = []
        for t in toks:
            if t.kind == "bn":
                opts = [_letters(S.romanise_bn(t.text))]
                opts += sorted(acronym_readings(t.text))
            else:
                opts = [_letters(t.text)]
            options.append(list(dict.fromkeys(o for o in opts if o)) or [""])
        combos = itertools.islice(itertools.product(*options), 64)
        for combo in combos:
            rom = "".join(combo)
            plain = rom == "".join(o[0] for o in options)
            for e in t1_en:
                target = _letters(normalise(e))
                if not rom or not target:
                    continue
                if S.is_similar(rom, target):
                    sim = S.edit_similarity(rom, target)
                    how = "similarity" if plain else "acronym"
                    cand = (sim, f"{b} ~ {e} ({how} {sim:.2f}, read as '{rom}')")
                    if best is None or cand[0] > best[0]:
                        best = cand
    if best:
        return True, best[1]
    return False, ""


# ----------------------------------------------------------------------------
# 5. build
# ----------------------------------------------------------------------------
def resolve_paths(args):
    d = Path(args.drive)
    assets = d / "results" / "assets"

    def pick(name, default):
        v = getattr(args, name, None)
        return Path(v) if v else default

    return {
        "data": pick("data", d / "data" / "banglaindoorvqa.json"),
        "counts": pick("counts", assets / "house_type_counts.csv"),
        "probes": pick("probes", d / "probes"),
        "tags": pick("tags", assets / "cultural_specificity.csv"),
        "clusters": pick("clusters", assets / "clusters_parsed.csv"),
        "cluster_counts": pick("cluster_counts", assets / "cluster_counts.csv"),
        "unmatched": pick("unmatched", assets / "unmatched_obj_name.csv"),
        "lexicon_txt": pick("lexicon_txt", d / "data" / "BanglaIndoorVQA_Lexicon.txt"),
        "review_5c": pick("review_5c", d / "probes" / "asset5c_review.csv"),
        "work": pick("work", d / "human" / "HT-1"),
    }


def self_checks(probes, pclust, form_to, tags, corpus_items, counts,
                unmatched_csv, obj_counter, compare_counts=True):
    """Hard checks that the inputs belong together. Each prints one line."""
    # 1. the name_id rule
    bad = [it["qa_id"] for it in probes["P-NAME"]
           if it.get("name_id") and a4.name_id(gold_string(it)) != it["name_id"]]
    if bad:
        raise SystemExit(f"[check] ERROR: name_id of {len(bad)} P-NAME items is not"
                         f" asset4a_specificity.name_id(gold), e.g. {bad[:3]}")
    say(f"[check] name_id rule reproduces all {len(probes['P-NAME'])} P-NAME ids")

    # 2. D43 table against the cluster fields of P-NAME
    bad = []
    for it in probes["P-NAME"]:
        if "cluster_id" not in it:
            continue
        mem = form_to.get(gold_string(it))
        got = (mem["cluster_id"], mem["member"]) if mem else ("", "")
        if (it["cluster_id"], it.get("cluster_member", got[1])) != got:
            bad.append((it["qa_id"], it["cluster_id"], got[0]))
    if bad:
        raise SystemExit(f"[check] ERROR: the D43 cluster table differs from the"
                         f" cluster fields of {len(bad)} P-NAME items, e.g."
                         f" {bad[:3]}. Stop and ask.")
    say("[check] cluster table (D43 applied) equals the cluster fields of every"
        " P-NAME item")

    # 3. the probe text is the dataset's text
    if corpus_items is not None:
        by_id = {it["qa_id"]: it for it in corpus_items}
        stale = []
        for qid, rec in unique_items(probes).items():
            it = rec["item"]
            src = by_id.get(qid)
            if src is None or src["qa_type"] != it["qa_type"] \
                    or normalise(src["answer_bn"]) != normalise(it["answer_bn"]):
                stale.append(qid)
        if stale:
            raise SystemExit(
                f"[check] ERROR: {len(stale)} probe items differ from the dataset"
                f" (missing, other type or other gold), e.g. {stale[:5]}. The"
                f" probe files are older than the dataset: run asset5_rerun.ipynb"
                f" first (DESIGN_ADDENDUM A3).")
        say(f"[check] all {len(unique_items(probes))} probe items carry the"
            f" dataset's current gold")

    # 4. obj_name counts against ASSET-1 and ASSET-4A (differences are listed)
    diffs = []
    if not compare_counts:
        return diffs
    for member, want in counts.items():
        forms = next((m["forms"] for m in set_of_members(form_to)
                      if normalise(m["member"]) == member), None)
        if forms is None:
            continue
        got = sum(obj_counter.get(f, 0) for f in forms)
        if got != want:
            diffs.append(f"cluster member {member}: ASSET-1 {want}, dataset now {got}")
    for t in tags.values():
        if str(t["n_obj_name"]).isdigit() and int(t["n_obj_name"]) != obj_counter.get(t["name_bn"], 0):
            diffs.append(f"4A name {t['name_bn']}: 4A {t['n_obj_name']}, dataset"
                         f" now {obj_counter.get(t['name_bn'], 0)}")
    if unmatched_csv and Path(unmatched_csv).exists():
        for r in read_table(unmatched_csv, ("normalised_answer", "count"),
                            "unmatched_obj_name.csv"):
            n = normalise(r["normalised_answer"])
            if int(r["count"]) != obj_counter.get(n, 0):
                diffs.append(f"unmatched name {n}: ASSET-1 {r['count']}, dataset"
                             f" now {obj_counter.get(n, 0)}")
    if diffs:
        say(f"[check] WARNING: {len(diffs)} obj_name counts differ from the ASSET-1"
            f" or 4A tables. The dataset counts are used. Differences can only"
            f" come from dataset corrections made after those tables; send this"
            f" list to the chat if it is long:")
        for d in diffs[:30]:
            say(f"          {d}")
    else:
        say("[check] obj_name counts equal the ASSET-1 and 4A tables for every"
            " listed name")
    return diffs


def set_of_members(form_to):
    seen, out = set(), []
    for m in form_to.values():
        if id(m) not in seen:
            seen.add(id(m))
            out.append(m)
    return out


def closed_gold_table(probes):
    """Every count, room, material and colour gold string, per 5C kind."""
    table = OrderedDict()
    for probe in ("P-CORE-CLOSED", "R-TEST"):
        for it in probes[probe]:
            if it["qa_type"] in S.NAMING_TYPES:
                continue
            k = (it.get("closed_kind", ""), gold_string(it))
            rec = table.setdefault(k, {"kind_5c": k[0], "gold": k[1],
                                       "n_pcore_closed": 0, "n_rtest": 0,
                                       "example_qa_id": it["qa_id"],
                                       "value_5c": it.get("gold_count",
                                                          it.get("gold_room", ""))})
            rec[COUNT_COL[probe]] += 1
    return table


def vocab_skeleton(probes):
    """Rows people start from. Only gold forms are pre-filled: the room golds
    by their 5C room and the count golds by their 5C number."""
    rooms, numbers = defaultdict(list), defaultdict(list)
    for rec in closed_gold_table(probes).values():
        if rec["kind_5c"] == "room":
            rooms[normalise(str(rec["value_5c"]))].append(rec["gold"])
        elif rec["kind_5c"] == "count":
            numbers[int(rec["value_5c"])].append(rec["gold"])
    rows = []
    for bn, cls in ROOM_CLASS.items():
        rows.append({"kind": "room", "class_id": cls,
                     "forms_bn": "|".join(dict.fromkeys([bn] + rooms.get(bn, []))),
                     "forms_en": "",
                     "note": "room name and gold forms pre-filled; add the other forms"})
    for bn in rooms:
        if bn not in ROOM_CLASS:
            raise SystemExit(f"[vocab] ERROR: gold_room {bn!r} is not one of the"
                             f" four rooms. Stop and ask.")
    wanted = sorted(set(range(0, 21)) | set(range(30, 101, 10)) | set(numbers))
    for n in wanted:
        rows.append({"kind": "number", "class_id": str(n),
                     "forms_bn": "|".join(dict.fromkeys(numbers.get(n, []))),
                     "forms_en": "",
                     "note": ("a gold number: required" if n in numbers
                              else "fill it or delete the row")})
    return rows


def cmd_build(args):
    P = resolve_paths(args)
    say(f"ASSET-4 v2 build ({SCRIPT_VERSION}). No model output is read.")
    work = P["work"]
    sheet_path = work / SHEET_FILE
    if sheet_path.exists() and not args.update:
        raise SystemExit(
            f"[guard] STOP: {sheet_path} exists. A new build would erase what"
            f" people wrote. To refresh it after a change in the dataset or"
            f" the probe files, run build --update. It keeps every human cell.")
    if args.update and not sheet_path.exists():
        raise SystemExit(f"[guard] STOP: --update needs an existing {sheet_path}")

    probes, pclust = load_probes(P["probes"])
    tags = load_tags(P["tags"])
    members, form_to, counts = load_clusters(P["clusters"], P["cluster_counts"])
    txt = parse_lexicon_txt(P["lexicon_txt"])
    corpus = load_corpus(P["data"], P["counts"], validate=not args.no_validate)
    obj_counter, objt_counter = corpus_counters(corpus)
    say(f"[data] {sum(obj_counter.values())} obj_name answers,"
        f" {len(obj_counter)} distinct after normalise()")
    want = getattr(a4, "EXPECTED_OBJ_NAME_ITEMS", None)
    if want and not args.no_validate and sum(obj_counter.values()) != want:
        say(f"[data] WARNING: the 4A script expects {want} obj_name items. Is this"
            f" the corrected merged dataset?")
    self_checks(probes, pclust, form_to, tags, corpus, counts, P["unmatched"],
                obj_counter)
    flags = read_5c_flags(P["review_5c"])
    scope = compute_scope(probes, pclust, members, form_to, tags, txt,
                          obj_counter, objt_counter, flags, corpus)
    rows = scope["rows"]

    work.mkdir(parents=True, exist_ok=True)
    if args.update:
        old = read_table(sheet_path, ("row_id", "name_bn"), "the lexicon sheet")
        stamp = utc_now().replace(":", "").replace("+0000", "Z")
        backup = work / f"lexicon_sheet.backup-{stamp}.csv"
        shutil.copy(sheet_path, backup)
        say(f"[update] backup of the sheet: {backup}")
        old_by = {normalise(r["name_bn"]): r for r in old}
        out, n_new, n_gone = [], 0, 0
        for n, r in rows.items():
            if n in old_by:
                row = {c: old_by[n].get(c, "") for c in SHEET_COLS}
                row.update(row_id=r["row_id"], name_bn=n)
                row.update(info_cells(r))
                if old_by[n].get("in_scope") == "no":
                    row["auto_note"] = "back in scope; " + row["auto_note"]
            else:
                row = new_sheet_row(r)
                row["auto_note"] = "NEW since the last build; " + row["auto_note"]
                n_new += 1
            out.append(row)
        for n, r in old_by.items():
            if n not in rows:
                row = {c: r.get(c, "") for c in SHEET_COLS}
                if row.get("in_scope") != "no":
                    n_gone += 1
                row["in_scope"] = "no"
                row["auto_note"] = "no longer a gold or a cluster form; ignored"
                for c in COUNT_COL.values():
                    row[c] = 0
                out.append(row)
        say(f"[update] {n_new} new rows, {n_gone} rows left the scope,"
            f" {len(out) - n_new} rows kept with their human cells")
    else:
        out = [new_sheet_row(r) for r in rows.values()]
    out = sort_sheet(out)
    write_table(sheet_path, out, SHEET_COLS)
    say(f"[write] {sheet_path} ({len(out)} rows)")

    vocab_path = work / VOCAB_FILE
    if not vocab_path.exists():
        vrows = vocab_skeleton(probes)
        write_table(vocab_path, vrows, VOCAB_SHEET_COLS)
        say(f"[write] {vocab_path} ({len(vrows)} skeleton rows: 4 rooms and the"
            f" numbers; materials and colours are written by people)")
    else:
        say(f"[keep]  {vocab_path} exists and is left as it is")
    excl_path = work / EXCLUSION_FILE
    if not excl_path.exists():
        write_table(excl_path, [], EXCLUSION_COLS)
        say(f"[write] {excl_path} (empty; reasons allowed: "
            + ", ".join(EXCLUSION_REASONS) + ")")

    gold_rows = list(closed_gold_table(probes).values())
    write_table(work / GOLD_FORMS_FILE, gold_rows,
                ["kind_5c", "gold", "value_5c", "n_pcore_closed", "n_rtest",
                 "example_qa_id"])
    say(f"[write] {work / GOLD_FORMS_FILE} ({len(gold_rows)} distinct count, room,"
        f" material and colour golds)")

    in_sheet = set(rows)
    others = []
    for n in sorted(set(obj_counter) | set(objt_counter)):
        if n in in_sheet or len(n.split()) > 3 or not key_of(n):
            continue
        others.append({"answer": n, "n_obj_name": obj_counter.get(n, 0),
                       "n_object": objt_counter.get(n, 0)})
    others.sort(key=lambda r: (-(r["n_obj_name"] + r["n_object"]), r["answer"]))
    write_table(work / OTHER_NAMES_FILE, others, ["answer", "n_obj_name", "n_object"])
    say(f"[write] {work / OTHER_NAMES_FILE} ({len(others)} other answers of one"
        f" to three words in the dataset; look here for attested spellings)")

    state = {"asset": "ASSET-4 v2", "script": SCRIPT_VERSION, "built_utc": utc_now(),
             "obj_name_counts": dict(obj_counter),
             "object_counts": dict(objt_counter)}
    with open(work / STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    say(f"[write] {work / STATE_FILE}")

    pri = Counter(r["priority"] for r in rows.values())
    src = Counter()
    for r in rows.values():
        src["gold" if "gold" in r["sources"] else "form only"] += 1
    pre = Counter()
    for r in rows.values():
        if r["suggest_same_as"]:
            pre[r["merge_kind"]] += 1
    out_scope = sorted(t["name_bn"] for i, t in tags.items()
                       if t["name_bn"] not in rows)
    say()
    say("=" * 72)
    say(f"Lexicon sheet: {len(rows)} rows = {src['gold']} naming gold strings of"
        f" the three probe sets + {src['form only']} cluster or colloquial forms"
        f" that are never a gold.")
    say(f"Priority 1 (P-NAME, P-CORE-CLOSED, qualifying clusters): {pri[1]};"
        f" priority 2 (R-TEST only): {pri[2]}; priority 3: {pri[3]}.")
    say(f"same_object_as pre-filled on {sum(pre.values())} rows: "
        + "; ".join(f"{v} {k}" for k, v in pre.items()))
    say(f"Rows not written in the lexicon file: "
        f"{sum(1 for r in rows.values() if r['in_lexicon_txt'] == 'no')}."
        f" Rows with a room from the file: "
        f"{sum(1 for r in rows.values() if r['room_prefill'])}.")
    say(f"4A tags: {sum(1 for r in rows.values() if r['tag_4a'])} rows carry one;"
        f" {len(out_scope)} tagged names are outside the sheet: {out_scope}")
    say(f"Qualifying clusters: {len(scope['qualifying'])}.")
    say("Next: two people fill the sheets (HT1_instructions.md), then run check.")
    say("=" * 72)


# ----------------------------------------------------------------------------
# 6. From the sheet to objects
# ----------------------------------------------------------------------------
def load_state_counters(work):
    p = Path(work) / STATE_FILE
    if not p.exists():
        raise SystemExit(f"[state] ERROR: {p} is missing. Run build first, or pass"
                         f" --data so the counts are read from the dataset.")
    with open(p, encoding="utf-8") as f:
        st = json.load(f)
    return Counter(st["obj_name_counts"]), Counter(st["object_counts"])


def assemble(sheet, scope, tags, txt, obj_counter, attested_keys, rep, final):
    """Turn sheet rows into objects and check every rule that concerns one
    row or one object. Returns (objects, name_to_object, excluded_names)."""
    srows = scope["rows"]

    by_name = {}
    for r in sheet:
        n = normalise(r["name_bn"])
        if not n:
            typed = [c for c in SHEET_COLS if (r.get(c) or "").strip()]
            rep.error(f"sheet line {r.get('_line', '?')} has no name_bn, but"
                      f" something is typed in: {', '.join(typed)}. This line is"
                      f" not a row of the sheet. Clear these cells")
            continue
        if n in by_name:
            rep.error(f"{n}: the sheet has this row twice (lines"
                      f" {by_name[n].get('_line', '?')} and {r.get('_line', '?')})")
            continue
        by_name[n] = r
    for n, r in by_name.items():
        if a4.name_id(n) != r["row_id"]:
            rep.error(f"{r['name_bn']}: name_bn or row_id was edited. The name is"
                      f" the data's spelling and is fixed; use same_object_as or"
                      f" variants_bn instead")
    missing = [n for n in srows if n not in by_name]
    extra = [n for n, r in by_name.items()
             if n not in srows and r.get("in_scope", "yes") != "no"]
    if missing:
        rep.error(f"{len(missing)} rows are missing from the sheet, e.g."
                  f" {missing[:4]}. Rows are never deleted. If the dataset or the"
                  f" probe files changed, run build --update")
    if extra:
        rep.error(f"{len(extra)} rows are in the sheet but are not a gold or a"
                  f" cluster form (any more), e.g. {extra[:4]}. Rows are added"
                  f" only by build --update; run it")
    stale = [n for n in srows if n in by_name and any(
        str(by_name[n].get(c, "")) != str(srows[n]["n"][p])
        for p, c in COUNT_COL.items())]
    if stale:
        rep.error(f"the item counts of {len(stale)} rows differ from the probe"
                  f" files, e.g. {stale[:4]}. The sheet is older than the probe"
                  f" files: run build --update")
    for n, r in by_name.items():
        if n not in srows and any(r.get(c) for c in CONTENT_COLS):
            rep.warn(f"{n}: is out of scope now, so what was written on it is"
                     f" ignored")

    rows = {n: by_name[n] for n in srows if n in by_name}
    is_name, target = {}, {}
    for n, r in rows.items():
        v = (r.get("is_name") or "yes").strip().lower()
        if v not in ("yes", "no"):
            rep.error(f"{n}: is_name must be yes or no, found {r['is_name']!r}")
            v = "yes"
        if v == "no" and srows[n]["cluster_member"]:
            rep.error(f"{n}: is_name = no, but clusters_parsed.csv lists it as a"
                      f" cluster member. It is a name")
            v = "yes"
        is_name[n] = v == "yes"
    for n, r in rows.items():
        s = normalise(r.get("same_object_as"))
        if not s:
            continue
        if not is_name[n]:
            rep.error(f"{n}: is_name = no and same_object_as are both set."
                      f" Choose one")
        elif s == n:
            rep.error(f"{n}: same_object_as points to itself")
        elif s not in rows:
            rep.error(f"{n}: same_object_as {s!r} is not a name_bn of the sheet")
        elif normalise(rows[s].get("same_object_as")):
            rep.error(f"{n}: same_object_as points to {s!r}, which is itself a"
                      f" spelling of another row. Point to the final row")
        elif not is_name[s]:
            rep.error(f"{n}: same_object_as points to {s!r}, which has is_name"
                      f" = no")
        else:
            target[n] = s
            filled = [c for c in CONTENT_COLS if (r.get(c) or "").strip()]
            if filled:
                rep.warn(f"{n}: is a spelling of {s!r}, so its {', '.join(filled)}"
                         f" cells are ignored. Write them on the row of {s!r}")

    canon = {n: target.get(n, n) for n in rows if is_name[n]}

    def one_object(group, reason):
        g = [n for n in group if n in rows]
        bad = [n for n in g if not is_name[n]]
        if bad:
            rep.error(f"{bad}: is_name = no is not allowed here: {reason}")
            return
        if len({canon[n] for n in g}) > 1:
            rep.error(f"{g} must be one object: {reason}. Set same_object_as on"
                      f" all but one of them")

    for g in scope["key_groups"]:
        one_object(g, "they are the same spelling for the scorer (spacing rule,"
                      " D103)")
    for g in scope["member_groups"]:
        one_object(g, "clusters_parsed.csv lists them as one cluster member."
                      " If you think they are two objects, tell the team lead:"
                      " it needs a decision row, not a sheet edit")
    for coll, main in txt["colloquial"].items():
        if coll in rows and main in rows and is_name.get(coll) \
                and canon.get(coll) != canon.get(main):
            rep.warn(f"{coll}: the lexicon file lists it as a colloquial form of"
                     f" {main}, but the sheet keeps them apart")

    for n, r in rows.items():
        w, c = (r.get("writer") or "").strip(), (r.get("checker") or "").strip()
        if not w or not c:
            rep.missing("writer or checker is empty", n)
        elif w.lower() == c.lower():
            rep.error(f"{n}: writer and checker are the same person ({w})."
                      f" The checker of a row is never its writer (HT-1)")

    groups = defaultdict(list)
    for n, c in canon.items():
        groups[c].append(n)
    all_names = set(rows)
    objects = OrderedDict()
    for c in sorted(groups):
        r = rows[c]
        spellings = [n for n in groups[c] if n != c]
        o = {"name": c, "row": r, "name_id": r["row_id"], "spellings": spellings}
        # ---- Bangla Tier 1
        extra_bn = [t for t in split_cell(r.get("variants_bn")) if t != c]
        ok_extra = []
        for t in extra_bn:
            if t in all_names:
                if canon.get(t) == c:
                    rep.warn(f"{c}: variants_bn lists {t!r}, which already is a"
                             f" row of this object")
                else:
                    rep.error(f"{c}: variants_bn lists {t!r}, which has its own"
                              f" row. Set same_object_as on that row instead")
                continue
            if key_of(t) not in attested_keys:
                rep.error(f"{c}: variants_bn term {t!r} is not a gold answer in"
                          f" the dataset and is not in the lexicon file. Design"
                          f" 4.2 allows only those. Remove it")
                continue
            ok_extra.append(t)
        spellings_sorted = sorted(
            spellings, key=lambda n: (-sum(srows[n]["n"].values()), n))
        o["t1_bn"] = [c] + spellings_sorted + ok_extra
        # ---- English Tier 1
        raw_en = (r.get("canonical_en") or "").strip()
        cen = split_cell(raw_en)
        if len(cen) > 1:
            rep.error(f"{c}: canonical_en must be ONE name; put the others in"
                      f" variants_en")
        if not cen:
            rep.missing("canonical_en is empty", c)
        ven = [t for t in split_cell(r.get("variants_en")) if t not in cen[:1]]
        o["canonical_en"] = cen[0] if cen else ""
        o["variants_en"] = ven
        o["t1_en"] = cen[:1] + ven
        # ---- Tier 2
        o["t2_bn"] = split_cell(r.get("super_bn"))
        o["t2_en"] = split_cell(r.get("super_en"))
        if bool(o["t2_bn"]) != bool(o["t2_en"]):
            rep.missing("super_bn and super_en must both be filled or both be"
                        " empty (Tier 2 is parallel in the two languages)", c)
        for t in o["t1_en"] + o["t2_en"]:
            if words_of(t)[:1] and words_of(t)[0] in ARTICLES:
                rep.error(f"{c}: {t!r} starts with an article. Write the name"
                          f" without a, an or the")
        for col in ("canonical_en", "variants_en", "variants_bn", "super_bn",
                    "super_en"):
            for term, ch in wrong_separators(r.get(col)):
                rep.error(f"{c}: {col} has '{ch}' inside {term!r}. Several values"
                          f" in one cell are separated with | only")
        vac = [t for t in o["t2_bn"] + o["t2_en"] if t in VACUOUS]
        if vac:
            rep.error(f"{c}: {vac} is too vague for Tier 2 (it fits every object)")
        both = ({key_of(t) for t in o["t1_bn"] + o["t1_en"]}
                & {key_of(t) for t in o["t2_bn"] + o["t2_en"]})
        if both:
            rep.error(f"{c}: {sorted(both)} is written both as a name of this"
                      f" object (name_bn, a spelling row, variants_bn,"
                      f" canonical_en or variants_en) and as its generic name"
                      f" (super_bn, super_en). A word can be in one of the two"
                      f" places only")
        # ---- room, origin, en_route
        rs = (r.get("room_scope") or "").strip().lower()
        if not rs:
            rep.missing("room_scope is empty", c)
        elif rs not in ROOM_SCOPES:
            rep.error(f"{c}: room_scope {rs!r} is not one of {list(ROOM_SCOPES)}")
        o["room_scope"] = rs if rs in ROOM_SCOPES else ""
        hn = normalise(r.get("head_noun"))
        if not hn:
            rep.missing("head_noun is empty", c)
        elif script_of(hn) != "bn":
            rep.error(f"{c}: head_noun {hn!r} is not in Bangla letters")
        o3 = (r.get("origin3") or "").strip().lower()
        note = (r.get("origin_note") or "").strip()
        if not o3:
            rep.missing("origin3 is empty", c)
        elif o3 not in ORIGINS:
            rep.error(f"{c}: origin3 {o3!r} is not one of {list(ORIGINS)}")
        if not note:
            rep.missing("origin_note is empty (write the dictionary label as it"
                        " is printed, or 'rule: modern English')", c)
        elif note.lower().startswith("rule") and o3 in ORIGINS and o3 != "english":
            rep.error(f"{c}: origin_note says a rule was used, but the only rule"
                      f" is 'modern English', which gives origin3 = english"
                      f" (Design 4.3). Anything else not in the dictionary is"
                      f" uncertain, with a plain note")
        o["origin3"] = o3 if o3 in ORIGINS else ""
        o["origin_note"] = note
        er = (r.get("en_route") or "").strip().lower()
        if not er:
            rep.missing("en_route is empty", c)
        elif er not in ("yes", "no"):
            rep.error(f"{c}: en_route must be yes or no, found {er!r}")
        o["en_route"] = er if er in ("yes", "no") else ""
        o["en_route_note"] = (r.get("en_route_note") or "").strip()
        # ---- cluster, tag, frequency
        clusters = {srows[n]["cluster_id"] for n in groups[c] if srows[n]["cluster_id"]}
        if len(clusters) > 1:
            rep.error(f"{c}: its spellings sit in two clusters {sorted(clusters)}."
                      f" One object, one cluster")
        o["cluster_id"] = sorted(clusters)[0] if clusters else ""
        members = defaultdict(list)
        for n in groups[c]:
            if srows[n]["cluster_member"]:
                members[srows[n]["cluster_member"]].append(n)
        if len(members) > 1:
            rep.error(f"{c}: the rows {sorted(n for v in members.values() for n in v)}"
                      f" are {len(members)} different members in"
                      f" clusters_parsed.csv, so they cannot be one object (the"
                      f" clusters are fixed since G1). Empty same_object_as on the"
                      f" rows of the other member. A general member may be the"
                      f" generic name (super_bn, super_en) of the specific ones")
        own = srows[c]["tag_4a"]
        merged_tags = {srows[n]["tag_4a"] for n in spellings if srows[n]["tag_4a"]}
        if own:
            o["cultural_specificity"] = own
            if merged_tags - {own}:
                rep.warn(f"{c}: a merged spelling has another 4A tag"
                         f" {sorted(merged_tags - {own})}; the tag of {c} is used")
        elif len(merged_tags) == 1:
            o["cultural_specificity"] = next(iter(merged_tags))
        else:
            o["cultural_specificity"] = NOT_RATED
            if merged_tags:
                rep.warn(f"{c}: its spellings carry different 4A tags"
                         f" {sorted(merged_tags)} and {c} has none: {NOT_RATED}")
        o["freq_obj_name"] = sum(obj_counter.get(t, 0) for t in dict.fromkeys(o["t1_bn"]))
        if srows[c]["in_lexicon_txt"] == "no" and any(
                srows[n]["in_lexicon_txt"] == "yes" and key_of(n) != key_of(c)
                for n in spellings):
            rep.warn(f"{c}: is the canonical name but is not in the lexicon file,"
                     f" while its spelling "
                     f"{[n for n in spellings if srows[n]['in_lexicon_txt'] == 'yes']}"
                     f" is. The lexicon file is the source of truth for name_bn")
        objects[c] = o

    # a general word made a spelling of one specific object
    forced = set()
    for g in scope["key_groups"] + scope["member_groups"]:
        forced.update(g)
    name_words = {c: words_of(c) for c in objects}
    for c, o in objects.items():
        for sp in o["spellings"]:
            if key_of(sp) == key_of(c) or (sp in forced and c in forced) \
                    or txt["colloquial"].get(sp) == c:
                continue
            w = words_of(sp)
            longer = sorted(x for x, xw in name_words.items()
                            if len(xw) > len(w) and xw[-len(w):] == w)
            others = [x for x in longer if x != c]
            if others:
                rep.warn(f"{sp}: is set as a spelling of {c!r}, but it is a more"
                         f" general word: it is also the last word of {others}."
                         f" As it is now, the answer {sp!r} is right for {c!r}"
                         f" and wrong for those. A general word is usually an"
                         f" object of its own, and may be the generic name"
                         f" (super_bn) of the specific ones. Look again")
            elif longer:
                rep.warn(f"{sp}: is set as a spelling of {c!r} and is its last"
                         f" word. Keep this only if {sp!r} always means this"
                         f" object. If it can also be another kind of {sp!r}, it"
                         f" is an object of its own. Look again")
    # two names that the lexicon file keeps apart, inside one object
    member_of_key = {}
    for n, sr in srows.items():
        if sr["cluster_member"]:
            member_of_key[key_of(n)] = sr["cluster_member"]
    for c, o in objects.items():
        seen_key, seen_member, apart = set(), set(), []
        for t in o["t1_bn"]:
            k = key_of(t)
            if k not in txt["keys"] or t in txt["colloquial"] or k in seen_key:
                continue
            mem = member_of_key.get(k, "")
            if mem and mem in seen_member:
                continue
            seen_key.add(k)
            if mem:
                seen_member.add(mem)
            apart.append(t)
        if len(apart) > 1:
            rep.warn(f"{c}: the lexicon file lists {apart} as separate names, but"
                     f" here they are one object. In the file one object has one"
                     f" name. If they are not the same object, take the extra"
                     f" ones out of variants_bn or same_object_as")

    excluded_names = sorted(n for n in rows if not is_name[n])
    return objects, canon, excluded_names


def lexicon_rows(objects, en_unique=None):
    """The 16 columns of Design 4.2, one row per object."""
    by_cluster = defaultdict(list)
    for o in objects.values():
        if o["cluster_id"]:
            by_cluster[o["cluster_id"]].append(o)
    out = []
    for o in objects.values():
        sibs = [s for s in by_cluster.get(o["cluster_id"], []) if s is not o]
        sibs.sort(key=lambda s: (-s["freq_obj_name"], s["name"]))
        eu = ""
        if en_unique is not None and en_unique.get(o["name_id"]) is not None:
            eu = "yes" if en_unique[o["name_id"]] else "no"
        out.append({
            "name_id": o["name_id"], "name_bn": o["name"],
            "variants_bn": "|".join(o["t1_bn"][1:]),
            "canonical_en": o["canonical_en"],
            "variants_en": "|".join(o["variants_en"]),
            "super_bn": "|".join(o["t2_bn"]), "super_en": "|".join(o["t2_en"]),
            "cluster_id": o["cluster_id"],
            "siblings": "|".join(s["name"] for s in sibs),
            "room_scope": o["room_scope"],
            "cultural_specificity": o["cultural_specificity"],
            "origin3": o["origin3"], "origin_note": o["origin_note"],
            "en_route": o["en_route"], "en_unique": eu,
            "freq_obj_name": o["freq_obj_name"]})
    out.sort(key=lambda r: r["name_bn"])
    return out


# ----------------------------------------------------------------------------
# 7. Checks that need the scorer
# ----------------------------------------------------------------------------
_REFUSAL_WORDS = {t.text for p in S.REFUSAL_PHRASES for t in S.tokenise(p)}


def en_route_review(objects, rep):
    """Design 4.4: compare en_route with the script cross-check. It needs no
    loaded lexicon, so it runs even when the sheet still has errors."""
    review = []
    for o in objects.values():
        if not o["t1_en"]:
            continue
        flag, evidence = en_route_script(o["t1_bn"], o["t1_en"])
        people = o["en_route"]
        differs = people in ("yes", "no") and (people == "yes") != flag
        if differs and not o["en_route_note"]:
            rep.missing(
                "en_route differs from the script cross-check and en_route_note"
                " is empty. Look again, then write the reason (Design 4.4)",
                f"{o['name']} (people {people}, script {'yes' if flag else 'no'}"
                f"{': ' + evidence if evidence else ''})")
        review.append({"name_id": o["name_id"], "name_bn": o["name"],
                       "bangla_tier1": "|".join(o["t1_bn"]),
                       "english_tier1": "|".join(o["t1_en"]),
                       "en_route_people": people,
                       "en_route_script": "yes" if flag else "no",
                       "evidence": evidence,
                       "disagree": "yes" if differs else "no",
                       "en_route_note": o["en_route_note"]})
    return review


def check_with_scorer(objects, rep):
    """Load the draft lexicon with the scorer's own loader. Returns (lex,
    rows); lex is None when the loader refuses the rows."""
    rows = lexicon_rows(objects)
    # one Bangla name, one object: said here with the names of the objects
    owner = defaultdict(list)
    for o in objects.values():
        for t in dict.fromkeys(o["t1_bn"]):
            owner[key_of(t)].append((o["name"], t))
    for k, pairs in owner.items():
        objs = sorted({a for a, _ in pairs})
        if len(objs) > 1:
            rep.error(f"the Bangla name {pairs[0][1]!r} is written on {len(objs)}"
                      f" objects: {objs}. One Bangla name can belong to one"
                      f" object only. Keep it on one of them and delete it from"
                      f" the others (look in variants_bn and same_object_as)")
    try:
        lex = S.build_lexicon(rows, source="lexicon sheet")
    except S.LexiconError as e:
        for line in str(e).splitlines()[1:]:
            if "belongs to both" in line or "both Tier 1 and Tier 2" in line:
                continue                  # already said above, in plain words
            rep.error("scorer loader: " + line.strip(" -"))
        rep.warn("the lexicon sheet still has errors, so these checks did NOT"
                 " run yet: naming golds, shared English names, P-CLUST, forced"
                 " choice, question leaks. New lines can appear once the errors"
                 " are fixed")
        return None, rows
    for w in lex.warnings:
        if "no English name yet" in w:
            continue                      # already counted as canonical_en TODO
        if "is written in" in w:
            rep.error("scorer loader: " + w)
        else:
            rep.warn("scorer loader: " + w)
    eu = S.compute_en_unique(lex.objects)
    rows = lexicon_rows(objects, eu)

    # English names shared by several objects (Design 4.4, "no stretching")
    owners = defaultdict(set)
    for oid, obj in lex.objects.items():
        for term in obj.t1_en:
            for f in S.english_forms(term):
                owners[f].add(obj.name_bn)
    shared = {f: sorted(v) for f, v in owners.items() if len(v) > 1}
    for f in sorted(shared):
        rep.info(f"English name '{f}' is shared by {shared[f]}: en_unique = no"
                 f" for them (Design 4.4). Do not add a rare English word to"
                 f" avoid this")

    # a Tier-2 term that is another object's Tier-1 name (D91)
    t1_owner = defaultdict(list)
    for obj in lex.objects.values():
        for k in obj.keys_t1:
            t1_owner[k].append(obj)
    generic_of = set()
    for obj in lex.objects.values():
        for term in obj.t2_bn + obj.t2_en:
            for other in t1_owner.get(key_of(term), []):
                if other.name_id == obj.name_id:
                    continue
                generic_of.add((obj.name_id, other.name_id))
                same = obj.cluster_id and obj.cluster_id == other.cluster_id
                rep.info(f"{obj.name_bn}: its generic name '{term}' is a Tier-1"
                         f" name of {other.name_bn}"
                         f" ({'same cluster' if same else 'NOT the same cluster'})."
                         f" The answer '{term}' then counts as Tier 2 for"
                         f" {obj.name_bn} (D91)")
    for a, b in sorted(generic_of):
        if a < b and (b, a) in generic_of:
            rep.error(f"{lex.objects[a].name_bn} and {lex.objects[b].name_bn} list"
                      f" each other's name as a generic name. Only the more"
                      f" generic of two objects can be Tier 2 for the other (D91)")

    # terms that are also words of a refusal phrase (D106)
    for obj in lex.objects.values():
        for k in sorted((obj.keys_t1 | obj.keys_t2) & _REFUSAL_WORDS):
            rep.warn(f"{obj.name_bn}: its term '{k}' is also a word of a refusal"
                     f" phrase in scoring.py. A refusal with this word is then"
                     f" not counted as a refusal (D106). Tell the ASSET-9 chat")

    return lex, rows


def read_exclusions(path, items, rep):
    """The hand-written exclusion list: qa_id, reason, note."""
    out = {}
    if not Path(path).exists():
        return out
    for r in read_table(path, ("qa_id", "reason"), "the exclusion list"):
        qid, reason = r["qa_id"], r["reason"].strip().lower()
        if not qid:
            continue
        if qid not in items:
            rep.error(f"exclusion list: {qid} is in none of the three probe sets")
            continue
        if reason not in EXCLUSION_REASONS:
            rep.error(f"exclusion list: {qid}: reason must be one of"
                      f" {list(EXCLUSION_REASONS)}, found {r['reason']!r}")
            continue
        if reason == "other" and not r.get("note"):
            rep.error(f"exclusion list: {qid}: reason 'other' needs a note")
            continue
        out[qid] = {"reason": reason, "source": "manual", "note": r.get("note", "")}
    return out


def check_vocab(vsheet, rep):
    """The closed-answer vocabulary sheet -> (vocab, clean rows)."""
    rows = []
    for i, r in enumerate(vsheet, 2):
        kind, cid = r.get("kind", "").lower(), r.get("class_id", "")
        has_forms = bool(split_cell(r.get("forms_bn")) or split_cell(r.get("forms_en")))
        tag = f"closed vocab row {i} ({kind} {cid})"
        if not has_forms:
            rep.missing("closed vocab: rows without forms (fill them or delete"
                        " them)", f"{kind} {cid}")
            continue
        w, c = r.get("writer", "").strip(), r.get("checker", "").strip()
        if not w or not c:
            rep.missing("closed vocab: writer or checker is empty", f"{kind} {cid}")
        elif w.lower() == c.lower():
            rep.error(f"{tag}: writer and checker are the same person")
        if not split_cell(r.get("forms_bn")) or not split_cell(r.get("forms_en")):
            rep.missing("closed vocab: a class needs forms in Bangla letters and"
                        " in Latin letters, or answers in one of the two"
                        " languages cannot be scored", f"{kind} {cid}")
        for col in ("forms_bn", "forms_en"):
            for term, ch in wrong_separators(r.get(col)):
                rep.error(f"{tag}: {col} has '{ch}' inside {term!r}. Several"
                          f" forms in one cell are separated with | only")
        for t in split_cell(r.get("forms_bn")) + split_cell(r.get("forms_en")):
            if key_of(t) not in _REFUSAL_WORDS:
                continue
            if kind == "number":
                rep.error(f"{tag}: the form '{t}' is also a word of a refusal"
                          f" phrase in scoring.py, so a refusal would be read as"
                          f" this number. List the forms with their classifiers"
                          f" instead (the word with টি, with টা, and so on)")
            else:
                rep.warn(f"{tag}: the form '{t}' is also a word of a refusal"
                         f" phrase in scoring.py. A refusal with this word is"
                         f" then not counted as a refusal. Tell the ASSET-9 chat")
        rows.append({k: r.get(k, "") for k in S.VOCAB_COLUMNS})
    order = {k: i for i, k in enumerate(S.VOCAB_KINDS)}
    rows.sort(key=lambda r: (order.get(r["kind"].lower(), 9),
                             int(r["class_id"]) if r["class_id"].isdigit() else 0,
                             r["class_id"]))
    try:
        voc = S.build_closed_vocab(rows, source="closed vocab sheet")
    except S.LexiconError as e:
        for line in str(e).splitlines()[1:]:
            rep.error("scorer loader: " + line.strip(" -"))
        return None, rows
    for w in voc.warnings:
        if "expects 4 room classes" in w or "is written in" in w \
                or "both a material and a colour" in w:
            rep.error("scorer loader: " + w)
        else:
            rep.warn("scorer loader: " + w)
    for kind in ("material", "colour"):
        if not voc.classes[kind]:
            rep.missing("closed vocab: a kind has no class yet", kind)
    return voc, rows


def audit_golds(items, lex, voc, excluded, rep, label="sheets"):
    """Resolve every gold with scoring.resolve_gold, the function the scorer
    uses. Naming golds need the lexicon; the others need the vocabulary.
    Returns the coverage rows (one per distinct gold string and kind)."""
    cov = OrderedDict()
    n_bad = Counter()
    problems = {}
    empty_voc = S.build_closed_vocab([])
    lx = lex if lex is not None else S.build_lexicon([])
    for qid, rec in items.items():
        it = rec["item"]
        naming = it["qa_type"] in S.NAMING_TYPES
        if naming and lex is None:
            continue
        use_voc = voc if voc is not None else empty_voc
        if not naming and voc is None:
            continue
        g = S.resolve_gold(it, lx, use_voc)
        problem = ""
        if g.status != "ok":
            problem = g.status + (f" ({g.hint})" if g.hint else "")
        elif it["qa_type"] == "count":
            if "gold_count" in it and g.number != int(it["gold_count"]):
                problem = f"number {g.number} but 5C read {it['gold_count']}"
        elif it["qa_type"] == "scene_id":
            if it.get("gold_room"):
                want = S.resolve_gold({"qa_id": qid, "qa_type": "scene_id",
                                       "answer_bn": it["gold_room"]}, lx, use_voc)
                if want.status != "ok" or want.classes != g.classes:
                    problem = (f"room class {sorted(g.classes)} but 5C room"
                               f" {it['gold_room']} is {sorted(want.classes)}")
        elif it["qa_type"] == "obj_property":
            if it.get("closed_kind") and g.kind != it["closed_kind"]:
                problem = (f"the vocabulary reads it as {g.kind}, but the question"
                           f" asks the {it['closed_kind']} (5C closed_kind)")
        kind = "name" if naming else it.get("closed_kind", it["qa_type"])
        key = (kind, gold_string(it))
        c = cov.setdefault(key, {
            "kind_5c": kind, "gold": key[1], "n_pname": 0, "n_pcore_closed": 0,
            "n_rtest": 0, "status": g.status, "resolves_to": "",
            "extra_words": " ".join(g.residual), "problem": "",
            "n_excluded": 0, "example_qa_id": qid})
        for p in rec["probes"]:
            c[COUNT_COL[p]] += 1
        if g.status == "ok":
            if naming:
                c["resolves_to"] = lex.objects[g.name_id].name_bn
            elif g.number is not None:
                c["resolves_to"] = str(g.number)
            else:
                c["resolves_to"] = "|".join(sorted(g.classes))
        if problem:
            problems[qid] = problem
        if qid in excluded:
            c["n_excluded"] += 1
        elif problem:
            if not c["problem"]:
                c["problem"] = problem
                c["example_qa_id"] = qid
            n_bad[kind] += 1
    for (kind, gold), c in cov.items():
        if c["problem"]:
            rep.missing(f"{kind} golds that do not resolve or disagree with 5C."
                        f" Give each a row or a form, or put its items on the"
                        f" exclusion list",
                        f"{gold!r} (e.g. {c['example_qa_id']}): {c['problem']}")
    total = len(items)
    n_ex = sum(1 for q in items if q in excluded)
    if lex is not None and voc is not None:
        if not n_bad:
            say(f"[audit] PASS on the {label}: every gold of the {total} items"
                f" resolves and agrees with the 5C fields, except the {n_ex}"
                f" excluded items")
        else:
            say(f"[audit] NOT PASSED on the {label}: {sum(n_bad.values())} items"
                f" do not resolve or disagree with 5C: {dict(n_bad)}")
    return list(cov.values()), n_bad, problems


def check_probes_against_lexicon(probes, pclust, lex, excluded, scope, rep):
    """P-CLUST items must sit in their cluster; list what the merges changed."""
    if lex is None:
        return
    outside = Counter()
    empty = S.build_closed_vocab([])
    for it in pclust:
        if it["qa_id"] in excluded:
            continue
        g = S.resolve_gold(it, lex, empty)
        if g.status != "ok":
            continue                          # reported by the audit
        got = lex.objects[g.name_id].cluster_id
        if got != it.get("cluster_id", got):
            rep.error(f"P-CLUST item {it['qa_id']}: its gold is in cluster"
                      f" {it['cluster_id']!r} in the probe file but in {got!r} in"
                      f" the lexicon")
    pc_ids = {it["qa_id"] for it in pclust}
    for it in probes["P-NAME"]:
        if it["qa_id"] in pc_ids or it["qa_id"] in excluded:
            continue
        g = S.resolve_gold(it, lex, empty)
        if g.status == "ok" and lex.objects[g.name_id].cluster_id in scope["qualifying"]:
            outside[(gold_string(it), lex.objects[g.name_id].name_bn)] += 1
    for (gold, obj), n in sorted(outside.items()):
        rep.warn(f"{n} P-NAME items with the gold {gold!r} now belong to the"
                 f" qualifying-cluster object {obj!r} but are NOT in P-CLUST"
                 f" (5B drew it by exact name). They get the sibling rule but no"
                 f" forced choice. Report this line to the team lead")


def forced_choice_feasible(lex, pclust, excluded, rep):
    """ASSET-11 needs 3 distractors for every P-CLUST gold (Design 4.7): up
    to 2 cluster siblings, the rest from the same room_scope outside the
    cluster, never a name of the gold's Tier 2. Warn before the freeze."""
    if lex is None:
        return
    golds = set()
    empty = S.build_closed_vocab([])
    for it in pclust:
        if it["qa_id"] in excluded:
            continue
        g = S.resolve_gold(it, lex, empty)
        if g.status == "ok":
            golds.add(g.name_id)
    for gid in sorted(golds):
        gold = lex.objects[gid]
        room = gold.row.get("room_scope", "")
        if not room:
            continue

        def allowed(o):
            names = {key_of(o.name_bn)} | {key_of(t) for t in o.t1_en[:1]}
            return not (names & gold.keys_t2)

        sibs = [o for o in lex.objects.values() if o.name_id != gid
                and o.cluster_id == gold.cluster_id and allowed(o)]
        outs = [o for o in lex.objects.values() if o.cluster_id != gold.cluster_id
                and o.row.get("room_scope", "") == room and allowed(o)]
        if min(len(sibs), 2) + len(outs) < 3 or not outs:
            rep.warn(f"{gold.name_bn}: forced choice needs 3 distractors (Design"
                     f" 4.7) but only {len(sibs)} cluster siblings and {len(outs)}"
                     f" objects with room_scope {room!r} outside the cluster are"
                     f" usable. ASSET-11 could not build its option set")


def question_leaks(items, lex, excluded):
    """Naming questions that contain a Bangla Tier-1 name of their own gold.
    Report only: the decision belongs to the ASSET-2 chat."""
    out = []
    if lex is None:
        return out
    empty = S.build_closed_vocab([])
    for qid, rec in items.items():
        it = rec["item"]
        if it["qa_type"] not in S.NAMING_TYPES or qid in excluded:
            continue
        g = S.resolve_gold(it, lex, empty)
        if g.status != "ok":
            continue
        toks = S.tokenise(normalise(it.get("question_bn", "")))
        hits = [m.surface for m in lex.index.find_all(toks)
                if any(s.name_id == g.name_id and s.tier == 1 for s in m.senses)]
        if hits:
            out.append({"qa_id": qid, "probes": "|".join(rec["probes"]),
                        "qa_type": it["qa_type"], "gold": gold_string(it),
                        "found_in_question": "|".join(hits)})
    return out


# ----------------------------------------------------------------------------
# 8. check and freeze
# ----------------------------------------------------------------------------
def run_check(args, final):
    P = resolve_paths(args)
    work = P["work"]
    rep = Report()
    say(f"ASSET-4 v2 {'freeze' if final else 'check'} ({SCRIPT_VERSION})."
        f" No model output is read.")
    probes, pclust = load_probes(P["probes"])
    tags = load_tags(P["tags"])
    members, form_to, counts = load_clusters(P["clusters"], P["cluster_counts"])
    txt = parse_lexicon_txt(P["lexicon_txt"])
    corpus = None
    if final or args.data:
        corpus = load_corpus(P["data"], P["counts"], validate=not args.no_validate)
        obj_counter, objt_counter = corpus_counters(corpus)
        old_obj, old_objt = (load_state_counters(work)
                             if (work / STATE_FILE).exists() else (None, None))
        if old_obj is not None and (old_obj != obj_counter or old_objt != objt_counter):
            rep.error("the naming answers of the dataset changed since the last"
                      " build: run build --update, then check again")
    else:
        obj_counter, objt_counter = load_state_counters(work)
        say("[data] counts taken from the last build (pass --data to re-read the"
            " dataset)")
    self_checks(probes, pclust, form_to, tags, corpus, counts, None, obj_counter,
                compare_counts=False)
    flags = read_5c_flags(P["review_5c"])
    scope = compute_scope(probes, pclust, members, form_to, tags, txt,
                          obj_counter, objt_counter, flags, corpus)
    attested = {key_of(n) for n in set(obj_counter) | set(objt_counter)} | txt["keys"]

    sheet = read_table(work / SHEET_FILE, SHEET_COLS, "the lexicon sheet")
    say(f"[sheet] {len(sheet)} rows read from {work / SHEET_FILE}")
    vsheet = read_table(work / VOCAB_FILE, VOCAB_SHEET_COLS, "the vocabulary sheet")
    say(f"[sheet] {len(vsheet)} rows read from {work / VOCAB_FILE}")

    items = unique_items(probes)
    objects, canon, excluded_names = assemble(sheet, scope, tags, txt, obj_counter,
                                              attested, rep, final)
    excluded = read_exclusions(work / EXCLUSION_FILE, items, rep)
    for qid, rec in items.items():
        it = rec["item"]
        if it["qa_type"] in S.NAMING_TYPES and gold_string(it) in excluded_names \
                and qid not in excluded:
            excluded[qid] = {"reason": "gold_not_one_object", "source": "is_name = no",
                             "note": ""}
    if getattr(args, "no_exclusions", False) and excluded:
        rep.error(f"--no-exclusions is set, but {len(excluded)} items are excluded"
                  f" (the exclusion list or is_name = no). Correct their golds in"
                  f" the dataset instead (DESIGN_ADDENDUM A3), then build --update")

    lex, lrows = check_with_scorer(objects, rep)
    review = en_route_review(objects, rep)
    voc, vrows = check_vocab(vsheet, rep)
    coverage, n_bad, problems = audit_golds(items, lex, voc, excluded, rep)
    check_probes_against_lexicon(probes, pclust, lex, excluded, scope, rep)
    forced_choice_feasible(lex, pclust, excluded, rep)
    leaks = question_leaks(items, lex, excluded)
    if leaks:
        rep.info(f"{len(leaks)} naming questions contain a Bangla name of their"
                 f" own gold (see bn_question_leaks.csv). The decision belongs to"
                 f" the ASSET-2 chat")
    for qid, e in excluded.items():
        it = items[qid]["item"]
        why = problems.get(qid)
        rep.info(f"excluded: {qid} ({'|'.join(items[qid]['probes'])},"
                 f" {it['qa_type']}): {e['reason']} [{e['source']}]; "
                 + (f"the audit says: {why}" if why else
                    "its gold could not be checked yet" if (
                        lex is None if it["qa_type"] in S.NAMING_TYPES
                        else voc is None) else
                    "the audit finds no fault with its gold, so the reason must"
                    " be in the note"))
    tag_diff = 0
    if lex is not None:
        empty = S.build_closed_vocab([])
        for it in probes["P-NAME"]:
            if it["qa_id"] in excluded or "cultural_specificity" not in it:
                continue
            g = S.resolve_gold(it, lex, empty)
            if g.status == "ok" and (it["cultural_specificity"] or NOT_RATED) != \
                    lex.objects[g.name_id].row["cultural_specificity"]:
                tag_diff += 1
        if tag_diff:
            rep.info(f"{tag_diff} P-NAME items carry another cultural tag in the"
                     f" probe file than their object has in the lexicon (a merged"
                     f" spelling takes the tag of its object). Statistics use the"
                     f" lexicon column")

    rep.close(final)

    # ---- reports
    rdir = work / "reports"
    rdir.mkdir(parents=True, exist_ok=True)
    write_table(rdir / "lexicon_preview.csv", lrows, list(S.LEXICON_COLUMNS))
    write_table(rdir / "gold_coverage.csv", coverage,
                ["kind_5c", "gold", "status", "resolves_to", "extra_words",
                 "problem", "n_pname", "n_pcore_closed", "n_rtest", "n_excluded",
                 "example_qa_id"])
    write_table(rdir / "en_route_review.csv", review,
                ["name_id", "name_bn", "bangla_tier1", "english_tier1",
                 "en_route_people", "en_route_script", "evidence", "disagree",
                 "en_route_note"])
    write_table(rdir / "bn_question_leaks.csv", leaks,
                ["qa_id", "probes", "qa_type", "gold", "found_in_question"])
    fixes = []
    for qid, e in sorted(excluded.items()):
        it = items[qid]["item"]
        fixes.append({"qa_id": qid, "probes": "|".join(items[qid]["probes"]),
                      "qa_type": it["qa_type"], "question_bn": it.get("question_bn", ""),
                      "answer_bn": it["answer_bn"], "reason": e["reason"],
                      "source": e["source"], "note": e["note"]})
    write_table(rdir / "gold_fix_candidates.csv", fixes,
                ["qa_id", "probes", "qa_type", "question_bn", "answer_bn", "reason",
                 "source", "note"])
    name = "freeze_report.txt" if final else "check_report.txt"
    head = [f"ASSET-4 v2 {'freeze' if final else 'check'}, {utc_now()},"
            f" {SCRIPT_VERSION}",
            f"objects {len(objects)}, sheet rows {len(sheet)}, excluded items"
            f" {len(excluded)}",
            f"ERROR {rep.n('ERROR')}, TODO {rep.n('TODO')} kinds ({rep.n_open}"
            f" open cells), WARNING {rep.n('WARNING')}, INFO {rep.n('INFO')}", ""]
    (rdir / name).write_text("\n".join(head + rep.text()) + "\n", encoding="utf-8")

    say()
    for line in rep.text(limit=40):
        say(line)
    say()
    n_rows_done = sum(1 for o in objects.values()
                      if o["canonical_en"] and o["room_scope"] and o["origin3"]
                      and o["en_route"])
    dis = sum(1 for r in review if r["disagree"] == "yes")
    say("=" * 72)
    say(f"Objects: {len(objects)} ({n_rows_done} with English name, room, origin"
        f" and en_route filled). Spellings merged into them:"
        f" {sum(len(o['spellings']) for o in objects.values())}."
        f" Rows marked is_name = no: {len(excluded_names)}.")
    say(f"Excluded items: {len(excluded)}"
        + (" (" + ", ".join(f"{v} {k}" for k, v in Counter(
            e['reason'] for e in excluded.values()).items()) + ")" if excluded else "")
        + f". en_route: script and people disagree on {dis} objects.")
    say(f"ERROR {rep.n('ERROR')}   OPEN CELLS {rep.n_open}   WARNING"
        f" {rep.n('WARNING')}   INFO {rep.n('INFO')}")
    say(f"Reports written to {rdir}: {name}, lexicon_preview.csv,"
        f" gold_coverage.csv, en_route_review.csv, bn_question_leaks.csv,"
        f" gold_fix_candidates.csv")
    say("=" * 72)
    return {"rep": rep, "objects": objects, "lrows": lrows, "vrows": vrows,
            "excluded": excluded, "items": items, "probes": probes,
            "pclust": pclust, "lex": lex, "voc": voc, "work": work, "scope": scope}


def cmd_check(args):
    r = run_check(args, final=False)
    rep = r["rep"]
    if rep.n("ERROR") or rep.n_open:
        say(f"NOT READY: {rep.n('ERROR')} errors and {rep.n_open} open cells."
            f" Nothing final was written.")
    else:
        say("READY: no error and nothing open. Read the WARNING and INFO lines"
            " once more, then run freeze.")


def _guard_frozen(path, text, force):
    if path.exists() and path.read_text(encoding="utf-8") != text and not force:
        raise SystemExit(
            f"[guard] STOP: {path} exists and would change. After Gate A nothing"
            f" in the lexicon changes (Design 4.2). If Gate A is not tagged yet,"
            f" rerun with --force.")


def _csv_text(rows, cols):
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore",
                       lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in cols})
    return buf.getvalue()


def cmd_freeze(args):
    r = run_check(args, final=True)
    rep = r["rep"]
    if rep.n("ERROR"):
        raise SystemExit(f"[freeze] {rep.n('ERROR')} errors. Nothing was written."
                         f" Fix them and run freeze again.")
    repo = Path(args.repo) if args.repo else _HERE.parent
    lex_dir = Path(args.out_lexicon) if args.out_lexicon else repo / "lexicon"
    probe_dir = Path(args.out_probes) if args.out_probes else repo / "probes"
    lex_path, voc_path = lex_dir / "lexicon.csv", lex_dir / "closed_vocab.csv"
    exc_path = probe_dir / "excluded_items.csv"

    frozen_excl = []
    for qid, e in sorted(r["excluded"].items()):
        frozen_excl.append({"qa_id": qid, "probes": "|".join(r["items"][qid]["probes"]),
                            "qa_type": r["items"][qid]["item"]["qa_type"],
                            "reason": e["reason"], "source": e["source"],
                            "added_at": "gate-A"})
    texts = {lex_path: _csv_text(r["lrows"], list(S.LEXICON_COLUMNS)),
             voc_path: _csv_text(r["vrows"], list(S.VOCAB_COLUMNS)),
             exc_path: _csv_text(frozen_excl, FROZEN_EXCLUSION_COLS)}
    for p, t in texts.items():
        _guard_frozen(p, t, args.force)

    # 1. write the three files next to the sheets and prove them there: the
    #    scorer loads them and the gold audit passes on them
    keep = r["work"] / "frozen"
    keep.mkdir(parents=True, exist_ok=True)
    for p, t in texts.items():
        (keep / p.name).write_text(t, encoding="utf-8")
    lex = S.load_lexicon(keep / lex_path.name)
    voc = S.load_closed_vocab(keep / voc_path.name)
    if lex.warnings or voc.warnings:
        raise SystemExit(f"[freeze] ERROR: the scorer warns about the written"
                         f" files: {lex.warnings + voc.warnings}. Stop and ask.")
    say(f"[scorer] lexicon.csv loads: {lex.describe()}")
    say(f"[scorer] closed_vocab.csv loads: {voc.describe()}")
    final_rep = Report()
    _, n_bad, _ = audit_golds(r["items"], lex, voc, r["excluded"], final_rep,
                              label="written files")
    if n_bad:
        raise SystemExit("[freeze] ERROR: the audit on the written files failed."
                         " Nothing was copied to the repo. Stop and ask.")
    say(f"[write] {keep} (lexicon.csv, closed_vocab.csv, excluded_items.csv)")
    # 2. only now copy them to the repo
    for p in texts:
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(keep / p.name, p)
        say(f"[write] {p}")

    # planning numbers for the next chats (computed after the files are written)
    empty = S.build_closed_vocab([])
    pc_ids = {it["qa_id"] for it in r["pclust"]}
    cells, fce = Counter(), 0
    for it in r["probes"]["P-NAME"]:
        if it["qa_id"] in r["excluded"]:
            continue
        o = lex.objects[S.resolve_gold(it, lex, empty).name_id]
        cells[("in P-CLUST" if it["qa_id"] in pc_ids else "not in P-CLUST",
               "en_route " + o.row["en_route"])] += 1
        if it["qa_id"] in pc_ids and o.row["en_unique"] == "yes":
            fce += 1
    objs = list(lex.objects.values())
    say()
    say("=" * 72)
    say(f"FROZEN. {len(objs)} objects;"
        f" en_route yes {sum(1 for o in objs if o.row['en_route'] == 'yes')},"
        f" en_unique yes {sum(1 for o in objs if o.row['en_unique'] == 'yes')};"
        f" origin3 " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(
            o.row['origin3'] for o in objs).items())) + ";"
        f" cultural " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(
            o.row['cultural_specificity'] for o in objs).items())) + ".")
    say(f"Excluded items: {len(frozen_excl)} (run, never scored; D110). Later"
        f" chats may add rows to probes/excluded_items.csv until Gate E.")
    say("For ASSET-12 (NAME-REF needs 50 per cell): "
        + "; ".join(f"{a}, {b}: {n}" for (a, b), n in sorted(cells.items())))
    say(f"For ASSET-11: P-CLUST items whose gold is en_unique = yes: {fce}"
        f" (upper bound of P-FCE; risk R10 starts below 100).")
    say("Gate A: commit the three files and tag the commit, from the repo:")
    say("  git add lexicon/lexicon.csv lexicon/closed_vocab.csv"
        " probes/excluded_items.csv scripts/asset4_lexicon.py")
    say('  git commit -m "Gate A: lexicon and closed-answer vocabulary frozen"')
    say("  git tag gate-A && git push && git push origin gate-A")
    say("=" * 72)


# ----------------------------------------------------------------------------
# 9. Self-test on made-up data (names printed in the design only)
# ----------------------------------------------------------------------------
def _syn_item(n, house, qa_type, q, a, cap="closeup", **extra):
    img = f"{house}_{n:04d}"
    slot = {"obj_name": 1, "object": 2, "obj_property": 3, "count": 4,
            "scene_id": 1}[qa_type]
    it = {"qa_id": f"{img}_q{slot}", "image_id": img, "house_id": house,
          "capture_type": cap, "qa_type": qa_type, "question_bn": q,
          "answer_bn": a, "image_file": f"{house}/{img}.jpg"}
    it.update(extra)
    return it


def _write_json(path, probe, items):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"probe": probe, "n": len(items), "items": items}, f,
                  ensure_ascii=False, indent=1)


def _selftest_files(tmp):
    """A tiny made-up project. Every object name here is printed in the design
    file, and every question is made up: nothing is dataset text."""
    d = Path(tmp)
    for sub in ("data", "probes", "results/assets", "human/HT-1", "repo"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    pots = "রান্নার পাত্র"
    tables = "টেবিলের ধরন"
    q = "SYN প্রশ্ন?"                      # made up; never a dataset question
    names = [  # gold, cluster of the P-NAME item
        ("ফ্রিজ।", ""), ("সোফা", ""), ("টেলিভিশন", ""), ("টিভি", ""),
        ("এসি", ""), ("বালতি", ""), ("সুইচ বোর্ড", ""), ("সুইচবোর্ড", ""),
        ("হাঁড়ি", pots), ("পাতিল", pots), ("টেবিল", tables),
        ("পড়ার টেবিল", tables), ("আলমারি", ""), ("ওয়ারড্রব", ""),
        ("চেয়ার ও টেবিল", ""), ("দরজা", "")]
    pname, n = [], 0
    for gold, cl in names:
        for rep_i in range(2):
            n += 1
            g = normalise(gold)
            pname.append(_syn_item(
                n, "H01" if rep_i == 0 else "H02", "obj_name", q, gold,
                name_bn=g, name_id=a4.name_id(g), cultural_specificity="",
                cluster_id=cl, cluster_member=g if cl else "",
                in_qualifying_cluster=bool(cl)))
    pclust = [it for it in pname if it["cluster_id"]]
    closed = [dict(it, split="train", closed_kind="name") for it in pname[::2]]
    for it in closed:
        for k in ("name_bn", "name_id", "cultural_specificity", "cluster_id",
                  "cluster_member", "in_qualifying_cluster"):
            it.pop(k, None)
    other = [
        ("count", "SYN গণনা প্রশ্ন?", "একটি।", {"closed_kind": "count", "gold_count": 1}),
        ("count", "SYN গণনা প্রশ্ন?", "দুইটি", {"closed_kind": "count", "gold_count": 2}),
        ("count", "SYN গণনা প্রশ্ন?", "দুটি", {"closed_kind": "count", "gold_count": 2}),
        ("scene_id", "SYN ঘর প্রশ্ন?", "শোবার ঘর।", {"closed_kind": "room", "gold_room": "শোবার ঘর"}),
        ("scene_id", "SYN ঘর প্রশ্ন?", "বাথরুমে", {"closed_kind": "room", "gold_room": "বাথরুম"}),
        ("scene_id", "SYN ঘর প্রশ্ন?", "রান্নাঘর", {"closed_kind": "room", "gold_room": "রান্নাঘর"}),
        ("scene_id", "SYN ঘর প্রশ্ন?", "বসার ঘর", {"closed_kind": "room", "gold_room": "বসার ঘর"}),
        ("obj_property", "SYN উপাদান প্রশ্ন?", "কাঠের", {"closed_kind": "material"}),
        ("obj_property", "SYN উপাদান প্রশ্ন?", "কাঁচ", {"closed_kind": "material"}),
        ("obj_property", "SYN রং প্রশ্ন?", "সবুজ-হলুদ", {"closed_kind": "colour"}),
        ("obj_property", "SYN রং প্রশ্ন?", "কাঁচ", {"closed_kind": "colour"}),
    ]
    for t, qq, a, extra in other:
        n += 1
        cap = "closeup" if t == "obj_property" else "group"
        closed.append(_syn_item(n, "H03", t, qq, a, cap=cap, split="test", **extra))
    n += 1
    extra_name = _syn_item(n, "H03", "object", "SYN বস্তু প্রশ্ন?", "চেয়ার",
                           cap="group", split="test", closed_kind="name",
                           in_pcore_closed=False)
    n += 1
    leak = _syn_item(n, "H03", "object", "SYN দরজা প্রশ্ন?", "দরজা", cap="group",
                     split="test", closed_kind="name", in_pcore_closed=False)
    rtest = [dict(it, in_pcore_closed=True) for it in closed if it["house_id"] == "H03"]
    rtest += [extra_name, leak]
    _write_json(d / "probes/P-NAME.json", "P-NAME", pname)
    _write_json(d / "probes/P-CLUST.json", "P-CLUST", pclust)
    _write_json(d / "probes/P-CORE-CLOSED.json", "P-CORE-CLOSED", closed)
    _write_json(d / "probes/R-TEST.json", "R-TEST", rtest)

    corpus = {it["qa_id"]: it for it in pname + closed + rtest}
    n += 1
    corpus["x"] = _syn_item(n, "H04", "obj_name", q, "কমোড")     # attested, no probe
    a5.write_synthetic_json(list(corpus.values()), str(d / "data/merged.json"))

    write_table(d / "results/assets/clusters_parsed.csv", [
        {"cluster_id": pots, "member": "হাঁড়ি", "surface_forms": "হাঁড়ি"},
        {"cluster_id": pots, "member": "পাতিল", "surface_forms": "পাতিল"},
        {"cluster_id": pots, "member": "দেকচি", "surface_forms": "দেকচি"},
        {"cluster_id": tables, "member": "টেবিল", "surface_forms": "টেবিল"},
        {"cluster_id": tables, "member": "পড়ার টেবিল", "surface_forms": "পড়ার টেবিল"},
        {"cluster_id": "বসার আসন", "member": "টুল/মোড়া", "surface_forms": "টুল | মোড়া"},
        {"cluster_id": "বসার আসন", "member": "চেয়ার", "surface_forms": "চেয়ার"},
    ], ["cluster_id", "member", "surface_forms", "description"])
    write_table(d / "results/assets/cluster_counts.csv", [
        {"cluster_id": pots, "member": "হাঁড়ি", "obj_name_count": 2},
        {"cluster_id": pots, "member": "পাতিল", "obj_name_count": 2},
    ], ["cluster_id", "member", "obj_name_count"])
    write_table(d / "results/assets/cultural_specificity.csv", [
        {"name_id": a4.name_id(normalise(x)), "name_bn": x, "n_obj_name": 2,
         "cultural_specificity": t}
        for x, t in (("হাঁড়ি", "ambiguous"), ("ফ্রিজ", "globally_common"),
                     ("টিভি", "globally_common"), ("দেকচি", "culturally_specific"))
    ], ["name_id", "name_bn", "n_obj_name", "cultural_specificity"])
    (d / "data/lexicon.txt").write_text(
        "================================================================\n"
        "LEXICON (made up)\n"
        "================================================================\n"
        "টেলিভিশন  (কথ্য: টিভি)\n"
        "================================================================\n"
        "\n"
        "===================== রান্নাঘর =====================\n"
        "◆ হাঁড়ি\n◆ পাতিল\n◆ দেকচি\n  ফ্রিজ\n\n"
        "===================== বসার ঘর =====================\n"
        "  সোফা\n  টেলিভিশন\n  চেয়ার\n◆ টুল\n◆ মোড়া\n\n"
        "===================== শোবার ঘর =====================\n"
        "  আলমারি\n  ওয়ারড্রব\n◆ টেবিল\n◆ পড়ার টেবিল\n\n"
        "================= সাধারণ / সব ঘর =================\n"
        "--- গঠন ---\n  দরজা\n  এসি\n  সুইচ বোর্ড\n  বালতি\n", encoding="utf-8")
    return d


def _selftest_args(d, **kw):
    base = dict(drive=str(d), data=str(d / "data/merged.json"), counts=None,
                probes=None, tags=None, clusters=None, cluster_counts=None,
                unmatched=None, lexicon_txt=str(d / "data/lexicon.txt"),
                review_5c=None, work=None, update=False, no_validate=True,
                no_exclusions=False, repo=str(d / "repo"), out_lexicon=None,
                out_probes=None, force=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _selftest_fill(d):
    """Fill the sheets the way a careful pair would, with made-up content."""
    work = d / "human/HT-1"
    rows = read_table(work / SHEET_FILE)
    by = {r["name_bn"]: r for r in rows}
    N = normalise
    fill = {
        "ফ্রিজ": ("fridge", "refrigerator", "", "", "english", "ইং", "yes"),
        "সোফা": ("sofa", "", "", "", "english", "ইং", "yes"),
        "টেলিভিশন": ("television", "tv", "", "", "english", "ইং", "yes"),
        "এসি": ("ac", "air conditioner", "", "", "english", "rule: modern English", "yes"),
        "বালতি": ("bucket", "", "পাত্র", "container", "other_loan", "পর্তু", "no"),
        "সুইচ বোর্ড": ("switch board", "", "", "", "english", "ইং", "yes"),
        "হাঁড়ি": ("pot", "", "পাত্র", "container", "inherited", "তদ্ভব", "no"),
        "পাতিল": ("pot", "", "পাত্র", "container", "inherited", "দেশি", "no"),
        "দেকচি": ("pot", "", "পাত্র", "container", "other_loan", "ফা", "no"),
        "টেবিল": ("table", "", "", "", "english", "ইং", "yes"),
        "পড়ার টেবিল": ("study table", "desk", "টেবিল", "table", "english", "ইং", "no"),
        "আলমারি": ("wardrobe", "almirah", "", "", "other_loan", "পর্তু", "no"),
        "ওয়ারড্রব": ("wardrobe", "", "", "", "english", "ইং", "yes"),
        "দরজা": ("door", "", "", "", "other_loan", "ফা", "no"),
        "চেয়ার": ("chair", "", "", "", "english", "ইং", "yes"),
        "টুল": ("stool", "", "", "", "english", "ইং", "yes"),
    }
    for name, (cen, ven, sbn, sen, o3, note, er) in fill.items():
        r = by[N(name)]
        r.update(canonical_en=cen, variants_en=ven, super_bn=sbn, super_en=sen,
                 origin3=o3, origin_note=note, en_route=er)
        if not r["room_scope"]:
            r["room_scope"] = "any"
    by[N("চেয়ার ও টেবিল")]["is_name"] = "no"
    by[N("টুল")]["variants_bn"] = ""
    for r in rows:
        r["writer"], r["checker"] = "P1", "P2"
    # en_route notes where the script and the people will disagree
    for name in ("ফ্রিজ", "টেবিল", "চেয়ার", "টুল"):
        by[N(name)]["en_route_note"] = "the English spelling is far from its sound"
    write_table(work / SHEET_FILE, rows, SHEET_COLS)

    v = read_table(work / VOCAB_FILE)
    more_bn = {"kitchen": "কিচেন", "bedroom": "বেডরুম", "living room": "ড্রয়িং রুম"}
    keep = []
    for r in v:
        if r["kind"] == "room":
            r["forms_en"] = r["class_id"]
            if r["class_id"] in more_bn:
                r["forms_bn"] += "|" + more_bn[r["class_id"]]
        elif r["class_id"] == "1":
            r["forms_bn"] += "|এক|একটা"
            r["forms_en"] = "one"
        elif r["class_id"] == "2":
            r["forms_bn"] += "|দুই|দুটো"
            r["forms_en"] = "two"
        else:
            continue
        keep.append(r)
    keep += [
        {"kind": "material", "class_id": "wood", "forms_bn": "কাঠ|পারটেক্স", "forms_en": "wood"},
        {"kind": "material", "class_id": "glass", "forms_bn": "কাচ|কাঁচ|গ্লাস", "forms_en": "glass"},
        {"kind": "colour", "class_id": "green", "forms_bn": "সবুজ", "forms_en": "green"},
        {"kind": "colour", "class_id": "yellow", "forms_bn": "হলুদ", "forms_en": "yellow"},
    ]
    for r in keep:
        r["writer"], r["checker"] = "P1", "P2"
    write_table(work / VOCAB_FILE, keep, VOCAB_SHEET_COLS)
    return rows


def cmd_selftest(args):
    say("#" * 72)
    say("SELF-TEST on made-up data. No number printed here is a project number.")
    say("#" * 72)
    N = normalise
    # ---- helpers
    assert acronym_readings("".join(words_of("এসি"))) == {"ac"}
    assert acronym_readings("".join(words_of("টিভি"))) == {"tv"}
    assert "ips" in acronym_readings("".join(words_of("আইপিএস")))
    assert acronym_readings("".join(words_of("বালতি"))) == set()
    assert en_route_script([N("সোফা")], ["sofa"])[0]
    assert en_route_script([N("টিভি")], ["television", "tv"])[0]
    assert en_route_script([N("এসি")], ["ac"])[0]
    assert not en_route_script([N("বালতি")], ["bucket"])[0]
    assert not en_route_script([N("ফ্রিজ")], ["fridge"])[0]   # known miss: 0.5
    say("[selftest] OK  en_route cross-check: similarity, acronyms, known miss")

    tmp = Path(tempfile.mkdtemp(prefix="asset4_selftest_"))
    import contextlib
    import io

    def quiet(fn, a):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                out = fn(a)
            except SystemExit as e:
                return buf.getvalue() + str(e), e
        return buf.getvalue(), out

    try:
        d = _selftest_files(tmp)
        work = d / "human/HT-1"
        text, _ = quiet(cmd_build, _selftest_args(d))
        rows = read_table(work / SHEET_FILE)
        by = {r["name_bn"]: r for r in rows}
        assert by[N("সুইচবোর্ড")]["same_object_as"] == N("সুইচ বোর্ড"), "D103 merge"
        assert by[N("টিভি")]["same_object_as"] == N("টেলিভিশন"), "colloquial merge"
        assert by[N("মোড়া")]["same_object_as"] == N("টুল"), "cluster member merge"
        assert by[N("দেকচি")]["priority"] == "1" and by[N("দেকচি")]["n_pname"] == "0"
        assert by[N("ফ্রিজ")]["room_scope"] == "kitchen"
        assert by[N("পড়ার টেবিল")]["head_noun"] == N("টেবিল")
        assert by[N("হাঁড়ি")]["tag_4a"] == "ambiguous"
        assert N("কমোড") not in by
        others = read_table(work / OTHER_NAMES_FILE)
        assert any(r["answer"] == N("কমোড") for r in others)
        say(f"[selftest] OK  build: {len(rows)} rows, three kinds of merge"
            f" pre-filled, room and head noun pre-filled, 4A tags attached")
        text, e = quiet(cmd_build, _selftest_args(d))
        assert isinstance(e, SystemExit) and "STOP" in text
        say("[selftest] OK  a second build refuses to erase the sheet")

        # ---- an empty sheet: nothing is an error, everything is open
        text, r = quiet(lambda a: run_check(a, final=False), _selftest_args(d))
        assert r["rep"].n("ERROR") == 0, r["rep"].lines["ERROR"]
        assert r["rep"].n_open > 20
        say(f"[selftest] OK  check on the untouched sheet: 0 errors,"
            f" {r['rep'].n_open} open cells in {r['rep'].n('TODO')} kinds")

        filled = _selftest_fill(d)
        excl = [{"qa_id": it["qa_id"], "reason": "gold_wrong_kind", "note": "made up"}
                for it in S.load_probe(d / "probes/R-TEST.json")
                if it.get("closed_kind") == "colour" and N(it["answer_bn"]) == N("কাঁচ")]
        write_table(work / EXCLUSION_FILE, excl, EXCLUSION_COLS)
        text, r = quiet(lambda a: run_check(a, final=True), _selftest_args(d))
        assert r["rep"].n("ERROR") == 0, r["rep"].lines["ERROR"]
        objs = r["objects"]
        assert len(objs) == 16, len(objs)
        assert objs[N("টেলিভিশন")]["cultural_specificity"] == "globally_common"
        assert objs[N("সোফা")]["cultural_specificity"] == NOT_RATED
        assert objs[N("টুল")]["t1_bn"] == [N("টুল"), N("মোড়া")]
        lr = {x["name_bn"]: x for x in r["lrows"]}
        assert lr[N("হাঁড়ি")]["en_unique"] == "no" and lr[N("ফ্রিজ")]["en_unique"] == "yes"
        assert lr[N("আলমারি")]["en_unique"] == "no"
        assert lr[N("হাঁড়ি")]["siblings"].split("|") == [N("পাতিল"), N("দেকচি")]
        assert len(r["excluded"]) == 3        # 2 two-object golds + 1 kind conflict
        assert any("NOT in P-CLUST" in w for w in r["rep"].lines["WARNING"]) is False
        assert any("own gold" in i for i in r["rep"].lines["INFO"])
        say("[selftest] OK  the carefully filled sheets pass in final mode:"
            " 16 objects, en_unique, siblings, tags, 3 excluded items, 1 leak")

        # ---- every rule must fire
        def mutate(change, vocab=None, exclusions=None):
            rows2 = [dict(x) for x in read_table(work / SHEET_FILE)]
            by2 = {x["name_bn"]: x for x in rows2}
            keep_v = (work / VOCAB_FILE).read_bytes()
            keep_s = (work / SHEET_FILE).read_bytes()
            keep_e = (work / EXCLUSION_FILE).read_bytes()
            try:
                if change:
                    change(by2, rows2)
                    write_table(work / SHEET_FILE, rows2, SHEET_COLS)
                if vocab:
                    v2 = read_table(work / VOCAB_FILE)
                    vocab(v2)
                    write_table(work / VOCAB_FILE, v2, VOCAB_SHEET_COLS)
                if exclusions is not None:
                    write_table(work / EXCLUSION_FILE, exclusions, EXCLUSION_COLS)
                _, res = quiet(lambda a: run_check(a, final=True), _selftest_args(d))
                return res["rep"].lines["ERROR"]
            finally:
                (work / VOCAB_FILE).write_bytes(keep_v)
                (work / SHEET_FILE).write_bytes(keep_s)
                (work / EXCLUSION_FILE).write_bytes(keep_e)

        def setv(name, col, val):
            return lambda by2, rows2: by2[N(name)].__setitem__(col, val)

        def vset(cls, col, val):
            def f(v2):
                for x in v2:
                    if x["class_id"] == cls:
                        x[col] = val
            return f

        cases = [
            ("writer equals checker", setv("সোফা", "checker", "p1"), None, "same person"),
            ("spacing pair split (D103)", setv("সুইচবোর্ড", "same_object_as", ""), None, "must be one object"),
            ("cluster member split", setv("মোড়া", "same_object_as", ""), None, "must be one object"),
            ("merge chain", setv("সোফা", "same_object_as", "টিভি"), None, "itself a spelling"),
            ("Bangla name in two objects", setv("সোফা", "variants_bn", "ফ্রিজ"), None, "has its own row"),
            ("unattested variant", setv("সোফা", "variants_bn", "কার্টেন"), None, "not a gold answer"),
            ("attested variant is fine", setv("সোফা", "variants_bn", "কমোড"), None, None),
            ("two canonical_en", setv("সোফা", "canonical_en", "sofa|couch"), None, "ONE name"),
            ("canonical_en missing", setv("সোফা", "canonical_en", ""), None, "canonical_en is empty"),
            ("article in an English name", setv("সোফা", "canonical_en", "a sofa"), None, "article"),
            ("Bangla letters in an English cell", setv("সোফা", "variants_en", "সোফা"), None, "written in bn letters"),
            ("vague Tier 2", setv("বালতি", "super_bn", "জিনিস"), None, "too vague"),
            ("Tier 2 on one side only", setv("বালতি", "super_en", ""), None, "both be filled"),
            ("a name that is also the generic name", setv("বালতি", "super_en", "bucket"), None, "generic name"),
            ("one Bangla name on two objects",
             lambda by2, rows2: (by2[N("সোফা")].__setitem__("variants_bn", "কমোড"),
                                 by2[N("ফ্রিজ")].__setitem__("variants_bn", "কমোড")),
             None, "is written on 2 objects"),
            ("two cluster members merged", setv("পাতিল", "same_object_as", "হাঁড়ি"), None, "different members"),
            ("comma instead of |", setv("সোফা", "variants_en", "couch, settee"), None, "separated with | only"),
            ("a typed line without a name",
             lambda by2, rows2: rows2.append(dict({c: "" for c in SHEET_COLS}, writer="P1", checker="P2")),
             None, "has no name_bn"),
            ("two objects generic for each other",
             lambda by2, rows2: (by2[N("টেবিল")].__setitem__("super_bn", "পড়ার টেবিল"),
                                 by2[N("টেবিল")].__setitem__("super_en", "study table")),
             None, "each other's name"),
            ("bad room_scope", setv("সোফা", "room_scope", "hall"), None, "room_scope"),
            ("bad origin3", setv("সোফা", "origin3", "persian"), None, "origin3"),
            ("rule note with a non-English origin", setv("এসি", "origin3", "inherited"), None, "modern English"),
            ("en_route differs, no note", setv("ফ্রিজ", "en_route_note", ""), None, "cross-check"),
            ("name edited", setv("সোফা", "name_bn", "সোফাসেট"), None, "was edited"),
            ("row deleted", lambda by2, rows2: rows2.remove(by2[N("সোফা")]), None, "missing from the sheet"),
            ("non-name with a cluster form", setv("হাঁড়ি", "is_name", "no"), None, None),
            ("gold without a row and not excluded", None, None, "exclusion list"),
            ("vocab: writer equals checker", None, vset("wood", "checker", "P1"), "same person"),
            ("vocab: form in two classes", None, vset("wood", "forms_bn", "কাঠ|কাঁচ"), "two material classes"),
            ("vocab: material also a colour", None, vset("green", "forms_bn", "সবুজ|কাঁচ"), "both a material and a colour"),
            ("vocab: room gold form removed", None, vset("bathroom", "forms_bn", "বাথরুম"), "gold_unmapped"),
            ("vocab: gold number missing", None, vset("2", "forms_bn", "দুই"), "gold_unmapped"),
            ("vocab: bare word of a refusal phrase", None, vset("1", "forms_en", "one|no"), "refusal"),
            ("vocab: danda instead of |", None, vset("wood", "forms_bn", "কাঠ।পারটেক্স"), "separated with | only"),
        ]
        for label, change, vocab, expect in cases:
            if label == "gold without a row and not excluded":
                errs = mutate(None, exclusions=[])
            else:
                errs = mutate(change, vocab)
            if expect is None and label == "attested variant is fine":
                assert not errs, (label, errs)
                say(f"[selftest] OK  {label}")
                continue
            if expect is None:
                assert errs, label
                say(f"[selftest] OK  {label:38s} -> {errs[0][:60]}")
                continue
            hit = [x for x in errs if expect in x]
            assert hit, (label, errs)
            say(f"[selftest] OK  {label:38s} -> {hit[0][:60]}")

        # ---- freeze, load with the scorer, score three answers
        text, _ = quiet(cmd_freeze, _selftest_args(d))
        assert "FROZEN" in text and "PASS on the written files" in text, text[-800:]
        lex = S.load_lexicon(d / "repo/lexicon/lexicon.csv")
        voc = S.load_closed_vocab(d / "repo/lexicon/closed_vocab.csv")
        with open(d / "repo/lexicon/lexicon.csv", encoding="utf-8", newline="") as fh:
            assert next(csv.reader(fh)) == list(S.LEXICON_COLUMNS)
        item = {"qa_id": "t", "qa_type": "obj_name", "answer_bn": "পড়ার টেবিল"}
        assert S.score_item(item, "N-BB", "টেবিল", lex, voc)["label"] == S.TIER2
        assert S.score_item(item, "N-BB", "পড়ার টেবিল", lex, voc)["strict"] == 1
        item = {"qa_id": "t", "qa_type": "obj_name", "answer_bn": "মোড়া"}
        assert S.score_item(item, "N-BE", "a stool", lex, voc)["strict"] == 1
        ex = read_table(d / "repo/probes/excluded_items.csv")
        assert len(ex) == 3 and set(ex[0]) - {"_line"} == set(FROZEN_EXCLUSION_COLS)
        say("[selftest] OK  freeze: files written, loaded by the scorer, audit"
            " PASS, three test answers scored as designed")
        rows2 = read_table(work / SHEET_FILE)
        next(x for x in rows2 if x["name_bn"] == N("সোফা"))["variants_en"] = "couch"
        write_table(work / SHEET_FILE, rows2, SHEET_COLS)
        text, e = quiet(cmd_freeze, _selftest_args(d))
        assert isinstance(e, SystemExit) and "STOP" in text
        say("[selftest] OK  a changed sheet cannot overwrite the frozen files")

        # ---- a dataset correction, then build --update
        for fname in ("P-NAME.json", "P-CLUST.json"):
            doc = json.load(open(d / "probes" / fname, encoding="utf-8"))
            for it in doc["items"]:
                if N(it["answer_bn"]) == N("সোফা") and it["house_id"] == "H02":
                    it["answer_bn"] = "কমোড"
                    it["name_bn"], it["name_id"] = N("কমোড"), a4.name_id(N("কমোড"))
            _write_json(d / "probes" / fname, fname[:-5], doc["items"])
        allitems = {}
        for _, fname in PROBES:
            for it in S.load_probe(d / "probes" / fname):
                allitems[it["qa_id"]] = it
        allitems["x"] = _syn_item(999, "H04", "obj_name", "SYN প্রশ্ন?", "কমোড")
        a5.write_synthetic_json(list(allitems.values()), str(d / "data/merged.json"))
        text, r = quiet(lambda a: run_check(a, final=True), _selftest_args(d))
        assert any("build --update" in x for x in r["rep"].lines["ERROR"])
        text, _ = quiet(cmd_build, _selftest_args(d, update=True))
        rows3 = {x["name_bn"]: x for x in read_table(work / SHEET_FILE)}
        assert rows3[N("সোফা")]["variants_en"] == "couch", "human cell kept"
        assert rows3[N("কমোড")]["auto_note"].startswith("NEW")
        assert rows3[N("সোফা")]["n_pname"] == "1"
        assert list(work.glob("lexicon_sheet.backup-*.csv"))
        say("[selftest] OK  after a gold change: check asks for build --update;"
            " the update adds the new row, keeps human cells, writes a backup")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    say()
    say("SELF-TEST PASSED: build, the merge rules, every validation rule, the")
    say("exclusion list, the freeze, the scorer round trip and build --update.")


# ----------------------------------------------------------------------------
# 10. Command line
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="ASSET-4 v2: lexicon.csv and closed_vocab.csv (Gate A)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")

    def common(s):
        s.add_argument("--drive", default="/content/drive/MyDrive/BanglaIndoorVQA",
                       help="project folder in Drive; every other path defaults"
                            " to a place inside it")
        s.add_argument("--data", help="merged dataset JSON (default:"
                                      " <drive>/data/banglaindoorvqa.json)")
        s.add_argument("--counts", help="house_type_counts.csv")
        s.add_argument("--probes", help="folder with the four probe files")
        s.add_argument("--tags", help="cultural_specificity.csv")
        s.add_argument("--clusters", help="clusters_parsed.csv")
        s.add_argument("--cluster-counts", dest="cluster_counts",
                       help="cluster_counts.csv")
        s.add_argument("--unmatched", help="unmatched_obj_name.csv")
        s.add_argument("--lexicon-txt", dest="lexicon_txt",
                       help="BanglaIndoorVQA_Lexicon.txt (default:"
                            " <drive>/data/BanglaIndoorVQA_Lexicon.txt)")
        s.add_argument("--review-5c", dest="review_5c", help="asset5c_review.csv")
        s.add_argument("--work", help="HT-1 folder (default: <drive>/human/HT-1)")
        s.add_argument("--no-validate", dest="no_validate", action="store_true",
                       help="skip the dataset checks of asset5a_probes (tests only)")

    s = sub.add_parser("build")
    common(s)
    s.add_argument("--update", action="store_true",
                   help="refresh an existing sheet and keep every human cell")
    for name in ("check", "freeze"):
        s = sub.add_parser(name)
        common(s)
        s.add_argument("--no-exclusions", dest="no_exclusions", action="store_true",
                       help="treat any excluded item as an error")
        if name == "freeze":
            s.add_argument("--repo", help="repo root (default: the folder above"
                                          " this script)")
            s.add_argument("--out-lexicon", dest="out_lexicon")
            s.add_argument("--out-probes", dest="out_probes")
            s.add_argument("--force", action="store_true")
    args = ap.parse_args()
    {"selftest": cmd_selftest, "build": cmd_build, "check": cmd_check,
     "freeze": cmd_freeze}[args.cmd](args)


if __name__ == "__main__":
    main()
