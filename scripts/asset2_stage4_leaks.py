#!/usr/bin/env python3
"""
asset2_stage4_leaks.py
BanglaIndoorVQA main paper, ASSET-2 stage 4: the leakage scan of Design 4.6.
Version 2026-10-12. Decision rows D215 to D218 (DESIGN_ADDENDUM section D).

Stage 4 is the second half of Gate B. The first half, stage 3b, passed on
2026-10-12 (146 of 150 faithful, DESIGN_ADDENDUM D7).

THE RULE (Design 4.6; D216)
    A naming question (qa_type obj_name or object) must not contain a Tier-1 term
    of its own gold object. The English question is searched for the gold's
    English Tier-1 terms (canonical_en and variants_en), the Bangla question for
    its Bangla Tier-1 terms (name_bn and variants_bn), because DESIGN_ADDENDUM A5
    ties the Bangla leaks to Gate B too.
    Scope: every naming item of P-CORE-CLOSED, R-TEST and P-NAME (P-CLUST is
    inside P-NAME), once per qa_id. Items in probes/excluded_items.csv are
    scanned and reported with the status "excluded"; they never count (D110).
    The leak half of Gate B passes when no item that is not excluded has a leak,
    every item has an English question, and every gold that is not excluded
    resolves to one object.
    Reported, never counted: (a) a Tier-2 (generic) term of the gold in the
    English question; (b) a gold Tier-1 term that the longest-match rule gives
    to a longer name of another object, such as "rack" inside "shoe rack".

NOTHING HERE IS A SECOND COPY OF PROJECT CODE
    probe files    harness.load_probe()
    English join   harness.load_english(), then the lookup that
                   harness.build_work() uses: table.get(normalise(question_bn))
    gold object    scoring.resolve_gold(); never the probe's name_id and never the
                   gold text (DESIGN_ADDENDUM C4)
    matching       scoring.prepare_answer() and the lexicon's TermIndex.find_all():
                   whole words, longest match first, the spacing rule (D103) and
                   the ending rule (D104: English plural and possessive). These
                   are the scorer's rules for answers, applied to the question.
    The scan is lexical. A question that gives the name away in other words,
    without the name itself, is not found by it.

COMMANDS (run from the repo root)
    check-inputs   one PASS or FAIL line per input; exit code 1 on any FAIL
    selftest       checks on made-up data; reads no dataset file
    scan           the scan; writes Drive english/stage4/<run>/
                       leak_scan_items.csv    one row per naming item (has text)
                       leak_scan_report.txt   counts, ids, sha256 of every input,
                                              the Gate B line
                       leak_fix_sheet.csv     only when a leak counts: one row per
                                              string, for the two post-editors
    check-fixes    --fix-sheet <filled sheet, .csv or .xlsx>: checks every new
                   English with the same rule; when all is clean it writes
                   english/amendments/amendments_stage4_leaks.csv for
                   asset2_stage23.py merge --scope all --overwrite-final (D9, D10)

PRINTING
    Counts and qa_ids only, never a question or an answer. Files with question
    text go to Drive only. No API call, no GPU, no cost.
"""

import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_VERSION = "2026-10-12"
SCRIPT_PATH = Path(__file__).resolve()
REPO = SCRIPT_PATH.parents[1]
DRIVE_DEFAULT = "/content/drive/MyDrive/BanglaIndoorVQA"

sys.path.insert(0, str(REPO / "src"))
IMPORT_ERRORS = {}
try:
    from normalise import normalise
except Exception as exc:                      # reported by check-inputs
    IMPORT_ERRORS["normalise"] = exc
try:
    import scoring
except Exception as exc:
    IMPORT_ERRORS["scoring"] = exc
try:
    import harness                            # also imports cost_ledger, images, prompts
except Exception as exc:
    IMPORT_ERRORS["harness"] = exc

NAMING_TYPES = ("obj_name", "object")
PROBE_FILES = ("P-CORE-CLOSED", "R-TEST", "P-NAME")
# Verified sizes (PROJECT_STATE; DESIGN_ADDENDUM A1 and C1).
EXPECT_PROBES = {
    "P-CORE-CLOSED": {"n": 1118, "obj_name": 255, "object": 255},
    "R-TEST": {"n": 1449, "obj_name": 280, "object": 378},
    "P-NAME": {"n": 1000, "obj_name": 1000, "object": 0},
}
EXPECT_PCLUST = 492
EXPECT_LEXICON_OBJECTS = 231
EXPECT_FINAL_ROWS = 653
EXPECT_PILOT_ROWS = 32
FINAL_V2_SHA256 = "99a67e08ca2910058c09596fffac0259d16c6b217e3cad138de8ed2e8168987c"
PILOT_V2_SHA256 = "fbeb4c3233c0446cc613f5a89f2da39025962aff783f33ee309dca5a45354c07"
GATE_A_TAG = "gate-A"
EXCLUDED_COLUMNS = ["qa_id", "probes", "qa_type", "reason", "source", "added_at"]
REASON_CODES = ("gold_not_one_object", "gold_wrong_kind", "gold_wrong", "other")
BN_LEAK_COLUMNS = ("qa_id", "probes", "qa_type", "gold", "found_in_question")
PILOT_SAME_COLUMNS = ("string_id", "question_bn", "question_bn_norm")
# Conditions that send the English question (Design 7). N-FCE and N-FCE0 run on
# P-FCE, the part of P-CLUST that ASSET-11 keeps (addendum C1).
ENGLISH_CONDITIONS = {"P-CORE-CLOSED": ("L-EB", "L-EE", "L-EE0"), "R-TEST": ("R-EE",),
                      "P-CLUST": ("N-FCE", "N-FCE0")}
STAGE3B_NOTE = "stage 3b passed on 2026-10-12, 146 of 150 faithful (DESIGN_ADDENDUM D7)"
QA_ID_RE = re.compile(r"H\d{2}_\d{4}_q\d+")
AMENDMENTS_FILE = "amendments_stage4_leaks.csv"
AMENDMENT_COLUMNS = ("string_id", "question_en_new", "reason")       # DESIGN_ADDENDUM D9
EXCLUDE_SOURCE = "stage 4 leak scan"
EXCLUDE_ADDED_AT = "ASSET-2 stage 4"
FIX_FILL_COLUMNS = ("action", "question_en_new", "agreed_by_both", "note")
ITEM_COLUMNS = (
    "qa_id", "house_id", "qa_type", "probes", "in_pclust", "english_conditions",
    "string_id", "in_pilot", "status", "leak_where", "excluded_reason",
    "excluded_added_at", "gold_status", "gold_name_id", "gold_name_bn", "gold_en_t1",
    "en_t1_found", "bn_t1_found", "en_t2_found", "en_inside_other_name",
    "question_bn", "question_en")
STATUS_ORDER = ("error_no_english", "error_no_gold", "leak", "excluded", "ok")


class Stop(Exception):
    """A problem that must stop the command. The message says what to do."""


def say(msg=""):
    print(msg, flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    """Run git in the repo. Returns (exit code, stdout bytes)."""
    try:
        r = subprocess.run(["git", "-C", str(REPO)] + list(args), capture_output=True,
                           timeout=60)
    except Exception:
        return None, b""
    return r.returncode, r.stdout


def repo_relative(path):
    try:
        return Path(path).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return None


GIT_OK = ("committed", "outside the repo")
GIT_HELP = {"LOCAL CHANGE": "commit it, or undo it with git checkout -- <file>",
            "NOT IN GIT": "add and commit it", "NO GIT": "run from the cloned repo"}


def git_state(path):
    """committed, outside the repo, LOCAL CHANGE, NOT IN GIT or NO GIT."""
    rel = repo_relative(path)
    if rel is None:
        return "outside the repo"
    code, out = git("status", "--porcelain", "--", rel)
    if code != 0:
        return "NO GIT"
    if out.strip():
        return "LOCAL CHANGE"
    code, _ = git("ls-files", "--error-unmatch", "--", rel)
    return "committed" if code == 0 else "NOT IN GIT"


def git_note(state):
    return state if state in GIT_OK else "%s: %s" % (state, GIT_HELP[state])


def read_csv(path):
    """(header, rows) of a UTF-8 CSV, with or without a byte order mark."""
    with open(str(path), "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = [{k: (v if v is not None else "") for k, v in r.items() if k is not None}
                for r in reader]
    return header, rows


def write_csv(path, columns, rows):
    """UTF-8 with a byte order mark, like every CSV in Drive english/ (addendum D4)."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([r.get(c, "") for c in columns])
    os.replace(tmp, str(path))


def write_text(path, lines):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, str(path))


def short(sha):
    return sha[:12] if sha else "-"


DRIVE_ROOT = [None]                 # set by main(): the --drive folder


def shown_path(path):
    """repo:<path> or drive:<path>, so the report stays short."""
    p = Path(path).resolve()
    for label, root in (("repo", REPO), ("drive", DRIVE_ROOT[0])):
        if root is None:
            continue
        try:
            return "%s:%s" % (label, p.relative_to(Path(root).resolve()).as_posix())
        except ValueError:
            pass
    return str(p)


# --------------------------------------------------------------------------
# 1. The search: the scorer's own rules, applied to a question
# --------------------------------------------------------------------------
def term_hits(text, lex, gold_id, lang):
    """Search `text` exactly as the scorer searches an answer, and keep what
    belongs to the gold object `gold_id` in language `lang` (en or bn).

    Returns (t1, t2, inside):
      t1      matches holding a Tier-1 term of the gold (a leak). A shared
              English name (pot) counts as the gold's own, as in scoring.
      t2      matches holding a Tier-2 term of the gold and no Tier-1 term
      inside  (gold term, [longer matches]) for each gold Tier-1 term that the
              longest-match rule gave to a longer name of another object
    """
    tokens = scoring.prepare_answer(text).norm_tokens
    matches = lex.index.find_all(tokens)
    t1, t2 = [], []
    for m in matches:
        mine = [s for s in m.senses if s.name_id == gold_id and s.lang == lang]
        if any(s.tier == 1 for s in mine):
            t1.append(m)
        elif any(s.tier == 2 for s in mine):
            t2.append(m)
    obj = lex.objects[gold_id]
    gold_index = scoring.TermIndex(lex.settings)      # the gold's Tier-1 terms only
    for term in (obj.t1_en if lang == "en" else obj.t1_bn):
        gold_index.add(term, gold_id)
    t1_spans = [(m.start, m.end) for m in t1]
    inside = []
    for g in gold_index.find_all(tokens):
        if any(a <= g.start and g.end <= b for a, b in t1_spans):
            continue
        over = [m for m in matches if m.start < g.end and g.start < m.end and m not in t1]
        if over:
            inside.append((g, over))
    return t1, t2, inside


def describe_inside(inside):
    parts = []
    for g, over in inside:
        for m in over:
            owners = sorted({s.name_id for s in m.senses})
            parts.append("%s in '%s' (%s)" % (g.surface, m.surface, "|".join(owners)))
    return "; ".join(parts)


def surfaces(matches):
    out = []
    for m in matches:
        if m.surface not in out:
            out.append(m.surface)
    return "|".join(out)


# --------------------------------------------------------------------------
# 2. The scan of all naming items (pure: no file is read or written here)
# --------------------------------------------------------------------------
def english_conditions(probes, in_pclust):
    conds = []
    for p in probes:
        conds.extend(ENGLISH_CONDITIONS.get(p, ()))
    if in_pclust:
        conds.extend(ENGLISH_CONDITIONS["P-CLUST"])
    return "|".join(conds) if conds else "none"


def scan_items(naming, lex, en_table, meta_by_norm, excluded, pclust, overrides=None):
    """One record per naming item.

    naming        {qa_id: {"item": probe record, "probes": [probe set names]}}
    en_table      the table of harness.load_english(): {question_bn_norm: English}
    meta_by_norm  {question_bn_norm: row of the final English file}
    excluded      {qa_id: row of probes/excluded_items.csv}
    pclust        set of P-CLUST qa_ids
    overrides     {string_id: new English}: a what-if for check-fixes
    """
    rows = []
    for qa_id in sorted(naming):
        rec = naming[qa_id]
        it = rec["item"]
        norm_q = normalise(it["question_bn"])
        q_en = en_table.get(norm_q)                 # the lookup of harness.build_work()
        meta = meta_by_norm.get(norm_q) or {}
        sid = meta.get("string_id", "")
        if overrides and sid in overrides:
            q_en = overrides[sid]
        # Naming golds never read the vocabulary (resolve_gold, NAMING_TYPES branch).
        gold = scoring.resolve_gold(it, lex, None)
        en_t1, en_t2, en_in, bn_t1 = [], [], [], []
        obj = lex.objects.get(gold.name_id) if gold.status == "ok" else None
        if obj is not None:
            if q_en is not None:
                en_t1, en_t2, en_in = term_hits(q_en, lex, obj.name_id, "en")
            bn_t1, _t2, _in = term_hits(it["question_bn"], lex, obj.name_id, "bn")
        where = "+".join(w for w, hit in (("en", en_t1), ("bn", bn_t1)) if hit)
        exc = excluded.get(qa_id)
        if q_en is None:
            status = "error_no_english"
        elif exc is not None:
            status = "excluded"
        elif gold.status != "ok":
            status = "error_no_gold"
        elif where:
            status = "leak"
        else:
            status = "ok"
        in_pc = qa_id in pclust
        rows.append({
            "qa_id": qa_id, "house_id": it.get("house_id", ""), "qa_type": it["qa_type"],
            "probes": "|".join(rec["probes"]), "in_pclust": "yes" if in_pc else "no",
            "english_conditions": english_conditions(rec["probes"], in_pc),
            "string_id": sid, "in_pilot": meta.get("in_pilot", ""), "status": status,
            "leak_where": where,
            "excluded_reason": exc.get("reason", "") if exc else "",
            "excluded_added_at": exc.get("added_at", "") if exc else "",
            "gold_status": gold.status, "gold_name_id": gold.name_id,
            "gold_name_bn": obj.name_bn if obj else "",
            "gold_en_t1": "|".join(obj.t1_en) if obj else "",
            "en_t1_found": surfaces(en_t1), "bn_t1_found": surfaces(bn_t1),
            "en_t2_found": surfaces(en_t2), "en_inside_other_name": describe_inside(en_in),
            "question_bn": it["question_bn"], "question_en": q_en or "",
        })
    return rows


def gate_b(rows):
    """(passed, leaks, errors). Leaks and errors of items that are not excluded."""
    leaks = [r for r in rows if r["status"] == "leak"]
    errors = [r for r in rows if r["status"].startswith("error")]
    return (not leaks and not errors), leaks, errors


def by_string(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["string_id"], []).append(r)
    return groups


# --------------------------------------------------------------------------
# 3. Inputs: one PASS or FAIL line each
# --------------------------------------------------------------------------
class Checks:
    def __init__(self):
        self.lines = []
        self.failed = 0
        self.inputs = []                  # (name, path, sha256) of every input read

    def add(self, ok, name, note, path=None, sha=None):
        tag = "PASS" if ok else "FAIL"
        if not ok:
            self.failed += 1
        line = "%s  %-36s %s" % (tag, name, note)
        self.lines.append(line)
        if path is not None:
            self.inputs.append((name, str(path), sha or ""))
        say(line)


class Ctx:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def default_paths(a):
    d = Path(a.drive)
    a.english_file = a.english_file or str(d / "english" / "english_questions_final.csv")
    a.pilot_file = a.pilot_file or str(d / "english" / "english_questions_final_PILOT.csv")
    a.probes_dir = a.probes_dir or str(d / "probes")
    a.bn_leaks = a.bn_leaks or str(d / "human" / "HT-1" / "reports" / "bn_question_leaks.csv")
    a.out_dir = a.out_dir or str(d / "english" / "stage4")
    a.amendments_dir = a.amendments_dir or str(d / "english" / "amendments")
    a.lexicon = a.lexicon or str(REPO / "lexicon" / "lexicon.csv")
    a.excluded = a.excluded or str(REPO / "probes" / "excluded_items.csv")
    a.pclust_ids = a.pclust_ids or str(REPO / "probes" / "P-CLUST.ids.txt")
    return a


def check_code(ck):
    """normalise.py, scoring.py, harness.py and this script: import and Git state."""
    files = (("src/normalise.py", "normalise"), ("src/scoring.py", "scoring"),
             ("src/harness.py", "harness"))
    for rel, mod in files:
        path = REPO / rel
        if not path.exists():
            ck.add(False, rel, "missing in the repo; pull first")
            continue
        sha = sha256_file(path)
        if mod in IMPORT_ERRORS:
            ck.add(False, rel, "import failed: %s" % IMPORT_ERRORS[mod], path, sha)
            continue
        state = git_state(path)
        extra = ""
        if mod == "scoring":
            extra = "scorer %s, " % scoring.SCORER_VERSION
        elif mod == "harness":
            if not (hasattr(harness, "load_english") and hasattr(harness, "load_probe")):
                ck.add(False, rel, "has no load_english() or load_probe(); pull the repo",
                       path, sha)
                continue
            extra = "harness %s, " % harness.HARNESS_VERSION
        ck.add(state in GIT_OK, rel, "%s%s, sha256 %s" % (extra, git_note(state), short(sha)),
               path, sha)
    sha = sha256_file(SCRIPT_PATH)
    state = git_state(SCRIPT_PATH)
    ck.add(state in GIT_OK, "scripts/asset2_stage4_leaks.py",
           "version %s, %s, sha256 %s" % (SCRIPT_VERSION, git_note(state), short(sha)),
           SCRIPT_PATH, sha)
    code, out = git("rev-parse", "HEAD")
    commit = out.decode().strip() if code == 0 else ""
    code_t, _ = git("rev-parse", "--verify", "--quiet", GATE_A_TAG + "^{commit}")
    code_a, _ = git("merge-base", "--is-ancestor", GATE_A_TAG, "HEAD")
    ok = bool(commit) and code_t == 0 and code_a == 0
    note = ("commit %s, contains tag %s" % (commit[:12], GATE_A_TAG) if ok else
            "commit %s; tag %s %s. Run: git fetch --tags, then pull"
            % (commit[:12] or "unknown", GATE_A_TAG,
               "not found" if code_t != 0 else "is not in this history"))
    ck.add(ok, "repo", note)
    return commit


def check_lexicon(ck, path):
    if not os.path.exists(path):
        ck.add(False, "lexicon/lexicon.csv", "missing: %s" % path)
        return None
    sha = sha256_file(path)
    code, blob = git("show", "%s:lexicon/lexicon.csv" % GATE_A_TAG)
    tag_sha = hashlib.sha256(blob).hexdigest() if code == 0 else ""
    if "scoring" in IMPORT_ERRORS:
        ck.add(False, "lexicon/lexicon.csv", "scoring.py did not import", path, sha)
        return None
    try:
        lex = scoring.load_lexicon(path)
    except Exception as exc:
        ck.add(False, "lexicon/lexicon.csv", "does not load: %s" % exc, path, sha)
        return None
    problems = []
    if not tag_sha:
        problems.append("cannot read it at tag %s" % GATE_A_TAG)
    elif sha != tag_sha:
        problems.append("differs from tag %s (the lexicon is frozen)" % GATE_A_TAG)
    if len(lex.objects) != EXPECT_LEXICON_OBJECTS:
        problems.append("%d objects, expected %d" % (len(lex.objects), EXPECT_LEXICON_OBJECTS))
    if lex.warnings:
        problems.append("%d loader warnings, expected 0" % len(lex.warnings))
    note = ("; ".join(problems) if problems else
            "identical to tag %s; %d objects, %d terms, 0 warnings; sha256 %s"
            % (GATE_A_TAG, len(lex.objects), len(lex.index.entries), short(sha)))
    ck.add(not problems, "lexicon/lexicon.csv", note, path, sha)
    return lex if not problems else None


def check_probes(ck, probes_dir):
    probes = {}
    for name in PROBE_FILES:
        path = os.path.join(probes_dir, name + ".json")
        label = name + ".json"
        if not os.path.exists(path):
            ck.add(False, label, "missing: %s" % path)
            continue
        sha = sha256_file(path)
        if "harness" in IMPORT_ERRORS:
            ck.add(False, label, "harness.py did not import, so load_probe() is missing",
                   path, sha)
            continue
        try:
            data = harness.load_probe(path)
        except Exception as exc:
            ck.add(False, label, "does not load: %s" % exc, path, sha)
            continue
        items = data["items"]
        types = Counter(it["qa_type"] for it in items)
        want = EXPECT_PROBES[name]
        problems = []
        if len(items) != want["n"]:
            problems.append("%d items, expected %d" % (len(items), want["n"]))
        for t in NAMING_TYPES:
            if types.get(t, 0) != want[t]:
                problems.append("%s %d, expected %d" % (t, types.get(t, 0), want[t]))
        no_gold = sum(1 for it in items if it["qa_type"] in NAMING_TYPES
                      and not str(it.get("answer_bn") or "").strip())
        if no_gold:
            problems.append("%d naming items have no answer_bn" % no_gold)
        note = ("; ".join(problems) if problems else
                "%d items; obj_name %d, object %d; sha256 %s"
                % (len(items), types.get("obj_name", 0), types.get("object", 0), short(sha)))
        ck.add(not problems, label, note, path, sha)
        if not problems:
            probes[name] = items
    return probes


def collect_naming(ck, probes):
    """One entry per naming qa_id, with every probe set it is in. Every copy of an
    item must have the same question, gold, type and house."""
    naming, differ, where_all = {}, [], {}
    for name in PROBE_FILES:
        for it in probes.get(name, []):
            where_all.setdefault(it["qa_id"], []).append(name)
            if it["qa_type"] not in NAMING_TYPES:
                continue
            rec = naming.get(it["qa_id"])
            if rec is None:
                naming[it["qa_id"]] = {"item": it, "probes": [name]}
                continue
            old = rec["item"]
            if any(old.get(k) != it.get(k) for k in ("qa_type", "question_bn", "answer_bn",
                                                       "house_id")):
                differ.append(it["qa_id"])
            rec["probes"].append(name)
    if len(probes) < len(PROBE_FILES):
        ck.add(False, "naming items", "not counted: a probe file failed above")
        return None, where_all
    multi = Counter(len(r["probes"]) for r in naming.values())
    ok = not differ
    note = ("%d distinct (%d in two files, %d in three); every copy has the same question, "
            "gold, type and house" % (len(naming), multi.get(2, 0), multi.get(3, 0))
            if ok else "%d items differ between probe files (first: %s)"
            % (len(differ), ", ".join(differ[:5])))
    ck.add(ok, "naming items", note)
    return (naming if ok else None), where_all


def check_pclust(ck, path, probes):
    label = "probes/P-CLUST.ids.txt"
    if not os.path.exists(path):
        ck.add(False, label, "missing: %s" % path)
        return None
    sha = sha256_file(path)
    with open(path, "r", encoding="utf-8-sig") as f:
        ids = QA_ID_RE.findall(f.read())
    problems = []
    if len(ids) != len(set(ids)):
        problems.append("a qa_id occurs twice")
    ids = set(ids)
    if len(ids) != EXPECT_PCLUST:
        problems.append("%d ids, expected %d" % (len(ids), EXPECT_PCLUST))
    pname = probes.get("P-NAME")
    if pname is None:
        problems.append("P-NAME.json failed above")
    else:
        pn_ids = {it["qa_id"] for it in pname}
        if not ids <= pn_ids:
            problems.append("%d ids are not in P-NAME" % len(ids - pn_ids))
        flagged = {it["qa_id"] for it in pname if it.get("in_qualifying_cluster") is True}
        if all("in_qualifying_cluster" in it for it in pname) and flagged != ids:
            problems.append("differs from the P-NAME items with in_qualifying_cluster true "
                            "(%d)" % len(flagged))
    state = git_state(path)
    if state not in GIT_OK:
        problems.append(git_note(state))
    note = ("; ".join(problems) if problems else
            "%d ids, all in P-NAME, equal to its in_qualifying_cluster items; %s; sha256 %s"
            % (len(ids), state, short(sha)))
    ck.add(not problems, label, note, path, sha)
    return ids if not problems else None


def check_excluded(ck, path, where_all, probes_ok):
    label = "probes/excluded_items.csv"
    if not os.path.exists(path):
        ck.add(False, label, "missing: %s" % path)
        return None
    sha = sha256_file(path)
    problems = []
    try:
        header, rows = read_csv(path)
    except UnicodeDecodeError:
        ck.add(False, label, "not UTF-8", path, sha)
        return None
    if header != EXCLUDED_COLUMNS:
        problems.append("columns must be %s" % ",".join(EXCLUDED_COLUMNS))
    seen = set()
    for n, r in enumerate(rows, 2):
        q = r.get("qa_id", "").strip()
        if not QA_ID_RE.fullmatch(q):
            problems.append("row %d: qa_id %r" % (n, q))
            continue
        if q in seen:
            problems.append("row %d: %s listed twice" % (n, q))
        seen.add(q)
        if r.get("reason", "") not in REASON_CODES:
            problems.append("row %d: reason must be one of %s" % (n, ", ".join(REASON_CODES)))
        if not r.get("source", "").strip() or not r.get("added_at", "").strip():
            problems.append("row %d: source and added_at must be filled" % n)
        if any(v != v.strip() for v in r.values()):
            problems.append("row %d: a cell starts or ends with a space" % n)
        if probes_ok:
            actual = sorted(where_all.get(q, []))
            listed = sorted(p for p in r.get("probes", "").split("|") if p)
            if not actual:
                problems.append("row %d: %s is in none of the three probe files" % (n, q))
            elif listed != actual:
                problems.append("row %d: %s probes says %s, the files say %s"
                                % (n, q, "|".join(listed), "|".join(actual)))
    code, blob = git("show", "%s:probes/excluded_items.csv" % GATE_A_TAG)
    if code != 0:
        problems.append("cannot read it at tag %s" % GATE_A_TAG)
        gate_rows = []
    else:
        text = blob.decode("utf-8-sig")
        gate_rows = list(csv.DictReader(text.splitlines()))
        if [dict(r) for r in rows[:len(gate_rows)]] != [dict(r) for r in gate_rows]:
            problems.append("the %d rows of tag %s are not unchanged at the top (rows are "
                            "only appended, never edited or removed)" % (len(gate_rows),
                                                                         GATE_A_TAG))
    added = rows[len(gate_rows):]
    if gate_rows and any(r.get("added_at") == GATE_A_TAG for r in added):
        problems.append("an added row says added_at %s" % GATE_A_TAG)
    state = git_state(path)
    if state not in GIT_OK:
        problems.append(git_note(state) + " (the scan must read a committed list)")
    note = ("; ".join(problems[:4]) if problems else
            "%d rows: the %d rows of %s unchanged, %d added%s; %s; sha256 %s"
            % (len(rows), len(gate_rows), GATE_A_TAG, len(added),
               (" (" + ", ".join(r["qa_id"] for r in added) + ")") if added else "",
               state, short(sha)))
    ck.add(not problems, label, note, path, sha)
    return {r["qa_id"]: r for r in rows} if not problems else None


def check_english(ck, path, column, want_sha):
    label = "english_questions_final.csv"
    if not os.path.exists(path):
        ck.add(False, label, "missing: %s" % path)
        return None
    sha = sha256_file(path)
    if "harness" in IMPORT_ERRORS:
        ck.add(False, label, "harness.py did not import, so load_english() is missing",
               path, sha)
        return None
    try:
        eng = harness.load_english(path, column)          # the harness's own reader
    except Exception as exc:
        ck.add(False, label, "harness.load_english() refused it: %s" % exc, path, sha)
        return None
    header, rows = read_csv(path)
    problems = []
    if sha != want_sha:
        problems.append("sha256 %s is not the recorded %s. If this is a new version, check "
                        "the merge report and pass --english-sha256 %s"
                        % (short(sha), short(want_sha), sha))
    if len(rows) != EXPECT_FINAL_ROWS:
        problems.append("%d rows, expected %d" % (len(rows), EXPECT_FINAL_ROWS))
    for col in ("string_id", "question_bn_norm", "in_pilot"):
        vals = [r.get(col, "") for r in rows]
        if col not in header:
            problems.append("no column %s" % col)
        elif col != "in_pilot" and len(set(vals)) != len(vals):
            problems.append("%s is not unique" % col)
    if len(eng["table"]) != len(rows):
        problems.append("%d rows have an empty %s" % (len(rows) - len(eng["table"]), column))
    versions = sorted({r.get("final_version", "") for r in rows})
    note = ("; ".join(problems) if problems else
            "%d rows, final_version %s, sha256 %s (the recorded version)"
            % (len(rows), "|".join(versions), short(sha)))
    ck.add(not problems, label, note, path, sha)
    if problems:
        return None
    return Ctx(table=eng["table"], rows=rows, sha256=sha, column=column, versions=versions,
               meta_by_norm={r["question_bn_norm"]: r for r in rows})


def check_pilot(ck, path, column, want_sha, final):
    label = "english_questions_final_PILOT.csv"
    if not os.path.exists(path):
        ck.add(False, label, "missing: %s" % path)
        return None
    sha = sha256_file(path)
    if final is None or "harness" in IMPORT_ERRORS:
        ck.add(False, label, "not compared: the final file failed above", path, sha)
        return None
    try:
        pil = harness.load_english(path, column)
    except Exception as exc:
        ck.add(False, label, "harness.load_english() refused it: %s" % exc, path, sha)
        return None
    _h, rows = read_csv(path)
    problems = []
    if sha != want_sha:
        problems.append("sha256 %s is not the recorded %s (the PILOT file is never replaced "
                        "once the pilot ran, D135)" % (short(sha), short(want_sha)))
    if len(rows) != EXPECT_PILOT_ROWS:
        problems.append("%d rows, expected %d" % (len(rows), EXPECT_PILOT_ROWS))
    by_sid = {r["string_id"]: r for r in final.rows}
    pilot_sids = [r.get("string_id", "") for r in rows]
    final_pilot = {r["string_id"] for r in final.rows if r.get("in_pilot") == "yes"}
    if set(pilot_sids) != final_pilot:
        problems.append("its string_ids are not the in_pilot rows of the final file")
    differ, other_cols = [], Counter()
    for r in rows:
        f = by_sid.get(r.get("string_id", ""))
        if f is None:
            differ.append(r.get("string_id", "?"))
            continue
        if any(r.get(c, "") != f.get(c, "") for c in PILOT_SAME_COLUMNS + (column,)):
            differ.append(r["string_id"])
        for c in r:
            if c not in PILOT_SAME_COLUMNS + (column,) and r.get(c, "") != f.get(c, ""):
                other_cols[c] += 1
    for norm, en in pil["table"].items():
        if final.table.get(norm) != en:
            if not differ:
                differ.append("(harness table)")
            break
    if differ:
        problems.append("%d rows differ from the final file in %s (first: %s)"
                        % (len(differ), ", ".join(PILOT_SAME_COLUMNS + (column,)), differ[0]))
    note = ("; ".join(problems) if problems else
            "%d rows, sha256 %s (the recorded version); all %d equal to the final file in "
            "%s%s" % (len(rows), short(sha), len(rows), ", ".join(PILOT_SAME_COLUMNS + (column,)),
                      ("; other columns differ: " + ", ".join(sorted(other_cols)))
                      if other_cols else ""))
    ck.add(not problems, label, note, path, sha)
    return pilot_sids if not problems else None


def check_bn_leaks(ck, path, naming):
    label = "bn_question_leaks.csv"
    if not os.path.exists(path):
        ck.add(False, label, "missing: %s" % path)
        return None
    sha = sha256_file(path)
    header, rows = read_csv(path)
    problems = []
    if any(c not in header for c in BN_LEAK_COLUMNS):
        problems.append("columns must include %s" % ", ".join(BN_LEAK_COLUMNS))
    ids = [r.get("qa_id", "") for r in rows]
    if naming is not None:
        unknown = [q for q in ids if q not in naming]
        if unknown:
            problems.append("not a naming item of the probe files: %s" % ", ".join(unknown))
    note = ("; ".join(problems) if problems else
            "%d rows (%s); sha256 %s" % (len(rows), ", ".join(ids) or "none", short(sha)))
    ck.add(not problems, label, note, path, sha)
    return set(ids) if not problems else None


def load_and_check(a):
    """Every input, one PASS or FAIL line each. Returns (checks, context)."""
    say("ASSET-2 stage 4 input checks (asset2_stage4_leaks.py %s, %s)"
        % (SCRIPT_VERSION, utc_now()))
    ck = Checks()
    commit = check_code(ck)
    lex = check_lexicon(ck, a.lexicon)
    probes = check_probes(ck, a.probes_dir)
    naming, where_all = collect_naming(ck, probes)
    pclust = check_pclust(ck, a.pclust_ids, probes)
    excluded = check_excluded(ck, a.excluded, where_all, len(probes) == len(PROBE_FILES))
    final = check_english(ck, a.english_file, a.english_column, a.english_sha256)
    pilot_sids = check_pilot(ck, a.pilot_file, a.english_column, a.pilot_sha256, final)
    bn_known = check_bn_leaks(ck, a.bn_leaks, naming)
    n = len(ck.lines)
    if ck.failed:
        say("STOP: %d of %d input checks failed. Nothing was scanned or written."
            % (ck.failed, n))
    else:
        say("ALL %d INPUT CHECKS PASS" % n)
    return ck, Ctx(commit=commit, lex=lex, probes=probes, naming=naming, pclust=pclust,
                   excluded=excluded, final=final, pilot_sids=pilot_sids, bn_known=bn_known)


# --------------------------------------------------------------------------
# 4. scan
# --------------------------------------------------------------------------
def run_dir(out_dir, en_sha, kind=""):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = Path(out_dir) / ("%s_en-%s%s" % (stamp, en_sha[:8], kind))
    d.mkdir(parents=True, exist_ok=False)
    return d


def probe_table(rows):
    """Counts per probe set, P-CLUST included."""
    sets = list(PROBE_FILES) + ["P-CLUST"]
    out = []
    for p in sets + ["all"]:
        if p == "all":
            rs = rows
        elif p == "P-CLUST":
            rs = [r for r in rows if r["in_pclust"] == "yes"]
        else:
            rs = [r for r in rows if p in r["probes"].split("|")]
        counted = [r for r in rs if r["status"] != "excluded"]
        out.append((p, len(rs), len(rs) - len(counted), len(counted),
                    sum(1 for r in counted if "en" in r["leak_where"]),
                    sum(1 for r in counted if "bn" in r["leak_where"]),
                    sum(1 for r in counted if r["status"].startswith("error")),
                    sum(1 for r in counted if r["en_t2_found"]),
                    sum(1 for r in counted if r["en_inside_other_name"])))
    return out


def report_lines(ck, ctx, rows, outdir, console=False):
    """The report. With console=True the terms found in the questions are left out,
    so the notebook output holds ids and counts only (D130)."""
    passed, leaks, errors = gate_b(rows)
    exc_leak = [r for r in rows if r["status"] == "excluded" and r["leak_where"]]
    bn_now = {r["qa_id"] for r in rows if "bn" in r["leak_where"]}
    leak_strings = sorted({r["string_id"] for r in leaks})
    pilot_leaks = sorted({r["string_id"] for r in leaks if r["in_pilot"] == "yes"})
    t2_rows = [r for r in rows if r["status"] != "excluded" and r["en_t2_found"]]
    t2_terms = Counter(t for r in t2_rows for t in r["en_t2_found"].split("|"))
    in_rows = [r for r in rows if r["status"] != "excluded" and r["en_inside_other_name"]]
    status = Counter(r["status"] for r in rows)
    L = []
    L.append("ASSET-2 stage 4: leakage scan of the naming questions (Design 4.6, Gate B)")
    L.append("script asset2_stage4_leaks.py version %s; run %s; repo commit %s"
             % (SCRIPT_VERSION, utc_now(), ctx.commit or "unknown"))
    L.append("scorer %s; harness %s; matching: %s"
             % (scoring.SCORER_VERSION, harness.HARNESS_VERSION, ctx.lex.settings.describe()))
    L.append("English file: final_version %s, column %s, sha256 %s"
             % ("|".join(ctx.final.versions), ctx.final.column, ctx.final.sha256))
    L.append("output folder: %s" % shown_path(outdir))
    L.append("")
    L.append("INPUTS (sha256)")
    for name, path, sha in ck.inputs:
        L.append("  %s  %s" % (sha, shown_path(path)))
    L.append("")
    L.append("ITEMS")
    L.append("  %-14s %7s %8s %7s %7s %7s %6s %9s %9s"
             % ("probe set", "naming", "excluded", "counted", "leak_en", "leak_bn", "error",
                "tier2(a)", "inside(b)"))
    for p, n, ex, c, le, lb, er, t2, ins in probe_table(rows):
        L.append("  %-14s %7d %8d %7d %7d %7d %6d %9d %9d" % (p, n, ex, c, le, lb, er, t2, ins))
    L.append("  'all' counts each qa_id once. Status: %s"
             % ", ".join("%s %d" % (s, status.get(s, 0)) for s in STATUS_ORDER))
    L.append("")
    L.append("LEAKS THAT COUNT (items not excluded)")
    if not leaks:
        L.append("  none")
    for r in sorted(leaks, key=lambda r: r["qa_id"]):
        found = "" if console else "  found en '%s' bn '%s'" % (r["en_t1_found"],
                                                               r["bn_t1_found"])
        L.append("  %s  %-14s %-6s string %s  gold %s  in %-5s pilot %s%s"
                 % (r["qa_id"], r["probes"], r["qa_type"], r["string_id"],
                    r["gold_name_id"], r["leak_where"], r["in_pilot"], found))
    L.append("  strings: %d; PILOT strings among them: %s"
             % (len(leak_strings), ", ".join(pilot_leaks) if pilot_leaks else "none"))
    for r in errors:
        L.append("  ERROR %s %s" % (r["qa_id"], r["status"]))
    L.append("")
    L.append("EXCLUDED ITEMS WITH A LEAK (reported, never counted, D110)")
    if not exc_leak:
        L.append("  none")
    for r in sorted(exc_leak, key=lambda r: r["qa_id"]):
        L.append("  %s  %-14s string %s  in %-5s added_at %s"
                 % (r["qa_id"], r["probes"], r["string_id"], r["leak_where"],
                    r["excluded_added_at"]))
    L.append("")
    L.append("INFORMATION LISTS (never decide Gate B; items not excluded)")
    L.append("  (a) a Tier-2 term of the gold in the English question: %d items, %d strings; "
             "terms: %s" % (len(t2_rows), len({r["string_id"] for r in t2_rows}),
                            ", ".join("%s %d" % kv for kv in t2_terms.most_common())
                            or "none"))
    L.append("  (b) a gold Tier-1 term inside a longer name of another object: %d items%s"
             % (len(in_rows), (": " + ", ".join(r["qa_id"] for r in in_rows)) if in_rows
                else ""))
    L.append("")
    L.append("CROSS-CHECKS")
    same = bn_now == ctx.bn_known
    L.append("  Bangla Tier-1 names in the Bangla question, found now: %d (%s); "
             "bn_question_leaks.csv lists %d; same: %s"
             % (len(bn_now), ", ".join(sorted(bn_now)) or "none", len(ctx.bn_known),
                "yes" if same else "NO: read both lists before deciding"))
    L.append("  PILOT file: 32 rows equal to the final file (see the input checks)")
    L.append("")
    if passed:
        L.append("GATE B LEAK SCAN: PASS (0 leaks, 0 errors in %d counted naming items)"
                 % sum(1 for r in rows if r["status"] != "excluded"))
        L.append("GATE B: PASS. %s; leak scan PASS on the English file with sha256 %s."
                 % (STAGE3B_NOTE, ctx.final.sha256))
        L.append("Every English-question run before Gate E uses this file version.")
    else:
        L.append("GATE B LEAK SCAN: FAIL (%d leaks in %d strings, %d errors)"
                 % (len(leaks), len(leak_strings), len(errors)))
        L.append("GATE B: FAIL. Fix route: leak_fix_sheet.csv, then check-fixes "
                 "(DESIGN_ADDENDUM D9, D10; D215 to D218).")
    return L, passed


def fix_sheet_rows(rows, lex):
    """One row per string with a counted leak, for the two post-editors."""
    groups = by_string(rows)
    out = []
    leak_sids = {r["string_id"] for r in rows if r["status"] == "leak"}
    for sid in sorted(leak_sids, key=lambda s: min(r["qa_id"] for r in groups[s])):
        rs = groups[sid]
        live = [r for r in rs if r["status"] != "excluded"]
        forbidden = []
        for r in live:
            for t in lex.objects[r["gold_name_id"]].t1_en if r["gold_name_id"] else []:
                if t not in forbidden:
                    forbidden.append(t)
        first = rs[0]
        out.append({
            "string_id": sid, "in_pilot": first["in_pilot"],
            "qa_ids": "|".join(r["qa_id"] for r in live),
            "leak_where": "|".join(sorted({r["leak_where"] for r in live if r["leak_where"]})),
            "gold_name_bn": "|".join(sorted({r["gold_name_bn"] for r in live})),
            "forbidden_en": "|".join(forbidden),
            "en_t1_found": "|".join(sorted({r["en_t1_found"] for r in live if r["en_t1_found"]})),
            "bn_t1_found": "|".join(sorted({r["bn_t1_found"] for r in live if r["bn_t1_found"]})),
            "question_bn": first["question_bn"], "question_en": first["question_en"],
            "action": "", "question_en_new": "", "agreed_by_both": "", "note": ""})
    return out


FIX_SHEET_COLUMNS = ("string_id", "in_pilot", "qa_ids", "leak_where", "gold_name_bn",
                     "forbidden_en", "en_t1_found", "bn_t1_found", "question_bn",
                     "question_en") + FIX_FILL_COLUMNS


def cmd_scan(a):
    ck, ctx = load_and_check(a)
    if ck.failed:
        return 1
    say("")
    say("Scanning %d naming items (obj_name, object) of %s, once per qa_id."
        % (len(ctx.naming), ", ".join(PROBE_FILES)))
    rows = scan_items(ctx.naming, ctx.lex, ctx.final.table, ctx.final.meta_by_norm,
                      ctx.excluded, ctx.pclust)
    rows.sort(key=lambda r: (STATUS_ORDER.index(r["status"]), r["qa_id"]))
    outdir = run_dir(a.out_dir, ctx.final.sha256)
    lines, passed = report_lines(ck, ctx, rows, outdir)
    write_csv(outdir / "leak_scan_items.csv", ITEM_COLUMNS, rows)
    write_text(outdir / "leak_scan_report.txt", lines)
    wrote = ["leak_scan_items.csv (%d rows)" % len(rows), "leak_scan_report.txt"]
    if not passed:
        fx = fix_sheet_rows(rows, ctx.lex)
        if fx:
            write_csv(outdir / "leak_fix_sheet.csv", FIX_SHEET_COLUMNS, fx)
            wrote.append("leak_fix_sheet.csv (%d strings)" % len(fx))
    shown, _p = report_lines(ck, ctx, rows, outdir, console=True)
    start = next(i for i, l in enumerate(shown) if l.startswith("ITEMS"))
    for line in shown[start:]:
        say(line)
    say("")
    say("Wrote to %s:" % outdir)
    for w in wrote:
        say("  " + w)
    return 0 if passed else 2


# --------------------------------------------------------------------------
# 5. check-fixes
# --------------------------------------------------------------------------
def read_sheet(path):
    """The filled fix sheet: CSV (UTF-8) or XLSX (first sheet, header in row 1)."""
    if str(path).lower().endswith(".xlsx"):
        import openpyxl
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        ws = wb.worksheets[0]
        it = ws.iter_rows(values_only=True)
        header = [str(c or "").strip() for c in next(it)]
        rows = []
        for vals in it:
            rows.append({h: ("" if v is None else str(v)) for h, v in zip(header, vals) if h})
        return header, rows
    try:
        return read_csv(path)
    except UnicodeDecodeError:
        raise Stop("%s is not UTF-8. Save it as 'CSV UTF-8' (Excel) or use File > Download "
                   "> Comma-separated values (Google Sheets)." % path)


def check_fix_rows(sheet, rows, lex, allow_pilot=False):
    """Check the filled rows against the current scan. Returns
    (problems, amend {string_id: English}, exclude_lines, info)."""
    groups = by_string(rows)
    leak_sids = {r["string_id"] for r in rows if r["status"] == "leak"}
    problems, info, amend, exclude_lines, seen = [], [], {}, [], set()
    for n, r in enumerate(sheet, 2):
        sid = (r.get("string_id") or "").strip()
        if not sid and not any(str(v).strip() for v in r.values()):
            continue
        where = "row %d (%s)" % (n, sid or "no string_id")
        if sid in seen:
            problems.append("%s: the string is in the sheet twice" % where)
            continue
        seen.add(sid)
        if sid not in leak_sids:
            problems.append("%s: no counted leak in the current scan (already fixed or "
                            "excluded?); delete the row" % where)
            continue
        live = [x for x in groups[sid] if x["status"] != "excluded"]
        action = (r.get("action") or "").strip().lower()
        if action == "exclude":
            for x in live:
                exclude_lines.append("%s,%s,%s,other,%s,%s" % (
                    x["qa_id"], x["probes"], x["qa_type"], EXCLUDE_SOURCE, EXCLUDE_ADDED_AT))
            continue
        if action != "amend":
            problems.append("%s: action must be amend or exclude" % where)
            continue
        raw = r.get("question_en_new") or ""
        new = " ".join(raw.split())
        if (r.get("agreed_by_both") or "").strip().lower() != "yes":
            problems.append("%s: agreed_by_both must be yes (both post-editors agree, D138)"
                            % where)
        if not new:
            problems.append("%s: question_en_new is empty" % where)
            continue
        if "\n" in raw.strip() or "\r" in raw.strip():
            problems.append("%s: question_en_new has a line break" % where)
        if new == " ".join(live[0]["question_en"].split()):
            problems.append("%s: question_en_new equals the current English" % where)
        if any("bn" in x["leak_where"] for x in live):
            problems.append("%s: the Bangla question names the gold; an English change cannot "
                            "fix that. Use exclude" % where)
        if live[0]["in_pilot"] == "yes" and not allow_pilot:
            problems.append("%s: a PILOT string. Changing it needs merge --scope pilot too, "
                            "allowed only if the pilot has not run (D135); add "
                            "--allow-pilot-string after checking" % where)
        for x in live:
            t1, t2, ins = term_hits(new, lex, x["gold_name_id"], "en")
            if t1:
                problems.append("%s: the new English still holds %s, a Tier-1 term of the gold "
                                "of %s" % (where, surfaces(t1), x["qa_id"]))
            if t2:
                info.append("%s: the new English holds the generic term %s (list a, allowed)"
                            % (where, surfaces(t2)))
            if ins:
                info.append("%s: the new English holds %s (list b, allowed)"
                            % (where, describe_inside(ins)))
        amend[sid] = new
    for sid in sorted(leak_sids - seen):
        problems.append("string %s leaks and has no row in the sheet" % sid)
    return problems, amend, exclude_lines, info


def cmd_check_fixes(a):
    if not a.fix_sheet:
        raise Stop("check-fixes needs --fix-sheet <the filled sheet, .csv or .xlsx>.")
    if not os.path.exists(a.fix_sheet):
        raise Stop("Missing fix sheet: %s" % a.fix_sheet)
    ck, ctx = load_and_check(a)
    if ck.failed:
        return 1
    sheet_sha = sha256_file(a.fix_sheet)
    header, sheet = read_sheet(a.fix_sheet)
    need = ("string_id",) + FIX_FILL_COLUMNS[:3]
    missing = [c for c in need if c not in header]
    if missing:
        raise Stop("The fix sheet lacks the column(s) %s." % ", ".join(missing))
    say("")
    amend_path = Path(a.amendments_dir) / AMENDMENTS_FILE
    if amend_path.exists() and not a.overwrite:
        raise Stop("%s exists. Read it first; to replace it add --overwrite."
                   % shown_path(amend_path))
    say("Checking the fix sheet %s (sha256 %s, %d rows)." % (shown_path(a.fix_sheet),
                                                             short(sheet_sha), len(sheet)))
    rows = scan_items(ctx.naming, ctx.lex, ctx.final.table, ctx.final.meta_by_norm,
                      ctx.excluded, ctx.pclust)
    problems, amend, exclude_lines, info = check_fix_rows(sheet, rows, ctx.lex,
                                                          a.allow_pilot_string)
    # What-if: the amendments applied and the exclusions added, then the full scan again.
    extra = {line.split(",", 1)[0]: {"reason": "other", "added_at": EXCLUDE_ADDED_AT}
             for line in exclude_lines}
    excluded2 = dict(ctx.excluded)
    excluded2.update(extra)
    rows2 = scan_items(ctx.naming, ctx.lex, ctx.final.table, ctx.final.meta_by_norm,
                       excluded2, ctx.pclust, overrides=amend)
    passed2, leaks2, errors2 = gate_b(rows2)
    if not passed2 and not problems:
        problems.append("after the fixes %d leaks and %d errors would remain (first: %s)"
                        % (len(leaks2), len(errors2), (leaks2 + errors2)[0]["qa_id"]))
    outdir = run_dir(a.out_dir, ctx.final.sha256, "_check-fixes")
    L = ["ASSET-2 stage 4: check of the fix sheet",
         "script version %s; run %s; repo commit %s" % (SCRIPT_VERSION, utc_now(), ctx.commit),
         "fix sheet %s, sha256 %s" % (shown_path(a.fix_sheet), sheet_sha),
         "English file sha256 %s" % ctx.final.sha256, ""]
    L.append("amend: %d strings (%s)" % (len(amend), ", ".join(sorted(amend)) or "none"))
    L.append("items to exclude: %d" % len(exclude_lines))
    L.append("after the fixes: %d leaks, %d errors" % (len(leaks2), len(errors2)))
    L += [""] + ["INFO " + x for x in info] + ["PROBLEM " + x for x in problems]
    if problems:
        L.append("CHECK-FIXES: FAIL. Nothing was written to %s." % a.amendments_dir)
    else:
        if amend:
            amend_path.parent.mkdir(parents=True, exist_ok=True)
            found = {r["string_id"]: r["en_t1_found"] for r in rows if r["status"] == "leak"}
            write_csv(amend_path, AMENDMENT_COLUMNS, [
                {"string_id": sid, "question_en_new": amend[sid],
                 "reason": "ASSET-2 stage 4 leak scan (D218): the English held the gold "
                           "name '%s'" % found.get(sid, "")} for sid in sorted(amend)])
            L.append("amendments file: %s (%d rows, sha256 %s)"
                     % (shown_path(amend_path), len(amend), sha256_file(amend_path)))
        if exclude_lines:
            L.append("Lines to append to probes/excluded_items.csv (under D218; name them "
                     "at the close):")
            L += ["  " + x for x in exclude_lines]
        L.append("CHECK-FIXES: PASS")
    write_text(outdir / "check_fixes_report.txt", L)
    for line in L[5:]:
        say(line)
    say("Wrote %s" % (outdir / "check_fixes_report.txt"))
    if not problems:
        say("Next: append and commit any lines above, run asset2_stage23.py merge --scope all "
            "--overwrite-final, then scan again with --english-sha256 <the new sha256>.")
    return 0 if not problems else 2


# --------------------------------------------------------------------------
# 6. selftest (made-up data only)
# --------------------------------------------------------------------------
def _lex_row(name_id, name_bn, canonical_en, variants_en="", super_bn="", super_en="",
             variants_bn=""):
    return {"name_id": name_id, "name_bn": name_bn, "variants_bn": variants_bn,
            "canonical_en": canonical_en, "variants_en": variants_en, "super_bn": super_bn,
            "super_en": super_en, "cluster_id": "", "siblings": "", "room_scope": "any",
            "cultural_specificity": "not_rated", "origin3": "english", "origin_note": "",
            "en_route": "", "en_unique": "", "freq_obj_name": "0"}


def cmd_selftest(_a):
    if IMPORT_ERRORS:
        raise Stop("import failed: %s" % "; ".join("%s: %s" % kv for kv in IMPORT_ERRORS.items()))
    say("Self test on made-up names and questions (no dataset file is read).")
    # Every name is invented, in both languages, so no dataset text enters the repo.
    lex = scoring.build_lexicon([
        _lex_row("X_zorb", "কলকা", "zorb", "blue zorb|quill"),
        _lex_row("X_plank", "ঝিমকা", "plank"),
        _lex_row("X_bootplank", "পুটলা", "boot plank", super_en="shelfo"),
        _lex_row("X_trestle", "টেবলা", "trestle"),
        _lex_row("X_study", "পড়া টেবলা", "study trestle", "desko", "টেবলা", "trestle"),
        _lex_row("X_crock1", "হাঁড়িকা", "crock"),
        _lex_row("X_crock2", "পাতিলা", "cooking crock", "crock"),
    ], source="selftest")
    # (qa_id, gold, Bangla question, English question or None for no English row)
    cases = [
        ("T01_0001_q2", "কলকা", "প্রশ্ন এক?", "What is the blue zorb on the left called?"),
        ("T01_0002_q2", "কলকা", "প্রশ্ন দুই?", "What are the zorbs here called?"),
        ("T01_0003_q2", "কলকা", "প্রশ্ন তিন?", "What is the quill's knob called?"),
        ("T01_0004_q2", "কলকা", "প্রশ্ন চার?", "What is the blue-zorb called?"),
        ("T01_0005_q2", "কলকা", "প্রশ্ন পাঁচ?", "What is the bluezorb called?"),
        ("T01_0006_q2", "কলকা", "প্রশ্ন ছয়?", "What is the device for frobbing called?"),
        ("T01_0007_q2", "ঝিমকা", "প্রশ্ন সাত?", "What is the thing beside the boot plank called?"),
        ("T01_0008_q2", "পড়া টেবলা।", "প্রশ্ন আট?", "What is the trestle for reading called?"),
        ("T01_0009_q2", "হাঁড়িকা", "প্রশ্ন নয়?", "What is the cooking crock called?"),
        ("T01_0010_q2", "হাঁড়িকা", "প্রশ্ন দশ?", "What is the black crock called?"),
        ("T01_0011_q2", "কলকা", "প্রশ্ন এগারো?", "What is the zorb called?"),
        ("T01_0012_q2", "ঝিমকা", "দেয়ালের ঝিমকাটিকে কী বলে?", "What is this called?"),
        ("T01_0013_q2", "কলকা", "প্রশ্ন তেরো?", None),
        ("T01_0014_q2", "অজানা জিনিস", "প্রশ্ন চৌদ্দ?", "What is this thing called?"),
        ("T01_0015_q2", "অজানা জিনিস", "প্রশ্ন পনেরো?", "What is that thing called?"),
        ("T01_0016_q2", "কলকা", "  প্রশ্ন   ষোলো?।", "WHAT IS THE ZORB CALLED?"),
        ("T01_0017_q2", "কলকা", "প্রশ্ন সতেরো?", "What is the zorbistry called?"),
    ]
    naming, en_rows = {}, []
    for k, (q, gold, q_bn, q_en) in enumerate(cases, 1):
        naming[q] = {"item": {"qa_id": q, "qa_type": "object", "question_bn": q_bn,
                              "answer_bn": gold, "house_id": "T01"},
                     "probes": ["R-TEST"] if k % 2 else ["P-CORE-CLOSED", "R-TEST"]}
        if q_en is not None:
            norm = normalise(q_bn)
            en_rows.append({"string_id": "s_%02d" % k, "in_pilot": "no", "question_bn": q_bn,
                            "question_en": q_en, "question_bn_norm": norm,
                            "final_version": "9"})
    with tempfile.TemporaryDirectory() as tmp:
        en_path = os.path.join(tmp, "english.csv")
        write_csv(en_path, ("string_id", "in_pilot", "question_bn", "question_en",
                            "question_bn_norm", "final_version"), en_rows)
        eng = harness.load_english(en_path, "question_en")      # the real harness reader
    meta = {r["question_bn_norm"]: r for r in en_rows}
    excluded = {"T01_0011_q2": {"reason": "other", "added_at": "test"},
                "T01_0015_q2": {"reason": "gold_not_one_object", "added_at": "test"}}
    rows = {r["qa_id"]: r for r in scan_items(naming, lex, eng["table"], meta, excluded,
                                              {"T01_0002_q2"})}
    fails = total = 0

    def check(label, ok):
        nonlocal fails, total
        total += 1
        fails += 0 if ok else 1
        say("%s %s" % ("OK  " if ok else "FAIL", label))

    def st(q):
        return rows[q]["status"], rows[q]["leak_where"]

    check("multi-word term, whole words: 'blue zorb' is a leak", st("T01_0001_q2") == ("leak", "en"))
    check("English plural: 'zorbs' is a leak", st("T01_0002_q2") == ("leak", "en"))
    check("English possessive: \"quill's\" is a leak", st("T01_0003_q2") == ("leak", "en"))
    check("spacing rule: 'blue-zorb' is a leak", st("T01_0004_q2") == ("leak", "en"))
    check("spacing rule: 'bluezorb' is a leak", st("T01_0005_q2") == ("leak", "en"))
    check("no gold term: ok", st("T01_0006_q2") == ("ok", ""))
    check("list (b): 'plank' inside 'boot plank' is reported, not a leak",
          st("T01_0007_q2") == ("ok", "") and "plank in 'boot plank'"
          in rows["T01_0007_q2"]["en_inside_other_name"])
    check("list (a) and generic-name precedence: 'trestle' for a study trestle is Tier 2",
          st("T01_0008_q2") == ("ok", "") and rows["T01_0008_q2"]["en_t2_found"] == "trestle")
    check("gold resolved through resolve_gold (the gold text had a dari)",
          rows["T01_0008_q2"]["gold_name_id"] == "X_study")
    check("list (b): 'crock' inside 'cooking crock' (another object) is not a leak",
          st("T01_0009_q2") == ("ok", "") and rows["T01_0009_q2"]["en_inside_other_name"])
    check("shared English name 'crock' counts as the gold's own: leak",
          st("T01_0010_q2") == ("leak", "en"))
    check("excluded item is scanned, reported, and has its own status",
          st("T01_0011_q2") == ("excluded", "en"))
    check("Bangla question with the gold name plus endings: Bangla leak",
          st("T01_0012_q2") == ("leak", "bn") and rows["T01_0012_q2"]["bn_t1_found"])
    check("no English row: error_no_english", st("T01_0013_q2")[0] == "error_no_english")
    check("gold not in the lexicon, not excluded: error_no_gold",
          st("T01_0014_q2")[0] == "error_no_gold")
    check("gold not in the lexicon, excluded: excluded",
          st("T01_0015_q2")[0] == "excluded" and rows["T01_0015_q2"]["gold_status"] != "ok")
    check("join through normalise(question_bn) and upper case English: leak",
          st("T01_0016_q2") == ("leak", "en") and rows["T01_0016_q2"]["string_id"] == "s_16")
    check("not a whole word: 'zorbistry' is ok", st("T01_0017_q2") == ("ok", ""))
    check("P-CLUST item gets the N-FCE conditions",
          rows["T01_0002_q2"]["english_conditions"] == "L-EB|L-EE|L-EE0|R-EE|N-FCE|N-FCE0")
    allrows = list(rows.values())
    passed, leaks, errors = gate_b(allrows)
    check("Gate B fails with leaks and errors", not passed and len(leaks) == 8
          and len(errors) == 2)
    clean = [r for r in allrows if r["status"] in ("ok", "excluded")]
    check("Gate B passes with only ok and excluded items", gate_b(clean)[0])
    # check-fixes on made-up sheets
    sheet = [{"string_id": "s_01", "action": "amend", "agreed_by_both": "yes",
              "question_en_new": "What is the device on the left called?"},
             {"string_id": "s_02", "action": "amend", "agreed_by_both": "yes",
              "question_en_new": "What are the quills here called?"},
             {"string_id": "s_12", "action": "amend", "agreed_by_both": "yes",
              "question_en_new": "What is the thing here called?"},
             {"string_id": "s_03", "action": "exclude"}]
    problems, amend, excl, _info = check_fix_rows(sheet, allrows, lex)
    check("check-fixes accepts a clean new English", "s_01" in amend
          and not any("(s_01)" in p for p in problems))
    check("check-fixes refuses a new English that still holds a gold term",
          any("(s_02)" in p and "quills" in p for p in problems))
    check("check-fixes refuses an English change for a Bangla leak",
          any("(s_12)" in p and "Use exclude" in p for p in problems))
    check("check-fixes writes the exclusion line for an excluded string",
          excl == ["T01_0003_q2,R-TEST,object,other,%s,%s" % (EXCLUDE_SOURCE, EXCLUDE_ADDED_AT)])
    check("check-fixes refuses a sheet that leaves a leaking string out",
          any("s_04" in p and "no row" in p for p in problems))
    say("")
    say("%d of %d self checks passed" % (total - fails, total))
    return 0 if fails == 0 else 1


# --------------------------------------------------------------------------
# 7. Command line
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="ASSET-2 stage 4: leakage scan of the naming questions (Gate B).")
    p.add_argument("command", choices=("check-inputs", "selftest", "scan", "check-fixes"))
    p.add_argument("--drive", default=DRIVE_DEFAULT, help="the project folder in Drive")
    p.add_argument("--english-file", help="default: <drive>/english/english_questions_final.csv")
    p.add_argument("--pilot-file", help="default: <drive>/english/english_questions_final_PILOT.csv")
    p.add_argument("--english-column", default="question_en")
    p.add_argument("--english-sha256", default=FINAL_V2_SHA256,
                   help="the recorded sha256 of the final file (default: version 2)")
    p.add_argument("--pilot-sha256", default=PILOT_V2_SHA256,
                   help="the recorded sha256 of the PILOT file (default: version 2)")
    p.add_argument("--probes-dir", help="default: <drive>/probes")
    p.add_argument("--bn-leaks", help="default: <drive>/human/HT-1/reports/bn_question_leaks.csv")
    p.add_argument("--out-dir", help="default: <drive>/english/stage4")
    p.add_argument("--amendments-dir", help="default: <drive>/english/amendments")
    p.add_argument("--lexicon", help="default: <repo>/lexicon/lexicon.csv")
    p.add_argument("--excluded", help="default: <repo>/probes/excluded_items.csv")
    p.add_argument("--pclust-ids", help="default: <repo>/probes/P-CLUST.ids.txt")
    p.add_argument("--fix-sheet", help="check-fixes: the filled sheet (.csv or .xlsx)")
    p.add_argument("--overwrite", action="store_true",
                   help="check-fixes: replace an existing amendments_stage4_leaks.csv")
    p.add_argument("--allow-pilot-string", action="store_true",
                   help="check-fixes: allow a change to a PILOT string (D135)")
    return p


def main(argv=None):
    a = default_paths(build_parser().parse_args(argv))
    DRIVE_ROOT[0] = a.drive
    try:
        if a.command == "selftest":
            return cmd_selftest(a)
        if a.command == "check-inputs":
            ck, _ctx = load_and_check(a)
            return 1 if ck.failed else 0
        if a.command == "scan":
            return cmd_scan(a)
        return cmd_check_fixes(a)
    except Stop as exc:
        say("STOP: %s" % exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
