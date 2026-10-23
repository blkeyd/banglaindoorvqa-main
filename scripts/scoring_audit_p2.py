#!/usr/bin/env python3
"""
scoring_audit_p2.py  (Scoring and audit, part 2)
BanglaIndoorVQA main paper. Design v2.1 at the tag gate-E, DESIGN_ADDENDUM K onward.

The HT-3 final scores (D208) and HT-5, the error audit (Design 9.3, 9.4 item 12 and 10;
DESIGN_ADDENDUM O5, I5, J3 (D234, D238), B7, B10, H3 (D204)). Decisions D280 to D289
(DESIGN_ADDENDUM P).

Commands (one notebook cell each; every printout holds ids and counts only)
  check          read-only. One PASS or FAIL line per input.                      Cell 0
  selftest       the HT-5 chain on made-up answers in a temporary folder.         Cell 1
  ht3-final      asset12 `check-inputs` and `score --final` from the gate-F
                 checkout, then ht3-check.                                         Cell 2
  ht3-check      the HT-3 final folder, read as the Statistics chat reads it.     (Cell 2)
  budget-items   analysis.py `budget-items --mode real` from the gate-E checkout,
                 then the frame check.                                             Cell 3
  ht5-draw       the 200 answers, the pools beside them, two packages, the key.   Cell 4
  ht5-receive    receipts of the returned sheets; the consensus package.          Cell 5
  ht5-consensus  the consensus; human/HT-5/ht5_audit.csv (D238).                  Cell 6
  ht5-read       the HT-5 reading: stats.read_ht5, stats.ht5_stats, counts.        Cell 7
  sens-template  the empty lexicon_sensitivity.csv and the worksheet.             Cell 8
  sens-check     lexicon_sensitivity.csv read and applied with the gate-E
                 stats.py, as the Statistics chat will.                           Cell 9
  status         what exists so far.                                              Cell 10

Rules built in (decision rows of DESIGN_ADDENDUM P)
  D280  HT-3 names are never shown to the auditors. The HT-3 look-first items (both
        namers named the same other object; a picture frame named as a painting) get
        one scorer-wrong N-BB answer each, beside the 200. After the consensus the
        reading counts missing_variant forms that equal a namer's name.
  D281  Two auditors, different people, neither the runner. Each labels every row
        alone, in an own seeded order with own codes. Rows that differ are settled at
        a consensus that never shows the scorer's label. ht5_audit.csv holds the
        consensus verdict; both raw verdicts are kept in ht5_verdicts.csv.
  D282  The D192 answers (touched and scored wrong, 14 as counted in addendum O2;
        copies of an EXP-L answer left out) are shown beside the 200, kept out of
        ht5_audit.csv. The D279 patterns are not added; the reading counts them.
  D283  HT-3 final scores from the detached gate-F checkout; budget-items from the
        detached gate-E checkout; this script from main, importing scoring,
        normalise and stats from the gate-E checkout. Nothing is written in a Git tree.
  D284  Frames and draw: C = category C of error_budget_items.csv (N-BB, gemini,
        qwen, gemma, pooled); english_copy and sibling = the scorer label in N-BB,
        N-BE and N-BBs of the same models, pooled; no excluded item, no copy; lowest
        seeded rank; each answer once (C, then english_copy, then sibling).
  D285  Verdicts and sheet content (ht5-rules-v1): five checks in a fixed order, the
        first that fits; picture, question as asked, gold, gold names, every sibling
        with every Tier 1 name; never the model, condition, stratum or scorer label.
  D286  Packages and receipts: byte-identical images named by code, no metadata; a
        read-only page with the pictures; the guide; receipts by sha256, never replaced.
  D287  lexicon_sensitivity.csv is written by people only, from an empty template;
        HT-5 rows only from consensus missing_variant forms; checked with stats.py.
  D288  Files in Drive human/HT-5/; ids and counts in printouts; text in TEXT_ files.

Files (Drive project folder D; nothing in the repo)
  D/human/HT-3/score_final_<utc>/  (asset12_nameref.py writes it)
  D/human/HT-3/final_check/        ht3_final_check_<utc>.json, TEXT_ht3_final_sample10.csv
  D/results/stats_inputs/error_budget_items.csv (+ _manifest.json; analysis.py writes it)
  D/human/HT-5/guide/              HT5_auditor_guide.md, HT5_sensitivity_instructions.md
  D/human/HT-5/to_send/            HT5_AU1.zip, HT5_AU2.zip, HT5_consensus.zip, blanks, rules
  D/human/HT-5/key/                ht5_key.csv, ht5_draw.json (never sent)
  D/human/HT-5/received/           returned sheets, consensus, receipts.jsonl
  D/human/HT-5/                    ht5_audit.csv (D238), ht5_verdicts.csv,
                                   TEXT_ht5_forms.csv, TEXT_ht5_notes.csv,
                                   lexicon_sensitivity_TEMPLATE.csv,
                                   HT5_sensitivity_worksheet.xlsx,
                                   lexicon_sensitivity.csv (people), reading/, _records/
"""

import sys
sys.dont_write_bytecode = True   # never write __pycache__ into a Git tree (D252)

import argparse                   # noqa: E402
import csv                        # noqa: E402
import hashlib                    # noqa: E402
import html                       # noqa: E402
import importlib.util             # noqa: E402
import io                         # noqa: E402
import json                       # noqa: E402
import os                         # noqa: E402
import re                         # noqa: E402
import shutil                     # noqa: E402
import subprocess                 # noqa: E402
import tempfile                   # noqa: E402
import zipfile                    # noqa: E402
from collections import Counter, OrderedDict, defaultdict   # noqa: E402
from datetime import datetime, timezone                     # noqa: E402
from pathlib import Path                                     # noqa: E402
from types import SimpleNamespace                            # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import scoring_audit as P1        # noqa: E402  part 1, in the tree of the tag gate-F

SCRIPT_VERSION = "scoring-audit-p2 2026-10-23"
RULES_VERSION = "ht5-rules-v1"
SEED = 42

Stop = P1.Stop
Lines = P1.Lines
sha256_file = P1.sha256_file
short = P1.short
pct = P1.pct
utc_now = P1.utc_now
stamp = P1.stamp
read_csv = P1.read_csv
write_json = P1.write_json
git = P1.git
tree_clean = P1.tree_clean
MODELS = P1.MODELS

# ---------------------------------------------------------------------------
# 0. Fixed values. Every number is in PROJECT_STATE, DEPENDENCY_MAP or
#    DESIGN_ADDENDUM; none is a result.
# ---------------------------------------------------------------------------

GATE_E_COMMIT = P1.GATE_E_COMMIT                  # 5db2bfff3fd3...
GATE_F_COMMIT = "a5164e4a0b82"                    # tag gate-F (D278, D279)
GATE_F_TAG_OBJECT = "c5de990d524c"
GATE_A_COMMIT = "9ee3eb0"                         # tag gate-A (lexicon frozen)
ASSET12_VERSION = "asset12-2026-10-13"
ANALYSIS_VERSION = "analysis-2026-10-17"
STATS_VERSION = "stats-2026-10-17"
HT5_MODELS = ("gemini", "qwen", "gemma")          # stats.MODELS
HT5_CONDS = ("N-BB", "N-BE", "N-BBs")             # stats.COND_N_OPEN
STRATA = ("C", "english_copy", "sibling")
ROLES = ("AU1", "AU2")
CODE_LETTER = {"AU1": "S", "AU2": "T"}
CHECK_ORDER = ("not_visible", "gold_wrong", "ambiguous", "missing_variant", "model_wrong")
KINDS_NAMING = ("tier1", "tier2")
KINDS_CLOSED = ("class_word",)
EARLIER_ROLES = ("none", "HT-1", "HT-3 N1", "HT-3 N2", "HT-4", "HT-6")
PICTURE_FRAME = "N_f330f487fb"                    # ছবির ফ্রেম (addendum I5)
PAINTING = "N_f13caeae85"                         # চিত্রকর্ম
LANG_NAME = {"bn": "Bangla", "en": "English"}
FRAME_COLUMNS = ["model_key", "qa_id", "house_id", "category", "condition_id"]
BUDGET_CATS = ("A1", "A2", "A3", "A4", "B", "C")
NO_ANSWER = "[no answer]"
GATE_F_PATHS = P1.GATE_F_PATHS
ZIP_TIME = (2026, 1, 1, 0, 0, 0)
ID_RE = re.compile(r"H\d{2}_\d{4}")
HOUSE_RE = re.compile(r"(?<![A-Za-z0-9])H\d{2}(?![0-9])")
MODEL_RE = re.compile(r"gemini|qwen|gemma", re.I)
COND_RE = re.compile(r"(?<![A-Za-z])[LNR]-(?:BB|BE|EB|EE|FCB|FCE)(?:s|0)?(?![A-Za-z])")

REAL = {
    "git_checks": True,
    "stats_world": True,
    "strata_n": {"C": 140, "english_copy": 30, "sibling": 30},          # Design 10
    # addendum O2: touched and scored wrong per B7 row; 14 answers in all
    "d192_expect": {"d192_1": 2, "d192_2": 0, "d192_3": 0, "d192_4": 2, "d192_5": 0,
                    "d192_6": 6, "d192_7": 4},
    "d192_distinct": 14,
    "ht3": {"n_drawn": 200, "n_counted": 199, "excluded": ["H54_0006_q1"], "rows": 400,
            "lexicon_writer": "N2", "primary_role": "N1"},
    "images_manifest_sha": "e726f6f3b10f2b57f75a2001bde98938321010b146e1502765d5076fa199738d",
    "images_manifest_rows": 4406,
    "d279_rows": {"a": 294, "b": 10},                                     # addendum O2
    "n_boot": 10000,
    "excluded_rows": P1.REAL["excluded_rows"],
    "lexicon": P1.REAL["lexicon"],
    "vocab": P1.REAL["vocab"],
    "gate_e_pins": P1.REAL["gate_e_pins"],
    "drive_pins": P1.REAL["drive_pins"],
    "d192_anchors": P1.REAL["d192_anchors"],
    "ht6_pin": "9b2f2a8569b9",
}

SHEET_COLS = ("code", "picture", "row_type", "asked_in", "answer_requested_in", "question",
              "gold_answer", "gold_names", "siblings", "model_answer", "verdict",
              "missing_form", "form_kind", "note")
CONTENT_COLS = ("row_type", "asked_in", "answer_requested_in", "question", "gold_answer",
                "gold_names", "siblings", "model_answer")
CONS_COLS = ("code", "picture") + CONTENT_COLS + (
    "au1_verdict", "au1_form", "au1_kind", "au1_note", "au2_verdict", "au2_form", "au2_kind",
    "au2_note", "consensus_verdict", "consensus_form", "consensus_kind", "note")
KEY_COLS = ("row_id", "code_au1", "code_au2", "key", "model_key", "condition_id", "qa_id",
            "house_id", "image_file", "image_sha256", "row_type", "requested_lang",
            "in_sample", "stratum", "pools", "d192_findings", "ht3_reason", "scorer_label",
            "strict", "named_ids", "gold_name_id", "gold_classes")
VERDICT_COLS = ("row_id", "in_sample", "stratum", "pools", "model_key", "condition_id", "qa_id",
                "house_id", "row_type", "scorer_label", "au1_verdict", "au2_verdict",
                "settled_at_consensus", "verdict", "form_given")
FORM_COLS = ("row_id", "in_sample", "stratum", "pools", "model_key", "condition_id", "qa_id",
             "row_type", "requested_lang", "gold_name_id", "gold_name_bn", "gold_classes",
             "form", "form_kind", "form_lang", "au1_form", "au2_form", "listed_for_gold",
             "listed_for_other", "namer_used")
WIDTHS = {"code": 8, "picture": 16, "row_type": 10, "asked_in": 9, "answer_requested_in": 11,
          "question": 30, "gold_answer": 22, "gold_names": 40, "siblings": 46,
          "model_answer": 40, "verdict": 16, "missing_form": 18, "form_kind": 11, "note": 30,
          "consensus_verdict": 16, "consensus_form": 18, "consensus_kind": 12}


# ---------------------------------------------------------------------------
# 1. Small helpers
# ---------------------------------------------------------------------------

def say(msg=""):
    print(msg, flush=True)


def write_csv(path, rows, columns):
    P1.write_csv(path, rows, columns)


def write_new(ctx, path, rows, columns):
    """Write a CSV once. An identical rerun changes nothing; a different file
    is never written over (D286)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".new")
    write_csv(ctx.guard(tmp), rows, columns)
    if path.exists():
        same = sha256_file(path) == sha256_file(tmp)
        tmp.unlink()
        if not same:
            raise Stop("%s exists and differs; it is never replaced" % path.name)
        return False
    tmp.rename(path)
    return True


def seeded_rank(*parts):
    return P1.seeded_rank("ht5", *parts)


def run_streamed(argv, cwd, log_path):
    """Run one command of another script, stream its printout and keep a log."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.Popen([str(a) for a in argv], cwd=str(cwd), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, env=env)
    out = []
    for line in p.stdout:
        print(line, end="", flush=True)
        out.append(line)
    rc = p.wait()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("".join(out), encoding="utf-8")
    return rc, out


def checkout_state(path, want):
    rc, head, _e = git(path, "rev-parse", "HEAD")
    detached = git(path, "symbolic-ref", "-q", "HEAD")[0] != 0
    clean, dirt = tree_clean(path)
    ok = rc == 0 and head.startswith(want) and detached and clean
    return ok, "HEAD %s, %s, %s" % (head[:12] or "-", "detached" if detached else "ON A BRANCH",
                                   "clean" if clean else "NOT clean: %s"
                                   % dirt.replace("\n", "; ")[:200])


def tag_message(repo, tag):
    return git(repo, "for-each-ref", "--format=%(contents)", "refs/tags/%s" % tag)[1]


def import_scorer(gate_e):
    """scoring and normalise from the gate-E checkout (D271, D283)."""
    src = str((Path(gate_e) / "src").resolve())
    if src in sys.path:
        sys.path.remove(src)
    sys.path.insert(0, src)
    for name in ("scoring", "normalise", "stats"):
        mod = sys.modules.get(name)
        if mod is not None and not str(Path(mod.__file__).resolve()).startswith(src):
            raise Stop("module %s is already loaded from %s, not from the gate-E checkout"
                       % (name, mod.__file__))
    import scoring as S          # noqa: E402
    import normalise as N        # noqa: E402
    for mod in (S, N):
        if not str(Path(mod.__file__).resolve()).startswith(src):
            raise Stop("%s was imported from %s, not from %s" % (mod.__name__, mod.__file__, src))
    return S, N


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 2. Context
# ---------------------------------------------------------------------------

class Ctx2:
    def __init__(self, root, gate_e, gate_f, main, cfg):
        self.root = Path(root)
        self.gate_e = Path(gate_e).resolve()
        self.gate_f = Path(gate_f).resolve() if gate_f else None
        self.main = Path(main).resolve() if main else None
        self.cfg = cfg
        self.S, self.N = import_scorer(self.gate_e)
        self.normalise = self.N.normalise
        self._lex = self._voc = self._ex = self._items = self._ST = None
        self._a12 = None
        self.mini = None                  # selftest only: (world, recs, budgets) stand-ins

    # ---- paths ------------------------------------------------------------------
    @property
    def scored_dir(self):
        return self.root / "scored" / "main"

    def scored_path(self, model):
        return self.scored_dir / ("%s_scored.jsonl" % model)

    @property
    def lexicon_path(self):
        return self.gate_e / "lexicon" / "lexicon.csv"

    @property
    def vocab_path(self):
        return self.gate_e / "lexicon" / "closed_vocab.csv"

    @property
    def excluded_path(self):
        return self.cfg.get("excluded_path") or (self.gate_e / "probes" / "excluded_items.csv")

    def guard(self, path):
        """Never write inside a Git tree (setup file Part 7, D271)."""
        p = Path(path).resolve()
        for tree in (self.gate_e, self.gate_f, self.main):
            if tree is None:
                continue
            t = Path(tree).resolve()
            if p == t or t in p.parents:
                raise Stop("%s is inside the Git tree %s. Data-derived files stay in Drive."
                           % (p, t))
        return p

    # ---- frozen inputs and data -------------------------------------------------
    def lexicon(self):
        if self._lex is None:
            self._lex = self.S.load_lexicon(self.lexicon_path)
        return self._lex

    def vocab(self):
        if self._voc is None:
            self._voc = self.S.load_closed_vocab(self.vocab_path)
        return self._voc

    def excluded(self):
        if self._ex is None:
            self._ex = self.S.load_excluded(self.excluded_path)
        return self._ex

    def items(self):
        """qa_id -> probe record, from the five Drive probe files."""
        if self._items is None:
            out = {}
            for name in P1.PROBE_ORDER:
                path = self.root / "probes" / ("%s.json" % name)
                if not path.exists():
                    continue
                for it in self.S.load_probe(path):
                    old = out.get(it["qa_id"])
                    if old is not None and (old["answer_bn"] != it["answer_bn"]
                                            or old["question_bn"] != it["question_bn"]):
                        raise Stop("%s differs between two probe files" % it["qa_id"])
                    out.setdefault(it["qa_id"], it)
            self._items = out
        return self._items

    def stats(self):
        """src/stats.py of the gate-E checkout, never a copy (D230, D271)."""
        if self._ST is None:
            src = str((self.gate_e / "src").resolve())
            import stats as ST                                   # noqa: E402
            if not str(Path(ST.__file__).resolve()).startswith(src):
                raise Stop("stats was imported from %s, not from the gate-E checkout" % ST.__file__)
            self._ST = ST
        return self._ST

    def asset12(self):
        """jpeg_metadata_blocks and friends of scripts/asset12_nameref.py (main)."""
        if self._a12 is None:
            self._a12 = load_module(HERE / "asset12_nameref.py", "asset12_nameref_p2")
        return self._a12

    # ---- records ----------------------------------------------------------------
    def provenance(self):
        d = {"script_version": SCRIPT_VERSION, "script_sha256": sha256_file(__file__),
             "part1_sha256": sha256_file(P1.__file__), "utc": utc_now(),
             "gate_e": str(self.gate_e), "gate_f": str(self.gate_f), "root": str(self.root),
             "scorer_sha256": sha256_file(self.S.__file__),
             "normalise_sha256": sha256_file(self.N.__file__)}
        if self.cfg.get("git_checks"):
            d["gate_e_head"] = git(self.gate_e, "rev-parse", "HEAD")[1]
            if self.gate_f:
                d["gate_f_head"] = git(self.gate_f, "rev-parse", "HEAD")[1]
            if self.main:
                d["main_head"] = git(self.main, "rev-parse", "HEAD")[1]
        return d

    def log_run(self, entry):
        path = self.guard(ht5_paths(self)["records"] / "runs.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(self.provenance(), **entry), ensure_ascii=False,
                                default=str) + "\n")


def ht5_paths(ctx):
    d = ctx.root / "human" / "HT-5"
    st = ctx.root / "results" / "stats_inputs"
    return {
        "dir": d, "guide": d / "guide" / "HT5_auditor_guide.md",
        "sens_guide": d / "guide" / "HT5_sensitivity_instructions.md",
        "to_send": d / "to_send", "key_dir": d / "key", "received": d / "received",
        "records": d / "_records", "reading_dir": d / "reading",
        "key": d / "key" / "ht5_key.csv", "draw": d / "key" / "ht5_draw.json",
        "rules": d / "to_send" / "HT5_rules.txt",
        "zip": {r: d / "to_send" / ("HT5_%s.zip" % r) for r in ROLES},
        "blank": {r: d / "to_send" / ("HT5_sheet_%s_blank.xlsx" % r) for r in ROLES},
        "returned": {r: d / "received" / ("HT5_sheet_%s_returned.xlsx" % r) for r in ROLES},
        "cons_zip": d / "to_send" / "HT5_consensus.zip",
        "cons_blank": d / "to_send" / "HT5_consensus_blank.xlsx",
        "cons_returned": d / "received" / "HT5_consensus_returned.xlsx",
        "receipts": d / "received" / "receipts.jsonl",
        "audit": d / "ht5_audit.csv", "verdicts": d / "ht5_verdicts.csv",
        "forms": d / "TEXT_ht5_forms.csv", "notes": d / "TEXT_ht5_notes.csv",
        "reading": d / "reading" / "ht5_reading.json",
        "sens": d / "lexicon_sensitivity.csv",
        "sens_template": d / "lexicon_sensitivity_TEMPLATE.csv",
        "sens_sheet": d / "HT5_sensitivity_worksheet.xlsx",
        "sens_check": d / "reading" / "sens_check.json",
        "sens_log": d / "_records" / "sens_checks.jsonl",
        "frame": st / "error_budget_items.csv",
        "frame_manifest": st / "error_budget_items_manifest.json",
        "ht3_dir": ctx.root / "human" / "HT-3",
        "ht3_check_dir": ctx.root / "human" / "HT-3" / "final_check",
        "images": ctx.root / "data" / "images_1024",
        "d279": ctx.scored_dir / "sensitivity_D279",
        "swaps": ctx.root / "human" / "HT-6" / "ht6_swaps.csv",
    }


def real_cfg():
    return dict(REAL)


# ---------------------------------------------------------------------------
# 3. The statistics reader (gate-E stats.py), as the Statistics chat runs it
# ---------------------------------------------------------------------------

def stats_inputs(ctx, echo=False, need_recs=True):
    """(ST, checks, world, recs, budgets). Real mode: build_world, load_scored and
    error_budget of stats.py, exactly as analysis.py run does. Selftest: stand-ins
    that carry the same fields."""
    ST = ctx.stats()
    if not ctx.cfg.get("stats_world"):
        world, recs, budgets = ctx.mini
        return ST, ST.Checks(echo=echo), world, recs, budgets
    checks = ST.Checks(echo=echo)
    paths = ST.Paths.make(ctx.gate_e, ctx.root)
    world = ST.build_world(paths, checks, need_dataset=False)
    if not need_recs:
        return ST, checks, world, None, None
    recs = ST.load_scored([str(ctx.scored_dir / "*_scored.jsonl")], world, checks, "real")
    boot = ST.HouseBoot(world.houses_of(world.scored_ids["P-NAME"]), n_boot=2, name="categories")
    budgets = {m: ST.error_budget(world, recs, m, boot) for m in ST.MODELS}
    return ST, checks, world, recs, budgets


def fails_of(checks):
    return [r for r in checks.rows if r[0] == "FAIL"]


# ---------------------------------------------------------------------------
# 4. check (Cell 0)
# ---------------------------------------------------------------------------

def ht3_final_dirs(ctx):
    d = ht5_paths(ctx)["ht3_dir"]
    return sorted(p for p in d.glob("score_final_*") if p.is_dir()) if d.is_dir() else []


def cmd_check(ctx, args):
    say("Scoring and audit, part 2: input check, %s, run %s" % (SCRIPT_VERSION, utc_now()))
    say("Drive %s; main %s; gate-E checkout %s; gate-F checkout %s"
        % (ctx.root, ctx.main, ctx.gate_e, ctx.gate_f))
    L = Lines()
    cfg = ctx.cfg
    P = ht5_paths(ctx)
    n = [0]

    def code():
        n[0] += 1
        return "C%02d" % n[0]

    # --- Git: clone, tags, the two checkouts, this script ---------------------
    if cfg.get("git_checks"):
        try:
            rc, url, _ = git(ctx.main, "remote", "get-url", "origin")
            url_ok = rc == 0 and re.search(r"/banglaindoorvqa-main(\.git)?$", url) is not None
            clean, dirt = tree_clean(ctx.main)
            head = git(ctx.main, "rev-parse", "HEAD")[1]
            L.add(code(), url_ok and clean, "main clone: origin %s, HEAD %s, %s" % (
                "banglaindoorvqa-main" if url_ok else "NOT banglaindoorvqa-main", head[:12],
                "clean" if clean else "NOT clean: %s" % dirt.replace("\n", "; ")[:200]))
            ka, ta = git(ctx.main, "cat-file", "-t", "gate-A")[1], git(ctx.main, "rev-parse",
                                                                      "gate-A^{commit}")[1]
            ke, te = git(ctx.main, "cat-file", "-t", "gate-E")[1], git(ctx.main, "rev-parse",
                                                                      "gate-E^{commit}")[1]
            kf, tf = git(ctx.main, "cat-file", "-t", "gate-F")[1], git(ctx.main, "rev-parse",
                                                                      "gate-F^{commit}")[1]
            of = git(ctx.main, "rev-parse", "refs/tags/gate-F")[1]
            ok = (ta.startswith(GATE_A_COMMIT) and ke == "tag" and te == GATE_E_COMMIT
                  and kf == "tag" and tf.startswith(GATE_F_COMMIT) and of.startswith(GATE_F_TAG_OBJECT))
            L.add(code(), ok, "tags: gate-A on %s (%s), gate-E %s on %s, gate-F %s on %s, tag object %s "
                  "(expected gate-A %s, gate-E annotated on %s, gate-F annotated on %s, object %s)"
                  % (ta[:7] or "missing", ka or "-", ke or "missing", te[:12], kf or "missing", tf[:12],
                     of[:12], GATE_A_COMMIT, GATE_E_COMMIT[:12], GATE_F_COMMIT, GATE_F_TAG_OBJECT))
            ok, txt = checkout_state(ctx.gate_e, GATE_E_COMMIT)
            L.add(code(), ok, "gate-E checkout %s: %s" % (ctx.gate_e, txt))
            ok, txt = checkout_state(ctx.gate_f, GATE_F_COMMIT)
            L.add(code(), ok, "gate-F checkout %s: %s" % (ctx.gate_f, txt))
            d1 = git(ctx.main, "diff", "--name-only", "gate-E", "gate-F", "--", *GATE_F_PATHS)
            d2 = git(ctx.main, "diff", "--name-only", "gate-A", "gate-E", "--",
                     "lexicon/lexicon.csv", "lexicon/closed_vocab.csv")
            ok = d1[0] == 0 and not d1[1].strip() and d2[0] == 0 and not d2[1].strip()
            L.add(code(), ok, "gate-E and gate-F hold the same scorer, normalise, fc_options, "
                  "asset9_scorer.py, lexicon, excluded list and conditions.yaml; the lexicon and "
                  "vocabulary equal gate-A%s" % ("" if ok else "; CHANGED: %s %s" % (
                      d1[1].replace("\n", ", "), d2[1].replace("\n", ", "))))
            d3 = git(ctx.main, "diff", "--name-only", "gate-E", "gate-F", "--", "src/stats.py",
                     "scripts/analysis.py")[1].strip()
            L.info(code(), "src/stats.py and scripts/analysis.py at gate-F: %s (budget-items and the "
                   "readers run from the gate-E checkout either way, D283)"
                   % ("same as gate-E" if not d3 else "changed: %s" % d3.replace("\n", ", ")))
            me = Path(__file__).resolve()
            bad = []
            for f in (me, Path(P1.__file__).resolve(), HERE / "asset12_nameref.py"):
                inside = ctx.main.resolve() in f.parents
                rel = f.relative_to(ctx.main.resolve()).as_posix() if inside else str(f)
                tracked = inside and git(ctx.main, "ls-files", "--error-unmatch", rel)[0] == 0
                unchanged = tracked and git(ctx.main, "diff", "--quiet", "HEAD", "--", rel)[0] == 0
                if not unchanged:
                    bad.append(rel)
            p1_same = git(ctx.main, "diff", "--name-only", "gate-F", "HEAD", "--",
                          "scripts/scoring_audit.py", "scripts/asset12_nameref.py")[1].strip()
            L.add(code(), not bad and not p1_same,
                  "this script (%s, sha %s), scripts/scoring_audit.py and scripts/asset12_nameref.py "
                  "committed on main with no local change; the last two equal gate-F%s"
                  % (SCRIPT_VERSION, short(sha256_file(me)),
                     "" if not bad and not p1_same else "; NOT: %s %s" % (", ".join(bad), p1_same)))
        except Exception as exc:                                  # report, never hide
            L.add(code(), False, "Git checks: %s: %s" % (type(exc).__name__, exc))
    else:
        L.info(code(), "Git checks skipped (selftest)")

    # --- gate-E scorer, lexicon, vocabulary, excluded list ----------------------
    S = ctx.S
    try:
        bad = []
        for rel in ("src/scoring.py", "lexicon/lexicon.csv", "lexicon/closed_vocab.csv",
                    "probes/excluded_items.csv"):
            pin = cfg["gate_e_pins"].get(rel)
            s = sha256_file(ctx.gate_e / rel)
            if not P1.pin_ok(s, pin):
                bad.append("%s %s (expected %s)" % (rel, short(s), pin[:12]))
        L.add(code(), not bad and S.SCORER_VERSION == P1.SCORER_VERSION,
              "gate-E scorer: scoring.py %s (%s), normalise.py %s, imported from %s%s" % (
                  short(sha256_file(S.__file__)), S.SCORER_VERSION, short(sha256_file(ctx.N.__file__)),
                  Path(S.__file__).parent, "" if not bad else "; DIFFER: " + "; ".join(bad)))
        lex, voc = ctx.lexicon(), ctx.vocab()
        want = cfg["lexicon"]
        L.add(code(), len(lex.objects) == want["objects"] and len(lex.index.entries) == want["terms"]
              and not lex.warnings, "lexicon.csv %s: %s; %d warnings" % (
                  short(sha256_file(ctx.lexicon_path)), lex.describe(), len(lex.warnings)))
        L.add(code(), voc.describe() == cfg["vocab"] and not voc.warnings,
              "closed_vocab.csv %s: %s; %d warnings" % (short(sha256_file(ctx.vocab_path)),
                                                        voc.describe(), len(voc.warnings)))
        ex = ctx.excluded()
        L.add(code(), len(ex) == cfg["excluded_rows"], "excluded_items.csv %s: %d rows" % (
            short(sha256_file(ctx.excluded_path)), len(ex)))
    except Exception as exc:
        L.add(code(), False, "scorer, lexicon or excluded list: %s: %s" % (type(exc).__name__, exc))
        return finish_check(ctx, L)

    # --- stats.py: the world and the six scored files ---------------------------
    try:
        ST = ctx.stats()
        same_cols = (tuple(ST.HT5_COLUMNS) == ("audit_id", "model_key", "condition_id", "qa_id",
                                                "house_id", "stratum", "verdict")
                     and set(ST.HT5_VERDICTS) == set(CHECK_ORDER) and tuple(ST.HT5_STRATA) == STRATA)
        L.add(code(), ST.STATS_VERSION == STATS_VERSION and same_cols,
              "stats.py %s (%s) from the gate-E checkout; D238 HT-5 columns, strata and verdicts as "
              "this script writes them: %s" % (short(sha256_file(ST.__file__)), ST.STATS_VERSION,
                                               same_cols))
    except Exception as exc:
        L.add(code(), False, "stats.py: %s: %s" % (type(exc).__name__, exc))
        return finish_check(ctx, L)
    shas = {}
    for m in MODELS:
        p = ctx.scored_path(m)
        shas[m] = sha256_file(p) if p.exists() else None
    if cfg.get("git_checks"):
        msg = tag_message(ctx.main, "gate-F")
        tagged = dict(re.findall(r"(qwen_ft_s4[234]|qwen|gemma|gemini) ([0-9a-f]{12})",
                                 msg.split("scored files:", 1)[-1])) if "scored files:" in msg else {}
        diff = [m for m in MODELS if not shas[m] or tagged.get(m) != shas[m][:12]]
        L.add(code(), not diff and len(tagged) == len(MODELS),
              "the six scored files in Drive scored/main/ equal the shas in the gate-F tag message%s"
              % ("" if not diff else "; DIFFER or missing: " + ", ".join(diff)))
    else:
        miss = [m for m in MODELS if not shas[m]]
        L.add(code(), not miss, "scored files present: %d of %d" % (len(MODELS) - len(miss), len(MODELS)))
    if cfg.get("stats_world"):
        try:
            _ST, checks, world, recs, budgets = stats_inputs(ctx)
            f = fails_of(checks)
            L.add(code(), not f, "stats.py build_world and load_scored on the Drive inputs and the six "
                  "scored files (as the Statistics chat runs them): %d check lines, %d FAIL; %d "
                  "model-condition sets%s" % (len(checks.rows), len(f), len(recs.data),
                                              "" if not f else "; " + "; ".join(
                                                  "%s: %s" % (x[1], x[2]) for x in f[:3])))
            cpool = sum(1 for m in HT5_MODELS for c in budgets[m]["categories"].values() if c == "C")
            L.info(code(), "error budget from the scored files (stats.error_budget): category C "
                   "pooled over gemini, qwen and gemma: %d answers (Design 10 samples 140)" % cpool)
        except Exception as exc:
            L.add(code(), False, "stats.py reader: %s: %s" % (type(exc).__name__, exc))
    else:
        L.info(code(), "stats.py reader skipped (selftest)")

    # --- HT-3 -----------------------------------------------------------------------
    try:
        if ctx.gate_f is not None and (ctx.gate_f / "scripts" / "asset12_nameref.py").exists():
            a12f = ctx.gate_f / "scripts" / "asset12_nameref.py"
            text = a12f.read_text("utf-8")
            ver = re.search(r'SCRIPT_VERSION = "([^"]+)"', text)
            same = sha256_file(a12f) == sha256_file(HERE / "asset12_nameref.py")
            L.add(code(), bool(ver) and ver.group(1) == ASSET12_VERSION,
                  "asset12_nameref.py in the gate-F checkout: %s, sha %s; %s main's copy"
                  % (ver.group(1) if ver else "?", short(sha256_file(a12f)),
                     "equal to" if same else "DIFFERENT from"))
        elif cfg.get("git_checks"):
            L.add(code(), False, "asset12_nameref.py missing in the gate-F checkout %s" % ctx.gate_f)
        man_p = P["ht3_dir"] / "ht3_manifest.json"
        man = json.loads(man_p.read_text("utf-8"))
        st = man.get("state", {})
        rec = st.get("received", {})
        finals = ht3_final_dirs(ctx)
        ok = "draw" in st and set(rec) == {"N1", "N2"}
        L.add(code(), ok, "HT-3 record %s: draw %s, receipts %s; final score folders: %s" % (
            man_p.name, "yes" if "draw" in st else "NO", ", ".join(sorted(rec)) or "none",
            "none yet (Cell 2)" if not finals else "%d, newest %s" % (len(finals), finals[-1].name)))
    except Exception as exc:
        L.add(code(), False, "HT-3 record: %s: %s" % (type(exc).__name__, exc))

    # --- HT-6, D192, D279, images, English questions ----------------------------------
    try:
        s = sha256_file(P["swaps"])
        rows = read_csv(P["swaps"])
        pairs = {(r["qa_id"], r["removed_id"]) for r in rows if r.get("reason") == "name"}
        pin = cfg.get("ht6_pin")
        L.add(code(), P1.pin_ok(s, pin) and (pin is None or (len(rows) == 53 and len(pairs) == 48)),
              "human/HT-6/ht6_swaps.csv %s: %d rows, %d item and sibling pairs with reason name (D204)"
              % (short(s), len(rows), len(pairs)))
    except Exception as exc:
        L.add(code(), False, "ht6_swaps.csv: %s: %s" % (type(exc).__name__, exc))
    try:
        d192 = P1.D192Rules(ctx)
        L.add(code(), True, "D192 inputs: extra_words.csv %s (%d rows), frozen_file_notes.csv %s (%d "
              "genitive-only rows, equal to the scorer's); anchors %s" % (
                  short(d192.sha_extra), d192.n_extra, short(d192.sha_notes), len(d192.gen),
                  ", ".join(cfg["d192_anchors"].values())))
    except Exception as exc:
        L.add(code(), False, "D192 inputs: %s: %s" % (type(exc).__name__, exc))
    try:
        ch = read_csv(P["d279"] / "changes.csv")
        sm = json.loads((P["d279"] / "summary.json").read_text("utf-8"))
        by = Counter(r["rule"] for r in ch)
        want = cfg.get("d279_rows")
        same = sm.get("scored_sha256") == shas
        L.add(code(), same and (want is None or dict(by) == want),
              "scored/main/sensitivity_D279/: changes.csv %d rows (rule a %d, rule b %d); summary.json "
              "%s the scored files" % (len(ch), by.get("a", 0), by.get("b", 0),
                                       "on" if same else "NOT on"))
    except Exception as exc:
        L.add(code(), False, "D279 change file: %s: %s" % (type(exc).__name__, exc))
    try:
        mp = P["images"] / "images_manifest.csv"
        s = sha256_file(mp)
        rows = read_csv(mp)
        want = cfg.get("images_manifest_sha")
        L.add(code(), (want is None or s == want) and (cfg.get("images_manifest_rows") is None
                                                       or len(rows) == cfg["images_manifest_rows"])
              and {"image_file", "sha256"} <= set(rows[0] if rows else {}),
              "data/images_1024/images_manifest.csv %s: %d rows (the pictures the models saw)"
              % (short(s), len(rows)))
    except Exception as exc:
        L.add(code(), False, "images manifest: %s: %s" % (type(exc).__name__, exc))
    try:
        _m, s = P1.english_questions(ctx)
        L.add(code(), True, "english/english_questions_final.csv %s: %d questions" % (short(s), len(_m)))
    except Exception as exc:
        L.add(code(), False, "English questions: %s: %s" % (type(exc).__name__, exc))

    # --- this chat's own files so far ----------------------------------------------------
    if P["frame"].exists():
        _rows, _man, prob = read_frame(ctx)
        L.add(code(), not prob, "results/stats_inputs/error_budget_items.csv: %d rows%s" % (
            len(_rows), "" if not prob else "; " + "; ".join(prob)))
    else:
        L.info(code(), "results/stats_inputs/error_budget_items.csv: not yet (Cell 3)")
    g = P["guide"]
    if g.exists():
        L.add(code(), True, "human/HT-5/guide/HT5_auditor_guide.md %s (in every package)"
              % short(sha256_file(g)))
    else:
        L.info(code(), "human/HT-5/guide/HT5_auditor_guide.md: not yet. Upload it before Cell 4")
    got = sorted({r["who"] for r in P1.receipts(P)}) if P["receipts"].exists() else []
    L.info(code(), "HT-5: draw %s; receipts %s; ht5_audit.csv %s; lexicon_sensitivity.csv %s" % (
        "yes (key %s)" % short(sha256_file(P["key"])) if P["key"].exists() else "not yet",
        ", ".join(got) or "none", "written" if P["audit"].exists() else "not yet",
        "present" if P["sens"].exists() else "not yet"))
    try:
        import openpyxl
        import PIL
        L.add(code(), True, "packages: openpyxl %s, Pillow %s" % (openpyxl.__version__, PIL.__version__))
    except Exception as exc:
        L.add(code(), False, "openpyxl and Pillow: %s" % exc)
    return finish_check(ctx, L)


def finish_check(ctx, L):
    fails = L.failed()
    say("CHECK: %s" % ("ALL PASS (%d lines)" % len(L.rows) if not fails else
                       "%d FAIL of %d lines" % (len(fails), len(L.rows))))
    try:
        path = ctx.guard(ht5_paths(ctx)["records"] / ("check_%s.json" % stamp()))
        write_json(path, {"command": "check", "provenance": ctx.provenance(), "lines": L.rows})
        say("Wrote %s" % path)
    except Exception as exc:
        say("Could not write the check record: %s" % exc)
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# 5. HT-3 final scores (Cell 2): asset12 from the gate-F checkout, then the check
# ---------------------------------------------------------------------------

def cmd_ht3_final(ctx, args):
    P = ht5_paths(ctx)
    say("Scoring and audit, part 2: HT-3 final scores (D208, D283), %s, run %s"
        % (SCRIPT_VERSION, utc_now()))
    if ctx.cfg.get("git_checks"):
        ok, txt = checkout_state(ctx.gate_f, GATE_F_COMMIT)
        say("%s gate-F checkout %s: %s" % ("PASS" if ok else "FAIL", ctx.gate_f, txt))
        if not ok:
            raise Stop("the HT-3 final scores run only from a clean, detached checkout of gate-F")
    finals = ht3_final_dirs(ctx)
    again = (args.again or "").strip()
    if finals and not again:
        say("INFO  final scores exist already (%s); not scored again. A second run needs "
            "--again \"<reason>\" and a decision row." % ", ".join(p.name for p in finals))
    else:
        if finals:
            say("Scoring again: %s" % again)
        script = ctx.gate_f / "scripts" / "asset12_nameref.py"
        base = [sys.executable, "-u", script]
        common = ["--repo", ctx.gate_f, "--drive", ctx.root]
        logs = P["ht3_check_dir"] / "logs"
        rc, _o = run_streamed(base + ["check-inputs"] + common, ctx.gate_f,
                              ctx.guard(logs / ("check_inputs_%s.log" % stamp())))
        if rc != 0:
            raise Stop("asset12 check-inputs ended with exit %d; nothing scored. Send the printout."
                       % rc)
        rc, _o = run_streamed(base + ["score", "--final"] + common, ctx.gate_f,
                              ctx.guard(logs / ("score_final_%s.log" % stamp())))
        if rc != 0:
            raise Stop("asset12 score --final ended with exit %d. Send the printout." % rc)
        if ctx.cfg.get("git_checks"):
            ok, txt = checkout_state(ctx.gate_f, GATE_F_COMMIT)
            say("%s gate-F checkout after the run: %s" % ("PASS" if ok else "FAIL", txt))
            if not ok:
                raise Stop("the gate-F checkout changed during the run")
        ctx.log_run({"command": "ht3-final", "again": again or None})
    return cmd_ht3_check(ctx, args)


def ht3_lookfirst(ctx, folder):
    """The D280 items: both namers named the same other object (D212), and items
    whose gold is the picture frame that a namer named as a painting (I5). Ids only."""
    same = read_csv(folder / "ht3_same_other_object.csv")
    rows = read_csv(folder / "ht3_scored_names.csv")
    lex, voc = ctx.lexicon(), ctx.vocab()
    items = ctx.items()
    out = defaultdict(list)
    for r in same:
        out[r["qa_id"]].append("same_other_object")
    for r in rows:
        q = r["qa_id"]
        if r.get("excluded") == "yes" or q not in items:
            continue
        g = ctx.S.resolve_gold(items[q], lex, voc)
        if g.name_id == PICTURE_FRAME and PAINTING in (r.get("named_ids") or "").split("|"):
            if "frame_as_painting" not in out[q]:
                out[q].append("frame_as_painting")
    return dict(out), rows


def cmd_ht3_check(ctx, args):
    P = ht5_paths(ctx)
    cfg = ctx.cfg["ht3"]
    say("Scoring and audit, part 2: HT-3 final check, %s, run %s" % (SCRIPT_VERSION, utc_now()))
    finals = ht3_final_dirs(ctx)
    folder = Path(args.folder) if getattr(args, "folder", None) else (finals[-1] if finals else None)
    if folder is None:
        raise Stop("no human/HT-3/score_final_* folder yet; run Cell 2")
    say("Folder: %s" % folder)
    L = Lines()
    names = ("ht3_agreement.json", "ht3_scored_names.csv", "ht3_same_other_object.csv", "ht3_report.txt")
    miss = [x for x in names if not (folder / x).exists()]
    L.add("T01", not miss, "files: %s" % ("all four present" if not miss else "MISSING " + ", ".join(miss)))
    if miss:
        return finish_ht3(ctx, L, folder, {})
    js = json.loads((folder / "ht3_agreement.json").read_text("utf-8"))
    rows = read_csv(folder / "ht3_scored_names.csv")
    summ = js.get("summary", {})
    sc_sha = sha256_file(ctx.gate_e / "src" / "scoring.py")
    no_sha = sha256_file(ctx.gate_e / "src" / "normalise.py")
    f_same = True
    if ctx.gate_f is not None and (ctx.gate_f / "src" / "scoring.py").exists():
        f_same = (sha256_file(ctx.gate_f / "src" / "scoring.py") == sc_sha
                  and sha256_file(ctx.gate_f / "src" / "normalise.py") == no_sha)
    ok = (js.get("mode") == "final" and js.get("reportable") is True
          and js.get("scorer_version") == ctx.S.SCORER_VERSION and js.get("scorer_sha256") == sc_sha
          and js.get("normalise_sha256") == no_sha and f_same)
    L.add("T02", ok, "mode %s, reportable %s, scorer %s (sha %s), normalise sha %s; equal to the "
          "gate-E and gate-F files: %s" % (js.get("mode"), js.get("reportable"), js.get("scorer_version"),
                                           short(js.get("scorer_sha256")), short(js.get("normalise_sha256")),
                                           js.get("scorer_sha256") == sc_sha and f_same))
    pins = ctx.cfg["gate_e_pins"]
    ok = (P1.pin_ok(js.get("lexicon_sha256", ""), pins.get("lexicon/lexicon.csv"))
          and P1.pin_ok(js.get("vocab_sha256", ""), pins.get("lexicon/closed_vocab.csv"))
          and P1.pin_ok(js.get("excluded_sha256", ""), pins.get("probes/excluded_items.csv"))
          and js.get("excluded_rows") == ctx.cfg["excluded_rows"]
          and js.get("lexicon_sha256") == sha256_file(ctx.lexicon_path))
    L.add("T03", ok, "lexicon %s, vocabulary %s (gate-A), excluded list %s with %s rows" % (
        short(js.get("lexicon_sha256")), short(js.get("vocab_sha256")), short(js.get("excluded_sha256")),
        js.get("excluded_rows")))
    ids = js.get("identity_checks") or []
    ok = len(ids) == 4 and all(x.get("ok") for x in ids)
    L.add("T04", ok, "identity checks (D208): %s" % "; ".join(
        "%s %s %s" % (x.get("tag"), x.get("file"), x.get("detail")) for x in ids))
    if ctx.cfg.get("git_checks"):
        ok = (str(js.get("git_head", "")).startswith(GATE_F_COMMIT)
              and str(js.get("gate_f_commit", "")).startswith(GATE_F_COMMIT))
        L.add("T05", ok, "run from the gate-F checkout: git HEAD %s, gate-F commit %s" % (
            js.get("git_head"), str(js.get("gate_f_commit"))[:12]))
    else:
        L.info("T05", "git HEAD not checked (selftest)")
    try:
        man = json.loads((P["ht3_dir"] / "ht3_manifest.json").read_text("utf-8"))
        st = man.get("state", {})
        rec = {r: v.get("sha256") for r, v in st.get("received", {}).items()}
        listed = [s for s in st.get("scores", []) if s.get("folder") == folder.name]
        ok = (st.get("draw", {}).get("key_sha256") == js.get("key_sha256")
              and rec == js.get("received_sha256") and len(listed) == 1 and listed[0].get("mode") == "final")
        L.add("T06", ok, "key and both received sheets as in ht3_manifest.json; the folder is listed "
              "there as a final score: %s" % ok)
    except Exception as exc:
        L.add("T06", False, "ht3_manifest.json: %s: %s" % (type(exc).__name__, exc))
    by_role = Counter(r["role"] for r in rows)
    ok = (summ.get("n_drawn") == cfg["n_drawn"] and summ.get("n_counted") == cfg["n_counted"]
          and sorted(summ.get("excluded_qa_ids") or []) == sorted(cfg["excluded"])
          and not summ.get("unscoreable_qa_ids") and len(rows) == cfg["rows"]
          and by_role == Counter({"N1": cfg["rows"] // 2, "N2": cfg["rows"] // 2})
          and js.get("primary_role") == cfg["primary_role"]
          and js.get("lexicon_writer") == cfg["lexicon_writer"])
    L.add("T07", ok, "items drawn %s, counted %s, excluded at scoring %s (%s), unscoreable %d; name rows "
          "%d (N1 %d, N2 %d); primary role %s, lexicon writer %s (D214)" % (
              summ.get("n_drawn"), summ.get("n_counted"), summ.get("n_excluded"),
              ", ".join(summ.get("excluded_qa_ids") or []) or "none",
              len(summ.get("unscoreable_qa_ids") or []), len(rows), by_role.get("N1", 0),
              by_role.get("N2", 0), js.get("primary_role"), js.get("lexicon_writer")))
    if ctx.cfg.get("stats_world"):
        try:
            ST, checks, world, _r, _b = stats_inputs(ctx, need_recs=False)
            n_before = len(checks.rows)
            f0 = len(fails_of(checks))
            ST.read_ht3(str(folder), world, checks, "real")
            f = fails_of(checks)
            last = checks.rows[n_before:]
            L.add("T08", not f, "stats.read_ht3 (mode real) as the Statistics chat runs it (inputs: %d check "
                  "lines, %d FAIL): %s" % (n_before, f0, "; ".join("%s %s" % (x[0], x[2]) for x in last)[:600]))
        except Exception as exc:
            L.add("T08", False, "stats.read_ht3: %s: %s" % (type(exc).__name__, exc))
    else:
        L.info("T08", "stats.read_ht3 skipped (selftest)")
    look, _rows = ht3_lookfirst(ctx, folder)
    reasons = Counter(x for v in look.values() for x in v)
    L.info("T09", "HT-5 look-first items (D280): both namers named the same other object %d, picture "
           "frame named as a painting %d, together %d items (excluded among them %d)" % (
               reasons.get("same_other_object", 0), reasons.get("frame_as_painting", 0), len(look),
               sum(1 for q in look if q in ctx.excluded())))
    # the 10-item check (setup rule): ten seeded-random counted items, text to Drive
    items = ctx.items()
    by_q = defaultdict(dict)
    for r in rows:
        if r.get("excluded") != "yes":
            by_q[r["qa_id"]][r["role"]] = r
    pick = sorted(by_q, key=lambda q: seeded_rank("ht3-final-10", q))[:10]
    sample = []
    for q in pick:
        d = by_q[q]
        sample.append({"qa_id": q, "gold": items.get(q, {}).get("answer_bn", ""),
                       "N1_name": d.get("N1", {}).get("name"), "N1_label": d.get("N1", {}).get("label"),
                       "N1_strict": d.get("N1", {}).get("strict"),
                       "N1_named_ids": d.get("N1", {}).get("named_ids"),
                       "N2_name": d.get("N2", {}).get("name"), "N2_label": d.get("N2", {}).get("label"),
                       "N2_strict": d.get("N2", {}).get("strict"),
                       "N2_named_ids": d.get("N2", {}).get("named_ids")})
    tp = ctx.guard(P["ht3_check_dir"] / "TEXT_ht3_final_sample10.csv")
    write_csv(tp, sample, list(sample[0]) if sample else ["qa_id"])
    L.info("T10", "the 10-item check: %s (%d items: %s)" % (tp.name, len(sample), ", ".join(pick)))
    extra = {"lookfirst": look, "sample10": pick}
    return finish_ht3(ctx, L, folder, extra)


def finish_ht3(ctx, L, folder, extra):
    P = ht5_paths(ctx)
    ok = not L.failed()
    path = ctx.guard(P["ht3_check_dir"] / ("ht3_final_check_%s.json" % stamp()))
    write_json(path, dict({"command": "ht3-check", "provenance": ctx.provenance(), "folder": str(folder),
                           "pass": ok, "lines": L.rows}, **extra))
    say("Wrote %s" % path)
    ctx.log_run({"command": "ht3-check", "folder": str(folder), "pass": ok})
    say("HT-3 FINAL: %s" % ("PASS. Send ht3_report.txt, ht3_same_other_object.csv and "
                            "TEXT_ht3_final_sample10.csv" if ok else "FAIL. Send this printout"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 6. budget-items (Cell 3): analysis.py from the gate-E checkout, then the frame check
# ---------------------------------------------------------------------------

def read_frame(ctx):
    """error_budget_items.csv and its manifest; (rows, manifest, problems)."""
    P = ht5_paths(ctx)
    prob = []
    with open(P["frame"], encoding="utf-8-sig", newline="") as fh:
        hdr = next(csv.reader(fh), [])
    rows = read_csv(P["frame"])
    if [h.strip() for h in hdr] != FRAME_COLUMNS:
        prob.append("columns %s" % hdr)
    man = {}
    try:
        man = json.loads(P["frame_manifest"].read_text("utf-8"))
    except Exception as exc:
        prob.append("manifest: %s" % exc)
    bad = sum(1 for r in rows if r.get("category") not in BUDGET_CATS or r.get("model_key") not in HT5_MODELS
              or r.get("condition_id") != "N-BB")
    dup = len(rows) - len({(r.get("model_key"), r.get("qa_id")) for r in rows})
    if bad or dup:
        prob.append("%d rows with an unknown category, model or condition, %d repeated" % (bad, dup))
    ex = ctx.excluded()
    if any(r.get("qa_id") in ex for r in rows):
        prob.append("excluded items in the frame")
    if man:
        if man.get("mode") != "real" or man.get("stage") != "main" or man.get("rows") != len(rows):
            prob.append("manifest mode %s, stage %s, rows %s" % (man.get("mode"), man.get("stage"),
                                                                  man.get("rows")))
        if man.get("script_version") != ANALYSIS_VERSION:
            prob.append("analysis.py version %s" % man.get("script_version"))
        files = {Path(f.get("file", "")).name: f.get("sha256") for f in man.get("scored_files", [])}
        now = {"%s_scored.jsonl" % m: (sha256_file(ctx.scored_path(m)) if ctx.scored_path(m).exists()
                                       else None) for m in MODELS}
        if files != now:
            prob.append("the frame was not made from the six scored files as they are now")
        inp = man.get("inputs", {})
        pins = ctx.cfg["gate_e_pins"]
        if not (P1.pin_ok(inp.get("lexicon.csv", ""), pins.get("lexicon/lexicon.csv"))
                and P1.pin_ok(inp.get("excluded_items.csv", ""), pins.get("probes/excluded_items.csv"))
                and inp.get("scoring.py") == sha256_file(ctx.gate_e / "src" / "scoring.py")):
            prob.append("manifest inputs differ from the gate-E files")
    return rows, man, prob


def frame_report(ctx):
    rows, man, prob = read_frame(ctx)
    L = Lines()
    L.add("B01", not prob, "error_budget_items.csv %s: %d rows, mode %s, stage %s, analysis %s, made from "
          "the six scored files as they are now%s" % (
              short(sha256_file(ht5_paths(ctx)["frame"])), len(rows), man.get("mode"), man.get("stage"),
              man.get("script_version"), "" if not prob else "; PROBLEMS: " + "; ".join(prob)))
    per = defaultdict(Counter)
    for r in rows:
        per[r["model_key"]][r["category"]] += 1
    for i, m in enumerate(HT5_MODELS):
        c = per[m]
        L.info("B%02d" % (i + 2), "%-7s not strict Tier 1 in N-BB %4d: %s" % (
            m, sum(c.values()), ", ".join("%s %d" % (k, c[k]) for k in BUDGET_CATS)))
    cpool = sum(per[m]["C"] for m in HT5_MODELS)
    want = ctx.cfg["strata_n"]["C"]
    L.add("B05", cpool > 0, "category C pooled over the three models: %d answers; the draw takes %d%s"
          % (cpool, min(cpool, want), "" if cpool >= want else " (FEWER than %d: all are taken, D284)"
             % want))
    if ctx.cfg.get("stats_world"):
        try:
            _ST, checks, _w, _r, budgets = stats_inputs(ctx)
            mine = {(m, q): c for m in HT5_MODELS for q, c in budgets[m]["categories"].items()}
            file_ = {(r["model_key"], r["qa_id"]): r["category"] for r in rows}
            f = fails_of(checks)
            L.add("B06", mine == file_, "the frame equals stats.error_budget on the scored files read now "
                  "(%d answers in the budget, %d in the file)" % (len(mine), len(file_)))
            L.add("B07", not f, "stats.py reader on the inputs and the scored files: %d check lines, %d FAIL%s"
                  % (len(checks.rows), len(f), "" if not f else "; " + "; ".join(
                      "%s: %s" % (x[1], x[2]) for x in f[:3])))
        except Exception as exc:
            L.add("B06", False, "stats.error_budget: %s: %s" % (type(exc).__name__, exc))
    ok = not L.failed()
    ctx.log_run({"command": "budget-frame", "pass": ok, "rows": len(rows), "c_pool": cpool})
    say("BUDGET FRAME: %s" % ("PASS. Send this printout and the analysis.py lines above it" if ok
                              else "FAIL. Send this printout"))
    return 0 if ok else 1


def cmd_budget_items(ctx, args):
    P = ht5_paths(ctx)
    say("Scoring and audit, part 2: the HT-5 frame (D234, D283), %s, run %s" % (SCRIPT_VERSION, utc_now()))
    if P["key"].exists():
        raise Stop("an HT-5 draw exists; its frame is never rewritten (D284)")
    if ctx.cfg.get("git_checks"):
        ok, txt = checkout_state(ctx.gate_e, GATE_E_COMMIT)
        say("%s gate-E checkout %s: %s" % ("PASS" if ok else "FAIL", ctx.gate_e, txt))
        if not ok:
            raise Stop("budget-items runs only from a clean, detached checkout of gate-E (D283)")
    argv = [sys.executable, "-u", ctx.gate_e / "scripts" / "analysis.py", "budget-items", "--mode", "real",
            "--repo", ctx.gate_e, "--drive", ctx.root, "--scored", ctx.scored_dir / "*_scored.jsonl",
            "--out", P["frame"]]
    rc, _o = run_streamed(argv, ctx.gate_e, ctx.guard(P["records"] / "logs" / ("budget_items_%s.log" % stamp())))
    if rc != 0:
        raise Stop("analysis.py budget-items ended with exit %d. Send the printout." % rc)
    if ctx.cfg.get("git_checks"):
        ok, txt = checkout_state(ctx.gate_e, GATE_E_COMMIT)
        say("%s gate-E checkout after the run: %s" % ("PASS" if ok else "FAIL", txt))
        if not ok:
            raise Stop("the gate-E checkout changed during the run")
    return frame_report(ctx)


# ---------------------------------------------------------------------------
# 7. HT-5 draw (Cell 4)
# ---------------------------------------------------------------------------

SLIM_KEEP = ("key", "model_key", "condition_id", "qa_id", "house_id", "qa_type", "task", "kind",
             "label", "strict", "lenient", "gold_status", "gold_name_id", "named_ids", "gold_value",
             "found_value", "answer_raw", "requested_lang", "content_lang_ok", "mode",
             "reused_from_key", "refusal")


def slim(r, model):
    d = {k: r.get(k) for k in SLIM_KEEP}
    d["model_key"] = model
    return d


def load_answers(ctx):
    """One pass over the six scored files: the HT-5 frames (open naming answers of
    gemini, qwen and gemma in N-BB, N-BE and N-BBs) and the D192 answers."""
    rules = P1.D192Rules(ctx)
    answers, by_mcq, d192, shas = {}, {}, [], {}
    counts = Counter()
    for m in MODELS:
        path = ctx.scored_path(m)
        recs, bad, inc, sha = P1.read_jsonl(path)
        if bad or inc:
            raise Stop("%s: %d unreadable line(s)" % (path.name, bad + inc))
        shas[m] = sha
        for r in recs:
            if r.get("model_key") not in (None, m):
                raise Stop("%s holds a record of model %s" % (path.name, r.get("model_key")))
            if r.get("gold_status") != "ok":
                continue
            if r.get("task") != "fc":
                hits = [(f, s) for f, _t, w, s in rules.apply(r) if w]
                for f, _s in hits:
                    counts[f] += 1
                if hits:
                    d192.append((slim(r, m), sorted({f for f, _s in hits})))
            if m in HT5_MODELS and r.get("condition_id") in HT5_CONDS and r.get("mode") != "reuse":
                s = slim(r, m)
                answers[s["key"]] = s
                by_mcq[(m, s["condition_id"], s["qa_id"])] = s["key"]
        del recs
    return answers, by_mcq, d192, counts, shas


def names_text(obj):
    return "\n".join([
        "Tier 1, Bangla: " + (" | ".join(obj.t1_bn) or "-"),
        "Tier 1, English: " + (" | ".join(obj.t1_en) or "-"),
        "Tier 2 (general), Bangla: " + (" | ".join(obj.t2_bn) or "-"),
        "Tier 2 (general), English: " + (" | ".join(obj.t2_en) or "-")])


def siblings_text(lex, obj):
    sib = sorted((lex.objects[s] for s in lex.siblings_of(obj.name_id)), key=lambda o: o.name_bn)
    if not sib:
        return "none (this object is in no group)"
    return "\n".join("%s: Tier 1 Bangla %s; Tier 1 English %s" % (
        o.name_bn, " | ".join(o.t1_bn) or "-", " | ".join(o.t1_en) or "-") for o in sib)


def classes_text(voc, kind, classes):
    out = ["Gold %s class(es):" % kind]
    for cls in classes or []:
        f = voc.forms.get(kind, {}).get(cls, {})
        out.append("%s: Bangla %s; English %s" % (cls, " | ".join(f.get("bn", [])) or "-",
                                                 " | ".join(f.get("en", [])) or "-"))
    return "\n".join(out)


def row_content(ctx, r, item, en_map):
    S, lex, voc = ctx.S, ctx.lexicon(), ctx.vocab()
    sp = S.condition_spec(r["condition_id"])
    q = item["question_bn"]
    if sp.question_lang == "en":
        qn = ctx.normalise(q)
        if qn not in en_map:
            raise Stop("no English question for %s (asked in English)" % r["qa_id"])
        q = en_map[qn]
    kind = r.get("kind")
    if kind == "naming":
        obj = lex.objects[r["gold_name_id"]]
        gold_names, siblings = names_text(obj), siblings_text(lex, obj)
    else:
        gold_names = classes_text(voc, kind, r.get("gold_value"))
        siblings = "-"
    ans = r.get("answer_raw") or ""
    return {"row_type": kind, "asked_in": LANG_NAME[sp.question_lang],
            "answer_requested_in": LANG_NAME.get(sp.answer_lang or sp.question_lang),
            "question": q, "gold_answer": item["answer_bn"], "gold_names": gold_names,
            "siblings": siblings, "model_answer": ans if ans.strip() else NO_ANSWER}


def kinds_for(row_type):
    return KINDS_NAMING if row_type == "naming" else KINDS_CLOSED


def ht5_rules():
    """The 'Read me' tab (ht5-rules-v1, D285)."""
    return [
        "HT-5 WRITTEN RULES (%s)" % RULES_VERSION,
        "",
        "YOUR TASK",
        "1. Each row of the tab 'Sheet' holds one model answer to a question about a photo of a room. Our "
        "scoring program scored every answer on this sheet as not fully right. Give each row one verdict "
        "that says why.",
        "2. For each row look at the picture (the file named in the column picture, or the page "
        "HT5_view_<your code>.html), the question, the gold answer, gold_names, siblings and the model answer.",
        "3. Work alone. Use no AI tool, no web search, no image search and no other person. Do not open the "
        "Drive folders scored/, results/, human/HT-3/ or human/HT-5/, and do not talk about rows with the "
        "other auditor, until both sheets are returned.",
        "4. Every row needs a verdict. Do not change the code column, the order of the columns or the tab names.",
        "5. Time: about 4 hours. You may stop and go on later.",
        "",
        "THE FIVE CHECKS. Take them in this order. The first one that fits gives the verdict.",
        "6. not_visible: in the picture you cannot tell what the object is (too dark, blurred, cut off, "
        "hidden or too small), or the object the gold names is not in the picture.",
        "7. gold_wrong: you can see the object clearly, and the gold answer is not a right answer for it.",
        "8. ambiguous: the question honestly has more than one right answer, and the model gave another right "
        "one. The question could mean another object in the picture, or the object fits two different names "
        "equally well. For a material or colour row: the object also has a material or colour that the gold "
        "leaves out, and the answer names it.",
        "9. missing_variant: the gold is right, and the model answer gives a right name (or material or colour "
        "word) for the same thing, in the language asked for, but our lists (gold_names and the tabs Lexicon "
        "and Vocabulary) do not have this name or this spelling. A spelling that many people use counts. A "
        "slip of the keyboard does not.",
        "10. model_wrong: everything else. For example: another object; a general name the lists already have; "
        "a description; a refusal; an empty answer; a right name only in the other language; a typo; an "
        "answer that names two different things.",
        "",
        "MISSING_VARIANT",
        "11. Write the name in missing_form as the model wrote it, without Bangla endings such as টি, টা, "
        "গুলো and without extra words.",
        "12. Choose form_kind: tier1 for a specific name of exactly this kind of object; tier2 for a general "
        "name that also covers other kinds of object; class_word for a material or colour word.",
        "13. Leave missing_form and form_kind empty for every other verdict.",
        "14. The name must be in the language of answer_requested_in: Bangla letters for Bangla, Latin letters "
        "for English. An English word written in Bangla letters counts as Bangla if Bangla speakers really "
        "use it that way for this object.",
        "",
        "GOOD TO KNOW",
        "15. gold_names shows the gold's names: Tier 1 (specific) and Tier 2 (general), Bangla and English. "
        "siblings lists similar objects of the same group, each with every Tier 1 name in both languages.",
        "16. The tab Lexicon lists every object we know with its names. The tab Vocabulary lists the "
        "material and colour words.",
        "17. A name or spelling that is already in the lists is never missing_variant.",
        "18. If you are unsure between two verdicts, take the one that comes first in the order above among "
        "those you think fit, and write the other one in note.",
        "19. %s in model_answer means the model wrote nothing." % NO_ANSWER,
    ]


def cons_rules():
    return [
        "HT-5 CONSENSUS (%s)" % RULES_VERSION,
        "",
        "1. The two auditors sit together. This tab lists only the rows where your verdicts differ, or where "
        "you both chose missing_variant but wrote a different name or form_kind. You see both answers and "
        "notes. You never see the scorer's label, the model or the condition.",
        "2. For each row look at the picture again (HT5_consensus_view.html), agree on one verdict under the "
        "written rules (tab 'Read me' of your own sheet) and write it in consensus_verdict. For "
        "missing_variant also agree on consensus_form and consensus_kind; leave both empty otherwise.",
        "3. A cell that is already filled holds what you already agree on. Leave it.",
        "4. Do not change the code column. Use no AI tool and no other person.",
        "5. Time: about 1 hour.",
    ]


def make_workbook(tabs, main_tab, columns, rows, dropdowns, links):
    """xlsx bytes. tabs: [(name, header or None, rows)]; dropdowns:
    {(row_index, column): values}; links: {row_index: target} for the picture."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    wb.remove(wb.active)
    ws = wb.create_sheet(main_tab)
    ws.append(list(columns))
    for c in ws[1]:
        c.font = Font(bold=True)
    for i, col in enumerate(columns):
        ws.column_dimensions[ws.cell(row=1, column=i + 1).column_letter].width = WIDTHS.get(col, 14)
    for row in rows:
        ws.append([row.get(c, "") for c in columns])
    wrap = Alignment(wrap_text=True, vertical="top")
    for line in ws.iter_rows(min_row=2):
        for c in line:
            c.alignment = wrap
            c.number_format = "@"
    if "picture" in columns:
        pc = columns.index("picture") + 1
        for ri, target in links.items():
            cell = ws.cell(row=ri + 2, column=pc)
            cell.hyperlink = target
            cell.font = Font(color="0563C1", underline="single")
    groups = defaultdict(list)
    for (ri, col), values in dropdowns.items():
        groups[(col, tuple(values))].append(ri)
    for (col, values), ris in groups.items():
        dv = DataValidation(type="list", formula1='"%s"' % ",".join(values), allow_blank=True)
        ws.add_data_validation(dv)
        letter = ws.cell(row=1, column=columns.index(col) + 1).column_letter
        for ri in ris:
            dv.add("%s%d" % (letter, ri + 2))
    ws.freeze_panes = "C2"
    for name, header, trows in tabs:
        t = wb.create_sheet(name)
        if header:
            t.append(list(header))
            for c in t[1]:
                c.font = Font(bold=True)
        for tr in trows:
            t.append(list(tr) if isinstance(tr, (list, tuple)) else [tr])
        t.column_dimensions["A"].width = 120 if header is None else 24
        for line in t.iter_rows():
            for c in line:
                c.alignment = wrap
    for k in ("creator", "lastModifiedBy"):
        setattr(wb.properties, k, "HT-5")
    wb.properties.title = "HT-5"
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def workbook_texts(data):
    from openpyxl import load_workbook
    wb = load_workbook(io.BytesIO(data))
    out = []
    for ws in wb.worksheets:
        hdr = [c.value for c in ws[1]] if ws.max_row else []
        out.append(("sheet name", ws.title, False))
        for row in ws.iter_rows():
            for c in row:
                if c.value is None:
                    continue
                col = hdr[c.column - 1] if c.column - 1 < len(hdr) else None
                answer = col in ("model_answer",) and c.row > 1
                out.append(("%s!%s" % (ws.title, c.coordinate), str(c.value), answer))
    return out


def leak_problems(texts):
    """(kind, where) for every id, house, model or condition pattern. A model
    answer is scanned for qa_id patterns only (D286)."""
    out = []
    for where, t, is_answer in texts:
        if ID_RE.search(t):
            out.append(("qa_id or image id", where))
        if is_answer:
            continue
        if HOUSE_RE.search(ID_RE.sub(" ", t)):
            out.append(("house id", where))
        if MODEL_RE.search(t):
            out.append(("model name", where))
        if COND_RE.search(t):
            out.append(("condition id", where))
    return out


def view_html(title, intro, rows, fields, extra_fields=()):
    """A read-only page: each row with its picture and its text (D286)."""
    esc = html.escape
    parts = ["<!doctype html>", '<html lang="bn"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             "<title>%s</title>" % esc(title),
             "<style>body{font-family:'Noto Sans Bengali','Kalpurush','SolaimanLipi',Arial,sans-serif;"
             "margin:16px;color:#222;background:#fff;line-height:1.45}"
             "section{border-top:2px solid #888;padding:12px 0 24px}h1{font-size:1.4em}"
             "h2{font-size:1.2em;margin:4px 0}img{max-width:100%;max-height:70vh;border:1px solid #ccc}"
             "table{border-collapse:collapse;margin-top:8px;width:100%;max-width:900px}"
             "th{text-align:left;vertical-align:top;width:190px;padding:4px 8px;background:#f2f2f2}"
             "td{vertical-align:top;padding:4px 8px;white-space:pre-wrap;font-size:1.08em}"
             "td.a{background:#fff8dc;font-size:1.2em}"
             "nav a{display:inline-block;margin:0 6px 4px 0}</style></head><body>",
             "<h1>%s</h1>" % esc(title)]
    for line in intro:
        parts.append("<p>%s</p>" % esc(line))
    parts.append("<nav>%s</nav>" % " ".join('<a href="#%s">%s</a>' % (esc(r["code"]), esc(r["code"]))
                                            for r in rows))
    for r in rows:
        parts.append('<section id="%s"><h2>%s</h2>' % (esc(r["code"]), esc(r["code"])))
        parts.append('<a href="%s"><img src="%s" alt="picture %s"></a>' % (
            esc(r["picture"]), esc(r["picture"]), esc(r["code"])))
        parts.append("<table>")
        for f, label in list(fields) + list(extra_fields):
            cls = ' class="a"' if f == "model_answer" else ""
            parts.append("<tr><th>%s</th><td%s>%s</td></tr>" % (esc(label), cls, esc(str(r.get(f, "")))))
        parts.append("</table></section>")
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


VIEW_FIELDS = (("row_type", "row type"), ("asked_in", "asked in"),
               ("answer_requested_in", "answer requested in"), ("question", "question"),
               ("gold_answer", "gold answer"), ("gold_names", "gold names"), ("siblings", "siblings"),
               ("model_answer", "model answer"))


def write_zip(path, folder, entries):
    """entries: [(name inside folder, bytes)]; fixed times, no compression."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    names = []
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
        for name, data in entries:
            zi = zipfile.ZipInfo("%s/%s" % (folder, name), date_time=ZIP_TIME)
            zi.external_attr = 0o644 << 16
            z.writestr(zi, data)
            names.append(zi.filename)
    tmp.replace(path)
    return names


def parse_roles(text, which):
    vals = [x.strip() for x in (text or "").split(",") if x.strip()]
    bad = [v for v in vals if v not in EARLIER_ROLES]
    if not vals or bad or ("none" in vals and len(vals) > 1):
        raise Stop("--%s-roles: give the auditor's earlier roles from %s, comma separated (or none); "
                   "got %r" % (which, ", ".join(EARLIER_ROLES), text))
    return vals


def check_images(ctx, rows_by_key, items):
    """Every picture is the copy the models saw: the sha256 of images_manifest.csv,
    no metadata block, the same size (D286)."""
    P = ht5_paths(ctx)
    man = {r["image_file"]: r for r in read_csv(P["images"] / "images_manifest.csv")}
    a12 = ctx.asset12()
    cache = {}
    from PIL import Image
    for k, r in rows_by_key.items():
        f = items[r["qa_id"]]["image_file"]
        if f in cache:
            continue
        m = man.get(f)
        if m is None:
            raise Stop("a picture of the draw is not in images_manifest.csv (%s)" % r["qa_id"])
        p = P["images"] / f
        if not p.exists():
            raise Stop("a picture of the draw is missing in %s (%s)" % (P["images"], r["qa_id"]))
        data = p.read_bytes()
        s = hashlib.sha256(data).hexdigest()
        if s != m["sha256"]:
            raise Stop("a picture differs from images_manifest.csv (%s)" % r["qa_id"])
        meta = a12.jpeg_metadata_blocks(data)
        if meta:
            raise Stop("a picture holds metadata %s (%s); stop and ask" % (meta, r["qa_id"]))
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            size = im.size
        if (m.get("width") and str(size[0]) != str(m["width"])) or \
                (m.get("height") and str(size[1]) != str(m["height"])):
            raise Stop("a picture size differs from images_manifest.csv (%s)" % r["qa_id"])
        cache[f] = (data, s)
    return cache


def cmd_ht5_draw(ctx, args):
    S = ctx.S
    P = ht5_paths(ctx)
    cfg = ctx.cfg
    say("Scoring and audit, part 2: HT-5 draw (D280 to D286), %s, run %s" % (SCRIPT_VERSION, utc_now()))
    if not args.auditors_cleared:
        raise Stop("ht5-draw writes the packages for the auditors. Give --auditors-cleared only after "
                   "both auditors confirmed the rules of D281: two different people, neither of them the "
                   "runner; each works alone; neither opens Drive scored/, results/, human/HT-3/ or "
                   "human/HT-5/ until both sheets are returned.")
    roles = {"AU1": parse_roles(args.au1_roles, "au1"), "AU2": parse_roles(args.au2_roles, "au2")}
    frame, fman, fprob = read_frame(ctx)
    if fprob:
        raise Stop("the HT-5 frame (Cell 3): %s" % "; ".join(fprob))
    finals = ht3_final_dirs(ctx)
    if not finals:
        raise Stop("no HT-3 final score folder yet; run Cell 2 first")
    if not P["guide"].exists():
        raise Stop("upload the auditor guide to %s first" % P["guide"])
    guide = P["guide"].read_bytes()
    items = ctx.items()
    excluded = ctx.excluded()
    lex = ctx.lexicon()

    answers, by_mcq, d192_all, d192_counts, shas = load_answers(ctx)
    say("Read the six scored files: %d open naming answers of %s in %s (copies left out)"
        % (len(answers), ", ".join(HT5_MODELS), ", ".join(HT5_CONDS)))

    # ---- the sample (D284) ------------------------------------------------------
    n_want = cfg["strata_n"]
    picked = OrderedDict()
    short_of = {}
    c_recs, c_bad = [], []
    for fr in frame:
        if fr["category"] != "C":
            continue
        k = by_mcq.get((fr["model_key"], "N-BB", fr["qa_id"]))
        r = answers.get(k)
        if r is None or r.get("strict") == 1 or fr["qa_id"] in excluded:
            c_bad.append(fr["qa_id"])
            continue
        c_recs.append(r)
    if c_bad:
        raise Stop("%d C answers of the frame are missing in the scored files or strict Tier 1 (first %s)"
                   % (len(c_bad), c_bad[0]))
    frames = {"C": c_recs,
              "english_copy": [r for r in answers.values() if r.get("label") == "english_copy"],
              "sibling": [r for r in answers.values() if r.get("label") == "sibling"]}
    repeats = Counter()
    for stratum in STRATA:
        got = 0
        for r in sorted(frames[stratum], key=lambda r: seeded_rank(stratum, r["key"])):
            if got == n_want[stratum]:
                break
            if r["key"] in picked:
                repeats[stratum] += 1
                continue
            if r["qa_id"] in excluded:
                raise Stop("an excluded item reached the frame (%s)" % r["qa_id"])
            picked[r["key"]] = dict(r, stratum=stratum, in_sample="yes", pools=[], d192_findings=[],
                                    ht3_reason=[])
            got += 1
        if got < n_want[stratum]:
            short_of[stratum] = got
    sample_keys = list(picked)
    for i, k in enumerate(sample_keys):
        picked[k]["row_id"] = "H5-%03d" % (i + 1)

    # ---- pool beside the sample: D192 (D282) --------------------------------------
    want = cfg.get("d192_expect")
    got_c = {f: d192_counts.get(f, 0) for f in P1.D192Rules.NAMES}
    distinct_all = len({s["key"] for s, _f in d192_all})
    if want is not None and (got_c != want or distinct_all != cfg["d192_distinct"]):
        raise Stop("the D192 answers differ from the counts of addendum O2: %s (%d distinct), expected %s "
                   "(%d)" % (got_c, distinct_all, want, cfg["d192_distinct"]))
    d192_keys = {s["key"] for s, _f in d192_all}
    pool_rows = []
    copies = 0
    for s, fnds in d192_all:
        if s.get("mode") == "reuse":
            copies += 1
            if s.get("reused_from_key") not in d192_keys:
                raise Stop("a D192 copy without its source answer (%s)" % s["qa_id"])
            continue
        pool_rows.append((s, fnds))
    n_pool_rows = 0
    for s, fnds in pool_rows:
        if s["key"] in picked:
            picked[s["key"]]["pools"].append("d192")
            picked[s["key"]]["d192_findings"] = fnds
        else:
            picked[s["key"]] = dict(s, stratum="", in_sample="no", pools=["d192"], d192_findings=fnds,
                                    ht3_reason=[])
            n_pool_rows += 1

    # ---- pool beside the sample: HT-3 look-first items (D280) -------------------------
    look, _rows = ht3_lookfirst(ctx, finals[-1])
    ht3_info = Counter()
    for q in sorted(look):
        reasons = look[q]
        if q in excluded:
            ht3_info["excluded"] += 1
            continue
        cands = [answers[by_mcq[(m, "N-BB", q)]] for m in HT5_MODELS if (m, "N-BB", q) in by_mcq]
        wrong = sorted([r for r in cands if r.get("strict") != 1],
                       key=lambda r: seeded_rank("ht3-pool", r["key"]))
        if not wrong:
            ht3_info["all_three_strict_tier1"] += 1
            continue
        hit = [r for r in wrong if r["key"] in picked]
        r = hit[0] if hit else wrong[0]
        if r["key"] not in picked:
            picked[r["key"]] = dict(r, stratum="", in_sample="no", pools=[], d192_findings=[], ht3_reason=[])
            ht3_info["rows_added"] += 1
        else:
            ht3_info["already_on_the_sheet"] += 1
        picked[r["key"]]["pools"].append("ht3")
        picked[r["key"]]["ht3_reason"] = list(reasons)
    k_pool = 0
    for k, r in picked.items():
        if r["in_sample"] == "no":
            k_pool += 1
            r["row_id"] = "H5P-%02d" % k_pool

    # ---- content, pictures, codes -----------------------------------------------------
    en_map, en_sha = P1.english_questions(ctx)
    content = {k: row_content(ctx, r, items[r["qa_id"]], en_map) for k, r in picked.items()}
    imgs = check_images(ctx, picked, items)
    order = {role: sorted(picked, key=lambda k: seeded_rank("order", role, k)) for role in ROLES}
    code = {role: {k: "%s%03d" % (CODE_LETTER[role], i + 1) for i, k in enumerate(order[role])}
            for role in ROLES}
    key_rows = []
    for k in sorted(picked, key=lambda k: picked[k]["row_id"]):
        r = picked[k]
        sp = S.condition_spec(r["condition_id"])
        f = items[r["qa_id"]]["image_file"]
        key_rows.append({"row_id": r["row_id"], "code_au1": code["AU1"][k], "code_au2": code["AU2"][k],
                         "key": k, "model_key": r["model_key"], "condition_id": r["condition_id"],
                         "qa_id": r["qa_id"], "house_id": r.get("house_id") or items[r["qa_id"]]["house_id"],
                         "image_file": f, "image_sha256": imgs[f][1], "row_type": r.get("kind"),
                         "requested_lang": sp.answer_lang or sp.question_lang,
                         "in_sample": r["in_sample"], "stratum": r["stratum"], "pools": "|".join(r["pools"]),
                         "d192_findings": "|".join(r["d192_findings"]),
                         "ht3_reason": "|".join(r["ht3_reason"]), "scorer_label": r.get("label"),
                         "strict": r.get("strict"), "named_ids": "|".join(r.get("named_ids") or []),
                         "gold_name_id": r.get("gold_name_id") or "",
                         "gold_classes": "|".join(r.get("gold_value") or [])
                         if isinstance(r.get("gold_value"), list) else ""})
    for d in (P["to_send"], P["key_dir"], P["received"], P["records"]):
        ctx.guard(d).mkdir(parents=True, exist_ok=True)
    if P["key"].exists():
        old = read_csv(P["key"])
        same = [tuple(r.get(c, "") for c in KEY_COLS) for r in old] == \
            [tuple("" if r[c] is None else str(r[c]) for c in KEY_COLS) for r in key_rows]
        if not same:
            raise Stop("an HT-5 draw exists and differs; a draw is never replaced (D286)")
        say("The HT-5 draw exists already and is identical; nothing rewritten.")
        return 0

    # ---- packages ---------------------------------------------------------------------------
    rules = ht5_rules()
    lex_rows, voc_rows = P1.lexicon_rows(ctx), P1.vocab_rows(ctx)
    tabs = [("Read me", None, [[x] for x in rules]),
            ("Lexicon", ("name_bn", "other Tier 1 Bangla", "Tier 1 English", "Tier 2 Bangla",
                         "Tier 2 English", "group"), lex_rows),
            ("Vocabulary", ("kind", "class", "words Bangla", "words English"), voc_rows)]
    leaks = []
    if any(pat.search(guide.decode("utf-8", "replace")) for pat in (ID_RE, HOUSE_RE, MODEL_RE, COND_RE)):
        leaks.append(("pattern", "the guide"))
    staging = ctx.guard(P["to_send"].with_name("to_send.tmp"))
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    built = {}
    for role in ROLES:
        srows, dd, links = [], {}, {}
        for i, k in enumerate(order[role]):
            c = code[role][k]
            d = dict(content[k], code=c, picture="images/%s.jpg" % c)
            srows.append(d)
            dd[(i, "verdict")] = CHECK_ORDER
            dd[(i, "form_kind")] = kinds_for(d["row_type"])
            links[i] = d["picture"]
        xlsx = make_workbook(tabs, "Sheet", SHEET_COLS, srows, dd, links)
        page = view_html("HT-5 rows with their pictures (%s)" % role, [
            "Read-only. Write your verdicts in HT5_sheet_%s.xlsx, not here." % role,
            "Each block shows one row of your sheet: its code, its picture and its text. Click a picture to "
            "open it full size. The guide HT5_auditor_guide.md explains the five checks."], srows,
            VIEW_FIELDS)
        entries = [("HT5_sheet_%s.xlsx" % role, xlsx), ("HT5_view_%s.html" % role, page.encode("utf-8")),
                   ("HT5_auditor_guide.md", guide)]
        for k in order[role]:
            entries.append(("images/%s.jpg" % code[role][k], imgs[items[picked[k]["qa_id"]]["image_file"]][0]))
        texts = workbook_texts(xlsx)
        # the page is built from the same rows: scan its fields, answers for ids only
        texts += [("view %s %s" % (d["code"], f), str(d.get(f, "")), f == "model_answer")
                  for d in srows for f, _label in VIEW_FIELDS]
        texts += [("zip entry", "HT5_%s/%s" % (role, n), False) for n, _d in entries]
        leaks += leak_problems(texts)
        zpath = staging / P["zip"][role].name
        write_zip(zpath, "HT5_%s" % role, entries)
        (staging / P["blank"][role].name).write_bytes(xlsx)
        built[role] = {"zip_sha256": sha256_file(zpath), "xlsx_sha256": hashlib.sha256(xlsx).hexdigest(),
                       "rows": len(srows), "zip_bytes": zpath.stat().st_size}
    model_in_answers = sum(1 for k in picked if MODEL_RE.search(content[k]["model_answer"]))
    if leaks:
        shutil.rmtree(staging)
        kinds = Counter(k for k, _w in leaks)
        raise Stop("leak scan FAILED: %s (first at %s). Nothing was written for sending."
                   % (dict(kinds), leaks[0][1]))
    (staging / P["rules"].name).write_text("\n".join(rules) + "\n", encoding="utf-8")
    if P["to_send"].exists() and any(P["to_send"].iterdir()):
        shutil.rmtree(staging)
        raise Stop("%s holds files but no key exists; move them away and run again" % P["to_send"])
    if P["to_send"].exists():
        P["to_send"].rmdir()
    staging.rename(P["to_send"])
    write_csv(ctx.guard(P["key"]), key_rows, KEY_COLS)
    by_stratum = Counter(r["stratum"] for r in picked.values() if r["in_sample"] == "yes")
    manifest = {"provenance": ctx.provenance(), "rules_version": RULES_VERSION, "auditor_roles": roles,
                "frame_sha256": sha256_file(P["frame"]), "frame_manifest": fman,
                "ht3_folder": finals[-1].name, "scored_sha256": shas, "english_sha256": en_sha,
                "guide_sha256": sha256_file(P["guide"]), "n_sample": len(sample_keys),
                "strata": dict(by_stratum), "strata_planned": n_want, "short_of": short_of,
                "repeats_skipped": dict(repeats), "d192_counts": got_c, "d192_distinct": distinct_all,
                "d192_copies_left_out": copies, "d192_rows_added": n_pool_rows,
                "d192_in_sample": sum(1 for r in picked.values() if "d192" in r["pools"]
                                      and r["in_sample"] == "yes"),
                "ht3_items": len(look), "ht3": dict(ht3_info), "rows": len(picked),
                "images": len(imgs), "packages": built, "key_sha256": sha256_file(P["key"]),
                "model_name_in_answers": model_in_answers}
    write_json(ctx.guard(P["draw"]), manifest)
    # ---- printout: ids and counts only ---------------------------------------------------------
    say("Sample (D284): %d answers: %s%s" % (len(sample_keys), ", ".join(
        "%s %d" % (s, by_stratum.get(s, 0)) for s in STRATA),
        "" if not short_of else "; SHORT: %s (all the frame had)" % short_of))
    mod = Counter((r["stratum"], r["model_key"]) for r in picked.values() if r["in_sample"] == "yes")
    for s in STRATA:
        say("  %-12s by model: %s; by condition: %s" % (s, ", ".join(
            "%s %d" % (m, mod[(s, m)]) for m in HT5_MODELS), ", ".join(
            "%s %d" % kv for kv in sorted(Counter(r["condition_id"] for r in picked.values()
                                                  if r["stratum"] == s).items()))))
    say("  answers drawn again in a later stratum and skipped: %s" % (dict(repeats) or "none"))
    say("D192 beside the sample (D282): touched and scored wrong %s, %d answers%s; %d copies of an "
        "EXP-L answer left out; %d rows added, %d already in the sample"
        % (", ".join("%s %d" % kv for kv in got_c.items()), distinct_all,
           " (equal to addendum O2)" if want is not None else "", copies, n_pool_rows,
           manifest["d192_in_sample"]))
    say("HT-3 look-first items (D280): %d; rows added %d, already on the sheet %d, all three models strict "
        "Tier 1 %d, excluded %d" % (len(look), ht3_info["rows_added"], ht3_info["already_on_the_sheet"],
                                    ht3_info["all_three_strict_tier1"], ht3_info["excluded"]))
    say("Rows per auditor: %d (%d sample + %d beside); pictures checked: %d distinct, sha256 as "
        "images_manifest.csv, no metadata" % (len(picked), len(sample_keys), len(picked) - len(sample_keys),
                                              len(imgs)))
    say("Leak scan: PASS (no qa_id, image or house id, model name or condition id; model answers scanned "
        "for ids only; %d answers name a model themselves)" % model_in_answers)
    for role in ROLES:
        say("  %s: %s, %d rows, %.1f MB, sha %s" % (role, P["zip"][role].name, built[role]["rows"],
                                                  built[role]["zip_bytes"] / 1e6, short(built[role]["zip_sha256"])))
    say("Wrote %s (two packages, the blank sheets, the rules) and %s (key: never sent)"
        % (P["to_send"], P["key_dir"]))
    ctx.log_run({"command": "ht5-draw", "rows": len(picked), "n_sample": len(sample_keys),
                 "key_sha256": manifest["key_sha256"]})
    say("HT-5 DRAW: DONE")
    return 0


# ---------------------------------------------------------------------------
# 8. Receipts and the consensus (Cells 5 and 6)
# ---------------------------------------------------------------------------

def form_lang(ctx, form):
    toks = ctx.S.tokenise(form)
    kinds = {t.kind for t in toks}
    if kinds == {"bn"}:
        return "bn"
    if kinds == {"lat"}:
        return "en"
    return None


def form_key(ctx, form):
    return ctx.lexicon().index.key_of(form) if form else ""


def read_entry(ctx, kr, verdict, form, kind):
    """(clean values, problem or None) for one verdict cell set."""
    v = (verdict or "").strip().lower()
    form_n = ctx.normalise(form or "")
    k = (kind or "").strip().lower()
    if not v:
        return None, "no verdict"
    if v not in CHECK_ORDER:
        return None, "unknown verdict %r" % v
    if v == "missing_variant":
        if not form_n:
            return None, "missing_variant needs missing_form"
        lang = form_lang(ctx, form_n)
        if lang is None:
            return None, "missing_form must be one name in one script"
        if lang != kr["requested_lang"]:
            return None, "missing_form must be in %s letters (answer_requested_in)" % (
                "Bangla" if kr["requested_lang"] == "bn" else "Latin")
        if k not in kinds_for(kr["row_type"]):
            return None, "form_kind must be one of %s" % ", ".join(kinds_for(kr["row_type"]))
        return {"verdict": v, "form": form_n, "kind": k, "lang": lang}, None
    if form_n or k:
        return None, "missing_form and form_kind only with missing_variant"
    return {"verdict": v, "form": "", "kind": "", "lang": ""}, None


def read_returned(ctx, P, role, key_rows):
    path = P["returned"][role]
    rows = P1.read_workbook(path, "Sheet", ("code", "verdict", "missing_form", "form_kind", "note"))
    col = "code_au1" if role == "AU1" else "code_au2"
    by_code = {kr[col]: kr for kr in key_rows}
    got, errors = {}, []
    for r in rows:
        c = r["code"]
        if c not in by_code:
            errors.append((c, "unknown code"))
            continue
        if c in got:
            errors.append((c, "code twice"))
            continue
        val, prob = read_entry(ctx, by_code[c], r["verdict"], r["missing_form"], r["form_kind"])
        if prob:
            errors.append((c, prob))
            val = {"verdict": "", "form": "", "kind": "", "lang": ""}
        val["note"] = r.get("note", "")
        got[c] = val
    for c in sorted(set(by_code) - set(got)):
        errors.append((c, "row missing"))
    return (not errors), {kr["key"]: got.get(kr[col]) for kr in key_rows}, errors, len(rows)


def differs(ctx, a, b):
    if a["verdict"] != b["verdict"]:
        return True
    return a["verdict"] == "missing_variant" and (form_key(ctx, a["form"]) != form_key(ctx, b["form"])
                                                  or a["kind"] != b["kind"])


def unpack_images(P, role, codes):
    out = {}
    with zipfile.ZipFile(P["zip"][role]) as z:
        for c in codes:
            out[c] = z.read("HT5_%s/images/%s.jpg" % (role, c))
    return out


def cmd_ht5_receive(ctx, args):
    P = ht5_paths(ctx)
    say("Scoring and audit, part 2: HT-5 receipt (D281, D286), %s, run %s" % (SCRIPT_VERSION, utc_now()))
    if not P["key"].exists():
        raise Stop("no HT-5 draw yet")
    key_rows = read_csv(P["key"])
    got = {}
    all_ok = True
    for role in ROLES:
        path = P["returned"][role]
        if not path.exists():
            say("%s: %s not received yet" % (role, path.name))
            all_ok = False
            continue
        ok, vals, errors, n = read_returned(ctx, P, role, key_rows)
        if not ok:
            say("%s: FAIL %s: %d rows read, %d problems. Send the auditor this list (codes only); nothing "
                "recorded:" % (role, path.name, n, len(errors)))
            for c, m in errors[:40]:
                say("    %s: %s" % (c, m))
            if len(errors) > 40:
                say("    ... %d more" % (len(errors) - 40))
            all_ok = False
            continue
        rec, new = P1.take_receipt(ctx, P, role, path, n)
        notes = sum(1 for v in vals.values() if v and v.get("note"))
        say("%s: PASS %s, %d rows, %d notes; sha %s, received %s%s" % (
            role, path.name, n, notes, short(rec["sha256"]), rec["received_utc"],
            "" if new else " (receipt already on file)"))
        got[role] = vals
    if not all_ok:
        say("RECEIPT: %d of 2" % len(got))
        return 1
    say("RECEIPT: 2 of 2")
    differ = [kr for kr in key_rows if differs(ctx, got["AU1"][kr["key"]], got["AU2"][kr["key"]])]
    for part, rows in (("sample", [kr for kr in key_rows if kr["in_sample"] == "yes"]),
                       ("beside", [kr for kr in key_rows if kr["in_sample"] != "yes"])):
        same_v = sum(1 for kr in rows if got["AU1"][kr["key"]]["verdict"] == got["AU2"][kr["key"]]["verdict"])
        say("Auditors agree on the verdict in %d of %d %s rows" % (same_v, len(rows), part))
    if P["cons_zip"].exists():
        say("The consensus package exists already: %s" % P["cons_zip"])
        return 0
    if not differ:
        say("No row differs: no consensus meeting is needed. Run Cell 6.")
        return 0
    blank = {r["code"]: r for r in P1.read_workbook(P["blank"]["AU1"], "Sheet", SHEET_COLS)}
    srows, dd = [], {}
    for i, kr in enumerate(sorted(differ, key=lambda kr: kr["code_au1"])):
        a, b = got["AU1"][kr["key"]], got["AU2"][kr["key"]]
        base = blank[kr["code_au1"]]
        d = {c: base.get(c, "") for c in CONTENT_COLS}
        same_v = a["verdict"] == b["verdict"]
        d.update(code=kr["code_au1"], picture="images/%s.jpg" % kr["code_au1"],
                 au1_verdict=a["verdict"], au1_form=a["form"], au1_kind=a["kind"], au1_note=a.get("note", ""),
                 au2_verdict=b["verdict"], au2_form=b["form"], au2_kind=b["kind"], au2_note=b.get("note", ""),
                 consensus_verdict=a["verdict"] if same_v else "", consensus_form="", consensus_kind="")
        srows.append(d)
        dd[(i, "consensus_verdict")] = CHECK_ORDER
        dd[(i, "consensus_kind")] = kinds_for(kr["row_type"])
    tabs = [("Read me", None, [[x] for x in cons_rules()]),
            ("Lexicon", ("name_bn", "other Tier 1 Bangla", "Tier 1 English", "Tier 2 Bangla",
                         "Tier 2 English", "group"), P1.lexicon_rows(ctx)),
            ("Vocabulary", ("kind", "class", "words Bangla", "words English"), P1.vocab_rows(ctx))]
    links = {i: d["picture"] for i, d in enumerate(srows)}
    xlsx = make_workbook(tabs, "Consensus", CONS_COLS, srows, dd, links)
    page = view_html("HT-5 consensus rows with their pictures", [
        "Read-only. Write the agreed verdicts in HT5_consensus.xlsx, not here."], srows, VIEW_FIELDS,
        (("au1_verdict", "auditor 1 verdict"), ("au1_form", "auditor 1 form"), ("au1_kind", "auditor 1 kind"),
         ("au1_note", "auditor 1 note"), ("au2_verdict", "auditor 2 verdict"), ("au2_form", "auditor 2 form"),
         ("au2_kind", "auditor 2 kind"), ("au2_note", "auditor 2 note")))
    imgs = unpack_images(P, "AU1", [d["code"] for d in srows])
    entries = [("HT5_consensus.xlsx", xlsx), ("HT5_consensus_view.html", page.encode("utf-8"))]
    entries += [("images/%s.jpg" % c, data) for c, data in imgs.items()]
    write_zip(ctx.guard(P["cons_zip"]), "HT5_consensus", entries)
    ctx.guard(P["cons_blank"]).write_bytes(xlsx)
    n_v = sum(1 for d in srows if d["au1_verdict"] != d["au2_verdict"])
    say("Wrote %s (%d rows: verdicts differ %d, same verdict missing_variant with another form or kind %d)"
        % (P["cons_zip"], len(srows), n_v, len(srows) - n_v))
    ctx.log_run({"command": "ht5-receive", "rows": len(key_rows), "differ": len(srows)})
    return 0


def cmd_ht5_consensus(ctx, args):
    P = ht5_paths(ctx)
    ST = ctx.stats()
    lex = ctx.lexicon()
    say("Scoring and audit, part 2: HT-5 consensus and the D238 file, %s, run %s" % (SCRIPT_VERSION, utc_now()))
    key_rows = read_csv(P["key"])
    got = {}
    for role in ROLES:
        if not [r for r in P1.receipts(P) if r["who"] == role]:
            raise Stop("no receipt for %s; run Cell 5 first" % role)
        ok, vals, _e, n = read_returned(ctx, P, role, key_rows)
        if not ok:
            raise Stop("%s no longer reads as a complete sheet" % P["returned"][role].name)
        P1.take_receipt(ctx, P, role, P["returned"][role], n)      # same sha or STOP
        got[role] = vals
    differ = [kr for kr in key_rows if differs(ctx, got["AU1"][kr["key"]], got["AU2"][kr["key"]])]
    differ_keys = {kr["key"] for kr in differ}
    final = {}
    for kr in key_rows:
        if kr["key"] not in differ_keys:
            final[kr["key"]] = dict(got["AU1"][kr["key"]])
    if differ:
        path = P["cons_returned"]
        if not path.exists():
            say("%s not received yet" % path.name)
            say("CONSENSUS: WAITING")
            return 1
        rows = P1.read_workbook(path, "Consensus", ("code", "consensus_verdict", "consensus_form",
                                                    "consensus_kind"))
        by = {r["code"]: r for r in rows}
        problems = []
        for kr in differ:
            r = by.get(kr["code_au1"])
            if r is None:
                problems.append((kr["code_au1"], "row missing"))
                continue
            val, prob = read_entry(ctx, kr, r["consensus_verdict"], r["consensus_form"], r["consensus_kind"])
            if prob:
                problems.append((kr["code_au1"], prob))
            else:
                final[kr["key"]] = val
        if problems or len(rows) != len(differ):
            say("CONSENSUS: FAIL: %d rows to fix; %d rows in the file, %d expected:" % (
                len(problems), len(rows), len(differ)))
            for c, m in problems[:40]:
                say("    %s: %s" % (c, m))
            return 1
        rec, _new = P1.take_receipt(ctx, P, "consensus", path, len(rows))
        say("Consensus sheet: %d rows settled, sha %s, received %s" % (len(rows), short(rec["sha256"]),
                                                                       rec["received_utc"]))
    audit = [{"audit_id": kr["row_id"], "model_key": kr["model_key"], "condition_id": kr["condition_id"],
              "qa_id": kr["qa_id"], "house_id": kr["house_id"], "stratum": kr["stratum"],
              "verdict": final[kr["key"]]["verdict"]}
             for kr in key_rows if kr["in_sample"] == "yes"]
    verdicts = [{"row_id": kr["row_id"], "in_sample": kr["in_sample"], "stratum": kr["stratum"],
                 "pools": kr["pools"], "model_key": kr["model_key"], "condition_id": kr["condition_id"],
                 "qa_id": kr["qa_id"], "house_id": kr["house_id"], "row_type": kr["row_type"],
                 "scorer_label": kr["scorer_label"], "au1_verdict": got["AU1"][kr["key"]]["verdict"],
                 "au2_verdict": got["AU2"][kr["key"]]["verdict"],
                 "settled_at_consensus": "yes" if kr["key"] in differ_keys else "no",
                 "verdict": final[kr["key"]]["verdict"],
                 "form_given": "yes" if final[kr["key"]]["form"] else "no"} for kr in key_rows]
    names = namer_names(ctx)
    forms = []
    for kr in key_rows:
        f = final[kr["key"]]
        if f["verdict"] != "missing_variant":
            continue
        gid = kr["gold_name_id"]
        k = form_key(ctx, f["form"])
        senses = lex.index.entries.get(k, [])
        own = sorted({s.tier for s in senses if s.name_id == gid})
        other = sorted({lex.objects[s.name_id].name_bn for s in senses if s.name_id != gid and s.tier == 1})
        if kr["row_type"] != "naming":
            idx = ctx.vocab().indexes.get(kr["row_type"])
            own = ["listed"] if idx is not None and idx.key_of(f["form"]) in idx.entries else []
            other = []
        used = names.get(gid, set())
        forms.append({"row_id": kr["row_id"], "in_sample": kr["in_sample"], "stratum": kr["stratum"],
                      "pools": kr["pools"], "model_key": kr["model_key"], "condition_id": kr["condition_id"],
                      "qa_id": kr["qa_id"], "row_type": kr["row_type"], "requested_lang": kr["requested_lang"],
                      "gold_name_id": gid, "gold_name_bn": lex.objects[gid].name_bn if gid in lex.objects else "",
                      "gold_classes": kr["gold_classes"], "form": f["form"], "form_kind": f["kind"],
                      "form_lang": f["lang"], "au1_form": got["AU1"][kr["key"]]["form"],
                      "au2_form": got["AU2"][kr["key"]]["form"],
                      "listed_for_gold": "|".join(str(x) for x in own),
                      "listed_for_other": "|".join(other),
                      "namer_used": ("yes" if k in used else "no") if kr["row_type"] == "naming" else "-"})
    notes = [{"row_id": kr["row_id"], "code_au1": kr["code_au1"], "code_au2": kr["code_au2"],
              "au1_note": got["AU1"][kr["key"]].get("note", ""), "au2_note": got["AU2"][kr["key"]].get("note", "")}
             for kr in key_rows if got["AU1"][kr["key"]].get("note") or got["AU2"][kr["key"]].get("note")]
    w1 = write_new(ctx, P["audit"], audit, list(ST.HT5_COLUMNS))
    write_new(ctx, P["verdicts"], verdicts, VERDICT_COLS)
    write_new(ctx, P["forms"], forms, FORM_COLS)
    write_new(ctx, P["notes"], notes, ["row_id", "code_au1", "code_au2", "au1_note", "au2_note"])
    say("Wrote %s (%d rows, D238 columns)%s, %s (%d rows, ids), %s (%d forms) and %s (%d rows with a note)"
        % (P["audit"], len(audit), "" if w1 else " (identical, kept)", P["verdicts"].name, len(verdicts),
           P["forms"].name, len(forms), P["notes"].name, len(notes)))
    ctx.log_run({"command": "ht5-consensus", "audit_sha256": sha256_file(P["audit"]), "forms": len(forms)})
    say("CONSENSUS: DONE. The auditors may now open Drive scored/, results/ and human/HT-3/.")
    return 0


def namer_names(ctx):
    """gold name_id -> set of index keys of the names the HT-3 namers wrote for
    items of that gold (D280: compared after the consensus, never shown)."""
    finals = ht3_final_dirs(ctx)
    out = defaultdict(set)
    if not finals:
        return out
    items, lex, voc = ctx.items(), ctx.lexicon(), ctx.vocab()
    for r in read_csv(finals[-1] / "ht3_scored_names.csv"):
        q = r["qa_id"]
        if r.get("excluded") == "yes" or q not in items or not r.get("name"):
            continue
        g = ctx.S.resolve_gold(items[q], lex, voc)
        if g.name_id:
            out[g.name_id].add(lex.index.key_of(ctx.S.strip_lead_in(ctx.normalise(r["name"]))))
    return out


# ---------------------------------------------------------------------------
# 9. The HT-5 reading (Cell 7)
# ---------------------------------------------------------------------------

def est_text(e):
    if not e or e.get("point") is None:
        return "-"
    lo, hi = e.get("lo"), e.get("hi")
    return "%.1f%% (%s to %s)" % (100 * e["point"], "-" if lo is None else "%.1f%%" % (100 * lo),
                                  "-" if hi is None else "%.1f%%" % (100 * hi))


def cmd_ht5_read(ctx, args):
    P = ht5_paths(ctx)
    cfg = ctx.cfg
    say("Scoring and audit, part 2: HT-5 reading (Design 9.3 and 10, D234, D238), %s, run %s"
        % (SCRIPT_VERSION, utc_now()))
    L = Lines()
    key_rows = read_csv(P["key"])
    draw = json.loads(P["draw"].read_text("utf-8"))
    rows = read_csv(P["audit"])
    ver = {r["row_id"]: r for r in read_csv(P["verdicts"])}
    forms = read_csv(P["forms"])
    ST = ctx.stats()
    with open(P["audit"], encoding="utf-8-sig", newline="") as fh:
        hdr = next(csv.reader(fh), [])
    sample = [kr for kr in key_rows if kr["in_sample"] == "yes"]
    by_id = {kr["row_id"]: kr for kr in key_rows}
    mism = sum(1 for r in rows if by_id.get(r["audit_id"]) is None or any(
        r[c] != by_id[r["audit_id"]][c] for c in ("model_key", "condition_id", "qa_id", "house_id", "stratum")))
    bad = sum(1 for r in rows if r["verdict"] not in CHECK_ORDER or r["stratum"] not in STRATA)
    ex = ctx.excluded()
    L.add("R01", hdr == list(ST.HT5_COLUMNS) and len(rows) == len(sample) and mism == 0 and bad == 0
          and not any(r["qa_id"] in ex for r in rows),
          "ht5_audit.csv %s: %d rows (D238 columns %s); strata %s; rows that differ from the key %d; unknown "
          "values %d" % (short(sha256_file(P["audit"])), len(rows), hdr == list(ST.HT5_COLUMNS), ", ".join(
              "%s %d" % (s, sum(1 for r in rows if r["stratum"] == s)) for s in STRATA), mism, bad))
    shas_now = {m: sha256_file(ctx.scored_path(m)) for m in MODELS}
    L.add("R02", shas_now == draw["scored_sha256"], "the six scored files are the ones the sheets were "
          "drawn from%s" % ("" if shas_now == draw["scored_sha256"] else ": CHANGED"))
    try:
        ST, checks, world, recs, budgets = stats_inputs(ctx)
        n0 = len(checks.rows)
        f0 = len(fails_of(checks))
        st_rows = ST.read_ht5(str(P["audit"]), world, recs, budgets, checks)
        new = checks.rows[n0:]
        f = fails_of(checks)
        L.add("R03", not f and len(st_rows) == len(rows), "stats.read_ht5 as the Statistics chat runs it "
              "(inputs and scored files: %d check lines, %d FAIL): %s" % (
                  n0, f0, "; ".join("%s %s %s" % x for x in new)[:700]))
    except Exception as exc:
        L.add("R03", False, "stats.read_ht5: %s: %s" % (type(exc).__name__, exc))
    res = ST.ht5_stats(rows, n_boot=cfg["n_boot"])
    L.info("R04", "F3 mark (D234): C answers judged gold_wrong or not_visible, pooled over the models: %s; "
           "gold-error rate over all %d audited answers: %s (house bootstrap, %d resamples, seed %d, %d houses)"
           % (est_text(res["C_gold_or_visibility"]), res["n"], est_text(res["gold_error_rate"]), cfg["n_boot"],
              SEED, res["houses"]))
    for s in STRATA:
        v = res["strata"][s]
        L.info("R05", "%-12s n %3d: %s" % (s, v["n"], "; ".join(
            "%s %d (%s)" % (x, sum(1 for r in rows if r["stratum"] == s and r["verdict"] == x),
                            est_text(v["verdicts"][x])) for x in CHECK_ORDER)))
    vs = list(ver.values())
    for part, rr in (("sample", [v for v in vs if v["in_sample"] == "yes"]),
                     ("beside", [v for v in vs if v["in_sample"] != "yes"])):
        k = sum(1 for v in rr if v["au1_verdict"] == v["au2_verdict"])
        L.info("R06", "auditor agreement (counts only, D281): %s %d of %d verdicts (%s); settled at the "
               "consensus %d" % (part, k, len(rr), pct(k, len(rr)),
                                 sum(1 for v in rr if v["settled_at_consensus"] == "yes")))
    sv = [v for v in vs if v["in_sample"] == "yes"]
    for s in STRATA:
        rr = [v for v in sv if v["stratum"] == s]
        k = sum(1 for v in rr if v["au1_verdict"] == v["au2_verdict"])
        L.info("R07", "  %-12s auditors agree %d of %d" % (s, k, len(rr)))
    cm = Counter((v["model_key"], v["verdict"]) for v in sv)
    L.info("R08", "verdicts by model (sample): %s" % "; ".join("%s: %s" % (m, ", ".join(
        "%s %d" % (x, cm[(m, x)]) for x in CHECK_ORDER if cm[(m, x)])) for m in HT5_MODELS))
    cl = Counter((v["scorer_label"], v["verdict"]) for v in sv)
    labs = sorted({v["scorer_label"] for v in sv})
    L.info("R09", "verdicts by scorer label (sample): %s" % "; ".join("%s: %s" % (lab, ", ".join(
        "%s %d" % (x, cl[(lab, x)]) for x in CHECK_ORDER if cl[(lab, x)])) for lab in labs))
    try:
        pairs = {(r["qa_id"], r["removed_id"]) for r in read_csv(P["swaps"]) if r.get("reason") == "name"}
    except Exception:
        pairs = set()
    on = [kr for kr in sample if kr["scorer_label"] == "sibling"
          and any((kr["qa_id"], s) in pairs for s in kr["named_ids"].split("|") if s)]
    L.info("R10", "D204: sample answers labelled sibling on an item and sibling pair that HT-6 judged also "
           "right: %d of %d; their verdicts: %s" % (
               len(on), sum(1 for kr in sample if kr["scorer_label"] == "sibling"),
               ", ".join("%s %d" % kv for kv in sorted(Counter(ver[kr["row_id"]]["verdict"] for kr in on).items()))
               or "-"))
    h3 = [kr for kr in key_rows if "ht3" in kr["pools"].split("|")]
    for reason in ("same_other_object", "frame_as_painting"):
        rr = [kr for kr in h3 if reason in kr["ht3_reason"].split("|")]
        L.info("R11", "HT-3 look-first rows (D280), %s: %d; verdicts: %s" % (reason, len(rr), ", ".join(
            "%s %d" % kv for kv in sorted(Counter(ver[kr["row_id"]]["verdict"] for kr in rr).items())) or "-"))
    try:
        ch = read_csv(P["d279"] / "changes.csv")
        ka = {r["key"] for r in ch if r["rule"] == "a"}
        kb = {r["key"] for r in ch if r["rule"] == "b"}
    except Exception:
        ka = kb = set()
    ra = [kr for kr in sample if kr["key"] in ka]
    L.info("R12", "D279: sample answers that rule a re-scores %d (verdicts: %s); rule b %d (forced choice "
           "only, so 0 is expected)" % (len(ra), ", ".join("%s %d" % kv for kv in sorted(Counter(
               ver[kr["row_id"]]["verdict"] for kr in ra).items())) or "-", sum(1 for kr in sample if kr["key"] in kb)))
    d1 = [kr for kr in key_rows if "d192" in kr["pools"].split("|")]
    for f in P1.D192Rules.NAMES:
        rr = [kr for kr in d1 if f in kr["d192_findings"].split("|")]
        if rr:
            L.info("R13", "D192 %s: %d answers; verdicts: %s" % (f, len(rr), ", ".join(
                "%s %d" % kv for kv in sorted(Counter(ver[kr["row_id"]]["verdict"] for kr in rr).items()))))
    fs = forms
    distinct = len({(r["gold_name_id"] or r["gold_classes"], r["form"]) for r in fs})
    L.info("R14", "missing-variant list (Design 9.4 item 12): %d answers (sample %d, beside %d), %d distinct "
           "forms; already listed for the gold %d; a Tier 1 name of another object %d (cannot be added); "
           "equal to an HT-3 namer's name for that object %d. Forms: %s" % (
               len(fs), sum(1 for r in fs if r["in_sample"] == "yes"), sum(1 for r in fs if r["in_sample"] != "yes"),
               distinct, sum(1 for r in fs if r["listed_for_gold"]), sum(1 for r in fs if r["listed_for_other"]),
               sum(1 for r in fs if r["namer_used"] == "yes"), P["forms"].name))
    passed = not L.failed()
    reading = {"provenance": ctx.provenance(), "pass": passed, "lines": L.rows, "ht5_stats": res,
               "audit_sha256": sha256_file(P["audit"]), "verdicts_sha256": sha256_file(P["verdicts"]),
               "forms_sha256": sha256_file(P["forms"]), "scored_sha256": shas_now}
    write_json(ctx.guard(P["reading"]), reading)
    say("Wrote %s" % P["reading"])
    ctx.log_run({"command": "ht5-read", "pass": passed})
    say("HT-5 READING: %s" % ("PASS. Send this printout" if passed else "FAIL. Send this printout"))
    return 0 if passed else 1


# ---------------------------------------------------------------------------
# 10. lexicon_sensitivity.csv (Cells 8 and 9): template, worksheet and check
# ---------------------------------------------------------------------------

def cmd_sens_template(ctx, args):
    P = ht5_paths(ctx)
    ST = ctx.stats()
    lex, voc = ctx.lexicon(), ctx.vocab()
    say("Scoring and audit, part 2: lexicon sensitivity template (D238, D287), %s, run %s"
        % (SCRIPT_VERSION, utc_now()))
    if not P["forms"].exists():
        raise Stop("no consensus yet (Cell 6)")
    tp = ctx.guard(P["sens_template"])
    with open(tp, "w", encoding="utf-8-sig", newline="") as fh:
        csv.writer(fh).writerow(list(ST.SENS_COLUMNS))
    forms = read_csv(P["forms"])
    frows = []
    for r in forms:
        g = lex.objects.get(r["gold_name_id"])
        frows.append([r["row_id"], r["row_type"], r["requested_lang"], r["gold_name_id"],
                      g.name_bn if g else "", " | ".join(g.t1_bn + g.t1_en) if g else "",
                      " | ".join(g.t2_bn + g.t2_en) if g else "", r["gold_classes"], r["form"], r["form_kind"],
                      r["form_lang"], r["listed_for_gold"], r["listed_for_other"], r["namer_used"], r["pools"]])
    key_rows = read_csv(P["key"])
    ver = {r["row_id"]: r for r in read_csv(P["verdicts"])}
    rules = P1.D192Rules(ctx)
    audit = ctx.root / "scored" / "asset9_stage2" / "audit"
    ew = {r["example_qa_id"]: r for r in read_csv(audit / "extra_words.csv")}
    drows = []
    for f, text in P1.D192Rules.NAMES.items():
        anchor = ctx.cfg["d192_anchors"].get(f, "")
        e = ew.get(anchor, {})
        rr = [kr for kr in key_rows if f in kr["d192_findings"].split("|")]
        vc = Counter(ver[kr["row_id"]]["verdict"] for kr in rr)
        drows.append([f, text, anchor, e.get("gold", ""), e.get("extra_words", ""), len(rr),
                      ", ".join("%s %d" % kv for kv in sorted(vc.items()))])
    for k, cid, form, bare in rules.gen:
        drows.append(["genitive-only form", "%s %s" % (k, cid), "", form, bare, "", ""])
    lrows = [[oid, o.name_bn, " | ".join(o.t1_bn[1:]), " | ".join(o.t1_en), " | ".join(o.t2_bn),
              " | ".join(o.t2_en), o.cluster_id or ""] for oid, o in sorted(lex.objects.items(),
                                                                            key=lambda kv: kv[1].name_bn)]
    vrows = [[kind, cls, " | ".join(voc.forms[kind][cls].get("bn", [])),
              " | ".join(voc.forms[kind][cls].get("en", []))]
             for kind in ctx.S.VOCAB_KINDS for cls in sorted(voc.forms[kind])]
    readme = [
        "HT-5 SENSITIVITY WORKSHEET (%s)" % RULES_VERSION,
        "",
        "Read HT5_sensitivity_instructions.md first. Write the rows in lexicon_sensitivity_TEMPLATE.csv and "
        "save it as human/HT-5/lexicon_sensitivity.csv. Nobody else, and no program, writes a row (D287).",
        "Tab 'HT-5 forms': every name or word the two auditors agreed was right but missing. An HT-5 row "
        "may only use a form from this tab.",
        "Tab 'D192 findings': the seven B7 gaps of DESIGN_ADDENDUM B7, the anchor gold, the extra words, and "
        "how the auditors judged the answers they touched. D192 rows come from the B7 table.",
        "Tabs 'Lexicon' and 'Vocabulary': the ids to write in target_id (name_id for the lexicon, class id "
        "for the vocabulary).",
    ]
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    wb = Workbook()
    wb.remove(wb.active)
    for name, header, trows in (
            ("Read me", None, [[x] for x in readme]),
            ("HT-5 forms", ("row_id", "row_type", "requested_lang", "gold_name_id", "gold_name_bn",
                            "gold Tier 1 names", "gold Tier 2 names", "gold_classes", "form", "form_kind",
                            "form_lang", "listed_for_gold", "listed_for_other", "namer_used", "pools"), frows),
            ("D192 findings", ("finding", "what", "anchor qa_id", "anchor gold", "extra words",
                               "answers on the HT-5 sheets", "consensus verdicts"), drows),
            ("Lexicon", ("name_id", "name_bn", "other Tier 1 Bangla", "Tier 1 English", "Tier 2 Bangla",
                         "Tier 2 English", "group"), lrows),
            ("Vocabulary", ("kind", "class_id", "words Bangla", "words English"), vrows)):
        ws = wb.create_sheet(name)
        if header:
            ws.append(list(header))
            for c in ws[1]:
                c.font = Font(bold=True)
        for tr in trows:
            ws.append(list(tr))
        ws.column_dimensions["A"].width = 120 if header is None else 22
        for line in ws.iter_rows():
            for c in line:
                c.alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(str(ctx.guard(P["sens_sheet"])))
    say("Wrote %s (header only) and %s (HT-5 forms %d, D192 findings %d, lexicon %d objects, vocabulary %d "
        "classes)" % (tp, P["sens_sheet"].name, len(frows), len(drows), len(lrows), len(vrows)))
    say("NEXT: the two auditors write the rows (HT5_sensitivity_instructions.md), save them as %s, then run "
        "Cell 9." % P["sens"])
    ctx.log_run({"command": "sens-template", "forms": len(frows)})
    return 0


def cmd_sens_check(ctx, args):
    P = ht5_paths(ctx)
    say("Scoring and audit, part 2: lexicon_sensitivity.csv check (Design 9.4 item 12, D238, D287), %s, "
        "run %s" % (SCRIPT_VERSION, utc_now()))
    if not P["sens"].exists():
        raise Stop("%s not there yet" % P["sens"])
    L = Lines()
    ST = ctx.stats()
    s = sha256_file(P["sens"])
    with open(P["sens"], encoding="utf-8-sig", newline="") as fh:
        hdr = [h.strip() for h in next(csv.reader(fh), [])]
    rows = ST.read_csv_dicts(P["sens"])
    ids = Counter(r.get("row_id", "") for r in rows)
    dup = [k for k, v in ids.items() if v > 1 or not k]
    L.add("X01", hdr == list(ST.SENS_COLUMNS) and not dup and rows,
          "lexicon_sensitivity.csv %s: %d rows (%s); D238 columns in order %s; row_id empty or repeated %d"
          % (short(s), len(rows), ", ".join("%s %d" % kv for kv in sorted(Counter(r.get("source", "")
                                                                                  for r in rows).items())),
             hdr == list(ST.SENS_COLUMNS), len(dup)))
    checks = ST.Checks(echo=False)
    ST.read_sensitivity(str(P["sens"]), checks)
    f = fails_of(checks)
    L.add("X02", not f, "stats.read_sensitivity: %s" % "; ".join("%s %s" % (x[0], x[2]) for x in checks.rows))
    forms = read_csv(P["forms"]) if P["forms"].exists() else []
    allowed = {form_key(ctx, ctx.normalise(r["form"])) for r in forms}
    stray = [r.get("row_id") for r in rows if r.get("source") == "HT-5" and r.get("op") != "item_any_class"
             and form_key(ctx, ctx.normalise(r.get("form", ""))) not in allowed]
    L.add("X03", not stray, "every HT-5 row uses a form the two auditors agreed on (TEXT_ht5_forms.csv, %d "
          "forms)%s" % (len(forms), "" if not stray else "; NOT: %s" % ", ".join(stray)))
    try:
        ST2, chk2, world, recs, _b = stats_inputs(ctx)
        lex2, voc2, any_class = ST2._apply_rows(world, rows)
        L.add("X04", True, "the rows apply to copies of the frozen files (stats._apply_rows): lexicon %d "
              "objects, %d terms; vocabulary %s; item_any_class items %d; the frozen files are not touched"
              % (len(lex2.objects), len(lex2.index.entries), voc2.describe(), len(any_class)))
        _new, info = ST2.sensitivity_records(world, recs, rows)
        L.info("X05", "answers that change when every row is applied: %d" % info["changed_total"])
        for k, v in sorted(info["changed_by_model_condition"].items()):
            L.info("X06", "  %s: %d" % (k, v))
        for pr in info["per_row"]:
            L.info("X07", "  row %s (%s, %s %s %s): %d answers change when this row is left out"
                   % (pr["row_id"], pr["source"], pr["op"], pr["target"], pr["target_id"], pr["changed"]))
        zero = [pr["row_id"] for pr in info["per_row"] if pr["changed"] == 0]
        if zero:
            L.info("X08", "rows that change no answer on their own: %s (allowed; say so in S9)" % ", ".join(zero))
    except Exception as exc:
        info = {}
        L.add("X04", False, "the rows do not apply: %s: %s" % (type(exc).__name__, exc))
    passed = not L.failed()
    log = ctx.guard(P["sens_log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    prev = [json.loads(x) for x in log.read_text("utf-8").splitlines() if x.strip()] if log.exists() else []
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"utc": utc_now(), "sha256": s, "rows": len(rows), "pass": passed}) + "\n")
    if prev and prev[-1]["sha256"] != s:
        say("INFO  the file changed since the last check (%s); the newest checked version is the one the "
            "Statistics chat reads" % short(prev[-1]["sha256"]))
    write_json(ctx.guard(P["sens_check"]), {"provenance": ctx.provenance(), "pass": passed, "sha256": s,
                                            "lines": L.rows, "info": info})
    ctx.log_run({"command": "sens-check", "pass": passed, "sha256": s})
    say("SENS-CHECK: %s" % ("PASS. Send this printout" if passed else "FAIL. Fix the rows and run again"))
    return 0 if passed else 1


# ---------------------------------------------------------------------------
# 11. status (Cell 10)
# ---------------------------------------------------------------------------

def cmd_status(ctx, args):
    P = ht5_paths(ctx)
    say("Scoring and audit, part 2: status, %s, run %s" % (SCRIPT_VERSION, utc_now()))
    finals = ht3_final_dirs(ctx)
    say("  HT-3 final folders: %s" % (", ".join(p.name for p in finals) or "none"))
    checks = sorted(P["ht3_check_dir"].glob("ht3_final_check_*.json")) if P["ht3_check_dir"].is_dir() else []
    if checks:
        say("  HT-3 final check: %s" % ("PASS" if json.loads(checks[-1].read_text("utf-8")).get("pass")
                                        else "FAIL"))
    say("  frame: %s" % ("present, %d rows" % len(read_csv(P["frame"])) if P["frame"].exists() else "not yet"))
    say("  guide: %s" % ("present" if P["guide"].exists() else "not yet"))
    say("  HT-5 draw: %s" % ("yes, %d rows" % len(read_csv(P["key"])) if P["key"].exists() else "no"))
    got = sorted({r["who"] for r in P1.receipts(P)}) if P["receipts"].exists() else []
    say("  receipts: %s" % (", ".join(got) or "none"))
    for name in ("audit", "verdicts", "forms", "sens_template", "sens"):
        say("  %-26s %s" % (P[name].name, "present" if P[name].exists() else "not yet"))
    for name in ("reading", "sens_check"):
        if P[name].exists():
            say("  %-26s %s" % (P[name].name, "PASS" if json.loads(P[name].read_text("utf-8")).get("pass")
                                else "FAIL"))
    return 0


# ---------------------------------------------------------------------------
# 12. selftest (Cell 1): the chain on made-up answers with the gate-E scorer
# ---------------------------------------------------------------------------

class MiniWorld:
    """The few World fields that read_ht5, _apply_rows and sensitivity_records use
    (selftest only; the real run uses stats.build_world)."""

    def __init__(self, ctx, items, excluded):
        self.sc, self.normalise = ctx.S, ctx.normalise
        self.lex, self.voc = ctx.lexicon(), ctx.vocab()
        self.paths = SimpleNamespace(repo=ctx.gate_e)
        self.items = {q: {"qa_type": it["qa_type"], "answer_bn": it["answer_bn"],
                          "house_id": it["house_id"]} for q, it in items.items()}
        self.excluded = excluded

    def scorer_item(self, q):
        it = self.items[q]
        return {"qa_id": q, "qa_type": it["qa_type"], "answer_bn": it["answer_bn"]}


def tiny_jpeg(seed):
    from PIL import Image
    im = Image.new("RGB", (24 + seed % 5, 16), ((seed * 37) % 255, (seed * 71) % 255, (seed * 13) % 255))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return buf.getvalue()


def selftest_world(ctx0, tmp):
    """A made-up Drive folder. Golds are names of the frozen lexicon and
    vocabulary; qa_ids start with H9; answers are scored by the gate-E scorer."""
    S, lex, voc = ctx0.S, ctx0.lexicon(), ctx0.vocab()
    norm = ctx0.normalise
    root = Path(tmp) / "drive"
    (root / "probes").mkdir(parents=True)
    an = load_module(ctx0.gate_e / "scripts" / "analysis.py", "analysis_gate_e_selftest")
    form = lambda kind, cid, lang="bn", i=0: voc.forms[kind][cid][lang][i]   # noqa: E731

    def item(q, t, gold, house):
        return {"qa_id": q, "image_id": q.rsplit("_", 1)[0], "house_id": house, "capture_type": "closeup",
                "qa_type": t, "question_bn": "প্রশ্ন %s" % t, "answer_bn": gold, "image_file": q.rsplit("_", 1)[0]
                + ".jpg"}

    # naming items: three objects whose English name, spelt in Bangla letters, the scorer
    # labels english_copy; cluster objects with siblings (the error budget needs en_unique);
    # two objects in no cluster
    skip = (PICTURE_FRAME, PAINTING, P1.BLANKET_BAG, P1.BAG)
    copies_bn, ec_objs = {}, []
    for o in sorted(lex.objects.values(), key=lambda o: o.name_id):
        if len(ec_objs) == 3:
            break
        if o.name_id in skip or not o.t1_en:
            continue
        probe = {"qa_id": "H90_0000_q1", "qa_type": "obj_name", "answer_bn": o.name_bn}
        for t in o.t1_en:
            a = an.transliterate(t)
            if S.score_item(probe, "N-BB", a, lex, voc)["label"] == "english_copy":
                copies_bn[o.name_id] = a
                ec_objs.append(o)
                break
    if len(ec_objs) < 2:
        raise Stop("selftest: fewer than two made-up English copies (%d)" % len(ec_objs))
    taken = {o.name_id for o in ec_objs}
    clus = sorted((o for o in lex.objects.values() if o.cluster_id and o.t1_en and o.en_unique
                   and lex.siblings_of(o.name_id) and o.name_id not in taken),
                  key=lambda o: o.name_id)[:6]
    plain = sorted((o for o in lex.objects.values() if not o.cluster_id and o.t1_en
                    and o.name_id not in skip and o.name_id not in taken),
                   key=lambda o: o.name_id)[:2]
    it, nm = {}, []
    for i, o in enumerate(ec_objs + clus + plain):
        k = "n%d" % (i + 1)
        it[k] = item("H9%d_%04d_q1" % (1 + i % 3, i + 1), "obj_name", o.name_bn, "H9%d" % (1 + i % 3))
        nm.append(k)
    it["nb"] = item("H93_0101_q1", "obj_name", lex.objects[P1.BLANKET_BAG].name_bn, "H93")
    it["nf"] = item("H92_0102_q1", "obj_name", lex.objects[PICTURE_FRAME].name_bn, "H92")
    it["nx"] = item("H92_0103_q1", "obj_name", lex.objects[clus[0].name_id].name_bn, "H92")   # excluded
    madeup = {"d192_1": "ঝঝঝঝ", "d192_2": "ঞঞঞঞ", "d192_3": "ঢঢঢঢ"}
    gen = {c: (f, b) for _k, c, f, b in S.genitive_only_forms(voc)}
    it["c5"] = item("H91_0015_q3", "obj_property", form("material", "plastic") + " " + madeup["d192_1"], "H91")
    it["c9"] = item("H93_0019_q3", "obj_property", form("material", "coconut") + " " + madeup["d192_2"], "H93")
    it["c8"] = item("H93_0018_q3", "obj_property", gen["plastic"][0] + " " + madeup["d192_3"], "H93")
    it["c7"] = item("H92_0017_q3", "obj_property", form("colour", "white"), "H92")
    it["c6"] = item("H93_0016_q3", "obj_property", gen["ceramic"][0], "H93")
    in_clus = [k for k in nm if lex.objects[S.resolve_gold(it[k], lex, voc).name_id].cluster_id]
    probes = {"P-CORE-CLOSED": ["c5", "c7"], "R-TEST": ["c5", "c6", "c8", "c9"],
              "P-NAME": nm + ["nb", "nf", "nx"], "P-CLUST": in_clus + ["nx"], "P-FCE": []}
    for p, ks in probes.items():
        (root / "probes" / ("%s.json" % p)).write_text(json.dumps({"items": [it[k] for k in ks]},
                                                                  ensure_ascii=False), "utf-8")
    exl = Path(tmp) / "excluded_items.csv"
    with open(exl, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["qa_id", "probes", "qa_type", "reason", "source", "added_at"])
        w.writerow([it["nx"]["qa_id"], "P-NAME|P-CLUST", "obj_name", "gold_wrong", "selftest", "selftest"])
    # pictures and their manifest
    img_dir = root / "data" / "images_1024"
    img_dir.mkdir(parents=True)
    mrows = []
    for i, k in enumerate(sorted(it)):
        f = it[k]["image_file"]
        if (img_dir / f).exists():
            continue
        data = tiny_jpeg(i + 1)
        (img_dir / f).write_bytes(data)
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            wdt, hgt = im.size
        mrows.append({"image_file": f, "sha256": hashlib.sha256(data).hexdigest(), "width": wdt, "height": hgt})
    write_csv(img_dir / "images_manifest.csv", mrows, ["image_file", "sha256", "width", "height"])
    # D192 inputs (as part 1's selftest)
    audit = root / "scored" / "asset9_stage2" / "audit"
    audit.mkdir(parents=True)
    ew = [{"kind": "material", "qa_type": "obj_property", "gold": it[k]["answer_bn"], "classes": "",
           "extra_words": madeup[f], "label_if_alone": "", "alone_scores_right": "False", "n_items": 1,
           "items_by_probe": "", "example_qa_id": it[k]["qa_id"]}
          for f, k in (("d192_1", "c5"), ("d192_2", "c9"), ("d192_3", "c8"))]
    ew.append({"kind": "colour", "qa_type": "obj_property", "gold": it["c7"]["answer_bn"], "classes": "white",
               "extra_words": "x", "label_if_alone": "", "alone_scores_right": "False", "n_items": 1,
               "items_by_probe": "", "example_qa_id": it["c7"]["qa_id"]})
    write_csv(audit / "extra_words.csv", ew, list(ew[0]))
    notes = [{"note": "genitive-only form", "source": k, "owner": c, "bn": f, "en": "", "how": "bare: %s" % b}
             for k, c, f, b in S.genitive_only_forms(voc)]
    write_csv(audit / "frozen_file_notes.csv", notes, ["note", "source", "owner", "bn", "en", "how"])
    en_rows = [{"string_id": "s%02d" % i, "question_bn_norm": norm(q), "question_en": "made-up question %d" % i}
               for i, q in enumerate(sorted({x["question_bn"] for x in it.values()}))]
    (root / "english").mkdir(parents=True)
    write_csv(root / "english" / "english_questions_final.csv", en_rows,
              ["string_id", "question_bn_norm", "question_en"])

    # ---- answers --------------------------------------------------------------------
    model_ids = {"qwen": "Qwen/Qwen3.5-4B", "gemma": "google/gemma-4-E4B-it", "gemini": "gemini-3.6-flash",
                 "qwen_ft_s42": "Qwen/Qwen3.5-4B+ft-s42", "qwen_ft_s43": "Qwen/Qwen3.5-4B+ft-s43",
                 "qwen_ft_s44": "Qwen/Qwen3.5-4B+ft-s44"}
    excluded = S.load_excluded(exl)
    copies_bn = {k: copies_bn[S.resolve_gold(it[k], lex, voc).name_id] for k in nm
                 if S.resolve_gold(it[k], lex, voc).name_id in copies_bn}

    def answer(m, cid, k):
        sp = S.condition_spec(cid)
        lang = sp.answer_lang or sp.question_lang
        i = it[k]
        g = S.resolve_gold(i, lex, voc)
        if g.kind == "naming":
            o = lex.objects[g.name_id]
            j = nm.index(k) if k in nm else 0
            pat = (j + HT5_MODELS.index(m) + HT5_CONDS.index(cid)) % 5 if m in HT5_MODELS else 0
            if k == "nf":
                return "" if m == "gemini" else o.name_bn
            if k == "nb":
                return lex.objects[P1.BAG].name_bn if m == "gemma" else o.name_bn
            sib = sorted(lex.siblings_of(o.name_id))
            if pat == 0:
                return o.name_bn if lang == "bn" else o.t1_en[0]
            if pat == 1 and sib:
                so = lex.objects[sib[0]]
                return so.name_bn if lang == "bn" else (so.t1_en or so.t1_bn)[0]
            if pat == 2 and k in copies_bn and lang == "bn":
                return copies_bn[k]
            if pat == 3:
                return "ছবি নেই"
            return "একটি কাঠের জিনিস"
        cid0 = sorted(g.classes)[0]
        return form(g.kind, cid0, "bn" if lang == "bn" else "en")

    special = {("gemma", "L-BB", "c5"): madeup["d192_1"], ("gemma", "L-BE", "c5"): "cane",
               ("gemma", "L-EE", "c5"): "cane", ("qwen", "L-BB", "c5"): madeup["d192_1"],
               ("qwen_ft_s42", "R-BB", "c6"): gen["ceramic"][1], ("qwen_ft_s42", "R-BB", "c8"): gen["plastic"][1],
               ("qwen_ft_s43", "R-BB", "c9"): madeup["d192_2"],
               ("gemini", "L-BB", "c7"): form("colour", "white") + " ও " + form("colour", "red")}
    plan = defaultdict(list)
    for m in HT5_MODELS:
        for c in HT5_CONDS:
            for k in nm + ["nb", "nf", "nx"]:
                plan[m].append((c, k))
        for c in ("L-BB", "L-BE", "L-EE"):
            plan[m].append((c, "c5"))
        plan[m].append(("L-BB", "c7"))
    plan["qwen"].append(("R-BB", "c5"))                                   # a copy of qwen L-BB
    for m in P1.FT_MODELS:
        for k in ("c5", "c6", "c8", "c9"):
            plan[m].append(("R-BB", k))
    root_scored = root / "scored" / "main"
    root_scored.mkdir(parents=True)
    all_recs = {}
    for m in MODELS:
        out, by_ck = [], {}
        for c, k in plan[m]:
            i = it[k]
            key = "|".join([model_ids[m], c, "exp-v1", i["qa_id"]])
            if m == "qwen" and c == "R-BB" and k == "c5":
                src = by_ck[("L-BB", "c5")]
                rec = dict(src, key=key, condition_id=c, mode="reuse", reused_from_key=src["key"])
            else:
                a = special.get((m, c, k), answer(m, c, k) if k in it else "")
                rec = S.score_item(i, c, a, lex, voc, excluded=excluded.get(i["qa_id"]))
                rec.update(key=key, model_key=m, model_id=model_ids[m], house_id=i["house_id"], stage="main",
                           prompt_version="exp-v1", mode="batch" if m == "gemini" else "hf")
            out.append(rec)
            by_ck[(c, k)] = rec
        with open(root_scored / ("%s_scored.jsonl" % m), "w", encoding="utf-8") as fh:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        all_recs[m] = out
    shas = {m: sha256_file(root_scored / ("%s_scored.jsonl" % m)) for m in MODELS}
    # the frame (made-up categories; real runs take analysis.py budget-items)
    pclust = {it[k]["qa_id"] for k in probes["P-CLUST"]}
    frame = []
    for m in HT5_MODELS:
        nbb = {r["qa_id"]: r for r in all_recs[m] if r["condition_id"] == "N-BB"}
        nbe = {r["qa_id"]: r for r in all_recs[m] if r["condition_id"] == "N-BE"}
        for q in sorted(pclust):
            r = nbb.get(q)
            if r is None or r["gold_status"] != "ok" or r["strict"] == 1:
                continue
            frame.append({"model_key": m, "qa_id": q, "house_id": r["house_id"],
                          "category": "A4" if nbe.get(q, {}).get("strict") == 1 else "C", "condition_id": "N-BB"})
    st = root / "results" / "stats_inputs"
    st.mkdir(parents=True)
    write_csv(st / "error_budget_items.csv", frame, FRAME_COLUMNS)
    write_json(st / "error_budget_items_manifest.json", {
        "utc": utc_now(), "script_version": ANALYSIS_VERSION, "mode": "real", "stage": "main", "rows": len(frame),
        "scored_files": [{"file": str(root_scored / ("%s_scored.jsonl" % m)), "sha256": shas[m]} for m in MODELS],
        "inputs": {"lexicon.csv": sha256_file(ctx0.lexicon_path), "excluded_items.csv": sha256_file(exl),
                   "scoring.py": sha256_file(ctx0.gate_e / "src" / "scoring.py")}})
    # HT-3: a made-up final folder (names are the gold or another object)
    h3 = root / "human" / "HT-3"
    fin = h3 / "score_final_20261023T000000Z"
    fin.mkdir(parents=True)
    nref = ["n1", "n2", "nf"]
    other = lex.objects[plain[0].name_id]
    names_rows = []
    for k in nref:
        i = it[k]
        for role in ("N1", "N2"):
            if k == "n2":
                nmv = other.name_bn
            elif k == "nf" and role == "N1":
                nmv = lex.objects[PAINTING].name_bn
            else:
                nmv = i["answer_bn"]
            rec = S.score_item(i, "N-BB", nmv, lex, voc)
            names_rows.append({"role": role, "code": "%s-%03d" % (role, nref.index(k) + 1), "qa_id": i["qa_id"],
                               "house_id": i["house_id"], "excluded": "no", "name": nmv, "label": rec["label"],
                               "strict": rec["strict"], "named_ids": "|".join(rec["named_ids"] or [])})
    write_csv(fin / "ht3_scored_names.csv", names_rows, list(names_rows[0]))
    write_csv(fin / "ht3_same_other_object.csv", [{"qa_id": it["n2"]["qa_id"], "label_N1": "other_wrong",
                                                    "label_N2": "other_wrong", "named_ids": other.name_id}],
              ["qa_id", "label_N1", "label_N2", "named_ids"])
    (fin / "ht3_report.txt").write_text("made-up report\n", "utf-8")
    js = {"mode": "final", "reportable": True, "scorer_version": S.SCORER_VERSION,
          "scorer_sha256": sha256_file(ctx0.gate_e / "src" / "scoring.py"),
          "normalise_sha256": sha256_file(ctx0.gate_e / "src" / "normalise.py"),
          "lexicon_sha256": sha256_file(ctx0.lexicon_path), "vocab_sha256": sha256_file(ctx0.vocab_path),
          "excluded_sha256": sha256_file(exl), "excluded_rows": 1, "git_head": GATE_F_COMMIT,
          "gate_f_commit": GATE_F_COMMIT, "key_sha256": "k" * 64, "received_sha256": {"N1": "a", "N2": "b"},
          "identity_checks": [{"tag": t, "file": f, "ok": True, "detail": "identical"} for t, f in (
              ("gate-F", "src/scoring.py"), ("gate-F", "src/normalise.py"), ("gate-A", "lexicon/lexicon.csv"),
              ("gate-A", "lexicon/closed_vocab.csv"))],
          "primary_role": "N1", "lexicon_writer": "N2",
          "summary": {"n_drawn": 3, "n_excluded": 0, "n_counted": 3, "excluded_qa_ids": [],
                      "unscoreable_qa_ids": []}}
    write_json(fin / "ht3_agreement.json", js)
    write_json(h3 / "ht3_manifest.json", {"state": {"draw": {"key_sha256": "k" * 64},
                                                    "received": {"N1": {"sha256": "a"}, "N2": {"sha256": "b"}},
                                                    "scores": [{"folder": fin.name, "mode": "final"}]}})
    # HT-6 pairs, D279 change file, the guide
    sib_items = [k for k in nm if lex.siblings_of(S.resolve_gold(it[k], lex, voc).name_id)]
    k0 = sib_items[0]
    s0 = sorted(lex.siblings_of(S.resolve_gold(it[k0], lex, voc).name_id))[0]
    (root / "human" / "HT-6").mkdir(parents=True)
    write_csv(root / "human" / "HT-6" / "ht6_swaps.csv", [{"seq": 1, "qa_id": it[k0]["qa_id"], "removed_id": s0,
                                                           "reason": "name"}], ["seq", "qa_id", "removed_id", "reason"])
    d279 = root_scored / "sensitivity_D279"
    d279.mkdir(parents=True)
    some = [r for r in all_recs["gemma"] if r["condition_id"] == "N-BB" and r.get("strict") == 0][:1]
    write_csv(d279 / "changes.csv", [{"rule": "a", "model_key": "gemma", "condition_id": "N-BB",
                                      "qa_id": r["qa_id"], "key": r["key"]} for r in some],
              ["rule", "model_key", "condition_id", "qa_id", "key"])
    write_json(d279 / "summary.json", {"scored_sha256": shas})
    g = root / "human" / "HT-5" / "guide"
    g.mkdir(parents=True)
    (g / "HT5_auditor_guide.md").write_text("# made-up guide for the selftest\n", "utf-8")
    cfg = dict(REAL)
    cfg.update({"git_checks": False, "stats_world": False, "excluded_path": exl, "excluded_rows": 1,
                "strata_n": {"C": 4, "english_copy": 2, "sibling": 3}, "d192_expect": None,
                "ht3": {"n_drawn": 3, "n_counted": 3, "excluded": [], "rows": 6, "lexicon_writer": "N2",
                        "primary_role": "N1"},
                "images_manifest_sha": None, "images_manifest_rows": None, "d279_rows": None, "n_boot": 200,
                "gate_e_pins": {k: (v if k in ("src/scoring.py",) else None) for k, v in REAL["gate_e_pins"].items()},
                "drive_pins": {}, "ht6_pin": None,
                "d192_anchors": {"d192_1": it["c5"]["qa_id"], "d192_2": it["c9"]["qa_id"],
                                 "d192_3": it["c8"]["qa_id"], "d192_4": it["c7"]["qa_id"]}})
    if not sha256_file(ctx0.gate_e / "src" / "scoring.py").startswith(REAL["gate_e_pins"]["src/scoring.py"]):
        cfg["gate_e_pins"]["src/scoring.py"] = None          # a sandbox may hold a stand-in
    world_items = {}
    for p in probes.values():
        for k in p:
            world_items[it[k]["qa_id"]] = it[k]
    data = defaultdict(dict)
    for m, recs in all_recs.items():
        for r in recs:
            data[(m, r["condition_id"])][r["qa_id"]] = r
    plantd = {"d192_rows": {("gemma", "L-BB", it["c5"]["qa_id"]), ("gemma", "L-BE", it["c5"]["qa_id"]),
                            ("gemma", "L-EE", it["c5"]["qa_id"]), ("qwen", "L-BB", it["c5"]["qa_id"]),
                            ("qwen_ft_s42", "R-BB", it["c6"]["qa_id"]), ("qwen_ft_s42", "R-BB", it["c8"]["qa_id"]),
                            ("qwen_ft_s43", "R-BB", it["c9"]["qa_id"]), ("gemini", "L-BB", it["c7"]["qa_id"]),
                            ("gemma", "N-BB", it["nb"]["qa_id"]), ("gemma", "N-BE", it["nb"]["qa_id"]),
                            ("gemma", "N-BBs", it["nb"]["qa_id"])},
              "ht3_items": {it["n2"]["qa_id"], it["nf"]["qa_id"]}, "excluded_q": it["nx"]["qa_id"],
              "frame": frame}
    return root, cfg, world_items, data, excluded, plantd


def fill_sheet(path_in, path_out, decide):
    """A made-up auditor: decide(code, row) -> (verdict, form, kind)."""
    from openpyxl import load_workbook
    wb = load_workbook(str(path_in))
    ws = wb["Sheet"]
    cols = [c.value for c in ws[1]]
    for row in ws.iter_rows(min_row=2):
        d = {cols[i]: row[i].value for i in range(len(cols))}
        v, f, k = decide(d["code"], d)
        row[cols.index("verdict")].value = v
        row[cols.index("missing_form")].value = f
        row[cols.index("form_kind")].value = k
    wb.save(str(path_out))


def cmd_selftest(_ctx, args):
    say("Scoring and audit, part 2: selftest on made-up answers, %s, run %s" % (SCRIPT_VERSION, utc_now()))
    tmp = Path(tempfile.mkdtemp(prefix="scoring_p2_selftest_"))
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok)))
        say("  %s %s%s" % ("PASS" if ok else "FAIL", name, "" if not detail else ": " + str(detail)))

    try:
        cfg0 = dict(REAL, git_checks=False, stats_world=False)
        ctx0 = Ctx2(tmp / "none", args.gate_e, None, None, cfg0)
        root, cfg, witems, data, excluded, plantd = selftest_world(ctx0, tmp)
        ctx = Ctx2(root, args.gate_e, None, None, cfg)
        ST = ctx.stats()
        world = MiniWorld(ctx, witems, excluded)
        recs = ST.Records(data=dict(data), stage="main", scorer_version=ctx.S.SCORER_VERSION,
                          prompt_version="exp-v1", files=[], manifests=[], seeds=())
        budgets = defaultdict(lambda: {"categories": {}})
        for fr in plantd["frame"]:
            budgets[fr["model_key"]]["categories"][fr["qa_id"]] = fr["category"]
        ctx.mini = (world, recs, budgets)
        P = ht5_paths(ctx)
        say("Made-up project in %s" % root)
        say("---- check, HT-3, frame ----")
        check("check: all lines pass", cmd_check(ctx, args) == 0)
        check("ht3-check: the made-up final folder passes", cmd_ht3_check(ctx, SimpleNamespace(folder=None)) == 0)
        check("the frame check passes", frame_report(ctx) == 0)
        say("---- draw ----")
        ns = SimpleNamespace(auditors_cleared=True, au1_roles="HT-4", au2_roles="none")
        try:
            cmd_ht5_draw(ctx, SimpleNamespace(auditors_cleared=False, au1_roles="none", au2_roles="none"))
            check("ht5-draw refuses without --auditors-cleared", False)
        except Stop:
            check("ht5-draw refuses without --auditors-cleared", True)
        check("ht5-draw: runs", cmd_ht5_draw(ctx, ns) == 0)
        key = read_csv(P["key"])
        samp = [k for k in key if k["in_sample"] == "yes"]
        nst = Counter(k["stratum"] for k in samp)
        want = cfg["strata_n"]
        check("sample: strata sizes as planned (or all the frame had)",
              all(0 < nst[s] <= want[s] for s in STRATA), dict(nst))
        check("sample: C rows are N-BB answers of frame category C", all(
            k["condition_id"] == "N-BB" and budgets[k["model_key"]]["categories"].get(k["qa_id"]) == "C"
            for k in samp if k["stratum"] == "C"))
        check("sample: english_copy and sibling rows carry that scorer label", all(
            k["scorer_label"] == k["stratum"] for k in samp if k["stratum"] != "C"))
        check("sample: no excluded item and no answer twice", not any(k["qa_id"] == plantd["excluded_q"] for k in key)
              and len({k["key"] for k in key}) == len(key))
        d1 = {(k["model_key"], k["condition_id"], k["qa_id"]) for k in key if "d192" in k["pools"].split("|")}
        want_d1 = {x for x in plantd["d192_rows"] if not (x[0] == "qwen" and x[1] == "R-BB")}
        check("D192 pool: the planted answers, the copy left out", want_d1 <= d1
              and not any(x[0] == "qwen" and x[1] == "R-BB" for x in d1), "%d rows" % len(d1))
        h3 = {k["qa_id"] for k in key if "ht3" in k["pools"].split("|")}
        check("HT-3 pool: both look-first items on the sheet", h3 == plantd["ht3_items"], sorted(h3))
        check("codes: two orders, unique codes", len({k["code_au1"] for k in key}) == len(key)
              and len({k["code_au2"] for k in key}) == len(key)
              and [k["key"] for k in sorted(key, key=lambda k: k["code_au1"])]
              != [k["key"] for k in sorted(key, key=lambda k: k["code_au2"])])
        check("ht5-draw: run again, nothing replaced", cmd_ht5_draw(ctx, ns) == 0)
        with zipfile.ZipFile(P["zip"]["AU1"]) as z:
            names = z.namelist()
            xl = z.read("HT5_AU1/HT5_sheet_AU1.xlsx")
            page = z.read("HT5_AU1/HT5_view_AU1.html").decode("utf-8")
            k1 = sorted(key, key=lambda k: k["code_au1"])[0]
            img = z.read("HT5_AU1/images/%s.jpg" % k1["code_au1"])
        src = (root / "data" / "images_1024" / k1["image_file"]).read_bytes()
        check("package: sheet, page, guide and one picture per row", "HT5_AU1/HT5_auditor_guide.md" in names
              and sum(1 for n in names if n.startswith("HT5_AU1/images/")) == len(key))
        check("package: pictures byte-identical to the source", img == src)
        texts = " ".join(t for _w, t, _a in workbook_texts(xl))
        check("sheet: no qa_id, model, condition or scorer-label column", not ID_RE.search(texts)
              and not ID_RE.search(page) and "scorer_label" not in texts and not COND_RE.search(page))
        from openpyxl import load_workbook
        wb = load_workbook(io.BytesIO(xl))
        check("sheet: tabs", wb.sheetnames == ["Sheet", "Read me", "Lexicon", "Vocabulary"], wb.sheetnames)
        ws = wb["Sheet"]
        cols = [c.value for c in ws[1]]
        sib_cells = [r[cols.index("siblings")].value for r in ws.iter_rows(min_row=2)
                     if r[cols.index("row_type")].value == "naming"]
        lex = ctx.lexicon()
        allt1 = True
        for k in key:
            if k["row_type"] == "naming":
                o = lex.objects[k["gold_name_id"]]
                for s in lex.siblings_of(o.name_id):
                    so = lex.objects[s]
                    if not any(all(n in (c or "") for n in so.t1_bn + so.t1_en) for c in sib_cells):
                        allt1 = False
        check("sheet: every Tier 1 name of every sibling is listed", allt1)
        say("---- two made-up auditors ----")
        by1 = {k["code_au1"]: k for k in key}
        by2 = {k["code_au2"]: k for k in key}
        ec = [k for k in key if k["scorer_label"] == "english_copy" and k["requested_lang"] == "bn"]
        flips = {k["key"] for k in samp if not ec or k["key"] != ec[0]["key"]}
        flips = set(sorted(flips)[:3])
        vform = ""
        if ec:
            blank1 = {r["code"]: r for r in P1.read_workbook(P["blank"]["AU1"], "Sheet", SHEET_COLS)}
            vform = blank1[ec[0]["code_au1"]]["model_answer"]

        def decide_for(by, who):
            def decide(c, d):
                k = by[c]
                if ec and k["key"] == ec[0]["key"]:
                    return "missing_variant", vform, "tier1"
                if who == "AU2" and k["key"] in flips:
                    return "not_visible", None, None
                if k["row_type"] != "naming":
                    return "ambiguous", None, None
                return "model_wrong", None, None
            return decide
        P["received"].mkdir(parents=True, exist_ok=True)
        fill_sheet(P["blank"]["AU1"], P["returned"]["AU1"],
                   lambda c, d: ("missing_variant", None, None) if c == "S001" else decide_for(by1, "AU1")(c, d))
        fill_sheet(P["blank"]["AU2"], P["returned"]["AU2"], decide_for(by2, "AU2"))
        check("ht5-receive: a broken sheet is sent back, nothing recorded", cmd_ht5_receive(ctx, args) == 1
              and not [r for r in P1.receipts(P) if r["who"] == "AU1"])
        fill_sheet(P["blank"]["AU1"], P["returned"]["AU1"], decide_for(by1, "AU1"))
        check("ht5-receive: both sheets, consensus package written", cmd_ht5_receive(ctx, args) == 0
              and P["cons_zip"].exists())
        with zipfile.ZipFile(P["cons_zip"]) as z:
            cx = z.read("HT5_consensus/HT5_consensus.xlsx")
        wb = load_workbook(io.BytesIO(cx))
        ws = wb["Consensus"]
        cols = [c.value for c in ws[1]]
        n_rows = 0
        for row in ws.iter_rows(min_row=2):
            n_rows += 1
            row[cols.index("consensus_verdict")].value = "model_wrong"
        cpath = P["cons_returned"]
        wb.save(str(cpath))
        check("consensus sheet: only the differing rows", n_rows == len(flips), n_rows)
        check("ht5-consensus: files written", cmd_ht5_consensus(ctx, args) == 0)
        rows = read_csv(P["audit"])
        check("ht5_audit.csv: D238 columns and one row per sample answer", list(rows[0]) == list(ST.HT5_COLUMNS)
              and len(rows) == len(samp))
        forms = read_csv(P["forms"])
        check("TEXT_ht5_forms.csv: the agreed form", len(forms) == (1 if ec else 0)
              and (not ec or forms[0]["form"] == ctx.normalise(vform)))
        check("ht5-read: passes", cmd_ht5_read(ctx, args) == 0)
        rd = json.loads(P["reading"].read_text("utf-8"))
        check("ht5-read: stats.ht5_stats ran", rd["ht5_stats"]["n"] == len(samp))
        say("---- lexicon sensitivity ----")
        check("sens-template: written", cmd_sens_template(ctx, args) == 0 and P["sens_template"].exists()
              and P["sens_sheet"].exists())
        with open(P["sens_template"], encoding="utf-8-sig") as fh:
            check("template: header only", fh.read().strip() == ",".join(ST.SENS_COLUMNS))
        srows = []
        if ec:
            srows.append({"row_id": "S01", "source": "HT-5", "op": "add_term", "target": "lexicon",
                          "target_id": ec[0]["gold_name_id"], "tier": "1", "lang": "bn", "form": vform,
                          "note": "selftest"})
        srows.append({"row_id": "S02", "source": "D192", "op": "item_any_class", "target": "",
                      "target_id": cfg["d192_anchors"]["d192_4"], "tier": "", "lang": "", "form": "",
                      "note": "selftest"})
        write_csv(P["sens"], srows, ST.SENS_COLUMNS)
        check("sens-check: made-up rows pass", cmd_sens_check(ctx, args) == 0)
        sc = json.loads(P["sens_check"].read_text("utf-8"))
        check("sens-check: answers change", sc["info"].get("changed_total", 0) >= 1,
              sc["info"].get("changed_total"))
        bad_rows = srows + [{"row_id": "S03", "source": "HT-5", "op": "add_term", "target": "lexicon",
                             "target_id": ec[0]["gold_name_id"] if ec else key[0]["gold_name_id"], "tier": "1",
                             "lang": "bn", "form": "ফেলুনি", "note": "no auditor marked this"}]
        write_csv(P["sens"], bad_rows, ST.SENS_COLUMNS)
        check("sens-check: an HT-5 form no auditor agreed on fails", cmd_sens_check(ctx, args) == 1)
        say("---- locks and guards ----")
        wb = load_workbook(str(P["returned"]["AU1"]))
        wb["Sheet"].cell(row=2, column=SHEET_COLS.index("note") + 1).value = "changed later"
        wb.save(str(P["returned"]["AU1"]))
        try:
            cmd_ht5_receive(ctx, args)
            check("a returned sheet that changed after its receipt stops the run", False)
        except Stop:
            check("a returned sheet that changed after its receipt stops the run", True)
        try:
            cmd_budget_items(ctx, args)
            check("budget-items refuses once an HT-5 draw exists", False)
        except Stop:
            check("budget-items refuses once an HT-5 draw exists", True)
        try:
            ctx.guard(ctx.gate_e / "x.txt")
            check("no write inside the gate-E checkout", False)
        except Stop:
            check("no write inside the gate-E checkout", True)
        check("leak scan finds an id, a model and a condition", len(leak_problems([
            ("a", "H12_0001_q1", False), ("b", "Gemma said", False), ("c", "N-BBs", False),
            ("d", "H12_0001_q1 in an answer", True), ("e", "Qwen in an answer", True)])) == 4)
    except Stop as exc:
        check("selftest stopped", False, str(exc))
    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    bad = [n for n, ok in results if not ok]
    say("SELFTEST: %d of %d checks PASS%s" % (len(results) - len(bad), len(results),
                                             "" if not bad else "; FAIL: " + "; ".join(bad)))
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# 13. main
# ---------------------------------------------------------------------------

def main(argv=None):
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description="Scoring and audit, part 2 (D280 to D288)")
    ap.add_argument("--root", default="/content/drive/MyDrive/BanglaIndoorVQA", help="the Drive project folder")
    ap.add_argument("--gate-e", default="/content/bivqa_gateE", help="the detached checkout of the tag gate-E")
    ap.add_argument("--gate-f", default="/content/bivqa_gateF", help="the detached checkout of the tag gate-F")
    ap.add_argument("--main", default=str(here.parents[1]), help="the main clone")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="read-only input check (Cell 0)")
    st = sub.add_parser("selftest", help="the chain on made-up answers (Cell 1)")
    st.add_argument("--keep-tmp", action="store_true")
    h3 = sub.add_parser("ht3-final", help="HT-3 final scores from the gate-F checkout (Cell 2)")
    h3.add_argument("--again", help="reason for scoring a second time (needs a decision row)")
    h3.add_argument("--folder", help=argparse.SUPPRESS)
    hc = sub.add_parser("ht3-check", help="check an HT-3 final folder")
    hc.add_argument("--folder", help="default: the newest human/HT-3/score_final_*")
    sub.add_parser("budget-items", help="the HT-5 frame from the gate-E checkout (Cell 3)")
    dr = sub.add_parser("ht5-draw", help="the HT-5 sample, pools and packages (Cell 4)")
    dr.add_argument("--auditors-cleared", action="store_true",
                    help="both auditors confirmed the rules of D281")
    dr.add_argument("--au1-roles", required=True, help="earlier roles of auditor 1, e.g. 'HT-4' or 'none'")
    dr.add_argument("--au2-roles", required=True, help="earlier roles of auditor 2")
    sub.add_parser("ht5-receive", help="receipts and the consensus package (Cell 5)")
    sub.add_parser("ht5-consensus", help="the consensus and ht5_audit.csv (Cell 6)")
    sub.add_parser("ht5-read", help="the HT-5 reading (Cell 7)")
    sub.add_parser("sens-template", help="the empty lexicon_sensitivity.csv and the worksheet (Cell 8)")
    sub.add_parser("sens-check", help="check lexicon_sensitivity.csv (Cell 9)")
    sub.add_parser("status", help="what exists (Cell 10)")
    args = ap.parse_args(argv)
    try:
        if args.command == "selftest":
            return cmd_selftest(None, args)
        ctx = Ctx2(args.root, args.gate_e, args.gate_f, args.main, real_cfg())
        return {"check": cmd_check, "ht3-final": cmd_ht3_final, "ht3-check": cmd_ht3_check,
                "budget-items": cmd_budget_items, "ht5-draw": cmd_ht5_draw, "ht5-receive": cmd_ht5_receive,
                "ht5-consensus": cmd_ht5_consensus, "ht5-read": cmd_ht5_read,
                "sens-template": cmd_sens_template, "sens-check": cmd_sens_check,
                "status": cmd_status}[args.command](ctx, args)
    except Stop as exc:
        say("STOP: %s" % exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
