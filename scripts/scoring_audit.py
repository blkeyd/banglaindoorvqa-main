#!/usr/bin/env python3
"""
scoring_audit.py  (Scoring and audit, part 1, stage 1)
BanglaIndoorVQA main paper. Design v2.1 at the tag gate-E, DESIGN_ADDENDUM K onward.

Stage 1 scores every stage-main cache with the gate-E scorer and reads what
needs no human: the checks of the scored files, the free check of the 738
copied base answers (addendum L10), the D192 and D194 counts with the open
issues, the D159 comparison against main, and the 10-item sample.
Stage 2 (the HT-4 sheet, receipt, consensus, reading and Gate F) is added to
this file after src/stats.py has been read (D238 makes stats.ht4_stats the one
pass rule).

Commands (one notebook cell each; every command prints ids and counts only)
  check     read-only. One PASS or FAIL line per input.                  Cell 0
  selftest  the whole chain on made-up answers in a temporary folder.    Cell 1
  score     one gate-E `asset9_scorer.py score` command per cache (D270). Cell 2
  verify    the scored files against their caches, and the free check.   Cell 3
  counts    D192, D194 and the open-issue counts (D272).                 Cell 4
  d159      the 97 D5 pairs: main against the pilot (D273).               Cell 5
  sample10  ten seeded-random scored answers for the 10-item check.       Cell 6
  status    what exists so far.                                          Cell 7

Rules built in (decision rows of DESIGN_ADDENDUM O)
  D270  one score command per cache, every probe of its conditions as a
        repeated --probe; a read-only probe-to-condition check before it and a
        one-record-per-answer check after it; a scored file is never replaced
        once an HT-4 sheet exists.
  D271  every score command runs from the detached gate-E checkout with
        PYTHONDONTWRITEBYTECODE=1; this script runs from main and imports
        scoring, fc_options and normalise from that checkout (path and sha
        checked). Nothing is written inside a Git tree.
  D272  the counting rules of the D192 findings, D194 and the open issues.
  D273  D159: main against the pilot on the 97 pairs, the repeat and the Batch
        test beside it.

Files (Drive project folder D; nothing in the repo)
  D/scored/main/<model>_scored.jsonl and _manifest.json   (the scorer writes them)
  D/scored/main/_records/   check_*.json, verify_*.json, runs.jsonl, logs/
  D/scored/main/reading/    counts and D159 files; files with answer text say so
"""

import sys
sys.dont_write_bytecode = True   # never write __pycache__ into the gate-E checkout (D252)

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_VERSION = "scoring-audit-p1 2026-10-22"
SEED = 42

# ---------------------------------------------------------------------------
# 0. Fixed values. Every number below is in PROJECT_STATE, DEPENDENCY_MAP or
#    DESIGN_ADDENDUM; none is a result.
# ---------------------------------------------------------------------------

GATE_E_COMMIT = "5db2bfff3fd3e31b83f8c4865d37388f25f931f0"
SCORER_VERSION = "0.2.2-stage2"
ASSET9_VERSION = "asset9-stage2 2026-10-16"
FC_RULES_VERSION = "asset11-v1 2026-10-10"

EXP_L = ("L-BB", "L-BE", "L-EB", "L-EE", "L-BB0", "L-EE0")
EXP_N = ("N-BB", "N-BE", "N-BBs", "N-FCB", "N-FCB0", "N-FCE", "N-FCE0")
EXP_R = ("R-BB", "R-EE", "R-BB0")
FT_MODELS = ("qwen_ft_s42", "qwen_ft_s43", "qwen_ft_s44")
MODELS = ("qwen", "gemma", "gemini") + FT_MODELS
PROBE_ORDER = ("P-CORE-CLOSED", "R-TEST", "P-NAME", "P-CLUST", "P-FCE")
COPY_SOURCE = {"R-BB": "L-BB", "R-EE": "L-EE", "R-BB0": "L-BB0"}

REAL = {
    "conditions_of": {"qwen": EXP_L + EXP_N + EXP_R, "gemma": EXP_L + EXP_N,
                      "gemini": EXP_L + EXP_N,
                      "qwen_ft_s42": EXP_R, "qwen_ft_s43": EXP_R, "qwen_ft_s44": EXP_R},
    "records_of": {"qwen": 15947, "gemma": 11600, "gemini": 11600,
                   "qwen_ft_s42": 4347, "qwen_ft_s43": 4347, "qwen_ft_s44": 4347},
    "copies": {"qwen": 738},
    "adapter_sha": {"qwen_ft_s42": "2746ab6cb568", "qwen_ft_s43": "75f4cf6d3a20",
                    "qwen_ft_s44": "7b2fcbc78868"},
    "probe_sizes": {"P-CORE-CLOSED": 1118, "R-TEST": 1449, "P-NAME": 1000,
                    "P-CLUST": 492, "P-FCE": 454},
    "scored_sizes": {"P-CORE-CLOSED": 1109, "R-TEST": 1435, "P-NAME": 991,
                     "P-CLUST": 483, "P-FCE": 446},
    "options_sets": {"bn": 492, "en": 454},
    "excluded_rows": 28,
    "lexicon": {"objects": 231, "terms": 864},
    "vocab": "room 4, material 16, colour 14, number 29",
    "gate_e_pins": {"src/scoring.py": "8563fa05f606",
                    "scripts/asset9_scorer.py": "c16f6c948589",
                    "lexicon/lexicon.csv": "73b6bc48f1bd",
                    "lexicon/closed_vocab.csv": "de36f2905e43",
                    "probes/excluded_items.csv":
                        "8c51763fad6e3c48f0ae1e137573d00e1d0e79274bf959d9bb8e0bca96b37cdb",
                    "configs/conditions.yaml": "40897bfc"},
    "drive_pins": {"probes/P-CORE-CLOSED.json": "ea242d5b8ade",
                   "probes/R-TEST.json": "30cd10956e56",
                   "probes/P-NAME.json": "fc4bfb1dbda1",
                   "probes/P-CLUST.json": "7759dcce19e4",
                   "probes/P-FCE.json":
                       "807e987b4b656586ee2955efb421e8e6cd0aae27790a11a4c4f3f4f7210479ae",
                   "probes/fc_options.jsonl":
                       "5288ba63033083b10beab8eb5949f3f6a2ab3968a4baf8eb8bb2e1b0fdcf2e96",
                   # sha of the copies attached in the Scoring chat
                   "scored/asset9_stage2/audit/extra_words.csv": "62183f0bea5b",
                   "scored/asset9_stage2/audit/frozen_file_notes.csv": "990935c55e24"},
    # Drive scored/pilot/ files, found by sha (addendum F8)
    "pilot_files": {"pilot_gemini": "7d096b2c2791", "pilot_qwen": "9d0080c33c79",
                    "pilot_gemma": "649eb38351a0", "d5repeat_gemini": "8b4bc1e2fd5c",
                    "batchtest_gemini": "2ca0f7d9835e"},
    "pilot_records": {"pilot_gemini": 560, "pilot_qwen": 560, "pilot_gemma": 560,
                      "d5repeat_gemini": 100, "batchtest_gemini": 100},
    "d159_pairs": {"L-BB": 58, "N-BB": 39},
    # B7 rows anchored on the example qa_id of extra_words.csv (D272)
    "d192_anchors": {"d192_1": "H25_0011_q3", "d192_2": "H23_0016_q3",
                     "d192_3": "H53_0051_q3", "d192_4": "H06_0018_q3"},
    "named_refusals": ("H06_0041_q1", "H45_0006_q1"),
    "git_checks": True,
}

KEEP_FROM = {"qwen": "pilot_qwen", "gemma": "pilot_gemma", "gemini": "pilot_gemini",
             "qwen_ft_s42": "pilot_qwen", "qwen_ft_s43": "pilot_qwen",
             "qwen_ft_s44": "pilot_qwen"}
KEEP_EXTRA = ("mode", "ts_utc", "model_version", "model_revision", "truncated",
              "finish_reason", "block_reason", "reused_from_key", "adapter_run",
              "adapter_sha256")
CACHE_REQUIRED = ("key", "model_id", "condition_id", "prompt_version", "qa_id", "qa_type",
                  "question_lang", "answer_lang", "forced_choice", "has_image",
                  "answer_text", "stage")
ENGLISH_FORMS_D192 = {"d192_1": ("cane",)}    # the English word B7 names for row 1
BLANKET_BAG = "N_5ad88fb244"                   # lexicon object of B7 row 7
BAG = "N_9973b18e2c"                           # the object ব্যাগ
RONG = "রঙ"                                    # the liquid form B7 row 5 names
WRONG_LABELS_OK = {"tier1", "correct"}


# ---------------------------------------------------------------------------
# 1. Small helpers
# ---------------------------------------------------------------------------

def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seeded_rank(*parts):
    return hashlib.sha256(("|".join([str(SEED), "scoring-audit"] + [str(p) for p in parts]))
                          .encode("utf-8")).hexdigest()


def pin_ok(sha, pin):
    return pin is None or sha.startswith(pin)


def short(sha):
    return sha[:12] if sha else "-"


def pct(k, n):
    return "%.1f%%" % (100.0 * k / n) if n else "-"


def wilson(k, n, z=1.959963984540054):
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else
                            (json.dumps(r.get(c), ensure_ascii=False)
                             if isinstance(r.get(c), (list, dict)) else r.get(c)))
                        for c in columns})


def read_csv(path):
    with open(str(path), encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def read_jsonl(path):
    """Records of a JSONL file: (records, unreadable lines, incomplete last line, sha256)."""
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8")
    lines = text.split("\n")
    recs, bad, incomplete = [], 0, 0
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            recs.append(json.loads(line))
        except ValueError:
            if n == len(lines) and not text.endswith("\n"):
                incomplete += 1
            else:
                bad += 1
    return recs, bad, incomplete, hashlib.sha256(raw).hexdigest()


def git(repo, *args):
    p = subprocess.run(["git", "-C", str(repo)] + list(args), capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def tree_clean(repo):
    rc, out, _err = git(repo, "status", "--porcelain", "--untracked-files=all")
    return rc == 0 and out == "", out


class Lines:
    """PASS / FAIL / INFO lines, printed and kept for the record file."""

    def __init__(self):
        self.rows = []

    def add(self, code, ok, text):
        self.rows.append({"code": code, "result": "PASS" if ok else "FAIL", "text": text})
        print("%s %s  %s" % (code, "PASS" if ok else "FAIL", text))
        return ok

    def info(self, code, text):
        self.rows.append({"code": code, "result": "INFO", "text": text})
        print("%s INFO  %s" % (code, text))

    def failed(self):
        return [r for r in self.rows if r["result"] == "FAIL"]


class Stop(Exception):
    """A rule of this script is broken; the message says which."""


# ---------------------------------------------------------------------------
# 2. Context: paths, configuration and the gate-E modules
# ---------------------------------------------------------------------------

def import_gate_e(gate_e):
    """scoring, fc_options, normalise and asset9_scorer from the gate-E checkout."""
    src = str((Path(gate_e) / "src").resolve())
    if src in sys.path:
        sys.path.remove(src)
    sys.path.insert(0, src)
    for name in ("scoring", "fc_options", "normalise"):
        mod = sys.modules.get(name)
        if mod is not None and not str(Path(mod.__file__).resolve()).startswith(src):
            raise Stop("module %s is already loaded from %s, not from the gate-E checkout"
                       % (name, mod.__file__))
    import scoring as S          # noqa: E402
    import fc_options as FC      # noqa: E402
    import normalise as N        # noqa: E402
    for mod in (S, FC, N):
        if not str(Path(mod.__file__).resolve()).startswith(src):
            raise Stop("%s was imported from %s, not from %s" % (mod.__name__, mod.__file__, src))
    path = Path(gate_e) / "scripts" / "asset9_scorer.py"
    spec = importlib.util.spec_from_file_location("asset9_scorer_gate_e", str(path))
    A9 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A9)
    if A9.S is not S:
        raise Stop("asset9_scorer.py loaded another copy of scoring.py")
    return S, FC, N, A9


class Ctx:
    def __init__(self, root, gate_e, main, cfg):
        self.root = Path(root).resolve()
        self.gate_e = Path(gate_e).resolve()          # the score command runs with this cwd
        self.main = Path(main).resolve() if main else None
        self.cfg = cfg
        self.S, self.FC, self.N, self.A9 = import_gate_e(self.gate_e)
        self.normalise = self.N.normalise
        self._lex = self._voc = self._excluded = None
        self._probes = None
        self._conditions = None
        self._options = None

    # ---- paths -------------------------------------------------------------
    @property
    def scored_dir(self):
        return self.root / "scored" / "main"

    @property
    def records_dir(self):
        return self.scored_dir / "_records"

    @property
    def reading_dir(self):
        return self.scored_dir / "reading"

    def cache_path(self, model):
        return self.root / "outputs" / "main" / model / "cache.jsonl"

    def scored_path(self, model):
        return self.scored_dir / ("%s_scored.jsonl" % model)

    def manifest_path(self, model):
        p = self.scored_path(model)
        return p.with_name(p.stem + "_manifest.json")

    def probe_path(self, probe):
        return self.root / "probes" / self.conditions()["files"][probe]

    @property
    def options_path(self):
        return self.root / "probes" / "fc_options.jsonl"

    @property
    def lexicon_path(self):
        return self.gate_e / "lexicon" / "lexicon.csv"

    @property
    def vocab_path(self):
        return self.gate_e / "lexicon" / "closed_vocab.csv"

    @property
    def excluded_path(self):
        return self.cfg.get("excluded_path") or (self.gate_e / "probes" / "excluded_items.csv")

    @property
    def ht4_dir(self):
        return self.root / "human" / "HT-4"

    def guard(self, path):
        """Never write inside a Git tree (setup file Part 7, rule 1)."""
        p = Path(path).resolve()
        for tree in (self.gate_e, self.main):
            if tree is None:
                continue
            t = Path(tree).resolve()
            if p == t or t in p.parents:
                raise Stop("%s is inside the Git tree %s. Data-derived files stay in Drive."
                           % (p, t))
        return p

    # ---- frozen inputs -------------------------------------------------------
    def lexicon(self):
        if self._lex is None:
            self._lex = self.S.load_lexicon(self.lexicon_path)
        return self._lex

    def vocab(self):
        if self._voc is None:
            self._voc = self.S.load_closed_vocab(self.vocab_path)
        return self._voc

    def excluded(self):
        if self._excluded is None:
            self._excluded = self.S.load_excluded(self.excluded_path)
        return self._excluded

    def conditions(self):
        """configs/conditions.yaml of the gate-E checkout: condition -> probe."""
        if self._conditions is None:
            import yaml
            data = yaml.safe_load((self.gate_e / "configs" / "conditions.yaml").read_text("utf-8"))
            cond = {cid: spec["probe"] for cid, spec in data["conditions"].items()}
            files = {name: spec["file"] for name, spec in data["probes"].items()}
            sizes = {name: spec.get("expected_n") for name, spec in data["probes"].items()}
            self._conditions = {"probe_of": cond, "files": files, "sizes": sizes,
                                "reuse_from": {cid: spec.get("reuse_from")
                                               for cid, spec in data["conditions"].items()}}
        return self._conditions

    def probes(self):
        """{probe: OrderedDict qa_id -> item} of the five probe files."""
        if self._probes is None:
            out = OrderedDict()
            for name in PROBE_ORDER:
                od = OrderedDict()
                for it in self.S.load_probe(self.probe_path(name)):
                    if it["qa_id"] in od:
                        raise Stop("%s holds %s twice" % (name, it["qa_id"]))
                    od[it["qa_id"]] = it
                out[name] = od
            self._probes = out
        return self._probes

    def options(self):
        if self._options is None:
            self._options = self.FC.read_option_sets(self.options_path)
        return self._options

    def probes_of(self, model):
        cond = self.conditions()["probe_of"]
        used = {cond[c] for c in self.cfg["conditions_of"][model]}
        return [p for p in PROBE_ORDER if p in used]

    def has_fc(self, model):
        return any(self.S.condition_spec(c).task == "fc" for c in self.cfg["conditions_of"][model])

    def excluded_in(self, probe):
        return sum(1 for q in self.probes()[probe] if q in self.excluded())

    # ---- records of this script -------------------------------------------------
    def provenance(self):
        d = {"script_version": SCRIPT_VERSION, "script_sha256": sha256_file(__file__),
             "utc": utc_now(), "gate_e": str(self.gate_e), "root": str(self.root),
             "scorer_sha256": sha256_file(self.S.__file__),
             "normalise_sha256": sha256_file(self.N.__file__),
             "fc_options_sha256": sha256_file(self.FC.__file__),
             "asset9_sha256": sha256_file(self.gate_e / "scripts" / "asset9_scorer.py")}
        if self.cfg.get("git_checks"):
            d["gate_e_commit"] = git(self.gate_e, "rev-parse", "HEAD")[1]
            if self.main:
                d["main_commit"] = git(self.main, "rev-parse", "HEAD")[1]
        return d

    def log_run(self, entry):
        path = self.guard(self.records_dir / "runs.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(self.provenance(), **entry), ensure_ascii=False,
                                default=str) + "\n")


def real_cfg():
    return dict(REAL)


# ---------------------------------------------------------------------------
# 3. Checks shared by check, score and verify
# ---------------------------------------------------------------------------

def pilot_files(ctx):
    """{role: {"path", "manifest", "keep", "records"}} from Drive scored/pilot/."""
    folder = ctx.root / "scored" / "pilot"
    if not folder.is_dir():
        raise Stop("Drive scored/pilot/ is missing")
    found = {}
    shas = {}
    for p in sorted(folder.glob("*.jsonl")):
        shas[p] = sha256_file(p)
    for role, pin in ctx.cfg["pilot_files"].items():
        if pin:
            hits = [p for p, s in shas.items() if s.startswith(pin)]
        else:
            hits = [p for p in shas if p.name == role + "_scored.jsonl"]
        if len(hits) != 1:
            raise Stop("scored/pilot/: %d file(s) for %s (sha %s)" % (len(hits), role, pin))
        p = hits[0]
        man = p.with_name(p.stem + "_manifest.json")
        if not man.exists():
            raise Stop("scored/pilot/: no manifest %s for %s" % (man.name, p.name))
        m = json.loads(man.read_text("utf-8"))
        if "keep" not in m:
            raise Stop("scored/pilot/%s has no 'keep' list" % man.name)
        found[role] = {"path": p, "manifest": man, "keep": list(m["keep"]), "sha": shas[p],
                       "scorer_version": m.get("scorer_version")}
    return found


def keep_list(ctx, model, present, pilot):
    """The --keep names for one cache: the pilot scoring's names and KEEP_EXTRA,
    present in the cache, never a scorer key, never a DEFAULT_KEEP name."""
    S, A9 = ctx.S, ctx.A9
    src = pilot[KEEP_FROM[model]]["keep"]
    wanted = list(dict.fromkeys(list(src) + list(KEEP_EXTRA)))
    keep, missing, collide = [], [], []
    for k in wanted:
        if k in A9.DEFAULT_KEEP:
            continue
        if k in S._RECORD_KEYS or k == "answer_text":
            collide.append(k)
            continue
        if k in present:
            keep.append(k)
        elif k in src:
            missing.append(k)
    return keep, missing, collide


def precheck_cache(ctx, model, recs, bad, incomplete):
    """Read-only checks of one cache before it is scored (D270). Returns
    (list of failure texts, summary dict)."""
    S, A9 = ctx.S, ctx.A9
    cfg = ctx.cfg
    cond_probe = ctx.conditions()["probe_of"]
    probes = ctx.probes()
    fails = []
    want_conds = cfg["conditions_of"][model]
    if len(recs) != cfg["records_of"][model]:
        fails.append("%d records, expected %d" % (len(recs), cfg["records_of"][model]))
    if bad or incomplete:
        fails.append("%d unreadable line(s), %d incomplete last line" % (bad, incomplete))
    missing = Counter(k for r in recs for k in CACHE_REQUIRED if k not in r)
    if missing:
        fails.append("records without %s" % ", ".join("%s (%d)" % kv for kv in missing.items()))
        return fails, {}
    keys = Counter(r["key"] for r in recs)
    dup = sum(1 for v in keys.values() if v > 1)
    if dup:
        fails.append("%d cache key(s) appear more than once" % dup)
    badkey = sum(1 for r in recs if r["key"] != "|".join(
        [r["model_id"], r["condition_id"], r["prompt_version"], r["qa_id"]]))
    if badkey:
        fails.append("%d key(s) are not model_id|condition_id|prompt_version|qa_id" % badkey)
    model_ids = sorted({r["model_id"] for r in recs})
    if len(model_ids) != 1:
        fails.append("%d model_id values" % len(model_ids))
    pv = Counter(r["prompt_version"] for r in recs)
    if set(pv) != {"exp-v1"}:
        fails.append("prompt_version %s" % dict(pv))
    st = Counter(r["stage"] for r in recs)
    if set(st) != {"main"}:
        fails.append("stage %s" % dict(st))
    by_cond = defaultdict(list)
    for r in recs:
        by_cond[r["condition_id"]].append(r)
    if set(by_cond) != set(want_conds):
        fails.append("conditions %s, expected %s" % (",".join(sorted(by_cond)),
                                                     ",".join(sorted(want_conds))))
    disagree = Counter()
    for cid, rs in by_cond.items():
        if cid not in cond_probe:
            fails.append("unknown condition %s" % cid)
            continue
        probe = cond_probe[cid]
        items = probes[probe]
        qa = [r["qa_id"] for r in rs]
        if len(qa) != len(items) or set(qa) != set(items):
            fails.append("%s: %d records, %d qa_ids outside %s, %d of its items missing"
                         % (cid, len(qa), len(set(qa) - set(items)), probe,
                            len(set(items) - set(qa))))
        spec = S.condition_spec(cid)
        for r in rs:
            item = items.get(r["qa_id"])
            if item is None:
                continue
            for fld in A9.condition_checks(r, spec, item):
                disagree["%s %s" % (cid, fld)] += 1
    if disagree:
        fails.append("records that disagree with the scorer's condition table: %s"
                     % ", ".join("%s %d" % kv for kv in sorted(disagree.items())))
    summary = {"model_id": model_ids[0] if model_ids else None, "records": len(recs),
               "conditions": {c: len(v) for c, v in sorted(by_cond.items())},
               "none_answers": sum(1 for r in recs if r["answer_text"] is None)}
    # copies of the base model (addendum L9, D228)
    want_copies = cfg["copies"].get(model, 0)
    if want_copies or any(r.get("mode") == "reuse" for r in recs):
        by_key = {r["key"]: r for r in recs}
        copies = [r for r in recs if r.get("mode") == "reuse"]
        shared = set(probes["R-TEST"]) & set(probes["P-CORE-CLOSED"])
        per = Counter(r["condition_id"] for r in copies)
        bad_copy = 0
        for r in copies:
            src = by_key.get(r.get("reused_from_key"))
            if src is None or src["condition_id"] != COPY_SOURCE.get(r["condition_id"]) \
                    or src["qa_id"] != r["qa_id"] or src["answer_text"] != r["answer_text"]:
                bad_copy += 1
        if len(copies) != want_copies:
            fails.append("%d copied answers, expected %d" % (len(copies), want_copies))
        for cid in EXP_R:
            got = {r["qa_id"] for r in copies if r["condition_id"] == cid}
            if want_copies and got != shared:
                fails.append("%s: the copies are not the %d R-TEST items in P-CORE-CLOSED"
                             % (cid, len(shared)))
        if bad_copy:
            fails.append("%d copies without a matching source record" % bad_copy)
        summary["copies"] = dict(per)
    # adapters (addendum L9)
    pin = cfg["adapter_sha"].get(model)
    if pin is not None:
        shas = Counter(r.get("adapter_sha256") for r in recs)
        runs = Counter(r.get("adapter_run") for r in recs)
        if len(shas) != 1 or not str(next(iter(shas))).startswith(pin):
            fails.append("adapter_sha256 %s, expected %s" % (dict(shas), pin))
        if len(runs) != 1 or next(iter(runs)) in (None, ""):
            fails.append("adapter_run %s" % dict(runs))
        summary["adapter"] = {"sha256": next(iter(shas)), "run": next(iter(runs))}
    return fails, summary


def expected_excluded(ctx, model):
    cond_probe = ctx.conditions()["probe_of"]
    return {c: ctx.excluded_in(cond_probe[c]) for c in ctx.cfg["conditions_of"][model]}


# ---------------------------------------------------------------------------
# 4. check (Cell 0)
# ---------------------------------------------------------------------------

def cmd_check(ctx, args):
    print("Scoring and audit, part 1: input check, %s, run %s" % (SCRIPT_VERSION, utc_now()))
    print("Drive folder %s; gate-E checkout %s; main clone %s" % (ctx.root, ctx.gate_e, ctx.main))
    L = Lines()
    S, FC, A9 = ctx.S, ctx.FC, ctx.A9
    cfg = ctx.cfg
    n = [0]

    def code():
        n[0] += 1
        return "C%02d" % n[0]

    # --- Git: the clone, the tag, the checkout, this script ---------------------
    if cfg.get("git_checks"):
        rc, url, _ = git(ctx.main, "remote", "get-url", "origin")
        url_ok = rc == 0 and re.search(r"/banglaindoorvqa-main(\.git)?$", url) is not None
        clean, dirt = tree_clean(ctx.main)
        head = git(ctx.main, "rev-parse", "HEAD")[1]
        L.add(code(), url_ok and clean, "main clone: origin %s, HEAD %s, %s"
              % ("banglaindoorvqa-main" if url_ok else "NOT banglaindoorvqa-main", head[:12],
                 "clean" if clean else "NOT clean: %s" % dirt.replace("\n", "; ")[:200]))
        kind = git(ctx.main, "cat-file", "-t", "gate-E")[1]
        target = git(ctx.main, "rev-parse", "gate-E^{commit}")[1]
        L.add(code(), kind == "tag" and target == GATE_E_COMMIT,
              "tag gate-E: %s, on %s (expected annotated, on %s)"
              % (kind or "missing", target[:12], GATE_E_COMMIT[:12]))
        ge_head = git(ctx.gate_e, "rev-parse", "HEAD")[1]
        detached = git(ctx.gate_e, "symbolic-ref", "-q", "HEAD")[0] != 0
        ge_clean, ge_dirt = tree_clean(ctx.gate_e)
        L.add(code(), ge_head == GATE_E_COMMIT and detached and ge_clean,
              "gate-E checkout %s: HEAD %s, %s, %s" % (
                  ctx.gate_e, ge_head[:12], "detached" if detached else "ON A BRANCH",
                  "clean" if ge_clean else "NOT clean: %s" % ge_dirt[:200]))
        paths = ["src/scoring.py", "src/normalise.py", "src/fc_options.py",
                 "scripts/asset9_scorer.py", "lexicon/lexicon.csv",
                 "lexicon/closed_vocab.csv", "probes/excluded_items.csv",
                 "configs/conditions.yaml"]
        changed = git(ctx.main, "diff", "--name-only", "gate-E", "HEAD", "--", *paths)
        same = changed[0] == 0 and not changed[1].strip()
        L.add(code(), same, "main equals gate-E for the scorer, normalise, fc_options, "
              "asset9_scorer.py, the lexicon, the excluded list and conditions.yaml (Gate F "
              "tags this scorer)%s" % ("" if same else "; CHANGED: " + changed[1].replace("\n", ", ")))
        tchanged = git(ctx.main, "diff", "--name-only", "gate-E", "HEAD", "--", "tests")[1].strip()
        L.info(code(), "tests/: %s" % (
            "same as gate-E" if not tchanged else "changed since gate-E: %s (Gate F runs the "
            "whole suite; say so in the chat)" % tchanged.replace("\n", ", ")))
        me = Path(__file__).resolve()
        inside = ctx.main.resolve() in me.parents
        rel = me.relative_to(ctx.main.resolve()).as_posix() if inside else str(me)
        tracked = inside and git(ctx.main, "ls-files", "--error-unmatch", rel)[0] == 0
        unchanged = tracked and git(ctx.main, "diff", "--quiet", "HEAD", "--", rel)[0] == 0
        L.add(code(), tracked and unchanged, "this script %s (%s, sha %s): %s" % (
            rel, SCRIPT_VERSION, short(sha256_file(me)),
            "committed on main, no local change" if unchanged else
            "NOT committed or changed locally: upload it to the repo, pull, run again"))
    else:
        L.info(code(), "Git checks skipped (selftest)")

    # --- gate-E files -------------------------------------------------------------
    bad = []
    for rel, pin in cfg["gate_e_pins"].items():
        s = sha256_file(ctx.gate_e / rel)
        if not pin_ok(s, pin):
            bad.append("%s %s (expected %s)" % (rel, short(s), pin[:12]))
    vers = (S.SCORER_VERSION == SCORER_VERSION and A9.SCRIPT_VERSION == ASSET9_VERSION
            and FC.RULES_VERSION == FC_RULES_VERSION)
    L.add(code(), not bad and vers,
          "gate-E scorer files: scoring.py %s (%s), asset9_scorer.py %s (%s), fc_options.py %s "
          "(%s), normalise.py %s%s" % (
              short(sha256_file(S.__file__)), S.SCORER_VERSION,
              short(sha256_file(ctx.gate_e / "scripts" / "asset9_scorer.py")), A9.SCRIPT_VERSION,
              short(sha256_file(FC.__file__)), FC.RULES_VERSION, short(sha256_file(ctx.N.__file__)),
              "" if not bad else "; DIFFER: " + "; ".join(bad)))
    try:
        lex, voc = ctx.lexicon(), ctx.vocab()
        lw, vw = len(lex.warnings), len(voc.warnings)
        want = cfg["lexicon"]
        ok = (len(lex.objects) == want["objects"] and len(lex.index.entries) == want["terms"]
              and lw == 0)
        L.add(code(), ok, "lexicon.csv %s: %s; %d warnings" % (
            short(sha256_file(ctx.lexicon_path)), lex.describe(), lw))
        L.add(code(), voc.describe() == cfg["vocab"] and vw == 0,
              "closed_vocab.csv %s: %s; %d warnings" % (short(sha256_file(ctx.vocab_path)),
                                                        voc.describe(), vw))
    except Exception as exc:                                   # report, never hide
        L.add(code(), False, "lexicon or vocabulary does not load: %s" % exc)
        return finish_check(ctx, L)
    try:
        ex = ctx.excluded()
        L.add(code(), len(ex) == cfg["excluded_rows"], "excluded_items.csv %s: %d rows (%s)" % (
            short(sha256_file(ctx.excluded_path)), len(ex),
            ", ".join("%s %d" % kv for kv in sorted(Counter(ex.values()).items()))))
    except Exception as exc:
        L.add(code(), False, "excluded list does not load: %s" % exc)
        return finish_check(ctx, L)
    try:
        cond = ctx.conditions()
        mismatch = [c for c in EXP_L + EXP_N + EXP_R if c not in cond["probe_of"]]
        L.add(code(), not mismatch, "conditions.yaml: %d conditions; probes: %s" % (
            len(cond["probe_of"]), ", ".join("%s %s" % (c, p) for c, p in
                                             sorted(Counter(cond["probe_of"].values()).items()))))
    except Exception as exc:
        L.add(code(), False, "conditions.yaml does not load: %s" % exc)
        return finish_check(ctx, L)

    # --- probe files ----------------------------------------------------------------
    try:
        probes = ctx.probes()
    except Exception as exc:
        L.add(code(), False, "probe files: %s" % exc)
        return finish_check(ctx, L)
    for name in PROBE_ORDER:
        path = ctx.probe_path(name)
        s = sha256_file(path)
        pin = cfg["drive_pins"].get("probes/" + path.name)
        size = len(probes[name])
        scored = size - ctx.excluded_in(name)
        want_scored = cfg["scored_sizes"].get(name)
        L.add(code(), pin_ok(s, pin) and size == cfg["probe_sizes"][name]
              and (want_scored is None or scored == want_scored),
              "%s %s: %d items, %d excluded, %d scored (expected %d of %s)" % (
                  path.name, short(s), size, size - scored, scored,
                  want_scored if want_scored is not None else scored, cfg["probe_sizes"][name]))
    seen, multi, differ = {}, 0, []
    for name in PROBE_ORDER:
        for q, it in probes[name].items():
            if q in seen:
                multi += 1
                o = seen[q]
                if o["qa_type"] != it["qa_type"] or o["answer_bn"] != it["answer_bn"]:
                    differ.append(q)
            else:
                seen[q] = it
    L.add(code(), not differ, "probe files together: %d distinct qa_ids, %d repeats in a "
          "second file, %d with another type or gold%s" % (
              len(seen), multi, len(differ), "" if not differ else " (first %s)" % differ[0]))
    try:
        table = ctx.options()
        s = sha256_file(ctx.options_path)
        bn = sum(1 for k in table if k[1] == "bn")
        en = sum(1 for k in table if k[1] == "en")
        miss_bn = [q for q in probes["P-CLUST"] if (q, "bn") not in table]
        miss_en = [q for q in probes["P-FCE"] if (q, "en") not in table]
        pin = cfg["drive_pins"].get("probes/fc_options.jsonl")
        L.add(code(), pin_ok(s, pin) and bn == cfg["options_sets"]["bn"]
              and en == cfg["options_sets"]["en"] and not miss_bn and not miss_en,
              "fc_options.jsonl %s: %d sets (%d bn, %d en); P-CLUST without a Bangla set %d, "
              "P-FCE without an English set %d" % (short(s), len(table), bn, en, len(miss_bn),
                                                   len(miss_en)))
    except Exception as exc:
        L.add(code(), False, "option file: %s" % exc)

    # --- pilot scored files (keep lists, D159) ----------------------------------------
    pilot = None
    try:
        pilot = pilot_files(ctx)
        bad = []
        for role, info in pilot.items():
            recs, b, inc, _s = read_jsonl(info["path"])
            vers = Counter(r.get("scorer_version") for r in recs)
            want = cfg["pilot_records"].get(role)
            if b or inc or set(vers) != {SCORER_VERSION} or (want and len(recs) != want):
                bad.append("%s: %d records, versions %s" % (role, len(recs), dict(vers)))
        L.add(code(), not bad, "scored/pilot/: %s%s" % (
            ", ".join("%s %s" % (r, short(i["sha"])) for r, i in pilot.items()),
            "" if not bad else "; " + "; ".join(bad)))
    except Exception as exc:
        L.add(code(), False, "scored/pilot/: %s" % exc)

    # --- caches -------------------------------------------------------------------------
    keeps = {}
    fields = {}
    for model in MODELS:
        path = ctx.cache_path(model)
        if not path.exists():
            L.add(code(), False, "%s cache: %s is missing" % (model, path))
            continue
        recs, b, inc, s = read_jsonl(path)
        fields[model] = Counter(k for r in recs for k in r)
        fails, summ = precheck_cache(ctx, model, recs, b, inc)
        extra = ""
        if summ.get("copies"):
            extra += "; copies %s" % ", ".join("%s %d" % kv for kv in sorted(summ["copies"].items()))
        if summ.get("adapter"):
            extra += "; adapter %s" % short(summ["adapter"]["sha256"])
        L.add(code(), not fails, "%s cache %s: %d records, %d conditions, model_id %s%s%s" % (
            model, short(s), len(recs), len(summ.get("conditions", {})), summ.get("model_id"),
            extra, "" if not fails else "; " + "; ".join(fails)))
        if pilot is not None:
            present = set().union(*(r.keys() for r in recs)) if recs else set()
            keep, missing, collide = keep_list(ctx, model, present, pilot)
            keeps[model] = (keep, missing, collide)
    if fields:
        allk = sorted(set().union(*fields.values()))
        not_all = []
        for k in allk:
            short_of = ["%s %d of %d" % (m, fields[m][k], ctx.cfg["records_of"][m])
                        for m in fields if fields[m][k] < ctx.cfg["records_of"][m]]
            if short_of:
                not_all.append("%s (%s)" % (k, ", ".join(short_of)))
        L.info(code(), "record keys in the caches: %s. Not in every record: %s" % (
            ", ".join(allk), "; ".join(not_all) if not_all else "none"))
    if pilot is not None and keeps:
        bad = {m: v[1] for m, v in keeps.items() if v[1]}
        col = sorted({k for v in keeps.values() for k in v[2]})
        L.add(code(), True, "kept harness fields: %s%s" % (
            "; ".join("%s %d (%s)" % (m, len(v[0]), ", ".join(v[0])) for m, v in keeps.items()),
            "" if not col else "; never kept because the scorer writes them: %s" % ", ".join(col)))
        if bad:
            L.info(code(), "NOTE: the pilot scoring kept fields that these caches lack: %s. They "
                   "cannot be kept. Send this line in the chat before HT-4 is drawn." % bad)

    # --- D192 inputs ---------------------------------------------------------------------
    try:
        d192 = D192Rules(ctx)
        L.add(code(), True, "D192 inputs: extra_words.csv %s (%d rows), frozen_file_notes.csv %s "
              "(%d genitive-only rows, equal to the scorer's); anchors %s" % (
                  short(d192.sha_extra), d192.n_extra, short(d192.sha_notes), len(d192.gen),
                  ", ".join(ctx.cfg["d192_anchors"].values())))
    except Exception as exc:
        L.add(code(), False, "D192 inputs: %s" % exc)

    # --- outputs and locks -------------------------------------------------------------------
    existing = [m for m in MODELS if ctx.scored_path(m).exists()]
    ht4 = ht4_drawn(ctx)
    L.add(code(), not ht4 or len(existing) == len(MODELS),
          "Drive scored/main/: %s%s" % (
              "no scored file yet" if not existing else
              "%d of 6 scored files exist (%s); score skips them unless --overwrite"
              % (len(existing), ", ".join(existing)),
              "; an HT-4 sheet exists, so no scored file may be replaced" if ht4 else ""))
    locks = ctx.root / "outputs" / "main" / "_locks"
    held = sorted(p.name for p in locks.glob("*")) if locks.is_dir() else []
    L.add(code(), not held, "Drive outputs/main/_locks/: %s" % (
        "empty (no session writes a cache)" if not held else "HELD: %s" % ", ".join(held)))
    return finish_check(ctx, L)


def finish_check(ctx, L):
    fails = L.failed()
    print("CHECK: %s" % ("ALL PASS (%d lines)" % len(L.rows) if not fails else
                         "%d FAIL of %d lines" % (len(fails), len(L.rows))))
    try:
        path = ctx.guard(ctx.records_dir / ("check_%s.json" % stamp()))
        write_json(path, {"command": "check", "provenance": ctx.provenance(), "lines": L.rows})
        print("Wrote %s" % path)
    except Exception as exc:
        print("Could not write the check record: %s" % exc)
    return 1 if fails else 0


def ht4_drawn(ctx):
    d = ctx.ht4_dir / "to_send"
    return d.is_dir() and any(d.iterdir())


# ---------------------------------------------------------------------------
# 5. score (Cell 2)
# ---------------------------------------------------------------------------

def score_argv(ctx, model, keep, out):
    argv = [sys.executable, str(ctx.gate_e / "scripts" / "asset9_scorer.py"), "score",
            "--outputs", str(ctx.cache_path(model))]
    for p in ctx.probes_of(model):
        argv += ["--probe", str(ctx.probe_path(p))]
    if ctx.has_fc(model):
        argv += ["--options-file", str(ctx.options_path)]
    argv += ["--lexicon", str(ctx.lexicon_path), "--vocab", str(ctx.vocab_path),
             "--excluded", str(ctx.excluded_path)]
    for k in keep:
        argv += ["--keep", k]
    argv += ["--counts-only", "--out", str(out)]
    return argv


def cmd_score(ctx, args):
    models = args.model or list(MODELS)
    print("Scoring and audit, part 1: score, %s, run %s" % (SCRIPT_VERSION, utc_now()))
    print("One gate-E `asset9_scorer.py score` command per cache (D270), run from %s (D271)."
          % ctx.gate_e)
    pilot = pilot_files(ctx)
    results = {}
    for model in models:
        print("\n==== %s ====" % model)
        out = ctx.guard(ctx.scored_path(model))
        if out.exists() and not args.overwrite:
            print("SKIP: %s exists. Run verify, or pass --overwrite to score again." % out.name)
            results[model] = "exists"
            continue
        if out.exists() and ht4_drawn(ctx):
            raise Stop("an HT-4 sheet exists (human/HT-4/to_send/); a scored file it was drawn "
                       "from is never replaced (D270)")
        recs, b, inc, cache_sha = read_jsonl(ctx.cache_path(model))
        fails, _summ = precheck_cache(ctx, model, recs, b, inc)
        if fails:
            print("STOP: the cache fails its check: %s" % "; ".join(fails))
            results[model] = "precheck failed"
            continue
        present = set().union(*(r.keys() for r in recs))
        keep, missing, _col = keep_list(ctx, model, present, pilot)
        if missing:
            print("NOTE: the pilot scoring kept %s, which no record of this cache has; not kept."
                  % ", ".join(missing))
        del recs
        argv = score_argv(ctx, model, keep, out)
        print("Probes: %s. Options file: %s. Kept fields: %s." % (
            ", ".join(ctx.probes_of(model)), "yes" if ctx.has_fc(model) else "no", ", ".join(keep)))
        out.parent.mkdir(parents=True, exist_ok=True)
        log = ctx.guard(ctx.records_dir / "logs" / ("score_%s_%s.log" % (model, stamp())))
        log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        t0 = datetime.now(timezone.utc)
        with open(log, "w", encoding="utf-8") as fh:
            fh.write("argv: %s\n" % json.dumps(argv, ensure_ascii=False))
            proc = subprocess.Popen(argv, cwd=str(ctx.gate_e), env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                fh.write(line)
                print("    " + line.rstrip("\n"))
            code = proc.wait()
        secs = (datetime.now(timezone.utc) - t0).total_seconds()
        if ctx.cfg.get("git_checks"):
            clean, dirt = tree_clean(ctx.gate_e)
            if not clean:
                print("FAIL: the gate-E checkout is no longer clean: %s" % dirt[:200])
                code = code or 9
        if code != 0:
            print("FAIL: the score command ended with exit code %d (log %s)" % (code, log))
            results[model] = "exit %d" % code
            ctx.log_run({"command": "score", "model": model, "exit": code, "log": str(log)})
            continue
        ok, lines, summ = verify_one(ctx, model)
        for line in lines:
            print("  " + line)
        print("SCORE %s: %s, %.0f s; wrote %s" % (model, "PASS" if ok else "FAIL", secs, out))
        results[model] = "PASS" if ok else "verify failed"
        ctx.log_run({"command": "score", "model": model, "exit": code, "seconds": round(secs, 1),
                     "cache_sha256": cache_sha, "out": str(out), "out_sha256": sha256_file(out),
                     "keep": keep, "argv": argv[1:], "log": str(log), "verify": summ})
    print("\nSCORE: %s" % ", ".join("%s %s" % kv for kv in results.items()))
    return 0 if all(v in ("PASS", "exists") for v in results.values()) else 1


# ---------------------------------------------------------------------------
# 6. verify (Cell 3) and the free check of the copies
# ---------------------------------------------------------------------------

def load_scored(ctx, model):
    recs, bad, inc, sha = read_jsonl(ctx.scored_path(model))
    if bad or inc:
        raise Stop("%s: %d unreadable line(s)" % (ctx.scored_path(model).name, bad + inc))
    return recs, sha


def verify_one(ctx, model):
    """The scored file of one cache against the cache (D270). Returns (ok, lines, summary)."""
    S = ctx.S
    lines, fails = [], []
    out, man_path = ctx.scored_path(model), ctx.manifest_path(model)
    if not out.exists() or not man_path.exists():
        return False, ["FAIL %s: scored file or manifest missing" % model], {}
    man = json.loads(man_path.read_text("utf-8"))
    cache_recs, _b, _i, cache_sha = read_jsonl(ctx.cache_path(model))
    recs, sha = load_scored(ctx, model)
    n = len(cache_recs)
    exp_ex = expected_excluded(ctx, model)
    want_ex = sum(exp_ex.values())
    if man.get("scorer_version") != SCORER_VERSION or man.get("script_version") != ASSET9_VERSION:
        fails.append("manifest scorer %s, script %s" % (man.get("scorer_version"),
                                                        man.get("script_version")))
    outs = man.get("inputs", {}).get("outputs", [])
    if len(outs) != 1 or outs[0].get("sha256") != cache_sha or outs[0].get("records") != n:
        fails.append("the manifest names another cache or size (the cache changed after scoring?)")
    if man.get("skipped") or man.get("later_records_differ"):
        fails.append("manifest skipped %s, later records differ %s"
                     % (man.get("skipped"), man.get("later_records_differ")))
    st = man.get("statuses", {})
    if man.get("records") != n or st.get("excluded", 0) != want_ex \
            or st.get("ok", 0) != n - want_ex or set(st) - {"ok", "excluded"}:
        fails.append("manifest records %s, statuses %s; expected %d with %d excluded"
                     % (man.get("records"), st, n, want_ex))
    ins = man.get("inputs", {})
    for name, pin_key in (("lexicon", "lexicon/lexicon.csv"), ("vocabulary", "lexicon/closed_vocab.csv"),
                          ("excluded", "probes/excluded_items.csv")):
        pin = ctx.cfg["gate_e_pins"].get(pin_key)
        if not pin_ok(ins.get(name, {}).get("sha256", ""), pin):
            fails.append("manifest %s sha %s" % (name, short(ins.get(name, {}).get("sha256"))))
    opt_sha = sha256_file(ctx.options_path) if ctx.has_fc(model) else None
    if ctx.has_fc(model) and ins.get("options", {}).get("sha256") != opt_sha:
        fails.append("manifest option file sha %s" % short(ins.get("options", {}).get("sha256")))
    got_probes = sorted(p.get("file") for p in ins.get("probes", []))
    want_probes = sorted(ctx.probe_path(p).name for p in ctx.probes_of(model))
    if got_probes != want_probes:
        fails.append("manifest probes %s, expected %s" % (got_probes, want_probes))
    # records
    cache_by_key = {r["key"]: r for r in cache_recs}
    keys = Counter(r.get("key") for r in recs)
    if len(recs) != n or len(keys) != n or set(keys) != set(cache_by_key):
        fails.append("%d scored records, %d distinct keys; %d cache keys not scored"
                     % (len(recs), len(keys), len(set(cache_by_key) - set(keys))))
    vers = Counter(r.get("scorer_version") for r in recs)
    if set(vers) != {SCORER_VERSION}:
        fails.append("scorer versions %s" % dict(vers))
    per = defaultdict(Counter)
    bad_status = bad_fc = bad_text = 0
    for r in recs:
        c = r.get("condition_id")
        per[c]["n"] += 1
        gs = r.get("gold_status")
        if gs == "excluded":
            per[c]["excluded"] += 1
        elif gs != "ok" or r.get("scoreable") is not True:
            bad_status += 1
        spec = S.condition_spec(c)
        if (spec.task == "fc") != (r.get("fc_options_sha256") is not None) or \
                (spec.task == "fc" and r.get("fc_options_sha256") != opt_sha):
            bad_fc += 1
        src = cache_by_key.get(r.get("key"))
        if src is not None:
            raw = "" if src.get("answer_text") is None else str(src.get("answer_text"))
            for a, b in S._QUOTES.items():
                raw = raw.replace(a, b)
            if raw != r.get("answer_raw"):
                bad_text += 1
    wrong_ex = {c: (per[c]["excluded"], exp_ex[c]) for c in exp_ex if per[c]["excluded"] != exp_ex[c]}
    if wrong_ex:
        fails.append("excluded per condition differ: %s" % wrong_ex)
    if bad_status:
        fails.append("%d records neither ok nor excluded" % bad_status)
    if bad_fc:
        fails.append("%d records with a wrong fc_options_sha256" % bad_fc)
    if bad_text:
        fails.append("%d records whose answer_raw is not the cache answer" % bad_text)
    summary = {"model": model, "records": len(recs), "excluded": sum(v["excluded"] for v in per.values()),
               "scored": len(recs) - sum(v["excluded"] for v in per.values()),
               "out_sha256": sha, "cache_sha256": cache_sha}
    lines.append("%s %s: %d records = cache, %d scored, %d excluded (expected %d), 0 skipped%s" % (
        "PASS" if not fails else "FAIL", model, len(recs), summary["scored"], summary["excluded"],
        want_ex, "" if not fails else "; " + "; ".join(fails)))
    # the free check of the copies (addendum L10)
    copies = [r for r in cache_recs if r.get("mode") == "reuse"]
    if copies or ctx.cfg["copies"].get(model):
        by_key = {r["key"]: r for r in recs}
        equal = differ = ex = 0
        first = None
        fields = [k for k in S._RECORD_KEYS if k != "condition_id"]
        for c in copies:
            a, b = by_key.get(c["key"]), by_key.get(c.get("reused_from_key"))
            if a is None or b is None:
                differ += 1
                first = first or c["key"]
                continue
            diff = [k for k in fields if a.get(k) != b.get(k)]
            if diff:
                differ += 1
                first = first or "%s (%s)" % (c["qa_id"], ",".join(diff[:4]))
            else:
                equal += 1
                ex += a.get("gold_status") == "excluded"
        want = ctx.cfg["copies"].get(model, 0)
        ok_copy = differ == 0 and equal == want
        if not ok_copy:
            fails.append("free check")
        lines.append("%s %s free check: %d of %d copies score exactly as their EXP-L source "
                     "(%d of them excluded items), %d differ%s" % (
                         "PASS" if ok_copy else "FAIL", model, equal, len(copies), ex, differ,
                         "" if not first else "; first %s" % first))
        summary["free_check"] = {"copies": len(copies), "equal": equal, "excluded": ex,
                                 "differ": differ}
    return not fails, lines, summary


def cmd_verify(ctx, args):
    print("Scoring and audit, part 1: verify, %s, run %s" % (SCRIPT_VERSION, utc_now()))
    allok = True
    out = {}
    total = Counter()
    for model in MODELS:
        ok, lines, summ = verify_one(ctx, model)
        for line in lines:
            print(line)
        allok = allok and ok
        out[model] = summ
        for k in ("records", "scored", "excluded"):
            total[k] += summ.get(k, 0)
    print("All six files: %d records, %d scored, %d excluded" % (total["records"], total["scored"],
                                                               total["excluded"]))
    print("VERIFY: %s" % ("ALL PASS" if allok else "FAIL"))
    path = ctx.guard(ctx.records_dir / ("verify_%s.json" % stamp()))
    write_json(path, {"command": "verify", "provenance": ctx.provenance(), "result": allok,
                      "files": out, "total": dict(total)})
    print("Wrote %s" % path)
    ctx.log_run({"command": "verify", "result": allok, "total": dict(total)})
    return 0 if allok else 1


# ---------------------------------------------------------------------------
# 7. counts (Cell 4): D192, D194 and the open issues (D272)
# ---------------------------------------------------------------------------

class D192Rules:
    """The seven rows of addendum B7, as counting rules on scored records."""

    NAMES = OrderedDict([
        ("d192_1", "unlisted word in the gold of row 1 (Bangla) or the English word B7 names"),
        ("d192_2", "unlisted word in the gold of row 2"),
        ("d192_3", "bare Bangla word of a genitive-only form, row 3 item"),
        ("d192_4", "a second colour named on the row 4 item"),
        ("d192_5", "the liquid form রঙ found in a material answer"),
        ("d192_6", "bare Bangla word of a genitive-only form, any material answer"),
        ("d192_7", "the object ব্যাগ named for the object কম্বল রাখার ব্যাগ"),
    ])

    def __init__(self, ctx):
        S = ctx.S
        self.S = S
        self.lex, self.voc = ctx.lexicon(), ctx.vocab()
        audit = ctx.root / "scored" / "asset9_stage2" / "audit"
        ew, fn = audit / "extra_words.csv", audit / "frozen_file_notes.csv"
        self.sha_extra, self.sha_notes = sha256_file(ew), sha256_file(fn)
        for path, s in ((ew, self.sha_extra), (fn, self.sha_notes)):
            pin = ctx.cfg["drive_pins"].get("scored/asset9_stage2/audit/" + path.name)
            if not pin_ok(s, pin):
                raise Stop("%s has sha %s; the copy attached in the Scoring chat had %s"
                           % (path.name, short(s), pin))
        rows = read_csv(ew)
        self.n_extra = len(rows)
        by_qa = {r["example_qa_id"]: r for r in rows}
        self.gold = {}
        for f, qa in ctx.cfg["d192_anchors"].items():
            if qa not in by_qa:
                raise Stop("extra_words.csv has no row with example_qa_id %s (%s)" % (qa, f))
            self.gold[f] = ctx.normalise(by_qa[qa]["gold"])
        self.word = {}
        for f in ("d192_1", "d192_2"):
            w = ctx.normalise(by_qa[ctx.cfg["d192_anchors"][f]]["extra_words"].split()[0])
            toks = S.tokenise(w)
            if len(toks) != 1 or toks[0].kind != "bn":
                raise Stop("%s: the first extra word is not one Bangla word" % f)
            if self.lex.index.key_of(w) in self.lex.index.entries or any(
                    self.voc.indexes[k].key_of(w) in self.voc.indexes[k].entries
                    for k in S.VOCAB_KINDS):
                raise Stop("%s: the extra word is a listed term" % f)
            self.word[f] = w
        self.gen = list(S.genitive_only_forms(self.voc))
        notes = [r for r in read_csv(fn) if r["note"] == "genitive-only form"]
        mine = sorted((k, c, f) for k, c, f, _b in self.gen)
        theirs = sorted((r["source"], r["owner"], ctx.normalise(r["bn"])) for r in notes)
        if mine != theirs:
            raise Stop("frozen_file_notes.csv genitive-only rows %s differ from the scorer's %s"
                       % (theirs, mine))
        self.mat = self.voc.indexes["material"]
        self.rong_key = self.mat.key_of(ctx.normalise(RONG))
        if ctx.normalise(RONG) not in self.voc.forms["material"].get("liquid", {}).get("bn", []):
            raise Stop("the form %s is not a Bangla form of the class liquid" % RONG)
        self.gen_keys = {self.mat.key_of(f): (cid, bare) for _k, cid, f, bare in self.gen}
        for oid in (BLANKET_BAG, BAG):
            if oid not in self.lex.objects:
                raise Stop("lexicon object %s is missing" % oid)
        self.english = ENGLISH_FORMS_D192
        self._gold_cache = {}

    @staticmethod
    def _holds(S, tokens, word, kind):
        for t in tokens:
            if t.kind != kind:
                continue
            stems = S.stems_bn(t.text) if kind == "bn" else S.stems_en(t.text)
            if t.text == word or word in stems:
                return True
        return False

    def _bare_hits(self, tokens, matches):
        covered = set()
        for m in matches:
            covered |= set(range(m.start, m.end))
        hits = []
        for i, t in enumerate(tokens):
            if i in covered or t.kind != "bn":
                continue
            cands = {t.text} | set(self.S.stems_bn(t.text))
            for _k, cid, form, bare in self.gen:
                if bare in cands:
                    hits.append((cid, form, bare))
        return hits

    def _gold_forms(self, gold_norm):
        if gold_norm not in self._gold_cache:
            toks = self.S.tokenise(self.S.strip_lead_in(gold_norm))
            self._gold_cache[gold_norm] = {m.key for m in self.mat.find_all(toks)}
        return self._gold_cache[gold_norm]

    def apply(self, r):
        """[(finding, touched, wrong, split)] for one scored open record."""
        S = self.S
        out = []
        prep = S.prepare_answer(r.get("answer_raw"))
        toks = prep.tokens
        gstrip = S.strip_lead_in(r.get("gold_norm") or "")
        wrong = r.get("label") not in WRONG_LABELS_OK
        kind = r.get("kind")
        mat_matches = self.mat.find_all(toks) if kind == "material" else []
        for f in ("d192_1", "d192_2"):
            if gstrip == self.gold[f]:
                t = self._holds(S, toks, self.word[f], "bn") or any(
                    self._holds(S, toks, w, "lat") for w in self.english.get(f, ()))
                out.append((f, t, t and wrong, ""))
        if gstrip == self.gold["d192_3"]:
            t = bool(self._bare_hits(toks, mat_matches)) if kind == "material" else False
            out.append(("d192_3", t, t and wrong, ""))
        if gstrip == self.gold["d192_4"]:
            g, fnd = set(r.get("gold_value") or []), set(r.get("found_value") or [])
            t = kind == "colour" and bool(fnd & g) and bool(fnd - g)
            out.append(("d192_4", t, t and wrong, ""))
        if kind == "material":
            rong = any(m.key == self.rong_key for m in mat_matches)
            liquid_gold = "liquid" in (r.get("gold_value") or [])
            out.append(("d192_5", rong, rong and wrong,
                        "gold holds liquid" if liquid_gold else "gold without liquid"))
            hits = self._bare_hits(toks, mat_matches)
            item_form = any(self.mat.key_of(form) in self._gold_forms(r.get("gold_norm") or "")
                            for _cid, form, _b in hits)
            out.append(("d192_6", bool(hits), bool(hits) and wrong,
                        "" if not hits else ("item gold holds the form" if item_form
                                             else "other item")))
        if r.get("gold_name_id") == BLANKET_BAG:
            t = BAG in (r.get("named_ids") or [])
            out.append(("d192_7", t, t and r.get("label") != "tier1", ""))
        return out


def fc_label_count(S, r, label_lang):
    prep = S.prepare_answer(r.get("answer_raw"))
    hits = S._label_tokens(prep.norm_tokens, prep.norm, label_lang)
    return len({k for _i, k in hits})


def cmd_counts(ctx, args):
    S = ctx.S
    print("Scoring and audit, part 1: counts (D272), %s, run %s" % (SCRIPT_VERSION, utc_now()))
    rules = D192Rules(ctx)
    d192 = defaultdict(Counter)            # (finding, model, condition, split) -> counts
    d194 = defaultdict(Counter)            # (model, condition) -> counts
    issues = defaultdict(Counter)          # (model, condition) -> counts
    touched_rows, phrase_rows, cand_rows, named_rows = [], [], [], []
    n_records = 0
    for model in MODELS:
        recs, _sha = load_scored(ctx, model)
        n_records += len(recs)
        for r in recs:
            if r.get("gold_status") != "ok":
                continue
            c = r["condition_id"]
            spec = S.condition_spec(c)
            key = (model, c)
            d194[key]["scored"] += 1
            phrase = r.get("refusal_phrase") is True
            if phrase:
                d194[key]["refusal_phrase"] += 1
                if r.get("lenient") == 1:
                    d194[key]["phrase_and_correct"] += 1
                    if r.get("strict") == 1:
                        d194[key]["phrase_and_strict"] += 1
                    if r.get("kind") == "count":
                        d194[key]["count_phrase_and_correct"] += 1
                    phrase_rows.append(row_text(model, r))
            if r.get("refusal") is True:
                d194[key]["refusal"] += 1
            if spec.task == "fc":
                iss = issues[key]
                iss["fc_scored"] += 1
                iss["invalid"] += r.get("label") == S.INVALID
                iss["invalid_refusal"] += r.get("label") == S.INVALID and r.get("refusal") is True
                k = fc_label_count(S, r, spec.question_lang)
                if k >= 2:
                    iss["two_or_more_labels"] += 1
                    iss["two_or_more_chosen_by_answer_word"] += r.get("fc_valid") is True
                    iss["two_or_more_invalid"] += r.get("fc_valid") is not True
                iss["other_script_label"] += r.get("fc_cross_label") is True
                cand = (r.get("label") == S.INVALID and r.get("refusal") is not True
                        and not r.get("is_empty") and r.get("fc_cross_label") is not True
                        and r.get("truncated") is not True)
                if cand:
                    iss["fc_possible_unlisted_refusal"] += 1
                    cand_rows.append(dict(row_text(model, r), pool="fc", labels_named=k))
                if model == "gemini" and r["qa_id"] in ctx.cfg["named_refusals"] \
                        and c == "N-FCB0":                 # addendum F11: both in N-FCB0
                    named_rows.append(dict(row_text(model, r), labels_named=k))
            else:
                iss = issues[key]
                iss["open_scored"] += 1
                if r.get("content_found") is False:
                    iss["no_content"] += 1
                    if r.get("refusal") is not True and not r.get("is_empty"):
                        iss["open_no_content_not_refusal"] += 1
                        cand_rows.append(dict(row_text(model, r), pool="open"))
                for f, t, w, split in rules.apply(r):
                    k2 = (f, model, c, split)
                    d192[k2]["considered"] += 1
                    d192[k2]["touched"] += bool(t)
                    d192[k2]["touched_and_wrong"] += bool(w)
                    if t:
                        touched_rows.append(dict(row_text(model, r), finding=f, split=split))
        del recs
    # ---- tables ---------------------------------------------------------------------
    rd = ctx.guard(ctx.reading_dir)
    rows192 = [dict(finding=f, model=m, condition=c, split=s, **v) for (f, m, c, s), v in
               sorted(d192.items())]
    write_csv(rd / "counts_d192.csv", rows192, ["finding", "model", "condition", "split",
                                                "considered", "touched", "touched_and_wrong"])
    rows194 = [dict(model=m, condition=c, **v) for (m, c), v in sorted(d194.items())]
    write_csv(rd / "counts_d194.csv", rows194, ["model", "condition", "scored", "refusal",
                                                "refusal_phrase", "phrase_and_correct",
                                                "phrase_and_strict", "count_phrase_and_correct"])
    cols_iss = ["model", "condition", "fc_scored", "invalid", "invalid_refusal",
                "two_or_more_labels", "two_or_more_chosen_by_answer_word", "two_or_more_invalid",
                "other_script_label", "fc_possible_unlisted_refusal", "open_scored", "no_content",
                "open_no_content_not_refusal"]
    rows_iss = [dict(model=m, condition=c, **v) for (m, c), v in sorted(issues.items())]
    write_csv(rd / "counts_open_issues.csv", rows_iss, cols_iss)
    tcols = ["model", "condition", "qa_id", "qa_type", "kind", "gold_norm", "answer_raw",
             "label", "strict", "lenient", "refusal", "refusal_phrase", "content_found",
             "matched", "named_ids", "found_value", "gold_value", "fc_choice",
             "fc_correct_index", "fc_valid", "truncated"]
    write_csv(rd / "TEXT_d192_touched.csv", touched_rows, ["finding", "split"] + tcols)
    write_csv(rd / "TEXT_d194_phrase_and_correct.csv", phrase_rows, tcols)
    write_csv(rd / "TEXT_possible_unlisted_refusals.csv", cand_rows, ["pool", "labels_named"] + tcols)
    write_csv(rd / "TEXT_named_pilot_refusals.csv", named_rows, ["labels_named"] + tcols)
    # ---- printout: counts only -------------------------------------------------------------
    print("Read %d scored records of 6 files; excluded items left out." % n_records)
    print("\nD192 findings (addendum B7): answers considered, touched, touched and scored wrong")
    agg = defaultdict(Counter)
    for (f, m, c, s), v in d192.items():
        agg[(f, s)].update(v)
    for (f, s), v in sorted(agg.items()):
        print("  %-7s %-26s considered %6d  touched %5d  touched and wrong %5d" % (
            f, s or "-", v["considered"], v["touched"], v["touched_and_wrong"]))
    print("  (per model and condition: counts_d192.csv)")
    print("\nD194: correct answers that hold a refusal phrase, per model (all conditions)")
    per_model = defaultdict(Counter)
    for (m, c), v in d194.items():
        per_model[m].update(v)
    for m in MODELS:
        v = per_model[m]
        print("  %-12s scored %6d  phrase %5d  phrase and lenient-correct %4d (strict %4d; count"
              " items %4d)  refusal %5d" % (m, v["scored"], v["refusal_phrase"],
                                           v["phrase_and_correct"], v["phrase_and_strict"],
                                           v["count_phrase_and_correct"], v["refusal"]))
    nb = sorted(((m, c), v["count_phrase_and_correct"]) for (m, c), v in d194.items()
                if v["count_phrase_and_correct"])
    if nb:
        print("  count answers scored correct with a refusal phrase, by condition: %s"
              % ", ".join("%s %s %d" % (m, c, k) for (m, c), k in nb))
    print("\nOpen issues (information): forced choice")
    for (m, c), v in sorted(issues.items()):
        if v["fc_scored"]:
            print("  %-12s %-7s scored %4d  invalid %4d (refusal %4d)  2+ labels %4d (answer word"
                  " %3d, invalid %3d)  other script %3d  possible unlisted refusals %4d" % (
                      m, c, v["fc_scored"], v["invalid"], v["invalid_refusal"],
                      v["two_or_more_labels"], v["two_or_more_chosen_by_answer_word"],
                      v["two_or_more_invalid"], v["other_script_label"],
                      v["fc_possible_unlisted_refusal"]))
    print("Open issues (information): open answers with no content that are not empty and not "
          "flagged as refusal")
    for m in MODELS:
        tot = sum(v["open_no_content_not_refusal"] for (mm, _c), v in issues.items() if mm == m)
        print("  %-12s %5d" % (m, tot))
    print("Named pilot refusals (%s; Gemini N-FCB0 in main): %d answers found "
          "(labels in TEXT_named_pilot_refusals.csv)" % (", ".join(ctx.cfg["named_refusals"]),
                                                        len(named_rows)))
    summary = {"records": n_records, "d192": {"%s|%s" % k: dict(v) for k, v in agg.items()},
               "d194_per_model": {m: dict(v) for m, v in per_model.items()},
               "named_pilot_refusals": len(named_rows),
               "possible_unlisted_refusals": len(cand_rows), "phrase_and_correct": len(phrase_rows)}
    write_json(rd / "counts_summary.json", dict(provenance=ctx.provenance(), **summary))
    print("\nWrote in %s: counts_d192.csv (%d rows), counts_d194.csv (%d rows), "
          "counts_open_issues.csv (%d rows), counts_summary.json; with answer text: "
          "TEXT_d192_touched.csv (%d), TEXT_d194_phrase_and_correct.csv (%d), "
          "TEXT_possible_unlisted_refusals.csv (%d), TEXT_named_pilot_refusals.csv (%d)" % (
              rd, len(rows192), len(rows194), len(rows_iss), len(touched_rows), len(phrase_rows),
              len(cand_rows), len(named_rows)))
    ctx.log_run({"command": "counts", "summary": summary})
    print("COUNTS: DONE")
    return 0


def row_text(model, r):
    keep = ("condition_id", "qa_id", "qa_type", "kind", "gold_norm", "answer_raw", "label", "strict",
            "lenient", "refusal", "refusal_phrase", "content_found", "matched", "named_ids",
            "found_value", "gold_value", "fc_choice", "fc_correct_index", "fc_valid", "truncated",
            "key", "house_id")
    d = {k: r.get(k) for k in keep}
    d["model"] = model
    d["condition"] = r.get("condition_id")
    return d


# ---------------------------------------------------------------------------
# 8. d159 (Cell 5)
# ---------------------------------------------------------------------------

def parse_ts(v):
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None


def cmd_d159(ctx, args):
    print("Scoring and audit, part 1: D159 against main (D273), %s, run %s"
          % (SCRIPT_VERSION, utc_now()))
    pilot = pilot_files(ctx)
    main_recs, main_sha = load_scored(ctx, "gemini")
    main = {(r["condition_id"], r["qa_id"]): r for r in main_recs
            if r["condition_id"] in ("L-BB", "N-BB")}
    base_recs, _b, _i, _s = read_jsonl(pilot["pilot_gemini"]["path"])
    base_cond = {}
    for r in base_recs:
        if r["condition_id"] in ("L-BB", "N-BB"):
            if r["qa_id"] in base_cond:
                raise Stop("%s is under L-BB and N-BB in the pilot" % r["qa_id"])
            base_cond[r["qa_id"]] = r["condition_id"]
    result = OrderedDict()
    pair_rows = []
    want = ctx.cfg["d159_pairs"]
    ok_all = True
    for role, label in (("pilot_gemini", "pilot (standard mode, 2026-10-14) against main"),
                        ("d5repeat_gemini", "D5 repeat (standard mode) against main"),
                        ("batchtest_gemini", "Batch test (Batch mode, 2026-10-05) against main")):
        recs, _b, _i, _s = read_jsonl(pilot[role]["path"])
        pairs = []
        for r in recs:
            q = r["qa_id"]
            if r["condition_id"] not in ("L-BB", "N-BB") or q not in base_cond:
                continue
            m = main.get((base_cond[q], q))
            if r.get("gold_status") != "ok" or m is None or m.get("gold_status") != "ok":
                continue
            pairs.append((q, base_cond[q], r, m))
        per = Counter(c for _q, c, _r, _m in pairs)
        size_ok = dict(per) == want
        ok_all = ok_all and size_ok
        res = OrderedDict(n=len(pairs), by_condition=dict(per), size_ok=size_ok)
        for name, f in (("same_text", lambda r, m: r.get("answer_norm") == m.get("answer_norm")),
                        ("same_strict", lambda r, m: r.get("strict") == m.get("strict")),
                        ("same_label", lambda r, m: r.get("label") == m.get("label"))):
            k = sum(1 for _q, _c, r, m in pairs if f(r, m))
            lo, hi = wilson(k, len(pairs))
            res[name] = {"k": k, "share": k / len(pairs) if pairs else None, "ci95": [lo, hi]}
            res[name + "_by_condition"] = {c: sum(1 for _q, cc, r, m in pairs if cc == c and f(r, m))
                                           for c in sorted(per)}
        res["strict_right_base"] = sum(1 for _q, _c, r, _m in pairs if r.get("strict") == 1)
        res["strict_right_main"] = sum(1 for _q, _c, _r, m in pairs if m.get("strict") == 1)
        res["right_to_wrong"] = sum(1 for _q, _c, r, m in pairs
                                    if r.get("strict") == 1 and m.get("strict") != 1)
        res["wrong_to_right"] = sum(1 for _q, _c, r, m in pairs
                                    if r.get("strict") != 1 and m.get("strict") == 1)
        gaps = sorted((parse_ts(m.get("ts_utc")) - parse_ts(r.get("ts_utc"))).total_seconds() / 3600
                      for _q, _c, r, m in pairs if parse_ts(m.get("ts_utc")) and parse_ts(r.get("ts_utc")))
        res["median_gap_hours"] = gaps[len(gaps) // 2] if gaps else None
        result[role] = res
        for q, c, r, m in pairs:
            pair_rows.append({"comparison": role, "qa_id": q, "condition": c,
                              "base_label": r.get("label"), "main_label": m.get("label"),
                              "base_strict": r.get("strict"), "main_strict": m.get("strict"),
                              "same_text": r.get("answer_norm") == m.get("answer_norm")})
        print("\n%s%s" % (label, "  [the D159 reading]" if role == "pilot_gemini" else
                          "  [information]"))
        print("  pairs %d (%s)%s" % (len(pairs), ", ".join("%s %d" % kv for kv in sorted(per.items())),
                                     "" if size_ok else "  FAIL: expected %s" % want))
        for name, text in (("same_text", "identical text after normalise"),
                           ("same_strict", "same strict score"), ("same_label", "same label")):
            v = res[name]
            print("  %-32s %3d of %d (%s, Wilson 95%% CI %s to %s)" % (
                text, v["k"], len(pairs), pct(v["k"], len(pairs)),
                pct(v["ci95"][0] * 1000, 1000) if v["ci95"][0] is not None else "-",
                pct(v["ci95"][1] * 1000, 1000) if v["ci95"][1] is not None else "-"))
        print("  strict right: %d in the earlier run, %d in main (right to wrong %d, wrong to right %d)"
              % (res["strict_right_base"], res["strict_right_main"], res["right_to_wrong"],
                 res["wrong_to_right"]))
        if res["median_gap_hours"] is not None:
            print("  median time between the two answers: %.1f h" % res["median_gap_hours"])
    rd = ctx.guard(ctx.reading_dir)
    write_json(rd / "d159_main.json", {"provenance": ctx.provenance(), "main_sha256": main_sha,
                                       "pilot_files": {k: v["sha"] for k, v in pilot.items()},
                                       "result": result})
    write_csv(rd / "d159_pairs.csv", pair_rows, ["comparison", "qa_id", "condition", "base_label",
                                                 "main_label", "base_strict", "main_strict",
                                                 "same_text"])
    print("\nWrote %s and %s" % (rd / "d159_main.json", rd / "d159_pairs.csv"))
    ctx.log_run({"command": "d159", "result": result})
    print("D159: %s" % ("DONE" if ok_all else "FAIL (pair counts)"))
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------
# 9. sample10 (Cell 6)
# ---------------------------------------------------------------------------

# The make-up of D166 (pilot): 4 open with the image, 2 without, 4 forced choice.
SLOTS = [
    ("S01", "Gemini, open, with the image, Bangla requested",
     lambda m, r, sp: m == "gemini" and sp.task == "open" and sp.image and sp.answer_lang == "bn"),
    ("S02", "Gemma, a naming answer, with the image, English requested",
     lambda m, r, sp: m == "gemma" and sp.task == "open" and sp.image and sp.answer_lang == "en"
     and r.get("kind") == "naming"),
    ("S03", "Qwen base, open, with the image",
     lambda m, r, sp: m == "qwen" and sp.task == "open" and sp.image),
    ("S04", "a fine-tuned seed, with the image (R-BB, R-EE)",
     lambda m, r, sp: m in FT_MODELS and sp.cid in ("R-BB", "R-EE")),
    ("S05", "no image, Bangla requested (L-BB0, R-BB0)",
     lambda m, r, sp: sp.cid in ("L-BB0", "R-BB0")),
    ("S06", "no image, English requested (L-EE0)", lambda m, r, sp: sp.cid == "L-EE0"),
    ("S07", "forced choice, Bangla, with the image (N-FCB)", lambda m, r, sp: sp.cid == "N-FCB"),
    ("S08", "forced choice, Bangla, no image (N-FCB0)", lambda m, r, sp: sp.cid == "N-FCB0"),
    ("S09", "forced choice, English, with the image (N-FCE)", lambda m, r, sp: sp.cid == "N-FCE"),
    ("S10", "forced choice, English, no image (N-FCE0)", lambda m, r, sp: sp.cid == "N-FCE0"),
]


def cmd_sample10(ctx, args):
    S = ctx.S
    print("Scoring and audit, part 1: the 10-item check, %s, run %s" % (SCRIPT_VERSION, utc_now()))
    best = {}
    for model in MODELS:
        recs, _sha = load_scored(ctx, model)
        for r in recs:
            if r.get("gold_status") != "ok":
                continue
            sp = S.condition_spec(r["condition_id"])
            rank = seeded_rank("sample10", model, r["key"])
            for code, _t, pred in SLOTS:
                if pred(model, r, sp) and (code not in best or rank < best[code][0]):
                    best[code] = (rank, model, r)
        del recs
    lex = ctx.lexicon()
    table = ctx.options()
    rows, used = [], set()
    for code, text, _p in SLOTS:
        if code not in best:
            print("%s  no record fits: %s" % (code, text))
            continue
        _rank, model, r = best[code]
        if r["key"] in used:
            continue
        used.add(r["key"])
        d = row_text(model, r)
        d.update(slot=code, slot_text=text, requested_lang=r.get("requested_lang"),
                 lang=r.get("lang"), fidelity=r.get("fidelity"), copy_term=r.get("copy_term"),
                 copy_sim=r.get("copy_sim"), fc_method=r.get("fc_method"))
        if r.get("gold_name_id"):
            d["gold_object"] = lex.objects[r["gold_name_id"]].name_bn
        sp = S.condition_spec(r["condition_id"])
        if sp.task == "fc":
            opts, _ci = ctx.FC.options_for(table, r["qa_id"], sp.question_lang)
            d["options"] = " | ".join(opts)
        rows.append(d)
        print("%s  %-12s %-7s %s" % (code, model, r["condition_id"], r["qa_id"]))
    rd = ctx.guard(ctx.reading_dir)
    cols = ["slot", "slot_text", "model", "condition", "qa_id", "qa_type", "kind", "gold_norm",
            "gold_object", "options", "answer_raw", "requested_lang", "lang", "label", "strict",
            "lenient", "fidelity", "refusal", "refusal_phrase", "content_found", "matched",
            "named_ids", "copy_term", "copy_sim", "found_value", "gold_value", "fc_choice",
            "fc_correct_index", "fc_method", "fc_valid", "truncated"]
    write_csv(rd / "TEXT_sample10.csv", rows, cols)
    print("Wrote %s (%d rows, with answer text: share it in the chat for the 10-item check)"
          % (rd / "TEXT_sample10.csv", len(rows)))
    ctx.log_run({"command": "sample10", "rows": [(r["slot"], r["model"], r["key"]) for r in rows]})
    return 0 if len(rows) == len(SLOTS) else 1


# ---------------------------------------------------------------------------
# 10. status (Cell 7)
# ---------------------------------------------------------------------------

def cmd_status(ctx, args):
    print("Scoring and audit, part 1: status, %s, run %s" % (SCRIPT_VERSION, utc_now()))
    for model in MODELS:
        p, m = ctx.scored_path(model), ctx.manifest_path(model)
        if p.exists():
            man = json.loads(m.read_text("utf-8")) if m.exists() else {}
            print("  %-12s scored %s, %s records, statuses %s" % (
                model, short(sha256_file(p)), man.get("records"), man.get("statuses")))
        else:
            print("  %-12s not scored yet" % model)
    for name in ("counts_summary.json", "d159_main.json", "TEXT_sample10.csv"):
        p = ctx.reading_dir / name
        print("  reading/%-24s %s" % (name, "present" if p.exists() else "not yet"))
    runs = ctx.records_dir / "runs.jsonl"
    if runs.exists():
        lines = [json.loads(x) for x in runs.read_text("utf-8").splitlines() if x.strip()]
        print("  runs.jsonl: %d entries; last: %s %s" % (
            len(lines), lines[-1].get("command"), lines[-1].get("utc")))
    print("  HT-4 sheet drawn: %s" % ("yes" if ht4_drawn(ctx) else "no (stage 2)"))
    return 0


# ---------------------------------------------------------------------------
# 11. selftest (Cell 1): the chain on made-up answers, real gate-E scorer
# ---------------------------------------------------------------------------

def selftest_world(gate_e, tmp):
    """A small made-up project folder: probe files, caches of the six models,
    an option file, pilot scored files, D192 inputs and an excluded list. Golds
    are names of the frozen lexicon and vocabulary; qa_ids start with H9."""
    S, FC, N, A9 = import_gate_e(gate_e)
    lex = S.load_lexicon(Path(gate_e) / "lexicon" / "lexicon.csv")
    voc = S.load_closed_vocab(Path(gate_e) / "lexicon" / "closed_vocab.csv")
    root = Path(tmp) / "drive"
    (root / "probes").mkdir(parents=True)
    obj = lex.objects
    form = lambda kind, cid, lang="bn", i=0: voc.forms[kind][cid][lang][i]   # noqa: E731
    madeup = {"d192_1": "ঝঝঝঝ", "d192_2": "ঞঞঞঞ", "d192_3": "ঢঢঢঢ"}
    jute = [f for k, c, f, b in S.genitive_only_forms(voc) if c == "plastic"][0]
    jute_bare = [b for k, c, f, b in S.genitive_only_forms(voc) if c == "plastic"][0]
    ceramic = [f for k, c, f, b in S.genitive_only_forms(voc) if c == "ceramic"][0]
    ceramic_bare = [b for k, c, f, b in S.genitive_only_forms(voc) if c == "ceramic"][0]

    def item(q, t, gold, house):
        return {"qa_id": q, "image_id": q.rsplit("_", 1)[0], "house_id": house, "capture_type":
                "close-up", "qa_type": t, "question_bn": "প্রশ্ন %s" % q, "answer_bn": gold,
                "image_file": q + ".jpg"}

    it = {
        "n1": item("H91_0001_q1", "obj_name", obj["N_cd3c2a98e1"].name_bn, "H91"),
        "n2": item("H91_0002_q1", "obj_name", obj["N_c84f4e8058"].name_bn, "H91"),
        "n3": item("H92_0003_q1", "obj_name", obj["N_4c3450ac9c"].name_bn, "H92"),
        "n4": item("H92_0004_q2", "object", obj["N_2f6f0442ed"].name_bn, "H92"),
        "n5": item("H93_0005_q1", "obj_name", obj[BLANKET_BAG].name_bn, "H93"),
        "n6": item("H93_0006_q1", "obj_name", obj["N_89cfba1d81"].name_bn, "H93"),
        "n7": item("H93_0007_q1", "obj_name", obj["N_f29daf73ee"].name_bn, "H93"),
        "c1": item("H91_0011_q4", "count", form("number", "1"), "H91"),
        "c2": item("H92_0012_q4", "count", form("number", "2"), "H92"),
        "c3": item("H92_0013_q5", "scene_id", form("room", "kitchen"), "H92"),
        "c4": item("H91_0014_q3", "obj_property", form("material", "wood"), "H91"),
        "c5": item("H91_0015_q3", "obj_property", form("material", "plastic") + " " + madeup["d192_1"], "H91"),
        "c6": item("H93_0016_q3", "obj_property", ceramic, "H93"),
        "c7": item("H92_0017_q3", "obj_property", form("colour", "white"), "H92"),
        "c8": item("H93_0018_q3", "obj_property", jute + " " + madeup["d192_3"], "H93"),
        "c9": item("H93_0019_q3", "obj_property", form("material", "coconut") + " " + madeup["d192_2"], "H93"),
    }
    probes = {"P-CORE-CLOSED": ["n1", "n4", "c1", "c2", "c3", "c4", "c5", "c7"],
              "R-TEST": ["n4", "n6", "c1", "c3", "c5", "c6", "c8", "c9"],
              "P-NAME": ["n1", "n2", "n3", "n5", "n7"],
              "P-CLUST": ["n1", "n2", "n3", "n7"]}
    builder = FC.Builder(lex)
    opt_lines, pfce = [], []
    for k in probes["P-CLUST"]:
        s = builder.make(it[k])
        bn, en = builder.records(s)
        opt_lines.append(FC.dump_record(bn))
        if en:
            opt_lines.append(FC.dump_record(en))
            pfce.append(k)
    probes["P-FCE"] = pfce
    if len(pfce) < 2 or "n1" not in pfce or "n2" not in pfce:
        raise Stop("selftest: the made-up P-FCE needs n1 and n2 (got %s)" % pfce)
    files = {"P-CORE-CLOSED": "P-CORE-CLOSED.json", "R-TEST": "R-TEST.json",
             "P-NAME": "P-NAME.json", "P-CLUST": "P-CLUST.json", "P-FCE": "P-FCE.json"}
    for p, ks in probes.items():
        (root / "probes" / files[p]).write_text(json.dumps({"items": [it[k] for k in ks]},
                                                           ensure_ascii=False), "utf-8")
    (root / "probes" / "fc_options.jsonl").write_text("\n".join(opt_lines) + "\n", "utf-8")
    table = FC.read_option_sets(root / "probes" / "fc_options.jsonl")
    exl = Path(tmp) / "excluded_items.csv"
    with open(exl, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["qa_id", "probes", "qa_type", "reason", "source", "added_at"])
        w.writerow([it["c3"]["qa_id"], "P-CORE-CLOSED|R-TEST", "scene_id", "other", "selftest", "selftest"])
        w.writerow([it["n7"]["qa_id"], "P-NAME|P-CLUST", "obj_name", "gold_wrong", "selftest", "selftest"])
    audit = root / "scored" / "asset9_stage2" / "audit"
    audit.mkdir(parents=True)
    rows = [{"kind": "material", "qa_type": "obj_property", "gold": it[k]["answer_bn"],
             "classes": "", "extra_words": madeup[f], "label_if_alone": "", "alone_scores_right": "False",
             "n_items": 1, "items_by_probe": "", "example_qa_id": it[k]["qa_id"]}
            for f, k in (("d192_1", "c5"), ("d192_2", "c9"), ("d192_3", "c8"))]
    rows.append({"kind": "colour", "qa_type": "obj_property", "gold": it["c7"]["answer_bn"] + " x",
                 "classes": "white", "extra_words": "x", "label_if_alone": "", "alone_scores_right":
                 "False", "n_items": 1, "items_by_probe": "", "example_qa_id": it["c7"]["qa_id"]})
    write_csv(audit / "extra_words.csv", rows, list(rows[0]))
    # the colour anchor gold is the plain white form in this made-up world
    ew = read_csv(audit / "extra_words.csv")
    for r in ew:
        if r["example_qa_id"] == it["c7"]["qa_id"]:
            r["gold"] = it["c7"]["answer_bn"]
    write_csv(audit / "extra_words.csv", ew, list(rows[0]))
    notes = [{"note": "genitive-only form", "source": k, "owner": c, "bn": f, "en": "",
              "how": "bare form not listed: %s" % b} for k, c, f, b in S.genitive_only_forms(voc)]
    write_csv(audit / "frozen_file_notes.csv", notes, ["note", "source", "owner", "bn", "en", "how"])

    # ---- answers --------------------------------------------------------------------
    refusal_bn = S.normalise("ছবি নেই")
    assert refusal_bn in S.REFUSAL_PHRASES, "selftest refusal phrase"
    one_bn = form("number", "1")

    def default_answer(k, cid):
        sp = S.condition_spec(cid)
        lang = sp.answer_lang or sp.question_lang
        i = it[k]
        if sp.task == "fc":
            opts, ci = FC.options_for(table, i["qa_id"], sp.question_lang)
            return (FC.BN_LABELS if sp.question_lang == "bn" else FC.EN_LABELS)[ci]
        if i["qa_type"] in ("obj_name", "object"):
            g = S.resolve_gold(i, lex, voc)
            o = obj[g.name_id]
            return o.name_bn if lang == "bn" else o.t1_en[0]
        g = S.resolve_gold(i, lex, voc)
        if g.kind == "count":
            return i["answer_bn"] if lang == "bn" else form("number", str(g.number), "en")
        cid0 = sorted(g.classes)[0]
        return form(g.kind, cid0, "bn" if lang == "bn" else "en")

    special = {
        ("gemini", "L-BB0", "c1"): refusal_bn + ", অনুগ্রহ করে " + one_bn + " ছবি দিন",   # P118 type
        ("gemma", "N-FCB0", "n1"): "উত্তর দেওয়া সম্ভব নয় (ক, খ, গ অথবা ঘ)",           # two+ labels, refusal
        ("gemini", "N-FCB0", "n2"): "ক বা খ, উত্তর খ",                                   # two+ labels, cue
        ("gemini", "N-FCE0", "n1"): "hard to say without a photo",                       # unlisted refusal
        ("qwen", "N-FCB", "n1"): "একটি কাঠের জিনিস দেখা যাচ্ছে যার উপর",                  # cut description
        ("gemma", "L-BB", "c4"): form("colour", "white") + " " + S.normalise("রঙের") + " " + form("material", "wood"),
        ("gemma", "L-BB", "c5"): madeup["d192_1"],
        ("gemma", "L-BE", "c5"): "cane",
        ("qwen_ft_s42", "R-BB", "c6"): ceramic_bare,
        ("qwen_ft_s42", "R-BB", "c8"): jute_bare,
        ("qwen_ft_s43", "R-BB", "c9"): madeup["d192_2"],
        ("gemini", "L-BB", "c7"): form("colour", "white") + " ও " + form("colour", "red"),
        ("gemma", "N-BB", "n5"): obj[BAG].name_bn,
        ("gemini", "N-BB", "n5"): "",                                                    # empty
    }
    trunc = {("qwen", "N-FCB", "n1")}
    conds = {"qwen": EXP_L + EXP_N + EXP_R, "gemma": EXP_L + EXP_N, "gemini": EXP_L + EXP_N}
    for m in FT_MODELS:
        conds[m] = EXP_R
    probe_of = {c: "P-CORE-CLOSED" for c in EXP_L}
    probe_of.update({"N-BB": "P-NAME", "N-BE": "P-NAME", "N-BBs": "P-NAME", "N-FCB": "P-CLUST",
                     "N-FCB0": "P-CLUST", "N-FCE": "P-FCE", "N-FCE0": "P-FCE",
                     "R-BB": "R-TEST", "R-EE": "R-TEST", "R-BB0": "R-TEST"})
    model_ids = {"qwen": "Qwen/Qwen3.5-4B", "gemma": "google/gemma-4-E4B-it",
                 "gemini": "gemini-3.6-flash", "qwen_ft_s42": "Qwen/Qwen3.5-4B+ft-s42",
                 "qwen_ft_s43": "Qwen/Qwen3.5-4B+ft-s43", "qwen_ft_s44": "Qwen/Qwen3.5-4B+ft-s44"}
    shared = set(probes["R-TEST"]) & set(probes["P-CORE-CLOSED"])

    def record(m, cid, k, answer, **extra):
        sp = S.condition_spec(cid)
        i = it[k]
        r = {"key": "|".join([model_ids[m], cid, "exp-v1", i["qa_id"]]), "model_id": model_ids[m],
             "model_key": m, "condition_id": cid, "prompt_version": "exp-v1", "qa_id": i["qa_id"],
             "house_id": i["house_id"], "qa_type": i["qa_type"], "question_lang": sp.question_lang,
             "answer_lang": sp.answer_lang or sp.question_lang, "forced_choice": sp.task == "fc",
             "has_image": sp.image, "answer_text": answer, "is_empty": not answer.strip(),
             "n_words": len(answer.split()), "finish_reason": "STOP", "block_reason": None,
             "truncated": (m, cid, k) in trunc, "mode": "batch" if m == "gemini" else "hf",
             "ts_utc": "2026-10-21T03:35:44Z", "stage": "main", "run_id": "selftest",
             "prompt_sha256": "p" * 12, "image_sha256": "i" * 12, "settings_sha256": "s" * 12}
        if m == "gemini":
            r["model_version"] = "gemini-3.6-flash"
        else:
            r["model_revision"] = "851bf6e806ef"
        if m in FT_MODELS:
            r["adapter_run"] = "ft-" + m[-3:]
            r["adapter_sha256"] = "a" * 64
        r.update(extra)
        return r

    expected = {"records": {}, "copies": 0}
    for m in MODELS:
        recs, by_key = [], {}
        for cid in conds[m]:
            for k in probes[probe_of[cid]]:
                if m == "qwen" and cid in EXP_R and k in shared:
                    src = by_key[(COPY_SOURCE[cid], k)]
                    r = dict(src, key="|".join([model_ids[m], cid, "exp-v1", it[k]["qa_id"]]),
                             condition_id=cid, mode="reuse", reused_from_key=src["key"],
                             ts_utc="2026-10-21T12:15:25Z")
                    expected["copies"] += 1
                else:
                    r = record(m, cid, k, special.get((m, cid, k), default_answer(k, cid)))
                recs.append(r)
                by_key[(cid, k)] = r
        d = root / "outputs" / "main" / m
        d.mkdir(parents=True)
        with open(d / "cache.jsonl", "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        expected["records"][m] = len(recs)

    # ---- pilot scored files (scored by the real scorer) ---------------------------------------
    pilot_items = ["c1", "c2", "c4", "n1", "n2", "n3"]
    pilot_cond = {k: ("L-BB" if k.startswith("c") else "N-BB") for k in pilot_items}
    pdir = root / "scored" / "pilot"
    pdir.mkdir(parents=True)
    (root / "outputs" / "main" / "_locks").mkdir(parents=True)
    pilot_answers = {
        "pilot_gemini": {}, "d5repeat_gemini": {"c2": form("number", "3")},
        "batchtest_gemini": {"n3": obj["N_9f5d811d75"].name_bn},
    }
    for role, model in (("pilot_gemini", "gemini"), ("d5repeat_gemini", "gemini"),
                        ("batchtest_gemini", "gemini"), ("pilot_qwen", "qwen"), ("pilot_gemma", "gemma")):
        recs = []
        for k in pilot_items:
            cid = "L-BB" if role == "batchtest_gemini" else pilot_cond[k]
            ans = pilot_answers.get(role, {}).get(k, default_answer(k, cid))
            r = record(model, cid, k, ans, stage="pilot", ts_utc="2026-10-14T11:00:00Z")
            recs.append(r)
        cache = Path(tmp) / ("pilot_%s.jsonl" % role)
        with open(cache, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        keep = ["prompt_sha256", "image_sha256", "settings_sha256", "mode", "ts_utc",
                "model_version" if model == "gemini" else "model_revision"]
        argv = [sys.executable, str(Path(gate_e) / "scripts" / "asset9_scorer.py"), "score",
                "--outputs", str(cache), "--probe", str(root / "probes" / "P-CORE-CLOSED.json"),
                "--probe", str(root / "probes" / "P-NAME.json"), "--excluded", str(exl),
                "--counts-only", "--out", str(pdir / ("%s_scored.jsonl" % role))]
        for kk in keep:
            argv += ["--keep", kk]
        p = subprocess.run(argv, capture_output=True, text=True, cwd=str(gate_e),
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        if p.returncode != 0:
            raise Stop("selftest pilot scoring failed: %s" % (p.stdout + p.stderr)[-800:])

    cfg = dict(REAL)
    cfg.update({
        "conditions_of": conds, "records_of": expected["records"], "copies": {"qwen": expected["copies"]},
        "adapter_sha": {m: "a" * 12 for m in FT_MODELS},
        "probe_sizes": {p: len(v) for p, v in probes.items()}, "scored_sizes": {},
        "options_sets": {"bn": len(probes["P-CLUST"]), "en": len(pfce)}, "excluded_rows": 2,
        "gate_e_pins": {k: (v if not k.startswith("probes/") else None)
                        for k, v in REAL["gate_e_pins"].items()},
        "drive_pins": {}, "pilot_files": {k: None for k in REAL["pilot_files"]},
        "pilot_records": {k: len(pilot_items) for k in REAL["pilot_files"]},
        "d159_pairs": {"L-BB": 3, "N-BB": 3},
        "d192_anchors": {"d192_1": it["c5"]["qa_id"], "d192_2": it["c9"]["qa_id"],
                         "d192_3": it["c8"]["qa_id"], "d192_4": it["c7"]["qa_id"]},
        "named_refusals": (it["n1"]["qa_id"],),
        "excluded_path": exl, "git_checks": False,
    })
    # gate-E pins of the code files only when they match (a sandbox may hold a stand-in)
    for rel in list(cfg["gate_e_pins"]):
        pin = cfg["gate_e_pins"][rel]
        if pin and not sha256_file(Path(gate_e) / rel).startswith(pin):
            cfg["gate_e_pins"][rel] = None
    want = {"copies": expected["copies"], "excluded_copies": 3, "p118": 1, "multi": 2,
            "multi_cue": 1, "fc_candidates": 1, "d192": {"d192_1": 2, "d192_2": 1, "d192_3": 1,
                                                         "d192_4": 1, "d192_5": 1, "d192_6": 2,
                                                         "d192_7": 1}}
    return root, cfg, want


def cmd_selftest(ctx_unused, args):
    gate_e = Path(args.gate_e).resolve()
    print("Scoring and audit, part 1: selftest on made-up answers, %s, run %s"
          % (SCRIPT_VERSION, utc_now()))
    tmp = Path(tempfile.mkdtemp(prefix="scoring_selftest_"))
    results = []

    def check(name, ok, detail=""):
        results.append((name, ok))
        print("  %s %s%s" % ("PASS" if ok else "FAIL", name, "" if not detail else ": " + detail))

    try:
        root, cfg, want = selftest_world(gate_e, tmp)
        ctx = Ctx(root, gate_e, None, cfg)
        print("Made-up project in %s" % root)
        print("---- check ----")
        check("check: all lines pass", cmd_check(ctx, args) == 0)
        print("---- score ----")
        sargs = argparse.Namespace(model=None, overwrite=False)
        check("score: six files", cmd_score(ctx, sargs) == 0)
        check("score again: nothing replaced", cmd_score(ctx, sargs) == 0)
        print("---- verify ----")
        check("verify: all pass", cmd_verify(ctx, args) == 0)
        _ok, _lines, summ = verify_one(ctx, "qwen")
        fc_ = summ.get("free_check", {})
        check("free check: every copy equal", fc_.get("equal") == want["copies"]
              and fc_.get("differ") == 0, str(fc_))
        check("free check: excluded copies counted", fc_.get("excluded") == want["excluded_copies"])
        print("---- counts ----")
        check("counts: runs", cmd_counts(ctx, args) == 0)
        summary = json.loads((ctx.reading_dir / "counts_summary.json").read_text("utf-8"))
        got192 = Counter()
        for k, v in summary["d192"].items():
            got192[k.split("|")[0]] += v.get("touched", 0)
        for f, n in want["d192"].items():
            check("counts: %s touched %d" % (f, n), got192[f] == n, "got %d" % got192[f])
        p118 = sum(v.get("count_phrase_and_correct", 0) for v in summary["d194_per_model"].values())
        check("counts: one count answer correct with a refusal phrase (P118 type)", p118 == want["p118"],
              "got %d" % p118)
        iss = read_csv(ctx.reading_dir / "counts_open_issues.csv")
        multi = sum(int(r["two_or_more_labels"] or 0) for r in iss)
        cue = sum(int(r["two_or_more_chosen_by_answer_word"] or 0) for r in iss)
        cand = sum(int(r["fc_possible_unlisted_refusal"] or 0) for r in iss)
        check("counts: forced-choice answers with two or more labels", multi == want["multi"], "got %d" % multi)
        check("counts: one of them chosen by an answer word", cue == want["multi_cue"], "got %d" % cue)
        check("counts: possible unlisted forced-choice refusal (cut answer left out)",
              cand == want["fc_candidates"], "got %d" % cand)
        check("counts: named pilot case found (Gemini N-FCB0 only)", summary["named_pilot_refusals"] == 1,
              "got %d" % summary["named_pilot_refusals"])
        print("---- d159 ----")
        check("d159: pair counts", cmd_d159(ctx, args) == 0)
        d = json.loads((ctx.reading_dir / "d159_main.json").read_text("utf-8"))["result"]
        check("d159: pilot against main identical", d["pilot_gemini"]["same_text"]["k"] == 6)
        check("d159: repeat differs on one item", d["d5repeat_gemini"]["same_strict"]["k"] == 5)
        check("d159: Batch test differs on one label", d["batchtest_gemini"]["same_label"]["k"] == 5)
        print("---- sample10 ----")
        check("sample10: ten rows", cmd_sample10(ctx, args) == 0)
        print("---- guard ----")
        try:
            ctx.guard(gate_e / "x.txt")
            check("no write inside the gate-E checkout", False)
        except Stop:
            check("no write inside the gate-E checkout", True)
        try:
            argparse_ns = argparse.Namespace(model=["gemma"], overwrite=True)
            (root / "human" / "HT-4" / "to_send").mkdir(parents=True)
            (root / "human" / "HT-4" / "to_send" / "x.xlsx").write_text("x")
            cmd_score(ctx, argparse_ns)
            check("no scored file replaced once an HT-4 sheet exists", False)
        except Stop:
            check("no scored file replaced once an HT-4 sheet exists", True)
        lo, hi = wilson(71, 97)
        check("Wilson interval equals F8 (71 of 97: 63.6% to 81.0%)",
              round(lo * 100, 1) == 63.6 and round(hi * 100, 1) == 81.0, "%.1f%% to %.1f%%" % (lo * 100, hi * 100))
    except Stop as exc:
        check("selftest stopped", False, str(exc))
    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    bad = [n for n, ok in results if not ok]
    print("SELFTEST: %d of %d checks PASS%s" % (len(results) - len(bad), len(results),
                                               "" if not bad else "; FAIL: " + "; ".join(bad)))
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# 12. main
# ---------------------------------------------------------------------------

def main(argv=None):
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description="Scoring and audit, part 1, stage 1 (D270 to D273)")
    ap.add_argument("--root", default="/content/drive/MyDrive/BanglaIndoorVQA",
                    help="the Drive project folder")
    ap.add_argument("--gate-e", default="/content/bivqa_gateE",
                    help="the detached checkout of the tag gate-E")
    ap.add_argument("--main", default=str(here.parents[1]), help="the main clone")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="read-only input check (Cell 0)")
    st = sub.add_parser("selftest", help="the chain on made-up answers (Cell 1)")
    st.add_argument("--keep-tmp", action="store_true")
    sc = sub.add_parser("score", help="score the six caches (Cell 2)")
    sc.add_argument("--model", action="append", choices=MODELS)
    sc.add_argument("--overwrite", action="store_true")
    sub.add_parser("verify", help="check the scored files (Cell 3)")
    sub.add_parser("counts", help="D192, D194 and open-issue counts (Cell 4)")
    sub.add_parser("d159", help="D5 pairs against main (Cell 5)")
    sub.add_parser("sample10", help="the 10-item check (Cell 6)")
    sub.add_parser("status", help="what exists (Cell 7)")
    args = ap.parse_args(argv)
    try:
        if args.command == "selftest":
            return cmd_selftest(None, args)
        ctx = Ctx(args.root, args.gate_e, args.main, real_cfg())
        return {"check": cmd_check, "score": cmd_score, "verify": cmd_verify,
                "counts": cmd_counts, "d159": cmd_d159, "sample10": cmd_sample10,
                "status": cmd_status}[args.command](ctx, args)
    except Stop as exc:
        print("STOP: %s" % exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
