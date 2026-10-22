#!/usr/bin/env python3
"""
scoring_audit.py  (Scoring and audit, part 1)
BanglaIndoorVQA main paper. Design v2.1 at the tag gate-E, DESIGN_ADDENDUM K onward.

Stage 1 scores every stage-main cache with the gate-E scorer and reads what
needs no human: the checks of the scored files, the free check of the 738
copied base answers (addendum L10), the D192 and D194 counts with the open
issues, the D159 comparison against main, and the 10-item sample.
Stage 2 runs the statistics reader on the scored files, draws the HT-4 sheets
with their written rules, takes the returned sheets, writes the consensus and
the D238 file, reads HT-4 with stats.ht4_stats, and creates the tag gate-F.

Commands (one notebook cell each; every command prints ids and counts only)
  check          read-only. One PASS or FAIL line per input.               Cell 0
  selftest       the whole chain on made-up answers in a temporary folder. Cell 1
  score          one gate-E `asset9_scorer.py score` command per cache.    Cell 2
  verify         the scored files against their caches; the free check.    Cell 3
  counts         D192, D194 and the open-issue counts (D272).              Cell 4
  d159           the 97 D5 pairs against main (D273).                      Cell 5
  sample10       ten seeded-random scored answers for the 10-item check.   Cell 6
  status         what exists so far.                                       Cell 7
  loadtest       stats.py build_world and load_scored on the six files.    Cell 8
  ht4-draw       the HT-4 sample, pools, two sheets, rules and key.        Cell 9
  ht4-receive    receipts of the returned sheets; the consensus sheet.     Cell 10
  ht4-consensus  the consensus; human/HT-4/ht4_labels.csv (D238).          Cell 11
  ht4-read       the HT-4 reading and its pass rule.                       Cell 12
  gate-f         tests and checks, then the annotated tag gate-F.          Cell 13

Rules built in (decision rows of DESIGN_ADDENDUM O)
  D270  one score command per cache, every probe of its conditions as a
        repeated --probe; checks before and after; no scored file replaced
        once an HT-4 sheet exists.
  D271  every score command runs from the detached gate-E checkout; this script
        imports scoring, fc_options, normalise and stats from that checkout.
        Nothing is written inside a Git tree.
  D272  the counting rules of the D192 findings, D194 and the open issues.
  D273  D159: main against the pilot on the 97 pairs, the repeat and the Batch
        test beside it.
  D274  HT-4 open-issue pools beside the 200 (pool 1 all 10 correct answers with
        a refusal phrase; pool 2 the 18 answer-word choices and 5 of 187; pool 3
        the 2 named cases and 5 of 26); fail above 10% only for pools of 10 or more.
  D275  HT-4 people: one preparer, two other checkers; separate orders and codes;
        receipts by sha256; consensus without the scorer's label.
  D276  HT-4 frame: gemini, qwen, gemma (no copies); 90% per consensus label only
        where n >= 10.
  D277  HT-4 sheet content and the written rules (ht4-rules-v1).
  D278  Gate F: HT-4 pass, unchanged files, main equal to gate-E for the scorer,
        445 tests with no failure and none skipped, then the annotated tag.

Files (Drive project folder D; nothing in the repo)
  D/scored/main/<model>_scored.jsonl and _manifest.json   (the scorer writes them)
  D/scored/main/_records/   check_*.json, verify_*.json, runs.jsonl, logs/
  D/scored/main/reading/    counts, D159 and HT-4 reading; files with answer text say so
  D/human/HT-4/             to_send/, key/ (never sent), received/, ht4_labels.csv,
                            ht4_open_issues.csv, ht4_refusal.csv
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

SCRIPT_VERSION = "scoring-audit-p1 2026-10-22b"
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
                   "scored/asset9_stage2/audit/frozen_file_notes.csv": "990935c55e24",
                   # Gate B file (addendum D11)
                   "english/english_questions_final.csv": "30f8286b4673"},
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
    "stats_world": True,
    # HT-4 (D274, D276, D277): sizes, the pools as counted in Cell 4, the pass rules
    "ht4": {"n": 200, "floor": 10, "judge_floor": 10, "n_boot": 10000,
            "pass_overall": 0.95, "pass_label": 0.90, "fail_share": 0.10, "r4": 0.90,
            "pool_expect": {"phrase_correct": 10, "multi_cued": 18, "multi_uncued": 187,
                            "fc_candidates": 28, "named": 2},
            "pool_take": {"multi_uncued": 5, "fc_candidates": 5},
            "tests_min": 445},
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
    P = ht4_paths(ctx)
    print("  HT-4 sheets drawn: %s" % ("yes" if P["key_csv"].exists() else "no"))
    got = sorted({r["who"] for r in receipts(P)}) if P["receipts"].exists() else []
    print("  HT-4 receipts: %s" % (", ".join(got) or "none"))
    print("  HT-4 labels (D238): %s" % ("written" if P["labels"].exists() else "not yet"))
    if P["reading"].exists():
        rd = json.loads(P["reading"].read_text("utf-8"))
        print("  HT-4 reading: %s" % ("PASS" if rd.get("pass") else "FAIL"))
    else:
        print("  HT-4 reading: not yet")
    if ctx.cfg.get("git_checks") and ctx.main is not None:
        t = git(ctx.main, "rev-parse", "-q", "--verify", "refs/tags/gate-F")[0] == 0
        print("  tag gate-F (local): %s" % ("present" if t else "not yet"))
    return 0


# ---------------------------------------------------------------------------
# 11b. Stage 2: the statistics reader, HT-4 and Gate F (D274 to D278)
# ---------------------------------------------------------------------------

HT4_RULES_VERSION = "ht4-rules-v1"
HT4_MODELS = ("gemini", "qwen", "gemma")                       # D276
HT4_NAMING = ("tier1", "tier2", "sibling", "ambiguous", "english_copy", "other_wrong")
HT4_CLOSED = ("correct", "wrong")
HT4_FC = ("correct", "wrong", "invalid")
HT4_ALL = HT4_NAMING + ("correct", "wrong", "invalid")
HT4_COLUMNS = ("sample_id", "model_key", "condition_id", "qa_id", "house_id",
               "scorer_label", "checker1_label", "checker2_label", "consensus_label")  # D238
SHEET_COLS = ("code", "row_type", "asked_in", "answer_requested_in", "question",
              "gold_answer", "reference", "model_answer", "your_label", "refusal_yes_no", "note")
CONS_COLS = ("code", "row_type", "asked_in", "answer_requested_in", "question", "gold_answer",
             "reference", "model_answer", "checker1_label", "checker2_label",
             "checker1_refusal", "checker2_refusal", "consensus_label", "consensus_refusal", "note")
KEY_COLS = ("code_c1", "code_c2", "key", "model_key", "condition_id", "qa_id", "house_id",
            "row_type", "scorer_label", "scorer_refusal", "in_sample", "sample_id", "pools")
LANG_NAME = {"bn": "Bangla", "en": "English"}
CHECKERS = ("C1", "C2")


def allowed_labels(row_type):
    return HT4_NAMING if row_type == "naming" else HT4_FC if row_type == "forced choice" \
        else HT4_CLOSED


def row_type_of(r):
    if r.get("task") == "fc":
        return "forced choice"
    return r.get("kind") or "?"


def stratum_lang(S, r):
    sp = S.condition_spec(r["condition_id"])
    return sp.answer_lang or sp.question_lang


def ht4_rules(S):
    """The written rules for the HT-4 checkers (D277). Every rule restates the
    scorer 0.2.2-stage2 (Design 8; D103 to D109, D114, D117, D120, D169, D185, D194)."""
    endings = " ".join(sorted(S.BN_SUFFIXES, key=len))
    return [
        "HT-4 WRITTEN RULES (%s, scorer %s)" % (HT4_RULES_VERSION, S.SCORER_VERSION),
        "",
        "YOUR TASK",
        "1. Each row of the tab 'Sheet' holds one model answer. Give it the label these rules give. "
        "Choose it from the list in the cell (column your_label).",
        "2. Also answer in refusal_yes_no: does the answer refuse, that is, say it cannot or will not "
        "answer (for example, there is no image, it cannot see, it does not know)? yes or no. Judge by "
        "meaning.",
        "3. You see the question as the model saw it, the gold answer, a reference box and the model "
        "answer. You do not see the picture or the scorer's label. Judge only by these rules, never "
        "by the picture or by what you think is true.",
        "4. If a rule gives a label you think is wrong, still give the rule's label and write why in "
        "the column note.",
        "5. Work alone. Do not use AI tools, web search or another person. Do not open the Drive "
        "folders scored/ or human/HT-4/key/, and do not talk about rows with the other checker, until "
        "both sheets are returned (D168).",
        "6. Excluded items never appear on this sheet (D185). Every row needs a label and a refusal "
        "answer. Do not change the code column, the order of columns or the tab names.",
        "7. Time: about 2 hours. You may stop and go on later.",
        "",
        "PREPARE THE ANSWER (every row)",
        "8. Ignore upper and lower case, extra spaces, quote marks, and the full stop, dari or other "
        "punctuation at the end.",
        "9. Remove one lead-in at the start: এটি একটি, এটা, এটি, it is a, this is a, it's a.",
        "10. Names match whole words only.",
        "11. Spacing rule (D103): a name matches with or without a space or hyphen between its words "
        "(সুইচ বোর্ড, সুইচবোর্ড, সুইচ-বোর্ড).",
        "12. Ending rule (D104): the last word of a name may carry up to two Bangla endings from the "
        "tab 'Endings' (বালতিটির = বালতি + টি + র); an English word may carry a plural (-s, -es, -ies) "
        "or 's. This counts only when what is left is a listed name. Any other spelling difference "
        "does not match.",
        "13. Language does not change the label (D105). A right name or value in the other language "
        "still gets tier1 or correct. The scorer checks the language separately.",
        "",
        "NAMING ROWS (row_type naming). Labels: tier1, tier2, sibling, ambiguous, english_copy, "
        "other_wrong",
        "14. The reference box shows the gold object's Tier 1 names (Bangla and English), its Tier 2 "
        "(generic) names and its cluster siblings. The tab 'Lexicon' lists every object with its names.",
        "15. Find every name of the Lexicon tab in the answer: Tier 1 or Tier 2, Bangla or English, of "
        "any object. Where names overlap, the longest match wins (পড়ার টেবিল, not টেবিল).",
        "16. Generic-name rule: a found name that is in the gold's Tier 2 list counts as a gold Tier 2 "
        "name, even if it is another object's Tier 1 name (টেবিল for a পড়ার টেবিল).",
        "17. Another object is named only by one of its Tier 1 names. A name that is only another "
        "object's Tier 2 name names nothing. A name shared by several objects (pot, cooking pot, bowl, "
        "jar, strainer, blanket, quilt, rug, hanger) counts as one; if it is in the gold's Tier 1 list "
        "it is the gold's own name.",
        "18. tier1: a gold Tier 1 name is found and no other object is named.",
        "19. ambiguous: a gold Tier 1 name and another object are named; or two or more different "
        "other objects are named.",
        "20. sibling: exactly one other object is named and it is a cluster sibling of the gold (a gold "
        "Tier 2 name beside it does not change this).",
        "21. other_wrong: exactly one other object is named and it is not a cluster sibling.",
        "22. tier2: no gold Tier 1 name, no other object named, and a gold Tier 2 name is found.",
        "23. english_copy (D109): nothing at all from the Lexicon tab is found, the answer is mostly in "
        "Bangla letters, and its last one, two or three Bangla words, read aloud, are one of the gold's "
        "English Tier 1 names written in Bangla letters, with no ending removed. The scorer measures "
        "this as a spelling similarity of 0.6 or more after a fixed romanisation.",
        "24. other_wrong: everything else, including empty answers, refusals and answers that name "
        "nothing listed.",
        "",
        "CLOSED ROWS (row_type count, room, material, colour). Labels: correct, wrong",
        "25. The reference box shows the gold number, or the gold classes with their accepted forms. "
        "The tab 'Vocabulary' lists every class and form.",
        "26. count: find the first number in the answer: a Bangla or Latin digit, or a number word of "
        "the Vocabulary tab (kind number), Bangla or English, with or without an ending (একটি, দুটো, "
        "তিনটা). correct if it equals the gold number; wrong if it differs or if there is no number. "
        "Only the first number counts. The bare word নয় is not a number (D120). An answer that also "
        "refuses is still correct when its first number equals the gold (D106, D194).",
        "27. room: find every room form of the Vocabulary tab. correct if the forms found name exactly "
        "one room class and it is the gold class; wrong if none, another class, or two classes.",
        "28. material and colour: find every form of that kind in the Vocabulary tab. correct if at "
        "least one class is found and every class found is in the gold list; else wrong. হালকা and গাঢ় "
        "are not colours. A gold such as সবুজ-হলুদ is the set green and yellow.",
        "",
        "FORCED-CHOICE ROWS (row_type forced choice). Labels: correct, wrong, invalid",
        "29. The reference box shows the four options with their letters, as the model saw them, and "
        "the gold option.",
        "30. Only the letters that were shown count (D107): ক খ গ ঘ for Bangla lists, A B C D for "
        "English lists. A letter of the other script, or a digit, chooses nothing by itself (rule "
        "34 may still apply).",
        "31. An English 'a' is a letter only when it stands alone, is marked like a letter (a) (a) a. "
        "a: [a]), comes right after option, choice or letter, or is the last word. In '(a wall-mounted "
        "cabinet)' it is the article (rule N, D169).",
        "32. One distinct letter named (it may be repeated): that option is chosen.",
        "33. Two or more different letters named: the option chosen is the one letter that has an "
        "answer word (উত্তর, উত্তরঃ, উত্তরটি, answer) within the three words before it. If no letter, "
        "or more than one, is marked this way, nothing is chosen (rule O, D169).",
        "34. No shown letter at all: if the answer names the text of exactly one option (rules 11 and 12 "
        "apply), that option is chosen; otherwise nothing is chosen.",
        "35. correct: the gold option is chosen. wrong: another option is chosen. invalid: nothing is "
        "chosen, which includes refusals and empty answers.",
        "",
        "BACKGROUND",
        "36. The tab 'Refusal phrases' lists the %d phrases the scorer reads (D106, D194). They do not "
        "change any label above." % len(S.REFUSAL_PHRASES),
        "37. Gold answers were checked to equal one listed name or class exactly (D108).",
        "38. A short name or an English word in Bangla letters accepted in one language is accepted in "
        "the other too; the lists already show this (D117). Whether a Bangla name is an English word "
        "(D114) does not change any label.",
        "",
        "Endings (rule 12): " + endings,
    ]


def cons_rules():
    return [
        "HT-4 CONSENSUS (%s)" % HT4_RULES_VERSION,
        "",
        "1. Both checkers sit together. This tab lists only the rows where your labels or your "
        "refusal answers differ. You see both answers; you never see the scorer's label.",
        "2. For each row agree on one label under the written rules (tab 'Read me' of your own sheet) "
        "and write it in consensus_label; agree on yes or no in consensus_refusal. Where you already "
        "agree, the cell is filled; leave it.",
        "3. Do not change the code column. Use no AI tool and no other person.",
        "4. Time: about 0.5 hours.",
    ]


def import_stats(ctx):
    """src/stats.py of the gate-E checkout: THE reader of the scored files and of
    the HT-4 file (J5, D238). Never a copy."""
    src = str((ctx.gate_e / "src").resolve())
    import stats as ST                                       # noqa: E402
    if not str(Path(ST.__file__).resolve()).startswith(src):
        raise Stop("stats was imported from %s, not from the gate-E checkout" % ST.__file__)
    return ST


def stats_reader(ctx, echo=True):
    """build_world and load_scored of stats.py on the six scored files, exactly as
    the Statistics chat will run them (mode real)."""
    ST = import_stats(ctx)
    checks = ST.Checks(echo=echo)
    paths = ST.Paths.make(ctx.gate_e, ctx.root)
    world = ST.build_world(paths, checks, need_dataset=False)
    recs = ST.load_scored([str(ctx.scored_dir / "*_scored.jsonl")], world, checks, "real")
    return ST, checks, world, recs


def cmd_loadtest(ctx, args):
    print("Scoring and audit, part 1: the statistics reader on the scored files, %s, run %s"
          % (SCRIPT_VERSION, utc_now()))
    if not ctx.cfg.get("stats_world"):
        print("LOADTEST: skipped (selftest)")
        return 0
    try:
        ST, checks, _world, recs = stats_reader(ctx)
    except Exception as exc:                                   # report, never hide
        print("LOADTEST: FAIL (%s: %s)" % (type(exc).__name__, exc))
        return 1
    n_fail = checks.count("FAIL")
    print("stats.py %s at %s; %d check lines, %d FAIL; %d model-condition sets"
          % (ST.STATS_VERSION, ST.__file__, len(checks.rows), n_fail, len(recs.data)))
    ctx.log_run({"command": "loadtest", "fail": n_fail, "sets": len(recs.data)})
    print("LOADTEST: %s" % ("PASS" if n_fail == 0 else "FAIL"))
    return 0 if n_fail == 0 else 1


# ---- the draw ---------------------------------------------------------------

def proportional(total, weights, caps):
    """Integer shares of total, proportional to weights, never above caps
    (largest remainder; ties in key order)."""
    alloc = {k: 0 for k in weights}
    remaining = total
    active = [k for k in sorted(weights, key=str) if caps[k] > 0 and weights[k] > 0]
    while remaining > 0 and active:
        wsum = float(sum(weights[k] for k in active))
        quota = {k: remaining * weights[k] / wsum for k in active}
        given = 0
        for k in active:
            add = min(int(quota[k]), caps[k] - alloc[k])
            alloc[k] += add
            given += add
        remaining -= given
        for k in sorted(active, key=lambda k: (-(quota[k] - int(quota[k])), str(k))):
            if remaining == 0:
                break
            if alloc[k] < caps[k]:
                alloc[k] += 1
                remaining -= 1
        active = [k for k in active if alloc[k] < caps[k]]
    return alloc


def ht4_allocate(cells, n, floor):
    """cells: {(model, lang, label): [records sorted by seeded rank]}. At least
    `floor` per scorer label (or all there are), then the rest split equally over
    the three models and, within a model, in proportion to the stratum sizes."""
    take = Counter()
    order = sorted(cells, key=lambda c: (HT4_MODELS.index(c[0]), c[1], c[2]))
    for lab in HT4_ALL:
        lc = [c for c in order if c[2] == lab]
        need = min(floor, sum(len(cells[c]) for c in lc))
        got = 0
        while got < need:
            moved = False
            for c in lc:
                if got < need and take[c] < len(cells[c]):
                    take[c] += 1
                    got += 1
                    moved = True
            if not moved:
                break
    rest = n - sum(take.values())
    if rest < 0:
        raise Stop("the label floor alone needs %d rows, more than %d" % (sum(take.values()), n))
    cap_m = {m: sum(len(cells[c]) - take[c] for c in order if c[0] == m) for m in HT4_MODELS}
    share = proportional(rest, {m: 1 for m in HT4_MODELS}, cap_m)
    for m in HT4_MODELS:
        mc = [c for c in order if c[0] == m]
        got = proportional(share[m], {c: len(cells[c]) for c in mc},
                           {c: len(cells[c]) - take[c] for c in mc})
        for c, k in got.items():
            take[c] += k
    if sum(take.values()) != min(n, sum(len(v) for v in cells.values())):
        raise Stop("allocation gave %d rows, not %d" % (sum(take.values()), n))
    return take


def ht4_frame(ctx):
    """Scored records of the three models: the sampling frame (no copies, no
    excluded items) and the D274 pools."""
    S = ctx.S
    frame, pools = [], {"phrase_correct": [], "multi_cued": [], "multi_uncued": [],
                        "fc_candidates": [], "named": []}
    shas = {}
    for model in HT4_MODELS:
        recs, sha = load_scored(ctx, model)
        shas[model] = sha
        for r in recs:
            if r.get("gold_status") != "ok" or r.get("mode") == "reuse":
                continue
            r = dict(r, model_key_run=model)
            frame.append(r)
            sp = S.condition_spec(r["condition_id"])
            if r.get("refusal_phrase") is True and r.get("lenient") == 1:
                pools["phrase_correct"].append(r)
            if sp.task == "fc":
                k = fc_label_count(S, r, sp.question_lang)
                if k >= 2:
                    pools["multi_cued" if r.get("fc_valid") is True else "multi_uncued"].append(r)
                if (r.get("label") == S.INVALID and r.get("refusal") is not True
                        and not r.get("is_empty") and r.get("fc_cross_label") is not True
                        and r.get("truncated") is not True):
                    pools["fc_candidates"].append(r)
                if model == "gemini" and r["qa_id"] in ctx.cfg["named_refusals"] \
                        and r["condition_id"] == "N-FCB0":
                    pools["named"].append(r)
        del recs
    for m in MODELS:
        if m not in shas:
            shas[m] = sha256_file(ctx.scored_path(m))
    return frame, pools, shas


def ht4_reference(ctx, r, table):
    S, lex, voc = ctx.S, ctx.lexicon(), ctx.vocab()
    rt = row_type_of(r)
    if rt == "forced choice":
        sp = S.condition_spec(r["condition_id"])
        opts, ci = ctx.FC.options_for(table, r["qa_id"], sp.question_lang)
        labs = ("ক", "খ", "গ", "ঘ") if sp.question_lang == "bn" else ("A", "B", "C", "D")
        return "\n".join(["%s) %s" % (labs[i], o) for i, o in enumerate(opts)]
                         + ["Gold: %s) %s" % (labs[ci], opts[ci])])
    if rt == "naming":
        obj = lex.objects[r["gold_name_id"]]
        sib = sorted(lex.objects[s].name_bn for s in lex.siblings_of(obj.name_id))
        return "\n".join([
            "Tier 1, Bangla: " + (" | ".join(obj.t1_bn) or "-"),
            "Tier 1, English: " + (" | ".join(obj.t1_en) or "-"),
            "Tier 2 (generic): " + (" | ".join(obj.t2_bn + obj.t2_en) or "-"),
            "Cluster siblings: " + (" | ".join(sib) or "none")])
    if rt == "count":
        return "Gold number: %s" % r.get("gold_value")
    out = []
    for cls in r.get("gold_value") or []:
        f = voc.forms[rt].get(cls, {})
        out.append("%s: %s / %s" % (cls, " | ".join(f.get("bn", [])) or "-",
                                    " | ".join(f.get("en", [])) or "-"))
    return "Gold %s class(es):\n%s" % (rt, "\n".join(out))


def english_questions(ctx):
    path = ctx.root / "english" / "english_questions_final.csv"
    s = sha256_file(path)
    pin = ctx.cfg["drive_pins"].get("english/english_questions_final.csv")
    if not pin_ok(s, pin):
        raise Stop("english_questions_final.csv has sha %s, expected %s" % (short(s), pin))
    rows = read_csv(path)
    if not rows or not {"question_bn_norm", "question_en"} <= set(rows[0]):
        raise Stop("english_questions_final.csv columns are %s" % (list(rows[0]) if rows else []))
    return {ctx.normalise(r["question_bn_norm"]): r["question_en"] for r in rows}, s


def lexicon_rows(ctx):
    lex = ctx.lexicon()
    rows = []
    for oid in sorted(lex.objects, key=lambda i: lex.objects[i].name_bn):
        o = lex.objects[oid]
        rows.append([o.name_bn, " | ".join(o.t1_bn[1:]), " | ".join(o.t1_en),
                     " | ".join(o.t2_bn), " | ".join(o.t2_en), o.cluster_id or ""])
    return rows


def vocab_rows(ctx):
    voc = ctx.vocab()
    rows = []
    for kind in ctx.S.VOCAB_KINDS:
        for cls in sorted(voc.forms[kind], key=lambda c: (len(c), c)):
            f = voc.forms[kind][cls]
            rows.append([kind, cls, " | ".join(f.get("bn", [])), " | ".join(f.get("en", []))])
    return rows


def write_workbook(path, tabs, main_tab, columns, rows, dropdowns):
    """tabs: [(name, header or None, rows)]; the main tab gets list validation
    per row (dropdowns: {(row_index, column_name): [values]})."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    wb.remove(wb.active)
    for name, header, trows in tabs:
        ws = wb.create_sheet(name)
        if header:
            ws.append(list(header))
            for c in ws[1]:
                c.font = Font(bold=True)
        for tr in trows:
            ws.append(list(tr) if isinstance(tr, (list, tuple)) else [tr])
        ws.column_dimensions["A"].width = 120 if header is None else 24
    ws = wb.create_sheet(main_tab, 0)
    ws.append(list(columns))
    for c in ws[1]:
        c.font = Font(bold=True)
    widths = {"code": 9, "row_type": 13, "asked_in": 9, "answer_requested_in": 11, "question": 40,
              "gold_answer": 18, "reference": 42, "model_answer": 48, "your_label": 14,
              "refusal_yes_no": 10, "note": 30, "checker1_label": 14, "checker2_label": 14,
              "checker1_refusal": 10, "checker2_refusal": 10, "consensus_label": 15,
              "consensus_refusal": 12}
    for i, col in enumerate(columns):
        ws.column_dimensions[ws.cell(row=1, column=i + 1).column_letter].width = widths.get(col, 14)
    for row in rows:
        ws.append([row.get(c, "") for c in columns])
    wrap = Alignment(wrap_text=True, vertical="top")
    for line in ws.iter_rows(min_row=2):
        for c in line:
            c.alignment = wrap
    groups = defaultdict(list)
    for (ri, col), values in dropdowns.items():
        groups[(col, tuple(values))].append(ri)
    for (col, values), ris in groups.items():
        dv = DataValidation(type="list", formula1='"%s"' % ",".join(values), allow_blank=True)
        ws.add_data_validation(dv)
        letter = ws.cell(row=1, column=columns.index(col) + 1).column_letter
        for ri in ris:
            dv.add("%s%d" % (letter, ri + 2))
    ws.freeze_panes = "B2"
    wb.save(str(path))


def read_workbook(path, tab, need):
    from openpyxl import load_workbook
    wb = load_workbook(str(path), data_only=True)
    if tab not in wb.sheetnames:
        raise Stop("%s has no tab %r (tabs: %s)" % (Path(path).name, tab, wb.sheetnames))
    ws = wb[tab]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    miss = [c for c in need if c not in header]
    if miss:
        raise Stop("%s, tab %s: missing columns %s" % (Path(path).name, tab, miss))
    out = []
    for r in rows[1:]:
        d = {h: ("" if v is None else str(v).strip()) for h, v in zip(header, r)}
        if any(d.values()):
            out.append(d)
    return out


def ht4_paths(ctx):
    d = ctx.ht4_dir
    return {"to_send": d / "to_send", "key": d / "key", "received": d / "received",
            "key_csv": d / "key" / "ht4_key.csv", "draw": d / "key" / "ht4_draw.json",
            "sheet": {c: d / "to_send" / ("HT4_sheet_%s.xlsx" % c) for c in CHECKERS},
            "returned": {c: d / "received" / ("HT4_sheet_%s_returned.xlsx" % c) for c in CHECKERS},
            "cons_sheet": d / "to_send" / "HT4_consensus.xlsx",
            "cons_returned": d / "received" / "HT4_consensus_returned.xlsx",
            "receipts": d / "received" / "receipts.jsonl",
            "labels": d / "ht4_labels.csv", "issues": d / "ht4_open_issues.csv",
            "refusal": d / "ht4_refusal.csv", "rules": d / "to_send" / "HT4_written_rules.txt",
            "reading": ctx.reading_dir / "ht4_reading.json"}


def cmd_ht4_draw(ctx, args):
    S = ctx.S
    hc = ctx.cfg["ht4"]
    P = ht4_paths(ctx)
    print("Scoring and audit, part 1: HT-4 draw (D274, D276, D277), %s, run %s"
          % (SCRIPT_VERSION, utc_now()))
    frame, pools, shas = ht4_frame(ctx)
    # pools as decided (D274): the sizes must be the counts read in this chat
    sizes = {k: len(v) for k, v in pools.items()}
    want = hc.get("pool_expect")
    print("Pools found: %s" % ", ".join("%s %d" % kv for kv in sizes.items()))
    if want and sizes != want:
        raise Stop("the pools differ from the counts D274 was decided on: %s, expected %s"
                   % (sizes, want))
    rank = lambda tag, r: seeded_rank("ht4", tag, r["key"])      # noqa: E731
    cells = defaultdict(list)
    for r in frame:
        cells[(r["model_key_run"], stratum_lang(S, r), r["label"])].append(r)
    for c in cells:
        cells[c].sort(key=lambda r: rank("sample", r))
    take = ht4_allocate(cells, hc["n"], hc["floor"])
    sample = [r for c in cells for r in cells[c][:take[c]]]
    sample.sort(key=lambda r: rank("sample-id", r))
    sid = {r["key"]: "H4-%03d" % (i + 1) for i, r in enumerate(sample)}
    pick = {"phrase_correct": pools["phrase_correct"],
            "multi_cued": pools["multi_cued"],
            "multi_uncued": sorted(pools["multi_uncued"],
                                   key=lambda r: rank("pool2", r))[:hc["pool_take"]["multi_uncued"]],
            "named": pools["named"]}
    named_keys = {r["key"] for r in pools["named"]}
    pick["fc_candidates"] = sorted([r for r in pools["fc_candidates"] if r["key"] not in named_keys],
                                   key=lambda r: rank("pool3", r))[:hc["pool_take"]["fc_candidates"]]
    pool_of = defaultdict(list)
    for name, rs in pick.items():
        tag = {"phrase_correct": "pool1", "multi_cued": "pool2", "multi_uncued": "pool2",
               "named": "pool3", "fc_candidates": "pool3"}[name]
        for r in rs:
            if tag not in pool_of[r["key"]]:
                pool_of[r["key"]].append(tag)
    rows = {r["key"]: r for r in sample}
    for name in ("phrase_correct", "multi_cued", "multi_uncued", "named", "fc_candidates"):
        for r in pick[name]:
            rows.setdefault(r["key"], r)
    # sheet content
    probes = ctx.probes()
    items = {}
    for name in PROBE_ORDER:
        for q, it in probes[name].items():
            items.setdefault(q, it)
    en_map, en_sha = english_questions(ctx)
    table = ctx.options()
    order = {c: sorted(rows.values(), key=lambda r: rank("order-" + c, r)) for c in CHECKERS}
    code = {c: {r["key"]: "%s%03d" % ("P" if c == "C1" else "Q", i + 1)
                for i, r in enumerate(order[c])} for c in CHECKERS}
    content = {}
    for k, r in rows.items():
        it = items[r["qa_id"]]
        sp = S.condition_spec(r["condition_id"])
        q = it["question_bn"]
        if sp.question_lang == "en":
            qn = ctx.normalise(q)
            if qn not in en_map:
                raise Stop("no English question for %s (asked in English)" % r["qa_id"])
            q = en_map[qn]
        content[k] = {"row_type": row_type_of(r), "asked_in": LANG_NAME[sp.question_lang],
                      "answer_requested_in": LANG_NAME.get(sp.answer_lang or sp.question_lang),
                      "question": q, "gold_answer": it["answer_bn"],
                      "reference": ht4_reference(ctx, r, table),
                      "model_answer": r.get("answer_raw") or ""}
    key_rows = []
    for k, r in sorted(rows.items(), key=lambda kv: code["C1"][kv[0]]):
        key_rows.append({"code_c1": code["C1"][k], "code_c2": code["C2"][k], "key": k,
                         "model_key": r["model_key_run"], "condition_id": r["condition_id"],
                         "qa_id": r["qa_id"], "house_id": r.get("house_id"),
                         "row_type": row_type_of(r), "scorer_label": r["label"],
                         "scorer_refusal": "yes" if r.get("refusal") is True else "no",
                         "in_sample": "yes" if k in sid else "no", "sample_id": sid.get(k, ""),
                         "pools": "|".join(pool_of.get(k, []))})
    for d in (P["to_send"], P["key"], P["received"]):
        ctx.guard(d).mkdir(parents=True, exist_ok=True)
    if P["key_csv"].exists():
        old = read_csv(P["key_csv"])
        same = [tuple(r.get(c, "") for c in KEY_COLS) for r in old] == \
            [tuple("" if r[c] is None else str(r[c]) for c in KEY_COLS) for r in key_rows]
        if not same:
            raise Stop("an HT-4 draw already exists and differs; a draw is never replaced (D277)")
        print("The HT-4 draw already exists and is identical; nothing rewritten.")
        return 0
    rules = ht4_rules(S)
    ctx.guard(P["rules"]).write_text("\n".join(rules) + "\n", encoding="utf-8")
    ref_rows = [[p] for p in S.REFUSAL_PHRASES]
    end_rows = [[e] for e in sorted(S.BN_SUFFIXES, key=len)]
    for c in CHECKERS:
        srows, dd = [], {}
        for i, r in enumerate(order[c]):
            d = dict(content[r["key"]], code=code[c][r["key"]])
            srows.append(d)
            dd[(i, "your_label")] = allowed_labels(d["row_type"])
            dd[(i, "refusal_yes_no")] = ("yes", "no")
        write_workbook(ctx.guard(P["sheet"][c]),
                       [("Read me", None, [[x] for x in rules]),
                        ("Lexicon", ("name_bn", "other Tier 1 Bangla", "Tier 1 English",
                                     "Tier 2 Bangla", "Tier 2 English", "cluster"),
                         lexicon_rows(ctx)),
                        ("Vocabulary", ("kind", "class", "forms Bangla", "forms English"),
                         vocab_rows(ctx)),
                        ("Refusal phrases", ("phrase",), ref_rows),
                        ("Endings", ("ending",), end_rows)],
                       "Sheet", SHEET_COLS, srows, dd)
    write_csv(ctx.guard(P["key_csv"]), key_rows, KEY_COLS)
    by_cell = sorted(((c[0], c[1], c[2], take[c], len(cells[c])) for c in cells if take[c]),
                     key=lambda x: (HT4_MODELS.index(x[0]), x[1], x[2]))
    manifest = {"provenance": ctx.provenance(), "rules_version": HT4_RULES_VERSION,
                "scored_sha256": shas, "english_sha256": en_sha,
                "options_sha256": sha256_file(ctx.options_path),
                "n_sample": len(sample), "n_rows": len(rows), "pools_found": sizes,
                "pools_shown": {k: len(v) for k, v in pick.items()},
                "allocation": [list(x) for x in by_cell],
                "sheets": {c: sha256_file(P["sheet"][c]) for c in CHECKERS},
                "key_sha256": sha256_file(P["key_csv"])}
    write_json(ctx.guard(P["draw"]), manifest)
    print("Sample: %d answers of %s (no copies, no excluded items); at least %d per scorer label "
          "where there are that many (D276)" % (len(sample), ", ".join(HT4_MODELS), hc["floor"]))
    lab = Counter(r["label"] for r in sample)
    print("  by scorer label: %s" % ", ".join("%s %d" % (k, lab[k]) for k in HT4_ALL if lab[k]))
    mod = Counter((r["model_key_run"], stratum_lang(S, r)) for r in sample)
    print("  by model and answer language: %s" % ", ".join("%s %s %d" % (m, l, n) for (m, l), n in
                                                          sorted(mod.items())))
    extra = len(rows) - len(sample)
    print("Pools shown beside the sample (D274): %s; %d of their answers are also in the sample"
          % (", ".join("%s %d" % (k, len(v)) for k, v in pick.items()),
             sum(1 for k in pool_of if k in sid)))
    print("Rows per checker sheet: %d (%d sample + %d beside)" % (len(rows), len(sample), extra))
    print("Wrote %s, %s, %s (rules), %s (key: never sent), %s"
          % (P["sheet"]["C1"], P["sheet"]["C2"], P["rules"], P["key_csv"], P["draw"]))
    ctx.log_run({"command": "ht4-draw", "n_sample": len(sample), "n_rows": len(rows),
                 "pools": manifest["pools_shown"], "key_sha256": manifest["key_sha256"]})
    print("HT-4 DRAW: DONE")
    return 0


# ---- receipt and consensus -----------------------------------------------------

def receipts(P):
    out = []
    if P["receipts"].exists():
        for line in P["receipts"].read_text("utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def take_receipt(ctx, P, who, path, n_rows):
    s = sha256_file(path)
    old = [r for r in receipts(P) if r["who"] == who]
    if old and old[-1]["sha256"] != s:
        raise Stop("%s was received before with sha %s; it is now %s. A returned sheet never "
                   "changes (D275)" % (path.name, short(old[-1]["sha256"]), short(s)))
    if not old:
        rec = {"who": who, "file": path.name, "sha256": s, "rows": n_rows, "received_utc": utc_now(),
               "file_mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
               .strftime("%Y-%m-%dT%H:%M:%SZ")}
        with open(ctx.guard(P["receipts"]), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec, True
    return old[-1], False


def read_returned(ctx, P, who, key_rows):
    path = P["returned"][who]
    rows = read_workbook(path, "Sheet", ("code", "your_label", "refusal_yes_no"))
    col = "code_c1" if who == "C1" else "code_c2"
    by_code = {r[col]: r for r in key_rows}
    got, bad, blank, notes = {}, [], 0, 0
    for r in rows:
        c = r["code"]
        if c not in by_code or c in got:
            bad.append(c)
            continue
        lab, ref = r["your_label"].lower(), r["refusal_yes_no"].lower()
        kr = by_code[c]
        if not lab or not ref:
            blank += 1
        elif lab not in allowed_labels(kr["row_type"]) or ref not in ("yes", "no"):
            bad.append(c)
        got[c] = (lab, ref)
        notes += bool(r.get("note"))
    missing = sorted(set(by_code) - set(got))
    ok = not bad and not missing and blank == 0
    return ok, {kr["key"]: got.get(kr[col], ("", "")) for kr in key_rows}, \
        {"rows": len(rows), "bad": bad[:5], "n_bad": len(bad), "missing": len(missing),
         "blank": blank, "notes": notes, "path": path}


def cmd_ht4_receive(ctx, args):
    P = ht4_paths(ctx)
    print("Scoring and audit, part 1: HT-4 receipt (D275), %s, run %s" % (SCRIPT_VERSION, utc_now()))
    if not P["key_csv"].exists():
        raise Stop("no HT-4 draw yet")
    key_rows = read_csv(P["key_csv"])
    lab = {}
    all_ok = True
    for who in CHECKERS:
        path = P["returned"][who]
        if not path.exists():
            print("%s: %s not received yet" % (who, path.name))
            all_ok = False
            continue
        ok, labels, info = read_returned(ctx, P, who, key_rows)
        if not ok:
            print("%s: FAIL %s: %d rows, unknown or wrong codes or labels %d %s, rows missing %d, "
                  "blank cells %d. Send it back to the checker; nothing recorded."
                  % (who, path.name, info["rows"], info["n_bad"], info["bad"], info["missing"],
                     info["blank"]))
            all_ok = False
            continue
        rec, new = take_receipt(ctx, P, who, path, info["rows"])
        print("%s: PASS %s, %d rows labelled, %d notes; sha %s, received %s%s"
              % (who, path.name, info["rows"], info["notes"], short(rec["sha256"]),
                 rec["received_utc"], "" if new else " (receipt already on file)"))
        lab[who] = labels
    if not all_ok:
        print("RECEIPT: %d of 2" % len(lab))
        return 1
    print("RECEIPT: 2 of 2")
    differ = [kr for kr in key_rows if lab["C1"][kr["key"]] != lab["C2"][kr["key"]]]
    n_lab = sum(1 for kr in key_rows if lab["C1"][kr["key"]][0] != lab["C2"][kr["key"]][0])
    n_ref = sum(1 for kr in key_rows if lab["C1"][kr["key"]][1] != lab["C2"][kr["key"]][1])
    print("Checkers agree on %d of %d labels and %d of %d refusal answers"
          % (len(key_rows) - n_lab, len(key_rows), len(key_rows) - n_ref, len(key_rows)))
    if P["cons_sheet"].exists():
        print("The consensus sheet exists already: %s" % P["cons_sheet"])
        return 0
    if not differ:
        print("No row differs: no consensus meeting is needed. Run ht4-consensus.")
        return 0
    rules = cons_rules()
    srows, dd = [], {}
    # the content columns are rebuilt from the C1 sheet (the same text)
    c1 = {r["code"]: r for r in read_workbook(P["sheet"]["C1"], "Sheet", SHEET_COLS)}
    for i, kr in enumerate(sorted(differ, key=lambda kr: kr["code_c1"])):
        base = c1[kr["code_c1"]]
        a, b = lab["C1"][kr["key"]], lab["C2"][kr["key"]]
        d = {c: base.get(c, "") for c in ("row_type", "asked_in", "answer_requested_in", "question",
                                          "gold_answer", "reference", "model_answer")}
        d.update(code=kr["code_c1"], checker1_label=a[0], checker2_label=b[0],
                 checker1_refusal=a[1], checker2_refusal=b[1],
                 consensus_label=a[0] if a[0] == b[0] else "",
                 consensus_refusal=a[1] if a[1] == b[1] else "")
        srows.append(d)
        dd[(i, "consensus_label")] = allowed_labels(kr["row_type"])
        dd[(i, "consensus_refusal")] = ("yes", "no")
    write_workbook(ctx.guard(P["cons_sheet"]), [("Read me", None, [[x] for x in rules])],
                   "Consensus", CONS_COLS, srows, dd)
    print("Wrote %s (%d rows: labels differ %d, refusal answers differ %d)"
          % (P["cons_sheet"], len(differ), n_lab, n_ref))
    ctx.log_run({"command": "ht4-receive", "rows": len(key_rows), "differ": len(differ)})
    return 0


def cmd_ht4_consensus(ctx, args):
    P = ht4_paths(ctx)
    print("Scoring and audit, part 1: HT-4 consensus and the D238 file, %s, run %s"
          % (SCRIPT_VERSION, utc_now()))
    key_rows = read_csv(P["key_csv"])
    lab = {}
    for who in CHECKERS:
        if not [r for r in receipts(P) if r["who"] == who]:
            raise Stop("no receipt for %s; run ht4-receive first" % who)
        ok, labels, info = read_returned(ctx, P, who, key_rows)
        if not ok:
            raise Stop("%s no longer reads as a complete sheet" % P["returned"][who].name)
        take_receipt(ctx, P, who, P["returned"][who], info["rows"])   # sha unchanged or STOP
        lab[who] = labels
    final = {}
    differ = [kr for kr in key_rows if lab["C1"][kr["key"]] != lab["C2"][kr["key"]]]
    for kr in key_rows:
        if lab["C1"][kr["key"]] == lab["C2"][kr["key"]]:
            final[kr["key"]] = lab["C1"][kr["key"]]
    if differ:
        path = P["cons_returned"]
        if not path.exists():
            print("%s not received yet" % path.name)
            print("CONSENSUS: WAITING")
            return 1
        rows = read_workbook(path, "Consensus", ("code", "consensus_label", "consensus_refusal"))
        by = {r["code"]: r for r in rows}
        open_ = []
        for kr in differ:
            r = by.get(kr["code_c1"])
            labv = (r or {}).get("consensus_label", "").lower()
            refv = (r or {}).get("consensus_refusal", "").lower()
            if labv not in allowed_labels(kr["row_type"]) or refv not in ("yes", "no"):
                open_.append(kr["code_c1"])
            else:
                final[kr["key"]] = (labv, refv)
        if open_ or len(rows) != len(differ):
            print("CONSENSUS: FAIL: %d rows without a valid consensus (%s); %d rows in the file, %d "
                  "expected" % (len(open_), ", ".join(open_[:5]), len(rows), len(differ)))
            return 1
        rec, _new = take_receipt(ctx, P, "consensus", path, len(rows))
        print("Consensus sheet: %d rows settled, sha %s, received %s"
              % (len(rows), short(rec["sha256"]), rec["received_utc"]))
    sample = sorted([kr for kr in key_rows if kr["in_sample"] == "yes"], key=lambda kr: kr["sample_id"])
    out = [{"sample_id": kr["sample_id"], "model_key": kr["model_key"],
            "condition_id": kr["condition_id"], "qa_id": kr["qa_id"], "house_id": kr["house_id"],
            "scorer_label": kr["scorer_label"], "checker1_label": lab["C1"][kr["key"]][0],
            "checker2_label": lab["C2"][kr["key"]][0], "consensus_label": final[kr["key"]][0]}
           for kr in sample]
    issues = [{"pools": kr["pools"], "sample_id": kr["sample_id"], "model_key": kr["model_key"],
               "condition_id": kr["condition_id"], "qa_id": kr["qa_id"], "house_id": kr["house_id"],
               "row_type": kr["row_type"], "scorer_label": kr["scorer_label"],
               "scorer_refusal": kr["scorer_refusal"], "checker1_label": lab["C1"][kr["key"]][0],
               "checker2_label": lab["C2"][kr["key"]][0], "consensus_label": final[kr["key"]][0],
               "consensus_refusal": final[kr["key"]][1]}
              for kr in key_rows if kr["pools"]]
    refusal = [{"in_sample": kr["in_sample"], "model_key": kr["model_key"],
                "condition_id": kr["condition_id"], "qa_id": kr["qa_id"], "row_type": kr["row_type"],
                "scorer_refusal": kr["scorer_refusal"], "checker1_refusal": lab["C1"][kr["key"]][1],
                "checker2_refusal": lab["C2"][kr["key"]][1], "consensus_refusal": final[kr["key"]][1]}
               for kr in key_rows]
    for path, rows, cols in ((P["labels"], out, HT4_COLUMNS),
                             (P["issues"], issues, list(issues[0]) if issues else ["pools"]),
                             (P["refusal"], refusal, list(refusal[0]))):
        tmp = path.with_name(path.name + ".new")
        write_csv(ctx.guard(tmp), rows, cols)
        if path.exists():
            if sha256_file(path) != sha256_file(tmp):
                tmp.unlink()
                raise Stop("%s exists and differs; it is never replaced" % path.name)
            tmp.unlink()
        else:
            tmp.rename(path)
    print("Wrote %s (%d rows, D238), %s (%d rows), %s (%d rows); ids and labels only"
          % (P["labels"], len(out), P["issues"], len(issues), P["refusal"], len(refusal)))
    ctx.log_run({"command": "ht4-consensus", "labels_sha256": sha256_file(P["labels"])})
    print("CONSENSUS: DONE. The two checkers may now open Drive scored/ (D168 ends).")
    return 0


# ---- the reading -----------------------------------------------------------------

def cmd_ht4_read(ctx, args):
    hc = ctx.cfg["ht4"]
    P = ht4_paths(ctx)
    print("Scoring and audit, part 1: HT-4 reading (Design 10, D274, D276), %s, run %s"
          % (SCRIPT_VERSION, utc_now()))
    L = Lines()
    rows = read_csv(P["labels"])
    draw = json.loads(P["draw"].read_text("utf-8"))
    # own checks of the D238 file
    hdr_ok = rows and list(rows[0].keys()) == list(HT4_COLUMNS)
    bad_lab = sum(1 for r in rows for c in HT4_COLUMNS[5:] if r[c] not in HT4_ALL)
    ex = ctx.excluded()
    n_ex = sum(1 for r in rows if r["qa_id"] in ex)
    scorer = {}
    for m in HT4_MODELS:
        recs, _s = load_scored(ctx, m)
        for r in recs:
            scorer[(m, r["condition_id"], r["qa_id"])] = r.get("label")
    mism = sum(1 for r in rows if scorer.get((r["model_key"], r["condition_id"], r["qa_id"]))
               != r["scorer_label"])
    shas_now = {m: sha256_file(ctx.scored_path(m)) for m in MODELS}
    same_files = shas_now == draw["scored_sha256"]
    L.add("H01", hdr_ok and len(rows) == draw["n_sample"] and bad_lab == 0 and n_ex == 0 and mism == 0,
          "ht4_labels.csv %s: %d rows (D238 columns: %s); unknown labels %d; excluded items %d; "
          "scorer label differs from the scored files %d"
          % (short(sha256_file(P["labels"])), len(rows), hdr_ok, bad_lab, n_ex, mism))
    L.add("H02", same_files, "the six scored files are the ones the sheet was drawn from%s"
          % ("" if same_files else ": CHANGED"))
    ST = import_stats(ctx)
    if ctx.cfg.get("stats_world"):
        try:
            _ST, checks, world, recs = stats_reader(ctx, echo=False)
            rows_st = ST.read_ht4(str(P["labels"]), world, recs, checks)
            f = [r for r in checks.rows if r[0] == "FAIL"]
            L.add("H03", not f and len(rows_st) == len(rows),
                  "stats.py reader (build_world, load_scored, read_ht4) as the Statistics chat runs "
                  "it: %d check lines, %d FAIL%s" % (len(checks.rows), len(f),
                                                     "" if not f else ": " + "; ".join(
                                                         "%s %s" % (x[1], x[2]) for x in f[:3])))
        except Exception as exc:
            L.add("H03", False, "stats.py reader: %s: %s" % (type(exc).__name__, exc))
    else:
        L.info("H03", "stats.py reader skipped (selftest)")
    res = ST.ht4_stats(rows, n_boot=hc["n_boot"])
    acc = res["accuracy"]
    k = sum(1 for r in rows if r["scorer_label"] == r["consensus_label"])
    pass_overall = acc["point"] is not None and acc["point"] >= hc["pass_overall"]
    ci = "-" if acc["lo"] is None else "%s to %s" % (pct(acc["lo"] * 1000, 1000),
                                                     pct(acc["hi"] * 1000, 1000))
    L.add("H04", pass_overall, "scorer = consensus on %d of %d (%s, house bootstrap 95%% CI %s); "
          "pass at %.0f%%" % (k, len(rows), pct(k, len(rows)), ci, 100 * hc["pass_overall"]))
    judged, small, fails = [], [], []
    for lab in HT4_ALL:
        v = res["accuracy_by_consensus_label"].get(lab)
        if not v:
            continue
        n = v["n"]
        kk = sum(1 for r in rows if r["consensus_label"] == lab and r["scorer_label"] == lab)
        txt = "%s %d of %d (%s)" % (lab, kk, n, pct(kk, n))
        if n >= hc["judge_floor"]:
            judged.append(txt)
            if v["point"] < hc["pass_label"]:
                fails.append(lab)
        else:
            small.append(txt)
    L.add("H05", not fails, "per consensus label, judged where n >= %d (D276), pass at %.0f%%: %s%s"
          % (hc["judge_floor"], 100 * hc["pass_label"], "; ".join(judged) or "none",
             "" if not fails else "; BELOW: " + ", ".join(fails)))
    L.info("H06", "consensus labels with n < %d, reported and not judged: %s"
           % (hc["judge_floor"], "; ".join(small) or "none"))
    per_s = []
    for lab in HT4_ALL:
        n = sum(1 for r in rows if r["scorer_label"] == lab)
        if n:
            kk = sum(1 for r in rows if r["scorer_label"] == lab and r["consensus_label"] == lab)
            per_s.append("%s %d of %d" % (lab, kk, n))
    L.info("H07", "per scorer label (information): %s" % "; ".join(per_s))
    kap = []
    for name in ("scorer_vs_checker1", "scorer_vs_checker2", "checker1_vs_checker2",
                 "scorer_vs_consensus"):
        v = res[name]
        kci = v["kappa_ci"]
        kap.append("%s agreement %s, kappa %s (%s to %s)" % (
            name.replace("_", " "), pct(v["agreement"] * 1000, 1000),
            "-" if v["kappa"] is None else "%.3f" % v["kappa"],
            "-" if kci["lo"] is None else "%.3f" % kci["lo"],
            "-" if kci["hi"] is None else "%.3f" % kci["hi"]))
    L.info("H08", "; ".join(kap))
    n_ec = sum(1 for r in rows if r["scorer_label"] == "english_copy")
    k_ec = sum(1 for r in rows if r["scorer_label"] == "english_copy"
               and r["consensus_label"] == "english_copy")
    if n_ec >= hc["judge_floor"]:
        r4 = "keep the label" if k_ec / n_ec >= hc["r4"] else "DROP the label (R4); H4 runs on tiers"
    else:
        r4 = "not judged (n < %d)" % hc["judge_floor"]
    L.info("H09", "R4, English copy: %d of %d scorer english_copy labels right%s: %s"
           % (k_ec, n_ec, "" if not n_ec else " (%s)" % pct(k_ec, n_ec), r4))
    issues = read_csv(P["issues"]) if P["issues"].exists() else []
    pool_fail = []
    for tag, text in (("pool1", "correct answers holding a refusal phrase"),
                      ("pool2", "forced choice naming two or more labels")):
        rs = [r for r in issues if tag in r["pools"].split("|")]
        d = sum(1 for r in rs if r["consensus_label"] != r["scorer_label"])
        if len(rs) >= hc["judge_floor"]:
            ok = d <= hc["fail_share"] * len(rs)
            if not ok:
                pool_fail.append(tag)
            L.add("H10" if tag == "pool1" else "H11", ok, "%s (%s): consensus differs from the scorer "
                  "on %d of %d; fails above %.0f%% (D274)" % (tag, text, d, len(rs), 100 * hc["fail_share"]))
        else:
            L.info("H10" if tag == "pool1" else "H11", "%s (%s): %d of %d differ; not judged (n < %d)"
                   % (tag, text, d, len(rs), hc["judge_floor"]))
    rs3 = [r for r in issues if "pool3" in r["pools"].split("|")]
    p118 = [r for r in issues if "pool1" in r["pools"].split("|") and r["row_type"] == "count"]
    L.info("H12", "pool3 (possible unlisted refusals, counted only): %d answers, the checkers call %d "
           "a refusal; P118 type (count answers in pool 1): %d, consensus correct %d"
           % (len(rs3), sum(1 for r in rs3 if r["consensus_refusal"] == "yes"), len(p118),
              sum(1 for r in p118 if r["consensus_label"] == "correct")))
    ref = read_csv(P["refusal"]) if P["refusal"].exists() else []
    rs = [r for r in ref if r["in_sample"] == "yes"]
    agree = sum(1 for r in rs if r["scorer_refusal"] == r["consensus_refusal"])
    L.info("H13", "refusal flag (information, D277): scorer = consensus on %d of %d sample answers "
           "(scorer yes %d, consensus yes %d)" % (agree, len(rs),
                                                   sum(1 for r in rs if r["scorer_refusal"] == "yes"),
                                                   sum(1 for r in rs if r["consensus_refusal"] == "yes")))
    passed = not L.failed()
    reading = {"provenance": ctx.provenance(), "pass": passed, "lines": L.rows,
               "ht4_stats": res, "labels_sha256": sha256_file(P["labels"]),
               "issues_sha256": sha256_file(P["issues"]) if P["issues"].exists() else None,
               "scored_sha256": shas_now, "accuracy_k": k, "n": len(rows), "r4": r4}
    write_json(ctx.guard(P["reading"]), reading)
    print("Wrote %s" % P["reading"])
    ctx.log_run({"command": "ht4-read", "pass": passed, "k": k, "n": len(rows)})
    print("HT-4: %s" % ("PASS" if passed else
                        "FAIL. Nothing is tagged. Design 10: fix the scorer logic (never the frozen "
                        "files), run all tests, re-score, check a fresh 100. Send this printout."))
    return 0 if passed else 1


# ---- Gate F ------------------------------------------------------------------------

GATE_F_PATHS = ("src/scoring.py", "src/normalise.py", "src/fc_options.py", "scripts/asset9_scorer.py",
                "lexicon/lexicon.csv", "lexicon/closed_vocab.csv", "probes/excluded_items.csv",
                "configs/conditions.yaml")


def run_tests(ctx):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", BIVQA_PROBES=str(ctx.root / "probes"))
    p = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                       cwd=str(ctx.main), env=env, capture_output=True, text=True)
    out = p.stdout + p.stderr
    log = ctx.guard(ctx.records_dir / "logs" / ("gate_f_tests_%s.log" % stamp()))
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(out, encoding="utf-8")
    m = re.search(r"Ran (\d+) tests?", out)
    ran = int(m.group(1)) if m else 0
    last = [x for x in out.strip().splitlines() if x.strip()][-1:] or [""]
    sk = re.search(r"skipped=(\d+)", last[0])
    return p.returncode, ran, int(sk.group(1)) if sk else 0, last[0], log


def cmd_gate_f(ctx, args):
    hc = ctx.cfg["ht4"]
    P = ht4_paths(ctx)
    print("Scoring and audit, part 1: Gate F (Design 8.7 and 11, D278), %s, run %s"
          % (SCRIPT_VERSION, utc_now()))
    L = Lines()
    rd = json.loads(P["reading"].read_text("utf-8")) if P["reading"].exists() else {}
    L.add("F01", rd.get("pass") is True, "HT-4 reading: %s" % (
        "PASS, %d of %d" % (rd.get("accuracy_k"), rd.get("n")) if rd.get("pass") else
        "not passed or not run"))
    now = {m: sha256_file(ctx.scored_path(m)) for m in MODELS}
    same = rd and now == rd.get("scored_sha256") and P["labels"].exists() \
        and sha256_file(P["labels"]) == rd.get("labels_sha256")
    L.add("F02", bool(same), "scored files and ht4_labels.csv unchanged since the reading")
    if ctx.cfg.get("git_checks"):
        clean, dirt = tree_clean(ctx.main)
        head = git(ctx.main, "rev-parse", "HEAD")[1]
        origin = git(ctx.main, "rev-parse", "refs/remotes/origin/main")[1]
        L.add("F03", clean and head == origin, "main clone clean and equal to origin/main: HEAD %s, "
              "origin/main %s%s" % (head[:12], origin[:12], "" if clean else "; NOT clean: " + dirt[:200]))
        ch = git(ctx.main, "diff", "--name-only", "gate-E", "HEAD", "--", *GATE_F_PATHS)
        L.add("F04", ch[0] == 0 and not ch[1].strip(), "scorer, normalise, fc_options, asset9_scorer.py, "
              "lexicon, excluded list and conditions.yaml equal to gate-E%s"
              % ("" if not ch[1].strip() else "; CHANGED: " + ch[1].replace("\n", ", ")))
        me = Path(__file__).resolve()
        rel = me.relative_to(ctx.main.resolve()).as_posix() if ctx.main.resolve() in me.parents else None
        committed = rel and git(ctx.main, "ls-files", "--error-unmatch", rel)[0] == 0 and \
            git(ctx.main, "diff", "--quiet", "HEAD", "--", rel)[0] == 0
        L.add("F05", bool(committed), "this script committed on main, no local change")
        exists = git(ctx.main, "rev-parse", "-q", "--verify", "refs/tags/gate-F")[0] == 0
        L.add("F06", not exists, "tag gate-F not there yet (local tags were fetched from GitHub in Cell 0)")
        rc, ran, skipped, last, log = run_tests(ctx)
        L.add("F07", rc == 0 and ran >= hc["tests_min"] and skipped == 0,
              "unit tests (python -m unittest discover -s tests, BIVQA_PROBES=Drive probes/): %d run, "
              "result '%s'; need %d or more, 0 failures, 0 skipped (log %s)"
              % (ran, last, hc["tests_min"], log))
        clean2, dirt2 = tree_clean(ctx.main)
        L.add("F08", clean2, "main clone still clean after the tests%s" % ("" if clean2 else ": " + dirt2[:200]))
    else:
        L.info("F03", "Git checks and tests skipped (selftest)")
    if L.failed():
        print("GATE F: FAIL. Nothing tagged. Send this printout.")
        return 1
    if args.check_only or not ctx.cfg.get("git_checks"):
        print("GATE F: ALL CHECKS PASS (check only; nothing tagged)")
        return 0
    if not args.tagger_name or not args.tagger_email:
        raise Stop("give --tagger-name and --tagger-email (Cell P)")
    head = git(ctx.main, "rev-parse", "HEAD")[1]
    msg = "\n".join([
        "Gate F: scorer frozen after HT-4 passed (Design 8.7 and 11).",
        "scoring.py %s sha256 %s" % (ctx.S.SCORER_VERSION, sha256_file(ctx.main / "src" / "scoring.py")),
        "normalise.py sha256 %s" % sha256_file(ctx.main / "src" / "normalise.py"),
        "asset9_scorer.py sha256 %s" % sha256_file(ctx.main / "scripts" / "asset9_scorer.py"),
        "HT-4: scorer = consensus on %d of %d; ht4_labels.csv sha256 %s"
        % (rd["accuracy_k"], rd["n"], rd["labels_sha256"]),
        "scored files: " + ", ".join("%s %s" % (m, now[m][:12]) for m in MODELS)])
    rc, _o, err = git(ctx.main, "-c", "user.name=%s" % args.tagger_name,
                      "-c", "user.email=%s" % args.tagger_email, "tag", "-a", "gate-F", "-m", msg, head)
    kind = git(ctx.main, "cat-file", "-t", "gate-F")[1]
    target = git(ctx.main, "rev-parse", "gate-F^{commit}")[1]
    ok = rc == 0 and kind == "tag" and target == head
    print("%s tag gate-F: %s, on %s%s" % ("PASS" if ok else "FAIL", kind or "missing", target[:12],
                                         "" if rc == 0 else "; " + err[:200]))
    ctx.log_run({"command": "gate-f", "tagged": ok, "commit": head})
    print("GATE F: %s" % ("TAG CREATED on %s; push it with the next cell" % head[:12] if ok
                          else "FAIL"))
    return 0 if ok else 1


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
                "close-up", "qa_type": t, "question_bn": "প্রশ্ন %s" % t, "answer_bn": gold,
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
    (root / "english").mkdir(parents=True)
    en_rows = [{"string_id": "s%02d" % i, "question_bn_norm": S.normalise(q),
                "question_en": "made-up English question %d" % i}
               for i, q in enumerate(sorted({it[k]["question_bn"] for k in it}))]
    write_csv(root / "english" / "english_questions_final.csv", en_rows,
              ["string_id", "question_bn_norm", "question_en"])
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
        "excluded_path": exl, "git_checks": False, "stats_world": False,
        "ht4": dict(REAL["ht4"], n=40, floor=2, n_boot=200, pool_expect=None),
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


def selftest_stage2(ctx, check):
    """HT-4 on made-up answers: draw, two filled sheets, receipt, consensus, the
    D238 file, the reading with stats.ht4_stats, the receipt lock, Gate F checks."""
    from openpyxl import load_workbook
    print("---- stage 2: HT-4 ----")
    ns = argparse.Namespace(check_only=True, tagger_name=None, tagger_email=None)
    P = ht4_paths(ctx)
    check("ht4-draw: runs", cmd_ht4_draw(ctx, ns) == 0)
    key = read_csv(P["key_csv"])
    n_s = sum(1 for k in key if k["in_sample"] == "yes")
    labs = Counter(k["scorer_label"] for k in key if k["in_sample"] == "yes")
    check("ht4-draw: %d sample rows, at least 2 per scorer label present" % ctx.cfg["ht4"]["n"],
          n_s == ctx.cfg["ht4"]["n"] and min(labs.values()) >= 2, str(dict(labs)))
    pools = Counter(t for k in key for t in k["pools"].split("|") if t)
    check("ht4-draw: pools 1 to 3 present", all(pools[t] for t in ("pool1", "pool2", "pool3")),
          str(dict(pools)))
    check("ht4-draw: run again, nothing replaced", cmd_ht4_draw(ctx, ns) == 0)
    wb = load_workbook(str(P["sheet"]["C1"]))
    hdr = [c.value for c in wb["Sheet"][1]]
    text = " ".join(str(c.value) for row in wb["Sheet"].iter_rows() for c in row if c.value)
    check("sheet: no scorer label column and no qa_id on the sheet",
          "scorer_label" not in hdr and not any(k["qa_id"] in text for k in key))
    check("sheet: tabs", wb.sheetnames == ["Sheet", "Read me", "Lexicon", "Vocabulary",
                                           "Refusal phrases", "Endings"], str(wb.sheetnames))
    # the two checkers (made up): C1 gives the scorer's labels, C2 differs on 3 rows
    by1 = {k["code_c1"]: k for k in key}
    by2 = {k["code_c2"]: k for k in key}
    changed = []
    for who, by in (("C1", by1), ("C2", by2)):
        wb = load_workbook(str(P["sheet"][who]))
        ws = wb["Sheet"]
        cols = [c.value for c in ws[1]]
        for row in ws.iter_rows(min_row=2):
            k = by[row[cols.index("code")].value]
            lab, ref = k["scorer_label"], k["scorer_refusal"]
            if who == "C2" and len(changed) < 3 and k["in_sample"] == "yes":
                alt = [x for x in allowed_labels(k["row_type"]) if x != lab]
                lab = alt[0]
                changed.append(k["key"])
            row[cols.index("your_label")].value = lab
            row[cols.index("refusal_yes_no")].value = ref
        wb.save(str(P["returned"][who]))
    check("ht4-receive: both sheets, consensus sheet written", cmd_ht4_receive(ctx, ns) == 0
          and P["cons_sheet"].exists())
    wb = load_workbook(str(P["cons_sheet"]))
    ws = wb["Consensus"]
    cols = [c.value for c in ws[1]]
    n_rows = 0
    for i, row in enumerate(ws.iter_rows(min_row=2)):
        n_rows += 1
        a = row[cols.index("checker1_label")].value
        b = row[cols.index("checker2_label")].value
        row[cols.index("consensus_label")].value = a if i else b     # one row keeps C2's label
    wb.save(str(P["cons_returned"]))
    check("consensus sheet: the 3 differing rows only", n_rows == 3, str(n_rows))
    check("ht4-consensus: D238 file written", cmd_ht4_consensus(ctx, ns) == 0)
    rows = read_csv(P["labels"])
    check("ht4_labels.csv: D238 columns, one disagreement with the scorer",
          list(rows[0]) == list(HT4_COLUMNS)
          and sum(1 for r in rows if r["scorer_label"] != r["consensus_label"]) == 1)
    check("ht4-read: pass (39 of 40)", cmd_ht4_read(ctx, ns) == 0)
    rd = json.loads(P["reading"].read_text("utf-8"))
    check("ht4-read: stats.ht4_stats accuracy 0.975", abs(rd["ht4_stats"]["accuracy"]["point"] - 0.975)
          < 1e-9, str(rd["ht4_stats"]["accuracy"]["point"]))
    check("gate-f: all checks pass (check only)", cmd_gate_f(ctx, ns) == 0)
    wb = load_workbook(str(P["returned"]["C1"]))
    wb["Sheet"].cell(row=2, column=SHEET_COLS.index("note") + 1).value = "changed later"
    wb.save(str(P["returned"]["C1"]))
    try:
        cmd_ht4_receive(ctx, ns)
        check("a returned sheet that changed after its receipt stops the run", False)
    except Stop:
        check("a returned sheet that changed after its receipt stops the run", True)
    with open(ctx.scored_path("gemma"), "a", encoding="utf-8") as fh:
        fh.write("\n")
    check("gate-f: a scored file changed after the reading fails", cmd_gate_f(ctx, ns) == 1)


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
        selftest_stage2(ctx, check)
        print("---- guard ----")
        try:
            ctx.guard(gate_e / "x.txt")
            check("no write inside the gate-E checkout", False)
        except Stop:
            check("no write inside the gate-E checkout", True)
        try:
            argparse_ns = argparse.Namespace(model=["gemma"], overwrite=True)
            (root / "human" / "HT-4" / "to_send").mkdir(parents=True, exist_ok=True)
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
    ap = argparse.ArgumentParser(description="Scoring and audit, part 1 (D270 to D278)")
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
    sub.add_parser("loadtest", help="stats.py reader on the scored files (Cell 8)")
    sub.add_parser("ht4-draw", help="draw the HT-4 sheets (Cell 9)")
    sub.add_parser("ht4-receive", help="receipts and the consensus sheet (Cell 10)")
    sub.add_parser("ht4-consensus", help="consensus and ht4_labels.csv (Cell 11)")
    sub.add_parser("ht4-read", help="the HT-4 reading (Cell 12)")
    gf = sub.add_parser("gate-f", help="checks, tests and the tag gate-F (Cell 13)")
    gf.add_argument("--check-only", action="store_true", help="all checks and tests, no tag")
    gf.add_argument("--tagger-name")
    gf.add_argument("--tagger-email")
    args = ap.parse_args(argv)
    try:
        if args.command == "selftest":
            return cmd_selftest(None, args)
        ctx = Ctx(args.root, args.gate_e, args.main, real_cfg())
        return {"check": cmd_check, "score": cmd_score, "verify": cmd_verify,
                "counts": cmd_counts, "d159": cmd_d159, "sample10": cmd_sample10,
                "status": cmd_status, "loadtest": cmd_loadtest, "ht4-draw": cmd_ht4_draw,
                "ht4-receive": cmd_ht4_receive, "ht4-consensus": cmd_ht4_consensus,
                "ht4-read": cmd_ht4_read, "gate-f": cmd_gate_f}[args.command](ctx, args)
    except Stop as exc:
        print("STOP: %s" % exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
