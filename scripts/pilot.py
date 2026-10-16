#!/usr/bin/env python3
"""
pilot.py
BanglaIndoorVQA, the Pilot (Design 11, 7.4, 5.4; DESIGN_ADDENDUM D9, D10, D15 and E;
decisions D155 to D163, DESIGN_ADDENDUM F) and the Pilot reading (DESIGN_ADDENDUM B10, F5,
H5; decisions D164 to D168, DESIGN_ADDENDUM F7 on).

The pilot runs the 100 PILOT items under the EXP-L and EXP-N conditions for the three
models, through the ASSET-8 harness (scripts/asset8_harness.py, src/harness.py). This
script adds no inference code and no scoring code. It
  - checks every input before the first run (Cell 0 of notebooks/pilot.ipynb),
  - runs harness commands, with the items of each condition taken from the fields of
    PILOT.json (D156), never from qa_id lists,
  - runs every finished command a second time and checks that the second run makes 0
    calls and changes neither the cache nor the cost ledger (the first half of Gate C),
  - counts what came out: answers, empty answers, answers of 8 words or fewer, seconds
    per item of the open models, tokens per Gemini call, projections against Design 12,
  - for the Pilot reading: checks the forced-choice prompts against the option file, starts
    scripts/asset9_scorer.py score once per cache, and reads Gates C and D from the scored
    records (D163, D190, D166).
It never prints a model answer. Question text is printed only by 'preview'. Files with
answer text go to Google Drive only (scored/pilot/).

Commands (run from the repo root, in Colab)
    check     one PASS or FAIL line per input; INFO lines are notes
    preview   the Bangla and English questions of the 3 items that the first English
              calls use, and one full prompt per condition (question text, no answer)
    run       one harness command for one model and one item group, then the same
              command again (Gate C)
    report    counts, seconds per item, tokens per call, projections (counts only)
    compare   D5 only: how often two runs of the same requests gave the same answer text
    exclusions  D160 and D161: list the probe items to exclude (the gold slip H68_0023_q2 and
              every question with the word for 'marked'); with --write, append them to
              probes/excluded_items.csv (never changes or removes a row)
    fc-verify D164: every forced-choice item is in the cache once, and every forced-choice
              prompt shows the scorer's four options in the order of the option file; with
              --dry-run on the dry-run caches (Cell 8a), before any billed forced-choice call
    score     D165: scripts/asset9_scorer.py score, once per cache (the three pilot caches, the
              D5 repeat, the Batch test of 2026-10-05), into Drive scored/pilot/
    read      the Pilot reading: Gate C (both halves), Gate D, D5 agreement, refusal rates,
              correct answers that hold a refusal phrase, forced choice, the 10-item sample

Item groups (D156)
    english3     L-EB, L-EE, L-EE0 on 3 closed items by seeded rank: the first English calls
    exp-l        L-BB, L-BE, L-EB, L-EE, L-BB0, L-EE0 on the 60 items with pilot_part = closed
    exp-n-open   N-BB, N-BE, N-BBs on the 40 items with pilot_part = naming
    exp-n-fcb    N-FCB, N-FCB0 on the 20 naming items with in_pclust = true (after ASSET-11, HT-6)
    exp-n-fce    N-FCE, N-FCE0 on the naming items that are in P-FCE (after ASSET-11, HT-6)
    d5-closed, d5-naming   the D5 repeat (stage smoke, run name pilot-repeat); only with
                 --d5-approved, after its decision row is accepted

Everything this script writes goes to Google Drive: outputs/ (model output, D150) and
scored/pilot/ (scored records and the reading).
"""

import argparse
import csv
import difflib
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True     # no src/__pycache__ in the Git tree (stage main needs a clean tree)
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
try:
    import cost_ledger
    import harness as H
    import images as imglib
    import prompts as promptlib
    from normalise import normalise
except ImportError as _exc:  # pragma: no cover
    sys.exit(f"STOP: cannot import the files in src/ ({_exc}). Run this script from the repo, "
             "with normalise.py, cost_ledger.py, prompts.py, images.py and harness.py in src/.")

SCRIPT_VERSION = "pilot-2026-10-16a"
DEFAULT_ROOT = "/content/drive/MyDrive/BanglaIndoorVQA"
STAGE = "pilot"
MODELS = ("gemini", "qwen", "gemma")
OPEN_MODELS = ("qwen", "gemma")
CLOSED_TYPES = ("obj_name", "object", "count", "scene_id", "obj_property")

# The values this chat started from (PROJECT_STATE, DEPENDENCY_MAP, DESIGN_ADDENDUM D, E, G).
EXPECTED = {
    "harness_version": "2026-10-07a",
    "prompt_version": "exp-v1",
    "prompt_sha256": "e0d006e7cd0e2559d3e80cd0cda24b9afa7ac2661c579916811a9cc765478b73",
    "pilot_sha256": "4d9e4cc0eaa839bfaec59c881ce520c95779743958a0766160655fa9fbc2b587",
    "pilot_n": 100,
    "pilot_houses": 49,
    "english_sha256": "fbeb4c3233c0446cc613f5a89f2da39025962aff783f33ee309dca5a45354c07",
    "english_column": "question_en",
    "english_rows": 32,
    "excluded_sha256": "799be4cfee0e2c2e356798acc767b16e57f811108eaa485dd02f32c7b22d37c8",
    "excluded_rows": 14,
    "revision": {"qwen": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                 "gemma": "ee0ef6023621cff504d758262d4e04895a5af4a2"},
    "qwen_settings_sha_prefix": "6ca7d8af597c",
    "gemini_version_before": "3.6-flash-07-2026",
    "ledger_before_usd": 0.1496,
    "transformers": "5.18.0",
    "torch_before": "2.11.0+cu130",
    # The Pilot reading (PROJECT_STATE and DESIGN_ADDENDUM B9, H2 of 2026-10-16)
    "excluded_now_sha256": "8c51763fad6e3c48f0ae1e137573d00e1d0e79274bf959d9bb8e0bca96b37cdb",
    "excluded_now_rows": 28,
    "fc_options_sha256": "5288ba63033083b10beab8eb5949f3f6a2ab3968a4baf8eb8bb2e1b0fdcf2e96",
    "fc_sets": {"bn": 492, "en": 454},
    "pfce_sha256": "807e987b4b656586ee2955efb421e8e6cd0aae27790a11a4c4f3f4f7210479ae",
    "pfce_n": 454,
    "pclust_n": 492,
    "pclust_sha_prefix": "7759dcce19e4",
    "scorer_version": "0.2.1-stage2",
    "scoring_sha256": "72781bfd879cd821d78ee8733ec891a9bc7061a4d8f123a34fb9af462e236ed2",
    "asset9_scorer_sha256": "c16f6c9485894d6a442777bdbd6ab6edd234474f6534f8df511f27f662752b50",
    "refusal_phrases": 57,
}

# The Gemini model ID and prices were read on Google's pages in the Pilot chat (Design 0.2 rule 5).
GOOGLE_CHECK = {
    "checked_utc": "2026-10-14",
    "pages": ("https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash",
              "https://ai.google.dev/gemini-api/docs/pricing",
              "https://ai.google.dev/gemini-api/docs/deprecations"),
    "batch_2027": {"in": 0.75, "out": 3.75},   # USD per 1M tokens from 2027-01-01 (pricing page)
}
MAX_DAYS_SINCE_CHECK = 7

# The 9 conditions of this step: template, image, question language, answer language, items.
SCOPE = {
    "L-BB":  ("open_bn_bn", True, "bn", "bn", "closed"),
    "L-BE":  ("open_bn_en", True, "bn", "en", "closed"),
    "L-EB":  ("open_en_bn", True, "en", "bn", "closed"),
    "L-EE":  ("open_en_en", True, "en", "en", "closed"),
    "L-BB0": ("open_bn_bn", False, "bn", "bn", "closed"),
    "L-EE0": ("open_en_en", False, "en", "en", "closed"),
    "N-BB":  ("open_bn_bn", True, "bn", "bn", "naming"),
    "N-BE":  ("open_bn_en", True, "bn", "en", "naming"),
    "N-BBs": ("open_bn_bn_specific", True, "bn", "bn", "naming"),
}
# Later in the Pilot chat, after ASSET-11 and HT-6.
LATER = {
    "N-FCB":  ("fc_bn", True, "bn", "bn", "pclust"),
    "N-FCB0": ("fc_bn", False, "bn", "bn", "pclust"),
    "N-FCE":  ("fc_en", True, "en", "en", "pfce"),
    "N-FCE0": ("fc_en", False, "en", "en", "pfce"),
}
ALL_CONDS = dict(SCOPE, **LATER)

GROUPS = {
    "english3":   {"conditions": "L-EB,L-EE,L-EE0", "where": [("pilot_part", "closed")], "sample": 3,
                   "note": "the first English-question calls: 3 closed items by seeded rank"},
    "exp-l":      {"conditions": "EXP-L", "where": [("pilot_part", "closed")],
                   "note": "EXP-L on the 60 items with pilot_part = closed"},
    "exp-n-open": {"conditions": "EXP-N-open", "where": [("pilot_part", "naming")],
                   "note": "EXP-N open naming on the 40 items with pilot_part = naming"},
    "exp-n-fcb":  {"conditions": "EXP-N-fcb", "where": [("pilot_part", "naming"), ("in_pclust", "True")],
                   "needs": ("options",),
                   "note": "N-FCB, N-FCB0 on the 20 naming items with in_pclust = true"},
    "exp-n-fce":  {"conditions": "EXP-N-fce", "where": [("pilot_part", "naming")],
                   "needs": ("options", "pfce"),
                   "note": "N-FCE, N-FCE0 on the naming items that are in P-FCE"},
    "d5-closed":  {"conditions": "L-BB", "where": [("pilot_part", "closed")], "stage": "smoke",
                   "run_name": "pilot-repeat", "gemini_only": True, "needs": ("d5",),
                   "note": "D5 repeat: the L-BB requests of the 60 closed items again"},
    "d5-naming":  {"conditions": "N-BB", "where": [("pilot_part", "naming")], "stage": "smoke",
                   "run_name": "pilot-repeat", "gemini_only": True, "needs": ("d5",),
                   "note": "D5 repeat: the N-BB requests of the 40 naming items again"},
}
SCOPE_GROUPS = ("english3", "exp-l", "exp-n-open")
FC_GROUPS = ("exp-n-fcb", "exp-n-fce")      # the Pilot reading, Cell 8 (D164)

# Excluded items (D110; DESIGN_ADDENDUM C4). Reason codes, the scored probe sets in the order
# of the 'probes' column, and the two exclusions decided in the Pilot chat.
REASONS = ("gold_not_one_object", "gold_wrong_kind", "gold_wrong", "other")
SCORED_PROBES = ("P-CORE-CLOSED", "R-TEST", "P-NAME", "P-CLUST")
EXCLUSION_STEP = {
    "added_at": "Pilot",
    # D160: the gold is a typing slip (the team lead checked the image)
    "gold": (("H68_0023_q2", "gold_wrong", "Pilot image check"),),
    # D161: the word for 'marked' points at the bounding box of the annotation tool, which the
    # models never see. Written with escapes so that no editor can change its code points.
    "word": ("চিহ্নিত", "other", "Pilot: question refers to the annotation mark"),
    "root": "চিহ্ন",     # other forms of the word are shown, never added
    "known_hit": "H13_0005_q2",                    # holds the word; the scan must find it
}

# Calls per condition and model in the full runs (DESIGN_ADDENDUM A1; Design 7.1, 7.2).
PFCE_N = 454   # P-FCE run size from ASSET-11 (DESIGN_ADDENDUM H1); replaces the Design 7.2 planning value 300
MAIN_N = {"L-BB": 1118, "L-BE": 1118, "L-EB": 1118, "L-EE": 1118, "L-BB0": 1118, "L-EE0": 1118,
          "N-BB": 1000, "N-BE": 1000, "N-BBs": 1000, "N-FCB": 492, "N-FCB0": 492,
          "N-FCE": PFCE_N, "N-FCE0": PFCE_N}
# EXP-R evaluation (G2): 16,650 generations = R-BB and R-EE with the image, R-BB0 without,
# 4 model versions, less the base answers reused from EXP-L.
EXP_R_EVAL = {"image": 11100, "noimage": 5550, "loads": 4}
DESIGN_12 = {
    "gpu_h_per_model": {"EXP-L": 2.0, "EXP-N": 1.5},   # Design 7.4
    "gpu_h_pilot_both": 1.0,                             # Design 7.4 and 12.2
    "gpu_h_exp_r_eval": 5.5,                             # D182
    "image_call_in": 1000, "text_call_in": 250, "call_out_max": 100,   # Design 12.1 planning sizes
    "main_usd_planning": 11.0,                           # Design 12.1: 9.30 + 1.70 at planning prices
    "pilot_calls": 560, "pilot_usd_planning": 1.30,      # Design 12.1
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
class Fail(Exception):
    """A step that must stop. The message says why and what to do."""


def say(msg=""):
    print(msg, flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(text):
    try:
        return datetime.strptime(str(text), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def run_start(run_id):
    """The start time of a harness run, from its run_id (YYYYmmddTHHMMSSZ, maybe '-dry')."""
    if not re.match(r"^\d{8}T\d{6}Z", str(run_id)):
        return None
    return datetime.strptime(str(run_id)[:16], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def load_yaml(path):
    import yaml
    with open(str(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def raw_lines(path):
    """Non-empty lines of a file (a stricter count than unique keys)."""
    if not os.path.exists(str(path)):
        return 0
    with open(str(path), "rb") as f:
        return sum(1 for line in f if line.strip())


def ledger_total(path):
    return float(cost_ledger.read_total(str(path))) if os.path.exists(str(path)) else 0.0


def ledger_count(path):
    return len(list(cost_ledger.read_lines(str(path)))) if os.path.exists(str(path)) else 0


def stage_spend(path, label):
    return float(H.stage_spend(path, label)) if os.path.exists(str(path)) else 0.0


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))
    return xs[i]


def share(k, n):
    return f"{100.0 * k / n:5.1f}%" if n else "    -"


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


class Paths:
    def __init__(self, root, repo, english="", design=""):
        self.root = Path(root)
        self.repo = Path(repo)
        self.probe = self.root / "probes" / "PILOT.json"
        self.english = Path(english) if english else self.root / "english" / "english_questions_final_PILOT.csv"
        self.design = Path(design) if design else self.root / "docs" / "BanglaIndoorVQA_Design.md"
        self.excluded = self.repo / "probes" / "excluded_items.csv"
        self.pilot_ids = self.repo / "probes" / "PILOT.ids.txt"
        self.images = self.root / "data" / "images_1024"
        self.ledger = self.root / "outputs" / "api_cost_ledger.jsonl"
        self.pilot_out = self.root / "outputs" / "pilot"
        self.dry_out = self.root / "outputs" / "dryrun" / "pilot"


def load_configs(paths):
    P = promptlib.load_prompts(paths.repo / "configs" / "prompts.yaml")
    cfg = promptlib.load_conditions(paths.repo / "configs" / "conditions.yaml", P)
    models = H.load_models(paths.repo / "configs" / "models.yaml")
    return P, cfg, models


def split_items(items):
    """D156: the items of each condition come from the fields of PILOT.json."""
    closed = H.select_items(items, where=[("pilot_part", "closed")])
    naming = H.select_items(items, where=[("pilot_part", "naming")])
    pclust = H.select_items(items, where=[("pilot_part", "naming"), ("in_pclust", "True")])
    return closed, naming, pclust


def read_ids(path):
    with open(str(path), "r", encoding="utf-8-sig") as f:
        return [line.strip() for line in f
                if line.strip() and not line.startswith("#") and line.strip() != "qa_id"]


def read_excluded(path):
    with open(str(path), "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames or []
    return rows, cols


def git_lines(repo, *args):
    try:
        r = subprocess.run(["git", "-C", str(repo)] + list(args), capture_output=True, text=True, timeout=30)
        return [line for line in r.stdout.splitlines() if line.strip()]
    except Exception:
        return []


# --------------------------------------------------------------------------
# Characters: what a template may hold, and how to show a difference
# --------------------------------------------------------------------------
def allowed_char(ch):
    cp = ord(ch)
    return ch == "\n" or 32 <= cp < 127 or 0x0980 <= cp <= 0x09FF or cp in (0x0964, 0x0965)


def show_char(ch):
    return f"U+{ord(ch):04X} {unicodedata.name(ch, 'NO NAME')}"


def visible(text):
    """Text with line breaks and hidden characters written out as codes."""
    out = []
    for ch in text:
        if ch == "\n":
            out.append("\\n")
        elif not allowed_char(ch) or (unicodedata.category(ch)[0] in "CZ" and ch != " "):
            out.append(f"<U+{ord(ch):04X}>")
        else:
            out.append(ch)
    return "".join(out)


def compare_text(name, want, got):
    """Code point by code point. want = the design text, got = the prompts.yaml text.
    Returns the lines that describe each difference (an empty list when they are equal)."""
    if want == got:
        return []
    nfc_equal = unicodedata.normalize("NFC", want) == unicodedata.normalize("NFC", got)
    ops = [op for op in difflib.SequenceMatcher(a=want, b=got, autojunk=False).get_opcodes()
           if op[0] != "equal"]
    lines = [f"{name}: {len(ops)} difference(s); the two texts are "
             + ("EQUAL after Unicode NFC normalisation (the same letters written with other code points)"
                if nfc_equal else "NOT equal even after Unicode NFC normalisation")]
    for _tag, i1, i2, j1, j2 in ops[:12]:
        d = ", ".join(show_char(c) for c in want[i1:i2]) or "nothing"
        y = ", ".join(show_char(c) for c in got[j1:j2]) or "nothing"
        lines.append(f"  design position {i1}: design has [{d}], prompts.yaml has [{y}]")
        lines.append(f"    design:       ...{visible(want[max(0, i1 - 15):i2 + 15])}...")
        lines.append(f"    prompts.yaml: ...{visible(got[max(0, j1 - 15):j2 + 15])}...")
    if len(ops) > 12:
        lines.append(f"  ... and {len(ops) - 12} more")
    return lines


DESIGN_BLOCKS = (
    # label at the start of the bold line above the block, kind, prompts.yaml target
    ("L-BB and N-BB", "full", ("open_bn_bn", "with_image")),
    ("L-BE and N-BE", "full", ("open_bn_en", "with_image")),
    ("L-EE (", "full", ("open_en_en", "with_image")),
    ("L-EB (", "full", ("open_en_bn", "with_image")),
    ("No-image versions", "starts", (("open_bn_bn", "no_image"), ("open_en_en", "no_image"))),
    ("N-BBs", "sentence", None),
    ("N-FCB (", "full", ("fc_bn", "with_image")),
    ("N-FCE (", "full", ("fc_en", "with_image")),
)


def read_design_62(path):
    """The code blocks of Design section 6.2, each with the bold label above it."""
    raw = open(str(path), "rb").read()
    text = raw.decode("utf-8-sig")
    crlf = "\r\n" in text
    lines = text.replace("\r\n", "\n").split("\n")
    starts = [i for i, line in enumerate(lines) if line.startswith("### 6.2")]
    if len(starts) != 1:
        raise Fail(f"the design file has {len(starts)} headings that start with '### 6.2' (expected 1)")
    s = starts[0]
    e = next((i for i in range(s + 1, len(lines)) if lines[i].startswith(("## ", "### "))), len(lines))
    blocks, label, inside, buf = [], None, False, []
    for line in lines[s + 1:e]:
        if line.lstrip().startswith("```"):
            if inside:
                blocks.append((label or "", "\n".join(buf)))
                buf, inside = [], False
            else:
                inside = True
            continue
        if inside:
            buf.append(line)
        elif line.startswith("**"):
            m = re.match(r"\*\*(.+?)\*\*", line)
            label = m.group(1).strip() if m else line
    if inside:
        raise Fail("a code block in Design 6.2 is not closed")
    head = "\n".join(lines[:10])
    return {"sha256": hashlib.sha256(raw).hexdigest(), "crlf": crlf,
            "version_ok": "Version 2.1" in head, "blocks": blocks}


def differing_middle(a, b):
    """The part where two texts differ, after their common start and common end."""
    p, n = 0, min(len(a), len(b))
    while p < n and a[p] == b[p]:
        p += 1
    s = 0
    while s < n - p and a[len(a) - 1 - s] == b[len(b) - 1 - s]:
        s += 1
    return p, a[p:len(a) - s], b[p:len(b) - s]


# --------------------------------------------------------------------------
# check: one PASS or FAIL line per input
# --------------------------------------------------------------------------
class Lines:
    def __init__(self):
        self.rows = []

    def add(self, status, cid, text):
        self.rows.append({"status": status, "id": cid, "text": text, "details": []})
        say(f"{status:<4} {cid}  {text}")

    def detail(self, text):
        say(f"           {text}")
        if self.rows:
            self.rows[-1]["details"].append(text)

    def count(self, status):
        return sum(1 for r in self.rows if r["status"] == status)


def check_repo(paths, a, L, ctx):
    g = H.git_info(paths.repo)
    ctx["git"] = g
    if not g["commit"]:
        L.add("FAIL", "C01", f"{paths.repo} is not a Git clone of the repo. Run Cell 0 from the start.")
        return
    tracked = git_lines(paths.repo, "status", "--porcelain", "--untracked-files=no")
    untracked = [x for x in git_lines(paths.repo, "status", "--porcelain") if x.startswith("??")]
    if H.HARNESS_VERSION != EXPECTED["harness_version"]:
        L.add("FAIL", "C01", f"harness version {H.HARNESS_VERSION}, expected {EXPECTED['harness_version']} "
                             "(pull the repo; a harness change needs the ASSET-8 rules and a decision row)")
    elif tracked:
        L.add("FAIL", "C01", f"repo at {g['commit'][:12]} has {len(tracked)} uncommitted change(s) "
                             f"({'; '.join(x.strip() for x in tracked[:3])}). Commit the change on GitHub, "
                             "then run Cell 0 again: pilot answers must come from committed code.")
    else:
        L.add("PASS", "C01", f"repo at commit {g['commit'][:12]}, no uncommitted change; harness "
                             f"{H.HARNESS_VERSION}; this script {SCRIPT_VERSION}")
    if untracked:
        L.add("INFO", "C01", f"{len(untracked)} untracked path(s) in the repo, for example "
                             f"{untracked[0][3:].strip()}. Stage main treats them as uncommitted (Gate E "
                             "needs a clean tree): add them to .gitignore before Gate E.")


def check_prompts(paths, a, L, ctx):
    ypath = paths.repo / "configs" / "prompts.yaml"
    ctx["prompts_bytes"] = open(ypath, "rb").read()
    ctx["P_raw"] = load_yaml(ypath)
    P = promptlib.load_prompts(ypath)          # stops if the text and the lock disagree
    ctx["P"] = P
    same = P["templates"] == ctx["P_raw"]["templates"]
    if P["prompt_version"] == EXPECTED["prompt_version"] and P["sha256"] == EXPECTED["prompt_sha256"] and same:
        L.add("PASS", "C02", f"configs/prompts.yaml: the lock holds, version {P['prompt_version']}, "
                             f"sha256 {P['sha256'][:12]} (as for the smoke runs and the FT dry run); "
                             "the harness loads the templates unchanged")
    else:
        L.add("FAIL", "C02", f"configs/prompts.yaml: version {P['prompt_version']}, sha256 {P['sha256'][:12]}; "
                             f"expected {EXPECTED['prompt_version']}, {EXPECTED['prompt_sha256'][:12]}"
                             + ("" if same else "; the loaded templates differ from the file")
                             + ". A prompt change needs a new prompt_version, a decision row and a new FT dry run.")


def check_design(paths, a, L, ctx):
    P_raw = ctx.get("P_raw")
    if P_raw is None:
        L.add("SKIP", "C03", "Design 6.2 comparison: prompts.yaml was not read")
        return
    if not paths.design.exists():
        L.add("FAIL", "C03", f"design file not found: {paths.design}. Download BanglaIndoorVQA_Design.md "
                             f"(v2.1) from the Claude project, upload it to Drive {paths.design.parent}/ "
                             "and run Cell 0 again.")
        return
    d = read_design_62(paths.design)
    ctx["design_sha256"] = d["sha256"]
    if not d["version_ok"]:
        L.add("FAIL", "C03", f"{paths.design.name} does not say 'Version 2.1' at the top. Use the design v2.1 "
                             "of the Claude project.")
        return
    labels = [lab for lab, _t in d["blocks"]]
    found, problems = {}, []
    for key, _kind, _target in DESIGN_BLOCKS:
        hits = [t for lab, t in d["blocks"] if lab.startswith(key)]
        if len(hits) != 1:
            problems.append(f"label '{key}' has {len(hits)} code blocks")
        else:
            found[key] = hits[0]
    if len(d["blocks"]) != len(DESIGN_BLOCKS) or problems:
        L.add("FAIL", "C03", f"Design 6.2 has {len(d['blocks'])} code blocks under the labels {labels}; "
                             f"expected {len(DESIGN_BLOCKS)} ({'; '.join(problems) or 'other labels'}). "
                             "Send this output to the Pilot chat.")
        return
    T = P_raw["templates"]
    diffs = []
    for key, kind, target in DESIGN_BLOCKS:
        text = found[key]
        if kind == "full":
            tid, var = target
            diffs += compare_text(f"{tid}.{var}", text, T[tid][var])
        elif kind == "starts":
            rows = text.split("\n")
            if len(rows) != len(target):
                diffs.append(f"the no-image block has {len(rows)} lines, expected {len(target)}")
                continue
            for row, (tid, var) in zip(rows, target):
                cut = row.find("…") if "…" in row else row.find("...")
                if cut <= 0:
                    diffs.append(f"no ellipsis in the no-image line '{visible(row)}'")
                    continue
                start = row[:cut]
                diffs += compare_text(f"{tid}.{var} (its start, up to the ellipsis of Design 6.2)",
                                      start, T[tid][var][:len(start)])
        else:
            diffs += compare_text("the N-BBs specificity sentence", text, P_raw["specificity_sentence"])
    note = " (the design file has CRLF line ends; they are not part of the templates)" if d["crlf"] else ""
    if not diffs:
        L.add("PASS", "C03", "Design 6.2 against configs/prompts.yaml, code point by code point: identical "
                             "(L-BB/N-BB, L-BE/N-BE, L-EE, L-EB, N-FCB, N-FCE in full; the starts of the "
                             f"two no-image versions; the N-BBs sentence). Design file sha256 {d['sha256'][:12]}{note}")
    else:
        L.add("FAIL", "C03", "Design 6.2 and configs/prompts.yaml differ (below). Do NOT run. Send this "
                             f"output to the Pilot chat. Design file sha256 {d['sha256'][:12]}{note}")
        for line in diffs:
            L.detail(line)


def check_chars(paths, a, L, ctx):
    P_raw, raw = ctx.get("P_raw"), ctx.get("prompts_bytes")
    if P_raw is None:
        L.add("SKIP", "C04", "hidden characters: prompts.yaml was not read")
        return
    problems = []
    if raw.startswith(b"\xef\xbb\xbf"):
        problems.append("a byte order mark at the start of the file")
    if b"\r" in raw:
        problems.append("carriage return (CR) characters in the file")
    texts = [(f"templates.{tid}.{var}", t) for tid, vs in P_raw["templates"].items() for var, t in vs.items()]
    texts += [("answer_language_sentence_bn", P_raw["answer_language_sentence_bn"]),
              ("specificity_sentence", P_raw["specificity_sentence"])]
    texts += [(f"image_clause.{k}", v) for k, v in (P_raw.get("image_clause") or {}).items()]
    for name, t in texts:
        bad = sorted({show_char(c) for c in t if not allowed_char(c)})
        if bad:
            problems.append(f"{name}: {', '.join(bad)}")
        if unicodedata.normalize("NFC", t) != t:
            problems.append(f"{name}: not in Unicode NFC form")
        if "\t" in t or "  " in t.replace("\n", " "):
            problems.append(f"{name}: a tab or a double space")
    if problems:
        L.add("FAIL", "C04", "configs/prompts.yaml holds hidden or unusual characters. Do NOT run; send "
                             "this output to the Pilot chat.")
        for p in problems:
            L.detail(p)
    else:
        L.add("PASS", "C04", f"configs/prompts.yaml: no hidden character in its {len(texts)} texts; only ASCII, "
                             "line feeds, the Bengali block U+0980 to U+09FF and the danda; no byte order mark, "
                             "no CR, no tab, no double space; all in Unicode NFC")


def check_derived(paths, a, L, ctx):
    P_raw = ctx.get("P_raw")
    if P_raw is None:
        L.add("SKIP", "C05", "derived templates: prompts.yaml was not read")
        return
    T = P_raw["templates"]
    problems, pairs = [], []
    clauses = P_raw.get("image_clause") or {}
    for tid, clause in clauses.items():
        w, n = T[tid]["with_image"], T[tid].get("no_image")
        if n is None or w.count(clause) != 1 or w.replace(clause, "", 1) != n:
            problems.append(f"{tid}: no_image is not with_image without '{visible(clause)}'")
    ans, spec = P_raw["answer_language_sentence_bn"], P_raw["specificity_sentence"]
    base = T["open_bn_bn"]["with_image"]
    if base.count(ans) != 1 or base.replace(ans, ans + " " + spec, 1) != T["open_bn_bn_specific"]["with_image"]:
        problems.append("open_bn_bn_specific is not open_bn_bn with the specificity sentence added after "
                        "the answer-language sentence")
    for a_id, b_id, anchor in (("open_bn_bn", "open_bn_en", ans),
                               ("open_en_en", "open_en_bn", "Write the answer in English only.")):
        x, y = T[a_id]["with_image"], T[b_id]["with_image"]
        p, mx, my = differing_middle(x, y)
        start = x.find(anchor)
        inside = start >= 0 and start <= p and p + len(mx) <= start + len(anchor)
        clean = all(ch not in m for m in (mx, my) for ch in ("\n", "।", "."))
        if inside and clean and mx and my:
            pairs.append(f"{a_id} and {b_id} differ only in '{mx}' against '{my}'")
        else:
            problems.append(f"{a_id} and {b_id} differ outside the answer-language sentence")
    if problems:
        L.add("FAIL", "C05", "the templates are not derived from each other as Design 6.1 and 6.2 say:")
        for p in problems:
            L.detail(p)
    else:
        L.add("PASS", "C05", f"no-image = with-image minus the image clause ({', '.join(clauses)}); N-BBs = N-BB "
                             "plus the specificity sentence after the answer-language sentence; "
                             + "; ".join(pairs))


def check_conditions(paths, a, L, ctx):
    P = ctx.get("P")
    if P is None:
        L.add("SKIP", "C06", "conditions: prompts.yaml was not loaded")
        return
    cfg = promptlib.load_conditions(paths.repo / "configs" / "conditions.yaml", P)
    ctx["cfg"] = cfg
    bad = []
    for cid, (tid, img, ql, al, _part) in SCOPE.items():
        c = cfg["conditions"].get(cid)
        if not c:
            bad.append(f"{cid} is missing")
            continue
        got = (c.get("template"), bool(c.get("image")), c.get("question_lang"), c.get("answer_lang"))
        if got != (tid, img, ql, al) or c.get("forced_choice"):
            bad.append(f"{cid} is {got}, expected {(tid, img, ql, al)}")
    st = (cfg.get("stages") or {}).get("pilot") or {}
    if list(st.get("gemini_modes") or []) != ["standard"]:
        bad.append(f"stage pilot Gemini modes {st.get('gemini_modes')} (D140: standard only)")
    if float(st.get("api_cap_usd") or 0) != 2.0:
        bad.append(f"stage pilot cap {st.get('api_cap_usd')} USD (D144: 2.00)")
    if st.get("probe_rule") != "PILOT":
        bad.append(f"stage pilot probe rule {st.get('probe_rule')}")
    pr = (cfg.get("probes") or {}).get("PILOT") or {}
    if pr.get("file") != "PILOT.json" or pr.get("expected_n") != 100:
        bad.append(f"probe PILOT is {pr}")
    if list(cfg["sets"].get("EXP-L") or []) != ["L-BB", "L-BE", "L-EB", "L-EE", "L-BB0", "L-EE0"]:
        bad.append(f"set EXP-L is {cfg['sets'].get('EXP-L')}")
    if list(cfg["sets"].get("EXP-N-open") or []) != ["N-BB", "N-BE", "N-BBs"]:
        bad.append(f"set EXP-N-open is {cfg['sets'].get('EXP-N-open')}")
    if bad:
        L.add("FAIL", "C06", "configs/conditions.yaml differs from Design 7 and addendum E: " + "; ".join(bad))
    else:
        L.add("PASS", "C06", "configs/conditions.yaml: the 9 conditions have the templates, image use and "
                             "languages of Design 7.1 and 7.2; stage pilot is standard mode only, cap 2.00 USD, "
                             "PILOT.json with 100 items")


def check_models(paths, a, L, ctx):
    models = H.load_models(paths.repo / "configs" / "models.yaml")
    ctx["models"] = models
    g, q, m = models["gemini"], models["qwen"], models["gemma"]
    bad = []
    for k, v in (("model_id", "gemini-3.6-flash"), ("thinking_level", "minimal"), ("max_output_tokens", 256),
                 ("media_resolution", "MEDIA_RESOLUTION_HIGH")):
        if g.get(k) != v:
            bad.append(f"gemini {k} = {g.get(k)} (expected {v})")
    pr = g.get("prices") or {}
    if (pr.get("standard") or {}) != {"in": 0.75, "out": 3.75} or (pr.get("batch") or {}) != {"in": 0.375, "out": 1.875}:
        bad.append(f"gemini prices {pr.get('standard')}, {pr.get('batch')} differ from Google's page "
                   f"(read {GOOGLE_CHECK['checked_utc']}: 0.75 and 3.75; Batch 0.375 and 1.875)")
    if str(pr.get("valid_until", "")) < datetime.now(timezone.utc).strftime("%Y-%m-%d"):
        bad.append(f"gemini prices valid until {pr.get('valid_until')}")
    for key, model_id, extra in (("qwen", "Qwen/Qwen3.5-4B", {"max_pixels": 786432}),
                                 ("gemma", "google/gemma-4-E4B-it", {"image_token_budget": 1120})):
        mm = models[key]
        want = dict({"model_id": model_id, "enable_thinking": False, "max_new_tokens": 32,
                     "dtype": "bfloat16", "revision": EXPECTED["revision"][key]}, **extra)
        for k, v in want.items():
            if mm.get(k) != v:
                bad.append(f"{key} {k} = {mm.get(k)!r} (expected {v!r}"
                           + (", D155: paste the commit before the first pilot run)" if k == "revision" else ")"))
    for key, mm in models.items():
        if mm.get("adapter") and mm.get("revision") != q.get("revision"):
            bad.append(f"{key} does not inherit the Qwen commit")
    qsha = H.canonical_sha(H.hf_settings(q))
    if not qsha.startswith(EXPECTED["qwen_settings_sha_prefix"]):
        bad.append(f"Qwen settings hash {qsha[:12]}, the smoke and FT runs had {EXPECTED['qwen_settings_sha_prefix']}")
    same = []
    for key, fn in (("gemini", H.gemini_settings), ("gemma", H.hf_settings)):
        now = H.canonical_sha(fn(models[key]))
        runs = [(p.parent.parent.name, H.read_json(p, {}).get("settings_sha256") == now)
                for p in sorted((paths.root / "outputs" / "smoke").glob(f"*/{key}/stage_state.json"))]
        if runs and not any(ok for _n, ok in runs):
            bad.append(f"the {key} settings (hash {now[:12]}) differ from every smoke run "
                       f"({', '.join(n for n, _ok in runs)})")
        elif runs:
            same.append(f"{key} as smoke {', '.join(n for n, ok in runs if ok)}")
    if bad:
        L.add("FAIL", "C07", "configs/models.yaml: " + "; ".join(bad))
    else:
        L.add("PASS", "C07", "configs/models.yaml: fixed settings of Design 5.2 and addendum E; commits pinned "
                             f"(D155): Qwen {q['revision'][:12]}, Gemma {m['revision'][:12]}, the adapters inherit "
                             f"Qwen's; settings hashes: Qwen {qsha[:12]} as the smoke and FT runs"
                             + (f", {', '.join(same)}" if same else ""))


def check_pilot(paths, a, L, ctx):
    if not paths.probe.exists():
        L.add("FAIL", "C08", f"missing {paths.probe}")
        return
    sha = H.file_sha256(paths.probe)
    probe = H.load_probe(paths.probe)
    items = probe["items"]
    closed, naming, pclust = split_items(items)
    ctx.update(probe=probe, items=items, closed=closed, naming=naming, pclust=pclust, pilot_sha256=sha)
    problems = []
    if sha != EXPECTED["pilot_sha256"]:
        problems.append(f"sha256 {sha[:12]} is not the copy attached to the Pilot chat "
                        f"({EXPECTED['pilot_sha256'][:12]}); send both files to the chat")
    if len(items) != EXPECTED["pilot_n"]:
        problems.append(f"{len(items)} items, expected {EXPECTED['pilot_n']}")
    parts = Counter(str(r.get("pilot_part")) for r in items)
    if parts != Counter({"closed": 60, "naming": 40}):
        problems.append(f"pilot_part counts {dict(parts)}, expected closed 60 and naming 40")
    types = Counter(r["qa_type"] for r in closed)
    if set(types) != set(CLOSED_TYPES) or any(types[t] != 12 for t in CLOSED_TYPES):
        problems.append(f"closed items by type {dict(types)}, expected 12 of each closed type")
    if not all(r.get("closed_kind") for r in closed):
        problems.append("a closed item has no closed_kind")
    if not all(r["qa_type"] == "obj_name" and r["capture_type"] == "closeup" for r in naming):
        problems.append("a naming item is not a close-up obj_name item")
    pc = Counter(r.get("in_pclust") for r in naming)
    if pc != Counter({True: 20, False: 20}):
        problems.append(f"in_pclust among the naming items {dict(pc)}, expected true 20 and false 20")
    if not all(bool(r.get("cluster_id")) == (r.get("in_pclust") is True) for r in naming):
        problems.append("cluster_id and in_pclust disagree on a naming item")
    if len(pclust) != 20:
        problems.append(f"{len(pclust)} naming items with in_pclust = true, expected 20")
    houses = {r["house_id"] for r in items}
    if len(houses) != EXPECTED["pilot_houses"]:
        problems.append(f"{len(houses)} houses, expected {EXPECTED['pilot_houses']}")
    if paths.pilot_ids.exists():
        listed = set(read_ids(paths.pilot_ids))
        if listed != {r["qa_id"] for r in items}:
            problems.append(f"the qa_ids differ from {paths.pilot_ids.name} ({len(listed)} ids)")
    else:
        problems.append(f"missing {paths.pilot_ids}")
    if problems:
        L.add("FAIL", "C08", "PILOT.json: " + "; ".join(problems))
        return
    images = len({r["image_file"] for r in items})
    L.add("PASS", "C08", f"PILOT.json: sha256 {sha[:12]}, {len(items)} items, {len(houses)} houses, {images} images, "
                         f"the same qa_ids as probes/PILOT.ids.txt. Split from the fields (D156): pilot_part = "
                         "closed 60 (12 per closed type) for the 6 EXP-L conditions; pilot_part = naming 40 (all "
                         "close-up obj_name; in_pclust true 20, false 20) for N-BB, N-BE, N-BBs; in_pclust = true "
                         "20 for N-FCB, N-FCB0 later. 480 answers per model now (360 + 120)")


def excluded_prefix_ok(path):
    """True when the header and the 14 rows this chat started from are unchanged (rows may only be added)."""
    lines = open(str(path), "rb").read().split(b"\n")
    n = EXPECTED["excluded_rows"] + 1
    return len(lines) > n and hashlib.sha256(b"\n".join(lines[:n]) + b"\n").hexdigest() == EXPECTED["excluded_sha256"]


def check_excluded(paths, a, L, ctx):
    if not paths.excluded.exists():
        L.add("FAIL", "C09", f"missing {paths.excluded}")
        return
    rows, cols = read_excluded(paths.excluded)
    sha = H.file_sha256(paths.excluded)
    ids = {r["qa_id"] for r in rows}
    ctx.update(excluded=ids, excluded_sha256=sha)
    want_cols = ["qa_id", "probes", "qa_type", "reason", "source", "added_at"]
    if cols != want_cols:
        L.add("FAIL", "C09", f"{paths.excluded.name} has the columns {cols}, expected {want_cols}")
        return
    problems = []
    if not excluded_prefix_ok(paths.excluded):
        problems.append(f"the first {EXPECTED['excluded_rows']} rows are not the list this chat started from "
                        f"(sha256 {EXPECTED['excluded_sha256'][:12]}): rows may only be added, never changed (C4)")
    if len(ids) != len(rows):
        problems.append("a qa_id is listed twice")
    bad = sorted({r["reason"] for r in rows} - set(REASONS))
    if bad:
        problems.append(f"unknown reason codes {bad}")
    if problems:
        L.add("FAIL", "C09", f"{paths.excluded.name}: " + "; ".join(problems))
        return
    items = ctx.get("items") or []
    overlap = sorted(ids & {r["qa_id"] for r in items})
    added = Counter(r["added_at"] for r in rows[EXPECTED["excluded_rows"]:])
    L.add("PASS", "C09",
          f"probes/excluded_items.csv: {len(rows)} rows, sha256 {sha[:12]}; the {EXPECTED['excluded_rows']} rows "
          "this chat started from are unchanged"
          + (f"; added since: {', '.join(f'{k} {v}' for k, v in added.items())}" if added else "")
          + f". PILOT items on the list: {len(overlap)} of {len(items)}"
          + (" (none)" if not overlap else ": " + ", ".join(overlap))
          + ". Under D110 every item is run; a listed item is left out of the Gate C and Gate D shares.")


def check_english(paths, a, L, ctx):
    if not paths.english.exists():
        L.add("FAIL", "C10", f"missing {paths.english}")
        return
    sha = H.file_sha256(paths.english)
    english = H.load_english(paths.english, EXPECTED["english_column"])
    ctx.update(english=english, english_sha256=sha)
    with open(str(paths.english), "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    versions = Counter(str(r.get("final_version", "")) for r in rows)
    closed, naming = ctx.get("closed") or [], ctx.get("naming") or []
    c_ok = sum(1 for r in closed if normalise(r["question_bn"]) in english["table"])
    n_ok = sum(1 for r in naming if normalise(r["question_bn"]) in english["table"])
    strings = len({normalise(r["question_bn"]) for r in closed})
    problems = []
    if sha != EXPECTED["english_sha256"]:
        problems.append(f"sha256 {sha[:12]} is not version 2 ({EXPECTED['english_sha256'][:12]}). The pilot runs "
                        "on version 2 only, and the PILOT file is never replaced once the pilot ran (D135)")
    if len(rows) != EXPECTED["english_rows"]:
        problems.append(f"{len(rows)} rows, expected {EXPECTED['english_rows']}")
    if closed and c_ok != len(closed):
        problems.append(f"only {c_ok} of {len(closed)} closed items find their English question")
    if problems:
        L.add("FAIL", "C10", f"{paths.english.name}: " + "; ".join(problems))
        return
    L.add("PASS", "C10", f"{paths.english.name}: sha256 {sha[:12]} (version 2"
                         + (f", final_version {dict(versions)}" if versions and "" not in versions else "")
                         + f"), {len(rows)} rows, column {EXPECTED['english_column']}; English found for "
                         f"{c_ok} of {len(closed)} closed items ({strings} strings) by the join rule of addendum D4; "
                         f"the naming items, for N-FCE later: {n_ok} of {len(naming)}")


def check_images(paths, a, L, ctx):
    items = ctx.get("items")
    if not items:
        L.add("SKIP", "C11", "images: PILOT.json was not read")
        return
    man = imglib.read_manifest(paths.images)
    ctx["man"] = man
    files = sorted({r["image_file"] for r in items})
    missing = [f for f in files if f not in man]
    bad = []
    for f in files:
        if f in man:
            try:
                imglib.load_prepared(paths.images, f, man)
            except Exception as exc:
                bad.append(f"{f}: {exc}")
    if missing or bad:
        L.add("FAIL", "C11", f"data/images_1024: {len(missing)} PILOT images missing from the manifest, "
                             f"{len(bad)} not matching it (first: {(missing + bad)[0]})")
    else:
        L.add("PASS", "C11", f"data/images_1024: all {len(files)} PILOT images are in images_manifest.csv "
                             f"({len(man)} rows) and match their sha256")


def check_built(paths, a, L, ctx):
    need = ("cfg", "P", "models", "english", "man", "closed", "naming")
    if any(ctx.get(k) is None for k in need):
        L.add("SKIP", "C12", "built prompts: an earlier input failed")
        return
    cfg, P, english = ctx["cfg"], ctx["P"], ctx["english"]
    counts, problems, en_hit, en_all = {}, [], 0, 0
    for cid, (tid, img, ql, _al, part) in SCOPE.items():
        cond = cfg["conditions"][cid]
        sel = ctx["closed"] if part == "closed" else ctx["naming"]
        work = H.build_work(ctx["models"]["gemini"]["model_id"], cond, sel, P, english, None, ctx["man"])
        tmpl = P["templates"][tid]["with_image" if img else "no_image"]
        prefix, suffix = tmpl[:tmpl.index("{")], tmpl[tmpl.rindex("}") + 1:]
        for w, it in zip(work, sel):
            t = w["prompt_text"]
            if (w["qa_id"] != it["qa_id"] or not t.startswith(prefix) or not t.endswith(suffix)
                    or "{question_" in t or "{option_" in t or bool(w["has_image"]) != img):
                problems.append(f"{cid} {it['qa_id']}")
            if ql == "en":
                en_all += 1
                en_hit += english["table"][normalise(it["question_bn"])] in t
        counts[cid] = len(work)
    if problems or len(counts) != len(SCOPE):
        L.add("FAIL", "C12", f"built prompts: {len(problems)} prompts do not keep the template text around the "
                             f"question (first: {problems[:3]}). Send this output to the Pilot chat.")
        return
    L.add("PASS", "C12", f"{sum(counts.values())} prompts built with the one build_prompt(): "
                         + ", ".join(f"{c} {n}" for c, n in counts.items())
                         + "; the template text around the question is unchanged, an image goes with the 7 "
                         f"with-image conditions only, no placeholder is left; the English question appears as "
                         f"written in {en_hit} of {en_all} English prompts")


def check_ledger(paths, a, L, ctx):
    total = ledger_total(paths.ledger)
    pilot = stage_spend(paths.ledger, "pilot")
    room = cost_ledger.HARD_CAP_USD - total
    ok = room > 2.0 and pilot + 2.5 * FC_USD_ESTIMATE < 2.0
    note = ""
    if pilot == 0 and abs(total - EXPECTED["ledger_before_usd"]) > 0.00005:
        note = f"; PROJECT_STATE says {EXPECTED['ledger_before_usd']:.4f} USD so far: bring both numbers"
    L.add("PASS" if ok else "FAIL", "C13",
          f"API ledger: project total {total:.4f} USD of {cost_ledger.HARD_CAP_USD:.0f}; stage pilot "
          f"{pilot:.4f} of 2.00 USD; the four forced-choice conditions add about {FC_USD_ESTIMATE:.2f} USD "
          f"(80 Gemini calls, estimated from the pilot's measured tokens; the harness prints a ceiling and "
          f"enforces the stage cap){note}. Prepaid credit is not visible to the API: compare with the AI "
          "Studio billing page after the run")


def check_env(paths, a, L, ctx):
    if a.session == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        L.add("PASS" if key else "FAIL", "C14",
              "GEMINI_API_KEY is set from Colab Secrets" if key else
              "GEMINI_API_KEY is not set: Cell 0 reads it from the Colab secret GEMINI_API_KEY")
        return
    if a.session != "open":
        L.add("INFO", "C14", "session 'any': no API key and no GPU check")
        return
    problems = []
    try:
        import torch
        import transformers
    except ImportError as exc:
        L.add("FAIL", "C14", f"torch or transformers cannot be imported ({exc})")
        return
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    if "L4" not in gpu:
        problems.append(f"GPU '{gpu or 'none'}': choose Runtime, Change runtime type, L4 GPU (Design 5.3)")
    if transformers.__version__ != EXPECTED["transformers"]:
        problems.append(f"transformers {transformers.__version__}; run pip install -q flash-linear-attention "
                        f"transformers=={EXPECTED['transformers']} (D153)")
    if importlib.util.find_spec("fla") is None:
        problems.append("flash-linear-attention is not installed (D152)")
    if not os.environ.get("HF_TOKEN"):
        problems.append("HF_TOKEN is not set")
    if problems:
        L.add("FAIL", "C14", "; ".join(problems))
    else:
        L.add("PASS", "C14", f"{gpu}; torch {torch.__version__} (smoke and FT runs: {EXPECTED['torch_before']}); "
                             f"transformers {transformers.__version__}; flash-linear-attention installed; HF_TOKEN set. "
                             "Each model's first pilot run fixes this environment for the stage (D147)")


DOC_SUFFIXES = (".md", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".txt")


def check_hub(paths, a, L, ctx):
    models = ctx.get("models")
    if models is None:
        L.add("SKIP", "C15", "Hugging Face commits: models.yaml was not read")
        return
    try:
        from huggingface_hub import HfApi
    except ImportError:
        L.add("INFO", "C15", "huggingface_hub is not installed here; the commit check runs in Session B")
        return
    api, token = HfApi(), (os.environ.get("HF_TOKEN") or None)
    for key in OPEN_MODELS:
        mm = models[key]
        mid, pinned = mm["model_id"], mm.get("revision")
        try:
            newest = api.model_info(mid, token=token).sha
        except Exception as exc:
            L.add("INFO", "C15", f"{mid}: the Hub could not be read ({type(exc).__name__}). The open-model runs "
                                 "need it to download the weights.")
            continue
        if newest == pinned:
            L.add("PASS", "C15", f"{mid}: the pinned commit {pinned[:12]} is the newest on the Hub")
            continue
        try:
            commits = api.list_repo_commits(mid, token=token)
            ids = [c.commit_id for c in commits]
            if pinned not in ids:
                L.add("FAIL", "C15", f"{mid}: the pinned commit {str(pinned)[:12]} is not in the history on the Hub")
                continue
            newer = commits[:ids.index(pinned)]
            old = {s.rfilename: s.blob_id for s in api.model_info(mid, revision=pinned, files_metadata=True,
                                                                   token=token).siblings}
            new = {s.rfilename: s.blob_id for s in api.model_info(mid, revision=newest, files_metadata=True,
                                                                   token=token).siblings}
        except Exception as exc:
            L.add("FAIL", "C15", f"{mid}: a newer commit {newest[:12]} exists and its changes could not be read "
                                 f"({type(exc).__name__}). Bring this line to the Pilot chat.")
            continue
        changed = sorted(f for f in set(old) | set(new) if old.get(f) != new.get(f))
        loaded = [f for f in changed if not (f.lower().endswith(DOC_SUFFIXES) or f in (".gitattributes", "LICENSE"))]
        status = "FAIL" if loaded else "PASS"
        L.add(status, "C15", f"{mid}: {len(newer)} newer commit(s) on the Hub; changed files: "
                             f"{', '.join(changed) or 'none'}. "
                             + ("They change files the model loads: keep the pinned commit unless a decision row "
                                "says otherwise; bring this output to the Pilot chat before Session B."
                                if loaded else "Documentation only: keep the pinned commit (D155)."))
        for c in newer[:10]:
            when = c.created_at.strftime("%Y-%m-%d") if getattr(c, "created_at", None) else "?"
            L.detail(f"{when}  {c.commit_id[:12]}  {getattr(c, 'title', '')}")


def check_outputs(paths, a, L, ctx):
    parts = []
    for key in MODELS:
        d = paths.pilot_out / key
        cache = d / "cache.jsonl"
        if not cache.exists():
            parts.append(f"{key} none")
            continue
        recs = H.Cache(cache).records.values()
        by = Counter(r["condition_id"] for r in recs)
        st = H.read_json(d / "stage_state.json", {})
        rev = (st.get("env") or {}).get("model_revision", "")
        parts.append(f"{key} {sum(by.values())} ({', '.join(f'{c} {n}' for c, n in sorted(by.items()))}"
                     + (f"; commit {rev[:12]}" if rev else "") + ")")
    L.add("INFO", "C16", "answers already in outputs/pilot/: " + "; ".join(parts)
          + ". A run goes on from what is there.")


# ---- the Pilot reading: forced-choice files, scorer and outputs (C17 to C24) ----------------------
def check_excluded_now(paths, a, L, ctx):
    if not paths.excluded.exists():
        L.add("SKIP", "C17", "the excluded list: see C09")
        return
    sha = H.file_sha256(paths.excluded)
    rows, _cols = read_excluded(paths.excluded)
    items = ctx.get("items") or []
    listed = sorted({r["qa_id"] for r in rows} & {r["qa_id"] for r in items})
    if sha == EXPECTED["excluded_now_sha256"] and len(rows) == EXPECTED["excluded_now_rows"]:
        L.add("PASS", "C17", f"the excluded list is the {len(rows)}-row list of PROJECT_STATE (sha256 {sha[:12]}, D203 "
                             f"included). PILOT items on it: {', '.join(listed) or 'none'}; "
                             f"{len(items) - len(listed)} PILOT items are scored")
    else:
        L.add("INFO", "C17", f"the excluded list has {len(rows)} rows (sha256 {sha[:12]}), not the 28-row list of "
                             "PROJECT_STATE (8c51763fad6e). Every count follows the list as it is (D110); bring this "
                             "line to the Pilot reading chat before Cell 11")


def check_fc_options(paths, a, L, ctx):
    p = paths.root / "probes" / "fc_options.jsonl"
    if not p.exists():
        L.add("FAIL", "C18", f"missing {p}, the final ASSET-11 option file")
        return
    _S, FCm = _scorer()
    sha = H.file_sha256(p)
    table = FCm.read_option_sets(str(p))
    n = Counter(k[1] for k in table)
    ctx["fc_table"], ctx["fc_module"] = table, FCm
    problems = []
    if sha != EXPECTED["fc_options_sha256"]:
        problems.append(f"sha256 {sha[:12]} is not the ASSET-11 stage 2 file ({EXPECTED['fc_options_sha256'][:12]})")
    if dict(n) != EXPECTED["fc_sets"]:
        problems.append(f"{dict(n)} sets by language, expected {EXPECTED['fc_sets']}")
    if problems:
        L.add("FAIL", "C18", "Drive probes/fc_options.jsonl: " + "; ".join(problems) + ". Do not run Cell 8.")
        return
    L.add("PASS", "C18", f"Drive probes/fc_options.jsonl: sha256 {sha[:12]}, the final file of ASSET-11 stage 2; "
                         f"{sum(n.values())} option sets ({n['bn']} Bangla, {n['en']} English) read through "
                         "src/fc_options.py")


def check_pfce(paths, a, L, ctx):
    p = paths.root / "probes" / "P-FCE.json"
    if not p.exists():
        L.add("FAIL", "C19", f"missing {p}")
        return
    sha = H.file_sha256(p)
    ids = [r["qa_id"] for r in H.load_probe(p)["items"]]
    ctx["pfce_ids"] = set(ids)
    problems = []
    if sha != EXPECTED["pfce_sha256"]:
        problems.append(f"sha256 {sha[:12]} is not the ASSET-11 file ({EXPECTED['pfce_sha256'][:12]})")
    if len(ids) != EXPECTED["pfce_n"] or len(set(ids)) != len(ids):
        problems.append(f"{len(ids)} items ({len(set(ids))} different), expected {EXPECTED['pfce_n']}")
    lst = paths.repo / "probes" / "P-FCE.ids.txt"
    if not lst.exists():
        problems.append(f"missing {lst}")
    elif set(read_ids(lst)) != set(ids):
        problems.append("the qa_ids differ from repo probes/P-FCE.ids.txt")
    if problems:
        L.add("FAIL", "C19", "Drive probes/P-FCE.json: " + "; ".join(problems) + ". Do not run Cell 8.")
        return
    L.add("PASS", "C19", f"Drive probes/P-FCE.json: sha256 {sha[:12]}, {len(ids)} items, the same qa_ids as repo "
                         "probes/P-FCE.ids.txt")


def check_fc_items(paths, a, L, ctx):
    need = ("items", "naming", "closed", "pclust", "excluded", "fc_table", "fc_module", "pfce_ids")
    if any(ctx.get(k) is None for k in need):
        L.add("SKIP", "C20", "forced-choice items of the pilot: an earlier input failed")
        return
    pc_path = paths.root / "probes" / "P-CLUST.json"
    if not pc_path.exists():
        L.add("FAIL", "C20", f"missing {pc_path}")
        return
    pc_sha = H.file_sha256(pc_path)
    pclust_ids = {r["qa_id"] for r in H.load_probe(pc_path)["items"]}
    FCm, table, pfce = ctx["fc_module"], ctx["fc_table"], ctx["pfce_ids"]
    problems = []
    if len(pclust_ids) != EXPECTED["pclust_n"] or not pc_sha.startswith(EXPECTED["pclust_sha_prefix"]):
        problems.append(f"P-CLUST.json has {len(pclust_ids)} items, sha256 {pc_sha[:12]} (expected "
                        f"{EXPECTED['pclust_n']}, {EXPECTED['pclust_sha_prefix']})")
    flag = [r["qa_id"] for r in ctx["naming"] if (r.get("in_pclust") is True) != (r["qa_id"] in pclust_ids)]
    if flag:
        problems.append(f"in_pclust of PILOT.json disagrees with P-CLUST.json for {flag[:5]}")
    fcb = [r["qa_id"] for r in ctx["pclust"]]
    fce = [r["qa_id"] for r in ctx["naming"] if r["qa_id"] in pfce]
    for qa_ids, lang in ((fcb, "bn"), (fce, "en")):
        for qa in qa_ids:
            try:
                FCm.options_for(table, qa, lang)
            except Exception as exc:
                problems.append(f"{qa} has no {lang} option set ({type(exc).__name__})")
    if not set(fce) <= set(fcb):
        problems.append(f"naming items in P-FCE but not in P-CLUST: {sorted(set(fce) - set(fcb))}")
    if problems:
        L.add("FAIL", "C20", "forced-choice items of the pilot: " + "; ".join(problems[:6]))
        return
    closed_in = sorted(r["qa_id"] for r in ctx["closed"] if r["qa_id"] in pclust_ids)
    listed = sorted((set(fcb) | set(fce)) & ctx["excluded"])
    runs = 2 * len(fcb) + 2 * len(fce)
    left = 2 * len([q for q in fcb if q in ctx["excluded"]]) + 2 * len([q for q in fce if q in ctx["excluded"]])
    L.add("PASS", "C20", f"forced choice per model (D156): N-FCB and N-FCB0 on {len(fcb)} items (naming, in_pclust true, "
                         f"all in P-CLUST.json), N-FCE and N-FCE0 on {len(fce)} (the naming items in P-FCE); "
                         f"{runs} answers per model, {runs - left} outside the excluded list ({', '.join(listed) or 'none'} "
                         f"listed); every item has its option set. The {len(closed_in)} closed-part P-CLUST items "
                         f"({', '.join(closed_in) or 'none'}) get no forced choice")


def check_scorer(paths, a, L, ctx):
    S, _FCm = _scorer()
    problems = []
    if S.SCORER_VERSION != EXPECTED["scorer_version"]:
        problems.append(f"scorer {S.SCORER_VERSION}, expected {EXPECTED['scorer_version']}")
    if len(S.REFUSAL_PHRASES) != EXPECTED["refusal_phrases"]:
        problems.append(f"{len(S.REFUSAL_PHRASES)} refusal phrases, expected {EXPECTED['refusal_phrases']} (D194)")
    shas = {}
    for rel, key in (("src/scoring.py", "scoring_sha256"), ("scripts/asset9_scorer.py", "asset9_scorer_sha256")):
        shas[rel] = H.file_sha256(paths.repo / rel)
        if shas[rel] != EXPECTED[key]:
            problems.append(f"{rel} sha256 {shas[rel][:12]} is not the delivered ASSET-9 stage 2 file "
                            f"({EXPECTED[key][:12]}, addendum B9)")
    if problems:
        L.add("FAIL", "C21", "scorer: " + "; ".join(problems) + ". A scorer change needs a new version, the 431 "
                             "tests and a decision row before any reading")
        return
    fc_sha = H.file_sha256(paths.repo / "src" / "fc_options.py")
    L.add("PASS", "C21", f"scorer {S.SCORER_VERSION}: src/scoring.py {shas['src/scoring.py'][:12]} and "
                         f"scripts/asset9_scorer.py {shas['scripts/asset9_scorer.py'][:12]} are the files of addendum "
                         f"B9; {len(S.REFUSAL_PHRASES)} refusal phrases; src/fc_options.py sha256 {fc_sha[:12]}")


def check_lexicon_tag(paths, a, L, ctx):
    problems = []
    for rel in ("lexicon/lexicon.csv", "lexicon/closed_vocab.csv"):
        try:
            r = subprocess.run(["git", "-C", str(paths.repo), "show", f"gate-A:{rel}"], capture_output=True, timeout=60)
        except Exception as exc:
            problems.append(f"git show gate-A:{rel} failed ({type(exc).__name__})")
            continue
        if r.returncode != 0:
            problems.append(f"git cannot show gate-A:{rel} (run: git -C {paths.repo} fetch --tags, then Cell 0 again)")
        elif r.stdout != (paths.repo / rel).read_bytes():
            problems.append(f"{rel} differs from the tag gate-A")
    if problems:
        L.add("FAIL", "C22", "frozen files: " + "; ".join(problems))
    else:
        L.add("PASS", "C22", "lexicon/lexicon.csv and lexicon/closed_vocab.csv equal the tag gate-A")


def check_reading_inputs(paths, a, L, ctx):
    parts, missing = [], []
    for name, label, key in SCORE_SOURCES:
        recs = cache_records(paths.root / "outputs" / label / key / "cache.jsonl")
        if recs is None:
            missing.append(f"outputs/{label}/{key}/cache.jsonl")
            continue
        n_fc = sum(1 for r in recs if r.get("condition_id") in FC_CONDS)
        parts.append(f"{name} {len(recs)}" + (f" (forced choice {n_fc})" if label == STAGE else ""))
    key_path = paths.root.joinpath(*PROOFREAD_KEY)
    parts.append("proofread key " + ("present" if key_path.exists() else "not found (the P118 check is skipped)"))
    if missing:
        L.add("FAIL", "C23", "outputs for the reading: missing " + ", ".join(missing) + "; " + "; ".join(parts))
        return
    L.add("INFO", "C23", "outputs for the reading: " + "; ".join(parts) + ". Scoring needs 560 answers per pilot "
                         "model (Cell 8 adds 80), 100 in the D5 repeat and 100 in the Batch test")


ENV_KEYS = {"torch": ("torch", "torch_version"), "transformers": ("transformers", "transformers_version"),
            "gpu": ("gpu", "gpu_name", "device_name"), "cuda": ("cuda", "cuda_version")}


def check_stage_env(paths, a, L, ctx):
    if a.session != "open":
        return
    try:
        import torch
        import transformers
    except ImportError as exc:
        L.add("FAIL", "C24", f"torch or transformers cannot be imported ({exc})")
        return
    now = {"torch": torch.__version__, "transformers": transformers.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
           "cuda": str(torch.version.cuda or "")}
    for key in OPEN_MODELS:
        st = H.read_json(paths.pilot_out / key / "stage_state.json", {})
        env = st.get("env") or {}
        if not env:
            L.add("INFO", "C24", f"{key}: outputs/pilot/{key}/stage_state.json holds no environment; the harness "
                                 "checks it at run time (D147)")
            continue
        compared, differ = [], []
        for k, v in sorted(env.items()):
            name = next((n for n, keys in ENV_KEYS.items() if k.lower() in keys), None)
            if name is None or isinstance(v, (dict, list)):
                continue
            compared.append(k)
            if str(v) != str(now[name]):
                differ.append(f"{k}: stage {v}, this session {now[name]}")
        if differ:
            L.add("FAIL", "C24", f"{key}: this session differs from the environment stage pilot was fixed to on "
                                 "2026-10-14 (D147): " + "; ".join(differ) + ". The harness will refuse the "
                                 "forced-choice run. Do not install another version yourself; send this output "
                                 "to the Pilot reading chat")
        elif compared:
            L.add("PASS", "C24", f"{key}: this session matches the environment of stage pilot on "
                                 + ", ".join(f"{k} {env[k]}" for k in compared) + " (D147)")
        else:
            L.add("INFO", "C24", f"{key}: the stage environment has the keys {', '.join(sorted(env))}; none names "
                                 "torch, transformers, the GPU or CUDA. The harness compares them at run time")


def cmd_check(a):
    paths = Paths(a.root, a.repo, a.english_file, a.design)
    L, ctx = Lines(), {}
    say("=" * 78)
    say(f"Pilot input check ({SCRIPT_VERSION}), session '{a.session}', {utc_now()}")
    say(f"Drive root {paths.root}   repo {paths.repo}")
    say("=" * 78)
    steps = (("C01", check_repo), ("C02", check_prompts), ("C03", check_design), ("C04", check_chars),
             ("C05", check_derived), ("C06", check_conditions), ("C07", check_models), ("C08", check_pilot),
             ("C09", check_excluded), ("C10", check_english), ("C11", check_images), ("C12", check_built),
             ("C13", check_ledger), ("C14", check_env), ("C15", check_hub), ("C16", check_outputs),
             ("C17", check_excluded_now), ("C18", check_fc_options), ("C19", check_pfce), ("C20", check_fc_items),
             ("C21", check_scorer), ("C22", check_lexicon_tag), ("C23", check_reading_inputs),
             ("C24", check_stage_env))
    for cid, fn in steps:
        try:
            fn(paths, a, L, ctx)
        except Exception as exc:
            L.add("FAIL", cid, f"{type(exc).__name__}: {exc}")
    n_fail = L.count("FAIL")
    say("-" * 78)
    say(f"{L.count('PASS')} PASS, {n_fail} FAIL, {L.count('INFO')} INFO, {L.count('SKIP')} SKIP")
    say("ALL INPUTS PASS: go on with the next cell." if not n_fail else
        "STOP: fix the FAIL lines first. If a line is unclear, send this whole output to the Pilot chat.")
    out = paths.pilot_out / "checks"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    H.write_json(out / f"{stamp}_{a.session}.json", {
        "script_version": SCRIPT_VERSION, "utc": utc_now(), "session": a.session, "git": ctx.get("git"),
        "inputs": {"pilot_sha256": ctx.get("pilot_sha256"), "english_sha256": ctx.get("english_sha256"),
                   "excluded_sha256": ctx.get("excluded_sha256"), "design_sha256": ctx.get("design_sha256")},
        "lines": L.rows, "n_fail": n_fail})
    say(f"wrote {out / (stamp + '_' + a.session + '.json')}")
    return 0 if not n_fail else 1


# --------------------------------------------------------------------------
# preview: the English questions before any English call
# --------------------------------------------------------------------------
def cmd_preview(a):
    paths = Paths(a.root, a.repo, a.english_file)
    P, cfg, models = load_configs(paths)
    items = H.load_probe(paths.probe)["items"]
    closed, naming, _pclust = split_items(items)
    english = H.load_english(paths.english, EXPECTED["english_column"])
    man = imglib.read_manifest(paths.images)
    say("QUESTION TEXT (dataset text, no model answer). Clear this output before the notebook is saved "
        "to GitHub.")
    three = H.select_items(closed, sample=GROUPS["english3"]["sample"])
    say("\nThe 3 closed items of the first English calls (group english3, seeded rank, seed 42):")
    for it in three:
        en = english["table"].get(normalise(it["question_bn"]), "(NO ENGLISH QUESTION)")
        say(f"  {it['qa_id']}  {it['qa_type']}")
        say(f"    Bangla : {it['question_bn']}")
        say(f"    English: {en}")
    first_naming = H.select_items(naming, sample=1)[0]
    say("\nOne prompt per condition, exactly as the models will get it:")
    for cid, (_tid, _img, _ql, _al, part) in SCOPE.items():
        it = three[0] if part == "closed" else first_naming
        w = H.build_work(models["gemini"]["model_id"], cfg["conditions"][cid], [it], P, english, None, man)[0]
        say(f"\n--- {cid}   item {it['qa_id']}   image {'yes' if w['has_image'] else 'no'}   prompt sha256 "
            f"{w['prompt_sha256'][:12]}")
        say(w["prompt_text"])
    say("\nRead each English question against its Bangla one: the same question, the same detail, the same "
        "difficulty. If one is wrong, stop and bring its qa_id to the Pilot chat.")
    return 0


# --------------------------------------------------------------------------
# run: one harness command, then the same command again (Gate C)
# --------------------------------------------------------------------------
def stream(argv, cwd):
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    p = subprocess.Popen(argv, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", bufsize=1, env=env)
    out = []
    for line in p.stdout:
        print(line, end="", flush=True)
        out.append(line)
    p.wait()
    return p.returncode, "".join(out), round(time.time() - t0, 1)


def parse_output(text):
    m = re.search(r"run_id (\S+)", text)
    plan = {}
    for mm in re.finditer(r"^\s+(\S+)\s+(\d+) items; in the cache\s+(\d+); to run\s+(\d+)", text, re.M):
        plan[mm.group(1)] = {"items": int(mm.group(2)), "cached": int(mm.group(3)), "to_run": int(mm.group(4))}
    return {"run_id": m.group(1) if m else "", "plan": plan,
            "complete": "COMPLETE for this command." in text,
            "nothing": "nothing to run: 0 calls" in text,
            "dry_ok": "DRY RUN OK." in text,
            "dry_resume_ok": "PASS  a second run would make 0 calls" in text,
            "stopped": "STOPPED:" in text or "\nSTOP:" in text}


def check_price_date(text):
    try:
        day = datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        raise Fail(f"--prices-checked needs a date like 2026-10-14, not '{text}'")
    age = (datetime.now(timezone.utc).date() - day).days
    if age > MAX_DAYS_SINCE_CHECK:
        raise Fail(f"The Gemini model ID and prices were last checked on {text}, {age} days ago. Design 0.2 rule 5: "
                   "check them in the week of each run. Read " + ", ".join(GOOGLE_CHECK["pages"])
                   + ". If nothing changed, run again with --prices-checked <today>; if a value changed, bring it "
                   "to the Pilot chat first.")


def check_fc_file(path):
    """D164: forced choice runs, and is scored, only on the final ASSET-11 option file."""
    p = Path(path)
    if not p.exists():
        raise Fail(f"the option file {p} does not exist; forced choice uses Drive probes/fc_options.jsonl (D164)")
    sha = H.file_sha256(p)
    if sha != EXPECTED["fc_options_sha256"]:
        raise Fail(f"{p.name} has sha256 {sha[:12]}: it is not the final ASSET-11 option file Drive "
                   f"probes/fc_options.jsonl (sha256 {EXPECTED['fc_options_sha256'][:12]}, D164). The draft file "
                   "is never run (D196).")
    return sha


def check_pfce_file(path):
    """D164: the P-FCE probe file of ASSET-11 stage 2."""
    p = Path(path)
    if p.name != "P-FCE.json" or not p.exists():
        raise Fail(f"P-FCE must be Drive probes/P-FCE.json, the ASSET-11 probe file (D164); got {p}")
    sha = H.file_sha256(p)
    if sha != EXPECTED["pfce_sha256"]:
        raise Fail(f"{p.name} has sha256 {sha[:12]}, not the ASSET-11 P-FCE file (sha256 "
                   f"{EXPECTED['pfce_sha256'][:12]}, D164)")
    return sha


def write_pfce_ids(paths, src, dry):
    """D156, D164: the qa_ids of the PILOT naming items that are in P-FCE, as the --ids-file of the
    harness. Only these ids are written, so the item set does not depend on how the harness combines
    --ids-file with --where."""
    if not src:
        raise Fail("--pfce is needed: Drive probes/P-FCE.json (D164)")
    p = Path(src)
    sha = check_pfce_file(p)
    pfce = {r["qa_id"] for r in H.load_probe(p)["items"]}
    naming = split_items(H.load_probe(paths.probe)["items"])[1]
    ids = [r["qa_id"] for r in naming if r["qa_id"] in pfce]
    out = (paths.dry_out if dry else paths.pilot_out) / "pilot_pfce_qa_ids.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(ids) + "\n")
    say(f"P-FCE: {p.name} (sha256 {sha[:12]}) holds {len(pfce)} items; {len(ids)} of them are PILOT naming "
        f"items; their qa_ids are written to {out}")
    return out


def require_fc_verify(paths):
    """D164: no forced-choice run before the dry run and fc-verify of this script version passed."""
    v = H.read_json(paths.dry_out / "fc_verify.json", {})
    ok = (v.get("result") == "PASS" and v.get("dry_run") is True and v.get("script_version") == SCRIPT_VERSION
          and v.get("options_sha256") == EXPECTED["fc_options_sha256"]
          and v.get("pfce_sha256") == EXPECTED["pfce_sha256"])
    if not ok:
        found = (f"result {v.get('result')}, script {v.get('script_version')}, {v.get('utc')}" if v
                 else f"no {paths.dry_out / 'fc_verify.json'}")
        raise Fail("Forced choice runs only after the dry run of the four conditions and 'fc-verify --dry-run' "
                   f"passed with this script version ({SCRIPT_VERSION}): run Cell 8a first (D164). Found: {found}")


def harness_argv(paths, a, g, cfg):
    stage, run_name = g.get("stage", STAGE), g.get("run_name", "")
    cond_ids = H.parse_conditions(g["conditions"], cfg)
    argv = [sys.executable, str(paths.repo / "scripts" / "asset8_harness.py"), "run",
            "--root", str(paths.root), "--repo", str(paths.repo), "--stage", stage, "--model", a.model,
            "--conditions", g["conditions"], "--probe", str(paths.probe)]
    if run_name:
        argv += ["--run-name", run_name]
    for k, v in g["where"]:
        argv += ["--where", f"{k}={v}"]
    if g.get("sample"):
        argv += ["--sample", str(g["sample"])]
    if any(cfg["conditions"][c]["question_lang"] == "en" for c in cond_ids):
        argv += ["--english-file", str(paths.english), "--english-column", EXPECTED["english_column"]]
    needs = g.get("needs", ())
    if "options" in needs:
        if not a.options_file:
            raise Fail("--options-file is needed: Drive probes/fc_options.jsonl (D164)")
        check_fc_file(a.options_file)
        argv += ["--options-file", a.options_file]
    if "pfce" in needs:
        argv += ["--ids-file", str(write_pfce_ids(paths, a.pfce, bool(a.dry_run)))]
    if a.model == "gemini":
        argv += ["--mode", "standard"]
        if not a.dry_run:
            if not a.paid_tier_confirmed:
                raise Fail("Billed run refused: add --paid-tier-confirmed after you saw a paid tier for this key's "
                           "project on the AI Studio Projects page (Design 5.5).")
            check_price_date(a.prices_checked)
            argv += ["--paid-tier-confirmed"]
    if a.dry_run:
        argv += ["--dry-run"]
    return argv, stage, run_name, cond_ids


def cmd_run(a):
    paths = Paths(a.root, a.repo, a.english_file)
    g = GROUPS[a.group]
    if g.get("gemini_only") and a.model != "gemini":
        raise Fail(f"group {a.group} is for Gemini only")
    if "d5" in g.get("needs", ()) and not a.d5_approved:
        raise Fail("The D5 repeat runs only after the team lead's yes and its decision row: add --d5-approved then.")
    if a.group in FC_GROUPS and not a.dry_run:
        require_fc_verify(paths)
    P, cfg, models = load_configs(paths)
    argv, stage, run_name, cond_ids = harness_argv(paths, a, g, cfg)
    out_dir, label = H.layout(paths.root, stage, run_name, a.model, bool(a.dry_run))
    cache = out_dir / "cache.jsonl"
    ledger = out_dir / "dryrun_ledger.jsonl" if a.dry_run else paths.ledger
    first_argv = argv + ([] if not a.dry_run or a.dry_fresh else ["--dry-keep"]) + (["--plan-only"] if a.plan_only else [])
    shown = " ".join(x if " " not in x else repr(x) for x in first_argv[1:])
    say("=" * 78)
    say(f"pilot.py run: model {a.model}, group {a.group} ({g['note']})")
    say(f"stage {label}; conditions {', '.join(cond_ids)}" + ("; DRY RUN" if a.dry_run else ""))
    say(f"harness command: python {shown}")
    say("=" * 78)
    rec = {"utc": utc_now(), "script_version": SCRIPT_VERSION, "model": a.model, "group": a.group,
           "stage": label, "dry_run": bool(a.dry_run), "conditions": cond_ids, "argv": first_argv[1:],
           "git": H.git_info(paths.repo)}
    c0, l0 = raw_lines(cache), ledger_total(ledger)
    rc1, text1, secs1 = stream(first_argv, paths.repo)
    i1 = parse_output(text1)
    c1, l1, n1 = raw_lines(cache), ledger_total(ledger), ledger_count(ledger)
    rec["first"] = {"rc": rc1, "run_id": i1["run_id"], "seconds": secs1, "plan": i1["plan"],
                    "complete": i1["complete"], "dry_ok": i1["dry_ok"], "cache_lines_before": c0,
                    "cache_lines_after": c1, "ledger_before": round(l0, 6), "ledger_after": round(l1, 6)}
    if a.plan_only:
        say("\n--plan-only: nothing was sent and nothing is checked twice.")
        return rc1
    first_ok = rc1 == 0 and (i1["dry_ok"] if a.dry_run else i1["complete"])
    second_ok = None
    if first_ok and (not a.dry_run or a.model in OPEN_MODELS):
        say("\n" + "-" * 78)
        say("Gate C: the same command again. It must make 0 calls and change neither the cache nor the ledger.")
        say("-" * 78)
        argv2 = argv + (["--dry-keep"] if a.dry_run else [])
        rc2, text2, secs2 = stream(argv2, paths.repo)
        i2 = parse_output(text2)
        c2, l2, n2 = raw_lines(cache), ledger_total(ledger), ledger_count(ledger)
        reasons = []
        if rc2 != 0:
            reasons.append(f"exit code {rc2}")
        if not i2["nothing"]:
            reasons.append("the line 'nothing to run: 0 calls' is missing")
        if c2 != c1:
            reasons.append(f"cache lines {c1} became {c2}")
        if n2 != n1 or abs(l2 - l1) > 1e-9:
            reasons.append(f"ledger {l1:.6f} USD in {n1} lines became {l2:.6f} USD in {n2} lines")
        second_ok = not reasons
        rec["second"] = {"rc": rc2, "run_id": i2["run_id"], "seconds": secs2, "nothing_to_run": i2["nothing"],
                         "cache_lines": c2, "ledger": round(l2, 6), "ledger_lines": n2, "problems": reasons}
        say("\nGATE C (second run): " + ("PASS. 0 calls; the cache stays at "
                                          f"{c2} lines; the ledger stays at {l2:.4f} USD."
                                          if second_ok else "FAIL: " + "; ".join(reasons)
                                          + ". Send this output to the Pilot chat."))
    elif first_ok and a.dry_run:
        second_ok = i1["dry_resume_ok"]
        say("\nGate C rehearsal (Gemini dry run): the harness's own check 'a second run would make 0 calls' "
            + ("PASSED." if second_ok else "FAILED.")
            + " A second Gemini dry run is not made: its fault checks need fresh calls.")
    rec["rerun_ok"] = second_ok
    H.append_jsonl(out_dir / "pilot_runs.jsonl", rec)
    say(f"logged in {out_dir / 'pilot_runs.jsonl'}")
    if not first_ok:
        say("\nNOT COMPLETE: the harness output above says why. Fix the cause and run this cell again; the cache "
            "keeps what is done.")
        return rc1 or 2
    return 0 if second_ok is not False else 4


# --------------------------------------------------------------------------
# report: counts only
# --------------------------------------------------------------------------
def hf_timing(recs):
    """Seconds per item of an open model. 'gen' is the generation time the harness measured
    (elapsed_s). 'wall' is the time between consecutive answers of one run (1 s resolution;
    summed over a block of answers, so the error stays near 1 s per block): image loading,
    the prompt and the record writing included."""
    by_run = defaultdict(list)
    for r in recs:
        if r.get("mode") == "hf":
            by_run[r["run_id"]].append(r)
    rows, startups = [], []
    for rid, rs in by_run.items():
        rs.sort(key=lambda r: str(r.get("call_id", "")))
        start, prev = run_start(rid), None
        for r in rs:
            ts = parse_ts(r.get("ts_utc"))
            gen = float(r.get("elapsed_s") or 0.0)
            if prev is None or ts is None:
                gap = gen
                if start and ts:
                    startups.append(max(0.0, (ts - start).total_seconds() - gen))
            else:
                gap = (ts - prev).total_seconds()
            prev = ts or prev
            rows.append((bool(r.get("has_image")), gen, gap))

    def agg(flag):
        sel = [x for x in rows if x[0] == flag]
        if not sel:
            return None
        gens = [x[1] for x in sel]
        return {"n": len(sel), "gen_mean": mean(gens), "gen_median": statistics.median(gens),
                "gen_p90": quantile(gens, 0.9), "wall_mean": sum(x[2] for x in sel) / len(sel)}
    return {"image": agg(True), "noimage": agg(False), "runs": len(by_run),
            "startup_median": statistics.median(startups) if startups else 0.0, "startups": startups}


def gpu_projection(t, model_key):
    img, no = t["image"], t["noimage"]
    if not img or not no:
        return None
    wi, wn, su = img["wall_mean"], no["wall_mean"], t["startup_median"]
    proj = {
        "EXP-L": (4 * 1118 * wi + 2 * 1118 * wn + 1 * su) / 3600,
        "EXP-N": ((3 * 1000 + 492 + PFCE_N) * wi + (492 + PFCE_N) * wn + 3 * su) / 3600,
    }
    if model_key == "qwen":
        proj["EXP-R evaluation"] = (EXP_R_EVAL["image"] * wi + EXP_R_EVAL["noimage"] * wn
                                    + EXP_R_EVAL["loads"] * su) / 3600
    return proj


def gemini_tokens(recs):
    by = defaultdict(list)
    for r in recs:
        if r.get("mode") in ("standard", "batch"):
            by[r["condition_id"]].append(r)
    rows = {}
    for cid, rs in by.items():
        rows[cid] = {
            "n": len(rs), "has_image": bool(rs[0].get("has_image")),
            "in": mean(int(r.get("input_tokens") or 0) for r in rs),
            "img": mean(int(r.get("image_tokens") or 0) for r in rs if r.get("has_image")),
            "out": mean(int(r.get("output_tokens") or 0) for r in rs),
            "think": mean(int(r.get("thoughts_tokens") or 0) for r in rs),
            "n_think": sum(1 for r in rs if (r.get("thoughts_tokens") or 0) > 0),
            "billed_out": mean(int(r.get("billed_output_tokens") or 0) for r in rs),
            "max_billed_out": max(int(r.get("billed_output_tokens") or 0) for r in rs),
            "usd": sum(float(r.get("usd") or 0.0) for r in rs),
            "no_usage": sum(1 for r in rs if r.get("usage_reported") is False)}
    return rows


def cmd_report(a):
    paths = Paths(a.root, a.repo, a.english_file)
    P, cfg, models = load_configs(paths)
    base = paths.dry_out if a.dry_run else paths.pilot_out
    items = H.load_probe(paths.probe)["items"]
    closed, naming, pclust = split_items(items)
    ex_rows, _cols = read_excluded(paths.excluded)
    excluded = {r["qa_id"] for r in ex_rows}
    pfce_file = base / "pilot_pfce_qa_ids.txt"
    pfce_n = (len(set(read_ids(pfce_file)) & {r["qa_id"] for r in naming}) if pfce_file.exists() else None)
    expected = {cid: len(closed) if spec[4] == "closed" else len(naming) for cid, spec in SCOPE.items()}
    expected.update({"N-FCB": len(pclust), "N-FCB0": len(pclust), "N-FCE": pfce_n, "N-FCE0": pfce_n})
    text = []

    def out(s=""):
        say(s)
        text.append(s)

    J = {"script_version": SCRIPT_VERSION, "utc": utc_now(), "dry_run": bool(a.dry_run), "models": {},
         "excluded_rows": len(ex_rows), "excluded_in_pilot": sorted(excluded & {r["qa_id"] for r in items})}
    out("=" * 78)
    out(f"PILOT REPORT ({SCRIPT_VERSION}), {utc_now()}" + ("   DRY RUN: MADE-UP ANSWERS AND NUMBERS" if a.dry_run else ""))
    out(f"outputs read from {base}")
    out("Counts only: no answer is shown. Gate C (98% non-empty AND parseable) and Gate D (the 20% floor and")
    out("the 8-word rule) are read in the Pilot chat after ASSET-9 stage 2; the shares below are information.")
    out(f"Excluded items (D110): {len(ex_rows)} on the list, {len(J['excluded_in_pilot'])} of them in PILOT; "
        "listed items are left out of every share.")
    out("=" * 78)
    timing_all = {}
    for key in MODELS:
        d = base / key
        cache = H.Cache(d / "cache.jsonl") if (d / "cache.jsonl").exists() else None
        recs = [r for r in (cache.records.values() if cache else []) if r["condition_id"] in ALL_CONDS]
        m = models[key]
        out(f"\n[{key}] {m['model_id']}")
        if not recs:
            out("  no answers yet")
            J["models"][key] = {"answers": 0}
            continue
        by = defaultdict(list)
        for r in recs:
            by[r["condition_id"]].append(r)
        out(f"  {'condition':<9}{'answers':>8}{'of':>5}{'empty':>7}{'blocked':>8}{'cut':>5}"
            f"{'<=8 words':>11}{'script ok':>11}{'think>0':>9}")
        tot = Counter()
        rows_j = {}
        for cid in [c for c in ALL_CONDS if c in by]:
            rs = by[cid]
            kept = [r for r in rs if r["qa_id"] not in excluded]
            n = len(kept)
            empty = sum(1 for r in kept if r.get("is_empty"))
            blocked = sum(1 for r in kept if r.get("block_reason"))
            cut = sum(1 for r in kept if r.get("truncated"))
            short = sum(1 for r in kept if int(r.get("n_words") or 0) <= 8)
            want = {"bn": "bangla", "en": "latin"}[ALL_CONDS[cid][3]]
            sc = Counter(H.script_of(r.get(H.ANSWER_FIELD) or "") for r in kept)
            script_ok = sc[want] + sc["digits"]
            think = sum(1 for r in kept if (r.get("thoughts_tokens") or 0) > 0)
            exp = expected.get(cid)
            out(f"  {cid:<9}{len(rs):>8}{(str(exp) if exp is not None else '-'):>5}{empty:>7}{blocked:>8}{cut:>5}"
                f"{share(short, n):>11}{share(script_ok, n):>11}{think:>9}")
            tot.update({"n": n, "empty": empty, "short": short, "excl": len(rs) - n,
                        "leak": sum(1 for r in kept if r.get("thinking_leak"))})
            rows_j[cid] = {"answers": len(rs), "expected": exp, "scored_base": n, "empty": empty,
                           "blocked": blocked, "cut": cut, "le8_words": short, "script_ok": script_ok,
                           "thinking_calls": think}
        missing = [c for c in SCOPE if c not in by]
        out(f"  all: {tot['n']} answers outside the excluded list; non-empty {tot['n'] - tot['empty']} "
            f"({share(tot['n'] - tot['empty'], tot['n']).strip()}; Gate C needs 98% non-empty and parseable); "
            f"8 words or fewer {tot['short']} ({share(tot['short'], tot['n']).strip()}; Gate D needs 90%)"
            + (f"; thinking markers in the text: {tot['leak']} (must be 0)" if key in OPEN_MODELS else ""))
        if missing:
            out(f"  not run yet: {', '.join(missing)}")
        versions = Counter(str(r.get("model_version") or r.get("model_revision") or "") for r in recs)
        days = Counter(str(r.get("ts_utc", ""))[:10] for r in recs)
        hv = Counter(str(r.get("harness_version", "")) for r in recs)
        pv = Counter(str(r.get("prompt_version", "")) for r in recs)
        out(f"  model {'version' if key == 'gemini' else 'commit'}: "
            + ", ".join(f"{k or '?'} ({v})" for k, v in versions.most_common())
            + (f"; before: {EXPECTED['gemini_version_before']}" if key == "gemini" else "")
            + f"; answer dates (UTC): {', '.join(f'{k} ({v})' for k, v in sorted(days.items()))}")
        out(f"  harness {', '.join(hv)}; prompts {', '.join(pv)}")
        man = H.read_json(d / "manifest.json", {"runs": []})
        runs = [r for r in man.get("runs", []) if r.get("stage") == STAGE]
        commits = sorted({str((r.get("git") or {}).get("commit", ""))[:12] for r in runs} - {""})
        en_sha = sorted({str((r.get("english") or {}).get("sha256", ""))[:12] for r in runs if r.get("english")})
        out(f"  code commits of the runs: {', '.join(commits) or '?'}; English file sha256: {', '.join(en_sha) or '-'}")
        mj = {"model_id": m["model_id"], "conditions": rows_j, "versions": dict(versions), "dates": dict(days),
              "commits": commits, "english_sha256": en_sha,
              "non_empty_share": (tot["n"] - tot["empty"]) / tot["n"] if tot["n"] else None,
              "le8_share": tot["short"] / tot["n"] if tot["n"] else None,
              "wall_seconds_pilot": sum(float(r.get("seconds") or 0) for r in runs)}
        if key in OPEN_MODELS:
            t = hf_timing(recs)
            proj = gpu_projection(t, key)
            timing_all[key] = mj["wall_seconds_pilot"]
            out("  seconds per item (gen = generation only; wall = with image loading and writing):")
            for flag, lab in (("image", "with image"), ("noimage", "no image")):
                x = t[flag]
                if x:
                    out(f"    {lab:<11} n {x['n']:>4}   gen mean {x['gen_mean']:.2f}  median {x['gen_median']:.2f}  "
                        f"p90 {x['gen_p90']:.2f}   wall mean {x['wall_mean']:.2f}")
            out(f"    start-up per command (model load and checks), median {t['startup_median']:.0f} s over "
                f"{len(t['startups'])} runs; wall time of all pilot commands {mj['wall_seconds_pilot'] / 60:.1f} min")
            if proj:
                parts = []
                for name, hours in proj.items():
                    plan = (DESIGN_12["gpu_h_per_model"].get(name) if name != "EXP-R evaluation"
                            else DESIGN_12["gpu_h_exp_r_eval"])
                    flag = "  OVER the design value: ask before the full runs" if hours > plan else ""
                    parts.append(f"{name} {hours:.2f} L4 h (design {plan}){flag}")
                out("  projection at these speeds: " + "; ".join(parts))
                out(f"    (EXP-N uses P-FCE = {PFCE_N}, the ASSET-11 size (H1), and the with-image "
                    "speed for forced choice; EXP-R: 16,650 generations as in G2, adapters may be slower)")
            mj.update(timing=t, projection=proj)
        else:
            rows_t = gemini_tokens(recs)
            out("  tokens per call (means; billed output = visible output + thinking):")
            out(f"    {'condition':<9}{'calls':>6}{'input':>8}{'image':>7}{'output':>8}{'thinking':>10}"
                f"{'think>0':>8}{'max billed':>11}{'USD':>9}")
            for cid in [c for c in ALL_CONDS if c in rows_t]:
                x = rows_t[cid]
                out(f"    {cid:<9}{x['n']:>6}{x['in']:>8.0f}{x['img']:>7.0f}{x['out']:>8.1f}{x['think']:>10.1f}"
                    f"{x['n_think']:>8}{x['max_billed_out']:>11}{x['usd']:>9.4f}")
            img_rows = [x for x in rows_t.values() if x["has_image"]]
            txt_rows = [x for x in rows_t.values() if not x["has_image"]]

            def wmean(rows, f):
                n = sum(x["n"] for x in rows)
                return sum(f(x) * x["n"] for x in rows) / n if n else 0.0
            in_img, in_txt = wmean(img_rows, lambda x: x["in"]), wmean(txt_rows, lambda x: x["in"])
            out_all = wmean(list(rows_t.values()), lambda x: x["billed_out"])
            out(f"    against Design 12.1 planning: input {in_img:.0f} per image call (plan about "
                f"{DESIGN_12['image_call_in']}), {in_txt:.0f} per text call (plan about {DESIGN_12['text_call_in']}); "
                f"billed output {out_all:.1f} per call (plan up to {DESIGN_12['call_out_max']})")
            pb, pb27 = m["prices"]["batch"], GOOGLE_CHECK["batch_2027"]
            usd, usd27, pending = 0.0, 0.0, []
            for cid, n_main in MAIN_N.items():
                x = rows_t.get(cid)
                n_main = n_main if n_main is not None else PFCE_N
                if not x:
                    pending.append(cid)
                    continue
                usd += n_main * (x["in"] * pb["in"] + x["billed_out"] * pb["out"]) / 1e6
                usd27 += n_main * (x["in"] * pb27["in"] + x["billed_out"] * pb27["out"]) / 1e6
            out(f"    full runs EXP-L and EXP-N in Batch mode, measured conditions: about {usd:.2f} USD at "
                f"{pb['in']}/{pb['out']} (to {m['prices']['valid_until']}), {usd27:.2f} USD at the 2027 prices; "
                f"Design 12.1 planned about {DESIGN_12['main_usd_planning']:.0f} USD"
                + (f"; not yet measured: {', '.join(pending)}" if pending else ""))
            q_img = wmean(img_rows, lambda x: x["in"] + x["billed_out"])
            q_txt = wmean(txt_rows, lambda x: x["in"] + x["billed_out"])
            bt = m["batch"]
            lbb = rows_t.get("L-BB")
            one = 1118 * (lbb["in"] + lbb["billed_out"]) if lbb else 0
            out(f"    Batch queue planning: {q_img:.0f} tokens per image request (models.yaml estimate "
                f"{bt['est_tokens_image_request']}), {q_txt:.0f} per text request (estimate "
                f"{bt['est_tokens_text_request']}); one L-BB condition of P-CORE-CLOSED about {one / 1e6:.2f} million "
                f"tokens of the {bt['max_enqueued_tokens'] / 1e6:.1f} million queue")
            mj.update(tokens=rows_t, main_usd_batch=usd, main_usd_batch_2027=usd27, pending=pending)
        J["models"][key] = mj
    if timing_all:
        both = sum(timing_all.values()) / 3600
        out(f"\nGPU time of the pilot so far (both open models, all commands): {both:.2f} L4 h "
            f"(Design 7.4 and 12.2: about {DESIGN_12['gpu_h_pilot_both']:.0f} for the whole pilot)")
    out("\nGate C, first half (every command run a second time with 0 calls):")
    all_ok, any_rec = True, False
    for key in MODELS:
        logs, _bad = H.read_jsonl(base / key / "pilot_runs.jsonl")
        latest = {}
        for rec in logs:
            if rec.get("stage") == STAGE:
                latest[rec["group"]] = rec
        for grp in [x for x in GROUPS if x in latest]:
            rec = latest[grp]
            any_rec = True
            ok = rec.get("rerun_ok")
            all_ok &= ok is True
            first = rec.get("first") or {}
            out(f"  {key:<7}{grp:<12}" + ("PASS" if ok else "FAIL" if ok is False else "not checked")
                + f"   first run {first.get('run_id', '?')}, cache {first.get('cache_lines_after', '?')} lines, "
                f"ledger {first.get('ledger_after', 0):.4f} USD")
        required = SCOPE_GROUPS + FC_GROUPS if not a.dry_run else ("exp-l", "exp-n-open")   # Cell 2 runs no english3
        todo = [x for x in required if x not in latest]
        if todo:
            all_ok = False
            out(f"  {key:<7}not run yet: {', '.join(todo)}")
    out("  verdict: " + ("every command so far re-ran from the cache at 0 calls" if any_rec and all_ok
                         else "NOT complete"))
    J["gate_c_rerun_ok"] = bool(any_rec and all_ok)
    if not a.dry_run:
        total, pilot = ledger_total(paths.ledger), stage_spend(paths.ledger, "pilot")
        rep = stage_spend(paths.ledger, "smoke/pilot-repeat")
        out(f"\nAPI cost (ledger): stage pilot {pilot:.4f} USD of 2.00 (Design 12.1 planned about "
            f"{DESIGN_12['pilot_usd_planning']:.2f} for about {DESIGN_12['pilot_calls']} calls)"
            + (f"; D5 repeat {rep:.4f} USD" if rep else "")
            + f"; project total {total:.4f} of {cost_ledger.HARD_CAP_USD:.0f} USD. Compare with the AI Studio "
            "billing page.")
        J.update(ledger_total=total, pilot_usd=pilot, d5_usd=rep)
    base.mkdir(parents=True, exist_ok=True)
    with open(base / "pilot_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(text) + "\n")
    H.write_json(base / "pilot_report.json", J)
    say(f"\nwrote {base / 'pilot_report.txt'} and pilot_report.json")
    return 0


# --------------------------------------------------------------------------
# compare: the D5 repeat (counts only)
# --------------------------------------------------------------------------
def cache_dir(paths, spec, dry):
    if ":" not in spec:
        raise Fail(f"'{spec}': give stage and model, for example pilot:gemini or smoke/pilot-repeat:gemini")
    label, model = spec.rsplit(":", 1)
    base = paths.root / "outputs" / "dryrun" if dry else paths.root / "outputs"
    d = base / label / model
    if not (d / "cache.jsonl").exists():
        raise Fail(f"no cache at {d}")
    return d


def cmd_compare(a):
    if not a.d5_approved:
        raise Fail("compare belongs to the D5 repeat: it runs only with --d5-approved, after the decision row.")
    paths = Paths(a.root, a.repo, a.english_file)
    ex_rows, _cols = read_excluded(paths.excluded)
    excluded = {r["qa_id"] for r in ex_rows}       # every count in the paper applies the list (D110)
    da, db = cache_dir(paths, a.a, a.dry_run), cache_dir(paths, a.b, a.dry_run)
    A, B = H.Cache(da / "cache.jsonl").records, H.Cache(db / "cache.jsonl").records
    conds = set(a.conditions.split(",")) if a.conditions else None
    index = {}
    for r in B.values():
        index.setdefault((r["qa_id"], r.get("prompt_sha256"), r.get("image_sha256")), r)
    per, gaps = defaultdict(Counter), []
    for r in A.values():
        if conds and r["condition_id"] not in conds:
            continue
        c = per[r["condition_id"]]
        if r["qa_id"] in excluded:
            c["listed"] += 1
            continue
        rb = index.get((r["qa_id"], r.get("prompt_sha256"), r.get("image_sha256")))
        if rb is None:
            c["no_partner"] += 1
            continue
        c["pairs"] += 1
        c["same_settings"] += rb.get("settings_sha256") == r.get("settings_sha256")
        c["same_mode"] += rb.get("mode") == r.get("mode")
        x, y = normalise(r.get(H.ANSWER_FIELD) or ""), normalise(rb.get(H.ANSWER_FIELD) or "")
        if x == y:
            c["identical"] += 1
            c["both_empty"] += not x
        elif not x or not y:
            c["one_empty"] += 1
        else:
            c["different"] += 1
        ta, tb = parse_ts(r.get("ts_utc")), parse_ts(rb.get("ts_utc"))
        if ta and tb:
            gaps.append(abs((ta - tb).total_seconds()) / 3600)
    say("=" * 78)
    say(f"D5 compare (counts only): A = {a.a}, B = {a.b}")
    say("Pairs are the same request: same qa_id, same prompt text, same image. Answers compared after normalise().")
    ex_sha = H.file_sha256(paths.excluded)
    say(f"Items on probes/excluded_items.csv ({len(ex_rows)} rows, sha256 {ex_sha[:12]}) are left out of every "
        "count (D110); column 'listed'.")
    say("=" * 78)
    say(f"  {'condition':<9}{'pairs':>7}{'identical':>11}{'different':>11}{'one empty':>11}{'no partner':>12}"
        f"{'same settings':>15}{'listed':>8}")
    tot = Counter()
    for cid in sorted(per):
        c = per[cid]
        tot.update(c)
        say(f"  {cid:<9}{c['pairs']:>7}{c['identical']:>11}{c['different']:>11}{c['one_empty']:>11}"
            f"{c['no_partner']:>12}{c['same_settings']:>15}{c['listed']:>8}")
    lo, hi = wilson(tot["identical"], tot["pairs"])
    say(f"  all: {tot['identical']} of {tot['pairs']} answers identical "
        f"({share(tot['identical'], tot['pairs']).strip()}, Wilson 95% CI {100 * lo:.1f}% to {100 * hi:.1f}%); "
        f"{tot['different']} different, {tot['one_empty']} with one side empty; requests of A without a partner "
        f"in B: {tot['no_partner']}; listed items left out: {tot['listed']}")
    if gaps:
        say(f"  time between the two answers of a pair: median {statistics.median(gaps):.2f} h")
    res = {"utc": utc_now(), "script_version": SCRIPT_VERSION, "a": a.a, "b": a.b,
           "excluded_rows": len(ex_rows), "excluded_sha256": ex_sha,
           "per_condition": {k: dict(v) for k, v in per.items()},
           "total": dict(tot), "identical_wilson95": [lo, hi],
           "median_hours_apart": statistics.median(gaps) if gaps else None}
    slug = lambda s: re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")
    out = (paths.dry_out if a.dry_run else paths.pilot_out) / f"d5_compare_{slug(a.a)}_vs_{slug(a.b)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    H.write_json(out, res)
    say(f"wrote {out}")
    return 0


# --------------------------------------------------------------------------
# exclusions: D160 and D161 (question text and gold only, never a model answer, D110)
# --------------------------------------------------------------------------
def strip_zw(text):
    return text.replace("‌", "").replace("‍", "")


def scored_sizes(probes, listed):
    out = {}
    for name, items in probes.items():
        kept = [r for r in items if r["qa_id"] not in listed]
        out[name] = {"run": len(items), "scored": len(kept), "by_type": dict(Counter(r["qa_type"] for r in kept))}
    return out


def cmd_exclusions(a):
    paths = Paths(a.root, a.repo, a.english_file)
    P, cfg, _models = load_configs(paths)
    rows, cols = read_excluded(paths.excluded)
    if not excluded_prefix_ok(paths.excluded):
        raise Fail(f"{paths.excluded} does not start with the {EXPECTED['excluded_rows']} rows this chat started from. "
                   "Pull the repo; never edit a listed row. Send this output to the Pilot chat.")
    listed = {r["qa_id"] for r in rows}
    probes, member, info = {}, defaultdict(list), {}
    for name in SCORED_PROBES:
        p = paths.root / "probes" / f"{name}.json"
        items = H.load_probe(p)["items"]
        want = (cfg["probes"].get(name) or {}).get("expected_n")
        if len(items) != want:
            raise Fail(f"{p.name} has {len(items)} items, configs/conditions.yaml says {want}")
        probes[name] = items
        for r in items:
            member[r["qa_id"]].append(name)
            info.setdefault(r["qa_id"], r)
    pilot_ids = {r["qa_id"] for r in H.load_probe(paths.probe)["items"]}
    nameref_path = paths.repo / "probes" / "NAME-REF.ids.txt"
    nameref = set(read_ids(nameref_path)) if nameref_path.exists() else set()
    word_raw, w_reason, w_source = EXCLUSION_STEP["word"]
    word = strip_zw(normalise(word_raw))           # the same normalise() as the questions below
    root = strip_zw(normalise(EXCLUSION_STEP["root"]))
    new = []
    for qa, reason, source in EXCLUSION_STEP["gold"]:
        if qa not in info:
            raise Fail(f"{qa} is in none of the scored probe sets {SCORED_PROBES}")
        new.append((qa, reason, source))
    question = {q: strip_zw(normalise(r["question_bn"])) for q, r in info.items()}
    hits = sorted(q for q, t in question.items() if word in t)
    if EXCLUSION_STEP["known_hit"] not in hits:
        raise Fail(f"the word scan did not find {EXCLUSION_STEP['known_hit']}, which holds the word. Nothing was "
                   "written. Send this output to the Pilot chat.")
    new += [(q, w_reason, w_source) for q in hits if q not in {n[0] for n in new}]
    others = sorted(q for q, t in question.items() if root in t and q not in hits)
    say("=" * 78)
    say(f"Exclusions of the Pilot chat (D160, D161), {SCRIPT_VERSION}, {utc_now()}")
    say("Read from the question text and the gold only, never from a model answer (D110). No text is printed.")
    say(f"Scanned the scored probe sets {', '.join(f'{n} {len(probes[n])}' for n in SCORED_PROBES)}; "
        f"current list {len(rows)} rows.")
    say("=" * 78)
    say(f"  {'qa_id':<14}{'probes':<30}{'qa_type':<14}{'capture':<10}{'house':<7}{'reason':<12}notes")
    for qa, reason, _source in new:
        r = info[qa]
        notes = [x for x, on in (("PILOT", qa in pilot_ids), ("NAME-REF", qa in nameref),
                                 ("already listed", qa in listed)) if on]
        say(f"  {qa:<14}{'|'.join(member[qa]):<30}{r['qa_type']:<14}{r['capture_type']:<10}{r['house_id']:<7}"
            f"{reason:<12}{', '.join(notes)}")
    to_add = [n for n in new if n[0] not in listed]
    by_probe = Counter(p for qa, _r, _s in to_add for p in member[qa])
    houses = Counter(info[qa]["house_id"] for qa, _r, _s in to_add)
    say(f"\n  to add: {len(to_add)} rows (gold slip {sum(1 for n in to_add if n[1] != w_reason)}, "
        f"word for 'marked' {sum(1 for n in to_add if n[1] == w_reason)}); by probe set: "
        + (", ".join(f"{k} {v}" for k, v in by_probe.items()) or "none")
        + f"; houses: {', '.join(f'{k} {v}' for k, v in sorted(houses.items())) or 'none'}")
    if any(qa in nameref for qa, _r, _s in to_add):
        say("  NOTE: a NAME-REF item is on the list. The HT-3 scores read the list again (D208), so their item "
            "count drops below 200; say so in the Methods.")
    if others:
        shown = ", ".join(f"{q} ({info[q]['qa_type']})" for q in others)
        say("  other forms of the root word (NOT added; look at them and say in the Pilot chat whether any is the "
            f"same fault): {shown}")
    before = scored_sizes(probes, listed)
    after = scored_sizes(probes, listed | {n[0] for n in to_add})
    say("\n  scored sizes (run, scored now, scored after):")
    for name in SCORED_PROBES:
        b, c = before[name], after[name]
        say(f"    {name:<14} run {b['run']:>5}   now {b['scored']:>5}   after {c['scored']:>5}")
    for name in ("P-CORE-CLOSED", "R-TEST"):
        say(f"    {name} after, by type: " + ", ".join(f"{t} {after[name]['by_type'].get(t, 0)}" for t in CLOSED_TYPES))
    if not a.write:
        say("\nNothing was written. If the list is right, run this again with --write (WRITE_EXCLUSIONS = True in "
            "the notebook), then commit with Cell 9.")
        return 0
    if not to_add:
        say("\nNothing to add: every row is on the list already.")
        return 0
    with open(paths.excluded, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        for qa, reason, source in to_add:
            w.writerow([qa, "|".join(member[qa]), info[qa]["qa_type"], reason, source, EXCLUSION_STEP["added_at"]])
    rows2, _cols = read_excluded(paths.excluded)
    ids2 = [r["qa_id"] for r in rows2]
    if not excluded_prefix_ok(paths.excluded) or len(rows2) != len(rows) + len(to_add) or len(set(ids2)) != len(ids2):
        raise Fail("the list did not come out as planned after the append. Do not commit it; run git checkout "
                   "probes/excluded_items.csv and send this output to the Pilot chat.")
    say(f"\nWROTE {len(to_add)} rows: {paths.excluded} now has {len(rows2)} rows, sha256 "
        f"{H.file_sha256(paths.excluded)}. Commit and push it with Cell 9 (message: "
        "'Pilot: excluded items, D160 and D161').")
    return 0


# --------------------------------------------------------------------------
# The Pilot reading (DESIGN_ADDENDUM B10, F5, H5; D163, D190; D164 to D168)
# --------------------------------------------------------------------------
FC_CONDS = ("N-FCB", "N-FCB0", "N-FCE", "N-FCE0")
PILOT_CONDS = tuple(SCOPE) + FC_CONDS            # the 13 pilot conditions
NOIMAGE_CONDS = ("L-BB0", "L-EE0")
D5_CONDS = ("L-BB", "N-BB")
GATE_C_MIN = 0.98      # Design 11, Gate C
GATE_D_MIN = 0.90      # Design 11, Gate D: 8 words or fewer
FLOOR_MIN = 0.20       # Design 5.4
FLOOR_COND = "L-EE"
FLOOR_N = 58           # D163, H5: the scored closed items under L-EE
# 40 image calls of about 1,160 input tokens and 40 text calls of about 90, at the standard price of
# 0.75 USD per 1M input tokens; about 2 output tokens per call at 3.75 USD per 1M (pilot measurements, F1).
FC_USD_ESTIMATE = 0.04
SAMPLE_SALT = "PILOT-READING"
SCORE_SOURCES = (      # D165: one score command per cache. Name, stage label, model key.
    ("pilot_gemini", "pilot", "gemini"),
    ("pilot_qwen", "pilot", "qwen"),
    ("pilot_gemma", "pilot", "gemma"),
    ("d5repeat_gemini", "smoke/pilot-repeat", "gemini"),
    ("batchtest_gemini", "smoke/batchtest", "gemini"),
)
KEEP_FIELDS = ("prompt_sha256", "image_sha256", "settings_sha256", "mode", "ts_utc",
               "model_version", "model_revision")
PROOFREAD_KEY = ("human", "ASSET-9", "proofread", "proofread_key.csv")
_OPTION_LINE = re.compile(r"^\s*(\S{1,3})\)\s*(.*?)\s*$")


def _scorer():
    """The repo's scorer and option-file reader, imported only when a command needs them."""
    try:
        import fc_options
        import scoring
    except ImportError as exc:
        raise Fail(f"cannot import src/scoring.py or src/fc_options.py ({exc}); pull the repo first")
    return scoring, fc_options


def file_sha(path):
    """SHA-256 of the raw bytes, as the scorer's manifest records it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fc_item_sets(paths, items):
    """D156, D164: the items of each forced-choice condition, from PILOT.json and P-FCE.json."""
    _closed, naming, pclust = split_items(items)
    pfce = {r["qa_id"] for r in H.load_probe(paths.root / "probes" / "P-FCE.json")["items"]}
    fce = [r for r in naming if r["qa_id"] in pfce]
    return {"N-FCB": pclust, "N-FCB0": pclust, "N-FCE": fce, "N-FCE0": fce}, pfce


def expected_cells(paths, items):
    """{condition: [qa_id, ...]} for the 13 pilot conditions."""
    closed, naming, _pclust = split_items(items)
    cells = {cid: [r["qa_id"] for r in (closed if spec[4] == "closed" else naming)] for cid, spec in SCOPE.items()}
    fc, _pfce = fc_item_sets(paths, items)
    cells.update({cid: [r["qa_id"] for r in sel] for cid, sel in fc.items()})
    return cells


def cache_records(path):
    """The records of one harness cache, the first record per key (D143), read by the harness."""
    path = Path(path)
    return list(H.Cache(path).records.values()) if path.exists() else None


def cells_of(recs):
    out = {}
    for r in recs:
        out.setdefault((r["condition_id"], r["qa_id"]), r)
    return out


def pct(k, n):
    return f"{100.0 * k / n:.1f}%" if n else "-"


def ci_text(k, n):
    lo, hi = wilson(k, n)
    return f"Wilson 95% CI {100 * lo:.1f}% to {100 * hi:.1f}%"


def write_rows(path, rows, columns):
    """A CSV for Drive (UTF-8 with BOM, so that Excel shows Bangla). Lists become JSON text."""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: (json.dumps(r.get(c), ensure_ascii=False) if isinstance(r.get(c), (list, dict))
                            else ("" if r.get(c) is None else r.get(c))) for c in columns})


def verify_fc_prompts(paths, base, cells_fc):
    """D164. For the forced-choice records of the three models under `base`: the prompt that the harness's own
    build_work makes from the option file must have the hash stored in the record, and it must show the
    four options the scorer reads (src/fc_options.py), in the order of the file, under the labels of the
    question language. Returns (ok, counts, problems); ok is None when a harness helper could not be called."""
    S, FCm = _scorer()
    P, cfg, models = load_configs(paths)
    opt_path = paths.root / "probes" / "fc_options.jsonl"
    table = FCm.read_option_sets(str(opt_path))
    english = H.load_english(paths.english, EXPECTED["english_column"])
    man = imglib.read_manifest(paths.images)
    try:
        hopts = H.load_options(str(opt_path))
    except Exception as exc:
        return None, Counter(), [f"harness.load_options({opt_path.name}) raised {type(exc).__name__}: {exc}"]
    # The hash comparison rests on one assumption: a cache record carries the prompt_sha256 that build_work
    # gives for that model. It is tested first, per model, on the 60 real L-BB answers of 2026-10-14.
    closed = split_items(H.load_probe(paths.probe)["items"])[0]
    for key in MODELS:
        lbb = cells_of(cache_records(paths.pilot_out / key / "cache.jsonl") or [])
        try:
            work = H.build_work(models[key]["model_id"], cfg["conditions"]["L-BB"], closed, P, english, None, man)
        except Exception as exc:
            return None, Counter(), [f"harness.build_work ({key}, L-BB) raised {type(exc).__name__}: {exc}"]
        same = sum(1 for w, it in zip(work, closed)
                   if (lbb.get(("L-BB", it["qa_id"])) or {}).get("prompt_sha256") == w.get("prompt_sha256"))
        if same != len(closed):
            return None, Counter(), [f"{key}: only {same} of {len(closed)} real L-BB records carry the prompt hash "
                                     "that harness.build_work gives, so the forced-choice hash check cannot be read"]
    labels = {"bn": [normalise(x) for x in getattr(FCm, "BN_LABELS", S.BN_LABELS)],
              "en": [normalise(x) for x in getattr(FCm, "EN_LABELS", S.EN_LABELS)]}
    counts, problems = Counter(), []
    for key in MODELS:
        have = cells_of(cache_records(base / key / "cache.jsonl") or [])
        for cid in FC_CONDS:
            cond = cfg["conditions"][cid]
            lang = cond["question_lang"]
            sel = cells_fc[cid]
            try:
                work = H.build_work(models[key]["model_id"], cond, sel, P, english, hopts, man)
            except Exception as exc:
                return None, counts, [f"harness.build_work ({key}, {cid}) raised {type(exc).__name__}: {exc}"]
            for w, it in zip(work, sel):
                qa = it["qa_id"]
                counts["prompts"] += 1
                rec = have.get((cid, qa))
                if rec is None:
                    problems.append(f"{key} {cid} {qa}: no record in the cache")
                    continue
                if not rec.get("prompt_sha256") or rec.get("prompt_sha256") != w.get("prompt_sha256"):
                    problems.append(f"{key} {cid} {qa}: the record's prompt hash is not the hash of the rebuilt prompt")
                    continue
                counts["hash_ok"] += 1
                if bool(w.get("has_image")) != bool(cond.get("image")) or bool(rec.get("has_image")) != bool(cond.get("image")):
                    problems.append(f"{key} {cid} {qa}: image use differs from the condition")
                    continue
                try:
                    opts, _ci = FCm.options_for(table, qa, lang)
                except Exception as exc:
                    problems.append(f"{key} {cid} {qa}: no {lang} option set ({type(exc).__name__})")
                    continue
                shown = [m.groups() for m in (_OPTION_LINE.match(x) for x in str(w.get("prompt_text", "")).split("\n")) if m]
                if len(shown) != 4:
                    problems.append(f"{key} {cid} {qa}: {len(shown)} option lines in the prompt, expected 4")
                elif any(normalise(lab) != labels[lang][i] for i, (lab, _t) in enumerate(shown)):
                    problems.append(f"{key} {cid} {qa}: the option labels are not those of the {lang} list")
                elif any(normalise(t) != normalise(opts[i]) for i, (_lab, t) in enumerate(shown)):
                    problems.append(f"{key} {cid} {qa}: the options shown differ from the option file (text or order)")
                else:
                    counts["order_ok"] += 1
    return not problems, counts, problems


def fc_coverage(base, cells_fc):
    """Every forced-choice item in each model's cache once, and nothing else (D156, D164)."""
    lines, ok = [], True
    for key in MODELS:
        recs = cache_records(base / key / "cache.jsonl")
        if recs is None:
            lines.append(f"  {key:<7}FAIL  no cache at {base / key}")
            ok = False
            continue
        have = cells_of(recs)
        parts, bad = [], []
        for cid in FC_CONDS:
            want = {r["qa_id"] for r in cells_fc[cid]}
            got = {q for (c, q) in have if c == cid}
            parts.append(f"{cid} {len(got & want)} of {len(want)}")
            if got != want:
                bad.append(f"{cid}: missing {sorted(want - got)[:3]}, not expected {sorted(got - want)[:3]}")
        ok = ok and not bad
        lines.append(f"  {key:<7}{'PASS' if not bad else 'FAIL'}  " + ", ".join(parts) + ("; " + "; ".join(bad) if bad else ""))
    return ok, lines


def cmd_fc_verify(a):
    paths = Paths(a.root, a.repo, a.english_file)
    base = paths.dry_out if a.dry_run else paths.pilot_out
    opt_sha = check_fc_file(paths.root / "probes" / "fc_options.jsonl")
    pfce_sha = check_pfce_file(paths.root / "probes" / "P-FCE.json")
    items = H.load_probe(paths.probe)["items"]
    cells_fc, _pfce = fc_item_sets(paths, items)
    say("=" * 78)
    say(f"fc-verify ({SCRIPT_VERSION}), {utc_now()}" + ("   DRY RUN CACHES (made-up answers)" if a.dry_run else ""))
    say(f"option file sha256 {opt_sha[:12]}; P-FCE.json sha256 {pfce_sha[:12]}; caches under {base}")
    say("=" * 78)
    say("[1] every forced-choice item in the cache once, and no other item (D156)")
    cov_ok, cov_lines = fc_coverage(base, cells_fc)
    for line in cov_lines:
        say(line)
    say("[2] each prompt, rebuilt by the harness from the option file, has the record's hash and shows the "
        "scorer's four options in the order of the file, under the labels of the question language")
    ok, counts, problems = verify_fc_prompts(paths, base, cells_fc)
    if ok is None:
        say(f"  SKIP  {problems[0]}")
        say("  The harness helper was not called as pilot.py expects. Send this output and src/harness.py to the "
            "Pilot reading chat.")
    else:
        say(f"  {'PASS' if ok else 'FAIL'}  {counts['order_ok']} of {counts['prompts']} prompts; the hash matched for "
            f"{counts['hash_ok']}")
        for p in problems[:15]:
            say(f"    - {p}")
        if len(problems) > 15:
            say(f"    ... and {len(problems) - 15} more")
    result = "PASS" if cov_ok and ok else "FAIL"
    H.write_json(base / "fc_verify.json", {
        "utc": utc_now(), "script_version": SCRIPT_VERSION, "dry_run": bool(a.dry_run), "result": result,
        "options_sha256": opt_sha, "pfce_sha256": pfce_sha, "coverage": cov_lines, "counts": dict(counts),
        "problems": problems})
    say(f"wrote {base / 'fc_verify.json'}")
    say(f"FC-VERIFY: {result}" + (": the billed and GPU forced-choice runs may start (Cells 8b, 8c)."
                                  if result == "PASS" and a.dry_run else ""))
    return 0 if result == "PASS" else 1


# ---- score: one score command per cache (D165) ---------------------------------------------------
def scored_dir(paths, a):
    d = Path(a.out_dir) if getattr(a, "out_dir", "") else paths.root / "scored" / "pilot"
    r = d.resolve()
    if r == paths.repo.resolve() or paths.repo.resolve() in r.parents:
        raise Fail(f"{d} is inside the Git repo; scored files hold answer text and stay in Drive (setup Part 7)")
    return d


def scorer_ready(paths):
    """The scorer files of addendum B9 with the 57 phrases of D194; returns (scoring, fc_options)."""
    S, FCm = _scorer()
    problems = []
    if S.SCORER_VERSION != EXPECTED["scorer_version"]:
        problems.append(f"scorer {S.SCORER_VERSION}, expected {EXPECTED['scorer_version']}")
    if len(S.REFUSAL_PHRASES) != EXPECTED["refusal_phrases"]:
        problems.append(f"{len(S.REFUSAL_PHRASES)} refusal phrases, expected {EXPECTED['refusal_phrases']}")
    for rel, key in (("src/scoring.py", "scoring_sha256"), ("scripts/asset9_scorer.py", "asset9_scorer_sha256")):
        if file_sha(paths.repo / rel) != EXPECTED[key]:
            problems.append(f"{rel} is not the delivered ASSET-9 stage 2 file (addendum B9)")
    if problems:
        raise Fail("; ".join(problems))
    return S, FCm


def same_scored_inputs(man_path, cache, paths, opt_path):
    """True when an existing scored file was made from exactly these inputs by this scorer."""
    m = H.read_json(man_path, {})
    inp = m.get("inputs") or {}
    outs = inp.get("outputs") or [{}]
    return (m.get("scorer_version") == EXPECTED["scorer_version"] and len(outs) == 1
            and outs[0].get("sha256") == file_sha(cache)
            and (inp.get("excluded") or {}).get("sha256") == file_sha(paths.excluded)
            and (inp.get("options") or {}).get("sha256") == file_sha(opt_path)
            and (inp.get("lexicon") or {}).get("sha256") == file_sha(paths.repo / "lexicon" / "lexicon.csv")
            and (inp.get("vocabulary") or {}).get("sha256") == file_sha(paths.repo / "lexicon" / "closed_vocab.csv"))


def cmd_score(a):
    paths = Paths(a.root, a.repo, a.english_file)
    out_dir = scored_dir(paths, a)
    S, _FCm = scorer_ready(paths)
    opt_path = paths.root / "probes" / "fc_options.jsonl"
    opt_sha = check_fc_file(opt_path)
    check_pfce_file(paths.root / "probes" / "P-FCE.json")
    if file_sha(paths.probe) != EXPECTED["pilot_sha256"]:
        raise Fail(f"{paths.probe} is not the PILOT.json of the pilot runs (sha256 {EXPECTED['pilot_sha256'][:12]})")
    items = H.load_probe(paths.probe)["items"]
    pilot_ids = {r["qa_id"] for r in items}
    closed, naming, _pclust = split_items(items)
    cells = expected_cells(paths, items)
    want_pilot = {(c, q) for c, qs in cells.items() for q in qs}
    want_d5 = {("L-BB", r["qa_id"]) for r in closed} | {("N-BB", r["qa_id"]) for r in naming}
    ex_rows, _cols = read_excluded(paths.excluded)
    ex_sha = file_sha(paths.excluded)
    say("=" * 78)
    say(f"pilot.py score ({SCRIPT_VERSION}), {utc_now()}: scorer {S.SCORER_VERSION}, one score command per cache (D165)")
    say(f"excluded list {len(ex_rows)} rows (sha256 {ex_sha[:12]}); option file sha256 {opt_sha[:12]}; out {out_dir}")
    say("=" * 78)
    plan = []
    for name, label, key in SCORE_SOURCES:
        cache = paths.root / "outputs" / label / key / "cache.jsonl"
        recs = cache_records(cache)
        if recs is None:
            raise Fail(f"no cache at {cache}")
        have = cells_of(recs)
        if len(have) != len(recs):
            raise Fail(f"{cache}: {len(recs)} records but {len(have)} different (condition, qa_id) pairs")
        if label == STAGE:
            missing, extra = sorted(want_pilot - set(have)), sorted(set(have) - want_pilot)
            if missing:
                n_fc = sum(1 for c, _q in missing if c in FC_CONDS)
                raise Fail(f"{name}: {len(missing)} pilot answers are missing"
                           + (f", {n_fc} of them forced choice: run Cells 8b and 8c first" if n_fc else "")
                           + f" (first: {missing[:3]})")
            if extra:
                raise Fail(f"{name}: {len(extra)} records are outside the 13 pilot conditions and their items "
                           f"(first: {extra[:3]}). Send this output to the Pilot reading chat.")
        else:
            per_qa = Counter(q for _c, q in have)
            if set(per_qa) != pilot_ids or any(v != 1 for v in per_qa.values()):
                raise Fail(f"{name}: expected each of the 100 PILOT items once, found {len(per_qa)} items in "
                           f"{len(have)} records")
            if label == "smoke/pilot-repeat" and set(have) != want_d5:
                raise Fail(f"{name}: the repeat is not L-BB on the 60 closed items and N-BB on the 40 naming items")
        keep = [k for k in KEEP_FIELDS if any(k in r for r in recs)]
        by = Counter(c for c, _q in have)
        say(f"  {name:<17}{len(recs):>5} records: " + ", ".join(f"{c} {n}" for c, n in sorted(by.items())))
        plan.append((name, label, key, cache, len(recs), keep))
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, label, key, cache, n_recs, keep in plan:
        out = out_dir / f"{name}_scored.jsonl"
        man_path = out_dir / f"{name}_scored_manifest.json"
        if out.exists() and man_path.exists() and same_scored_inputs(man_path, cache, paths, opt_path) \
                and not a.overwrite:
            say(f"\n{name}: already scored from these inputs by scorer {EXPECTED['scorer_version']}; kept {out.name}")
            m = H.read_json(man_path, {})
            results.append({"name": name, "out": str(out), "records": m.get("records"), "statuses": m.get("statuses"),
                            "sha256": file_sha(out), "kept": True})
            continue
        if out.exists() and not a.overwrite:
            raise Fail(f"{out} exists and was scored from other inputs. Bring this line to the Pilot reading chat; "
                       "re-scoring needs OVERWRITE = True in Cell 10.")
        argv = [sys.executable, str(paths.repo / "scripts" / "asset9_scorer.py"), "score",
                "--probe", str(paths.probe), "--outputs", str(cache), "--options-file", str(opt_path),
                "--out", str(out), "--sample", "0"]
        for k in keep:
            argv += ["--keep", k]
        say("\n" + "-" * 78)
        say(f"{name}: scripts/asset9_scorer.py score on {cache}; kept fields {', '.join(keep)}")
        say("-" * 78)
        rc, _text, secs = stream(argv, paths.repo)
        if rc != 0:
            raise Fail(f"the score command for {name} stopped with exit code {rc}; files scored before it are kept")
        m = H.read_json(man_path, {})
        inp = m.get("inputs") or {}
        problems = []
        if m.get("scorer_version") != EXPECTED["scorer_version"]:
            problems.append(f"scorer {m.get('scorer_version')}")
        if m.get("records") != n_recs:
            problems.append(f"{m.get('records')} scored records for {n_recs} cache records")
        bad_skip = {k: v for k, v in (m.get("skipped") or {}).items() if "same cache key" not in k}
        if bad_skip:
            problems.append(f"skipped {bad_skip}")
        if (inp.get("excluded") or {}).get("sha256") != ex_sha:
            problems.append("scored with another excluded list")
        if problems:
            raise Fail(f"{name}: " + "; ".join(problems) + ". Send this output to the Pilot reading chat.")
        results.append({"name": name, "out": str(out), "records": m.get("records"), "statuses": m.get("statuses"),
                        "sha256": file_sha(out), "seconds": secs, "kept": False})
    H.write_json(out_dir / "score_run.json", {
        "utc": utc_now(), "script_version": SCRIPT_VERSION, "scorer_version": EXPECTED["scorer_version"],
        "excluded": {"rows": len(ex_rows), "sha256": ex_sha}, "options_sha256": opt_sha,
        "pilot_sha256": file_sha(paths.probe), "results": results})
    say("\n" + "=" * 78)
    for r in results:
        st = r.get("statuses") or {}
        say(f"  {r['name']:<17}{r['records']:>5} records ({', '.join(f'{k} {v}' for k, v in sorted(st.items()))})"
            f"  sha256 {r['sha256'][:12]}" + ("  (kept)" if r.get("kept") else ""))
    say(f"wrote {out_dir / 'score_run.json'}")
    say("SCORING DONE: run Cell 11 (read).")
    return 0


# ---- read: the Pilot reading ---------------------------------------------------------------------
def load_scored(out_dir, name):
    path = out_dir / f"{name}_scored.jsonl"
    man_path = out_dir / f"{name}_scored_manifest.json"
    if not path.exists() or not man_path.exists():
        raise Fail(f"{path.name} or its manifest is not in {out_dir}: run Cell 10 (score) first")
    recs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return recs, H.read_json(man_path, {}), file_sha(path)


def gate_c_first_half(paths):
    """{model: {group: rerun_ok or None}} from pilot_runs.jsonl, the latest real run per group (D158)."""
    out = {}
    for key in MODELS:
        path = paths.pilot_out / key / "pilot_runs.jsonl"
        logs = H.read_jsonl(path)[0] if path.exists() else []
        latest = {}
        for rec in logs:
            if rec.get("stage") == STAGE and not rec.get("dry_run"):
                latest[rec["group"]] = rec
        out[key] = {g: (latest[g].get("rerun_ok") if g in latest else None) for g in SCOPE_GROUPS + FC_GROUPS}
    return out


def seeded_rank(text):
    return hashlib.sha256(f"42|{SAMPLE_SALT}|{text}".encode("utf-8")).hexdigest()


def pick_sample(kept):
    """D166: 10 scored pilot records for the check by Claude: the lowest seeded rank in each forced-choice
    condition (4), in each no-image condition (2), and in 4 different with-image open conditions."""
    pool = sorted(((seeded_rank(f"{key}|{r['condition_id']}|{r['qa_id']}"), key, r)
                   for key in MODELS for r in kept[key] if r["scoreable"]), key=lambda x: x[0])
    picked, used = [], set()
    for stratum, conds in (("forced choice", FC_CONDS), ("no image", NOIMAGE_CONDS)):
        for cid in conds:
            x = next((x for x in pool if x[2]["condition_id"] == cid), None)
            if x:
                picked.append((stratum, x[1], x[2]))
    for _rank, key, r in pool:
        cid = r["condition_id"]
        if cid in FC_CONDS or cid in NOIMAGE_CONDS or cid in used:
            continue
        used.add(cid)
        picked.append(("with image", key, r))
        if len(used) == 4:
            break
    return picked


SAMPLE_COLS = ("n", "stratum", "model", "condition_id", "qa_id", "qa_type", "kind", "requested_lang", "gold_norm",
               "options_shown", "answer_raw", "label", "strict", "lenient", "fidelity", "lang", "refusal",
               "refusal_phrase", "content_found", "matched", "named_ids", "gold_value", "found_value",
               "fc_choice", "fc_correct_index", "fc_method", "fc_valid", "fc_cross_label", "n_words")
REFUSAL_COLS = ("source", "model", "condition_id", "qa_id", "qa_type", "kind", "gold_norm", "answer_raw", "label",
                "strict", "lenient", "refusal", "refusal_phrase", "found_value", "proofread_code")
PAIR_COLS = ("comparison", "condition_a", "condition_b", "qa_id", "strict_a", "strict_b", "label_a", "label_b",
             "text_identical", "hours_apart")


def cmd_read(a):
    paths = Paths(a.root, a.repo, a.english_file)
    S, FCm = scorer_ready(paths)
    out_dir = scored_dir(paths, a)
    opt_path = paths.root / "probes" / "fc_options.jsonl"
    opt_sha = check_fc_file(opt_path)
    check_pfce_file(paths.root / "probes" / "P-FCE.json")
    table = FCm.read_option_sets(str(opt_path))
    items = H.load_probe(paths.probe)["items"]
    info = {r["qa_id"]: r for r in items}
    cells = expected_cells(paths, items)
    cells_fc, _pfce = fc_item_sets(paths, items)
    ex_rows, _cols = read_excluded(paths.excluded)
    excluded = {r["qa_id"] for r in ex_rows}
    ex_sha = file_sha(paths.excluded)
    listed_pilot = sorted(excluded & set(info))
    EX = S.EXCLUDED_STATUS
    text = []

    def out(s=""):
        say(s)
        text.append(s)

    # ---- the scored files and their inputs ----
    sc = {}
    for name, label, key in SCORE_SOURCES:
        recs, man, sha = load_scored(out_dir, name)
        inp = man.get("inputs") or {}
        outs = inp.get("outputs") or [{}]
        cache = paths.root / "outputs" / label / key / "cache.jsonl"
        problems = []
        if man.get("scorer_version") != EXPECTED["scorer_version"]:
            problems.append(f"scorer {man.get('scorer_version')}")
        if (inp.get("excluded") or {}).get("sha256") != ex_sha:
            problems.append("scored with another excluded list")
        if (inp.get("options") or {}).get("sha256") != opt_sha:
            problems.append("scored with another option file")
        if outs[0].get("sha256") != file_sha(cache):
            problems.append("the cache changed after scoring")
        if problems:
            raise Fail(f"{name}: " + "; ".join(problems) + ". Run Cell 10 again with OVERWRITE = True, then Cell 11.")
        sc[name] = {"recs": recs, "cells": cells_of(recs), "sha": sha, "label": label, "key": key}
    want = {(c, q) for c, qs in cells.items() for q in qs}
    selfcheck = []
    for key in MODELS:
        s = sc[f"pilot_{key}"]
        if set(s["cells"]) != want or len(s["recs"]) != len(want):
            selfcheck.append(f"pilot_{key}: the scored records are not the {len(want)} answers of the 13 conditions")
        wrong = [k for k, r in s["cells"].items() if (r["gold_status"] == EX) != (k[1] in excluded)]
        if wrong:
            selfcheck.append(f"pilot_{key}: {len(wrong)} records whose excluded status does not follow the list")
        bad = [k for k, r in s["cells"].items() if k[0] in FC_CONDS and r.get("fc_options_sha256") != opt_sha]
        if bad:
            selfcheck.append(f"pilot_{key}: {len(bad)} forced-choice records without the option file's sha256")
    if selfcheck:
        raise Fail("the scored files do not fit the pilot: " + "; ".join(selfcheck))
    kept = {key: [r for r in sc[f"pilot_{key}"]["recs"] if r["gold_status"] != EX] for key in MODELS}
    J = {"script_version": SCRIPT_VERSION, "utc": utc_now(), "scorer_version": S.SCORER_VERSION,
         "inputs": {"excluded": {"rows": len(ex_rows), "sha256": ex_sha, "pilot_items_listed": listed_pilot},
                    "options_sha256": opt_sha, "pilot_sha256": file_sha(paths.probe),
                    "scored": {n: v["sha"] for n, v in sc.items()}}}
    out("=" * 78)
    out(f"PILOT READING ({SCRIPT_VERSION}), {utc_now()}; scorer {S.SCORER_VERSION} ({len(S.REFUSAL_PHRASES)} refusal phrases)")
    out(f"scored files in {out_dir}; excluded list {len(ex_rows)} rows (sha256 {ex_sha[:12]}); PILOT items "
        f"listed: {', '.join(listed_pilot)}")
    out("Counts and ids only. Every number leaves out the listed items (D110, D185). Text is in the CSV files.")
    out("=" * 78)
    for key in MODELS:
        out(f"  {key:<7}{len(sc[f'pilot_{key}']['recs'])} pilot answers, {len(kept[key])} outside the list")

    # [1] Gate C, first half
    out("\n[1] Gate C, first half (D158): every pilot command ran a second time with 0 calls")
    gc1 = gate_c_first_half(paths)
    gc1_ok = {}
    for key in MODELS:
        st = gc1[key]
        gc1_ok[key] = all(v is True for v in st.values())
        out(f"  {key:<7}{'PASS' if gc1_ok[key] else 'FAIL':<6}" + ", ".join(
            f"{g} {'ok' if v is True else ('not run' if v is None else 'FAILED')}" for g, v in st.items()))
    J["gate_c_first_half"] = {"per_model": gc1, "ok": gc1_ok}

    # [2] Gate C, second half
    out("\n[2] Gate C, second half (D163, D190): open answers not empty; forced-choice answers not empty and a "
        "valid label, one option named, or a refusal; per model, outside the list")
    gc2 = {}
    for key in MODELS:
        rs = kept[key]
        op = [r for r in rs if r["task"] == "open"]
        op_apart = [r for r in op if r["gold_status"] != "ok"]
        op_base = [r for r in op if r["gold_status"] == "ok"]
        op_ok = sum(1 for r in op_base if not r["is_empty"])
        no_content = sum(1 for r in op_base if r.get("content_found") is False)
        fc = [r for r in rs if r["task"] == "fc"]
        fc_valid = sum(1 for r in fc if not r["is_empty"] and r["fc_valid"])
        fc_ref = sum(1 for r in fc if not r["is_empty"] and not r["fc_valid"] and r["refusal"])
        fc_empty = sum(1 for r in fc if r["is_empty"])
        fc_cross = sum(1 for r in fc if not r["is_empty"] and not r["fc_valid"] and not r["refusal"]
                       and r.get("fc_cross_label"))
        fc_none = sum(1 for r in fc if not r["is_empty"] and not r["fc_valid"] and not r["refusal"]
                      and not r.get("fc_cross_label"))
        n, k = len(op_base) + len(fc), op_ok + fc_valid + fc_ref
        ok = n > 0 and k / n >= GATE_C_MIN
        failing = Counter(r["condition_id"] for r in op_base if r["is_empty"])
        failing.update(r["condition_id"] for r in fc if r["is_empty"] or (not r["fc_valid"] and not r["refusal"]))
        gc2[key] = {"ok": ok, "k": k, "n": n, "open_ok": op_ok, "open_n": len(op_base), "open_no_content": no_content,
                    "open_unscoreable_apart": len(op_apart), "fc_n": len(fc), "fc_valid": fc_valid,
                    "fc_refusal": fc_ref, "fc_empty": fc_empty, "fc_other_script_or_digit": fc_cross,
                    "fc_no_single_option": fc_none, "not_counted_by_condition": dict(failing)}
        out(f"  {key:<7}{'PASS' if ok else 'FAIL':<6}{k} of {n} = {pct(k, n)} (needs {GATE_C_MIN:.0%})")
        out(f"         open: {op_ok} of {len(op_base)} not empty; {no_content} hold no scoreable content "
            f"(information, D190); unscoreable gold, counted apart: {len(op_apart)}")
        out(f"         forced choice: {fc_valid + fc_ref} of {len(fc)} parseable (label or option text {fc_valid}, "
            f"refusal {fc_ref}); not parseable: empty {fc_empty}, other-script label or digit {fc_cross}, no single "
            f"option {fc_none}")
        if failing:
            out("         not counted, by condition: " + ", ".join(f"{c} {v}" for c, v in sorted(failing.items())))
    J["gate_c_second_half"] = gc2

    # [3] Gate D
    out("\n[3] Gate D (D163): 8 words or fewer over all of the model's pilot answers, forced choice included; "
        "floor: strict accuracy on L-EE over the scored closed items (Design 5.4 applies it to the open models)")
    gd = {}
    for key in MODELS:
        rs = kept[key]
        short = sum(1 for r in rs if (r.get("n_words") or 0) <= 8)
        ok8 = bool(rs) and short / len(rs) >= GATE_D_MIN
        lee = [r for r in rs if r["condition_id"] == FLOOR_COND and r["scoreable"]]
        kf, nf = sum(1 for r in lee if r["strict"] == 1), len(lee)
        okf = nf > 0 and kf / nf >= FLOOR_MIN
        applies = key in OPEN_MODELS
        long_by = Counter(r["condition_id"] for r in rs if (r.get("n_words") or 0) > 8)
        gd[key] = {"ok": ok8 and (okf or not applies), "short": short, "n": len(rs), "ok_8_words": ok8,
                   "floor_k": kf, "floor_n": nf, "floor_ok": okf, "floor_applies": applies,
                   "floor_ci95": list(wilson(kf, nf)), "over_8_by_condition": dict(long_by)}
        out(f"  {key:<7}8-word rule {'PASS' if ok8 else 'FAIL'}: {short} of {len(rs)} = {pct(short, len(rs))} "
            f"(needs {GATE_D_MIN:.0%})" + (f"; over 8 words: " + ", ".join(f"{c} {v}" for c, v in sorted(long_by.items()))
                                          if long_by else ""))
        verdict = ("PASS" if okf else "FAIL") if applies else ("above 20%" if okf else "below 20%") + ", information"
        out(f"         floor {verdict}: L-EE strict {kf} of {nf} = {pct(kf, nf)}, {ci_text(kf, nf)}"
            + ("" if nf == FLOOR_N else f"  (D163 expects {FLOOR_N} items)"))
    target = next((k for k in OPEN_MODELS if gd[k]["ok"]), None)
    J["gate_d"] = gd
    J["exp_r_target"] = target

    # [4] D5
    out("\n[4] D5 repeat (D159, D167): the same requests to Gemini (L-BB, N-BB), paired by qa_id, prompt and "
        "image hash, outside the list")
    a_cache = cache_records(paths.pilot_out / "gemini" / "cache.jsonl")
    pair_rows, d5 = [], {}
    for bname, title in (("d5repeat_gemini", "pilot against the repeat"),
                         ("batchtest_gemini", "pilot against the Batch test of 2026-10-05")):
        b_cache = cache_records(paths.root / "outputs" / sc[bname]["label"] / "gemini" / "cache.jsonl")
        index = {}
        for r in b_cache:
            index.setdefault((r["qa_id"], r.get("prompt_sha256"), r.get("image_sha256")), r)
        sa, sb = sc["pilot_gemini"]["cells"], sc[bname]["cells"]
        per, gaps = defaultdict(Counter), []
        for r in a_cache:
            cid = r["condition_id"]
            if cid not in D5_CONDS:
                continue
            c = per[cid]
            if r["qa_id"] in excluded:
                c["listed"] += 1
                continue
            rb = index.get((r["qa_id"], r.get("prompt_sha256"), r.get("image_sha256")))
            if rb is None:
                c["no_partner"] += 1
                continue
            xa, xb = sa.get((cid, r["qa_id"])), sb.get((rb["condition_id"], rb["qa_id"]))
            if not xa or not xb or not xa["scoreable"] or not xb["scoreable"]:
                c["not_scored"] += 1
                continue
            same_text = normalise(r.get(H.ANSWER_FIELD) or "") == normalise(rb.get(H.ANSWER_FIELD) or "")
            c["pairs"] += 1
            c["text_same"] += same_text
            c["strict_same"] += xa["strict"] == xb["strict"]
            c["label_same"] += xa["label"] == xb["label"]
            c["strict_a"] += xa["strict"] == 1
            c["strict_b"] += xb["strict"] == 1
            c["a1_b0"] += xa["strict"] == 1 and xb["strict"] == 0
            c["a0_b1"] += xa["strict"] == 0 and xb["strict"] == 1
            c["same_text_other_score"] += same_text and (xa["label"] != xb["label"] or xa["strict"] != xb["strict"])
            ta, tb = parse_ts(r.get("ts_utc")), parse_ts(rb.get("ts_utc"))
            hours = abs((ta - tb).total_seconds()) / 3600 if ta and tb else None
            if hours is not None:
                gaps.append(hours)
            pair_rows.append({"comparison": bname, "condition_a": cid, "condition_b": rb["condition_id"],
                              "qa_id": r["qa_id"], "strict_a": xa["strict"], "strict_b": xb["strict"],
                              "label_a": xa["label"], "label_b": xb["label"], "text_identical": same_text,
                              "hours_apart": None if hours is None else round(hours, 2)})
        tot = Counter()
        for v in per.values():
            tot.update(v)
        if tot["same_text_other_score"]:
            selfcheck.append(f"{bname}: {tot['same_text_other_score']} pairs with the same text but another score")
        med = statistics.median(gaps) if gaps else None
        d5[bname] = {"per_condition": {c: dict(v) for c, v in per.items()}, "total": dict(tot),
                     "median_hours_apart": med}
        n = tot["pairs"]

        def split(field):
            return ", ".join(f"{c} {per[c][field]} of {per[c]['pairs']}" for c in D5_CONDS if c in per)
        out(f"  {title}" + (f" (median {med:.2f} h apart)" if med is not None else "") + f": {n} pairs ("
            + ", ".join(f"{c} {per[c]['pairs']}" for c in D5_CONDS if c in per) + f"); listed and left out "
            f"{tot['listed']}; without a partner {tot['no_partner']}; not scoreable {tot['not_scored']}")
        for field, what in (("text_same", "identical answer text"), ("strict_same", "same strict score"),
                            ("label_same", "same scorer label")):
            out(f"    {what:<22}{tot[field]} of {n} = {pct(tot[field], n)}, {ci_text(tot[field], n)} ({split(field)})")
        out(f"    strict right: pilot {tot['strict_a']} of {n}, other run {tot['strict_b']} of {n}; changed right to "
            f"wrong {tot['a1_b0']}, wrong to right {tot['a0_b1']}")
    J["d5"] = d5

    # [5] refusal without the image
    out("\n[5] Without the image: refusal rate (D106; scorer 0.2.1-stage2, 57 phrases) per model, strict "
        "accuracy beside it")
    ref = {}
    for key in MODELS:
        for cid in NOIMAGE_CONDS:
            rs = [r for r in kept[key] if r["condition_id"] == cid and r["scoreable"]]
            n = len(rs)
            k_ref = sum(1 for r in rs if r["refusal"])
            k_st = sum(1 for r in rs if r["strict"] == 1)
            k_ph = sum(1 for r in rs if r.get("refusal_phrase"))
            k_nc = sum(1 for r in rs if r.get("content_found") is False)
            ref[f"{key}|{cid}"] = {"n": n, "refusal": k_ref, "strict": k_st, "refusal_phrase": k_ph, "no_content": k_nc}
            out(f"  {key:<7}{cid:<7}refusal {k_ref} of {n} = {pct(k_ref, n)}; strict {k_st} of {n} = {pct(k_st, n)}; "
                f"a refusal phrase anywhere {k_ph}; no scoreable content {k_nc}")
    J["refusal_no_image"] = ref

    # [6] correct answers that hold a refusal phrase
    out("\n[6] Answers scored right that hold a refusal phrase (D194: counted, never re-scored)")
    codes = {}
    key_path = paths.root.joinpath(*PROOFREAD_KEY)
    if key_path.exists():
        with open(key_path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                for rid in (row.get("records") or "").split():
                    codes[rid] = row.get("code")
    rc_rows, rc_count = [], {}
    for name, label, key in SCORE_SOURCES:
        hits = [r for r in sc[name]["recs"] if r["gold_status"] != EX and r.get("refusal_phrase")
                and r.get("lenient") == 1]
        cnt = Counter(r["condition_id"] for r in hits)
        cnt_strict = Counter(r["condition_id"] for r in hits if r.get("strict") == 1)
        rc_count[name] = {"lenient": len(hits), "strict": sum(cnt_strict.values()),
                          "by_condition": {c: [cnt[c], cnt_strict[c]] for c in cnt}}
        for r in hits:
            rid = f"{r.get('model_key') or key}:{r['condition_id']}:{r['qa_id']}"
            rc_rows.append(dict(r, source=name, model=key, proofread_code=codes.get(rid, "")))
        out(f"  {name:<17}{len(hits)} right on content (lenient 1), {sum(cnt_strict.values())} of them strict 1"
            + (": " + ", ".join(f"{c} {cnt[c]} (strict {cnt_strict[c]})" for c in sorted(cnt)) if cnt else ""))
    p118 = sorted(rid for rid, code in codes.items() if code == "P118")
    if not key_path.exists():
        out("  P118: the proofread key is not in Drive human/ASSET-9/proofread/; not checked")
        p118_ok = None
    else:
        found = {f"{r.get('model_key') or r['model']}:{r['condition_id']}:{r['qa_id']}" for r in rc_rows
                 if r["source"].startswith("pilot_")}
        hit = [rid for rid in p118 if rid in found]
        p118_ok = bool(hit)
        out(f"  P118 (record{'s' if len(p118) != 1 else ''} {', '.join(p118) or 'none in the key'}): "
            + (f"{len(hit)} of {len(p118)} counted above, as D194 expects" if p118_ok
               else "NOT among the answers counted above"))
        if not p118_ok:
            selfcheck.append("P118 is not among the correct answers that hold a refusal phrase")
    J["refusal_phrase_correct"] = {"counts": rc_count, "p118": p118, "p118_found": p118_ok}

    # [7] forced choice
    out("\n[7] Forced choice (information; chance is 25%)")
    fcj = {}
    for key in MODELS:
        for cid in FC_CONDS:
            rs = [r for r in kept[key] if r["condition_id"] == cid]
            n = len(rs)
            right = sum(1 for r in rs if r["strict"] == 1)
            by_label = sum(1 for r in rs if r.get("fc_method") == "label")
            by_text = sum(1 for r in rs if r.get("fc_method") == "text")
            inv = sum(1 for r in rs if r["label"] == S.INVALID)
            fcj[f"{key}|{cid}"] = {"n": n, "right": right, "by_label": by_label, "by_text": by_text, "invalid": inv}
            out(f"  {key:<7}{cid:<7}right {right} of {n} = {pct(right, n)}; read from a label {by_label}, from the "
                f"option text {by_text}; invalid {inv}")
    J["forced_choice"] = fcj

    # [8] the prompts the models saw
    out("\n[8] Forced-choice prompts against the option file (D164), the real caches of the three models")
    cov_ok, cov_lines = fc_coverage(paths.pilot_out, cells_fc)
    for line in cov_lines:
        out(line)
    ok_p, counts, problems = verify_fc_prompts(paths, paths.pilot_out, cells_fc)
    if ok_p is None:
        out(f"  SKIP  {problems[0]}")
    else:
        out(f"  {'PASS' if ok_p else 'FAIL'}  {counts['order_ok']} of {counts['prompts']} prompts show the scorer's "
            f"options in the order of the file; the hash matched for {counts['hash_ok']}")
        for p in problems[:10]:
            out(f"    - {p}")
    if not cov_ok or ok_p is False:
        selfcheck.append("forced-choice prompts or items")
    J["fc_prompts"] = {"coverage_ok": cov_ok, "ok": ok_p, "counts": dict(counts), "problems": problems[:50]}

    # [9] the 10-item check
    out("\n[9] The 10-item check (D166): ids here, text in pilot_reading_sample.csv")
    sample_rows = []
    for i, (stratum, key, r) in enumerate(pick_sample(kept), 1):
        row = dict(r, n=i, stratum=stratum, model=key)
        if r["condition_id"] in FC_CONDS:
            lang = S.condition_spec(r["condition_id"]).question_lang
            opts, _ci = FCm.options_for(table, r["qa_id"], lang)
            labs = getattr(FCm, "BN_LABELS" if lang == "bn" else "EN_LABELS",
                           S.BN_LABELS if lang == "bn" else S.EN_LABELS)
            row["options_shown"] = " | ".join(f"{labs[j]}) {o}" for j, o in enumerate(opts))
        sample_rows.append(row)
        out(f"  {i:>2}. {stratum:<14}{key:<7}{r['condition_id']:<7}{r['qa_id']}")

    # verdict
    out("\n" + "=" * 78)
    out("VERDICT (per model, outside the excluded list)")
    verdict = {}
    for key in MODELS:
        c_ok = gc1_ok[key] and gc2[key]["ok"]
        d_ok = gd[key]["ok"]
        verdict[key] = {"gate_c": c_ok, "gate_d": d_ok}
        out(f"  {key:<7}Gate C {'PASS' if c_ok else 'FAIL'} (first half {'PASS' if gc1_ok[key] else 'FAIL'}, "
            f"second half {pct(gc2[key]['k'], gc2[key]['n'])})   Gate D {'PASS' if d_ok else 'FAIL'} "
            f"(8 words or fewer {pct(gd[key]['short'], gd[key]['n'])}; L-EE strict "
            f"{pct(gd[key]['floor_k'], gd[key]['floor_n'])}" + ("" if gd[key]["floor_applies"] else ", information") + ")")
    out(f"  fine-tuning target by Design 5.4 (the lighter model above the floor, Qwen first): "
        + ({"qwen": "Qwen3.5-4B", "gemma": "Gemma 4 E4B"}[target] if target else "NONE: bring this to the chat"))
    if selfcheck:
        out("  SELF-CHECK FAILED: " + "; ".join(selfcheck) + ". The reading is not valid; send this output.")
    else:
        out("  self-checks: all passed")
    J["verdict"] = verdict
    J["selfcheck"] = selfcheck
    out("=" * 78)

    # files (Drive)
    write_rows(out_dir / "pilot_reading_sample.csv", sample_rows, SAMPLE_COLS)
    write_rows(out_dir / "pilot_reading_refusal_correct.csv", rc_rows, REFUSAL_COLS)
    write_rows(out_dir / "pilot_reading_d5_pairs.csv", pair_rows, PAIR_COLS)
    J["summaries"] = {name: S.summarise(v["recs"], by=("condition_id",)) for name, v in sc.items()}
    H.write_json(out_dir / "pilot_reading.json", J)
    with open(out_dir / "pilot_reading.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(text) + "\n")
    say(f"\nwrote in {out_dir}: pilot_reading.txt and .json, pilot_reading_sample.csv ({len(sample_rows)} rows), "
        f"pilot_reading_refusal_correct.csv ({len(rc_rows)} rows), pilot_reading_d5_pairs.csv ({len(pair_rows)} rows)")
    return 1 if selfcheck else 0


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="The pilot (Design 11) on top of the ASSET-8 harness.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--root", default=DEFAULT_ROOT, help="the Drive folder of the project")
        p.add_argument("--repo", default=str(REPO), help="the repo root")
        p.add_argument("--english-file", default="", help="default <root>/english/english_questions_final_PILOT.csv")

    p = sub.add_parser("check", help="one PASS or FAIL line per input")
    common(p)
    p.add_argument("--session", choices=("gemini", "open", "any"), default="any")
    p.add_argument("--design", default="", help="default <root>/docs/BanglaIndoorVQA_Design.md")
    p = sub.add_parser("preview", help="the English questions and one prompt per condition")
    common(p)
    p = sub.add_parser("run", help="one harness command for one model and group, then again (Gate C)")
    common(p)
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--group", required=True, choices=tuple(GROUPS))
    p.add_argument("--dry-run", action="store_true", help="no network, no GPU, no cost")
    p.add_argument("--dry-fresh", action="store_true", help="dry run: start this model's dry folder empty")
    p.add_argument("--paid-tier-confirmed", action="store_true")
    p.add_argument("--prices-checked", default=GOOGLE_CHECK["checked_utc"],
                   help="the day you last read Google's model and pricing pages (YYYY-MM-DD)")
    p.add_argument("--options-file", default="", help="forced choice: Drive probes/fc_options.jsonl (D164)")
    p.add_argument("--pfce", default="", help="N-FCE: Drive probes/P-FCE.json (D164)")
    p.add_argument("--d5-approved", action="store_true")
    p.add_argument("--plan-only", action="store_true", help="print the harness plan and stop")
    p = sub.add_parser("report", help="counts, seconds per item, tokens per call (counts only)")
    common(p)
    p.add_argument("--dry-run", action="store_true", help="read the dry-run outputs")
    p = sub.add_parser("compare", help="D5 only: two runs of the same requests")
    common(p)
    p.add_argument("--a", required=True, help="for example pilot:gemini")
    p.add_argument("--b", required=True, help="for example smoke/pilot-repeat:gemini")
    p.add_argument("--conditions", default="")
    p.add_argument("--d5-approved", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("exclusions", help="D160, D161: list the items to exclude; --write appends them")
    common(p)
    p.add_argument("--write", action="store_true", help="append the rows to probes/excluded_items.csv")
    p = sub.add_parser("fc-verify", help="D164: forced-choice items and prompts against the option file")
    common(p)
    p.add_argument("--dry-run", action="store_true", help="check the dry-run caches (Cell 8a)")
    p = sub.add_parser("score", help="D165: asset9_scorer.py score, once per cache, into Drive scored/pilot/")
    common(p)
    p.add_argument("--out-dir", default="", help="default <root>/scored/pilot")
    p.add_argument("--overwrite", action="store_true", help="score again although a scored file exists")
    p = sub.add_parser("read", help="the Pilot reading: Gates C and D, D5, refusals, forced choice, 10-item sample")
    common(p)
    p.add_argument("--out-dir", default="", help="the scored files; default <root>/scored/pilot")
    a = ap.parse_args(argv)
    fn = {"check": cmd_check, "preview": cmd_preview, "run": cmd_run, "report": cmd_report,
          "compare": cmd_compare, "exclusions": cmd_exclusions, "fc-verify": cmd_fc_verify,
          "score": cmd_score, "read": cmd_read}[a.cmd]
    try:
        return fn(a)
    except (Fail, H.Stop, promptlib.PromptError, imglib.ImageError, cost_ledger.BudgetExceeded,
            cost_ledger.LedgerError) as exc:
        say(f"\nSTOP: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
