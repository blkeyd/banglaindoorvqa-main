#!/usr/bin/env python3
"""
pilot.py
BanglaIndoorVQA, the Pilot (Design 11, 7.4, 5.4; DESIGN_ADDENDUM D9, D10, D15 and E;
decisions D155 to D159, DESIGN_ADDENDUM F).

The pilot runs the 100 PILOT items under the EXP-L and EXP-N conditions for the three
models, through the ASSET-8 harness (scripts/asset8_harness.py, src/harness.py). This
script adds no inference code. It
  - checks every input before the first run (Cell 0 of notebooks/pilot.ipynb),
  - runs harness commands, with the items of each condition taken from the fields of
    PILOT.json (D156), never from qa_id lists,
  - runs every finished command a second time and checks that the second run makes 0
    calls and changes neither the cache nor the cost ledger (the first half of Gate C),
  - counts what came out: answers, empty answers, answers of 8 words or fewer, seconds
    per item of the open models, tokens per Gemini call, projections against Design 12.
It never prints a model answer. Question text is printed only by 'preview'.

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

Item groups (D156)
    english3     L-EB, L-EE, L-EE0 on 3 closed items by seeded rank: the first English calls
    exp-l        L-BB, L-BE, L-EB, L-EE, L-BB0, L-EE0 on the 60 items with pilot_part = closed
    exp-n-open   N-BB, N-BE, N-BBs on the 40 items with pilot_part = naming
    exp-n-fcb    N-FCB, N-FCB0 on the 20 naming items with in_pclust = true (after ASSET-11, HT-6)
    exp-n-fce    N-FCE, N-FCE0 on the naming items that are in P-FCE (after ASSET-11, HT-6)
    d5-closed, d5-naming   the D5 repeat (stage smoke, run name pilot-repeat); only with
                 --d5-approved, after its decision row is accepted

Everything this script writes goes to Google Drive outputs/ (model output, D150).
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

SCRIPT_VERSION = "pilot-2026-10-14c"
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
MAIN_N = {"L-BB": 1118, "L-BE": 1118, "L-EB": 1118, "L-EE": 1118, "L-BB0": 1118, "L-EE0": 1118,
          "N-BB": 1000, "N-BE": 1000, "N-BBs": 1000, "N-FCB": 492, "N-FCB0": 492,
          "N-FCE": None, "N-FCE0": None}
PFCE_PLANNING = 300   # Design 7.2 planning value until ASSET-11 gives the P-FCE size (upper bound 471, C1)
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
    ok = room > 2.0 and pilot < 2.0
    note = ""
    if pilot == 0 and abs(total - EXPECTED["ledger_before_usd"]) > 0.00005:
        note = f"; PROJECT_STATE says {EXPECTED['ledger_before_usd']:.4f} USD so far: bring both numbers"
    L.add("PASS" if ok else "FAIL", "C13",
          f"API ledger: project total {total:.4f} USD of {cost_ledger.HARD_CAP_USD:.0f}; stage pilot "
          f"{pilot:.4f} of 2.00 USD; the 9 Gemini conditions are expected near 0.35 USD (estimate from the "
          f"smoke-run costs; the harness prints a ceiling){note}. Prepaid credit is not visible to the API: "
          "compare with the AI Studio billing page after the run")


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
             ("C13", check_ledger), ("C14", check_env), ("C15", check_hub), ("C16", check_outputs))
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


def write_pfce_ids(paths, src, dry):
    if not src:
        raise Fail("--pfce is needed: the P-FCE probe file (P-FCE.json) or its qa_id list from ASSET-11")
    p = Path(src)
    ids = [r["qa_id"] for r in H.load_probe(p)["items"]] if p.suffix == ".json" else read_ids(p)
    out = (paths.dry_out if dry else paths.pilot_out) / "pilot_pfce_qa_ids.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(ids) + "\n")
    say(f"P-FCE: {len(ids)} qa_ids from {p.name} written to {out}; the harness keeps the PILOT naming items "
        "among them")
    return out


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
            raise Fail("--options-file is needed: the ASSET-11 option sets in the D149 format")
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
        "EXP-N": ((3 * 1000 + 492 + PFCE_PLANNING) * wi + (492 + PFCE_PLANNING) * wn + 3 * su) / 3600,
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
                out(f"    (EXP-N uses P-FCE = {PFCE_PLANNING}, the Design 7.2 planning value, and the with-image "
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
                n_main = n_main if n_main is not None else PFCE_PLANNING
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
        required = SCOPE_GROUPS if not a.dry_run else ("exp-l", "exp-n-open")   # Cell 2 runs no english3
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
    p.add_argument("--options-file", default="", help="forced choice: the ASSET-11 option sets")
    p.add_argument("--pfce", default="", help="N-FCE: the P-FCE probe file or its qa_id list")
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
    a = ap.parse_args(argv)
    fn = {"check": cmd_check, "preview": cmd_preview, "run": cmd_run, "report": cmd_report,
          "compare": cmd_compare, "exclusions": cmd_exclusions}[a.cmd]
    try:
        return fn(a)
    except (Fail, H.Stop, promptlib.PromptError, imglib.ImageError, cost_ledger.BudgetExceeded,
            cost_ledger.LedgerError) as exc:
        say(f"\nSTOP: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
