#!/usr/bin/env python3
"""
scripts/stats_checks.py  (Statistics chat, D291)
Read-only checks around the Statistics runs of scripts/analysis.py. Standard
library only. It never writes into a Git tree except with `stage`, which copies
aggregate tables and figures into <main>/results/stats/ (addendum J4).

Commands
    pins     Cell 0, before any run: one PASS or FAIL line per input. The two
             clones and the tags, the code files, the packages, and every Drive
             input of the runs against the sha256 recorded by earlier chats.
             Also the D172 count: excluded items in the training split.
    verify   Cells 3 and 5: reads a finished folder of `analysis.py run` (--run)
             and of `analysis.py sens-d279` (--d279) and compares it with the
             values known before the run (addenda O2, P2, P5, R; D303). Prints the
             key estimates.
    stage    Cell 6: copies the aggregate tables, figures, reports, manifests and
             checks of verified folders into <main>/results/stats/<folder>/.
             Never numbers JSON, per-item files, TEXT_ files or placeholder files.

Output: Drive results/stats/stats_checks/ (pins_<UTC>, verify_<UTC>, .txt and
.json). Printouts hold ids, counts, hashes and estimates; never a question,
gold or answer.
"""

import argparse
import csv
import glob
import hashlib
import importlib.metadata as md
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

VERSION = "stats-checks-2026-10-24"
DRIVE_DEFAULT = "/content/drive/MyDrive/BanglaIndoorVQA"
MAIN_DEFAULT = "/content/banglaindoorvqa-main"
GATE_E_DEFAULT = "/content/bivqa_gateE"
HT3_DIR_NAME = "human/HT-3/score_final_20261023T061119Z"          # addendum P2

# ---- values recorded by earlier chats --------------------------------------

TAGS = {"gate-A": "9ee3eb0a535b5f5b799777d410e6cef38cc29f64",
        "gate-E": "5db2bfff3fd3e31b83f8c4865d37388f25f931f0",
        "gate-F": "a5164e4a0b8244f16e1e793203444e638b4e28c6"}
GATE_E_FILES = {   # in the detached gate-E checkout (addenda J, K, O2)
    "src/stats.py": "125668b0e712eb219470dfe271fe4a68ace93362c4f0e195a674428c08be68a5",
    "scripts/analysis.py": "9d90f3ff477a3bb44c682a0184eb2e3224e492f51c91eb69d75dbe5fe2915f52",
    "src/scoring.py": "8563fa05f606fd1d0899b3e96e7c06ed120fac77c5fd2343b1c1bb82a5e82b2a",
    "src/normalise.py": "488a644d30d130586aa61737ae271a3ed2ddbefb6990454b3542ac09cccc231a",
    "lexicon/lexicon.csv": "73b6bc48f1bda27863fbad69e26830c5f484d50702543b66297679bc07bc7448",
    "lexicon/closed_vocab.csv": "de36f2905e43784704f8765c9886da1e9a9524d295ff10fac6c75b21f50e4b90",
    "probes/excluded_items.csv": "8c51763fad6e3c48f0ae1e137573d00e1d0e79274bf959d9bb8e0bca96b37cdb"}
# main must equal gate-E for these, so sens-d279 reads the same world (D230, D271)
SAME_AS_GATE_E = ("src/stats.py", "src/scoring.py", "src/normalise.py", "lexicon/lexicon.csv",
                  "lexicon/closed_vocab.csv", "probes/excluded_items.csv", "configs/conditions.yaml",
                  "probes/P-CORE-CLOSED.ids.txt", "probes/R-TEST.ids.txt", "probes/P-NAME.ids.txt",
                  "probes/P-CLUST.ids.txt", "probes/P-FCE.ids.txt", "probes/NAME-REF.ids.txt")
ANALYSIS_D290 = {"sha256": "3a261ba8a587b16bf4cdb0d9d095af1dd96b34f767a0fd31b779fed2cffbbced", "deleted_vs_gate_e": 2,
                 "script_version": "analysis-2026-10-24", "d279_version": "analysis-d279-2026-10-24"}
SCORED = {   # addendum O2: sha256 and records per file
    "qwen": ("2088f0a91be62518f25832dd989e47fd7a805da8bec9ef0be694b43f774d115e", 15947),
    "gemma": ("8d64916f320835d5757fd66f50f1ea3b08c03dc35bac49cd5a29a9f22789039d", 11600),
    "gemini": ("3e0ed6cb4d404676cbe1ce74364a70efb7660713bfa3656bac42227eecdc5143", 11600),
    "qwen_ft_s42": ("30224f1e67e6b52fb1d88c53dbfe87cac0404990d585b2003b5e9aadaea05913", 4347),
    "qwen_ft_s43": ("b4cb6740f7996e5e5b736295e7283f611102be7cf5d40bbb8e316263421611d2", 4347),
    "qwen_ft_s44": ("e6e08d76ecd9797bd912ae337081e0766ef79162b0f8f20886acbc0986c0b540", 4347)}
DRIVE_FILES = (   # (path under Drive, sha256 or recorded prefix, what)
    ("data/banglaindoorvqa.json", "30017e400212d3949928599766513ee462802f0878484cab7dda6a73ec9daa58",
     "version of record (D300)"),
    ("data/house_assignment.csv", "1dde22c04bddadae85f262e97c8429eb1b404ec22a4fe12050716287d478941c", "D206"),
    ("data/splits.csv", "cc5089f2a5545e876b01442c6da8ae09ccbcdb47ed62c8233faacb64a8c986bf", "addendum R2"),
    ("probes/P-CORE-CLOSED.json", "ea242d5b8ade", "pinned in stats.py"),
    ("probes/R-TEST.json", "30cd10956e56", "pinned in stats.py"),
    ("probes/P-NAME.json", "fc4bfb1dbda1f6f3a554e196dc049c90a9a72af4376170420f50b5decd7e9035", "pinned in stats.py"),
    ("probes/P-CLUST.json", "7759dcce19e4c36596d6b39a39be8feefa467cf2088d5c03bdcace99a70f296f", "pinned in stats.py"),
    ("probes/P-FCE.json", "807e987b4b656586ee2955efb421e8e6cd0aae27790a11a4c4f3f4f7210479ae", "pinned in stats.py"),
    ("probes/probe_manifest_11.json", "5916b56ba317807fdac827edbb6cb5b4ffb3cb6ac62c1ef68fc5ee1904be3f4e",
     "pinned in stats.py"),
    ("human/HT-6/ht6_swaps.csv", "9b2f2a8569b9a8682cbf011a05295c501bd3071b8d5a7c1dcc51690c46f4f957", "D204"),
    (HT3_DIR_NAME + "/ht3_agreement.json", "ffa70e6a10ab52c2978c906f6c7895a9b9741e6c26f780f812a1c06d2ece3683",
     "HT-3 final (P2)"),
    (HT3_DIR_NAME + "/ht3_scored_names.csv", "4982aeef2ca74077f2d20da98e3e160b4cb2a0791e1deb8d755b9ee764adefd5",
     "HT-3 final (P2)"),
    ("human/HT-4/ht4_labels.csv", "e5bcc7a056a5", "D238, addendum O2"),
    ("human/HT-5/ht5_audit.csv", "f6c9de862d2e", "D238, R01"),
    ("human/HT-5/lexicon_sensitivity.csv", "12d0289c39e2", "D238, the last checked version"),
    ("results/stats_inputs/token_counts.csv", "", "no sha recorded; rows checked"),
    ("scored/main/sensitivity_D279/changes.csv", "6f8e9bc759328f3142c30e427f4c0ff5372a1f4a7113b663de950bafdcc9672a",
     "D279"),
    ("scored/main/sensitivity_D279/summary.json", "1a8d15ed57c671a4c310341f3e6151acf8743201c1baea6101f76b636094205e",
     "D279"),
    ("scored/main/sensitivity_D279/by_condition.csv",
     "741b6e9d95fdbedb8c54abf6f9359edf720f10f2f984d043523c2925973303d6", "D279"))
PIN_OF = {os.path.basename(rel): pin for rel, pin, _ in DRIVE_FILES}
PACKAGES = (("transformers", "5.18.0"), ("statsmodels", None), ("sacrebleu", None), ("numpy", None),
            ("matplotlib", None), ("PyYAML", None), ("torch", None))
D303 = {"images": 4406, "houses": 68, "qa": 18276, "objects": 3620, "train_qa": 13668,
        "splits": {"train": 43, "val": 10, "test": 15}}
D172_EXPECT = {"train": 13, "val": 1, "test": 14}
CONTENT = {"ht4_rows": 200, "ht5_strata": {"C": 140, "english_copy": 30, "sibling": 30}, "sens_rows": 25,
           "sens_sources": {"HT-5": 17, "D192": 8}, "token_by_model": {"gemini": 1118, "qwen": 1118, "gemma": 1118},
           "d279_rules": {"a": 294, "b": 10}, "excluded_rows": 28}

EXPECT = {
    "manifest": {"mode": "real", "seed": 42, "n_boot": 10000, "failed": [], "deviation": "",
                 "git_head": TAGS["gate-E"][:12], "scorer_version": "0.2.2-stage2", "prompt_version": "exp-v1",
                 "stage": "main", "script_version": "analysis-2026-10-17", "stats_version": "stats-2026-10-17"},
    "inputs": {"scoring.py": GATE_E_FILES["src/scoring.py"], "normalise.py": GATE_E_FILES["src/normalise.py"],
               "lexicon.csv": GATE_E_FILES["lexicon/lexicon.csv"],
               "closed_vocab.csv": GATE_E_FILES["lexicon/closed_vocab.csv"],
               "excluded_items.csv": GATE_E_FILES["probes/excluded_items.csv"]},
    "t1_sets": {   # set: (items run, items scored, houses run, houses scored or None, images or None)
        "P-CORE-CLOSED": (1118, 1109, 68, 68, 979), "R-TEST": (1449, 1435, 15, 15, None),
        "P-NAME": (1000, 991, 61, 61, 1000), "P-CLUST": (492, 483, 57, None, 492),
        "P-FCE": (454, 446, 57, None, 454), "NAME-REF": (200, 199, 22, 22, None)},
    "ht2": ("146 of 150", "97.33 [93.34, 98.96]"),
    "ht3": {"n": "199", "six-label agreement (primary)": "70.4 [64.7, 75.7]",
            "six-label kappa (primary)": "0.274 [0.180, 0.357]",
            "strict agreement (secondary)": "72.4 [67.3, 77.0]",
            "strict kappa (secondary)": "0.271 [0.162, 0.361]",
            "strict, N1 (main row, D214)": "68.3 [60.4, 76.5]",
            "strict, N2 (secondary row, D214)": "82.9 [75.4, 88.8]"},
    "ht4": {"n": 200, "correct": 187, "accuracy": "93.5 [89.8, 97.0]",
            "kappa": {"scorer_vs_consensus": "0.924", "scorer_vs_checker1": "0.877", "scorer_vs_checker2": "0.900",
                      "checker1_vs_checker2": "0.905"},
            "kappa_ci_scorer_vs_consensus": "0.881 0.964",
            "per_label": {"tier1": (31, 35), "tier2": (12, 12), "sibling": (12, 12), "english_copy": (10, 11),
                          "other_wrong": (36, 43), "correct": (37, 37), "wrong": (32, 32), "invalid": (14, 14),
                          "ambiguous": (3, 4)}},
    "ht5": {"n": 200, "houses": 49, "gold_error": (3, "1.5 [0.0, 4.1]"), "f3_mark": (1, 140, "0.7 [0.0, 2.3]"),
            "strata": {   # R05: count and the CI text of each verdict share
                "C": (140, {"not_visible": (0, "0.0 [0.0, 0.0]"), "gold_wrong": (1, "0.7 [0.0, 2.3]"),
                            "ambiguous": (25, "17.9 [10.2, 24.7]"), "missing_variant": (10, "7.1 [3.4, 11.3]"),
                            "model_wrong": (104, "74.3 [67.1, 82.9]")}),
                "english_copy": (30, {"not_visible": (0, "0.0 [0.0, 0.0]"), "gold_wrong": (1, "3.3 [0.0, 11.1]"),
                                      "ambiguous": (0, "0.0 [0.0, 0.0]"), "missing_variant": (20, "66.7 [48.0, 82.1]"),
                                      "model_wrong": (9, "30.0 [14.3, 48.3]")}),
                "sibling": (30, {"not_visible": (0, "0.0 [0.0, 0.0]"), "gold_wrong": (1, "3.3 [0.0, 11.1]"),
                                 "ambiguous": (4, "13.3 [2.9, 29.2]"), "missing_variant": (0, "0.0 [0.0, 0.0]"),
                                 "model_wrong": (25, "83.3 [66.7, 95.5]")})}},
    "s9": {"total": 352,
           "rows": {"S01": 9, "S02": 28, "S03": 48, "S04": 13, "S05": 8, "S06": 50, "S07": 8, "S08": 5, "S09": 71,
                    "S10": 9, "S11": 12, "S12": 24, "S13": 0, "S14": 38, "S15": 3, "S16": 1, "S17": 1, "S18": 5,
                    "S19": 7, "S20": 0, "S21": 0, "S22": 0, "S23": 6, "S24": 1, "S25": 2},
           "cells": {"gemini|L-BB": 29, "gemini|L-BB0": 4, "gemini|L-BE": 3, "gemini|L-EB": 21, "gemini|L-EE": 1,
                     "gemini|N-BB": 80, "gemini|N-BBs": 31, "gemma|L-BB": 11, "gemma|L-BB0": 4, "gemma|L-BE": 1,
                     "gemma|L-EB": 10, "gemma|L-EE": 1, "gemma|N-BB": 30, "gemma|N-BBs": 29,
                     "qwen_ft_s42|R-EE": 1, "qwen_ft_s44|R-EE": 1, "qwen|L-BB": 6, "qwen|L-BB0": 1, "qwen|L-BE": 2,
                     "qwen|L-EB": 13, "qwen|L-EE": 2, "qwen|N-BB": 30, "qwen|N-BBs": 29, "qwen|N-BE": 3,
                     "qwen|R-BB": 9}},
    "budget": {"gemini": (258, {"A1": 1, "A2": 11, "A3": 6, "A4": 57, "B": 155, "C": 28}),
               "qwen": (460, {"A1": 0, "A2": 13, "A3": 5, "A4": 90, "B": 165, "C": 187}),
               "gemma": (337, {"A1": 0, "A2": 9, "A3": 7, "A4": 54, "B": 199, "C": 68})},
    "exp": {"exp_l": (1109, 68), "exp_n": (991, 61), "exp_r": (1435, 15),
            "seeds": ["qwen_ft_s42", "qwen_ft_s43", "qwen_ft_s44"]},
    "files": {"figures": ["F2_exp_l", "F3_error_budget", "F4_word_form", "F5_tiers", "F6_exp_r"],
              "tables": ["T1_probe_sets", "T2_models_record_columns", "T4_contrasts", "T5_instrument_checks"],
              "supplement": ["S%d_" % i for i in range(1, 11)],
              "figure_data": ["F2_data", "F3_data", "F4_data", "F5_data", "F6_data"]},
    "d279": {"rows": 304, "a": 294, "b": 10, "cells": 37, "s11_rows": 31,
             "fc_rule_b": {"gemini|N-FCB": 1, "gemini|N-FCB0": 6, "gemini|N-FCE0": 3},
             "change_points": {   # summary.json, exp_l_contrast_change_points
                 "gemini": {"P1": 0.0013525698827772767, "P2": 0.0013525698827772767, "P12": 0.0,
                            "P3": 0.0027051397655545534},
                 "qwen": {"P1": -0.0018034265103697023, "P2": -0.0009017132551848512,
                          "P12": 0.0009017132551848512, "P3": -0.0009017132551848512},
                 "gemma": {"P1": 0.0009017132551848512, "P2": 0.0, "P12": -0.0009017132551848512,
                           "P3": 0.0018034265103697023}}}}

MODELS = ("gemini", "qwen", "gemma")
MODEL_NAMES = {"gemini": "Gemini 3.6 Flash", "qwen": "Qwen3.5-4B", "gemma": "Gemma 4 E4B"}
CONTRASTS = ("P1", "P2", "P12", "P3", "P4", "P5", "P6", "P7")
COND_L = ("L-BB", "L-BE", "L-EB", "L-EE", "L-BB0", "L-EE0")
STAGE_OK = (re.compile(r"\.(png|pdf)$"), re.compile(r"\.(csv|md)$"))
STAGE_TOP = ("analysis_report.md", "d279_report.md", "explore_report.md", "manifest.json", "checks.txt")
D293_VERSION = "explore-p7-d293-2026-10-24"


# ---- small tools --------------------------------------------------------------

def say(msg=""):
    print(msg, flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo, *args):
    try:
        r = subprocess.run(["git", "-C", str(repo)] + list(args), capture_output=True, text=True)
    except OSError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    return (rows[0], rows[1:]) if rows else ([], [])


def read_dicts(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(fh)]


def pct(x, digits=1):
    return "-" if x is None else ("%." + str(digits) + "f") % (100.0 * x)


def ci_text(s, scale=100.0, digits=1):
    """As stats.ci_text."""
    if not s or s.get("point") is None:
        return "-"
    f = "%." + str(digits) + "f"
    lo = "-" if s.get("lo") is None else f % (s["lo"] * scale)
    hi = "-" if s.get("hi") is None else f % (s["hi"] * scale)
    return "%s [%s, %s]" % (f % (s["point"] * scale), lo, hi)


def fmt_p(p, floor=None):
    if p is None:
        return "-"
    if p == 0:
        return "< %.4f" % floor if floor else "< 1e-16"
    return "%.4f" % p if p >= 0.0001 else "%.1e" % p


def fmt_holm(h):
    if not h:
        return "-"
    return ("< %.4f" % h["p"]) if h["bound"] else fmt_p(h["p"])


def count_of(s, n):
    return None if not s or s.get("point") is None else int(round(s["point"] * n))


class Lines:
    """One PASS, FAIL, WARN or INFO line per check, with an id."""

    def __init__(self, prefix):
        self.prefix, self.rows, self.k = prefix, [], 0

    def add(self, status, what, detail=""):
        self.k += 1
        row = ("%s%02d" % (self.prefix, self.k), status, what, str(detail))
        self.rows.append(row)
        say("%-4s %-4s  %-44s %s" % row)
        return status == "PASS"

    def check(self, ok, what, detail):
        return self.add("PASS" if ok else "FAIL", what, detail)

    def count(self, status):
        return sum(1 for r in self.rows if r[1] == status)

    def text(self):
        return "".join("%-4s %-4s  %-44s %s\n" % r for r in self.rows)


def write_out(drive, name, lines, extra):
    d = Path(drive) / "results" / "stats" / "stats_checks"
    d.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    (d / ("%s_%s.txt" % (name, stamp))).write_text(lines.text(), encoding="utf-8")
    js = dict(extra, version=VERSION, utc=utc_now(), fail=lines.count("FAIL"), warn=lines.count("WARN"),
              lines=[list(r) for r in lines.rows])
    (d / ("%s_%s.json" % (name, stamp))).write_text(json.dumps(js, ensure_ascii=False, indent=1), encoding="utf-8")
    return d / ("%s_%s.txt" % (name, stamp))


def newest(base, prefix):
    found = sorted(p for p in Path(base).glob(prefix + "*") if p.is_dir() and "PLACEHOLDER" not in p.name)
    return found[-1] if found else None


def resolve(arg, base, prefix):
    if arg in (None, ""):
        return None
    if arg == "latest":
        return newest(base, prefix)
    return Path(arg)


# ---- pins (Cell 0) ------------------------------------------------------------

def d172_count(drive, main, L):
    """Excluded items by split (D172: the list is not applied to training or
    validation data), from the dataset, splits.csv and the gate-E excluded list."""
    with open(Path(drive) / "data" / "banglaindoorvqa.json", encoding="utf-8") as fh:
        data = json.load(fh)
    split_of = {r["house_id"]: r["split"] for r in read_dicts(Path(drive) / "data" / "splits.csv")}
    house_of = {}
    for im in data:
        for qa in im.get("qa_pairs") or []:
            house_of[qa["qa_id"]] = im["house_id"]
    qa_split = Counter(split_of.get(h) for h in house_of.values())
    ex = read_dicts(Path(main) / "probes" / "excluded_items.csv")
    by_split, reasons = Counter(), {}
    missing = [r["qa_id"] for r in ex if r["qa_id"] not in house_of]
    prefix_bad = [r["qa_id"] for r in ex if r["qa_id"] in house_of and house_of[r["qa_id"]] != r["qa_id"][:3]]
    val_ids = []
    for r in ex:
        sp = split_of.get(house_of.get(r["qa_id"], r["qa_id"][:3]))
        by_split[sp] += 1
        reasons.setdefault(sp, Counter())[r["reason"]] += 1
        if sp == "val":
            val_ids.append(r["qa_id"])
    ftval = set()
    for p in (Path(main) / "probes" / "FT-VAL400.ids.txt", Path(drive) / "data" / "ft" / "FT-VAL400.ids.txt"):
        if p.exists():
            ftval = set(re.findall(r"H\d{2}_\d{4}_q\d+", p.read_text(encoding="utf-8-sig")))
            break
    ok = (dict(by_split) == D172_EXPECT and qa_split.get("train") == D303["train_qa"] and not missing
          and not prefix_bad and len(ex) == CONTENT["excluded_rows"])
    L.check(ok, "D172: excluded items by split",
            "train %d of %s training pairs (%s); val %d (%s); test %d; list rows %d; not in the dataset %d;"
            " house differs from the id prefix %d" % (
                by_split.get("train", 0), "{:,}".format(qa_split.get("train", 0)),
                ", ".join("%s %d" % kv for kv in sorted(reasons.get("train", Counter()).items(),
                                                        key=lambda kv: -kv[1])),
                by_split.get("val", 0), ", ".join("%s in FT-VAL400: %s" % (q, "yes" if q in ftval else "no")
                                                  for q in val_ids) or "-",
                by_split.get("test", 0), len(ex), len(missing), len(prefix_bad)))
    return {"by_split": dict(by_split), "reasons_by_split": {k: dict(v) for k, v in reasons.items()},
            "train_qa": qa_split.get("train"), "val_ids": val_ids, "val_in_ftval400": [q in ftval for q in val_ids],
            "ftval400_n": len(ftval)}


def cmd_pins(args):
    say("stats_checks pins (%s, UTC %s)" % (VERSION, utc_now()))
    L = Lines("P")
    main, gate_e, drive = Path(args.main), Path(args.gate_e), Path(args.drive)
    head = git(gate_e, "rev-parse", "HEAD")
    detached = git(gate_e, "symbolic-ref", "-q", "HEAD") is None
    st = git(gate_e, "status", "--porcelain")
    L.check(head == TAGS["gate-E"] and detached and st == "", "gate-E checkout",
            "%s: HEAD %s, detached %s, clean %s" % (gate_e, (head or "none")[:12], detached, st == ""))
    for tag, commit in TAGS.items():
        a, b = git(main, "rev-parse", tag + "^{commit}"), git(gate_e, "rev-parse", tag + "^{commit}")
        L.check(a == commit and b == commit, "tag %s" % tag, "commit %s in main %s, in the gate-E checkout %s" % (
            commit[:12], (a or "missing")[:12], (b or "missing")[:12]))
    for rel, want in GATE_E_FILES.items():
        p = gate_e / rel
        got = sha256_file(p) if p.exists() else "missing"
        L.check(got == want, "gate-E %s" % rel, "sha256 %s..." % got[:12])
    br = git(main, "rev-parse", "--abbrev-ref", "HEAD")
    mh, oh = git(main, "rev-parse", "HEAD"), git(main, "rev-parse", "origin/main")
    ms = git(main, "status", "--porcelain")
    L.check(br == "main" and mh and mh == oh and ms == "", "main clone",
            "%s: branch %s, HEAD %s, equal to origin/main %s, clean %s" % (main, br, (mh or "none")[:12],
                                                                           mh == oh, ms == ""))
    ap = main / "scripts" / "analysis.py"
    got = sha256_file(ap) if ap.exists() else "missing"
    ns = (git(main, "diff", "--numstat", "gate-E", "HEAD", "--", "scripts/analysis.py") or "").split("\t")
    added, deleted = (ns[0], ns[1]) if len(ns) >= 2 else ("?", "?")
    L.check(got == ANALYSIS_D290["sha256"] and deleted == str(ANALYSIS_D290["deleted_vs_gate_e"]),
            "main scripts/analysis.py (D290)", "sha256 %s...; against gate-E %s lines added, %s deleted (expected"
            " %s...: 2 deleted)" % (got[:12], added, deleted, ANALYSIS_D290["sha256"][:12]))
    diff = git(main, "diff", "--name-only", "gate-E", "HEAD", "--", *SAME_AS_GATE_E)
    L.check(diff == "", "main equals gate-E (13 files)", "stats.py, scorer, normalise, lexicon, excluded list,"
            " conditions.yaml, the six id lists: %s" % ("identical" if diff == "" else "differ: %s" % diff))
    sc = main / "scripts" / "stats_checks.py"
    L.add("INFO", "main scripts/stats_checks.py", "sha256 %s..., %s" % (
        sha256_file(sc)[:12] if sc.exists() else "missing", VERSION))
    ex = main / "scripts" / "explore_p7.py"
    L.add("INFO", "main scripts/explore_p7.py (D293)", "sha256 %s..." % (sha256_file(ex)[:12] if ex.exists()
                                                                        else "missing"))
    for dist, want in PACKAGES:
        try:
            v = md.version(dist)
        except md.PackageNotFoundError:
            v = None
        L.check(v is not None and (want is None or v == want), "package %s" % dist,
                "%s%s" % (v or "not installed", (" (needs %s, D153)" % want) if want and v != want else ""))
    for rel, pin, what in DRIVE_FILES:
        p = drive / rel
        if not p.exists():
            L.check(False, "Drive %s" % rel, "missing")
            continue
        got = sha256_file(p)
        if pin:
            L.check(got.startswith(pin), "Drive %s" % rel, "sha256 %s... (%s)%s" % (
                got[:12], what, "" if got.startswith(pin) else "; expected %s..." % pin[:12]))
        else:
            L.add("INFO", "Drive %s" % rel, "sha256 %s... (%s)" % (got[:12], what))
    files = sorted(glob.glob(str(drive / "scored" / "main" / "*_scored.jsonl")))
    names = {os.path.basename(f)[:-len("_scored.jsonl")]: f for f in files}
    L.check(set(names) == set(SCORED), "Drive scored/main/*_scored.jsonl", "%d files: %s" % (
        len(files), ", ".join(sorted(names))))
    for m, (sha, n) in SCORED.items():
        f = names.get(m)
        if not f:
            L.check(False, "scored %s" % m, "missing")
            continue
        got = sha256_file(f)
        with open(f, "rb") as fh:
            lines = sum(1 for line in fh if line.strip())
        L.check(got == sha and lines == n, "scored %s" % m, "sha256 %s..., %d records (addendum O2: %s..., %d)" % (
            got[:12], lines, sha[:12], n))
    # content of the human and stats inputs (counts only)
    sp = read_dicts(drive / "data" / "splits.csv") if (drive / "data" / "splits.csv").exists() else []
    L.check(dict(Counter(r["split"] for r in sp)) == D303["splits"], "splits.csv houses",
            "%s (D303: train 43, val 10, test 15)" % dict(Counter(r["split"] for r in sp)))
    p4 = drive / "human" / "HT-4" / "ht4_labels.csv"
    if p4.exists():
        r4 = read_dicts(p4)
        L.check(len(r4) == CONTENT["ht4_rows"], "HT-4 labels rows", "%d answers, %d houses, agree with consensus %d" % (
            len(r4), len({r["house_id"] for r in r4}), sum(1 for r in r4 if r["scorer_label"] == r["consensus_label"])))
    p5 = drive / "human" / "HT-5" / "ht5_audit.csv"
    if p5.exists():
        r5 = read_dicts(p5)
        st5 = dict(Counter(r["stratum"] for r in r5))
        L.check(st5 == CONTENT["ht5_strata"], "HT-5 audit rows",
                "%d answers, %d houses, strata %s" % (len(r5), len({r["house_id"] for r in r5}), st5))
    ps = drive / "human" / "HT-5" / "lexicon_sensitivity.csv"
    if ps.exists():
        rs = read_dicts(ps)
        src = dict(Counter(r["source"] for r in rs))
        L.check(len(rs) == CONTENT["sens_rows"] and src == CONTENT["sens_sources"], "lexicon sensitivity rows",
                "%d rows %s; row ids %s to %s" % (len(rs), src, rs[0]["row_id"] if rs else "-",
                                                  rs[-1]["row_id"] if rs else "-"))
    pt = drive / "results" / "stats_inputs" / "token_counts.csv"
    if pt.exists():
        rt = read_dicts(pt)
        bym = dict(Counter(r["model_key"] for r in rt))
        L.check(bym == CONTENT["token_by_model"], "token counts rows",
                "%d rows %s (PROJECT_STATE: 3,354)" % (len(rt), bym))
    pc = drive / "scored" / "main" / "sensitivity_D279" / "changes.csv"
    if pc.exists():
        rc = read_dicts(pc)
        L.check(dict(Counter(r["rule"] for r in rc)) == CONTENT["d279_rules"],
                "D279 change rows", "%d rows %s (addendum O2: 304, a 294, b 10)" % (
                    len(rc), dict(Counter(r["rule"] for r in rc))))
    ht3 = Path(args.ht3_dir) if args.ht3_dir else drive / HT3_DIR_NAME
    L.check((ht3 / "ht3_agreement.json").exists() and (ht3 / "ht3_scored_names.csv").exists(),
            "HT-3 input folder", "%s holds ht3_agreement.json and ht3_scored_names.csv" % ht3.name)
    d172 = None
    if (drive / "data" / "banglaindoorvqa.json").exists() and (drive / "data" / "splits.csv").exists():
        try:
            d172 = d172_count(drive, main, L)
        except Exception as exc:
            L.check(False, "D172: excluded items by split", "%s: %s" % (type(exc).__name__, exc))
    out = write_out(drive, "pins", L, {"main_head": mh, "d172": d172})
    n_fail = L.count("FAIL")
    say("pins: %s; written to %s" % ("ALL PASS" if not n_fail else "%d FAIL" % n_fail, out))
    return 1 if n_fail else 0


# ---- verify (Cells 3 and 5) ----------------------------------------------------

def check_lines(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append((line[:4].strip(), line[6:52].strip(), line[52:].strip()))
    return rows


def verify_primary(run, L):
    E = EXPECT
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    nums = json.loads((run / "numbers.json").read_text(encoding="utf-8"))
    bad = [k for k, v in E["manifest"].items() if man.get(k) != v]
    L.check(not bad, "manifest: mode, code, settings", "%s; mode %s, script %s, stats %s, scorer %s, prompt %s,"
            " stage %s, %s resamples, seed %s, git head %s, failed stages %d, deviation %s%s" % (
                run.name, man.get("mode"), man.get("script_version"), man.get("stats_version"),
                man.get("scorer_version"), man.get("prompt_version"), man.get("stage"), man.get("n_boot"),
                man.get("seed"), man.get("git_head"), len(man.get("failed") or []), man.get("deviation") or "none",
                ("; differs: %s" % ", ".join(bad)) if bad else ""))
    got = {re.sub(r"_scored\.jsonl$", "", os.path.basename(f["file"])): (f["sha256"], f["records"])
           for f in man.get("scored_files", [])}
    L.check(got == SCORED, "manifest: the six scored files", "sha256 and records as addendum O2: %s" % (got == SCORED))
    inp = man.get("inputs", {})
    bad = [k for k, v in E["inputs"].items() if inp.get(k) != v]
    bad += [os.path.basename(rel) for rel, pin, _ in DRIVE_FILES
            if os.path.basename(rel) in inp and pin and not str(inp[os.path.basename(rel)]).startswith(pin)]
    L.check(not bad, "manifest: input hashes", "scorer, normalise, lexicon, vocabulary, excluded list, dataset,"
            " probes, house assignment, HT-6 swaps as pinned%s" % (("; differ: %s" % ", ".join(bad)) if bad else ""))
    h3 = man.get("ht3") or {}
    ok3 = h3.get("json") == PIN_OF["ht3_agreement.json"] and h3.get("csv") == PIN_OF["ht3_scored_names.csv"]
    L.check(ok3, "manifest: HT-3 final files", "json %s..., csv %s..." % (str(h3.get("json"))[:12],
                                                                       str(h3.get("csv"))[:12]))
    cl = check_lines(run / "checks.txt")
    st = Counter(r[0] for r in cl)
    warns = ["%s (%s)" % (r[1], r[2][:80]) for r in cl if r[0] in ("WARN", "FAIL")]
    L.check(st.get("FAIL", 0) == 0 and st.get("WARN", 0) == 0, "checks.txt of the run",
            "%d lines: PASS %d, INFO %d, WARN %d, FAIL %d%s" % (len(cl), st.get("PASS", 0), st.get("INFO", 0),
                                                               st.get("WARN", 0), st.get("FAIL", 0),
                                                               ("; " + "; ".join(warns)) if warns else ""))
    ds = [r for r in cl if r[1].startswith("Drive data/banglaindoorvqa.json")]
    vals = re.findall(r"images (\d+), houses (\d+), QA (\d+), boxes (\d+)", ds[0][2]) if ds else []
    got = tuple(int(x) for x in vals[0]) if vals else None
    want = (D303["images"], D303["houses"], D303["qa"], D303["objects"])
    L.check(got == want, "dataset values printed by the run = D303", "images, houses, QA pairs, boxes: %s (D303:"
            " 4,406, 68, 18,276, 3,620)" % (got,))
    for r in cl:
        if r[1].startswith("dataset annotator ids"):
            L.add("INFO", "dataset annotator ids (printed by the run)", "%s (not a D303 value)" % r[2])
    hdr, rows = read_csv(run / "tables" / "T1_probe_sets.csv")
    t1 = {r[0]: r for r in rows}
    drow = t1.get("dataset (QA pairs as items)")
    got = (drow[1], drow[2], drow[3], drow[7]) if drow else None
    L.check(got == (str(D303["images"]), str(D303["houses"]), str(D303["qa"]), "bounding boxes %d" % D303["objects"]),
            "T1 dataset row = D303", "images %s, houses %s, QA pairs %s, %s" % (got if got else ("-",) * 4))
    bad = []
    for name, (run_n, sco_n, h_run, h_sco, imgs) in E["t1_sets"].items():
        r = t1.get(name)
        if not r or r[3] != str(run_n) or r[4] != str(sco_n) or r[5] != str(run_n - sco_n) or r[2] != str(h_run) \
                or (h_sco is not None and r[6] != str(h_sco)) or (imgs is not None and r[1] != str(imgs)):
            bad.append(name)
    L.check(not bad, "T1 probe-set rows", "run and scored items, excluded, houses and images as recorded%s" % (
        ("; differ: %s" % ", ".join(bad)) if bad else ""))
    hdr, rows = read_csv(run / "tables" / "T5_instrument_checks.csv")
    t5 = {(r[0], r[1]): r for r in rows}
    r2 = t5.get(("HT-2 translation", "faithful pairs"))
    L.check(r2 is not None and (r2[2], r2[3]) == E["ht2"], "T5 HT-2", "%s, Wilson %s" % ((r2[2], r2[3]) if r2 else
                                                                                       ("-", "-")))
    bad = [k for k, v in E["ht3"].items() if k != "n" and (t5.get(("HT-3 naming reference", k)) or [""] * 4)[3] != v]
    n3 = (t5.get(("HT-3 naming reference", "six-label agreement (primary)")) or [""] * 3)[2]
    L.check(not bad and n3 == E["ht3"]["n"], "T5 HT-3 (addendum P2 values)",
            "n %s; six-label %s, kappa %s; strict N1 %s, N2 %s%s" % (
                n3, *[(t5.get(("HT-3 naming reference", k)) or [""] * 4)[3] for k in (
                    "six-label agreement (primary)", "six-label kappa (primary)", "strict, N1 (main row, D214)",
                    "strict, N2 (secondary row, D214)")], ("; differ: %s" % ", ".join(bad)) if bad else ""))
    h4 = nums.get("ht4") or {}
    E4 = E["ht4"]
    acc = h4.get("accuracy")
    ok = h4.get("n") == E4["n"] and count_of(acc, E4["n"]) == E4["correct"] and ci_text(acc) == E4["accuracy"]
    L.check(ok, "HT-4 accuracy", "%s of %s, %s (addendum O2: 187 of 200, 93.5 [89.8, 97.0])" % (
        count_of(acc, h4.get("n") or 0), h4.get("n"), ci_text(acc)))
    kap = {k: (h4.get(k) or {}).get("kappa") for k in E4["kappa"]}
    kc = (h4.get("scorer_vs_consensus") or {}).get("kappa_ci") or {}
    kci = "%.3f %.3f" % (kc["lo"], kc["hi"]) if kc.get("lo") is not None and kc.get("hi") is not None else "-"
    ok = all(v is not None and "%.3f" % v == E4["kappa"][k] for k, v in kap.items()) and \
        kci == E4["kappa_ci_scorer_vs_consensus"]
    L.check(ok, "HT-4 kappa", "; ".join("%s %s" % (k.replace("_", " "), "-" if v is None else "%.3f" % v)
                                       for k, v in kap.items()) + "; scorer vs consensus CI %s" % kci)
    per = h4.get("accuracy_by_consensus_label") or {}
    got = {lab: (count_of(s, s.get("n") or 0), s.get("n")) for lab, s in per.items()}
    L.check(got == E4["per_label"], "HT-4 per consensus label", ", ".join(
        "%s %s of %s" % (lab, v[0], v[1]) for lab, v in sorted(got.items())))
    h5 = nums.get("ht5") or {}
    E5 = E["ht5"]
    ge, f3 = h5.get("gold_error_rate"), h5.get("C_gold_or_visibility")
    nC = ((h5.get("strata") or {}).get("C") or {}).get("n") or 0
    ok = (h5.get("n") == E5["n"] and h5.get("houses") == E5["houses"]
          and count_of(ge, E5["n"]) == E5["gold_error"][0] and ci_text(ge) == E5["gold_error"][1]
          and nC == E5["f3_mark"][1] and count_of(f3, nC) == E5["f3_mark"][0] and ci_text(f3) == E5["f3_mark"][2])
    L.check(ok, "HT-5 gold-error rate and F3 mark", "gold-error %s of %s, %s; F3 mark %s of %s, %s; %s houses"
            " (P5: 3 of 200, 1.5 [0.0, 4.1]; 1 of 140, 0.7 [0.0, 2.3]; 49)" % (
                count_of(ge, h5.get("n") or 0), h5.get("n"), ci_text(ge), count_of(f3, nC), nC, ci_text(f3),
                h5.get("houses")))
    bad = []
    for stt, (n, verdicts) in E5["strata"].items():
        b = (h5.get("strata") or {}).get(stt) or {}
        for v, (k, txt) in verdicts.items():
            s = (b.get("verdicts") or {}).get(v)
            if b.get("n") != n or count_of(s, n) != k or ci_text(s) != txt:
                bad.append("%s %s" % (stt, v))
    L.check(not bad, "HT-5 verdicts per stratum (R05)", "15 counts and CIs equal the HT-5 reading%s" % (
        ("; differ: %s" % ", ".join(bad)) if bad else ""))
    sens = nums.get("sensitivity") or {}
    rows_ = {r["row_id"]: r["changed"] for r in sens.get("per_row", [])}
    L.check(sens.get("changed_total") == E["s9"]["total"] and rows_ == E["s9"]["rows"]
            and sens.get("changed_by_model_condition") == E["s9"]["cells"], "S9 lexicon sensitivity (X05 to X07)",
            "answers changed %s (Cell 9: 352); per-row counts equal X07: %s; per-cell counts equal X06: %s" % (
                sens.get("changed_total"), rows_ == E["s9"]["rows"],
                sens.get("changed_by_model_condition") == E["s9"]["cells"]))
    zero = sorted(k for k, v in rows_.items() if v == 0)
    L.add("INFO", "S9 rows that change no answer on their own", ", ".join(zero) or "none")
    bud = nums.get("budget") or {}
    got = {m: ((bud.get(m) or {}).get("n_denominator"), (bud.get(m) or {}).get("counts")) for m in MODELS}
    L.check(got == E["budget"], "error budget denominators and counts", "; ".join(
        "%s %s (%s)" % (m, got[m][0], ", ".join("%s %d" % kv for kv in (got[m][1] or {}).items())) for m in MODELS)
        + "; C pooled %d (P2: 283)" % sum((got[m][1] or {}).get("C", 0) for m in MODELS))
    xl, xn, xr = nums.get("exp_l") or {}, nums.get("exp_n") or {}, nums.get("exp_r") or {}
    got = ((xl.get("n_items"), xl.get("houses")), (xn.get("n_items"), xn.get("houses")),
           (xr.get("n_items"), xr.get("houses")), xr.get("seeds"))
    want = (E["exp"]["exp_l"], E["exp"]["exp_n"], E["exp"]["exp_r"], E["exp"]["seeds"])
    L.check(got == want, "items, houses and seeds", "EXP-L %s items, %s houses; EXP-N %s, %s; EXP-R %s, %s;"
            " seeds %s" % (got[0][0], got[0][1], got[1][0], got[1][1], got[2][0], got[2][1], got[3]))
    miss = []
    for f in E["files"]["figures"]:
        for ext in (".png", ".pdf"):
            if not (run / "figures" / (f + ext)).exists():
                miss.append(f + ext)
    for sub, names in (("tables", E["files"]["tables"]), ("figures", E["files"]["figure_data"])):
        for f in names:
            for ext in (".csv", ".md"):
                if not (run / sub / (f + ext)).exists():
                    miss.append("%s/%s%s" % (sub, f, ext))
    for pre in E["files"]["supplement"]:
        if len(list((run / "supplement").glob(pre + "*.csv"))) != 1:
            miss.append("supplement/%s*.csv" % pre)
    L.check(not miss, "figures and tables", "5 figures (PNG, PDF), T1 T2 T4 T5, S1 to S10, F2 to F6 data%s" % (
        ("; missing: %s" % ", ".join(miss)) if miss else ""))
    mi = (nums.get("metric") or {}).get("info") or {}
    L.add("INFO", "metric check settings", "transformers %s, sacrebleu %s, %s layer %s, idf %s, rescale %s" % (
        mi.get("transformers"), mi.get("sacrebleu"), mi.get("bertscore_model"), mi.get("bertscore_layers"),
        mi.get("idf"), mi.get("rescale_with_baseline")))
    return nums


def print_estimates(nums):
    say("")
    say("Key estimates of the primary run (percent or points, 95% CI; the full set is in analysis_report.md):")
    xl = nums["exp_l"]
    say("  B0 strict %s" % ci_text(xl["b0"]["all"]))
    for m in MODELS:
        cells = xl["models"][m]["cells"]
        say("  %-17s EXP-L strict: %s" % (MODEL_NAMES[m], "; ".join(
            "%s %s" % (c, ci_text(cells[c]["strict"])) for c in COND_L)))
    for m in MODELS:
        c, h = nums["contrasts"][m], nums["holm"][m]
        say("  %-17s %s" % (MODEL_NAMES[m], "; ".join(
            "%s %s%s" % (k, ci_text(c[k]), (" Holm %s" % fmt_holm(h.get(k))) if k in h else "")
            for k in CONTRASTS)))
    r = nums.get("exp_r") or {}
    if "P8" in r:
        say("  P8 (fine-tuned, %d seeds) %s, p %s" % (len(r["seeds"]), ci_text(r["P8"]),
                                                    fmt_p(r["P8"].get("p"), r["P8"].get("p_floor"))))
    for m in MODELS:
        b = nums["budget"][m]
        say("  %-17s error budget A share %s (denominator %d)" % (MODEL_NAMES[m], ci_text(b["A_share"]),
                                                                  b["n_denominator"]))
    for m in MODELS:
        parts = []
        for probe, block in nums["fc"][m].items():
            for c in [k for k in block if k.startswith("N-FC")]:
                v = block[c]
                parts.append("%s %s (invalid %s, refusal %s)" % (c, ci_text(v["accuracy"]), pct(v["invalid"]["point"]),
                                                                 pct(v["refusal"]["point"])))
        say("  %-17s forced choice: %s" % (MODEL_NAMES[m], "; ".join(parts)))


def verify_d279(folder, primary, main, L):
    E = EXPECT["d279"]
    man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    dn = json.loads((folder / "d279_numbers.json").read_text(encoding="utf-8"))
    head = git(main, "rev-parse", "--short=12", "HEAD") if main else None
    mh = man.get("git_head") or ""
    in_main = head is None or mh == head or (bool(re.fullmatch(r"[0-9a-f]{7,40}", mh))
                                             and git(main, "merge-base", "--is-ancestor", mh, "HEAD") is not None)
    ns = man.get("analysis_py_vs_gate_e_numstat") or ["?", "?"]
    ok = (man.get("mode") == "real" and str(man.get("deviation", "")).startswith("D290")
          and man.get("script_version") == ANALYSIS_D290["script_version"]
          and man.get("d279_version") == ANALYSIS_D290["d279_version"]
          and man.get("stats_version") == EXPECT["manifest"]["stats_version"]
          and man.get("scorer_version") == EXPECT["manifest"]["scorer_version"]
          and man.get("n_boot") == 10000 and man.get("seed") == 42
          and (primary is None or man.get("primary_run") == primary.name)
          and in_main and ns[1] == str(ANALYSIS_D290["deleted_vs_gate_e"]))
    L.check(ok, "D279 manifest", "%s; mode %s, script %s, pass %s, %s resamples, primary run %s, git head %s"
            " (in main %s), analysis.py against gate-E +%s -%s; deviation: %s" % (
                folder.name, man.get("mode"), man.get("script_version"), man.get("d279_version"), man.get("n_boot"),
                man.get("primary_run"), mh, in_main, ns[0], ns[1], man.get("deviation") or "none"))
    got = {re.sub(r"_scored\.jsonl$", "", os.path.basename(f["file"])): (f["sha256"], f["records"])
           for f in man.get("scored_files", [])}
    pins = {k: v["sha256"] for k, v in (man.get("d279_files") or {}).items()}
    want = {os.path.basename(rel): pin for rel, pin, _ in DRIVE_FILES if "sensitivity_D279" in rel}
    L.check(got == SCORED and pins == want, "D279 inputs", "six scored files as addendum O2: %s; changes.csv,"
            " summary.json, by_condition.csv as pinned: %s" % (got == SCORED, pins == want))
    cl = check_lines(folder / "checks.txt")
    st = Counter(r[0] for r in cl)
    warns = [r for r in cl if r[0] == "WARN"]
    dev = [r for r in warns if r[1].startswith("scripts/analysis.py at tag gate-E")]
    other = [r for r in warns if r not in dev]
    near = bool(other) and all(r[1] == "primary numbers reproduced" for r in other)
    detail = "%d lines: PASS %d, INFO %d, WARN %d, FAIL %d; the D290 deviation WARN %s%s" % (
        len(cl), st.get("PASS", 0), st.get("INFO", 0), st.get("WARN", 0), st.get("FAIL", 0),
        "present" if len(dev) == 1 else "missing",
        "; other WARN: %s" % "; ".join("%s (%s)" % (r[1], r[2][:90]) for r in other) if other else "")
    if st.get("FAIL", 0) == 0 and len(dev) == 1 and near:
        L.add("WARN", "checks.txt of the D279 pass", detail + " (last digits only; S11 shows the reported values)")
    else:
        L.check(st.get("FAIL", 0) == 0 and len(dev) == 1 and not other, "checks.txt of the D279 pass", detail)
    for r in cl:
        if r[1] in ("primary numbers reproduced", "EXP-L contrast changes = summary.json", "cell accuracy changes",
                    "D279 changes.csv rows", "D279 by_condition.csv"):
            L.add("INFO" if r[0] == "PASS" else "FAIL", "pass line: %s" % r[1], "%s: %s" % (r[0], r[2][:110]))
    ov = dn.get("overlay") or {}
    fc = {k: v.get("changed_b", 0) for k, v in (ov.get("by_cell") or {}).items() if v.get("changed_b")}
    ok = (ov.get("rows"), ov.get("a"), ov.get("b"), ov.get("cells")) == (E["rows"], E["a"], E["b"], E["cells"]) \
        and fc == E["fc_rule_b"]
    L.check(ok, "D279 answers changed", "%s (rule a %s, rule b %s) in %s cells; rule b by cell %s (addendum O2: 304,"
            " 294, 10; 37 cells)" % (ov.get("rows"), ov.get("a"), ov.get("b"), ov.get("cells"), fc))
    prim, sens = dn.get("primary") or {}, dn.get("d279") or {}
    bad = []
    for m, pts in E["change_points"].items():
        for k, v in pts.items():
            try:
                d = sens["contrasts"][m][k]["point"] - prim["contrasts"][m][k]["point"]
            except (KeyError, TypeError):
                d = None
            if d is None or abs(d - v) > 1e-9:
                bad.append("%s %s" % (m, k))
    L.check(not bad, "EXP-L contrast changes (addendum O2)", "P1, P2, P12, P3 point changes equal summary.json%s" % (
        ("; differ: %s" % ", ".join(bad)) if bad else ""))
    meta = dn.get("meta") or {}
    L.check(meta.get("primary_diffs") == [] and (meta.get("primary_compared") or 0) > 0,
            "primary reproduced inside the pass", "%s quantities, differences %d, last digits only %d" % (
                meta.get("primary_compared"), len(meta.get("primary_diffs", [None])),
                len(meta.get("primary_last_digit_diffs") or [])))
    hdr, rows = read_csv(folder / "supplement" / "S11_scorer_sensitivity_D279.csv")
    L.check(len(rows) == E["s11_rows"] and len(hdr) == 11, "S11 table", "%d rows, %d columns (expected %d rows:"
            " 24 contrasts, P8, 3 A shares, 3 forced-choice cells)" % (len(rows), len(hdr), E["s11_rows"]))
    say("")
    say("S11 (percentage points [95% CI]; primary -> D279 rules a and b; change; Holm p or p):")
    for r in rows:
        say("  %-31s %-34s %-22s -> %-22s %7s  %s" % (r[0], r[1], r[2], r[3], r[4], (
            "Holm %s -> %s" % (r[7], r[8])) if r[7] else (("p %s -> %s" % (r[5], r[6])) if r[5] else "")))
    flips = dn.get("readings_that_change") or []
    say("readings that change between the two scorings: %s" % ("; ".join(flips) if flips else "none"))
    return dn


def verify_d293(folder, primary, L):
    man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    ok = (man.get("script_version") == D293_VERSION and man.get("mode") == "real" and man.get("n_boot") == 10000
          and man.get("seed") == 42 and (primary is None or man.get("primary_run") == primary.name))
    L.check(ok, "D293 manifest", "%s; script %s, %s resamples, primary run %s" % (
        folder.name, man.get("script_version"), man.get("n_boot"), man.get("primary_run")))
    cl = check_lines(folder / "checks.txt")
    st = Counter(r[0] for r in cl)
    rep_ = [r for r in cl if r[1] == "primary P7 reproduced"]
    L.check(st.get("FAIL", 0) == 0 and st.get("WARN", 0) == 0 and rep_ and rep_[0][0] == "PASS",
            "checks.txt of the D293 bound", "%d lines: PASS %d, INFO %d, WARN %d, FAIL %d; primary P7 reproduced %s" % (
                len(cl), st.get("PASS", 0), st.get("INFO", 0), st.get("WARN", 0), st.get("FAIL", 0),
                rep_[0][0] if rep_ else "missing"))
    hdr, rows = read_csv(folder / "supplement" / "S3_rows_D293_exploratory.csv")
    L.check(len(rows) == 3, "D293 rows", "%d rows (one per model)" % len(rows))
    say("")
    say("D293 exploratory bound (P7, points; primary -> bound; p and Holm of the bound; English copies yes / no):")
    for r in rows:
        say("  %-17s %-22s -> %-22s p %s, Holm %s; copies %s" % (r[0], r[3], r[4], r[5], r[6], r[8]))


def cmd_verify(args):
    say("stats_checks verify (%s, UTC %s)" % (VERSION, utc_now()))
    base = Path(args.drive) / "results" / "stats"
    run = resolve(args.run, base, "run_")
    d279 = resolve(args.d279, base, "sens_D279_")
    d293 = resolve(args.d293, base, "explore_D293_")
    if run is None and d279 is None and d293 is None:
        say("STOP: give --run, --d279 and/or --d293 (a folder or latest)")
        return 2
    L = Lines("V")
    parts = {}
    nums = None
    if run is not None:
        if not (run / "numbers.json").exists():
            L.check(False, "primary run folder", "%s has no numbers.json" % run)
        else:
            k0 = len(L.rows)
            nums = verify_primary(run, L)
            parts["primary"] = {"folder": run.name, "fail": sum(1 for r in L.rows[k0:] if r[1] == "FAIL")}
    if d279 is not None:
        if not (d279 / "d279_numbers.json").exists():
            L.check(False, "D279 folder", "%s has no d279_numbers.json" % d279)
        else:
            k0 = len(L.rows)
            prim_run = run if run is not None else newest(base, "run_")
            verify_d279(d279, prim_run, Path(args.main) if args.main else None, L)
            parts["d279"] = {"folder": d279.name, "fail": sum(1 for r in L.rows[k0:] if r[1] == "FAIL")}
    if d293 is not None:
        if not (d293 / "explore_numbers.json").exists():
            L.check(False, "D293 folder", "%s has no explore_numbers.json" % d293)
        else:
            k0 = len(L.rows)
            verify_d293(d293, run if run is not None else newest(base, "run_"), L)
            parts["d293"] = {"folder": d293.name, "fail": sum(1 for r in L.rows[k0:] if r[1] == "FAIL")}
    if nums is not None and args.estimates:
        try:
            print_estimates(nums)
        except (KeyError, TypeError) as exc:
            say("estimates not printed: %s: %s" % (type(exc).__name__, exc))
    out = write_out(args.drive, "verify", L, {"parts": parts})
    n_fail = L.count("FAIL")
    say("")
    say("verify: %s; written to %s" % ("ALL PASS" if not n_fail else "%d FAIL" % n_fail, out))
    return 1 if n_fail else 0


# ---- stage (Cell 6) -----------------------------------------------------------

def verified(drive, folder_name, part):
    files = sorted((Path(drive) / "results" / "stats" / "stats_checks").glob("verify_*.json"))
    for f in reversed(files):
        js = json.loads(f.read_text(encoding="utf-8"))
        p = (js.get("parts") or {}).get(part)
        if p and p.get("folder") == folder_name:
            return p.get("fail") == 0, f.name
    return False, None


def stage_files(folder):
    out = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(folder)
        name = p.name
        if "PLACEHOLDER" in name or name.startswith("TEXT_") or "numbers" in name or "_items" in name \
                or name.endswith(".tmp"):
            continue
        top = len(rel.parts) == 1
        if (top and name in STAGE_TOP) or (not top and rel.parts[0] in ("figures", "tables", "supplement")
                                            and any(rx.search(name) for rx in STAGE_OK)):
            out.append(rel)
    return out


def cmd_stage(args):
    say("stats_checks stage (%s, UTC %s)" % (VERSION, utc_now()))
    base = Path(args.drive) / "results" / "stats"
    dest_root = Path(args.main) / "results" / "stats"
    n_files, n_bytes, refused = 0, 0, []
    for part, arg, prefix in (("primary", args.run, "run_"), ("d279", args.d279, "sens_D279_"),
                              ("d293", args.d293, "explore_D293_")):
        folder = resolve(arg, base, prefix)
        if folder is None:
            continue
        ok, vf = verified(args.drive, folder.name, part)
        if not ok:
            refused.append("%s (%s)" % (folder.name, "no verify file" if vf is None else "FAIL in %s" % vf))
            continue
        files = stage_files(folder)
        for rel in files:
            dst = dest_root / folder.name / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(folder / rel, dst)
            n_files += 1
            n_bytes += dst.stat().st_size
        say("staged %s: %d files (verified in %s)" % (folder.name, len(files), vf))
    if refused:
        say("REFUSED (verify first, with 0 FAIL): %s" % "; ".join(refused))
        return 1
    say("stage: %d files, %.1f MB in %s; numbers JSON, per-item files and TEXT_ files stay in Drive" % (
        n_files, n_bytes / 1e6, dest_root))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-only checks for the Statistics runs (D291)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pins", "verify", "stage"):
        p = sub.add_parser(name)
        p.add_argument("--drive", default=DRIVE_DEFAULT)
        p.add_argument("--main", default=MAIN_DEFAULT)
        if name == "pins":
            p.add_argument("--gate-e", default=GATE_E_DEFAULT)
            p.add_argument("--ht3-dir")
        else:
            p.add_argument("--run", help="a run_<UTC> folder, or latest")
            p.add_argument("--d279", help="a sens_D279_<UTC> folder, or latest")
            p.add_argument("--d293", help="an explore_D293_<UTC> folder, or latest")
        if name == "verify":
            p.add_argument("--no-estimates", dest="estimates", action="store_false")
    args = ap.parse_args(argv)
    return {"pins": cmd_pins, "verify": cmd_verify, "stage": cmd_stage}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
