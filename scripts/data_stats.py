#!/usr/bin/env python3
"""
scripts/data_stats.py
BanglaIndoorVQA main paper: DATA-STATS (D44; decisions D300 to D304, DESIGN_ADDENDUM R).

Recounts every dataset value of Design 3.1 on the version of record (D300), with
normalise() applied to every text field first (Design 3.1, D44). Each value is shown as
written (raw), after normalise(), beside the Design 3.1 value and the difference. It also
lists the QA pairs whose question or answer is empty after normalise() (D301).

Commands
  check     one PASS or FAIL line per input (Cell 0 of notebooks/data_stats.ipynb)
  selftest  the counting code on small made-up data; no Drive, no git
  run       the checks again (stops on any FAIL), the counts, the report; writes only
            Drive results/data_stats/run_<UTC>/

Rules this script keeps
  - It reads Drive data/ and the repo, and writes nothing under Drive data/, outputs/,
    scored/, probes/ or human/, and nothing inside the repo.
  - normalise() is imported from src/normalise.py, byte-equal to the tag gate-F. It is
    never copied. Ids (image_id, house_id, qa_id, object_id) are used as written.
  - Printouts give counts, ids and code points only. Names and other dataset text go only
    to the file TEXT_box_names.csv in the run folder (Drive).
  - Standard library only. No random draw.
"""

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True  # importing src/normalise.py must not write src/__pycache__

VERSION = "data-stats-2026-10-24"
REPO = Path(__file__).resolve().parents[1]
SCRIPT_REL = "scripts/data_stats.py"
NORMALISE_REL = "src/normalise.py"

# ---------------------------------------------------------------- pins ------------
# D300: the version of record (Drive data/banglaindoorvqa.json).
DATASET_SHA256 = "30017e400212d3949928599766513ee462802f0878484cab7dda6a73ec9daa58"
# normalise.py at the tag gate-F (the attached copy; addendum O2 records 488a644d30d1).
NORMALISE_SHA256 = "488a644d30d130586aa61737ae271a3ed2ddbefb6990454b3542ac09cccc231a"
# data/splits.csv as attached to the DATA-STATS chat: bytes (CRLF) and content.
SPLITS_BYTES_SHA256 = "cc5089f2a5545e876b01442c6da8ae09ccbcdb47ed62c8233faacb64a8c986bf"
SPLITS_CONTENT_SHA256 = "9f14f9098aa030019c57da7b20a1a5f16576933692c396e25c2f0cd32e266a13"
SPLIT_COLUMNS = ["house_id", "annotator_id", "split", "n_images"]
SPLIT_NAMES = ("train", "val", "test")
# probes/excluded_items.csv, final at Gate E (28 rows).
EXCLUDED_SHA256 = "8c51763fad6e3c48f0ae1e137573d00e1d0e79274bf959d9bb8e0bca96b37cdb"
EXCLUDED_ROWS = 28
GATE_E_COMMIT = "5db2bfff3fd3e31b83f8c4865d37388f25f931f0"
GATE_F_COMMIT_PREFIX = "a5164e4a0b82"
# The nine qa_id lists in repo probes/ and their sizes (all in the tree of gate-E).
ID_LISTS = {
    "P-CORE": 2040, "P-CORE-CLOSED": 1118, "R-TEST": 1449, "PILOT": 100,
    "P-NAME": 1000, "P-CLUST": 492, "FT-VAL400": 400, "NAME-REF": 200, "P-FCE": 454,
}
FORBIDDEN_DRIVE_DIRS = ("data", "outputs", "scored", "probes", "human")
MIN_INSTANCES = 5  # Design 3.1: "with 5 or more instances"
ITEM_D301 = "H13_0032_q2"

# ---------------------------------------------------------------- values ----------
# Design 3.1 of the gate-E design (the "old" values).
DESIGN = {
    "images": 4406, "houses": 68, "qa": 18276, "qa_types": 8, "boxes": 3620,
    "split": {"train": (43, 3296, 13668), "val": (10, 449, 1863), "test": (15, 661, 2745)},
    "capture": (("overview", 728), ("group", 1801), ("close-up", 1877)),
    "box_names": 472, "box_names_ge5": 135,
}
# Values recorded elsewhere in the project files. Cross-checks only, never "old" values.
RECORDS = {
    # addendum G2 (FT data step, 2026-10-09): QA types in the train split
    "train_types": {"assistive": 1890, "count": 1890, "obj_function": 1406, "obj_name": 1406,
                    "obj_property": 1406, "object": 1890, "scene_id": 1890, "spatial": 1890},
    "objname_qa": 1877,           # addendum C2: obj_name answers in the full dataset
    "objname_distinct": 393,      # addendum C2: distinct obj_name answers (reading not stated)
    "question_strings": 2956,     # addendum J1: distinct question strings of "the dataset"
    # Design 3.5: houses with no close-up image (EXP-N therefore has 61 houses)
    "no_closeup_houses": ["H39", "H55", "H64", "H65", "H66", "H67", "H68"],
}
QA_TYPES = sorted(RECORDS["train_types"])
IMAGE_KEYS = ("image_id", "house_id", "annotator_id", "image_file", "image_w", "image_h",
              "capture_type", "room_type", "objects", "qa_pairs")
OBJECT_KEYS = ("object_id", "bangla_name", "bbox", "bbox_pct")
QA_KEYS = ("qa_id", "type", "question", "answer")


# ================================================================ small helpers ====
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def git(args, repo=None):
    """Run git in the repo. Returns (code, stdout bytes, stderr text)."""
    r = subprocess.run(["git", "-C", str(repo or REPO)] + list(args), capture_output=True)
    return r.returncode, r.stdout, r.stderr.decode("utf-8", "replace").strip()


def git_text(args, repo=None):
    code, out, err = git(args, repo)
    return code, out.decode("utf-8", "replace").strip(), err


def fmt(n):
    if n is None:
        return "-"
    if isinstance(n, int):
        return f"{n:,}"
    return str(n)


def fmt_diff(n):
    if n is None:
        return "-"
    return f"{n:+,}" if n else "0"


def codepoints(s, limit=12):
    """Code points of a short string, for printouts that must not show dataset text."""
    if not isinstance(s, str):
        return f"<{type(s).__name__}>"
    cps = [f"U+{ord(c):04X}" for c in s[:limit]]
    if len(s) > limit:
        cps.append(f"... ({len(s)} chars)")
    return " ".join(cps) if cps else "<empty>"


def label(s):
    """A schema label for a printout: quoted if ASCII, else code points only."""
    return repr(s) if isinstance(s, str) and s.isascii() else codepoints(s, 10)


def visible(s):
    """Show invisible characters as \\uXXXX and the string edges as [ ]. For TEXT_ files."""
    out = []
    for c in s:
        if c != " " and unicodedata.category(c) in ("Cf", "Cc", "Zs", "Zl", "Zp"):
            out.append(f"\\u{ord(c):04x}")
        else:
            out.append(c)
    return "[" + "".join(out) + "]"


def has_content(s):
    """True if s holds at least one letter, digit or combining mark (Unicode L, N, M)."""
    return any(unicodedata.category(c)[0] in "LNM" for c in s)


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def valid_box(b):
    return isinstance(b, (list, tuple)) and len(b) == 4 and all(is_number(v) for v in b)


def letters_key(s):
    """Comparison key for the three capture labels, applied AFTER normalise():
    only a to z are kept, so close-up, closeup and close_up meet. Used only to match
    Design 3.1's labels; every value is also printed as it is."""
    return re.sub(r"[^a-z]", "", s)


def find_value_paths(obj, needle, path="$"):
    """JSON paths of every string value that contains needle (lower-case hex)."""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            hits += find_value_paths(v, needle, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += find_value_paths(v, needle, f"{path}[{i}]")
    elif isinstance(obj, str) and needle in obj.lower():
        hits.append(path)
    return hits


def parse_splits_text(text):
    """Rows of splits.csv as dicts; raises ValueError on a wrong header."""
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != SPLIT_COLUMNS:
        raise ValueError(f"header {reader.fieldnames}, expected {SPLIT_COLUMNS}")
    return [{k: (r[k] or "").strip() for k in SPLIT_COLUMNS} for r in reader]


def splits_content_sha256(rows):
    lines = sorted(",".join(r[k] for k in SPLIT_COLUMNS) for r in rows)
    return sha256_bytes("\n".join(lines).encode("utf-8"))


def parse_id_list(text):
    """qa_ids of a probes/*.ids.txt file: one per line; blank and # lines skipped;
    a first line that is exactly 'qa_id' is a header and is skipped."""
    ids = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if not ids and s == "qa_id":
            continue
        ids.append(s)
    return ids


def check_out_base(base, root, repo):
    """Refuse an output folder inside the repo or under a forbidden Drive folder."""
    base, root, repo = Path(base).resolve(), Path(root).resolve(), Path(repo).resolve()
    if base == repo or repo in base.parents:
        raise ValueError(f"output folder {base} is inside the repo")
    for d in FORBIDDEN_DRIVE_DIRS:
        f = (root / d).resolve()
        if base == f or f in base.parents:
            raise ValueError(f"output folder {base} is under Drive {d}/")
    want = (root / "results" / "data_stats").resolve()
    if base != want:
        raise ValueError(f"output folder {base} is not Drive results/data_stats/")
    return base


def load_normalise(path):
    """Import normalise() from exactly this file (never from sys.path)."""
    spec = importlib.util.spec_from_file_location("bivqa_normalise", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if Path(mod.__file__).resolve() != Path(path).resolve():
        raise ImportError("normalise was not loaded from the expected file")
    return mod.normalise


# ================================================================ counting ========
def compute(records, splits, id_lists, excluded, nf, min_instances=MIN_INSTANCES):
    """All counts, from parsed inputs only. Pure: no file, no git.

    records   the dataset: a list of image records
    splits    {house_id: {"annotator_id", "split", "n_images" (int)}}
    id_lists  {list name: set of qa_ids}
    excluded  set of qa_ids on probes/excluded_items.csv
    nf        normalise()
    """
    st = {"non_dict_records": 0, "non_dict_objects": 0, "non_dict_qa": 0,
          "objects_not_list": 0, "qa_pairs_not_list": 0,
          "images_without_objects": 0, "images_without_qa": 0,
          "missing_image_keys": Counter(), "extra_image_keys": Counter(),
          "missing_object_keys": Counter(), "extra_object_keys": Counter(),
          "missing_qa_keys": Counter(), "extra_qa_keys": Counter(),
          "non_string_question": 0, "non_string_answer": 0, "non_string_name": 0}
    image_ids, houses_img = Counter(), Counter()
    capture_raw, capture_norm = Counter(), Counter()
    house_capture_keys = defaultdict(set)
    house_annotators = defaultdict(Counter)
    n_obj = box_valid = box_pos = boxpct_valid = 0
    name_raw_boxes = Counter()
    name_norm_boxes = Counter()
    name_norm_images = defaultdict(set)
    norm_raw_forms = defaultdict(Counter)
    empty_name_boxes = []
    raw_empty_string_names = 0
    qa_ids = Counter()
    qa = []
    type_raw, type_norm = Counter(), Counter()

    for r in records:
        if not isinstance(r, dict):
            st["non_dict_records"] += 1
            continue
        for k in IMAGE_KEYS:
            if k not in r:
                st["missing_image_keys"][k] += 1
        for k in set(r) - set(IMAGE_KEYS):
            st["extra_image_keys"][k] += 1
        iid, hid = r.get("image_id"), r.get("house_id")
        image_ids[iid] += 1
        houses_img[hid] += 1
        house_annotators[hid][r.get("annotator_id")] += 1

        cap = r.get("capture_type")
        cap_raw = cap if isinstance(cap, str) else "<missing>"
        cap_n = nf(cap) if isinstance(cap, str) else ""
        capture_raw[cap_raw] += 1
        capture_norm[cap_n or "<empty>"] += 1
        house_capture_keys[hid].add(letters_key(cap_n))

        objs = r.get("objects")
        if objs is None:
            objs = []
        if not isinstance(objs, list):
            st["objects_not_list"] += 1
            objs = []
        if not objs:
            st["images_without_objects"] += 1
        for o in objs:
            n_obj += 1
            if not isinstance(o, dict):
                st["non_dict_objects"] += 1
                continue
            for k in OBJECT_KEYS:
                if k not in o:
                    st["missing_object_keys"][k] += 1
            for k in set(o) - set(OBJECT_KEYS):
                st["extra_object_keys"][k] += 1
            b = o.get("bbox")
            if valid_box(b):
                box_valid += 1
                if b[2] > 0 and b[3] > 0:  # [x, y, w, h] (addendum J1)
                    box_pos += 1
            if valid_box(o.get("bbox_pct")):
                boxpct_valid += 1
            name = o.get("bangla_name")
            if isinstance(name, str):
                if name == "":
                    raw_empty_string_names += 1
                else:
                    name_raw_boxes[name] += 1
                nn = nf(name)
            else:
                st["non_string_name"] += 1
                nn = ""
            if nn == "":
                empty_name_boxes.append({"object_id": o.get("object_id"), "image_id": iid,
                                         "house_id": hid, "raw": codepoints(name)})
            else:
                name_norm_boxes[nn] += 1
                name_norm_images[nn].add(iid)
                norm_raw_forms[nn][name] += 1

        qas = r.get("qa_pairs")
        if qas is None:
            qas = []
        if not isinstance(qas, list):
            st["qa_pairs_not_list"] += 1
            qas = []
        if not qas:
            st["images_without_qa"] += 1
        for q in qas:
            if not isinstance(q, dict):
                st["non_dict_qa"] += 1
                continue
            for k in QA_KEYS:
                if k not in q:
                    st["missing_qa_keys"][k] += 1
            for k in set(q) - set(QA_KEYS):
                st["extra_qa_keys"][k] += 1
            qid = q.get("qa_id")
            qa_ids[qid] += 1
            t = q.get("type")
            type_raw[t if isinstance(t, str) else "<missing>"] += 1
            t_n = nf(t) if isinstance(t, str) else ""
            type_norm[t_n or "<empty>"] += 1
            q_raw, a_raw = q.get("question"), q.get("answer")
            if q_raw is not None and not isinstance(q_raw, str):
                st["non_string_question"] += 1
            if a_raw is not None and not isinstance(a_raw, str):
                st["non_string_answer"] += 1
            qa.append({"qa_id": qid, "image_id": iid, "house_id": hid, "type": t_n,
                       "q_raw": q_raw, "a_raw": a_raw, "q": nf(q_raw), "a": nf(a_raw)})

    res = {"structure": st}
    split_of = {h: v["split"] for h, v in splits.items()}

    # ---- images and houses
    res["images"] = {"records": sum(image_ids.values()), "distinct_image_id": len(image_ids),
                     "duplicate_image_ids": sorted(str(k) for k, c in image_ids.items() if c > 1)}
    res["houses"] = {"distinct": len(houses_img), "list": sorted(str(h) for h in houses_img)}

    # ---- QA
    qa_by_id = {}
    for rec in qa:
        qa_by_id.setdefault(rec["qa_id"], rec)
    any_empty = [rec for rec in qa if rec["q"] == "" or rec["a"] == ""]
    res["qa"] = {
        "records": len(qa), "distinct_qa_id": len(qa_ids),
        "duplicate_qa_ids": sorted(str(k) for k, c in qa_ids.items() if c > 1),
        "types_raw": dict(sorted(type_raw.items())), "types_norm": dict(sorted(type_norm.items())),
        "nonempty_q_and_a": len(qa) - len(any_empty),
    }
    ts = Counter((split_of.get(rec["house_id"], "unassigned"), rec["type"]) for rec in qa)
    res["qa"]["type_by_split"] = {f"{s}|{t}": c for (s, t), c in sorted(ts.items())}

    # ---- empty after normalise(), and no letter or digit after normalise()
    def member(qid):
        return sorted(n for n, s in id_lists.items() if qid in s)

    def item_row(rec, fields):
        sp = split_of.get(rec["house_id"], "unassigned")
        lists = member(rec["qa_id"])
        return {"qa_id": rec["qa_id"], "image_id": rec["image_id"], "house_id": rec["house_id"],
                "split": sp, "type": rec["type"], "fields": "+".join(fields),
                "q_raw_codepoints": codepoints(rec["q_raw"]) if "question" in fields else "",
                "a_raw_codepoints": codepoints(rec["a_raw"]) if "answer" in fields else "",
                "id_lists": "|".join(lists) or "none",
                "in_probe_list": any(n != "FT-VAL400" for n in lists),
                "in_ft_val400": "FT-VAL400" in lists,
                "excluded": rec["qa_id"] in excluded,
                "exp_r_training_pair": sp == "train"}

    empty_rows, nocontent_rows = [], []
    for rec in qa:
        e = [f for f, v in (("question", rec["q"]), ("answer", rec["a"])) if v == ""]
        nc = [f for f, v in (("question", rec["q"]), ("answer", rec["a"])) if v and not has_content(v)]
        if e:
            empty_rows.append(item_row(rec, e))
        if nc:
            nocontent_rows.append(item_row(rec, nc))
    res["empty"] = empty_rows
    res["no_content"] = nocontent_rows
    res["empty_by_field"] = dict(Counter(r["fields"] for r in empty_rows))
    ets = Counter((r["type"], r["split"]) for r in empty_rows)
    res["empty_by_type_split"] = {f"{t}|{s}": c for (t, s), c in sorted(ets.items())}

    # ---- QA pairs repeated on one image after normalise() (information)
    seen = defaultdict(list)
    for rec in qa:
        seen[(rec["image_id"], rec["type"], rec["q"], rec["a"])].append(rec["qa_id"])
    res["qa_duplicates"] = [{"image_id": k[0], "type": k[1], "qa_ids": "|".join(map(str, v))}
                            for k, v in seen.items() if len(v) > 1]

    # ---- question strings and obj_name answers (cross-checks)
    res["questions"] = {
        "distinct_raw": len({rec["q_raw"] for rec in qa if isinstance(rec["q_raw"], str)}),
        "distinct_norm": len({rec["q"] for rec in qa if rec["q"]}),
    }
    on = [rec for rec in qa if rec["type"] == "obj_name"]
    res["obj_name"] = {
        "records": len(on),
        "distinct_answers_raw": len({rec["a_raw"] for rec in on if isinstance(rec["a_raw"], str)}),
        "distinct_answers_norm": len({rec["a"] for rec in on if rec["a"]}),
    }

    # ---- splits
    sp = {}
    for s in SPLIT_NAMES:
        hs = sorted(h for h, v in splits.items() if v["split"] == s)
        sp[s] = {"houses": len(hs),
                 "houses_in_dataset": sum(1 for h in hs if h in houses_img),
                 "images": sum(houses_img.get(h, 0) for h in hs),
                 "qa": sum(1 for rec in qa if split_of.get(rec["house_id"]) == s)}
    res["splits"] = sp
    res["split_checks"] = {
        "houses_compared": sum(1 for h in splits if h in houses_img),
        "houses_missing_in_splits": sorted(str(h) for h in houses_img if h not in splits),
        "houses_missing_in_dataset": sorted(h for h in splits if h not in houses_img),
        "n_images_mismatch": sorted(h for h, v in splits.items()
                                    if h in houses_img and v["n_images"] != houses_img[h]),
        "annotator_mismatch": sorted(h for h, v in splits.items()
                                     if h in houses_img
                                     and set(house_annotators[h]) != {v["annotator_id"]}),
        "unassigned_images": sum(c for h, c in houses_img.items() if h not in splits),
        "unassigned_qa": sum(1 for rec in qa if rec["house_id"] not in split_of),
    }

    # ---- capture types
    cap_rows = []
    keys = set()
    for label, _ in DESIGN["capture"]:
        key = letters_key(label)
        keys.add(key)
        vals = sorted(v for v in capture_norm if letters_key(v) == key)
        cap_rows.append({"label": label, "raw_exact": capture_raw.get(label, 0),
                         "norm_matched": sum(capture_norm[v] for v in vals),
                         "matched_values": vals})
    res["capture"] = {"rows": cap_rows,
                      "raw_values": dict(sorted(capture_raw.items())),
                      "norm_values": dict(sorted(capture_norm.items())),
                      "other_norm_values": {v: c for v, c in sorted(capture_norm.items())
                                            if letters_key(v) not in keys}}
    res["houses_without_closeup"] = sorted(str(h) for h in houses_img
                                           if "closeup" not in house_capture_keys[h])

    # ---- boxes
    res["boxes"] = {"object_records": n_obj, "valid_bbox": box_valid,
                    "valid_bbox_positive_wh": box_pos, "valid_bbox_pct": boxpct_valid,
                    "images_with_objects": res["images"]["records"] - st["images_without_objects"]}

    # ---- names on box labels
    changed = [f for f in name_raw_boxes if nf(f) != f]
    diag = Counter()
    for f in changed:
        hit = False
        if not unicodedata.is_normalized("NFC", f):
            diag["not_nfc"] += 1
            hit = True
        if "‌" in f or "‍" in f:
            diag["zwnj_or_zwj"] += 1
            hit = True
        if f != f.strip():
            diag["edge_whitespace"] += 1
            hit = True
        if re.search(r"\s{2,}|[^\S ]", f.strip()):
            diag["inner_whitespace"] += 1
            hit = True
        if re.search(r"[A-Z]", f):
            diag["upper_latin"] += 1
            hit = True
        if not hit:
            diag["edge_punctuation_only"] += 1
    groups = {nn: forms for nn, forms in norm_raw_forms.items() if len(forms) >= 2}
    res["names"] = {
        "raw_distinct": len(name_raw_boxes),
        "raw_empty_string_boxes": raw_empty_string_names,
        "raw_ge_min_boxes": sum(1 for c in name_raw_boxes.values() if c >= min_instances),
        "norm_distinct": len(name_norm_boxes),
        "norm_ge_min_boxes": sum(1 for c in name_norm_boxes.values() if c >= min_instances),
        "norm_ge_min_images": sum(1 for s in name_norm_images.values() if len(s) >= min_instances),
        "boxes_empty_after_normalise": len(empty_name_boxes),
        "raw_forms_changed_by_normalise": len(changed),
        "raw_forms_empty_after_normalise": sum(1 for f in name_raw_boxes if nf(f) == ""),
        "merge_groups": len(groups),
        "raw_forms_in_merge_groups": sum(len(v) for v in groups.values()),
        "changed_form_diagnostics": dict(sorted(diag.items())),
        "min_instances": min_instances,
    }
    res["_name_table"] = [
        {"name_norm": nn, "boxes": name_norm_boxes[nn], "images": len(name_norm_images[nn]),
         "raw_forms": norm_raw_forms[nn]} for nn in sorted(name_norm_boxes)]
    res["_name_boxes_raw"] = dict(name_raw_boxes)
    res["_name_images_norm"] = {nn: len(s) for nn, s in name_norm_images.items()}
    res["empty_box_names"] = empty_name_boxes

    # ---- id lists and the excluded list against the dataset (cross-checks)
    res["id_list_missing"] = {n: sum(1 for q in s if q not in qa_ids) for n, s in sorted(id_lists.items())}
    res["excluded_missing"] = sum(1 for q in excluded if q not in qa_ids)

    # ---- facts about one item (D301)
    rec = qa_by_id.get(ITEM_D301)
    if rec is None:
        res["item_d301"] = {"present": 0}
    else:
        res["item_d301"] = {
            "present": qa_ids[ITEM_D301], "house_id": rec["house_id"],
            "split": split_of.get(rec["house_id"], "unassigned"), "type": rec["type"],
            "answer_empty": rec["a"] == "", "question_empty": rec["q"] == "",
            "answer_raw_codepoints": codepoints(rec["a_raw"]),
            "id_lists": member(ITEM_D301), "excluded": ITEM_D301 in excluded}
    return res


# ================================================================ the table =======
def build_table(res):
    """Rows of the old (Design 3.1) / raw / after normalise() / difference table.
    kind: run  = a count the runs used or the gate-E design fixes (Design 3.5, 7.3)
          desc = a descriptive Design 3.1 value
          info = no Design 3.1 value"""
    rows = []

    def add(rid, row, reading, design, raw, norm, kind, note=""):
        diff = (norm - design) if (design is not None and norm is not None) else None
        rows.append({"id": rid, "row": row, "reading": reading, "design": design, "raw": raw,
                     "norm": norm, "diff": diff, "kind": kind, "note": note})

    im, qa, bx, nm = res["images"], res["qa"], res["boxes"], res["names"]
    add("R01", "Images", "image records", DESIGN["images"], im["records"], im["records"], "run",
        "normalise() does not apply to a record count")
    add("R02", "Images", "distinct image_id", DESIGN["images"], im["distinct_image_id"],
        im["distinct_image_id"], "info", "ids used as written")
    add("R03", "Houses", "distinct house_id", DESIGN["houses"], res["houses"]["distinct"],
        res["houses"]["distinct"], "run")
    add("R04", "QA pairs", "QA records, each counted as it is (D301)", DESIGN["qa"],
        qa["records"], qa["records"], "run")
    add("R05", "QA pairs", "question and answer both non-empty after normalise()", None,
        qa["records"], qa["nonempty_q_and_a"], "info", "raw column: all records")
    add("R06", "QA types", "distinct type labels", DESIGN["qa_types"], len(qa["types_raw"]),
        len(qa["types_norm"]), "desc")
    extra_types = sorted(t for t in qa["types_norm"] if t not in QA_TYPES)
    for i, t in enumerate(QA_TYPES + extra_types):
        add(f"R{7 + i:02d}", "QA type", t if t.isascii() else label(t), None, qa["types_raw"].get(t, 0),
            qa["types_norm"].get(t, 0), "info", "Design 3.1 gives only the total")
    n = 7 + len(QA_TYPES) + len(extra_types)
    add(f"R{n:02d}", "Bounding boxes", "object records", DESIGN["boxes"], bx["object_records"],
        bx["object_records"], "desc")
    add(f"R{n + 1:02d}", "Bounding boxes", "object records with a four-number bbox",
        DESIGN["boxes"], bx["valid_bbox"], bx["valid_bbox"], "desc")
    n += 2
    for s in SPLIT_NAMES:
        dh, di, dq = DESIGN["split"][s]
        v = res["splits"][s]
        add(f"R{n:02d}", f"Split {s}", "houses (splits.csv)", dh, v["houses"], v["houses"], "run")
        add(f"R{n + 1:02d}", f"Split {s}", "images", di, v["images"], v["images"], "run")
        add(f"R{n + 2:02d}", f"Split {s}", "QA pairs", dq, v["qa"], v["qa"], "run")
        n += 3
    design_cap = dict(DESIGN["capture"])
    for c in res["capture"]["rows"]:
        add(f"R{n:02d}", "Capture type", c["label"], design_cap[c["label"]], c["raw_exact"],
            c["norm_matched"], "desc",
            "raw: exact label; normalise(): values "
            + ", ".join([label(v) for v in c["matched_values"]] or ["none"]))
        n += 1
    add(f"R{n:02d}", "Names on box labels", "distinct names", DESIGN["box_names"],
        nm["raw_distinct"], nm["norm_distinct"], "desc",
        "raw: as written, the empty string not counted; normalise(): empty results not counted")
    add(f"R{n + 1:02d}", "Names on box labels", f"names with {MIN_INSTANCES} or more boxes",
        DESIGN["box_names_ge5"], nm["raw_ge_min_boxes"], nm["norm_ge_min_boxes"], "desc",
        "instance = one box (object record)")
    add(f"R{n + 2:02d}", "Names on box labels", f"names in {MIN_INSTANCES} or more images",
        DESIGN["box_names_ge5"], None, nm["norm_ge_min_images"], "info",
        "the other reading of 'instances'")
    return rows


def cross_checks(res, splits_in_manifest):
    """Checks against values recorded outside Design 3.1. Each: (id, text, ok, detail)."""
    out = []
    tr = {t: c for k, c in res["qa"]["type_by_split"].items()
          for s, t in [k.split("|", 1)] if s == "train"}
    eq = [t for t in QA_TYPES if tr.get(t, 0) == RECORDS["train_types"][t]]
    extra = sorted(t for t in tr if t not in QA_TYPES)
    out.append(("X01", "train split QA per type = addendum G2 (FT data step)",
                len(eq) == len(QA_TYPES) and not extra,
                f"{len(eq)} of {len(QA_TYPES)} types equal"
                + (f"; other types {', '.join(label(t) for t in extra)}" if extra else "")))
    on = res["obj_name"]
    out.append(("X02", "obj_name QA records = addendum C2 (1,877)",
                on["records"] == RECORDS["objname_qa"], fmt(on["records"])))
    out.append(("X03", "distinct obj_name answers, addendum C2 recorded 393 (reading not stated)",
                None, f"raw {fmt(on['distinct_answers_raw'])}, after normalise() "
                      f"{fmt(on['distinct_answers_norm'])}"))
    qs = res["questions"]
    out.append(("X04", "distinct question strings, addendum J1 recorded 2,956 (scope not stated)",
                None, f"raw {fmt(qs['distinct_raw'])}, after normalise() {fmt(qs['distinct_norm'])}"))
    hw = res["houses_without_closeup"]
    out.append(("X05", "houses without a close-up image = Design 3.5 (7 houses; EXP-N 61)",
                hw == RECORDS["no_closeup_houses"],
                f"{len(hw)} houses: {', '.join(hw) if hw else 'none'}"))
    sc = res["split_checks"]
    nh, mm = sc["houses_compared"], sc["n_images_mismatch"]
    out.append(("X06", "splits.csv n_images = images per house in the dataset", not mm,
                f"{nh - len(mm)} of {nh} houses agree" + (f"; differ: {', '.join(mm)}" if mm else "")))
    out.append(("X07", "splits.csv annotator_id = annotator_id of every image of the house",
                not sc["annotator_mismatch"],
                "all houses agree" if not sc["annotator_mismatch"]
                else f"{len(sc['annotator_mismatch'])} differ: {', '.join(sc['annotator_mismatch'])}"))
    out.append(("X08", "every dataset house is in splits.csv and the reverse",
                not sc["houses_missing_in_splits"] and not sc["houses_missing_in_dataset"],
                f"missing in splits.csv {sc['houses_missing_in_splits'] or 'none'}; "
                f"missing in the dataset {sc['houses_missing_in_dataset'] or 'none'}"))
    out.append(("X09", "the FT data manifest records this splits.csv (bytes sha256)",
                splits_in_manifest, "found" if splits_in_manifest else "not found as a value"))
    miss = {k: v for k, v in res["id_list_missing"].items() if v}
    out.append(("X10", "every qa_id of the nine id lists is in the dataset", not miss,
                "all found" if not miss else f"missing: {miss}"))
    out.append(("X11", "every qa_id of the excluded list is in the dataset",
                res["excluded_missing"] == 0, f"missing {res['excluded_missing']}"))
    out.append(("X12", "qa_id and image_id unique",
                not res["qa"]["duplicate_qa_ids"] and not res["images"]["duplicate_image_ids"],
                f"repeated qa_ids {len(res['qa']['duplicate_qa_ids'])}, "
                f"repeated image_ids {len(res['images']['duplicate_image_ids'])}"))
    return out


# ================================================================ report ==========
def render(res, rows, xchecks, meta):
    L = []
    w = L.append
    w(f"DATA-STATS {VERSION} | run {meta['run_utc']} | repo HEAD {meta['head'][:12]} | "
      f"normalise {meta['normalise_sha256'][:12]} | dataset {meta['dataset_sha256'][:12]}")
    w("")
    w("A. Design 3.1 values: as written (raw), after normalise(), and the difference "
      "(after normalise() minus Design 3.1)")
    w(f"{'id':<4} {'row':<20} {'reading':<52} {'Design':>7} {'raw':>7} {'normal.':>7} "
      f"{'diff':>6}  kind")
    for r in rows:
        w(f"{r['id']:<4} {r['row']:<20} {r['reading'][:52]:<52} {fmt(r['design']):>7} "
          f"{fmt(r['raw']):>7} {fmt(r['norm']):>7} {fmt_diff(r['diff']):>6}  {r['kind']}")
    w("   kind: run = a count the runs used or the gate-E design fixes; desc = descriptive; "
      "info = no Design value")
    by_note = defaultdict(list)
    for r in rows:
        if r["note"]:
            by_note[r["note"]].append(r["id"])
    for note, ids in by_note.items():
        span = ids[0] if len(ids) == 1 else f"{ids[0]} to {ids[-1]}"
        w(f"   {span}: {note}")
    cap = res["capture"]
    w("   capture_type as written: " + "; ".join(f"{label(k)} {fmt(v)}"
                                                for k, v in cap["raw_values"].items()))
    if cap["other_norm_values"]:
        w("   capture_type after normalise() matching no Design label: "
          + "; ".join(f"{label(k)} {fmt(v)}" for k, v in cap["other_norm_values"].items()))
    w("   QA types by split (after normalise()):")
    tbs = res["qa"]["type_by_split"]
    splits_present = [s for s in SPLIT_NAMES] + (["unassigned"] if any(
        k.startswith("unassigned|") for k in tbs) else [])
    types_present = sorted({k.split("|", 1)[1] for k in tbs})
    w("     " + f"{'type':<14}" + "".join(f"{s:>11}" for s in splits_present))
    for t in types_present:
        shown = t if t.isascii() else label(t)
        w("     " + f"{shown:<14}" + "".join(f"{fmt(tbs.get(f'{s}|{t}', 0)):>11}" for s in splits_present))
    nm = res["names"]
    w(f"   names on box labels: {fmt(nm['raw_forms_changed_by_normalise'])} raw forms change under "
      f"normalise(); {fmt(nm['merge_groups'])} names gather {fmt(nm['raw_forms_in_merge_groups'])} "
      f"raw forms; {fmt(nm['raw_forms_empty_after_normalise'])} raw forms become empty; "
      f"boxes with an empty name after normalise() {fmt(nm['boxes_empty_after_normalise'])} "
      f"(the empty string as written {fmt(nm['raw_empty_string_boxes'])})")
    if nm["changed_form_diagnostics"]:
        w("   changed raw forms hold: " + ", ".join(f"{k} {v}" for k, v in
                                                   nm["changed_form_diagnostics"].items()))
    bx = res["boxes"]
    w(f"   boxes: bbox with w and h above 0 {fmt(bx['valid_bbox_positive_wh'])}; "
      f"four-number bbox_pct {fmt(bx['valid_bbox_pct'])}; images with at least one box "
      f"{fmt(bx['images_with_objects'])}")
    w("")
    w("B. Cross-checks against other records (not Design 3.1 values)")
    for xid, text, ok, detail in xchecks:
        tag = "INFO" if ok is None else ("PASS" if ok else "FAIL")
        w(f"{tag}  {xid} {text}: {detail}")
    w("")
    em = res["empty"]
    w("C. QA pairs empty after normalise() (D301: counted as they are)")
    w(f"   items {len(em)}; by field: "
      + (", ".join(f"{k} {v}" for k, v in sorted(res["empty_by_field"].items())) or "none"))
    if res["empty_by_type_split"]:
        w("   by type and split: " + ", ".join(f"{k.replace('|', ' ')} {v}"
                                               for k, v in res["empty_by_type_split"].items()))
    for title, items in (("", em), ("Not empty after normalise() but no letter or digit "
                                    "(information)", res["no_content"])):
        if title:
            w(f"   {title}: {len(items)} items")
        if items:
            w(f"   {'qa_id':<14} {'type':<13} {'house':<6} {'split':<6} {'field':<16} "
              f"{'raw (code points)':<24} {'id lists':<16} {'excl.':<5} {'FT-VAL400':<9} EXP-R train")
            for r in items:
                rawcp = (r["a_raw_codepoints"] or r["q_raw_codepoints"])[:24]
                tshown = r["type"] if r["type"].isascii() else label(r["type"])
                w(f"   {str(r['qa_id']):<14} {tshown:<13} {str(r['house_id']):<6} {r['split']:<6} "
                  f"{r['fields']:<16} {rawcp:<24} {r['id_lists'][:16]:<16} "
                  f"{'yes' if r['excluded'] else 'no':<5} {'yes' if r['in_ft_val400'] else 'no':<9} "
                  f"{'yes' if r['exp_r_training_pair'] else 'no'}")
    w("")
    w(f"D. {ITEM_D301} (D301)")
    for tag, text in item_facts(res):
        w(f"{tag}  {text}")
    w("")
    st = res["structure"]
    w("E. Structure (counts only)")
    bits = []
    for k in ("non_dict_records", "non_dict_objects", "non_dict_qa", "objects_not_list",
              "qa_pairs_not_list", "images_without_objects", "images_without_qa",
              "non_string_question", "non_string_answer", "non_string_name"):
        bits.append(f"{k} {fmt(st[k])}")
    w("   " + "; ".join(bits))
    for k in ("missing_image_keys", "extra_image_keys", "missing_object_keys",
              "extra_object_keys", "missing_qa_keys", "extra_qa_keys"):
        if st[k]:
            w(f"   {k}: " + ", ".join(f"{kk} {fmt(v)}" for kk, v in sorted(st[k].items())))
    w(f"   QA pairs repeated on one image after normalise() (same type, question and answer): "
      f"{len(res['qa_duplicates'])} groups")
    w("")
    w("F. Verdict")
    run_bad = [r["id"] for r in rows if r["kind"] == "run" and r["diff"] != 0]
    x_bad = [x[0] for x in xchecks if x[2] is False]
    if run_bad or "X05" in x_bad or "X01" in x_bad:
        w("   RUN-DEFINING VALUES DIFFER: " + ", ".join(run_bad + [x for x in ("X01", "X05")
                                                            if x in x_bad])
          + ". Stop: send this printout before anything is recorded.")
    else:
        w("   Run-defining values (houses, images, QA pairs, the three splits, the train types,"
          " the houses without a close-up image): all equal to the gate-E design and the run records.")
    desc_bad = [r["id"] for r in rows if r["kind"] == "desc" and r["diff"] not in (0, None)]
    w("   Descriptive Design 3.1 values that differ after normalise(): "
      + (", ".join(desc_bad) if desc_bad else "none"))
    other_bad = [x for x in x_bad if x not in ("X01", "X05")]
    if other_bad:
        w("   Cross-checks that FAIL: " + ", ".join(other_bad))
    w(f"   Empty after normalise(): {len(em)} item(s)"
      + (f" ({', '.join(str(r['qa_id']) for r in em[:10])}"
         + (", ..." if len(em) > 10 else "") + ")" if em else ""))
    return "\n".join(L), bool(run_bad or "X05" in x_bad or "X01" in x_bad)


def item_facts(res):
    """PASS or FAIL lines for the facts PROJECT_STATE records about H13_0032_q2."""
    f = res["item_d301"]
    out = []
    if not f.get("present"):
        return [("FAIL", f"{ITEM_D301} is not in the dataset")]
    out.append(("PASS" if f["present"] == 1 else "FAIL", f"present in the dataset {f['present']} time(s)"))
    out.append(("PASS" if f["answer_empty"] else "FAIL",
                f"answer empty after normalise() (raw: {f['answer_raw_codepoints']})"))
    out.append(("PASS" if f["house_id"] == "H13" and f["split"] == "train" else "FAIL",
                f"house {f['house_id']}, split {f['split']} (so one of the EXP-R training pairs, D172)"))
    out.append(("PASS" if not f["id_lists"] else "FAIL",
                f"in no id list (lists holding it: {', '.join(f['id_lists']) or 'none'})"))
    out.append(("PASS" if not f["excluded"] else "FAIL", "not on probes/excluded_items.csv"))
    out.append(("INFO", f"type {f['type'] if f['type'].isascii() else label(f['type'])}; "
                        f"question empty after normalise(): "
                        f"{'yes' if f['question_empty'] else 'no'}"))
    return out


# ================================================================ inputs ==========
class Inputs:
    """The checks of Cell 0 and the parsed inputs for the run."""

    def __init__(self, root):
        self.root = Path(root)
        self.dataset_path = self.root / "data" / "banglaindoorvqa.json"
        self.splits_path = self.root / "data" / "splits.csv"
        self.manifest_path = self.root / "data" / "ft" / "ft_data_manifest.json"
        self.out_base = self.root / "results" / "data_stats"
        self.checks = []
        self.records = None
        self.splits = None
        self.id_lists = {}
        self.excluded = set()
        self.nf = None
        self.meta = {}
        self.splits_bytes_sha = None
        self.manifest = None

    def add(self, cid, ok, text):
        self.checks.append((cid, bool(ok), text))

    def run_all(self):
        steps = [self.c01_repo, self.c02_script, self.c03_tags, self.c04_normalise,
                 self.c05_dataset, self.c06_manifest, self.c07_splits]
        for f in steps:
            try:
                f()
            except Exception as e:  # a crash in a check is a FAIL of that check
                self.add(f.__name__[:3].upper(), False, f"{f.__name__}: error {type(e).__name__}: {e}")
        for i, name in enumerate(ID_LISTS):
            try:
                self.c_idlist(f"C{8 + i:02d}", name)
            except Exception as e:
                self.add(f"C{8 + i:02d}", False, f"probes/{name}.ids.txt: error {type(e).__name__}: {e}")
        for cid, f in (("C17", self.c17_excluded), ("C18", self.c18_out)):
            try:
                f(cid)
            except Exception as e:
                self.add(cid, False, f"error {type(e).__name__}: {e}")
        return all(ok for _, ok, _ in self.checks)

    # -- C01 the clone
    def c01_repo(self):
        code, top, err = git_text(["rev-parse", "--show-toplevel"])
        if code != 0 or Path(top).resolve() != REPO.resolve():
            return self.add("C01", False, f"repo: {REPO} is not the top of a git clone ({err or top})")
        _, branch, _ = git_text(["rev-parse", "--abbrev-ref", "HEAD"])
        _, head, _ = git_text(["rev-parse", "HEAD"])
        code, origin, _ = git_text(["rev-parse", "--verify", "--quiet", "refs/remotes/origin/main"])
        self.meta["head"] = head
        ok = branch == "main" and code == 0 and origin == head
        self.add("C01", ok, f"repo {REPO.name}: branch {branch}, HEAD {head[:12]}, origin/main "
                            f"{origin[:12] if code == 0 else 'missing'}"
                            + ("" if ok else " (need branch main, pulled: HEAD = origin/main)"))

    # -- C02 this script is the committed one
    def c02_script(self):
        me = Path(__file__).resolve()
        code_t, _, _ = git(["ls-files", "--error-unmatch", SCRIPT_REL])
        _, dirty, _ = git_text(["status", "--porcelain", "--", SCRIPT_REL])
        _, last, _ = git_text(["log", "-1", "--format=%H", "--", SCRIPT_REL])
        sha = sha256_bytes(me.read_bytes())
        self.meta["script_sha256"] = sha
        self.meta["script_commit"] = last
        ok = me == (REPO / SCRIPT_REL).resolve() and code_t == 0 and not dirty
        self.add("C02", ok, f"{SCRIPT_REL} version {VERSION}, sha256 {sha[:12]}, "
                            + ("committed and unmodified" if ok else
                               "NOT committed or modified: commit it, pull, run again")
                            + (f", last commit {last[:12]}" if last else ""))

    # -- C03 the two tags
    def c03_tags(self):
        ce, e, _ = git_text(["rev-parse", "--verify", "--quiet", "refs/tags/gate-E^{commit}"])
        cf, f, _ = git_text(["rev-parse", "--verify", "--quiet", "refs/tags/gate-F^{commit}"])
        ok = ce == 0 and e == GATE_E_COMMIT and cf == 0 and f.startswith(GATE_F_COMMIT_PREFIX)
        self.add("C03", ok, f"tags: gate-E -> {e[:12] if ce == 0 else 'missing'} "
                            f"(expected {GATE_E_COMMIT[:12]}), gate-F -> {f[:12] if cf == 0 else 'missing'} "
                            f"(expected {GATE_F_COMMIT_PREFIX})"
                            + ("" if ok else "; run git fetch --tags"))

    # -- C04 normalise.py
    def c04_normalise(self):
        p = REPO / NORMALISE_REL
        work = sha256_bytes(p.read_bytes())
        code, blob, _ = git(["show", f"gate-F:{NORMALISE_REL}"])
        tagged = sha256_bytes(blob) if code == 0 else None
        _, dirty, _ = git_text(["status", "--porcelain", "--", NORMALISE_REL])
        _, msg, _ = git_text(["for-each-ref", "--format=%(contents)", "refs/tags/gate-F"])
        in_msg = work[:12] in msg
        ok = work == NORMALISE_SHA256 and tagged == work and not dirty
        if ok:
            self.nf = load_normalise(p)
            probe = self.nf(None) == "" and self.nf("  Ab। ") == "ab"
            ok = ok and probe
        self.meta["normalise_sha256"] = work
        self.add("C04", ok, f"{NORMALISE_REL} sha256 {work[:12]}; gate-F blob "
                            f"{tagged[:12] if tagged else 'missing'}; pin {NORMALISE_SHA256[:12]}; "
                            f"{'unmodified' if not dirty else 'MODIFIED'}; named in the gate-F tag "
                            f"message: {'yes' if in_msg else 'no'}"
                            + ("; imported from this file" if ok else ""))

    # -- C05 the dataset (D300)
    def c05_dataset(self):
        p = self.dataset_path
        if not p.is_file():
            return self.add("C05", False, f"dataset {p} not found")
        b = p.read_bytes()
        sha = sha256_bytes(b)
        self.meta["dataset_sha256"] = sha
        self.meta["dataset_bytes"] = len(b)
        if sha != DATASET_SHA256:
            return self.add("C05", False, f"dataset sha256 {sha} is not the version of record "
                                          f"{DATASET_SHA256} (D300). Stop.")
        data = json.loads(b.decode("utf-8-sig"))
        if not isinstance(data, list):
            return self.add("C05", False, "dataset top level is not a list")
        self.records = data
        self.add("C05", True, f"data/banglaindoorvqa.json sha256 {sha[:12]} = version of record "
                              f"(D300); {len(b) / 1e6:.1f} MB; {fmt(len(data))} image records")

    # -- C06 the FT data manifest holds the same dataset sha256
    def c06_manifest(self):
        p = self.manifest_path
        if not p.is_file():
            return self.add("C06", False, f"{p} not found")
        self.manifest = json.loads(p.read_bytes().decode("utf-8-sig"))
        hits = find_value_paths(self.manifest, DATASET_SHA256)
        self.add("C06", bool(hits), "data/ft/ft_data_manifest.json holds the dataset sha256"
                                    + (f" at {hits[0]}" if hits else ": NOT found"))

    # -- C07 splits.csv
    def c07_splits(self):
        p = self.splits_path
        if not p.is_file():
            return self.add("C07", False, f"{p} not found")
        b = p.read_bytes()
        self.splits_bytes_sha = sha256_bytes(b)
        rows = parse_splits_text(b.decode("utf-8-sig"))
        content = splits_content_sha256(rows)
        houses = [r["house_id"] for r in rows]
        bad_split = sorted({r["split"] for r in rows} - set(SPLIT_NAMES))
        ints = all(re.fullmatch(r"\d+", r["n_images"]) for r in rows)
        per = Counter(r["split"] for r in rows)
        ok = (content == SPLITS_CONTENT_SHA256 and len(rows) == 68 and len(set(houses)) == 68
              and not bad_split and ints)
        if ok:
            self.splits = {r["house_id"]: {"annotator_id": r["annotator_id"], "split": r["split"],
                                           "n_images": int(r["n_images"])} for r in rows}
        same_bytes = self.splits_bytes_sha == SPLITS_BYTES_SHA256
        self.meta["splits_sha256"] = self.splits_bytes_sha
        self.add("C07", ok, f"data/splits.csv content {'=' if content == SPLITS_CONTENT_SHA256 else '!='} "
                            f"the attached copy (bytes {'equal' if same_bytes else 'differ'}, sha256 "
                            f"{self.splits_bytes_sha[:12]}); {len(rows)} houses: "
                            + ", ".join(f"{s} {per.get(s, 0)}" for s in SPLIT_NAMES)
                            + (f"; bad split values {bad_split}" if bad_split else ""))

    # -- C08 to C16 the nine id lists
    def c_idlist(self, cid, name):
        rel = f"probes/{name}.ids.txt"
        p = REPO / rel
        if not p.is_file():
            return self.add(cid, False, f"{rel} not found")
        b = p.read_bytes()
        code, blob, _ = git(["show", f"gate-E:{rel}"])
        same = code == 0 and blob == b
        _, dirty, _ = git_text(["status", "--porcelain", "--", rel])
        ids = parse_id_list(b.decode("utf-8-sig"))
        ok = same and not dirty and len(ids) == ID_LISTS[name] and len(set(ids)) == len(ids)
        if ok:
            self.id_lists[name] = set(ids)
        self.add(cid, ok, f"{rel}: {fmt(len(ids))} qa_ids (expected {fmt(ID_LISTS[name])}), "
                          f"{'= gate-E' if same else 'NOT equal to gate-E'}"
                          + ("" if len(set(ids)) == len(ids) else ", REPEATED ids"))

    # -- C17 the excluded list
    def c17_excluded(self, cid):
        rel = "probes/excluded_items.csv"
        b = (REPO / rel).read_bytes()
        sha = sha256_bytes(b)
        code, blob, _ = git(["show", f"gate-E:{rel}"])
        rows = list(csv.DictReader(io.StringIO(b.decode("utf-8-sig"))))
        ids = [r.get("qa_id", "").strip() for r in rows]
        ok = sha == EXCLUDED_SHA256 and code == 0 and blob == b and len(ids) == EXCLUDED_ROWS \
            and all(ids) and len(set(ids)) == len(ids)
        if ok:
            self.excluded = set(ids)
        self.add(cid, ok, f"{rel}: sha256 {sha[:12]} (pin {EXCLUDED_SHA256[:12]}), "
                          f"{'= gate-E' if code == 0 and blob == b else 'NOT equal to gate-E'}, "
                          f"{len(ids)} rows (expected {EXCLUDED_ROWS})")

    # -- C18 the output folder
    def c18_out(self, cid):
        base = check_out_base(self.out_base, self.root, REPO)
        parent = base.parent
        ok = parent.is_dir() and os.access(str(parent), os.W_OK)
        self.add(cid, ok, f"output: Drive results/data_stats/run_<UTC>/ "
                          f"({'results/ is writable' if ok else 'results/ missing or not writable'}); "
                          f"not inside the repo or Drive data/, outputs/, scored/, probes/, human/")


def print_checks(inp):
    for cid, ok, text in inp.checks:
        print(f"{'PASS' if ok else 'FAIL'}  {cid} {text}")
    n_ok = sum(1 for _, ok, _ in inp.checks if ok)
    print(f"CHECK: {n_ok} of {len(inp.checks)} PASS" + ("" if n_ok == len(inp.checks) else
                                                          " -- stop and send these lines"))


# ================================================================ commands ========
def cmd_check(args):
    print(f"DATA-STATS {VERSION}: input checks (reads only)")
    inp = Inputs(args.root)
    ok = inp.run_all()
    print_checks(inp)
    return 0 if ok else 1


def write_csv(path, rows, fields):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


def cmd_run(args):
    print(f"DATA-STATS {VERSION}: input checks, then the counts")
    inp = Inputs(args.root)
    ok = inp.run_all()
    print_checks(inp)
    if not ok:
        print("RUN STOPPED: an input check failed; nothing was written.")
        return 1
    _, status_before, _ = git_text(["status", "--porcelain"])
    now = dt.datetime.now(dt.timezone.utc)
    run_utc = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = dict(inp.meta, run_utc=run_utc, version=VERSION)

    res = compute(inp.records, inp.splits, inp.id_lists, inp.excluded, inp.nf)
    rows = build_table(res)
    in_manifest = bool(find_value_paths(inp.manifest, inp.splits_bytes_sha))
    xchecks = cross_checks(res, in_manifest)
    text, run_bad = render(res, rows, xchecks, meta)
    print()
    print(text)

    base = check_out_base(inp.out_base, inp.root, REPO)
    out = base / f"run_{now.strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True, exist_ok=False)
    written = []

    def put(name, writer):
        p = out / name
        assert p.parent == out
        writer(p)
        written.append(p)

    put("data_stats_report.txt", lambda p: p.write_text(text + "\n", encoding="utf-8"))
    put("data_stats_table.csv", lambda p: write_csv(
        p, [dict(r, design=r["design"], raw=r["raw"], norm=r["norm"]) for r in rows],
        ["id", "row", "reading", "design", "raw", "norm", "diff", "kind", "note"]))
    item_fields = ["qa_id", "image_id", "house_id", "split", "type", "fields", "q_raw_codepoints",
                   "a_raw_codepoints", "id_lists", "in_probe_list", "in_ft_val400", "excluded",
                   "exp_r_training_pair"]
    put("empty_after_normalise.csv", lambda p: write_csv(p, res["empty"], item_fields))
    put("no_content_after_normalise.csv", lambda p: write_csv(p, res["no_content"], item_fields))
    put("empty_box_names.csv", lambda p: write_csv(p, res["empty_box_names"],
                                                   ["object_id", "image_id", "house_id", "raw"]))
    put("qa_duplicates_after_normalise.csv", lambda p: write_csv(
        p, res["qa_duplicates"], ["image_id", "type", "qa_ids"]))
    name_rows = []
    for r in res["_name_table"]:
        forms = sorted(r["raw_forms"].items(), key=lambda kv: (-kv[1], kv[0]))
        name_rows.append({"name_after_normalise": r["name_norm"], "boxes": r["boxes"],
                          "images": r["images"],
                          f"boxes_ge_{MIN_INSTANCES}": "yes" if r["boxes"] >= MIN_INSTANCES else "no",
                          "n_raw_forms": len(forms),
                          "raw_forms": " | ".join(f"{visible(f)} x{c}" for f, c in forms)})
    put("TEXT_box_names.csv", lambda p: write_csv(
        p, name_rows, ["name_after_normalise", "boxes", "images", f"boxes_ge_{MIN_INSTANCES}",
                       "n_raw_forms", "raw_forms"]))
    # Counts only: no qa_id, no name, no house id. The one file that may go to repo results/.
    counts = {"version": VERSION, "run_utc": run_utc,
              "dataset_sha256": meta["dataset_sha256"], "normalise_sha256": meta["normalise_sha256"],
              "table": [{k: r[k] for k in ("id", "row", "reading", "design", "raw", "norm", "diff",
                                           "kind")} for r in rows],
              "cross_checks": [{"id": x[0], "check": x[1], "ok": x[2]} for x in xchecks],
              "empty_after_normalise": len(res["empty"]),
              "empty_by_field": res["empty_by_field"],
              "empty_by_type_split": res["empty_by_type_split"],
              "no_content_after_normalise": len(res["no_content"]),
              "qa_type_by_split": res["qa"]["type_by_split"],
              "names": {k: v for k, v in res["names"].items()},
              "boxes": res["boxes"]}
    put("data_stats_counts.json", lambda p: p.write_text(
        json.dumps(counts, ensure_ascii=False, indent=1), encoding="utf-8"))
    full = {k: v for k, v in res.items() if not k.startswith("_")}
    full["structure"] = {k: (dict(v) if isinstance(v, Counter) else v)
                         for k, v in res["structure"].items()}
    put("data_stats.json", lambda p: p.write_text(
        json.dumps({"meta": meta, "results": full,
                    "checks": [{"id": c, "ok": o, "text": t} for c, o, t in inp.checks],
                    "cross_checks": [{"id": x[0], "check": x[1], "ok": x[2], "detail": x[3]}
                                     for x in xchecks]},
                   ensure_ascii=False, indent=1, default=str), encoding="utf-8"))
    manifest = {"version": VERSION, "run_utc": run_utc, "meta": meta,
                "inputs": {"dataset": str(inp.dataset_path), "splits": str(inp.splits_path),
                           "ft_manifest": str(inp.manifest_path), "repo": str(REPO)},
                "outputs": {p.name: sha256_bytes(p.read_bytes()) for p in written}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    written.append(out / "manifest.json")

    _, status_after, _ = git_text(["status", "--porcelain"])
    repo_ok = status_after == status_before
    print()
    print(f"{'PASS' if repo_ok else 'FAIL'}  repo working tree unchanged by the run")
    print(f"RUN DONE: wrote {len(written)} files to {out}")
    print("   send this printout; data_stats_counts.json holds counts only (repo-safe); "
          "TEXT_box_names.csv holds names (Drive only)")
    return 2 if (run_bad or not repo_ok) else 0


# ================================================================ selftest ========
def _st_records():
    Z = "‌"

    def img(iid, house, ann, cap, objs, qas):
        return {"image_id": iid, "house_id": house, "annotator_id": ann, "image_file": iid + ".jpg",
                "image_w": 100, "image_h": 100, "capture_type": cap, "room_type": "x",
                "objects": [{"object_id": o[0], "bangla_name": o[1], "bbox": o[2], "bbox_pct": o[2]}
                            for o in objs],
                "qa_pairs": [{"qa_id": q[0], "type": q[1], "question": q[2], "answer": q[3]}
                             for q in qas]}
    return [
        img("HA_0001", "HA", "A1", "close-up",
            [("o1", "কখগ", [0, 0, 10, 10]), ("o2", "কখগ।", [1, 1, 5, 5])],
            [("HA_0001_q1", "obj_name", "প্রশ্ন ক?", "কখগ।"), ("HA_0001_q2", "count", "প্রশ্ন খ?", "।")]),
        img("HA_0002", "HA", "A1", "Close-up ",
            [("o3", " কখগ", [0, 0, 0, 5]), ("o4", "ঘঙচ", [0, 0, 3])],
            [("HA_0002_q1", "obj_name", "প্রশ্ন ক?", "ঘঙচ"), ("HA_0002_q2", "object", "প্রশ্ন গ", "-")]),
        img("HB_0001", "HB", "A2", "overview", [("o5", "", [0, 0, 1, 1])],
            [("HB_0001_q1", "scene_id", "?", "রুম")]),
        img("HC_0001", "HC", "A3", "group",
            [("o6", "কখ" + Z + "গ", [0, 0, 2, 2]), ("o7", "ক্" + Z + "ষ", [0, 0, 2, 2])],
            [("HC_0001_q1", " Count", "প্রশ্ন খ?", "২"), ("HC_0001_q2", " Count", "প্রশ্ন খ?", "২")]),
        img("HC_0002", "HC", "A3", "closeup", [], [("HC_0002_q1", "obj_name", "প্রশ্ন ক", "কখগ")]),
    ]


def cmd_selftest(args):
    print(f"DATA-STATS {VERSION}: self test on made-up data (no Drive, no git)")
    nf = load_normalise(REPO / NORMALISE_REL)
    results = []

    def t(name, got, want):
        results.append(got == want)
        print(f"{'OK  ' if got == want else 'FAIL'} {name}: got {got!r}"
              + ("" if got == want else f", expected {want!r}"))

    splits = {"HA": {"annotator_id": "A1", "split": "train", "n_images": 2},
              "HB": {"annotator_id": "A2", "split": "val", "n_images": 1},
              "HC": {"annotator_id": "A3", "split": "test", "n_images": 2}}
    lists = {"P-NAME": {"HA_0001_q1", "HC_0002_q1"}, "R-TEST": {"HC_0001_q1"}}
    r = compute(_st_records(), splits, lists, {"HA_0002_q2"}, nf, min_instances=4)
    t("image records", r["images"]["records"], 5)
    t("houses", r["houses"]["distinct"], 3)
    t("QA records", r["qa"]["records"], 8)
    t("type labels as written", len(r["qa"]["types_raw"]), 5)
    t("type labels after normalise()", r["qa"]["types_norm"],
      {"count": 3, "obj_name": 3, "object": 1, "scene_id": 1})
    t("empty after normalise()", [(e["qa_id"], e["fields"]) for e in r["empty"]],
      [("HA_0001_q2", "answer"), ("HB_0001_q1", "question")])
    t("empty items: split, lists, excluded",
      [(e["split"], e["id_lists"], e["excluded"], e["exp_r_training_pair"]) for e in r["empty"]],
      [("train", "none", False, True), ("val", "none", False, False)])
    t("no letter or digit after normalise()", [(e["qa_id"], e["excluded"]) for e in r["no_content"]],
      [("HA_0002_q2", True)])
    t("non-empty question and answer", r["qa"]["nonempty_q_and_a"], 6)
    t("repeated QA on one image", [d["qa_ids"] for d in r["qa_duplicates"]], ["HC_0001_q1|HC_0001_q2"])
    t("boxes", (r["boxes"]["object_records"], r["boxes"]["valid_bbox"],
                r["boxes"]["valid_bbox_positive_wh"], r["boxes"]["images_with_objects"]), (7, 6, 5, 4))
    n = r["names"]
    t("names as written (empty string not counted)", (n["raw_distinct"], n["raw_empty_string_boxes"]), (6, 1))
    t("names after normalise()", n["norm_distinct"], 3)
    t("boxes per name after normalise()", sorted(x["boxes"] for x in r["_name_table"]), [1, 1, 4])
    t("names with 4 or more boxes (raw, normalise())", (n["raw_ge_min_boxes"], n["norm_ge_min_boxes"]), (0, 1))
    t("images per name after normalise()", sorted(r["_name_images_norm"].values()), [1, 1, 3])
    t("merge groups and their raw forms", (n["merge_groups"], n["raw_forms_in_merge_groups"]), (1, 4))
    t("raw forms changed by normalise()", n["raw_forms_changed_by_normalise"], 3)
    t("diagnostics", n["changed_form_diagnostics"],
      {"edge_punctuation_only": 1, "edge_whitespace": 1, "zwnj_or_zwj": 1})
    t("ZWNJ after hasant kept as its own name", "ক্‌ষ" in r["_name_boxes_raw"]
      and any(x["name_norm"] == "ক্‌ষ" for x in r["_name_table"]), True)
    t("boxes with an empty name", len(r["empty_box_names"]), 1)
    cap = {c["label"]: (c["raw_exact"], c["norm_matched"]) for c in r["capture"]["rows"]}
    t("capture types (exact, after normalise() by letters)", cap,
      {"overview": (1, 1), "group": (1, 1), "close-up": (1, 3)})
    t("houses without a close-up image", r["houses_without_closeup"], ["HB"])
    t("splits", {s: (v["houses"], v["images"], v["qa"]) for s, v in r["splits"].items()},
      {"train": (1, 2, 4), "val": (1, 1, 1), "test": (1, 2, 3)})
    t("split checks clean", (r["split_checks"]["n_images_mismatch"], r["split_checks"]["annotator_mismatch"]),
      ([], []))
    t("question strings (raw, normalise())", (r["questions"]["distinct_raw"], r["questions"]["distinct_norm"]),
      (5, 3))
    t("obj_name (records, raw, normalise())", (r["obj_name"]["records"], r["obj_name"]["distinct_answers_raw"],
                                              r["obj_name"]["distinct_answers_norm"]), (3, 3, 2))
    t("table rows have no missing kind", all(x["kind"] in ("run", "desc", "info") for x in build_table(r)), True)

    # error paths
    bad = [
        {"image_id": "X1", "house_id": "HX", "annotator_id": "A1", "capture_type": "group",
         "objects": [], "qa_pairs": [{"qa_id": "D1", "type": "count", "question": "প্রশ্ন", "answer": "১"}]},
        {"image_id": "X1", "house_id": "HY", "annotator_id": "A1", "capture_type": "group",
         "objects": None, "qa_pairs": [{"qa_id": "D1", "type": "count", "question": "প্রশ্ন", "answer": "১"}]},
        {"image_id": "X2", "house_id": "HX", "annotator_id": "A1", "capture_type": "close-up",
         "qa_pairs": [{"qa_id": "D2", "type": "count", "question": None, "answer": 3}]},
    ]
    sp2 = {"HX": {"annotator_id": "A9", "split": "train", "n_images": 5},
           "HZ": {"annotator_id": "A1", "split": "test", "n_images": 1}}
    r2 = compute(bad, sp2, {}, set(), nf)
    t("repeated image_id and qa_id", (r2["images"]["duplicate_image_ids"], r2["qa"]["duplicate_qa_ids"]),
      (["X1"], ["D1"]))
    sc = r2["split_checks"]
    t("split mismatches", (sc["houses_missing_in_splits"], sc["houses_missing_in_dataset"],
                           sc["n_images_mismatch"], sc["annotator_mismatch"]),
      (["HY"], ["HZ"], ["HX"], ["HX"]))
    t("missing keys counted", r2["structure"]["missing_image_keys"]["objects"], 1)
    t("None question is empty, int answer counted", ([e["qa_id"] for e in r2["empty"]],
                                                     r2["structure"]["non_string_answer"]), (["D2"], 1))
    t("item D301 absent is reported", r2["item_d301"], {"present": 0})

    # helpers
    a = "house_id,annotator_id,split,n_images\r\nH1,A1,train,3\r\n"
    b = "﻿house_id,annotator_id,split,n_images\nH1,A1,train,3\n"
    t("split content hash ignores BOM and line ends",
      splits_content_sha256(parse_splits_text(a)) == splits_content_sha256(parse_splits_text(b.lstrip("﻿"))),
      True)
    t("id list parsing", parse_id_list("qa_id\nA_1\n\n# note\nA_2\n"), ["A_1", "A_2"])
    t("value paths in a manifest", find_value_paths({"x": {"sha256": "AB12"}, "y": ["cd"]}, "ab12"),
      ["$.x.sha256"])
    t("visible()", visible("a‌b "), "[a\\u200cb ]")
    t("has_content", (has_content("-"), has_content("২"), has_content("​"), has_content("ক")),
      (False, True, False, True))
    root = Path("/x/drive/proj")
    for nm_, base, want in (("good", root / "results" / "data_stats", True),
                            ("under data/", root / "data" / "data_stats", False),
                            ("inside repo", REPO / "results" / "data_stats", False)):
        try:
            check_out_base(base, root, REPO)
            got = True
        except ValueError:
            got = False
        t(f"output guard ({nm_})", got, want)
    n_ok = sum(results)
    print(f"SELFTEST: {n_ok} of {len(results)} passed")
    return 0 if n_ok == len(results) else 1


# ================================================================ main ============
def main(argv=None):
    ap = argparse.ArgumentParser(description="DATA-STATS: Design 3.1 dataset values after normalise()")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "run"):
        s = sub.add_parser(name)
        s.add_argument("--root", required=True, help="the project folder in Drive "
                                                     "(holds data/ and results/)")
    sub.add_parser("selftest")
    args = ap.parse_args(argv)
    return {"check": cmd_check, "run": cmd_run, "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
