#!/usr/bin/env python3
"""
asset11_options.py  (ASSET-11)
BanglaIndoorVQA main paper: forced-choice option sets, the HT-6 review sheet and
P-FCE. Design v2.1 section 4.7; DESIGN_ADDENDUM C4 and C5; decisions D195 to D202.

The rules are in src/fc_options.py. This file is the command line.

Commands
    inputs     One PASS or FAIL line per input. Run it first (Cell 0).
    selftest   Every rule on a made-up lexicon. Needs no project file.
    build      STAGE 1. Candidates, draft option sets and the HT-6 review sheet.
    apply      STAGE 2. The HT-6 log -> final option sets, P-FCE.json, its id
               list and its size for configs/conditions.yaml.
    check      Read the final files from disk again and repeat every check.

Where the files go (D195)
    Drive human/HT-6/   ht6_review.html (the sheet, with the pictures),
                        fc_candidates.csv, fc_options_DRAFT.jsonl,
                        asset11_state.json; then ht6_log.csv (from the reviewer),
                        ht6_swaps.csv, ht6_flags.csv
    Drive probes/       fc_options.jsonl (final), P-FCE.json, asset11_report.txt,
                        probe_manifest_11.json
    repo probes/        P-FCE.ids.txt (qa_ids only)
    repo configs/       conditions.yaml: the one line of P-FCE

Typical use in Colab
    python scripts/asset11_options.py inputs   --root "$ROOT"
    python scripts/asset11_options.py selftest
    python scripts/asset11_options.py build    --root "$ROOT"
    ... one person does HT-6 and puts ht6_log.csv into Drive human/HT-6/ ...
    python scripts/asset11_options.py apply    --root "$ROOT"
    python scripts/asset11_options.py check    --root "$ROOT"

Nothing here reads a model answer. Printed lines hold counts and lexicon names.
The sheet holds photos of homes beside their gold names: it stays in Drive
human/HT-6/ and on the reviewer's own computer only (D202).
"""

import argparse
import base64
import csv
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DEFAULT / "src"))

import scoring                       # noqa: E402  (the one scorer)
from normalise import normalise      # noqa: E402  (the one normalise)
import fc_options as fc              # noqa: E402  (the rules of ASSET-11)

SCRIPT_VERSION = "asset11-v1 2026-10-15"
ROOT_DEFAULT = os.environ.get("BIVQA_ROOT", "/content/drive/MyDrive/BanglaIndoorVQA")

PROBE = "P-CLUST"
EXCLUDED_MIN_ROWS = 19   # rows of probes/excluded_items.csv on 2026-10-15; rows are never removed (C4)
PFCE = "P-FCE"
OPTIONS_FILE = "fc_options.jsonl"
DRAFT_FILE = "fc_options_DRAFT.jsonl"
SHEET_FILE = "ht6_review.html"
LOG_FILE = "ht6_log.csv"
STATE_FILE = "asset11_state.json"
TEMPLATE_FILE = "asset11_ht6_page.html"

LOG_COLUMNS = ("seq", "qa_id", "cluster", "gold_id", "gold_bn", "verdict", "rejected_ids",
               "rejected_bn", "reasons", "final_ids", "final_bn", "flag", "note", "reviewer",
               "draft_sha", "rules")
REASONS = {"name": "also a right name for the main object",
           "visible": "also a main object in the picture",
           "other": "other reason (see the note)"}


# Values of the two seeded ranks, recorded when the rules were written. They are
# checked in every run, so the order cannot differ from one session to the next.
RECORDED_ORDER = {"T01_0001_q1": [1, 2, 3, 0], "T01_0002_q1": [3, 0, 2, 1],
                  "H01_0004_q1": [1, 2, 0, 3]}
RECORDED_TIE = {"N_selftest01": "7d1ae3b0960db0f3", "N_0616a73a8b": "3cb0d2ebb907d11c"}


def ranks_as_recorded():
    return all(fc.shown_order(q) == v for q, v in RECORDED_ORDER.items()) and \
        all(fc.tie_rank(i)[:16] == v for i, v in RECORDED_TIE.items())


class Stop(Exception):
    """A condition that must stop the command. The message says what to do."""


def say(msg=""):
    print(msg, flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def mark(ok):
    return "PASS  " if ok else "FAIL  "


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------

class Paths:
    def __init__(self, repo, root, images=None):
        self.repo, self.root = Path(repo), Path(root)
        r, d = self.repo, self.root
        self.lexicon = r / "lexicon" / "lexicon.csv"
        self.conditions = r / "configs" / "conditions.yaml"
        self.prompts = r / "configs" / "prompts.yaml"
        self.template = r / "scripts" / TEMPLATE_FILE
        self.ids_pclust = r / "probes" / "P-CLUST.ids.txt"
        self.ids_pilot = r / "probes" / "PILOT.ids.txt"
        self.ids_pfce = r / "probes" / "P-FCE.ids.txt"
        self.excluded = r / "probes" / "excluded_items.csv"
        self.pclust = d / "probes" / "P-CLUST.json"
        self.pname = d / "probes" / "P-NAME.json"
        self.pilot_json = d / "probes" / "PILOT.json"
        self.frozen_copy = d / "human" / "HT-1" / "frozen" / "lexicon.csv"
        self.images = Path(images) if images else d / "data" / "images_1024"
        self.ht6 = d / "human" / "HT-6"
        self.sheet = self.ht6 / SHEET_FILE
        self.draft = self.ht6 / DRAFT_FILE
        self.candidates = self.ht6 / "fc_candidates.csv"
        self.state = self.ht6 / STATE_FILE
        self.log = self.ht6 / LOG_FILE
        self.swaps = self.ht6 / "ht6_swaps.csv"
        self.flags = self.ht6 / "ht6_flags.csv"
        self.options = d / "probes" / OPTIONS_FILE
        self.pfce = d / "probes" / "P-FCE.json"
        self.report = d / "probes" / "asset11_report.txt"
        self.manifest = d / "probes" / "probe_manifest_11.json"


def write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_text(path, text):
    write_bytes(path, text.encode("utf-8"))


def write_json(path, obj):
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=1) + "\n")


def read_json(path):
    with open(str(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_csv(path, columns, rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(columns), lineterminator="\r\n")
    w.writeheader()
    for row in rows:
        w.writerow({c: row.get(c, "") for c in columns})
    write_bytes(path, ("﻿" + buf.getvalue()).encode("utf-8"))


def write_final(path, data, replace):
    """Write a final file. An existing file with other content is kept unless
    --replace is given: a forced-choice run may already have used it."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() == data:
            return "unchanged"
        if not replace:
            raise Stop(
                "%s exists and would change. If no forced-choice run has used it yet, run "
                "the command again with --replace. If a run has used it, stop and ask in "
                "the ASSET-11 chat: the harness refuses answers made with other options."
                % path)
        write_bytes(path, data)
        return "REPLACED"
    write_bytes(path, data)
    return "written"


def git(repo, *args):
    try:
        r = subprocess.run(["git", "-C", str(repo)] + list(args), capture_output=True,
                           text=True, timeout=60)
        return r.returncode, r.stdout.strip()
    except Exception:
        return None, ""


def frozen_status(p):
    """Is repo lexicon/lexicon.csv the file frozen at Gate A? Returns
    ('same' | 'differs' | 'unknown', words). Two witnesses: the Git tag gate-A
    and the copy in Drive human/HT-1/frozen/."""
    code, _ = git(p.repo, "diff", "--quiet", "gate-A", "--", "lexicon/lexicon.csv")
    copy_ok = None
    if p.frozen_copy.exists():
        copy_ok = fc.file_sha256(p.frozen_copy) == fc.file_sha256(p.lexicon)
    if code == 1 or copy_ok is False:
        return "differs", "it DIFFERS from %s. Nobody edits lexicon.csv after Gate A." % (
            "the tag gate-A" if code == 1 else "Drive human/HT-1/frozen/lexicon.csv")
    if code == 0 or copy_ok:
        return "same", " and ".join(x for x in (
            "same as the tag gate-A" if code == 0 else "",
            "same as Drive human/HT-1/frozen/lexicon.csv" if copy_ok else "") if x)
    return "unknown", ("cannot be confirmed: the tag gate-A is not in this clone and Drive "
                       "human/HT-1/frozen/lexicon.csv is missing. Run  git fetch --tags  in "
                       "the repo.")


def read_yaml(path):
    import yaml  # PyYAML, in Colab by default
    with open(str(path), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------
# Loading the world: lexicon, probe, sets
# --------------------------------------------------------------------------

def load_probe_file(path, name):
    if not Path(path).exists():
        raise Stop("Missing probe file: %s" % path)
    data = read_json(path)
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        raise Stop("%s: no 'items' list. Is this the ASSET-5 file %s.json?" % (path, name))
    header = {k: v for k, v in data.items() if k != "items"}
    if header.get("n") not in (None, len(items)):
        raise Stop("%s: header n=%s but %d items." % (path, header.get("n"), len(items)))
    ids = [it.get("qa_id") for it in items]
    if len(set(ids)) != len(ids) or not all(ids):
        raise Stop("%s: an empty or repeated qa_id." % path)
    return header, items


def load_world(p):
    if not p.lexicon.exists():
        raise Stop("Missing lexicon: %s (pull the repo)." % p.lexicon)
    status, words = frozen_status(p)
    if status == "differs":
        raise Stop("repo lexicon/lexicon.csv is not the frozen file: %s Restore it from Git "
                   "(git checkout gate-A -- lexicon/lexicon.csv)." % words)
    say("  lexicon file: %s" % words)
    lex = scoring.load_lexicon(p.lexicon)
    builder = fc.Builder(lex)
    header, items = load_probe_file(p.pclust, PROBE)
    return lex, builder, header, items


def make_sets(builder, items, rejected=None):
    sets = [builder.make(it, (rejected or {}).get(it["qa_id"], ())) for it in items]
    table = {}
    for s in sets:
        bn, en = builder.records(s)
        table[(s.qa_id, "bn")] = bn
        if en is not None:
            table[(s.qa_id, "en")] = en
    return sets, table


def table_lines(items, table, options_key="options"):
    lines = []
    for it in items:
        for lang in fc.LANGS:
            rec = table.get((it["qa_id"], lang))
            if rec is not None:
                lines.append(fc.dump_record(rec, options_key))
    return lines


def draft_fingerprint(builder, sets):
    """One hash for everything the review depends on: the rules version, the
    candidate lists of every gold object, and each item's gold and order."""
    golds, used = {}, set()
    for s in sets:
        if s.gold_id not in golds:
            q = builder.queues(s.gold_id)
            golds[s.gold_id] = {"sib": q["sib"], "out": q["out"], "draft": list(s.draft)}
            used.update(q["sib"] + q["out"] + [s.gold_id])
    objs = builder.objects
    names = {i: [fc.text_bn(objs[i]), fc.text_en(objs[i]),
                 str(objs[i].row.get("en_unique", "")).strip().lower()] for i in sorted(used)}
    return fc.canonical_sha({"rules": fc.RULES_VERSION, "golds": golds, "names": names,
                             "items": [[s.qa_id, s.gold_id, list(s.order)] for s in sets]})


def review_order(builder, sets):
    """The order of the sheet: cluster by cluster, then gold by gold, so that
    the reviewer sees the same four names many times in a row."""
    objs = builder.objects
    n_cl = Counter(objs[s.gold_id].cluster_id for s in sets)
    n_gold = Counter(s.gold_id for s in sets)
    return sorted(sets, key=lambda s: (-n_cl[objs[s.gold_id].cluster_id],
                                       objs[s.gold_id].cluster_id,
                                       -n_gold[s.gold_id], s.gold_id, s.qa_id))


def name_of(builder, oid, with_freq=False):
    o = builder.objects[oid]
    text = "%s [%s]" % (fc.text_bn(o), fc.text_en(o))
    return text + " (%d)" % fc.freq_of(o) if with_freq else text


def pilot_ids(p):
    if p.ids_pilot.exists():
        return fc.read_ids(p.ids_pilot), "repo probes/PILOT.ids.txt"
    if p.pilot_json.exists():
        return [it["qa_id"] for it in load_probe_file(p.pilot_json, "PILOT")[1]], \
            "Drive probes/PILOT.json"
    return None, ""


def excluded_ids(p):
    if not p.excluded.exists():
        return None
    with open(p.excluded, "r", encoding="utf-8-sig", newline="") as fh:
        return [(row.get("qa_id") or "").strip() for row in csv.DictReader(fh)
                if (row.get("qa_id") or "").strip()]


def excluded_info(p):
    """Rows and SHA-256 of repo probes/excluded_items.csv, for the records."""
    ids = excluded_ids(p)
    if ids is None:
        return {"rows": None, "sha256": ""}
    return {"rows": len(ids), "sha256": fc.file_sha256(p.excluded)}


# --------------------------------------------------------------------------
# Printing the checks
# --------------------------------------------------------------------------

def print_checks(builder, items, table, p, rejected=None, out=None):
    """Print every check of fc.check_sets and the per-cluster and PILOT lines.
    Returns (all passed, numbers)."""
    def emit(line=""):
        say(line)
        if out is not None:
            out.append(line)

    res = fc.check_sets(builder, items, table, rejected)
    num = res["numbers"]
    ok_all = ranks_as_recorded()
    emit("  " + mark(ok_all) + "the seeded ranks (order of the options, ties) give the values "
         "recorded with the rules: the same in every session")
    for _key, text, passed in res["checks"]:
        ok_all = ok_all and passed
        emit("  " + mark(passed) + text)
    emit("        P-FCE by cluster (in P-FCE / en_unique items / all items):")
    for cl in sorted(num["items_by_cluster"], key=lambda c: -num["items_by_cluster"][c]):
        emit("          %4d / %3d / %3d   %s"
             % (num["pfce_by_cluster"].get(cl, 0), num["en_unique_by_cluster"].get(cl, 0),
                num["items_by_cluster"][cl], cl))
    emit("        gold position inside P-FCE: %s"
         % ", ".join("%s %d" % (fc.EN_LABELS[k], num["gold_position_pfce"][k]) for k in range(4)))
    emit("        shape of the sets: %s"
         % "; ".join("%d sets with %s" % (v, k) for k, v in sorted(num["shape"].items())))
    emit("        sets where a distractor is a more specific kind of the gold (kept by "
         "Design 4.7, marked for the reviewer): %d" % num["n_kind_of_gold"])
    pil, src = pilot_ids(p)
    pfce = set(num["pfce_ids"])
    item_ids = {it["qa_id"] for it in items}
    if pil is None:
        ok_all = False
        emit("  " + mark(False) + "PILOT: no id list found (repo probes/PILOT.ids.txt or Drive "
             "probes/PILOT.json), so the PILOT line cannot be printed")
    else:
        in_cl = [q for q in pil if q in item_ids]
        passed = len(in_cl) >= 20
        ok_all = ok_all and passed
        emit("  " + mark(passed) + "PILOT (%s, %d ids): %d items are in P-CLUST by qa_id; %d of "
             "them are in P-FCE%s"
             % (src, len(pil), len(in_cl), sum(1 for q in in_cl if q in pfce),
                "" if len(in_cl) == 20 else
                ". Design 3.4 draws 20 P-CLUST items for the naming part; a number above 20 "
                "means that items of the closed part are P-CLUST members too. Send this line "
                "to the ASSET-11 chat"))
    exc = excluded_ids(p)
    if exc is None:
        emit("        excluded items (D110): repo probes/excluded_items.csv not found; "
             "not counted")
    else:
        emit("        excluded items (D110): %d rows in the list (sha256 %s...); %d are in "
             "P-CLUST, %d in P-FCE. Excluded items keep their option sets: they are run and "
             "never scored."
             % (len(exc), fc.file_sha256(p.excluded)[:12], sum(1 for q in exc if q in item_ids),
                sum(1 for q in exc if q in pfce)))
    return ok_all, num


def print_gold_table(builder, sets, out=None):
    """One line per gold object: the draft set. Lexicon names only."""
    objs = builder.objects
    n_gold = Counter(s.gold_id for s in sets)
    first = {}
    for s in sets:
        first.setdefault(s.gold_id, s)
    lines = ["  items  gold  ->  distractor 1 | distractor 2 | distractor 3      "
             "(s = same cluster, o = outside)"]
    cluster = None
    for gid in sorted(first, key=lambda g: (objs[g].cluster_id, -n_gold[g], g)):
        s = first[gid]
        if objs[gid].cluster_id != cluster:
            cluster = objs[gid].cluster_id
            lines.append("  -- %s" % cluster)
        parts = ["%s:%s" % ("s" if builder.role_of(gid, d) == fc.ROLE_SIBLING else "o",
                            name_of(builder, d)) for d in s.draft]
        tag = "" if builder.in_pfce(gid, [gid] + list(s.draft)) else "   (not in P-FCE)"
        lines.append("  %5d  %s  ->  %s%s" % (n_gold[gid], name_of(builder, gid),
                                              " | ".join(parts), tag))
    for line in lines:
        say(line)
    if out is not None:
        out.extend(lines)


def boundary_ties(builder, sets):
    """Places in a draft set that the tie rule (rule A) decided."""
    objs = builder.objects
    n_gold = Counter(s.gold_id for s in sets)
    found = []
    seen = set()
    for s in sets:
        if s.gold_id in seen:
            continue
        seen.add(s.gold_id)
        q = builder.queues(s.gold_id)
        for name, queue in (("cluster", q["sib"]), ("outside", q["out"][:q["n_same"]])):
            took = [c for c in queue if c in s.draft]
            left = [c for c in queue if c not in s.draft]
            if took and left and fc.freq_of(objs[took[-1]]) == fc.freq_of(objs[left[0]]):
                found.append((s.gold_id, n_gold[s.gold_id], name, took[-1], left[0]))
    return found


# --------------------------------------------------------------------------
# The HT-6 sheet
# --------------------------------------------------------------------------

def page_data(builder, sets, items, sha, with_images):
    objs = builder.objects
    by_qa = {it["qa_id"]: it for it in items}
    ordered = review_order(builder, sets)
    clusters, golds, used = [], {}, set()
    rows = []
    for s in ordered:
        gold = objs[s.gold_id]
        if gold.cluster_id not in clusters:
            clusters.append(gold.cluster_id)
        if s.gold_id not in golds:
            q = builder.queues(s.gold_id)
            pool = q["sib"] + q["out"]
            golds[s.gold_id] = {
                "sib": q["sib"], "out": q["out"], "draft": list(s.draft),
                "kinds": [i for i in pool if fc.is_kind_of(gold, objs[i])],
                "en": str(gold.row.get("en_unique", "")).strip().lower() == "yes"}
            used.update(pool)
            used.add(s.gold_id)
        question = " ".join(str(by_qa[s.qa_id].get("question_bn") or "").split())
        written = normalise(by_qa[s.qa_id]["answer_bn"])       # shown only when it is a merged spelling
        rows.append([s.qa_id, s.gold_id, question, list(s.order), clusters.index(gold.cluster_id),
                     "" if written == gold.name_bn else written])
    return ordered, {
        "sha": sha, "rules": fc.RULES_VERSION, "images": bool(with_images),
        "labels_bn": list(fc.BN_LABELS), "labels_en": list(fc.EN_LABELS),
        "log_columns": list(LOG_COLUMNS), "log_name": LOG_FILE, "clusters": clusters,
        "objects": {i: [fc.text_bn(objs[i]), fc.text_en(objs[i])] for i in sorted(used)},
        "golds": golds, "items": rows}


def render_page(template, data, thumbs):
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    images = "\n".join('<script type="text/plain" id="im-%d">%s</script>' % (i, b64)
                       for i, b64 in enumerate(thumbs))
    for token in ("/*__DATA__*/", "<!--__IMAGES__-->"):
        if template.count(token) != 1:
            raise Stop("scripts/%s is not the ASSET-11 template (the mark %s is missing). "
                       "Pull the repo." % (TEMPLATE_FILE, token))
    return template.replace("/*__DATA__*/", blob).replace("<!--__IMAGES__-->", images)


def page_json(html):
    """The data block of a sheet, read back from the HTML text."""
    m = re.search(r'<script type="application/json" id="d">(.*?)</script>', html, re.S)
    if not m:
        raise Stop("The sheet holds no data block.")
    return json.loads(m.group(1))


def load_image_tools(p):
    """The one image loader of the project (src/images.py) and Pillow."""
    try:
        import images as imglib
    except Exception as exc:
        raise Stop("src/images.py cannot be imported (%s: %s). It is the image loader of "
                   "ASSET-8; pull the repo." % (type(exc).__name__, exc))
    try:
        from PIL import Image
    except Exception as exc:
        raise Stop("Pillow is missing (%s). In Colab it is installed by default." % exc)
    if not p.images.exists():
        raise Stop("Missing the prepared images: %s" % p.images)
    manifest = imglib.read_manifest(p.images)
    return imglib, Image, manifest


def make_thumbs(p, ordered, items, px, quality):
    imglib, Image, manifest = load_image_tools(p)
    by_qa = {it["qa_id"]: it for it in items}
    missing = sorted({by_qa[s.qa_id]["image_file"] for s in ordered
                      if manifest.get(by_qa[s.qa_id]["image_file"]) is None})
    if missing:
        raise Stop("%d P-CLUST images are not in %s (first: %s). All 4,406 images were "
                   "prepared on 2026-10-09 (D175); is --root the right Drive folder?"
                   % (len(missing), p.images, missing[0]))
    thumbs, total = [], 0
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    for n, s in enumerate(ordered, 1):
        data, _sha = imglib.load_prepared(p.images, by_qa[s.qa_id]["image_file"], manifest)
        im = Image.open(io.BytesIO(data))
        im = im.convert("RGB")
        im.thumbnail((px, px), resample)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        thumbs.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        total += buf.tell()
        if n % 100 == 0 or n == len(ordered):
            say("    %d of %d pictures read and reduced" % (n, len(ordered)))
    return thumbs, total


# --------------------------------------------------------------------------
# Command: build (STAGE 1)
# --------------------------------------------------------------------------

def cmd_build(a):
    p = Paths(a.repo, a.root, a.images_1024)
    say("=" * 78)
    say("ASSET-11 STAGE 1: candidates, draft option sets, HT-6 sheet")
    say("script %s   rules %s   %s" % (SCRIPT_VERSION, fc.RULES_VERSION, utc_now()))
    say("=" * 78)
    if not p.template.exists():
        raise Stop("Missing scripts/%s. Upload it to the repo with this script." % TEMPLATE_FILE)

    say("\n[1] Inputs")
    lex, builder, header, items = load_world(p)
    say("  lexicon: %s" % lex.describe())
    say("  %s: %d items (created %s)" % (p.pclust.name, len(items), header.get("created_utc", "?")))
    sets, table = make_sets(builder, items)
    sha = draft_fingerprint(builder, sets)
    n_gold = len({s.gold_id for s in sets})
    say("  every gold answer joined to its object by the scorer: %d of %d, %d gold objects"
        % (len(sets), len(items), n_gold))
    old = read_json(p.state) if p.state.exists() else {}
    if old.get("draft_sha") not in (None, sha) and p.log.exists():
        raise Stop(
            "Drive human/HT-6/ holds an HT-6 log made with another draft (%s...), and this "
            "build gives %s.... The lexicon, the probe file or the rules changed. Move the "
            "old folder content into a folder named old and ask in the ASSET-11 chat before "
            "a new review." % (str(old.get("draft_sha"))[:12], sha[:12]))

    report = []
    say("\n[2] Draft sets, one line per gold object (lexicon names only)")
    print_gold_table(builder, sets, report)
    ties = boundary_ties(builder, sets)
    say("  places decided by the tie rule (equal freq_obj_name, rule A): %d gold objects"
        % len({t[0] for t in ties}))
    for gid, n, where, took, left in ties:
        say("    %s (%d item%s), %s: %s taken before %s"
            % (name_of(builder, gid), n, "" if n == 1 else "s", where,
               name_of(builder, took, True), name_of(builder, left, True)))
    generic = Counter()
    for s in sets:
        for g in builder.queues(s.gold_id)["generic"]:
            generic[(s.gold_id, g)] += 1
    say("  candidates skipped as generic names of the gold (rule B): %d gold objects"
        % len({k[0] for k in generic}))
    for (gid, g), n in sorted(generic.items(), key=lambda kv: (-kv[1], kv[0])):
        say("    %s never gets %s (%d item%s)"
            % (name_of(builder, gid), name_of(builder, g), n, "" if n == 1 else "s"))

    say("\n[3] Checks on the draft")
    ok, num = print_checks(builder, items, table, p, out=report)
    again = draft_fingerprint(builder, make_sets(fc.Builder(scoring.load_lexicon(p.lexicon)),
                                                items)[0])
    same = again == sha
    ok = ok and same
    say("  " + mark(same) + "a second build from the files gives the same draft (%s...)" % sha[:12])
    if not ok:
        say("\nRESULT: STAGE 1 NOT OK: a check failed. No sheet was written. Send this output "
            "to the ASSET-11 chat.")
        return 1

    say("\n[4] HT-6 sheet")
    ordered, data = page_data(builder, sets, items, sha, not a.no_images)
    if a.no_images:
        thumbs, nbytes = [], 0
        say("  --no-images: a TEST sheet without pictures. It cannot be used for HT-6.")
    else:
        say("  reading %d prepared pictures from %s" % (len(ordered), p.images))
        thumbs, nbytes = make_thumbs(p, ordered, items, a.thumb_px, a.thumb_quality)
    html = render_page(p.template.read_text(encoding="utf-8"), data, thumbs)
    back = page_json(html)
    if back != json.loads(json.dumps(data)):
        raise Stop("Internal error: the data block of the sheet does not read back.")

    say("\n[5] Writing to %s" % p.ht6)
    p.ht6.mkdir(parents=True, exist_ok=True)
    write_text(p.sheet, html)
    write_text(p.draft, "\n".join(table_lines(items, table, fc.DRAFT_OPTIONS_KEY)) + "\n")
    cand_rows = []
    for gid in sorted({s.gold_id for s in sets},
                      key=lambda g: (builder.objects[g].cluster_id, g)):
        g, q = builder.objects[gid], builder.queues(gid)
        draft = next(s.draft for s in sets if s.gold_id == gid)
        cand_rows.append({
            "gold_id": gid, "gold_bn": fc.text_bn(g), "gold_en": fc.text_en(g),
            "cluster": g.cluster_id, "room_scope": fc.scope_of(g),
            "n_items": sum(1 for s in sets if s.gold_id == gid),
            "en_unique": g.row.get("en_unique", ""),
            "draft_1": name_of(builder, draft[0]), "draft_2": name_of(builder, draft[1]),
            "draft_3": name_of(builder, draft[2]),
            "skipped_generic": " | ".join(name_of(builder, i) for i in q["generic"]),
            "cluster_queue": " | ".join(name_of(builder, i, True) for i in q["sib"]),
            "outside_queue_first_12": " | ".join(name_of(builder, i, True) for i in q["out"][:12]),
            "outside_same_room": q["n_same"]})
    write_csv(p.candidates, list(cand_rows[0].keys()), cand_rows)
    state = {
        "asset": "ASSET-11", "stage": 1, "script_version": SCRIPT_VERSION,
        "rules_version": fc.RULES_VERSION, "built_utc": utc_now(), "draft_sha": sha,
        "n_items": len(items), "n_gold_objects": n_gold, "n_pfce_draft": num["n_pfce"],
        "gold_position_draft": num["gold_position_bn"], "checks_passed": bool(ok),
        "images_embedded": not a.no_images, "thumb_px": a.thumb_px,
        "thumb_quality": a.thumb_quality,
        "inputs": {"lexicon_sha256": fc.file_sha256(p.lexicon),
                   "pclust_sha256": fc.file_sha256(p.pclust),
                   "pclust_created_utc": header.get("created_utc", ""),
                   "excluded_items": excluded_info(p)},
        "sheet": {"file": SHEET_FILE, "bytes": len(html.encode("utf-8")),
                  "sha256": fc.sha256_text(html)},
        "repo_commit": git(p.repo, "rev-parse", "HEAD")[1]}
    write_json(p.state, state)
    mb = len(html.encode("utf-8")) / 1e6
    say("  %s   %.1f MB, %d sets%s" % (SHEET_FILE, mb, len(ordered),
                                      "" if a.no_images else ", pictures %d px, %.1f MB before "
                                      "encoding" % (a.thumb_px, nbytes / 1e6)))
    say("  %s   %d records (key '%s': the harness refuses this file on purpose)"
        % (DRAFT_FILE, len(table), fc.DRAFT_OPTIONS_KEY))
    say("  fc_candidates.csv   %d gold objects" % len(cand_rows))
    say("  %s   draft %s" % (STATE_FILE, sha))
    if mb > 60:
        say("  note: the sheet is large. If the browser is slow, build again with "
            "--thumb-px 512.")
    say("\nRESULT: STAGE 1 OK. Draft P-FCE size %d (final after HT-6)." % num["n_pfce"])
    if not a.no_images:
        say("Next: one reviewer downloads %s from Drive human/HT-6/, follows "
            "HT6_instructions.md, and uploads %s to the same folder." % (SHEET_FILE, LOG_FILE))
    return 0


# --------------------------------------------------------------------------
# The HT-6 log
# --------------------------------------------------------------------------

def read_log(path):
    if not Path(path).exists():
        raise Stop("Missing the HT-6 log: %s. The reviewer downloads it from the sheet "
                   "(button 'Download the log') and uploads it to Drive human/HT-6/." % path)
    try:
        with open(str(path), "r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            cols = [(c or "").strip() for c in (reader.fieldnames or [])]
            missing = [c for c in LOG_COLUMNS if c not in cols]
            if missing:
                raise Stop("%s is not an HT-6 log: column(s) missing: %s."
                           % (path, ", ".join(missing)))
            return [{(k or "").strip(): (v or "").strip() for k, v in row.items()
                     if k is not None} for row in reader]
    except UnicodeDecodeError:
        raise Stop("%s is not saved as UTF-8. Use the file the sheet downloads, unchanged."
                   % path)


def judge_log(builder, items, rows, sha):
    """Check every row of the log against the rules. Returns the decisions, or
    stops with a list of what is wrong, in plain words."""
    by_qa = {it["qa_id"]: it for it in items}
    seen, problems, open_rows = {}, [], []
    out = {"rejected": {}, "reasons": {}, "notes": {}, "flags": [], "moot": 0, "reviewers": Counter(),
           "seq": {}}

    def bad(row, text):
        problems.append("row %s (%s): %s" % (row.get("seq") or "?", row.get("qa_id") or "?", text))

    for row in rows:
        qa = row.get("qa_id", "")
        if qa not in by_qa:
            bad(row, "this qa_id is not in P-CLUST")
            continue
        if qa in seen:
            bad(row, "this item is in the log twice")
            continue
        seen[qa] = row
        if row["draft_sha"] != sha:
            bad(row, "made with another version of the sheet (draft %s..., now %s...)"
                % (row["draft_sha"][:12], sha[:12]))
            continue
        if not row["verdict"]:
            open_rows.append(row)
            continue
        if row["verdict"] not in ("ok", "swap"):
            bad(row, "the verdict must be ok or swap, found %r" % row["verdict"])
            continue
        ids = [x for x in row["rejected_ids"].split("|") if x]
        reasons = [x for x in row["reasons"].split("|") if x]
        if len(ids) != len(reasons) or len(set(ids)) != len(ids):
            bad(row, "rejected_ids and reasons do not fit together")
            continue
        if any(r not in REASONS for r in reasons):
            bad(row, "an unknown reason; the three reasons are %s" % ", ".join(REASONS))
            continue
        if row["flag"] not in ("0", "1"):
            bad(row, "flag must be 0 or 1")
            continue
        if ("other" in reasons or row["flag"] == "1") and not row["note"]:
            bad(row, "'other reason' and the tick box both need a note")
            continue
        if not row["reviewer"]:
            bad(row, "the reviewer's initials are missing")
            continue
        try:
            s = builder.make(by_qa[qa], ids)
        except fc.OptionError as exc:
            bad(row, str(exc))
            continue
        if row["gold_id"] != s.gold_id:
            bad(row, "the gold object in the log is not the one the scorer resolves")
            continue
        if row["final_ids"].split("|") != s.ids_shown():
            bad(row, "the four options in the log are not what the rules give for these "
                     "rejections. The log was changed by hand: open the sheet, load the log, "
                     "confirm the set again and download a new log")
            continue
        if (row["verdict"] == "swap") != s.changed:
            bad(row, "the verdict does not fit the options")
            continue
        out["moot"] += len(s.moot)
        out["rejected"][qa] = list(s.rejected)
        out["reasons"][qa] = {i: r for i, r in zip(ids, reasons) if i in s.rejected}
        out["notes"][qa] = row["note"]
        out["seq"][qa] = row["seq"]
        out["reviewers"][row["reviewer"]] += 1
        if row["flag"] == "1":
            out["flags"].append(qa)
    if open_rows:
        problems.insert(0, "%d of %d sets are not reviewed yet (the first is set %s, %s). The "
                           "review is not finished%s"
                        % (len(open_rows), len(by_qa), open_rows[0].get("seq") or "?",
                           open_rows[0].get("qa_id"),
                           ": this is a partial log." if len(open_rows) > 1 else "."))
    missing = [q for q in by_qa if q not in seen]
    if missing:
        problems.append("%d of %d items are not in the log at all (first: %s). Use the "
                        "complete log, not a partial one." % (len(missing), len(by_qa), missing[0]))
    if problems:
        shown = "\n  - ".join(problems[:20])
        more = "" if len(problems) <= 20 else "\n  ... and %d more" % (len(problems) - 20)
        raise Stop("The HT-6 log cannot be applied: %d problem(s).\n  - %s%s\n"
                   "Nothing was written. The reviewer opens the sheet, loads this log with "
                   "'Load a saved log', finishes or repairs the rows named above, and "
                   "downloads a new log." % (len(problems), shown, more))
    if len(out["reviewers"]) != 1:
        raise Stop("The log names %d reviewers (%s). HT-6 is done by one person (Design 10)."
                   % (len(out["reviewers"]), ", ".join(sorted(out["reviewers"]))))
    return out


def swap_rows(builder, sets, items, judged):
    """One row per replaced distractor: the logged change of Design 4.7."""
    objs = builder.objects
    rows = []
    for s in sets:
        if not s.changed:
            continue
        cluster = objs[s.gold_id].cluster_id
        for r in s.rejected:
            rest = [x for x in s.rejected if x != r]
            slot = builder.fill(s.gold_id, rest)[1].index(r)
            new = s.slots[slot]
            place = s.order.index(slot + 1)
            rows.append({
                "seq": judged["seq"].get(s.qa_id, ""), "qa_id": s.qa_id, "cluster": cluster,
                "gold_id": s.gold_id, "gold_bn": fc.text_bn(objs[s.gold_id]),
                "removed_id": r, "removed_bn": fc.text_bn(objs[r]),
                "removed_en": fc.text_en(objs[r]), "removed_role": builder.role_of(s.gold_id, r),
                "was_in_draft": "yes" if r in s.draft else "no",
                "reason": judged["reasons"][s.qa_id].get(r, ""),
                "final_id": new, "final_bn": fc.text_bn(objs[new]),
                "final_en": fc.text_en(objs[new]), "final_role": builder.role_of(s.gold_id, new),
                "place": "%s / %s" % (fc.BN_LABELS[place], fc.EN_LABELS[place]),
                "note": judged["notes"].get(s.qa_id, "")})
    return rows


# --------------------------------------------------------------------------
# Checks that use the harness and the prompts (final files only)
# --------------------------------------------------------------------------

def _lines_in_order(text, options):
    """True if the four option texts end four lines of the prompt, in order."""
    lines = text.split("\n")
    at = -1
    for opt in options:
        hit = [i for i, line in enumerate(lines)
               if i > at and line.endswith(opt) and len(line) - len(opt) <= 4]
        if not hit:
            return False
        at = hit[0]
    return True


def harness_checks(p, items, pfce_ids, repo=True):
    """PASS/FAIL lines on the files as the harness and build_prompt see them.
    repo=False leaves out the two lines about the repo files."""
    out = []
    try:
        import harness
        import prompts as promptlib
    except Exception as exc:
        return [("the harness (src/harness.py) can be imported: %s: %s"
                 % (type(exc).__name__, exc), False)]
    try:
        opt = harness.load_options(str(p.options))
        n_bn = sum(1 for (_q, lang) in opt["table"] if lang == "bn")
        n_en = len(opt["table"]) - n_bn
        have = all((it["qa_id"], "bn") in opt["table"] for it in items) and \
            all((q, "en") in opt["table"] for q in pfce_ids)
        out.append(("harness.load_options reads %s: %d Bangla and %d English sets; no item "
                    "of P-CLUST or P-FCE is without its set" % (OPTIONS_FILE, n_bn, n_en),
                    have and n_bn == len(items) and n_en == len(pfce_ids)))
    except Exception as exc:
        return out + [("harness.load_options reads %s: %s" % (OPTIONS_FILE, exc), False)]
    try:
        try:
            harness.load_options(str(p.draft))
            out.append(("the harness refuses the DRAFT file", False))
        except harness.Stop:
            out.append(("the harness refuses the DRAFT file %s (so no run can use it)"
                        % DRAFT_FILE, True))
        except FileNotFoundError:
            pass
    except Exception:
        pass
    try:
        prompts = promptlib.load_prompts(p.prompts)
        cfg = promptlib.load_conditions(p.conditions, prompts)
    except Exception as exc:
        return out + [("configs/prompts.yaml and conditions.yaml load: %s" % exc, False)]
    conds = cfg["conditions"]
    try:
        work = harness.build_work("asset11-check", conds["N-FCB0"], items, prompts, None, opt, {})
        good = sum(1 for w, it in zip(work, items)
                   if _lines_in_order(w["prompt_text"], opt["table"][(it["qa_id"], "bn")]))
        out.append(("the harness builds an N-FCB0 prompt for every P-CLUST item, with the four "
                    "options in the order of the file: %d of %d" % (good, len(items)),
                    len(work) == len(items) and good == len(items)))
    except Exception as exc:
        out.append(("the harness builds the N-FCB0 prompts: %s" % exc, False))
    try:
        by_qa = {it["qa_id"]: it for it in items}
        good = 0
        for q in pfce_ids:
            opts = opt["table"][(q, "en")]
            pr = promptlib.build_prompt(conds["N-FCE0"], by_qa[q], prompts,
                                        question_en="What is the main object here?", options=opts)
            good += _lines_in_order(pr["text"], opts)
        out.append(("build_prompt accepts every English set (N-FCE0 template, a stand-in "
                    "question): %d of %d" % (good, len(pfce_ids)), good == len(pfce_ids)))
    except Exception as exc:
        out.append(("build_prompt accepts the English sets: %s" % exc, False))
    try:
        probe = harness.load_probe(str(p.pfce))
        ids = [it["qa_id"] for it in probe["items"]]
        out.append(("harness.load_probe reads P-FCE.json: %d items, the P-FCE items in "
                    "P-CLUST order" % len(ids), ids == list(pfce_ids)))
        if not repo:
            return out
        want = (cfg["probes"].get(PFCE) or {})
        cfg_ok = want.get("file") == p.pfce.name and want.get("expected_n") == len(ids)
        out.append(("configs/conditions.yaml names %s with expected_n %d (it says: file %s, "
                    "expected_n %s)" % (p.pfce.name, len(ids), want.get("file"),
                                        want.get("expected_n")), cfg_ok))
        if cfg_ok:
            harness.check_probe_rule("main", cfg, probe, ["N-FCE", "N-FCE0"], False, False)
            out.append(("stage main will accept P-FCE.json for N-FCE and N-FCE0", True))
    except Exception as exc:
        out.append(("P-FCE.json and its line in conditions.yaml: %s" % exc, False))
    if p.ids_pfce.exists():
        out.append(("repo probes/P-FCE.ids.txt holds the same %d qa_ids" % len(pfce_ids),
                    fc.read_ids(p.ids_pfce) == list(pfce_ids)))
    else:
        out.append(("repo probes/P-FCE.ids.txt exists (the apply command writes it)", False))
    return out


def check_final(p, builder, items, rejected=None, out=None, repo=True):
    """Read the final files from disk and run every check on them."""
    def emit(line):
        say(line)
        if out is not None:
            out.append(line)

    if not p.options.exists():
        raise Stop("Missing %s. Stage 2 (apply) writes it." % p.options)
    table = fc.read_option_sets(p.options)
    ok, num = print_checks(builder, items, table, p, rejected, out)
    for text, passed in harness_checks(p, items, num["pfce_ids"], repo):
        ok = ok and passed
        emit("  " + mark(passed) + text)
    return ok, num, table


# --------------------------------------------------------------------------
# Command: apply (STAGE 2)
# --------------------------------------------------------------------------

def pfce_file(header, items, pfce_ids, created):
    keep = set(pfce_ids)
    chosen = [it for it in items if it["qa_id"] in keep]
    return {
        "probe": PFCE, "n": len(chosen), "seed": fc.SEED,
        "sampling": "every P-CLUST item whose gold is en_unique = yes and whose four "
                    "English option labels differ, after HT-6 (Design 4.7; D199)",
        "source_probe": PROBE, "rules_version": fc.RULES_VERSION,
        "dataset_sha256": header.get("dataset_sha256", ""), "created_utc": created,
        "qa_ids_sha256": fc.sha256_text("\n".join(it["qa_id"] for it in chosen)),
        "items": chosen}


def cmd_apply(a):
    p = Paths(a.repo, a.root)
    log_path = p.log
    say("=" * 78)
    say("ASSET-11 STAGE 2: HT-6 log -> final option sets, P-FCE")
    say("script %s   rules %s   %s" % (SCRIPT_VERSION, fc.RULES_VERSION, utc_now()))
    say("=" * 78)

    say("\n[1] Inputs and the draft")
    if not p.state.exists():
        raise Stop("Missing %s: stage 1 (build) has not run for this Drive folder." % p.state)
    state = read_json(p.state)
    lex, builder, header, items = load_world(p)
    say("  lexicon: %s" % lex.describe())
    say("  %s: %d items" % (p.pclust.name, len(items)))
    for key, path in (("lexicon_sha256", p.lexicon), ("pclust_sha256", p.pclust)):
        if (state.get("inputs") or {}).get(key) != fc.file_sha256(path):
            raise Stop("%s is not the file stage 1 read (its SHA-256 changed). Nothing was "
                       "written; ask in the ASSET-11 chat." % path)
    draft_sets, _draft_table = make_sets(builder, items)
    sha = draft_fingerprint(builder, draft_sets)
    if sha != state.get("draft_sha"):
        raise Stop("The draft built now (%s...) is not the draft of stage 1 (%s...). The "
                   "lexicon, P-CLUST.json or the rules changed after the sheet was made. "
                   "Nothing was written; ask in the ASSET-11 chat."
                   % (sha[:12], str(state.get("draft_sha"))[:12]))
    say("  draft %s... is the draft the sheet was made from" % sha[:12])
    if not state.get("images_embedded") and not a.allow_test_sheet:
        raise Stop("The sheet of stage 1 was a TEST sheet without pictures (--no-images). "
                   "HT-6 is a check against the pictures. Run build again without that flag.")

    say("\n[2] HT-6 log: %s" % log_path)
    rows = read_log(log_path)
    judged = judge_log(builder, items, rows, sha)
    reviewer = next(iter(judged["reviewers"]))
    sets, table = make_sets(builder, items, judged["rejected"])
    changed = [s for s in sets if s.changed]
    reason_n = Counter(r for d in judged["reasons"].values() for r in d.values())
    report = ["ASSET-11 report   script %s   rules %s   %s"
              % (SCRIPT_VERSION, fc.RULES_VERSION, utc_now()), "draft %s" % sha, "",
              "HT-6 (Design 4.7, D201)"]

    def tell(line):
        say(line)
        report.append(line)

    tell("  %d rows, all reviewed, by one reviewer (%s)" % (len(rows), reviewer))
    tell("  sets left as drafted: %d   sets changed: %d   distractors replaced: %d"
         % (len(sets) - len(changed), len(changed), sum(reason_n.values())))
    for code in REASONS:
        tell("    %-8s %3d   %s" % (code, reason_n.get(code, 0), REASONS[code]))
    tell("  rejections that changed nothing (ignored): %d" % judged["moot"])
    tell("  sets flagged for the team lead: %d" % len(judged["flags"]))
    by_gold = Counter(builder.objects[s.gold_id].name_bn for s in changed)
    if by_gold:
        tell("  changed sets by gold object: "
             + ", ".join("%s %d" % (k, v) for k, v in by_gold.most_common()))
    report += ["", "Checks on the final files"]

    say("\n[3] Writing the final files")
    lines = table_lines(items, table)
    pre = fc.check_sets(builder, items, table, judged["rejected"])
    failed = [text for _key, text, passed in pre["checks"] if not passed]
    if failed or not ranks_as_recorded():
        raise Stop("The final sets fail %d check(s), so nothing was written:\n  - %s\n"
                   "Send this output to the ASSET-11 chat."
                   % (len(failed), "\n  - ".join(failed) or "the seeded ranks changed"))
    num = pre["numbers"]
    pfce_ids = num["pfce_ids"]
    created = utc_now()
    if p.pfce.exists():
        old = read_json(p.pfce)
        new = pfce_file(header, items, pfce_ids, old.get("created_utc", created))
        if old == new:
            created = old.get("created_utc", created)
    pf = pfce_file(header, items, pfce_ids, created)
    st1 = write_final(p.options, ("\n".join(lines) + "\n").encode("utf-8"), a.replace)
    st2 = write_final(p.pfce, (json.dumps(pf, ensure_ascii=False, indent=1) + "\n").encode("utf-8"),
                      a.replace)
    say("  Drive probes/%s   %d records (%d Bangla, %d English)   %s"
        % (OPTIONS_FILE, len(lines), num["n_bn"], num["n_en"], st1))
    say("  Drive probes/%s   %d items   %s" % (p.pfce.name, len(pfce_ids), st2))
    sw = swap_rows(builder, sets, items, judged)
    sw_cols = ["seq", "qa_id", "cluster", "gold_id", "gold_bn", "removed_id", "removed_bn",
               "removed_en", "removed_role", "was_in_draft", "reason", "final_id", "final_bn",
               "final_en", "final_role", "place", "note"]
    write_csv(p.swaps, sw_cols, sw)
    by_qa = {s.qa_id: s for s in sets}
    fl = [{"seq": judged["seq"].get(q, ""), "qa_id": q,
           "gold_bn": fc.text_bn(builder.objects[by_qa[q].gold_id]),
           "set_changed": "yes" if by_qa[q].changed else "no", "note": judged["notes"].get(q, "")}
          for q in judged["flags"]]
    write_csv(p.flags, ["seq", "qa_id", "gold_bn", "set_changed", "note"], fl)
    say("  Drive human/HT-6/%s   %d rows (one per replaced distractor)" % (p.swaps.name, len(sw)))
    say("  Drive human/HT-6/%s   %d flagged sets (for the team lead; nothing is excluded here)"
        % (p.flags.name, len(fl)))

    line = "  P-FCE:         {file: %s,         expected_n: %d}" % (p.pfce.name, len(pfce_ids))
    say("\n[4] Repo files (in the clone %s)" % p.repo)
    if not a.no_repo:
        write_text(p.ids_pfce, "\n".join(pfce_ids) + "\n")
        with open(p.conditions, "r", encoding="utf-8", newline="") as fh:
            text = fh.read()
        new_text = fc.patch_conditions_text(text, p.pfce.name, len(pfce_ids))
        import yaml
        before, after = yaml.safe_load(text), yaml.safe_load(new_text)
        want = {"file": p.pfce.name, "expected_n": len(pfce_ids)}
        before["probes"][PFCE] = want
        if after != before or after["probes"][PFCE] != want:
            raise Stop("Internal error: the edit of configs/conditions.yaml would change more "
                       "than the P-FCE line. The file was not touched. Paste this line by "
                       "hand over the P-FCE line:\n" + line)
        if new_text != text:
            with open(p.conditions, "w", encoding="utf-8", newline="") as fh:
                fh.write(new_text)
        say("  repo probes/%s   %d qa_ids" % (p.ids_pfce.name, len(pfce_ids)))
        say("  repo configs/conditions.yaml   the P-FCE line now reads:")
        say("    " + line.strip())
        say("  Commit and push these two files (the notebook has the cell).")
    else:
        say("  --no-repo: the clone was not touched. The line for configs/conditions.yaml is:")
        say("    " + line.strip())

    say("\n[5] Checks on the final files, read again from disk")
    ok, num2, _table = check_final(p, builder, items, judged["rejected"], report,
                                   repo=not a.no_repo)
    same = num2["pfce_ids"] == pfce_ids and fc.file_sha256(p.options) == \
        fc.sha256_text("\n".join(lines) + "\n")
    ok = ok and same
    say("  " + mark(same) + "the file on disk is the file that was built")
    per_reason = {code: reason_n.get(code, 0) for code in REASONS}
    manifest = {
        "asset": "ASSET-11", "script_version": SCRIPT_VERSION, "rules_version": fc.RULES_VERSION,
        "applied_utc": utc_now(), "repo_commit": git(p.repo, "rev-parse", "HEAD")[1],
        "draft_sha": sha, "stage1_built_utc": state.get("built_utc", ""),
        "inputs": {"lexicon_sha256": fc.file_sha256(p.lexicon),
                   "pclust_sha256": fc.file_sha256(p.pclust),
                   "ht6_log_sha256": fc.file_sha256(log_path),
                   "excluded_items": excluded_info(p)},
        "outputs": {OPTIONS_FILE: {"sha256": fc.file_sha256(p.options), "records": len(lines),
                                   "bn": num["n_bn"], "en": num["n_en"]},
                    p.pfce.name: {"sha256": fc.file_sha256(p.pfce), "n": len(pfce_ids),
                                  "created_utc": created}},
        "ht6": {"reviewer": reviewer, "sets_reviewed": len(rows), "sets_changed": len(changed),
                "distractors_replaced": sum(reason_n.values()), "by_reason": per_reason,
                "rejections_without_effect": judged["moot"], "sets_flagged": len(judged["flags"])},
        "pfce": {"n": len(pfce_ids), "n_draft": state.get("n_pfce_draft"),
                 "by_cluster": num["pfce_by_cluster"],
                 "en_unique_by_cluster": num["en_unique_by_cluster"],
                 "gold_position": num["gold_position_pfce"]},
        "pclust": {"n": len(items), "gold_position": num["gold_position_bn"],
                   "shape": num["shape"]},
        "checks_passed": bool(ok)}
    write_json(p.manifest, manifest)
    write_text(p.report, "\n".join(report) + "\n")
    say("  wrote Drive probes/%s and %s" % (p.manifest.name, p.report.name))
    say("\nRESULT: " + (
        "STAGE 2 OK. P-FCE = %d items (draft %s). %d sets changed by HT-6."
        % (len(pfce_ids), state.get("n_pfce_draft"), len(changed)) if ok else
        "STAGE 2 NOT OK: a check failed. Do not run a forced-choice condition; send this "
        "output to the ASSET-11 chat."))
    if ok and a.no_repo:
        say("One step is left: run apply again without --no-repo, then commit and push "
            "probes/P-FCE.ids.txt and configs/conditions.yaml.")
    return 0 if ok else 1


def cmd_check(a):
    p = Paths(a.repo, a.root)
    say("ASSET-11 check of the final files   script %s   rules %s" % (SCRIPT_VERSION, fc.RULES_VERSION))
    _lex, builder, _header, items = load_world(p)
    if not (p.log.exists() and p.state.exists()):
        raise Stop("Drive human/HT-6/ must still hold %s and %s: they are the record of HT-6, "
                   "and the check reads the rejected options from them." % (LOG_FILE, STATE_FILE))
    sha = read_json(p.state).get("draft_sha")
    rejected = judge_log(builder, items, read_log(p.log), sha)["rejected"]
    ok, num, _table = check_final(p, builder, items, rejected)
    if p.manifest.exists():
        man = read_json(p.manifest)
        same = (man.get("outputs", {}).get(OPTIONS_FILE, {}).get("sha256") == fc.file_sha256(p.options)
                and man.get("outputs", {}).get(p.pfce.name, {}).get("sha256") == fc.file_sha256(p.pfce)
                and man.get("inputs", {}).get("ht6_log_sha256") == fc.file_sha256(p.log)
                and man.get("draft_sha") == sha)
        say("  " + mark(same) + "the option file, P-FCE.json and the HT-6 log are the files of "
            "stage 2 (%s, reviewer %s, %s sets changed)"
            % (man.get("applied_utc"), man.get("ht6", {}).get("reviewer"),
               man.get("ht6", {}).get("sets_changed")))
        ok = ok and same
    else:
        say("  " + mark(False) + "Drive probes/%s is missing" % p.manifest.name)
        ok = False
    say("\nRESULT: " + ("FINAL FILES OK. P-FCE = %d items." % num["n_pfce"] if ok
                        else "NOT OK: a check failed."))
    return 0 if ok else 1


# --------------------------------------------------------------------------
# Command: inputs (Cell 0)
# --------------------------------------------------------------------------

def cmd_inputs(a):
    p = Paths(a.repo, a.root, a.images_1024)
    say("ASSET-11 inputs   script %s   rules %s" % (SCRIPT_VERSION, fc.RULES_VERSION))
    say("repo %s   (commit %s)" % (p.repo, git(p.repo, "rev-parse", "--short", "HEAD")[1] or "unknown"))
    say("Drive root %s" % p.root)
    results = []

    def line(ok, text):
        results.append(bool(ok))
        say(mark(ok) + text)

    def guard(label, fn):
        try:
            fn()
        except (Stop, fc.OptionError, scoring.LexiconError) as exc:
            line(False, "%s: %s" % (label, exc))
        except Exception as exc:
            line(False, "%s: %s: %s" % (label, type(exc).__name__, exc))

    world = {}

    def code():
        need = [p.template, p.repo / "src" / "fc_options.py", p.repo / "src" / "scoring.py",
                p.repo / "src" / "normalise.py"]
        missing = [str(x.relative_to(p.repo)) for x in need if not x.exists()]
        line(not missing, "code: scripts/asset11_options.py, scripts/%s, src/fc_options.py; "
             "scorer %s%s" % (TEMPLATE_FILE, scoring.SCORER_VERSION,
                              "" if not missing else "; MISSING " + ", ".join(missing)))

    def lexicon():
        lex = scoring.load_lexicon(p.lexicon)
        world["builder"] = fc.Builder(lex)
        n_in = sum(1 for o in lex.objects.values() if o.cluster_id)
        counts_ok = (len(lex.objects), n_in, len(lex.clusters), len(lex.index.entries)) == \
            (231, 86, 16, 864)
        line(counts_ok and not lex.warnings,
             "repo lexicon/lexicon.csv loads in the scorer: %d objects, %d in %d clusters, %d "
             "terms, %d warnings (PROJECT_STATE: 231, 86 in 16, 864, 0)"
             % (len(lex.objects), n_in, len(lex.clusters), len(lex.index.entries),
                len(lex.warnings)))
        status, words = frozen_status(p)
        line(status == "same", "the lexicon is the file frozen at Gate A: %s" % words)

    def config():
        cfg = read_yaml(p.conditions)
        world["cfg"] = cfg
        pf = (cfg.get("probes") or {}).get(PFCE)
        fce = [c for c, v in (cfg.get("conditions") or {}).items() if v.get("probe") == PFCE]
        now = "not set yet (file: null), as expected before stage 2"
        if isinstance(pf, dict) and pf.get("file"):
            now = "set to %s with expected_n %s" % (pf.get("file"), pf.get("expected_n"))
        line(isinstance(pf, dict) and sorted(fce) == ["N-FCE", "N-FCE0"],
             "repo configs/conditions.yaml: P-FCE is %s; the conditions on it are %s"
             % (now, " and ".join(sorted(fce))))

    def pclust():
        header, items = load_probe_file(p.pclust, PROBE)
        world["items"] = items
        want = ((world.get("cfg") or {}).get("probes") or {}).get(PROBE, {}).get("expected_n")
        b = world.get("builder")
        golds = [b.gold_of(it) for it in items] if b else []
        uniq = sum(1 for g in golds if str(g.row.get("en_unique", "")).lower() == "yes")
        line(len(items) == want and len(golds) == len(items)
             and len({g.cluster_id for g in golds}) == 7,
             "Drive probes/P-CLUST.json: %d items (conditions.yaml says %s); the scorer joins "
             "%d golds to %d objects in %d clusters; %d are en_unique (the upper bound of P-FCE)"
             % (len(items), want, len(golds), len({g.name_id for g in golds}),
                len({g.cluster_id for g in golds}), uniq))
        if p.ids_pclust.exists():
            ids = fc.read_ids(p.ids_pclust)
            line(set(ids) == {it["qa_id"] for it in items} and len(ids) == len(items),
                 "repo probes/P-CLUST.ids.txt: %d ids, the same items as the Drive file" % len(ids))
        else:
            line(False, "repo probes/P-CLUST.ids.txt is missing")

    def pname():
        _h, items = load_probe_file(p.pname, "P-NAME")
        want = ((world.get("cfg") or {}).get("probes") or {}).get("P-NAME", {}).get("expected_n")
        ids = {it["qa_id"] for it in items}
        inside = sum(1 for it in world.get("items", []) if it["qa_id"] in ids)
        line(len(items) == want and inside == len(world.get("items", [])),
             "Drive probes/P-NAME.json: %d items (conditions.yaml says %s); %d of the %d P-CLUST "
             "items are in it" % (len(items), want, inside, len(world.get("items", []))))

    def pilot():
        ids, src = pilot_ids(p)
        if ids is None:
            line(False, "PILOT id list: neither repo probes/PILOT.ids.txt nor Drive "
                        "probes/PILOT.json found")
            return
        in_cl = len(set(ids) & {it["qa_id"] for it in world.get("items", [])})
        line(len(ids) == 100 and in_cl >= 20,
             "%s: %d ids, %d of them in P-CLUST by qa_id (Design 3.4: 100 items, 20 drawn from "
             "P-CLUST for the naming part)" % (src, len(ids), in_cl))

    def excluded():
        exc = excluded_ids(p)
        if exc is None:
            line(False, "repo probes/excluded_items.csv is missing (frozen at gate-A with 4 rows)")
            return
        n_in = len(set(exc) & {it["qa_id"] for it in world.get("items", [])})
        line(len(exc) >= EXCLUDED_MIN_ROWS,
             "repo probes/excluded_items.csv: %d rows (sha256 %s...; at least %d expected, rows "
             "are never removed), %d in P-CLUST (excluded items still get option sets)%s"
             % (len(exc), fc.file_sha256(p.excluded)[:12], EXCLUDED_MIN_ROWS, n_in,
                "" if len(exc) >= EXCLUDED_MIN_ROWS else ". This clone is out of date: pull"))

    def pictures():
        imglib, Image, manifest = load_image_tools(p)
        files = [it["image_file"] for it in world.get("items", [])]
        missing = [f for f in files if manifest.get(f) is None]
        if missing or not files:
            line(False, "Drive data/images_1024/: %d of %d P-CLUST pictures are not in the "
                        "manifest (first: %s)" % (len(missing), len(files),
                                                  missing[0] if missing else "-"))
            return
        sizes = []
        for f in (files[0], files[len(files) // 2], files[-1]):
            data, _sha = imglib.load_prepared(p.images, f, manifest)
            sizes.append("%dx%d" % Image.open(io.BytesIO(data)).size)
        line(True, "Drive data/images_1024/: all %d P-CLUST pictures are in the manifest (%d "
                   "rows); 3 read through src/images.py (%s)" % (len(files), len(manifest),
                                                              ", ".join(sizes)))

    def harness_ok():
        import harness
        import prompts as promptlib
        prompts = promptlib.load_prompts(p.prompts)
        promptlib.load_conditions(p.conditions, prompts)
        line(True, "src/harness.py %s and configs/prompts.yaml (%s) load; they are used for "
                   "the last checks of stage 2" % (harness.HARNESS_VERSION, prompts["prompt_version"]))

    def folders():
        for d in (p.ht6, p.options.parent):
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".asset11_write_test"
            probe.write_text("x", encoding="utf-8")
            probe.unlink()
        line(True, "Drive human/HT-6/ and probes/ can be written")

    for label, fn in (("code", code), ("repo lexicon/lexicon.csv", lexicon),
                      ("repo configs/conditions.yaml", config),
                      ("Drive probes/P-CLUST.json", pclust), ("Drive probes/P-NAME.json", pname),
                      ("PILOT id list", pilot), ("repo probes/excluded_items.csv", excluded),
                      ("Drive data/images_1024/", pictures),
                      ("src/harness.py and configs/prompts.yaml", harness_ok),
                      ("Drive folders", folders)):
        guard(label, fn)

    if p.state.exists():
        st = read_json(p.state)
        say("NOTE  stage 1 was built %s (draft %s...); HT-6 log in the folder: %s; final "
            "option file: %s" % (st.get("built_utc"), str(st.get("draft_sha"))[:12],
                                 "yes" if p.log.exists() else "no",
                                 "yes" if p.options.exists() else "no"))
    else:
        say("NOTE  stage 1 has not run yet for this Drive folder")
    bad = results.count(False)
    say("\n%d of %d inputs PASS. %s" % (len(results) - bad, len(results),
                                      "Go on with selftest and build." if not bad else
                                      "Fix every FAIL line before build."))
    return 0 if not bad else 1


# --------------------------------------------------------------------------
# Command: selftest (made-up lexicon; no project file, no dataset text)
# --------------------------------------------------------------------------

def _fake_rows():
    """A made-up lexicon. The names are letter strings without meaning."""
    def row(i, bn, en, freq, scope, cluster="", var_bn="", sup_bn="", sup_en=""):
        return {"name_id": "N_t%02d" % i, "name_bn": bn, "variants_bn": var_bn,
                "canonical_en": en, "variants_en": "x" + en if i == 2 else "", "super_bn": sup_bn,
                "super_en": sup_en,
                "cluster_id": cluster, "siblings": "", "room_scope": scope,
                "cultural_specificity": "not_rated", "origin3": "uncertain", "origin_note": "",
                "en_route": "no", "en_unique": "", "freq_obj_name": str(freq)}
    return [
        # cluster C1, kitchen: a generic member (05) and a tie (02, 03)
        row(1, "কাফু", "zorb", 10, "kitchen", "C1", sup_bn="গোমা", sup_en="blen"),
        row(2, "খিমো", "quix", 8, "kitchen", "C1", var_bn="খিমু"),
        row(3, "গুনা", "plim", 8, "kitchen", "C1"),
        row(4, "ঘেরি", "drav", 3, "kitchen", "C1"),
        row(5, "গোমা", "blen", 20, "kitchen", "C1"),
        # cluster C2, any: two members only
        row(6, "চাপো", "merk", 5, "any", "C2"),
        row(7, "ছুটা", "vund", 4, "any", "C2"),
        # cluster C3, bedroom: two members share an English name
        row(8, "জিলু", "tarp", 9, "bedroom", "C3"),
        row(9, "ঝোমা", "snib", 7, "bedroom", "C3"),
        row(10, "টুপা", "snibs", 6, "bedroom", "C3"),
        # cluster C4, bathroom: two members, and one other bathroom object only
        row(11, "নিবু", "lorn", 4, "bathroom", "C4"),
        row(12, "পেটু", "mabe", 3, "bathroom", "C4"),
        row(13, "ধোপু", "wesk", 2, "bathroom"),
        # objects outside every cluster
        row(14, "ডালু", "frep", 30, "kitchen"),
        row(15, "ঢেকো", "glon", 12, "kitchen"),
        row(16, "তিরু", "hask", 50, "any"),
        row(17, "থোবা", "jilp", 40, "any"),
        row(18, "দিনু", "kreb", 15, "bedroom"),
        row(19, "রিসু", "ombe", 11, "bedroom"),
        # cluster C5, bedroom: its first member shares an English name with a kitchen object
        row(20, "লোকা", "frep", 2, "bedroom", "C5"),
        row(21, "শিমু", "yarl", 1, "bedroom", "C5"),
    ]


def _fake_lexicon():
    rows = _fake_rows()
    lex = scoring.build_lexicon(rows, source="selftest")
    uniq = scoring.compute_en_unique(lex.objects)
    for r in rows:
        r["en_unique"] = "yes" if uniq[r["name_id"]] else "no"
    return scoring.build_lexicon(rows, source="selftest")


def _fake_items(lex):
    golds = ["কাফু", "কাফু", "খিমু", "গুনা", "ঘেরি", "গোমা", "চাপো", "ছুটা", "জিলু", "ঝোমা",
             "টুপা", "নিবু", "পেটু", "কাফু।", "খিমো", "লোকা", "শিমু"]
    items = []
    for n, g in enumerate(golds, 1):
        oid = lex.bn_t1_key_to_id[lex.index.key_of(normalise(g))]
        items.append({"qa_id": "T%02d_%04d_q1" % (1 + n % 3, n), "image_id": "T_%04d" % n,
                      "house_id": "T%02d" % (1 + n % 3), "capture_type": "closeup",
                      "qa_type": "obj_name", "question_bn": "ক খ গ?", "answer_bn": g,
                      "image_file": "T/T_%04d.jpg" % n,
                      "cluster_id": lex.objects[oid].cluster_id})
    return items


def cmd_selftest(a):
    say("ASSET-11 self test   script %s   rules %s" % (SCRIPT_VERSION, fc.RULES_VERSION))
    checks = []
    try:
        return _selftest(a, checks)
    except Exception as exc:          # a rule that breaks so badly that a check cannot run
        say("FAIL the self test stopped after %d checks with %s: %s"
            % (len(checks), type(exc).__name__, exc))
        say("\n%d/%d self checks passed" % (checks.count(True), len(checks) + 1))
        return 1


def _selftest(a, checks):

    def ck(name, passed):
        checks.append(bool(passed))
        say(("OK   " if passed else "FAIL ") + name)

    def raises(fn, exc=(fc.OptionError, Stop)):
        try:
            fn()
        except exc:
            return True
        return False

    ID = lambda n: "N_t%02d" % n            # noqa: E731
    lex = _fake_lexicon()
    b = fc.Builder(lex)
    items = _fake_items(lex)
    by_gold = {}
    for it in items:
        by_gold.setdefault(b.gold_of(it).name_id, it)

    # --- seeded ranks: fixed values, so a change of rule or platform is seen ---
    ck("the order of three known qa_ids is the recorded one",
       all(fc.shown_order(q) == v for q, v in RECORDED_ORDER.items()))
    ck("the tie rank of two known name_ids is the recorded one",
       all(fc.tie_rank(i)[:16] == v for i, v in RECORDED_TIE.items()))
    place = Counter(fc.shown_order("X%02d_%04d_q1" % (h, i)).index(0)
                    for h in range(40) for i in range(100))
    ck("over 4,000 made-up ids the gold falls on each place 900 to 1,100 times (%s)"
       % ", ".join(str(place[k]) for k in range(4)), all(900 <= place[k] <= 1100 for k in range(4)))

    # --- candidates ---
    q1 = b.queues(ID(1))
    tie = sorted([ID(2), ID(3)], key=fc.tie_rank)
    ck("cluster candidates come by frequency, and a tie by the seeded rank",
       q1["sib"] == tie + [ID(4)])
    ck("a cluster member that is a generic name of the gold is skipped (rule B)",
       ID(5) in q1["generic"] and ID(5) not in q1["sib"] and ID(5) not in q1["out"])
    ck("the same member is offered when the gold has no such generic name",
       b.queues(ID(2))["sib"][0] == ID(5))
    ck("outside candidates: same room_scope first, by frequency",
       q1["out"][:q1["n_same"]] == [ID(14), ID(15)] and q1["n_same"] == 2)
    d1, s1 = b.fill(ID(1))
    ck("a gold in a large cluster gets two cluster members and one same-room object",
       list(d1) == tie + [ID(14)] and d1 == s1)
    d6, _ = b.fill(ID(6))
    ck("a cluster with one other member: that member, then two 'any' objects for an 'any' gold",
       list(d6) == [ID(7), ID(16), ID(17)])
    d11, _ = b.fill(ID(11))
    ck("when the same-room list ends, the most frequent other object is used (rule D)",
       list(d11) == [ID(12), ID(13), ID(16)])
    bad_rows = _fake_rows()
    bad_rows[0]["room_scope"] = "kitchen|bedroom"
    ck("a room_scope with several values stops the build (rule C)",
       raises(lambda: fc.Builder(scoring.build_lexicon(bad_rows)).queues(ID(2))))

    # --- sets and records ---
    sets, table = make_sets(b, items)
    s_var = next(s for s, it in zip(sets, items) if it["answer_bn"] == "খিমু")
    rec = table[(s_var.qa_id, "bn")]
    ck("a gold written as a merged spelling is shown under the main name of its object",
       s_var.gold_id == ID(2) and rec["options"][rec["correct_index"]] == "খিমো")
    wrong_id = dict(by_gold[ID(1)], name_id="N_wrong", name_bn="x")
    ck("name_id and name_bn fields of the probe record are never read (C4)",
       b.make(wrong_id) == b.make(by_gold[ID(1)]))
    ck("a gold the scorer cannot resolve stops the build",
       raises(lambda: b.make(dict(by_gold[ID(1)], answer_bn="মমম"))))
    s8 = b.make(by_gold[ID(8)])
    ck("P-FCE: an en_unique gold drops out when two distractors share an English label",
       lex.objects[ID(8)].row["en_unique"] == "yes" and not b.in_pfce(ID(8), s8.ids_shown())
       and (s8.qa_id, "en") not in table)
    s20 = b.make(by_gold[ID(20)])
    ck("P-FCE: a gold that is not en_unique gets no English set, also when its four labels differ",
       all((s.qa_id, "en") not in table for s in sets if s.gold_id in (ID(9), ID(10), ID(20)))
       and lex.objects[ID(20)].row["en_unique"] == "no"
       and fc.english_labels_differ([fc.text_en(lex.objects[i]) for i in s20.ids_shown()])
       and (by_gold[ID(21)]["qa_id"], "en") in table)
    ck("the English option is canonical_en, never another accepted English name",
       all(o in {r["canonical_en"] for r in _fake_rows()}
           for (_q, lang), r_ in table.items() if lang == "en" for o in r_["options"])
       and any("quix" in r_["options"] for (_q, lang), r_ in table.items() if lang == "en"))
    ck("P-FCE: labels that differ only by a plural ending count as the same",
       not fc.english_labels_differ(["tarp", "snib", "snibs", "kreb"])
       and fc.english_labels_differ(["tarp", "snib", "kreb", "ombe"]))
    s2 = b.make(by_gold[ID(1)])
    en = table[(s2.qa_id, "en")]
    bn = table[(s2.qa_id, "bn")]
    ck("the English set holds the same four objects in the same places as the Bangla set",
       en["option_ids"] == bn["option_ids"] and en["correct_index"] == bn["correct_index"]
       and en["options"][en["correct_index"]] == "zorb")
    res = fc.check_sets(b, items, table)
    ck("all checks pass on the made-up draft, except the room_scope rule (2 sets use the "
       "last resort of rule D)",
       [k for k, _t, ok in res["checks"] if not ok] == ["frequency"]
       and "2 sets" in next(t for k, t, _ok in res["checks"] if k == "frequency"))
    ck("the scorer reads every option text as that option and every label as its place",
       not fc.scorer_round_trip(table))
    trap = {("x", "en"): {"options": ["vex c", "zorb", "quix", "plim"], "correct_index": 1}}
    ck("an option text that holds a label letter is caught (D107)",
       len(fc.scorer_round_trip(trap)) >= 1)

    # --- HT-6 replacement ---
    g = ID(1)
    _d, s = b.fill(g, [tie[0]])
    ck("a rejected distractor is replaced in its own slot by the next cluster member",
       list(s) == [ID(4), tie[1], ID(14)])
    _d, s = b.fill(g, [tie[0], ID(4)])
    ck("when the cluster has no member left, the slot takes the next same-room object",
       list(s) == [ID(15), tie[1], ID(14)])
    _d, s = b.fill(g, [ID(14)])
    ck("the outside slot takes the next same-room object", list(s) == tie + [ID(15)])
    ck("the result does not depend on the order of the rejections",
       b.fill(g, [tie[1], tie[0]]) == b.fill(g, [tie[0], tie[1]]))
    keep, moot = b.prune(g, [ID(4)])
    ck("a rejection of an option that is not in the set changes nothing and is dropped",
       keep == [] and moot == [ID(4)])
    ck("the gold and its generic names cannot be rejected",
       raises(lambda: b.prune(g, [g])) and raises(lambda: b.prune(g, [ID(5)])))
    it1 = by_gold[g]
    a_set, b_set = b.make(it1), b.make(it1, [tie[0]])
    ck("a replacement keeps the place of the option it replaces; the gold does not move",
       a_set.correct_index == b_set.correct_index
       and [x for x in a_set.ids_shown() if x != tie[0]]
       == [x for x in b_set.ids_shown() if x != ID(4)])

    # --- files ---
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        final = tmp / OPTIONS_FILE
        write_text(final, "\n".join(table_lines(items, table)) + "\n")
        back = fc.read_option_sets(final)
        ck("the option file reads back as it was written (%d records)" % len(back), back == table)
        ck("options_for gives the four texts and the gold place to the scorer",
           fc.options_for(back, s2.qa_id, "en") == (en["options"], en["correct_index"])
           and raises(lambda: fc.options_for(back, s8.qa_id, "en")))
        draft = tmp / DRAFT_FILE
        write_text(draft, "\n".join(table_lines(items, table, fc.DRAFT_OPTIONS_KEY)) + "\n")
        ck("the reader refuses a draft file", raises(lambda: fc.read_option_sets(draft)))
        lines = table_lines(items, table)
        three = json.loads(lines[0])
        three["options"] = three["options"][:3]
        write_text(final, json.dumps(three, ensure_ascii=False) + "\n")
        ck("the reader refuses a set with three options",
           raises(lambda: fc.read_option_sets(final)))
        write_text(final, lines[0] + "\n" + lines[0] + "\n")
        ck("the reader refuses two sets for one item and language",
           raises(lambda: fc.read_option_sets(final)))
        moved = json.loads(lines[0])
        moved["correct_index"] = (moved["correct_index"] + 1) % 4
        write_text(final, json.dumps(moved, ensure_ascii=False) + "\n")
        ck("the reader refuses a gold place that is not the place of the gold",
           raises(lambda: fc.read_option_sets(final)))
        try:
            import harness
            write_text(final, "\n".join(lines) + "\n")
            n = len(harness.load_options(str(final))["table"])
            ck("harness.load_options reads the final file (%d sets) and refuses the draft" % n,
               n == len(table) and raises(lambda: harness.load_options(str(draft)),
                                          (harness.Stop,)))
        except ImportError as exc:
            say("SKIP harness.load_options: src/harness.py is not importable here (%s)" % exc)

    # --- the HT-6 log ---
    sha = draft_fingerprint(b, sets)

    def log_rows(decide):
        out = []
        for n, s in enumerate(review_order(b, sets), 1):
            rej = decide(s)
            f = b.make(next(it for it in items if it["qa_id"] == s.qa_id), list(rej))
            out.append({"seq": str(n), "qa_id": s.qa_id, "cluster": "", "gold_id": s.gold_id,
                        "gold_bn": "", "verdict": "swap" if f.changed else "ok",
                        "rejected_ids": "|".join(rej), "rejected_bn": "",
                        "reasons": "|".join(rej[x] for x in rej),
                        "final_ids": "|".join(f.ids_shown()), "final_bn": "", "flag": "0",
                        "note": "n" if "other" in rej.values() else "", "reviewer": "ab",
                        "draft_sha": sha, "rules": fc.RULES_VERSION})
        return out

    j = judge_log(b, items, log_rows(lambda s: {}), sha)
    ck("a log with no rejection is accepted and changes nothing",
       not any(j["rejected"].values()) and j["moot"] == 0)
    one = log_rows(lambda s: {tie[0]: "name"} if s.gold_id == g else {})
    j = judge_log(b, items, one, sha)
    f_sets, f_table = make_sets(b, items, j["rejected"])
    ck("a log with one rejected name changes exactly the sets of that gold",
       sorted(s.gold_id for s in f_sets if s.changed) == [g] * 3
       and all(tie[0] not in f_table[(s.qa_id, "bn")]["option_ids"]
               for s in f_sets if s.gold_id == g))
    res2 = {k: ok for k, _t, ok in fc.check_sets(b, items, f_table, j["rejected"])["checks"]}
    ck("the checks pass on the changed sets, and a rejected option is never shown",
       res2["rejected"] and [k for k, ok in res2.items() if not ok] == ["frequency"])
    shown = dict(f_table)
    put_back = dict(shown[(it1["qa_id"], "bn")], option_ids=table[(it1["qa_id"], "bn")]["option_ids"],
                    options=table[(it1["qa_id"], "bn")]["options"])
    shown[(it1["qa_id"], "bn")] = put_back
    res3 = {k: ok for k, _t, ok in fc.check_sets(b, items, shown, j["rejected"])["checks"]}
    ck("a file that still shows a rejected option fails the check", not res3["rejected"])
    ck("swap rows name the removed option, its reason and the option now in its place",
       [(r["removed_id"], r["reason"], r["final_id"]) for r in swap_rows(b, f_sets, items, j)]
       == [(tie[0], "name", ID(4))] * 3)

    def broken(change):
        rows = log_rows(lambda s: {tie[0]: "other"} if s.gold_id == g else {})
        change(rows)
        return raises(lambda: judge_log(b, items, rows, sha), (Stop,))

    ck("a log with 'other reason' and a note is accepted", not broken(lambda r: None))

    ck("the log is refused when a row is missing", broken(lambda r: r.pop()))
    ck("the log is refused when a set is not reviewed",
       broken(lambda r: r[0].update(verdict="")))
    ck("the log is refused when it belongs to another draft",
       broken(lambda r: r[0].update(draft_sha="0" * 64)))
    ck("the log is refused when the final set was changed by hand",
       broken(lambda r: r[0].update(final_ids="|".join(reversed(r[0]["final_ids"].split("|"))))))
    ck("the log is refused when 'other reason' has no note",
       broken(lambda r: [x.update(note="") for x in r]))
    ck("the log is refused when two reviewers are named",
       broken(lambda r: r[0].update(reviewer="cd")))

    # --- conditions.yaml and the sheet ---
    text = ("probes:\n  P-CLUST:       {file: P-CLUST.json,       expected_n: 492}\n"
            "  P-FCE:         {file: null,               expected_n: null}   # written by "
            "ASSET-11 after Gate A\n  R-TEST:        {file: R-TEST.json,        expected_n: 1449}\n")
    new = fc.patch_conditions_text(text, "P-FCE.json", 123)
    ck("the edit of conditions.yaml changes the P-FCE line and nothing else",
       [x for x, y in zip(text.split("\n"), new.split("\n")) if x != y]
       == [text.split("\n")[2]] and "{file: P-FCE.json,         expected_n: 123}" in new
       and fc.patch_conditions_text(new, "P-FCE.json", 123) == new)
    template = Path(a.repo) / "scripts" / TEMPLATE_FILE
    if template.exists():
        ordered, data = page_data(b, sets, items, sha, False)
        html = render_page(template.read_text(encoding="utf-8"), data, [])
        ck("the sheet holds every set, and its data block reads back",
           page_json(html) == json.loads(json.dumps(data)) and len(data["items"]) == len(items)
           and [r[0] for r in data["items"]] == [s.qa_id for s in ordered])
    else:
        ck("scripts/%s is in the repo" % TEMPLATE_FILE, False)

    bad = checks.count(False)
    say("\n%d/%d self checks passed" % (len(checks) - bad, len(checks)))
    return 0 if not bad else 1


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="ASSET-11: forced-choice option sets, HT-6 sheet, P-FCE")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp, root=True):
        sp.add_argument("--repo", default=str(REPO_DEFAULT), help="the repo clone")
        if root:
            sp.add_argument("--root", default=ROOT_DEFAULT,
                            help="the project folder in Google Drive (holds probes/, data/, human/)")

    sp = sub.add_parser("inputs", help="one PASS or FAIL line per input")
    common(sp)
    sp.add_argument("--images-1024", default=None, help="prepared pictures (default ROOT/data/images_1024)")
    sp.set_defaults(fn=cmd_inputs)

    sp = sub.add_parser("selftest", help="the rules on a made-up lexicon")
    common(sp, root=False)
    sp.set_defaults(fn=cmd_selftest)

    sp = sub.add_parser("build", help="STAGE 1: candidates, draft sets, HT-6 sheet")
    common(sp)
    sp.add_argument("--images-1024", default=None, help="prepared pictures (default ROOT/data/images_1024)")
    sp.add_argument("--thumb-px", type=int, default=640, help="longer side of a picture in the sheet")
    sp.add_argument("--thumb-quality", type=int, default=75, help="JPEG quality in the sheet")
    sp.add_argument("--no-images", action="store_true", help="a test sheet without pictures; not for HT-6")
    sp.set_defaults(fn=cmd_build)

    sp = sub.add_parser("apply", help="STAGE 2: HT-6 log -> final sets, P-FCE")
    common(sp)
    sp.add_argument("--no-repo", action="store_true",
                    help="do not write probes/P-FCE.ids.txt and the P-FCE line of "
                         "configs/conditions.yaml into the clone")
    sp.add_argument("--replace", action="store_true", help="overwrite final files whose content changes")
    sp.add_argument("--allow-test-sheet", action="store_true", help=argparse.SUPPRESS)
    sp.set_defaults(fn=cmd_apply)

    sp = sub.add_parser("check", help="read the final files again and repeat every check")
    common(sp)
    sp.set_defaults(fn=cmd_check)

    a = ap.parse_args(argv)
    src = str(Path(a.repo).resolve() / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    if fc.RULES_VERSION.split()[0] != SCRIPT_VERSION.split()[0]:
        say("STOPPED: scripts/asset11_options.py (%s) and src/fc_options.py (%s) are not from "
            "the same delivery. Pull the repo." % (SCRIPT_VERSION, fc.RULES_VERSION))
        return 2
    try:
        return a.fn(a)
    except (Stop, fc.OptionError, scoring.LexiconError) as exc:
        say("\nSTOPPED: %s" % exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
