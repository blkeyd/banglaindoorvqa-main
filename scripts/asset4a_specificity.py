"""
asset4a_specificity.py
BanglaIndoorVQA, ASSET-4A: cultural_specificity tags by three-way vote.

Tags every object name that can enter P-NAME with one of three labels:
culturally_specific, globally_common, ambiguous. The tags feed CTRL-8 (G7),
the D42 frequency floor for P-NAME (ASSET-5B), and the cultural_specificity
column of lexicon.csv (ASSET-4B).

Subcommands:

  selftest  Checks the agreement statistics against textbook values, then
            runs build and merge end to end on SYNTHETIC data. Needs only
            house_type_counts.csv. Writes only to a temporary folder.
  build     Lists every normalise()d object name that reaches 5 instances as
            a box label OR as an obj_name answer (D50). Writes a master table,
            three rater sheets (each in its own shuffled order, with no counts
            shown) and the rater instructions.
  merge     Reads the three filled sheets. Majority of three decides; a
            three-way split becomes ambiguous unless a session file settles it
            (D51). Reports Fleiss' kappa, Gwet's AC1, pairwise Cohen's kappa,
            stratum sizes and a P-NAME preview under D42. Writes a DRAFT;
            with --freeze it writes results/assets/cultural_specificity.csv.

Run order on real data:
  1. selftest
  2. build
  3. the three raters read the instructions together, then vote ALONE
  4. merge                         (draft: look at agreement and splits)
  5. short session on three-way splits only, if there are any
  6. merge --session ... --freeze  (or merge --freeze if no session)

Needs normalise.py and asset5a_probes.py in the same folder. The QA loader is
imported from asset5a_probes.py, so any field-name fix made there applies
here too. Never copy normalise() into this file.
"""

import argparse
import datetime
import json
import os
import platform
import shutil
import sys
import tempfile
from collections import Counter, defaultdict

from normalise import normalise
import asset5a_probes as a5

say = a5.say
hkey = a5.hkey
sha256_file = a5.sha256_file

# ----------------------------------------------------------------------------
# Fixed design constants. Changing any of these changes the pre-registration.
# ----------------------------------------------------------------------------
SEED = 42
LABELS = ["culturally_specific", "globally_common", "ambiguous"]
SHORT_CODES = {"cs": "culturally_specific", "gc": "globally_common", "amb": "ambiguous"}
N_RATERS = 3
LIST_FLOOR = 5            # lowest threshold any downstream use applies (D42, D50)
PNAME_FLOOR_CS = 5        # D42
PNAME_FLOOR_OTHER = 10    # Guide section 3.2, kept by D42
PNAME_SIZE = 1000
N_EXAMPLE_IMAGES = 3

# Examples printed in the rater instructions (locked design, section 6, A8).
# Kappa is also reported with these rows left out, because raters saw them.
EXAMPLES_CS = ["বটি", "শিল-পাটা", "বদনা", "কুলা", "মোড়া", "গামছা", "লেপ", "পিঁড়ি", "হামানদিস্তা"]
EXAMPLES_GC = ["চেয়ার", "ফ্রিজ", "টেলিভিশন", "সোফা", "বেসিন"]
EXAMPLE_NAMES = {normalise(x) for x in EXAMPLES_CS + EXAMPLES_GC}

# Verified v2 figures from stats.json. Totals are hard checks; the name
# counts are warnings, because stats.json counted raw surface forms (D44).
EXPECTED_BOXES = 3620
EXPECTED_OBJ_NAME_ITEMS = 1877
EXPECTED_RAW_BOX_NAMES = 472
EXPECTED_RAW_BOX_NAMES_GE5 = 135
EXPECTED_RAW_BOX_SINGLETONS = 222

# Box field names. The loader tries each in order; add yours to the front.
BOX_LIST_KEYS = ("boxes", "bboxes", "bounding_boxes", "objects", "regions", "box")
BOX_LABEL_KEYS = ("bangla_name", "label", "name", "object_name", "label_bn", "name_bn", "object", "category")
ROOM_KEYS = ("room_type", "room")
NO_ROOM = {"", "na", "none", "null"}

RATER_INSTRUCTIONS = """\
ASSET-4A RATER INSTRUCTIONS: cultural specificity

What you are doing
You will give every object name in your sheet one of three labels.
Work alone. Do not discuss any row of the sheet with the other two raters
until all three sheets are handed in. Do not look at any AI model output on
these objects before you finish.

The question to ask for every object
"Is this kind of object, in the form it has in Bangladeshi homes, found in
much the same form in homes in many parts of the world outside South Asia?"

Labels (write exactly one in the label column)
  globally_common      Yes. Homes in many parts of the world have this object
                       in much the same form.
                       Examples: চেয়ার, ফ্রিজ, টেলিভিশন, সোফা, বেসিন
  culturally_specific  No. It is mainly found in Bangladeshi or other South
                       Asian homes, or its local form clearly differs in
                       shape, material or use from the nearest item found
                       elsewhere.
                       Examples: বটি, শিল-পাটা, বদনা, কুলা, মোড়া, গামছা, লেপ,
                       পিঁড়ি, হামানদিস্তা
  ambiguous            You cannot decide, or the answer depends on which form
                       of the object is in the photo.

Rules
1. Judge the object, not the word. A global object with a Bangla name is
   still globally_common.
2. Do not think about how hard the object is to spot in a photo, or whether
   an AI model would name it correctly. Only the question above matters.
3. If you are not sure what a name refers to, open the example images listed
   in its row before you decide.
4. If a row is not an object name, or is the same object as another row,
   still give it a label, and write what you noticed in the note column.
5. Short codes are accepted: cs, gc, amb.
6. Do not change the name_id column. Save as CSV UTF-8.
"""


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def name_id(name_norm):
    return "N_" + hkey(name_norm)[:10]


def clean_label(v):
    v = (v or "").strip().lower().replace("-", "_").replace(" ", "_")
    v = SHORT_CODES.get(v, v)
    return v if v in LABELS else None


def write_csv(path, rows, cols):
    a5.write_csv(path, rows, cols)
    say(f"[write] {path} ({len(rows)} rows)")


# ----------------------------------------------------------------------------
# Agreement statistics
# ----------------------------------------------------------------------------
def category_counts(votes):
    """votes: list of per-item label lists -> per-item category count lists."""
    return [[v.count(lab) for lab in LABELS] for v in votes]


def fleiss_kappa(counts):
    """Fleiss' kappa. counts: per-item lists of how many raters chose each
    category; every item must have the same number of raters."""
    N = len(counts)
    n = sum(counts[0])
    q = len(counts[0])
    p_i = [(sum(c * c for c in row) - n) / (n * (n - 1)) for row in counts]
    p_bar = sum(p_i) / N
    p_j = [sum(row[j] for row in counts) / (N * n) for j in range(q)]
    p_e = sum(p * p for p in p_j)
    return None if p_e >= 1 else (p_bar - p_e) / (1 - p_e)


def gwet_ac1(counts, q=None):
    """Gwet's AC1 for several raters with the same number of raters per item.
    Less affected than kappa when one label dominates."""
    N = len(counts)
    n = sum(counts[0])
    q = q or len(counts[0])
    p_a = sum((sum(c * c for c in row) - n) / (n * (n - 1)) for row in counts) / N
    pi_k = [sum(row[k] for row in counts) / (N * n) for k in range(len(counts[0]))]
    p_e = sum(p * (1 - p) for p in pi_k) / (q - 1)
    return None if p_e >= 1 else (p_a - p_e) / (1 - p_e)


def cohen_kappa(a, b):
    n = len(a)
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    cats = set(a) | set(b)
    p_e = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return None if p_e >= 1 else (p_o - p_e) / (1 - p_e)


def r4(x):
    return None if x is None else round(x, 4)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def find_records(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        key = a5._pick(data, a5.RECORD_LIST_KEYS)
        if key is None:
            for k, v in data.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    key = k
                    break
        if key is not None:
            return data[key]
        if data and all(isinstance(v, dict) for v in data.values()):
            return [dict(v, image_id=v.get("image_id", k)) for k, v in data.items()]
    raise SystemExit("[boxes] ERROR: could not find the record list in the JSON.")


def load_boxes(path):
    """One dict per bounding box: image, house, capture, room, raw label."""
    say(f"[boxes] reading {path}")
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    records = find_records(data)
    first = records[0]
    k_img = a5._pick(first, a5.IMAGE_ID_KEYS)
    k_house = a5._pick(first, a5.HOUSE_KEYS)
    k_cap = a5._pick(first, a5.CAPTURE_KEYS)
    k_room = a5._pick(first, ROOM_KEYS)
    k_file = a5._pick(first, a5.IMAGE_FILE_KEYS)
    k_box, sample = None, None
    for rec in records:
        k = a5._pick(rec, BOX_LIST_KEYS)
        if k and rec[k]:
            k_box, sample = k, rec[k]
            break
    if k_box is None:
        raise SystemExit(f"[boxes] ERROR: no record holds a box list under any of "
                         f"{BOX_LIST_KEYS}. The first record's keys are {sorted(first)}. "
                         f"Add the right name to BOX_LIST_KEYS at the top of this script.")
    sample = sample[0] if isinstance(sample, list) else sample
    if not isinstance(sample, dict):
        raise SystemExit(f"[boxes] ERROR: boxes under '{k_box}' are not objects: {sample!r:.80}")
    k_lab = next((k for k in BOX_LABEL_KEYS if isinstance(sample.get(k), str)), None)
    if k_lab is None:
        raise SystemExit(f"[boxes] ERROR: no label field in a box. Its keys are "
                         f"{sorted(sample)}. Add the right name to BOX_LABEL_KEYS.")
    say(f"[boxes] fields: box_list='{k_box}' label='{k_lab}' room='{k_room}'")
    out = []
    for rec in records:
        img = str(rec[k_img])
        if k_house and rec.get(k_house):
            house = str(rec[k_house])
        else:
            m = a5.HOUSE_RE.match(img)
            if not m:
                raise SystemExit(f"[boxes] ERROR: no house id for image {img}")
            house = m.group(1)
        bx = rec.get(k_box) or []
        if isinstance(bx, dict):
            bx = [bx]
        for b in bx:
            out.append({"image_id": img, "house_id": house,
                        "capture_type": str(rec.get(k_cap)),
                        "room_type": str(rec.get(k_room)) if k_room else "",
                        "image_file": rec.get(k_file) if k_file else None,
                        "label_raw": b.get(k_lab)})
    say(f"[boxes] {len(records)} records, {len(out)} boxes")
    return out


# ----------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------
MASTER_COLS = ["name_id", "name_bn", "n_obj_name", "n_box", "in_obj_name_list", "in_box_list",
               "n_houses_obj_name", "houses_obj_name", "rooms_seen", "is_instruction_example",
               "example_images"]
RATER_COLS = ["name_id", "name_bn", "rooms_seen", "example_images", "label", "note"]


def build_table(items, boxes):
    obj = [it for it in items if it["qa_type"] == "obj_name"]
    n_obj = Counter(normalise(it["answer_bn"]) for it in obj)
    n_box = Counter(normalise(b["label_raw"]) for b in boxes)
    houses = defaultdict(set)
    for it in obj:
        houses[normalise(it["answer_bn"])].add(it["house_id"])
    rooms = defaultdict(set)
    for b in boxes:
        if b["room_type"].strip().lower() not in NO_ROOM:
            rooms[normalise(b["label_raw"])].add(b["room_type"])
    shots = defaultdict(dict)   # name -> {image_id: shown path}
    for it in obj:
        shots[normalise(it["answer_bn"])][it["image_id"]] = it.get("image_file") or it["image_id"]
    for b in boxes:
        d = shots[normalise(b["label_raw"])]
        if len(d) < 50:
            d.setdefault(b["image_id"], b.get("image_file") or b["image_id"])

    universe = sorted({n for n, c in n_obj.items() if c >= LIST_FLOOR}
                      | {n for n, c in n_box.items() if c >= LIST_FLOOR})
    rows = []
    for n in universe:
        ex = sorted(shots[n], key=lambda img: hkey(SEED, "4A-example", img))[:N_EXAMPLE_IMAGES]
        rows.append({
            "name_id": name_id(n),
            "name_bn": n,
            "n_obj_name": n_obj.get(n, 0),
            "n_box": n_box.get(n, 0),
            "in_obj_name_list": "yes" if n_obj.get(n, 0) >= LIST_FLOOR else "no",
            "in_box_list": "yes" if n_box.get(n, 0) >= LIST_FLOOR else "no",
            "n_houses_obj_name": len(houses[n]),
            "houses_obj_name": " ".join(sorted(houses[n])),
            "rooms_seen": "/".join(sorted(rooms[n])),
            "is_instruction_example": "yes" if n in EXAMPLE_NAMES else "no",
            "example_images": " ".join(shots[n][img] for img in ex),
        })
    ids = Counter(r["name_id"] for r in rows)
    clash = [i for i, c in ids.items() if c > 1]
    if clash:
        raise SystemExit(f"[build] ERROR: name_id clash {clash}; stop and ask.")
    return rows, n_obj, n_box


def cmd_build(args):
    items = a5.load_items(args.data)
    a5.validate_items(items, args.counts)
    boxes = load_boxes(args.data)

    say("[check] comparing with the verified figures in stats.json")
    n_obj_items = sum(1 for it in items if it["qa_type"] == "obj_name")
    if n_obj_items != EXPECTED_OBJ_NAME_ITEMS:
        raise SystemExit(f"[check] ERROR: {n_obj_items} obj_name items, stats.json says "
                         f"{EXPECTED_OBJ_NAME_ITEMS}")
    if len(boxes) != EXPECTED_BOXES:
        raise SystemExit(f"[check] ERROR: {len(boxes)} boxes, stats.json says {EXPECTED_BOXES}. "
                         f"The loader is probably reading the wrong box field.")
    blank = [b["image_id"] for b in boxes if not str(b["label_raw"] or "").strip()]
    if blank:
        raise SystemExit(f"[check] ERROR: {len(blank)} boxes have no label, e.g. {blank[:3]}; "
                         f"stats.json says every box is named. Check BOX_LABEL_KEYS.")
    raw = Counter(str(b["label_raw"]) for b in boxes)
    raw_ge5 = sum(1 for c in raw.values() if c >= 5)
    raw_single = sum(1 for c in raw.values() if c == 1)
    for label, got, want in (("distinct raw box labels", len(raw), EXPECTED_RAW_BOX_NAMES),
                             ("raw box labels with 5 or more", raw_ge5, EXPECTED_RAW_BOX_NAMES_GE5),
                             ("raw box-label singletons", raw_single, EXPECTED_RAW_BOX_SINGLETONS)):
        flag = "matches" if got == want else "WARNING: differs from"
        say(f"[check]   {label}: {got} ({flag} stats.json {want})")
    say(f"[check]   obj_name items {n_obj_items} and boxes {len(boxes)} match stats.json")

    rows, n_obj, n_box = build_table(items, boxes)
    both = sum(1 for r in rows if r["in_obj_name_list"] == "yes" and r["in_box_list"] == "yes")
    only_obj = [r for r in rows if r["in_obj_name_list"] == "yes" and r["in_box_list"] == "no"]
    only_box = [r for r in rows if r["in_obj_name_list"] == "no" and r["in_box_list"] == "yes"]
    say()
    say(f"[build] after normalise(): {sum(1 for c in n_box.values() if c >= LIST_FLOOR)} box-label "
        f"names and {sum(1 for c in n_obj.values() if c >= LIST_FLOOR)} obj_name names reach "
        f"{LIST_FLOOR} instances")
    say(f"[build] names to tag (union): {len(rows)}  = {both} on both lists, "
        f"{len(only_obj)} on the obj_name list only, {len(only_box)} on the box list only")
    if only_obj:
        say(f"[build] a box-only list would have missed: "
            f"{', '.join(r['name_bn'] for r in only_obj[:25])}{' ...' if len(only_obj) > 25 else ''}")

    out = args.out
    os.makedirs(out, exist_ok=True)
    master = os.path.join(out, "asset4a_master.csv")
    if os.path.exists(master) and not args.force:
        old = {r["name_id"] for r in a5.read_csv(master)}
        if old != {r["name_id"] for r in rows}:
            raise SystemExit(f"[guard] STOP: {master} exists with a different name list. "
                             f"If raters already have sheets, a new build would split the "
                             f"vote. Rerun with --force only if no sheet has been handed out.")
    write_csv(master, rows, MASTER_COLS)
    sheets = []
    for k in range(1, N_RATERS + 1):
        order = sorted(rows, key=lambda r: hkey(SEED, "4A-order", k, r["name_id"]))
        p = os.path.join(out, f"asset4a_rater_{k}.csv")
        write_csv(p, [dict(r, label="", note="") for r in order], RATER_COLS)
        sheets.append(p)
    instr = os.path.join(out, "asset4a_rater_instructions.txt")
    with open(instr, "w", encoding="utf-8") as f:
        f.write(RATER_INSTRUCTIONS)
    say(f"[write] {instr}")

    here = os.path.dirname(os.path.abspath(__file__))
    manifest = {
        "asset": "ASSET-4A build",
        "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "normalise_py_sha256": sha256_file(os.path.join(here, "normalise.py")),
        "asset5a_probes_py_sha256": sha256_file(os.path.join(here, "asset5a_probes.py")),
        "dataset": {"path": os.path.basename(args.data), "sha256": sha256_file(args.data)},
        "rule": "Tag every normalise()d object name with at least 5 box labels or at least 5 "
                "obj_name answers (D50). Raters see no counts; each sheet has its own order "
                "(SHA-256 of '42|4A-order|<rater>|<name_id>').",
        "stats_json_checks": {"obj_name_items": n_obj_items, "boxes": len(boxes),
                              "raw_box_names": len(raw), "raw_box_names_ge5": raw_ge5,
                              "raw_box_singletons": raw_single},
        "names": {"total": len(rows), "both_lists": both, "obj_name_list_only": len(only_obj),
                  "box_list_only": len(only_box),
                  "obj_name_list_only_names": [r["name_bn"] for r in only_obj]},
        "outputs_sha256": {os.path.basename(p): sha256_file(p) for p in [master, instr] + sheets},
    }
    mpath = os.path.join(out, "asset4a_build_manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    say(f"[write] {mpath}")
    say()
    say("=" * 70)
    say(f"ASSET-4A build: {len(rows)} names to tag. Give each rater ONE sheet and the")
    say("instructions file. Nobody sees another rater's sheet until merge.")
    say("=" * 70)


# ----------------------------------------------------------------------------
# merge
# ----------------------------------------------------------------------------
def read_votes(master, paths):
    ids = [r["name_id"] for r in master]
    names = {r["name_id"]: r["name_bn"] for r in master}
    votes = {i: [] for i in ids}
    notes = defaultdict(list)
    for k, p in enumerate(paths, start=1):
        rows = a5.read_csv(p)
        got = {r.get("name_id") for r in rows}
        missing, extra = sorted(set(ids) - got), sorted(got - set(ids))
        if missing or extra:
            raise SystemExit(f"[merge] ERROR: sheet {k} ({p}) has {len(missing)} missing and "
                             f"{len(extra)} unknown name_ids, e.g. {(missing + extra)[:3]}. "
                             f"Was it made by a different build?")
        bad, damaged = [], 0
        by_id = {r["name_id"]: r for r in rows}
        for i in ids:
            lab = clean_label(by_id[i].get("label"))
            if lab is None:
                bad.append((names[i], by_id[i].get("label")))
            votes[i].append(lab)
            if normalise(by_id[i].get("name_bn", "")) != names[i]:
                damaged += 1
            if (by_id[i].get("note") or "").strip():
                notes[i].append(f"rater {k}: {by_id[i]['note'].strip()}")
        if bad:
            raise SystemExit(f"[merge] ERROR: sheet {k} has {len(bad)} rows without a valid "
                             f"label, e.g. {bad[:4]}. Allowed: {LABELS} or cs, gc, amb.")
        if damaged:
            say(f"[merge] WARNING: {damaged} Bangla names in sheet {k} differ from the master. "
                f"Usually Excel saved the wrong encoding. Labels are still matched by name_id.")
        say(f"[merge] sheet {k}: {len(ids)} labels read from {p}")
    return votes, notes


def decide(v):
    c = Counter(v)
    top, n = c.most_common(1)[0]
    if n == N_RATERS:
        return top, "unanimous"
    if n == 2:
        return top, "majority"
    return "ambiguous", "three-way split"


def agreement_block(ids, votes):
    vs = [votes[i] for i in ids]
    if not vs:
        return {"names": 0}
    counts = category_counts(vs)
    pairs = {}
    for a in range(N_RATERS):
        for b in range(a + 1, N_RATERS):
            pairs[f"rater{a + 1}_vs_rater{b + 1}"] = r4(cohen_kappa([v[a] for v in vs],
                                                                    [v[b] for v in vs]))
    return {"names": len(vs),
            "fleiss_kappa": r4(fleiss_kappa(counts)),
            "gwet_ac1": r4(gwet_ac1(counts, q=len(LABELS))),
            "pairwise_cohen_kappa": pairs,
            "unanimous": sum(1 for v in vs if len(set(v)) == 1),
            "two_to_one": sum(1 for v in vs if len(set(v)) == 2),
            "three_way_split": sum(1 for v in vs if len(set(v)) == 3),
            "cs_vs_gc_disagreements": sum(1 for v in vs
                                          if "culturally_specific" in v and "globally_common" in v)}


def pname_preview(master, final):
    out = {}
    for basis in ("n_obj_name", "n_box"):
        strata = {}
        for lab in LABELS:
            elig = [r for r in master if final[r["name_id"]] == lab
                    and int(r[basis]) >= (PNAME_FLOOR_CS if lab == "culturally_specific"
                                          else PNAME_FLOOR_OTHER)]
            houses = set()
            for r in elig:
                houses.update(r["houses_obj_name"].split())
            strata[lab] = {"eligible_names": len(elig),
                           "eligible_obj_name_items": sum(int(r["n_obj_name"]) for r in elig),
                           "houses": len(houses)}
        total = sum(s["eligible_obj_name_items"] for s in strata.values())
        out["floor counted on " + ("obj_name answers" if basis == "n_obj_name" else "box labels")] = {
            "strata": strata, "eligible_obj_name_items_total": total,
            "enough_for_P-NAME_1000": total >= PNAME_SIZE}
    return out


def cmd_merge(args):
    master_path = os.path.join(args.dir, "asset4a_master.csv")
    master = a5.read_csv(master_path)
    if len(args.raters) != N_RATERS:
        raise SystemExit(f"[merge] ERROR: need exactly {N_RATERS} rater sheets")
    votes, notes = read_votes(master, args.raters)
    ids = [r["name_id"] for r in master]
    names = {r["name_id"]: r["name_bn"] for r in master}

    final, pattern, decided_by = {}, {}, {}
    for i in ids:
        final[i], pattern[i] = decide(votes[i])
        decided_by[i] = "vote" if pattern[i] != "three-way split" else "split, left ambiguous"

    if args.session:
        srows = a5.read_csv(args.session)
        applied = 0
        for r in srows:
            i = r.get("name_id")
            tag = (r.get("session_tag") or "").strip()
            if not tag:
                continue
            if i not in pattern:
                raise SystemExit(f"[merge] ERROR: session file names unknown name_id {i}")
            if pattern[i] != "three-way split":
                raise SystemExit(f"[merge] ERROR: session file changes {names[i]}, which was "
                                 f"decided by a {pattern[i]} vote. Only three-way splits go "
                                 f"to the session (D51).")
            lab = clean_label(tag)
            if lab is None:
                raise SystemExit(f"[merge] ERROR: session_tag '{tag}' for {names[i]} is not a label")
            final[i], decided_by[i] = lab, "session, unanimous"
            applied += 1
        say(f"[merge] session file applied to {applied} three-way splits")

    agree_all = agreement_block(ids, votes)
    non_ex = [r["name_id"] for r in master if r["is_instruction_example"] == "no"]
    agree_nonex = agreement_block(non_ex, votes)
    strata = {}
    for lab in LABELS:
        rs = [r for r in master if final[r["name_id"]] == lab]
        hs = set()
        for r in rs:
            hs.update(r["houses_obj_name"].split())
        strata[lab] = {"names": len(rs), "obj_name_items": sum(int(r["n_obj_name"]) for r in rs),
                       "box_instances": sum(int(r["n_box"]) for r in rs), "houses": len(hs)}
    unanimous_only = {lab: sum(1 for i in ids if pattern[i] == "unanimous" and final[i] == lab)
                      for lab in LABELS}
    preview = pname_preview(master, final)

    rows = []
    for r in master:
        i = r["name_id"]
        rows.append({"name_id": i, "name_bn": r["name_bn"], "n_obj_name": r["n_obj_name"],
                     "n_box": r["n_box"], "vote_1": votes[i][0], "vote_2": votes[i][1],
                     "vote_3": votes[i][2], "pattern": pattern[i],
                     "cultural_specificity": final[i], "decided_by": decided_by[i],
                     "notes": " | ".join(notes[i])})
    cols = ["name_id", "name_bn", "n_obj_name", "n_box", "vote_1", "vote_2", "vote_3",
            "pattern", "cultural_specificity", "decided_by", "notes"]

    splits = [{"name_id": i, "name_bn": names[i], "vote_1": votes[i][0], "vote_2": votes[i][1],
               "vote_3": votes[i][2], "session_tag": ""} for i in ids
              if pattern[i] == "three-way split"]
    write_csv(os.path.join(args.dir, "asset4a_splits.csv"), splits,
              ["name_id", "name_bn", "vote_1", "vote_2", "vote_3", "session_tag"])
    draft = os.path.join(args.dir, "cultural_specificity_DRAFT.csv")
    write_csv(draft, rows, cols)

    report = {
        "asset": "ASSET-4A merge",
        "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "master_sha256": sha256_file(master_path),
        "rater_sheets_sha256": {os.path.basename(p): sha256_file(p) for p in args.raters},
        "session_file_sha256": sha256_file(args.session) if args.session else None,
        "rule": "Majority of three decides; a three-way split is ambiguous unless the three "
                "agree unanimously in a session. Agreement is computed on the independent "
                "votes only (D51).",
        "agreement_all_names": agree_all,
        "agreement_without_instruction_examples": agree_nonex,
        "strata_final": strata,
        "unanimous_only_names": unanimous_only,
        "pname_preview_D42": preview,
        "rows_with_rater_notes": sum(1 for i in ids if notes[i]),
    }

    if args.freeze:
        final_path = args.final
        if os.path.exists(final_path):
            old = {r["name_id"]: r["cultural_specificity"] for r in a5.read_csv(final_path)}
            new = {r["name_id"]: r["cultural_specificity"] for r in rows}
            changed = sorted(i for i in set(old) | set(new) if old.get(i) != new.get(i))
            if changed and not args.force:
                raise SystemExit(f"[guard] STOP: {final_path} is frozen and {len(changed)} tags "
                                 f"would change, e.g. {[names.get(i, i) for i in changed[:3]]}. "
                                 f"Rerun with --force only with a decision row that says why.")
        write_csv(final_path, rows, cols)
        report["frozen_file"] = os.path.basename(final_path)
        report["frozen_file_sha256"] = sha256_file(final_path)

    rpath = os.path.join(args.dir, "asset4a_agreement.json")
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    say(f"[write] {rpath}")

    k, ac1 = agree_all["fleiss_kappa"], agree_all["gwet_ac1"]
    say()
    say("=" * 70)
    say(f"Agreement on {agree_all['names']} names (independent votes): Fleiss kappa {k}, "
        f"Gwet AC1 {ac1}")
    say(f"  without the instruction examples ({agree_nonex['names']} names): Fleiss kappa "
        f"{agree_nonex.get('fleiss_kappa')}, AC1 {agree_nonex.get('gwet_ac1')}")
    say(f"  unanimous {agree_all['unanimous']}, two-to-one {agree_all['two_to_one']}, "
        f"three-way splits {agree_all['three_way_split']}, culturally_specific vs "
        f"globally_common disagreements {agree_all['cs_vs_gc_disagreements']}")
    if k is not None and k < 0.40:
        if ac1 is not None and ac1 < 0.40:
            say("  READING: kappa and AC1 are both below 0.40. The definition is not working.")
            say("  Per D51: revise the test question once and re-vote ALL names. Stop here.")
        else:
            say("  READING: kappa below 0.40 but AC1 is not. Most names share one label, which")
            say("  pulls kappa down. Report both values and keep the tags (D51).")
    for lab in LABELS:
        s = strata[lab]
        say(f"  {lab:20s} {s['names']:4d} names, {s['obj_name_items']:5d} obj_name items, "
            f"{s['houses']:3d} houses")
    for basis, p in preview.items():
        cs = p["strata"]["culturally_specific"]
        say(f"P-NAME preview, D42 {basis}: {p['eligible_obj_name_items_total']} eligible items "
            f"(culturally_specific {cs['eligible_obj_name_items']} items, {cs['houses']} houses)"
            f"{'' if p['enough_for_P-NAME_1000'] else '  BELOW 1,000'}")
    if splits:
        say(f"Three-way splits: {len(splits)}, listed in asset4a_splits.csv for the session.")
    say(f"{'FROZEN: ' + args.final if args.freeze else 'Draft only. Add --freeze when final.'}")
    say("=" * 70)


# ----------------------------------------------------------------------------
# Self-test on SYNTHETIC data
# ----------------------------------------------------------------------------
SYN_OBJ = [("চেয়ার", 300), ("ফ্রিজ", 200), ("বিছানা", 200), ("টেলিভিশন", 150), ("সোফা", 150),
           ("বেসিন", 150), ("আলমারি", 150), ("সুইচ বোর্ড", 120), ("বালতি", 100), ("টব", 60),
           ("ফিল্টার", 50), ("পাতিল", 40), ("বটি", 30), ("হাঁড়ি", 25), ("দেকচি", 20),
           ("মোড়া", 12), ("বদনা", 8), ("গামছা", 6), ("কুলা", 3), ("দরজা", 2)]
SYN_SCENE = [("চেয়ার", 400), ("বিছানা", 300), ("সোফা", 200), ("পর্দা", 150), ("টেবিল", 100),
             ("আলমারি", 100), ("ফ্রিজ", 90), ("সিলিং ফ্যান", 80), ("বেসিন", 80), ("কমোড", 80),
             ("দরজা", 60), ("জানালা", 40), ("মশারি", 4)]
SYN_NO_BOX = {"বদনা", "গামছা"}
SYN_TRUTH = {"culturally_specific": {"বটি", "হাঁড়ি", "দেকচি", "মোড়া", "বদনা", "গামছা", "পাতিল"},
             "ambiguous": {"আলমারি", "সুইচ বোর্ড", "ফিল্টার"}}


def synthetic_dataset(counts_csv):
    """SYNTHETIC records shaped like the real counts, with 3,620 boxes."""
    avail, capinfo = a5.availability_from_counts_csv(counts_csv)
    items = a5.synthetic_items(avail, capinfo)
    close = sorted({it["image_id"] for it in items if it["capture_type"] == "closeup"},
                   key=lambda i: hkey("syn4a", i))
    names = [n for n, c in SYN_OBJ for _ in range(c)]
    names += [f"সিনথেটিক বস্তু {k:03d}" for k in range(len(close) - len(names))]
    close_name = dict(zip(close, names))
    for it in items:
        if it["qa_type"] == "obj_name":
            it["answer_bn"] = close_name[it["image_id"]] + "।"
    scene = sorted({it["image_id"] for it in items if it["capture_type"] != "closeup"},
                   key=lambda i: hkey("syn4a-scene", i))
    snames = [n for n, c in SYN_SCENE for _ in range(c)]
    snames += [f"সিনথেটিক দৃশ্য {k:04d}" for k in range(len(scene) - len(snames))]
    scene_name = dict(zip(scene, snames))
    # Raw-form variants that normalise() must merge: trailing space, decomposed nukta.
    variants = {scene[0]: "চেয়ার ", scene[1]: "চে\u09af\u09bcার", scene[2]: "চেয়ার।"}
    for img, lab in variants.items():
        if normalise(scene_name[img]) == normalise("চেয়ার"):
            scene_name[img] = lab
    budget = EXPECTED_BOXES - len(scene)
    boxed = set()
    for img in close:
        if len(boxed) == budget:
            break
        if normalise(close_name[img]) not in {normalise(x) for x in SYN_NO_BOX}:
            boxed.add(img)
    recs = {}
    for it in items:
        r = recs.setdefault(it["image_id"], {
            "image_id": it["image_id"], "house_id": it["house_id"],
            "capture_type": it["capture_type"],
            "room_type": "NA" if it["capture_type"] == "closeup" else "bedroom",
            "qa": [], "boxes": []})
        r["qa"].append({"qa_id": it["qa_id"], "qa_type": it["qa_type"],
                        "question": it["question_bn"], "answer": it["answer_bn"]})
    for img, r in recs.items():
        if r["capture_type"] != "closeup":
            r["boxes"] = [{"object_id": img + "_b1", "label": scene_name[img]}]
        elif img in boxed:
            r["boxes"] = [{"object_id": img + "_b1", "label": close_name[img]}]
    return {"images": list(recs.values())}


def simulated_votes(master_rows):
    N = normalise
    truth_sets = {lab: {N(x) for x in s} for lab, s in SYN_TRUTH.items()}

    def truth(n):
        for lab, s in truth_sets.items():
            if n in s:
                return lab
        return "globally_common"
    out = {1: {}, 2: {}, 3: {}}
    for r in master_rows:
        n, t = r["name_bn"], truth(r["name_bn"])
        out[1][r["name_id"]] = t
        out[2][r["name_id"]] = "culturally_specific" if n == N("আলমারি") else t
        v3 = "gc" if n == N("সুইচ বোর্ড") else t.upper()
        out[3][r["name_id"]] = v3
        if n == N("ফিল্টার"):
            out[2][r["name_id"]] = "culturally_specific"
            out[3][r["name_id"]] = "globally_common"
    return out


def fill_sheets(out, votes):
    paths = []
    for k in range(1, N_RATERS + 1):
        rows = a5.read_csv(os.path.join(out, f"asset4a_rater_{k}.csv"))
        for r in rows:
            r["label"] = votes[k][r["name_id"]]
        p = os.path.join(out, f"filled_{k}.csv")
        a5.write_csv(p, rows, RATER_COLS)
        paths.append(p)
    return paths


def cmd_selftest(args):
    say("#" * 70)
    say("SELF-TEST. Part 1 checks the agreement formulas against textbook values.")
    say("Part 2 uses SYNTHETIC names and votes: no number it prints is a dataset figure.")
    say("#" * 70)
    wiki = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
            [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    fk = fleiss_kappa(wiki)
    assert abs(fk - 0.210) < 0.001, fk
    say(f"[selftest] OK  Fleiss kappa on the standard 10-item, 14-rater table: {fk:.3f} (expected 0.210)")
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    ck = cohen_kappa(a, b)
    assert abs(ck - 0.4) < 1e-9, ck
    say(f"[selftest] OK  Cohen kappa on the standard 50-item example: {ck:.3f} (expected 0.400)")
    two = [[(p == "y") + (q == "y"), (p == "n") + (q == "n")] for p, q in zip(a, b)]
    ac = gwet_ac1(two, q=2)
    assert abs(ac - 0.205 / 0.505) < 1e-9, ac
    say(f"[selftest] OK  Gwet AC1 on the same example: {ac:.4f} (expected 0.4059)")

    tmp = tempfile.mkdtemp(prefix="asset4a_selftest_")
    try:
        data = os.path.join(tmp, "synthetic_merged.json")
        with open(data, "w", encoding="utf-8") as f:
            json.dump(synthetic_dataset(args.counts), f, ensure_ascii=False)
        counts = os.path.join(tmp, "house_type_counts.csv")
        shutil.copy(args.counts, counts)
        out = os.path.join(tmp, "asset4a")
        say()
        say("[selftest] --- build ---")
        cmd_build(argparse.Namespace(data=data, counts=counts, out=out, force=False))
        master = a5.read_csv(os.path.join(out, "asset4a_master.csv"))
        N = normalise
        byname = {r["name_bn"]: r for r in master}
        for n in ("বদনা", "গামছা"):
            assert N(n) in byname and byname[N(n)]["in_box_list"] == "no", n
        assert N("দরজা") in byname and byname[N("দরজা")]["in_obj_name_list"] == "no"
        assert N("কুলা") not in byname and N("মশারি") not in byname
        with open(data, encoding="utf-8") as f:
            syn = json.load(f)
        raw_forms = Counter(b["label"] for r in syn["images"] for b in r["boxes"]
                            if normalise(b["label"]) == N("চেয়ার"))
        assert len(raw_forms) >= 3, raw_forms
        assert int(byname[N("চেয়ার")]["n_box"]) == sum(raw_forms.values())
        assert byname[N("বটি")]["is_instruction_example"] == "yes"
        s1 = a5.read_csv(os.path.join(out, "asset4a_rater_1.csv"))
        s2 = a5.read_csv(os.path.join(out, "asset4a_rater_2.csv"))
        assert "n_obj_name" not in s1[0] and "n_box" not in s1[0], "a sheet shows counts"
        assert [r["name_id"] for r in s1] != [r["name_id"] for r in s2], "sheets share an order"
        say("[selftest] OK  close-up-only names are on the list; names under 5 on both are not")
        say("[selftest] OK  raw variants merged by normalise(); sheets hide counts; orders differ")

        say("[selftest] --- merge (draft) ---")
        votes = simulated_votes(master)
        paths = fill_sheets(out, votes)
        final_path = os.path.join(tmp, "cultural_specificity.csv")
        margs = dict(dir=out, raters=paths, session=None, freeze=False, final=final_path, force=False)
        cmd_merge(argparse.Namespace(**margs))
        draft = {r["name_bn"]: r for r in a5.read_csv(os.path.join(out, "cultural_specificity_DRAFT.csv"))}
        assert draft[N("ফিল্টার")]["pattern"] == "three-way split"
        assert draft[N("ফিল্টার")]["cultural_specificity"] == "ambiguous"
        assert draft[N("আলমারি")]["cultural_specificity"] == "ambiguous"
        assert draft[N("আলমারি")]["pattern"] == "majority"
        assert draft[N("বদনা")]["cultural_specificity"] == "culturally_specific"
        assert not os.path.exists(final_path), "a draft merge wrote the frozen file"
        say("[selftest] OK  majority rule, split default and short codes all work; draft is not frozen")

        say("[selftest] --- session on a non-split row: must stop ---")
        bad = os.path.join(out, "bad_session.csv")
        a5.write_csv(bad, [{"name_id": byname[N("আলমারি")]["name_id"], "session_tag": "gc"}],
                     ["name_id", "session_tag"])
        try:
            cmd_merge(argparse.Namespace(**dict(margs, session=bad)))
            raise AssertionError("a majority decision was re-opened")
        except SystemExit as e:
            say(f"[selftest] OK  stopped: {str(e)[:70]}...")

        say("[selftest] --- session, then freeze ---")
        sp = os.path.join(out, "asset4a_splits.csv")
        srows = a5.read_csv(sp)
        for r in srows:
            r["session_tag"] = "globally_common"
        a5.write_csv(sp, srows, list(srows[0].keys()))
        cmd_merge(argparse.Namespace(**dict(margs, session=sp, freeze=True)))
        final = {r["name_bn"]: r for r in a5.read_csv(final_path)}
        assert final[N("ফিল্টার")]["cultural_specificity"] == "globally_common"
        assert final[N("ফিল্টার")]["decided_by"] == "session, unanimous"
        say("[selftest] OK  the session settles only the split; the file is frozen")

        say("[selftest] --- change one vote and freeze again: must stop ---")
        votes[2][byname[N("চেয়ার")]["name_id"]] = "culturally_specific"
        votes[3][byname[N("চেয়ার")]["name_id"]] = "culturally_specific"
        paths = fill_sheets(out, votes)
        try:
            cmd_merge(argparse.Namespace(**dict(margs, raters=paths, session=sp, freeze=True)))
            raise AssertionError("the frozen file was overwritten")
        except SystemExit as e:
            say(f"[selftest] OK  the frozen-file guard stopped the run: {str(e)[:60]}...")

        say("[selftest] --- a blank label: must stop ---")
        rows = a5.read_csv(paths[0])
        rows[0]["label"] = ""
        a5.write_csv(paths[0], rows, RATER_COLS)
        try:
            cmd_merge(argparse.Namespace(**margs))
            raise AssertionError("a blank label was accepted")
        except SystemExit as e:
            say(f"[selftest] OK  stopped: {str(e)[:70]}...")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    say()
    say("SELF-TEST PASSED: agreement formulas match textbook values; build, merge,")
    say("session, freeze and both guards work end to end on synthetic data.")


def main():
    ap = argparse.ArgumentParser(description="ASSET-4A: cultural_specificity tags")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--counts", default="results/assets/house_type_counts.csv")
    s = sub.add_parser("build")
    s.add_argument("--data", required=True)
    s.add_argument("--counts", default="results/assets/house_type_counts.csv")
    s.add_argument("--out", default="results/assets/asset4a")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("merge")
    s.add_argument("--raters", nargs=N_RATERS, required=True)
    s.add_argument("--dir", default="results/assets/asset4a")
    s.add_argument("--session", default=None)
    s.add_argument("--freeze", action="store_true")
    s.add_argument("--final", default="results/assets/cultural_specificity.csv")
    s.add_argument("--force", action="store_true")
    args = ap.parse_args()
    {"selftest": cmd_selftest, "build": cmd_build, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
