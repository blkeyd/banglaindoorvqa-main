"""
asset1_cluster_counts.py
BanglaIndoorVQA  ASSET-1: confusable cluster instance counts.

What it does, in order:
  1. Parses the confusable section of BanglaIndoorVQA_Lexicon.txt into clusters.
  2. Loads the merged dataset JSON.
  3. Counts, for every lexicon term: how many bounding box labels carry it,
     and how many obj_name gold answers carry it.
  4. Rolls those up per cluster and writes cluster_counts.csv.
  5. Prints the G1 gate reading.
  6. Counts QA per house per question type, which is the P-CORE feasibility
     check that ASSET-5 needs before probe sets can be built.

Outputs, all written under --outdir:
  clusters_parsed.csv       what the lexicon parser found. CHECK THIS BY EYE.
  cluster_counts.csv        the ASSET-1 deliverable.
  unmatched_obj_name.csv    obj_name answers that matched no lexicon term.
  house_type_counts.csv     QA per house per type per capture type.
  gate_G1.txt               the G1 reading, in words.
  pcore_feasibility.txt     whether 30 per house balanced across 8 types works.

Run:
  python asset1_cluster_counts.py \
      --lexicon BanglaIndoorVQA_Lexicon.txt \
      --data merged_dataset.json \
      --outdir results/assets
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict

from normalise import normalise

random.seed(42)

EM_DASH = "\u2014"
CLUSTER_HEADING = re.compile(r"^-{2,}\s*(.+?)\s*-{2,}\s*$")
MEMBER_LINE = re.compile(r"^(?P<term>\S.*?)\s+" + EM_DASH + r"\s+(?P<desc>.*)$")
COLLOQUIAL = re.compile(r"[\[(]\s*কথ্য[^\]\)]*[:：]\s*([^\]\)]+)[\]\)]")

# Gate thresholds. Fixed by the Guide, section 3.5. Do not tune these.
MIN_CLUSTER_TOTAL = 30
MIN_MEMBER_COUNT = 8
MIN_REPORTABLE_MEMBERS = 2   # see the note in the README section of the reply
MIN_QUALIFYING_CLUSTERS = 3

# P-CORE target from the Guide, section 3.2.
PCORE_PER_HOUSE = 30
PCORE_N_TYPES = 8

EXPECTED_RECORD_FIELDS = ["house_id", "capture_type", "objects", "qa_pairs"]


def log(msg=""):
    print(msg, flush=True)


# ----------------------------------------------------------------------
# Step 1: parse the lexicon
# ----------------------------------------------------------------------

def parse_lexicon(path):
    """Return a list of dicts: cluster_id, member, surface_forms."""
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()

    text = unicodedata.normalize("NFC", raw)
    lines = text.split("\n")

    # The confusable section is the block that runs from the first dashed
    # cluster heading  --- name ---  to the next long ==== rule after it.
    # Dashed headings also appear later in the room by room lists, so the
    # closing ==== is what keeps us inside the right block.
    start = None
    for i, line in enumerate(lines):
        if CLUSTER_HEADING.match(line.strip()):
            start = i
            break
    if start is None:
        raise SystemExit(
            "ERROR: no cluster heading of the form  --- name ---  found in the "
            "lexicon. The file format has changed. Stop and check it by hand."
        )

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("===="):
            end = j
            break

    log(f"  lexicon confusable section: lines {start + 1} to {end}")

    rows = []
    current_cluster = None
    for line in lines[start:end]:
        stripped = line.strip()
        if not stripped or stripped.startswith("="):
            continue

        heading = CLUSTER_HEADING.match(stripped)
        if heading:
            current_cluster = heading.group(1).strip()
            continue

        member = MEMBER_LINE.match(stripped)
        if member and current_cluster:
            term = member.group("term").strip()
            desc = member.group("desc").strip()

            # Surface forms: the term itself, plus anything separated by / or =,
            # plus a colloquial synonym recorded in the description.
            forms = [term]
            for sep in ("/", "="):
                expanded = []
                for f in forms:
                    expanded.extend(part.strip() for part in f.split(sep))
                forms = [f for f in expanded if f]
            for extra in COLLOQUIAL.findall(desc):
                for part in extra.split(","):
                    part = part.strip()
                    if part:
                        forms.append(part)

            seen = set()
            unique_forms = []
            for f in forms:
                n = normalise(f)
                if n and n not in seen:
                    seen.add(n)
                    unique_forms.append(f)

            rows.append({
                "cluster_id": current_cluster,
                "member": term,
                "surface_forms": unique_forms,
                "description": desc,
            })

    return rows


# ----------------------------------------------------------------------
# Step 2: load and validate the dataset
# ----------------------------------------------------------------------

def load_records(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        list_keys = [k for k, v in data.items() if isinstance(v, list)]
        if len(list_keys) != 1:
            raise SystemExit(
                "ERROR: the JSON is an object, and I cannot tell which key holds "
                f"the records. Keys with lists: {list_keys}\n"
                "Re-run with a plain list at the top level, or tell me the key."
            )
        log(f"  top level is an object; using key '{list_keys[0]}'")
        records = data[list_keys[0]]
    elif isinstance(data, list):
        records = data
    else:
        raise SystemExit(f"ERROR: top level of the JSON is {type(data).__name__}, "
                         "expected a list of image records.")

    if not records:
        raise SystemExit("ERROR: the JSON contains zero records.")

    missing = [f for f in EXPECTED_RECORD_FIELDS if f not in records[0]]
    if missing:
        raise SystemExit(
            "ERROR: the first record is missing these expected fields: "
            f"{missing}\nFields it does have: {sorted(records[0].keys())}\n"
            "The field names in this script no longer match your data. Stop and "
            "fix the names at the top of the file rather than guessing."
        )

    return records


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default="BanglaIndoorVQA_Lexicon.txt")
    ap.add_argument("--data", default="merged_dataset.json")
    ap.add_argument("--outdir", default="results/assets")
    args = ap.parse_args()

    for p in (args.lexicon, args.data):
        if not os.path.exists(p):
            raise SystemExit(f"ERROR: file not found: {p}\n"
                             "Check the path. In Colab, run  !ls  to see what is "
                             "actually in the current folder.")

    os.makedirs(args.outdir, exist_ok=True)

    log("=" * 68)
    log("ASSET-1: confusable cluster instance counts")
    log("=" * 68)
    log(f"lexicon : {args.lexicon}")
    log(f"data    : {args.data}")
    log(f"outdir  : {args.outdir}")
    log()

    # -- Step 1 ---------------------------------------------------------
    log("STEP 1  parsing the lexicon confusable section")
    lex_rows = parse_lexicon(args.lexicon)
    clusters = sorted({r["cluster_id"] for r in lex_rows})
    log(f"  found {len(clusters)} clusters and {len(lex_rows)} member terms")
    if len(lex_rows) < 20:
        raise SystemExit(
            f"ERROR: only {len(lex_rows)} member terms parsed, which is far too "
            "few. The lexicon layout has changed and the parser is reading the "
            "wrong block. Do not trust any count from this run."
        )

    # A term that appears in two clusters would be counted twice.
    by_norm = defaultdict(list)
    for r in lex_rows:
        by_norm[normalise(r["member"])].append(r["cluster_id"])
    duplicates = {k: v for k, v in by_norm.items() if len(v) > 1}
    if duplicates:
        log("  WARNING: these terms appear in more than one cluster and will be "
            "counted in each:")
        for term, cl in duplicates.items():
            log(f"    {term}  ->  {cl}")
        log("  Decide which cluster owns each of them before using the counts.")
    log()

    # Map every normalised surface form to (cluster_id, member).
    form_to_member = {}
    form_collisions = []
    for r in lex_rows:
        for f in r["surface_forms"]:
            n = normalise(f)
            if n in form_to_member and form_to_member[n] != (r["cluster_id"], r["member"]):
                form_collisions.append((n, form_to_member[n], (r["cluster_id"], r["member"])))
            form_to_member[n] = (r["cluster_id"], r["member"])
    if form_collisions:
        log(f"  NOTE: {len(form_collisions)} surface form collisions, "
            "last definition wins. Listed in clusters_parsed.csv.")
        log()

    # -- Step 2 ---------------------------------------------------------
    log("STEP 2  loading the dataset")
    records = load_records(args.data)
    log(f"  loaded {len(records)} image records")

    n_objects = sum(len(r.get("objects") or []) for r in records)
    n_qa = sum(len(r.get("qa_pairs") or []) for r in records)
    n_houses = len({r["house_id"] for r in records})
    log(f"  {n_objects} bounding box objects, {n_qa} QA pairs, {n_houses} houses")
    log()

    # -- Step 3 ---------------------------------------------------------
    log("STEP 3  counting instances")
    box_exact = Counter()
    name_exact = Counter()
    box_loose = Counter()
    name_loose = Counter()
    unmatched = Counter()
    n_obj_name = 0

    # Loose matching is WHOLE WORD, never substring. A substring rule makes the
    # short term  দা  match inside  পর্দা  and  ফুলদানি , which is five false
    # positives out of five on the sample file. Whole word matching costs
    # nothing and removes that entire class of error.
    form_tokens = {f: f.split() for f in form_to_member}

    def loose_hits(norm_text):
        """Every lexicon form that appears as a complete word or word run."""
        tokens = norm_text.split()
        hits = set()
        for form, ftok in form_tokens.items():
            if not ftok:
                continue
            span = len(ftok)
            for i in range(len(tokens) - span + 1):
                if tokens[i:i + span] == ftok:
                    hits.add(form_to_member[form])
                    break
        return hits

    for rec in records:
        for obj in (rec.get("objects") or []):
            n = normalise(obj.get("bangla_name"))
            if n in form_to_member:
                box_exact[form_to_member[n]] += 1
            for key in loose_hits(n):
                box_loose[key] += 1

        for qa in (rec.get("qa_pairs") or []):
            if qa.get("type") != "obj_name":
                continue
            n_obj_name += 1
            n = normalise(qa.get("answer"))
            if n in form_to_member:
                name_exact[form_to_member[n]] += 1
            hits = loose_hits(n)
            for key in hits:
                name_loose[key] += 1
            if not hits:
                unmatched[n] += 1

    log(f"  scanned {n_objects} box labels and {n_obj_name} obj_name answers")
    log(f"  exact matches: {sum(box_exact.values())} boxes, "
        f"{sum(name_exact.values())} obj_name answers")
    log(f"  obj_name answers matching no lexicon term at all: "
        f"{sum(unmatched.values())}")
    log()

    # -- Step 4 ---------------------------------------------------------
    log("STEP 4  writing CSV outputs")

    parsed_path = os.path.join(args.outdir, "clusters_parsed.csv")
    with open(parsed_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cluster_id", "member", "surface_forms", "description"])
        for r in lex_rows:
            w.writerow([r["cluster_id"], r["member"],
                        " | ".join(r["surface_forms"]), r["description"]])
    log(f"  wrote {parsed_path}  ({len(lex_rows)} rows)")

    counts_path = os.path.join(args.outdir, "cluster_counts.csv")
    rows_out = []
    for r in lex_rows:
        key = (r["cluster_id"], r["member"])
        b, nm = box_exact.get(key, 0), name_exact.get(key, 0)
        bl, nl = box_loose.get(key, 0), name_loose.get(key, 0)
        rows_out.append({
            "cluster_id": r["cluster_id"],
            "member": r["member"],
            "box_count": b,
            "obj_name_count": nm,
            "total": b + nm,
            "box_count_loose": bl,
            "obj_name_count_loose": nl,
            "total_loose": bl + nl,
        })
    rows_out.sort(key=lambda x: (-sum(
        y["total"] for y in rows_out if y["cluster_id"] == x["cluster_id"]),
        x["cluster_id"], -x["total"]))
    with open(counts_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    log(f"  wrote {counts_path}  ({len(rows_out)} rows)")

    unmatched_path = os.path.join(args.outdir, "unmatched_obj_name.csv")
    with open(unmatched_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["normalised_answer", "count"])
        for text, c in unmatched.most_common():
            w.writerow([text, c])
    log(f"  wrote {unmatched_path}  ({len(unmatched)} distinct answers)")
    log()

    # -- Step 5 ---------------------------------------------------------
    log("STEP 5  G1 gate reading")
    cluster_total = Counter()
    cluster_members_ok = defaultdict(list)
    for row in rows_out:
        cluster_total[row["cluster_id"]] += row["total"]
        if row["total"] >= MIN_MEMBER_COUNT:
            cluster_members_ok[row["cluster_id"]].append(row["member"])

    qualifying = []
    for cid in clusters:
        total = cluster_total.get(cid, 0)
        ok_members = cluster_members_ok.get(cid, [])
        if total >= MIN_CLUSTER_TOTAL and len(ok_members) >= MIN_REPORTABLE_MEMBERS:
            qualifying.append((cid, total, ok_members))

    lines = []
    lines.append("G1  do at least 3 confusable clusters clear the threshold?")
    lines.append("")
    lines.append(f"rule: cluster total >= {MIN_CLUSTER_TOTAL} AND at least "
                 f"{MIN_REPORTABLE_MEMBERS} members with >= {MIN_MEMBER_COUNT} each")
    lines.append("")
    for cid in sorted(clusters, key=lambda c: -cluster_total.get(c, 0)):
        total = cluster_total.get(cid, 0)
        ok = cluster_members_ok.get(cid, [])
        mark = "PASS" if (cid, total, ok) in qualifying else "no  "
        lines.append(f"  {mark}  total={total:5d}  members>=8: {len(ok):2d}   {cid}")
    lines.append("")
    lines.append(f"qualifying clusters: {len(qualifying)} "
                 f"(threshold is {MIN_QUALIFYING_CLUSTERS})")
    if len(qualifying) >= MIN_QUALIFYING_CLUSTERS:
        lines.append("READING: G1 PASSES. The specificity analysis stays at "
                     "cluster level. P-CLUST is drawn from the qualifying clusters.")
    else:
        lines.append("READING: G1 FAILS. Fall back to object level scoring over "
                     "the 48 names with 20 or more instances, with automatically "
                     "constructed distractors. Contribution 2 survives in reduced "
                     "form. Tell the team and record it in the decisions log.")
    gate_text = "\n".join(lines)
    log(gate_text)
    gate_path = os.path.join(args.outdir, "gate_G1.txt")
    with open(gate_path, "w", encoding="utf-8") as fh:
        fh.write(gate_text + "\n")
    log()
    log(f"  wrote {gate_path}")
    log()

    # -- Step 6 ---------------------------------------------------------
    log("STEP 6  P-CORE feasibility check, per house")
    house_type = defaultdict(Counter)
    house_capture = defaultdict(Counter)
    all_types = set()
    for rec in records:
        h = rec["house_id"]
        house_capture[h][rec.get("capture_type", "NA")] += 1
        for qa in (rec.get("qa_pairs") or []):
            t = qa.get("type", "NA")
            all_types.add(t)
            house_type[h][t] += 1
    all_types = sorted(all_types)

    per_type_needed = -(-PCORE_PER_HOUSE // PCORE_N_TYPES)   # ceiling, = 4
    houses_short_total = []
    houses_short_type = []
    hrows = []
    for h in sorted(house_type):
        counts = house_type[h]
        total = sum(counts.values())
        min_type = min(counts.get(t, 0) for t in all_types)
        max_balanced = min_type * len(all_types)
        row = {"house_id": h, "qa_total": total}
        for t in all_types:
            row[t] = counts.get(t, 0)
        for c in ("overview", "group", "closeup"):
            row["img_" + c] = house_capture[h].get(c, 0)
        row["min_type_count"] = min_type
        row["max_balanced_draw"] = max_balanced
        hrows.append(row)
        if total < PCORE_PER_HOUSE:
            houses_short_total.append((h, total))
        if min_type < per_type_needed:
            houses_short_type.append((h, min_type))

    hpath = os.path.join(args.outdir, "house_type_counts.csv")
    with open(hpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hrows[0].keys()))
        w.writeheader()
        w.writerows(hrows)
    log(f"  wrote {hpath}  ({len(hrows)} houses)")

    flines = []
    flines.append("P-CORE feasibility: 30 QA per house, balanced across 8 types")
    flines.append("")
    flines.append(f"houses seen: {len(hrows)}")
    flines.append(f"per type quota if balanced: {per_type_needed} "
                  f"({PCORE_PER_HOUSE} / {PCORE_N_TYPES}, rounded up)")
    flines.append("")
    flines.append(f"houses with fewer than {PCORE_PER_HOUSE} QA in total: "
                  f"{len(houses_short_total)}")
    for h, t in houses_short_total:
        flines.append(f"    {h}: {t} QA")
    flines.append("")
    flines.append(f"houses where at least one question type has fewer than "
                  f"{per_type_needed} items: {len(houses_short_type)}")
    for h, m in sorted(houses_short_type, key=lambda x: x[1]):
        flines.append(f"    {h}: smallest type has {m} items")
    flines.append("")
    if not houses_short_total and not houses_short_type:
        flines.append("READING: every house can supply a balanced draw of 30. "
                      "P-CORE as specified is constructible. ASSET-5 proceeds "
                      "unchanged.")
    else:
        flines.append("READING: P-CORE as specified is NOT constructible for "
                      "every house. ASSET-5 needs a documented rule for the "
                      "short houses, and that rule must be fixed and written "
                      "into the OSF pre-registration BEFORE any probe is drawn.")
    ftext = "\n".join(flines)
    log()
    log(ftext)
    fpath = os.path.join(args.outdir, "pcore_feasibility.txt")
    with open(fpath, "w", encoding="utf-8") as fh:
        fh.write(ftext + "\n")
    log()
    log(f"  wrote {fpath}")
    log()
    log("=" * 68)
    log(f"ASSET-1 done. 6 files in {args.outdir}")
    log("Next: open clusters_parsed.csv and check the parser by eye before "
        "trusting any count.")
    log("=" * 68)


if __name__ == "__main__":
    main()
