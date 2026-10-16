"""
asset9_scorer.py  (ASSET-9, stage 2)
Commands around src/scoring.py. Script version: see SCRIPT_VERSION.

  audit            Check the lexicon and the vocabulary, then every gold of
                   one or more probe files. Stage 2 (Gate A): the frozen
                   files, --require-full, and probes/excluded_items.csv.
                   Excluded items have their own status: counted, never
                   scored (D110). Also checks the ASSET-5C fields against the
                   scorer (addendum A2) and lists the golds with extra words
                   (addendum C5).
  score            Score harness outputs: lines of cache.jsonl (addendum E4),
                   answer in answer_text. Forced choice reads the option file
                   only through src/fc_options.py (read_option_sets,
                   options_for) and refuses the draft file.
  dryrun-fc        Forced-choice wiring test: made-up answers for every option
                   set of the real option file go through the score command,
                   and every label is checked. No model output is read.
  proofread-sheet  The native-speaker proofreading of the ending list and the
                   refusal phrases (addendum B5, F5): one Excel sheet with
                   every distinct no-image pilot answer, blind to the scorer.
  proofread-read   Reads the filled sheet back: format check, and the
                   proofreader's refusal marks against the scorer's.

Printouts give ids and counts only, never question, gold or answer text.
Every file with dataset text or model output goes to Google Drive; the script
refuses to write inside the Git repo (setup file Part 7, rule 1).

Examples (from the repo root; D is the Drive project folder)
  python scripts/asset9_scorer.py audit --require-full \
      --probe "$D/probes/P-CORE-CLOSED.json" --probe "$D/probes/R-TEST.json" \
      --probe "$D/probes/P-NAME.json" --probe "$D/probes/PILOT.json" \
      --lexicon lexicon/lexicon.csv --vocab lexicon/closed_vocab.csv \
      --out-dir "$D/scored/asset9_stage2/audit"

  python scripts/asset9_scorer.py score \
      --outputs "$D/outputs/pilot/gemini/cache.jsonl" \
      --probe "$D/probes/PILOT.json" --options-file "$D/probes/fc_options.jsonl" \
      --lexicon lexicon/lexicon.csv --vocab lexicon/closed_vocab.csv \
      --out "$D/scored/asset9_stage2/pilot_test/pilot_scored.jsonl"
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import scoring as S                      # noqa: E402
import fc_options as FC                  # noqa: E402
from normalise import normalise          # noqa: E402

SCRIPT_VERSION = "asset9-stage2 2026-10-16"
SEED = 42
QA_TYPE_OF_KIND = {"room": "scene_id", "material": "obj_property",
                   "colour": "obj_property", "number": "count"}
DEFAULT_EXCLUDED = REPO / "probes" / "excluded_items.csv"
DEFAULT_KEEP = ("key", "model_id", "model_key", "prompt_version", "stage",
                "run_id", "house_id")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_of(path):
    return FC.file_sha256(path)


def refuse_inside_repo(path):
    """Data-derived files never go into the Git repo."""
    p = Path(path).resolve()
    if p == REPO or REPO in p.parents:
        sys.exit("STOP: %s is inside the Git repo (%s).\n"
                 "Files with dataset text or model output stay in Google"
                 " Drive. Choose a path under MyDrive/BanglaIndoorVQA/."
                 % (p, REPO))
    return p


def print_table(rows, columns, say=print):
    if not rows:
        say("    (no rows)")
        return
    cells = [[("" if r.get(c) is None else str(r.get(c))) for c in columns]
             for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells))
              for i, c in enumerate(columns)]
    say("    " + "  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    say("    " + "  ".join("-" * w for w in widths))
    for row in cells:
        say("    " + "  ".join(v.ljust(w) for v, w in zip(row, widths)))


def seeded_rank(key):
    """Same sampling method as the probes: lowest SHA-256 rank, seed 42."""
    return hashlib.sha256(("%d|ASSET-9|%s" % (SEED, key)).encode("utf-8")) \
        .hexdigest()


def fake_item(qa_type, gold, qa_id="selftest"):
    return {"qa_id": qa_id, "qa_type": qa_type, "answer_bn": gold}


def write_csv(path, rows, columns):
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c))
                        for c in columns})


class Report:
    """Prints a line and keeps it for the summary file. Lines hold ids and
    counts only."""

    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)

    def write(self, path):
        Path(path).write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def load_settings(args):
    return S.MatchSettings(spacing_insensitive=not args.no_spacing_rule,
                           suffix_tolerance=not args.no_suffix_rule)


def load_excluded_or_stop(path):
    try:
        return S.load_excluded(path)
    except (S.ExcludedListError, OSError) as exc:
        sys.exit("STOP: the excluded-items list cannot be read.\n%s" % exc)


def load_probes(paths):
    """{qa_id: item} over several probe files, and one info row per file.
    A qa_id found in two files must have the same type and gold."""
    items = OrderedDict()
    info = []
    for p in paths:
        recs = S.load_probe(p)
        info.append({"file": Path(p).name, "path": str(p),
                     "sha256": sha256_of(p), "items": len(recs)})
        for it in recs:
            old = items.get(it["qa_id"])
            if old is not None and (old["qa_type"] != it["qa_type"]
                                    or old["answer_bn"] != it["answer_bn"]):
                sys.exit("STOP: %s has another type or gold in %s than in an"
                         " earlier probe file." % (it["qa_id"], Path(p).name))
            items.setdefault(it["qa_id"], it)
    return items, info


# --------------------------------------------------------------------------
# audit, part 1: the lexicon and the vocabulary against the scoring rules
# --------------------------------------------------------------------------

def self_test_lexicon(lex, voc):
    """Every row of the lexicon must behave as Design 8.3 says: its own names
    are Tier 1, its generic names are Tier 2, its siblings are siblings,
    another object is wrong. Returns (number of checks, failures). A failure
    is a dict with an id-only message and a detail that may hold terms."""
    n = 0
    fails = []
    ids = sorted(lex.objects)

    def check(obj, cond, answer, want_label, want_strict, what):
        nonlocal n
        n += 1
        r = S.score_item(fake_item("obj_name", obj.name_bn), cond, answer,
                         lex, voc)
        if not r["scoreable"] or r["label"] != want_label \
                or (want_strict is not None and r["strict"] != want_strict):
            fails.append({
                "check": "lexicon self-test", "id": obj.name_id,
                "message": "%s: %s in %s gave %s, strict %s; expected %s,"
                           " strict %s" % (obj.name_id, what, cond, r["label"],
                                           r["strict"], want_label,
                                           want_strict),
                "detail": "answer %r" % answer})

    def other_label(obj, term, fallback):
        key = lex.index.key_of(term)
        if key in obj.keys_t1:
            return S.TIER1
        if key in obj.keys_t2:
            return S.TIER2
        return fallback

    for oid in ids:
        obj = lex.objects[oid]
        for i, t in enumerate(obj.t1_bn):
            check(obj, "N-BB", t, S.TIER1, 1, "own Bangla name %d" % i)
            check(obj, "N-BE", t, S.TIER1, 0, "own Bangla name %d" % i)
        for i, t in enumerate(obj.t1_en):
            check(obj, "N-BE", t, S.TIER1, 1, "own English name %d" % i)
            check(obj, "N-BB", t, S.TIER1, 0, "own English name %d" % i)
        for i, t in enumerate(obj.t2_bn):
            check(obj, "N-BB", t, S.TIER2, 0, "own generic Bangla name %d" % i)
        for i, t in enumerate(obj.t2_en):
            check(obj, "N-BE", t, S.TIER2, 0, "own generic English name %d" % i)
        sibs = lex.siblings_of(oid)
        for sid in sorted(sibs):
            sib = lex.objects[sid]
            check(obj, "N-BB", sib.name_bn,
                  other_label(obj, sib.name_bn, S.SIBLING), None,
                  "Bangla name of sibling %s" % sid)
            if sib.t1_en:
                check(obj, "N-BE", sib.t1_en[0],
                      other_label(obj, sib.t1_en[0], S.SIBLING), None,
                      "English name of sibling %s" % sid)
        for xid in ids:
            if xid != oid and xid not in sibs:
                other = lex.objects[xid]
                check(obj, "N-BB", other.name_bn,
                      other_label(obj, other.name_bn, S.OTHER_WRONG), None,
                      "Bangla name of object %s" % xid)
                break
    return n, fails


def self_test_vocab(lex, voc):
    """Every form must score correct for its own class and wrong for another
    class of the same kind. Returns (number of checks, failures)."""
    n = 0
    fails = []
    for kind in S.VOCAB_KINDS:
        by_class = {}
        for key, senses in voc.indexes[kind].entries.items():
            for s in senses:
                by_class.setdefault(s.class_id, []).append((key, s.lang))
        qa_type = QA_TYPE_OF_KIND[kind]
        for cid in sorted(by_class):
            golds = [k for k, lang in by_class[cid] if lang == "bn"]
            if not golds:
                fails.append({"check": "vocabulary self-test",
                              "id": "%s %s" % (kind, cid),
                              "message": "%s %s: no Bangla form, so no gold"
                                         " can map to this class" % (kind, cid),
                              "detail": ""})
                continue
            it = fake_item(qa_type, golds[0])
            for i, (key, lang) in enumerate(by_class[cid]):
                n += 1
                r = S.score_item(it, "L-BB" if lang == "bn" else "L-BE", key,
                                 lex, voc)
                if not r["scoreable"] or r["strict"] != 1:
                    fails.append({"check": "vocabulary self-test",
                                  "id": "%s %s" % (kind, cid),
                                  "message": "%s %s: own %s form %d gave %s (%s)"
                                             % (kind, cid, lang, i, r["label"],
                                                r["gold_status"]),
                                  "detail": "form %r" % key})
            for other in sorted(by_class):
                if other == cid:
                    continue
                key = by_class[other][0][0]
                n += 1
                r = S.score_item(it, "L-BB", key, lex, voc)
                if r["scoreable"] and r["lenient"] != 0:
                    fails.append({"check": "vocabulary self-test",
                                  "id": "%s %s" % (kind, cid),
                                  "message": "%s %s: a form of class %s scored"
                                             " correct" % (kind, cid, other),
                                  "detail": "form %r" % key})
                break
    return n, fails


# --------------------------------------------------------------------------
# audit, part 2: real gold answers
# --------------------------------------------------------------------------

def robustness(items):
    """Every answer and question must pass through the scorer's text steps."""
    fails = []
    for it in items:
        for fld in ("answer_bn", "question_bn"):
            if fld in it:
                try:
                    S.prepare_answer(it[fld])
                    S.romanise_bn(it[fld])
                except Exception as exc:               # report, never hide
                    fails.append({"check": "text steps", "id": it["qa_id"],
                                  "message": "%s %s: %s" % (
                                      it["qa_id"], fld, type(exc).__name__),
                                  "detail": repr(exc)})
    return fails


def identity_check(items, settings):
    """Naming golds as their own lexicon: each distinct gold name becomes one
    made-up object, in memory only. Scoring each gold as an answer must give
    Tier 1. This tests the matcher on every real gold string; it says nothing
    about the real lexicon."""
    naming = [it for it in items if it["qa_type"] in S.NAMING_TYPES]
    probe_index = S.TermIndex(settings)
    groups = OrderedDict()
    for it in naming:
        gold = S.strip_lead_in(normalise(it["answer_bn"]))
        key = probe_index.key_of(gold)
        if key:
            groups.setdefault(key, [])
            if gold not in groups[key]:
                groups[key].append(gold)
    rows = []
    for i, (key, golds) in enumerate(groups.items()):
        row = dict.fromkeys(S.LEXICON_COLUMNS, "")
        row.update(name_id="ID%04d" % i, name_bn=golds[0])
        rows.append(row)
    lex = S.build_lexicon(rows, settings, "identity")
    voc = S.build_closed_vocab([], settings, "identity")
    variants = [("plain", lambda g: g),
                ("with lead-in and dari", lambda g: "এটি একটি " + g + "।")]
    if settings.suffix_tolerance:
        variants.append(("with a classifier ending", lambda g: g + "টি"))
    result = OrderedDict((name, 0) for name, _ in variants)
    fails = []
    for it in naming:
        gold = normalise(it["answer_bn"])
        for name, make in variants:
            r = S.score_item(it, "N-BB", make(gold), lex, voc)
            if r["scoreable"] and r["label"] == S.TIER1 and r["strict"] == 1:
                result[name] += 1
            else:
                fails.append({"check": "identity", "id": it["qa_id"],
                              "message": "%s [%s] -> %s"
                                         % (it["qa_id"], name, r["label"]),
                              "detail": "gold %r matched %s"
                                        % (gold, r["matched"])})
    merged = [g for g in groups.values() if len(g) > 1]
    return {"items": len(naming),
            "distinct": sum(len(g) for g in groups.values()),
            "names": len(groups), "merged": len(merged), "passed": result,
            "fails": fails}


def audit_golds(items, lex, voc, excluded):
    """Resolve every gold that is not excluded, then score each resolved gold
    as if it were a model answer: it must be strict-correct. Excluded items
    are counted under their own status and enter no check."""
    coverage = OrderedDict()
    stats = OrderedDict()
    self_fails = []
    langs = Counter()
    n_self = 0
    for it in items:
        qa_type = it["qa_type"]
        st = stats.setdefault(qa_type, Counter())
        st["items"] += 1
        if it["qa_id"] in excluded:
            st["excluded"] += 1
            key = (qa_type, normalise(it["answer_bn"]), "excluded")
            row = coverage.setdefault(key, {
                "qa_type": qa_type, "kind": "", "gold": key[1],
                "n_items": 0, "status": S.EXCLUDED_STATUS, "mapped_to": "",
                "hint": excluded[it["qa_id"]], "extra_words": "",
                "example_qa_id": it["qa_id"]})
            row["n_items"] += 1
            continue
        if qa_type not in S.CLOSED_TYPES:
            st["not a closed type"] += 1
            continue
        langs[S.prepare_answer(it["answer_bn"]).lang["lang"]] += 1
        gold = S.resolve_gold(it, lex, voc)
        st["scoreable" if gold.status == "ok" else gold.status] += 1
        mapped = ""
        if gold.status == "ok":
            mapped = gold.name_id or (
                str(gold.number) if gold.kind == "count"
                else "|".join(sorted(gold.classes)))
            n_self += 1
            r = S.score_item(it, "L-BB", it["answer_bn"], lex, voc)
            if r["strict"] != 1:
                self_fails.append({
                    "check": "gold as its own answer", "id": it["qa_id"],
                    "message": "%s -> %s" % (it["qa_id"], r["label"]),
                    "detail": "gold %r found %s" % (
                        gold.norm, r["matched"] or r["found_value"])})
        key = (qa_type, gold.norm, gold.status)
        row = coverage.setdefault(key, {
            "qa_type": qa_type, "kind": gold.kind, "gold": gold.norm,
            "n_items": 0, "status": gold.status, "mapped_to": mapped,
            "hint": gold.hint, "extra_words": " ".join(gold.residual),
            "example_qa_id": it["qa_id"]})
        row["n_items"] += 1
    return {"coverage": list(coverage.values()), "stats": stats,
            "self_n": n_self, "self_fails": self_fails, "langs": langs}


def a2_check(items, lex, voc, excluded, probe_name):
    """Addendum A2 (B5 task 2): on every item that is not excluded and carries
    closed_kind, the scorer's kind, count and room class must equal the
    ASSET-5C fields. Returns (rows for the CSV, counts)."""
    rows = []
    counts = Counter()
    for it in items:
        if it["qa_id"] in excluded:
            counts["excluded"] += 1
            continue
        gold = S.resolve_gold(it, lex, voc)
        bad = S.check_5c_fields(it, gold, voc)
        if bad is None:
            counts["no 5C fields"] += 1
            continue
        status = "agree" if not bad else "disagree"
        counts[status] += 1
        if gold.status == "ok":
            value = gold.name_id or (
                str(gold.number) if gold.kind == "count"
                else "|".join(sorted(gold.classes)))
        else:
            value = ""
        rows.append({"probe": probe_name, "qa_id": it["qa_id"],
                     "qa_type": it["qa_type"],
                     "closed_kind": it.get("closed_kind"),
                     "gold_count": it.get("gold_count"),
                     "scorer_kind": gold.kind, "scorer_value": value,
                     "status": status, "detail": "; ".join(bad)})
    return rows, counts


def extra_word_rows(all_items, lex, voc, excluded):
    """B5 task 3: every distinct scoreable closed gold that holds words
    outside every listed term, with the label that an answer made of one
    such word alone receives (L-BB). Distinct over all probe files."""
    rows = OrderedDict()
    for probe_name, items in all_items:
        for it in items:
            if it["qa_id"] in excluded or it["qa_type"] not in S.CLOSED_TYPES:
                continue
            gold = S.resolve_gold(it, lex, voc)
            if gold.status != "ok" or not gold.residual:
                continue
            key = (it["qa_type"], gold.norm)
            row = rows.get(key)
            if row is None:
                alone = []
                for w in gold.residual:
                    r = S.score_item(it, "L-BB", w, lex, voc)
                    alone.append((w, r["label"], r["refusal"]))
                row = rows[key] = {
                    "qa_type": it["qa_type"], "kind": gold.kind,
                    "gold": gold.norm,
                    "classes": "|".join(sorted(gold.classes))
                    if gold.kind != "count" else str(gold.number),
                    "extra_words": " ".join(gold.residual),
                    "label_if_alone": "; ".join(
                        "%s=%s%s" % (w, lab, " (refusal)" if ref else "")
                        for w, lab, ref in alone),
                    "alone_scores_right": any(
                        lab in (S.CORRECT, S.TIER1) for _w, lab, _r in alone),
                    "example_qa_id": it["qa_id"], "probes": Counter(),
                    "qa_ids": set()}
            row["probes"][probe_name] += 1
            row["qa_ids"].add(it["qa_id"])
    out = []
    for row in rows.values():
        row["n_items"] = len(row["qa_ids"])
        row["items_by_probe"] = " ".join(
            "%s:%d" % (p, n) for p, n in sorted(row["probes"].items()))
        out.append(row)
    return out


def parse_expect(text):
    """--expect: JSON {probe file name: {"total": n, qa_type: n, ...}}."""
    try:
        data = json.loads(text)
    except ValueError as exc:
        sys.exit("STOP: --expect is not JSON: %s" % exc)
    if not isinstance(data, dict):
        sys.exit("STOP: --expect must be a JSON object")
    return data


def cmd_audit(args):
    out_dir = refuse_inside_repo(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = load_settings(args)
    say = Report()
    failed = False
    all_fails = []

    say("ASSET-9 audit, scorer %s, script %s, run %s"
        % (S.SCORER_VERSION, SCRIPT_VERSION, utc_now()))
    say("Matching rules: %s" % settings.describe())
    try:
        lex = S.load_lexicon(args.lexicon, settings)
        voc = S.load_closed_vocab(args.vocab, settings)
    except S.LexiconError as exc:
        say("\nFILE ERROR (see the message below; it names rows, not items)")
        print(exc)
        say("\nRESULT: FAIL. Fix the file and run again.")
        say.write(out_dir / "audit_summary.txt")
        return 1
    excluded = load_excluded_or_stop(args.excluded)
    inputs = OrderedDict()
    for name, path in (("lexicon", args.lexicon), ("vocabulary", args.vocab),
                       ("excluded list", args.excluded)):
        inputs[name] = {"path": str(path), "sha256": sha256_of(path)}
    say("Lexicon      : %s, sha256 %s -> %s"
        % (Path(args.lexicon).name, inputs["lexicon"]["sha256"][:12],
           lex.describe()))
    say("Vocabulary   : %s, sha256 %s -> %s"
        % (Path(args.vocab).name, inputs["vocabulary"]["sha256"][:12],
           voc.describe()))
    say("Excluded list: %s, sha256 %s -> %d rows (%s)"
        % (Path(args.excluded).name, inputs["excluded list"]["sha256"][:12],
           len(excluded), ", ".join("%s %d" % kv for kv in
                                    sorted(Counter(excluded.values()).items()))))
    for name, obj in (("Lexicon", lex), ("Vocabulary", voc)):
        say("%s warnings: %d" % (name, len(obj.warnings)))
        if obj.warnings:
            say("    (warnings name rows of the frozen file; see"
                " audit_failures.csv)")
            for w in obj.warnings:
                all_fails.append({"check": "%s warning" % name, "id": "",
                                  "message": "warning", "detail": w})

    def report(title, n, fails):
        nonlocal failed
        say("%s: %d checks, %d failures" % (title, n, len(fails)))
        for f in fails[:args.max_list]:
            say("    - " + f["message"])
        if len(fails) > args.max_list:
            say("    ... %d more (audit_failures.csv)"
                % (len(fails) - args.max_list))
        all_fails.extend(fails)
        failed = failed or bool(fails)

    say()
    report("[1] Lexicon self-test", *self_test_lexicon(lex, voc))
    report("[2] Vocabulary self-test", *self_test_vocab(lex, voc))

    summary = OrderedDict()
    all_rows = []
    all_a2 = []
    all_items = []
    for probe_path in args.probe:
        items = S.load_probe(probe_path)
        name = Path(probe_path).name
        all_items.append((name, items))
        n_ex = sum(1 for it in items if it["qa_id"] in excluded)
        say()
        say("Probe: %s, sha256 %s, %d items, %d excluded, %d checked"
            % (name, sha256_of(probe_path)[:12], len(items), n_ex,
               len(items) - n_ex))
        live = [it for it in items if it["qa_id"] not in excluded]

        fails = robustness(items)
        report("[3] Text steps on every answer and question (all %d items)"
               % len(items), len(items), fails)

        ident = identity_check(live, settings)
        say("[4] Identity check on naming golds not excluded: %d items, %d"
            " distinct gold strings, %d names" % (ident["items"],
                                                 ident["distinct"],
                                                 ident["names"]))
        for vname, passed in ident["passed"].items():
            say("    %-26s %d / %d" % (vname, passed, ident["items"]))
        if ident["merged"]:
            say("    Spellings the spacing rule treats as one name: %d"
                " group(s) (information)" % ident["merged"])
        for f in ident["fails"][:args.max_list]:
            say("    - FAIL " + f["message"])
        all_fails.extend(ident["fails"])
        failed = failed or bool(ident["fails"])

        res = audit_golds(items, lex, voc, excluded)
        say("[5] Gold answers against the supplied lexicon and vocabulary")
        table = []
        not_scoreable = 0
        by_type = OrderedDict()
        for qa_type, st in res["stats"].items():
            row = {"qa_type": qa_type, "items": st["items"],
                   "excluded": st["excluded"], "scored": st["scoreable"],
                   "unmapped": st["gold_unmapped"],
                   "ambiguous": st["gold_ambiguous"],
                   "not closed": st["not a closed type"]}
            table.append(row)
            by_type[qa_type] = row
            not_scoreable += st["gold_unmapped"] + st["gold_ambiguous"]
        table.append({"qa_type": "total",
                      **{c: sum(r[c] for r in table)
                         for c in ("items", "excluded", "scored", "unmapped",
                                   "ambiguous", "not closed")}})
        print_table(table, ["qa_type", "items", "excluded", "scored",
                            "unmapped", "ambiguous", "not closed"], say)
        say("    Gold language, closed types not excluded: %s"
            % ", ".join("%s %d" % kv for kv in sorted(res["langs"].items())))
        n_extra = sum(1 for r in res["coverage"]
                      if r["status"] == "ok" and r["extra_words"])
        say("    Scoreable golds with words outside every listed term: %d"
            " distinct (see [8])" % n_extra)
        report("[6] Each scoreable gold scored as its own answer",
               res["self_n"], res["self_fails"])
        if args.require_full and not_scoreable:
            say("    --require-full: %d closed-type item(s) not excluded are"
                " not scoreable" % not_scoreable)
            failed = True

        a2_rows, a2 = a2_check(items, lex, voc, excluded, name)
        say("[7] ASSET-5C fields against the scorer (addendum A2): %d agree,"
            " %d disagree, %d excluded, %d without 5C fields"
            % (a2["agree"], a2["disagree"], a2["excluded"],
               a2["no 5C fields"]))
        bad = [r for r in a2_rows if r["status"] != "agree"]
        for r in bad[:args.max_list]:
            say("    - %s: %s" % (r["qa_id"], r["detail"]))
        if bad:
            failed = True
            all_fails.extend({"check": "A2", "id": r["qa_id"],
                              "message": r["detail"], "detail": ""}
                             for r in bad)
        all_a2.extend(a2_rows)

        for r in res["coverage"]:
            r["probe"] = name
        all_rows.extend(res["coverage"])
        summary[name] = OrderedDict(
            sha256=sha256_of(probe_path), run=len(items), excluded=n_ex,
            scored=sum(r["scored"] for r in by_type.values()),
            by_type=by_type, a2=dict(a2),
            identity={k: v for k, v in ident["passed"].items()},
            identity_items=ident["items"])

    extra = extra_word_rows(all_items, lex, voc, excluded)
    kinds = Counter(r["kind"] for r in extra)
    right_alone = [r for r in extra if r["alone_scores_right"]]
    say()
    say("[8] Closed golds with extra words (B5 task 3), distinct over all"
        " probes: %d (%s) in %d distinct items; an answer made of one extra"
        " word alone scores right in %d of them"
        % (len(extra), ", ".join("%s %d" % kv for kv in sorted(kinds.items())),
           len(set().union(*(r["qa_ids"] for r in extra))) if extra else 0,
           len(right_alone)))
    say("    Table with text: extra_words.csv")

    pairs = S.parallel_pairs(lex, voc)
    p_count = Counter((p[0], p[4]) for p in pairs)
    gaps = S.genitive_only_forms(voc)
    say("[9] Frozen-file notes for D117 (information, never a failure):")
    say("    Same word in two scripts, both accepted: lexicon %d pairs"
        " (acronyms %d), vocabulary %d pairs"
        % (p_count[("lexicon", "romanised")] + p_count[("lexicon", "acronym")],
           p_count[("lexicon", "acronym")],
           sum(v for (src, _h), v in p_count.items() if src != "lexicon")))
    say("    Bangla forms listed only with a genitive ending (bare form not"
        " accepted, English form accepted): %d (%s)"
        % (len(gaps), ", ".join("%s %s" % (k, c) for k, c, _f, _b in gaps)
           or "none"))
    say("    Details: frozen_file_notes.csv")

    expect_failed = False
    if args.expect:
        want = parse_expect(args.expect)
        say("[10] Scored sizes against --expect")
        lines = []
        for pname, exp in want.items():
            got = summary.get(pname)
            if got is None:
                lines.append("    FAIL %s: probe not audited" % pname)
                expect_failed = True
                continue
            for k, v in exp.items():
                have = got["scored"] if k == "total" else \
                    got["by_type"].get(k, {}).get("scored", 0)
                ok = have == v
                expect_failed = expect_failed or not ok
                lines.append("    %s %s %s: scored %d, expected %d"
                             % ("PASS" if ok else "FAIL", pname, k, have, v))
        for line in lines:
            say(line)
        failed = failed or expect_failed

    # ---- files (Drive) -----------------------------------------------------
    write_csv(out_dir / "gold_coverage.csv", all_rows,
              ["probe", "qa_type", "kind", "gold", "n_items", "status",
               "mapped_to", "hint", "extra_words", "example_qa_id"])
    write_csv(out_dir / "a2_check.csv", all_a2,
              ["probe", "qa_id", "qa_type", "closed_kind", "gold_count",
               "scorer_kind", "scorer_value", "status", "detail"])
    write_csv(out_dir / "extra_words.csv", extra,
              ["kind", "qa_type", "gold", "classes", "extra_words",
               "label_if_alone", "alone_scores_right", "n_items",
               "items_by_probe", "example_qa_id"])
    notes = [{"note": "parallel pair", "source": p[0], "owner": p[1],
              "bn": p[2], "en": p[3], "how": p[4]} for p in pairs]
    notes += [{"note": "genitive-only form", "source": k, "owner": c,
               "bn": f, "en": "", "how": "bare form not listed: %s" % b}
              for k, c, f, b in gaps]
    write_csv(out_dir / "frozen_file_notes.csv", notes,
              ["note", "source", "owner", "bn", "en", "how"])
    write_csv(out_dir / "audit_failures.csv", all_fails,
              ["check", "id", "message", "detail"])
    result = "FAIL" if failed else "PASS"
    js = OrderedDict(
        command="audit", script_version=SCRIPT_VERSION,
        scorer_version=S.SCORER_VERSION, run_utc=utc_now(),
        settings=settings.describe(), require_full=args.require_full,
        inputs=inputs, probes=summary,
        extra_words={"rows": len(extra), "by_kind": dict(kinds),
                     "alone_scores_right": len(right_alone)},
        frozen_file_notes={"pairs": {"%s/%s" % k: v
                                     for k, v in p_count.items()},
                           "genitive_only": [[k, c] for k, c, _f, _b in gaps]},
        expect_failed=expect_failed, result=result)
    (out_dir / "audit_summary.json").write_text(
        json.dumps(js, ensure_ascii=False, indent=1), encoding="utf-8")
    say()
    say("Wrote in %s: gold_coverage.csv (%d rows), a2_check.csv (%d rows),"
        " extra_words.csv (%d rows), frozen_file_notes.csv (%d rows),"
        " audit_failures.csv (%d rows), audit_summary.json"
        % (out_dir, len(all_rows), len(all_a2), len(extra), len(notes),
           len(all_fails)))
    say("RESULT: %s" % result)
    if not failed and not args.require_full:
        say("(Without --require-full unmapped golds are allowed. At Gate A,"
            " run with the frozen files and --require-full.)")
    say.write(out_dir / "audit_summary.txt")
    print("Wrote %s" % (out_dir / "audit_summary.txt"))
    return 1 if failed else 0


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------

def read_jsonl(path):
    """Lines of a JSONL file (a harness cache.jsonl). An unreadable last line
    without a line break is a line still being written by another session
    (the pilot reading may run Cell 8 at the same time): it is skipped and
    counted. Any other unreadable line stops the run."""
    recs = []
    incomplete = 0
    with open(path, "rb") as fh:
        raw = fh.read()
    text = raw.decode("utf-8")
    lines = text.split("\n")
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            recs.append((n, json.loads(line)))
        except ValueError as exc:
            if n == len(lines) and not text.endswith("\n"):
                incomplete += 1
                continue
            sys.exit("STOP: %s line %d is not JSON (%s)." % (path, n, exc))
    return recs, incomplete, hashlib.sha256(raw).hexdigest()


def read_outputs(path):
    """Records of one outputs file: .jsonl lines, or a .json list (stage 1
    format, kept for old files)."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        data = data["items"]
    if not isinstance(data, list):
        sys.exit("%s: expected JSON Lines, a JSON list, or {'items': [...]}"
                 % path)
    return list(enumerate(data, 1)), 0, hashlib.sha256(raw).hexdigest()


def load_options(path):
    """The option file through src/fc_options.py only. Refuses the draft."""
    if "draft" in Path(path).name.lower():
        sys.exit("STOP: %s is the DRAFT option file (before HT-6). Scoring uses"
                 " Drive probes/fc_options.jsonl only (D196)." % path)
    try:
        table = FC.read_option_sets(path)
    except (FC.OptionError, OSError) as exc:
        sys.exit("STOP: the option file cannot be used: %s" % exc)
    return table, sha256_of(path)


def condition_checks(rec, cond, item):
    """Fields of a harness record that must agree with the scorer's table of
    conditions (scoring.CONDITIONS) and with the probe item. Returns the
    names of the fields that disagree; absent fields are not checked."""
    bad = []
    if "qa_type" in rec and rec["qa_type"] != item["qa_type"]:
        bad.append("qa_type")
    if "question_lang" in rec and rec["question_lang"] != cond.question_lang:
        bad.append("question_lang")
    if cond.task == "open" and "answer_lang" in rec \
            and rec["answer_lang"] != cond.answer_lang:
        bad.append("answer_lang")
    if "has_image" in rec and bool(rec["has_image"]) != cond.image:
        bad.append("has_image")
    if "forced_choice" in rec and bool(rec["forced_choice"]) != (cond.task == "fc"):
        bad.append("forced_choice")
    return bad


def run_score(args):
    """The score command. Returns 0, or exits with a STOP message."""
    out_path = refuse_inside_repo(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    settings = load_settings(args)
    print("ASSET-9 score, scorer %s, script %s, run %s"
          % (S.SCORER_VERSION, SCRIPT_VERSION, utc_now()))
    print("Matching rules: %s" % settings.describe())
    lex = S.load_lexicon(args.lexicon, settings)
    voc = S.load_closed_vocab(args.vocab, settings)
    excluded = load_excluded_or_stop(args.excluded)
    inputs = OrderedDict(
        lexicon={"path": str(args.lexicon), "sha256": sha256_of(args.lexicon)},
        vocabulary={"path": str(args.vocab), "sha256": sha256_of(args.vocab)},
        excluded={"path": str(args.excluded),
                  "sha256": sha256_of(args.excluded), "rows": len(excluded)})
    print("Lexicon      : sha256 %s -> %s"
          % (inputs["lexicon"]["sha256"][:12], lex.describe()))
    print("Vocabulary   : sha256 %s -> %s"
          % (inputs["vocabulary"]["sha256"][:12], voc.describe()))
    print("Excluded list: sha256 %s -> %d rows"
          % (inputs["excluded"]["sha256"][:12], len(excluded)))
    items, probe_info = load_probes(args.probe)
    inputs["probes"] = probe_info
    print("Probe items  : %d from %s" % (len(items), ", ".join(
        "%s (%d, sha256 %s)" % (p["file"], p["items"], p["sha256"][:12])
        for p in probe_info)))
    table, opt_sha = None, None
    if args.options_file:
        table, opt_sha = load_options(args.options_file)
        inputs["options"] = {"path": str(args.options_file), "sha256": opt_sha,
                             "sets": len(table)}
        print("Option file  : %s, sha256 %s, %d sets (%d bn, %d en)"
              % (Path(args.options_file).name, opt_sha[:12], len(table),
                 sum(1 for k in table if k[1] == "bn"),
                 sum(1 for k in table if k[1] == "en")))

    recs = []
    inputs["outputs"] = []
    for p in args.outputs:
        got, incomplete, sha = read_outputs(p)
        inputs["outputs"].append({"path": str(p), "sha256": sha,
                                  "records": len(got),
                                  "incomplete_last_line": incomplete})
        print("Outputs      : %s, %d records%s"
              % (p, len(got), ", 1 incomplete last line skipped (being"
                 " written; run again later)" if incomplete else ""))
        recs.extend(got)
    if not recs:
        sys.exit("STOP: no records to score.")
    keep = list(dict.fromkeys(list(DEFAULT_KEEP) + list(args.keep or [])))
    for k in args.keep or []:
        if not any(k in r for _n, r in recs):
            sys.exit("STOP: --keep %s is in no output record." % k)
    need = [args.qa_id_field, args.condition_field, args.answer_field]
    for fld in need:
        missing = sum(1 for _n, r in recs if fld not in r)
        if missing:
            sys.exit("STOP: %d record(s) have no field %r. Fields of the first"
                     " record: %s" % (missing, fld, ", ".join(sorted(recs[0][1]))))

    scored = []
    skipped = Counter()
    seen = {}
    disagree = []
    no_set = []
    dup_differs = 0
    for _n, rec in recs:
        qa_id = rec[args.qa_id_field]
        cid = rec[args.condition_field]
        try:
            cond = S.condition_spec(cid)
        except KeyError as exc:
            sys.exit("STOP: %s" % exc)
        if qa_id not in items:
            skipped["qa_id not in the probe file(s)"] += 1
            continue
        key = rec.get("key") or "|".join(str(rec.get(k)) for k in (
            "model_id", args.condition_field, "prompt_version",
            args.qa_id_field))
        if key in seen:
            skipped["later record with the same cache key (the first stands,"
                    " D143)"] += 1
            if seen[key] != rec.get(args.answer_field):
                dup_differs += 1
            continue
        seen[key] = rec.get(args.answer_field)
        item = items[qa_id]
        bad = condition_checks(rec, cond, item)
        if bad:
            disagree.append((qa_id, cid, ",".join(bad)))
            continue
        reason = excluded.get(qa_id)
        options, ci, sha = None, None, None
        if cond.task == "fc":
            sha = opt_sha
            if not reason:
                if table is None:
                    sys.exit("STOP: forced-choice records found (%s). Pass"
                             " --options-file with Drive probes/"
                             "fc_options.jsonl." % cid)
                try:
                    options, ci = FC.options_for(table, qa_id,
                                                 cond.question_lang)
                except FC.OptionError as exc:
                    no_set.append((qa_id, cid, str(exc)))
                    continue
        out = S.score_item(item, cid, rec.get(args.answer_field), lex, voc,
                           options=options, correct_index=ci,
                           excluded=reason)
        for k in keep:
            out[k] = rec.get(k)
        out["fc_options_sha256"] = sha
        scored.append(out)

    if disagree:
        print("\nSTOP: %d record(s) disagree with the scorer's condition table"
              " or the probe item; nothing was written. First ones (qa_id,"
              " condition, fields):" % len(disagree))
        for d in disagree[:10]:
            print("    %s %s %s" % d)
        sys.exit(1)
    if no_set:
        print("\nSTOP: %d forced-choice record(s) of items that are not"
              " excluded have no option set in the option file; nothing was"
              " written. First ones:" % len(no_set))
        for d in no_set[:10]:
            print("    %s %s" % d[:2])
        sys.exit(1)

    with open(out_path, "w", encoding="utf-8") as fh:
        for r in scored:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    statuses = Counter(r["gold_status"] for r in scored)
    print("\nScored file : %d records (%s)"
          % (len(scored), ", ".join("%s %d" % kv
                                     for kv in sorted(statuses.items()))))
    for why, n in skipped.items():
        print("Skipped     : %d, %s" % (n, why))
    if dup_differs:
        print("Note        : %d later record(s) with a known key hold another"
              " answer; the first record was scored" % dup_differs)

    by = ("model_id", "condition_id")
    rows = S.summarise(scored, by)
    if args.counts_only:
        print("\nCounts per model and condition (rates are left to the chat"
              " that reads them):")
        print_table(rows, list(by) + ["n", "excluded", "scoreable"])
    else:
        print("\nSummary (percent). For eyeballing only; statistics come"
              " later.")
        print_table(rows, list(by) + ["n", "excluded", "scoreable", "strict",
                                      "lenient", "fidelity", "refusal",
                                      "empty", "le8_words", "mixed_script",
                                      "invalid", "no_content"])
        labels = Counter((r["kind"], r["label"]) for r in scored
                         if r["gold_status"] != S.EXCLUDED_STATUS)
        print("\nLabels (excluded records left out):")
        for (kind, label), n in sorted(labels.items(),
                                       key=lambda kv: str(kv[0])):
            print("    %-14s %-13s %d" % (kind, label, n))

    sample_path = None
    if args.sample:
        pick = sorted((r for r in scored
                       if r["gold_status"] != S.EXCLUDED_STATUS),
                      key=lambda r: seeded_rank("%s|%s|%s" % (
                          r["qa_id"], r["condition_id"], r.get("model_id"))))
        pick = pick[:args.sample]
        sample_path = out_path.with_name(out_path.stem + "_sample.csv")
        write_csv(sample_path, pick,
                  ["qa_id", "model_id", "condition_id", "qa_type", "kind",
                   "gold_norm", "answer_raw", "label", "strict", "lenient",
                   "fidelity", "lang", "refusal", "content_found", "matched",
                   "named_ids", "gold_value", "found_value", "fc_choice",
                   "fc_correct_index", "fc_method"])
        print("\n%d seeded-random scored records (seed %d), ids only here;"
              " text in %s:" % (len(pick), SEED, sample_path.name))
        for r in pick:
            print("    %s %s %s" % (r["qa_id"], r["condition_id"],
                                    r.get("model_id")))

    manifest = OrderedDict(
        command="score", script_version=SCRIPT_VERSION,
        scorer_version=S.SCORER_VERSION, run_utc=utc_now(),
        settings=settings.describe(), answer_field=args.answer_field,
        inputs=inputs, records=len(scored), statuses=dict(statuses),
        skipped=dict(skipped), later_records_differ=dup_differs,
        keep=keep, out=str(out_path),
        sample=str(sample_path) if sample_path else None)
    man_path = out_path.with_name(out_path.stem + "_manifest.json")
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print("\nWrote %s" % out_path)
    print("Wrote %s" % man_path)
    return 0


def cmd_score(args):
    return run_score(args)


# --------------------------------------------------------------------------
# dryrun-fc: the forced-choice wiring on the real option file
# --------------------------------------------------------------------------

def cmd_dryrun_fc(args):
    """Made-up answers for every option set whose item is in the probe file:
    the right label, a wrong label, the text of the right option, the right
    label in the other script, and an empty answer, each under the image and
    the no-image condition. They go through the score command unchanged, and
    every scored record is checked against what Design 8.5 and D107 say."""
    out_dir = refuse_inside_repo(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("ASSET-9 dryrun-fc, scorer %s, script %s, run %s"
          % (S.SCORER_VERSION, SCRIPT_VERSION, utc_now()))
    table, opt_sha = load_options(args.options_file)
    items, _info = load_probes(args.probe)
    excluded = load_excluded_or_stop(args.excluded)
    conds = {"bn": ("N-FCB", "N-FCB0"), "en": ("N-FCE", "N-FCE0")}
    labels = {"bn": FC.BN_LABELS, "en": FC.EN_LABELS}
    expect = {}
    lines = []
    missing = 0
    for (qa_id, lang), rec in sorted(table.items()):
        if qa_id not in items:
            missing += 1
            continue
        ci = rec["correct_index"]
        other = "en" if lang == "bn" else "bn"
        variants = (
            ("label_right", labels[lang][ci], S.CORRECT, ci, "label"),
            ("label_wrong", labels[lang][(ci + 1) % 4], S.WRONG,
             (ci + 1) % 4, "label"),
            ("text_right", rec["options"][ci], S.CORRECT, ci, "text"),
            ("other_script", labels[other][ci], S.INVALID, None, None),
            ("empty", "", S.INVALID, None, None))
        for cid in conds[lang]:
            for vname, answer, label, choice, method in variants:
                key = "dryrun-%s|%s|exp-v1|%s" % (vname, cid, qa_id)
                lines.append({"key": key, "model_id": "dryrun-%s" % vname,
                              "condition_id": cid, "prompt_version": "exp-v1",
                              "qa_id": qa_id, "qa_type": items[qa_id]["qa_type"],
                              "question_lang": lang, "forced_choice": True,
                              "has_image": not cid.endswith("0"),
                              "stage": "dryrun", "answer_text": answer})
                expect[key] = (vname, label, choice, method,
                               excluded.get(qa_id))
    cache = out_dir / "dryrun_fc_cache.jsonl"
    with open(cache, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    print("Option sets: %d; with an item in the probe file: %d; made-up"
          " records: %d" % (len(table), len(table) - missing, len(lines)))
    if not lines:
        sys.exit("STOP: no option set has an item in the probe file(s).")

    scored_path = out_dir / "dryrun_fc_scored.jsonl"
    ns = argparse.Namespace(
        outputs=[str(cache)], probe=args.probe, lexicon=args.lexicon,
        vocab=args.vocab, excluded=args.excluded,
        options_file=args.options_file, out=str(scored_path),
        answer_field="answer_text", qa_id_field="qa_id",
        condition_field="condition_id", keep=[], counts_only=True,
        sample=0, no_spacing_rule=False, no_suffix_rule=False)
    print("\n---- the score command on the made-up records ----")
    run_score(ns)
    print("---- end of the score command ----\n")

    fails = Counter()
    first = {}
    checked = Counter()
    sha_bad = 0
    with open(scored_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            vname, label, choice, method, reason = expect[r["key"]]
            checked[vname] += 1
            if reason:
                ok = (r["gold_status"] == S.EXCLUDED_STATUS
                      and r["label"] is None and r["strict"] is None)
                tag = "excluded"
            else:
                ok = (r["label"] == label and r["fc_choice"] == choice
                      and r["fc_method"] == method
                      and r["fc_valid"] == (label != S.INVALID)
                      and r["strict"] == int(label == S.CORRECT)
                      and r["fidelity"] is None)
                if vname == "other_script":
                    ok = ok and r["fc_cross_label"] is True
                if vname == "empty":
                    ok = ok and r["refusal"] is True and r["is_empty"] is True
                tag = vname
            if r["fc_options_sha256"] != opt_sha:
                sha_bad += 1
            if not ok:
                fails[tag] += 1
                first.setdefault(tag, "%s %s" % (r["qa_id"],
                                                 r["condition_id"]))
    n_ex = sum(1 for k, v in expect.items() if v[4])
    print("Checked %d scored records (%s); %d of them belong to excluded"
          " items and must carry status excluded"
          % (sum(checked.values()), ", ".join("%s %d" % kv for kv in
                                              sorted(checked.items())), n_ex))

    # The draft file and a file whose records lack "options" are refused.
    draft_ok = False
    draft = out_dir / "fc_options_DRAFT_copy.jsonl"
    with open(args.options_file, encoding="utf-8") as src, \
            open(draft, "w", encoding="utf-8") as dst:
        for line in src:
            if line.strip():
                rec = json.loads(line)
                rec[FC.DRAFT_OPTIONS_KEY] = rec.pop("options")
                dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
    renamed = out_dir / "options_without_options_key.jsonl"
    renamed.write_bytes(draft.read_bytes())
    refused = 0
    for path in (draft, renamed):
        try:
            load_options(str(path))
        except SystemExit as exc:
            refused += 1
            print("Refused as it must be: %s (%s)"
                  % (path.name, str(exc).split("\n")[0][:90]))
    draft_ok = refused == 2
    draft.unlink()
    renamed.unlink()

    print()
    for tag, n in sorted(fails.items()):
        print("FAIL %s: %d record(s), first %s" % (tag, n, first[tag]))
    print("Option file sha256 %s in every scored record: %s"
          % (opt_sha[:12], "yes" if not sha_bad else
             "NO (%d records differ)" % sha_bad))
    ok = not fails and draft_ok and not sha_bad
    print("DRYRUN-FC: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# --------------------------------------------------------------------------
# proofread-sheet and proofread-read (addendum B5, F5)
# --------------------------------------------------------------------------

SHEET_TABS = ("instructions", "answers", "endings", "refusal_phrases")
ANSWER_COLS = ("code", "asked_in", "answer", "is_refusal", "refusal_words",
               "note")
ENDING_COLS = ("no", "ending", "marks", "keep", "note")
PHRASE_COLS = ("no", "language", "phrase", "correct_and_natural",
               "could_mean_something_else", "note")
N_ADD_ENDINGS = 6
N_ADD_PHRASES = 12

INSTRUCTIONS = [
    "ASSET-9: proofreading of the scorer's word lists (about 30 minutes)",
    "",
    "Who: one native Bangla speaker who also reads English. Work alone. Use no"
    " dictionary, no internet and no AI tool. Do not discuss the rows with"
    " anyone until you hand the file back.",
    "",
    "TAB answers. Each row is an answer that a computer model gave to a"
    " question about a photo of a home, when it was NOT shown the photo."
    " Column asked_in says whether the model was asked to answer in Bangla or"
    " in English. Fill is_refusal for every row:",
    "   yes     the answer gives no answer of its own: it declines, says it"
    " cannot see the photo or does not know, or asks for the photo.",
    "   no      the answer names something (an object, a number, a room, a"
    " material, a colour), even if it may be wrong or says 'maybe'.",
    "   unsure  you cannot decide. Say why in note.",
    "If yes or unsure: copy into refusal_words the exact words of the answer"
    " that make it a refusal, as they are written there (do not correct"
    " spelling). Do not judge whether an answer is right.",
    "",
    "TAB endings. The scorer cuts these endings off the last word of a Bangla"
    " answer before it looks the word up, so that a word with an ending counts"
    " as the word itself. Column marks says what the ending does. Fill keep:",
    "   yes  it is an ending that people write after the name of a household"
    " object.",
    "   no   it is wrong, or it can cut a real word into a different word."
    " Say why in note, with an example word.",
    "In the empty rows at the bottom you may add an ending that is missing,"
    " one per row, with an example word in note.",
    "",
    "TAB refusal_phrases. An answer that names nothing and holds one of these"
    " phrases counts as a refusal. Fill both columns for every row:",
    "   correct_and_natural        yes if this is how people or a chatbot"
    " write it; no if it is misspelt or unnatural (say the better form in"
    " note).",
    "   could_mean_something_else  yes if the phrase can also appear in an"
    " answer that is not a refusal (give an example in note).",
    "In the empty rows you may add a refusal phrase that is missing, for"
    " example one you saw in the answers tab, copied exactly, one per row;"
    " fill language (bn or en).",
    "",
    "Do not rename, move or delete tabs, columns or rows. Save the file and"
    " hand it back to the team lead.",
]


def _need_openpyxl():
    try:
        import openpyxl                                   # noqa: F401
    except ImportError:
        sys.exit("STOP: openpyxl is needed for the sheet (pip install"
                 " openpyxl). Colab has it.")
    import openpyxl
    return openpyxl


def collect_answers(paths, conditions, items, excluded):
    """Records of the given conditions: the first per cache key, items not
    excluded, qa_id in the probe files. Returns (records, counts)."""
    recs = []
    counts = Counter()
    seen = set()
    for p in paths:
        got, incomplete, _sha = read_outputs(p)
        counts["incomplete last lines skipped"] += incomplete
        for _n, r in got:
            if r.get("condition_id") not in conditions:
                continue
            key = r.get("key") or "|".join(str(r.get(k)) for k in (
                "model_id", "condition_id", "prompt_version", "qa_id"))
            if key in seen:
                counts["later record with the same key"] += 1
                continue
            seen.add(key)
            if r.get("qa_id") not in items:
                counts["qa_id not in the probe file"] += 1
                continue
            if r["qa_id"] in excluded:
                counts["excluded item"] += 1
                continue
            recs.append(r)
            counts["%s %s" % (r.get("model_key") or r.get("model_id"),
                              r["condition_id"])] += 1
    return recs, counts


def cmd_proofread_sheet(args):
    openpyxl = _need_openpyxl()
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation
    out_dir = refuse_inside_repo(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = out_dir / "ASSET9_proofread_sheet.xlsx"
    key_path = out_dir / "proofread_key.csv"
    if sheet.exists() and not args.overwrite:
        sys.exit("STOP: %s exists. A sheet that went out is never replaced;"
                 " use --overwrite only if it was not sent." % sheet)
    print("ASSET-9 proofread-sheet, scorer %s, script %s, run %s"
          % (S.SCORER_VERSION, SCRIPT_VERSION, utc_now()))
    items, info = load_probes(args.probe)
    excluded = load_excluded_or_stop(args.excluded)
    recs, counts = collect_answers(args.outputs, set(args.conditions), items,
                                   excluded)
    print("Answers read: %d (%s)" % (len(recs), ", ".join(
        "%s %d" % kv for kv in sorted(counts.items()))))
    groups = OrderedDict()
    for r in recs:
        asked = S.condition_spec(r["condition_id"]).answer_lang
        norm = normalise(r.get("answer_text"))
        g = groups.setdefault((asked, norm), {"raw": None, "records": []})
        if g["raw"] is None:
            g["raw"] = (r.get("answer_text") or "").strip()
        g["records"].append("%s:%s:%s" % (r.get("model_key") or
                                          r.get("model_id"),
                                          r["condition_id"], r["qa_id"]))
    order = sorted(groups, key=lambda k: seeded_rank(
        "proofread|%s|%s" % (k[0], k[1])))
    print("Distinct answers (after normalise, per requested language): %d"
          " (bn %d, en %d)" % (len(order),
                               sum(1 for k in order if k[0] == "bn"),
                               sum(1 for k in order if k[0] == "en")))
    if len(order) > args.max_rows:
        print("NOTE: more than %d rows; the sheet will take longer than"
              " 30 minutes." % args.max_rows)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = SHEET_TABS[0]
    for i, text in enumerate(INSTRUCTIONS, 1):
        ws.cell(row=i, column=1, value=text)
        ws.cell(row=i, column=1).alignment = Alignment(wrap_text=True)
    ws.column_dimensions["A"].width = 120
    ws.cell(row=1, column=1).font = Font(bold=True)

    def tab(name, cols, widths):
        t = wb.create_sheet(name)
        t.append(list(cols))
        for c in range(1, len(cols) + 1):
            t.cell(row=1, column=c).font = Font(bold=True)
        for letter, w in widths.items():
            t.column_dimensions[letter].width = w
        t.freeze_panes = "A2"
        return t

    ta = tab("answers", ANSWER_COLS, {"A": 8, "B": 9, "C": 70, "D": 11,
                                       "E": 40, "F": 40})
    key_rows = []
    for i, k in enumerate(order, 1):
        code = "P%03d" % i
        g = groups[k]
        ta.append([code, "Bangla" if k[0] == "bn" else "English", g["raw"],
                   "", "", ""])
        ta.cell(row=i + 1, column=3).alignment = Alignment(wrap_text=True)
        key_rows.append({"code": code, "asked_in": k[0], "answer_norm": k[1],
                         "n_records": len(g["records"]),
                         "records": " ".join(g["records"])})
    if order:
        dv = DataValidation(type="list", formula1='"yes,no,unsure"',
                            allow_blank=True)
        ta.add_data_validation(dv)
        dv.add("D2:D%d" % (len(order) + 1))
    else:
        print("NOTE: no answer of %s was found; the answers tab is empty."
              % ", ".join(args.conditions))

    te = tab("endings", ENDING_COLS, {"A": 8, "B": 10, "C": 12, "D": 8,
                                       "E": 60})
    ends = sorted(S.BN_SUFFIXES, key=lambda e: (S.BN_SUFFIX_ROLES[e], e))
    for i, e in enumerate(ends, 1):
        te.append([i, e, S.BN_SUFFIX_ROLES[e], "", ""])
    for j in range(1, N_ADD_ENDINGS + 1):
        te.append(["add-%d" % j, "", "", "", ""])
    dv2 = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
    te.add_data_validation(dv2)
    dv2.add("D2:D%d" % (len(ends) + 1))

    tp = tab("refusal_phrases", PHRASE_COLS, {"A": 8, "B": 10, "C": 30,
                                               "D": 20, "E": 26, "F": 60})
    phrases = [("en" if S.detect_language(S.tokenise(p))["lang"] == "en"
                else "bn", p) for p in S.REFUSAL_PHRASES]
    for i, (lang, p) in enumerate(phrases, 1):
        tp.append([i, lang, p, "", "", ""])
    for j in range(1, N_ADD_PHRASES + 1):
        tp.append(["add-%d" % j, "", "", "", "", ""])
    dv3 = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
    tp.add_data_validation(dv3)
    dv3.add("D2:E%d" % (len(phrases) + 1))
    wb.save(sheet)
    write_csv(key_path, key_rows, ["code", "asked_in", "answer_norm",
                                   "n_records", "records"])
    manifest = OrderedDict(
        command="proofread-sheet", script_version=SCRIPT_VERSION,
        scorer_version=S.SCORER_VERSION, run_utc=utc_now(),
        conditions=list(args.conditions),
        outputs=[{"path": str(p), "sha256": sha256_of(p)}
                 for p in args.outputs],
        probes=info, excluded={"path": str(args.excluded),
                               "sha256": sha256_of(args.excluded)},
        answers=len(recs), rows=len(order), endings=len(ends),
        phrases=len(phrases), sheet_sha256=sha256_of(sheet),
        order="sorted by SHA-256 of '42|ASSET-9|proofread|<lang>|<answer>'")
    (out_dir / "proofread_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Endings: %d. Refusal phrases: %d (en %d, bn %d)."
          % (len(ends), len(phrases), sum(1 for l, _p in phrases if l == "en"),
             sum(1 for l, _p in phrases if l == "bn")))
    print("Wrote %s (send this file only)" % sheet)
    print("Wrote %s (the key; never send it)" % key_path)
    print("Wrote %s" % (out_dir / "proofread_manifest.json"))
    return 0


def _read_tab(wb, name, cols):
    if name not in wb.sheetnames:
        sys.exit("STOP: the sheet has no tab %r." % name)
    rows = list(wb[name].iter_rows(values_only=True))
    head = [str(c or "").strip() for c in rows[0][:len(cols)]]
    if head != list(cols):
        sys.exit("STOP: tab %r has columns %s, expected %s."
                 % (name, head, list(cols)))
    out = []
    for r in rows[1:]:
        vals = ["" if v is None else str(v).strip() for v in r[:len(cols)]]
        vals += [""] * (len(cols) - len(vals))
        out.append(dict(zip(cols, vals)))
    return out


def cmd_proofread_read(args):
    openpyxl = _need_openpyxl()
    out_dir = refuse_inside_repo(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("ASSET-9 proofread-read, scorer %s, script %s, run %s"
          % (S.SCORER_VERSION, SCRIPT_VERSION, utc_now()))
    print("Filled sheet: %s, sha256 %s" % (Path(args.sheet).name,
                                          sha256_of(args.sheet)[:12]))
    wb = openpyxl.load_workbook(args.sheet, read_only=True)
    answers = _read_tab(wb, "answers", ANSWER_COLS)
    endings = _read_tab(wb, "endings", ENDING_COLS)
    phrases = _read_tab(wb, "refusal_phrases", PHRASE_COLS)
    with open(args.key, encoding="utf-8-sig") as fh:
        key = {r["code"]: r for r in csv.DictReader(fh)}

    problems = []
    for a in answers:
        if a["code"] not in key:
            problems.append("answers: unknown code %r" % a["code"])
        elif a["is_refusal"] not in ("yes", "no", "unsure"):
            problems.append("answers %s: is_refusal is %r" %
                            (a["code"], a["is_refusal"]))
    if len(answers) != len(key):
        problems.append("answers: %d rows, the key has %d"
                        % (len(answers), len(key)))
    for e in endings:
        if not e["no"].startswith("add") and e["keep"] not in ("yes", "no"):
            problems.append("endings %s: keep is %r" % (e["no"], e["keep"]))
    for p in phrases:
        if not p["no"].startswith("add") and (
                p["correct_and_natural"] not in ("yes", "no")
                or p["could_mean_something_else"] not in ("yes", "no")):
            problems.append("refusal_phrases %s: a yes/no cell is empty"
                            % p["no"])
    if problems:
        print("SHEET NOT COMPLETE: %d problem(s); first ones:" % len(problems))
        for p in problems[:15]:
            print("    - " + p)
        return 1

    lex = S.load_lexicon(args.lexicon)
    voc = S.load_closed_vocab(args.vocab)
    items, _info = load_probes(args.probe)
    excluded = load_excluded_or_stop(args.excluded)
    recs, _counts = collect_answers(args.outputs, set(args.conditions), items,
                                    excluded)
    by_key = {"%s:%s:%s" % (r.get("model_key") or r.get("model_id"),
                            r["condition_id"], r["qa_id"]): r for r in recs}
    mark = {a["code"]: a for a in answers}
    table = Counter()
    rows = []
    for code, k in key.items():
        for rid in k["records"].split():
            r = by_key.get(rid)
            if r is None:
                table["record not found again"] += 1
                continue
            out = S.score_item(items[r["qa_id"]], r["condition_id"],
                               r.get("answer_text"), lex, voc)
            person = mark[code]["is_refusal"]
            scorer = "yes" if out["refusal"] else "no"
            cell = "person %s, scorer %s" % (person, scorer)
            table[cell] += 1
            if person != scorer:
                rows.append({"code": code, "record": rid,
                             "person": person, "scorer": scorer,
                             "content_found": out["content_found"],
                             "label": out["label"],
                             "answer": r.get("answer_text"),
                             "refusal_words": mark[code]["refusal_words"],
                             "note": mark[code]["note"]})
    print("\nRefusal marks, per answer record (person against scorer %s):"
          % S.SCORER_VERSION)
    for cell, n in sorted(table.items()):
        print("    %-32s %d" % (cell, n))
    print("Disagreeing records: %d (codes: %s)"
          % (len(rows), " ".join(sorted({r["code"] for r in rows})) or "none"))

    changes = []
    for e in endings:
        if e["no"].startswith("add") and e["ending"]:
            changes.append({"list": "endings", "row": e["no"],
                            "what": "add", "text": e["ending"],
                            "note": e["note"]})
        elif not e["no"].startswith("add") and e["keep"] == "no":
            changes.append({"list": "endings", "row": e["no"],
                            "what": "remove or change", "text": e["ending"],
                            "note": e["note"]})
    for p in phrases:
        if p["no"].startswith("add") and p["phrase"]:
            changes.append({"list": "refusal_phrases", "row": p["no"],
                            "what": "add (%s)" % (p["language"] or "?"),
                            "text": p["phrase"], "note": p["note"]})
        elif not p["no"].startswith("add") and (
                p["correct_and_natural"] == "no"
                or p["could_mean_something_else"] == "yes"):
            changes.append({"list": "refusal_phrases", "row": p["no"],
                            "what": "check: natural %s, other meaning %s"
                                    % (p["correct_and_natural"],
                                       p["could_mean_something_else"]),
                            "text": p["phrase"], "note": p["note"]})
    # A proposed phrase that holds a listed term would turn content into a
    # refusal (or the reverse); D120 forbids the number case.
    for c in changes:
        if c["list"] == "refusal_phrases" and c["what"].startswith("add"):
            toks = S.tokenise(normalise(c["text"]))
            hits = [m for m in lex.index.find_all(toks)]
            for kind in S.VOCAB_KINDS:
                hits += voc.indexes[kind].find_all(toks)
            c["holds_listed_term"] = bool(hits)
    print("Proposed changes: endings %d, refusal phrases %d; added phrases"
          " that hold a listed term (D120 conflict): %d"
          % (sum(1 for c in changes if c["list"] == "endings"),
             sum(1 for c in changes if c["list"] == "refusal_phrases"),
             sum(1 for c in changes if c.get("holds_listed_term"))))
    write_csv(out_dir / "proofread_disagreements.csv", rows,
              ["code", "record", "person", "scorer", "content_found", "label",
               "answer", "refusal_words", "note"])
    write_csv(out_dir / "proofread_changes.csv", changes,
              ["list", "row", "what", "text", "holds_listed_term", "note"])
    print("Wrote %s" % (out_dir / "proofread_disagreements.csv"))
    print("Wrote %s" % (out_dir / "proofread_changes.csv"))
    print("A change to either list is a scorer change: new scorer version,"
          " all tests again, and a decision row saying it was made after"
          " reading pilot answers.")
    return 0


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="ASSET-9 stage 2: audit, score, forced-choice dry run,"
                    " proofreading sheet.")
    sub = ap.add_subparsers(dest="command", required=True)

    def files(p, probe_required=True):
        p.add_argument("--probe", action="append", required=probe_required,
                       help="probe file in the P-CORE.json record format;"
                            " repeat for several files")
        p.add_argument("--lexicon", default=str(REPO / "lexicon" /
                                                "lexicon.csv"))
        p.add_argument("--vocab", default=str(REPO / "lexicon" /
                                              "closed_vocab.csv"))
        p.add_argument("--excluded", default=str(DEFAULT_EXCLUDED),
                       help="probes/excluded_items.csv (D110)")

    def rules(p):
        p.add_argument("--no-spacing-rule", action="store_true",
                       help="switch the spacing rule off (sensitivity check)")
        p.add_argument("--no-suffix-rule", action="store_true",
                       help="switch the ending rule off (sensitivity check)")

    a = sub.add_parser("audit", help="check lexicon, vocabulary and golds")
    files(a)
    rules(a)
    a.add_argument("--out-dir", required=True, help="a folder in Drive")
    a.add_argument("--require-full", action="store_true",
                   help="fail unless every closed gold that is not excluded"
                        " is scoreable")
    a.add_argument("--expect", default="",
                   help='JSON {"P-CORE-CLOSED.json": {"total": 1109,'
                        ' "obj_name": 252, ...}, ...}: scored sizes to check')
    a.add_argument("--max-list", type=int, default=15,
                   help="how many findings to print per list (ids only)")
    a.set_defaults(func=cmd_audit)

    s = sub.add_parser("score", help="score harness outputs")
    files(s)
    rules(s)
    s.add_argument("--outputs", action="append", required=True,
                   help="a harness cache.jsonl (or a .json list); repeat for"
                        " several files")
    s.add_argument("--answer-field", default="answer_text",
                   help="field with the model's answer (ASSET-8: answer_text)")
    s.add_argument("--qa-id-field", default="qa_id")
    s.add_argument("--condition-field", default="condition_id")
    s.add_argument("--options-file", default="",
                   help="Drive probes/fc_options.jsonl; needed when forced-"
                        "choice records are present. The draft is refused.")
    s.add_argument("--keep", action="append", default=[],
                   help="another record field to copy into the scored file;"
                        " always kept: %s" % ", ".join(DEFAULT_KEEP))
    s.add_argument("--out", required=True, help="scored .jsonl, in Drive")
    s.add_argument("--sample", type=int, default=0,
                   help="write this many seeded-random scored records, with"
                        " text, to <out>_sample.csv (ids only on screen)")
    s.add_argument("--counts-only", action="store_true",
                   help="print counts only, no rates or labels")
    s.set_defaults(func=cmd_score)

    d = sub.add_parser("dryrun-fc", help="forced-choice wiring test")
    files(d)
    d.add_argument("--options-file", required=True,
                   help="Drive probes/fc_options.jsonl")
    d.add_argument("--out-dir", required=True, help="a folder in Drive")
    d.set_defaults(func=cmd_dryrun_fc)

    p = sub.add_parser("proofread-sheet",
                       help="sheet for the native-speaker proofreading")
    files(p)
    p.add_argument("--outputs", action="append", required=True,
                   help="a pilot cache.jsonl; repeat for the three models")
    p.add_argument("--conditions", nargs="+", default=["L-BB0", "L-EE0"])
    p.add_argument("--out-dir", required=True, help="a folder in Drive")
    p.add_argument("--max-rows", type=int, default=220)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_proofread_sheet)

    r = sub.add_parser("proofread-read", help="read the filled sheet back")
    files(r)
    r.add_argument("--sheet", required=True, help="the filled .xlsx")
    r.add_argument("--key", required=True, help="proofread_key.csv")
    r.add_argument("--outputs", action="append", required=True)
    r.add_argument("--conditions", nargs="+", default=["L-BB0", "L-EE0"])
    r.add_argument("--out-dir", required=True, help="a folder in Drive")
    r.set_defaults(func=cmd_proofread_read)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
