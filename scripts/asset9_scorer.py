"""
asset9_scorer.py  (ASSET-9)
Two commands around src/scoring.py.

  audit   Check a lexicon file and a vocabulary file, then check every gold
          answer of one or more probe files against them.
          Stage 1: run with the made-up fixture files, to exercise the scorer
                   on real gold answers. Partial coverage is expected.
          Stage 2 (Gate A): run with the frozen lexicon.csv and
                   closed_vocab.csv and add --require-full. Every gold of a
                   closed-answer probe must then be scoreable.

  score   Score a file of raw model outputs (open conditions). Forced choice
          needs the ASSET-11 option file, whose format does not exist yet, so
          those records are counted and skipped for now.

Everything this script writes holds dataset text. It refuses to write inside
the Git repo: outputs belong in Google Drive (setup file Part 7, rule 1).

Examples (from the repo root)
  python scripts/asset9_scorer.py audit \
      --probe  "$DRIVE/data/P-CORE.json" \
      --lexicon tests/fixtures/lexicon_FIXTURE.csv \
      --vocab   tests/fixtures/closed_vocab_FIXTURE.csv \
      --out-dir "$DRIVE/scored/asset9_audit"

  python scripts/asset9_scorer.py score \
      --outputs "$DRIVE/outputs/pilot.jsonl" --answer-field <name> \
      --probe   "$DRIVE/probes/PILOT.json" \
      --lexicon lexicon/lexicon.csv --vocab lexicon/closed_vocab.csv \
      --keep model_id --out "$DRIVE/scored/pilot_scored.jsonl" --show 10
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import scoring as S                      # noqa: E402
from normalise import normalise          # noqa: E402

SEED = 42
QA_TYPE_OF_KIND = {"room": "scene_id", "material": "obj_property",
                   "colour": "obj_property", "number": "count"}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def refuse_inside_repo(path):
    """Data-derived files never go into the Git repo."""
    p = Path(path).resolve()
    if p == REPO or REPO in p.parents:
        sys.exit("STOP: %s is inside the Git repo (%s).\n"
                 "Files with dataset text stay in Google Drive. Choose a path"
                 " under MyDrive/BanglaIndoorVQA/." % (p, REPO))
    return p


def print_table(rows, columns):
    if not rows:
        print("    (no rows)")
        return
    cells = [[("" if r.get(c) is None else str(r.get(c))) for c in columns]
             for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells))
              for i, c in enumerate(columns)]
    line = "    " + "  ".join(c.ljust(w) for c, w in zip(columns, widths))
    print(line)
    print("    " + "  ".join("-" * w for w in widths))
    for row in cells:
        print("    " + "  ".join(v.ljust(w) for v, w in zip(row, widths)))


def seeded_rank(key):
    """Same sampling method as the probes: lowest SHA-256 rank, seed 42."""
    return hashlib.sha256(("%d|ASSET-9|%s" % (SEED, key)).encode()).hexdigest()


def fake_item(qa_type, gold):
    return {"qa_id": "selftest", "qa_type": qa_type, "answer_bn": gold}


# --------------------------------------------------------------------------
# audit, part 1: the lexicon and the vocabulary against the scoring rules
# --------------------------------------------------------------------------

def self_test_lexicon(lex, voc):
    """Every row of the lexicon must behave as Design 8.3 says:
    its own names are Tier 1, its generic names are Tier 2, its siblings are
    siblings, another object is wrong. Returns (number of checks, failures)."""
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
            fails.append("%s (%s): %s %r in %s gave %s, strict %s; expected"
                         " %s, strict %s"
                         % (obj.name_id, obj.name_bn, what, answer, cond,
                            r["label"], r["strict"], want_label, want_strict))

    def other_label(obj, term, fallback):
        key = lex.index.key_of(term)
        if key in obj.keys_t1:
            return S.TIER1
        if key in obj.keys_t2:
            return S.TIER2
        return fallback

    for oid in ids:
        obj = lex.objects[oid]
        for t in obj.t1_bn:
            check(obj, "N-BB", t, S.TIER1, 1, "own Bangla name")
            check(obj, "N-BE", t, S.TIER1, 0, "own Bangla name")
        for t in obj.t1_en:
            check(obj, "N-BE", t, S.TIER1, 1, "own English name")
            check(obj, "N-BB", t, S.TIER1, 0, "own English name")
        for t in obj.t2_bn:
            check(obj, "N-BB", t, S.TIER2, 0, "own generic Bangla name")
        for t in obj.t2_en:
            check(obj, "N-BE", t, S.TIER2, 0, "own generic English name")
        sibs = lex.siblings_of(oid)
        for sid in sorted(sibs):
            sib = lex.objects[sid]
            check(obj, "N-BB", sib.name_bn,
                  other_label(obj, sib.name_bn, S.SIBLING), None,
                  "sibling's Bangla name")
            if sib.t1_en:
                check(obj, "N-BE", sib.t1_en[0],
                      other_label(obj, sib.t1_en[0], S.SIBLING), None,
                      "sibling's English name")
        for xid in ids:
            if xid != oid and xid not in sibs:
                other = lex.objects[xid]
                check(obj, "N-BB", other.name_bn,
                      other_label(obj, other.name_bn, S.OTHER_WRONG), None,
                      "another object's Bangla name")
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
                fails.append("%s %s: no Bangla form, so no gold answer can"
                             " map to this class" % (kind, cid))
                continue
            it = fake_item(qa_type, golds[0])
            for key, lang in by_class[cid]:
                n += 1
                r = S.score_item(it, "L-BB" if lang == "bn" else "L-BE", key,
                                 lex, voc)
                if not r["scoreable"] or r["strict"] != 1:
                    fails.append("%s %s: own form %r gave %s (%s)"
                                 % (kind, cid, key, r["label"],
                                    r["gold_status"]))
            for other in sorted(by_class):
                if other == cid:
                    continue
                key = by_class[other][0][0]
                n += 1
                r = S.score_item(it, "L-BB", key, lex, voc)
                if r["scoreable"] and r["lenient"] != 0:
                    fails.append("%s %s: form %r of class %s scored correct"
                                 % (kind, cid, key, other))
                break
    return n, fails


# --------------------------------------------------------------------------
# audit, part 2: real gold answers
# --------------------------------------------------------------------------

def robustness(items):
    """Every answer and question must pass through the scorer's text steps."""
    errors = []
    for it in items:
        for fld in ("answer_bn", "question_bn"):
            if fld in it:
                try:
                    S.prepare_answer(it[fld])
                    S.romanise_bn(it[fld])
                except Exception as exc:               # report, never hide
                    errors.append("%s %s: %r" % (it["qa_id"], fld, exc))
    return errors


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
                fails.append("%s [%s] gold %r -> %s %s"
                             % (it["qa_id"], name, gold, r["label"],
                                r["matched"]))
    merged = [g for g in groups.values() if len(g) > 1]
    return {"items": len(naming),
            "distinct": sum(len(g) for g in groups.values()),
            "names": len(groups), "merged": merged, "passed": result,
            "fails": fails}


def audit_golds(items, lex, voc):
    """Resolve every gold with the supplied files, then score each resolved
    gold as if it were a model answer: it must be strict-correct."""
    coverage = OrderedDict()
    stats = OrderedDict()
    self_fails = []
    langs = Counter()
    n_self = 0
    for it in items:
        qa_type = it["qa_type"]
        st = stats.setdefault(qa_type, Counter())
        st["items"] += 1
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
                self_fails.append("%s gold %r -> %s (found %s)"
                                  % (it["qa_id"], gold.norm, r["label"],
                                     r["matched"] or r["found_value"]))
        key = (qa_type, gold.norm)
        row = coverage.setdefault(key, {
            "qa_type": qa_type, "kind": gold.kind, "gold": gold.norm,
            "n_items": 0, "status": gold.status, "mapped_to": mapped,
            "hint": gold.hint, "extra_words": " ".join(gold.residual),
            "example_qa_id": it["qa_id"]})
        row["n_items"] += 1
    return {"coverage": list(coverage.values()), "stats": stats,
            "self_n": n_self, "self_fails": self_fails, "langs": langs}


def cmd_audit(args):
    out_dir = refuse_inside_repo(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = S.MatchSettings(
        spacing_insensitive=not args.no_spacing_rule,
        suffix_tolerance=not args.no_suffix_rule)
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    say("ASSET-9 audit, scorer %s" % S.SCORER_VERSION)
    say("Matching rules: %s" % settings.describe())
    try:
        lex = S.load_lexicon(args.lexicon, settings)
        voc = S.load_closed_vocab(args.vocab, settings)
    except S.LexiconError as exc:
        say("\nFILE ERROR\n%s" % exc)
        say("\nRESULT: FAIL. Fix the file and run again.")
        return 1
    say("Lexicon   : %s -> %s" % (args.lexicon, lex.describe()))
    say("Vocabulary: %s -> %s" % (args.vocab, voc.describe()))
    for name, obj in (("Lexicon", lex), ("Vocabulary", voc)):
        say("%s warnings: %d" % (name, len(obj.warnings)))
        for w in obj.warnings[:args.max_list]:
            say("    - " + w)
        if len(obj.warnings) > args.max_list:
            say("    ... %d more" % (len(obj.warnings) - args.max_list))

    failed = False

    def report(title, n, fails):
        nonlocal failed
        say("%s: %d checks, %d failures" % (title, n, len(fails)))
        for f in fails[:args.max_list]:
            say("    - " + f)
        if len(fails) > args.max_list:
            say("    ... %d more" % (len(fails) - args.max_list))
        failed = failed or bool(fails)

    say()
    report("[1] Lexicon self-test", *self_test_lexicon(lex, voc))
    report("[2] Vocabulary self-test", *self_test_vocab(lex, voc))

    all_rows = []
    for probe_path in args.probe:
        items = S.load_probe(probe_path)
        name = Path(probe_path).name
        say()
        say("Probe: %s, %d items" % (name, len(items)))

        errors = robustness(items)
        report("[3] Text steps on every answer and question", len(items),
               errors)

        ident = identity_check(items, settings)
        say("[4] Identity check on naming golds: %d items, %d distinct gold"
            " strings, %d names" % (ident["items"], ident["distinct"],
                                    ident["names"]))
        for vname, passed in ident["passed"].items():
            say("    %-26s %d / %d" % (vname, passed, ident["items"]))
        if ident["merged"]:
            say("    Spellings the spacing rule treats as one name: %d group(s)"
                % len(ident["merged"]))
            for g in ident["merged"][:args.max_list]:
                say("    - " + " = ".join(g))
        for f in ident["fails"][:args.max_list]:
            say("    - FAIL " + f)
        failed = failed or bool(ident["fails"])

        res = audit_golds(items, lex, voc)
        say("[5] Gold answers against the supplied lexicon and vocabulary")
        table = []
        not_scoreable = 0
        for qa_type, st in res["stats"].items():
            row = {"qa_type": qa_type, "items": st["items"],
                   "scoreable": st["scoreable"],
                   "unmapped": st["gold_unmapped"],
                   "ambiguous": st["gold_ambiguous"],
                   "not closed": st["not a closed type"]}
            table.append(row)
            not_scoreable += st["gold_unmapped"] + st["gold_ambiguous"]
        print_table(table, ["qa_type", "items", "scoreable", "unmapped",
                            "ambiguous", "not closed"])
        lines.append("    " + json.dumps(table, ensure_ascii=False))
        say("    Gold language, closed types: %s"
            % ", ".join("%s %d" % kv for kv in sorted(res["langs"].items())))
        extra = [r for r in res["coverage"]
                 if r["status"] == "ok" and r["extra_words"]]
        say("    Scoreable golds with words outside every listed term: %d"
            " distinct (see extra_words in the CSV)" % len(extra))
        report("[6] Each scoreable gold scored as its own answer",
               res["self_n"], res["self_fails"])
        if args.require_full and not_scoreable:
            say("    --require-full: %d closed-type item(s) are not scoreable"
                % not_scoreable)
            failed = True
        for r in res["coverage"]:
            r["probe"] = name
        all_rows.extend(res["coverage"])

    csv_path = out_dir / "gold_coverage.csv"
    columns = ["probe", "qa_type", "kind", "gold", "n_items", "status",
               "mapped_to", "hint", "extra_words", "example_qa_id"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerows(all_rows)
    say()
    say("Wrote %s (%d rows: one per distinct gold answer and type)"
        % (csv_path, len(all_rows)))
    say("RESULT: %s" % ("FAIL" if failed else "PASS"))
    if not failed and not args.require_full:
        say("(Unmapped golds are expected with fixture files. At Gate A, run"
            " with the frozen files and --require-full.)")
    summary = out_dir / "audit_summary.txt"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote %s" % summary)
    return 1 if failed else 0


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------

def read_outputs(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        recs = []
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        sys.exit("%s line %d is not JSON: %s"
                                 % (path, n, exc))
        return recs
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        data = data["items"]
    if not isinstance(data, list):
        sys.exit("%s: expected JSON Lines, a JSON list, or {'items': [...]}"
                 % path)
    return data


def cmd_score(args):
    out_path = refuse_inside_repo(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    settings = S.MatchSettings(
        spacing_insensitive=not args.no_spacing_rule,
        suffix_tolerance=not args.no_suffix_rule)
    print("ASSET-9 score, scorer %s" % S.SCORER_VERSION)
    print("Matching rules: %s" % settings.describe())
    lex = S.load_lexicon(args.lexicon, settings)
    voc = S.load_closed_vocab(args.vocab, settings)
    print("Lexicon   : %s -> %s" % (args.lexicon, lex.describe()))
    print("Vocabulary: %s -> %s" % (args.vocab, voc.describe()))

    items = {}
    for probe_path in args.probe:
        for it in S.load_probe(probe_path):
            items[it["qa_id"]] = it
    print("Probe items: %d from %d file(s)" % (len(items), len(args.probe)))

    recs = read_outputs(args.outputs)
    print("Raw outputs: %d records from %s" % (len(recs), args.outputs))
    if not recs:
        sys.exit("No records to score.")
    for fld in [args.qa_id_field, args.condition_field, args.answer_field] \
            + list(args.keep):
        if fld not in recs[0]:
            sys.exit("Field %r is not in the output records. Fields found: %s"
                     % (fld, ", ".join(sorted(recs[0]))))

    scored = []
    skipped = Counter()
    seen = set()
    for rec in recs:
        qa_id = rec[args.qa_id_field]
        cond = rec[args.condition_field]
        if qa_id not in items:
            skipped["qa_id not in the probe file(s)"] += 1
            continue
        if S.condition_spec(cond).task == "fc":
            skipped["forced choice (needs the ASSET-11 option file)"] += 1
            continue
        dup = (qa_id, cond) + tuple(str(rec.get(k)) for k in args.keep)
        if dup in seen:
            skipped["duplicate of an earlier record"] += 1
            continue
        seen.add(dup)
        out = S.score_item(items[qa_id], cond, rec.get(args.answer_field),
                           lex, voc)
        for k in args.keep:
            out[k] = rec.get(k)
        scored.append(out)

    with open(out_path, "w", encoding="utf-8") as fh:
        for r in scored:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Scored: %d records" % len(scored))
    for why, n in skipped.items():
        print("Skipped: %d, %s" % (n, why))
    by = tuple(args.keep) + ("condition_id",)
    print("\nSummary (percent). For eyeballing only; statistics come later.")
    print_table(S.summarise(scored, by),
                list(by) + ["n", "scoreable", "strict", "lenient", "fidelity",
                            "refusal", "empty", "le8_words", "mixed_script"])
    labels = Counter((r["kind"], r["label"]) for r in scored)
    print("\nLabels:")
    for (kind, label), n in sorted(labels.items(), key=lambda kv: str(kv[0])):
        print("    %-10s %-13s %d" % (kind, label, n))
    if args.show:
        pick = sorted(scored, key=lambda r: seeded_rank(
            "%s|%s|%s" % (r["qa_id"], r["condition_id"],
                          "|".join(str(r.get(k)) for k in args.keep))))
        print("\n%d seeded-random items (seed %d):" % (args.show, SEED))
        for r in pick[:args.show]:
            print("    %s %s | gold %r | answer %r | %s | strict %s lenient"
                  " %s | lang %s"
                  % (r["qa_id"], r["condition_id"], r["gold_norm"],
                     r["answer_norm"], r["label"], r["strict"], r["lenient"],
                     r["lang"]))
    print("\nWrote %s" % out_path)
    return 0


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="ASSET-9: audit the scorer's inputs, or score outputs.")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--probe", action="append", required=True,
                       help="probe file in the P-CORE.json record format;"
                            " repeat for several files")
        p.add_argument("--lexicon", required=True, help="lexicon.csv")
        p.add_argument("--vocab", required=True, help="closed_vocab.csv")
        p.add_argument("--no-spacing-rule", action="store_true",
                       help="switch the spacing rule off (sensitivity check)")
        p.add_argument("--no-suffix-rule", action="store_true",
                       help="switch the ending rule off (sensitivity check)")

    a = sub.add_parser("audit", help="check lexicon, vocabulary and golds")
    common(a)
    a.add_argument("--out-dir", required=True, help="a folder in Drive")
    a.add_argument("--require-full", action="store_true",
                   help="fail unless every closed-type gold is scoreable")
    a.add_argument("--max-list", type=int, default=15,
                   help="how many findings to print per list")
    a.set_defaults(func=cmd_audit)

    s = sub.add_parser("score", help="score raw model outputs")
    common(s)
    s.add_argument("--outputs", required=True,
                   help="raw outputs: .jsonl, or a .json list")
    s.add_argument("--answer-field", required=True,
                   help="name of the field holding the model's answer text")
    s.add_argument("--qa-id-field", default="qa_id")
    s.add_argument("--condition-field", default="condition_id")
    s.add_argument("--keep", action="append", default=[],
                   help="field to copy into the scored file, for example"
                        " model_id; repeat for several")
    s.add_argument("--out", required=True, help="scored .jsonl, in Drive")
    s.add_argument("--show", type=int, default=0,
                   help="print this many seeded-random scored items")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
