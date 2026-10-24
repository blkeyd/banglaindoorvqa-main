#!/usr/bin/env python3
"""
scripts/explore_p7.py  (Statistics chat, D293; exploratory, fixed before its result)
The word-form bound: P7 recomputed with every Bangla answer labelled
english_copy counted as transferred. The most generous reading of the view that
spelling variants missing from the frozen lexicon explain the negative P7. If P7
stays below 0 under it, the reversal does not come from spelling.

    python scripts/explore_p7.py --scored "<Drive>/scored/main/*_scored.jsonl" \
        --primary-run latest --out <Drive>/results/stats

Rules kept here
    - The pre-specified P7 (Design 9.1, H4) stays primary; this is reported as
      exploratory (D293), never as a test of H4.
    - Same items (stats.p7_items), same house resamples (P7 pool, 10,000, seed 42),
      same Holm rule (within the model, with this P7 in place of the primary P7).
    - The script first recomputes the primary P7 and its two transfer rates; they
      must equal the primary run's numbers.json, or nothing is written.
    - src/stats.py, scoring.py, normalise.py and the lexicon must equal their tags;
      this script changes no input and imports the functions, it copies none.
    - Printouts hold ids, counts and estimates only.
"""

import argparse
import json
import sys
import time
from pathlib import Path

VERSION = "explore-p7-d293-2026-10-24"
REPO = Path(__file__).resolve().parents[1]
DRIVE_DEFAULT = Path("/content/drive/MyDrive/BanglaIndoorVQA")
PRIMARY_SCRIPT = "analysis-2026-10-17"
TESTED = ("P12", "P3", "P4", "P5", "P6", "P7")
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import stats as S  # noqa: E402


def say(msg=""):
    print(msg, flush=True)


def same(a, b, tol=1e-12):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="D293: exploratory word-form bound (P7 with English copies)")
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--drive", default=str(DRIVE_DEFAULT))
    ap.add_argument("--scored", nargs="+", required=True)
    ap.add_argument("--primary-run", required=True, help="a run_<UTC> folder, or latest (newest run_ in --out)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    t0 = time.time()
    say("explore_p7 (%s, %s, UTC %s)" % (VERSION, S.STATS_VERSION, S.now_utc()))
    checks = S.Checks()
    paths = S.Paths.make(args.repo, args.drive)
    S.guard_out(args.out, "real", paths.repo, paths.drive)
    for tag, rel in (("gate-E", "src/stats.py"), ("gate-F", "src/scoring.py"), ("gate-F", "src/normalise.py"),
                     ("gate-A", "lexicon/lexicon.csv"), ("gate-A", "lexicon/closed_vocab.csv")):
        ok = S.same_as_tag(paths.repo, tag, rel)
        checks.check(ok is True, "%s at tag %s" % (rel, tag), "identical" if ok else "differs or tag missing")
    checks.check(Path(S.__file__).resolve() == (paths.repo / "src" / "stats.py").resolve(), "running src/stats.py",
                 "the file in --repo")
    world = S.build_world(paths, checks, need_dataset=False)
    recs = S.load_scored(args.scored, world, checks, "real")
    base = Path(args.out)
    prun = Path(args.primary_run)
    if args.primary_run == "latest":
        found = sorted(p for p in base.glob("run_*") if p.is_dir() and "PLACEHOLDER" not in p.name)
        prun = found[-1] if found else base / "run_missing"
    nums = man = None
    if (prun / "numbers.json").exists() and (prun / "manifest.json").exists():
        nums = json.loads((prun / "numbers.json").read_text(encoding="utf-8"))
        man = json.loads((prun / "manifest.json").read_text(encoding="utf-8"))
        same_files = sorted((Path(f["file"]).name, f["sha256"]) for f in man.get("scored_files", [])) == \
            sorted((Path(f["file"]).name, f["sha256"]) for f in recs.files)
        ok = (man.get("mode") == "real" and man.get("failed") == [] and man.get("script_version") == PRIMARY_SCRIPT
              and man.get("n_boot") == S.N_BOOT and man.get("seed") == S.SEED and not man.get("deviation")
              and same_files)
        checks.check(ok, "primary run (gate-E code)", "%s: mode %s, script %s, failed stages %d, same scored files %s"
                     % (prun.name, man.get("mode"), man.get("script_version"), len(man.get("failed") or []),
                        same_files))
    else:
        checks.add("FAIL", "primary run (gate-E code)", "no numbers.json and manifest.json in %s" % prun)
    if checks.count("FAIL"):
        say("explore_p7: %d FAIL; nothing written" % checks.count("FAIL"))
        return 1
    pitems = S.p7_items(world)
    ids = [p[0] for p in pitems]
    boot = S.HouseBoot(world.houses_of(ids), n_boot=S.N_BOOT, name="P7")
    hidx = boot.hidx(world.houses_of(ids))
    route = np.array([S._b(world.items[q]["en_route"]) for q in ids])
    res, rows, bad = {}, [], []
    for m in S.MODELS:
        E = np.array([S.strict_of(recs.get(m, e)[q]) for q, e, b, _ in pitems])
        Bn = np.array([S.strict_of(recs.get(m, b)[q]) for q, e, b, _ in pitems])
        EC = np.array([S._b(recs.get(m, b)[q]["label"] == "english_copy") for q, e, b, _ in pitems])
        Bb = np.maximum(Bn, EC)
        out = {}
        for name, Bv in (("primary", Bn), ("bound", Bb)):
            tr = {g: boot.ratio(hidx, E * Bv * (route == v), E * (route == v)) for g, v in (("yes", 1), ("no", 0))}
            out[name] = {"P7": S.summarise_est(tr["yes"] - tr["no"], test=True),
                         "transfer": {g: S.summarise_est(tr[g]) for g in tr}}
        pp = nums["contrasts"][m]["P7"]
        tp = nums["exp_n"]["models"][m]["transfer"]
        for k in ("point", "lo", "hi", "p"):
            if not same(out["primary"]["P7"].get(k), pp.get(k)):
                bad.append("%s P7 %s" % (m, k))
        for g in ("yes", "no"):
            for k in ("point", "lo", "hi"):
                if not same(out["primary"]["transfer"][g].get(k), tp[g].get(k)):
                    bad.append("%s transfer %s %s" % (m, g, k))
        fam = {k: nums["contrasts"][m][k] for k in TESTED if k != "P7"}
        fam["P7"] = out["bound"]["P7"]
        out["holm_with_bound"] = S.holm_bootstrap(fam)
        out["counts"] = {g: {"named_in_english": int((E * (route == v)).sum()),
                             "english_copy_among_them": int((E * EC * (route == v)).sum()),
                             "strict_bangla_among_them": int((E * Bn * (route == v)).sum())}
                         for g, v in (("yes", 1), ("no", 0))}
        res[m] = out
    checks.check(not bad, "primary P7 reproduced", "P7 and both transfer rates of the three models equal"
                 " %s/numbers.json: %s" % (prun.name, "yes" if not bad else "no (%s)" % ", ".join(bad[:6])))
    if bad:
        say("explore_p7: the primary P7 is not reproduced; nothing written")
        return 1
    folder = S.guard_out(args.out, "real", paths.repo, paths.drive) / ("explore_D293_%s" % S.utc_stamp())
    folder.mkdir(parents=True, exist_ok=False)
    for m in S.MODELS:
        o = res[m]
        c = o["counts"]
        rows.append([S.MODEL_NAMES[m], "P7 with English copies counted as transferred (exploratory, D293)",
                     "%d / %d" % (c["yes"]["named_in_english"], c["no"]["named_in_english"]),
                     S.ci_text(o["primary"]["P7"]), S.ci_text(o["bound"]["P7"]), S.fmt_ps(o["bound"]["P7"]),
                     S.fmt_holm(o["holm_with_bound"].get("P7")),
                     "%s / %s" % (S.ci_text(o["bound"]["transfer"]["yes"]), S.ci_text(o["bound"]["transfer"]["no"])),
                     "%d / %d" % (c["yes"]["english_copy_among_them"], c["no"]["english_copy_among_them"])])
    tab = S.write_table(folder / "supplement" / "S3_rows_D293_exploratory",
                        ["model", "group", "items named in English (en_route yes / no)", "primary P7 pp [95% CI]",
                         "bound P7 pp [95% CI]", "p (bound)", "Holm p (bound in the family)",
                         "bound transfer % yes / no [95% CI]", "English copies among them (yes / no)"],
                        rows, False, "S3, exploratory rows (D293): P7 with every Bangla answer labelled English copy"
                                     " counted as transferred; the pre-specified P7 stays primary")
    lines = ["# Exploratory word-form bound (D293)", "",
             "script %s, stats %s, scorer %s, UTC %s; primary run %s; 10,000 resamples, seed 42, the P7 pool (%d"
             " items, %d houses)" % (VERSION, S.STATS_VERSION, recs.scorer_version, S.now_utc(), prun.name,
                                     len(ids), boot.H), "",
             "The pre-specified P7 is reproduced exactly. Bound: an answer labelled English copy in the Bangla"
             " condition counts as transferred. Exploratory; H4 is read from the primary P7 only.", ""]
    for r in rows:
        lines.append("- %s: primary P7 %s; bound %s, p %s, Holm %s; bound transfer yes / no %s; English copies"
                     " among the items named in English %s (yes / no)" % (r[0], r[3], r[4], r[5], r[6], r[7], r[8]))
    keep = ["%s (bound CI %s)" % (S.MODEL_NAMES[m], "excludes 0" if (res[m]["bound"]["P7"]["hi"] < 0
                                                                     or res[m]["bound"]["P7"]["lo"] > 0)
                                   else "includes 0") for m in S.MODELS]
    lines += ["", "Reading: %s" % "; ".join(keep), ""]
    S.write_text(folder / "explore_report.md", "\n".join(lines))
    S.write_json(folder / "explore_numbers.json", {"version": VERSION, "primary_run": prun.name, "models": res})
    S.write_json(folder / "manifest.json", {"script_version": VERSION, "stats_version": S.STATS_VERSION,
                                            "utc": S.now_utc(), "mode": "real", "n_boot": S.N_BOOT, "seed": S.SEED,
                                            "primary_run": prun.name, "decision": "D293 (exploratory)",
                                            "git_head": S.git_out(paths.repo, "rev-parse", "--short=12", "HEAD"),
                                            "inputs": world.hashes, "scored_files": recs.files, "tables": [tab],
                                            "seconds": round(time.time() - t0, 1)})
    S.write_text(folder / "checks.txt", checks.text())
    say("")
    for line in lines[6:]:
        if line:
            say(line)
    say("explore_p7: %d FAIL, %d WARN; written to %s (%.0f s)" % (checks.count("FAIL"), checks.count("WARN"),
                                                                 folder, time.time() - t0))
    return 1 if checks.count("FAIL") else 0


if __name__ == "__main__":
    sys.exit(main())
