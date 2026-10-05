#!/usr/bin/env python3
"""
asset8_harness.py
BanglaIndoorVQA, ASSET-8: command line of the inference harness (Design 4.9).
The logic lives in src/harness.py, src/prompts.py and src/images.py.

Commands (run from the repo root, in Colab)
    selftest         no network, no GPU, no Drive: every line must print OK
    prompts          print the prompt version, its lock and every template
    prepare-images   make the one shared 1,024 px copy of each image (Design 5.2)
    run              run conditions for one model: resumable, cached, metered
    report           counts for one stage and model; never shows an answer
    peek             show a few answers (model output: needs a flag)

Stages (configs/conditions.yaml, decision D144)
    smoke   harness tests; any probe file and subset; never reported
    pilot   PILOT.json; Gemini in standard mode
    main    EXP-L, EXP-N, EXP-R; Gemini in Batch mode; needs the tag gate-E

Examples
    python scripts/asset8_harness.py selftest
    python scripts/asset8_harness.py prepare-images --probe "$P/P-CORE.json"
    # a 100-example dry run: no network, no GPU, no cost
    python scripts/asset8_harness.py run --stage smoke --run-name dry100 --model gemini \\
        --mode batch --conditions L-BB,L-BB0 --probe "$P/P-CORE.json" --sample 100 --dry-run
    # an open model on the L4
    python scripts/asset8_harness.py run --stage smoke --run-name t1 --model qwen \\
        --conditions L-BB,L-BB0 --probe "$P/P-CORE.json" --sample 20
    # Gemini, billed
    python scripts/asset8_harness.py run --stage smoke --run-name t1 --model gemini \\
        --mode standard --conditions L-BB,L-BB0 --probe "$P/P-CORE.json" --sample 10 \\
        --paid-tier-confirmed

Keys come from the environment (GEMINI_API_KEY, HF_TOKEN), which the notebook
fills from Colab Secrets. They are never written to a file.
Everything a run writes is model output and goes to Google Drive only.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
try:
    import cost_ledger
    import harness as H
    import images as imglib
    import prompts as promptlib
except ImportError as _exc:  # pragma: no cover
    sys.exit(f"STOP: cannot import the files in src/ ({_exc}). Run this script from the "
             "repo, with normalise.py, cost_ledger.py, prompts.py, images.py and harness.py "
             "in src/.")

DEFAULT_ROOT = "/content/drive/MyDrive/BanglaIndoorVQA"


def cmd_prompts(a):
    path = Path(a.repo) / "configs" / "prompts.yaml"
    p = promptlib.load_prompts(path, check_lock=False)
    H.say(f"{path}")
    H.say(f"prompt_version: {p['prompt_version']}")
    H.say(f"sha256 of the template texts: {p['sha256']}")
    H.say("lock: " + ("holds" if p["locked"] else "DOES NOT HOLD. If the change is meant: give "
                      "prompt_version a new value and copy that value and the sha256 above "
                      "into the lock (decision row needed)."))
    cfg = promptlib.load_conditions(Path(a.repo) / "configs" / "conditions.yaml", p)
    users = {}
    for cid, c in cfg["conditions"].items():
        users.setdefault((c["template"], "with_image" if c["image"] else "no_image"), []).append(cid)
    for tid, variants in p["templates"].items():
        for variant, text in variants.items():
            H.say(f"\n--- {tid}.{variant}   used by: {', '.join(users.get((tid, variant), [])) or 'no condition'}")
            H.say(text)
    return 0 if p["locked"] else 1


def ns(**kw):
    """The arguments of 'run', with every default, for the selftest."""
    base = dict(repo=str(REPO), root="", stage="smoke", run_name="t", model="gemini",
                conditions="L-BB", probe="", where=[], ids_file="", sample=0, limit=0,
                mode=None, english_file="", english_column="", options_file="",
                images_1024="", ledger="", paid_tier_confirmed=False, dry_run=True,
                dry_keep=False, plan_only=False, no_submit=False, wait=False, poll_seconds=0,
                stage_cap_usd=None, price_in=None, price_out=None, allow_env_change=False)
    base.update(kw)
    return argparse.Namespace(**base)


def cmd_selftest(a):
    """Everything that can be checked without the network, a GPU or Drive."""
    from PIL import Image
    checks, logs = [], {}
    root = Path(tempfile.mkdtemp(prefix="asset8_selftest_"))
    (root / "probes").mkdir()
    gold = "GOLD-NEVER-SENT"
    items = []
    types = ["obj_name", "object", "count", "scene_id", "obj_property"]
    for i in range(30):
        house = f"H{1 + i % 3:02d}"
        img = f"{house}_{1 + i % 4:04d}"
        items.append({"qa_id": f"{img}_q{1 + i // 12}x{i}", "image_id": img, "house_id": house,
                      "capture_type": "closeup", "qa_type": types[i % 5],
                      "question_bn": f"  পরীক্ষার প্রশ্ন নম্বর {i % 7} কী? ",
                      "answer_bn": f"{gold}-{i}", "image_file": f"{house}/{img}.jpg"})
        d = root / "data" / "images" / house
        d.mkdir(parents=True, exist_ok=True)
        if not (d / f"{img}.jpg").exists():
            kw = {}
            if i % 4 == 1:
                ex = Image.Exif()
                ex[0x0112] = 6
                kw["exif"] = ex.tobytes()
            Image.new("RGB", (1600, 1200), (40 * (i % 5), 90, 160)).save(d / f"{img}.jpg", quality=92, **kw)
    probe = root / "probes" / "T.json"
    probe.write_text(json.dumps({"probe": "T", "n": len(items), "created_utc": "selftest",
                                 "items": items}, ensure_ascii=False), encoding="utf-8")
    from normalise import normalise
    en = root / "english.csv"
    norms = sorted({normalise(it["question_bn"]) for it in items})
    with open(en, "w", encoding="utf-8-sig", newline="") as f:
        f.write("string_id,question_bn_norm,en_final,mt_en\n")
        for k, n in enumerate(norms):
            f.write(f'q{k},"{n}","What is test question {k}?","What is test question {k}?"\n')
    opts = root / "options.jsonl"
    with open(opts, "w", encoding="utf-8") as f:
        for it in items:
            for lang in ("bn", "en"):
                f.write(json.dumps({"qa_id": it["qa_id"], "lang": lang,
                                    "options": ["opt-1", "opt-2", "opt-3", "opt-4"]}) + "\n")

    def quiet(name, fn, *args):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                out = fn(*args)
        except Exception as exc:        # a check below decides whether this was expected
            out = exc
        logs[name] = buf.getvalue()
        return out

    def expect_stop(fn, *args):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fn(*args)
        except (H.Stop, promptlib.PromptError, cost_ledger.BudgetExceeded):
            return True
        return False

    # ---- images ---------------------------------------------------------------
    rc = quiet("prepare", H.prepare_images, argparse.Namespace(
        root=str(root), images="", images_1024="", probe=[str(probe)], all=False, verify=False))
    man = imglib.read_manifest(root / "data" / "images_1024")
    checks.append(("prepare-images makes one copy per image, 1024 px on the longer side",
                   rc == 0 and len(man) == 12 and all(max(int(r["width"]), int(r["height"])) == 1024
                                                      for r in man.values())))
    checks.append(("EXIF orientation is applied first (rotated images are 768x1024)",
                   sum(1 for r in man.values() if (r["width"], r["height"]) == ("768", "1024")) == 3))

    # ---- end-to-end dry runs ---------------------------------------------------
    common = dict(root=str(root), probe=str(probe), english_file=str(en),
                  english_column="en_final", options_file=str(opts))
    rc = quiet("std", H.run, ns(conditions="L-BB,L-BB0,L-EE,N-FCB,N-FCE0", mode="standard", **common))
    checks.append(("Gemini standard path: 5 conditions x 30 items, faults survived, resume = 0 calls", rc == 0))
    rc = quiet("batch", H.run, ns(conditions="L-BB,L-EE0,N-BBs", mode="batch", run_name="b", **common))
    checks.append(("Gemini Batch path: lost create call, adopted job, error line, missing line, "
                   "three reply shapes; no item submitted twice; reservations released", rc == 0))
    bdir = root / "outputs" / "dryrun" / "smoke" / "b" / "gemini"
    evs, _bad = H.read_jsonl(bdir / "batch" / "jobs.jsonl")
    kinds = {e["event"] for e in evs}
    checks.append(("the batch registry saw: planned, reserved, uploaded, create_sent, "
                   "create_unknown, adopted, not_on_server, submitted, state, collected",
                   {"planned", "reserved", "uploaded", "create_sent", "create_unknown", "adopted",
                    "not_on_server", "submitted", "state", "collected"} <= kinds))
    led = cost_ledger.read_lines(str(bdir / "dryrun_ledger.jsonl"))
    modes = {r["mode"] for r in led}
    checks.append(("the ledger holds reserve, release and real-cost lines, and no negative total",
                   {"batch-reserve", "batch-release", "batch"} <= modes
                   and cost_ledger.read_total(str(bdir / "dryrun_ledger.jsonl")) > 0))
    cache_b = H.Cache(bdir / "cache.jsonl")
    checks.append(("one answer per key in Batch mode (90 keys, 0 repeated)",
                   len(cache_b.records) == 90 and cache_b.duplicates == 0))
    rc = quiet("hf", H.run, ns(conditions="L-BB,N-BBs,L-EB", model="qwen", **common))
    checks.append(("open-model path (stand-in backend): 3 conditions x 30 items", rc == 0))

    # ---- resume and reuse (two commands on one folder) -------------------------
    rc1 = quiet("res1", H.run, ns(conditions="L-BB", model="gemma", run_name="r", limit=7, **common))
    rc2 = quiet("res2", H.run, ns(conditions="L-BB", model="gemma", run_name="r", dry_keep=True, **common))
    c = H.Cache(root / "outputs" / "dryrun" / "smoke" / "r" / "gemma" / "cache.jsonl")
    runs = {}
    for r in c.records.values():
        runs[r["run_id"]] = runs.get(r["run_id"], 0) + 1
    checks.append(("a second command runs only what is missing (7, then 23)",
                   rc1 == 0 and rc2 == 0 and len(c.records) == 30 and sum(runs.values()) == 30
                   and "to run     7" in logs["res1"] and "to run    23" in logs["res2"]))
    rc3 = quiet("reuse", H.run, ns(conditions="R-BB", model="gemma", run_name="r", dry_keep=True, **common))
    c = H.Cache(root / "outputs" / "dryrun" / "smoke" / "r" / "gemma" / "cache.jsonl")
    reused = [r for r in c.records.values() if r["condition_id"] == "R-BB"]
    checks.append(("R-BB copies L-BB answers made with the same prompt and image (reuse_from)",
                   rc3 == 0 and len(reused) == 30 and all(r["mode"] == "reuse" for r in reused)))

    # ---- the gold answer never leaves ------------------------------------------
    leaks = 0
    for base in (root / "outputs", Path(tempfile.gettempdir()) / "asset8_tmp"):
        for dirpath, _d, files in os.walk(base):
            for name in files:
                with open(os.path.join(dirpath, name), "rb") as f:
                    leaks += gold.encode() in f.read()
    checks.append(("no request, log or output file holds a gold answer", leaks == 0))
    rec = next(iter(cache_b.records.values()))
    checks.append((f"an output record carries qa_id, condition_id and '{H.ANSWER_FIELD}'",
                   all(k in rec for k in ("qa_id", "condition_id", H.ANSWER_FIELD, "prompt_sha256",
                                          "image_sha256", "settings_sha256", "model_id"))))

    # ---- rules that must stop a run ---------------------------------------------
    checks.append(("stage pilot refuses Batch mode (D140)",
                   expect_stop(H.run, ns(stage="pilot", run_name="", mode="batch", **common))))
    checks.append(("stage main refuses standard mode (D140)",
                   expect_stop(H.run, ns(stage="main", run_name="", mode="standard", **common))))
    checks.append(("stage pilot refuses a probe file that is not PILOT.json",
                   expect_stop(H.run, ns(stage="pilot", run_name="", mode="standard", **common))))
    checks.append(("the machine translation column is refused outside stage smoke (D149)",
                   expect_stop(H.run, ns(stage="pilot", run_name="", mode="standard", conditions="L-EE",
                                         **dict(common, english_column="mt_en")))))
    checks.append(("an English condition without --english-file is refused",
                   expect_stop(H.run, ns(conditions="L-EE", root=str(root), probe=str(probe)))))
    checks.append(("forced choice without --options-file is refused",
                   expect_stop(H.run, ns(conditions="N-FCB", root=str(root), probe=str(probe)))))
    P = promptlib.load_prompts(REPO / "configs" / "prompts.yaml")
    cfg = promptlib.load_conditions(REPO / "configs" / "conditions.yaml", P)
    pr = H.load_probe(probe)
    checks.append(("stage main refuses a subset and a wrong probe file",
                   expect_stop(H.check_probe_rule, "main", cfg, pr, ["L-BB"], True, False)
                   and expect_stop(H.check_probe_rule, "main", cfg, pr, ["L-BB"], False, False)
                   and expect_stop(H.check_probe_rule, "main", cfg, pr, ["N-FCE"], False, False)))
    m = H.load_models(REPO / "configs" / "models.yaml")["gemini"]
    good = H.build_gemini_body(m, "x", "QUJD")
    with contextlib.redirect_stdout(io.StringIO()):
        H.audit_gemini_body(good, m, True)
    bad_bodies = []
    b = json.loads(json.dumps(good)); b["generationConfig"]["temperature"] = 0; bad_bodies.append(b)
    b = json.loads(json.dumps(good)); b["systemInstruction"] = {"parts": [{"text": "s"}]}; bad_bodies.append(b)
    b = json.loads(json.dumps(good)); b["contents"].append({"role": "model", "parts": [{"text": "a"}]}); bad_bodies.append(b)
    b = json.loads(json.dumps(good)); b["generationConfig"]["candidateCount"] = 1; bad_bodies.append(b)
    b = json.loads(json.dumps(good)); b["generationConfig"]["maxOutputTokens"] = 512; bad_bodies.append(b)
    checks.append(("a body with a sampling parameter, a system prompt, a prefilled model turn, "
                   "candidate_count or another output limit is refused",
                   all(expect_stop(H.audit_gemini_body, x, m, True) for x in bad_bodies)))
    checks.append(("the fixed Gemini body: image part first, minimal thinking, 256 tokens, "
                   "MEDIA_RESOLUTION_HIGH",
                   list(good["contents"][0]["parts"][0]) == ["inlineData"]
                   and good["generationConfig"] == {"maxOutputTokens": 256,
                                                    "thinkingConfig": {"thinkingLevel": "minimal"},
                                                    "mediaResolution": "MEDIA_RESOLUTION_HIGH"}))
    w = H.build_work("m", cfg["conditions"]["L-BB"], pr["items"][:3], P, None, None, man)
    cache = H.Cache(root / "guard.jsonl")
    cache.add({"key": w[0]["key"], H.ANSWER_FIELD: "a", "prompt_sha256": "other",
               "image_sha256": w[0]["image_sha256"], "settings_sha256": "s"})
    checks.append(("a cache hit made with another prompt text stops the run (D145)",
                   expect_stop(H.check_hits, cache, w, "s")))
    led = root / "cap.jsonl"
    stub = dict(script="other", call_id="c", model_id="m", mode="standard", input_tokens=0,
                output_tokens=0, thoughts_tokens=0)
    cost_ledger.append(str(led), dict(stub, usd=39.99))
    ctx = H.Ctx(ledger=led, stage_label="x", stage_cap=5.0)
    checks.append(("the 40 USD hard stop holds across scripts",
                   expect_stop(H.check_budget, ctx, 0.02) and not expect_stop(H.check_budget, ctx, 0.005)))
    led2 = root / "cap2.jsonl"
    cost_ledger.append(str(led2), dict(stub, script=H.SCRIPT_NAME, stage="x", usd=0.45))
    ctx2 = H.Ctx(ledger=led2, stage_label="x", stage_cap=0.5)
    checks.append(("the stage cap stops a step that would pass it",
                   expect_stop(H.check_budget, ctx2, 0.06) and not expect_stop(H.check_budget, ctx2, 0.04)))

    # ---- a job on the server that the registry does not know ----------------------
    class _Stranger(H.FakeGemini):
        def __init__(self, display):
            super().__init__()
            self.batches["batches/x-1"] = {"display": display, "file": "none", "polls": 0, "n": 2}
            self.files["none"] = []

    def _submit_with(display):
        t = _Stranger(display)
        cx = H.Ctx(out_dir=root / "stranger" / (display or "nameless"), dry=True, model=m,
                   model_key="gemini", stage_tag="smoke.t", stage_label="smoke/t", stage_cap=1.0,
                   ledger=root / "stranger.jsonl", transport=t, cache=H.Cache(root / "none.jsonl"),
                   prices={"in": 0.375, "out": 1.875}, run_id="r", usd_run=0.0, stats={},
                   img_root=root / "data" / "images_1024", img_manifest=man,
                   tmp_dir=root / "tmp", rawlog=root / "raw.jsonl", prompts=P,
                   settings_sha="s", model_id="m")
        cx.out_dir.mkdir(parents=True, exist_ok=True)
        H.Batch(cx).submit("L-BB", w)
        return t.n_create

    checks.append(("a live job of the same stage, model and condition that the registry does "
                   "not know stops the submit, and so does a live job without a display name",
                   expect_stop(_submit_with, "a8-smoke.t-gemini-L-BB-abcdef12-261006T000000-01")
                   and expect_stop(_submit_with, "")))
    with contextlib.redirect_stdout(io.StringIO()):
        other = _submit_with("somebody-elses-job")
    checks.append(("a job with another display name does not block", other == 1))

    # ---- small parts --------------------------------------------------------------
    op1 = {"name": "batches/1", "metadata": {"state": "BATCH_STATE_SUCCEEDED"}, "response": {"responsesFile": "files/a"}}
    op2 = {"name": "batches/2", "state": "JOB_STATE_RUNNING", "dest": {"fileName": "files/b"}}
    checks.append(("both documented job-status shapes are read",
                   H.job_state(op1) == "SUCCEEDED" and H.results_file_of(op1) == "files/a"
                   and H.job_state(op2) == "RUNNING" and H.results_file_of(op2) == "files/b"))
    s1 = [r["qa_id"] for r in H.select_items(items, sample=10)]
    s2 = [r["qa_id"] for r in H.select_items(list(reversed(items)), sample=10)]
    checks.append(("--sample is seeded: the same 10 items whatever the file order",
                   len(s1) == 10 and set(s1) == set(s2)))
    checks.append(("--where filters on a field", len(H.select_items(items, where=[("qa_type", "count")])) == 6))
    torn = root / "torn.jsonl"
    H.append_jsonl(torn, {"a": 1})
    with open(torn, "a", encoding="utf-8") as f:
        f.write('{"a": 2')                       # a line cut by a crash
    H._checked_tail.discard(str(torn))
    H.append_jsonl(torn, {"a": 3})
    recs, bad = H.read_jsonl(torn)
    checks.append(("a line cut by a crash is skipped and does not damage the next line",
                   [r["a"] for r in recs] == [1, 3] and bad == 1))
    changed = root / "prompts_changed.yaml"
    text = (REPO / "configs" / "prompts.yaml").read_text(encoding="utf-8")
    changed.write_text(text.replace("Do not explain.", "Do not explain!"), encoding="utf-8")
    checks.append(("a changed prompt text without a new lock is refused",
                   expect_stop(promptlib.load_prompts, changed)))
    checks.append(("script check: Bangla, Latin, digits",
                   [H.script_of(t) for t in ("বালতি", "bucket", "৩", "")] == ["bangla", "latin", "digits", "none"]))

    bad = 0
    for name, passed in checks:
        print(("OK   " if passed else "FAIL ") + name)
        bad += 0 if passed else 1
    print()
    print(f"{len(checks) - bad}/{len(checks)} self checks passed")
    if bad and a.verbose:
        for name, text in logs.items():
            print(f"\n===== log of '{name}' =====\n{text}")
    elif bad:
        print("Run again with --verbose and send the output to the ASSET-8 chat.")
    return 0 if not bad else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="ASSET-8: the inference harness (Design 4.9).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, stage=True):
        p.add_argument("--root", default=DEFAULT_ROOT, help="the Drive folder of the project")
        p.add_argument("--repo", default=str(REPO), help="the repo root (configs/ are read from it)")
        if stage:
            p.add_argument("--stage", required=True, choices=("smoke", "pilot", "main"))
            p.add_argument("--run-name", default="", help="stage smoke only: a name for this test")
            p.add_argument("--model", required=True, help="a key of configs/models.yaml: gemini, qwen, gemma")
            p.add_argument("--dry-run", action="store_true",
                           help="no network, no GPU, no cost; writes to outputs/dryrun/")

    p = sub.add_parser("selftest", help="checks without network, GPU or Drive")
    p.add_argument("--verbose", action="store_true")
    p = sub.add_parser("prompts", help="print the prompts and their lock")
    p.add_argument("--repo", default=str(REPO))

    p = sub.add_parser("prepare-images", help="make the shared 1,024 px copies")
    common(p, stage=False)
    p.add_argument("--probe", action="append", help="a probe file; may be given several times")
    p.add_argument("--all", action="store_true", help="every .jpg under data/images/H*/")
    p.add_argument("--images", default="", help="source folder (default <root>/data/images)")
    p.add_argument("--images-1024", default="", help="target folder (default <root>/data/images_1024)")
    p.add_argument("--verify", action="store_true", help="re-hash the copies that already exist")

    p = sub.add_parser("run", help="run conditions for one model")
    common(p)
    p.add_argument("--conditions", required=True,
                   help="comma list of conditions or sets, for example L-BB,L-BB0 or EXP-L")
    p.add_argument("--probe", required=True, help="the probe file (Drive probes/)")
    p.add_argument("--where", action="append", default=[], help="field=value; may be repeated")
    p.add_argument("--ids-file", default="", help="a text file with one qa_id per line")
    p.add_argument("--sample", type=int, default=0, help="keep N items by seeded rank (seed 42)")
    p.add_argument("--limit", type=int, default=0, help="keep the first N items")
    p.add_argument("--mode", choices=("standard", "batch"), default=None, help="Gemini only")
    p.add_argument("--english-file", default="")
    p.add_argument("--english-column", default="", help="the column with the English question")
    p.add_argument("--options-file", default="")
    p.add_argument("--images-1024", default="")
    p.add_argument("--ledger", default="", help="default <root>/outputs/api_cost_ledger.jsonl")
    p.add_argument("--paid-tier-confirmed", action="store_true",
                   help="you checked on the AI Studio Projects page that the key's project "
                        "is on a paid tier (Design 5.5). Needed for every billed run.")
    p.add_argument("--plan-only", action="store_true", help="print the plan and stop")
    p.add_argument("--no-submit", action="store_true",
                   help="Batch mode: poll and collect, submit nothing new")
    p.add_argument("--wait", action="store_true",
                   help="Batch mode: keep polling, collecting and submitting until done")
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--stage-cap-usd", type=float, default=None, help="needs a decision row")
    p.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens")
    p.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens")
    p.add_argument("--allow-env-change", action="store_true", help="needs a decision row (Design 5.3)")
    p.add_argument("--dry-keep", action="store_true", help=argparse.SUPPRESS)

    for name in ("report", "peek"):
        p = sub.add_parser(name)
        common(p)
        p.add_argument("--conditions", default="")
        if name == "peek":
            p.add_argument("--n", type=int, default=10)
            p.add_argument("--i-am-not-in-the-ht1-pair", action="store_true")

    a = ap.parse_args(argv)
    fn = {"selftest": cmd_selftest, "prompts": cmd_prompts, "prepare-images": H.prepare_images,
          "run": H.run, "report": H.report, "peek": H.peek}[a.cmd]
    return fn(a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (H.Stop, promptlib.PromptError, imglib.ImageError) as exc:
        H.say(f"\nSTOP: {exc}")
        sys.exit(1)
    except (cost_ledger.BudgetExceeded, cost_ledger.LedgerError) as exc:
        H.say(f"\nSTOP: {exc}")
        sys.exit(1)
