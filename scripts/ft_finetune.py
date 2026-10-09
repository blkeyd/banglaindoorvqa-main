#!/usr/bin/env python3
"""
ft_finetune.py
BanglaIndoorVQA, FT scripts: command line of EXP-R training and evaluation
(Design 7.3; DESIGN_ADDENDUM G; decisions D170 to D178).
The logic lives in src/finetune.py. The model is loaded by the harness
(src/harness.py), the prompt comes from build_prompt() (src/prompts.py) and the
images from the one image loader (src/images.py).

Commands (run from the repo root, in Colab)
    selftest   no network, no GPU, no Drive: every line must print OK
    data       define the training and validation data, check them, prepare images
    train      train one run of configs/finetune.yaml (resumable):
                 ft-dry                  the Gate G dry run: 100 examples, 1 epoch
                 ft-s42, ft-s43, ft-s44  the three seeds of EXP-R (after Gate E)
    reload     Gate G: the saved adapter and the base model through the harness
    gate-g     print the Gate G report: checks, seconds per item, projections
    eval       EXP-R evaluation: base model and adapters on R-TEST (harness)
    status     where every run stands

The Gate G dry run, in order
    python scripts/ft_finetune.py selftest
    python scripts/ft_finetune.py data --prepare-images all
    python scripts/ft_finetune.py train --run ft-dry
    python scripts/ft_finetune.py reload --run ft-dry
    python scripts/ft_finetune.py gate-g --run ft-dry

Keys come from the environment (HF_TOKEN), which the notebook fills from Colab
Secrets. Everything a run writes goes to Google Drive only. Answers are model
output: not for the two HT-1 people before gate-A (D150). Counts only are printed.
"""

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
try:
    import harness as H
    import images as imglib
    import prompts as promptlib
    import finetune as F
except ImportError as _exc:  # pragma: no cover
    sys.exit(f"STOP: cannot import the files in src/ ({_exc}). Run this script from the repo, "
             "with normalise.py, cost_ledger.py, prompts.py, images.py, harness.py and "
             "finetune.py in src/.")

DEFAULT_ROOT = "/content/drive/MyDrive/BanglaIndoorVQA"
GOLD = "সোনা"            # marks every made-up gold answer of the self check


def cmd_gate_g(a):
    old = a.base_old or str(Path(a.root) / "outputs" / "smoke" / "t1"
                            / F.load_ft(Path(a.repo) / "configs" / "finetune.yaml")["base_model"]
                            / "cache.jsonl")
    return F.gate_report(a.root, a.repo, a.run, base_old=old)


# --------------------------------------------------------------------------
# Self check: everything that can be checked without a GPU, the network or Drive
# --------------------------------------------------------------------------
def _made_up_project(root):
    """A small made-up dataset in the format of the real one: 4 train houses, 2
    validation houses, 1 test house. No real question or answer is in it."""
    from PIL import Image
    types = ["obj_name", "object", "count", "scene_id", "obj_property", "spatial",
             "obj_function", "assistive"]
    plan = [("H01", "train", 40), ("H02", "train", 40), ("H03", "train", 40), ("H04", "train", 40),
            ("H05", "val", 45), ("H06", "val", 45), ("H07", "test", 30)]
    data, rows, n = [], ["house_id,annotator_id,split,n_images"], 0
    for house, split, n_img in plan:
        rows.append(f"{house},A1,{split},{n_img}")
        (root / "data" / "images" / house).mkdir(parents=True, exist_ok=True)
        for i in range(1, n_img + 1):
            img = f"{house}_{i:04d}"
            Image.new("RGB", (64, 48), (30 * (i % 7), 90, 160)).save(
                root / "data" / "images" / house / f"{img}.jpg", quality=90)
            qa = []
            for k in range(1, 6):
                n += 1
                qa.append({"qa_id": f"{img}_q{k}", "type": types[n % 8],
                           "question": f"পরীক্ষার প্রশ্ন নম্বর {n} কী?", "answer": f"{GOLD}{n}"})
            data.append({"image_id": img, "house_id": house, "annotator_id": "A1",
                         "image_file": f"{house}/{img}.jpg", "image_w": 64, "image_h": 48,
                         "capture_type": "closeup", "room_type": "kitchen", "objects": [],
                         "qa_pairs": qa})
    data[0]["qa_pairs"][0]["question"] = "প্রথম লাইন\nদ্বিতীয় লাইন কী?"       # a line break inside
    data[1]["qa_pairs"][0]["answer"] = "  "                                   # an empty answer
    (root / "data" / "banglaindoorvqa.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    (root / "data" / "splits.csv").write_text("\r\n".join(rows) + "\r\n", encoding="utf-8")
    (root / "probes").mkdir(parents=True, exist_ok=True)
    items = []
    for rec in data:
        if rec["house_id"] != "H07":
            continue
        for q in rec["qa_pairs"][:2]:
            items.append({"qa_id": q["qa_id"], "image_id": rec["image_id"], "house_id": "H07",
                          "capture_type": "closeup", "qa_type": q["type"], "question_bn": q["question"],
                          "answer_bn": q["answer"], "image_file": rec["image_file"], "split": "test"})
    (root / "probes" / "R-TEST.json").write_text(json.dumps(
        {"probe": "R-TEST", "n": len(items), "created_utc": "selftest", "items": items},
        ensure_ascii=False), encoding="utf-8")
    return {"train": (4, 160, 800), "val": (2, 90, 450), "test": (1, 30, 150)}, data, items


def cmd_selftest(a):
    checks, logs = [], {}
    design_3_1 = F.DESIGN_3_1
    root = Path(tempfile.mkdtemp(prefix="ft_selftest_"))
    (root / "data").mkdir()

    def quiet(name, fn, *args, **kw):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                out = fn(*args, **kw)
        except Exception as exc:        # a check below decides whether this was expected
            out = exc
        logs[name] = buf.getvalue() + (f"\n[{type(out).__name__}] {out}" if isinstance(out, Exception) else "")
        return out

    def stops(fn, *args, **kw):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fn(*args, **kw)
        except (H.Stop, promptlib.PromptError, imglib.ImageError):
            return True
        return False

    def targs(**kw):
        base = dict(root=str(root), repo=str(REPO), run="ft-dry", fake=True, fake_opts={},
                    restart=False, image_cache="", allow_a100=False, grad_checkpointing=False,
                    no_prefetch=False, stop_after_minutes=0, keep_state=False,
                    allow_env_change=False, dataset="", rtest="")
        base.update(kw)
        return argparse.Namespace(**base)

    def dargs(**kw):
        base = dict(root=str(root), repo=str(REPO), dataset="", rtest="", prepare_images="dry",
                    verify=False, force=False, repo_list=False)
        base.update(kw)
        return argparse.Namespace(**base)

    try:
        # ---- settings ------------------------------------------------------------
        ft = F.load_ft(REPO / "configs" / "finetune.yaml")
        checks.append(("configs/finetune.yaml holds the values of Design 7.3", True))
        text = (REPO / "configs" / "finetune.yaml").read_text(encoding="utf-8")
        for name, old, new in (("r", "  r: 16", "  r: 8"), ("lr", "lr: 1.0e-4 ", "lr: 2.0e-4 "),
                               ("seed", "ft-s43: {profile: seed, seed: 43}", "ft-s43: {profile: seed, seed: 45}")):
            changed = root / f"finetune_{name}.yaml"
            changed.write_text(text.replace(old, new, 1), encoding="utf-8")
            checks.append((f"a changed design value is refused ({name})",
                           old in text and stops(F.load_ft, changed)))
        models = H.load_models(REPO / "configs" / "models.yaml")
        base = models[ft["base_model"]]
        ads = {k: m for k, m in models.items() if m.get("adapter")}
        checks.append(("every adapter entry inherits the base entry and adds cache_id and adapter",
                       len(ads) == 4 and all(H.base_entry(m) == base for m in ads.values())
                       and len({m["cache_id"] for m in ads.values()}) == 4))
        checks.append(("an adapter has its own settings hash; the base hash does not depend on adapters",
                       len({H.canonical_sha(H.hf_settings(m)) for m in list(ads.values()) + [base]}) == 5
                       and "adapter" not in H.hf_settings(base)))
        ytext = (REPO / "configs" / "models.yaml").read_text(encoding="utf-8")
        y1, y2 = root / "m1.yaml", root / "m2.yaml"
        y1.write_text(ytext.replace('cache_id: "Qwen/Qwen3.5-4B+ft-s43", ', "", 1), encoding="utf-8")
        y2.write_text(ytext.replace("+ft-s43", "+ft-s42", 1), encoding="utf-8")
        checks.append(("an adapter entry without its own cache_id, or with one already used, is refused",
                       stops(H.load_models, y1) and stops(H.load_models, y2)))

        # ---- data ------------------------------------------------------------------
        made, data, probe_items = _made_up_project(root)
        dpath, spath = root / "data" / "banglaindoorvqa.json", root / "data" / "splits.csv"
        checks.append(("numbers that are not those of Design 3.1 stop the run",
                       stops(F.define_data, dpath, spath)))
        F.DESIGN_3_1 = made
        d = F.define_data(dpath, spath)
        checks.append(("splits: houses, images and QA pairs are counted per split",
                       {k: (v["houses"], v["images"], v["qa"]) for k, v in d["stats"].items()} == made))
        first = d["by_id"]["H01_0001_q1"]
        checks.append(("a line break inside a question becomes one space; an empty answer is left out",
                       "\n" not in first["question_bn"] and first["line_break_fixed"]
                       and d["unusable"] == ["H01_0002_q1"] and len(d["train"]) == 799))
        rev = root / "reversed.json"
        rev.write_text(json.dumps(list(reversed(data)), ensure_ascii=False), encoding="utf-8")
        d2 = F.define_data(rev, spath)
        checks.append(("the fixed lists do not depend on the order of the dataset file",
                       [x["qa_id"] for x in d["val"]] == [x["qa_id"] for x in d2["val"]]
                       and [x["qa_id"] for x in d["dry"]] == [x["qa_id"] for x in d2["dry"]]
                       and d["hashes"]["train_ids_sha256"] == d2["hashes"]["train_ids_sha256"]
                       and len(d["val"]) == 400 and len(d["dry"]) == 100
                       and {x["split"] for x in d["val"]} == {"val"}
                       and {x["split"] for x in d["dry"]} == {"train"}))
        bad_probe = [dict(probe_items[0], question_bn="অন্য প্রশ্ন?")] + probe_items[1:]
        checks.append(("R-TEST is found in the test split with the same text; a changed text stops",
                       F.crosscheck_probe(d["by_id"], probe_items) == len(probe_items)
                       and stops(F.crosscheck_probe, d["by_id"], bad_probe)
                       and stops(F.crosscheck_probe, d["by_id"], probe_items, "train")))
        rc1 = quiet("data", F.data_command, dargs())
        man = H.read_json(root / "data" / "ft" / F.DATA_MANIFEST, {})
        vlist = (root / "data" / "ft" / F.VAL_LIST).read_text(encoding="utf-8").split()
        checks.append(("the data command writes the manifest and the validation list and prepares images",
                       rc1 == 0 and man.get("n_train") == 799 and len(vlist) == 400
                       and man["image_coverage"]["dry run"]["prepared"] == man["image_coverage"]["dry run"]["wanted"]
                       and man["image_coverage"]["train"]["prepared"] < 160))
        rc2 = quiet("data_all", F.data_command, dargs(prepare_images="all"))
        man = H.read_json(root / "data" / "ft" / F.DATA_MANIFEST, {})
        rc3 = quiet("data_again", F.data_command, dargs(prepare_images="all"))
        checks.append(("--prepare-images all prepares every image of the dataset, and only once",
                       rc2 == 0 and rc3 == 0
                       and man["image_coverage"]["whole dataset"] == {"wanted": 280, "prepared": 280}
                       and "already 280, prepared 0" in logs["data_again"]))
        other = root / "other.json"
        changed = json.loads(dpath.read_text(encoding="utf-8"))
        changed[5]["qa_pairs"][0]["answer"] = GOLD + "অন্য"
        other.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
        checks.append(("a dataset that changed after the data command is refused (D110)",
                       stops(F.data_command, dargs(dataset=str(other), prepare_images="none"))
                       and stops(F.train_command, targs(dataset=str(other)))))

        # ---- plan --------------------------------------------------------------------
        full = F.make_plan(13668, ft, {"epochs": 2})
        checks.append(("the plan of a seed: 6,834 batches and 855 updates per epoch, 1,710 updates, "
                       "warmup 52, validation after 214, 428, 642, 855, 1069, 1283, 1497, 1710",
                       (full["batches_per_epoch"], full["updates_per_epoch"], full["total_updates"],
                        full["warmup_updates"]) == (6834, 855, 1710, 52)
                       and full["eval_points"] == [214, 428, 642, 855, 1069, 1283, 1497, 1710]))
        lrs = [F.lr_at(u, full) for u in range(full["total_updates"])]
        checks.append(("learning rate: linear warmup to 1e-4, then cosine down, never 0 or above 1e-4",
                       abs(lrs[0] - 1e-4 / 52) < 1e-12 and abs(lrs[51] - 1e-4) < 1e-12
                       and abs(lrs[52] - 1e-4) < 1e-12 and all(x >= y for x, y in zip(lrs[52:], lrs[53:]))
                       and 0 < lrs[-1] < 1e-8 and max(lrs) <= 1e-4 + 1e-15))
        dryplan = F.make_plan(100, ft, {"epochs": 1})
        orders, seen = {}, []
        for u in range(dryplan["total_updates"]):
            _e, _fb, batches = F.group_for(u, dryplan, orders, 42, d["dry"])
            seen += [it["qa_id"] for b in batches for it in b]
        o42 = [it["qa_id"] for it in F.order_for_epoch(42, 0, d["dry"])]
        checks.append(("the dry run: 50 batches, 7 updates, every example once; the order depends "
                       "on the seed and the epoch",
                       (dryplan["batches_per_epoch"], dryplan["total_updates"]) == (50, 7)
                       and dryplan["eval_points"] == [2, 4, 6, 7] and seen == o42 and len(set(seen)) == 100
                       and o42 != [it["qa_id"] for it in F.order_for_epoch(43, 0, d["dry"])]
                       and o42 != [it["qa_id"] for it in F.order_for_epoch(42, 1, d["dry"])]))

        # ---- the end of an answer, and the layers that get LoRA -------------------------
        checks.append(("end of an answer: the template's end of turn, cut at the stop token, or "
                       "followed by it; no stop token is refused",
                       F.ending_rule([7, 8], {7}, 7)[0] == [7] and F.ending_rule([7, 8], {9}, 9)[0] == [7, 8, 9]
                       and F.ending_rule([7, 8], {8, 5}, 3)[0] == [7, 8]
                       and stops(F.ending_rule, [7], {5, 6}, 3) and stops(F.ending_rule, [7], set(), 3)))
        qwen = [(f"model.visual.blocks.{i}.attn.qkv", 8, 8) for i in range(3)] \
            + [("model.visual.merger.mlp.0", 8, 8), ("lm_head", 8, 9)]
        for i in range(4):
            leaves = ["self_attn.q_proj", "self_attn.o_proj"] if i % 2 else ["linear_attn.in_proj_qkv", "linear_attn.out_proj"]
            qwen += [(f"model.language_model.layers.{i}.{x}", 8, 8) for x in leaves + ["mlp.gate_proj", "mlp.down_proj"]]
        sel = F.select_targets(qwen, 4)
        checks.append(("LoRA layers, Qwen-like names: every linear layer of the 4 language layers, "
                       "nothing in the vision encoder, the merger or the output head",
                       len(sel["targets"]) == 16 and all(t.startswith("model.language_model.layers.") for t in sel["targets"])
                       and sel["table"]["vision"]["layers"] == 4 and sel["table"]["head"]["layers"] == 1
                       and sel["leaves"]["mlp.gate_proj"] == 4 and sel["leaves"]["linear_attn.out_proj"] == 2))
        gemma = [(f"model.vision_tower.encoder.layers.{i}.self_attn.q_proj", 8, 8) for i in range(2)] \
            + [(f"model.audio_tower.layers.{i}.fc", 8, 8) for i in range(2)] \
            + [("model.multi_modal_projector.linear", 8, 8), ("model.embed_audio.proj", 8, 8), ("lm_head", 8, 9)] \
            + [(f"model.language_model.layers.{i}.{x}", 8, 8) for i in range(3) for x in ("self_attn.k_proj", "mlp.up_proj")]
        sel2 = F.select_targets(gemma, 3)
        checks.append(("LoRA layers, Gemma-like names: vision, audio and projector layers stay out",
                       len(sel2["targets"]) == 6 and sel2["stack_prefix"] == "model.language_model.layers"
                       and sel2["table"]["audio"]["layers"] == 2 and sel2["table"]["projector"]["layers"] == 2))
        checks.append(("a layer count that does not fit the model config stops the run",
                       stops(F.select_targets, qwen, 6) and stops(F.select_targets, qwen[:5], 4)))

        # ---- the training loop, with the stand-in engine -----------------------------------
        rc = quiet("dry", F.train_command, targs())
        run_dir, smoke = root / "models" / "ft-dry", root / "outputs" / "smoke" / "ft-dry"
        man = H.read_json(run_dir / H.FT_MANIFEST, {})
        batches = F.read_log(run_dir / "train_log.jsonl", "batch", "batch")
        evals = F.read_log(run_dir / "eval_log.jsonl", "eval", "update")
        mem, _b = H.read_jsonl(smoke / "from_memory" / "answers.jsonl")
        checks.append(("dry run end to end: 100 examples in 50 batches, 7 weight updates, validation "
                       "after updates 2, 4, 6, 7, adapter and manifest written",
                       rc == 0 and man.get("complete") and len(batches) == 50
                       and sum(len(b["qa_ids"]) for b in batches) == 100
                       and man["result"]["updates_done"] == 7 and [e["update"] for e in evals] == [2, 4, 6, 7]
                       and H.file_sha256(run_dir / "adapter" / H.ADAPTER_FILE) == man["adapter"]["sha256"]
                       and not list(run_dir.glob("best_update*"))))
        checks.append(("every batch has its loss on the answer and the end of turn only",
                       all(lab == ans + end and tot == pr + lab for b in batches for pr, ans, end, lab, tot
                           in zip(b["n_prompt"], b["n_answer"], b["n_ending"], b["n_labels"], b["n_tokens"]))))
        gg = H.read_json(run_dir / "gate_g.json", {})
        checks.append(("Gate G extras: 20 R-TEST answers from memory, base probe, overfit probe on "
                       "8 examples with different answers",
                       len(mem) == 20 and gg["from_memory"]["n"] == 20 and gg["base_probe"]["n"] == 3
                       and gg["overfit_probe"]["n"] == 8 and gg["overfit_probe"]["both"] == 8))
        cfg = promptlib.load_conditions(REPO / "configs" / "conditions.yaml",
                                        promptlib.load_prompts(REPO / "configs" / "prompts.yaml"))
        P = promptlib.load_prompts(REPO / "configs" / "prompts.yaml")
        prompts_clean = all(GOLD not in promptlib.build_prompt(cfg["conditions"]["L-BB"], it, P)["text"]
                            for it in d["train"])
        leaks = [p.name for p in list(run_dir.rglob("*")) + [root / "data" / "ft" / F.DATA_MANIFEST,
                                                             root / "data" / "ft" / F.VAL_LIST]
                 if p.is_file() and GOLD.encode() in p.read_bytes()]
        checks.append(("no gold answer is in a prompt, a log, a manifest or the adapter folder",
                       prompts_clean and not leaks))
        # resume after a crash gives the same weights as an uninterrupted run
        shutil.copytree(run_dir, root / "keep_dry")
        r1 = quiet("crash", F.train_command, targs(restart=True, fake_opts={"fail_at_update": 4}))
        r2 = quiet("resume", F.train_command, targs())
        man2 = H.read_json(run_dir / H.FT_MANIFEST, {})
        b2 = F.read_log(run_dir / "train_log.jsonl", "batch", "batch")
        checks.append(("a run that died in update 4 resumes after update 2 and ends with the same "
                       "weights and the same 50 batches",
                       isinstance(r1, RuntimeError) and r2 == 0 and "resuming after weight update 2" in logs["resume"]
                       and man2["adapter"]["sha256"] == man["adapter"]["sha256"]
                       and [b["qa_ids"] for b in b2] == [b["qa_ids"] for b in batches]
                       and man2["result"]["sessions"] == 2))
        r3 = quiet("budget", F.train_command, targs(restart=True, stop_after_minutes=1e-9))
        r4 = quiet("budget2", F.train_command, targs())
        man3 = H.read_json(run_dir / H.FT_MANIFEST, {})
        checks.append(("--stop-after-minutes saves and ends with NOT COMPLETE; the next start finishes",
                       r3 == 2 and r4 == 0 and man3["adapter"]["sha256"] == man["adapter"]["sha256"]
                       and man3["result"]["sessions"] >= 2))
        checks.append(("a run that is complete is not trained again",
                       quiet("again", F.train_command, targs()) == 0 and "already COMPLETE" in logs["again"]))
        # an unfinished run made differently is refused
        quiet("part", F.train_command, targs(restart=True, stop_after_minutes=1e-9))
        mpath = run_dir / H.FT_MANIFEST
        mm = H.read_json(mpath, {})
        mm["ft_settings_sha256"] = "other"
        H.write_json(mpath, mm)
        checks.append(("an unfinished run folder made under other settings is refused",
                       stops(F.train_command, targs())))
        # a too long example, and early stopping that is only recorded in the dry run
        long_id = d["dry"][0]["qa_id"]
        r5 = quiet("long", F.train_command, targs(restart=True, fake_opts={
            "long_ids": [long_id], "val_curve": [1.0, 1.1, 1.2, 1.3]}))
        man5 = H.read_json(run_dir / H.FT_MANIFEST, {})
        b5 = F.read_log(run_dir / "train_log.jsonl", "batch", "batch")
        checks.append(("an example over 1,536 tokens is left out and counted; the dry run records "
                       "where early stopping would end it and goes on",
                       r5 == 0 and man5["result"]["skipped"] == {long_id: "too_long"}
                       and sum(len(b["qa_ids"]) for b in b5) == 99 and man5["result"]["would_stop_at"] == 6
                       and man5["result"]["stopped"] is None and man5["result"]["updates_done"] == 7
                       and man5["adapter"]["from_update"] == 2))
        # a seed: early stopping with patience 2, best checkpoint kept
        r6 = quiet("seed", F.train_command, targs(run="ft-s42", fake_opts={
            "val_curve": [1.0, 0.9, 0.95, 0.97, 0.5]}))
        sdir = root / "models" / "ft-s42"
        man6 = H.read_json(sdir / H.FT_MANIFEST, {})
        checks.append(("a seed: all 799 usable training QA, 400 validation items, early stopping "
                       "after 2 evaluations without a better loss, best checkpoint kept",
                       r6 == 0 and man6["data"]["n_train"] == 799 and man6["data"]["n_val"] == 400
                       and man6["plan"]["eval_points"] == [13, 25, 38, 50, 63, 75, 88, 100]
                       and man6["result"]["stopped"]["update"] == 50 and man6["adapter"]["from_update"] == 25
                       and man6["result"]["updates_done"] == 50 and not (sdir / "state").exists()
                       and man6["ft_settings"]["seed"] == 42))

        # ---- the harness: adapter guards that need no GPU -----------------------------------
        shutil.rmtree(run_dir)
        shutil.copytree(root / "keep_dry", run_dir)
        ok = quiet("read", H.read_adapter, root, "smoke", models["qwen_ft_dry"], P)
        checks.append(("the harness accepts the dry-run adapter in stage smoke",
                       isinstance(ok, dict) and ok["sha256"] == man["adapter"]["sha256"]
                       and ok["profile"] == "dry" and ok["n_modules"] == 4))
        checks.append(("the dry-run adapter is refused in stages pilot and main; a seed adapter of "
                       "an unknown run is refused",
                       stops(H.read_adapter, root, "main", models["qwen_ft_dry"], P)
                       and stops(H.read_adapter, root, "pilot", models["qwen_ft_dry"], P)
                       and stops(H.read_adapter, root, "main", models["qwen_ft_s43"], P)))
        P2 = dict(P, templates=dict(P["templates"], open_bn_bn=dict(
            P["templates"]["open_bn_bn"], with_image=P["templates"]["open_bn_bn"]["with_image"] + " ")))
        checks.append(("an adapter is refused under another prompt text, other fixed settings or "
                       "another pinned commit",
                       stops(H.read_adapter, root, "smoke", models["qwen_ft_dry"], P2)
                       and stops(H.read_adapter, root, "smoke", dict(models["qwen_ft_dry"], max_new_tokens=64), P)
                       and stops(H.read_adapter, root, "smoke", dict(models["qwen_ft_dry"], revision="abc"), P)))
        wfile = run_dir / "adapter" / H.ADAPTER_FILE
        keep = wfile.read_bytes()
        wfile.write_bytes(keep + b" ")
        changed_refused = stops(H.read_adapter, root, "smoke", models["qwen_ft_dry"], P)
        wfile.write_bytes(keep)
        mm = H.read_json(mpath, {})
        H.write_json(mpath, dict(mm, complete=False))
        unfinished_refused = stops(H.read_adapter, root, "smoke", models["qwen_ft_dry"], P)
        H.write_json(mpath, mm)
        checks.append(("a changed weights file and an unfinished training are refused",
                       changed_refused and unfinished_refused))
        sys.path.insert(0, str(REPO / "scripts"))
        import asset8_harness as A8
        rc = quiet("h_adapter", H.run, A8.ns(root=str(root), probe=str(root / "probes" / "R-TEST.json"),
                                             model="qwen_ft_dry", conditions="R-BB,R-BB0", sample=20,
                                             run_name="sf"))
        hc = H.Cache(root / "outputs" / "dryrun" / "smoke" / "sf" / "qwen_ft_dry" / "cache.jsonl")
        hrec = list(hc.records.values())
        checks.append(("harness dry run with an adapter entry: its own cache id in every key, its "
                       "own settings hash",
                       rc == 0 and len(hrec) == 40 and all(r["model_id"] == "Qwen/Qwen3.5-4B+ft-dry"
                                                           and r["key"].startswith("Qwen/Qwen3.5-4B+ft-dry|")
                                                           for r in hrec)
                       and {r["settings_sha256"] for r in hrec} == {H.canonical_sha(H.hf_settings(models["qwen_ft_dry"]))}))

        # ---- reload, the Gate G report, eval ---------------------------------------------------
        old_dir = root / "outputs" / "smoke" / "t1" / "qwen"
        old_cache = H.Cache(old_dir / "cache.jsonl")
        for it in probe_items[:6]:
            for cid in ("L-BB", "L-BB0"):
                old_cache.add({"key": f"Qwen/Qwen3.5-4B|{cid}|exp-v1|{it['qa_id']}", "condition_id": cid,
                               "qa_id": it["qa_id"], H.ANSWER_FIELD: "উত্তর", "elapsed_s": 1.0})
        H.write_json(old_dir / "manifest.json", {"runs": [{"dry_run": False, "probe": {"file": "R-TEST.json"}}]})
        H.write_json(old_dir / "stage_state.json", {"settings_sha256": H.canonical_sha(H.hf_settings(base)),
                                                    "env": {"gpu": "x"}})
        rl = quiet("reload", F.reload_command, argparse.Namespace(
            root=str(root), repo=str(REPO), run="ft-dry", base_old="", base_probe="", rtest="",
            harness_dry_run=True))
        base_dry = H.Cache(root / "outputs" / "dryrun" / "smoke" / "ft-base" / "qwen" / "cache.jsonl")
        checks.append(("reload: three harness runs (adapter, base, base on the items of the earlier "
                       "smoke run) are started with arguments the harness accepts",
                       rl == 0 and len(base_dry.records) == 12
                       and len(H.Cache(root / "outputs" / "dryrun" / "smoke" / "ft-dry" / "qwen_ft_dry" / "cache.jsonl").records) == 20))
        ev = quiet("eval", F.eval_command, argparse.Namespace(
            root=str(root), repo=str(REPO), stage="smoke", run_name="ev", models="qwen,qwen_ft_dry",
            conditions="R-BB,R-BB0", probe="", english_file="", english_column="", sample=5,
            plan_only=False, harness_dry_run=True))
        checks.append(("eval: the harness is started once per model version", ev == 0
                       and len(H.Cache(root / "outputs" / "dryrun" / "smoke" / "ev" / "qwen_ft_dry" / "cache.jsonl").records) == 10))

        def harness_side(change=None):
            for sub, akey in (("qwen_ft_dry", man["adapter"]["sha256"]), ("qwen", None)):
                p = smoke / sub / "cache.jsonl"
                if p.exists():
                    p.unlink()
                H._checked_tail.discard(str(p))
                c = H.Cache(p)
                for i, r in enumerate(mem):
                    ans = r[H.ANSWER_FIELD] + ("!" if change is not None and i == change and akey else "")
                    rec = {"key": f"{sub}|R-BB|exp-v1|{r['qa_id']}", "condition_id": "R-BB",
                           "qa_id": r["qa_id"], H.ANSWER_FIELD: ans, "elapsed_s": 1.5}
                    c.add(dict(rec, adapter_sha256=akey) if akey else rec)
            newp = root / "outputs" / "smoke" / "ft-base" / "qwen"
            shutil.rmtree(newp, ignore_errors=True)
            shutil.copytree(old_dir, newp)

        shutil.rmtree(smoke, ignore_errors=True)
        (smoke / "from_memory").mkdir(parents=True)
        for r in mem:
            H.append_jsonl(smoke / "from_memory" / "answers.jsonl", r)
        harness_side()
        g1 = quiet("gate", F.gate_report, root, REPO, "ft-dry", str(old_dir / "cache.jsonl"))
        rep = H.read_json(run_dir / "gate_g_report.json", {})
        checks.append(("Gate G report: G1 to G5 pass on a correct dry run, the base answers are "
                       "compared, seconds per item and projections are printed",
                       g1 == 0 and rep.get("gate_g_passed") and all(rep["verdict"].get(k) for k in ("G1", "G2", "G3", "G4", "G5", "H1", "H2"))
                       and "projection, training" in logs["gate"] and "12 of 12 identical" in logs["gate"]))
        harness_side(change=3)
        g2 = quiet("gate_diff", F.gate_report, root, REPO, "ft-dry", str(old_dir / "cache.jsonl"))
        checks.append(("Gate G report: one reloaded answer that differs fails G3",
                       g2 == 2 and "19 of 20 identical" in logs["gate_diff"]
                       and H.read_json(run_dir / "gate_g_report.json", {})["verdict"]["G3"] is False))
        harness_side()
        tl = run_dir / "train_log.jsonl"
        keep_log = tl.read_text(encoding="utf-8")
        recs, _b = H.read_jsonl(tl)
        tl.write_text("\n".join(json.dumps(dict(r, loss=0.001) if r.get("kind") == "batch" and r["batch"] >= 30 else r,
                                           ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
        g3 = quiet("gate_loss", F.gate_report, root, REPO, "ft-dry", "")
        tl.write_text(keep_log, encoding="utf-8")
        checks.append(("Gate G report: a loss that falls near 0 within 50 steps fails G1",
                       g3 == 2 and H.read_json(run_dir / "gate_g_report.json", {})["verdict"]["G1"] is False))
        cmp_ = F.compare_caches(smoke / "qwen" / "cache.jsonl", smoke / "qwen_ft_dry" / "cache.jsonl")
        checks.append(("compare: two caches are joined on condition and item",
                       cmp_["in_both"] == 20 and cmp_["identical"] == 20))
        quiet("status", F.status_command, argparse.Namespace(root=str(root), repo=str(REPO)))
        checks.append(("status lists the four runs", all(k in logs["status"] for k in ("ft-dry", "ft-s42", "ft-s43", "ft-s44"))
                       and "COMPLETE" in logs["status"] and "not started" in logs["status"]))
    finally:
        F.DESIGN_3_1 = design_3_1

    bad = 0
    for name, passed in checks:
        print(("OK   " if passed else "FAIL ") + name)
        bad += 0 if passed else 1
    print()
    print(f"{len(checks) - bad}/{len(checks)} self checks passed")
    if (bad and a.verbose) or a.show:
        for name, text in logs.items():
            if not a.show or name in a.show.split(","):
                print(f"\n===== log of '{name}' =====\n{text}")
    elif bad:
        print("Run again with --verbose and send the output to the FT chat.")
    return 0 if not bad else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="FT scripts: EXP-R training and evaluation (Design 7.3).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--root", default=DEFAULT_ROOT, help="the Drive folder of the project")
        p.add_argument("--repo", default=str(REPO), help="the repo root (configs/ are read from it)")

    p = sub.add_parser("selftest", help="checks without network, GPU or Drive")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--show", default="", help=argparse.SUPPRESS)

    p = sub.add_parser("data", help="define the data, check them, prepare the images")
    common(p)
    p.add_argument("--dataset", default="", help="default <root>/data/banglaindoorvqa.json")
    p.add_argument("--rtest", default="", help="default <root>/probes/R-TEST.json")
    p.add_argument("--prepare-images", choices=("all", "dry", "none"), default="all",
                   help="all: every image of the dataset (D175). dry: only what the dry run needs")
    p.add_argument("--verify", action="store_true", help="re-hash the copies that already exist")
    p.add_argument("--force", action="store_true", help="accept changed data (needs a decision row)")
    p.set_defaults(repo_list=True)

    p = sub.add_parser("train", help="train one run (resumable)")
    common(p)
    p.add_argument("--run", required=True, help="ft-dry, ft-s42, ft-s43 or ft-s44")
    p.add_argument("--image-cache", default="/content/ft_images_1024" if os.path.isdir("/content") else "",
                   help="local folder for copies of the prepared images; '' reads from Drive")
    p.add_argument("--stop-after-minutes", type=float, default=0,
                   help="save and end after this time; start the same command again to go on")
    p.add_argument("--allow-a100", action="store_true",
                   help="a seed may train on an A100 (Design 5.3). Evaluation stays on the L4")
    p.add_argument("--grad-checkpointing", action="store_true", help="less GPU memory, slower")
    p.add_argument("--no-prefetch", action="store_true",
                   help="prepare examples in the main thread only")
    p.add_argument("--restart", action="store_true", help="dry run only: delete it and start again")
    p.add_argument("--keep-state", action="store_true", help="keep the resume state of a finished seed")
    p.add_argument("--allow-env-change", action="store_true", help="needs a decision row")
    p.add_argument("--dataset", default="")
    p.add_argument("--rtest", default="")
    p.add_argument("--fake", action="store_true", help=argparse.SUPPRESS)

    p = sub.add_parser("reload", help="Gate G: adapter and base model through the harness")
    common(p)
    p.add_argument("--run", default="ft-dry")
    p.add_argument("--base-old", default="",
                   help="cache.jsonl of the earlier Qwen smoke run (default outputs/smoke/t1/<base>/)")
    p.add_argument("--base-probe", default="", help="the probe file of that run, if its manifest is gone")
    p.add_argument("--rtest", default="")
    p.add_argument("--harness-dry-run", action="store_true", help=argparse.SUPPRESS)

    p = sub.add_parser("gate-g", help="print the Gate G report")
    common(p)
    p.add_argument("--run", default="ft-dry")
    p.add_argument("--base-old", default="")

    p = sub.add_parser("eval", help="EXP-R evaluation through the harness")
    common(p)
    p.add_argument("--stage", choices=("smoke", "main"), default="main")
    p.add_argument("--run-name", default="", help="stage smoke only")
    p.add_argument("--models", default="", help="default: the base model and the three seed adapters")
    p.add_argument("--conditions", default="EXP-R", help="R-BB,R-EE,R-BB0 (R-EE needs Gate B)")
    p.add_argument("--probe", default="", help="default <root>/probes/R-TEST.json")
    p.add_argument("--english-file", default="")
    p.add_argument("--english-column", default="")
    p.add_argument("--sample", type=int, default=0, help="stage smoke only")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--harness-dry-run", action="store_true", help=argparse.SUPPRESS)

    p = sub.add_parser("status", help="where every run stands")
    common(p)

    a = ap.parse_args(argv)
    fn = {"selftest": cmd_selftest, "data": F.data_command, "train": F.train_command,
          "reload": F.reload_command, "gate-g": cmd_gate_g, "eval": F.eval_command,
          "status": F.status_command}[a.cmd]
    return fn(a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (H.Stop, promptlib.PromptError, imglib.ImageError) as exc:
        H.say(f"\nSTOP: {exc}")
        sys.exit(1)
