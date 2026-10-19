#!/usr/bin/env python3
"""
exp_r_memcheck.py
BanglaIndoorVQA, EXP-R training: the D221 memory check. Read-only.

Run once, from the pinned EXP-R commit, on the L4, before seed 42 (notebooks/exp_r.ipynb,
Cell 4). It decides --grad-checkpointing for all three seeds with the train command's own
code (src/finetune.py, TorchEngine) and changes none of it.

  1. Lengths. Every training and validation example is measured the way the train
     command builds it: the build_prompt() text, the image tokens of its prepared image
     (through the harness's own processor call), the gold answer and the end of turn.
     The 8 longest training examples within the 1,536-token cap, the next 8 and the
     examples over the cap are then built with TorchEngine.build(); every length must
     agree, or nothing is measured.
  2. Memory. LoRA is attached as for seed 42. One update at learning rate 0 creates the
     AdamW state, as in training after its first update (the LoRA weights are checked to
     be unchanged). The 8 examples run as single-example training passes, with images,
     their gradients summed as in one weight update: checkpointing off, off again (to see
     run-to-run noise, and as the time reference once the GPU kernels are compiled), then
     on. Peak reserved memory is read for each run.
  3. Verdict (D221). Off if the peak reserved memory with checkpointing off is at most
     20.0 GiB. Otherwise on, if every loss and the summed LoRA gradient of the on run
     match the off run within the tolerance below. Otherwise STOP: the chat decides (D224).

Tolerance, fixed before any run: each example's loss within one bf16 unit, 2^-8 = 0.39%,
relative; the summed LoRA gradient within 1% relative L2 distance, ||on - off|| / ||off||.

It writes nothing in the repo and nothing in Drive; the notebook keeps the verdict from
the last line, "MEMCHECK_RESULT {...}". It copies the training and validation images to
the local image cache of the train command, with the train command's own function.
Printouts: counts, losses, times, memory and hashes. No question, answer or model output.
Exit code: 0 verdict off or on; 3 STOP (D224); 1 error.
"""

import argparse
import gc
import json
import math
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import harness as H            # noqa: E402
import images as imglib        # noqa: E402
import prompts as promptlib    # noqa: E402
import finetune as F           # noqa: E402

VERSION = "exp-r-memcheck 2026-10-19b"
SEED_RUN = "ft-s42"            # LoRA start and dropout seed of the first seed
N_LONGEST = 8                  # D221
LIMIT_GIB = 20.0               # D221
LOSS_RTOL = 2.0 ** -8          # one bf16 unit
GRAD_RTOL = 0.01               # relative L2 distance of the summed LoRA gradient
say = H.say


def gib(n_bytes):
    return n_bytes / 2 ** 30


def count_tokens(tok, texts, batch=512):
    out = []
    for i in range(0, len(texts), batch):
        out += [len(ids) for ids in tok(list(texts[i:i + batch]), add_special_tokens=False)["input_ids"]]
    return out


def grad_vector(engine):
    """The LoRA gradients summed over the passes so far, as one float32 vector on the CPU."""
    torch = engine.torch
    return torch.cat([p.grad.detach().reshape(-1).to("cpu", torch.float32) if p.grad is not None
                      else torch.zeros(p.numel(), dtype=torch.float32) for p in engine.params])


def grad_distance(ref, other):
    a, b = ref.double(), other.double()
    na, nb = float(a.norm()), float(b.norm())
    rel = float((b - a).norm()) / na if na > 0 else math.inf
    cos = float(a.dot(b)) / (na * nb) if na > 0 and nb > 0 else float("nan")
    return rel, cos


def run_passes(engine, exs, seed, snap_after=None):
    """Single-example training passes, as the train command runs them, with the gradients
    of all passes summed. The gradient is kept after snap_after examples (default: after
    the last example that ran). An out-of-memory pass ends the run and is recorded."""
    torch = engine.torch
    engine.set_train(True)
    engine.opt.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rows, oom_at, snap = [], None, None
    for k, ex in enumerate(exs):
        engine.reseed(seed, k)                 # the same dropout draws in every run
        t0 = time.time()
        try:
            nll = engine.train_pass(ex, 1.0 / (ex["n_labels"] * len(exs)))
            engine.sync()
        except H.Stop as exc:                  # TorchEngine turns a CUDA OOM into Stop
            if "out of memory" not in str(exc).lower():
                raise
            oom_at = k
            break
        rows.append({"qa_id": ex["qa_id"], "n_tokens": ex["n_tokens"], "loss": nll / ex["n_labels"],
                     "seconds": round(time.time() - t0, 3),
                     "peak_reserved_gib": round(gib(torch.cuda.max_memory_reserved()), 3)})
        if snap_after is None or len(rows) == snap_after:
            snap = grad_vector(engine)
    out = {"rows": rows, "oom_at": oom_at, "peak_reserved_gib": round(gib(torch.cuda.max_memory_reserved()), 3),
           "peak_allocated_gib": round(gib(torch.cuda.max_memory_allocated()), 3), "grad": snap}
    engine.opt.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    return out


def say_run(name, run):
    say(f"  {name}:")
    for r in run["rows"]:
        say(f"    {r['qa_id']:<16} {r['n_tokens']:>5} tokens   loss {r['loss']:.6f}   {r['seconds']:6.2f} s"
            f"   peak reserved so far {r['peak_reserved_gib']:.2f} GiB")
    if run["oom_at"] is not None:
        say(f"    OUT OF MEMORY in the pass of example {run['oom_at'] + 1}")
    say(f"    peak reserved {run['peak_reserved_gib']:.2f} GiB, peak allocated {run['peak_allocated_gib']:.2f} GiB")


def clean(obj):
    """JSON without NaN or infinity, and without tensors."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items() if k != "grad"}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def main(argv=None):
    ap = argparse.ArgumentParser(description="EXP-R: the D221 memory check (read-only).")
    ap.add_argument("--root", default="/content/drive/MyDrive/BanglaIndoorVQA")
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--image-cache", default="/content/ft_images_1024",
                    help="the local image folder of the train command (its --image-cache default)")
    a = ap.parse_args(argv)
    root, repo, cache = Path(a.root), Path(a.repo), Path(a.image_cache)
    t_start = time.time()
    git = H.git_info(repo)
    say("=" * 78)
    say(f"{VERSION}   the D221 memory check of EXP-R   {H.utc_now()}")
    say(f"code: commit {str(git.get('commit') or '')[:12] or 'unknown'}"
        + ("" if git.get("dirty") is False else "   NOT A CLEAN CHECKOUT"))
    say(f"rule (D221): checkpointing off if the peak reserved memory with it off is at most {LIMIT_GIB} GiB; "
        f"otherwise on, if each loss matches within {LOSS_RTOL:.5f} relative (one bf16 unit) and the summed "
        f"LoRA gradient within {GRAD_RTOL:.0%} relative L2; otherwise STOP (D224)")
    say("=" * 78)
    if git.get("dirty") is not False or not git.get("gate_e"):
        raise H.Stop("The memory check runs only from a clean checkout with the tag gate-E in its history "
                     "(D221). Run Cells 1 and 3 of notebooks/exp_r.ipynb first.")

    say("\n[1] Settings, data and images (the train command's own functions)")
    ft = F.load_ft(repo / "configs" / "finetune.yaml")
    rc = F.run_cfg(ft, SEED_RUN)
    seed, cap = rc["seed"], int(ft["max_seq_tokens"])
    prompts = promptlib.load_prompts(repo / "configs" / "prompts.yaml")
    cfg = promptlib.load_conditions(repo / "configs" / "conditions.yaml", prompts)
    cond = cfg["conditions"][ft["condition"]]
    models = H.load_models(repo / "configs" / "models.yaml")
    d = F.define_data(root / "data" / "banglaindoorvqa.json", root / "data" / "splits.csv",
                      val_n=int(ft["val_n"]), dry_n=int(ft["runs"]["ft-dry"]["n_train"]))
    F.check_data_manifest(d, root / "data" / "ft" / F.DATA_MANIFEST)
    train, val = d["train"], d["val"][:rc["n_val"]]
    say(f"  {len(train)} training examples, {len(val)} validation items; the hashes of the data command agree")
    img_drive = root / "data" / "images_1024"
    manifest = imglib.read_manifest(img_drive)
    needed = F.images_of(train + val)
    lost = [f for f in needed if f not in manifest]
    if lost:
        raise H.Stop(f"{len(lost)} of {len(needed)} images are not prepared (first: {lost[0]}).")
    copied = F.cache_images(needed, img_drive, cache, manifest)
    say(f"  {len(needed)} prepared images; {copied} copied now to {cache} (the train command's local cache)")

    say("\n[2] Model and LoRA, as for seed 42 (TorchEngine of src/finetune.py)")
    engine = F.TorchEngine(models, ft, rc, prompts, cond, cache, manifest,
                           Path(tempfile.mkdtemp(prefix="exp_r_memcheck_")),
                           allow_a100=False, grad_checkpointing=False)
    torch = engine.torch
    env = engine.env()
    say("  environment: " + json.dumps(env))
    lora = engine.attach_lora(seed)
    dry = (H.read_json(root / "models" / "ft-dry" / H.FT_MANIFEST, {}) or {}).get("lora") or {}
    if (lora["n_modules"], lora["trainable_params"]) != (dry.get("n_modules"), dry.get("trainable_params")):
        raise H.Stop(f"LoRA sits on {lora['n_modules']} layers with {lora['trainable_params']:,} weights; the "
                     f"Gate G dry run had {dry.get('n_modules')} and {dry.get('trainable_params')}. Nothing was "
                     "measured. Send this line to the EXP-R chat.")
    total_gib = gib(torch.cuda.get_device_properties(0).total_memory)
    say(f"  LoRA on {lora['n_modules']} linear layers, {lora['trainable_params']:,} trainable weights "
        f"({lora['param_dtype']}), as in the Gate G dry run; PEFT {lora['peft']}")
    say(f"  GPU {env.get('gpu')}: {total_gib:.2f} GiB in all; {gib(torch.cuda.memory_reserved()):.2f} GiB "
        "reserved after the model load")

    say("\n[3] Token length of every training and validation example")
    from PIL import Image
    sizes = {}
    for rel in needed:
        with Image.open(cache / rel) as im:
            sizes[rel] = tuple(im.size)
    items = train + val
    texts = [promptlib.build_prompt(cond, it, prompts)["text"] for it in items]
    n_text = count_tokens(engine.tok, texts)
    n_ans = count_tokens(engine.tok, [it["answer_bn"] for it in items])
    n_end = len(engine.ending_ids)
    img_tokens, overhead = {}, set()
    for s in sorted(set(sizes.values())):
        enc = engine.backend._inputs(texts[0], Image.new("RGB", s, (128, 128, 128)))
        img_tokens[s] = int(engine.backend._count_image_tokens(enc))
        overhead.add(int(enc["input_ids"].shape[-1]) - img_tokens[s] - n_text[0])
    if len(overhead) != 1:
        raise H.Stop(f"The chat template adds {sorted(overhead)} tokens for different image sizes. Nothing "
                     "was measured. Send this line to the EXP-R chat.")
    c = overhead.pop()
    length = [n_text[i] + img_tokens[sizes[it["image_file"]]] + c + n_ans[i] + n_end for i, it in enumerate(items)]
    nt = len(train)
    order = sorted(range(nt), key=lambda i: (-length[i], items[i]["qa_id"]))
    over = [i for i in order if length[i] > cap]
    fit = [i for i in order if length[i] <= cap]
    val_over = sorted((i for i in range(nt, len(items)) if length[i] > cap), key=lambda i: items[i]["qa_id"])
    tl = sorted(length[:nt])
    say(f"  {len(img_tokens)} image sizes: {min(img_tokens.values())} to {max(img_tokens.values())} image tokens; "
        f"the template adds {c} tokens and the end of turn {n_end}")
    say(f"  training examples: longest {tl[-1]} tokens, median {tl[nt // 2]}, mean {sum(tl) / nt:.0f}; over the "
        f"cap of {cap}: {len(over)} (the train command leaves them out and counts them)")
    say(f"  validation items over the cap: {len(val_over)} of {len(val)}")
    if len(fit) < N_LONGEST:
        raise H.Stop(f"Only {len(fit)} training examples fit the cap of {cap} tokens.")

    say(f"\n[4] The {N_LONGEST} longest training examples within the cap, built with TorchEngine.build()")
    pick, nxt = fit[:N_LONGEST], fit[N_LONGEST:2 * N_LONGEST]
    check_ids = pick + nxt + over[:16] + val_over[:8]
    built, wrong = {}, []
    for i in check_ids:
        ex = engine.build(items[i])
        want_img = img_tokens[sizes[items[i]["image_file"]]]
        if ex["n_tokens"] != length[i] or ex["image_tokens"] != want_img:
            wrong.append(f"{items[i]['qa_id']}: computed {length[i]} tokens ({want_img} image), built "
                         f"{ex['n_tokens']} ({ex['image_tokens']} image)")
        if i in pick:
            if ex.get("skip"):
                raise H.Stop(f"{items[i]['qa_id']} fits the cap but the train command leaves it out ({ex['skip']}).")
            built[i] = ex
    if wrong:
        raise H.Stop(f"For {len(wrong)} of {len(check_ids)} examples the computed length differs from the train "
                     f"command's build (first: {wrong[0]}). Nothing was measured. Send this line to the EXP-R chat.")
    say(f"  {len(check_ids)} examples built ({len(pick)} longest, {len(nxt)} next, "
        f"{min(len(over), 16)} over the cap, {min(len(val_over), 8)} validation over the cap): every computed "
        "length equals the built length")
    exs = sorted((built[i] for i in pick), key=lambda ex: (ex["n_tokens"], ex["qa_id"]))
    qtype = {it["qa_id"]: it["qa_type"] for it in items}
    for ex in exs:
        say(f"    {ex['qa_id']:<16} {str(qtype[ex['qa_id']]):<13} {ex['n_tokens']:>5} tokens: prompt {ex['n_prompt']} "
            f"({ex['image_tokens']} image), answer {ex['n_answer']}, end {ex['n_ending']}")

    say("\n[5] The AdamW state, as in training after its first weight update")
    before = engine.trainable_state()
    engine.set_train(True)
    engine.reseed(seed, 10 ** 6)
    engine.train_pass(exs[0], 1.0 / exs[0]["n_labels"])
    engine.step(0.0)                           # learning rate 0: the state is created, no weight moves
    after = engine.trainable_state()
    if set(before) != set(after) or not all(torch.equal(before[n], after[n]) for n in before):
        raise H.Stop("A weight update at learning rate 0 changed the LoRA weights. Nothing was measured. "
                     "Send this line to the EXP-R chat.")
    state_mib = sum(t.numel() * t.element_size() for st in engine.opt.state.values()
                    for t in st.values() if torch.is_tensor(t)) / 2 ** 20
    say(f"  AdamW state on the GPU: {state_mib:.1f} MiB; the LoRA weights are unchanged")
    del before, after

    say(f"\n[6] Training passes on the {N_LONGEST} examples, shortest first (single-example passes, "
        "gradients summed as in one weight update)")
    off = run_passes(engine, exs, seed)
    say_run("checkpointing off", off)
    m = len(off["rows"])
    off2 = None
    if off["oom_at"] is None:
        off2 = run_passes(engine, exs, seed)
        say_run("checkpointing off, again (run-to-run noise only)", off2)
    on, gc_error = None, ""
    try:                                       # the call attach_lora makes for --grad-checkpointing
        engine.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception as exc:
        gc_error = f"{type(exc).__name__}: {str(exc)[:200]}"
    if not gc_error:
        on = run_passes(engine, exs, seed, snap_after=m or None)
        say_run("checkpointing on", on)

    cmp_, noise = {}, {}
    if on is not None and m and len(on["rows"]) >= m and on["grad"] is not None and off["grad"] is not None:
        rel = [abs(on["rows"][k]["loss"] - off["rows"][k]["loss"]) / max(abs(off["rows"][k]["loss"]), 1e-12)
               for k in range(m)]
        g_rel, g_cos = grad_distance(off["grad"], on["grad"])
        cmp_ = {"examples": m, "max_loss_rel_diff": max(rel), "grad_rel_l2": g_rel, "grad_cosine": g_cos,
                "loss_ok": max(rel) <= LOSS_RTOL, "grad_ok": g_rel <= GRAD_RTOL}
    if off2 is not None and off2["oom_at"] is None and off2["grad"] is not None:
        rel2 = [abs(off2["rows"][k]["loss"] - off["rows"][k]["loss"]) / max(abs(off["rows"][k]["loss"]), 1e-12)
                for k in range(m)]
        g2, c2 = grad_distance(off["grad"], off2["grad"])
        noise = {"max_loss_rel_diff": max(rel2), "grad_rel_l2": g2, "grad_cosine": c2}

    say("\n[7] Verdict (D221)")
    if cmp_:
        say(f"  on against off, {m} examples: largest loss difference {cmp_['max_loss_rel_diff']:.2e} relative "
            f"(limit {LOSS_RTOL:.2e}); summed LoRA gradient {cmp_['grad_rel_l2']:.2e} relative L2 (limit "
            f"{GRAD_RTOL:.0e}), cosine {cmp_['grad_cosine']:.8f}")
    if noise:
        say(f"  off against off again (noise only): loss {noise['max_loss_rel_diff']:.2e}, gradient "
            f"{noise['grad_rel_l2']:.2e} relative L2")
    off_txt = ("ran out of memory" if off["oom_at"] is not None
               else f"peak reserved memory {off['peak_reserved_gib']:.2f} GiB")
    if off["oom_at"] is None and off["peak_reserved_gib"] <= LIMIT_GIB:
        verdict, why = "off", f"with checkpointing off: {off_txt}, at most {LIMIT_GIB} GiB"
    elif gc_error:
        verdict, why = "stop", f"with checkpointing off: {off_txt}; it could not be switched on ({gc_error})"
    elif on is None or on["oom_at"] is not None:
        verdict, why = "stop", f"with checkpointing off: {off_txt}; with it on: ran out of memory too"
    elif not cmp_:
        verdict, why = "stop", "no example ran with checkpointing off, so loss and gradients cannot be compared"
    elif cmp_["loss_ok"] and cmp_["grad_ok"]:
        verdict, why = "on", (f"with checkpointing off: {off_txt}, more than {LIMIT_GIB} GiB; with it on: peak reserved "
                              f"memory {on['peak_reserved_gib']:.2f} GiB, and loss and gradients match the off run "
                              "within the tolerance")
    else:
        verdict, why = "stop", (f"with checkpointing off: {off_txt}; with it on, loss or gradients differ from the "
                                "off run beyond the tolerance")
    if verdict == "on" and on["peak_reserved_gib"] > total_gib - 1.0:
        say(f"  WARNING: with checkpointing on the peak is {on['peak_reserved_gib']:.2f} GiB, within 1 GiB of the "
            f"card ({total_gib:.2f} GiB). A longer session may still run out of memory (D221: then D224).")

    rep = H.read_json(root / "models" / "ft-dry" / "gate_g_report.json", {}) or {}
    secs = rep.get("seconds") or {}
    t_tr, t_val = float(secs.get("train_per_example") or "nan"), float(secs.get("val_per_item") or "nan")
    plan = F.make_plan(nt, ft, {"epochs": int(ft["epochs"])})
    ratio = None
    if on is not None and m and len(on["rows"]) >= m:
        # the warm off run is the time reference: the first off run may include first-use kernel compiles
        ref = off2 if off2 is not None and len(off2["rows"]) >= m else off
        ratio = sum(r["seconds"] for r in on["rows"][:m]) / max(1e-9, sum(r["seconds"] for r in ref["rows"][:m]))

    def seed_hours(r):
        return (nt * int(ft["epochs"]) * t_tr * r + len(plan["eval_points"]) * int(ft["val_n"]) * t_val) / 3600

    say(f"  one seed at the Gate G speed: {seed_hours(1.0):.1f} L4 hours without checkpointing"
        + (f"; with it about {seed_hours(ratio):.1f} (a pass took {ratio:.2f} times as long here)" if ratio else ""))
    say("")
    if verdict in ("off", "on"):
        say(f"VERDICT: gradient checkpointing {verdict.upper()} for all three seeds (D221): {why}.")
    else:
        say(f"STOP (D221, D224): {why}. No seed is trained. Send this whole output to the EXP-R chat.")
    result = {
        "helper": VERSION, "utc": H.utc_now(), "commit": git.get("commit"), "verdict": verdict, "why": why,
        "limit_gib": LIMIT_GIB, "tolerance": {"loss_rel": LOSS_RTOL, "grad_rel_l2": GRAD_RTOL},
        "gpu_total_gib": round(total_gib, 3), "env_engine": env,
        "lora": {k: lora[k] for k in ("n_modules", "n_tensors", "trainable_params", "param_dtype", "peft",
                                     "targets_sha256")},
        "examples": [{"qa_id": ex["qa_id"], "qa_type": qtype[ex["qa_id"]], "n_tokens": ex["n_tokens"],
                      "image_tokens": ex["image_tokens"], "n_answer": ex["n_answer"]} for ex in exs],
        "off": off, "off_repeat": off2, "on": on, "compare": cmp_, "noise": noise,
        "lengths": {"image_sizes": len(img_tokens), "image_tokens_min": min(img_tokens.values()),
                    "image_tokens_max": max(img_tokens.values()), "template_tokens": c, "end_tokens": n_end,
                    "train_longest": tl[-1], "train_median": tl[nt // 2], "train_mean": round(sum(tl) / nt, 1)},
        "predicted": {"train_over_cap": len(over), "train_over_cap_ids": [items[i]["qa_id"] for i in over],
                      "val_over_cap": len(val_over), "val_over_cap_ids": [items[i]["qa_id"] for i in val_over]},
        "projection": {"seed_hours_off": seed_hours(1.0), "seed_hours_on": seed_hours(ratio) if ratio else None,
                       "time_ratio_on_off": ratio, "gate_g_seconds": {"train_per_example": t_tr, "val_per_item": t_val}},
        "seconds": round(time.time() - t_start, 1)}
    say("MEMCHECK_RESULT " + json.dumps(clean(result), ensure_ascii=True, allow_nan=False))
    return 0 if verdict in ("off", "on") else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (H.Stop, promptlib.PromptError, imglib.ImageError) as exc:
        H.say(f"\nSTOP: {exc}")
        sys.exit(1)
