"""
harness.py
BanglaIndoorVQA, ASSET-8: the inference harness (Design 4.9, 5, 6, 7;
DESIGN_ADDENDUM D1 and D5; decisions D140 to D150 in DESIGN_ADDENDUM E).

What is in this file
    1. Reading the configs, a probe file, the English questions and the option sets.
    2. The per-call cache, keyed by (model_id, condition_id, prompt_version, qa_id).
    3. Gemini, standard mode (generateContent): used for the pilot.
    4. Gemini, Batch mode (batchGenerateContent): used for EXP-L and EXP-N.
    5. The open models through Hugging Face transformers (greedy): Qwen3.5, Gemma 4.
       5b. The LoRA adapters of EXP-R: loaded on top of an open model (D174).
    6. Counting reports that never show a question or an answer.

Rules this file keeps
    - One normalise(), one cost ledger, one build_prompt(), one image loader: all
      imported, none copied.
    - The gold answer is never read and never sent.
    - No sampling parameter, no system prompt and no prefilled model turn is sent to
      Gemini (D126). Every request body is checked before it leaves.
    - A reply the model gave is final, even if it is empty or cut. Only transport
      errors and server errors are tried again (D143).
    - Everything written here is model output. It goes to Google Drive only.

The command line is scripts/asset8_harness.py.
"""

import base64
import copy
import csv
import hashlib
import importlib
import json
import os
import random
import subprocess
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from normalise import normalise
import cost_ledger
import images as imglib
import prompts as promptlib

HARNESS_VERSION = "2026-10-07a"
SCRIPT_NAME = "asset8_harness"          # the 'script' value of this harness in the ledger
ANSWER_FIELD = "answer_text"            # the field of an output record that holds the answer
API_BASE = "https://generativelanguage.googleapis.com/v1beta"
UPLOAD_BASE = "https://generativelanguage.googleapis.com/upload/v1beta"
DOWNLOAD_BASE = "https://generativelanguage.googleapis.com/download/v1beta"
ITEM_FIELDS = ("qa_id", "image_id", "house_id", "capture_type", "qa_type", "question_bn",
               "image_file")
MT_COLUMN = "mt_en"                     # the stage 1 machine translation column (D4)
SEED = 42


class Stop(Exception):
    """A condition that must stop the run. The message tells the user why."""


def say(msg=""):
    print(msg, flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_hex(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha(obj):
    return sha256_hex(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


_checked_tail = set()


def append_jsonl(path, record):
    """Append one JSON line and flush it to disk. A torn last line is closed first."""
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path not in _checked_tail:
        _checked_tail.add(path)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                torn = f.read(1) != b"\n"
            if torn:
                with open(path, "a", encoding="utf-8") as f:
                    f.write("\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    """Return (records, n_bad_lines). A line that is not JSON is counted, not used."""
    records, bad = [], 0
    if not os.path.exists(str(path)):
        return records, bad
    with open(str(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(rec, dict):
                records.append(rec)
            else:
                bad += 1
    return records, bad


def write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, str(path))


def read_json(path, default):
    if not os.path.exists(str(path)):
        return default
    with open(str(path), "r", encoding="utf-8") as f:
        return json.load(f)


def file_sha256(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Ctx:
    """Everything one run needs, as plain attributes."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


# --------------------------------------------------------------------------
# 1. Configs, probe files, English questions, option sets
# --------------------------------------------------------------------------
def load_models(path):
    import yaml
    with open(str(path), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    models = data.get("models") or {}
    owner = {}
    for key, m in models.items():
        if m.get("backend") not in ("gemini", "hf") or not m.get("model_id"):
            raise Stop(f"{path}: model {key} needs backend (gemini or hf) and model_id.")
        if m.get("adapter") is not None:
            ad = m["adapter"]
            if m["backend"] != "hf" or not isinstance(ad, dict) or not str(ad.get("run") or "").strip():
                raise Stop(f"{path}: model {key}: 'adapter' needs backend hf and a run name, "
                           "for example  adapter: {run: ft-s42}.")
            if not m.get("cache_id") or m["cache_id"] == m["model_id"]:
                raise Stop(f"{path}: model {key} has an adapter, so it needs its own cache_id. "
                           "Without one its answers would share cache keys with the base "
                           "model (DESIGN_ADDENDUM E5).")
        cid = m.get("cache_id") or m["model_id"]
        if cid in owner:
            raise Stop(f"{path}: models {owner[cid]} and {key} would both write answers under "
                       f"the cache id '{cid}'. Give each its own cache_id.")
        owner[cid] = key
    return models


def load_probe(path):
    """Read an ASSET-5 probe file. Every such file keeps the P-CORE.json record format."""
    path = str(path)
    if not os.path.exists(path):
        raise Stop(f"Missing probe file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        raise Stop(f"{path}: no 'items' list. Is this an ASSET-5 output file?")
    if data.get("n") is not None and data.get("n") != len(items):
        raise Stop(f"{path}: header n={data.get('n')} but {len(items)} items.")
    for field in ITEM_FIELDS:
        missing = sum(1 for r in items if not str(r.get(field) or "").strip())
        if missing:
            raise Stop(f"{path}: {missing} items lack the field '{field}'.")
    ids = [r["qa_id"] for r in items]
    if len(ids) != len(set(ids)):
        raise Stop(f"{path}: a qa_id occurs twice.")
    return {"path": path, "file": os.path.basename(path), "items": items,
            "created_utc": data.get("created_utc", ""),
            "qa_ids_sha256": sha256_hex("\n".join(ids))}


def select_items(items, where=(), ids=None, sample=0, limit=0):
    """Filter in file order. --sample N keeps the N lowest seeded ranks (seed 42)."""
    out = list(items)
    for key, val in where:
        if not any(key in r for r in out):
            raise Stop(f"--where {key}=...: no item has a field named '{key}'.")
        out = [r for r in out if str(r.get(key)) == val]
    if ids is not None:
        out = [r for r in out if r["qa_id"] in ids]
    if sample and sample < len(out):
        ranked = sorted(out, key=lambda r: sha256_hex(f"{SEED}|asset8-sample|{r['qa_id']}"))
        keep = {r["qa_id"] for r in ranked[:sample]}
        out = [r for r in out if r["qa_id"] in keep]
    if limit and limit < len(out):
        out = out[:limit]
    return out


def load_english(path, column):
    """English questions (ASSET-2). Join rule of DESIGN_ADDENDUM D4:
    the row whose question_bn_norm equals normalise(question_bn) of the item."""
    path = str(path)
    if not os.path.exists(path):
        raise Stop(f"Missing English questions file: {path}")
    table = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if "question_bn_norm" not in cols or column not in cols:
            raise Stop(f"{path}: needs the columns 'question_bn_norm' and '{column}'. "
                       f"It has {len(cols)} columns; give the English column with "
                       "--english-column.")
        for row in reader:
            norm = row["question_bn_norm"]
            en = " ".join((row[column] or "").split())
            if norm != normalise(norm):
                raise Stop(f"{path}: a question_bn_norm value is not in normalise() form.")
            if norm in table and table[norm] != en:
                raise Stop(f"{path}: one question_bn_norm has two different English strings.")
            if en:
                table[norm] = en
    return {"path": path, "column": column, "table": table, "sha256": file_sha256(path)}


def load_options(path):
    """Forced-choice option sets (ASSET-11). The harness reads one record per item and
    option language: {"qa_id": ..., "lang": "bn" or "en", "options": [four texts in the
    order shown]}. A .jsonl file holds one record per line; a .json file holds a list,
    or an object with an "items" list. Other keys are ignored. The harness never
    builds, shuffles or judges an option set."""
    path = str(path)
    if not os.path.exists(path):
        raise Stop(f"Missing options file: {path}")
    if path.endswith(".jsonl"):
        recs, bad = read_jsonl(path)
        if bad:
            raise Stop(f"{path}: {bad} lines are not JSON.")
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        recs = data.get("items") if isinstance(data, dict) else data
    table = {}
    for r in recs or []:
        opts = r.get("options") if isinstance(r, dict) else None
        if not r.get("qa_id") or r.get("lang") not in ("bn", "en") or \
                not isinstance(opts, list) or len(opts) != 4:
            raise Stop(f"{path}: every record needs qa_id, lang (bn or en) and four options.")
        if (r["qa_id"], r["lang"]) in table:
            raise Stop(f"{path}: two option sets for one item and language.")
        table[(r["qa_id"], r["lang"])] = [str(o) for o in opts]
    return {"path": path, "table": table, "sha256": file_sha256(path)}


def cache_key(model_id, condition_id, prompt_version, qa_id):
    return f"{model_id}|{condition_id}|{prompt_version}|{qa_id}"


def build_work(model_id, cond, items, prompts, english, options, img_manifest):
    """One work entry per item: the exact prompt and the image the model will get."""
    work, no_en, no_opt, no_img = [], [], [], []
    for it in items:
        q_en = None
        if cond["question_lang"] == "en":
            q_en = english["table"].get(normalise(it["question_bn"])) if english else None
            if not q_en:
                no_en.append(it["qa_id"])
                continue
        opts = None
        if cond["forced_choice"]:
            opts = options["table"].get((it["qa_id"], cond["options_lang"])) if options else None
            if not opts:
                no_opt.append(it["qa_id"])
                continue
        p = promptlib.build_prompt(cond, it, prompts, question_en=q_en, options=opts)
        row = img_manifest.get(it["image_file"]) if p["has_image"] else None
        if p["has_image"] and row is None:
            no_img.append(it["image_file"])
            continue
        work.append({
            "key": cache_key(model_id, cond["condition_id"], prompts["prompt_version"], it["qa_id"]),
            "qa_id": it["qa_id"], "house_id": it["house_id"], "qa_type": it["qa_type"],
            "condition_id": cond["condition_id"], "question_lang": cond["question_lang"],
            "answer_lang": cond["answer_lang"], "forced_choice": cond["forced_choice"],
            "prompt_text": p["text"], "prompt_sha256": p["sha256"], "has_image": p["has_image"],
            "image_file": it["image_file"] if p["has_image"] else "",
            "image_sha256": row["sha256"] if row else "",
            "image_bytes": int(row["bytes"]) if row else 0,
        })
    cid = cond["condition_id"]
    if no_en:
        raise Stop(f"{cid}: {len(no_en)} items have no English question in the file given "
                   f"with --english-file (first ids: {', '.join(no_en[:5])}). English-question "
                   "conditions wait for ASSET-2 (Gate B).")
    if no_opt:
        raise Stop(f"{cid}: {len(no_opt)} items have no option set in the file given with "
                   f"--options-file (first ids: {', '.join(no_opt[:5])}). Forced choice "
                   "waits for ASSET-11 and HT-6.")
    if no_img:
        raise Stop(f"{cid}: {len(set(no_img))} images are not prepared yet (first: "
                   f"{sorted(set(no_img))[0]}). Run the prepare-images command for this "
                   "probe file first.")
    return work


# --------------------------------------------------------------------------
# 2. Cache, stage state, Git, budget
# --------------------------------------------------------------------------
class Cache:
    """cache.jsonl: one line per answered key. The first answer for a key stands."""

    def __init__(self, path):
        self.path = str(path)
        self.records = {}
        self.duplicates = 0
        recs, self.bad_lines = read_jsonl(self.path)
        for rec in recs:
            k = rec.get("key")
            if not k or ANSWER_FIELD not in rec:
                self.bad_lines += 1
            elif k in self.records:
                self.duplicates += 1
            else:
                self.records[k] = rec

    def add(self, rec):
        if rec["key"] in self.records:
            raise Stop(f"Internal error: a second answer for {rec['key']}.")
        append_jsonl(self.path, rec)
        self.records[rec["key"]] = rec


def check_hits(cache, work, settings_sha):
    """A cache hit must have been made with the same prompt, image and settings."""
    wrong = Counter()
    first = {}
    for w in work:
        rec = cache.records.get(w["key"])
        if rec is None:
            continue
        for field, now in (("prompt_sha256", w["prompt_sha256"]),
                           ("image_sha256", w["image_sha256"]),
                           ("settings_sha256", settings_sha)):
            if rec.get(field) != now:
                wrong[field] += 1
                first.setdefault(field, w["qa_id"])
    if wrong:
        detail = "; ".join(f"{n} answers with another {f.replace('_sha256', '')} "
                           f"(first: {first[f]})" for f, n in wrong.items())
        raise Stop("The cache holds answers that were made with something else than this "
                   f"run would send: {detail}. A changed question text or template needs a "
                   "new prompt_version; changed settings or images need a decision row. "
                   "Nothing was sent.")


def check_state(ctx, env=None):
    """stage_state.json fixes, for one stage and model: the model, the settings, the
    prompt text of each prompt version and (open models) the environment."""
    path = ctx.out_dir / "stage_state.json"
    state = read_json(path, {})
    pv, sha = ctx.prompts["prompt_version"], ctx.prompts["sha256"]
    problems = []
    if state.get("model_id") not in (None, ctx.model_id):
        problems.append(f"model was {state['model_id']}, now {ctx.model_id}")
    if state.get("settings_sha256") not in (None, ctx.settings_sha):
        problems.append("the fixed settings changed (configs/models.yaml)")
    if (state.get("prompt_versions") or {}).get(pv, sha) != sha:
        problems.append(f"prompt text changed under the same version {pv}")
    if problems:
        raise Stop(f"This stage already holds runs made differently: {'; '.join(problems)}. "
                   f"State file: {path}. One stage and model must use one setting. "
                   "Nothing was sent.")
    if env is not None:
        old = state.get("env")
        diff = [k for k in sorted(set(env) | set(old or {})) if (old or {}).get(k) != env.get(k)]
        if old and diff and not ctx.allow_env_change:
            raise Stop("The environment differs from the earlier runs of this stage and "
                       f"model in: {', '.join(diff)}. Design 5.3: never split one model's "
                       "runs across platforms. If this is intended, write a decision row "
                       "and add --allow-env-change. Earlier: "
                       + json.dumps({k: old.get(k) for k in diff}) + "  Now: "
                       + json.dumps({k: env.get(k) for k in diff}))
        state["env"] = env if not old or ctx.allow_env_change else old
        if old and diff:
            state.setdefault("env_changes", []).append({"utc": utc_now(), "now": env})
    state.update({"model_id": ctx.model_id, "settings_sha256": ctx.settings_sha,
                  "settings": ctx.settings})
    state.setdefault("prompt_versions", {})[pv] = sha
    state.setdefault("created_utc", utc_now())
    write_json(path, state)
    return state


def git_info(repo_root):
    def run(*a):
        try:
            r = subprocess.run(["git", "-C", str(repo_root)] + list(a), capture_output=True,
                               text=True, timeout=30)
            return r.returncode, r.stdout.strip()
        except Exception:
            return None, ""
    code, commit = run("rev-parse", "HEAD")
    if code != 0:
        return {"commit": "", "dirty": None, "gate_e": None}
    _c, status = run("status", "--porcelain")
    code_e, _o = run("merge-base", "--is-ancestor", "gate-E", "HEAD")
    return {"commit": commit, "dirty": bool(status), "gate_e": code_e == 0}


def stage_spend(ledger_path, stage_label):
    total = 0.0
    for rec in cost_ledger.read_lines(str(ledger_path)):
        if rec.get("script") == SCRIPT_NAME and rec.get("stage") == stage_label:
            total += float(rec["usd"])
    return total


def check_budget(ctx, next_max_usd):
    """The 40 USD hard stop of the project (D128), then this stage's own cap."""
    cost_ledger.assert_room(str(ctx.ledger), next_call_max_usd=next_max_usd)
    spent = stage_spend(ctx.ledger, ctx.stage_label)
    if spent + next_max_usd > ctx.stage_cap:
        raise cost_ledger.BudgetExceeded(
            f"STAGE STOP: stage '{ctx.stage_label}' has {spent:.4f} USD in the ledger "
            f"(open Batch reservations included) and the next step could add up to "
            f"{next_max_usd:.4f} USD. The stage cap is {ctx.stage_cap:.2f} USD "
            "(configs/conditions.yaml). Ask the team lead before raising it "
            "(--stage-cap-usd, with a decision row).")


def ledger_line(ctx, call_id, mode, usd, usage=None, note="", **extra):
    u = usage or {}
    rec = {"ts_utc": utc_now(), "script": SCRIPT_NAME, "run_id": ctx.run_id,
           "stage": ctx.stage_label, "call_id": call_id, "model_id": ctx.model["model_id"],
           "mode": mode, "input_tokens": int(u.get("input_tokens", 0)),
           "output_tokens": int(u.get("output_tokens", 0)),
           "thoughts_tokens": int(u.get("thoughts_tokens", 0)),
           "billed_output_tokens": int(u.get("billed_output_tokens", 0)),
           "price_in_per_m": ctx.prices["in"], "price_out_per_m": ctx.prices["out"],
           "usd": usd, "note": note}
    rec.update(extra)
    cost_ledger.append(str(ctx.ledger), rec)
    ctx.usd_run += usd


# --------------------------------------------------------------------------
# Output records and the counting report
# --------------------------------------------------------------------------
def base_record(ctx, w, mode):
    return {"key": w["key"], "model_key": ctx.model_key, "model_id": ctx.model_id,
            "condition_id": w["condition_id"], "prompt_version": ctx.prompts["prompt_version"],
            "qa_id": w["qa_id"], "house_id": w["house_id"], "qa_type": w["qa_type"],
            "question_lang": w["question_lang"], "answer_lang": w["answer_lang"],
            "forced_choice": w["forced_choice"], "has_image": w["has_image"],
            "image_file": w["image_file"], "image_sha256": w["image_sha256"],
            "prompt_sha256": w["prompt_sha256"], "settings_sha256": ctx.settings_sha,
            "stage": ctx.stage_label, "mode": mode, "run_id": ctx.run_id,
            "harness_version": HARNESS_VERSION}


def finish_record(rec, answer_text, **fields):
    answer_text = answer_text if isinstance(answer_text, str) else ""
    rec[ANSWER_FIELD] = answer_text
    rec["is_empty"] = normalise(answer_text) == ""
    rec["n_words"] = len(answer_text.split())
    rec.update(fields)
    rec["ts_utc"] = utc_now()
    return rec


def script_of(text):
    """Bangla or Latin letters in the majority. An engineering check only: answer
    language and fidelity are computed by the scorer (Design 8.2)."""
    bn = sum(1 for ch in text if "\u0980" <= ch <= "\u09ff"
             and not ("\u09e6" <= ch <= "\u09ef") and ch not in "\u0964\u0965")
    la = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    if bn + la == 0:
        return "digits" if any(ch.isdigit() for ch in text) else "none"
    return "bangla" if bn * 2 >= bn + la else "latin"


def report_table(records, expected=None):
    """Counts per condition. Never prints a question or an answer."""
    by = {}
    for r in records:
        by.setdefault(r["condition_id"], []).append(r)
    lines = []
    head = (f"  {'condition':<8} {'answers':>7} {'of':>6} {'empty':>5} {'blocked':>7} "
            f"{'cut':>4} {'<=8w':>6} {'script ok':>9} {'think>0':>7} {'in tok':>7} "
            f"{'img tok':>7} {'USD':>8}")
    lines.append(head)
    totals = Counter()
    for cid in sorted(by, key=lambda c: (c[0], c)):
        rs = by[cid]
        n = len(rs)
        empty = sum(1 for r in rs if r.get("is_empty"))
        blocked = sum(1 for r in rs if r.get("block_reason"))
        cut = sum(1 for r in rs if r.get("truncated"))
        short = sum(1 for r in rs if r.get("n_words", 0) <= 8)
        want = {"bn": "bangla", "en": "latin"}[rs[0]["answer_lang"]]
        sc = Counter(script_of(r.get(ANSWER_FIELD, "")) for r in rs)
        ok = sc[want] + sc["digits"]
        think = sum(1 for r in rs if (r.get("thoughts_tokens") or 0) > 0)
        intok = [r["input_tokens"] for r in rs if r.get("input_tokens")]
        imgtok = [r["image_tokens"] for r in rs if r.get("image_tokens")]
        usd = sum(float(r.get("usd") or 0) for r in rs)
        exp = (expected or {}).get(cid)
        lines.append(
            f"  {cid:<8} {n:>7} {(str(exp) if exp is not None else '-'):>6} {empty:>5} "
            f"{blocked:>7} {cut:>4} {100 * short / n:>5.1f}% {100 * ok / n:>8.1f}% "
            f"{think:>7} {(sum(intok) / len(intok) if intok else 0):>7.0f} "
            f"{(sum(imgtok) / len(imgtok) if imgtok else 0):>7.0f} {usd:>8.4f}")
        totals.update({"n": n, "empty": empty, "leak": sum(1 for r in rs if r.get("thinking_leak"))})
    n = totals["n"]
    if n:
        lines.append(f"  all: {n} answers; non-empty {n - totals['empty']} "
                     f"({100 * (n - totals['empty']) / n:.1f}%); answers with thinking "
                     f"markers in the text: {totals['leak']} (must be 0)")
    return lines


# --------------------------------------------------------------------------
# 3. Gemini: request body, reply parsing, transport
# --------------------------------------------------------------------------
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
FATAL_HINTS = {
    400: "The API refused the request body. Send the message below to the ASSET-8 chat; "
         "do not change a setting yourself.",
    401: "The API key was refused. Check the Colab secret GEMINI_API_KEY.",
    402: "No credit: the prepaid balance is 0. Add credit in AI Studio (Billing). "
         "Never switch to a free-tier key.",
    403: "Permission denied for this key or project.",
    404: "The model or the resource was not found for this key. Do not change the model "
         "yourself: Design 5.5 fixes the fallback order and needs a decision row.",
}


def gemini_settings(m):
    return {"backend": "gemini", "api": "generateContent, REST v1beta",
            "model_id": m["model_id"], "thinking_level": m["thinking_level"],
            "max_output_tokens": int(m["max_output_tokens"]),
            "media_resolution": m.get("media_resolution") or "not sent",
            "sampling_parameters": "none sent (D126)", "system_prompt": "none",
            "turns": "one user turn: image part, then text part",
            "image_prep": imglib.PREP_VERSION}


def build_gemini_body(m, prompt_text, image_b64=None):
    """The one request body, used by the standard path and by the Batch path."""
    parts = []
    if image_b64 is not None:
        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": image_b64}})
    parts.append({"text": prompt_text})
    gen = {"maxOutputTokens": int(m["max_output_tokens"]),
           "thinkingConfig": {"thinkingLevel": m["thinking_level"]}}
    if image_b64 is not None and m.get("media_resolution"):
        gen["mediaResolution"] = m["media_resolution"]
    return {"contents": [{"role": "user", "parts": parts}], "generationConfig": gen}


def audit_gemini_body(body, m, has_image):
    """Stop unless the body is exactly what the fixed settings allow (Design 5.2, D126)."""
    def bad(why):
        raise Stop(f"Internal error: a Gemini request body breaks the fixed settings ({why}). "
                   "Nothing was sent.")
    if set(body) != {"contents", "generationConfig"}:
        bad("a field other than contents and generationConfig")
    turns = body["contents"]
    if len(turns) != 1 or turns[0].get("role") != "user":
        bad("not exactly one user turn")
    parts = turns[0]["parts"]
    kinds = ["image" if "inlineData" in p else "text" if "text" in p else "?" for p in parts]
    if kinds != (["image", "text"] if has_image else ["text"]):
        bad("the parts are not [image, text] or [text]")
    if not str(parts[-1]["text"]).strip():
        bad("empty text")
    if has_image and (parts[0]["inlineData"].get("mimeType") != "image/jpeg"
                      or not parts[0]["inlineData"].get("data")):
        bad("the image part is not a JPEG")
    want = {"maxOutputTokens": int(m["max_output_tokens"]),
            "thinkingConfig": {"thinkingLevel": m["thinking_level"]}}
    if has_image and m.get("media_resolution"):
        want["mediaResolution"] = m["media_resolution"]
    if body["generationConfig"] != want:
        bad("generationConfig differs from configs/models.yaml")


def log_body(m, w):
    """The body as it goes into the logs: the image bytes are replaced by their hash."""
    return build_gemini_body(m, w["prompt_text"],
                             f"<jpeg sha256:{w['image_sha256']}>" if w["has_image"] else None)


def api_error_message(text):
    try:
        return str(json.loads(text).get("error", {}).get("message", ""))[:500]
    except Exception:
        return (text or "")[:300]


def parse_usage(resp):
    """Token counts of one reply. Billed output = visible output + thinking."""
    u = resp.get("usageMetadata") or {}
    prompt = int(u.get("promptTokenCount") or 0)
    cand = int(u.get("candidatesTokenCount") or 0)
    thoughts = int(u.get("thoughtsTokenCount") or 0)
    total = int(u.get("totalTokenCount") or 0)
    billed = total - prompt if total >= prompt and total > 0 else cand + thoughts
    image = sum(int(d.get("tokenCount") or 0) for d in (u.get("promptTokensDetails") or [])
                if str(d.get("modality", "")).upper() == "IMAGE")
    return {"input_tokens": prompt, "output_tokens": cand, "thoughts_tokens": thoughts,
            "billed_output_tokens": billed, "image_tokens": image, "total_tokens": total,
            "reported": bool(u)}


def extract_text(resp):
    """Return (text, finish_reason, block_reason) of a generateContent reply."""
    block = str((resp.get("promptFeedback") or {}).get("blockReason", "") or "")
    cands = resp.get("candidates") or []
    if not cands:
        return "", "", block or "NO_CANDIDATE"
    parts = (cands[0].get("content") or {}).get("parts") or []
    text = "".join(str(p.get("text", "")) for p in parts if not p.get("thought"))
    return text, str(cands[0].get("finishReason", "")), block


def worst_usage(m, w):
    """An upper bound on the tokens of one request, used before the real count is known."""
    b = m["bounds"]
    tokens_in = int(len(w["prompt_text"]) * float(b["text_tokens_per_char"])) \
        + int(b["text_tokens_extra"]) + (int(b["image_tokens"]) if w["has_image"] else 0)
    return {"input_tokens": tokens_in, "output_tokens": int(m["max_output_tokens"]),
            "thoughts_tokens": 0, "billed_output_tokens": int(m["max_output_tokens"]),
            "image_tokens": 0, "total_tokens": 0, "reported": False}


def usage_usd(usage, prices):
    return cost_ledger.cost_usd(usage["input_tokens"], usage["billed_output_tokens"],
                                prices["in"], prices["out"])


def gemini_record(ctx, w, resp, usage, usd, mode, call_id, **extra):
    text, finish, block = extract_text(resp)
    return finish_record(
        base_record(ctx, w, mode), text, finish_reason=finish, block_reason=block,
        truncated=(finish == "MAX_TOKENS"), input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"], thoughts_tokens=usage["thoughts_tokens"],
        billed_output_tokens=usage["billed_output_tokens"], image_tokens=usage["image_tokens"],
        usage_reported=usage["reported"], usd=round(usd, 8),
        model_version=str(resp.get("modelVersion", "")),
        response_id=str(resp.get("responseId", "")), call_id=call_id, **extra)


class TransportProblem(Exception):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind            # "timeout" or "connection"


class GeminiHTTP:
    """The Gemini Developer API over REST. The key travels in a header only."""

    dry = False

    def __init__(self, api_key, timeout_s=120):
        import requests  # in Colab by default
        self._rq = requests
        self._s = requests.Session()
        self._key = api_key
        self._h = {"x-goog-api-key": api_key}
        self.timeout = timeout_s
        self.version = getattr(requests, "__version__", "")

    def scrub(self, text):
        return (text or "").replace(self._key, "***")

    def _send(self, method, url, headers=None, data=None, timeout=None):
        h = dict(self._h)
        h.update(headers or {})
        try:
            r = self._s.request(method, url, headers=h, data=data,
                                timeout=timeout or self.timeout)
        except self._rq.exceptions.Timeout as exc:
            raise TransportProblem("timeout", self.scrub(str(exc)))
        except self._rq.exceptions.RequestException as exc:
            raise TransportProblem("connection", self.scrub(str(exc)))
        return (r.status_code, {k.lower(): v for k, v in r.headers.items()},
                self.scrub(r.content.decode("utf-8", errors="replace")))

    def _json(self, method, url, payload=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self._send(method, url, {"Content-Type": "application/json"}, data)

    def get_model(self, model_id):
        return self._json("GET", f"{API_BASE}/models/{model_id}")

    def generate(self, model_id, body):
        return self._json("POST", f"{API_BASE}/models/{model_id}:generateContent", body)

    def upload_file(self, path, display_name, mime):
        """File API, resumable upload in one piece (Batch API guide, REST example)."""
        size = os.path.getsize(path)
        meta = json.dumps({"file": {"display_name": display_name}}).encode("utf-8")
        status, h, text = 0, {}, ""
        for content_type in ("application/jsonl", "application/json"):
            status, h, text = self._send("POST", f"{UPLOAD_BASE}/files", {
                "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(size),
                "X-Goog-Upload-Header-Content-Type": mime, "Content-Type": content_type}, meta)
            if status == 200 and h.get("x-goog-upload-url"):
                break
        url = h.get("x-goog-upload-url")
        if status != 200 or not url:
            return status or 0, h, text or "no upload URL in the reply"
        with open(path, "rb") as f:
            return self._send("POST", url, {"Content-Length": str(size),
                                            "X-Goog-Upload-Offset": "0",
                                            "X-Goog-Upload-Command": "upload, finalize"},
                              f, timeout=max(self.timeout, 1800))

    def batch_create(self, model_id, display_name, file_name):
        body = {"batch": {"display_name": display_name,
                          "input_config": {"file_name": file_name}}}
        return self._json("POST", f"{API_BASE}/models/{model_id}:batchGenerateContent", body)

    def batch_get(self, name):
        return self._json("GET", f"{API_BASE}/{name}")

    def batch_list(self, page_token=""):
        url = f"{API_BASE}/batches?pageSize=100"
        return self._json("GET", url + (f"&pageToken={page_token}" if page_token else ""))

    def download(self, file_name, dst_path):
        url = f"{DOWNLOAD_BASE}/{file_name}:download?alt=media"
        try:
            with self._s.get(url, headers=self._h, stream=True,
                             timeout=max(self.timeout, 600)) as r:
                if r.status_code != 200:
                    return r.status_code, {}, self.scrub(r.text[:2000])
                with open(dst_path, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
        except self._rq.exceptions.Timeout as exc:
            raise TransportProblem("timeout", self.scrub(str(exc)))
        except self._rq.exceptions.RequestException as exc:
            raise TransportProblem("connection", self.scrub(str(exc)))
        return 200, {}, ""

    def file_delete(self, file_name):
        return self._json("DELETE", f"{API_BASE}/{file_name}")


def gemini_preflight(ctx, need_batch):
    """Everything that must hold before a billed Gemini call. Returns the model info."""
    if ctx.dry:
        return {"name": "models/" + ctx.model["model_id"], "displayName": "DRY RUN (no network)"}
    m = ctx.model
    if not ctx.paid_tier_confirmed:
        raise Stop("Billed run refused: add --paid-tier-confirmed after you have checked, on "
                   "the AI Studio Projects page, that the project of this key shows a paid "
                   "tier. gemini-3.6-flash also has a free tier, and free-tier content is "
                   "used to improve Google's products (Design 5.5).")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if today > str(m["prices"]["valid_until"]) and not ctx.price_override:
        raise Stop(f"The prices in configs/models.yaml were valid until "
                   f"{m['prices']['valid_until']}. Read "
                   "https://ai.google.dev/gemini-api/docs/pricing and pass --price-in and "
                   "--price-out for the mode you run.")
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise Stop("GEMINI_API_KEY is not set. In the notebook, before this command: "
                   "os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')")
    ctx.transport = GeminiHTTP(key, int(m["standard"]["timeout_s"]))
    total = cost_ledger.assert_room(str(ctx.ledger), 0.0)
    say(f"  prices used ({ctx.mode}): {ctx.prices['in']} USD per 1M input, {ctx.prices['out']} "
        f"USD per 1M output (thinking included). Source: {ctx.price_source}")
    say(f"  ledger: {ctx.ledger}   project total {total:.4f} USD of "
        f"{cost_ledger.HARD_CAP_USD:.0f}; stage '{ctx.stage_label}' "
        f"{stage_spend(ctx.ledger, ctx.stage_label):.4f} USD of {ctx.stage_cap:.2f}")
    try:
        status, _h, text = ctx.transport.get_model(m["model_id"])
    except TransportProblem as exc:
        raise Stop(f"Model check could not reach the Gemini API ({exc.kind}). Check the "
                   "network and run again. Nothing was spent.")
    if status != 200:
        raise Stop(f"Model check failed with HTTP {status}. {FATAL_HINTS.get(status, '')} "
                   f"API message: {api_error_message(text)}")
    try:
        info = json.loads(text)
    except ValueError:
        raise Stop("Model check: HTTP 200 without JSON. Run again. Nothing was spent.")
    methods = info.get("supportedGenerationMethods") or []
    say(f"  model check: {info.get('name')} is served (displayName {info.get('displayName')}, "
        f"version {info.get('version')})")
    if need_batch and methods and "batchGenerateContent" not in methods:
        raise Stop(f"The API does not list batchGenerateContent for {m['model_id']} "
                   f"(it lists: {', '.join(methods)}). Nothing was submitted.")
    return info


def fetch_standard(ctx, call_id, w, body, sleep):
    """One generateContent call with retries. Returns the parsed reply, or None."""
    m = ctx.model
    worst = worst_usage(m, w)
    bound = usage_usd(worst, ctx.prices)
    base = {"run_id": ctx.run_id, "call_id": call_id, "key": w["key"], "qa_id": w["qa_id"],
            "condition_id": w["condition_id"], "mode": "standard", "dry_run": ctx.dry}
    ids = {"condition_id": w["condition_id"], "qa_id": w["qa_id"]}
    for attempt in range(1, int(m["standard"]["max_attempts"]) + 1):
        check_budget(ctx, bound)
        t0 = time.time()
        try:
            status, headers, text = ctx.transport.generate(m["model_id"], body)
        except TransportProblem as exc:
            ctx.stats["http_" + exc.kind] += 1
            note = ""
            if exc.kind == "timeout":
                # The server may have answered and billed: keep the meter an upper bound.
                ledger_line(ctx, call_id, "standard", bound, worst,
                            note="timeout: billed amount unknown, worst case charged", **ids)
                note = "worst case charged"
            append_jsonl(ctx.rawlog, dict(base, kind="transport_problem", ts_utc=utc_now(),
                                          attempt=attempt, problem=exc.kind,
                                          message=str(exc)[:500], note=note,
                                          request=log_body(m, w)))
            wait = min(60, 2 ** attempt) + random.uniform(0, 1)
            say(f"    {call_id} attempt {attempt}: {exc.kind}, wait {wait:.0f} s")
            sleep(wait)
            continue
        append_jsonl(ctx.rawlog, dict(base, kind="response" if status == 200 else "http_error",
                                      ts_utc=utc_now(), attempt=attempt, http_status=status,
                                      elapsed_s=round(time.time() - t0, 2),
                                      request=log_body(m, w), response_text=text))
        if status == 200:
            try:
                resp = json.loads(text)
            except ValueError:
                resp = None
            if isinstance(resp, dict):
                return resp
            ctx.stats["http_200_not_json"] += 1
            ledger_line(ctx, call_id, "standard", bound, worst,
                        note="HTTP 200 without JSON: worst case charged", **ids)
            sleep(min(60, 2 ** attempt))
            continue
        if status in RETRY_STATUS:
            ctx.stats[f"http_{status}"] += 1
            wait = min(60, 2 ** attempt) + random.uniform(0, 1)
            ra = headers.get("retry-after")
            if ra and str(ra).isdigit():
                wait = max(wait, min(120, int(ra)))
            say(f"    {call_id} attempt {attempt}: HTTP {status}, wait {wait:.0f} s")
            sleep(wait)
            continue
        raise Stop(f"HTTP {status} from the Gemini API on {call_id}. "
                   f"{FATAL_HINTS.get(status, '')} API message: {api_error_message(text)}")
    return None


def run_gemini_standard(ctx, work):
    """Standard mode (Design 12.1: the pilot). One call per item, in file order."""
    m = ctx.model
    todo = [w for w in work if w["key"] not in ctx.cache.records]
    sleep = (lambda s: None) if ctx.dry else time.sleep
    gave_up_in_a_row = 0
    for n, w in enumerate(todo, 1):
        ctx.seq += 1
        call_id = f"{ctx.run_id}-c{ctx.seq:05d}"
        image_b64 = None
        if w["has_image"]:
            data, _sha = imglib.load_prepared(ctx.img_root, w["image_file"], ctx.img_manifest)
            image_b64 = base64.b64encode(data).decode("ascii")
        body = build_gemini_body(m, w["prompt_text"], image_b64)
        audit_gemini_body(body, m, w["has_image"])
        resp = fetch_standard(ctx, call_id, w, body, sleep)
        if resp is None:
            ctx.stats["calls_gave_up"] += 1
            gave_up_in_a_row += 1
            if gave_up_in_a_row >= 3:
                raise Stop("Three calls in a row got no answer. The API is not reachable or "
                           "a quota is exhausted (a repeated HTTP 429 can mean a free-tier "
                           "quota: check the tier). Try later; the cache keeps what is done.")
            continue
        gave_up_in_a_row = 0
        usage = parse_usage(resp)
        note = ""
        if not usage["reported"] and not ctx.dry:
            usage, note = worst_usage(m, w), "no usageMetadata: worst case charged"
            ctx.stats["usage_missing"] += 1
        usd = usage_usd(usage, ctx.prices)
        # Ledger first, then the cache: a crash in between can never undercount spend.
        ledger_line(ctx, call_id, "standard", usd, usage, note=note,
                    condition_id=w["condition_id"], qa_id=w["qa_id"])
        ctx.cache.add(gemini_record(ctx, w, resp, usage, usd, "standard", call_id))
        ctx.stats["answered"] += 1
        if n % 25 == 0 or n == len(todo):
            say(f"    {n} of {len(todo)} calls done; ledger change in this run "
                f"{ctx.usd_run:.4f} USD")
        sleep(float(m["standard"]["pause_s"]))


# --------------------------------------------------------------------------
# 4. Gemini, Batch mode (decision D142)
# --------------------------------------------------------------------------
TERMINAL = ("collected", "create_failed", "not_on_server", "closed_failed")
DEAD_STATES = ("FAILED", "CANCELLED", "EXPIRED")
W_META = ("key", "qa_id", "house_id", "qa_type", "condition_id", "question_lang",
          "answer_lang", "forced_choice", "prompt_sha256", "has_image", "image_file",
          "image_sha256", "image_bytes")


def deep_find(obj, key):
    """The first value stored under `key` in a nested JSON value, nearest level first.
    Google documents the job status in two shapes; this reads both."""
    queue = [obj]
    while queue:
        cur = queue.pop(0)
        if isinstance(cur, dict):
            if key in cur:
                return cur[key]
            queue.extend(cur.values())
        elif isinstance(cur, list):
            queue.extend(cur)
    return None


def job_state(op):
    s = deep_find(op, "state")
    if isinstance(s, str) and "_STATE_" in s:       # JOB_STATE_x or BATCH_STATE_x
        return s.split("_STATE_", 1)[1]
    return "UNKNOWN"


def results_file_of(op):
    name = deep_find(op, "responsesFile")
    if not name:
        dest = deep_find(op, "dest")
        name = dest.get("fileName") if isinstance(dest, dict) else None
    return name if isinstance(name, str) and name else None


class Batch:
    """The job registry (batch/jobs.jsonl) and the steps submit, poll, collect.

    A job is never submitted twice because:
      1. an item that has an answer, or sits in an open job, is never planned again;
      2. the API's own job list is read first, and a live job for the same stage, model
         and condition that this registry does not know stops the submit;
      3. the line 'create_sent' is on disk before the create call goes out;
      4. the create call is never retried. If its outcome is unknown, the next pass
         looks the job up by its display name and adopts it or declares it absent.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.cfg = ctx.model["batch"]
        self.dir = ctx.out_dir / "batch"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.dir / "jobs.jsonl"
        self.jobs = {}
        recs, self.bad_lines = read_jsonl(self.events_path)
        for ev in recs:
            self._apply(ev)
        self.sleep = (lambda s: None) if ctx.dry else time.sleep
        self._server = None
        self.submitted_now = 0

    # ---- registry ---------------------------------------------------------
    def _apply(self, ev):
        j = self.jobs.setdefault(ev["job_id"], {"job_id": ev["job_id"], "status": "", "state": ""})
        for k, v in ev.items():
            if k not in ("event", "ts_utc", "job_id"):
                j[k] = v
        if ev["event"] != "state":
            j["status"] = ev["event"]
        if ev["event"] == "create_sent":
            j["create_ts"] = ev.get("ts_utc", "")
        j["last_ts"] = ev.get("ts_utc", "")

    def _event(self, job_id, event, **fields):
        ev = dict(fields, job_id=job_id, event=event, ts_utc=utc_now())
        append_jsonl(self.events_path, ev)
        self._apply(ev)

    def open_jobs(self):
        return [j for j in self.jobs.values() if j["status"] not in TERMINAL]

    def flight_keys(self):
        keys = set()
        for j in self.open_jobs():
            keys.update(j.get("keys") or [])
        return keys

    def attempts(self):
        c = Counter()
        for j in self.jobs.values():
            if j["status"] in ("collected", "closed_failed"):
                c.update(j.get("keys") or [])
        return c

    # ---- ledger -----------------------------------------------------------
    def _ledger_ids(self):
        return {r.get("call_id") for r in cost_ledger.read_lines(str(self.ctx.ledger))}

    def _release(self, j, note):
        ids = self._ledger_ids()
        jid = j["job_id"]
        if jid + ":reserve" in ids and jid + ":release" not in ids:
            ledger_line(self.ctx, jid + ":release", "batch-release", -float(j["bound_usd"]),
                        note=note, job_id=jid)

    # ---- API calls that are safe to repeat --------------------------------
    def _call(self, fn, what):
        last = ""
        for attempt in range(1, 5):
            try:
                status, h, text = fn()
            except TransportProblem as exc:
                last = f"{exc.kind}: {str(exc)[:200]}"
                self.sleep(min(60, 2 ** attempt))
                continue
            if status in RETRY_STATUS:
                last = f"HTTP {status}: {api_error_message(text)}"
                self.sleep(min(60, 2 ** attempt))
                continue
            return status, h, text
        raise Stop(f"The Gemini API did not answer the {what} request after 4 tries ({last}). "
                   "Nothing was submitted in this step. Run the same command again later.")

    def _list_server(self):
        """Every batch job the API lists for this key: (name, display name, state)."""
        if self._server is not None:
            return self._server
        out, token = [], ""
        for _page in range(30):
            status, _h, text = self._call(lambda: self.ctx.transport.batch_list(token), "list")
            try:
                data = json.loads(text) if status == 200 else None
            except ValueError:
                data = None
            if not isinstance(data, dict):
                raise Stop(f"The list of batch jobs could not be read (HTTP {status}: "
                           f"{api_error_message(text)}). Without it a double submit cannot be "
                           "ruled out, so nothing was submitted.")
            for op in data.get("operations") or data.get("batches") or []:
                out.append((str(op.get("name", "")), str(deep_find(op, "displayName") or ""),
                            job_state(op)))
            token = data.get("nextPageToken") or ""
            if not token:
                break
        self._server = out
        return out

    def _unknown_live(self, prefix):
        """Live jobs on the server that this registry does not know: those whose display
        name starts with `prefix`, and those the API lists without any display name
        (they cannot be told apart, so they count as possible doubles)."""
        names = {j.get("batch_name") for j in self.jobs.values() if j.get("batch_name")}
        return [(n, d, s) for n, d, s in self._list_server()
                if s not in DEAD_STATES and n not in names and d not in self.jobs
                and (not d or d.startswith(prefix))]

    # ---- step 1: settle jobs whose create call has no known outcome --------
    def reconcile(self):
        self._server = None
        waiting = 0
        for j in list(self.open_jobs()):
            if j.get("batch_name"):
                continue
            jid = j["job_id"]
            on_server = {d: n for n, d, _s in self._list_server()}
            if jid in on_server:
                self._event(jid, "adopted", batch_name=on_server[jid])
                say(f"    {jid}: found on the server as {on_server[jid]}; adopted, not resubmitted")
                continue
            age = 1e9
            nameless = [x for x in self._unknown_live("\0") if not x[1]]
            if j.get("create_ts") and nameless:
                raise Stop(f"The create call of {jid} got no answer, and the API lists a live "
                           f"batch job without a display name ({nameless[0][0]}, "
                           f"{nameless[0][2]}). It may be this job. It is NOT submitted again. "
                           "Look at the Batch page of AI Studio and bring both names to the "
                           "ASSET-8 chat.")
            if j.get("create_ts"):
                sent = datetime.strptime(j["create_ts"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - sent).total_seconds()
            if not self.ctx.dry and age < float(self.cfg["unknown_wait_s"]):
                waiting += 1
                say(f"    {jid}: the create call has no known outcome yet; waiting "
                    f"{float(self.cfg['unknown_wait_s']) - age:.0f} s more before deciding")
                continue
            self._event(jid, "not_on_server")
            self._release(self.jobs[jid], "job is not on the server: reservation released")
            say(f"    {jid}: not on the server; its items are free again")
        return waiting

    # ---- step 2: status of open jobs; collect the finished ones ------------
    def poll(self):
        for j in list(self.open_jobs()):
            name = j.get("batch_name")
            if not name:
                continue
            status, _h, text = self._call(lambda: self.ctx.transport.batch_get(name), "status")
            try:
                op = json.loads(text) if status == 200 else None
            except ValueError:
                op = None
            if not isinstance(op, dict):
                say(f"    {j['job_id']}: status not readable (HTTP {status}); will ask again")
                continue
            state = job_state(op)
            if state != j.get("state"):
                self._event(j["job_id"], "state", state=state,
                            batch_stats=deep_find(op, "batchStats") or {})
                write_json(self.dir / f"{j['job_id']}.status.json", op)
            if state == "SUCCEEDED":
                self._collect(j, op, "collected")
            elif state in DEAD_STATES:
                self._close_dead(j, op, state)

    def _collect(self, j, op, final_event):
        ctx, m, jid = self.ctx, self.ctx.model, j["job_id"]
        fname = results_file_of(op)
        if not fname:
            raise Stop(f"Job {jid} ended as {j.get('state')} but its status names no results "
                       f"file. The raw status is in {self.dir / (jid + '.status.json')}. Do "
                       "not submit again; bring that file to the ASSET-8 chat.")
        dst = self.dir / f"{jid}.results.jsonl"
        if not dst.exists():
            part = str(dst) + ".part"
            status, _h, text = self._call(lambda: ctx.transport.download(fname, part), "download")
            if status != 200:
                raise Stop(f"The results of job {jid} could not be downloaded (HTTP {status}: "
                           f"{text[:300]}). Google keeps them for weeks: run the same "
                           "command again later. Nothing is lost and nothing is resubmitted.")
            os.replace(part, dst)
        lines, bad = read_jsonl(dst)
        light, _bad = read_jsonl(self.dir / j["light_file"])
        meta = {r["key"]: dict(r["meta"], prompt_text=r["request"]["contents"][0]["parts"][-1]["text"])
                for r in light}
        by_key = {}
        for line in lines:
            k = line.get("key") or (line.get("metadata") or {}).get("key")
            if k:
                by_key[str(k)] = line
        if not by_key and len(lines) == len(j["keys"]):
            by_key = dict(zip(j["keys"], lines))   # the API writes replies in input order
        tot = Counter()
        usd_actual = 0.0
        n = Counter()
        for key in j["keys"]:
            line, w = by_key.get(key), meta.get(key)
            if line is None or w is None:
                n["missing"] += 1
                continue
            resp = line.get("response")
            if not isinstance(resp, dict):
                n["error"] += 1
                append_jsonl(ctx.rawlog, {"kind": "batch_error", "ts_utc": utc_now(),
                                          "job_id": jid, "key": key, "qa_id": w["qa_id"],
                                          "condition_id": w["condition_id"],
                                          "error": line.get("error")})
                continue
            usage = parse_usage(resp)
            if not usage["reported"]:
                usage = worst_usage(m, w)
                n["no_usage"] += 1
            usd = usage_usd(usage, ctx.prices)
            usd_actual += usd
            for k2 in ("input_tokens", "output_tokens", "thoughts_tokens", "billed_output_tokens"):
                tot[k2] += usage[k2]
            n["answered"] += 1
            if key in ctx.cache.records:
                n["already_cached"] += 1
                continue
            ctx.cache.add(gemini_record(ctx, w, resp, usage, usd, "batch", f"{jid}#{w['qa_id']}",
                                        job_id=jid, batch_name=j.get("batch_name", "")))
            n["new"] += 1
        # The reservation stays in the ledger until the real cost is written.
        if jid + ":actual" not in self._ledger_ids():
            ledger_line(ctx, jid + ":actual", "batch", usd_actual, dict(tot),
                        note=f"{n['answered']} answered requests; {n['error']} error lines and "
                             f"{n['missing']} missing lines charged 0; {n['no_usage']} replies "
                             "without usage data charged at the worst case",
                        job_id=jid, n_requests=len(j["keys"]))
        self._release(j, "job collected: the reservation is replaced by the real cost")
        self._event(jid, final_event, n_new=n["new"], n_answered=n["answered"],
                    n_error=n["error"], n_missing=n["missing"], usd_actual=round(usd_actual, 6),
                    results_file=dst.name, bad_lines=bad)
        ctx.stats["answered"] += n["new"]
        ctx.stats["batch_error_lines"] += n["error"] + n["missing"]
        say(f"    {jid}: collected. {n['new']} new answers, {n['error']} error lines, "
            f"{n['missing']} missing; real cost {usd_actual:.4f} USD (reserved "
            f"{float(j['bound_usd']):.4f})")
        if self.cfg.get("delete_input_file_after_collect") and j.get("file_name"):
            try:        # tidy up: the uploaded input file holds the images
                ctx.transport.file_delete(j["file_name"])
            except Exception:
                pass

    def _close_dead(self, j, op, state):
        jid = j["job_id"]
        if results_file_of(op):
            self._collect(j, op, "closed_failed")
            return
        stats = deep_find(op, "batchStats")
        # Numbers that are 0 are left out of the JSON, so a present batchStats without
        # successfulRequestCount means 0 successful requests.
        succ = int(stats.get("successfulRequestCount") or 0) if isinstance(stats, dict) and stats else None
        released = succ == 0
        if released:
            self._release(j, f"job ended as {state} with 0 successful requests")
        self._event(jid, "closed_failed", final_state=state, released=released,
                    error=deep_find(op, "error"))
        say(f"    {jid}: ended as {state}. Its items are free again. "
            + ("The reservation was released." if released else
               f"Its reservation of {float(j['bound_usd']):.4f} USD STAYS in the ledger, "
               "because the cost of this job is not known. Compare with the AI Studio "
               "billing page and bring the numbers to the ASSET-8 chat."))

    # ---- step 3: plan and submit ------------------------------------------
    def plan_and_submit(self, works_by_cond):
        out = Counter()
        cfg = self.cfg
        for cond_id, works in works_by_cond.items():
            flight, tries = self.flight_keys(), self.attempts()
            todo = [w for w in works if w["key"] not in self.ctx.cache.records
                    and w["key"] not in flight]
            out["given_up"] += sum(1 for w in todo if tries[w["key"]] >= int(cfg["max_attempts_per_item"]))
            todo = [w for w in todo if tries[w["key"]] < int(cfg["max_attempts_per_item"])]
            chunks, chunk, size = [], [], 0
            for w in todo:
                wsize = int(w["image_bytes"] * 4 / 3) + 6 * len(w["prompt_text"]) + 400
                if chunk and (len(chunk) >= int(cfg["max_requests_per_job"])
                              or size + wsize > float(cfg["max_file_mb"]) * 1e6):
                    chunks.append(chunk)
                    chunk, size = [], 0
                chunk.append(w)
                size += wsize
            if chunk:
                chunks.append(chunk)
            for ch in chunks:
                result = self.submit(cond_id, ch)
                out[result] += 1
                if result in ("queue_full", "unknown"):
                    return out          # stop here; the next pass goes on
        return out

    def submit(self, cond_id, works):
        ctx, m, cfg = self.ctx, self.ctx.model, self.cfg
        keys = [w["key"] for w in works]
        prefix = f"a8-{ctx.stage_tag}-{ctx.model_key}-{cond_id}-"
        job_id = (prefix + sha256_hex("\n".join(sorted(keys)))[:8] + "-"
                  + datetime.now(timezone.utc).strftime("%y%m%dT%H%M%S") + f"-{len(self.jobs) + 1:02d}")
        est = sum(int(cfg["est_tokens_image_request"]) if w["has_image"]
                  else int(cfg["est_tokens_text_request"]) for w in works)
        queued = sum(int(j.get("est_tokens") or 0) for j in self.open_jobs())
        if queued and queued + est > float(cfg["max_enqueued_tokens"]) * float(cfg["enqueue_safety"]):
            say(f"    {cond_id}: {len(keys)} requests (about {est:,} tokens) wait: about "
                f"{queued:,} tokens are queued and the limit is {int(cfg['max_enqueued_tokens']):,}")
            return "queue_full"
        # (a) local registry: none of these items may be in an open job
        if set(keys) & self.flight_keys():
            raise Stop("Internal error: an item of the new job is already in an open job.")
        # (b) the API's own list: no live job for this stage, model and condition that
        #     this registry does not know
        strangers = self._unknown_live(prefix)
        if strangers:
            raise Stop(f"A batch job exists on the server that this folder does not know, and "
                       f"it may hold items of {cond_id}: {strangers[0][1] or 'no display name'} "
                       f"({strangers[0][0]}, {strangers[0][2]}). It "
                       "was probably submitted from another session whose files have not "
                       "reached this Drive mount yet. Do NOT submit again. Wait a few minutes "
                       "and run the same command with --no-submit. (In stage smoke you can "
                       "also start a new --run-name.)")
        # (c) write the input file, check it, and put a ceiling on its cost
        tmp_dir = Path(ctx.tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        heavy = tmp_dir / f"{job_id}.jsonl"
        light = self.dir / f"{job_id}.input_light.jsonl"
        bound = 0.0
        with open(heavy, "w", encoding="utf-8") as fh, open(light, "w", encoding="utf-8") as fl:
            for w in works:
                image_b64 = None
                if w["has_image"]:
                    data, _sha = imglib.load_prepared(ctx.img_root, w["image_file"], ctx.img_manifest)
                    image_b64 = base64.b64encode(data).decode("ascii")
                body = build_gemini_body(m, w["prompt_text"], image_b64)
                audit_gemini_body(body, m, w["has_image"])
                fh.write(json.dumps({"key": w["key"], "request": body}) + "\n")
                fl.write(json.dumps({"key": w["key"], "meta": {k: w[k] for k in W_META},
                                     "request": log_body(m, w)}, ensure_ascii=False) + "\n")
                bound += usage_usd(worst_usage(m, w), ctx.prices)
            bound = round(bound, 6)     # the reservation and its release are this one number
            for f in (fh, fl):
                f.flush()
                os.fsync(f.fileno())
        seen = []
        with open(heavy, "r", encoding="utf-8") as f:
            for line in f:
                seen.append(json.loads(line)["key"])
        size = os.path.getsize(heavy)
        if seen != keys or size > 2 * 1024 ** 3:
            heavy.unlink()
            raise Stop("Internal error: the batch input file is not what was planned, or is "
                       "larger than 2 GB. Nothing was submitted.")
        try:
            check_budget(ctx, bound)
        except cost_ledger.BudgetExceeded:
            heavy.unlink()
            raise
        # (d) registry line, then the reservation in the ledger
        self._event(job_id, "planned", condition_id=cond_id, keys=keys, n_requests=len(keys),
                    est_tokens=est, bound_usd=bound, input_bytes=size,
                    input_sha256=file_sha256(heavy), light_file=light.name)
        ledger_line(ctx, job_id + ":reserve", "batch-reserve", bound,
                    note=f"reservation: worst case of {len(keys)} requests, released when the "
                         "job is collected", job_id=job_id, n_requests=len(keys))
        self._event(job_id, "reserved")
        # (e) upload (safe to repeat)
        file_name, err = "", ""
        for attempt in (1, 2, 3):
            try:
                status, _h, text = ctx.transport.upload_file(str(heavy), job_id,
                                                             cfg["upload_mime_type"])
            except TransportProblem as exc:
                err = f"{exc.kind}: {str(exc)[:200]}"
                self.sleep(10 * attempt)
                continue
            if status == 200:
                try:
                    file_name = str((json.loads(text).get("file") or {}).get("name") or "")
                except ValueError:
                    file_name = ""
                if file_name:
                    break
            err = f"HTTP {status}: {api_error_message(text)}"
            if status not in RETRY_STATUS:
                break
            self.sleep(10 * attempt)
        if not file_name:
            heavy.unlink()
            self._event(job_id, "create_failed", step="upload", message=err[:300])
            self._release(self.jobs[job_id], "upload failed: reservation released")
            raise Stop(f"The batch input file could not be uploaded ({err}). Nothing was "
                       "submitted and the reservation was released. If the message is about "
                       "the file type, set batch.upload_mime_type to text/plain in "
                       "configs/models.yaml and run again.")
        self._event(job_id, "uploaded", file_name=file_name)
        # (f) the create call: sent once, never repeated
        self._event(job_id, "create_sent")
        try:
            status, _h, text = ctx.transport.batch_create(m["model_id"], job_id, file_name)
        except TransportProblem as exc:
            self._event(job_id, "create_unknown", message=f"{exc.kind}: {str(exc)[:200]}")
            say(f"    {job_id}: the create call got no answer ({exc.kind}). Its outcome is "
                "unknown, so it is NOT sent again. The next pass looks the job up.")
            return "unknown"
        finally:
            if heavy.exists():
                heavy.unlink()
        name = ""
        if status == 200:
            try:
                name = str(json.loads(text).get("name") or "")
            except ValueError:
                name = ""
        if status == 200 and name:
            self._event(job_id, "submitted", batch_name=name)
            self.submitted_now += 1
            say(f"    {job_id}: submitted as {name}; {len(keys)} requests, {size / 1e6:.1f} MB, "
                f"about {est:,} tokens, reserved {bound:.4f} USD")
            return "submitted"
        if status == 200 or status >= 500 or status == 408:
            self._event(job_id, "create_unknown",
                        message=f"HTTP {status}: {api_error_message(text)[:200]}")
            say(f"    {job_id}: the create call ended with HTTP {status}. Its outcome is "
                "unknown, so it is NOT sent again. The next pass looks the job up.")
            return "unknown"
        self._event(job_id, "create_failed", step="create", http_status=status,
                    message=api_error_message(text)[:300])
        self._release(self.jobs[job_id], "batch creation refused: reservation released")
        if status == 429:
            say(f"    {cond_id}: the API refused the job (HTTP 429: "
                f"{api_error_message(text)[:160]}). This is usually the enqueued-token "
                "limit. Nothing is queued for it; the next pass tries again.")
            return "queue_full"
        raise Stop(f"Batch creation for {cond_id} was refused with HTTP {status}. "
                   f"{FATAL_HINTS.get(status, '')} API message: {api_error_message(text)} "
                   "Nothing is queued and the reservation was released.")

    def table(self):
        lines = []
        for j in sorted(self.jobs.values(), key=lambda x: x["job_id"].rsplit("-", 2)[-2:]):
            real = f"  real {float(j['usd_actual']):.4f}" if "usd_actual" in j else ""
            lines.append(f"    {j['job_id']}  {j['status']:<14} {(j.get('state') or '-'):<10}"
                         f" {int(j.get('n_requests') or 0):>5} requests  reserved "
                         f"{float(j.get('bound_usd') or 0):.4f}{real}")
        return lines


def run_gemini_batch(ctx, works_by_cond, submit=True, wait=False, poll_seconds=60):
    """One pass: settle, poll and collect, then submit what is free. With wait=True the
    passes repeat until nothing is open and nothing more can be submitted."""
    b = Batch(ctx)
    if b.bad_lines:
        say(f"  note: {b.bad_lines} torn lines in {b.events_path} were skipped")
    sleep = (lambda s: None) if ctx.dry else time.sleep
    passes = 0
    while True:
        passes += 1
        b.reconcile()
        b.poll()
        result = b.plan_and_submit(works_by_cond) if submit else Counter()
        n_open = len(b.open_jobs())
        n_todo = sum(1 for ws in works_by_cond.values() for w in ws
                     if w["key"] not in ctx.cache.records)
        say(f"  pass {passes}: {n_open} open jobs; {n_todo} items without an answer"
            + (f"; {result['given_up']} items failed {b.cfg['max_attempts_per_item']} times and "
               "are not sent again" if result["given_up"] else ""))
        ctx.stats["given_up"] = result["given_up"]
        if not wait or (n_open == 0 and not result["submitted"]) or passes >= 3000:
            break
        sleep(poll_seconds)
    return b


def fake_answer(prompt_text):
    """A made-up answer in the form the prompt asks for (dry runs only)."""
    if "বিকল্পের অক্ষরটি" in prompt_text:
        return "খ"
    if "option letter" in prompt_text:
        return "B"
    if "শুধু বাংলায়" in prompt_text or "Bangla (Bengali script)" in prompt_text:
        return "নকল উত্তর।"
    return "Dry answer."


class FakeGemini:
    """Dry run: no network, no cost. It answers in the API's own shapes, in both
    documented status shapes, and injects each fault once, so that the retry, the
    lost create call, the adopted job, the error line and the missing line are all
    exercised: one HTTP 503, one timeout, two create calls without an answer."""

    dry = True
    version = "dry-run"

    def __init__(self):
        self.n_gen = 0
        self.n_create = 0
        self.files = {}
        self.batches = {}
        self.faults = Counter()

    def scrub(self, text):
        return text or ""

    def get_model(self, model_id):
        return 200, {}, json.dumps({"name": "models/" + model_id, "displayName": "DRY RUN"})

    @staticmethod
    def _reply(has_image, n, text="Dry answer."):
        p = 100 + (1120 if has_image else 0)
        details = [{"modality": "TEXT", "tokenCount": 100}]
        if has_image:
            details.append({"modality": "IMAGE", "tokenCount": 1120})
        if n % 23 == 0:        # a blocked prompt: final, kept as an empty answer
            return {"promptFeedback": {"blockReason": "OTHER"},
                    "usageMetadata": {"promptTokenCount": p, "totalTokenCount": p}}
        if n % 17 == 0:        # thinking used the whole output limit: final, empty
            return {"candidates": [{"finishReason": "MAX_TOKENS", "index": 0}],
                    "usageMetadata": {"promptTokenCount": p, "thoughtsTokenCount": 256,
                                      "totalTokenCount": p + 256}}
        return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]},
                                "finishReason": "STOP", "index": 0}],
                "usageMetadata": {"promptTokenCount": p, "candidatesTokenCount": 3,
                                  "thoughtsTokenCount": 0, "totalTokenCount": p + 3,
                                  "promptTokensDetails": details},
                "modelVersion": "dry-run", "responseId": f"dry-{n}"}

    def generate(self, model_id, body):
        self.n_gen += 1
        if self.n_gen == 1:
            self.faults["http_503"] += 1
            return 503, {}, json.dumps({"error": {"code": 503, "message": "dry-run injected fault"}})
        if self.n_gen == 3:
            self.faults["timeout"] += 1
            raise TransportProblem("timeout", "dry-run injected timeout")
        parts = body["contents"][0]["parts"]
        has_image = any("inlineData" in p for p in parts)
        return 200, {}, json.dumps(self._reply(has_image, self.n_gen, fake_answer(parts[-1]["text"])))

    def upload_file(self, path, display_name, mime):
        keys = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                parts = o["request"]["contents"][0]["parts"]
                keys.append((o["key"], any("inlineData" in p for p in parts),
                             fake_answer(parts[-1]["text"])))
        name = f"files/dry-in-{len(self.files) + 1}"
        self.files[name] = keys
        return 200, {}, json.dumps({"file": {"name": name}})

    def batch_create(self, model_id, display_name, file_name):
        self.n_create += 1
        if self.n_create == 2:      # the call never reached the server
            self.faults["create_lost"] += 1
            raise TransportProblem("timeout", "dry-run: the create call was lost")
        name = f"batches/dry-{self.n_create}"
        self.batches[name] = {"display": display_name, "file": file_name, "polls": 0,
                              "n": self.n_create}
        if self.n_create == 1:      # the job was made, but the answer was lost
            self.faults["create_answer_lost"] += 1
            raise TransportProblem("timeout", "dry-run: the answer to the create call was lost")
        return 200, {}, json.dumps({"name": name, "metadata": {"state": "BATCH_STATE_PENDING"}})

    def _op(self, name):
        b = self.batches[name]
        state = ("PENDING", "RUNNING", "SUCCEEDED")[min(b["polls"], 2)]
        out = f"files/dry-out-{b['n']}"
        if b["n"] % 2:              # shape 1: a long-running operation
            op = {"name": name, "done": state == "SUCCEEDED",
                  "metadata": {"displayName": b["display"], "state": "BATCH_STATE_" + state,
                               "batchStats": {"requestCount": str(len(self.files[b["file"]]))}}}
            if state == "SUCCEEDED":
                op["response"] = {"responsesFile": out}
        else:                       # shape 2: a flat job object
            op = {"name": name, "displayName": b["display"], "state": "JOB_STATE_" + state}
            if state == "SUCCEEDED":
                op["dest"] = {"fileName": out}
        return op

    def batch_get(self, name):
        if name not in self.batches:
            return 404, {}, json.dumps({"error": {"message": "not found"}})
        op = self._op(name)
        self.batches[name]["polls"] += 1
        return 200, {}, json.dumps(op)

    def batch_list(self, page_token=""):
        return 200, {}, json.dumps({"operations": [self._op(n) for n in self.batches]})

    def download(self, file_name, dst_path):
        n = int(file_name.rsplit("-", 1)[1])
        b = [x for x in self.batches.values() if x["n"] == n][0]
        keys = self.files[b["file"]]
        with open(dst_path, "w", encoding="utf-8") as f:
            for i, (key, has_image, answer) in enumerate(keys):
                self.n_gen += 1
                reply = self._reply(has_image, self.n_gen, answer)
                if n == 1 and i == 0 and len(keys) > 2:
                    self.faults["error_line"] += 1
                    line = {"key": key, "error": {"code": 13, "message": "dry-run injected error"}}
                elif n == 1 and i == len(keys) - 1 and len(keys) > 2:
                    self.faults["missing_line"] += 1
                    continue
                elif n % 3 == 0:
                    line = {"response": reply}                  # no key: input order
                elif n % 2 == 0:
                    line = {"metadata": {"key": key}, "response": reply}
                else:
                    line = {"key": key, "response": reply}
                f.write(json.dumps(line) + "\n")
        return 200, {}, ""

    def file_delete(self, file_name):
        return 200, {}, "{}"


# --------------------------------------------------------------------------
# 5. The open models: Hugging Face transformers, greedy (Design 5.2, D147)
# --------------------------------------------------------------------------
THINK_MARKERS = ("<think>", "</think>", "<|channel>", "<channel|>", "<|think|>")


def hf_settings(m):
    return {"backend": "hf", "model_id": m["model_id"], "family": m.get("family", ""),
            "dtype": m["dtype"], "max_new_tokens": int(m["max_new_tokens"]),
            "enable_thinking": bool(m["enable_thinking"]),
            "max_pixels": m.get("max_pixels") or "processor default",
            "decoding": "greedy: do_sample False, num_beams 1, repetition_penalty 1.0, "
                        "no shipped sampling value",
            "batch_size": 1, "system_prompt": "none",
            "turns": "one user turn: image, then text", "image_prep": imglib.PREP_VERSION,
            **({"image_token_budget": int(m["image_token_budget"])}
               if m.get("image_token_budget") else {}),
            **({"adapter": str(m["adapter"]["run"])} if m.get("adapter") else {})}


def qwen_fast_path():
    """Design 5.3: record whether the fast kernels of the Qwen3.5 linear-attention
    layers are there. Without them transformers falls back, silently, to slower code."""
    info = {}
    for mod in ("fla", "causal_conv1d"):
        try:
            importlib.import_module(mod)
            info[mod] = True
        except Exception:
            info[mod] = False
    flag = None
    for name in ("transformers.models.qwen3_5.modeling_qwen3_5",
                 "transformers.models.qwen3_next.modeling_qwen3_next"):
        try:
            flag = getattr(importlib.import_module(name), "is_fast_path_available", flag)
        except Exception:
            pass
    info["fast_path_flag"] = flag if isinstance(flag, bool) else None
    return info


# --- 5b. LoRA adapters of EXP-R (Design 7.3; decisions D170 to D178) --------------------
# Nothing below runs for a model entry without 'adapter' in configs/models.yaml.
ADAPTER_FILE = "adapter_model.safetensors"
FT_MANIFEST = "ft_manifest.json"


def base_entry(m):
    """A model entry without the two fields that only an adapter entry has."""
    return {k: v for k, v in m.items() if k not in ("adapter", "cache_id")}


def read_adapter(root, stage, m, prompts):
    """Find the adapter of the model entry m in Drive models/<run>/ and check that it
    fits this run: training finished, same base model, same fixed settings, same
    training prompt text, same base commit, unchanged weights file. No GPU is needed.
    Returns a dict; raises Stop with the reason otherwise."""
    run_name = str(m["adapter"]["run"])
    run_dir = Path(root) / "models" / run_name
    path = run_dir / FT_MANIFEST
    if not path.exists():
        raise Stop(f"Adapter {run_name}: {path} does not exist. Train it first "
                   "(scripts/ft_finetune.py train).")
    man = read_json(path, {})
    if not man.get("complete") or man.get("run") != run_name:
        raise Stop(f"Adapter {run_name}: its training is not finished ({path}). Run the "
                   "train command again until it prints COMPLETE.")
    profile = man.get("profile")
    if profile == "dry" and stage != "smoke":
        raise Stop(f"Adapter {run_name} is the Gate G dry-run adapter. It runs in stage "
                   "smoke only and is never reported (Design 7.3).")
    if stage == "main" and profile != "seed":
        raise Stop(f"Adapter {run_name}: stage main runs the adapters of the three seeds only.")
    base = man.get("base") or {}
    if base.get("model_id") != m["model_id"]:
        raise Stop(f"Adapter {run_name} was trained on {base.get('model_id')}, but this model "
                   f"entry loads {m['model_id']}.")
    now = hf_settings(base_entry(m))
    if base.get("settings_sha256") != canonical_sha(now):
        then = base.get("settings") or {}
        diff = sorted(k for k in set(now) | set(then) if then.get(k) != now.get(k))
        raise Stop(f"Adapter {run_name} was trained under other fixed settings of the base "
                   f"model (changed: {', '.join(diff) or 'unknown'}). A change of "
                   "configs/models.yaml after training needs a decision row, and the "
                   "adapter must be trained again.")
    pr = man.get("prompt") or {}
    text = ((prompts.get("templates") or {}).get(pr.get("template_id")) or {}).get(pr.get("variant"))
    if text is None or sha256_hex(text) != pr.get("template_sha256"):
        raise Stop(f"Adapter {run_name} was trained with another text of the prompt template "
                   f"{pr.get('template_id')} (prompt version {pr.get('prompt_version')}; now "
                   f"{prompts.get('prompt_version')}). An adapter is evaluated with the prompt "
                   "it was trained with (Design 7.3). Train it again after Gate E.")
    if m.get("revision") and m["revision"] != base.get("model_revision"):
        raise Stop(f"Adapter {run_name} was trained on commit {base.get('model_revision')} of "
                   f"{m['model_id']}, but configs/models.yaml fixes commit {m['revision']}.")
    weights = run_dir / "adapter" / ADAPTER_FILE
    if not weights.exists():
        raise Stop(f"Adapter {run_name}: {weights} is missing.")
    digest = file_sha256(weights)
    if digest != (man.get("adapter") or {}).get("sha256"):
        raise Stop(f"Adapter {run_name}: {weights} is not the file written by the training "
                   "run (other SHA-256). Restore it or train again; do not edit it.")
    lora = man.get("lora") or {}
    return {"run": run_name, "dir": str(run_dir / "adapter"), "sha256": digest,
            "profile": profile, "seed": man.get("seed"),
            "base_revision": base.get("model_revision"),
            "peft": (man.get("env") or {}).get("peft"),
            "n_modules": int(lora.get("n_modules") or 0),
            "param_dtype": lora.get("param_dtype"),
            "script_version": man.get("script_version")}


class HFBackend:
    """One open model on one GPU. Batch size 1, so padding can never change an answer."""

    def __init__(self, ctx):
        m = ctx.model
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise Stop(f"torch and transformers are needed for the open models ({exc}).")
        self.torch, self.m = torch, m
        if int(transformers.__version__.split(".")[0]) < 5:
            raise Stop(f"transformers {transformers.__version__} is too old: Qwen3.5 and "
                       "Gemma 4 need version 5 or newer. Run: pip install -U transformers")
        if not torch.cuda.is_available():
            raise Stop("No GPU. In Colab choose Runtime, Change runtime type, L4 GPU.")
        gpu = torch.cuda.get_device_name(0)
        if "T4" in gpu and m.get("family") == "gemma4":
            raise Stop("Design 5.3: Gemma 4 never runs on a T4 (known fp16 overflow issue).")
        if "L4" not in gpu and not ctx.allow_env_change:
            raise Stop(f"The GPU is '{gpu}'. Design 5.3: inference runs on an L4, and one "
                       "model's runs are never split across platforms. Qwen3.5 may run on a "
                       "Kaggle T4 only after a 20-item smoke test and with a decision row "
                       "(then add --allow-env-change).")
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        token = os.environ.get("HF_TOKEN") or None
        earlier = (read_json(ctx.out_dir / "stage_state.json", {}).get("env") or {})
        self.revision = m.get("revision") or earlier.get("model_revision")
        self.adapter, self.adapter_fields = None, {}
        if m.get("adapter"):
            self.adapter = read_adapter(ctx.root, ctx.stage, m, ctx.prompts)
            if self.revision and self.revision != self.adapter["base_revision"]:
                raise Stop(f"Adapter {self.adapter['run']} was trained on commit "
                           f"{self.adapter['base_revision']} of {m['model_id']}, but this "
                           f"stage is fixed to commit {self.revision}. Base model and "
                           "adapter must be one commit.")
            self.revision = self.adapter["base_revision"]
        if not self.revision:
            from huggingface_hub import HfApi
            self.revision = HfApi().model_info(m["model_id"], token=token).sha
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForMultimodalLM as auto_model
        except ImportError:
            from transformers import AutoModelForImageTextToText as auto_model
        say(f"  loading {m['model_id']} at commit {self.revision} on {gpu} ({m['dtype']})")
        kw = {"revision": self.revision, "token": token}
        try:
            self.processor = AutoProcessor.from_pretrained(
                m["model_id"], **(dict(kw, max_pixels=int(m["max_pixels"])) if m.get("max_pixels") else kw))
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(m["model_id"], **kw)
        if m.get("max_pixels"):
            self._set_max_pixels(int(m["max_pixels"]))
        self.budget_set_on = self._set_image_budget(int(m["image_token_budget"])) \
            if m.get("image_token_budget") else []
        self.model = auto_model.from_pretrained(m["model_id"], dtype=getattr(torch, m["dtype"]),
                                                device_map={"": 0}, **kw)
        self.model.eval()
        if self.adapter:
            self._attach_adapter()
        gc = copy.deepcopy(self.model.generation_config)
        self.shipped_generation_config = json.loads(gc.to_json_string())
        gc.do_sample = False
        gc.num_beams = 1
        for name in ("temperature", "top_p", "top_k", "typical_p", "min_p", "epsilon_cutoff",
                     "eta_cutoff", "penalty_alpha"):
            if getattr(gc, name, None) is not None:
                setattr(gc, name, None)
        gc.repetition_penalty = 1.0
        if getattr(gc, "no_repeat_ngram_size", None):
            gc.no_repeat_ngram_size = 0
        gc.max_new_tokens = int(m["max_new_tokens"])
        self.gc = gc
        # generate() fills unset values from the model's own config: make that config
        # the greedy one too, so no shipped sampling value can come back.
        self.model.generation_config = copy.deepcopy(gc)
        self.generation_config_used = json.loads(gc.to_json_string())
        eos = gc.eos_token_id
        self.eos = set([eos] if isinstance(eos, int) else list(eos or []))
        self.image_token_id = next(
            (getattr(self.model.config, n) for n in ("image_token_id", "image_token_index")
             if getattr(self.model.config, n, None) is not None), None)
        self.probe = self._probe()

    def _attach_adapter(self):
        """Put the LoRA weights of EXP-R into the loaded base model (D174). PEFT changes
        the base model object in place, and that same object goes on generating, so the
        code below this point is the code a base model runs."""
        a, torch = self.adapter, self.torch
        try:
            import peft
            from peft import PeftModel
        except ImportError as exc:
            raise Stop(f"peft is needed to load the adapter {a['run']} ({exc}). Run: "
                       f"pip install -q peft=={a['peft']}")
        if a["peft"] and peft.__version__ != a["peft"]:
            raise Stop(f"Adapter {a['run']} was trained with peft {a['peft']}; this session "
                       f"has peft {peft.__version__}. Training and evaluation use one PEFT "
                       f"version (D171). Run: pip install -q peft=={a['peft']}")
        say(f"  attaching the adapter {a['run']} (sha256 {a['sha256'][:12]}) from {a['dir']}")
        self.peft_model = PeftModel.from_pretrained(self.model, a["dir"], is_trainable=False,
                                                    autocast_adapter_dtype=True)
        n_lora = sum(1 for _n, mod in self.model.named_modules()
                     if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"))
        if not n_lora or n_lora != a["n_modules"]:
            raise Stop(f"Adapter {a['run']}: {n_lora} layers of the loaded model carry LoRA "
                       f"weights, the training run adapted {a['n_modules']}. Send this line "
                       "to the FT chat. Nothing was run.")
        dtypes = sorted({str(p.dtype) for n, p in self.model.named_parameters()
                         if ".lora_A." in n or ".lora_B." in n})
        if a["param_dtype"] and dtypes != [a["param_dtype"]]:
            raise Stop(f"Adapter {a['run']}: the LoRA weights are loaded as {dtypes}, they "
                       f"were trained as {a['param_dtype']}. Send this line to the FT chat.")
        a["weights_check"] = self._check_adapter_weights()
        self.model.eval()
        self.adapter_fields = {"adapter_run": a["run"], "adapter_sha256": a["sha256"]}

    def _check_adapter_weights(self):
        """Compare the loaded LoRA tensors with the file, tensor by tensor."""
        torch = self.torch
        try:
            from safetensors.torch import load_file
            saved = load_file(os.path.join(self.adapter["dir"], ADAPTER_FILE))
        except Exception as exc:
            return f"not compared ({type(exc).__name__})"
        params = dict(self.model.named_parameters())
        matched = 0
        for key, val in saved.items():
            name = key[len("base_model.model."):] if key.startswith("base_model.model.") else key
            for tag in (".lora_A.", ".lora_B."):
                if tag in name and tag + "default." not in name:
                    name = name.replace(tag, tag + "default.", 1)
            p = params.get(name)
            if p is None:
                continue
            matched += 1
            if tuple(p.shape) != tuple(val.shape) or \
                    not torch.equal(p.detach().to("cpu", dtype=val.dtype), val):
                raise Stop(f"Adapter {self.adapter['run']}: the loaded weights of {name} are "
                           "not the weights in the file. Send this line to the FT chat.")
        if not matched:
            return f"not compared (none of the {len(saved)} names in the file was recognised)"
        return f"{matched} of {len(saved)} tensors compared with the file, all equal"

    def _set_max_pixels(self, max_pixels):
        ip = getattr(self.processor, "image_processor", None)
        if ip is None:
            return
        size = getattr(ip, "size", None)
        if isinstance(size, dict) and "longest_edge" in size:
            ip.size = dict(size, longest_edge=max_pixels)
        if getattr(ip, "max_pixels", None) is not None:
            ip.max_pixels = max_pixels

    def _set_image_budget(self, budget):
        """Gemma 4: the visual token budget per image (D151). The value sits in
        processor_config.json as max_soft_tokens and image_seq_length."""
        if budget not in (70, 140, 280, 560, 1120):
            raise Stop("image_token_budget must be 70, 140, 280, 560 or 1120 (Gemma 4 card).")
        done = []
        for label, obj in (("image_processor", getattr(self.processor, "image_processor", None)),
                           ("processor", self.processor)):
            for name in ("max_soft_tokens", "image_seq_length"):
                if obj is not None and getattr(obj, name, None) is not None:
                    setattr(obj, name, budget)
                    done.append(f"{label}.{name}")
        return done

    def _inputs(self, text, image):
        content = ([{"type": "image", "image": image}] if image is not None else []) \
            + [{"type": "text", "text": text}]
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=bool(self.m["enable_thinking"]))

    def _count_image_tokens(self, inputs):
        grid = inputs.get("image_grid_thw") if hasattr(inputs, "get") else None
        vc = getattr(self.model.config, "vision_config", None)
        if grid is not None:
            merge = int(getattr(vc, "spatial_merge_size", 2) or 2)
            t, h, w = [int(x) for x in grid[0].tolist()]
            return t * h * w // (merge * merge)
        if self.image_token_id is not None:
            return int((inputs["input_ids"] == self.image_token_id).sum())
        return None

    def _probe(self):
        """Checks on made-up grey images: image tokens, max_pixels, the thinking switch."""
        from PIL import Image
        info = {}
        for name, size in (("1024x768", (1024, 768)), ("2048x1536", (2048, 1536))):
            inputs = self._inputs("x", Image.new("RGB", size, (127, 127, 127)))
            info["image_tokens_" + name] = self._count_image_tokens(inputs)
        if self.m.get("max_pixels"):
            vc = getattr(self.model.config, "vision_config", None)
            patch = getattr(vc, "patch_size", 16) or 16
            patch = int(patch if isinstance(patch, int) else patch[0])
            px = (patch * int(getattr(vc, "spatial_merge_size", 2) or 2)) ** 2
            limit, big = int(self.m["max_pixels"]) // px, info["image_tokens_2048x1536"]
            if big is None or big > limit:
                raise Stop(f"max_pixels = {self.m['max_pixels']} did not take effect: a "
                           f"2048x1536 test image gives {big} image tokens, the limit is "
                           f"{limit} (Design 5.2). Send this line and the transformers "
                           "version to the ASSET-8 chat. Nothing was run.")
        if self.m.get("image_token_budget"):
            budget, got = int(self.m["image_token_budget"]), info["image_tokens_1024x768"]
            info["image_budget_set_on"] = self.budget_set_on
            if got is None or not (0.6 * budget <= got <= budget):
                raise Stop(f"image_token_budget = {budget} did not take effect: a 1024x768 "
                           f"test image gives {got} image tokens (set on: {self.budget_set_on}). "
                           "Send this line to the ASSET-8 chat. Nothing was run.")
        try:
            msgs = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
            off, on = [self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=flag)
                for flag in (False, True)]
            info["thinking_switch_changes_prompt"] = off != on
            info["prompt_tail_thinking_off"] = off[-80:]
            if "<|think|>" in off:
                raise Stop("The chat template still switches thinking on. Design 5.2 needs it "
                           "off. Send this line to the ASSET-8 chat. Nothing was run.")
        except Stop:
            raise
        except Exception as exc:
            info["thinking_switch_changes_prompt"] = None
            info["thinking_probe_error"] = type(exc).__name__
        return info

    def env(self):
        import transformers
        torch = self.torch
        e = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
             "cuda": torch.version.cuda, "transformers": transformers.__version__,
             "dtype": self.m["dtype"], "model_revision": self.revision,
             "model_class": type(self.model).__name__,
             "processor_class": type(self.processor).__name__}
        if self.m.get("family") == "qwen3_5":
            e.update(qwen_fast_path())
        if self.adapter:
            import peft
            e.update({"peft": peft.__version__, "adapter_run": self.adapter["run"],
                      "adapter_sha256": self.adapter["sha256"]})
        return e

    def generate(self, text, image):
        torch = self.torch
        inputs = self._inputs(text, image).to(self.model.device)
        n_in = int(inputs["input_ids"].shape[-1])
        n_img = self._count_image_tokens(inputs) if image is not None else 0
        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(**inputs, generation_config=self.gc)
        new = out[0][n_in:]
        n_new = int(new.shape[-1])
        raw = self.processor.decode(new, skip_special_tokens=False)
        answer = self.processor.decode(new, skip_special_tokens=True)
        stopped = n_new < int(self.gc.max_new_tokens) or (n_new > 0 and int(new[-1]) in self.eos)
        return {"answer_text": answer, "raw_text": raw, "n_new_tokens": n_new,
                "n_input_tokens": n_in, "image_tokens": n_img,
                "finish_reason": "stop" if stopped else "length",
                "thinking_leak": any(mk in raw for mk in THINK_MARKERS),
                "elapsed_s": round(time.time() - t0, 3)}


class FakeHF:
    """Dry run: no GPU and no download. Made-up answers in the real record shape."""

    revision = "dry-run"
    probe = {}
    shipped_generation_config = {}
    generation_config_used = {}

    def __init__(self):
        self.n = 0

    def env(self):
        return {"gpu": "dry-run", "model_revision": "dry-run"}

    def generate(self, text, image):
        self.n += 1
        cut = self.n % 19 == 0
        answer = fake_answer(text) + (" x" * 12 if cut else "")
        return {"answer_text": answer, "raw_text": answer + "<end>",
                "n_new_tokens": 32 if cut else 3,
                "n_input_tokens": (808 if image is not None else 40) + len(text) // 3,
                "image_tokens": 768 if image is not None else 0,
                "finish_reason": "length" if cut else "stop", "thinking_leak": False,
                "elapsed_s": 0.0}


def run_hf(ctx, work):
    todo = [w for w in work if w["key"] not in ctx.cache.records]
    if not todo:
        return
    backend = FakeHF() if ctx.dry else HFBackend(ctx)
    env = backend.env()
    check_state(ctx, env=env)
    ctx.hf_info = {"env": env, "probe": backend.probe,
                   "shipped_generation_config": backend.shipped_generation_config,
                   "generation_config_used": backend.generation_config_used}
    if getattr(backend, "adapter", None):
        ctx.hf_info["adapter"] = backend.adapter
    say("  environment: " + json.dumps(env))
    say("  checks at load: " + json.dumps(backend.probe, ensure_ascii=False))
    t_start = time.time()
    for n, w in enumerate(todo, 1):
        image = None
        if w["has_image"]:
            data, _sha = imglib.load_prepared(ctx.img_root, w["image_file"], ctx.img_manifest)
            image = imglib.decode_rgb(data)
        ids = {"run_id": ctx.run_id, "key": w["key"], "qa_id": w["qa_id"],
               "condition_id": w["condition_id"], "ts_utc": utc_now()}
        try:
            g = backend.generate(w["prompt_text"], image)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {str(exc)[:300]}"
            append_jsonl(ctx.rawlog, dict(ids, kind="hf_error", error=msg))
            if "out of memory" in msg.lower():
                raise Stop("CUDA out of memory. Restart the runtime, run only this model in "
                           "the session, and run the same command again. " + msg)
            ctx.stats["item_errors"] += 1
            if ctx.stats["item_errors"] >= 5 and not ctx.stats["answered"]:
                raise Stop("The first five items all failed. Last error: " + msg)
            continue
        append_jsonl(ctx.rawlog, dict(ids, kind="hf_response", raw_text=g["raw_text"],
                                      n_new_tokens=g["n_new_tokens"],
                                      finish_reason=g["finish_reason"]))
        ctx.cache.add(finish_record(
            base_record(ctx, w, "hf"), g["answer_text"], finish_reason=g["finish_reason"],
            block_reason="", truncated=g["finish_reason"] == "length",
            input_tokens=g["n_input_tokens"], output_tokens=g["n_new_tokens"],
            thoughts_tokens=0, image_tokens=g["image_tokens"],
            thinking_leak=g["thinking_leak"], elapsed_s=g["elapsed_s"], usd=0.0,
            model_revision=backend.revision, call_id=f"{ctx.run_id}-g{n:05d}",
            **getattr(backend, "adapter_fields", {})))
        ctx.stats["answered"] += 1
        if n % 50 == 0 or n == len(todo):
            mins = (time.time() - t_start) / 60
            say(f"    {n} of {len(todo)} items done in {mins:.1f} min")
    ctx.gpu_seconds = time.time() - t_start


# --------------------------------------------------------------------------
# 6. One run, from the command line arguments to the result
# --------------------------------------------------------------------------
def parse_conditions(spec, cfg):
    out = []
    for name in [s.strip() for s in spec.split(",") if s.strip()]:
        members = cfg["sets"].get(name) or ([name] if name in cfg["conditions"] else None)
        if not members:
            raise Stop(f"--conditions: '{name}' is neither a condition nor a set. Conditions: "
                       f"{', '.join(cfg['conditions'])}. Sets: {', '.join(cfg['sets'])}.")
        out += [c for c in members if c not in out]
    return out


def layout(root, stage, run_name, model_key, dry):
    if stage == "smoke":
        label = f"smoke/{run_name or 'default'}"
    elif run_name:
        raise Stop("--run-name is for stage smoke only. Pilot and main have one cache each.")
    else:
        label = stage
    base = Path(root) / "outputs" / "dryrun" if dry else Path(root) / "outputs"
    return base / label / model_key, label


def check_probe_rule(stage, cfg, probe, cond_ids, selection_used, dry):
    rule = cfg["stages"][stage]["probe_rule"]
    probes = cfg["probes"]

    def same(name):
        want = probes.get(name) or {}
        return probe["file"] == want.get("file") and len(probe["items"]) == want.get("expected_n")

    if rule == "PILOT" and not same("PILOT"):
        raise Stop(f"Stage pilot runs PILOT.json with {probes['PILOT']['expected_n']} items "
                   f"(Design 11). Given: {probe['file']} with {len(probe['items'])} items.")
    if rule == "condition":
        if selection_used and not dry:
            raise Stop("Stage main runs the full probe set of each condition. Remove --where, "
                       "--ids-file, --sample and --limit. A stopped run is simply started "
                       "again: the cache keeps what is done.")
        for cid in cond_ids:
            pname = cfg["conditions"][cid]["probe"]
            if not (probes.get(pname) or {}).get("file"):
                raise Stop(f"{cid}: its probe set {pname} is not defined yet in "
                           "configs/conditions.yaml (ASSET-11 writes P-FCE after Gate A).")
            if not same(pname) and not dry:
                raise Stop(f"{cid} runs on {probes[pname]['file']} with "
                           f"{probes[pname]['expected_n']} items (DESIGN_ADDENDUM A1). Given: "
                           f"{probe['file']} with {len(probe['items'])} items. In stage main, "
                           "run the conditions of one probe set per command.")


def run(a):
    """a: the parsed command line of 'run'. Returns the exit code."""
    repo, root = Path(a.repo), Path(a.root)
    prompts = promptlib.load_prompts(repo / "configs" / "prompts.yaml")
    cfg = promptlib.load_conditions(repo / "configs" / "conditions.yaml", prompts)
    models = load_models(repo / "configs" / "models.yaml")
    if a.model not in models:
        raise Stop(f"--model: '{a.model}' is not in configs/models.yaml ({', '.join(models)}).")
    m = models[a.model]
    dry = bool(a.dry_run)
    stage_cfg = cfg["stages"][a.stage]
    cond_ids = parse_conditions(a.conditions, cfg)
    is_gemini = m["backend"] == "gemini"
    if is_gemini:
        allowed = stage_cfg["gemini_modes"]
        mode = a.mode or (allowed[0] if len(allowed) == 1 else None)
        if mode not in allowed:
            raise Stop(f"Stage {a.stage} runs Gemini in {' or '.join(allowed)} mode (D140: "
                       "standard for the pilot, Batch for EXP-L and EXP-N). Give --mode.")
    else:
        mode = "hf"
    out_dir, label = layout(root, a.stage, a.run_name, a.model, dry)
    if dry and not getattr(a, "dry_keep", False) and out_dir.exists() and "dryrun" in out_dir.parts:
        import shutil
        shutil.rmtree(out_dir)            # every dry run starts clean
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ("-dry" if dry else "")
    git = git_info(repo)

    say("=" * 78)
    say(f"ASSET-8 harness {HARNESS_VERSION}   {'DRY RUN (no network, no GPU, no cost)' if dry else 'REAL RUN'}"
        f"   run_id {run_id}")
    say(f"stage {label}   model {a.model} = {m['model_id']}   mode {mode}   prompts "
        f"{prompts['prompt_version']} (sha256 {prompts['sha256'][:12]})")
    say(f"code: commit {git['commit'][:12] or 'unknown'}"
        + ("" if not git["dirty"] else "  WITH UNCOMMITTED CHANGES"))
    say("Everything this run writes is model output: Drive only, and not for the two HT-1 "
        "people before gate-A (Design 4.2, D125).")
    say("=" * 78)
    if a.stage == "main" and not dry:
        if not git["gate_e"] or git["dirty"] is not False:
            raise Stop("Stage main needs the tag gate-E on this commit or before it, and no "
                       "uncommitted change (Design 9.7, Gate E). Full runs wait for the "
                       "pre-specification commit.")

    # ---- items ----------------------------------------------------------------
    say("\n[1] Items")
    probe = load_probe(a.probe)
    where = []
    for pair in a.where or []:
        if "=" not in pair:
            raise Stop(f"--where needs field=value, got '{pair}'.")
        where.append(tuple(pair.split("=", 1)))
    ids = None
    if a.ids_file:
        with open(a.ids_file, "r", encoding="utf-8") as f:
            ids = {line.strip() for line in f if line.strip()}
    selection_used = bool(where or ids is not None or a.sample or a.limit)
    check_probe_rule(a.stage, cfg, probe, cond_ids, selection_used, dry)
    items = select_items(probe["items"], where, ids, a.sample, a.limit)
    say(f"  read {probe['file']}: {len(probe['items'])} items (created {probe['created_utc']}); "
        f"selected {len(items)}"
        + (f" (where {a.where or '-'}, ids-file {'yes' if ids is not None else 'no'}, "
           f"sample {a.sample}, limit {a.limit})" if selection_used else ""))
    if not items:
        raise Stop("No item is selected.")
    conds = [cfg["conditions"][c] for c in cond_ids]
    english = options = None
    if any(c["question_lang"] == "en" for c in conds):
        if not a.english_file or not a.english_column:
            raise Stop("An English-question condition needs --english-file and "
                       "--english-column (the final file of ASSET-2 stage 2).")
        if a.english_column == MT_COLUMN and a.stage != "smoke":
            raise Stop(f"The column {MT_COLUMN} is the machine translation of stage 1. The "
                       "pilot and the full runs use the post-edited English (Gate B). It is "
                       "accepted in stage smoke only (D149).")
        english = load_english(a.english_file, a.english_column)
        say(f"  English questions: {len(english['table'])} strings from column "
            f"'{english['column']}'")
    if any(c["forced_choice"] for c in conds):
        if not a.options_file:
            raise Stop("A forced-choice condition needs --options-file (ASSET-11).")
        options = load_options(a.options_file)
        say(f"  option sets: {len(options['table'])}")
    img_root = Path(a.images_1024) if a.images_1024 else root / "data" / "images_1024"
    img_manifest = imglib.read_manifest(img_root) if any(c["image"] for c in conds) else {}
    model_id = m.get("cache_id") or m["model_id"]
    works = {c["condition_id"]: build_work(model_id, c, items, prompts, english, options,
                                           img_manifest) for c in conds}
    all_work = [w for ws in works.values() for w in ws]

    # ---- context, cache, guards ------------------------------------------------
    settings = gemini_settings(m) if is_gemini else hf_settings(m)
    prices, price_source = {"in": 0.0, "out": 0.0}, ""
    if is_gemini:
        prices = dict(m["prices"][mode])
        price_source = m["prices"]["source"]
        if a.price_in is not None or a.price_out is not None:
            prices = {"in": a.price_in if a.price_in is not None else prices["in"],
                      "out": a.price_out if a.price_out is not None else prices["out"]}
            price_source = "given on the command line by the runner"
    ledger = out_dir / "dryrun_ledger.jsonl" if dry else \
        (Path(a.ledger) if a.ledger else root / "outputs" / "api_cost_ledger.jsonl")
    tmp_dir = Path("/content/asset8_tmp") if os.path.isdir("/content") else \
        Path(tempfile.gettempdir()) / "asset8_tmp"
    ctx = Ctx(root=root, out_dir=out_dir, stage=a.stage, stage_label=label,
              stage_tag=label.replace("/", "."), model_key=a.model, model=m, model_id=model_id,
              mode=mode, dry=dry, run_id=run_id, prompts=prompts, settings=settings,
              settings_sha=canonical_sha(settings), cache=Cache(out_dir / "cache.jsonl"),
              rawlog=out_dir / "raw_calls.jsonl", ledger=ledger, prices=prices,
              price_source=price_source,
              price_override=a.price_in is not None and a.price_out is not None,
              stage_cap=a.stage_cap_usd if a.stage_cap_usd is not None
              else float(stage_cfg["api_cap_usd"]),
              paid_tier_confirmed=bool(a.paid_tier_confirmed),
              allow_env_change=bool(a.allow_env_change), img_root=img_root,
              img_manifest=img_manifest, transport=FakeGemini() if (dry and is_gemini) else None,
              stats=Counter(), usd_run=0.0, seq=0, tmp_dir=tmp_dir, hf_info={}, gpu_seconds=0.0)
    check_state(ctx)
    check_hits(ctx.cache, all_work, ctx.settings_sha)
    if ctx.cache.duplicates or ctx.cache.bad_lines:
        say(f"  note: the cache file has {ctx.cache.duplicates} repeated keys (the first answer "
            f"stands) and {ctx.cache.bad_lines} unreadable lines (not used)")
    # Design 7.3: an item already answered under the same prompt, image and settings in
    # the condition named by reuse_from is copied, not run again.
    for c in conds:
        src = c.get("reuse_from")
        for w in works[c["condition_id"]] if src else []:
            old = ctx.cache.records.get(cache_key(model_id, src, prompts["prompt_version"], w["qa_id"]))
            if w["key"] not in ctx.cache.records and old and all(
                    old.get(f) == v for f, v in (("prompt_sha256", w["prompt_sha256"]),
                                                 ("image_sha256", w["image_sha256"]),
                                                 ("settings_sha256", ctx.settings_sha))):
                rec = dict(old, key=w["key"], condition_id=w["condition_id"], mode="reuse",
                           reused_from_key=old["key"], run_id=run_id, usd=0.0, ts_utc=utc_now())
                ctx.cache.add(rec)
                ctx.stats["reused"] += 1

    say("\n[2] Plan")
    bound_total = 0.0
    for cid, ws in works.items():
        todo = [w for w in ws if w["key"] not in ctx.cache.records]
        line = f"  {cid:<7} {len(ws):>5} items; in the cache {len(ws) - len(todo):>5}; to run {len(todo):>5}"
        if is_gemini:
            b = sum(usage_usd(worst_usage(m, w), prices) for w in todo)
            bound_total += b
            line += f"; at most {b:.4f} USD"
        say(line)
    n_todo = sum(1 for w in all_work if w["key"] not in ctx.cache.records)
    if ctx.stats["reused"]:
        say(f"  {ctx.stats['reused']} answers copied from the cache of another condition (reuse_from)")
    if is_gemini:
        say(f"  ceiling for this command: {bound_total:.4f} USD ({mode} prices, every answer at "
            f"the {m['max_output_tokens']}-token limit). The real cost is much lower.")
    if a.plan_only:
        say("\n  --plan-only: nothing was sent.")
        return 0

    # ---- run ---------------------------------------------------------------------
    say("\n[3] Running" + (" (dry run: made-up replies; a few are empty, blocked or cut on "
                           "purpose, and faults are injected)" if dry else ""))
    random.seed(SEED)
    stopped, batch = "", None
    t0 = time.time()
    try:
        need = n_todo > 0
        if is_gemini and mode == "batch" and not need:
            need = bool(Batch(ctx).open_jobs())     # jobs to settle or collect
        if is_gemini and need:
            info = gemini_preflight(ctx, need_batch=(mode == "batch"))
            ctx.hf_info = {"model_info": info}
            if mode == "standard":
                run_gemini_standard(ctx, all_work)
            else:
                batch = run_gemini_batch(ctx, works, submit=not a.no_submit, wait=bool(a.wait) or dry,
                                         poll_seconds=a.poll_seconds)
        elif not is_gemini and n_todo:
            run_hf(ctx, all_work)
        else:
            say("  nothing to run: 0 calls, 0 USD (everything is in the cache)")
    except (Stop, cost_ledger.BudgetExceeded, imglib.ImageError) as exc:
        stopped = str(exc)
        say(f"\n  STOPPED: {exc}")
        say("  What was answered before the stop is in the cache. Fix the cause and run the "
            "same command again.")

    # ---- result ------------------------------------------------------------------
    say("\n[4] Result")
    done = [ctx.cache.records[w["key"]] for w in all_work if w["key"] in ctx.cache.records]
    for line in report_table(done, {cid: len(ws) for cid, ws in works.items()}):
        say(line)
    missing = len(all_work) - len(done)
    if batch is not None:
        say("  batch jobs of this stage and model:")
        for line in batch.table() or ["    none"]:
            say(line)
    faults = {k: v for k, v in sorted(ctx.stats.items()) if v and k.startswith(("http_", "calls_", "usage_", "item_", "batch_", "given_"))}
    say(f"  retries and faults: {faults or 'none'}")
    if is_gemini:
        say(f"  {'DRY-RUN ledger (made-up numbers)' if dry else 'ledger'} change in this run: "
            f"{ctx.usd_run:.5f} USD (open Batch reservations included)   stage total {stage_spend(ledger, label):.5f} of {ctx.stage_cap:.2f}   "
            f"project total {cost_ledger.read_total(str(ledger)):.5f} of "
            f"{cost_ledger.HARD_CAP_USD:.0f} USD")
    complete = not stopped and missing == 0
    manifest = read_json(out_dir / "manifest.json", {"asset": "ASSET-8", "runs": []})
    manifest["runs"].append({
        "run_id": run_id, "utc": utc_now(), "dry_run": dry, "harness_version": HARNESS_VERSION,
        "git": git, "stage": label, "model_key": a.model, "model_id": m["model_id"],
        "mode": mode, "conditions": cond_ids, "prompt_version": prompts["prompt_version"],
        "prompts_sha256": prompts["sha256"], "settings": settings,
        "settings_sha256": ctx.settings_sha, "prices": prices, "price_source": price_source,
        "probe": {"file": probe["file"], "n": len(probe["items"]),
                  "created_utc": probe["created_utc"], "qa_ids_sha256": probe["qa_ids_sha256"]},
        "selection": {"where": a.where or [], "ids_file": a.ids_file or "", "sample": a.sample,
                      "limit": a.limit, "n_selected": len(items)},
        "english": {k: english[k] for k in ("path", "column", "sha256")} if english else None,
        "options": {k: options[k] for k in ("path", "sha256")} if options else None,
        "images": {"root": str(img_root), "prep": imglib.PREP_VERSION},
        "paid_tier_confirmed_by_runner": bool(a.paid_tier_confirmed),
        "n_items_x_conditions": len(all_work), "n_answered": len(done), "n_missing": missing,
        "stats": dict(ctx.stats), "usd_ledger_change": round(ctx.usd_run, 6),
        "seconds": round(time.time() - t0, 1), "backend_info": ctx.hf_info,
        "answer_field": ANSWER_FIELD, "stopped": stopped, "complete": complete})
    write_json(out_dir / "manifest.json", manifest)
    say(f"  wrote: {out_dir / 'cache.jsonl'}  (answers; the answer field is '{ANSWER_FIELD}')")
    say(f"         {out_dir / 'raw_calls.jsonl'}, manifest.json, stage_state.json"
        + (", batch/" if batch is not None else ""))

    if dry:
        say("\n[5] Dry-run checks")
        checks = [("every selected item has an answer", missing == 0 and not stopped)]
        again = Cache(out_dir / "cache.jsonl")
        checks.append(("a second run would make 0 calls (resume from the cache)",
                       all(w["key"] in again.records for w in all_work)))
        if is_gemini:
            led = cost_ledger.read_lines(str(ledger))
            res = sum(float(r["usd"]) for r in led if str(r.get("mode", "")).startswith("batch-"))
            checks.append(("every Batch reservation was released", abs(res) < 1e-9))
            f = ctx.transport.faults
            if mode == "standard" and len(all_work) >= 4:
                checks.append(("the HTTP 503 and the timeout were survived",
                               f["http_503"] == 1 and f["timeout"] == 1))
            if mode == "batch" and ctx.transport.n_create >= 3:
                checks.append(("a lost create call and a lost create answer did not cause "
                               "a second job for the same items",
                               f["create_lost"] == 1 and f["create_answer_lost"] == 1
                               and len({k for j in batch.jobs.values() if j.get("batch_name")
                                        for k in j["keys"]}) == len(all_work)))
        bad = 0
        for name, passed in checks:
            say(("  PASS  " if passed else "  FAIL  ") + name)
            bad += 0 if passed else 1
        say("  DRY RUN OK. Nothing was sent and nothing was spent." if not bad
            else "  DRY RUN FAILED: send this output to the ASSET-8 chat.")
        return 0 if not bad else 1
    if stopped:
        return 3
    if missing:
        say(f"\n  NOT COMPLETE: {missing} answers are missing. "
            + ("Batch jobs are still open: run the same command again later (add --wait to "
               "keep polling)." if batch is not None and batch.open_jobs()
               else "Run the same command again."))
        return 2
    say("\n  COMPLETE for this command.")
    return 0


def report(a):
    """Counts for one stage and model. Never shows a question or an answer."""
    root = Path(a.root)
    out_dir, label = layout(root, a.stage, a.run_name, a.model, bool(a.dry_run))
    cache = Cache(out_dir / "cache.jsonl")
    recs = list(cache.records.values())
    if a.conditions:
        keep = set(a.conditions.split(","))
        recs = [r for r in recs if r["condition_id"] in keep]
    say(f"Report for stage {label}, model {a.model}: {out_dir}")
    if not recs:
        say("  no answers yet")
        return 0
    for line in report_table(recs):
        say(line)
    for field in ("mode", "model_id", "model_version", "model_revision", "prompt_version",
                  "harness_version", "finish_reason"):
        c = Counter(str(r.get(field, "")) for r in recs if r.get(field) not in (None, ""))
        if c:
            say(f"  {field}: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    days = Counter(r.get("ts_utc", "")[:10] for r in recs)
    say("  answer dates (UTC): " + ", ".join(f"{k} {v}" for k, v in sorted(days.items())))
    say(f"  repeated keys in the file: {cache.duplicates}   unreadable lines: {cache.bad_lines}")
    say("  columns: cut = stopped by the length limit; <=8w = answers of 8 words or fewer "
        "(Gate D asks for 90%); script ok = letters are in the script of the requested "
        "language (an engineering check; fidelity comes from the scorer); think>0 = answers "
        "with thinking tokens.")
    return 0


def peek(a):
    """Show a few answers. Model output: not for the HT-1 pair before gate-A."""
    if not a.i_am_not_in_the_ht1_pair:
        raise Stop("peek shows model answers. Until gate-A is tagged, the two people who "
                   "write and check the lexicon must not see any model output (Design 4.2, "
                   "D125). If you are not one of them, add --i-am-not-in-the-ht1-pair.")
    out_dir, label = layout(Path(a.root), a.stage, a.run_name, a.model, bool(a.dry_run))
    recs = list(Cache(out_dir / "cache.jsonl").records.values())
    if a.conditions:
        recs = [r for r in recs if r["condition_id"] in set(a.conditions.split(","))]
    recs.sort(key=lambda r: sha256_hex(f"{SEED}|asset8-peek|{r['key']}"))
    say(f"MODEL OUTPUT. {min(a.n, len(recs))} of {len(recs)} answers, stage {label}, model {a.model}")
    for r in recs[:a.n]:
        say(f"  {r['qa_id']:<14} {r['condition_id']:<7} {r['qa_type']:<12} "
            f"finish={r.get('finish_reason') or '-':<10} words={r.get('n_words')}  "
            f"answer={r.get(ANSWER_FIELD)!r}")
    return 0


def prepare_images(a):
    root = Path(a.root)
    src = Path(a.images) if a.images else root / "data" / "images"
    dst = Path(a.images_1024) if a.images_1024 else root / "data" / "images_1024"
    files = set()
    for p in a.probe or []:
        files.update(r["image_file"] for r in load_probe(p)["items"])
    if a.all:
        for house in sorted(os.listdir(src)):
            if os.path.isdir(src / house):
                files.update(f"{house}/{name}" for name in os.listdir(src / house)
                             if name.lower().endswith((".jpg", ".jpeg")))
    if not files:
        raise Stop("Give --probe FILE (one or more) or --all.")
    say(f"prepare-images: {len(files)} images wanted   from {src}   to {dst}")
    say(f"rule: {imglib.PREP_VERSION}")
    s = imglib.prepare_many(sorted(files), src, dst, say=say, verify=bool(a.verify))
    problems = s.pop("problems")
    say("  " + ", ".join(f"{k} {v}" for k, v in s.items()))
    for rel, why in problems[:20]:
        say(f"  PROBLEM {rel}: {why}")
    say(f"  manifest: {imglib.manifest_path(dst)} ({len(imglib.read_manifest(dst))} rows)")
    if s["icc_failed"] or s["not_resized"]:
        say("  note: icc_failed = a colour profile could not be converted (pixels kept as "
            "stored); not_resized = images whose longer side is 1,024 px or less. Report "
            "both counts in the chat if they are not 0.")
    ok = not problems
    say("  RESULT: " + ("all wanted images are prepared" if ok else "NOT COMPLETE"))
    return 0 if ok else 2
