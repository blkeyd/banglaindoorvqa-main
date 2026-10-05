#!/usr/bin/env python3
"""
asset2_stage1_mt.py
BanglaIndoorVQA, ASSET-2 stage 1: machine translation of the Bangla question
strings into English (Design 4.6, DESIGN_ADDENDUM A1).

What it does
    1. Reads PILOT.json, P-CORE-CLOSED.json, R-TEST.json and P-NAME.json from
       Drive probes/ and builds the list of unique question strings, keyed by
       normalise(question_bn). Expected: 653 strings (649 + 4), 32 of them in
       PILOT. It stops if the counts differ from DESIGN_ADDENDUM A1.
    2. Sends the strings to Gemini in calls of about 20 strings, PILOT strings
       first, with one fixed prompt. A call holds question strings only: no
       answers, no images, no qa_ids. Naming questions (obj_name, object)
       never share a call with count, room or property questions, because
       those name objects and a riddle must not sit beside its answer.
    3. Keeps a cache keyed by (model_id, condition_id, prompt_version,
       string_id), logs every raw response, retries with backoff, can be
       stopped and started again, and meters cost against the shared ledger
       (40 USD hard stop for the project, 0.50 USD for this stage).
    4. Writes the stage 1 file english_questions_mt.csv, a PILOT-only copy,
       a report with counts and checks, and a manifest with the model ID, the
       settings and the run date.

What it never does
    It never prints a question or a translation. Stage 1 output is model
    output. Until Gate A it must not be opened by the two HT-1 people
    (Design 4.2, hard rule). The report file holds ids and counts only.

Where things live
    Code (this file, src/normalise.py, src/cost_ledger.py): GitHub.
    Everything this script writes: Google Drive only (question text and model
    output never enter Git).

Usage (from the repo root, in Colab)
    # 1. dry run, no network, no cost, 100 strings end to end
    python scripts/asset2_stage1_mt.py --scope all --limit 100 --dry-run
    # 2. billed: the 32 PILOT strings (2 calls)
    python scripts/asset2_stage1_mt.py --scope pilot --paid-tier-confirmed
    # 3. billed: the rest (32 calls); PILOT strings come from the cache
    python scripts/asset2_stage1_mt.py --scope all --paid-tier-confirmed

The API key is read from the environment variable GEMINI_API_KEY, which the
notebook fills from Colab Secrets. It is never written to any file.

NOTE: build_mt_request_text() below is the translation prompt of ASSET-2. It
is not the experiment prompt function build_prompt() of ASSET-8.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# Imports from src/ (one normalise(), one cost ledger, imported everywhere)
# --------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
for _cand in (_HERE.parent.parent / "src", _HERE.parent / "src", Path.cwd() / "src"):
    if (_cand / "normalise.py").exists() and (_cand / "cost_ledger.py").exists():
        sys.path.insert(0, str(_cand))
        break
try:
    from normalise import normalise
    import cost_ledger
except ImportError as _exc:  # pragma: no cover
    sys.exit(
        "STOP: cannot import src/normalise.py and src/cost_ledger.py "
        f"({_exc}). Run this script from the repo root, with both files in src/."
    )

# --------------------------------------------------------------------------
# Fixed settings. Changing any of these is a decision, not a tweak.
# --------------------------------------------------------------------------
SCRIPT_NAME = "asset2_stage1_mt"
SCRIPT_VERSION = "2026-10-05"

MODEL_ID = "gemini-3.6-flash"      # Design 5.1. Fallback order: Design 5.5.
CONDITION_ID = "A2-MT"             # cache key part; not an experiment condition
PROMPT_VERSION = "mt-v2"           # bump when PROMPT_HEAD changes (D129: rule 9 added)
THINKING_LEVEL = "minimal"         # Design 5.2
MAX_OUTPUT_TOKENS = 4096           # includes thinking tokens (Google docs)
BATCH_SIZE = 20                    # Design 4.6: about 20 strings per call
ROUND_SIZES = (BATCH_SIZE, 10, 5, 1)   # strings not accepted go to a smaller call
MODE = "standard"                  # Design 12.1: translation first pass, standard API
API_BASE = "https://generativelanguage.googleapis.com/v1beta"
# temperature, top_p and top_k are NOT sent: Google ignores them for this
# model and asks callers to remove them (developer guide of 23 July 2026).

# Paid tier, standard mode, USD per 1M tokens. Output price includes thinking.
PRICE_IN_PER_M = 0.75
PRICE_OUT_PER_M = 3.75
PRICE_VALID_UNTIL = "2026-12-31"   # Google: 1.50 and 7.50 from 2027-01-01
PRICE_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing (read 2026-10-04)"

STAGE_CAP_USD = 0.50               # Design 12.1: planning cost of this stage
HTTP_TIMEOUT_S = 120
MAX_HTTP_ATTEMPTS = 6
PAUSE_BETWEEN_CALLS_S = 1.0

# Expected sizes (PROJECT_STATE and DESIGN_ADDENDUM A1). The script stops if
# the files do not match, so that out-of-date probe files are caught here.
EXPECTED_ITEMS = {"PILOT": 100, "P-CORE-CLOSED": 1118, "R-TEST": 1449, "P-NAME": 1000}
EXPECTED_STRINGS = {
    "P-CORE-CLOSED": 342,
    "R-TEST adds": 307,
    "P-CORE-CLOSED and R-TEST": 649,
    "P-NAME adds": 4,
    "all": 653,
    "PILOT": 32,
}
PROBE_FILES = (  # the order fixes which file names a string first
    ("P-CORE-CLOSED", "P-CORE-CLOSED.json"),
    ("R-TEST", "R-TEST.json"),
    ("P-NAME", "P-NAME.json"),
    ("PILOT", "PILOT.json"),
)
TYPE_ORDER = ("scene_id", "obj_property", "count", "obj_name", "object")
# Naming questions (obj_name, object) never share a call with the other types.
# Count and property questions name objects; a riddle must not sit beside them.
NAMING_TYPES = ("obj_name", "object")

DEFAULT_ROOT = "/content/drive/MyDrive/BanglaIndoorVQA"

# --------------------------------------------------------------------------
# The one fixed translation prompt (Design 4.6). No dataset text in it.
# --------------------------------------------------------------------------
IMAGE_WORD_BN = "ছবি"            # the Bangla word for the image a question is about

PROMPT_HEAD = """You translate questions from Bangla into English. Each question is asked about one image: a photo taken inside a home in Bangladesh. You cannot see the image, and you do not need it.

Translate every question in the list at the end. Do not answer the questions. Follow all of these rules.

1. Faithful. Keep the meaning, the amount of detail and the difficulty exactly as in the Bangla. Do not add, remove or explain anything.
2. No new names. Never name an object that the Bangla question does not name. If the Bangla describes a thing without naming it (by its use, place, shape or material), describe it in the same way in English, and keep general words general.
3. Named objects. If the Bangla question itself names an object, use the ordinary everyday English word for that object. Do not make it more general or more specific.
4. Plain English. Write simple household English, the way one person asks another at home. Write one natural question that ends with a question mark.
5. The image. When the Bangla word ছবি means the image that the question is asked about, translate it as "image". When it means something else, such as a picture that hangs in the room, translate it normally.
6. Spelling. If the Bangla has a spelling mistake or lacks a question mark, translate the question that was meant. Do not comment on it.
7. Independent. Translate each question on its own. Never use one question in the list to work out what another question is about.
8. English only. The translation contains no Bangla letters, no notes, no brackets and no alternatives.
9. Fixed wording. Always translate দেখা যাচ্ছে with the word "visible" (is visible, are visible). Always translate জিনিস as "thing" and বস্তু as "object".

Output: JSON only. Return an object with one key, "translations", holding a list with exactly one entry for each input question, in the same order. Each entry has "id" (copied exactly from the input) and "en" (the English question).

Questions:
"""
PROMPT_SHA256 = hashlib.sha256(PROMPT_HEAD.encode("utf-8")).hexdigest()

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "en": {"type": "string"}},
                "required": ["id", "en"],
            },
        }
    },
    "required": ["translations"],
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_hex(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def say(msg=""):
    print(msg, flush=True)


class Stop(Exception):
    """A condition that must stop the run. The message tells the user why."""


def write_csv(path, header, rows):
    """Write a CSV in UTF-8 with BOM (opens cleanly in Excel and Sheets)."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)
    os.replace(tmp, path)


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def balanced_chunks(seq, size):
    """Split seq into ceil(n/size) chunks whose sizes differ by at most 1."""
    n = len(seq)
    if n == 0:
        return []
    k = math.ceil(n / size)
    base, extra = divmod(n, k)
    out, start = [], 0
    for i in range(k):
        step = base + (1 if i < extra else 0)
        out.append(seq[start:start + step])
        start += step
    return out


# --------------------------------------------------------------------------
# Step 1: the registry of unique question strings
# --------------------------------------------------------------------------
def load_probe(path, name):
    if not os.path.exists(path):
        raise Stop(f"Missing probe file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        raise Stop(f"{path}: no 'items' list. Is this the ASSET-5 output file?")
    if data.get("n") != len(items):
        raise Stop(f"{path}: header n={data.get('n')} but {len(items)} items.")
    for field in ("qa_id", "qa_type", "question_bn"):
        if any(field not in r for r in items):
            raise Stop(f"{path}: an item lacks the field '{field}'.")
    return {"name": name, "path": str(path), "items": items,
            "created_utc": data.get("created_utc", "")}


def build_registry(probes):
    """Return (strings, checks). strings is the ordered list of entries."""
    checks = []          # (label, passed, detail)
    text_of = {}         # qa_id -> question_bn, to prove the files agree
    conflicts = []
    reg = {}             # normalised string -> entry
    sets = {name: set() for name, _ in PROBE_FILES}

    for name, _fn in PROBE_FILES:
        probe = probes[name]
        ids = [r["qa_id"] for r in probe["items"]]
        if len(ids) != len(set(ids)):
            raise Stop(f"{name}: duplicate qa_id inside the file.")
        for r in probe["items"]:
            qa_id, raw = r["qa_id"], r["question_bn"]
            norm = normalise(raw)
            if not norm:
                raise Stop(f"{name}: empty question for {qa_id}.")
            if qa_id in text_of and text_of[qa_id] != raw:
                conflicts.append(qa_id)
            text_of.setdefault(qa_id, raw)
            sets[name].add(norm)
            e = reg.setdefault(norm, {
                "norm": norm, "surface": {}, "types": {}, "probes": [],
                "qa_ids": set(), "in_pilot": False, "in_pclust": False,
            })
            if name not in e["probes"]:
                e["probes"].append(name)
            if name == "PILOT":
                e["in_pilot"] = True
            if name == "P-NAME" and r.get("in_qualifying_cluster") is True:
                e["in_pclust"] = True
            if qa_id not in e["qa_ids"]:        # count every item once
                e["qa_ids"].add(qa_id)
                e["surface"][raw] = e["surface"].get(raw, 0) + 1
                e["types"][r["qa_type"]] = e["types"].get(r["qa_type"], 0) + 1

    # Item counts
    for name, _fn in PROBE_FILES:
        got, want = len(probes[name]["items"]), EXPECTED_ITEMS[name]
        checks.append((f"items in {name}", got == want, f"{got} (expected {want})"))

    # String counts (DESIGN_ADDENDUM A1, row for Design 4.6)
    pcc, rt, pn, pilot = (sets["P-CORE-CLOSED"], sets["R-TEST"], sets["P-NAME"],
                          sets["PILOT"])
    counted = {
        "P-CORE-CLOSED": len(pcc),
        "R-TEST adds": len(rt - pcc),
        "P-CORE-CLOSED and R-TEST": len(pcc | rt),
        "P-NAME adds": len(pn - (pcc | rt)),
        "all": len(reg),
        "PILOT": len(pilot),
    }
    for label, want in EXPECTED_STRINGS.items():
        got = counted[label]
        checks.append((f"unique strings, {label}", got == want,
                       f"{got} (expected {want})"))
    checks.append(("every PILOT string is in the full list",
                   pilot <= (pcc | rt | pn), f"{len(pilot - (pcc | rt | pn))} outside"))
    checks.append(("one question text per qa_id across the four files",
                   not conflicts, f"{len(conflicts)} conflicts"
                   + (": " + ", ".join(sorted(conflicts)[:10]) if conflicts else "")))

    # Finish the entries
    strings = []
    seen_ids = {}
    for norm, e in reg.items():
        # the surface form sent to the model: the most frequent one as written
        best = sorted(e["surface"].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        e["question_bn"] = best.strip()
        e["n_surface_forms"] = len(e["surface"])
        e["n_items"] = len(e["qa_ids"])
        e["example_qa_id"] = sorted(e["qa_ids"])[0]
        types_sorted = sorted(
            e["types"].items(),
            key=lambda kv: (-kv[1], TYPE_ORDER.index(kv[0]) if kv[0] in TYPE_ORDER else 99))
        e["primary_type"] = types_sorted[0][0]
        e["qa_types"] = "|".join(t for t, _ in types_sorted)
        e["family"] = "naming" if e["primary_type"] in NAMING_TYPES else "other"
        e["string_id"] = "q_" + sha256_hex(norm)[:12]
        if e["string_id"] in seen_ids:
            raise Stop("Two strings share one string_id. Lengthen the id hash.")
        seen_ids[e["string_id"]] = norm
        strings.append(e)

    def order_key(e):
        t = e["primary_type"]
        return (0 if e["in_pilot"] else 1,
                0 if e["family"] == "other" else 1,
                TYPE_ORDER.index(t) if t in TYPE_ORDER else 99,
                e["norm"])

    strings.sort(key=order_key)
    for i, e in enumerate(strings, 1):
        e["seq"] = i
    mixed = [e["string_id"] for e in strings
             if any(t in NAMING_TYPES for t in e["types"])
             and any(t not in NAMING_TYPES for t in e["types"])]
    checks.append(("no string is used by a naming type and another type",
                   not mixed, f"{len(mixed)} such strings"))
    return strings, checks


def call_groups(entries):
    """Split entries into the four call groups, in running order."""
    groups = []
    for pilot_flag, pname in ((True, "pilot"), (False, "rest")):
        for family in ("other", "naming"):
            part = [e for e in entries
                    if e["in_pilot"] == pilot_flag and e["family"] == family]
            if part:
                groups.append((f"{pname}-{family}", part))
    return groups


# --------------------------------------------------------------------------
# Step 2: the request
# --------------------------------------------------------------------------
def build_mt_request_text(batch):
    """batch: list of (local_id, bangla_question). Returns the full prompt."""
    lines = [json.dumps({"id": lid, "bn": bn}, ensure_ascii=False) for lid, bn in batch]
    return PROMPT_HEAD + "[\n" + ",\n".join(lines) + "\n]\n"


def build_body(prompt_text, use_schema):
    gen = {
        "maxOutputTokens": MAX_OUTPUT_TOKENS,
        "thinkingConfig": {"thinkingLevel": THINKING_LEVEL},
        "responseMimeType": "application/json",
    }
    if use_schema:
        gen["responseJsonSchema"] = RESPONSE_SCHEMA
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
        "generationConfig": gen,
    }


def worst_case_usd(prompt_text, price_in, price_out):
    """Upper bound on the cost of one call (used before the call)."""
    max_in = int(len(prompt_text) * 1.5) + 64
    return cost_ledger.cost_usd(max_in, MAX_OUTPUT_TOKENS, price_in, price_out)


# --------------------------------------------------------------------------
# Transports: the real API, and a dry-run stand-in with the same interface
# --------------------------------------------------------------------------
class TransportProblem(Exception):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind            # "timeout" or "connection"


class RealTransport:
    """Gemini API, generateContent, REST. The key travels in a header only."""

    dry = False

    def __init__(self, api_key):
        import requests  # in Colab by default
        self._requests = requests
        self._session = requests.Session()
        self._key = api_key
        self._headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        self.requests_version = getattr(requests, "__version__", "")

    def scrub(self, text):
        return (text or "").replace(self._key, "***")

    def _send(self, method, url, payload=None):
        try:
            if method == "GET":
                r = self._session.get(url, headers=self._headers, timeout=HTTP_TIMEOUT_S)
            else:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                r = self._session.post(url, headers=self._headers, data=data,
                                       timeout=HTTP_TIMEOUT_S)
        except self._requests.exceptions.Timeout as exc:
            raise TransportProblem("timeout", self.scrub(str(exc)))
        except self._requests.exceptions.RequestException as exc:
            raise TransportProblem("connection", self.scrub(str(exc)))
        text = r.content.decode("utf-8", errors="replace")
        return r.status_code, dict(r.headers), self.scrub(text)

    def get_model(self):
        return self._send("GET", f"{API_BASE}/models/{MODEL_ID}")

    def post_generate(self, body, meta):
        return self._send("POST", f"{API_BASE}/models/{MODEL_ID}:generateContent", body)


class DryRunTransport:
    """No network, no cost. Answers in the API's own response shape and
    injects three faults once each, so that the retry, re-queue and parsing
    paths are exercised: an HTTP 503, a dropped entry, a fenced JSON reply."""

    dry = True
    requests_version = "dry-run"

    def __init__(self):
        self.n_posts = 0
        self.did_503 = False
        self.did_drop = False
        self.did_fence = False

    def scrub(self, text):
        return text or ""

    def get_model(self):
        body = {"name": f"models/{MODEL_ID}", "displayName": "DRY RUN (no network)"}
        return 200, {}, json.dumps(body)

    def post_generate(self, body, meta):
        self.n_posts += 1
        if not self.did_503:
            self.did_503 = True
            return 503, {}, json.dumps({"error": {"code": 503, "status": "UNAVAILABLE",
                                                  "message": "dry-run injected fault"}})
        entries = [{"id": lid, "en": f"Dry run question {lid}?"}
                   for lid in meta["local_ids"]]
        if not self.did_drop and len(entries) > 1:
            self.did_drop = True
            entries = entries[:-1]
        text = json.dumps({"translations": entries})
        if self.did_drop and not self.did_fence and self.n_posts >= 4:
            self.did_fence = True
            text = "```json\n" + text + "\n```"
        resp = {
            "candidates": [{"content": {"role": "model", "parts": [{"text": text}]},
                            "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 0, "candidatesTokenCount": 0,
                              "thoughtsTokenCount": 0, "totalTokenCount": 0},
            "modelVersion": MODEL_ID + " (dry run)",
            "responseId": f"dry-{self.n_posts}",
        }
        return 200, {}, json.dumps(resp)


# --------------------------------------------------------------------------
# One call: HTTP retries, schema fallback, parsing, metering
# --------------------------------------------------------------------------
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
FATAL_HINTS = {
    401: "The API key was refused. Check the Colab secret GEMINI_API_KEY.",
    402: "No credit: the Prepay balance of the billing account is 0. Add "
         "credit in AI Studio (Billing). Do not switch to a free-tier key.",
    403: "Permission denied for this key or project. Check the key and that "
         "the Gemini API is enabled for its project.",
    404: f"The model '{MODEL_ID}' was not found for this key. Do not change "
         "the model yourself: Design 5.5 fixes the fallback order and needs "
         "a decision row.",
}


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
    # total - prompt is right whether or not the API counts thinking inside
    # candidatesTokenCount; fall back to the sum if no total is given.
    billed_out = total - prompt if total >= prompt and total > 0 else cand + thoughts
    return {"input_tokens": prompt, "output_tokens": cand, "thoughts_tokens": thoughts,
            "total_tokens": total, "billed_output_tokens": billed_out,
            "cached_tokens": int(u.get("cachedContentTokenCount") or 0),
            "reported": bool(u)}


def extract_text(resp):
    """Return (text, finish_reason, block_reason) from a generateContent reply."""
    block = (resp.get("promptFeedback") or {}).get("blockReason", "")
    cands = resp.get("candidates") or []
    if not cands:
        return "", "", block or "NO_CANDIDATE"
    c0 = cands[0]
    parts = (c0.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    return text, str(c0.get("finishReason", "")), block


def parse_translations(text):
    """Return a list of (id, en) pairs, or raise ValueError."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    data = json.loads(t)
    if isinstance(data, dict):
        data = data.get("translations")
    if not isinstance(data, list):
        raise ValueError("no list of translations in the reply")
    pairs = []
    for entry in data:
        if isinstance(entry, dict) and "id" in entry:
            pairs.append((str(entry.get("id")), entry.get("en")))
    return pairs


class Runner:
    """Holds the state of one run: transport, files, counters."""

    def __init__(self, transport, paths, run_id, price_in, price_out, stage_cap):
        self.t = transport
        self.paths = paths
        self.run_id = run_id
        self.price_in = price_in
        self.price_out = price_out
        self.stage_cap = stage_cap
        self.use_schema = True
        self.seq = 0
        self.gave_up_in_a_row = 0
        self.stats = Counter()
        self.usd_run = 0.0
        self.model_versions = Counter()
        self.sleep = (lambda s: None) if transport.dry else time.sleep

    # ---- budget -----------------------------------------------------------
    def check_budget(self, next_max):
        ledger = self.paths["ledger"]
        cost_ledger.assert_room(ledger, next_call_max_usd=next_max)
        stage_total = cost_ledger.read_total(ledger, script=SCRIPT_NAME)
        if stage_total + next_max > self.stage_cap:
            raise cost_ledger.BudgetExceeded(
                f"STAGE STOP: this stage has spent {stage_total:.4f} USD and the "
                f"next call could add up to {next_max:.4f} USD. The stage cap is "
                f"{self.stage_cap:.2f} USD (Design 12.1 plans {STAGE_CAP_USD:.2f} USD "
                "for the translation first pass). Ask the team lead before raising "
                "it (--stage-cap-usd).")

    def charge(self, call_id, usage, note=""):
        usd = cost_ledger.cost_usd(usage["input_tokens"], usage["billed_output_tokens"],
                                   self.price_in, self.price_out)
        cost_ledger.append(self.paths["ledger"], {
            "ts_utc": utc_now(), "script": SCRIPT_NAME, "run_id": self.run_id,
            "call_id": call_id, "model_id": MODEL_ID, "mode": MODE,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "thoughts_tokens": usage["thoughts_tokens"],
            "billed_output_tokens": usage["billed_output_tokens"],
            "price_in_per_m": self.price_in, "price_out_per_m": self.price_out,
            "usd": usd, "note": note,
        })
        self.usd_run += usd
        return usd

    def worst_usage(self, prompt_text):
        return {"input_tokens": int(len(prompt_text) * 1.5) + 64,
                "output_tokens": MAX_OUTPUT_TOKENS, "thoughts_tokens": 0,
                "billed_output_tokens": MAX_OUTPUT_TOKENS}

    # ---- HTTP level: one batch, with retries -------------------------------
    def fetch(self, call_id, prompt_text, meta, base_log):
        """Return the parsed reply (dict) of an HTTP 200, or None."""
        attempt = 0
        while attempt < MAX_HTTP_ATTEMPTS:
            attempt += 1
            body = build_body(prompt_text, self.use_schema)
            self.check_budget(worst_case_usd(prompt_text, self.price_in, self.price_out))
            t0 = time.time()
            try:
                status, headers, text = self.t.post_generate(body, meta)
            except TransportProblem as exc:
                self.stats["http_" + exc.kind] += 1
                note = ""
                if exc.kind == "timeout":
                    # The server may have answered and billed. Charge the worst
                    # case so that the meter stays an upper bound.
                    self.charge(call_id, self.worst_usage(prompt_text),
                                note="timeout: billed amount unknown, worst case charged")
                    note = "worst case charged"
                append_jsonl(self.paths["rawlog"], dict(
                    base_log, kind="transport_problem", ts_utc=utc_now(), attempt=attempt,
                    problem=exc.kind, message=str(exc)[:500], note=note,
                    schema_mode=self.use_schema, request=body))
                wait = min(60, 2 ** attempt) + random.uniform(0, 1)
                say(f"    {call_id} attempt {attempt}: {exc.kind}, wait {wait:.0f} s")
                self.sleep(wait)
                continue
            # Log the raw response first, exactly as received.
            append_jsonl(self.paths["rawlog"], dict(
                base_log, kind="response" if status == 200 else "http_error",
                ts_utc=utc_now(), attempt=attempt, http_status=status,
                elapsed_s=round(time.time() - t0, 2), schema_mode=self.use_schema,
                request=body, response_text=text))
            if status == 200:
                try:
                    resp = json.loads(text)
                except ValueError:
                    resp = None
                if isinstance(resp, dict):
                    return resp
                self.stats["http_200_not_json"] += 1
                self.charge(call_id, self.worst_usage(prompt_text),
                            note="HTTP 200 without JSON: worst case charged")
                self.sleep(min(60, 2 ** attempt))
                continue
            msg = api_error_message(text)
            if status == 400 and self.use_schema and "schema" in msg.lower():
                # The schema field was refused. Fall back to plain JSON mode: the
                # prompt already describes the format and the code checks it.
                self.use_schema = False
                self.stats["schema_fallback"] += 1
                say(f"    {call_id}: responseJsonSchema refused, using JSON mode only")
                attempt -= 1
                continue
            if status in RETRY_STATUS:
                self.stats[f"http_{status}"] += 1
                wait = min(60, 2 ** attempt) + random.uniform(0, 1)
                ra = headers.get("Retry-After") or headers.get("retry-after")
                if ra and str(ra).isdigit():
                    wait = max(wait, min(120, int(ra)))
                say(f"    {call_id} attempt {attempt}: HTTP {status}, wait {wait:.0f} s")
                self.sleep(wait)
                continue
            hint = FATAL_HINTS.get(status, "")
            raise Stop(f"HTTP {status} from the Gemini API on {call_id}. "
                       + (hint + " " if hint else "") + f"API message: {msg}")
        return None

    # ---- one API call for one batch ----------------------------------------
    def translate_batch(self, batch, group, round_no):
        """batch: list of registry entries. Returns {string_id: cache record}."""
        self.seq += 1
        call_id = f"{self.run_id}-c{self.seq:03d}"
        local = [(f"s{i}", e) for i, e in enumerate(batch, 1)]
        prompt_text = build_mt_request_text([(lid, e["question_bn"]) for lid, e in local])
        meta = {"local_ids": [lid for lid, _ in local]}
        base_log = {
            "run_id": self.run_id, "call_id": call_id, "group": group,
            "round": round_no, "model_id": MODEL_ID, "condition_id": CONDITION_ID,
            "prompt_version": PROMPT_VERSION, "dry_run": self.t.dry,
            "local_to_string_id": {lid: e["string_id"] for lid, e in local},
        }
        resp = self.fetch(call_id, prompt_text, meta, base_log)
        if resp is None:
            self.stats["calls_gave_up"] += 1
            self.gave_up_in_a_row += 1
            say(f"    {call_id}: no answer after {MAX_HTTP_ATTEMPTS} attempts")
            if self.gave_up_in_a_row >= 3:
                raise Stop("Three calls in a row got no answer. The API is not "
                           "reachable, or a quota is exhausted (a repeated HTTP 429 "
                           "can mean a free-tier quota: check the tier). Try later.")
            return {}
        self.gave_up_in_a_row = 0
        self.stats["calls_ok"] += 1

        usage = parse_usage(resp)
        if not usage["reported"] and not self.t.dry:
            usage = dict(self.worst_usage(prompt_text), total_tokens=0, cached_tokens=0,
                         reported=False)
            self.stats["usage_missing"] += 1
        usd = self.charge(call_id, usage,
                          note="" if usage["reported"] or self.t.dry
                          else "no usageMetadata: worst case charged")
        for k in ("input_tokens", "output_tokens", "thoughts_tokens", "billed_output_tokens"):
            self.stats[k] += usage[k]
        text, finish, block = extract_text(resp)
        model_version = str(resp.get("modelVersion", ""))
        self.model_versions[model_version] += 1

        accepted, problems = {}, {}
        pairs = None
        if block:
            problems["_reply"] = f"blocked ({block})"
            self.stats["reply_blocked"] += 1
        elif not text.strip():
            problems["_reply"] = "empty reply"
            self.stats["reply_empty"] += 1
        else:
            try:
                pairs = parse_translations(text)
            except ValueError as exc:
                problems["_reply"] = f"unparseable ({type(exc).__name__})"
                self.stats["reply_unparseable"] += 1
        if pairs is not None:
            seen = Counter(lid for lid, _ in pairs)
            value = dict(pairs)
            for lid, e in local:
                if seen[lid] == 0:
                    problems[e["string_id"]] = "missing"
                elif seen[lid] > 1:
                    problems[e["string_id"]] = "duplicate id"
                elif not isinstance(value[lid], str) or not value[lid].strip():
                    problems[e["string_id"]] = "empty"
                else:
                    accepted[e["string_id"]] = {
                        "key": cache_key(e["string_id"]), "model_id": MODEL_ID,
                        "condition_id": CONDITION_ID, "prompt_version": PROMPT_VERSION,
                        "prompt_sha256": PROMPT_SHA256, "thinking_level": THINKING_LEVEL,
                        "string_id": e["string_id"],
                        "mt_en": " ".join(value[lid].split()),
                        "call_id": call_id, "model_version": model_version,
                        "ts_utc": utc_now(),
                    }
            self.stats["reply_entries_rejected"] += sum(
                1 for k in problems if not k.startswith("_"))
        for rec in accepted.values():
            append_jsonl(self.paths["cache"], rec)
        append_jsonl(self.paths["rawlog"], dict(
            base_log, kind="result", ts_utc=utc_now(), usage=usage, usd=round(usd, 8),
            finish_reason=finish, block_reason=block, model_version=model_version,
            response_id=resp.get("responseId", ""), n_accepted=len(accepted),
            problems=problems))
        if finish and finish != "STOP":
            self.stats["finish_" + finish] += 1
        say(f"    {call_id} [{group}, round {round_no}]: {len(batch)} sent, "
            f"{len(accepted)} accepted; tokens in {usage['input_tokens']} out "
            f"{usage['output_tokens']} thinking {usage['thoughts_tokens']}; "
            f"{usd:.5f} USD" + (f"; finish {finish}" if finish not in ("", "STOP") else "")
            + (f"; blocked {block}" if block else ""))
        return accepted

    # ---- a group of strings: rounds with smaller and smaller calls ---------
    def translate_group(self, entries, group, done):
        pending = [e for e in entries if e["string_id"] not in done]
        for round_no, size in enumerate(ROUND_SIZES, 1):
            if not pending:
                break
            batches = balanced_chunks(pending, size)
            say(f"  {group}: round {round_no}, {len(pending)} strings in "
                f"{len(batches)} calls")
            for batch in batches:
                done.update(self.translate_batch(batch, group, round_no))
                self.sleep(PAUSE_BETWEEN_CALLS_S)
            pending = [e for e in pending if e["string_id"] not in done]
        return pending      # strings that failed in every round


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------
def cache_key(string_id):
    return f"{MODEL_ID}|{CONDITION_ID}|{PROMPT_VERSION}|{string_id}"


def load_cache(path, valid_ids):
    """Return ({string_id: record} for the current key, n_other_lines)."""
    done, other = {}, 0
    if not os.path.exists(path):
        return done, other
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                raise Stop(f"Cache line {n} in {path} is not JSON. Remove that line.")
            sid = rec.get("string_id")
            if rec.get("key") == cache_key(sid):
                if (rec.get("prompt_sha256") != PROMPT_SHA256
                        or rec.get("thinking_level") != THINKING_LEVEL):
                    raise Stop(
                        f"The cache holds translations made as {PROMPT_VERSION} with "
                        "another prompt text or thinking level. One version must mean "
                        "one prompt. Give the new prompt a new PROMPT_VERSION (for "
                        "example mt-v2) and run again; every string is then "
                        "translated again with it.")
                if sid in valid_ids and rec.get("mt_en"):
                    done[sid] = rec
                    continue
            other += 1          # another prompt version or model, or an old string
    return done, other


# --------------------------------------------------------------------------
# Quality flags for the post-editors (attention marks, not judgements)
# --------------------------------------------------------------------------
def qc_flags(bn, en):
    flags = []
    if any("\u0980" <= ch <= "\u09ff" for ch in en):
        flags.append("bangla_letters")
    if not en.endswith("?"):
        flags.append("no_question_mark")
    if en.count("?") > 1:
        flags.append("several_questions")
    if any(ch in en for ch in "()[]{}"):
        flags.append("brackets")
    if "/" in en:
        flags.append("slash")
    low = en.lower()
    if IMAGE_WORD_BN in bn and "image" not in low:
        flags.append("no_image_word")
    if any(w in low.replace("?", " ").replace(",", " ").split()
           for w in ("picture", "pictures", "photo", "photos", "photograph")):
        flags.append("picture_or_photo")
    nb, ne = len(bn.split()), len(en.split())
    if ne > 2.5 * nb + 4 or ne < 0.6 * nb:
        flags.append("length_outlier")
    if any(ord(ch) > 126 for ch in en) and "bangla_letters" not in flags:
        flags.append("non_ascii")
    return flags


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------
MT_HEADER = ["seq", "string_id", "in_pilot", "in_pclust", "qa_types", "probes",
             "n_items", "question_bn", "mt_en", "mt_status", "qc_flags",
             "n_surface_forms", "question_bn_norm", "model_id", "prompt_version",
             "call_id", "translated_utc"]
REG_HEADER = ["seq", "string_id", "in_pilot", "in_pclust", "qa_types", "probes",
              "n_items", "n_surface_forms", "example_qa_id", "question_bn",
              "question_bn_norm"]


def yn(flag):
    return "yes" if flag else "no"


def write_outputs(paths, strings, done, failed_ids):
    reg_rows, mt_rows, flag_counts, flagged = [], [], Counter(), []
    for e in strings:
        common = [e["seq"], e["string_id"], yn(e["in_pilot"]), yn(e["in_pclust"]),
                  e["qa_types"], "|".join(e["probes"]), e["n_items"]]
        reg_rows.append(common + [e["n_surface_forms"], e["example_qa_id"],
                                  e["question_bn"], e["norm"]])
        rec = done.get(e["string_id"])
        if rec:
            flags = qc_flags(e["question_bn"], rec["mt_en"])
            for fl in flags:
                flag_counts[fl] += 1
            if flags:
                flagged.append((e["string_id"], flags))
            mt_rows.append(common + [e["question_bn"], rec["mt_en"], "ok", "|".join(flags),
                                     e["n_surface_forms"], e["norm"], MODEL_ID,
                                     PROMPT_VERSION, rec.get("call_id", ""),
                                     rec.get("ts_utc", "")])
        else:
            status = "failed" if e["string_id"] in failed_ids else "pending"
            mt_rows.append(common + [e["question_bn"], "", status, "",
                                     e["n_surface_forms"], e["norm"], MODEL_ID,
                                     PROMPT_VERSION, "", ""])
    write_csv(paths["registry"], REG_HEADER, reg_rows)
    write_csv(paths["mt_all"], MT_HEADER, mt_rows)
    write_csv(paths["mt_pilot"], MT_HEADER, [r for r in mt_rows if r[2] == "yes"])
    return flag_counts, flagged


def main(argv=None):
    ap = argparse.ArgumentParser(description="ASSET-2 stage 1: machine translation "
                                 "of the Bangla question strings (Design 4.6).")
    ap.add_argument("--scope", choices=("pilot", "all"), required=True,
                    help="pilot: the 32 PILOT strings. all: every string; PILOT first.")
    ap.add_argument("--probes-dir", default=f"{DEFAULT_ROOT}/probes")
    ap.add_argument("--out-dir", default=f"{DEFAULT_ROOT}/english")
    ap.add_argument("--ledger", default=f"{DEFAULT_ROOT}/outputs/api_cost_ledger.jsonl")
    ap.add_argument("--limit", type=int, default=0,
                    help="translate only the first N strings of the scope (tests)")
    ap.add_argument("--dry-run", action="store_true",
                    help="no network and no cost; writes to <out-dir>/stage1_dryrun")
    ap.add_argument("--paid-tier-confirmed", action="store_true",
                    help="you checked in AI Studio that the key's project is on a "
                         "paid tier (Design 5.5). Needed for every billed run.")
    ap.add_argument("--stage-cap-usd", type=float, default=STAGE_CAP_USD)
    ap.add_argument("--price-in", type=float, default=None,
                    help="USD per 1M input tokens, if Google's price has changed")
    ap.add_argument("--price-out", type=float, default=None,
                    help="USD per 1M output tokens (thinking included)")
    ap.add_argument("--allow-count-mismatch", action="store_true",
                    help="run although the counts differ from DESIGN_ADDENDUM A1. "
                         "Needs a decision row.")
    args = ap.parse_args(argv)

    dry = args.dry_run
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ("-dry" if dry else "")
    say("=" * 74)
    say(f"ASSET-2 stage 1: machine translation   script {SCRIPT_VERSION}")
    say(f"mode: {'DRY RUN (no network, no cost)' if dry else 'BILLED RUN'}   "
        f"scope: {args.scope}   run_id: {run_id}")
    say(f"model: {MODEL_ID}   thinking: {THINKING_LEVEL}   prompt: {PROMPT_VERSION} "
        f"(sha256 {PROMPT_SHA256[:12]})")
    say("=" * 74)

    # ---- paths -----------------------------------------------------------
    out_root = Path(args.out_dir)
    if dry:
        out_root = out_root / "stage1_dryrun"
        if out_root.exists() and out_root.name == "stage1_dryrun":
            shutil.rmtree(out_root)     # every dry run starts clean
    work = out_root / "stage1"
    work.mkdir(parents=True, exist_ok=True)
    paths = {
        "registry": out_root / "asset2_strings.csv",
        "mt_all": out_root / "english_questions_mt.csv",
        "mt_pilot": out_root / "english_questions_mt_PILOT.csv",
        "cache": work / "mt_cache.jsonl",
        "rawlog": work / "raw_calls.jsonl",
        "manifest": work / "stage1_manifest.json",
        "report": work / "asset2_stage1_report.txt",
        "ledger": (work / "dryrun_ledger.jsonl") if dry else Path(args.ledger),
    }

    # ---- step 1: registry -------------------------------------------------
    say("\n[1] Reading the probe files and building the string list")
    probes = {name: load_probe(Path(args.probes_dir) / fn, name) for name, fn in PROBE_FILES}
    for name, _fn in PROBE_FILES:
        p = probes[name]
        say(f"  read {name}: {len(p['items'])} items   created {p['created_utc']}   {p['path']}")
    strings, checks = build_registry(probes)
    report = []
    all_pass = True
    for label, passed, detail in checks:
        line = f"  {'PASS' if passed else 'FAIL'}  {label}: {detail}"
        say(line)
        report.append(line)
        all_pass = all_pass and passed
    if not all_pass and not args.allow_count_mismatch:
        raise Stop("A check failed. The probe files do not match DESIGN_ADDENDUM A1. "
                   "Use the files in Drive probes/ (ASSET-5 output of 3 Oct 2026). "
                   "Nothing was sent and nothing was spent.")
    by_id = {e["string_id"]: e for e in strings}
    type_counts = Counter(e["primary_type"] for e in strings)
    say("  strings by main type: " + ", ".join(f"{t} {type_counts[t]}" for t in TYPE_ORDER))

    # ---- scope -------------------------------------------------------------
    scope = [e for e in strings if e["in_pilot"]] if args.scope == "pilot" else list(strings)
    if args.limit and args.limit > 0:
        scope = scope[:args.limit]
    done, other_lines = load_cache(paths["cache"], set(by_id))
    todo = [e for e in scope if e["string_id"] not in done]
    n_calls_plan = sum(len(balanced_chunks(part, BATCH_SIZE))
                       for _name, part in call_groups(todo))
    say(f"\n[2] Scope '{args.scope}': {len(scope)} strings; in the cache {len(scope) - len(todo)}; "
        f"to translate {len(todo)}; planned calls {n_calls_plan}")
    if other_lines:
        say(f"  note: {other_lines} cache lines belong to another prompt version, "
            "model or string list and are not used")

    # ---- prices and safety checks before any billed call ------------------
    price_in = args.price_in if args.price_in is not None else PRICE_IN_PER_M
    price_out = args.price_out if args.price_out is not None else PRICE_OUT_PER_M
    price_source = PRICE_SOURCE
    if args.price_in is not None or args.price_out is not None:
        price_source = "given on the command line by the runner"
    transport = None
    model_info = {}
    if dry:
        transport = DryRunTransport()
    elif todo:
        if not args.paid_tier_confirmed:
            raise Stop("Billed run refused: add --paid-tier-confirmed after you have "
                       "checked, in AI Studio (API keys or Projects page), that the "
                       "project of this key shows a paid tier. gemini-3.6-flash also "
                       "has a free tier, and free-tier content is used to improve "
                       "Google's products (Design 5.5).")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today > PRICE_VALID_UNTIL and (args.price_in is None or args.price_out is None):
            raise Stop(f"The built-in prices were valid until {PRICE_VALID_UNTIL}. Read "
                       "https://ai.google.dev/gemini-api/docs/pricing and pass "
                       "--price-in and --price-out.")
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise Stop("GEMINI_API_KEY is not set. In the notebook, before this "
                       "command: os.environ['GEMINI_API_KEY'] = "
                       "userdata.get('GEMINI_API_KEY')  (Colab Secrets).")
        transport = RealTransport(api_key)
        say(f"  prices used: {price_in} USD per 1M input, {price_out} USD per 1M output "
            f"(thinking included). Source: {price_source}")
        total_before = cost_ledger.assert_room(paths["ledger"], 0.0)
        say(f"  ledger: {paths['ledger']}   project total so far {total_before:.4f} USD "
            f"(hard stop {cost_ledger.HARD_CAP_USD:.0f} USD)")
        try:
            status, _h, text = transport.get_model()
        except TransportProblem as exc:
            raise Stop(f"Model check could not reach the Gemini API ({exc.kind}). "
                       "Check the network and run again. Nothing was spent.")
        if status != 200:
            hint = FATAL_HINTS.get(status, "")
            raise Stop(f"Model check failed with HTTP {status}. "
                       + (hint + " " if hint else "")
                       + f"API message: {api_error_message(text)}")
        try:
            model_info = json.loads(text)
        except ValueError:
            raise Stop("Model check: the API answered HTTP 200 without JSON. "
                       "Run again. Nothing was spent.")
        say(f"  model check: {model_info.get('name')} is served "
            f"(displayName {model_info.get('displayName')}, version {model_info.get('version')})")

    # ---- step 3: translate --------------------------------------------------
    failed = []
    runner = None
    if todo:
        say("\n[3] Translating" + (" (dry run: made-up text, three injected faults)" if dry else ""))
        random.seed(42)
        runner = Runner(transport, paths, run_id, price_in, price_out, args.stage_cap_usd)
        try:
            for gname, part in call_groups(scope):
                failed += runner.translate_group(part, gname, done)
        except (cost_ledger.BudgetExceeded, Stop) as exc:
            say(f"\n  STOPPED: {exc}")
            say("  What was translated before the stop is in the cache. Fix the cause "
                "and run the same command again.")
            failed = None
    else:
        say("\n[3] Nothing to translate: 0 calls, 0 USD (everything is in the cache)")

    # ---- step 4: outputs ----------------------------------------------------
    say("\n[4] Writing the output files")
    failed_ids = {e["string_id"] for e in failed} if failed else set()
    flag_counts, flagged = write_outputs(paths, strings, done, failed_ids)
    n_ok_all = sum(1 for e in strings if e["string_id"] in done)
    n_ok_scope = sum(1 for e in scope if e["string_id"] in done)
    n_pilot = sum(1 for e in strings if e["in_pilot"])
    n_ok_pilot = sum(1 for e in strings if e["in_pilot"] and e["string_id"] in done)
    english = Counter(done[e["string_id"]]["mt_en"].lower() for e in strings
                      if e["string_id"] in done)
    shared = sum(1 for _t, c in english.items() if c > 1)

    stats = runner.stats if runner else Counter()
    usd_run = runner.usd_run if runner else 0.0
    report += [
        "",
        f"run_id {run_id}   utc {utc_now()}   script {SCRIPT_NAME} {SCRIPT_VERSION}",
        f"mode {'dry run' if dry else 'billed'}   scope {args.scope}"
        + (f"   limit {args.limit}" if args.limit else ""),
        f"model_id {MODEL_ID}   api generateContent (REST v1beta)   mode {MODE}",
        f"model_version returned: {dict(runner.model_versions) if runner else {}}",
        f"thinking_level {THINKING_LEVEL}   max_output_tokens {MAX_OUTPUT_TOKENS}   "
        f"sampling parameters: none sent (ignored by this model)",
        f"prompt_version {PROMPT_VERSION}   prompt_sha256 {PROMPT_SHA256}",
        f"structured output: {'JSON schema' if (runner is None or runner.use_schema) else 'JSON mode only (schema refused)'}",
        "",
        f"strings in the list: {len(strings)}",
        f"translated so far (all scopes): {n_ok_all} of {len(strings)}",
        f"translated in this scope: {n_ok_scope} of {len(scope)}",
        f"PILOT strings translated: {n_ok_pilot} of {n_pilot}",
        f"failed in this run: {len(failed_ids)}" + (
            "  " + ", ".join(sorted(failed_ids)) if failed_ids else ""),
        f"calls answered: {stats['calls_ok']}   calls given up: {stats['calls_gave_up']}",
        "retries and faults: " + (", ".join(
            f"{k} {v}" for k, v in sorted(stats.items())
            if v and k.startswith(("http_", "reply_", "finish_", "schema_", "usage_")))
            or "none"),
        f"tokens: input {stats['input_tokens']}, output {stats['output_tokens']}, "
        f"thinking {stats['thoughts_tokens']}, billed output {stats['billed_output_tokens']}",
        f"cost of this run: {usd_run:.5f} USD   (prices {price_in} in, {price_out} out per 1M)",
    ]
    if not dry:
        report.append(
            f"cost of this stage so far: "
            f"{cost_ledger.read_total(paths['ledger'], script=SCRIPT_NAME):.5f} USD "
            f"(stage cap {args.stage_cap_usd:.2f})   project total: "
            f"{cost_ledger.read_total(paths['ledger']):.5f} USD (hard stop "
            f"{cost_ledger.HARD_CAP_USD:.0f})")
    report += [
        "",
        "attention flags for the post-editors (counts; the ids are in the report file)"
        + (" [dry run: the text is made up, so these counts mean nothing]:" if dry else ":"),
        "  " + (", ".join(f"{k} {v}" for k, v in sorted(flag_counts.items())) or "none"),
        f"English strings shared by two or more Bangla strings: {shared}",
    ]
    complete = (failed is not None) and not failed_ids and n_ok_scope == len(scope)
    report += ["", f"RESULT: {'COMPLETE for this scope' if complete else 'NOT COMPLETE'}"]
    with open(paths["report"], "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
        f.write("\nflagged strings (ids only; the text is in english_questions_mt.csv):\n")
        for sid, fl in flagged:
            f.write(f"  {sid}: {'|'.join(fl)}\n")

    manifest = {"asset": "ASSET-2 stage 1", "runs": []}
    if os.path.exists(paths["manifest"]):
        with open(paths["manifest"], "r", encoding="utf-8") as f:
            manifest = json.load(f)
    manifest["runs"].append({
        "run_id": run_id, "utc": utc_now(), "dry_run": dry, "scope": args.scope,
        "limit": args.limit, "script": SCRIPT_NAME, "script_version": SCRIPT_VERSION,
        "model_id": MODEL_ID, "model_info": model_info,
        "model_versions_returned": dict(runner.model_versions) if runner else {},
        "api": "generateContent, REST v1beta", "mode": MODE,
        "thinking_level": THINKING_LEVEL, "max_output_tokens": MAX_OUTPUT_TOKENS,
        "sampling_parameters_sent": "none", "batch_size": BATCH_SIZE,
        "condition_id": CONDITION_ID, "prompt_version": PROMPT_VERSION,
        "prompt_sha256": PROMPT_SHA256, "prompt_head": PROMPT_HEAD,
        "json_schema_used": bool(runner is None or runner.use_schema),
        "paid_tier_confirmed_by_runner": bool(args.paid_tier_confirmed),
        "price_in_per_m": price_in, "price_out_per_m": price_out,
        "price_source": price_source,
        "probe_files": {n: {"path": probes[n]["path"], "n": len(probes[n]["items"]),
                            "created_utc": probes[n]["created_utc"]} for n, _ in PROBE_FILES},
        "n_strings": len(strings), "n_scope": len(scope),
        "n_translated_scope": n_ok_scope, "n_translated_all": n_ok_all,
        "failed_string_ids": sorted(failed_ids), "stats": dict(stats),
        "usd_run": round(usd_run, 6), "complete": complete,
        "python": platform.python_version(),
        "requests": getattr(transport, "requests_version", "") if transport else "",
    })
    with open(paths["manifest"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    for line in report[len(checks):]:
        say("  " + line if line else "")
    say("\n  wrote:")
    for key in ("registry", "mt_all", "mt_pilot", "report", "manifest", "cache", "rawlog"):
        say(f"    {paths[key]}")
    say(f"    ledger: {paths['ledger']}")

    # ---- dry run only: prove that a second run costs nothing ---------------
    if dry and failed is not None:
        again, _ = load_cache(paths["cache"], set(by_id))
        left = [e for e in scope if e["string_id"] not in again]
        faults = (transport.did_503 and (transport.did_drop or len(scope) < 2)
                  and (transport.did_fence or transport.n_posts < 4))
        say("\n[5] Dry-run checks")
        say(f"  {'PASS' if not left else 'FAIL'}  resume: a second run would make "
            f"{len(balanced_chunks(left, BATCH_SIZE))} calls for {len(left)} strings")
        say(f"  {'PASS' if faults else 'FAIL'}  injected faults were hit and survived "
            f"(HTTP 503: {transport.did_503}, dropped entry: {transport.did_drop}, "
            f"fenced reply: {transport.did_fence})")
        say(f"  {'PASS' if complete else 'FAIL'}  every string in the scope has a translation")
        if left or not faults or not complete:
            return 1
        say("  DRY RUN OK. No network call was made and nothing was spent.")
        return 0

    if failed is None:
        return 3
    if not complete:
        say("\n  NOT COMPLETE: run the same command again. If the same strings fail "
            "again, send the report file (it holds no question text).")
        return 2
    say("\n  COMPLETE for this scope. Do not open the translations if you are one "
        "of the two HT-1 people and Gate A is not tagged yet.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Stop as exc:
        say(f"\nSTOP: {exc}")
        sys.exit(1)
    except (cost_ledger.BudgetExceeded, cost_ledger.LedgerError) as exc:
        say(f"\nSTOP: {exc}")
        sys.exit(1)
