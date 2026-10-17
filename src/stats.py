"""
src/stats.py  (Analysis notebooks, before Gate E)
BanglaIndoorVQA main paper: THE shared statistics functions.
Design v2.1 sections 9.1 to 9.4 and 9.7, the figures and tables of section 13;
DESIGN_ADDENDUM B4, B6, B9, C4, C5, F1, F7, F11, H1, H5, I5; decisions D230 to
D239 (addendum J).

scripts/analysis.py calls these functions. The HT-4 chat imports ht4_stats() from
here for its pass reading, so there is one implementation. Import this file; never
copy a function out of it.

What it implements
    9.1  P1 to P8 and P12; Holm within each model over P12, P3, P4, P5, P6, P7;
         P1 and P2 as estimates with CIs; P8 a single test.
    9.2  the house-clustered paired bootstrap (10,000 resamples, seed 42); the GEE
         checks (statsmodels, binomial, exchangeable); the template_id fit; EXP-R
         per seed and on seed-averaged item scores.
    9.3  tiers, fallback, transfer rate, the error budget (A1 to A4, B, C) with the
         N-FCB0 guessing level, the two-way split and the D204 count.
    9.4  every pre-specified secondary analysis (items 1 to 13).
    13   F2 to F6, T1, T2 (the columns the records hold), T4, T5 and the
         supplement tables S1 to S10. F1 and T3 are made by hand.

Rules kept here
    - Nothing printed holds a question, an answer, a gold or a model answer. Files
      with text are written only to the run folder in Drive.
    - An item joins its lexicon object through the scorer: gold_name_id of the
      record, or resolve_gold on the probe record for forced choice (C4).
    - Excluded items carry gold_status excluded and enter no number (D110, D185).
    - Placeholder mode is read from the records (stage "placeholder"); every
      figure and table then carries a visible PLACEHOLDER mark.
"""

import csv
import glob
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import unicodedata
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STATS_VERSION = "stats-2026-10-17"
SEED = 42
N_BOOT = 10000
PLACEHOLDER = "PLACEHOLDER"
STAGE_PLACEHOLDER = "placeholder"
STAGE_MAIN = "main"

# --------------------------------------------------------------------------
# 0. Fixed values (Design 3, 7, 9; PROJECT_STATE; addendum C1, F7, H1, I2)
# --------------------------------------------------------------------------

MODELS = ("gemini", "qwen", "gemma")
FT_BASE = "qwen"
FT_SEEDS = ("qwen_ft_s42", "qwen_ft_s43", "qwen_ft_s44")
MODEL_NAMES = {"gemini": "Gemini 3.6 Flash", "qwen": "Qwen3.5-4B",
               "gemma": "Gemma 4 E4B", "qwen_ft_s42": "Qwen3.5-4B LoRA s42",
               "qwen_ft_s43": "Qwen3.5-4B LoRA s43",
               "qwen_ft_s44": "Qwen3.5-4B LoRA s44"}

COND_L = ("L-BB", "L-BE", "L-EB", "L-EE", "L-BB0", "L-EE0")
COND_L_IMG = ("L-BB", "L-BE", "L-EB", "L-EE")
COND_N_OPEN = ("N-BB", "N-BE", "N-BBs")
COND_N_FC = ("N-FCB", "N-FCB0", "N-FCE", "N-FCE0")
COND_R = ("R-BB", "R-EE", "R-BB0")
ZS_CONDS = COND_L + COND_N_OPEN + COND_N_FC
ALL_CONDS = ZS_CONDS + COND_R
EXPERIMENT = dict([(c, "EXP-L") for c in COND_L] + [(c, "EXP-N") for c in
                  COND_N_OPEN + COND_N_FC] + [(c, "EXP-R") for c in COND_R])
NO_IMAGE = ("L-BB0", "L-EE0", "N-FCB0", "N-FCE0", "R-BB0")

PROBE_FILES = {"P-CORE-CLOSED": "P-CORE-CLOSED.json", "R-TEST": "R-TEST.json",
               "P-NAME": "P-NAME.json", "P-CLUST": "P-CLUST.json",
               "P-FCE": "P-FCE.json"}
ID_LISTS = {"P-CORE-CLOSED": "P-CORE-CLOSED.ids.txt", "R-TEST": "R-TEST.ids.txt",
            "P-NAME": "P-NAME.ids.txt", "P-CLUST": "P-CLUST.ids.txt",
            "P-FCE": "P-FCE.ids.txt", "NAME-REF": "NAME-REF.ids.txt"}
RUN_SIZES = {"P-CORE-CLOSED": 1118, "R-TEST": 1449, "P-NAME": 1000,
             "P-CLUST": 492, "P-FCE": 454, "NAME-REF": 200}
# Scored sizes under the 28-row excluded list (PROJECT_STATE, addendum H1). They
# are pinned together with the list's sha256: a new row fails the check until the
# design-merge chat updates both pins with its decision row (D231).
SCORED_SIZES = {"P-CORE-CLOSED": 1109, "R-TEST": 1435, "P-NAME": 991,
                "P-CLUST": 483, "P-FCE": 446, "NAME-REF": 199}
RUN_HOUSES = {"P-CORE-CLOSED": 68, "R-TEST": 15, "P-NAME": 61, "P-CLUST": 57,
              "P-FCE": 57, "NAME-REF": 22}
BOOT_HOUSES = {"EXP-L": 68, "EXP-N": 61, "EXP-R": 15}       # Design 3.5
DATASET_COUNTS = {"images": 4406, "houses": 68, "qa": 18276, "objects": 3620}

PIN_SHA = {   # a value shorter than 64 characters is a prefix (as recorded)
    "lexicon.csv": "73b6bc48f1bda27863fbad69e26830c5f484d50702543b66297679bc07bc7448",
    "closed_vocab.csv": "de36f2905e43784704f8765c9886da1e9a9524d295ff10fac6c75b21f50e4b90",
    "excluded_items.csv": "8c51763fad6e3c48f0ae1e137573d00e1d0e79274bf959d9bb8e0bca96b37cdb",
    "house_assignment.csv": "1dde22c04bddadae85f262e97c8429eb1b404ec22a4fe12050716287d478941c",
    "P-CORE-CLOSED.json": "ea242d5b8ade",
    "R-TEST.json": "30cd10956e56",
    "P-NAME.json": "fc4bfb1dbda1f6f3a554e196dc049c90a9a72af4376170420f50b5decd7e9035",
    "P-CLUST.json": "7759dcce19e4c36596d6b39a39be8feefa467cf2088d5c03bdcace99a70f296f",
    "P-FCE.json": "807e987b4b656586ee2955efb421e8e6cd0aae27790a11a4c4f3f4f7210479ae",
    "probe_manifest_11.json": "5916b56ba317807fdac827edbb6cb5b4ffb3cb6ac62c1ef68fc5ee1904be3f4e",
    "ht6_swaps.csv": "9b2f2a8569b9a8682cbf011a05295c501bd3071b8d5a7c1dcc51690c46f4f957",
    "banglaindoorvqa.json": "30017e400212",
    "english_questions_final.csv": "30f8286b4673cdf1a50a3ae1f40db8762d0fbb378b2a82ee0a9240806f87361c",
    "fc_options.jsonl": "5288ba63033083b10beab8eb5949f3f6a2ab3968a4baf8eb8bb2e1b0fdcf2e96",
}
SCORER_VERSION_EXPECTED = "0.2.2-stage2"

# P-CLUST clusters (Design 3.4, run counts), matched on the start of the lexicon
# cluster_id after NFC.
PCLUST_CLUSTERS = (("সংরক্ষণ-আসবাব", 138), ("বৈদ্যুতিক সংযোগ", 88),
                   ("বাথরুমের কল", 79), ("টেবিলের ধরন", 61),
                   ("টব / ফুলদানি", 43), ("রান্নার পাত্র", 42),
                   ("ফ্যানের ধরন", 41))
COOKING_POTS = "রান্নার পাত্র"      # P-FCE: 4 items, a count only (H5)

NAMING_TYPES = ("obj_name", "object")
CLOSED_TYPES = ("count", "object", "obj_name", "obj_property", "scene_id")
LABELS_NAMING = ("tier1", "tier2", "sibling", "ambiguous", "english_copy",
                 "other_wrong")
LABELS_OTHER = ("correct", "wrong", "invalid")
# Seven segments of F5 (D233): Tier 1 split by the language of the matched term.
SEGMENTS = ("tier1_req", "tier1_other", "tier2", "sibling", "ambiguous",
            "english_copy", "other_wrong")
SEGMENT_TEXT = {"tier1_req": "Tier 1, requested language",
                "tier1_other": "Tier 1, other language", "tier2": "Tier 2 (generic)",
                "sibling": "Sibling", "ambiguous": "Ambiguous",
                "english_copy": "English copy", "other_wrong": "Other wrong"}
BUDGET_CATS = ("A1", "A2", "A3", "A4", "B", "C")
# origin3 groups; uncertain is left out of the origin analysis (Design 4.3 step 5)
ORIGINS = ("english", "other_loan", "inherited")
BUDGET_TEXT = {"A1": "A1 wrong language", "A2": "A2 English copy",
               "A3": "A3 generic name", "A4": "A4 sibling or other",
               "B": "B lexical access", "C": "C not known"}

# Contracts for files written by later chats (D238).
HT4_COLUMNS = ("sample_id", "model_key", "condition_id", "qa_id", "house_id",
               "scorer_label", "checker1_label", "checker2_label",
               "consensus_label")
HT4_LABELS = LABELS_NAMING + LABELS_OTHER
HT5_COLUMNS = ("audit_id", "model_key", "condition_id", "qa_id", "house_id",
               "stratum", "verdict")
HT5_STRATA = ("C", "english_copy", "sibling")
HT5_STRATA_N = {"C": 140, "english_copy": 30, "sibling": 30}      # Design 10
HT5_VERDICTS = ("gold_wrong", "missing_variant", "not_visible", "ambiguous",
                "model_wrong")
HT5_GOLD_OR_VISIBILITY = ("gold_wrong", "not_visible")
SENS_COLUMNS = ("row_id", "source", "op", "target", "target_id", "tier",
                "lang", "form", "note")
SENS_SOURCES = ("HT-5", "D192")
SENS_OPS = ("add_term", "remove_term", "item_any_class")
SENS_TARGETS = ("lexicon", "room", "material", "colour", "number")
TOKEN_COLUMNS = ("qa_id", "model_key", "tokens_bn", "tokens_en", "counter",
                 "counted_utc")
HARNESS_KEYS = ("key", "model_id", "model_key", "prompt_version", "stage",
                "run_id", "house_id", "mode", "ts_utc")
HT2_FAITHFUL = (146, 150)       # addendum D7 (stage 3b); Wilson recomputed here

# Template fold (D236): applied after normalise(), for grouping questions only.
TEMPLATE_FOLD = (("ঁ", ""),                         # chandrabindu
                 ("্যা", "্য"),  # ya-phala + aa
                 ("ী", "ি"), ("ঈ", "ই"),
                 ("ূ", "ু"), ("ঊ", "উ"),
                 ("ণ", "ন"),                    # ণ to ন
                 ("ষ", "শ"), ("স", "শ"),   # ষ, স to শ
                 ("ঙ", "ং"),                    # ঙ to ং
                 ("‌", ""), ("‍", ""))


class StatsError(Exception):
    """A rule of the analysis is broken; the message says what to do."""


# --------------------------------------------------------------------------
# 1. Small tools
# --------------------------------------------------------------------------

def say(msg=""):
    print(msg, flush=True)


class Checks:
    """One PASS, FAIL, WARN or INFO line per input or rule (Cell 0)."""

    def __init__(self, echo=True, soften=False):
        self.rows = []
        self.echo = echo
        self.soften = soften     # tests only (--allow-other-inputs)

    def add(self, status, what, detail=""):
        self.rows.append((status, what, str(detail)))
        if self.echo:
            say("%-4s  %-46s %s" % (status, what, detail))

    def check(self, ok, what, detail, pinned=False):
        if ok:
            self.add("PASS", what, detail)
        elif pinned and self.soften:
            self.add("WARN", what, "%s  [test inputs: --allow-other-inputs]" % detail)
        else:
            self.add("FAIL", what, detail)
        return bool(ok)

    def count(self, status):
        return sum(1 for r in self.rows if r[0] == status)

    def text(self):
        return "".join("%-4s  %-46s %s\n" % r for r in self.rows)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pin_ok(name, got):
    want = PIN_SHA.get(name)
    return bool(want) and got.startswith(want)


def now_utc():
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp():
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json(path, obj):
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=1, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if math.isnan(float(o)) else float(o)
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def read_csv_dicts(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return [{(k or "").strip(): (v or "").strip() for k, v in row.items()}
                for row in csv.DictReader(fh)]


def csv_header(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return [(h or "").strip() for h in next(csv.reader(fh), [])]


def write_csv(path, header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow(["" if v is None else v for v in r])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(buf.getvalue(), encoding="utf-8-sig")
    tmp.replace(path)


def git_show(repo, rev, rel):
    try:
        r = subprocess.run(["git", "-C", str(repo), "show", "%s:%s" % (rev, rel)],
                           capture_output=True)
    except OSError:
        return None
    return r.stdout if r.returncode == 0 else None


def git_out(repo, *args):
    try:
        r = subprocess.run(["git", "-C", str(repo)] + list(args), capture_output=True,
                           text=True)
    except OSError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def same_as_tag(repo, tag, rel):
    """True, False, or None when the tag or the file at the tag is missing."""
    tagged = git_show(repo, tag, rel)
    if tagged is None:
        return None
    local = Path(repo) / rel
    return local.exists() and local.read_bytes() == tagged


def guard_out(path, mode, repo, drive):
    """Placeholder output only under a folder named placeholder; real output
    never there; never inside the repo, Drive scored/ or Drive outputs/ (D230)."""
    p = Path(path).resolve()
    parts = [x.lower() for x in p.parts]
    for root, what in ((Path(repo).resolve(), "the repo"),
                       ((Path(drive) / "scored").resolve(), "Drive scored/"),
                       ((Path(drive) / "outputs").resolve(), "Drive outputs/")):
        if p == root or root in p.parents:
            raise StatsError("refusing to write inside %s: %s" % (what, p))
    if mode == "placeholder" and "placeholder" not in parts:
        raise StatsError("placeholder output must sit in a folder named placeholder: %s" % p)
    if mode == "real" and "placeholder" in parts:
        raise StatsError("real output must not sit in a folder named placeholder: %s" % p)
    return p


def load_scorer(repo):
    src = str(Path(repo) / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import scoring  # noqa: E402  (the one scorer)
    from normalise import normalise  # noqa: E402  (the one normalise())
    return scoring, normalise


def nfc(s):
    return unicodedata.normalize("NFC", s or "")


# --------------------------------------------------------------------------
# 2. Bootstrap, CIs, p-values, Holm, Wilson, kappa, AUC (Design 9.2, D232)
# --------------------------------------------------------------------------

class Est:
    """A point estimate with its bootstrap replicates (same resamples for
    every estimate of one house set, so differences stay paired)."""
    __slots__ = ("point", "boot")

    def __init__(self, point, boot):
        self.point = float(point) if point is not None else float("nan")
        self.boot = boot

    def _wrap(self, other, op):
        if isinstance(other, Est):
            return Est(op(self.point, other.point), op(self.boot, other.boot))
        return Est(op(self.point, other), op(self.boot, other))

    def __add__(self, o):
        return self._wrap(o, lambda a, b: a + b)

    def __sub__(self, o):
        return self._wrap(o, lambda a, b: a - b)

    def __mul__(self, o):
        return self._wrap(o, lambda a, b: a * b)

    def __neg__(self):
        return Est(-self.point, -self.boot)


def est_mean(ests):
    """Mean of several Est (for example over seeds), resample by resample."""
    ests = list(ests)
    point = sum(e.point for e in ests) / len(ests)
    boot = sum(e.boot for e in ests) / len(ests)
    return Est(point, boot)


class HouseBoot:
    """House-clustered bootstrap: resample houses with replacement, B times.
    numpy default_rng(seed), integers over the sorted house list; one generator
    per house set, so a set always gets the same resamples (D232)."""

    def __init__(self, houses, n_boot=N_BOOT, seed=SEED, name=""):
        self.houses = sorted(set(houses))
        self.H = len(self.houses)
        if self.H == 0:
            raise StatsError("bootstrap %s: no houses" % name)
        self.index = {h: i for i, h in enumerate(self.houses)}
        self.n_boot = n_boot
        self.name = name
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, self.H, size=(n_boot, self.H))
        flat = (draws + (np.arange(n_boot) * self.H)[:, None]).ravel()
        self.W = np.bincount(flat, minlength=n_boot * self.H).reshape(
            n_boot, self.H).astype(np.float64)

    def hidx(self, house_ids):
        try:
            return np.array([self.index[h] for h in house_ids], dtype=np.int64)
        except KeyError as exc:
            raise StatsError("bootstrap %s: house %s is not in its house set"
                             % (self.name, exc))

    def sums(self, hidx, values):
        return np.bincount(hidx, weights=np.asarray(values, dtype=np.float64),
                           minlength=self.H)

    def ratio_sums(self, num_h, den_h):
        den = float(den_h.sum())
        point = float(num_h.sum()) / den if den > 0 else float("nan")
        nb = self.W @ num_h
        db = self.W @ den_h
        boot = np.full(nb.shape, np.nan)
        np.divide(nb, db, out=boot, where=db > 0)
        return Est(point, boot)

    def ratio(self, hidx, num, den):
        return self.ratio_sums(self.sums(hidx, num), self.sums(hidx, den))

    def mean(self, hidx, values, mask=None):
        values = np.asarray(values, dtype=np.float64)
        if mask is None:
            mask = np.ones(len(values))
        mask = np.asarray(mask, dtype=np.float64)
        return self.ratio(hidx, values * mask, mask)

    def const(self, value):
        return Est(value, np.full(self.n_boot, float(value)))


def p_two_sided(boot):
    """Design 9.2: p = 2 x the smaller share of resamples at or below 0 and at
    or above 0; resamples with an undefined value are left out and counted."""
    b = boot[~np.isnan(boot)]
    if b.size == 0:
        return None
    return float(min(1.0, 2.0 * min(np.mean(b <= 0), np.mean(b >= 0))))


def summarise_est(est, test=False):
    b = est.boot
    ok = b[~np.isnan(b)]
    lo = hi = None
    if ok.size:
        lo, hi = (float(x) for x in np.percentile(ok, [2.5, 97.5]))
    point = None if math.isnan(est.point) else est.point
    out = {"point": point, "lo": lo, "hi": hi, "n_undefined": int(np.isnan(b).sum())}
    if test:
        out["p"] = p_two_sided(b) if point is not None else None
        out["p_floor"] = 2.0 / ok.size if ok.size else None     # smallest p above 0
    return out


def holm(pvals):
    """Holm step-down adjustment of a dict {name: p}; None values are skipped."""
    items = sorted(((k, p) for k, p in pvals.items() if p is not None),
                   key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def holm_bootstrap(tests):
    """Holm over bootstrap tests {name: summarise_est(..., test=True)}. A p of 0
    only says p < its floor, so the floor enters as an upper bound; an adjusted
    value that comes from such a bound is marked. Returns {name: {"p", "bound"}}."""
    items = sorted(((k, (s["p"] if s["p"] > 0 else s["p_floor"]), s["p"] == 0)
                    for k, s in tests.items() if s and s.get("p") is not None),
                   key=lambda kv: kv[1])
    m = len(items)
    out, running, run_bound = {}, 0.0, False
    for i, (k, p, bound) in enumerate(items):
        v = min(1.0, (m - i) * p)
        if v > running:
            running, run_bound = v, bound
        elif v == running:
            run_bound = run_bound and bound
        out[k] = {"p": running, "bound": run_bound}
    return out


def fmt_holm(h):
    if not h:
        return "-"
    return ("< %.4f" % h["p"]) if h["bound"] else fmt_p(h["p"])


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def kappa_from_confusion(conf):
    """Cohen's kappa from a K x K count matrix (or B x K x K). None or nan where
    chance agreement is 1."""
    conf = np.asarray(conf, dtype=np.float64)
    n = conf.sum(axis=(-2, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        po = np.trace(conf, axis1=-2, axis2=-1) / n
        pe = (conf.sum(axis=-1) * conf.sum(axis=-2)).sum(axis=-1) / (n * n)
        k = (po - pe) / (1 - pe)
    k = np.where(np.abs(1 - pe) < 1e-12, np.nan, k)
    return po, k


def auc(pos, neg):
    """Area under the ROC curve: P(pos > neg) + 0.5 P(tie)."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return None
    allv = np.concatenate([pos, neg])
    uniq, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    avg_rank = ends - (counts - 1) / 2.0
    ranks = avg_rank[inv]
    u = ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def pct(x, digits=1):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else \
        ("%." + str(digits) + "f") % (100.0 * x)


def fmt_p(p, floor=None):
    """A bootstrap p of 0 is shown as below its floor, 2 / (defined resamples);
    a model-based p (GEE) has no floor."""
    if p is None:
        return "-"
    if p == 0:
        return "< %.4f" % floor if floor else "< 1e-16"
    return "%.4f" % p if p >= 0.0001 else "%.1e" % p


def fmt_ps(s):
    """fmt_p of a summarise_est(..., test=True) result."""
    return fmt_p((s or {}).get("p"), (s or {}).get("p_floor"))


# --------------------------------------------------------------------------
# 3. Templates (D236)
# --------------------------------------------------------------------------

def template_key(question, normalise):
    t = normalise(question)
    for a, b in TEMPLATE_FOLD:
        t = t.replace(a, b)
    return re.sub(r"\s+", "", t)


def template_id_of(key):
    return "T_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]


# --------------------------------------------------------------------------
# 4. The fixed inputs (the "world")
# --------------------------------------------------------------------------

@dataclass
class Paths:
    repo: Path
    drive: Path
    dataset: Path = None
    houses: Path = None
    probes_dir: Path = None
    manifest11: Path = None
    swaps: Path = None
    english: Path = None

    @classmethod
    def make(cls, repo, drive, **kw):
        repo, drive = Path(repo).resolve(), Path(drive)
        p = cls(repo=repo, drive=drive)
        p.dataset = Path(kw.get("dataset") or drive / "data" / "banglaindoorvqa.json")
        p.houses = Path(kw.get("houses") or drive / "data" / "house_assignment.csv")
        p.probes_dir = Path(kw.get("probes_dir") or drive / "probes")
        p.manifest11 = Path(kw.get("manifest11") or drive / "probes" / "probe_manifest_11.json")
        p.swaps = Path(kw.get("swaps") or drive / "human" / "HT-6" / "ht6_swaps.csv")
        p.english = Path(kw.get("english") or drive / "english" / "english_questions_final.csv")
        return p


@dataclass
class World:
    paths: Paths
    sc: object
    normalise: object
    lex: object
    voc: object
    excluded: dict
    conditions: dict
    run_ids: dict
    scored_ids: dict
    items: dict
    house_annotator: dict
    swap_rows: list
    swap_pairs: set
    manifest11: dict
    dataset_qa: list
    box_format: str
    hashes: dict
    soften: bool = False
    dataset_counts: dict = None

    def scorer_item(self, qa_id):
        it = self.items[qa_id]
        return {"qa_id": qa_id, "qa_type": it["qa_type"], "answer_bn": it["answer_bn"]}

    def probe_of(self, condition_id):
        return self.conditions[condition_id]["probe"]

    def houses_of(self, qa_ids):
        return [self.items[q]["house_id"] for q in qa_ids]


def _load_probe(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    items = data["items"] if isinstance(data, dict) and "items" in data else data
    if not isinstance(items, list):
        raise StatsError("%s: expected a list of records or {'items': [...]}" % path)
    return items


def _read_ids(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    return [x for x in re.findall(r"H\d{2}_\d{4}_q\d+", text)]


def read_house_assignment(path):
    """house_id -> annotator_id. Only these two columns are read (D206)."""
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = [h.strip().lower() for h in next(reader)]
        ih, ia = header.index("house_id"), header.index("annotator_id")
        for row in reader:
            if row and len(row) > max(ih, ia) and row[ih].strip():
                out[row[ih].strip()] = row[ia].strip()
    return out


def box_format_of(objects):
    """Decide whether bbox_pct is [x, y, w, h] or [x1, y1, x2, y2] from the
    data (D239): exactly one reading must fit every box."""
    xywh = xyxy = True
    for o in objects:
        x, y, a, b = (float(v) for v in o["bbox_pct"])
        if not (x >= -0.5 and y >= -0.5 and a > 0 and b > 0
                and x + a <= 100.5 and y + b <= 100.5):
            xywh = False
        if not (a > x and b > y and a <= 100.5 and b <= 100.5):
            xyxy = False
    if xywh and not xyxy:
        return "xywh"
    if xyxy and not xywh:
        return "xyxy"
    return "ambiguous" if xywh and xyxy else "neither"


def box_area(bbox_pct, fmt):
    x, y, a, b = (float(v) for v in bbox_pct)
    if fmt == "xywh":
        return (a * b) / 1e4
    return ((a - x) * (b - y)) / 1e4


def build_world(paths, checks, need_dataset=True, soften=False):
    """Read and check every fixed input. Each check adds one line to checks."""
    import yaml
    hashes = {}
    repo = paths.repo

    # Scorer and normalise()
    sc, normalise = load_scorer(repo)
    hashes["scoring.py"] = sha256_file(repo / "src" / "scoring.py")
    hashes["normalise.py"] = sha256_file(repo / "src" / "normalise.py") \
        if (repo / "src" / "normalise.py").exists() else "missing"
    checks.check(sc.SCORER_VERSION == SCORER_VERSION_EXPECTED, "src/scoring.py",
                 "version %s, sha256 %s..." % (sc.SCORER_VERSION, hashes["scoring.py"][:12]),
                 pinned=True)

    # Lexicon and vocabulary (frozen at gate-A)
    lexp, vocp = repo / "lexicon" / "lexicon.csv", repo / "lexicon" / "closed_vocab.csv"
    hashes["lexicon.csv"], hashes["closed_vocab.csv"] = sha256_file(lexp), sha256_file(vocp)
    lex = sc.load_lexicon(lexp)
    voc = sc.load_closed_vocab(vocp)
    checks.check(pin_ok("lexicon.csv", hashes["lexicon.csv"]), "lexicon/lexicon.csv",
                 "sha256 %s..., %s, warnings %d" % (hashes["lexicon.csv"][:12],
                                                    lex.describe(), len(lex.warnings)),
                 pinned=True)
    checks.check(pin_ok("closed_vocab.csv", hashes["closed_vocab.csv"]),
                 "lexicon/closed_vocab.csv",
                 "sha256 %s..., %s" % (hashes["closed_vocab.csv"][:12], voc.describe()),
                 pinned=True)

    # Excluded list
    exp_ = repo / "probes" / "excluded_items.csv"
    hashes["excluded_items.csv"] = sha256_file(exp_)
    excluded = sc.load_excluded(exp_)
    checks.check(pin_ok("excluded_items.csv", hashes["excluded_items.csv"]),
                 "probes/excluded_items.csv",
                 "%d rows, sha256 %s... (pinned with the scored sizes, D231)"
                 % (len(excluded), hashes["excluded_items.csv"][:12]), pinned=True)

    # Conditions
    with open(repo / "configs" / "conditions.yaml", encoding="utf-8") as fh:
        cy = yaml.safe_load(fh)
    conditions = cy["conditions"]
    want = {"L-BB": ("P-CORE-CLOSED", True, "bn", "bn"), "L-BE": ("P-CORE-CLOSED", True, "bn", "en"),
            "L-EB": ("P-CORE-CLOSED", True, "en", "bn"), "L-EE": ("P-CORE-CLOSED", True, "en", "en"),
            "L-BB0": ("P-CORE-CLOSED", False, "bn", "bn"), "L-EE0": ("P-CORE-CLOSED", False, "en", "en"),
            "N-BB": ("P-NAME", True, "bn", "bn"), "N-BE": ("P-NAME", True, "bn", "en"),
            "N-BBs": ("P-NAME", True, "bn", "bn"), "N-FCB": ("P-CLUST", True, "bn", "bn"),
            "N-FCB0": ("P-CLUST", False, "bn", "bn"), "N-FCE": ("P-FCE", True, "en", "en"),
            "N-FCE0": ("P-FCE", False, "en", "en"), "R-BB": ("R-TEST", True, "bn", "bn"),
            "R-EE": ("R-TEST", True, "en", "en"), "R-BB0": ("R-TEST", False, "bn", "bn")}
    bad = [c for c, (pr, img, ql, al) in want.items()
           if c not in conditions or conditions[c].get("probe") != pr
           or bool(conditions[c].get("image")) != img
           or conditions[c].get("question_lang") != ql or conditions[c].get("answer_lang") != al]
    pr_ok = all(cy["probes"].get(p, {}).get("expected_n") == RUN_SIZES[p] for p in PROBE_FILES)
    checks.check(not bad and pr_ok, "configs/conditions.yaml",
                 "16 conditions and 5 probe sizes as Design 7" if not bad and pr_ok
                 else "differs: %s" % (", ".join(bad) or "probe sizes"))

    # Id lists (repo) and probe files (Drive)
    id_lists = {}
    for p, fname in ID_LISTS.items():
        path = repo / "probes" / fname
        ids = _read_ids(path) if path.exists() else []
        id_lists[p] = ids
        checks.check(len(ids) == RUN_SIZES[p] and len(set(ids)) == len(ids),
                     "probes/%s" % fname, "%d qa_ids" % len(ids), pinned=True)
    items, run_ids = {}, {}
    fields = ("qa_id", "house_id", "image_id", "qa_type", "capture_type",
              "question_bn", "answer_bn", "image_file")
    for p, fname in PROBE_FILES.items():
        path = paths.probes_dir / fname
        hashes[fname] = sha256_file(path)
        recs = _load_probe(path)
        ids = [r["qa_id"] for r in recs]
        houses = {r["house_id"] for r in recs}
        miss = [f for f in fields if any(f not in r for r in recs)]
        clash = 0
        for r in recs:
            it = items.get(r["qa_id"])
            if it is None:
                items[r["qa_id"]] = {f: r.get(f) for f in fields}
                items[r["qa_id"]]["probes"] = {p}
            else:
                if any(it[f] != r.get(f) for f in fields):
                    clash += 1
                it["probes"].add(p)
        run_ids[p] = ids
        ok = (pin_ok(fname, hashes[fname]) and len(ids) == RUN_SIZES[p]
              and set(ids) == set(id_lists[p]) and len(houses) == RUN_HOUSES[p]
              and not miss and clash == 0)
        checks.check(ok, "Drive probes/%s" % fname,
                     "%d items, %d houses, sha256 %s..., same ids as the repo list: %s%s%s"
                     % (len(ids), len(houses), hashes[fname][:12],
                        set(ids) == set(id_lists[p]),
                        ", missing fields %s" % miss if miss else "",
                        ", %d records differ from another probe file" % clash if clash else ""),
                     pinned=True)
    nameref = id_lists["NAME-REF"]
    run_ids["NAME-REF"] = nameref
    nr_bad = [q for q in nameref if q not in items or "P-NAME" not in items[q]["probes"]]
    checks.check(not nr_bad, "NAME-REF in P-NAME", "%d of %d NAME-REF ids are P-NAME items"
                 % (len(nameref) - len(nr_bad), len(nameref)))
    for q in nameref:
        if q in items:
            items[q]["probes"].add("NAME-REF")

    # Derived item fields: excluded, template, gold object through the scorer
    for q, it in items.items():
        it["excluded"] = q in excluded
        it["excl_reason"] = excluded.get(q)
        it["template_key"] = template_key(it["question_bn"], normalise)
        it["template_id"] = template_id_of(it["template_key"])
        it["is_overview"] = it["capture_type"] == "overview"
        gold = sc.resolve_gold(it, lex, voc)
        it["gold_status"] = gold.status
        it["gold_kind"] = gold.kind
        it["name_id"] = gold.name_id or None
        obj = lex.objects.get(gold.name_id) if gold.name_id else None
        it["cluster_id"] = obj.cluster_id if obj else None
        it["in_cluster"] = bool(obj.cluster_id) if obj else None
        it["en_route"] = obj.en_route if obj else None
        it["en_unique"] = obj.en_unique if obj else None
        it["origin3"] = obj.row.get("origin3") if obj else None
        it["cultural"] = obj.row.get("cultural_specificity") if obj else None
        it["freq"] = int(obj.row.get("freq_obj_name") or 0) if obj else None
    unresolved = [q for q, it in items.items() if not it["excluded"] and it["gold_status"] != "ok"]
    checks.check(not unresolved, "golds resolve through the scorer",
                 "%d of %d items outside the list; unresolved %d%s"
                 % (sum(1 for it in items.values() if not it["excluded"]), len(items),
                    len(unresolved), (" (first %s)" % unresolved[0]) if unresolved else ""))
    naming_no_obj = [q for q, it in items.items() if not it["excluded"]
                     and it["qa_type"] in NAMING_TYPES and not it["name_id"]]
    checks.check(not naming_no_obj, "naming items joined to an object",
                 "gold_name_id from resolve_gold for every naming item (C4); missing %d"
                 % len(naming_no_obj))

    # Scored sets and sizes (D231)
    scored_ids = {p: [q for q in ids if q not in excluded] for p, ids in run_ids.items()}
    sizes_ok = all(len(scored_ids[p]) == SCORED_SIZES[p] for p in SCORED_SIZES)
    checks.check(sizes_ok, "scored sizes (PROJECT_STATE)", ", ".join(
        "%s %d" % (p, len(scored_ids[p])) for p in SCORED_SIZES), pinned=True)
    bh = {"EXP-L": len({items[q]["house_id"] for q in scored_ids["P-CORE-CLOSED"]}),
          "EXP-N": len({items[q]["house_id"] for q in scored_ids["P-NAME"]}),
          "EXP-R": len({items[q]["house_id"] for q in scored_ids["R-TEST"]})}
    checks.check(bh == BOOT_HOUSES, "bootstrap houses (Design 3.5)",
                 "EXP-L %(EXP-L)d, EXP-N %(EXP-N)d, EXP-R %(EXP-R)d" % bh, pinned=True)

    # Clusters (lexicon join) against Design 3.4 and the ASSET-11 manifest
    hashes["probe_manifest_11.json"] = sha256_file(paths.manifest11)
    with open(paths.manifest11, encoding="utf-8") as fh:
        manifest11 = json.load(fh)
    pc = Counter(nfc(items[q]["cluster_id"]) for q in run_ids["P-CLUST"])
    want_pc = {}
    for prefix, n in PCLUST_CLUSTERS:
        names = [c for c in pc if c.startswith(nfc(prefix))]
        want_pc[prefix] = (names[0] if len(names) == 1 else None, n)
    pc_ok = all(name and pc[name] == n for name, n in want_pc.values()) \
        and sum(pc.values()) == RUN_SIZES["P-CLUST"] and len(pc) == 7
    checks.check(pc_ok, "P-CLUST clusters (lexicon join)", ", ".join(
        "%d" % pc.get(name, 0) for name, _ in want_pc.values()) + " (Design 3.4: 138, 88, 79, 61, 43, 42, 41)",
        pinned=True)
    pf = Counter(nfc(items[q]["cluster_id"]) for q in run_ids["P-FCE"])
    mf = {nfc(k): v for k, v in manifest11.get("pfce", {}).get("by_cluster", {}).items()}
    checks.check(pin_ok("probe_manifest_11.json", hashes["probe_manifest_11.json"]) and pf == mf,
                 "P-FCE clusters = probe_manifest_11.json",
                 "%s; manifest sha256 %s..." % (", ".join(str(pf[k]) for k in sorted(pf)),
                                                 hashes["probe_manifest_11.json"][:12]),
                 pinned=True)

    # Houses and annotators (ids only)
    hashes["house_assignment.csv"] = sha256_file(paths.houses)
    house_annotator = read_house_assignment(paths.houses)
    checks.check(pin_ok("house_assignment.csv", hashes["house_assignment.csv"])
                 and len(house_annotator) == 68, "Drive data/house_assignment.csv",
                 "%d houses, annotator ids %s (names not read)" % (
                     len(house_annotator), ", ".join("%s %d" % kv for kv in sorted(
                         Counter(house_annotator.values()).items()))), pinned=True)
    no_ann = [q for q, it in items.items() if it["house_id"] not in house_annotator]
    for q, it in items.items():
        it["annotator"] = house_annotator.get(it["house_id"])

    # HT-6 swaps (D204)
    hashes["ht6_swaps.csv"] = sha256_file(paths.swaps)
    swap_rows = read_csv_dicts(paths.swaps)
    swap_pairs = {(r["qa_id"], r["removed_id"]) for r in swap_rows if r.get("reason") == "name"}
    checks.check(pin_ok("ht6_swaps.csv", hashes["ht6_swaps.csv"]) and len(swap_rows) == 53
                 and len(swap_pairs) == 48, "Drive human/HT-6/ht6_swaps.csv",
                 "%d rows, %d item and sibling pairs with reason name (D204)"
                 % (len(swap_rows), len(swap_pairs)), pinned=True)
    swap_out = [q for q, _ in swap_pairs if q not in items or "P-CLUST" not in items[q]["probes"]]
    checks.check(not swap_out and not no_ann, "joins of houses and HT-6 pairs",
                 "every item has an annotator id; every HT-6 pair is a P-CLUST item"
                 if not swap_out and not no_ann else
                 "%d items without annotator, %d pairs outside P-CLUST" % (len(no_ann), len(swap_out)))

    # Dataset (B0 pool, box areas)
    dataset_qa, box_fmt, counts = [], "not read", None
    if need_dataset:
        hashes["banglaindoorvqa.json"] = sha256_file(paths.dataset)
        with open(paths.dataset, encoding="utf-8") as fh:
            data = json.load(fh)
        n_img = len(data)
        houses = {im["house_id"] for im in data}
        objects = [o for im in data for o in (im.get("objects") or [])]
        by_qa, img_of = {}, {}
        for im in data:
            img_of[im["image_id"]] = im
            for qa in im.get("qa_pairs") or []:
                by_qa[qa["qa_id"]] = (qa, im)
                dataset_qa.append((qa["qa_id"], qa["type"], qa["question"], qa["answer"],
                                   im["house_id"]))
        counts = {"images": n_img, "houses": len(houses), "qa": len(by_qa),
                  "objects": len(objects)}
        box_fmt = box_format_of(objects)
        checks.check(pin_ok("banglaindoorvqa.json", hashes["banglaindoorvqa.json"])
                     and counts == DATASET_COUNTS, "Drive data/banglaindoorvqa.json",
                     "images %(images)d, houses %(houses)d, QA %(qa)d, boxes %(objects)d" % counts
                     + ", sha256 %s..." % hashes["banglaindoorvqa.json"][:12], pinned=True)
        checks.check(box_fmt in ("xywh", "xyxy"), "box format (from the data, D239)",
                     "bbox_pct read as %s" % box_fmt)
        # The probe items must equal the dataset items (B0 compares with dataset golds)
        diff = 0
        for q, it in items.items():
            got = by_qa.get(q)
            if got is None:
                diff += 1
                continue
            qa, im = got
            if (qa["type"] != it["qa_type"] or im["house_id"] != it["house_id"]
                    or im["image_id"] != it["image_id"]
                    or normalise(qa["question"]) != normalise(it["question_bn"])
                    or normalise(qa["answer"]) != normalise(it["answer_bn"])):
                diff += 1
        checks.check(diff == 0, "probe items equal the dataset",
                     "%d of %d probe items differ (type, house, image, question or gold)"
                     % (diff, len(items)))
        ann_diff = sum(1 for im in data if house_annotator.get(im["house_id"]) != im.get("annotator_id"))
        checks.add("INFO", "dataset annotator ids", "%d of %d images differ from house_assignment.csv"
                   % (ann_diff, n_img))
        # Box area: items on close-up images whose image holds a box (D239)
        for q, it in items.items():
            im = img_of.get(it["image_id"])
            objs = (im or {}).get("objects") or []
            it["box_area"] = box_area(objs[0]["bbox_pct"], box_fmt) \
                if (it["capture_type"] == "closeup" and len(objs) == 1
                    and box_fmt in ("xywh", "xyxy")) else None
    else:
        for it in items.values():
            it["box_area"] = None

    return World(paths=paths, sc=sc, normalise=normalise, lex=lex, voc=voc,
                 excluded=excluded, conditions=conditions, run_ids=run_ids,
                 scored_ids=scored_ids, items=items, house_annotator=house_annotator,
                 swap_rows=swap_rows, swap_pairs=swap_pairs, manifest11=manifest11,
                 dataset_qa=dataset_qa, box_format=box_fmt, hashes=hashes, soften=soften,
                 dataset_counts=counts)


def template_report(world):
    """Counts and merged groups of the template fold (D236). The groups hold
    question text and are written to Drive only."""
    norm = world.normalise
    out = {}
    scopes = {"P-CORE-CLOSED": [(q, world.items[q]["question_bn"])
                                for q in world.run_ids["P-CORE-CLOSED"]],
              "dataset": [(q, t) for q, _, t, _, _ in world.dataset_qa]}
    groups_rows = []
    for scope, pairs in scopes.items():
        by_key = defaultdict(lambda: defaultdict(int))
        for q, text in pairs:
            by_key[template_key(text, norm)][norm(text)] += 1
        merged = {k: v for k, v in by_key.items() if len(v) > 1}
        out[scope] = {"items": len(pairs), "strings": sum(len(v) for v in by_key.values()),
                      "templates": len(by_key), "merged_groups": len(merged),
                      "strings_in_merged": sum(len(v) for v in merged.values())}
        for k, v in sorted(merged.items(), key=lambda kv: -sum(kv[1].values())):
            groups_rows.append((scope, template_id_of(k), len(v), sum(v.values()),
                                " || ".join(sorted(v))))
    return out, groups_rows


# --------------------------------------------------------------------------
# 5. Scored records (B4, B6, B9; D231)
# --------------------------------------------------------------------------

SLIM = ("label", "strict", "lenient", "fidelity", "refusal", "refusal_phrase",
        "is_empty", "n_words", "mixed_script", "content_found", "content_lang_ok",
        "label_lang", "gold_name_id", "named_ids", "fc_valid", "fc_choice",
        "gold_status", "scoreable", "kind", "task", "answer_raw", "ts_utc", "mode",
        "en_route", "en_unique", "origin3", "requested_lang", "lang", "found_value",
        "gold_value", "model_id", "model_version", "model_revision", "prompt_version",
        "scorer_version", "stage", "fc_options_sha256", "qa_type", "house_id")
FINGERPRINT = ("label", "strict", "lenient", "fidelity", "refusal", "refusal_phrase",
               "is_empty", "n_words", "answer_raw", "scorer_version")


@dataclass
class Records:
    data: dict            # (model_key, condition_id) -> {qa_id: slim record}
    stage: str
    scorer_version: str
    prompt_version: str
    files: list
    manifests: list
    seeds: tuple

    def get(self, model, cond):
        return self.data[(model, cond)]

    def has(self, model, cond):
        return (model, cond) in self.data

    def vec(self, model, cond, qa_ids, fn):
        r = self.data[(model, cond)]
        return np.array([fn(r[q]) for q in qa_ids], dtype=np.float64)


def _b(x):
    return 1.0 if x else 0.0


def load_scored(patterns, world, checks, mode):
    """Read scored JSONL files (one record per line, the scorer's keys plus the
    kept harness fields). Every rule here is a check line; a FAIL stops the run."""
    sc = world.sc
    files = sorted({f for pat in patterns for f in glob.glob(str(pat))})
    if not files:
        checks.add("FAIL", "scored files", "no file matches %s" % ", ".join(map(str, patterns)))
        raise StatsError("no scored files")
    required = set(sc._RECORD_KEYS) | {"fc_options_sha256"} | set(HARNESS_KEYS)
    data, finger = {}, {}
    n_lines = bad_json = missing_keys = unknown_cond = foreign = dup_same = dup_diff = 0
    house_diff = type_diff = 0
    file_info, manifests = [], []
    for f in files:
        n_f = 0
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                n_lines += 1
                try:
                    rec = json.loads(line)
                except ValueError:
                    bad_json += 1
                    continue
                n_f += 1
                if not required <= set(rec):
                    missing_keys += 1
                    continue
                cid, m, q = rec["condition_id"], rec["model_key"], rec["qa_id"]
                if cid not in world.conditions or cid not in ALL_CONDS:
                    unknown_cond += 1
                    continue
                if q not in world.items or world.probe_of(cid) not in world.items[q]["probes"]:
                    foreign += 1
                    continue
                if rec["house_id"] != world.items[q]["house_id"]:
                    house_diff += 1
                if rec["qa_type"] != world.items[q]["qa_type"]:
                    type_diff += 1
                key = (m, cid)
                fp = tuple(json.dumps(rec.get(k), sort_keys=True) for k in FINGERPRINT)
                slot = data.setdefault(key, {})
                if q in slot:
                    if finger[(m, cid, q)] == fp:
                        dup_same += 1
                    else:
                        dup_diff += 1
                    continue
                slot[q] = {k: rec.get(k) for k in SLIM}
                finger[(m, cid, q)] = fp
        file_info.append({"file": str(f), "sha256": sha256_file(f), "records": n_f})
        man = f[:-len(".jsonl")] + "_manifest.json" if f.endswith(".jsonl") else None
        if man and os.path.exists(man):
            with open(man, encoding="utf-8") as fh:
                m_ = json.load(fh)
            manifests.append({"file": os.path.basename(man), "scorer_version": m_.get("scorer_version"),
                              "run_utc": m_.get("run_utc"), "inputs": m_.get("inputs")})
    checks.check(bad_json == 0 and missing_keys == 0 and unknown_cond == 0 and foreign == 0,
                 "scored files readable", "%d files, %d records; unreadable %d, missing keys %d,"
                 " unknown condition %d, item not in the condition's probe %d"
                 % (len(files), n_lines, bad_json, missing_keys, unknown_cond, foreign))
    checks.check(dup_diff == 0 and house_diff == 0 and type_diff == 0,
                 "one record per model, condition, item",
                 "identical duplicates kept once %d, conflicting duplicates %d, house differs %d,"
                 " type differs %d" % (dup_same, dup_diff, house_diff, type_diff))

    stages = Counter(r["stage"] for s in data.values() for r in s.values())
    scorers = Counter(r["scorer_version"] for s in data.values() for r in s.values())
    prompts = Counter(r["prompt_version"] for s in data.values() for r in s.values())
    want_stage = STAGE_PLACEHOLDER if mode == "placeholder" else STAGE_MAIN
    checks.check(set(stages) == {want_stage}, "record stage",
                 "%s (mode %s needs every record at stage %s)" % (dict(stages), mode, want_stage))
    checks.check(len(scorers) == 1 and sc.SCORER_VERSION in scorers, "one scorer version",
                 "%s; imported scorer %s" % (dict(scorers), sc.SCORER_VERSION))
    checks.check(len(prompts) == 1, "one prompt version", str(dict(prompts)))

    # Expected (model, condition) pairs and completeness
    seeds = tuple(s for s in FT_SEEDS if any((s, c) in data for c in COND_R))
    expected = [(m, c) for m in MODELS for c in ZS_CONDS] + [(FT_BASE, c) for c in COND_R] + \
               [(s, c) for s in seeds for c in COND_R]
    missing_pairs = [p for p in expected if p not in data]
    extra_pairs = [p for p in data if p not in expected]
    incomplete = []
    for p in expected:
        if p in data:
            want_ids = set(world.run_ids[world.probe_of(p[1])])
            if set(data[p]) != want_ids:
                incomplete.append("%s %s %d of %d" % (p[0], p[1], len(set(data[p]) & want_ids),
                                                      len(want_ids)))
    checks.check(not missing_pairs and not extra_pairs and not incomplete,
                 "every model and condition complete",
                 "%d model-condition sets; EXP-R seeds %s" % (len(data), ", ".join(seeds) or "none")
                 if not (missing_pairs or extra_pairs or incomplete) else
                 "missing %s; unexpected %s; incomplete %s" % (missing_pairs[:4], extra_pairs[:4],
                                                              incomplete[:4]))
    if len(seeds) < 3:
        checks.add("WARN", "EXP-R seeds", "%d of 3 seeds present: report the seeds that ran"
                   " (Design 7.3)" % len(seeds))

    # Excluded items, scoreability, field agreement with the lexicon join
    ex_bad = sc_bad = gid_bad = obj_bad = lang_bad = fc_bad = 0
    want_fc = "placeholder" if mode == "placeholder" else PIN_SHA["fc_options.jsonl"]
    for (m, cid), slot in data.items():
        cond = world.conditions[cid]
        for q, r in slot.items():
            it = world.items[q]
            is_ex = r["gold_status"] == "excluded"
            if is_ex != it["excluded"]:
                ex_bad += 1
            if is_ex:
                continue
            if not r["scoreable"] or r["gold_status"] != "ok" or r["strict"] not in (0, 1):
                sc_bad += 1
            if r["task"] == "fc":
                if r["fc_options_sha256"] != want_fc:
                    fc_bad += 1
                if r["requested_lang"] is not None:
                    lang_bad += 1
                continue
            if r["requested_lang"] != cond["answer_lang"]:
                lang_bad += 1
            if it["qa_type"] in NAMING_TYPES:
                if r["gold_name_id"] != it["name_id"]:
                    gid_bad += 1
                if (r["en_route"] or None) != _yn(it["en_route"]) or \
                        (r["en_unique"] or None) != _yn(it["en_unique"]) or \
                        (r["origin3"] or None) != (it["origin3"] or None):
                    obj_bad += 1
    checks.check(ex_bad == 0, "excluded items as the list (D185)",
                 "records whose excluded status differs from probes/excluded_items.csv: %d" % ex_bad)
    checks.check(sc_bad == 0, "scoreable outside the list",
                 "records not scoreable, not ok, or without strict: %d (B4: counted, not dropped)" % sc_bad)
    checks.check(gid_bad == 0 and obj_bad == 0, "gold object as the lexicon join",
                 "gold_name_id differs %d; en_route, en_unique or origin3 differ %d" % (gid_bad, obj_bad))
    checks.check(lang_bad == 0 and fc_bad == 0, "requested language and option file",
                 "requested_lang differs %d; fc_options_sha256 not %s: %d"
                 % (lang_bad, want_fc[:12], fc_bad))
    ids = Counter()
    for (m, cid), slot in data.items():
        for r in slot.values():
            ids[(m, r["model_id"])] += 1
    per_model = defaultdict(set)
    for (m, mid) in ids:
        per_model[m].add(mid)
    checks.check(all(len(v) == 1 for v in per_model.values()), "one model_id per model key",
                 "; ".join("%s: %s" % (m, ", ".join(sorted(v))) for m, v in sorted(per_model.items())))
    return Records(data=data, stage=next(iter(stages)) if stages else "",
                   scorer_version=next(iter(scorers)) if scorers else "",
                   prompt_version=next(iter(prompts)) if prompts else "",
                   files=file_info, manifests=manifests, seeds=seeds)


def _yn(v):
    return None if v is None else ("yes" if v else "no")


# Derived per-record indicators -------------------------------------------

def t1_req(r):
    return _b(r["label"] == "tier1" and r["content_lang_ok"])


def t2_req(r):
    return _b(r["label"] == "tier2" and r["content_lang_ok"])


def segment(r):
    lab = r["label"]
    if lab == "tier1":
        return "tier1_req" if r["content_lang_ok"] else "tier1_other"
    return lab if lab in SEGMENTS else "other_wrong"


def strict_of(r):
    return float(r["strict"])


def invalid_of(r):
    return _b(r["label"] == "invalid")


def refusal_of(r):
    return _b(r["refusal"])


# --------------------------------------------------------------------------
# 6. B0 baseline (Design 7.1; D236)
# --------------------------------------------------------------------------

def compute_b0(world):
    """Per P-CORE-CLOSED item: the most common normalise()d gold among dataset
    items with the same qa_type and template_id from other houses; if no other
    house has the template, the most common gold of that type from other
    houses. Excluded items are out of the pool. Ties: SHA-256("42|B0|<gold>").
    Scored with score_item under L-BB (strict)."""
    norm, sc = world.normalise, world.sc
    by_tt = defaultdict(lambda: defaultdict(Counter))   # (type, tid) -> house -> Counter
    by_t = defaultdict(lambda: defaultdict(Counter))    # type -> house -> Counter
    for q, typ, question, answer, house in world.dataset_qa:
        if q in world.excluded:
            continue
        a = norm(answer)
        tid = template_id_of(template_key(question, norm))
        by_tt[(typ, tid)][house][a] += 1
        by_t[typ][house][a] += 1

    def mode_excluding(per_house, house):
        tot = Counter()
        for h, cnt in per_house.items():
            if h != house:
                tot.update(cnt)
        if not tot:
            return None
        best = max(tot.values())
        cands = [a for a, n in tot.items() if n == best]
        return min(cands, key=lambda a: hashlib.sha256(("42|B0|" + a).encode("utf-8")).hexdigest())

    out = {}
    for q in world.scored_ids["P-CORE-CLOSED"]:
        it = world.items[q]
        ans = mode_excluding(by_tt.get((it["qa_type"], it["template_id"]), {}), it["house_id"])
        src = "template"
        if ans is None:
            ans, src = mode_excluding(by_t.get(it["qa_type"], {}), it["house_id"]), "type"
        rec = sc.score_item(world.scorer_item(q), "L-BB", ans or "", world.lex, world.voc)
        out[q] = {"source": src, "strict": int(rec["strict"] or 0)}
    return out


# --------------------------------------------------------------------------
# 7. EXP-L (Design 7.1, 9.1 P1 to P4, 9.4 items 1, 2, 9)
# --------------------------------------------------------------------------

def cell_summary(boot, hidx, recs_slot, ids, mask=None):
    v = lambda fn: np.array([fn(recs_slot[q]) for q in ids], dtype=np.float64)
    m = np.ones(len(ids)) if mask is None else np.asarray(mask, dtype=np.float64)
    sub = [q for q, k in zip(ids, m) if k]
    return {"n": int(m.sum()),
            "strict": summarise_est(boot.mean(hidx, v(strict_of), m)),
            "lenient": summarise_est(boot.mean(hidx, v(lambda r: float(r["lenient"])), m)),
            "fidelity": summarise_est(boot.mean(hidx, v(lambda r: _b(r["fidelity"])), m)),
            "refusal": summarise_est(boot.mean(hidx, v(refusal_of), m)),
            "empty": float(np.mean([_b(recs_slot[q]["is_empty"]) for q in sub])) if sub else None,
            "le8_words": float(np.mean([_b(recs_slot[q]["n_words"] <= 8) for q in sub])) if sub else None,
            "mixed_script_n": int(sum(1 for q in sub if recs_slot[q]["mixed_script"])),
            "correct_with_refusal_phrase_n": int(sum(1 for q in sub if recs_slot[q]["strict"] == 1
                                                     and recs_slot[q]["refusal_phrase"])),
            "no_content": float(np.mean([_b(recs_slot[q]["content_found"] is False) for q in sub]))
            if sub else None}


def analyse_exp_l(world, recs, boot, b0):
    ids = world.scored_ids["P-CORE-CLOSED"]
    hidx = boot.hidx(world.houses_of(ids))
    types = np.array([world.items[q]["qa_type"] for q in ids])
    caps = np.array([world.items[q]["capture_type"] for q in ids])
    b0v = np.array([b0[q]["strict"] for q in ids], dtype=np.float64)
    res = {"n_items": len(ids), "houses": boot.H,
           "capture_n": {k: int((caps == k).sum()) for k in ("closeup", "group", "overview")},
           "b0": {
        "all": summarise_est(boot.mean(hidx, b0v)),
        "by_type": {t: summarise_est(boot.mean(hidx, b0v, types == t)) for t in CLOSED_TYPES},
        "source": dict(Counter(b0[q]["source"] for q in ids))}, "models": {}}
    for m in MODELS:
        acc, fid, cells, per_type, per_cap = {}, {}, {}, {}, {}
        for c in COND_L:
            slot = recs.get(m, c)
            acc[c] = boot.mean(hidx, recs.vec(m, c, ids, strict_of))
            fid[c] = boot.mean(hidx, recs.vec(m, c, ids, lambda r: _b(r["fidelity"])))
            cells[c] = cell_summary(boot, hidx, slot, ids)
            per_type[c] = {t: cell_summary(boot, hidx, slot, ids, types == t) for t in CLOSED_TYPES}
            per_cap[c] = {k: summarise_est(boot.mean(hidx, recs.vec(m, c, ids, strict_of), caps == k))
                          for k in ("closeup", "group", "overview")}
        P1 = ((acc["L-EB"] - acc["L-BB"]) + (acc["L-EE"] - acc["L-BE"])) * 0.5
        P2 = ((acc["L-BE"] - acc["L-BB"]) + (acc["L-EE"] - acc["L-EB"])) * 0.5
        P12 = P2 - P1
        P3 = (acc["L-EE"] - acc["L-EE0"]) - (acc["L-BB"] - acc["L-BB0"])
        P4 = fid["L-BB0"] - fid["L-BB"]
        res["models"][m] = {
            "cells": cells, "per_type": per_type, "per_capture": per_cap,
            "contrasts": {"P1": summarise_est(P1), "P2": summarise_est(P2),
                          "P12": summarise_est(P12, test=True), "P3": summarise_est(P3, test=True),
                          "P4": summarise_est(P4, test=True)}}
    return res


# --------------------------------------------------------------------------
# 8. EXP-N (Design 7.2, 9.1 P5 to P7, 9.3, 9.4 items 3 to 7)
# --------------------------------------------------------------------------

def p7_items(world):
    """Pooled word-form items (Design 7.2): en_unique naming items of P-NAME
    (N-BE, N-BB) and of P-CORE-CLOSED that are not in P-NAME (L-BE, L-BB); a
    qa_id in both counts once, from P-NAME (D233)."""
    pname = set(world.scored_ids["P-NAME"])
    out = []
    for q in world.scored_ids["P-NAME"]:
        if world.items[q]["en_unique"]:
            out.append((q, "N-BE", "N-BB", "N_obj_name"))
    for q in world.scored_ids["P-CORE-CLOSED"]:
        it = world.items[q]
        if it["qa_type"] in NAMING_TYPES and it["en_unique"] and q not in pname:
            out.append((q, "L-BE", "L-BB", "L_" + it["qa_type"]))
    return out


def composition(slot, ids):
    cnt = Counter(segment(slot[q]) for q in ids)
    n = len(ids)
    return {"n": n, "counts": {s: cnt.get(s, 0) for s in SEGMENTS},
            "shares": {s: (cnt.get(s, 0) / n if n else None) for s in SEGMENTS}}


def analyse_exp_n(world, recs, boot_n, boot_p7):
    ids = world.scored_ids["P-NAME"]
    hidx = boot_n.hidx(world.houses_of(ids))
    u = np.array([_b(world.items[q]["en_unique"]) for q in ids])
    clus = np.array([_b(world.items[q]["in_cluster"]) for q in ids])
    ids_u = [q for q, k in zip(ids, u) if k]
    pitems = p7_items(world)
    p7_ids = [p[0] for p in pitems]
    p7_hidx = boot_p7.hidx(world.houses_of(p7_ids))
    route = np.array([_b(world.items[q]["en_route"]) for q in p7_ids])
    origin = np.array([world.items[q]["origin3"] for q in p7_ids])
    res = {"n_items": len(ids), "n_en_unique": int(u.sum()), "houses": boot_n.H,
           "p7_items": {"n": len(pitems), "houses": boot_p7.H,
                        "by_source": dict(Counter(p[3] for p in pitems)),
                        "by_source_route": {"%s|%s" % k: v for k, v in sorted(Counter(
                            (p[3], _yn(world.items[p[0]]["en_route"])) for p in pitems).items())},
                        "origin_uncertain": int(sum(1 for q in p7_ids if world.items[q]["origin3"] not in ORIGINS)),
                        "en_route_yes": int(route.sum()), "en_route_no": int(len(route) - route.sum()),
                        "overlap_pname_pcc": sum(1 for q in world.scored_ids["P-CORE-CLOSED"]
                                                 if q in set(ids) and world.items[q]["qa_type"] in NAMING_TYPES)},
           "models": {}}
    for m in MODELS:
        t1 = {c: recs.vec(m, c, ids, t1_req) for c in COND_N_OPEN}
        t2 = {c: recs.vec(m, c, ids, t2_req) for c in COND_N_OPEN}
        t1a = {c: recs.vec(m, c, ids, lambda r: _b(r["label"] == "tier1")) for c in COND_N_OPEN}
        t2a = {c: recs.vec(m, c, ids, lambda r: _b(r["label"] == "tier2")) for c in COND_N_OPEN}
        sib = {c: recs.vec(m, c, ids, lambda r: _b(r["label"] == "sibling")) for c in COND_N_OPEN}
        fb = {c: boot_n.ratio(hidx, t2[c] * u, (t1[c] + t2[c]) * u) for c in COND_N_OPEN}
        fba = {c: boot_n.ratio(hidx, t2a[c] * u, (t1a[c] + t2a[c]) * u) for c in COND_N_OPEN}
        t1rate = {c: boot_n.mean(hidx, t1[c]) for c in COND_N_OPEN}
        P5 = fb["N-BB"] - fb["N-BE"]
        P5_any = fba["N-BB"] - fba["N-BE"]
        P6 = t1rate["N-BBs"] - t1rate["N-BB"]
        be = recs.vec(m, "N-BE", ids, strict_of)
        diff = t1["N-BBs"] - t1["N-BB"]
        g1, g0 = u * be, u * (1 - be)
        derived = {}
        for c in COND_N_OPEN:
            for scope, msk in (("all", np.ones(len(ids))), ("en_unique", u)):
                derived[(c, scope)] = {
                    "identification": summarise_est(boot_n.ratio(hidx, (t1[c] + t2[c]) * msk, msk)),
                    "specificity": summarise_est(boot_n.ratio(hidx, t1[c] * msk, (t1[c] + t2[c]) * msk)),
                    "fallback": summarise_est(boot_n.ratio(hidx, t2[c] * msk, (t1[c] + t2[c]) * msk)),
                    "sibling_confusion": summarise_est(boot_n.ratio(hidx, sib[c] * msk * clus, msk * clus)),
                    "tier1_rate": summarise_est(boot_n.ratio(hidx, t1[c] * msk, msk))}
        # Word form (P7) on the pooled items
        E = np.array([strict_of(recs.get(m, e)[q]) for q, e, b, _ in pitems])
        Bn = np.array([strict_of(recs.get(m, b)[q]) for q, e, b, _ in pitems])
        tr = {g: boot_p7.ratio(p7_hidx, E * Bn * (route == v), E * (route == v))
              for g, v in (("yes", 1), ("no", 0))}
        P7 = tr["yes"] - tr["no"]
        by_origin = {o: summarise_est(boot_p7.ratio(p7_hidx, E * Bn * (origin == o), E * (origin == o)))
                     for o in ORIGINS}
        origin_n = {o: int((E * (origin == o)).sum()) for o in ORIGINS}
        res["models"][m] = {
            "contrasts": {"P5": summarise_est(P5, test=True), "P6": summarise_est(P6, test=True),
                          "P7": summarise_est(P7, test=True)},
            "info": {"P5_any_language": summarise_est(P5_any),
                     "P6_known_in_english": summarise_est(boot_n.ratio(hidx, diff * g1, g1)),
                     "P6_not_known_in_english": summarise_est(boot_n.ratio(hidx, diff * g0, g0)),
                     "P6_known_n": int(g1.sum()), "P6_not_known_n": int(g0.sum())},
            "fallback": {c: summarise_est(fb[c]) for c in COND_N_OPEN},
            "derived": {"%s|%s" % k: v for k, v in derived.items()},
            "composition_en_unique": {c: composition(recs.get(m, c), ids_u) for c in COND_N_OPEN},
            "mixed_script_n": {c: int(sum(1 for q in ids if recs.get(m, c)[q]["mixed_script"])) for c in COND_N_OPEN},
            "composition_all": {c: composition(recs.get(m, c), ids) for c in COND_N_OPEN},
            "transfer": {g: summarise_est(tr[g]) for g in tr},
            "transfer_n": {"yes": int((E * (route == 1)).sum()), "no": int((E * (route == 0)).sum())},
            "transfer_by_origin": by_origin, "transfer_by_origin_n": origin_n}
    return res


def error_budget(world, recs, m, boot_n):
    """Design 9.3 with D234: en_unique P-CLUST items; D = not strict Tier 1 in
    N-BB; A = strict Tier 1 in N-BE, split by the N-BB label; B = correct in
    N-FCB; C = the rest. Guessing level: N-FCB0 on the items that are not A."""
    ids = [q for q in world.scored_ids["P-CLUST"] if world.items[q]["en_unique"]]
    hidx = boot_n.hidx(world.houses_of(ids))
    nbb, nbe = recs.get(m, "N-BB"), recs.get(m, "N-BE")
    fcb, fcb0 = recs.get(m, "N-FCB"), recs.get(m, "N-FCB0")
    cat = {}
    for q in ids:
        if nbb[q]["strict"] == 1:
            cat[q] = None
        elif nbe[q]["strict"] == 1:
            lab = nbb[q]["label"]
            cat[q] = {"tier1": "A1", "english_copy": "A2", "tier2": "A3"}.get(lab, "A4")
        elif fcb[q]["strict"] == 1:
            cat[q] = "B"
        else:
            cat[q] = "C"
    inD = np.array([_b(cat[q] is not None) for q in ids])
    shares = {k: summarise_est(boot_n.ratio(hidx, np.array([_b(cat[q] == k) for q in ids]), inD))
              for k in BUDGET_CATS}
    notA = np.array([_b(cat[q] in ("B", "C")) for q in ids])
    a_share = summarise_est(boot_n.ratio(hidx, np.array([_b(cat[q] in ("A1", "A2", "A3", "A4"))
                                                         for q in ids]), inD))
    guess = {"accuracy": summarise_est(boot_n.ratio(hidx, notA * np.array([strict_of(fcb0[q]) for q in ids]), notA)),
             "invalid": summarise_est(boot_n.ratio(hidx, notA * np.array([invalid_of(fcb0[q]) for q in ids]), notA)),
             "refusal": summarise_est(boot_n.ratio(hidx, notA * np.array([refusal_of(fcb0[q]) for q in ids]), notA)),
             "n": int(notA.sum())}
    a4_empty_or_refusal = sum(1 for q in ids if cat[q] == "A4" and (nbb[q]["is_empty"] or nbb[q]["refusal"]))
    a1_lang = Counter(nbb[q]["label_lang"] for q in ids if cat[q] == "A3")
    # Two-way split on all en_unique P-NAME items
    ids2 = [q for q in world.scored_ids["P-NAME"] if world.items[q]["en_unique"]]
    h2 = boot_n.hidx(world.houses_of(ids2))
    d2 = np.array([_b(nbb[q]["strict"] == 0) for q in ids2])
    a2 = d2 * np.array([strict_of(nbe[q]) for q in ids2])
    sib_pairs = sum(1 for q in ids if cat[q] is not None and nbb[q]["label"] == "sibling"
                    and any((q, s) in world.swap_pairs for s in (nbb[q]["named_ids"] or [])))
    return {"n_items": len(ids), "n_denominator": int(inD.sum()),
            "counts": {k: sum(1 for q in ids if cat[q] == k) for k in BUDGET_CATS},
            "shares": shares, "A_share": a_share, "guessing_level": guess,
            "A4_empty_or_refusal": a4_empty_or_refusal,
            "A3_by_term_language": dict(a1_lang),
            "two_way": {"n_items": len(ids2), "n_denominator": int(d2.sum()),
                        "A_share": summarise_est(boot_n.ratio(h2, a2, d2))},
            "siblings_on_ht6_pairs_in_budget": sib_pairs,
            "categories": {q: c for q, c in cat.items() if c is not None}}


def d204_counts(world, recs, m):
    """D204: sibling labels in the open naming conditions that fall on the 48
    item and sibling pairs HT-6 judged also right."""
    out = {}
    for c in COND_N_OPEN:
        slot = recs.get(m, c)
        sib = [q for q in world.scored_ids["P-NAME"] if slot[q]["label"] == "sibling"]
        on = [q for q in sib if any((q, s) in world.swap_pairs for s in (slot[q]["named_ids"] or []))]
        out[c] = {"sibling_labels": len(sib), "on_ht6_pairs": len(on)}
    return out


def analyse_fc(world, recs, m, boot_n):
    """Forced choice with and without the image, against chance (Design 9.4
    item 4), with the invalid and refusal shares (F11) and per cluster."""
    out = {}
    for c_img, c_no, probe in (("N-FCB", "N-FCB0", "P-CLUST"), ("N-FCE", "N-FCE0", "P-FCE")):
        ids = world.scored_ids[probe]
        hidx = boot_n.hidx(world.houses_of(ids))
        acc = {c: boot_n.mean(hidx, recs.vec(m, c, ids, strict_of)) for c in (c_img, c_no)}
        block = {}
        for c in (c_img, c_no):
            block[c] = {"n": len(ids), "accuracy": summarise_est(acc[c]),
                        "invalid": summarise_est(boot_n.mean(hidx, recs.vec(m, c, ids, invalid_of))),
                        "refusal": summarise_est(boot_n.mean(hidx, recs.vec(m, c, ids, refusal_of))),
                        "vs_chance": summarise_est(acc[c] - boot_n.const(0.25), test=True)}
        block["image_effect"] = summarise_est(acc[c_img] - acc[c_no], test=True)
        clusters = sorted({nfc(world.items[q]["cluster_id"]) for q in ids})
        by_cl = {}
        for cl in clusters:
            msk = np.array([_b(nfc(world.items[q]["cluster_id"]) == cl) for q in ids])
            row = {"n": int(msk.sum())}
            for c in (c_img, c_no):
                v = recs.vec(m, c, ids, strict_of)
                if probe == "P-FCE" and cl.startswith(nfc(COOKING_POTS)):
                    row[c] = {"correct": int((v * msk).sum()), "count_only": True}
                else:
                    row[c] = summarise_est(boot_n.mean(hidx, v, msk))
            by_cl[cl] = row
        block["by_cluster"] = by_cl
        out[probe] = block
    return out


def analyse_cultural(world, recs, boot_n):
    """Design 9.4 item 6 (exploratory, descriptive): culturally specific
    against globally common; N-BE on en_unique items only (D77)."""
    ids = world.scored_ids["P-NAME"]
    hidx = boot_n.hidx(world.houses_of(ids))
    u = np.array([_b(world.items[q]["en_unique"]) for q in ids])
    tag = np.array([world.items[q]["cultural"] for q in ids])
    out = {"counts": dict(Counter(tag.tolist())), "models": {}}
    for m in MODELS:
        row = {}
        for c in COND_N_OPEN:
            v = recs.vec(m, c, ids, strict_of)
            base = u if c == "N-BE" else np.ones(len(ids))
            row[c] = {t: dict(summarise_est(boot_n.mean(hidx, v, base * (tag == t))),
                              n=int((base * (tag == t)).sum()))
                      for t in ("culturally_specific", "globally_common")}
        out["models"][m] = row
    return out


def analyse_visual(world, recs, boot_l, boot_n):
    """Design 9.4 item 9: strict accuracy by box-area tertile, for items on
    close-up images with a box (EXP-L obj_name and obj_property; P-NAME)."""
    out = {}
    for name, ids_all, conds, boot in (
            ("EXP-L", world.scored_ids["P-CORE-CLOSED"], COND_L_IMG, boot_l),
            ("EXP-N", world.scored_ids["P-NAME"], ("N-BB", "N-BE"), boot_n)):
        ids = [q for q in ids_all if world.items[q]["box_area"] is not None]
        if not ids:
            out[name] = {"n_with_box": 0}
            continue
        areas = np.array([world.items[q]["box_area"] for q in ids])
        q1, q2 = (float(x) for x in np.quantile(areas, [1 / 3, 2 / 3]))
        tert = np.where(areas <= q1, "low", np.where(areas <= q2, "mid", "high"))
        hidx = boot.hidx(world.houses_of(ids))
        u = np.array([_b(world.items[q]["en_unique"]) for q in ids])
        block = {"n_with_box": len(ids), "n_close_up": sum(1 for q in ids_all if world.items[q]["capture_type"] == "closeup"),
                 "cuts": [q1, q2], "models": {}}
        for m in MODELS:
            block["models"][m] = {}
            for c in conds:
                v = recs.vec(m, c, ids, strict_of)
                base = u if c == "N-BE" else np.ones(len(ids))
                block["models"][m][c] = {t: dict(summarise_est(boot.mean(hidx, v, base * (tert == t))),
                                                 n=int((base * (tert == t)).sum()))
                                         for t in ("low", "mid", "high")}
        out[name] = block
    return out


def analyse_annotators(world, recs, n_boot=N_BOOT):
    """Design 9.4 item 10: strict accuracy per annotator (house_assignment.csv),
    with a bootstrap over that annotator's houses."""
    out = {}
    sets = (("EXP-L", world.scored_ids["P-CORE-CLOSED"], COND_L),
            ("EXP-N", world.scored_ids["P-NAME"], ("N-BB", "N-BE", "N-BBs")))
    for name, ids, conds in sets:
        anns = sorted({world.items[q]["annotator"] for q in ids})
        block = {}
        for a in anns:
            sub = [q for q in ids if world.items[q]["annotator"] == a]
            boot = HouseBoot(world.houses_of(sub), n_boot=n_boot, name="%s %s" % (name, a))
            hidx = boot.hidx(world.houses_of(sub))
            u = np.array([_b(world.items[q]["en_unique"]) for q in sub])
            block[a] = {"houses": boot.H, "n": len(sub), "models": {}}
            for m in MODELS:
                block[a]["models"][m] = {}
                for c in conds:
                    base = u if c == "N-BE" else None
                    block[a]["models"][m][c] = summarise_est(boot.mean(hidx, recs.vec(m, c, sub, strict_of), base))
        out[name] = block
    return out


# --------------------------------------------------------------------------
# 9. EXP-R (Design 7.3, 9.1 P8, 9.2 seeds; D237)
# --------------------------------------------------------------------------

def analyse_exp_r(world, recs, boot):
    ids = world.scored_ids["R-TEST"]
    hidx = boot.hidx(world.houses_of(ids))
    naming = np.array([_b(world.items[q]["qa_type"] in NAMING_TYPES) for q in ids])
    seeds = list(recs.seeds)
    res = {"n_items": len(ids), "houses": boot.H, "seeds": seeds,
           "house_items": dict(Counter(world.houses_of(ids)).most_common(3))}
    if not seeds:
        res["skipped"] = "no fine-tuned seed in the records"
        return res
    versions = [FT_BASE] + seeds
    V = {}
    for v in versions:
        V[v] = {"BN": recs.vec(v, "R-BB", ids, strict_of), "EN": recs.vec(v, "R-EE", ids, strict_of),
                "NOIMG": recs.vec(v, "R-BB0", ids, strict_of),
                "FID": recs.vec(v, "R-BB", ids, lambda r: _b(r["fidelity"])),
                "T1": recs.vec(v, "R-BB", ids, t1_req), "T2": recs.vec(v, "R-BB", ids, t2_req),
                "LEN": recs.vec(v, "R-BB", ids, lambda r: float(r["n_words"])),
                "REF0": recs.vec(v, "R-BB0", ids, refusal_of)}
    deltas, per_seed = {}, {}
    for key in ("BN", "EN", "NOIMG", "FID", "LEN"):
        base = boot.mean(hidx, V[FT_BASE][key])
        seed_est = {s: boot.mean(hidx, V[s][key]) - base for s in seeds}
        avg = boot.mean(hidx, sum(V[s][key] for s in seeds) / len(seeds)) - base
        deltas[key] = summarise_est(avg, test=(key in ("BN", "EN")))
        per_seed[key] = {s: seed_est[s].point for s in seeds}
    base_t1 = boot.mean(hidx, V[FT_BASE]["T1"], naming)
    deltas["TIER1"] = summarise_est(boot.mean(hidx, sum(V[s]["T1"] for s in seeds) / len(seeds), naming) - base_t1)
    per_seed["TIER1"] = {s: (boot.mean(hidx, V[s]["T1"], naming) - base_t1).point for s in seeds}
    fb = {v: boot.ratio(hidx, V[v]["T2"] * naming, (V[v]["T1"] + V[v]["T2"]) * naming) for v in versions}
    fb_d = {s: fb[s] - fb[FT_BASE] for s in seeds}
    deltas["FALLBACK"] = summarise_est(est_mean(fb_d.values()))
    per_seed["FALLBACK"] = {s: fb_d[s].point for s in seeds}
    avg_bb = sum(V[s]["BN"] for s in seeds) / len(seeds)
    avg_ee = sum(V[s]["EN"] for s in seeds) / len(seeds)
    P8 = boot.mean(hidx, (avg_bb - V[FT_BASE]["BN"]) - (avg_ee - V[FT_BASE]["EN"]))
    res.update({"deltas": deltas, "per_seed": per_seed,
                "P8": summarise_est(P8, test=True),
                "P8_per_seed": {s: (boot.mean(hidx, (V[s]["BN"] - V[FT_BASE]["BN"])
                                              - (V[s]["EN"] - V[FT_BASE]["EN"]))).point for s in seeds},
                "levels": {v: {k: summarise_est(boot.mean(hidx, V[v][k])) for k in
                               ("BN", "EN", "NOIMG", "FID", "REF0")} for v in versions}})
    return res


# --------------------------------------------------------------------------
# 10. GEE (Design 9.2; D235)
# --------------------------------------------------------------------------

GEE_BACKEND = None     # tests may set a callable(y, X, names, groups) -> dict


def gee_fit(y, X, names, groups):
    if GEE_BACKEND is not None:
        return GEE_BACKEND(y, X, names, groups)
    import statsmodels.api as sm
    labels, codes = np.unique(np.asarray(groups), return_inverse=True)
    model = sm.GEE(np.asarray(y, dtype=float), np.asarray(X, dtype=float), groups=codes,
                   family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = model.fit(maxiter=100)
    ci = np.asarray(res.conf_int(alpha=0.05))
    dep = getattr(model.cov_struct, "dep_params", None)
    return {"terms": {n: {"estimate": float(res.params[i]), "se": float(res.bse[i]),
                          "lo": float(ci[i, 0]), "hi": float(ci[i, 1]), "p": float(res.pvalues[i])}
                      for i, n in enumerate(names)},
            "n_obs": int(len(y)), "n_groups": int(len(labels)),
            "dependence": None if dep is None else float(np.ravel(dep)[0]),
            "warnings": sorted({"%s: %s" % (w.category.__name__, str(w.message)[:120]) for w in caught})}


def _safe_gee(y, X, names, groups):
    try:
        return gee_fit(y, X, names, groups)
    except Exception as exc:          # a failed fit is a result, recorded
        return {"error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}


def design_exp_l(world, recs, m, group_by="house", tokens=None):
    """correct ~ q_bn * a_bn + C(qa_type) + capture (D235). Close-up is the same
    as qa_type in {obj_name, obj_property} in P-CORE-CLOSED, so capture enters
    as one indicator, overview against group; if that identity does not hold,
    capture enters with full dummies. With tokens: + log ratio + q_bn:log ratio."""
    ids = world.scored_ids["P-CORE-CLOSED"]
    closeup_identity = all((world.items[q]["capture_type"] == "closeup")
                           == (world.items[q]["qa_type"] in ("obj_name", "obj_property")) for q in ids)
    types = [t for t in CLOSED_TYPES if t != "count"]
    names = ["const", "q_bn", "a_bn", "q_bn:a_bn"] + ["type[%s]" % t for t in types]
    names += ["overview"] if closeup_identity else ["capture[group]", "capture[overview]"]
    if tokens is not None:
        names += ["log_tok_ratio", "q_bn:log_tok_ratio"]
    y, X, g = [], [], []
    for c in COND_L_IMG:
        qb = 1.0 if c in ("L-BB", "L-BE") else 0.0
        ab = 1.0 if c in ("L-BB", "L-EB") else 0.0
        slot = recs.get(m, c)
        for q in ids:
            it = world.items[q]
            row = [1.0, qb, ab, qb * ab] + [_b(it["qa_type"] == t) for t in types]
            if closeup_identity:
                row.append(_b(it["capture_type"] == "overview"))
            else:
                row += [_b(it["capture_type"] == "group"), _b(it["capture_type"] == "overview")]
            if tokens is not None:
                lr = tokens[q]
                row += [lr, qb * lr]
            y.append(float(slot[q]["strict"]))
            X.append(row)
            g.append(it["house_id"] if group_by == "house" else it["template_id"])
    return np.array(y), np.array(X), names, g, closeup_identity


def design_word_form(world, recs, m):
    """correct ~ a_bn * en_route + log(freq_obj_name + 1) + C(item_type) +
    in_cluster on the pooled en_unique naming items with a Bangla question."""
    pitems = p7_items(world)
    kinds = ["L_obj_name", "L_object"]
    names = ["const", "a_bn", "en_route", "a_bn:en_route", "log_freq"] + \
            ["item_type[%s]" % k for k in kinds] + ["in_cluster"]
    y, X, g = [], [], []
    for q, ce, cb, kind in pitems:
        it = world.items[q]
        er = _b(it["en_route"])
        for c, ab in ((cb, 1.0), (ce, 0.0)):
            y.append(float(recs.get(m, c)[q]["strict"]))
            X.append([1.0, ab, er, ab * er, math.log(it["freq"] + 1)] +
                     [_b(kind == k) for k in kinds] + [_b(it["in_cluster"])])
            g.append(it["house_id"])
    return np.array(y), np.array(X), names, g


def analyse_gee(world, recs, tokens_by_model=None):
    out = {}
    for m in MODELS:
        y, X, names, g, ident = design_exp_l(world, recs, m, "house")
        house = _safe_gee(y, X, names, g)
        y2, X2, names2, g2, _ = design_exp_l(world, recs, m, "template")
        templ = _safe_gee(y2, X2, names2, g2)
        yw, Xw, nw, gw = design_word_form(world, recs, m)
        wf = _safe_gee(yw, Xw, nw, gw)
        tok = None
        if tokens_by_model and m in tokens_by_model:
            y3, X3, n3, g3, _ = design_exp_l(world, recs, m, "house", tokens=tokens_by_model[m])
            tok = _safe_gee(y3, X3, n3, g3)
        out[m] = {"exp_l_house": house, "exp_l_template": templ, "word_form": wf,
                  "token_cost": tok, "closeup_identity": ident}
    return out


def _sign(x):
    return 0 if x is None or (isinstance(x, float) and math.isnan(x)) or x == 0 else (1 if x > 0 else -1)


def direction_checks(contrasts, gee):
    """Design 9.1 with D235: P1 against -(b_q + b_qa/2), P2 against
    -(b_a + b_qa/2), P7 against b(a_bn:en_route), on the log-odds scale."""
    def term(fit, name):
        return (fit or {}).get("terms", {}).get(name, {}).get("estimate")
    house, wf = gee.get("exp_l_house"), gee.get("word_form")
    bq, ba, bqa = term(house, "q_bn"), term(house, "a_bn"), term(house, "q_bn:a_bn")
    vals = {"P1": None if None in (bq, bqa) else -(bq + bqa / 2.0),
            "P2": None if None in (ba, bqa) else -(ba + bqa / 2.0),
            "P7": term(wf, "a_bn:en_route")}
    out = {}
    for k, v in vals.items():
        s1, s2 = _sign((contrasts.get(k) or {}).get("point")), _sign(v)
        out[k] = {"gee_value": v, "agrees": "n/a" if v is None else
                  ("agrees" if s1 == s2 and s1 != 0 else ("zero" if 0 in (s1, s2) else "disagrees"))}
    return out


# --------------------------------------------------------------------------
# 11. Human checks: HT-2, HT-3, HT-4, HT-5 (Design 9.5 as used by T5; D238)
# --------------------------------------------------------------------------

def ht2_row():
    k, n = HT2_FAITHFUL
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "point": k / n, "lo": lo, "hi": hi, "source": "addendum D7"}


def read_ht3(ht3_dir, world, checks, mode):
    """The HT-3 score folder of asset12_nameref.py (final after gate-F, D208).
    Each counted name is scored again in memory with the imported scorer, so the
    human rows use exactly the model rows' segments; the label and strict
    values must equal the file's."""
    d = Path(ht3_dir) if ht3_dir else None
    jp, cp = (d / "ht3_agreement.json", d / "ht3_scored_names.csv") if d else (None, None)
    if d is None or not jp.exists() or not cp.exists():
        checks.add("FAIL", "HT-3 files", "missing ht3_agreement.json or ht3_scored_names.csv in %s" % d)
        raise StatsError("HT-3 files missing")
    js = json.loads(jp.read_text(encoding="utf-8"))
    rows = read_csv_dicts(cp)
    want_mode = "placeholder" if mode == "placeholder" else "final"
    summ = js.get("summary", {})
    comp = summ.get("composition", {})
    counted = set(world.scored_ids["NAME-REF"])
    labels = defaultdict(dict)
    differ = 0
    for r in rows:
        q = r["qa_id"]
        if q not in counted or r.get("excluded") == "yes":
            continue
        rec = world.sc.score_item(world.scorer_item(q), "N-BB", r["name"], world.lex, world.voc)
        if rec["label"] != r["label"] or str(rec["strict"]) != str(r["strict"]):
            differ += 1
        labels[r["role"]][q] = {k: rec.get(k) for k in SLIM}
    full = all(set(labels[role]) == counted for role in ("N1", "N2"))
    cross = full and all(Counter(labels["N1"][q]["label"] for q in counted).get(lab, 0)
                         == comp.get("N1", {}).get("counts", {}).get(lab, -1) for lab in LABELS_NAMING)
    ok = (js.get("mode") == want_mode and bool(js.get("reportable")) == (mode == "real")
          and summ.get("n_counted") == SCORED_SIZES["NAME-REF"] and len(counted) == SCORED_SIZES["NAME-REF"]
          and js.get("primary_role") == "N1" and js.get("lexicon_sha256") == world.hashes["lexicon.csv"]
          and js.get("scorer_version") == world.sc.SCORER_VERSION and full and cross and differ == 0)
    checks.check(ok, "HT-3 %s files" % want_mode,
                 "mode %s, reportable %s, counted %s, primary role %s, lexicon as gate-A %s, scorer %s;"
                 " both namers on every counted item %s; N1 labels equal the json %s; labels that differ"
                 " when scored again %d" % (js.get("mode"), js.get("reportable"), summ.get("n_counted"),
                                           js.get("primary_role"),
                                           js.get("lexicon_sha256") == world.hashes["lexicon.csv"],
                                           js.get("scorer_version"), full, cross, differ),
                 pinned=(mode == "placeholder"))
    return {"json": js, "labels": labels, "sha256": {"json": sha256_file(jp), "csv": sha256_file(cp)}}


def human_reference(world, ht3, recs, boot_nr):
    """Design 9.4 item 7 on the en_unique NAME-REF items: tier composition and
    generic fallback (Bangla) for N1 (main row), N2, both pooled, and the three
    models on the same items (N-BB), with a bootstrap over the NAME-REF houses."""
    ids = [q for q in world.scored_ids["NAME-REF"] if world.items[q]["en_unique"]]
    hidx = boot_nr.hidx(world.houses_of(ids))

    def block(slots):
        t1 = sum(np.array([t1_req(s[q]) for q in ids]) for s in slots)
        t2 = sum(np.array([t2_req(s[q]) for q in ids]) for s in slots)
        cnt = Counter(segment(s[q]) for s in slots for q in ids)
        n = len(ids) * len(slots)
        return {"n": n, "counts": {g: cnt.get(g, 0) for g in SEGMENTS},
                "shares": {g: (cnt.get(g, 0) / n if n else None) for g in SEGMENTS},
                "fallback": summarise_est(boot_nr.ratio(hidx, t2, t1 + t2)),
                "tier1_rate": summarise_est(boot_nr.ratio(hidx, t1, np.full(len(ids), float(len(slots)))))}
    out = {"n_items": len(ids), "houses": boot_nr.H,
           "human N1": block([ht3["labels"]["N1"]]), "human N2": block([ht3["labels"]["N2"]]),
           "human pooled": block([ht3["labels"]["N1"], ht3["labels"]["N2"]])}
    for m in MODELS:
        out[m] = block([recs.get(m, "N-BB")])
    return out


def _need_file(path, what, checks):
    if path is None or not Path(path).exists():
        checks.add("FAIL", what, "missing: %s" % path)
        return False
    return True


def read_ht4(path, world, recs, checks):
    if not _need_file(path, "HT-4 labels (D238)", checks):
        return []
    rows = read_csv_dicts(path)
    hdr = csv_header(path)
    bad_hdr = [c for c in HT4_COLUMNS if c not in hdr]
    bad_lab = sum(1 for r in rows for c in ("scorer_label", "checker1_label", "checker2_label",
                                            "consensus_label") if r[c] not in HT4_LABELS)
    excl = sum(1 for r in rows if r["qa_id"] in world.excluded)
    mismatch = 0
    for r in rows:
        key = (r["model_key"], r["condition_id"])
        rec = recs.data.get(key, {}).get(r["qa_id"])
        if rec is None or rec["label"] != r["scorer_label"]:
            mismatch += 1
    checks.check(not bad_hdr and bad_lab == 0 and excl == 0 and mismatch == 0 and rows,
                 "HT-4 labels (D238)", "%d answers; unknown labels %d; excluded items %d (D185);"
                 " scorer label differs from the scored file %d%s"
                 % (len(rows), bad_lab, excl, mismatch, (", missing columns %s" % bad_hdr) if bad_hdr else ""))
    return rows


def ht4_stats(rows, n_boot=N_BOOT, seed=SEED):
    """Scorer accuracy against the consensus label (overall and per consensus
    label) and Cohen's kappa for scorer-checker1, scorer-checker2,
    checker1-checker2 and scorer-consensus, each with a house-clustered
    bootstrap CI. The HT-4 chat reads its pass rule from these values."""
    labels = sorted({r[c] for r in rows for c in ("scorer_label", "checker1_label",
                                                  "checker2_label", "consensus_label")})
    li = {l: i for i, l in enumerate(labels)}
    boot = HouseBoot([r["house_id"] for r in rows], n_boot=n_boot, seed=seed, name="HT-4")
    hidx = boot.hidx([r["house_id"] for r in rows])
    K = len(labels)
    out = {"n": len(rows), "houses": boot.H}
    corr = np.array([_b(r["scorer_label"] == r["consensus_label"]) for r in rows])
    out["accuracy"] = summarise_est(boot.mean(hidx, corr))
    per = {}
    for lab in labels:
        msk = np.array([_b(r["consensus_label"] == lab) for r in rows])
        if msk.sum():
            per[lab] = dict(summarise_est(boot.mean(hidx, corr, msk)), n=int(msk.sum()))
    out["accuracy_by_consensus_label"] = per
    for a, b, name in (("scorer_label", "checker1_label", "scorer_vs_checker1"),
                       ("scorer_label", "checker2_label", "scorer_vs_checker2"),
                       ("checker1_label", "checker2_label", "checker1_vs_checker2"),
                       ("scorer_label", "consensus_label", "scorer_vs_consensus")):
        C = np.zeros((boot.H, K, K))
        for h, r in zip(hidx, rows):
            C[h, li[r[a]], li[r[b]]] += 1
        po, k = kappa_from_confusion(C.sum(axis=0))
        bo = np.tensordot(boot.W, C, axes=(1, 0))
        _, kb = kappa_from_confusion(bo)
        out[name] = {"agreement": float(po), "kappa": None if np.isnan(k) else float(k),
                     "kappa_ci": summarise_est(Est(float(k), kb))}
    return out


def read_ht5(path, world, recs, budgets, checks):
    if not _need_file(path, "HT-5 audit (D238)", checks):
        return []
    rows = read_csv_dicts(path)
    hdr = csv_header(path)
    bad_hdr = [c for c in HT5_COLUMNS if c not in hdr]
    bad_val = sum(1 for r in rows if r["stratum"] not in HT5_STRATA or r["verdict"] not in HT5_VERDICTS)
    excl = sum(1 for r in rows if r["qa_id"] in world.excluded)
    missing = sum(1 for r in rows if r["qa_id"] not in recs.data.get((r["model_key"], r["condition_id"]), {}))
    not_c = sum(1 for r in rows if r["stratum"] == "C" and (
        r["condition_id"] != "N-BB" or budgets.get(r["model_key"], {}).get("categories", {}).get(r["qa_id"]) != "C"))
    n_by = Counter(r["stratum"] for r in rows)
    checks.check(not bad_hdr and bad_val == 0 and excl == 0 and missing == 0 and not_c == 0 and rows,
                 "HT-5 audit (D238)", "%d answers (%s); unknown values %d; excluded %d; not in the"
                 " scored files %d; C answers outside the error-budget C %d"
                 % (len(rows), ", ".join("%s %d" % kv for kv in sorted(n_by.items())), bad_val, excl,
                    missing, not_c))
    if dict(n_by) != HT5_STRATA_N:
        checks.add("WARN", "HT-5 strata sizes", "%s; Design 10 plans C 140, english_copy 30, sibling 30"
                   % dict(n_by))
    return rows


def ht5_stats(rows, n_boot=N_BOOT, seed=SEED):
    boot = HouseBoot([r["house_id"] for r in rows], n_boot=n_boot, seed=seed, name="HT-5")
    hidx = boot.hidx([r["house_id"] for r in rows])
    out = {"n": len(rows), "houses": boot.H, "strata": {}}
    for s in HT5_STRATA:
        msk = np.array([_b(r["stratum"] == s) for r in rows])
        out["strata"][s] = {"n": int(msk.sum()), "verdicts": {
            v: summarise_est(boot.mean(hidx, np.array([_b(r["verdict"] == v) for r in rows]), msk))
            for v in HT5_VERDICTS}}
    c = np.array([_b(r["stratum"] == "C") for r in rows])
    gv = np.array([_b(r["verdict"] in HT5_GOLD_OR_VISIBILITY) for r in rows])
    out["C_gold_or_visibility"] = summarise_est(boot.mean(hidx, gv, c))
    out["gold_error_rate"] = summarise_est(boot.mean(hidx, np.array([_b(r["verdict"] == "gold_wrong") for r in rows])))
    return out


# --------------------------------------------------------------------------
# 12. Lexicon sensitivity (Design 9.4 item 12; D192, D238)
# --------------------------------------------------------------------------

def read_sensitivity(path, checks):
    if not _need_file(path, "lexicon sensitivity rows (D238)", checks):
        return []
    rows = read_csv_dicts(path)
    hdr = csv_header(path)
    bad = []
    for r in rows:
        if r["source"] not in SENS_SOURCES or r["op"] not in SENS_OPS:
            bad.append(r.get("row_id"))
        elif r["op"] != "item_any_class" and (r["target"] not in SENS_TARGETS or not r["target_id"]
                                              or r["lang"] not in ("bn", "en") or not r["form"]
                                              or (r["target"] == "lexicon" and r["tier"] not in ("1", "2"))):
            bad.append(r.get("row_id"))
        elif r["op"] == "item_any_class" and not re.fullmatch(r"H\d{2}_\d{4}_q\d+", r["target_id"]):
            bad.append(r.get("row_id"))
    miss = [c for c in SENS_COLUMNS if c not in hdr]
    checks.check(not bad and not miss and rows, "lexicon sensitivity rows (D238)",
                 "%d rows (%s); rows that break the format %s%s"
                 % (len(rows), ", ".join("%s %d" % kv for kv in sorted(Counter(r["source"] for r in rows).items())),
                    bad or "none", (", missing columns %s" % miss) if miss else ""))
    return rows


def _apply_rows(world, rows, strict=True):
    """Copies of the frozen rows with the sensitivity rows applied in file
    order (never written back). Returns (lexicon, vocabulary, item_any_class
    qa_ids). strict=False (the leave-one-out counts) lets a remove_term whose
    class or object is missing pass as a no-op."""
    sc = world.sc
    lex_rows = sc._read_csv(world.paths.repo / "lexicon" / "lexicon.csv", sc.LEXICON_COLUMNS)
    voc_rows = sc._read_csv(world.paths.repo / "lexicon" / "closed_vocab.csv", sc.VOCAB_COLUMNS)
    lex_by = {r["name_id"]: r for r in lex_rows}
    frozen_classes = {(v["kind"], v["class_id"]) for v in voc_rows}
    any_class = set()

    def edit(cell, form, add):
        parts = [p for p in re.split(r"[|;\n]", cell or "") if p.strip()]
        if add:
            if form not in parts:
                parts.append(form)
        else:
            parts = [p for p in parts if world.normalise(p) != world.normalise(form)]
        return "|".join(parts)

    for r in rows:
        if r["op"] == "item_any_class":
            any_class.add(r["target_id"])
            continue
        add = r["op"] == "add_term"
        if r["target"] == "lexicon":
            row = lex_by.get(r["target_id"])
            if row is None:
                raise StatsError("sensitivity row %s: no object %s in the lexicon" % (r["row_id"], r["target_id"]))
            col = {("1", "bn"): "variants_bn", ("1", "en"): "variants_en",
                   ("2", "bn"): "super_bn", ("2", "en"): "super_en"}[(r["tier"], r["lang"])]
            if not add and col == "variants_bn" and world.normalise(row["name_bn"]) == world.normalise(r["form"]):
                raise StatsError("sensitivity row %s: the main Bangla name cannot be removed" % r["row_id"])
            if not add and col == "variants_en":
                row["canonical_en"] = edit(row["canonical_en"], r["form"], False)
            row[col] = edit(row[col], r["form"], add)
        else:
            col = "forms_bn" if r["lang"] == "bn" else "forms_en"
            hit = [v for v in voc_rows if v["kind"] == r["target"] and v["class_id"] == r["target_id"]]
            if not hit:
                if not add:
                    if not strict:
                        continue
                    raise StatsError("sensitivity row %s: no %s class %s to remove a form from"
                                     % (r["row_id"], r["target"], r["target_id"]))
                voc_rows.append({"kind": r["target"], "class_id": r["target_id"], "forms_bn": "",
                                 "forms_en": "", "note": "sensitivity"})
                hit = voc_rows[-1:]
            hit[0][col] = edit(hit[0][col], r["form"], add)
    empty = [v for v in voc_rows if not v["forms_bn"] and not v["forms_en"]]
    lost = [v for v in empty if (v["kind"], v["class_id"]) in frozen_classes]
    if lost and strict:
        raise StatsError("sensitivity rows remove every form of %s; fix the rows"
                         % ", ".join("%s %s" % (v["kind"], v["class_id"]) for v in lost))
    voc_rows = [v for v in voc_rows if v["forms_bn"] or v["forms_en"]]
    lex = sc.build_lexicon(lex_rows, world.lex.settings, "lexicon (sensitivity copy)")
    voc = sc.build_closed_vocab(voc_rows, world.voc.settings, "closed_vocab (sensitivity copy)")
    return lex, voc, any_class


def _rescore(world, rec_slot, qa_id, cid, lex, voc, any_class):
    sc = world.sc
    r = rec_slot[qa_id]
    new = sc.score_item(world.scorer_item(qa_id), cid, r["answer_raw"], lex, voc)
    if qa_id in any_class and new.get("kind") in ("material", "colour") and new.get("found_value"):
        found = set(new["found_value"])
        gold = set(new.get("gold_value") or [])
        ok = bool(found & gold)
        new["label"] = "correct" if ok else "wrong"
        new["lenient"] = int(ok)
        new["strict"] = int(ok and bool(new.get("content_lang_ok")))
    return {k: new.get(k) for k in ("label", "strict", "lenient", "content_lang_ok", "named_ids",
                                    "is_empty", "refusal")}


def sensitivity_records(world, recs, rows):
    """Re-score every open answer outside the excluded list with all the
    sensitivity rows applied (forced choice is unchanged: it reads no
    lexicon). Returns new Records and the change counts: per model and
    condition, and per row as a leave-one-out count (answers whose label or
    score differs between all rows and all rows but this one)."""
    lex, voc, any_class = _apply_rows(world, rows)
    new_data, changed = {}, Counter()
    for (m, cid), slot in recs.data.items():
        if cid in COND_N_FC:
            new_data[(m, cid)] = slot
            continue
        ns = {}
        for q, r in slot.items():
            if r["gold_status"] == "excluded":
                ns[q] = r
                continue
            upd = _rescore(world, slot, q, cid, lex, voc, any_class)
            nr = dict(r)
            nr.update(upd)
            if (nr["label"], nr["strict"], nr["lenient"]) != (r["label"], r["strict"], r["lenient"]):
                changed[(m, cid)] += 1
            ns[q] = nr
        new_data[(m, cid)] = ns
    per_row = []
    norm = world.normalise
    for i, r in enumerate(rows):
        lex1, voc1, any1 = _apply_rows(world, rows[:i] + rows[i + 1:], strict=False)
        form = norm(r.get("form", ""))
        n = 0
        for (m, cid), slot in recs.data.items():
            if cid in COND_N_FC:
                continue
            for q, rr in slot.items():
                if rr["gold_status"] == "excluded":
                    continue
                if r["op"] == "item_any_class":
                    if q != r["target_id"]:
                        continue
                elif form not in norm(rr["answer_raw"] or "") and form not in norm(world.items[q]["answer_bn"]):
                    continue
                upd = _rescore(world, slot, q, cid, lex1, voc1, any1)
                full = new_data[(m, cid)][q]
                if (upd["label"], upd["strict"], upd["lenient"]) != (full["label"], full["strict"], full["lenient"]):
                    n += 1
        per_row.append({"row_id": r["row_id"], "source": r["source"], "op": r["op"],
                        "target": r["target"], "target_id": r["target_id"], "changed": n})
    new = Records(data=new_data, stage=recs.stage, scorer_version=recs.scorer_version,
                  prompt_version=recs.prompt_version, files=recs.files, manifests=recs.manifests,
                  seeds=recs.seeds)
    return new, {"changed_by_model_condition": {"%s|%s" % k: v for k, v in sorted(changed.items())},
                 "changed_total": int(sum(changed.values())), "per_row": per_row}


# --------------------------------------------------------------------------
# 13. Metric check (Design 9.4 item 13; D239)
# --------------------------------------------------------------------------

BERTSCORE_MODEL = "bert-base-multilingual-cased"
BERTSCORE_LAYERS = 9
METRIC_BACKEND = None   # tests may set a callable(cands, refs) -> (chrf list, bert list, info)


def _bertscore_f1(cands, refs, batch=64):
    """BERTScore F1 (greedy cosine matching of token embeddings; Zhang et al.)
    with transformers directly, because the bert_score package does not run
    under transformers 5 (D239). Layer 9 of bert-base-multilingual-cased, no
    idf, no baseline rescaling, special tokens left out of the matching, as
    in bert_score. An empty answer scores 0."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BERTSCORE_MODEL)
    model = AutoModel.from_pretrained(BERTSCORE_MODEL)
    model.eval()

    def embed(texts):
        out = []
        for i in range(0, len(texts), batch):
            part = list(texts[i:i + batch])
            enc = tok(part, padding=True, truncation=True, max_length=510, return_tensors="pt",
                      return_special_tokens_mask=True)
            special = enc.pop("special_tokens_mask").bool()
            with torch.no_grad():
                h = model(**enc, output_hidden_states=True).hidden_states[BERTSCORE_LAYERS]
            h = torch.nn.functional.normalize(h.float(), dim=-1)
            keep = enc["attention_mask"].bool() & ~special
            out += [h[j][keep[j]] for j in range(len(part))]
        return out
    C, R = embed(cands), embed(refs)
    f1 = []
    for c, r in zip(C, R):
        if c.shape[0] == 0 or r.shape[0] == 0:
            f1.append(0.0)
            continue
        sim = c @ r.T
        p, rc = float(sim.max(dim=1).values.mean()), float(sim.max(dim=0).values.mean())
        f1.append(2 * p * rc / (p + rc) if p + rc > 0 else 0.0)
    return f1


def _metric_backend(cands, refs):
    if METRIC_BACKEND is not None:
        return METRIC_BACKEND(cands, refs)
    from sacrebleu.metrics import CHRF
    import sacrebleu
    import transformers
    chrf = CHRF(word_order=2)
    chrf_scores = [chrf.sentence_score(c, [r]).score for c, r in zip(cands, refs)]
    bert = _bertscore_f1(cands, refs)
    info = {"sacrebleu": sacrebleu.__version__, "transformers": transformers.__version__,
            "chrf_signature": str(chrf.get_signature()), "bertscore": "own code, bert_score method",
            "bertscore_model": BERTSCORE_MODEL, "bertscore_layers": BERTSCORE_LAYERS, "idf": False,
            "rescale_with_baseline": False}
    return chrf_scores, bert, info


def metric_check(world, recs):
    norm = world.normalise
    ids = world.scored_ids["P-NAME"]
    out = {"models": {}}
    for m in MODELS:
        slot = recs.get(m, "N-BB")
        cands = [norm(slot[q]["answer_raw"] or "") for q in ids]
        refs = [norm(world.items[q]["answer_bn"]) for q in ids]
        chrf, bert, info = _metric_backend(cands, refs)
        seg = [segment(slot[q]) for q in ids]
        per = {}
        for s in SEGMENTS:
            idx = [i for i, x in enumerate(seg) if x == s]
            per[s] = {"n": len(idx), "chrf": float(np.mean([chrf[i] for i in idx])) if idx else None,
                      "bertscore_f1": float(np.mean([bert[i] for i in idx])) if idx else None}
        t1 = [i for i, x in enumerate(seg) if x == "tier1_req"]
        aucs = {}
        for other in ("tier2", "sibling"):
            o = [i for i, x in enumerate(seg) if x == other]
            aucs["tier1_vs_%s" % other] = {
                "chrf": auc([chrf[i] for i in t1], [chrf[i] for i in o]),
                "bertscore_f1": auc([bert[i] for i in t1], [bert[i] for i in o]), "n": [len(t1), len(o)]}
        out["models"][m] = {"per_segment": per, "auc": aucs}
        out["info"] = info
    return out


# --------------------------------------------------------------------------
# 14. Token counts (Design 9.4 item 8; D238)
# --------------------------------------------------------------------------

def read_tokens(path, world, checks):
    if not _need_file(path, "token counts (D238)", checks):
        return {}
    rows = read_csv_dicts(path)
    hdr = csv_header(path)
    miss = [c for c in TOKEN_COLUMNS if c not in hdr]
    by = defaultdict(dict)
    bad = 0
    for r in rows:
        try:
            bn, en = int(r["tokens_bn"]), int(r["tokens_en"])
        except ValueError:
            bad += 1
            continue
        if bn <= 0 or en <= 0:
            bad += 1
            continue
        by[r["model_key"]][r["qa_id"]] = math.log(bn / en)
    ids = world.scored_ids["P-CORE-CLOSED"]
    complete = all(all(q in by.get(m, {}) for q in ids) for m in MODELS)
    checks.check(not miss and bad == 0 and complete, "token counts (D238)",
                 "%d rows, models %s, every scored P-CORE-CLOSED item for all three: %s, bad rows %d"
                 % (len(rows), ", ".join(sorted(by)), complete, bad))
    return {m: by[m] for m in MODELS if m in by}


# --------------------------------------------------------------------------
# 15. Figures (Design 13; dataviz palette, light surface)
# --------------------------------------------------------------------------

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CRITICAL = "#d03b3b"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9,
                         "axes.titlecolor": INK, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "legend.frameon": False, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    return plt


def _grid(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    ax.xaxis.grid(False)


def mark_placeholder(fig, placeholder):
    """Every placeholder figure carries a large visible mark (D230)."""
    if not placeholder:
        return
    fig.text(0.5, 0.5, "PLACEHOLDER\nmade-up data, not results", ha="center", va="center",
             rotation=24, fontsize=34, color=CRITICAL, alpha=0.22, fontweight="bold",
             zorder=1000, gid="placeholder-mark")


def _title(fig, text, placeholder):
    fig.suptitle(("[PLACEHOLDER: made-up data, not results]\n" if placeholder else "") + text, fontsize=9,
                 color=INK, x=0.01, ha="left")


def _save(fig, base, placeholder):
    plt = _plt()
    suffix = "_PLACEHOLDER" if placeholder else ""
    paths = []
    for ext in ("png", "pdf"):
        p = Path("%s%s.%s" % (base, suffix, ext))
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200)
        paths.append(p)
    plt.close(fig)
    return paths


def _err(s):
    if s is None or s.get("point") is None or s.get("lo") is None:
        return [[0], [0]]
    return [[max(0.0, s["point"] - s["lo"]) * 100], [max(0.0, s["hi"] - s["point"]) * 100]]


def _figure(plt, ncols, size, **kw):
    """Constrained layout, so titles, legends and labels never overlap."""
    return plt.subplots(1, ncols, figsize=size, layout="constrained", **kw)


def _legend_below(fig, ax_or_none, handles, ncol):
    try:
        fig.legend(handles=handles, loc="outside lower center", ncol=ncol, fontsize=7)
    except (ValueError, TypeError):           # matplotlib older than 3.7
        fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=7,
                   bbox_to_anchor=(0.5, -0.08))


def fig_f2(res_l, base, placeholder):
    plt = _plt()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig, axes = _figure(plt, 3, (7.2, 3.0), sharey=True)
    order = [("L-BB", 0.0), ("L-BB0", 0.7), ("L-BE", 1.8), ("L-EB", 2.9), ("L-EE", 4.0), ("L-EE0", 4.7)]
    b0 = res_l["b0"]["all"]["point"]
    for ax, m in zip(axes, MODELS):
        cells = res_l["models"][m]["cells"]
        for c, x in order:
            s = cells[c]["strict"]
            noimg = c.endswith("0")
            ax.bar(x, (s["point"] or 0) * 100, width=0.6, color=SLOTS[1] if noimg else SLOTS[0],
                   edgecolor=SURFACE, linewidth=1)
            ax.errorbar(x, (s["point"] or 0) * 100, yerr=_err(s), fmt="none", ecolor=INK2,
                        elinewidth=0.8, capsize=2)
            if noimg:
                ax.text(x, (s["hi"] or s["point"] or 0) * 100 + 3,
                        "ref.\n%s%%" % pct(cells[c]["refusal"]["point"], 0),
                        ha="center", va="bottom", fontsize=6, color=INK2)
        if b0 is not None:
            ax.axhline(b0 * 100, color=MUTED, linewidth=1)
        ax.set_xticks([x for _, x in order])
        ax.set_xticklabels([c[2:] for c, _ in order], fontsize=7)
        ax.set_title(MODEL_NAMES[m], fontsize=8)
        ax.set_ylim(0, 110)
        _grid(ax)
    axes[0].set_ylabel("Strict accuracy (%)")
    _legend_below(fig, None, [Patch(color=SLOTS[0], label="With image"),
                              Patch(color=SLOTS[1], label="Without image"),
                              Line2D([0], [0], color=MUTED, lw=1, label="B0 baseline")], 3)
    _title(fig, "F2. EXP-L strict accuracy, 95% CI (question then answer language: B Bangla, E English;"
                " 0 no image); ref. = refusal rate", placeholder)
    mark_placeholder(fig, placeholder)
    return _save(fig, base, placeholder)


def fig_f3(budgets, ht5, base, placeholder):
    plt = _plt()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(5.6, 3.2), layout="constrained")
    xs = np.arange(len(MODELS))
    show_mark = False
    for i, m in enumerate(MODELS):
        b = budgets[m]
        bottom = 0.0
        for j, k in enumerate(BUDGET_CATS):
            v = (b["shares"][k]["point"] or 0) * 100
            ax.bar(i, v, bottom=bottom, width=0.5, color=SLOTS[j], edgecolor=SURFACE, linewidth=1)
            if k == "C" and ht5 is not None and ht5["C_gold_or_visibility"]["point"] is not None:
                h = v * ht5["C_gold_or_visibility"]["point"]
                ax.bar(i, h, bottom=bottom + v - h, width=0.5, color="none", edgecolor=INK,
                       hatch="////", linewidth=0)
                show_mark = True
            bottom += v
        a = (b["A_share"]["point"] or 0) * 100
        g = b["guessing_level"]["accuracy"]["point"]
        if g is not None:
            y = a + g * (100 - a)
            ax.plot([i - 0.32, i + 0.32], [y, y], color=INK, linewidth=1.2)
        ax.text(i, 102, "n=%d" % b["n_denominator"], ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_NAMES[m] for m in MODELS])
    ax.set_ylabel("Share of Bangla naming errors (%)")
    ax.set_ylim(0, 110)
    _grid(ax)
    handles = [Patch(color=SLOTS[j], label=BUDGET_TEXT[k]) for j, k in enumerate(BUDGET_CATS)][::-1]
    if show_mark:
        handles.append(Patch(facecolor="none", edgecolor=INK, hatch="////", linewidth=0,
                             label="C: gold or visibility (HT-5)"))
    handles.append(Line2D([0], [0], color=INK, lw=1.2, label="Guessing level (N-FCB0)"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
    _title(fig, "F3. Error budget, en_unique P-CLUST items not at Tier 1 in N-BB", placeholder)
    mark_placeholder(fig, placeholder)
    return _save(fig, base, placeholder)


def fig_f4(res_n, base, placeholder):
    plt = _plt()
    fig, axes = _figure(plt, 3, (7.2, 2.9), sharey=True)
    for ax, m in zip(axes, MODELS):
        tr = res_n["models"][m]["transfer"]
        for x, g, col in ((0, "yes", SLOTS[0]), (1, "no", SLOTS[1])):
            s = tr[g]
            ax.bar(x, (s["point"] or 0) * 100, width=0.6, color=col, edgecolor=SURFACE, linewidth=1)
            ax.errorbar(x, (s["point"] or 0) * 100, yerr=_err(s), fmt="none", ecolor=INK2, elinewidth=0.8,
                        capsize=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["yes", "no"], fontsize=7)
        ax.set_xlim(-0.6, 3.4)
        ax.set_ylim(0, 110)
        ax.set_title(MODEL_NAMES[m], fontsize=8)
        _grid(ax)
        ins = ax.inset_axes([0.56, 0.52, 0.42, 0.42])
        bo = res_n["models"][m]["transfer_by_origin"]
        for x, o in enumerate(ORIGINS):
            s = bo[o]
            ins.bar(x, (s["point"] or 0) * 100, width=0.6, color=MUTED, edgecolor=SURFACE, linewidth=0.8)
            ins.errorbar(x, (s["point"] or 0) * 100, yerr=_err(s), fmt="none", ecolor=INK2, elinewidth=0.6,
                         capsize=1.5)
        ins.set_xticks([0, 1, 2])
        ins.set_xticklabels(["Eng.", "Other", "Inh."], fontsize=5.5)
        ins.set_ylim(0, 110)
        ins.tick_params(labelsize=5.5)
        ins.set_title("by origin3", fontsize=6, color=INK2)
    axes[0].set_ylabel("Transfer rate (%)")
    fig.supxlabel("English-route Bangla name (en_route)", fontsize=7, color=INK2)
    _title(fig, "F4. Word form: Tier 1 in Bangla among items at Tier 1 in English, 95% CI", placeholder)
    mark_placeholder(fig, placeholder)
    return _save(fig, base, placeholder)


def fig_f5(res_n, human, base, placeholder):
    plt = _plt()
    from matplotlib.patches import Patch
    fig, axes = _figure(plt, 3, (7.2, 3.1), sharey=True)
    rows = [("N-BB", "N-BB"), ("N-BE", "N-BE"), ("N-BBs", "N-BBs")]
    for ax, m in zip(axes, MODELS):
        comp = res_n["models"][m]["composition_en_unique"]
        data = [(lab, comp[c]["shares"]) for c, lab in rows]
        if human is not None:
            data.append(("Human N1", human["shares"]))
        for yi, (lab, shares) in enumerate(reversed(data)):
            left = 0.0
            for j, s in enumerate(SEGMENTS):
                v = (shares.get(s) or 0) * 100
                ax.barh(yi, v, left=left, height=0.6, color=SLOTS[j], edgecolor=SURFACE, linewidth=1)
                left += v
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([lab for lab, _ in reversed(data)], fontsize=7)
        ax.set_xlim(0, 100)
        ax.set_title(MODEL_NAMES[m], fontsize=8)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[1].set_xlabel("Share of items (%)", fontsize=7)
    _legend_below(fig, None, [Patch(color=SLOTS[j], label=SEGMENT_TEXT[s]) for j, s in enumerate(SEGMENTS)], 4)
    _title(fig, "F5. Tier composition, en_unique P-NAME items; human row: N1 on NAME-REF (a reference)",
           placeholder)
    mark_placeholder(fig, placeholder)
    return _save(fig, base, placeholder)


def fig_f6(res_r, base, placeholder):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.8, 3.0), layout="constrained")
    keys = (("BN", "ΔBN\n(R-BB)"), ("EN", "ΔEN\n(R-EE)"), ("NOIMG", "Δno-image\n(R-BB0)"),
            ("FID", "Δfidelity\n(R-BB)"))
    if "deltas" not in res_r:
        ax.text(0.5, 0.5, "No fine-tuned seed in the records", ha="center", va="center", transform=ax.transAxes)
    else:
        for x, (k, lab) in enumerate(keys):
            s = res_r["deltas"][k]
            ax.bar(x, (s["point"] or 0) * 100, width=0.55, color=SLOTS[0], edgecolor=SURFACE, linewidth=1)
            ax.errorbar(x, (s["point"] or 0) * 100, yerr=_err(s), fmt="none", ecolor=INK2, elinewidth=0.8,
                        capsize=2)
            seeds = list(res_r["per_seed"][k].values())
            if seeds:
                ax.plot([x + 0.38] * 2, [min(seeds) * 100, max(seeds) * 100], color=INK, linewidth=1.2,
                        label="Seed range" if x == 0 else None)
                ax.scatter([x + 0.38] * len(seeds), [v * 100 for v in seeds], s=14, color=INK, zorder=5,
                           edgecolor=SURFACE, linewidth=0.8, label="Single seed" if x == 0 else None)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([lab for _, lab in keys], fontsize=7)
        ax.legend(fontsize=7, loc="upper right")
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_ylabel("Fine-tuned minus base (points)")
    _grid(ax)
    _title(fig, "F6. EXP-R, mean of %d seeds: 95%% CI and seed range" % len(res_r.get("seeds", [])),
           placeholder)
    mark_placeholder(fig, placeholder)
    return _save(fig, base, placeholder)


# --------------------------------------------------------------------------
# 16. Tables and the report
# --------------------------------------------------------------------------

def write_table(base, header, rows, placeholder, title):
    """CSV and Markdown. Placeholder tables get a status column on every row
    and a heading line (D230)."""
    suffix = "_PLACEHOLDER" if placeholder else ""
    header = list(header)
    rows = [list(r) for r in rows]
    if placeholder:
        header = ["status"] + header
        rows = [[PLACEHOLDER] + r for r in rows]
    write_csv("%s%s.csv" % (base, suffix), header, rows)
    md = ["# %s%s" % ("[PLACEHOLDER: made-up data, not results] " if placeholder else "", title), "",
          "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rows]
    write_text("%s%s.md" % (base, suffix), "\n".join(md) + "\n")
    return "%s%s.csv" % (base, suffix)


def ci_text(s, scale=100.0, digits=1):
    if s is None or s.get("point") is None:
        return "-"
    f = "%." + str(digits) + "f"
    lo = "-" if s.get("lo") is None else f % (s["lo"] * scale)
    hi = "-" if s.get("hi") is None else f % (s["hi"] * scale)
    return "%s [%s, %s]" % (f % (s["point"] * scale), lo, hi)
