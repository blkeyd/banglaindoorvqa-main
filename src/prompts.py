"""
prompts.py
BanglaIndoorVQA: THE one prompt function, build_prompt() (Design 4.9 and 6).

The inference harness and the fine-tuning scripts both import this file.
Do not write a second copy of build_prompt() anywhere. The prompt texts live in
configs/prompts.yaml and the conditions in configs/conditions.yaml.

Usage:
    from prompts import load_prompts, load_conditions, build_prompt
    prompts = load_prompts("configs/prompts.yaml")
    conds = load_conditions("configs/conditions.yaml", prompts)["conditions"]
    p = build_prompt(conds["L-BB"], item, prompts)
    p["text"]        the user turn (there is no system prompt, Design 5.2)
    p["has_image"]   True: attach the image before the text

Fine-tuning (Design 7.3) formats its training examples with
build_prompt(conds["L-BB"], item, prompts) and adds the gold answer itself.
build_prompt() never reads the gold answer.

Self check:  python src/prompts.py   (every line must print OK)
"""

import hashlib
import json
import re
from pathlib import Path

PLACEHOLDERS = ("question_bn", "question_en", "option_1", "option_2", "option_3", "option_4")
_PH = re.compile(r"\{(" + "|".join(PLACEHOLDERS) + r")\}")
_ANY_BRACE = re.compile(r"\{[^{}\n]*\}")
OPTION_PH = ("option_1", "option_2", "option_3", "option_4")


class PromptError(Exception):
    """A prompt cannot be built, or prompts.yaml / conditions.yaml is not valid."""


def _read_yaml(path):
    import yaml  # PyYAML, in Colab by default
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def templates_sha256(templates):
    """One hash for all template texts. It changes if any character changes."""
    blob = json.dumps(templates, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(blob)


def _check_template_text(tid, variant, text):
    if not isinstance(text, str) or not text.strip():
        raise PromptError(f"prompts.yaml: template {tid}.{variant} is empty.")
    if text != text.strip() or "\r" in text:
        raise PromptError(f"prompts.yaml: template {tid}.{variant} has stray white space "
                          "at an end, or Windows line ends.")
    found = _PH.findall(text)
    other = [m for m in _ANY_BRACE.findall(text) if not _PH.fullmatch(m)]
    if other:
        raise PromptError(f"prompts.yaml: template {tid}.{variant} has an unknown "
                          f"placeholder: {other[0]}")
    questions = [p for p in found if p.startswith("question_")]
    if len(questions) != 1:
        raise PromptError(f"prompts.yaml: template {tid}.{variant} must hold exactly one "
                          "question placeholder.")
    options = [p for p in found if p.startswith("option_")]
    if options and sorted(options) != list(OPTION_PH):
        raise PromptError(f"prompts.yaml: template {tid}.{variant} must hold each of the "
                          "four option placeholders once, or none.")
    return questions[0], bool(options)


def load_prompts(path, check_lock=True):
    """Read and check configs/prompts.yaml. Returns a dict used by build_prompt()."""
    data = _read_yaml(path)
    try:
        version = str(data["prompt_version"])
        templates = data["templates"]
    except (KeyError, TypeError):
        raise PromptError(f"{path}: needs 'prompt_version' and 'templates'.")
    sha = templates_sha256(templates)
    lock = data.get("lock") or {}
    locked = (str(lock.get("prompt_version")) == version and lock.get("sha256") == sha)
    if check_lock and not locked:
        raise PromptError(
            f"{path}: the prompt text and the lock disagree.\n"
            f"  prompt_version {version!r}, sha256 of the text now: {sha}\n"
            f"  lock: prompt_version {lock.get('prompt_version')!r}, sha256 {lock.get('sha256')}\n"
            "  If nobody meant to change a prompt, restore the file from Git.\n"
            "  If a prompt was changed on purpose (allowed only before Gate E): give "
            "prompt_version a new value, copy that value and the sha256 above into the "
            "lock, and write a decision row. One version must mean one text.")
    info = {}
    for tid, variants in templates.items():
        if not isinstance(variants, dict) or "with_image" not in variants:
            raise PromptError(f"{path}: template {tid} needs a 'with_image' text.")
        for variant, text in variants.items():
            if variant not in ("with_image", "no_image"):
                raise PromptError(f"{path}: template {tid} has an unknown variant {variant}.")
            q, has_opt = _check_template_text(tid, variant, text)
            info.setdefault(tid, {"question": q, "options": has_opt})
            if info[tid] != {"question": q, "options": has_opt}:
                raise PromptError(f"{path}: the two variants of {tid} use different placeholders.")
    # Design 6.1: a no-image version removes only the clause that mentions the image.
    clauses = data.get("image_clause") or {}
    for tid, variants in templates.items():
        if "no_image" not in variants:
            continue
        clause = clauses.get(tid)
        if not clause or variants["with_image"].count(clause) != 1:
            raise PromptError(f"{path}: image_clause for {tid} is missing, or does not occur "
                              "exactly once in the with_image text.")
        if variants["with_image"].replace(clause, "", 1) != variants["no_image"]:
            raise PromptError(f"{path}: {tid}.no_image is not {tid}.with_image minus the image "
                              "clause (Design 6.1).")
    # Design 6.2: N-BBs is the N-BB text with one sentence added.
    spec, ans = data.get("specificity_sentence"), data.get("answer_language_sentence_bn")
    if "open_bn_bn_specific" in templates:
        base = templates["open_bn_bn"]["with_image"]
        if not spec or not ans or base.count(ans) != 1 or \
                templates["open_bn_bn_specific"]["with_image"] != base.replace(ans, ans + " " + spec, 1):
            raise PromptError(f"{path}: open_bn_bn_specific is not open_bn_bn plus the "
                              "specificity sentence after the answer-language sentence.")
    return {"prompt_version": version, "templates": templates, "sha256": sha,
            "locked": locked, "info": info, "path": str(path)}


def load_conditions(path, prompts):
    """Read and check configs/conditions.yaml against the loaded prompts."""
    data = _read_yaml(path)
    conds = {}
    for cid, raw in (data.get("conditions") or {}).items():
        c = dict(raw)
        c["condition_id"] = cid
        for key in ("experiment", "template", "image", "question_lang", "answer_lang", "probe"):
            if key not in c:
                raise PromptError(f"{path}: condition {cid} lacks '{key}'.")
        tid = c["template"]
        if tid not in prompts["templates"]:
            raise PromptError(f"{path}: condition {cid} names an unknown template {tid}.")
        if not isinstance(c["image"], bool):
            raise PromptError(f"{path}: condition {cid}: image must be true or false.")
        if not c["image"] and "no_image" not in prompts["templates"][tid]:
            raise PromptError(f"{path}: condition {cid} has no image, but template {tid} has "
                              "no no_image text.")
        if c["question_lang"] not in ("bn", "en") or c["answer_lang"] not in ("bn", "en"):
            raise PromptError(f"{path}: condition {cid}: languages must be bn or en.")
        if prompts["info"][tid]["question"] != "question_" + c["question_lang"]:
            raise PromptError(f"{path}: condition {cid}: question language "
                              f"{c['question_lang']} does not fit template {tid}.")
        c["forced_choice"] = bool(c.get("options_lang"))
        if c["forced_choice"] != prompts["info"][tid]["options"]:
            raise PromptError(f"{path}: condition {cid}: options_lang and template {tid} "
                              "do not agree on forced choice.")
        if c["forced_choice"] and c["options_lang"] not in ("bn", "en"):
            raise PromptError(f"{path}: condition {cid}: options_lang must be bn or en.")
        conds[cid] = c
    if not conds:
        raise PromptError(f"{path}: no conditions.")
    for cid, c in conds.items():
        src = c.get("reuse_from")
        if src and (src not in conds or conds[src]["template"] != c["template"]
                    or conds[src]["image"] != c["image"]):
            raise PromptError(f"{path}: condition {cid}: reuse_from must name a condition "
                              "with the same template and image setting.")
    sets = data.get("sets") or {}
    for name, members in sets.items():
        for cid in members:
            if cid not in conds:
                raise PromptError(f"{path}: set {name} names an unknown condition {cid}.")
    return {"conditions": conds, "sets": sets, "probes": data.get("probes") or {},
            "stages": data.get("stages") or {}, "path": str(path)}


def build_prompt(condition, item, prompts, question_en=None, options=None):
    """Return the prompt of one item under one condition.

    condition    one entry of load_conditions(...)["conditions"]
    item         one probe record; only its question_bn is read
    prompts      the dict from load_prompts()
    question_en  the English question; needed when the question language is English
    options      the four option texts in the order shown; forced choice only

    Returns a dict: text, has_image, template_id, variant, prompt_version, sha256.
    """
    tid = condition["template"]
    variant = "with_image" if condition["image"] else "no_image"
    template = prompts["templates"][tid][variant]
    values = {}
    if condition["question_lang"] == "bn":
        q = str(item.get("question_bn") or "").strip()
        if not q:
            raise PromptError(f"{item.get('qa_id')}: empty Bangla question.")
        values["question_bn"] = q
    else:
        q = str(question_en or "").strip()
        if not q:
            raise PromptError(f"{item.get('qa_id')}: no English question for condition "
                              f"{condition['condition_id']}.")
        values["question_en"] = q
    if condition["forced_choice"]:
        if not isinstance(options, (list, tuple)) or len(options) != 4 or \
                any(not str(o or "").strip() for o in options):
            raise PromptError(f"{item.get('qa_id')}: condition {condition['condition_id']} "
                              "needs exactly four option texts.")
        for name, opt in zip(OPTION_PH, options):
            values[name] = str(opt).strip()
    elif options is not None:
        raise PromptError(f"{item.get('qa_id')}: options given for an open condition.")
    for name, val in values.items():
        if "\n" in val or "\r" in val:
            raise PromptError(f"{item.get('qa_id')}: a line break inside {name}.")
    # One pass: text that was put in is never scanned again for placeholders.
    text = _PH.sub(lambda m: values[m.group(1)], template)
    return {"text": text, "has_image": bool(condition["image"]), "template_id": tid,
            "variant": variant, "prompt_version": prompts["prompt_version"],
            "sha256": sha256_text(text)}


if __name__ == "__main__":
    # Self check on the real config files. Every line must print OK.
    root = Path(__file__).resolve().parent.parent
    checks = []
    P = load_prompts(root / "configs" / "prompts.yaml")
    C = load_conditions(root / "configs" / "conditions.yaml", P)["conditions"]
    checks.append(("prompts.yaml loads and its lock holds", P["locked"]))
    checks.append(("16 conditions load", len(C) == 16))
    item = {"qa_id": "T_0001_q1", "question_bn": "  ক খ গ?  ", "answer_bn": "GOLD"}
    p = build_prompt(C["L-BB"], item, P)
    checks.append(("L-BB holds the question, stripped", "প্রশ্ন: ক খ গ?\n" in p["text"]))
    checks.append(("L-BB ends with the answer cue", p["text"].endswith("উত্তর:")))
    checks.append(("the gold answer is never in a prompt", "GOLD" not in p["text"]))
    checks.append(("L-BB and N-BB and R-BB are one text",
                   p["text"] == build_prompt(C["N-BB"], item, P)["text"]
                   == build_prompt(C["R-BB"], item, P)["text"]))
    p0 = build_prompt(C["L-BB0"], item, P)
    checks.append(("L-BB0 has no image and a shorter text",
                   not p0["has_image"] and len(p0["text"]) < len(p["text"])))
    ps = build_prompt(C["N-BBs"], item, P)
    checks.append(("N-BBs is longer than N-BB", len(ps["text"]) > len(p["text"])))
    e = build_prompt(C["L-EE"], item, P, question_en="What is this?")
    checks.append(("L-EE holds the English question", "Question: What is this?\n" in e["text"]))
    try:
        build_prompt(C["L-EE"], item, P)
        ok = False
    except PromptError:
        ok = True
    checks.append(("an English condition without an English question is refused", ok))
    f = build_prompt(C["N-FCB"], item, P, options=["w", "x", "y", "z"])
    checks.append(("N-FCB lists four options in order",
                   "ক) w\nখ) x\nগ) y\nঘ) z\n" in f["text"]))
    try:
        build_prompt(C["N-FCB"], item, P, options=["w", "x", "y"])
        ok = False
    except PromptError:
        ok = True
    checks.append(("forced choice with three options is refused", ok))
    tricky = dict(item, question_bn="{option_1} {question_en}?")
    t = build_prompt(C["L-BB"], tricky, P)
    checks.append(("placeholder-like text in a question is kept as written",
                   "{option_1} {question_en}?" in t["text"]))
    checks.append(("no placeholder is left in a built prompt",
                   not _PH.search(p["text"]) and not _PH.search(f["text"])))
    bad = 0
    for name, passed in checks:
        print(("OK   " if passed else "FAIL ") + name)
        bad += 0 if passed else 1
    print()
    print(f"{len(checks) - bad}/{len(checks)} self checks passed")
