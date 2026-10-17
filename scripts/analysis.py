#!/usr/bin/env python3
"""
scripts/analysis.py  (Analysis notebooks, before Gate E)
BanglaIndoorVQA main paper: the analysis of Design 9.1 to 9.4 and the figures
and tables of Design 13, run end to end on placeholder data now and on the real
scored files later with no change other than input paths (Design 9.7).

The functions live in src/stats.py. This script reads the inputs, runs the
analyses and writes the figures, tables and report.

Commands
    selftest      checks on invented data; needs no Drive.
    check-inputs  Cell 0: one PASS or FAIL line per input.
    templates     the template_id fold (D236): counts, and the merged groups
                  (question text) written to Drive for one look before Gate E.
    tokens        question token counts per item and model tokenizer (Design
                  9.4 item 8): Qwen and Gemma tokenizers at the pinned commits,
                  Gemini countTokens (not billed). Real data, run once.
    placeholder   writes made-up inputs in the exact real formats, from the real
                  probe ids, houses and fields, to a folder named placeholder.
    budget-items  the HT-5 sampling frame: error-budget category per model and
                  item (ids only), from the scored files alone.
    run           every analysis, figure and table; --mode placeholder or real.

Rules kept here
    - Seed 42; 10,000 bootstrap resamples (the real run refuses other values).
    - Printouts hold ids and counts only, never a question, gold or answer.
    - Placeholder output goes only to a folder named placeholder; nothing is
      written to Drive scored/, Drive outputs/ or the repo (D230).
    - Real mode needs every record at stage main, the gate-F scorer, the gate-A
      lexicon and this code equal to the tag gate-E, or a written --deviation.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

SCRIPT_VERSION = "analysis-2026-10-17"
REPO_DEFAULT = Path(__file__).resolve().parents[1]
DRIVE_DEFAULT = Path("/content/drive/MyDrive/BanglaIndoorVQA")
sys.path.insert(0, str(REPO_DEFAULT / "src"))

import numpy as np  # noqa: E402
import stats as S  # noqa: E402

TOKENIZERS = {"qwen": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
              "gemma": ("google/gemma-4-E4B-it", "ee0ef6023621cff504d758262d4e04895a5af4a2")}
GEMINI_MODEL = "gemini-3.6-flash"
TRANSFORMERS_VERSION = "5.18.0"       # D153
# D235: the template_id fit counts as more conservative when a language term's
# robust SE grows by 10% or more, or a CI that excluded 0 no longer does.
TEMPLATE_SE_RATIO = 1.10


EXP_R_TEXT = {"BN": "dBN (R-BB strict)", "EN": "dEN (R-EE strict)", "NOIMG": "d no-image (R-BB0 strict)",
              "FID": "d fidelity (R-BB)", "LEN": "d answer length (R-BB)", "TIER1": "d Tier-1 rate (R-BB naming)",
              "FALLBACK": "d generic fallback (R-BB naming)"}


def say(msg=""):
    print(msg, flush=True)


def paths_from(args):
    return S.Paths.make(args.repo, args.drive, dataset=args.dataset, houses=args.houses,
                        probes_dir=args.probes_dir, manifest11=args.manifest11, swaps=args.swaps,
                        english=args.english)


# --------------------------------------------------------------------------
# 1. Placeholder inputs (made-up answers and labels in the exact formats)
# --------------------------------------------------------------------------

PH_MODEL_IDS = {"gemini": "gemini-3.6-flash", "qwen": "Qwen/Qwen3.5-4B",
                "gemma": "google/gemma-4-E4B-it", "qwen_ft_s42": "Qwen/Qwen3.5-4B+ft-s42",
                "qwen_ft_s43": "Qwen/Qwen3.5-4B+ft-s43", "qwen_ft_s44": "Qwen/Qwen3.5-4B+ft-s44"}
PH_SKILL = {"gemini": 0.80, "qwen": 0.62, "gemma": 0.55, "qwen_ft_s42": 0.62,
            "qwen_ft_s43": 0.62, "qwen_ft_s44": 0.62, "human_N1": 0.80, "human_N2": 0.86}
PH_FT_GAIN = {"qwen_ft_s42": 0.10, "qwen_ft_s43": 0.08, "qwen_ft_s44": 0.12}
PH_COND = {"L-BB": 0.72, "L-BE": 0.86, "L-EB": 0.82, "L-EE": 1.0, "L-BB0": 0.35, "L-EE0": 0.45,
           "N-BB": 0.68, "N-BE": 0.95, "N-BBs": 0.76, "R-BB": 0.72, "R-EE": 1.0, "R-BB0": 0.35}
PH_REFUSE = {"L-BB0": 0.35, "L-EE0": 0.25, "R-BB0": 0.35}
PH_REFUSALS = {"bn": ("জানি না", "বলা সম্ভব নয়", "ছবি নেই"),
               "en": ("cannot tell", "not visible", "sorry unknown")}
PH_OPTIONS = ["PH option one", "PH option two", "PH option three", "PH option four"]
PH_LABELS = {"bn": ("ক", "খ", "গ", "ঘ"), "en": ("A", "B", "C", "D")}
# Invented words for the placeholder sensitivity rows; the code checks that no
# lexicon or vocabulary term matches them.
PH_FORMS = {"t1_bn": "ঝিমকালি", "t2_en": "zorbex", "mat_bn": "ঘুমরিক", "mat_bn2": "ফেলুনি"}

_EN2BN = (("sh", "শ"), ("ch", "চ"), ("th", "থ"), ("ph", "ফ"), ("ck", "ক"), ("ee", "ী"),
          ("oo", "ু"), ("b", "ব"), ("c", "ক"), ("d", "ড"), ("f", "ফ"), ("g", "গ"), ("h", "হ"),
          ("j", "জ"), ("k", "ক"), ("l", "ল"), ("m", "ম"), ("n", "ন"), ("p", "প"), ("q", "ক"),
          ("r", "র"), ("s", "স"), ("t", "ট"), ("v", "ভ"), ("w", "ও"), ("x", "ক্স"), ("z", "জ"))
_VOW_SIGN = {"a": "া", "e": "ে", "i": "ি", "o": "ো", "u": "া", "y": "ি"}
_VOW_IND = {"a": "আ", "e": "এ", "i": "ই", "o": "ও", "u": "আ", "y": "ই"}


def ph_rng(*parts):
    h = hashlib.sha256(("42|PH|" + "|".join(parts)).encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


def transliterate(en):
    """A crude English-to-Bangla-letters spelling, used only to make a few
    placeholder answers that the scorer labels English copy (checked)."""
    out, i, prev_cons = [], 0, False
    w = en.lower()
    while i < len(w):
        ch = w[i]
        if ch == " ":
            out.append(" ")
            prev_cons = False
            i += 1
            continue
        hit = next(((a, b) for a, b in _EN2BN if w.startswith(a, i)), None)
        if hit and hit[0] in ("ee", "oo"):
            out.append(hit[1] if prev_cons else ("ই" if hit[0] == "ee" else "উ"))
            prev_cons = False
            i += 2
            continue
        if ch in _VOW_SIGN:
            out.append(_VOW_SIGN[ch] if prev_cons else _VOW_IND[ch])
            prev_cons = False
            i += 1
            continue
        if hit:
            out.append(hit[1])
            prev_cons = True
            i += len(hit[0])
            continue
        i += 1
    return "".join(out)


def ph_answer(world, rng, who, cid, qa_id, lang, gain=0.0):
    """One made-up answer. who is a model key or human_N1 / human_N2."""
    it = world.items[qa_id]
    lex, voc = world.lex, world.voc
    noimg = cid in S.NO_IMAGE
    p = PH_SKILL[who] * PH_COND.get(cid, 0.8) + gain
    if who.startswith("human"):
        p = PH_SKILL[who]
    if lang == "bn" and it.get("en_route") is False:
        p *= 0.82
    refuse = PH_REFUSE.get(cid, 0.04)
    u = rng.random()
    if u < refuse * (1.0 if noimg else 0.5):
        return rng.choice(PH_REFUSALS[lang])
    if rng.random() < 0.02:
        return ""
    other = "en" if lang == "bn" else "bn"
    kind = it["gold_kind"]
    correct = rng.random() < p
    lead = "এটি একটি " if (lang == "bn" and rng.random() < 0.08) else ""
    if kind == "naming":
        obj = lex.objects[it["name_id"]]
        names = {"bn": obj.t1_bn, "en": obj.t1_en or obj.t1_bn}
        if correct:
            if rng.random() < 0.10 and obj.t1_en and lang == "bn":
                return rng.choice(obj.t1_en)                      # wrong language
            if rng.random() < 0.05 and obj.t1_en and lang == "bn":
                return "%s (%s)" % (rng.choice(obj.t1_bn), rng.choice(obj.t1_en))    # mixed script
            return lead + rng.choice(names[lang])
        options = ["other", "other", "ambig"]
        if (obj.t2_bn if lang == "bn" else obj.t2_en):
            options += ["generic"] * 3
        if lex.siblings_of(obj.name_id):
            options += ["sibling"] * 3
        if lang == "bn" and obj.t1_en:
            options += ["copy"]
        pick = rng.choice(options)
        if pick == "generic":
            return lead + rng.choice(obj.t2_bn if lang == "bn" else obj.t2_en)
        if pick == "sibling":
            sib = lex.objects[rng.choice(sorted(lex.siblings_of(obj.name_id)))]
            return rng.choice(sib.t1_bn if lang == "bn" else (sib.t1_en or sib.t1_bn))
        if pick == "copy":
            return transliterate(obj.t1_en[0])
        if pick == "ambig":
            oid = rng.choice(sorted(set(lex.objects) - {obj.name_id}))
            o = lex.objects[oid]
            return "%s %s %s" % (rng.choice(names[lang]), "অথবা" if lang == "bn" else "or",
                                 rng.choice(o.t1_bn if lang == "bn" else (o.t1_en or o.t1_bn)))
        oid = rng.choice(sorted(lex.objects))
        o = lex.objects[oid]
        return rng.choice(o.t1_bn if lang == "bn" else (o.t1_en or o.t1_bn))
    if kind == "count":
        gold = world.sc.resolve_gold(world.scorer_item(qa_id), lex, voc).number
        n = gold if correct else max(0, gold + rng.choice([-1, 1, 2]))
        forms = voc.forms["number"].get(str(n), {}).get(lang) or [str(n)]
        if rng.random() < 0.2:
            return str(n)
        if correct and lang == "bn" and rng.random() < 0.05:
            return rng.choice(forms) + " " + PH_REFUSALS["bn"][0]       # content and a refusal phrase
        if not correct and rng.random() < 0.15:
            forms = voc.forms["number"].get(str(n), {}).get(other) or forms
        return rng.choice(forms)
    if kind in ("room", "material", "colour"):
        gold = world.sc.resolve_gold(world.scorer_item(qa_id), lex, voc)
        classes = sorted(voc.forms[kind])
        cls = sorted(gold.classes)[0] if correct else rng.choice([c for c in classes if c not in gold.classes] or classes)
        lg = lang if rng.random() > 0.08 else other
        forms = voc.forms[kind][cls].get(lg) or voc.forms[kind][cls].get(lang) or [cls]
        return rng.choice(forms)
    return rng.choice(PH_REFUSALS[lang])


def ph_fc_answer(rng, who, cid, lang, correct_index):
    noimg = cid in S.NO_IMAGE
    p = {"gemini": 0.82, "qwen": 0.60, "gemma": 0.70}[who] * (0.45 if noimg else 1.0)
    u = rng.random()
    labels = PH_LABELS[lang]
    if u < (0.30 if noimg else 0.05):
        return rng.choice(PH_REFUSALS[lang])
    if u < (0.36 if noimg else 0.08):
        a, b = [x for x in labels if x != labels[correct_index]][:2]
        return ("%s অথবা %s" % (a, b)) if lang == "bn" else ("%s or %s" % (a, b))
    if rng.random() < p:
        return labels[correct_index]
    return rng.choice([x for i, x in enumerate(labels) if i != correct_index])


def ph_correct_index(qa_id):
    return int(hashlib.sha256(("42|PH|ci|" + qa_id).encode()).hexdigest()[:8], 16) % 4


def _ph_material_items(world, cls):
    out = []
    for q in world.scored_ids["P-CORE-CLOSED"]:
        if world.items[q]["gold_kind"] == "material":
            g = world.sc.resolve_gold(world.scorer_item(q), world.lex, world.voc)
            if cls in g.classes:
                out.append(q)
    return out


def ph_plant(world, rows_sens):
    """Answers that make the placeholder sensitivity rows change a label."""
    plant = {}
    t1 = next(r for r in rows_sens if r["row_id"] == "PH1")
    for q in world.scored_ids["P-NAME"]:
        if world.items[q]["name_id"] == t1["target_id"]:
            plant[("N-BB", q)] = PH_FORMS["t1_bn"]
            plant[("N-BE", q)] = PH_FORMS["t2_en"]
    mat = next(r for r in rows_sens if r["row_id"] == "PH3")["target_id"]
    items = _ph_material_items(world, mat)
    for q in items[:3]:
        plant[("L-BB", q)] = PH_FORMS["mat_bn"]          # accepted only with PH3
    for q in items[3:5]:
        plant[("L-EB", q)] = PH_FORMS["mat_bn2"]         # added by PH4, removed again by PH5
    item = next(r for r in rows_sens if r["op"] == "item_any_class")["target_id"]
    gold = world.sc.resolve_gold(world.scorer_item(item), world.lex, world.voc)
    other = [c for c in sorted(world.voc.forms["colour"]) if c not in gold.classes][0]
    text = "%s %s" % (world.voc.forms["colour"][sorted(gold.classes)[0]]["en"][0],
                      world.voc.forms["colour"][other]["en"][0])
    plant[("L-EE", item)] = text
    plant[("R-EE", item)] = text
    return plant


def ph_sensitivity_rows(world):
    """Placeholder sensitivity rows: invented forms only (PH_FORMS), on targets
    chosen by code. Real rows are written by people in the HT-5 chat."""
    lex, voc = world.lex, world.voc
    for f in PH_FORMS.values():
        toks = world.sc.tokenise(world.normalise(f))
        if lex.index.find_all(toks) or any(voc.indexes[k].find_all(toks) for k in voc.indexes):
            raise S.StatsError("placeholder form %r matches a real term; change PH_FORMS" % f)
    counts = Counter(world.items[q]["name_id"] for q in world.scored_ids["P-NAME"])
    top = sorted(counts, key=lambda k: (-counts[k], k))[0]
    mats = Counter(c for q in world.scored_ids["P-CORE-CLOSED"] if world.items[q]["gold_kind"] == "material"
                   for c in world.sc.resolve_gold(world.scorer_item(q), lex, voc).classes)
    mat = sorted(mats, key=lambda c: (-mats[c], c))[0]
    colour = [q for q in world.scored_ids["P-CORE-CLOSED"] if world.items[q]["gold_kind"] == "colour"]
    colour_item = ([q for q in colour if q in set(world.scored_ids["R-TEST"])] or colour)[0]
    note = "PLACEHOLDER test row, invented form"
    return [
        {"row_id": "PH1", "source": "HT-5", "op": "add_term", "target": "lexicon", "target_id": top,
         "tier": "1", "lang": "bn", "form": PH_FORMS["t1_bn"], "note": note},
        {"row_id": "PH2", "source": "HT-5", "op": "add_term", "target": "lexicon", "target_id": top,
         "tier": "2", "lang": "en", "form": PH_FORMS["t2_en"], "note": note},
        {"row_id": "PH3", "source": "D192", "op": "add_term", "target": "material",
         "target_id": mat, "tier": "", "lang": "bn", "form": PH_FORMS["mat_bn"], "note": note},
        {"row_id": "PH4", "source": "D192", "op": "add_term", "target": "material",
         "target_id": mat, "tier": "", "lang": "bn", "form": PH_FORMS["mat_bn2"], "note": note},
        {"row_id": "PH5", "source": "D192", "op": "remove_term", "target": "material",
         "target_id": mat, "tier": "", "lang": "bn", "form": PH_FORMS["mat_bn2"], "note": note},
        {"row_id": "PH6", "source": "D192", "op": "item_any_class", "target": "", "target_id": colour_item,
         "tier": "", "lang": "", "form": "", "note": note}]


def ph_record(world, rec, model, cid, qa_id, utc):
    rec = dict(rec)
    rec.update({"key": hashlib.sha256(("%s|%s|exp-v1|%s" % (PH_MODEL_IDS.get(model, model), cid, qa_id))
                                      .encode()).hexdigest()[:24],
                "model_id": PH_MODEL_IDS.get(model, model), "model_key": model,
                "model_version" if model == "gemini" else "model_revision": "placeholder",
                "prompt_version": "exp-v1", "stage": S.STAGE_PLACEHOLDER, "run_id": "placeholder",
                "house_id": world.items[qa_id]["house_id"], "image_sha256": "placeholder",
                "prompt_sha256": "placeholder", "settings_sha256": "placeholder",
                "mode": "placeholder", "ts_utc": utc,
                "fc_options_sha256": "placeholder" if cid in S.COND_N_FC else None})
    return rec


def cmd_placeholder(args):
    t0 = time.time()
    checks = S.Checks(soften=args.allow_other_inputs)
    paths = paths_from(args)
    world = S.build_world(paths, checks, need_dataset=False, soften=args.allow_other_inputs)
    if checks.count("FAIL"):
        say("placeholder: %d FAIL in the inputs; nothing written" % checks.count("FAIL"))
        return 1
    out = S.guard_out(args.out, "placeholder", paths.repo, paths.drive)
    sc, lex, voc = world.sc, world.lex, world.voc
    utc = S.now_utc()
    rows_sens = ph_sensitivity_rows(world)
    plant = ph_plant(world, rows_sens)
    files = []
    plan = {m: S.ZS_CONDS for m in S.MODELS}
    plan[S.FT_BASE] = S.ZS_CONDS + S.COND_R
    for s in S.FT_SEEDS:
        plan[s] = S.COND_R
    n_total = 0
    reuse = {}
    for model in list(S.MODELS) + list(S.FT_SEEDS):
        recs = []
        for cid in plan[model]:
            cond = world.conditions[cid]
            lang = cond["answer_lang"]
            for q in world.run_ids[world.probe_of(cid)]:
                excl = world.excluded.get(q)
                rng = ph_rng(model, cid, q)
                if cid in S.COND_N_FC:
                    ci = ph_correct_index(q)
                    ans = ph_fc_answer(rng, model, cid, cond["question_lang"], ci)
                    rec = sc.score_item(world.scorer_item(q), cid, ans, lex, voc, options=PH_OPTIONS,
                                        correct_index=ci, excluded=excl)
                else:
                    gain = PH_FT_GAIN.get(model, 0.0) * (1.0 if cid != "R-EE" else 0.25)
                    if model == S.FT_BASE and cid in S.COND_R and (cid.replace("R-", "L-")) in reuse \
                            and q in reuse[cid.replace("R-", "L-")]:
                        ans = reuse[cid.replace("R-", "L-")][q]       # reuse_from, as the harness
                    elif excl:
                        ans = "PH excluded item"     # never scored (D110); its gold may not resolve
                    elif (cid, q) in plant and model in S.MODELS + (S.FT_BASE,):
                        ans = plant[(cid, q)]
                    else:
                        ans = ph_answer(world, rng, model, cid, q, lang, gain)
                    if model == S.FT_BASE and cid in ("L-BB", "L-EE", "L-BB0"):
                        reuse.setdefault(cid, {})[q] = ans
                    rec = sc.score_item(world.scorer_item(q), cid, ans, lex, voc, excluded=excl)
                    if not excl and world.items[q]["gold_kind"] == "naming" and rec["label"] != "english_copy" \
                            and lang == "bn" and ans and ans == transliterate(
                                (lex.objects[world.items[q]["name_id"]].t1_en or [""])[0]):
                        ans = ph_answer(world, ph_rng(model, cid, q, "retry"), model, cid, q, lang, gain)
                        rec = sc.score_item(world.scorer_item(q), cid, ans, lex, voc, excluded=excl)
                recs.append(ph_record(world, rec, model, cid, q, utc))
        fp = out / "scored" / ("ph_%s_scored.jsonl" % model)
        fp.parent.mkdir(parents=True, exist_ok=True)
        with open(fp, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        S.write_json(str(fp)[:-len(".jsonl")] + "_manifest.json",
                     {"command": "placeholder", "script_version": SCRIPT_VERSION,
                      "scorer_version": sc.SCORER_VERSION, "run_utc": utc, "records": len(recs),
                      "inputs": {"placeholder": True}})
        files.append(fp)
        n_total += len(recs)
        say("  wrote %s: %d records" % (fp.name, len(recs)))

    # Error-budget C items, for the HT-5 placeholder sample
    q_checks = S.Checks(echo=False)
    recs = S.load_scored([str(f) for f in files], world, q_checks, "placeholder")
    if q_checks.count("FAIL"):
        say(q_checks.text())
        raise S.StatsError("the placeholder records fail their own checks")
    boot_n = S.HouseBoot(world.houses_of(world.scored_ids["P-NAME"]), n_boot=200, name="EXP-N")
    budgets = {m: S.error_budget(world, recs, m, boot_n) for m in S.MODELS}

    # HT-3 in the exact asset12_nameref.py format, made with its own functions
    a12 = load_asset12(paths.repo)
    nameref = world.run_ids["NAME-REF"]
    key_rows = []
    for role in a12.ROLES:
        for code, q in a12.assign_codes(nameref, role):
            it = world.items[q]
            cell = ("PY" if "P-CLUST" in it["probes"] else "PN") + "-" + ("EY" if it["en_route"] else "EN")
            key_rows.append({"role": role, "code": code, "qa_id": q, "house_id": it["house_id"],
                             "cell": cell, "in_pclust": "yes" if "P-CLUST" in it["probes"] else "no",
                             "en_route": "yes" if it["en_route"] else "no", "drawn_as": "regular"})
    names = defaultdict(dict)
    for r in key_rows:
        rng = ph_rng("human", r["role"], r["qa_id"])
        names[r["role"]][r["code"]] = "PH excluded item" if r["qa_id"] in world.excluded else \
            ph_answer(world, rng, "human_" + r["role"], "N-BB", r["qa_id"], "bn")
    inp = SimpleNamespace(sc=sc, normalise=world.normalise, excluded=set(world.excluded),
                          pname_by_id={q: world.scorer_item(q) for q in nameref}, lex=lex, voc=voc)
    scored = a12.score_rows(inp, key_rows, names)
    summary = a12.summarise_scores(scored, n_boot=args.boot)
    ht3_dir = out / "human" / "HT-3" / ("score_placeholder_%s" % S.utc_stamp())
    ht3_dir.mkdir(parents=True, exist_ok=True)
    a12.write_csv(ht3_dir / "ht3_scored_names.csv", a12.SCORED_HEADER,
                  [[s[k] for k in a12.SCORED_HEADER] for s in scored])
    meta = {"utc": utc, "mode": "placeholder", "reportable": False, "scorer_version": sc.SCORER_VERSION,
            "scorer_sha256": world.hashes["scoring.py"], "lexicon_sha256": world.hashes["lexicon.csv"],
            "vocab_sha256": world.hashes["closed_vocab.csv"],
            "excluded_sha256": world.hashes["excluded_items.csv"], "excluded_rows": len(world.excluded),
            "condition": "N-BB", "script_version": "placeholder via %s" % a12.SCRIPT_VERSION,
            "lexicon_writer": "N2", "primary_role": a12.roles_note("N2")[0]}
    js = dict(meta)
    js["summary"] = {k: v for k, v in summary.items() if k != "same_other_object"}
    js["summary"]["same_other_object_n"] = len(summary["same_other_object"])
    S.write_text(ht3_dir / "ht3_agreement.json", json.dumps(js, ensure_ascii=False, indent=1, default=str))
    S.write_text(ht3_dir / "README_PLACEHOLDER.txt", "PLACEHOLDER: made-up names scored with the real scorer;"
                 " never results.\n")

    # HT-4: 200 answers, at least 10 per scorer label where available
    pool = [(m, c, q, r) for (m, c), slot in recs.data.items() if m in S.MODELS
            for q, r in slot.items() if r["gold_status"] != "excluded"]
    rng = random.Random(42)
    rng.shuffle(pool)
    pick, seen = [], Counter()
    for m, c, q, r in pool:
        if seen[r["label"]] < 10:
            pick.append((m, c, q, r))
            seen[r["label"]] += 1
    for item in pool:
        if len(pick) >= 200:
            break
        if item not in pick:
            pick.append(item)
    rows4 = []
    for i, (m, c, q, r) in enumerate(pick[:200], 1):
        lab = r["label"]
        alt = [x for x in (S.LABELS_NAMING if r["task"] == "open" and r["kind"] == "naming" else
                           (("correct", "wrong", "invalid") if r["task"] == "fc" else ("correct", "wrong")))
               if x != lab]
        c1 = lab if rng.random() > 0.06 else rng.choice(alt)
        c2 = lab if rng.random() > 0.08 else rng.choice(alt)
        cons = lab if rng.random() > 0.03 else rng.choice(alt)
        rows4.append(["H4-%03d" % i, m, c, q, world.items[q]["house_id"], lab, c1, c2, cons])
    S.write_csv(out / "human" / "HT-4" / "ht4_labels.csv", S.HT4_COLUMNS, rows4)

    # HT-5: 140 C answers (pooled over models), 30 English copies, 30 siblings
    c_pool = sorted((m, q) for m in S.MODELS for q, cat in budgets[m]["categories"].items() if cat == "C")
    ec = sorted((m, c, q) for (m, c), slot in recs.data.items() if m in S.MODELS and c in S.COND_N_OPEN
                for q, r in slot.items() if r["label"] == "english_copy")
    sb = sorted((m, c, q) for (m, c), slot in recs.data.items() if m in S.MODELS and c in S.COND_N_OPEN
                for q, r in slot.items() if r["label"] == "sibling")
    rng = random.Random(4242)
    rows5, k = [], 0
    for (m, q) in rng.sample(c_pool, min(140, len(c_pool))):
        k += 1
        rows5.append(["H5-%03d" % k, m, "N-BB", q, world.items[q]["house_id"], "C",
                      rng.choices(S.HT5_VERDICTS, weights=(2, 2, 2, 1, 8))[0]])
    for stratum, src in (("english_copy", ec), ("sibling", sb)):
        for (m, c, q) in rng.sample(src, min(30, len(src))):
            k += 1
            rows5.append(["H5-%03d" % k, m, c, q, world.items[q]["house_id"], stratum,
                          rng.choices(S.HT5_VERDICTS, weights=(1, 3, 1, 1, 6))[0]])
    S.write_csv(out / "human" / "HT-5" / "ht5_audit.csv", S.HT5_COLUMNS, rows5)
    S.write_csv(out / "human" / "HT-5" / "lexicon_sensitivity.csv", S.SENS_COLUMNS,
                [[r[c] for c in S.SENS_COLUMNS] for r in rows_sens])

    # Token counts: made-up numbers from the string length
    rows_t = []
    for q in world.run_ids["P-CORE-CLOSED"]:
        L = len(world.items[q]["question_bn"])
        for m, f in (("gemini", 2.6), ("qwen", 2.1), ("gemma", 2.3)):
            r = ph_rng("tok", m, q)
            rows_t.append([q, m, max(2, int(L / f) + r.randint(0, 4)), max(2, int(L / 5.0) + r.randint(0, 2)),
                           "placeholder", utc])
    S.write_csv(out / "stats_inputs" / "token_counts.csv", S.TOKEN_COLUMNS, rows_t)
    S.write_json(out / "placeholder_manifest.json",
                 {"script_version": SCRIPT_VERSION, "stats_version": S.STATS_VERSION, "utc": utc,
                  "records": n_total, "files": [str(f) for f in files], "ht3_dir": str(ht3_dir),
                  "ht4_rows": len(rows4), "ht5_rows": len(rows5), "sensitivity_rows": len(rows_sens),
                  "token_rows": len(rows_t), "inputs": world.hashes,
                  "note": "PLACEHOLDER: made-up answers and labels; never results"})
    say("placeholder: %d records in %d files, HT-3 %s, HT-4 %d rows, HT-5 %d rows, sensitivity %d rows,"
        " token rows %d (%.0f s)" % (n_total, len(files), ht3_dir.name, len(rows4), len(rows5),
                                     len(rows_sens), len(rows_t), time.time() - t0))
    say("Wrote %s" % out)
    return 0


def load_asset12(repo):
    path = Path(repo) / "scripts" / "asset12_nameref.py"
    spec = importlib.util.spec_from_file_location("asset12_nameref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# 2. Tables (T1, T2, T4, T5, S1 to S10) and the report
# --------------------------------------------------------------------------

def table_t1(world, out, ph):
    rows = []
    dc = world.dataset_counts or {}
    rows.append(["dataset (QA pairs as items)", dc.get("images"), dc.get("houses"), dc.get("qa"), "", "", "",
                 "bounding boxes %s" % dc.get("objects", "")])
    for p in ("P-CORE-CLOSED", "R-TEST", "P-NAME", "P-CLUST", "P-FCE", "NAME-REF"):
        run, sco = world.run_ids[p], world.scored_ids[p]
        images = len({world.items[q]["image_id"] for q in run})
        houses = len({world.items[q]["house_id"] for q in run})
        types = Counter(world.items[q]["qa_type"] for q in sco)
        rows.append([p, images, houses, len(run), len(sco), len(run) - len(sco),
                     len({world.items[q]["house_id"] for q in sco}),
                     ", ".join("%s %d" % kv for kv in sorted(types.items()))])
    return S.write_table(out / "tables" / "T1_probe_sets",
                         ["set", "images", "houses_run", "items_run", "items_scored", "excluded",
                          "houses_scored", "scored by type"], rows, ph,
                         "T1. Dataset and probe sets (run and scored sizes, D110)")


def table_t2(recs, out, ph):
    info = defaultdict(lambda: {"ids": set(), "ver": set(), "rev": set(), "mode": set(), "prompt": set(),
                                "exp": defaultdict(lambda: [0, None, None])})
    for (m, c), slot in recs.data.items():
        d = info[m]
        for r in slot.values():
            d["ids"].add(r["model_id"])
            if r.get("model_version"):
                d["ver"].add(r["model_version"])
            if r.get("model_revision"):
                d["rev"].add(r["model_revision"])
            d["mode"].add(r["mode"])
            d["prompt"].add(r["prompt_version"])
            e = d["exp"][S.EXPERIMENT[c]]
            e[0] += 1
            t = r["ts_utc"] or ""
            e[1] = t if e[1] is None or t < e[1] else e[1]
            e[2] = t if e[2] is None or t > e[2] else e[2]
    rows = []
    for m in list(S.MODELS) + list(S.FT_SEEDS):
        if m not in info:
            continue
        d = info[m]
        for exp, (n, t1, t2) in sorted(d["exp"].items()):
            rows.append([S.MODEL_NAMES.get(m, m), m, "; ".join(sorted(d["ids"])), "; ".join(sorted(d["ver"])) or "-",
                         "; ".join(sorted(d["rev"])) or "-", "; ".join(sorted(d["mode"])),
                         "; ".join(sorted(d["prompt"])), exp, n, t1, t2])
    return S.write_table(out / "tables" / "T2_models_record_columns",
                         ["model", "model_key", "model_id", "model_version", "model_revision", "mode",
                          "prompt_version", "experiment", "answers", "first_answer_utc", "last_answer_utc"],
                         rows, ph, "T2 (record columns). Models, versions and run dates; the settings"
                                   " columns come from Design 5.2 and addendum E")


def table_t4(res, out, ph):
    rows = []
    for m in S.MODELS:
        c = res["contrasts"][m]
        h = res["holm"][m]
        d = res["direction"][m]
        for k in ("P1", "P2", "P12", "P3", "P4", "P5", "P6", "P7"):
            s = c[k]
            rows.append([S.MODEL_NAMES[m], k, S.ci_text(s), "" if "p" not in s else S.fmt_ps(s),
                         S.fmt_holm(h.get(k)) if k in h else "", s.get("n_undefined"),
                         d.get(k, {}).get("agrees", "") if k in ("P1", "P2", "P7") else ""])
    r = res.get("exp_r", {})
    if "P8" in r:
        rows.append(["%s fine-tuned (%d seeds)" % (S.MODEL_NAMES[S.FT_BASE], len(r["seeds"])), "P8",
                     S.ci_text(r["P8"]), S.fmt_ps(r["P8"]), "", r["P8"].get("n_undefined"), ""])
    return S.write_table(out / "tables" / "T4_contrasts",
                         ["model", "contrast", "estimate pp [95% CI]", "p (two-sided)", "Holm p",
                          "undefined resamples", "GEE direction"], rows, ph,
                         "T4. Contrasts P1 to P8 and P12; Holm within each model over P12, P3 to P7")


def table_t5(res, out, ph):
    rows = []
    h2 = res["ht2"]
    rows.append(["HT-2 translation", "faithful pairs", "%d of %d" % (h2["k"], h2["n"]),
                 "%s [%s, %s]" % (S.pct(h2["point"], 2), S.pct(h2["lo"], 2), S.pct(h2["hi"], 2)), "Wilson 95% CI"])
    js = res["ht3"]["json"]
    p, ci = js["summary"]["point"], js["summary"]["ci95"]

    def c3(k, kappa=False):
        lo, hi = (ci.get(k) or [None, None])
        if kappa:
            return "%s [%s, %s]" % tuple("-" if v is None else "%.3f" % v for v in (p.get(k), lo, hi))
        return "%s [%s, %s]" % (S.pct(p.get(k)), S.pct(lo), S.pct(hi))
    rows.append(["HT-3 naming reference", "six-label agreement (primary)", js["summary"]["n_counted"], c3("agree6"), "house bootstrap"])
    rows.append(["HT-3 naming reference", "six-label kappa (primary)", "", c3("kappa6", True), ""])
    rows.append(["HT-3 naming reference", "strict agreement (secondary)", "", c3("agree_strict"), ""])
    rows.append(["HT-3 naming reference", "strict kappa (secondary)", "", c3("kappa_strict", True), ""])
    rows.append(["HT-3 naming reference", "strict, N1 (main row, D214)", "", c3("strict_N1"), ""])
    rows.append(["HT-3 naming reference", "strict, N2 (secondary row, D214)", "", c3("strict_N2"), ""])
    h4 = res["ht4"]
    rows.append(["HT-4 scorer check", "scorer accuracy against consensus", h4["n"], S.ci_text(h4["accuracy"]), "house bootstrap"])
    for k in ("scorer_vs_checker1", "scorer_vs_checker2", "checker1_vs_checker2", "scorer_vs_consensus"):
        s = h4[k]["kappa_ci"]
        rows.append(["HT-4 scorer check", "kappa %s" % k.replace("_", " "), "",
                     "-" if s["point"] is None else "%.3f [%s, %s]" % (s["point"], "-" if s["lo"] is None else "%.3f" % s["lo"],
                                                                       "-" if s["hi"] is None else "%.3f" % s["hi"]), ""])
    h5 = res["ht5"]
    rows.append(["HT-5 error audit", "gold-error rate (all audited)", h5["n"], S.ci_text(h5["gold_error_rate"]), "house bootstrap"])
    rows.append(["HT-5 error audit", "C answers: gold or visibility", h5["strata"]["C"]["n"],
                 S.ci_text(h5["C_gold_or_visibility"]), "the F3 mark"])
    for st in S.HT5_STRATA:
        for v in S.HT5_VERDICTS:
            rows.append(["HT-5 error audit", "%s: %s" % (st, v), h5["strata"][st]["n"],
                         S.ci_text(h5["strata"][st]["verdicts"][v]), ""])
    return S.write_table(out / "tables" / "T5_instrument_checks", ["check", "quantity", "n", "value [95% CI]", "note"],
                         rows, ph, "T5. Instrument checks")


def supplement_tables(res, out, ph):
    files = []
    L, N = res["exp_l"], res["exp_n"]
    # S1 per type and cell, with B0, refusal and the F11 columns
    rows = []
    for m in S.MODELS:
        for c in S.COND_L:
            cs = L["models"][m]["cells"][c]
            rows.append([S.MODEL_NAMES[m], c, "all", cs["n"], S.ci_text(cs["strict"]), S.pct(cs["lenient"]["point"]),
                         S.pct(cs["fidelity"]["point"]), S.ci_text(cs["refusal"]), S.pct(cs["empty"]),
                         S.pct(cs["le8_words"]), cs["mixed_script_n"], cs["correct_with_refusal_phrase_n"],
                         S.pct(cs["no_content"]), S.ci_text(L["b0"]["all"])])
            for t in S.CLOSED_TYPES:
                cs = L["models"][m]["per_type"][c][t]
                rows.append([S.MODEL_NAMES[m], c, t, cs["n"], S.ci_text(cs["strict"]), S.pct(cs["lenient"]["point"]),
                             S.pct(cs["fidelity"]["point"]), S.ci_text(cs["refusal"]), S.pct(cs["empty"]),
                             S.pct(cs["le8_words"]), cs["mixed_script_n"], cs["correct_with_refusal_phrase_n"],
                             S.pct(cs["no_content"]), S.ci_text(L["b0"]["by_type"][t])])
    files.append(S.write_table(out / "supplement" / "S1_per_type_and_cell",
                               ["model", "condition", "qa_type", "n", "strict % [CI]", "lenient %", "fidelity %",
                                "refusal % [CI]", "empty %", "8 words or fewer %", "mixed script n",
                                "correct with refusal phrase n", "no content %", "B0 strict % [CI]"],
                               rows, ph, "S1. EXP-L per type and cell, against B0; refusal beside every accuracy"))
    # S2 forced choice
    rows = []
    for m in S.MODELS:
        for probe, block in res["fc"][m].items():
            for c in [k for k in block if k.startswith("N-FC")]:
                b = block[c]
                rows.append([S.MODEL_NAMES[m], probe, c, "all", b["n"], S.ci_text(b["accuracy"]), S.ci_text(b["invalid"]),
                             S.ci_text(b["refusal"]), S.ci_text(b["vs_chance"]), S.fmt_ps(b["vs_chance"]), ""])
            rows.append([S.MODEL_NAMES[m], probe, "with minus without image", "all", "", S.ci_text(block["image_effect"]),
                         "", "", "", S.fmt_ps(block["image_effect"]), ""])
            for cl, row in block["by_cluster"].items():
                for c in [k for k in row if k.startswith("N-FC")]:
                    v = row[c]
                    val = ("%d of %d correct (count only, H5)" % (v["correct"], row["n"])) if v.get("count_only") \
                        else S.ci_text(v)
                    rows.append([S.MODEL_NAMES[m], probe, c, cl, row["n"], val, "", "", "", "",
                                 "count only" if v.get("count_only") else ""])
    files.append(S.write_table(out / "supplement" / "S2_forced_choice",
                               ["model", "probe", "condition", "cluster", "n", "accuracy % [CI]", "invalid % [CI]",
                                "refusal % [CI]", "minus chance pp [CI]", "p", "note"], rows, ph,
                               "S2. Forced choice with and without the image, against chance (25%); invalid and refusal shares (F11)"))
    # S3 origin
    rows = []
    for m in S.MODELS:
        nm = N["models"][m]
        for g in ("yes", "no"):
            rows.append([S.MODEL_NAMES[m], "en_route %s" % g, nm["transfer_n"][g], S.ci_text(nm["transfer"][g])])
        for o in S.ORIGINS:
            rows.append([S.MODEL_NAMES[m], "origin3 %s" % o, nm["transfer_by_origin_n"][o], S.ci_text(nm["transfer_by_origin"][o])])
        rows.append([S.MODEL_NAMES[m], "origin3 uncertain (left out, Design 4.3)", N["p7_items"]["origin_uncertain"], ""])
    files.append(S.write_table(out / "supplement" / "S3_origin", ["model", "group", "items at Tier 1 in English",
                                                                  "transfer % [CI]"], rows, ph,
                               "S3. Word form by en_route and by origin3 (pooled EXP-N and EXP-L naming items)"))
    # S4 cultural
    rows = []
    cu = res["cultural"]
    for m in S.MODELS:
        for c, block in cu["models"][m].items():
            for t, s in block.items():
                rows.append([S.MODEL_NAMES[m], c, t, s["n"], S.ci_text(s)])
    files.append(S.write_table(out / "supplement" / "S4_cultural_stratum", ["model", "condition", "tag", "n", "strict % [CI]"],
                               rows, ph, "S4. Cultural stratum (exploratory, descriptive only; not_rated outside it)"))
    # S5 token cost
    rows = []
    for m in S.MODELS:
        g = res["gee"][m]["token_cost"] or {}
        lr = res["token_ratio"].get(m, {})
        rows.append([S.MODEL_NAMES[m], "Bangla to English token ratio, median [IQR]", lr.get("text", "-"), "", ""])
        for term in ("log_tok_ratio", "q_bn:log_tok_ratio"):
            t = g.get("terms", {}).get(term)
            rows.append([S.MODEL_NAMES[m], term, "-" if not t else "%.3f" % t["estimate"],
                         "-" if not t else "[%.3f, %.3f]" % (t["lo"], t["hi"]), "-" if not t else S.fmt_p(t["p"])])
        if g.get("error"):
            rows.append([S.MODEL_NAMES[m], "fit failed", g["error"], "", ""])
    files.append(S.write_table(out / "supplement" / "S5_token_cost", ["model", "quantity", "estimate", "95% CI", "p"],
                               rows, ph, "S5. Token cost: secondary EXP-L GEE with the log token ratio (log-odds)"))
    # S6 visual covariates
    rows = []
    for m in S.MODELS:
        for c in S.COND_L:
            for k, s in L["models"][m]["per_capture"][c].items():
                rows.append([S.MODEL_NAMES[m], "EXP-L", c, "capture %s" % k, L["capture_n"][k], S.ci_text(s)])
    for name, block in res["visual"].items():
        if block.get("cuts"):
            rows.append(["-", name, "-", "box area tertile cuts (share of the image)", block["n_with_box"],
                         "%.3f and %.3f; close-up items %d" % (block["cuts"][0], block["cuts"][1], block["n_close_up"])])
        for m in S.MODELS:
            for c, tert in block.get("models", {}).get(m, {}).items():
                for t, s in tert.items():
                    rows.append([S.MODEL_NAMES[m], name, c, "box area %s" % t, s["n"], S.ci_text(s)])
    files.append(S.write_table(out / "supplement" / "S6_visual_covariates", ["model", "set", "condition", "group", "n",
                                                                             "strict % [CI]"], rows, ph,
                               "S6. Visual covariates: capture type, and box-area tertiles on close-up items with a box"))
    # S7 annotators
    rows = []
    for name, block in res["annotators"].items():
        for a, b in block.items():
            for m in S.MODELS:
                for c, s in b["models"][m].items():
                    rows.append([name, a, b["houses"], b["n"], S.MODEL_NAMES[m], c, S.ci_text(s)])
    files.append(S.write_table(out / "supplement" / "S7_per_annotator", ["set", "annotator", "houses", "items", "model",
                                                                         "condition", "strict % [CI]"], rows, ph,
                               "S7. Per-annotator results (house_assignment.csv)"))
    # S8 GEE: house and template groups, and the word-form model
    rows = []
    for m in S.MODELS:
        g = res["gee"][m]
        for fit_name in ("exp_l_house", "exp_l_template", "word_form"):
            f = g[fit_name] or {}
            if f.get("error"):
                rows.append([S.MODEL_NAMES[m], fit_name, "fit failed", f["error"], "", "", "", ""])
                continue
            for term, t in f.get("terms", {}).items():
                rows.append([S.MODEL_NAMES[m], fit_name, term, "%.3f" % t["estimate"], "%.3f" % t["se"],
                             "[%.3f, %.3f]" % (t["lo"], t["hi"]), S.fmt_p(t["p"]),
                             "groups %d; %s" % (f.get("n_groups", 0), "; ".join(f.get("warnings", [])) or "no warnings")])
        cmp_ = res["template_conservative"].get(m, {})
        if cmp_:
            ex = cmp_["ci_excludes_0"]
            rows.append([S.MODEL_NAMES[m], "template vs house", "robust SE ratio (template / house)",
                         "; ".join("%s %.2f" % kv for kv in cmp_["se_ratio"].items()), "", "", "",
                         "%s; language terms with a CI that excludes 0: house %d of 3, template %d of 3" % (
                             "more conservative" if max(cmp_["se_ratio"].values()) >= TEMPLATE_SE_RATIO
                             or ex["template"] < ex["house"] else "not more conservative", ex["house"], ex["template"])])
    files.append(S.write_table(out / "supplement" / "S8_gee_and_template_sensitivity",
                               ["model", "fit", "term", "estimate (log-odds)", "robust SE", "95% CI", "p", "note"],
                               rows, ph, "S8. GEE checks: EXP-L with house and template_id groups; word form (H4 interaction)"))
    # S9 lexicon sensitivity
    rows = []
    sens = res["sensitivity"]
    for r in sens["per_row"]:
        rows.append(["row", r["row_id"], r["source"], r["op"], r["target"], r["target_id"], r["changed"], "", ""])
    for k, (prim, sen) in sens["compare"].items():
        rows.append(["quantity", k, "", "", "", "", "", S.ci_text(prim), S.ci_text(sen)])
    rows.append(["total", "answers whose label or score changed", "", "", "", "", sens["changed_total"], "", ""])
    files.append(S.write_table(out / "supplement" / "S9_lexicon_sensitivity",
                               ["kind", "id", "source", "op", "target", "target_id",
                                "answers changed (row left out vs all rows)", "primary [CI]", "sensitivity [CI]"], rows, ph,
                               "S9. Lexicon sensitivity: HT-5 variants and the D192 gaps; the frozen lexicon stays primary"))
    # S10 metric check
    rows = []
    mc = res["metric"]
    for m in S.MODELS:
        b = mc["models"][m]
        for s in S.SEGMENTS:
            v = b["per_segment"][s]
            rows.append([S.MODEL_NAMES[m], "mean", s, v["n"], "-" if v["chrf"] is None else "%.1f" % v["chrf"],
                         "-" if v["bertscore_f1"] is None else "%.3f" % v["bertscore_f1"]])
        for k, a in b["auc"].items():
            rows.append([S.MODEL_NAMES[m], "AUC", k, "%d / %d" % tuple(a["n"]),
                         "-" if a["chrf"] is None else "%.3f" % a["chrf"],
                         "-" if a["bertscore_f1"] is None else "%.3f" % a["bertscore_f1"]])
    files.append(S.write_table(out / "supplement" / "S10_metric_check", ["model", "statistic", "group", "n", "chrF++",
                                                                         "BERTScore F1"], rows, ph,
                               "S10. Metric check on N-BB answers against the tiers (a known finding replicated)"))
    return files


def figure_data(res, out, ph):
    files = []
    L, N = res["exp_l"], res["exp_n"]
    rows = []
    for m in S.MODELS:
        for c in S.COND_L:
            cs = L["models"][m]["cells"][c]
            rows.append([m, c, cs["n"], S.ci_text(cs["strict"]), S.ci_text(cs["refusal"])])
    rows.append(["B0", "baseline", L["n_items"], S.ci_text(L["b0"]["all"]), ""])
    files.append(S.write_table(out / "figures" / "F2_data", ["model", "condition", "n", "strict % [CI]", "refusal % [CI]"],
                               rows, ph, "F2 data (table view)"))
    rows = []
    for m in S.MODELS:
        b = res["budget"][m]
        for k in S.BUDGET_CATS:
            rows.append([m, k, b["counts"][k], b["n_denominator"], S.ci_text(b["shares"][k])])
        g = b["guessing_level"]
        rows.append([m, "guessing level N-FCB0 (not A)", "", g["n"], S.ci_text(g["accuracy"])])
        rows.append([m, "guessing: invalid share", "", g["n"], S.ci_text(g["invalid"])])
        rows.append([m, "guessing: refusal share", "", g["n"], S.ci_text(g["refusal"])])
        rows.append([m, "A4 empty or refusal", b["A4_empty_or_refusal"], "", ""])
        rows.append([m, "two-way split A share (en_unique P-NAME)", "", b["two_way"]["n_denominator"],
                     S.ci_text(b["two_way"]["A_share"])])
        rows.append([m, "D204 sibling labels on HT-6 pairs, in the budget", b["siblings_on_ht6_pairs_in_budget"], "", ""])
        for c, v in res["d204"][m].items():
            rows.append([m, "D204 %s sibling labels on HT-6 pairs" % c, v["on_ht6_pairs"], v["sibling_labels"], ""])
    if res.get("ht5"):
        rows.append(["HT-5", "C answers gold or visibility (pooled)", "", res["ht5"]["strata"]["C"]["n"],
                     S.ci_text(res["ht5"]["C_gold_or_visibility"])])
    files.append(S.write_table(out / "figures" / "F3_data", ["model", "category", "count", "denominator", "share % [CI]"],
                               rows, ph, "F3 data (table view)"))
    rows = []
    for m in S.MODELS:
        nm = N["models"][m]
        for g in ("yes", "no"):
            rows.append([m, "en_route %s" % g, nm["transfer_n"][g], S.ci_text(nm["transfer"][g])])
        for o, s in nm["transfer_by_origin"].items():
            rows.append([m, "origin3 %s" % o, nm["transfer_by_origin_n"][o], S.ci_text(s)])
    files.append(S.write_table(out / "figures" / "F4_data", ["model", "group", "n at Tier 1 in English", "transfer % [CI]"],
                               rows, ph, "F4 data (table view)"))
    rows = []
    for m in S.MODELS:
        for c in S.COND_N_OPEN:
            comp = N["models"][m]["composition_en_unique"][c]
            der = N["models"][m]["derived"]["%s|en_unique" % c]
            rows.append([m, c, "P-NAME en_unique", comp["n"]] + [comp["counts"][s] for s in S.SEGMENTS] +
                        [S.ci_text(der["tier1_rate"]), S.ci_text(der["fallback"])])
    hr = res.get("human_ref") or {}
    for who in ("human N1", "human N2", "human pooled") + tuple(S.MODELS):
        if who not in hr:
            continue
        comp = hr[who]
        rows.append([who if who.startswith("human") else who, "N-BB",
                     "NAME-REF en_unique (%d items, %d houses)" % (hr["n_items"], hr["houses"]), comp["n"]] +
                    [comp["counts"][s] for s in S.SEGMENTS] + [S.ci_text(comp["tier1_rate"]), S.ci_text(comp["fallback"])])
    files.append(S.write_table(out / "figures" / "F5_data", ["who", "condition", "items", "n"] + list(S.SEGMENTS) +
                               ["Tier-1 rate % [CI]", "generic fallback % [CI]"], rows, ph,
                               "F5 data (table view): counts per segment; human rows N1 (main, D214), N2 and pooled,"
                               " with the models on the same NAME-REF items (Design 9.4 item 7)"))
    rows = []
    r = res["exp_r"]
    if "deltas" in r:
        for k, s in r["deltas"].items():
            words = k == "LEN"
            rows.append(["mean of %d seeds" % len(r["seeds"]), EXP_R_TEXT.get(k, k), "words" if words else "points",
                         S.ci_text(s, scale=1.0 if words else 100.0, digits=2 if words else 1)] +
                        [("%.2f" % v) if words else ("%.1f" % (100 * v)) for v in r["per_seed"].get(k, {}).values()])
        rows.append(["P8", "dBN minus dEN", "points", S.ci_text(r["P8"])] +
                    ["%.1f" % (100 * v) for v in r["P8_per_seed"].values()])
    files.append(S.write_table(out / "figures" / "F6_data", ["what", "readout", "unit", "estimate [95% CI]"] +
                               list(r.get("seeds", [])), rows, ph,
                               "F6 data (table view): fine-tuned minus base; one column per seed after the CI"))
    return files


def report(res, ph, meta):
    P = "[PLACEHOLDER] " if ph else ""
    L, N = res["exp_l"], res["exp_n"]
    lines = ["# %sAnalysis report" % ("PLACEHOLDER: made-up data, these numbers are not results. " if ph else ""), "",
             "script %s, stats %s, scorer %s, prompt %s, UTC %s, mode %s" % (
                 SCRIPT_VERSION, S.STATS_VERSION, meta["scorer_version"], meta["prompt_version"], meta["utc"],
                 meta["mode"]),
             "bootstrap: %d resamples, seed %d; houses EXP-L %d, EXP-N %d, P7 pooled %d, EXP-R %d" % (
                 meta["n_boot"], S.SEED, L["houses"], N["houses"], N["p7_items"]["houses"], res["exp_r"].get("houses", 0)),
             ""]
    lines.append("## EXP-L (P-CORE-CLOSED, %d scored items)" % L["n_items"])
    lines.append("%sB0 strict: %s (template %s, type fallback %s)" % (P, S.ci_text(L["b0"]["all"]),
                                                                    L["b0"]["source"].get("template", 0),
                                                                    L["b0"]["source"].get("type", 0)))
    for m in S.MODELS:
        cells = L["models"][m]["cells"]
        lines.append("%s%s cells: %s" % (P, S.MODEL_NAMES[m], "; ".join(
            "%s %s%s" % (c, S.ci_text(cells[c]["strict"]),
                         (" (refusal %s)" % S.pct(cells[c]["refusal"]["point"])) if c in S.NO_IMAGE else "")
            for c in S.COND_L)))
    lines += ["", "## Contrasts (T4)"]
    for m in S.MODELS:
        for k in ("P1", "P2", "P12", "P3", "P4", "P5", "P6", "P7"):
            s = res["contrasts"][m][k]
            lines.append("%s%s %s: %s pp, p %s, Holm %s%s" % (
                P, S.MODEL_NAMES[m], k, S.ci_text(s), S.fmt_ps(s) if "p" in s else "-",
                S.fmt_holm(res["holm"][m].get(k)) if k in res["holm"][m] else "-",
                (", GEE direction %s" % res["direction"][m][k]["agrees"]) if k in res["direction"][m] else ""))
        info = N["models"][m]["info"]
        lines.append("%s%s information: P5 any language %s; P6 known in English %s (n %d), not known %s (n %d)" % (
            P, S.MODEL_NAMES[m], S.ci_text(info["P5_any_language"]), S.ci_text(info["P6_known_in_english"]),
            info["P6_known_n"], S.ci_text(info["P6_not_known_in_english"]), info["P6_not_known_n"]))
    pi = N["p7_items"]
    lines += ["", "## Word-form items (P7 pool, counts recomputed through resolve_gold; C5, F1, H1)",
              "%s%d items in %d houses; by source and en_route: %s; origin3 uncertain %d (left out of the"
              " origin split); P-NAME and P-CORE-CLOSED overlap counted once %d" % (
                  P, pi["n"], pi["houses"], ", ".join("%s %d" % kv for kv in pi["by_source_route"].items()),
                  pi["origin_uncertain"], pi["overlap_pname_pcc"])]
    lines += ["", "## Mixed-script answers (B5: where strict and fidelity can disagree)"]
    for m in S.MODELS:
        lines.append("%s%s: %s" % (P, S.MODEL_NAMES[m], "; ".join(
            ["%s %d" % (c, L["models"][m]["cells"][c]["mixed_script_n"]) for c in S.COND_L] +
            ["%s %d" % (c, N["models"][m]["mixed_script_n"][c]) for c in S.COND_N_OPEN])))
    lines += ["", "## Error budget (F3) and D204"]
    for m in S.MODELS:
        b = res["budget"][m]
        lines.append("%s%s: denominator %d of %d; %s; guessing level %s (invalid %s, refusal %s); two-way A %s;"
                     " D204 sibling labels on HT-6 pairs: %s" % (
                         P, S.MODEL_NAMES[m], b["n_denominator"], b["n_items"],
                         ", ".join("%s %s" % (k, S.ci_text(b["shares"][k])) for k in S.BUDGET_CATS),
                         S.ci_text(b["guessing_level"]["accuracy"]), S.pct(b["guessing_level"]["invalid"]["point"]),
                         S.pct(b["guessing_level"]["refusal"]["point"]), S.ci_text(b["two_way"]["A_share"]),
                         ", ".join("%s %d of %d" % (c, v["on_ht6_pairs"], v["sibling_labels"])
                                   for c, v in res["d204"][m].items())))
    if res.get("ht5"):
        lines.append("%sHT-5: C answers marked gold or visibility %s" % (P, S.ci_text(res["ht5"]["C_gold_or_visibility"])))
    lines += ["", "## Forced choice (F11 shares beside every accuracy)"]
    for m in S.MODELS:
        for probe, block in res["fc"][m].items():
            lines.append("%s%s %s: %s" % (P, S.MODEL_NAMES[m], probe, "; ".join(
                "%s %s (invalid %s, refusal %s)" % (c, S.ci_text(block[c]["accuracy"]), S.pct(block[c]["invalid"]["point"]),
                                                   S.pct(block[c]["refusal"]["point"]))
                for c in block if c.startswith("N-FC"))))
    r = res["exp_r"]
    lines += ["", "## EXP-R (R-TEST, %d items, %d houses)" % (r.get("n_items", 0), r.get("houses", 0))]
    if "deltas" in r:
        lines.append("%sseeds %s" % (P, ", ".join(r["seeds"])))
        for k, s in r["deltas"].items():
            words = k == "LEN"
            lines.append("%s%s: %s %s (per seed %s)" % (
                P, EXP_R_TEXT.get(k, k), S.ci_text(s, scale=1.0 if words else 100.0, digits=2 if words else 1),
                "words" if words else "points", ", ".join(("%.2f" % v) if words else ("%.1f" % (100 * v))
                                                          for v in r["per_seed"][k].values())))
        lines.append("%sP8: %s, p %s" % (P, S.ci_text(r["P8"]), S.fmt_ps(r["P8"])))
        lines.append("%sR-BB0 refusal: %s" % (P, "; ".join("%s %s" % (v, S.pct(lv["REF0"]["point"]))
                                                           for v, lv in r["levels"].items())))
    else:
        lines.append(r.get("skipped", ""))
    lines += ["", "## Instrument checks (T5)", "%sHT-2 %d of %d faithful, Wilson %s to %s" % (
        P, res["ht2"]["k"], res["ht2"]["n"], S.pct(res["ht2"]["lo"], 2), S.pct(res["ht2"]["hi"], 2))]
    js = res["ht3"]["json"]["summary"]
    k6 = js["point"].get("kappa6")
    lines.append("%sHT-3 six-label agreement %s, kappa %s (primary); strict N1 %s (main row), N2 %s (D214)" % (
        P, S.pct(js["point"].get("agree6")), "-" if k6 is None else "%.3f" % k6,
        S.pct(js["point"].get("strict_N1")), S.pct(js["point"].get("strict_N2"))))
    hr = res.get("human_ref") or {}
    if hr:
        lines.append("%sHuman reference on %d en_unique NAME-REF items (%d houses), Tier-1 rate and generic"
                     " fallback: %s" % (P, hr["n_items"], hr["houses"], "; ".join(
                         "%s %s, %s" % (who, S.ci_text(hr[who]["tier1_rate"]), S.ci_text(hr[who]["fallback"]))
                         for who in ("human N1", "human N2", "human pooled") + tuple(S.MODELS))))
    lines.append("%sHT-4 scorer accuracy %s" % (P, S.ci_text(res["ht4"]["accuracy"])))
    lines += ["", "## Lexicon sensitivity (S9)", "%sanswers changed: %d" % (P, res["sensitivity"]["changed_total"])]
    for k, (prim, sen) in res["sensitivity"]["compare"].items():
        lines.append("%s%s: primary %s, sensitivity %s" % (P, k, S.ci_text(prim), S.ci_text(sen)))
    lines += ["", "## Failed stages", "\n".join(res["failed"]) or "none", ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 3. run
# --------------------------------------------------------------------------

def real_mode_checks(paths, checks, deviation):
    repo = paths.repo
    running = {"scripts/analysis.py": Path(__file__), "src/stats.py": Path(S.__file__)}
    for name in ("scoring", "normalise"):
        mod = sys.modules.get(name)
        if mod is not None:
            running["src/%s.py" % name] = Path(mod.__file__)
    for rel, f in sorted(running.items()):
        same = f.resolve() == (repo / rel).resolve()
        checks.check(same, "running %s" % rel, "the file in --repo" if same else
                     "runs from %s, not from --repo; run the script inside the repo" % f)
    for tag, rel in (("gate-A", "lexicon/lexicon.csv"), ("gate-A", "lexicon/closed_vocab.csv"),
                     ("gate-F", "src/scoring.py"), ("gate-F", "src/normalise.py")):
        same = S.same_as_tag(repo, tag, rel)
        checks.check(same is True, "%s at tag %s" % (rel, tag),
                     "identical" if same else ("tag missing" if same is None else "differs from the tagged file"))
    for rel in ("src/stats.py", "scripts/analysis.py"):
        same = S.same_as_tag(repo, "gate-E", rel)
        if same is True:
            checks.add("PASS", "%s at tag gate-E" % rel, "identical: the pre-specified code")
        elif deviation:
            checks.add("WARN", "%s at tag gate-E" % rel, "%s; deviation recorded: %s" % (
                "tag missing" if same is None else "differs", deviation))
        else:
            checks.add("FAIL", "%s at tag gate-E" % rel, "%s; a change after Gate E needs --deviation"
                       " \"<reason>\" and a decision row" % ("tag missing" if same is None else "differs"))


def cmd_check_inputs(args):
    say("analysis check-inputs (%s, %s, mode %s, UTC %s)" % (SCRIPT_VERSION, S.STATS_VERSION, args.mode, S.now_utc()))
    checks = S.Checks(soften=args.allow_other_inputs and args.mode == "placeholder")
    paths = paths_from(args)
    try:
        world = S.build_world(paths, checks, need_dataset=True, soften=checks.soften)
    except Exception as exc:
        checks.add("FAIL", "fixed inputs", "%s: %s" % (type(exc).__name__, exc))
        say("check-inputs: stopped")
        return 1
    for mod in ("numpy", "statsmodels", "matplotlib", "yaml", "sacrebleu", "transformers", "torch"):
        try:
            m = __import__(mod)
            checks.add("PASS", "package %s" % mod, getattr(m, "__version__", "?"))
        except Exception as exc:
            checks.add("FAIL", "package %s" % mod, "%s (install it in Cell 0)" % type(exc).__name__)
    inputs = resolve_inputs(args, paths)
    for k, v in inputs.items():
        if k == "scored":
            n = sum(len(__import__("glob").glob(str(p))) for p in v)
            checks.add("PASS" if n else ("INFO" if args.mode == "placeholder" else "FAIL"), "input %s" % k,
                       "%d files match" % n if n else "no file yet%s" % (
                           " (written by the placeholder command)" if args.mode == "placeholder" else ""))
        else:
            ok = v is not None and Path(v).exists()
            checks.add("PASS" if ok else ("INFO" if args.mode == "placeholder" else "FAIL"), "input %s" % k,
                       str(v) if ok else "missing%s" % (" (written by the placeholder command)"
                                                         if args.mode == "placeholder" else ""))
    if args.mode == "real":
        real_mode_checks(paths, checks, args.deviation)
    n_fail = checks.count("FAIL")
    say("check-inputs: %s" % ("ALL PASS" if not n_fail else "%d FAIL" % n_fail))
    return 1 if n_fail else 0


def resolve_inputs(args, paths):
    base = Path(args.inputs) if args.inputs else None
    ht3 = args.ht3_dir
    if ht3 is None and base is not None:
        found = sorted((base / "human" / "HT-3").glob("score_placeholder_*")) if args.mode == "placeholder" else []
        ht3 = str(found[-1]) if found else None
    return {"scored": args.scored or ([str(base / "scored" / "*_scored.jsonl")] if base else []),
            "ht3_dir": ht3,
            "ht4": args.ht4 or (str(base / "human" / "HT-4" / "ht4_labels.csv") if base else None),
            "ht5": args.ht5 or (str(base / "human" / "HT-5" / "ht5_audit.csv") if base else None),
            "sensitivity": args.sensitivity or (str(base / "human" / "HT-5" / "lexicon_sensitivity.csv") if base else None),
            "tokens": args.tokens or (str(base / "stats_inputs" / "token_counts.csv") if base else None)}


def cmd_run(args):
    t0 = time.time()
    mode = args.mode
    if mode == "real" and (args.boot != S.N_BOOT or args.allow_other_inputs):
        say("STOP: the real run uses 10,000 resamples and the pinned inputs (D232)")
        return 2
    say("analysis run (%s, %s, mode %s, UTC %s)" % (SCRIPT_VERSION, S.STATS_VERSION, mode, S.now_utc()))
    checks = S.Checks(soften=args.allow_other_inputs and mode == "placeholder")
    paths = paths_from(args)
    S.guard_out(args.out, mode, paths.repo, paths.drive)          # before any work
    world = S.build_world(paths, checks, need_dataset=True, soften=checks.soften)
    inp = resolve_inputs(args, paths)
    recs = S.load_scored(inp["scored"], world, checks, mode)
    if mode == "real":
        real_mode_checks(paths, checks, args.deviation)
    ht3 = S.read_ht3(inp["ht3_dir"], world, checks, mode)
    tokens = S.read_tokens(inp["tokens"], world, checks)
    sens_rows = S.read_sensitivity(inp["sensitivity"], checks)
    ht4_rows = S.read_ht4(inp["ht4"], world, recs, checks)
    cats = {m: S.error_budget(world, recs, m, S.HouseBoot(world.houses_of(world.scored_ids["P-NAME"]), n_boot=2,
                                                          name="categories")) for m in S.MODELS}
    ht5_rows = S.read_ht5(inp["ht5"], world, recs, cats, checks)
    ph = mode == "placeholder"
    if checks.count("FAIL"):
        say("run: %d FAIL before the analysis; nothing written. Fix the inputs first." % checks.count("FAIL"))
        return 1
    out = S.guard_out(args.out, mode, paths.repo, paths.drive) / ("run_%s%s" % (S.utc_stamp(), "_PLACEHOLDER" if ph else ""))
    out.mkdir(parents=True, exist_ok=False)
    say("output folder: %s" % out)
    B = args.boot
    boot_l = S.HouseBoot(world.houses_of(world.scored_ids["P-CORE-CLOSED"]), n_boot=B, name="EXP-L")
    boot_n = S.HouseBoot(world.houses_of(world.scored_ids["P-NAME"]), n_boot=B, name="EXP-N")
    boot_r = S.HouseBoot(world.houses_of(world.scored_ids["R-TEST"]), n_boot=B, name="EXP-R")
    boot_p7 = S.HouseBoot(world.houses_of([p[0] for p in S.p7_items(world)]), n_boot=B, name="P7")
    boot_nr = S.HouseBoot(world.houses_of(world.scored_ids["NAME-REF"]), n_boot=B, name="NAME-REF")
    res = {"failed": []}

    def stage(name, fn):
        t = time.time()
        try:
            v = fn()
            say("  %-28s done (%.1f s)" % (name, time.time() - t))
            return v
        except Exception as exc:
            res["failed"].append("%s: %s: %s" % (name, type(exc).__name__, str(exc)[:300]))
            say("  %-28s FAILED: %s: %s" % (name, type(exc).__name__, exc))
            traceback.print_exc()
            return None

    b0 = stage("B0 baseline", lambda: S.compute_b0(world))
    res["exp_l"] = stage("EXP-L", lambda: S.analyse_exp_l(world, recs, boot_l, b0))
    res["exp_n"] = stage("EXP-N", lambda: S.analyse_exp_n(world, recs, boot_n, boot_p7))
    res["budget"] = stage("error budget", lambda: {m: S.error_budget(world, recs, m, boot_n) for m in S.MODELS})
    res["d204"] = stage("D204 count", lambda: {m: S.d204_counts(world, recs, m) for m in S.MODELS})
    res["fc"] = stage("forced choice", lambda: {m: S.analyse_fc(world, recs, m, boot_n) for m in S.MODELS})
    res["cultural"] = stage("cultural stratum", lambda: S.analyse_cultural(world, recs, boot_n))
    res["visual"] = stage("visual covariates", lambda: S.analyse_visual(world, recs, boot_l, boot_n))
    res["annotators"] = stage("per annotator", lambda: S.analyse_annotators(world, recs, n_boot=B))
    res["exp_r"] = stage("EXP-R", lambda: S.analyse_exp_r(world, recs, boot_r)) or {}
    res["gee"] = stage("GEE fits", lambda: S.analyse_gee(world, recs, tokens))
    res["ht2"] = S.ht2_row()
    res["ht3"] = ht3
    res["ht4"] = stage("HT-4", lambda: S.ht4_stats(ht4_rows, n_boot=B))
    res["ht5"] = stage("HT-5", lambda: S.ht5_stats(ht5_rows, n_boot=B))

    res["human_ref"] = stage("human naming reference", lambda: S.human_reference(world, ht3, recs, boot_nr))

    # T4: contrasts, Holm within each model, GEE direction
    contrasts, holm_, direction = {}, {}, {}
    for m in S.MODELS:
        c = dict(res["exp_l"]["models"][m]["contrasts"]) if res.get("exp_l") else {}
        c.update(res["exp_n"]["models"][m]["contrasts"] if res.get("exp_n") else {})
        contrasts[m] = c
        holm_[m] = S.holm_bootstrap({k: c[k] for k in ("P12", "P3", "P4", "P5", "P6", "P7") if k in c})
        direction[m] = S.direction_checks(c, (res.get("gee") or {}).get(m, {}))
    res.update({"contrasts": contrasts, "holm": holm_, "direction": direction})

    def tconserv():
        """Design 9.2: is the template_id fit more conservative? Robust SE ratios
        and how many of the three language terms keep a CI that excludes 0."""
        out_ = {}
        keys = ("q_bn", "a_bn", "q_bn:a_bn")
        for m in S.MODELS:
            h, t = res["gee"][m]["exp_l_house"], res["gee"][m]["exp_l_template"]
            if h.get("terms") and t.get("terms"):
                excl = lambda f: sum(1 for k in keys if f["terms"][k]["lo"] > 0 or f["terms"][k]["hi"] < 0)
                out_[m] = {"se_ratio": {k: t["terms"][k]["se"] / h["terms"][k]["se"] for k in keys
                                        if h["terms"][k]["se"] > 0},
                           "ci_excludes_0": {"house": excl(h), "template": excl(t)}}
        return out_
    res["template_conservative"] = stage("template comparison", tconserv) or {}

    def token_ratio():
        out_ = {}
        for m, d in (tokens or {}).items():
            v = np.exp(np.array([d[q] for q in world.scored_ids["P-CORE-CLOSED"]]))
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            out_[m] = {"median": float(med), "q1": float(q1), "q3": float(q3),
                       "text": "%.2f [%.2f, %.2f]" % (med, q1, q3)}
        return out_
    res["token_ratio"] = stage("token ratio", token_ratio) or {}

    def sensitivity():
        new, info = S.sensitivity_records(world, recs, sens_rows)
        l2 = S.analyse_exp_l(world, new, boot_l, b0)
        n2 = S.analyse_exp_n(world, new, boot_n, boot_p7)
        r2 = S.analyse_exp_r(world, new, boot_r)
        cmp_ = {}
        for m in S.MODELS:
            for k in ("P1", "P2", "P12", "P3", "P4"):
                cmp_["%s %s" % (m, k)] = (res["exp_l"]["models"][m]["contrasts"][k], l2["models"][m]["contrasts"][k])
            for c in S.COND_L:
                cmp_["%s %s strict" % (m, c)] = (res["exp_l"]["models"][m]["cells"][c]["strict"],
                                                 l2["models"][m]["cells"][c]["strict"])
            for k in ("P5", "P6", "P7"):
                cmp_["%s %s" % (m, k)] = (res["exp_n"]["models"][m]["contrasts"][k], n2["models"][m]["contrasts"][k])
            b2 = S.error_budget(world, new, m, boot_n)
            cmp_["%s error budget A share" % m] = (res["budget"][m]["A_share"], b2["A_share"])
        if "P8" in r2 and "P8" in res["exp_r"]:
            cmp_["P8"] = (res["exp_r"]["P8"], r2["P8"])
        info["compare"] = cmp_
        return info
    res["sensitivity"] = stage("lexicon sensitivity", sensitivity) or {"per_row": [], "compare": {}, "changed_total": 0}
    res["metric"] = stage("metric check", lambda: S.metric_check(world, recs)) or {"models": {
        m: {"per_segment": {s: {"n": 0, "chrf": None, "bertscore_f1": None} for s in S.SEGMENTS}, "auc": {}}
        for m in S.MODELS}}

    figs, tabs = [], []
    figs += stage("F2", lambda: S.fig_f2(res["exp_l"], out / "figures" / "F2_exp_l", ph)) or []
    figs += stage("F3", lambda: S.fig_f3(res["budget"], res.get("ht5"), out / "figures" / "F3_error_budget", ph)) or []
    figs += stage("F4", lambda: S.fig_f4(res["exp_n"], out / "figures" / "F4_word_form", ph)) or []
    figs += stage("F5", lambda: S.fig_f5(res["exp_n"], (res.get("human_ref") or {}).get("human N1"),
                                          out / "figures" / "F5_tiers", ph)) or []
    figs += stage("F6", lambda: S.fig_f6(res["exp_r"], out / "figures" / "F6_exp_r", ph)) or []
    for name, fn in (("T1", lambda: table_t1(world, out, ph)), ("T2", lambda: table_t2(recs, out, ph)),
                     ("T4", lambda: table_t4(res, out, ph)), ("T5", lambda: table_t5(res, out, ph))):
        v = stage(name, fn)
        if v:
            tabs.append(v)
    tabs += stage("supplement S1 to S10", lambda: supplement_tables(res, out, ph)) or []
    tabs += stage("figure data", lambda: figure_data(res, out, ph)) or []
    budget_rows = [[m, q, cat] for m in S.MODELS for q, cat in sorted((res.get("budget") or {}).get(m, {})
                                                                         .get("categories", {}).items())]
    S.write_csv(out / ("error_budget_items%s.csv" % ("_PLACEHOLDER" if ph else "")),
                ["model_key", "qa_id", "category"], budget_rows)
    meta = {"mode": mode, "utc": S.now_utc(), "script_version": SCRIPT_VERSION, "stats_version": S.STATS_VERSION,
            "scorer_version": recs.scorer_version, "prompt_version": recs.prompt_version, "stage": recs.stage,
            "n_boot": B, "seed": S.SEED, "deviation": args.deviation or "",
            "git_head": S.git_out(paths.repo, "rev-parse", "--short=12", "HEAD") or "unknown"}
    rep = report(res, ph, meta)
    S.write_text(out / ("analysis_report%s.md" % ("_PLACEHOLDER" if ph else "")), rep)
    S.write_json(out / ("numbers%s.json" % ("_PLACEHOLDER" if ph else "")),
                 {k: v for k, v in res.items() if k not in ("ht3",)})
    versions = {}
    for mod in ("numpy", "statsmodels", "matplotlib", "yaml", "sacrebleu", "transformers", "torch"):
        try:
            versions[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:
            versions[mod] = "not installed"
    versions["python"] = sys.version.split()[0]
    S.write_json(out / "manifest.json", dict(meta, inputs=world.hashes, scored_files=recs.files,
                                             scored_manifests=recs.manifests, ht3=ht3["sha256"],
                                             resolved_inputs=inp, versions=versions, seeds=recs.seeds,
                                             box_format=world.box_format, figures=[str(f) for f in figs],
                                             tables=[str(t) for t in tabs], failed=res["failed"],
                                             seconds=round(time.time() - t0, 1)))
    S.write_text(out / "checks.txt", checks.text())
    say("run: %d figures, %d tables, report, numbers, manifest in %s (%.0f s)" % (
        len(figs), len(tabs), out, time.time() - t0))
    if res["failed"]:
        say("run: %d stage(s) FAILED: %s" % (len(res["failed"]), "; ".join(f.split(":")[0] for f in res["failed"])))
        return 1
    say("run: ALL STAGES DONE%s" % (" (PLACEHOLDER: never results)" if ph else ""))
    return 0


# --------------------------------------------------------------------------
# 4. templates and tokens (real data, no model output)
# --------------------------------------------------------------------------

def cmd_templates(args):
    checks = S.Checks(soften=args.allow_other_inputs)
    paths = paths_from(args)
    world = S.build_world(paths, checks, need_dataset=True, soften=checks.soften)
    if checks.count("FAIL"):
        say("templates: %d FAIL in the inputs" % checks.count("FAIL"))
        return 1
    stats_, groups = S.template_report(world)
    out = S.guard_out(args.out, "real", paths.repo, paths.drive)
    out.mkdir(parents=True, exist_ok=True)
    S.write_csv(out / "template_merges.csv", ["scope", "template_id", "n_strings", "n_items", "strings"], groups)
    S.write_json(out / "template_report.json", {"utc": S.now_utc(), "script_version": SCRIPT_VERSION,
                                                "fold": [list(x) for x in S.TEMPLATE_FOLD], "counts": stats_})
    for scope, c in stats_.items():
        say("templates %s: %d items, %d distinct strings, %d templates, %d merged groups holding %d strings"
            % (scope, c["items"], c["strings"], c["templates"], c["merged_groups"], c["strings_in_merged"]))
    say("Wrote %s (template_merges.csv holds question text: Drive only; a native speaker reads it once)" % out)
    return 0


def cmd_budget_items(args):
    """The HT-5 sampling frame: error-budget categories per model and item
    (ids only), from the scored files alone. HT-5 needs it before the full run
    can read the HT-5 file, so it is its own command."""
    mode = args.mode
    checks = S.Checks(soften=args.allow_other_inputs and mode == "placeholder")
    paths = paths_from(args)
    out = S.guard_out(Path(args.out).parent, mode, paths.repo, paths.drive) / Path(args.out).name
    world = S.build_world(paths, checks, need_dataset=False, soften=checks.soften)
    recs = S.load_scored(resolve_inputs(args, paths)["scored"], world, checks, mode)
    if mode == "real":
        real_mode_checks(paths, checks, args.deviation)
    if checks.count("FAIL"):
        say("budget-items: %d FAIL; nothing written" % checks.count("FAIL"))
        return 1
    boot = S.HouseBoot(world.houses_of(world.scored_ids["P-NAME"]), n_boot=2, name="categories")
    rows = []
    for m in S.MODELS:
        b = S.error_budget(world, recs, m, boot)
        for q, cat in sorted(b["categories"].items()):
            rows.append([m, q, world.items[q]["house_id"], cat, "N-BB"])
        say("budget-items %s: %d items not at Tier 1 in N-BB; %s" % (
            m, b["n_denominator"], ", ".join("%s %d" % (k, b["counts"][k]) for k in S.BUDGET_CATS)))
    S.write_csv(out, ["model_key", "qa_id", "house_id", "category", "condition_id"], rows)
    S.write_json(str(out)[:-4] + "_manifest.json", {"utc": S.now_utc(), "script_version": SCRIPT_VERSION,
                                                    "mode": mode, "stage": recs.stage, "rows": len(rows),
                                                    "scored_files": recs.files, "inputs": world.hashes})
    say("Wrote %s (%d rows; ids and categories only)" % (out, len(rows)))
    return 0


def _secret(name):
    v = os.environ.get(name)
    if v:
        return v
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None


def _gemini_count(text, key, tries=6):
    import urllib.request
    import urllib.error
    url = "https://generativelanguage.googleapis.com/v1beta/models/%s:countTokens" % GEMINI_MODEL
    body = json.dumps({"contents": [{"role": "user", "parts": [{"text": text}]}]}).encode("utf-8")
    wait = 2.0
    for k in range(tries):
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return int(json.loads(resp.read().decode("utf-8"))["totalTokens"])
        except urllib.error.HTTPError as exc:
            if exc.code in (408, 429) or exc.code >= 500:
                time.sleep(wait)
                wait *= 2
                continue
            raise
        except urllib.error.URLError:
            time.sleep(wait)
            wait *= 2
    raise S.StatsError("countTokens failed after %d tries" % tries)


def cmd_tokens(args):
    if not args.paid_tier_confirmed:
        say("STOP: give --paid-tier-confirmed after checking the AI Studio project is on the paid tier"
            " (Design 5.5): countTokens sends the question text to Google")
        return 2
    import transformers
    if transformers.__version__ != TRANSFORMERS_VERSION:
        say("STOP: transformers %s; install transformers==%s first (D153)" % (transformers.__version__,
                                                                          TRANSFORMERS_VERSION))
        return 2
    from transformers import AutoTokenizer
    checks = S.Checks()
    paths = paths_from(args)
    world = S.build_world(paths, checks, need_dataset=False)
    sha = S.sha256_file(paths.english)
    checks.check(S.pin_ok("english_questions_final.csv", sha), "Drive english/english_questions_final.csv",
                 "sha256 %s... (version 3, Gate B)" % sha[:12])
    if checks.count("FAIL"):
        return 1
    eng = {}
    for r in S.read_csv_dicts(paths.english):
        eng[r["question_bn_norm"]] = r["question_en"]
    ids = world.run_ids["P-CORE-CLOSED"]
    texts = {}
    for q in ids:
        bn = world.items[q]["question_bn"]
        en = eng.get(world.normalise(bn))
        if en is None:
            raise S.StatsError("no English question for %s" % q)
        texts[q] = (bn, en)
    say("tokens: %d P-CORE-CLOSED items, %d distinct Bangla and %d distinct English strings"
        % (len(ids), len({t[0] for t in texts.values()}), len({t[1] for t in texts.values()})))
    hf = _secret("HF_TOKEN")
    rows, utc = [], S.now_utc()
    for m, (repo_id, rev) in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(repo_id, revision=rev, token=hf)
        cnt = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
        cache = {}
        for q in ids:
            bn, en = texts[q]
            for s in (bn, en):
                if s not in cache:
                    cache[s] = cnt(s)
            rows.append([q, m, cache[bn], cache[en], "%s@%s" % (repo_id, rev[:12]), utc])
        say("tokens: %s done (%d distinct strings)" % (m, len(cache)))
    key = _secret("GEMINI_API_KEY")
    if not key:
        raise S.StatsError("no GEMINI_API_KEY in Colab Secrets")
    cache = {}
    strings = sorted({s for t in texts.values() for s in t})
    for i, s in enumerate(strings, 1):
        cache[s] = _gemini_count(s, key)
        time.sleep(0.1)
        if i % 100 == 0:
            say("tokens: gemini %d of %d strings" % (i, len(strings)))
    for q in ids:
        bn, en = texts[q]
        rows.append([q, "gemini", cache[bn], cache[en], "%s countTokens" % GEMINI_MODEL, utc])
    out = S.guard_out(Path(args.out).parent, "real", paths.repo, paths.drive) / Path(args.out).name
    S.write_csv(out, S.TOKEN_COLUMNS, rows)
    S.write_json(str(out)[:-4] + "_manifest.json", {"utc": utc, "script_version": SCRIPT_VERSION,
                                                    "transformers": transformers.__version__,
                                                    "tokenizers": TOKENIZERS, "gemini_model": GEMINI_MODEL,
                                                    "gemini_requests": len(strings), "rows": len(rows),
                                                    "english_sha256": sha, "probe_sha256": world.hashes.get("P-CORE-CLOSED.json")})
    say("tokens: %d rows written to %s (%d countTokens requests, not billed)" % (len(rows), out, len(strings)))
    return 0


# --------------------------------------------------------------------------
# 5. selftest (invented data only)
# --------------------------------------------------------------------------

def _st_world(tmp):
    """A tiny invented world: lexicon, vocabulary, items and a dataset."""
    sc, normalise = S.load_scorer(REPO_DEFAULT)

    def lrow(nid, bn, en, cluster="", route="no", t2bn="", t2en="", uniq="yes", freq="3"):
        return {"name_id": nid, "name_bn": bn, "variants_bn": "", "canonical_en": en, "variants_en": "",
                "super_bn": t2bn, "super_en": t2en, "cluster_id": cluster, "siblings": "", "room_scope": "any",
                "cultural_specificity": "globally_common", "origin3": "inherited", "origin_note": "",
                "en_route": route, "en_unique": uniq, "freq_obj_name": freq}
    lex_rows = [lrow("X1", "ঝুমকাল", "zumkal", "K1", "yes", "পাত্রক", "vessa"),
                lrow("X2", "টপারি", "topari", "K1", "no", "পাত্রক", "vessa"),
                lrow("X3", "মিলনো", "milno", "", "yes"), lrow("X4", "কাবলু", "pindrop")]
    voc_rows = [{"kind": "room", "class_id": c, "forms_bn": b, "forms_en": c, "note": ""}
                for c, b in (("bathroom", "স্নানঘরক"), ("bedroom", "শোয়ারঘরক"), ("kitchen", "রাঁধাঘরক"),
                             ("living room", "বসাঘরক"))]
    voc_rows += [{"kind": "number", "class_id": str(i), "forms_bn": f, "forms_en": e, "note": ""}
                 for i, f, e in ((1, "একটি", "one"), (2, "দুইটি", "two"), (3, "তিনটি", "three"))]
    voc_rows += [{"kind": "colour", "class_id": c, "forms_bn": b, "forms_en": c, "note": ""}
                 for c, b in (("white", "শাদাক"), ("red", "লালক"))]
    voc_rows += [{"kind": "material", "class_id": "plastic", "forms_bn": "প্লাস্টিকক", "forms_en": "plastic", "note": ""}]
    lexdir = Path(tmp) / "lexicon"
    lexdir.mkdir(parents=True, exist_ok=True)
    S.write_csv(lexdir / "lexicon.csv", sc.LEXICON_COLUMNS, [[r[c] for c in sc.LEXICON_COLUMNS] for r in lex_rows])
    S.write_csv(lexdir / "closed_vocab.csv", sc.VOCAB_COLUMNS, [[r[c] for c in sc.VOCAB_COLUMNS] for r in voc_rows])
    lex = sc.load_lexicon(lexdir / "lexicon.csv")
    voc = sc.load_closed_vocab(lexdir / "closed_vocab.csv")
    paths = SimpleNamespace(repo=Path(tmp))
    return sc, normalise, lex, voc, paths


def cmd_selftest(args):
    import tempfile as _tf
    results = []

    def check(name, cond):
        results.append(bool(cond))
        say("%s  %s" % ("OK  " if cond else "FAIL", name))

    # Bootstrap engine
    hb = S.HouseBoot(["H1", "H2", "H3", "H4", "H5"], n_boot=500)
    check("bootstrap: every resample draws 5 houses", set(hb.W.sum(axis=1)) == {5.0})
    hb2 = S.HouseBoot(["H5", "H4", "H3", "H2", "H1"], n_boot=500)
    check("bootstrap: same resamples for the same house set (seed 42)", np.array_equal(hb.W, hb2.W))
    hidx = hb.hidx(["H1", "H1", "H2", "H3", "H4", "H5"])
    e = hb.mean(hidx, [1, 0, 1, 1, 0, 0])
    check("bootstrap: point estimate is the plain mean", abs(e.point - 0.5) < 1e-12)
    naive = []
    for row in hb.W:
        num = den = 0.0
        for h, k in zip(hb.houses, row):
            vals = [v for hh, v in zip(["H1", "H1", "H2", "H3", "H4", "H5"], [1, 0, 1, 1, 0, 0]) if hh == h]
            num += k * sum(vals)
            den += k * len(vals)
        naive.append(num / den)
    check("bootstrap: equals a naive house resampling", np.allclose(naive, e.boot))
    r = hb.ratio(hidx, [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0])
    check("bootstrap: undefined ratios are counted, not dropped silently",
          S.summarise_est(r)["n_undefined"] == int((hb.W[:, 0] == 0).sum()))
    d = S.Est(0.1, np.array([0.2, -0.1, 0.3, 0.0]))
    check("p-value: 2 x smaller share at or below and at or above 0", S.p_two_sided(d.boot) == 1.0
          and S.p_two_sided(np.array([0.1, 0.2, 0.3, -0.1])) == 0.5)
    check("Holm: step-down with monotone adjustment",
          S.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "c": 0.06, "b": 0.06})
    lo, hi = S.wilson(146, 150)
    check("Wilson: 146 of 150 gives 93.34% to 98.96% (addendum D7)", ("%.2f %.2f" % (lo * 100, hi * 100)) == "93.34 98.96")
    C = np.array([[20, 5], [10, 15]])
    po, k = S.kappa_from_confusion(C)
    check("kappa: textbook table 0.70 and 0.40", abs(po - 0.7) < 1e-12 and abs(k - 0.4) < 1e-12)
    _, k1 = S.kappa_from_confusion(np.array([[5, 0], [0, 0]]))
    check("kappa: undefined when chance agreement is 1", np.isnan(k1))
    check("AUC: separable 1.0, ties 0.5", S.auc([2, 3], [0, 1]) == 1.0 and S.auc([1, 1], [1, 1]) == 0.5)
    # Templates
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    check("template fold merges ব্যবহারে and ব্যাবহারে", S.template_key("এটি কী কাজে ব্যবহারে লাগে", norm)
          == S.template_key("এটি কি কাজে ব্যাবহারে লাগে", norm))
    check("template fold merges ঝুঁকি and ঝুকি", S.template_key("কোনো ঝুঁকি আছে", norm) == S.template_key("কোনো ঝুকি আছে", norm))
    check("template fold keeps different questions apart", S.template_key("কয়টি চেয়ার আছে", norm)
          != S.template_key("কয়টি টেবিল আছে", norm))
    # Box format
    xywh = [{"bbox_pct": [80, 10, 15, 20]}, {"bbox_pct": [5, 5, 30, 30]}]
    xyxy = [{"bbox_pct": [10, 10, 60, 70]}, {"bbox_pct": [50, 40, 90, 95]}]
    check("box format: [x, y, w, h] found", S.box_format_of(xywh) == "xywh")
    check("box format: [x1, y1, x2, y2] found", S.box_format_of(xyxy) == "xyxy")
    check("box format: stops when both readings fit", S.box_format_of([{"bbox_pct": [10, 10, 30, 30]}]) == "ambiguous")
    check("box area from percent", abs(S.box_area([10, 10, 20, 50], "xywh") - 0.10) < 1e-12
          and abs(S.box_area([10, 10, 20, 50], "xyxy") - 0.04) < 1e-12)
    # Invented world for B0, error budget, sensitivity
    with _tf.TemporaryDirectory() as tmp:
        sc, normalise, lex, voc, paths = _st_world(tmp)
        items = {}

        def add(q, house, typ, question, answer, probes, uniq=True):
            g = sc.resolve_gold({"qa_id": q, "qa_type": typ, "answer_bn": answer}, lex, voc)
            obj = lex.objects.get(g.name_id) if g.name_id else None
            items[q] = {"qa_id": q, "house_id": house, "image_id": q[:8], "qa_type": typ, "capture_type": "closeup",
                        "question_bn": question, "answer_bn": answer, "probes": set(probes), "excluded": False,
                        "template_id": S.template_id_of(S.template_key(question, normalise)),
                        "gold_kind": g.kind, "name_id": g.name_id or None,
                        "en_unique": obj.en_unique if obj else None, "en_route": obj.en_route if obj else None,
                        "in_cluster": bool(obj.cluster_id) if obj else None, "cluster_id": obj.cluster_id if obj else None,
                        "origin3": "inherited", "freq": 3, "cultural": "globally_common", "annotator": "A1",
                        "box_area": None, "is_overview": False}
        add("H01_0001_q1", "H01", "count", "কয়টি আছে", "একটি", ["P-CORE-CLOSED"])
        add("H02_0001_q1", "H02", "count", "কয়টি আছে", "দুইটি", ["P-CORE-CLOSED"])
        add("H03_0001_q1", "H03", "count", "কয়টি আছে", "দুইটি", ["P-CORE-CLOSED"])
        add("H04_0001_q1", "H04", "count", "অন্য প্রশ্ন", "তিনটি", ["P-CORE-CLOSED"])
        dataset_qa = [(q, it["qa_type"], it["question_bn"], it["answer_bn"], it["house_id"]) for q, it in items.items()]
        dataset_qa.append(("H01_0002_q1", "count", "কয়টি আছে", "তিনটি", "H01"))
        w = SimpleNamespace(sc=sc, normalise=normalise, lex=lex, voc=voc, items=items, excluded={},
                            dataset_qa=dataset_qa, scored_ids={"P-CORE-CLOSED": sorted(items)},
                            scorer_item=lambda q: {"qa_id": q, "qa_type": items[q]["qa_type"],
                                                   "answer_bn": items[q]["answer_bn"]}, paths=paths)
        b0 = S.compute_b0(w)
        check("B0: leave one house out (H01 takes দুইটি from H02 and H03, not its own)", b0["H01_0001_q1"]["strict"] == 0)
        check("B0: H02 gets the tie between একটি, দুইটি and তিনটি by the seeded rank",
              b0["H02_0001_q1"]["source"] == "template")
        check("B0: a template no other house has falls back to the type", b0["H04_0001_q1"]["source"] == "type")
        # Error budget categories on crafted labels
        for i in range(1, 7):
            add("H0%d_0009_q1" % i, "H0%d" % i, "obj_name", "এটা কী", "ঝুমকাল", ["P-NAME", "P-CLUST"])
        ids = sorted(q for q in items if q.endswith("0009_q1"))

        def rec(label, strict, lang_ok=False, empty=False, named=None):
            return {"label": label, "strict": strict, "content_lang_ok": lang_ok, "is_empty": empty, "refusal": empty,
                    "named_ids": named or [], "label_lang": "en", "gold_status": "ok"}
        nbb = dict(zip(ids, [rec("tier1", 1, True), rec("tier1", 0), rec("english_copy", 0), rec("tier2", 0),
                             rec("sibling", 0, named=["X2"]), rec("other_wrong", 0, empty=True)]))
        nbe = dict(zip(ids, [rec("tier1", 1, True), rec("tier1", 1, True), rec("tier1", 1, True),
                             rec("tier1", 1, True), rec("sibling", 0), rec("other_wrong", 0)]))
        fcb = dict(zip(ids, [rec("correct", 1)] * 4 + [rec("correct", 1), rec("wrong", 0)]))
        fcb0 = dict(zip(ids, [rec("wrong", 0)] * 6))
        recs = SimpleNamespace(get=lambda m, c: {"N-BB": nbb, "N-BE": nbe, "N-FCB": fcb, "N-FCB0": fcb0}[c])
        w.scored_ids.update({"P-CLUST": ids, "P-NAME": ids})
        w.swap_pairs = {(ids[4], "X2")}
        w.houses_of = lambda qs: [items[q]["house_id"] for q in qs]
        bn = S.HouseBoot([items[q]["house_id"] for q in ids], n_boot=200)
        eb = S.error_budget(w, recs, "m", bn)
        check("error budget: strict Tier 1 in N-BB is outside the denominator", eb["n_denominator"] == 5)
        check("error budget: A1 A2 A3 from the N-BB label, B from N-FCB, C the rest",
              eb["categories"] == {ids[1]: "A1", ids[2]: "A2", ids[3]: "A3", ids[4]: "B", ids[5]: "C"})
        check("error budget: D204 sibling on an HT-6 pair counted", eb["siblings_on_ht6_pairs_in_budget"] == 1)
        # P7 precedence
        add("H07_0001_q1", "H07", "obj_name", "এটা কী", "মিলনো", ["P-NAME", "P-CORE-CLOSED"])
        add("H08_0001_q1", "H08", "object", "কোনটি", "কাবলু", ["P-CORE-CLOSED"])
        w.scored_ids = {"P-NAME": ["H07_0001_q1"], "P-CORE-CLOSED": ["H07_0001_q1", "H08_0001_q1"]}
        p7 = S.p7_items(w)
        check("P7: a qa_id in P-NAME and P-CORE-CLOSED counts once, from N-BB and N-BE",
              [(q, e_, b_) for q, e_, b_, _ in p7] == [("H07_0001_q1", "N-BE", "N-BB"), ("H08_0001_q1", "L-BE", "L-BB")])
        # Direction rule
        dc = S.direction_checks({"P1": {"point": 0.05}, "P2": {"point": 0.10}, "P7": {"point": -0.02}},
                                {"exp_l_house": {"terms": {"q_bn": {"estimate": -0.3}, "a_bn": {"estimate": -0.6},
                                                           "q_bn:a_bn": {"estimate": 0.2}}},
                                 "word_form": {"terms": {"a_bn:en_route": {"estimate": 0.4}}}})
        check("GEE direction: P1 and P2 agree, P7 disagrees", (dc["P1"]["agrees"], dc["P2"]["agrees"], dc["P7"]["agrees"])
              == ("agrees", "agrees", "disagrees"))
        # Lexicon sensitivity on the invented lexicon
        rows = [{"row_id": "r1", "source": "HT-5", "op": "add_term", "target": "lexicon", "target_id": "X4", "tier": "1",
                 "lang": "bn", "form": "কাবলুক", "note": ""},
                {"row_id": "r2", "source": "D192", "op": "add_term", "target": "material", "target_id": "cane",
                 "tier": "", "lang": "bn", "form": "বেতক", "note": ""},
                {"row_id": "r3", "source": "D192", "op": "remove_term", "target": "colour", "target_id": "red",
                 "tier": "", "lang": "en", "form": "red", "note": ""},
                {"row_id": "r4", "source": "D192", "op": "item_any_class", "target": "", "target_id": "H09_0001_q1",
                 "tier": "", "lang": "", "form": "", "note": ""}]
        lex2, voc2, anyc = S._apply_rows(w, rows)
        before = sc.score_item({"qa_id": "q", "qa_type": "obj_name", "answer_bn": "কাবলু"}, "N-BB", "কাবলুক", lex, voc)
        after = sc.score_item({"qa_id": "q", "qa_type": "obj_name", "answer_bn": "কাবলু"}, "N-BB", "কাবলুক", lex2, voc2)
        check("sensitivity: an added Tier-1 variant turns other_wrong into tier1",
              before["label"] == "other_wrong" and after["label"] == "tier1")
        check("sensitivity: a new vocabulary class and a removed form", "cane" in voc2.classes["material"]
              and not voc2.indexes["colour"].find_all(sc.tokenise("red")))
        check("sensitivity: the frozen files are untouched", (Path(tmp) / "lexicon" / "lexicon.csv").read_text(
            encoding="utf-8-sig").count("কাবলুক") == 0 and anyc == {"H09_0001_q1"})
        it = {"qa_id": "H09_0001_q1", "qa_type": "obj_property", "answer_bn": "শাদাক"}
        items["H09_0001_q1"] = dict(items["H01_0001_q1"], qa_id="H09_0001_q1", qa_type="obj_property", answer_bn="শাদাক")
        slot = {"H09_0001_q1": {"answer_raw": "white red"}}
        primary = sc.score_item(it, "L-EE", "white red", lex, voc)
        upd = S._rescore(w, slot, "H09_0001_q1", "L-EE", lex2, voc2, anyc)
        check("sensitivity: item_any_class accepts a second colour on that item only",
              primary["strict"] == 0 and upd["strict"] == 1)
    # HT-4 statistics
    rows = [{"house_id": "H%02d" % (i % 7), "scorer_label": "tier1" if i % 3 else "tier2",
             "checker1_label": "tier1" if i % 4 else "tier2", "checker2_label": "tier1" if i % 3 else "tier2",
             "consensus_label": "tier1" if i % 3 else "tier2"} for i in range(60)]
    h4a, h4b = S.ht4_stats(rows, n_boot=300), S.ht4_stats(rows, n_boot=300)
    check("HT-4: accuracy 1.0 when scorer equals consensus; reproducible kappa CIs",
          h4a["accuracy"]["point"] == 1.0 and h4a["scorer_vs_checker1"]["kappa_ci"] == h4b["scorer_vs_checker1"]["kappa_ci"])
    # GEE on simulated data (real statsmodels when present)
    try:
        import statsmodels  # noqa: F401
        rng = np.random.default_rng(42)
        n = 2000
        g = rng.integers(0, 40, n)
        q = rng.integers(0, 2, n)
        lin = 0.5 - 1.0 * q + rng.normal(0, 0.3, 40)[g]
        y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(float)
        fit = S.gee_fit(y, np.column_stack([np.ones(n), q]), ["const", "q_bn"], g)
        check("GEE (statsmodels): a negative effect is recovered, robust SE > 0",
              fit["terms"]["q_bn"]["estimate"] < -0.5 and fit["terms"]["q_bn"]["se"] > 0)
    except ImportError:
        say("SKIP  GEE (statsmodels not installed here; Colab has it)")
    # Placeholder marks and the output guard
    with _tf.TemporaryDirectory() as tmp:
        plt = S._plt()
        fig, ax = plt.subplots()
        S.mark_placeholder(fig, True)
        check("placeholder: figures carry the mark", any(t.get_gid() == "placeholder-mark" for t in fig.texts))
        plt.close(fig)
        f = S.write_table(Path(tmp) / "placeholder" / "T", ["a"], [[1]], True, "t")
        check("placeholder: tables carry a status column", open(f, encoding="utf-8-sig").read().splitlines()[1]
              .startswith("PLACEHOLDER"))
        try:
            S.guard_out(Path(tmp) / "out", "placeholder", Path(tmp) / "repo", Path(tmp) / "drive")
            check("guard: placeholder output outside a placeholder folder is refused", False)
        except S.StatsError:
            check("guard: placeholder output outside a placeholder folder is refused", True)
        try:
            S.guard_out(Path(tmp) / "drive" / "scored" / "placeholder", "placeholder", Path(tmp) / "repo", Path(tmp) / "drive")
            check("guard: Drive scored/ is refused", False)
        except S.StatsError:
            check("guard: Drive scored/ is refused", True)
    check("transliteration makes Bangla letters", re.fullmatch(r"[ঀ-৿ ]+", transliterate("bucket")) is not None)
    try:
        from sacrebleu.metrics import CHRF
        ch = CHRF(word_order=2)
        same = "দুইটি কাঠের চেয়ার আছে"
        check("chrF++ (sacrebleu): identical 100, disjoint 0",
              abs(ch.sentence_score(same, [same]).score - 100) < 1e-6
              and ch.sentence_score("two wooden chairs", [same]).score < 1e-6)
    except ImportError:
        say("SKIP  chrF++ (sacrebleu not installed here; Cell 0 installs it)")
    n_ok = sum(results)
    say("")
    say("selftest: %d of %d checks OK" % (n_ok, len(results)))
    return 0 if n_ok == len(results) else 1


# --------------------------------------------------------------------------
# 6. Command line
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="BanglaIndoorVQA analysis (Design 9, 13)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--repo", default=str(REPO_DEFAULT))
        p.add_argument("--drive", default=str(DRIVE_DEFAULT))
        p.add_argument("--dataset")
        p.add_argument("--houses")
        p.add_argument("--probes-dir")
        p.add_argument("--manifest11")
        p.add_argument("--swaps")
        p.add_argument("--english")
        p.add_argument("--allow-other-inputs", action="store_true",
                       help="tests only: pinned values become WARN; never for the real run")

    def inputs(p):
        p.add_argument("--inputs", help="folder holding scored/, human/ and stats_inputs/ (placeholder layout)")
        p.add_argument("--scored", nargs="*", help="glob(s) of scored .jsonl files")
        p.add_argument("--ht3-dir")
        p.add_argument("--ht4")
        p.add_argument("--ht5")
        p.add_argument("--sensitivity")
        p.add_argument("--tokens")
        p.add_argument("--deviation", help="real mode: the reason for any change after the tag gate-E")

    p = sub.add_parser("selftest")
    common(p)
    p = sub.add_parser("check-inputs")
    common(p)
    inputs(p)
    p.add_argument("--mode", choices=("placeholder", "real"), required=True)
    p = sub.add_parser("templates")
    common(p)
    p.add_argument("--out", required=True)
    p = sub.add_parser("tokens")
    common(p)
    p.add_argument("--out", required=True, help="the token_counts.csv to write")
    p.add_argument("--paid-tier-confirmed", action="store_true")
    p = sub.add_parser("placeholder")
    common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--boot", type=int, default=S.N_BOOT)
    p = sub.add_parser("budget-items")
    common(p)
    inputs(p)
    p.add_argument("--mode", choices=("placeholder", "real"), required=True)
    p.add_argument("--out", required=True, help="the error_budget_items.csv to write")
    p = sub.add_parser("run")
    common(p)
    inputs(p)
    p.add_argument("--mode", choices=("placeholder", "real"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--boot", type=int, default=S.N_BOOT)
    args = ap.parse_args(argv)
    fn = {"selftest": cmd_selftest, "check-inputs": cmd_check_inputs, "templates": cmd_templates,
          "tokens": cmd_tokens, "placeholder": cmd_placeholder, "budget-items": cmd_budget_items,
          "run": cmd_run}[args.cmd]
    try:
        return fn(args)
    except S.StatsError as exc:
        say("STOP: %s" % exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
