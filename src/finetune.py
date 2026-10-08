"""
finetune.py
BanglaIndoorVQA, FT scripts: EXP-R training, its Gate G dry run and the evaluation
runs (Design 7.3; DESIGN_ADDENDUM G; decisions D170 to D178).

What is in this file
    1. The fixed settings (configs/finetune.yaml), compared with Design 7.3.
    2. The training and validation data: from the dataset and data/splits.csv only.
    3. The training target: the prompt of build_prompt(), the gold answer, the end
       of turn. The loss is on the answer positions only.
    4. Which layers get LoRA: every linear layer inside the language-model layers.
    5. Two engines with one interface: the real one (torch, PEFT, the model that the
       harness loads) and a stand-in without a GPU for the self checks.
    6. The training loop: resumable, validation every quarter epoch, early stopping.
    7. Gate G: answers from memory, the overfit probe, the report.
    8. Evaluation: the harness, one model version after the other.

Rules this file keeps
    - One normalise(), one build_prompt(), one image loader, one model loader (the
      harness): all imported, none copied.
    - The gold answer is read here and nowhere else. It never enters a prompt.
    - Training and inference use the same model commit, processor settings, chat
      template call and image bytes, because both go through harness.HFBackend.
    - Everything a run writes goes to Google Drive. Answers are model output: not for
      the two HT-1 people before gate-A (D150). This file prints counts only.

The command line is scripts/ft_finetune.py.
"""

import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from normalise import normalise
import harness as H
import images as imglib
import prompts as promptlib

FT_VERSION = "2026-10-07a"
SEED = 42
Stop = H.Stop
say = H.say

# Design 3.1: houses, images and QA pairs per split.
DESIGN_3_1 = {"train": (43, 3296, 13668), "val": (10, 449, 1863), "test": (15, 661, 2745)}
# Design 7.3: values that configs/finetune.yaml must hold.
DESIGN_7_3 = (("condition", "L-BB"), ("lora.r", 16), ("lora.alpha", 32), ("lora.dropout", 0.05),
              ("optimiser.lr", 1e-4), ("optimiser.schedule", "cosine"),
              ("optimiser.warmup_ratio", 0.03), ("optimiser.weight_decay", 0.01),
              ("batch.per_device", 2), ("batch.accumulation", 8), ("max_seq_tokens", 1536),
              ("epochs", 2), ("eval_every_epochs", 0.25), ("patience", 2), ("val_n", 400),
              ("gate_g.n_generate", 20), ("gate_g.loss_window", 50))
DESIGN_SEEDS = {"ft-s42": 42, "ft-s43": 43, "ft-s44": 44}
# Design 12.2 planning hours on an L4.
PLAN_TRAIN_HOURS_3_SEEDS = 10.5
PLAN_EVAL_HOURS = 5.0
VAL_LIST = "FT-VAL400.ids.txt"
DATA_MANIFEST = "ft_data_manifest.json"


# --------------------------------------------------------------------------
# 1. Settings
# --------------------------------------------------------------------------
def _get(d, dotted):
    for part in dotted.split("."):
        d = d[part]
    return d


def load_ft(path):
    """Read configs/finetune.yaml and compare it with Design 7.3."""
    import yaml
    if not os.path.exists(str(path)):
        raise Stop(f"Missing {path}.")
    with open(str(path), "r", encoding="utf-8") as f:
        ft = yaml.safe_load(f)
    for dotted, want in DESIGN_7_3:
        try:
            got = _get(ft, dotted)
        except (KeyError, TypeError):
            raise Stop(f"{path}: '{dotted}' is missing.")
        if got != want:
            raise Stop(f"{path}: {dotted} is {got!r}, Design 7.3 fixes {want!r}. Changing it "
                       "is a design change: ask in the chat first (decision row).")
    runs = ft.get("runs") or {}
    for name, seed in DESIGN_SEEDS.items():
        r = runs.get(name) or {}
        if r.get("profile") != "seed" or r.get("seed") != seed or set(r) - {"profile", "seed"}:
            raise Stop(f"{path}: run {name} must be exactly {{profile: seed, seed: {seed}}} "
                       "(Design 7.3: one setting, seeds 42, 43, 44).")
    dry = runs.get("ft-dry") or {}
    if dry.get("profile") != "dry" or dry.get("n_train") != 100 or dry.get("epochs") != 1:
        raise Stop(f"{path}: run ft-dry must be the Gate G dry run: 100 training examples, "
                   "1 epoch (Design 7.3).")
    for name, r in runs.items():
        if r.get("profile") not in ("dry", "seed"):
            raise Stop(f"{path}: run {name}: profile must be dry or seed.")
    return ft


def run_cfg(ft, name):
    runs = ft.get("runs") or {}
    if name not in runs:
        raise Stop(f"--run: '{name}' is not in configs/finetune.yaml ({', '.join(runs)}).")
    r = runs[name]
    return {"run": name, "profile": r["profile"], "seed": int(r["seed"]),
            "epochs": int(r.get("epochs", ft["epochs"])),
            "n_train": int(r["n_train"]) if r.get("n_train") else None,
            "n_val": int(r.get("n_val", ft["val_n"])),
            "early_stop": bool(r.get("early_stop", True))}


def ft_settings(ft, rc):
    """Everything that defines one training run. Its hash fixes a run folder."""
    return {
        "method": "LoRA (PEFT) on every nn.Linear inside the language-model layers; no LoRA on "
                  "the output head, the embeddings, the vision encoder or the projector; all "
                  "base weights frozen",
        "lora": {"r": ft["lora"]["r"], "alpha": ft["lora"]["alpha"],
                 "dropout": ft["lora"]["dropout"], "bias": "none"},
        "optimiser": {"name": "AdamW", "lr": ft["optimiser"]["lr"],
                      "betas": list(ft["optimiser"]["betas"]), "eps": ft["optimiser"]["eps"],
                      "weight_decay": ft["optimiser"]["weight_decay"],
                      "max_grad_norm": ft["optimiser"]["max_grad_norm"],
                      "schedule": "linear warmup over ceil(warmup_ratio x planned updates), "
                                  "then cosine to zero at the last planned update",
                      "warmup_ratio": ft["optimiser"]["warmup_ratio"]},
        "batch": {"per_device": ft["batch"]["per_device"],
                  "accumulation": ft["batch"]["accumulation"],
                  "how": "a batch of 2 is two single-example passes whose token losses are "
                         "pooled, so no padding token is used; a weight update is the mean "
                         "over its batches"},
        "precision": "base weights bfloat16 and frozen; LoRA weights float32; training and "
                     "validation forward passes under bfloat16 autocast",
        "max_seq_tokens": ft["max_seq_tokens"], "epochs": rc["epochs"],
        "eval_every_epochs": ft["eval_every_epochs"], "patience": ft["patience"],
        "early_stop": rc["early_stop"],
        "validation_loss": "sum of token losses over the validation items / number of loss "
                           "positions",
        "condition": ft["condition"],
        "target": "the gold answer as written (ends stripped; a line break inside becomes one "
                  "space), then the end of turn of the chat template",
        "loss_positions": "the answer and the end of turn; every prompt position is left out",
        "seed": rc["seed"], "n_train": rc["n_train"] or "all", "n_val": rc["n_val"],
        "profile": rc["profile"]}


# --------------------------------------------------------------------------
# 2. Data: the dataset and data/splits.csv, nothing else (D172, D173)
# --------------------------------------------------------------------------
def read_splits(path):
    if not os.path.exists(str(path)):
        raise Stop(f"Missing split file: {path}")
    rows = {}
    with open(str(path), "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            house = (r.get("house_id") or "").strip()
            if not house:
                continue
            if house in rows or (r.get("split") or "").strip() not in ("train", "val", "test"):
                raise Stop(f"{path}: house {house} occurs twice or has no valid split.")
            rows[house] = {"split": r["split"].strip(),
                           "annotator_id": (r.get("annotator_id") or "").strip(),
                           "n_images": int(r.get("n_images") or 0)}
    if not rows:
        raise Stop(f"{path}: no rows.")
    return rows


def read_dataset(path):
    if not os.path.exists(str(path)):
        raise Stop(f"Missing dataset file: {path}")
    with open(str(path), "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data or "qa_pairs" not in data[0]:
        raise Stop(f"{path}: expected the merged dataset, a list of image records with "
                   "'qa_pairs'.")
    return data


def clean_text(value):
    """The text as written, ends stripped. A line break inside becomes one space,
    because build_prompt() refuses a line break inside a question."""
    s = str(value if value is not None else "").strip()
    t = re.sub(r"\s*[\r\n]+\s*", " ", s)
    return t, t != s


def image_file_of(rec):
    s = str(rec.get("image_file") or "").replace("\\", "/").strip()
    if s and "/" not in s:
        s = f"{rec.get('house_id')}/{s}"
    return s


def build_items(dataset, splits):
    """One record per QA pair, in the record format of the probe files. 'answer_bn'
    is the training target text. Nothing is dropped here; 'usable' marks what can be
    trained on."""
    items, seen, images = [], set(), {}
    for rec in dataset:
        house = rec.get("house_id")
        if house not in splits:
            raise Stop(f"House {house} of the dataset is not in data/splits.csv.")
        img = image_file_of(rec)
        if not img or img in images:
            raise Stop(f"Image record {rec.get('image_id')}: no image_file, or it occurs twice.")
        images[img] = house
        for qa in rec.get("qa_pairs") or []:
            qid = qa.get("qa_id")
            if not qid or qid in seen:
                raise Stop(f"qa_id {qid!r} is missing or occurs twice in the dataset.")
            seen.add(qid)
            q, q_fixed = clean_text(qa.get("question"))
            a, a_fixed = clean_text(qa.get("answer"))
            items.append({"qa_id": qid, "image_id": rec.get("image_id"), "house_id": house,
                          "capture_type": rec.get("capture_type"), "qa_type": qa.get("type"),
                          "question_bn": q, "answer_bn": a, "image_file": img,
                          "split": splits[house]["split"],
                          "line_break_fixed": bool(q_fixed or a_fixed),
                          "usable": bool(q and a)})
    return items, images


def ids_sha(ids):
    return H.sha256_hex("\n".join(ids))


def rank(tag, qa_id):
    return H.sha256_hex(f"{SEED}|{tag}|{qa_id}")


def seeded(items, tag, n=None):
    """Seeded rank (seed 42), the project's sampling method (D89)."""
    out = sorted(items, key=lambda r: rank(tag, r["qa_id"]))
    return out if n is None else out[:n]


def define_data(dataset_path, splits_path, expected="design", val_n=400, dry_n=100):
    """Read the dataset and the split file, check them against Design 3.1 and derive
    the fixed lists. Returns a dict; raises Stop when a number is not the design's."""
    expected = DESIGN_3_1 if expected == "design" else expected
    splits = read_splits(splits_path)
    dataset = read_dataset(dataset_path)
    items, images = build_items(dataset, splits)
    stats, bad = {}, []
    for sp in ("train", "val", "test"):
        houses = sorted(h for h, r in splits.items() if r["split"] == sp)
        n_img = sum(1 for h in images.values() if splits[h]["split"] == sp)
        its = [it for it in items if it["split"] == sp]
        stats[sp] = {"houses": len(houses), "images": n_img, "qa": len(its),
                     "usable": sum(1 for it in its if it["usable"]),
                     "line_break_fixed": sum(1 for it in its if it["line_break_fixed"]),
                     "types": dict(sorted(Counter(str(it["qa_type"]) for it in its).items()))}
        if expected and (len(houses), n_img, len(its)) != tuple(expected[sp]):
            bad.append(f"{sp}: {len(houses)} houses, {n_img} images, {len(its)} QA "
                       f"(Design 3.1: {expected[sp][0]}, {expected[sp][1]}, {expected[sp][2]})")
    per_house = Counter(images.values())
    wrong_n = sorted(h for h, r in splits.items() if r["n_images"] != per_house.get(h, 0))
    if wrong_n:
        bad.append(f"n_images of splits.csv differs from the dataset for {len(wrong_n)} houses "
                   f"(first: {wrong_n[0]})")
    if bad:
        raise Stop("The dataset and data/splits.csv do not give the numbers of Design 3.1: "
                   + "; ".join(bad) + ". Nothing was written. Send this line to the FT chat.")
    train = [it for it in items if it["split"] == "train" and it["usable"]]
    val = [it for it in items if it["split"] == "val" and it["usable"]]
    if len(val) < val_n or len(train) < dry_n:
        raise Stop(f"Too few usable items: {len(train)} train, {len(val)} val.")
    val_list = seeded(val, "ft-val400", val_n)
    dry_list = seeded(train, "ft-dry100", dry_n)
    unusable = sorted(it["qa_id"] for it in items if not it["usable"])
    return {
        "splits": splits, "items": items, "images": images, "by_id": {it["qa_id"]: it for it in items},
        "train": train, "val": val_list, "dry": dry_list, "stats": stats,
        "unusable": unusable,
        "hashes": {"dataset_sha256": H.file_sha256(dataset_path),
                   "splits_sha256": H.file_sha256(splits_path),
                   "train_ids_sha256": ids_sha(sorted(it["qa_id"] for it in train)),
                   "val_ids_sha256": ids_sha([it["qa_id"] for it in val_list]),
                   "dry_ids_sha256": ids_sha([it["qa_id"] for it in dry_list])}}


def crosscheck_probe(by_id, probe_items, split="test"):
    """A probe file was drawn from the same dataset: every item must be found here
    with the same image, type and text. This also proves that the dataset fields are
    read the way ASSET-5 read them."""
    c = Counter()
    first = {}
    for p in probe_items:
        it = by_id.get(p["qa_id"])
        if it is None:
            c["missing"] += 1
            first.setdefault("missing", p["qa_id"])
            continue
        checks = (("split", it["split"] == split),
                  ("image_file", it["image_file"] == p["image_file"]),
                  ("qa_type", it["qa_type"] == p["qa_type"]),
                  ("question", it["question_bn"] == clean_text(p["question_bn"])[0]),
                  ("answer", it["answer_bn"] == clean_text(p.get("answer_bn"))[0]))
        for name, ok in checks:
            if not ok:
                c[name] += 1
                first.setdefault(name, p["qa_id"])
        c["ok"] += all(ok for _n, ok in checks)
    if c["ok"] != len(probe_items):
        detail = ", ".join(f"{k} {v} (first {first[k]})" for k, v in c.items() if k != "ok")
        raise Stop(f"The probe file and the dataset disagree on {len(probe_items) - c['ok']} of "
                   f"{len(probe_items)} items: {detail}. Either the dataset changed after the "
                   "probe was drawn (D110 forbids that), or a dataset field is read wrongly. "
                   "Nothing was written. Send this line to the FT chat.")
    return c["ok"]


def data_manifest(d, extra=None):
    m = {"asset": "FT", "script_version": FT_VERSION, "rule": "D172, D173: the dataset and "
         "data/splits.csv only; probes/excluded_items.csv is not applied; seeded rank, seed 42",
         "stats": d["stats"], "n_train": len(d["train"]), "n_val": len(d["val"]),
         "n_dry": len(d["dry"]), "unusable_qa_ids": d["unusable"],
         "line_break_fixed": sum(1 for it in d["items"] if it["line_break_fixed"]),
         **d["hashes"]}
    m.update(extra or {})
    return m


def check_data_manifest(d, path):
    """The data definition written by the data command must be the one found now."""
    old = H.read_json(path, None)
    if old is None:
        raise Stop(f"{path} does not exist. Run the data command first.")
    diff = [k for k, v in d["hashes"].items() if old.get(k) != v]
    if diff:
        raise Stop("The training data are not the ones fixed by the data command: "
                   f"{', '.join(diff)} changed ({path}). The dataset is not edited after "
                   "3 October (D110). If the change is intended, write a decision row and run "
                   "the data command with --force.")
    return old


# --------------------------------------------------------------------------
# Images (D175): the prepared copies of the one image loader, hash-checked
# --------------------------------------------------------------------------
def images_of(items):
    return sorted({it["image_file"] for it in items})


def cache_images(files, drive_root, local_root, manifest, say=say):
    """Copy prepared images from Drive to the local disk once. Every later read is
    still checked against the manifest by images.load_prepared()."""
    copied = 0
    for n, rel in enumerate(files, 1):
        row = manifest.get(rel)
        if row is None:
            raise imglib.ImageError(f"{rel}: not in {imglib.MANIFEST_NAME}. Run the data "
                                    "command with --prepare-images first.")
        dst = os.path.join(str(local_root), rel)
        if os.path.exists(dst) and os.path.getsize(dst) == int(row["bytes"]):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(str(drive_root), rel), dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        copied += 1
        if copied % 500 == 0:
            say(f"    {n} of {len(files)} images checked, {copied} copied to the local disk")
    return copied


# --------------------------------------------------------------------------
# The plan of a run: updates, schedule, validation points
# --------------------------------------------------------------------------
def make_plan(n_items, ft, rc):
    pd, acc = int(ft["batch"]["per_device"]), int(ft["batch"]["accumulation"])
    pairs = math.ceil(n_items / pd)
    per_epoch = math.ceil(pairs / acc)
    total = per_epoch * rc["epochs"]
    frac = float(ft["eval_every_epochs"])
    points = sorted({min(total, math.ceil(k * per_epoch * frac - 1e-9))
                     for k in range(1, int(round(rc["epochs"] / frac)) + 1)} | {total})
    return {"n_items": n_items, "per_device": pd, "accumulation": acc, "batches_per_epoch": pairs,
            "updates_per_epoch": per_epoch, "epochs": rc["epochs"], "total_updates": total,
            "warmup_updates": max(1, math.ceil(float(ft["optimiser"]["warmup_ratio"]) * total)),
            "eval_points": [p for p in points if p > 0], "base_lr": float(ft["optimiser"]["lr"])}


def lr_at(u, plan):
    """Learning rate of weight update number u + 1 (u counts from 0)."""
    w, total, base = plan["warmup_updates"], plan["total_updates"], plan["base_lr"]
    if u < w:
        return base * (u + 1) / w
    return base * 0.5 * (1.0 + math.cos(math.pi * (u - w) / max(1, total - w)))


def order_for_epoch(seed, epoch, items):
    """The data order of one epoch: a function of the seed and the epoch only, so a
    resumed run continues in exactly the order of an uninterrupted one."""
    return sorted(items, key=lambda r: H.sha256_hex(f"{seed}|ft-order|{epoch}|{r['qa_id']}"))


def group_for(u, plan, orders, seed, items):
    """The batches of weight update u (from 0): a list of lists of items."""
    epoch, k = divmod(u, plan["updates_per_epoch"])
    if epoch not in orders:
        orders.clear()
        orders[epoch] = order_for_epoch(seed, epoch, items)
    size = plan["per_device"] * plan["accumulation"]
    chunk = orders[epoch][k * size:(k + 1) * size]
    first_batch = epoch * plan["batches_per_epoch"] + k * plan["accumulation"]
    return epoch, first_batch, [chunk[i:i + plan["per_device"]]
                                for i in range(0, len(chunk), plan["per_device"])]


# --------------------------------------------------------------------------
# 3. The end of an answer, and 4. which layers get LoRA (both without torch)
# --------------------------------------------------------------------------
def ending_rule(tail_ids, eos_ids, tok_eos_id):
    """What follows the gold answer in a training sequence.

    tail_ids: the tokens the chat template writes after an assistant answer.
    eos_ids:  the tokens on which generation stops (the harness's greedy settings).
    The model must learn to stop, so the sequence ends with a token that stops
    generation: the template's own tokens up to the first such token, or, if the
    template writes none, the template's tokens and then the stop token."""
    tail = [int(t) for t in tail_ids]
    eos = {int(t) for t in eos_ids}
    if not eos:
        raise Stop("The generation settings name no token that stops generation.")
    for k, t in enumerate(tail):
        if t in eos:
            return tail[:k + 1], "the template's end of turn, up to the first token that stops generation"
    if tok_eos_id is not None and int(tok_eos_id) in eos:
        extra = int(tok_eos_id)
    elif len(eos) == 1:
        extra = next(iter(eos))
    else:
        raise Stop("The chat template writes no stop token after an answer, and the "
                   f"generation settings name several ({sorted(eos)}). Send this line to "
                   "the FT chat. Nothing was run.")
    return tail + [extra], "the template's end of turn, then the token that stops generation"


_VISION = ("visual", "vision_tower", "vision_model", "vision_encoder", "image_encoder", "vit")
_AUDIO = ("audio_tower", "audio_model", "audio_encoder")
_PROJECTOR = ("multi_modal_projector", "mm_projector", "merger", "connector", "aligner",
              "embed_vision", "embed_audio")
_HEAD = ("lm_head", "embed_out", "output_projection")
_STACK = re.compile(r"^(?P<prefix>(?:.*?\.)?(?:layers|layer|h|blocks|block))\.(?P<idx>\d+)\.(?P<rest>.+)$")
_LANG_HINT = ("language_model", "text_model", "decoder", "llm")


def classify_linear(name):
    """vision, audio, projector, head, or '' (decided by the layer stack afterwards)."""
    segs = name.split(".")
    if any(s in _VISION or s.startswith("vision") or s.startswith("visual") for s in segs):
        return "vision"
    if any(s in _AUDIO or s.startswith("audio") for s in segs):
        return "audio"
    if any(s in _PROJECTOR or "projector" in s for s in segs):
        return "projector"
    if any(s in _HEAD for s in segs):
        return "head"
    return ""


def select_targets(linears, n_layers_cfg):
    """linears: (name, in_features, out_features) of every nn.Linear of the base model.
    Returns the names that get LoRA: every linear layer inside the layers of the
    language model, and a table of everything else for the record."""
    comp, rest = {}, []
    for name, _i, _o in linears:
        c = classify_linear(name)
        comp[name] = c
        if not c:
            rest.append(name)
    stacks = {}
    for name in rest:
        mt = _STACK.match(name)
        if mt:
            stacks.setdefault(mt.group("prefix"), {}).setdefault(int(mt.group("idx")), []).append(name)
    cands = sorted(stacks)
    if n_layers_cfg:
        cands = [p for p in cands if sorted(stacks[p]) == list(range(int(n_layers_cfg)))]
    else:
        most = max((len(v) for v in stacks.values()), default=0)
        cands = [p for p in cands if len(stacks[p]) == most]
    if len(cands) > 1:
        hinted = [p for p in cands if any(h in p for h in _LANG_HINT)]
        cands = hinted if len(hinted) == 1 else cands
    if len(cands) != 1:
        seen = {p: len(v) for p, v in stacks.items()}
        raise Stop("The layers of the language model could not be identified. Layer stacks "
                   f"found among the linear layers (name: number of layers): {seen}; the "
                   f"model config names {n_layers_cfg} language layers. Send this line to the "
                   "FT chat. Nothing was run.")
    prefix = cands[0]
    targets = sorted(n for names in stacks[prefix].values() for n in names)
    for n in targets:
        comp[n] = "language"
    for n in rest:
        comp.setdefault(n, "")
        if not comp[n]:
            comp[n] = "other"
    size = {name: i * o for name, i, o in linears}
    table = {}
    for name, c in comp.items():
        t = table.setdefault(c, {"layers": 0, "weights": 0})
        t["layers"] += 1
        t["weights"] += size[name]
    leaves = Counter(_STACK.match(n).group("rest") for n in targets)
    per_layer = Counter(len(v) for v in stacks[prefix].values())
    return {"targets": targets, "stack_prefix": prefix, "n_layers": len(stacks[prefix]),
            "n_layers_config": n_layers_cfg, "table": dict(sorted(table.items())),
            "leaves": dict(sorted(leaves.items())),
            "linear_per_layer": {str(k): v for k, v in sorted(per_layer.items())},
            "other": sorted(n for n, c in comp.items() if c == "other")[:40],
            "targets_sha256": ids_sha(targets)}


# --------------------------------------------------------------------------
# 5. Engines
# --------------------------------------------------------------------------
class TorchEngine:
    """The real engine. The base model, its processor, its chat template call and its
    greedy generation are those of harness.HFBackend; PEFT adds the LoRA weights."""

    name = "torch"

    def __init__(self, models, ft, rc, prompts, cond, img_root, img_manifest, run_dir,
                 allow_a100=False, grad_checkpointing=False):
        try:
            import torch
        except ImportError as exc:
            raise Stop(f"torch is needed for training ({exc}). Run this in Colab on an L4.")
        self.torch, self.ft, self.rc = torch, ft, rc
        self.prompts, self.cond = prompts, cond
        self.img_root, self.img_manifest = img_root, img_manifest
        self.m = m = models[ft["base_model"]]
        if m.get("backend") != "hf" or m.get("adapter"):
            raise Stop(f"configs/finetune.yaml: base_model '{ft['base_model']}' must be an open "
                       "base model of configs/models.yaml, not an adapter entry.")
        if not torch.cuda.is_available():
            raise Stop("No GPU. In Colab choose Runtime, Change runtime type, L4 GPU.")
        gpu = torch.cuda.get_device_name(0)
        if "T4" in gpu:
            raise Stop("Design 5.3: fine-tuning never runs on a T4.")
        a100 = "A100" in gpu
        if "L4" not in gpu and not (a100 and allow_a100 and rc["profile"] == "seed"):
            raise Stop(f"The GPU is '{gpu}'. The dry run needs an L4, because its answers are "
                       "compared with the harness on an L4. A seed may train on an A100 with "
                       "--allow-a100 (Design 5.3 allows an A100 for fine-tuning only).")
        ctx = H.Ctx(model=m, allow_env_change=a100, out_dir=Path(run_dir), prompts=prompts)
        self.backend = H.HFBackend(ctx)            # the same loader as inference
        self.model = self.backend.model
        self.tok = self.backend.processor.tokenizer
        self.grad_checkpointing = bool(grad_checkpointing)
        self.use_cache_arg = True
        self.extended_keys = set()
        self.best_mem = None
        self.opt = self.peft_model = self.params = None
        self._ending()

    # ---- environment -------------------------------------------------------
    def env(self):
        e = dict(self.backend.env())
        try:
            import peft
            e["peft"] = peft.__version__
        except ImportError:
            e["peft"] = ""
        e["python"] = sys.version.split()[0]
        return e

    def peak_gb(self):
        return round(self.torch.cuda.max_memory_allocated() / 2 ** 30, 2)

    # ---- the end of an answer ----------------------------------------------
    def _ending(self):
        proc, sentinel = self.backend.processor, "ZQXJ9KV"
        think = bool(self.m["enable_thinking"])
        user = {"role": "user", "content": [{"type": "text", "text": "x"}]}
        text, err = None, ""
        for content in ([{"type": "text", "text": sentinel}], sentinel):
            try:
                text = proc.apply_chat_template(
                    [user, {"role": "assistant", "content": content}], tokenize=False,
                    add_generation_prompt=False, enable_thinking=think)
            except Exception as exc:
                text, err = None, f"{type(exc).__name__}: {str(exc)[:200]}"
                continue
            text = text[0] if isinstance(text, (list, tuple)) else text
            if isinstance(text, str) and sentinel in text:
                break
            text = None
        if text is None:
            raise Stop("The chat template could not write a finished assistant turn, so the "
                       f"end of an answer is unknown ({err or 'no text'}). Send this line to "
                       "the FT chat. Nothing was run.")
        head, tail = text.split(sentinel, 1)
        gen = proc.apply_chat_template([user], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=think)
        gen = gen[0] if isinstance(gen, (list, tuple)) else gen
        tail_ids = self.tok(tail, add_special_tokens=False)["input_ids"] if tail else []
        ids, rule = ending_rule(tail_ids, self.backend.eos, self.tok.eos_token_id)
        self.ending_ids = ids
        self.ending = {
            "ids": ids, "tokens": [str(t) for t in self.tok.convert_ids_to_tokens(ids)],
            "text": self.tok.decode(ids, skip_special_tokens=False), "rule": rule,
            "template_tail": tail, "stop_token_ids": sorted(self.backend.eos),
            "generation_prompt_tail": gen[-80:], "finished_turn_head_tail": head[-80:],
            "generation_prompt_equals_finished_turn_head": gen == head}

    # ---- one example ---------------------------------------------------------
    def prompt_and_image(self, item):
        p = promptlib.build_prompt(self.cond, item, self.prompts)
        data, sha = imglib.load_prepared(self.img_root, item["image_file"], self.img_manifest)
        return p, imglib.decode_rgb(data), sha

    def build(self, item, keep_prompt=False):
        """The training sequence of one item: the inference prompt exactly as the
        harness builds it, then the gold answer and the end of turn."""
        torch, t0 = self.torch, time.time()
        p, image, sha = self.prompt_and_image(item)
        enc = self.backend._inputs(p["text"], image)
        ids = enc["input_ids"]
        n_prompt = int(ids.shape[-1])
        answer = self.tok(item["answer_bn"], add_special_tokens=False)["input_ids"]
        tail = [int(t) for t in answer] + list(self.ending_ids)
        base = {"qa_id": item["qa_id"], "n_prompt": n_prompt, "n_answer": len(answer),
                "n_ending": len(self.ending_ids), "n_labels": len(tail),
                "n_tokens": n_prompt + len(tail), "first_label": n_prompt,
                "image_tokens": self.backend._count_image_tokens(enc),
                "prompt_sha256": p["sha256"], "image_sha256": sha}
        if not answer:
            return dict(base, skip="empty_target")
        if base["n_tokens"] > int(self.ft["max_seq_tokens"]):
            return dict(base, skip="too_long")
        add = torch.tensor([tail], dtype=ids.dtype)
        inputs = {}
        for k, v in enc.items():
            if torch.is_tensor(v) and v.dim() == 2 and tuple(v.shape) == tuple(ids.shape) \
                    and not v.is_floating_point():
                if k == "input_ids":
                    inputs[k] = torch.cat([v, add], dim=1)
                elif k == "attention_mask":
                    inputs[k] = torch.cat([v, torch.ones_like(add).to(v.dtype)], dim=1)
                else:               # another per-token tensor: the new positions are text
                    inputs[k] = torch.cat([v, torch.zeros_like(add).to(v.dtype)], dim=1)
                    self.extended_keys.add(k)
            else:
                inputs[k] = v
        ex = dict(base, inputs=inputs, prompt_text=p["text"], build_s=round(time.time() - t0, 4))
        if keep_prompt:
            ex["prompt_inputs"] = {k: v for k, v in enc.items()}
            ex["image"] = image
        return ex

    def _to_device(self, inputs):
        dev = self.model.device
        return {k: (v.to(dev) if self.torch.is_tensor(v) else v) for k, v in inputs.items()}

    def _forward(self, inputs):
        if self.use_cache_arg:
            try:
                return self.model(**inputs, use_cache=False)
            except TypeError as exc:
                if "use_cache" not in str(exc):
                    raise
                self.use_cache_arg = False
        return self.model(**inputs)

    def _nll(self, ex, inputs):
        """Sum of the token losses on the answer positions. The prompt positions are
        never read, so they cannot enter the loss."""
        torch = self.torch
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self._forward(inputs)
        if out.logits.shape[1] != inputs["input_ids"].shape[1]:
            raise Stop(f"{ex['qa_id']}: the model returned {out.logits.shape[1]} positions for "
                       f"{inputs['input_ids'].shape[1]} input tokens. Send this line to the "
                       "FT chat.")
        a, n = ex["first_label"], ex["n_labels"]
        logits = out.logits[0, a - 1:a - 1 + n].float()
        target = inputs["input_ids"][0, a:a + n]
        return torch.nn.functional.cross_entropy(logits, target, reduction="sum")

    def _oom(self, exc):
        if "out of memory" in str(exc).lower():
            raise Stop("CUDA out of memory in a training pass. Restart the runtime and run "
                       "the same command with --grad-checkpointing (slower, less memory); "
                       "what was trained is kept. If it happens again, send the line to the "
                       f"FT chat. ({str(exc)[:160]})")
        raise exc

    def train_pass(self, ex, weight):
        torch = self.torch
        try:
            nll = self._nll(ex, self._to_device(ex["inputs"]))
            if not bool(torch.isfinite(nll)):
                raise Stop(f"{ex['qa_id']}: the loss is not a finite number. Nothing was "
                           "updated. Send this line to the FT chat.")
            (nll * float(weight)).backward()
        except RuntimeError as exc:
            self._oom(exc)
        return float(nll.detach())

    def eval_pass(self, ex):
        torch = self.torch
        with torch.no_grad():
            return float(self._nll(ex, self._to_device(ex["inputs"])))

    def step(self, lr):
        torch = self.torch
        for g in self.opt.param_groups:
            g["lr"] = float(lr)
        norm = torch.nn.utils.clip_grad_norm_(self.params, float(self.ft["optimiser"]["max_grad_norm"]))
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return float(norm)

    def sync(self):
        self.torch.cuda.synchronize()

    def reseed(self, seed, update):
        """Dropout draws depend on the seed and the update number only."""
        self.torch.manual_seed((int(seed) * 1000003 + int(update)) % (2 ** 63 - 1))

    def set_train(self, flag):
        self.model.train(bool(flag))

    # ---- LoRA ------------------------------------------------------------------
    def attach_lora(self, seed):
        torch = self.torch
        try:
            import peft
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise Stop(f"peft is needed for training ({exc}). Run the install cell of the "
                       "notebook first.")
        cfgm = self.model.config
        tc = getattr(cfgm, "text_config", None)
        n_layers = (tc.get("num_hidden_layers") if isinstance(tc, dict)
                    else getattr(tc, "num_hidden_layers", None)) \
            or getattr(cfgm, "num_hidden_layers", None)
        linears = [(n, int(mod.in_features), int(mod.out_features))
                   for n, mod in self.model.named_modules() if isinstance(mod, torch.nn.Linear)]
        self.sel = sel = select_targets(linears, n_layers)
        lora = self.ft["lora"]
        config = LoraConfig(r=int(lora["r"]), lora_alpha=int(lora["alpha"]),
                            lora_dropout=float(lora["dropout"]), bias="none",
                            target_modules=list(sel["targets"]))
        torch.manual_seed(int(seed))               # the seed sets the LoRA initialisation
        torch.cuda.manual_seed_all(int(seed))
        self.peft_model = get_peft_model(self.model, config, autocast_adapter_dtype=True)
        got = sorted(n for n, mod in self.model.named_modules()
                     if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"))
        if got != sel["targets"]:
            extra, lost = sorted(set(got) - set(sel["targets"])), sorted(set(sel["targets"]) - set(got))
            raise Stop(f"LoRA was put on {len(got)} layers, the rule names {len(sel['targets'])} "
                       f"(not wanted: {extra[:3]}; not reached: {lost[:3]}). Send this line "
                       "to the FT chat. Nothing was trained.")
        trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        stray = [n for n, _p in trainable if ".lora_A." not in n and ".lora_B." not in n]
        dtypes = sorted({str(p.dtype) for _n, p in trainable})
        if stray or dtypes != ["torch.float32"]:
            raise Stop(f"After PEFT, {len(stray)} weights outside LoRA are trainable (first: "
                       f"{stray[:2]}) and the LoRA weights are {dtypes} (float32 expected). "
                       "Send this line to the FT chat. Nothing was trained.")
        self.params = [p for _n, p in trainable]
        self.lora = {"peft": peft.__version__, "n_modules": len(got),
                     "n_tensors": len(trainable),
                     "trainable_params": int(sum(p.numel() for p in self.params)),
                     "total_params": int(sum(p.numel() for p in self.model.parameters())),
                     "param_dtype": dtypes[0], "targets": sel["targets"],
                     "targets_sha256": sel["targets_sha256"], "stack_prefix": sel["stack_prefix"],
                     "n_layers": sel["n_layers"], "n_layers_config": sel["n_layers_config"],
                     "leaves": sel["leaves"], "linear_per_layer": sel["linear_per_layer"],
                     "components": sel["table"], "other_linear_not_adapted": sel["other"]}
        if self.grad_checkpointing:
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            except Exception as exc:
                raise Stop(f"--grad-checkpointing could not be switched on "
                           f"({type(exc).__name__}: {str(exc)[:160]}). Send this line to the "
                           "FT chat.")
        self.new_optimizer()
        self.model.train()
        return self.lora

    def new_optimizer(self):
        o = self.ft["optimiser"]
        self.opt = self.torch.optim.AdamW(self.params, lr=float(o["lr"]),
                                          betas=(float(o["betas"][0]), float(o["betas"][1])),
                                          eps=float(o["eps"]), weight_decay=float(o["weight_decay"]))

    def reset_lora(self, seed):
        """Fresh LoRA weights (A random, B zero), for the overfit probe only."""
        torch = self.torch
        torch.manual_seed(int(seed))
        with torch.no_grad():
            for _n, mod in self.model.named_modules():
                if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
                    for key in mod.lora_A.keys():
                        torch.nn.init.kaiming_uniform_(mod.lora_A[key].weight, a=math.sqrt(5))
                        torch.nn.init.zeros_(mod.lora_B[key].weight)
        self.new_optimizer()

    def check_backward(self, ex):
        """One training pass on one example before anything long starts."""
        torch = self.torch
        torch.cuda.reset_peak_memory_stats()
        self.set_train(True)
        t0 = time.time()
        nll = self.train_pass(ex, 1.0 / ex["n_labels"])
        self.sync()
        with_grad = sum(1 for n, p in self.model.named_parameters()
                        if ".lora_B." in n and p.grad is not None and float(p.grad.abs().sum()) > 0)
        n_b = sum(1 for n, _p in self.model.named_parameters() if ".lora_B." in n)
        self.opt.zero_grad(set_to_none=True)
        if not with_grad:
            raise Stop("No gradient reaches the LoRA weights. Send this line to the FT chat. "
                       "Nothing was trained.")
        return {"loss": round(nll / ex["n_labels"], 4), "lora_B_with_gradient": with_grad,
                "lora_B_tensors": n_b, "seconds": round(time.time() - t0, 2),
                "peak_gpu_gb": self.peak_gb()}

    # ---- state -------------------------------------------------------------------
    def trainable_state(self):
        return {n: p.detach().to("cpu").clone()
                for n, p in self.model.named_parameters() if p.requires_grad}

    def load_trainable(self, state):
        torch = self.torch
        named = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        if set(named) != set(state):
            raise Stop("The saved LoRA weights do not fit this model (other names). The run "
                       "folder holds a run made differently.")
        with torch.no_grad():
            for n, p in named.items():
                p.copy_(state[n].to(p.device, dtype=p.dtype))

    def keep_best(self, path):
        """Dry run only: the best LoRA weights stay in memory for the Gate G answers,
        and go to a state file in case the session ends before they are used."""
        self.best_mem = self.trainable_state()
        self.torch.save(self.best_mem, str(path) + ".tmp")
        os.replace(str(path) + ".tmp", str(path))

    def use_best(self, path):
        """Returns where the weights came from: 'memory' or 'state file'."""
        if self.best_mem is not None:
            self.load_trainable(self.best_mem)
            return "memory"
        try:
            state = self.torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            state = self.torch.load(str(path), map_location="cpu")
        self.load_trainable(state)
        return "state file"

    def save_state(self, path, update, progress):
        torch = self.torch
        tmp = str(path) + ".tmp"
        torch.save({"update": int(update), "lora": self.trainable_state(),
                    "opt": self.opt.state_dict(), "progress": json.dumps(progress)}, tmp)
        os.replace(tmp, str(path))

    def load_state(self, path):
        torch = self.torch
        try:
            d = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            d = torch.load(str(path), map_location="cpu")
        self.load_trainable(d["lora"])
        self.opt.load_state_dict(d["opt"])
        return int(d["update"]), json.loads(d["progress"])

    def save_adapter(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.peft_model.save_pretrained(str(directory), safe_serialization=True,
                                        save_embedding_layers=False)
        weights = directory / H.ADAPTER_FILE
        if not weights.exists():
            raise Stop(f"PEFT did not write {weights}. Send this line to the FT chat.")
        return H.file_sha256(weights), os.path.getsize(weights)

    # ---- generation: the harness's own function ------------------------------------
    def generate(self, text, image):
        self.model.eval()
        return self.backend.generate(text, image)

    def probe_base(self, items):
        """Before LoRA is attached: (1) teacher forcing is aligned with generation;
        (2) the base model ends its own answers the way the training target ends;
        (3) the frozen base model computes the same in training mode as in inference
        mode, so LoRA dropout is the only random part of training."""
        torch = self.torch
        out = {"n": 0, "forward_prompt_equals_forward_full": 0, "forward_equals_generate": 0,
               "stopped_by_itself": 0, "ends_like_target": 0, "tails": [],
               "train_mode_same_next_token": 0, "train_mode_max_logit_diff": 0.0}
        self.model.eval()
        for it in items:
            ex = self.build(it, keep_prompt=True)
            if ex.get("skip"):
                continue
            out["n"] += 1
            full, prompt = self._to_device(ex["inputs"]), self._to_device(ex["prompt_inputs"])
            n = ex["first_label"]
            with torch.inference_mode():
                l_eval = self._forward(prompt).logits[0, -1].float()
                a_prompt = int(l_eval.argmax())
                a_full = int(self._forward(full).logits[0, n - 1].argmax())
                gen = self.model.generate(**prompt, generation_config=self.backend.gc,
                                          max_new_tokens=1)
                self.model.train()
                l_train = self._forward(prompt).logits[0, -1].float()
                self.model.eval()
            a_gen = int(gen[0, n])
            out["forward_prompt_equals_forward_full"] += a_prompt == a_full
            out["forward_equals_generate"] += a_prompt == a_gen
            out["train_mode_same_next_token"] += int(l_train.argmax()) == a_prompt
            out["train_mode_max_logit_diff"] = round(max(out["train_mode_max_logit_diff"],
                                                         float((l_train - l_eval).abs().max())), 5)
            g = self.backend.generate(ex["prompt_text"], ex["image"])
            if g["finish_reason"] == "stop":
                out["stopped_by_itself"] += 1
                if g["raw_text"].endswith(self.ending["text"]):
                    out["ends_like_target"] += 1
                else:           # show the shape of the ending, never the answer itself
                    ids = self.tok(g["raw_text"], add_special_tokens=False)["input_ids"][-4:]
                    special = set(self.tok.all_special_ids) | set(self.backend.eos)
                    out["tails"].append([str(self.tok.convert_ids_to_tokens(i))
                                         if i in special or not self.tok.decode([i]).strip()
                                         else "(text)" for i in ids])
        return out


class FakeEngine:
    """A stand-in without torch, for the self checks. Four numbers stand for the LoRA
    weights; each item pulls them towards a point derived from its qa_id. It shares
    the whole training loop, the logs and the files with the real engine."""

    name = "fake"

    def __init__(self, ft, rc, prompts, cond, val_curve=None, fail_at_update=None, long_ids=()):
        self.ft, self.rc, self.prompts, self.cond = ft, rc, prompts, cond
        self.theta = [0.0, 0.0, 0.0, 0.0]
        self.grad = [0.0, 0.0, 0.0, 0.0]
        self.training, self.best_mem = True, None
        self.val_curve, self.fail_at_update, self.long_ids = val_curve, fail_at_update, set(long_ids)
        self.updates_done, self.memorised = 0, {}
        self.ending_ids = [9]
        self.ending = {"ids": [9], "tokens": ["<end>"], "text": "<end>", "rule": "stand-in",
                       "template_tail": "<end>\n", "stop_token_ids": [9],
                       "generation_prompt_tail": "", "finished_turn_head_tail": "",
                       "generation_prompt_equals_finished_turn_head": True}
        self.extended_keys = set()
        self.lora = None

    def env(self):
        return {"gpu": "stand-in", "torch": "none", "transformers": "none", "peft": "stand-in",
                "model_revision": "stand-in", "python": sys.version.split()[0]}

    def peak_gb(self):
        return 0.0

    @staticmethod
    def _point(qa_id):
        h = H.sha256_hex("fake|" + qa_id)
        return [int(h[i * 4:i * 4 + 4], 16) / 32768.0 - 1.0 for i in range(4)]

    def prompt_and_image(self, item):
        p = promptlib.build_prompt(self.cond, item, self.prompts)
        return p, None, "0" * 64

    def build(self, item, keep_prompt=False):
        p, image, sha = self.prompt_and_image(item)
        n_prompt = 40 + len(p["text"]) // 3
        n_answer = max(1, len(item["answer_bn"].split()))
        base = {"qa_id": item["qa_id"], "n_prompt": n_prompt, "n_answer": n_answer, "n_ending": 1,
                "n_labels": n_answer + 1, "n_tokens": n_prompt + n_answer + 1,
                "first_label": n_prompt, "image_tokens": 768, "prompt_sha256": p["sha256"],
                "image_sha256": sha}
        if item["qa_id"] in self.long_ids:
            return dict(base, skip="too_long", n_tokens=int(self.ft["max_seq_tokens"]) + 1)
        ex = dict(base, inputs={}, prompt_text=p["text"], build_s=0.0, point=self._point(item["qa_id"]))
        if keep_prompt:
            ex["prompt_inputs"], ex["image"] = {}, None
        return ex

    def _nll(self, ex):
        d = sum((t - q) ** 2 for t, q in zip(self.theta, ex["point"]))
        return ex["n_labels"] * (0.8 + 0.5 * d)

    def train_pass(self, ex, weight):
        for i in range(4):
            self.grad[i] += weight * ex["n_labels"] * (self.theta[i] - ex["point"][i])
        return self._nll(ex)

    def eval_pass(self, ex):
        return self._nll(ex)

    def step(self, lr):
        if self.fail_at_update is not None and self.updates_done + 1 == self.fail_at_update:
            self.fail_at_update = None
            raise RuntimeError("stand-in crash")
        norm = math.sqrt(sum(g * g for g in self.grad))
        for i in range(4):
            self.theta[i] -= lr * 1500.0 * self.grad[i]
        self.grad = [0.0, 0.0, 0.0, 0.0]
        self.updates_done += 1
        return norm

    def sync(self):
        pass

    def reseed(self, seed, update):
        pass

    def set_train(self, flag):
        self.training = bool(flag)

    def attach_lora(self, seed):
        self.lora = {"peft": "stand-in", "n_modules": 4, "n_tensors": 8, "trainable_params": 4,
                     "total_params": 1000, "param_dtype": "torch.float32", "targets": ["a", "b"],
                     "targets_sha256": ids_sha(["a", "b"]), "stack_prefix": "m.layers",
                     "n_layers": 2, "n_layers_config": 2, "leaves": {"x": 4},
                     "linear_per_layer": {"2": 2}, "components": {}, "other_linear_not_adapted": []}
        return self.lora

    def new_optimizer(self):
        self.grad = [0.0, 0.0, 0.0, 0.0]

    def reset_lora(self, seed):
        self.theta = [0.0, 0.0, 0.0, 0.0]
        self.new_optimizer()

    def check_backward(self, ex):
        return {"loss": round(self._nll(ex) / ex["n_labels"], 4), "lora_B_with_gradient": 4,
                "lora_B_tensors": 4, "seconds": 0.0, "peak_gpu_gb": 0.0}

    def trainable_state(self):
        return {"theta": list(self.theta)}

    def load_trainable(self, state):
        self.theta = list(state["theta"])

    def keep_best(self, path):
        self.best_mem = self.trainable_state()
        H.write_json(path, self.best_mem)

    def use_best(self, path):
        if self.best_mem is not None:
            self.load_trainable(self.best_mem)
            return "memory"
        self.load_trainable(H.read_json(path, {}))
        return "state file"

    def save_state(self, path, update, progress):
        H.write_json(path, {"update": int(update), "lora": self.trainable_state(),
                            "updates_done": self.updates_done, "progress": json.dumps(progress)})

    def load_state(self, path):
        d = H.read_json(path, {})
        self.load_trainable(d["lora"])
        self.updates_done = d["updates_done"]
        return int(d["update"]), json.loads(d["progress"])

    def save_adapter(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        weights = directory / H.ADAPTER_FILE
        weights.write_text(json.dumps({"theta": self.theta}), encoding="utf-8")
        (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
        return H.file_sha256(weights), os.path.getsize(weights)

    def memorise(self, pairs):
        """Stand-in for overfitting: the probe prompts now give back their gold answer."""
        self.memorised = {H.sha256_hex(ex["prompt_text"]): it["answer_bn"] for it, ex in pairs}

    def generate(self, text, image):
        key = H.sha256_hex(text)
        answer = self.memorised.get(key, "উত্তর " + str(int(key[:2], 16) % 7))
        return {"answer_text": answer, "raw_text": answer + "<end>", "n_new_tokens": 3,
                "n_input_tokens": 800, "image_tokens": 768, "finish_reason": "stop",
                "thinking_leak": False, "elapsed_s": 0.01}

    def probe_base(self, items):
        n = len(items)
        return {"n": n, "forward_prompt_equals_forward_full": n, "forward_equals_generate": n,
                "stopped_by_itself": n, "ends_like_target": n, "tails": [],
                "train_mode_same_next_token": n, "train_mode_max_logit_diff": 0.0}


# --------------------------------------------------------------------------
# 6. The training loop (the same for both engines)
# --------------------------------------------------------------------------
class Log:
    """A JSON-lines file written in blocks, so Drive is not hit once per line."""

    def __init__(self, path):
        self.path, self.buf = str(path), []
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)

    def add(self, rec):
        self.buf.append(json.dumps(rec, ensure_ascii=False))

    def flush(self):
        if not self.buf:
            return
        torn = False
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            with open(self.path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                torn = f.read(1) != b"\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(("\n" if torn else "") + "\n".join(self.buf) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.buf = []


def read_log(path, kind, key):
    """Lines of one kind; a later line for the same key replaces an earlier one (a
    resumed session repeats the steps after its last checkpoint)."""
    recs, _bad = H.read_jsonl(path)
    out = {}
    for r in recs:
        if r.get("kind") == kind:
            out[r[key]] = r
    return [out[k] for k in sorted(out)]


def new_progress():
    return {"update": 0, "evals": [], "best": None, "bad_evals": 0, "stopped": None,
            "would_stop_at": None, "skipped": {}, "val_skipped": [], "train_seconds": 0.0,
            "sessions": 0, "finished": False}


def evaluate(engine, val_items):
    engine.set_train(False)
    t0, nll, n_tok, n_items, skipped = time.time(), 0.0, 0, 0, []
    for it in val_items:
        ex = engine.build(it)
        if ex.get("skip"):
            skipped.append(it["qa_id"])
            continue
        nll += engine.eval_pass(ex)
        n_tok += ex["n_labels"]
        n_items += 1
    engine.sync()
    engine.set_train(True)
    if not n_tok:
        raise Stop("No validation item could be used.")
    return {"val_loss": nll / n_tok, "n_items": n_items, "n_tokens": n_tok,
            "seconds": round(time.time() - t0, 2), "skipped": skipped}


def train_run(engine, T):
    """T: plan, rc, ft, train_items, val_items, run_dir, prefetch, stop_after_s, say.
    Returns the progress dict. Can be started again at any time."""
    plan, rc, run_dir = T.plan, T.rc, Path(T.run_dir)
    state_dir = run_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    ext = "json" if engine.name == "fake" else "pt"
    state_path, best_state = state_dir / f"last.{ext}", state_dir / f"best_lora.{ext}"
    tlog, elog = Log(run_dir / "train_log.jsonl"), Log(run_dir / "eval_log.jsonl")
    progress = new_progress()
    if state_path.exists():
        _u, progress = engine.load_state(state_path)
        T.say(f"  resuming after weight update {progress['update']} of {plan['total_updates']}")
    for d in run_dir.glob("best_update*"):          # left over by a session that died
        if not progress["best"] or d.name != progress["best"]["dir"]:
            shutil.rmtree(d, ignore_errors=True)
    if progress["best"] and not (run_dir / progress["best"]["dir"] / H.ADAPTER_FILE).exists() \
            and not progress["finished"]:
        raise Stop(f"{run_dir}: the best adapter named by the saved state is missing. Send "
                   "this line to the FT chat.")
    progress["sessions"] += 1
    session = f"{H.utc_now()}#{progress['sessions']}"
    orders, t_session = {}, time.time()
    pool = ThreadPoolExecutor(max_workers=1) if T.prefetch else None
    ckpt_every = int(T.ft["checkpoint_every_updates"])

    def build_group(u):
        epoch, first_batch, batches = group_for(u, plan, orders, rc["seed"], T.train_items)
        return epoch, first_batch, [[engine.build(it) for it in b] for b in batches]

    def save():
        tlog.flush()
        elog.flush()
        engine.save_state(state_path, progress["update"], progress)

    u = progress["update"]
    pending = pool.submit(build_group, u) if pool and u < plan["total_updates"] else None
    engine.set_train(True)
    stop_now = ""
    while u < plan["total_updates"] and not progress["stopped"]:
        t_update = time.time()
        epoch, first_batch, built = pending.result() if pending else build_group(u)
        pending = pool.submit(build_group, u + 1) if pool and u + 1 < plan["total_updates"] else None
        usable = []
        for b in built:
            good = [ex for ex in b if not ex.get("skip")]
            for ex in b:
                if ex.get("skip"):
                    progress["skipped"][ex["qa_id"]] = ex["skip"]
            usable.append(good)
        n_batches = sum(1 for b in usable if b)
        lr = lr_at(u, plan)
        engine.reseed(rc["seed"], u)
        losses = []
        for j, b in enumerate(usable):
            if not b:
                continue
            n_lab = sum(ex["n_labels"] for ex in b)
            t0 = time.time()
            nll = sum(engine.train_pass(ex, 1.0 / (n_lab * n_batches)) for ex in b)
            engine.sync()
            losses.append(nll / n_lab)
            tlog.add({"kind": "batch", "session": session, "batch": first_batch + j, "update": u + 1,
                      "epoch": epoch, "qa_ids": [ex["qa_id"] for ex in b],
                      "n_prompt": [ex["n_prompt"] for ex in b], "n_answer": [ex["n_answer"] for ex in b],
                      "n_ending": [ex["n_ending"] for ex in b], "n_labels": [ex["n_labels"] for ex in b],
                      "n_tokens": [ex["n_tokens"] for ex in b],
                      "image_tokens": [ex["image_tokens"] for ex in b],
                      "loss": round(nll / n_lab, 6), "lr": lr,
                      "gpu_s": round(time.time() - t0, 4),
                      "build_s": round(sum(ex["build_s"] for ex in b), 4)})
        grad_norm = engine.step(lr) if n_batches else 0.0
        engine.sync()
        u += 1
        progress["update"] = u
        wall = time.time() - t_update
        progress["train_seconds"] += wall
        tlog.add({"kind": "update", "session": session, "update": u, "epoch": epoch, "lr": lr,
                  "loss": round(sum(losses) / len(losses), 6) if losses else None,
                  "batches": n_batches, "examples": sum(len(b) for b in usable),
                  "grad_norm": round(grad_norm, 5), "wall_s": round(wall, 3),
                  "peak_gpu_gb": engine.peak_gb(), "ts_utc": H.utc_now()})
        if u % 10 == 0 or u == plan["total_updates"] or u <= 3:
            T.say(f"    update {u:>5} of {plan['total_updates']}   loss {losses and sum(losses) / len(losses) or 0:.4f}"
                  f"   lr {lr:.2e}   {wall / max(1, sum(len(b) for b in usable)):.2f} s per example")
        did_eval = u in plan["eval_points"]
        if did_eval:
            if pending:
                pending.result()          # the processor is never used by two threads at once
            ev = evaluate(engine, T.val_items)
            forced = getattr(engine, "val_curve", None)
            if forced:
                ev["val_loss"] = forced[min(len(progress["evals"]), len(forced) - 1)]
            improved = progress["best"] is None or ev["val_loss"] < progress["best"]["val_loss"]
            if improved:
                name = f"best_update{u:06d}"
                sha, size = engine.save_adapter(run_dir / name)
                old = progress["best"]
                progress["best"] = {"update": u, "val_loss": ev["val_loss"], "dir": name,
                                    "sha256": sha, "bytes": size}
                progress["bad_evals"] = 0
                if rc["profile"] == "dry":
                    engine.keep_best(best_state)
            else:
                old = None
                progress["bad_evals"] += 1
            rec = {"kind": "eval", "session": session, "update": u,
                   "epoch": round(u / plan["updates_per_epoch"], 4), "val_loss": round(ev["val_loss"], 6),
                   "n_items": ev["n_items"], "n_tokens": ev["n_tokens"], "seconds": ev["seconds"],
                   "improved": improved, "bad_evals": progress["bad_evals"],
                   "best_update": progress["best"]["update"], "ts_utc": H.utc_now()}
            progress["evals"].append({k: rec[k] for k in ("update", "epoch", "val_loss", "n_items",
                                                          "n_tokens", "seconds", "improved")})
            progress["val_skipped"] = ev["skipped"]
            elog.add(rec)
            T.say(f"    validation after update {u}: loss {ev['val_loss']:.4f} on {ev['n_items']} items"
                  f" ({'better' if improved else 'not better'}; best is update {progress['best']['update']})")
            if progress["bad_evals"] >= int(T.ft["patience"]):
                if progress["would_stop_at"] is None:
                    progress["would_stop_at"] = u
                if rc["early_stop"]:
                    progress["stopped"] = {"update": u, "reason": f"validation loss not better "
                                           f"for {progress['bad_evals']} evaluations"}
            save()
            if improved and old:
                shutil.rmtree(run_dir / old["dir"], ignore_errors=True)
        elif u % ckpt_every == 0:
            save()
        if T.stop_after_s and time.time() - t_session > T.stop_after_s and u < plan["total_updates"] \
                and not progress["stopped"]:
            stop_now = "the time given with --stop-after-minutes is used up"
            break
    if pool:
        if pending:
            pending.cancel()
        pool.shutdown(wait=True)
    if not stop_now:
        progress["finished"] = True
    save()
    progress["stop_now"] = stop_now
    return progress


def finalize(engine, T, progress):
    """The best checkpoint becomes models/<run>/adapter/. Safe to repeat."""
    run_dir, best = Path(T.run_dir), progress["best"]
    final, src = run_dir / "adapter", run_dir / best["dir"]
    if src.exists():
        if final.exists():
            shutil.rmtree(final)
        os.rename(src, final)
    weights = final / H.ADAPTER_FILE
    if not weights.exists() or H.file_sha256(weights) != best["sha256"]:
        raise Stop(f"{weights} is missing or is not the best checkpoint that was saved. Send "
                   "this line to the FT chat.")
    return {"dir": "adapter", "file": H.ADAPTER_FILE, "sha256": best["sha256"],
            "bytes": best["bytes"], "from_update": best["update"], "val_loss": best["val_loss"]}


# --------------------------------------------------------------------------
# 7. Gate G: answers from memory and the overfit probe (D178)
# --------------------------------------------------------------------------
def pick_probe(dry_items, cfg):
    """The first training examples by rank whose gold answers are short and all
    different, so that reproducing them cannot come from one constant answer."""
    out, seen = [], set()
    for it in dry_items:
        key = normalise(it["answer_bn"])
        if key and key not in seen and len(it["answer_bn"]) <= int(cfg["max_target_chars"]):
            seen.add(key)
            out.append(it)
        if len(out) == int(cfg["n"]):
            break
    return out


def overfit_probe(engine, items, cfg, out_path):
    """Train fresh LoRA weights on a few training examples until the loss is small,
    then generate on the same examples with the harness's function. If the target is
    built correctly, the model gives back the gold answers and stops by itself."""
    exs = [engine.build(it, keep_prompt=True) for it in items]
    pairs = [(it, ex) for it, ex in zip(items, exs) if not ex.get("skip")]
    if len(pairs) < 4:
        return {"ran": False, "why": f"only {len(pairs)} usable probe examples"}
    engine.reset_lora(SEED + 1)
    engine.set_train(True)
    n_tok = sum(ex["n_labels"] for _it, ex in pairs)
    losses, t0 = [], time.time()
    for u in range(int(cfg["max_updates"])):
        engine.reseed(SEED + 1, u)
        nll = sum(engine.train_pass(ex, 1.0 / n_tok) for _it, ex in pairs)
        engine.step(float(cfg["lr"]))
        losses.append(nll / n_tok)
        if losses[-1] < float(cfg["stop_loss"]):
            break
    engine.sync()
    train_s = time.time() - t0
    if hasattr(engine, "memorise"):
        engine.memorise(pairs)
    rows = []
    for it, ex in pairs:
        g = engine.generate(ex["prompt_text"], ex.get("image"))
        rows.append({"qa_id": it["qa_id"],
                     "reproduced": normalise(g["answer_text"]) == normalise(it["answer_bn"]),
                     "stopped": g["finish_reason"] == "stop", "n_new_tokens": g["n_new_tokens"],
                     "answer_text": g["answer_text"]})
    for r in rows:                                   # answers are model output: Drive only
        H.append_jsonl(out_path, dict(r, kind="overfit_probe", ts_utc=H.utc_now()))
    return {"ran": True, "n": len(rows), "updates": len(losses),
            "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
            "reproduced": sum(r["reproduced"] for r in rows),
            "stopped": sum(r["stopped"] for r in rows),
            "both": sum(r["reproduced"] and r["stopped"] for r in rows),
            "train_seconds": round(train_s, 1),
            "items": [{k: r[k] for k in ("qa_id", "reproduced", "stopped", "n_new_tokens")} for r in rows]}


def generate_from_memory(engine, works, img_root, img_manifest, out_path):
    """The Gate G items, answered by the trained model that is still in memory."""
    if os.path.exists(str(out_path)):
        os.remove(str(out_path))
    n, secs = 0, []
    for w in works:
        image = None
        if w["has_image"] and engine.name != "fake":
            data, _sha = imglib.load_prepared(img_root, w["image_file"], img_manifest)
            image = imglib.decode_rgb(data)
        t0 = time.time()
        g = engine.generate(w["prompt_text"], image)
        secs.append(time.time() - t0)
        H.append_jsonl(out_path, {"kind": "from_memory", "qa_id": w["qa_id"],
                                  "condition_id": w["condition_id"], "answer_lang": w["answer_lang"],
                                  H.ANSWER_FIELD: g["answer_text"], "finish_reason": g["finish_reason"],
                                  "n_new_tokens": g["n_new_tokens"], "elapsed_s": g["elapsed_s"],
                                  "prompt_sha256": w["prompt_sha256"],
                                  "image_sha256": w["image_sha256"], "ts_utc": H.utc_now()})
        n += 1
    return {"n": n, "seconds_per_item": round(sum(secs) / max(1, n), 3)}


def compare_caches(path_a, path_b, conditions=None):
    """Two harness caches, joined on (condition, qa_id). Counts only."""
    a, b = H.Cache(path_a), H.Cache(path_b)
    ka = {(r["condition_id"], r["qa_id"]): r for r in a.records.values()}
    kb = {(r["condition_id"], r["qa_id"]): r for r in b.records.values()}
    both = sorted(k for k in ka if k in kb and (not conditions or k[0] in conditions))
    same = [k for k in both if ka[k][H.ANSWER_FIELD] == kb[k][H.ANSWER_FIELD]]
    env = {}
    for label, p in (("a", path_a), ("b", path_b)):
        env[label] = (H.read_json(Path(p).parent / "stage_state.json", {}) or {})
    ea, eb = env["a"].get("env") or {}, env["b"].get("env") or {}
    return {"in_a": len(ka), "in_b": len(kb), "in_both": len(both), "identical": len(same),
            "per_condition": dict(sorted(Counter(k[0] for k in both).items())),
            "identical_per_condition": dict(sorted(Counter(k[0] for k in same).items())),
            "first_different": next((f"{k[0]} {k[1]}" for k in both if k not in set(same)), ""),
            "settings_sha_a": env["a"].get("settings_sha256", ""),
            "settings_sha_b": env["b"].get("settings_sha256", ""),
            "env_differs_in": sorted(k for k in set(ea) | set(eb) if ea.get(k) != eb.get(k))}


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def gate_report(root, repo, run_name, base_old="", say=say):
    """Read what the dry run wrote and print the Gate G report. Counts only."""
    root, repo = Path(root), Path(repo)
    run_dir = root / "models" / run_name
    man = H.read_json(run_dir / H.FT_MANIFEST, None)
    gg = H.read_json(run_dir / "gate_g.json", None)
    if not man or not man.get("complete") or gg is None:
        raise Stop(f"{run_dir}: the dry run is not complete. Run  train --run {run_name}  first.")
    ft = load_ft(repo / "configs" / "finetune.yaml")
    g = ft["gate_g"]
    smoke = root / "outputs" / "smoke" / run_name
    lines, verdict = [], {}

    def line(code, ok, text):
        tag = {True: "PASS", False: "FAIL", None: "INFO"}[ok]
        lines.append(f"  {tag}  {code}  {text}")
        if ok is not None:
            verdict[code] = ok

    # G1: the loss does not fall near 0 within 50 steps
    batches = read_log(run_dir / "train_log.jsonl", "batch", "batch")
    win = int(g["loss_window"])
    first = [b["loss"] for b in batches[:win]]
    ok = len(first) == win and _mean(first[-10:]) >= float(g["loss_floor_last10"]) \
        and min(first) >= float(g["loss_floor_any"])
    line("G1", ok, f"the loss does not fall near 0 within {win} steps: {len(first)} steps logged; "
         f"mean of the first 10 {_mean(first[:10]):.3f}, of the last 10 {_mean(first[-10:]):.3f}, "
         f"lowest {min(first) if first else float('nan'):.3f} (floors {g['loss_floor_last10']} and "
         f"{g['loss_floor_any']})")
    # G2: 20 R-TEST items, short Bangla answers (from memory)
    mem, _bad = H.read_jsonl(smoke / "from_memory" / "answers.jsonl")
    mem = {r["qa_id"]: r for r in mem if r.get("kind") == "from_memory"}
    n_gen = int(g["n_generate"])
    not_empty = sum(1 for r in mem.values() if normalise(r[H.ANSWER_FIELD]))
    script = sum(1 for r in mem.values() if H.script_of(r[H.ANSWER_FIELD]) in ("bangla", "digits"))
    short = sum(1 for r in mem.values() if len(r[H.ANSWER_FIELD].split()) <= 8)
    cut = sum(1 for r in mem.values() if r["finish_reason"] != "stop")
    good = sum(1 for r in mem.values() if normalise(r[H.ANSWER_FIELD])
               and H.script_of(r[H.ANSWER_FIELD]) in ("bangla", "digits")
               and len(r[H.ANSWER_FIELD].split()) <= 8)
    line("G2", len(mem) == n_gen and good >= int(g["short_answers_min"]),
         f"generation on {n_gen} R-TEST items returns short Bangla answers: {len(mem)} answers; "
         f"{not_empty} not empty, {script} in Bangla script or digits, {short} with 8 words or "
         f"fewer, {cut} cut by the length limit; {good} meet all three (need {g['short_answers_min']})")
    # G3: saved, reloaded through the harness, same answers
    models = H.load_models(repo / "configs" / "models.yaml")
    key = next((k for k, m in models.items() if (m.get("adapter") or {}).get("run") == run_name), "")
    cond = g["generate_condition"]
    cache_path = smoke / key / "cache.jsonl"
    reload = {r["qa_id"]: r for r in H.Cache(cache_path).records.values() if r["condition_id"] == cond}
    same = sum(1 for q, r in mem.items() if q in reload and reload[q][H.ANSWER_FIELD] == r[H.ANSWER_FIELD])
    sha_ok = all(r.get("adapter_sha256") == man["adapter"]["sha256"] for r in reload.values())
    line("G3", bool(mem) and len(reload) >= len(mem) and same == len(mem) and sha_ok,
         f"the adapter saves, reloads and gives the same answers: {same} of {len(mem)} identical "
         f"({len(reload)} answers of model {key or '?'} under {cond} in the harness cache; adapter "
         f"file {'as written by the run' if sha_ok and reload else 'NOT the one of the run' if reload else 'not run yet'})")
    # G4: loss positions
    seen = {}
    for b in batches:
        for i, q in enumerate(b["qa_ids"]):
            seen[q] = (b["n_prompt"][i], b["n_answer"][i], b["n_ending"][i], b["n_labels"][i], b["n_tokens"][i])
    right = sum(1 for p, a, e, lab, tot in seen.values() if lab == a + e and tot == p + lab and a > 0)
    share = 100.0 * sum(v[3] for v in seen.values()) / max(1, sum(v[4] for v in seen.values()))
    n_train = man["data"]["n_train"] - len(man["result"]["skipped"])
    line("G4", bool(seen) and right == len(seen) == n_train,
         f"loss positions: {right} of {len(seen)} trained examples have the answer and the end "
         f"of turn as their only loss positions ({n_train} expected); mean "
         f"{_mean(v[3] for v in seen.values()):.1f} of {_mean(v[4] for v in seen.values()):.0f} "
         f"tokens ({share:.2f}%)")
    # G5: overfit probe
    pr, pc = gg.get("overfit_probe") or {}, g["probe"]
    line("G5", bool(pr.get("ran")) and pr.get("both", 0) >= int(pc["min_reproduced"]),
         (f"overfit probe: {pr.get('reproduced')} of {pr.get('n')} gold answers reproduced, "
          f"{pr.get('stopped')} stopped by themselves, {pr.get('both')} both (need "
          f"{pc['min_reproduced']}); loss {pr.get('loss_first')} to {pr.get('loss_last')} in "
          f"{pr.get('updates')} updates") if pr.get("ran") else f"overfit probe: not run ({pr.get('why')})")
    # G6: alignment of training and generation, on the base model
    bp = gg.get("base_probe") or {}
    n = bp.get("n", 0)
    line("G6", (bp.get("forward_prompt_equals_forward_full", 0) >= max(1, n - 1)
                and bp.get("forward_equals_generate", 0) >= max(1, n - 1)
                and bp.get("stopped_by_itself", 0) >= 1
                and bp.get("ends_like_target", 0) == bp.get("stopped_by_itself", -1)
                and bp.get("train_mode_same_next_token", n) == n) if n else None,
         f"base model, {n} training prompts: next token equal with and without the target "
         f"appended {bp.get('forward_prompt_equals_forward_full')} of {n}; equal to the first "
         f"generated token {bp.get('forward_equals_generate')} of {n}; answers that stop by "
         f"themselves {bp.get('stopped_by_itself')}, of these ending like the training target "
         f"{bp.get('ends_like_target')}; frozen base model in training mode: same next token "
         f"{bp.get('train_mode_same_next_token')} of {n}, largest logit difference "
         f"{bp.get('train_mode_max_logit_diff')}"
         + (f"; other endings seen: {bp.get('tails')}" if bp.get("tails") else ""))
    say("=" * 78)
    say(f"FT Gate G report   run {run_name}   script {man['script_version']}   {H.utc_now()}")
    say(f"base {man['base']['model_id']} at commit {str(man['base']['model_revision'])[:12]}   prompt "
        f"{man['prompt']['prompt_version']} ({man['prompt']['template_id']})   peft {man['env'].get('peft')}"
        f"   {man['env'].get('gpu')}")
    say(f"LoRA on {man['lora']['n_modules']} linear layers of {man['lora']['n_layers']} language layers; "
        f"{man['lora']['trainable_params']:,} trainable of {man['lora']['total_params']:,} weights; "
        f"end of turn {man['target']['tokens']}")
    say("=" * 78)
    say("[A] The three checks of Design 7.3")
    for ln in lines[:3]:
        say(ln)
    say("[B] Added by decision D178")
    for ln in lines[3:]:
        say(ln)
    # H: the harness edit
    say("[C] The harness edit (D174)")
    base_key = ft["base_model"]
    now_sha = H.canonical_sha(H.hf_settings(models[base_key]))
    old_path = Path(base_old) if base_old else None
    new_path = root / "outputs" / "smoke" / "ft-base" / base_key / "cache.jsonl"
    if old_path and old_path.exists() and new_path.exists():
        c = compare_caches(old_path, new_path)
        ok1 = c["settings_sha_a"] == now_sha
        say(f"  {'PASS' if ok1 else 'FAIL'}  H1  fixed settings of the base model: hash now {now_sha[:12]}, "
            f"stored by the earlier smoke run {c['settings_sha_a'][:12]}")
        ok2 = c["in_both"] > 0 and c["identical"] == c["in_both"] == c["in_a"]
        say(f"  {'PASS' if ok2 else 'FAIL'}  H2  base answers after the edit: {c['identical']} of "
            f"{c['in_both']} identical to the earlier smoke run ({c['in_a']} answers there; per "
            f"condition {c['identical_per_condition']})"
            + (f"; the environment differs in {c['env_differs_in']}, so a difference need not "
               "come from the edit" if c["env_differs_in"] else "; same environment")
            + (f"; first different: {c['first_different']}" if c["first_different"] else ""))
        verdict["H1"], verdict["H2"] = ok1, ok2
    else:
        say(f"  INFO  H1  fixed settings of the base model: hash now {now_sha[:12]}")
        say("  INFO  H2  base answers after the edit: not compared (give --base-old and run the "
            "reload command first)")
    base_same = root / "outputs" / "smoke" / run_name / base_key / "cache.jsonl"
    if base_same.exists() and cache_path.exists():
        c = compare_caches(base_same, cache_path, conditions={cond})
        say(f"  INFO  H3  on the {c['in_both']} R-TEST items the dry-run adapter changes "
            f"{c['in_both'] - c['identical']} base answers (few are expected after "
            f"{man['result']['updates_done']} weight updates)")
    # D: seconds per item
    say(f"[D] Seconds per item on {man['env'].get('gpu')}")
    ups = [x for x in read_log(run_dir / "train_log.jsonl", "update", "update") if x["update"] > 1]
    t_train = _mean(x["wall_s"] / max(1, x["examples"]) for x in ups)
    b_late = [b for b in batches if b["update"] > 1]
    t_gpu = _mean(b["gpu_s"] / len(b["qa_ids"]) for b in b_late)
    t_build = _mean(b["build_s"] / len(b["qa_ids"]) for b in b_late)
    evs = man["result"]["evals"][1:] or man["result"]["evals"]
    t_val = _mean(e["seconds"] / max(1, e["n_items"]) for e in evs)
    say(f"  training    {t_train:.2f} s per example (GPU {t_gpu:.2f} s, preparing the example "
        f"{t_build:.2f} s; first update left out)   peak GPU memory {man['result']['peak_gpu_gb']} GB")
    say(f"  validation  {t_val:.2f} s per item")
    t_mem = _mean(r["elapsed_s"] for r in mem.values())
    t_rel = _mean(r.get("elapsed_s", float("nan")) for r in reload.values()) if reload else float("nan")
    say(f"  generation  {t_mem:.2f} s per item from memory, {t_rel:.2f} s through the harness "
        f"({len(reload)} items, with image)")
    stats = H.read_json(root / "data" / "ft" / DATA_MANIFEST, {})
    n_all = int(stats.get("n_train") or DESIGN_3_1["train"][2])
    full = make_plan(n_all, ft, {"epochs": int(ft["epochs"])})
    seed_h = (n_all * ft["epochs"] * t_train + len(full["eval_points"]) * ft["val_n"] * t_val) / 3600
    say(f"  projection, training: one seed = {ft['epochs']} epochs x {n_all:,} examples + "
        f"{len(full['eval_points'])} x {ft['val_n']} validation items = {seed_h:.1f} h; three seeds "
        f"{3 * seed_h:.1f} h. Design 12.2 plans {PLAN_TRAIN_HOURS_3_SEEDS} L4 hours: "
        + ("WITHIN the plan" if 3 * seed_h <= PLAN_TRAIN_HOURS_3_SEEDS else "OVER the plan, ask before training"))
    n_test, n_shared = 1449, 246                      # DESIGN_ADDENDUM A1
    rtest = root / "probes" / "R-TEST.json"
    if rtest.exists():
        its = H.load_probe(rtest)["items"]
        n_test, n_shared = len(its), sum(1 for it in its if it.get("in_pcore_closed"))
    n_eval = 3 * 3 * n_test + 3 * (n_test - n_shared)
    eval_h = n_eval * (t_rel if t_rel == t_rel else t_mem) / 3600
    say(f"  projection, evaluation: {n_eval:,} generations (R-TEST x 3 conditions x 4 model "
        f"versions, less the base answers reused from EXP-L) = {eval_h:.1f} h at the with-image "
        f"speed, an upper bound. Design 12.2 plans {PLAN_EVAL_HOURS} L4 hours: "
        + ("WITHIN the plan" if eval_h <= PLAN_EVAL_HOURS else "OVER the plan, ask before the evaluation"))
    need = ("G1", "G2", "G3", "G4", "G5")
    passed = all(verdict.get(k) for k in need)
    other = [k for k, v in verdict.items() if k not in need and v is False]
    say("")
    say("GATE G: " + ("PASS" if passed else "NOT PASSED") + f" ({', '.join(k + ' ' + ('ok' if verdict.get(k) else 'no') for k in need)})"
        + (f"; also look at {', '.join(other)}" if other else "")
        + ". The team lead signs the gate off; send this whole report to the FT chat.")
    report = {"run": run_name, "utc": H.utc_now(), "script_version": FT_VERSION, "verdict": verdict,
              "gate_g_passed": passed, "ft_settings_sha256": man["ft_settings_sha256"],
              "peft": man["env"].get("peft"), "seconds": {"train_per_example": t_train,
                                                         "val_per_item": t_val, "generate_per_item": t_rel},
              "projection_hours": {"one_seed": seed_h, "three_seeds": 3 * seed_h, "evaluation": eval_h}}
    H.write_json(run_dir / "gate_g_report.json", report)
    say(f"wrote: {run_dir / 'gate_g_report.json'}")
    if passed and not ft.get("peft_version"):
        say(f"After the sign-off: write  peft_version: \"{man['env'].get('peft')}\"  into "
            "configs/finetune.yaml (D171) and commit it.")
    return 0 if passed else 2


# --------------------------------------------------------------------------
# 8. The harness as a subprocess: one model version after the other
# --------------------------------------------------------------------------
def harness_run(repo, args, say=say, dry=False, quiet=False):
    args = [str(x) for x in args] + (["--dry-run"] if dry else [])
    cmd = [sys.executable, str(Path(repo) / "scripts" / "asset8_harness.py"), "run"] + args
    say("  $ python scripts/asset8_harness.py run " + " ".join(args))
    if quiet:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    return subprocess.run(cmd).returncode


def _answers_table(root, stage, run_name, keys, say=say):
    say(f"  {'model':<14} {'condition':<9} {'answers':>7} {'empty':>5} {'cut':>4} {'s per item':>10}")
    for key in keys:
        out_dir, _label = H.layout(root, stage, run_name, key, False)
        recs = list(H.Cache(out_dir / "cache.jsonl").records.values())
        by = {}
        for r in recs:
            by.setdefault(r["condition_id"], []).append(r)
        if not by:
            say(f"  {key:<14} {'-':<9} {0:>7}")
        for cid in sorted(by):
            rs = by[cid]
            secs = [r["elapsed_s"] for r in rs if r.get("elapsed_s") is not None and r.get("mode") != "reuse"]
            say(f"  {key:<14} {cid:<9} {len(rs):>7} {sum(1 for r in rs if r.get('is_empty')):>5} "
                f"{sum(1 for r in rs if r.get('truncated')):>4} {(_mean(secs) if secs else float('nan')):>10.2f}")


# --------------------------------------------------------------------------
# Commands (called by scripts/ft_finetune.py)
# --------------------------------------------------------------------------
def _header(title, git):
    say("=" * 78)
    say(f"FT scripts {FT_VERSION}   {title}")
    say(f"code: commit {(git.get('commit') or '')[:12] or 'unknown'}"
        + ("" if not git.get("dirty") else "  WITH UNCOMMITTED CHANGES"))
    say("=" * 78)


def _installed(package):
    try:
        from importlib import metadata
        return metadata.version(package)
    except Exception:
        return ""


def _dig(d, path):
    for k in path:
        d = (d or {}).get(k)
    return d


def data_command(a):
    """Define the training and validation data, check them, prepare the images."""
    root, repo = Path(a.root), Path(a.repo)
    ft = load_ft(repo / "configs" / "finetune.yaml")
    git = H.git_info(repo)
    dataset_path = Path(a.dataset) if a.dataset else root / "data" / "banglaindoorvqa.json"
    splits_path = root / "data" / "splits.csv"
    _header("data: training and validation data of EXP-R (D172, D173, D175)", git)

    say("\n[1] Dataset and split file")
    d = define_data(dataset_path, splits_path, val_n=int(ft["val_n"]),
                    dry_n=int(ft["runs"]["ft-dry"]["n_train"]))
    say(f"  read {dataset_path.name}: {len(d['images'])} images, {len(d['items'])} QA pairs;  "
        f"{splits_path.name}: {len(d['splits'])} houses")
    say(f"  {'split':<6} {'houses':>6} {'images':>7} {'QA':>7} {'usable':>7} {'line break fixed':>17}")
    for sp in ("train", "val", "test"):
        st = d["stats"][sp]
        say(f"  {sp:<6} {st['houses']:>6} {st['images']:>7} {st['qa']:>7} {st['usable']:>7} "
            f"{st['line_break_fixed']:>17}")
    say("  houses, images and QA are those of Design 3.1 (checked; the run would have stopped)")
    say("  QA types in train: " + ", ".join(f"{k} {v}" for k, v in d["stats"]["train"]["types"].items()))
    n_fixed = sum(1 for it in d["items"] if it["line_break_fixed"])
    if d["unusable"] or n_fixed:
        say(f"  NOTE: {len(d['unusable'])} QA pairs have an empty question or answer and are left "
            f"out; in {n_fixed} a line break inside the text became one space. Report both "
            "counts in the FT chat: they need a decision row if they are not 0.")

    say("\n[2] Cross-check with R-TEST.json")
    probe = H.load_probe(Path(a.rtest) if a.rtest else root / "probes" / "R-TEST.json")
    n_ok = crosscheck_probe(d["by_id"], probe["items"], "test")
    say(f"  {n_ok} of {len(probe['items'])} R-TEST items are in the test split of the dataset with "
        "the same image, type, question and answer")
    used = {it["house_id"] for it in d["train"]} | {it["house_id"] for it in d["val"]}
    overlap = sorted(used & {it["house_id"] for it in probe["items"]})
    if overlap:
        raise Stop(f"R-TEST houses are in the train or validation data: {overlap}. Nothing was written.")
    say("  no R-TEST house is in the training or validation data")

    say("\n[3] Fixed lists (seeded rank, seed 42)")
    out_dir = root / "data" / "ft"
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / DATA_MANIFEST
    old = H.read_json(man_path, None)
    if old is not None:
        diff = [k for k, v in d["hashes"].items() if old.get(k) != v]
        if diff and not a.force:
            raise Stop(f"{man_path} was written for other data ({', '.join(diff)} changed). The "
                       "dataset is not edited after 3 October (D110). If the change is "
                       "intended, write a decision row and add --force. Nothing was written.")
    val_ids = [it["qa_id"] for it in d["val"]]
    targets = [out_dir / VAL_LIST]
    if getattr(a, "repo_list", True) and (repo / "probes").is_dir():
        targets.append(repo / "probes" / VAL_LIST)
    for t in targets:
        t.write_text("\n".join(val_ids) + "\n", encoding="utf-8")
    say(f"  training: {len(d['train'])} QA pairs, all 8 types; probes/excluded_items.csv is not "
        f"applied (D172); sha256 of the id list {d['hashes']['train_ids_sha256'][:12]}")
    say(f"  validation list: {len(val_ids)} qa_ids, sha256 {d['hashes']['val_ids_sha256'][:12]}")
    for t in targets:
        say(f"    wrote {t}")
    if len(targets) == 2:
        say(f"    commit probes/{VAL_LIST} to the repo (qa_ids only, no text)")
    n_val_dry = int(ft["runs"]["ft-dry"].get("n_val", ft["val_n"]))
    say(f"  dry run: {len(d['dry'])} training QA pairs; validation on the first {n_val_dry} of the list")

    say("\n[4] Images (the one image loader: D148, D175)")
    src, dst = root / "data" / "images", root / "data" / "images_1024"
    gate_items = H.select_items(probe["items"], sample=int(ft["gate_g"]["n_generate"]))
    sets = {"train": images_of(d["train"]), "validation list": images_of(d["val"]),
            "R-TEST": images_of(probe["items"]),
            "dry run": images_of(d["dry"] + d["val"][:n_val_dry] + gate_items),
            "whole dataset": sorted(d["images"])}
    if a.prepare_images != "none":
        files = sets["whole dataset"] if a.prepare_images == "all" else sets["dry run"]
        say(f"  preparing {len(files)} images ({a.prepare_images}) from {src} to {dst}")
        say(f"  rule: {imglib.PREP_VERSION}")
        st = imglib.prepare_many(files, src, dst, say=say, verify=bool(a.verify))
        problems = st.pop("problems")
        say("  " + ", ".join(f"{k} {v}" for k, v in st.items()))
        for rel, why in problems[:20]:
            say(f"  PROBLEM {rel}: {why}")
    manifest = imglib.read_manifest(dst)
    cover = {}
    for name, files in sets.items():
        have = sum(1 for f in files if f in manifest)
        cover[name] = {"wanted": len(files), "prepared": have}
        say(f"  {name:<16} {have:>5} of {len(files):>5} images prepared")
    pillow = Counter(manifest[f].get("pillow", "?") for f in sets["whole dataset"] if f in manifest)
    say("  Pillow version of the prepared copies: " + (", ".join(f"{k} ({v})" for k, v in sorted(pillow.items())) or "none yet"))

    H.write_json(man_path, data_manifest(d, {
        "created_utc": (old or {}).get("created_utc") or H.utc_now(), "updated_utc": H.utc_now(),
        "dataset_file": dataset_path.name, "git": git, "image_coverage": cover,
        "rtest_items_checked": n_ok, "val_list": VAL_LIST}))
    say(f"\n  wrote: {man_path}")
    ready_dry = cover["dry run"]["prepared"] == cover["dry run"]["wanted"]
    ready_all = all(cover[k]["prepared"] == cover[k]["wanted"] for k in ("train", "validation list", "R-TEST"))
    say("  RESULT: " + ("ready for the dry run and for training" if ready_all else
                        "ready for the dry run; run again with --prepare-images all before a seed is trained"
                        if ready_dry else "NOT READY: prepared images are missing"))
    return 0 if ready_dry else 2


def check_seed_requirements(ft, m, git, root, repo, d, installed_peft):
    """D176: a seed is trained only when everything it depends on is final."""
    missing = []
    if not git.get("gate_e") or git.get("dirty") is not False:
        missing.append("the tag gate-E on this commit or before it and no uncommitted change")
    if not m.get("revision"):
        missing.append("the model commit pasted into configs/models.yaml (revision)")
    if not ft.get("peft_version") or str(ft["peft_version"]) != installed_peft:
        missing.append(f"peft_version in configs/finetune.yaml equal to the installed PEFT "
                       f"('{installed_peft or 'not installed'}'), as fixed at Gate G (D171)")
    rep = H.read_json(root / "models" / "ft-dry" / "gate_g_report.json", {})
    if not rep.get("gate_g_passed") or rep.get("script_version") != FT_VERSION \
            or rep.get("peft") != str(ft.get("peft_version")):
        missing.append(f"a passed Gate G report made with this script version ({FT_VERSION}) and "
                       "this PEFT version (run the dry run again after a script change)")
    listed = repo / "probes" / VAL_LIST
    ids = [ln.strip() for ln in listed.read_text(encoding="utf-8").splitlines() if ln.strip()] \
        if listed.exists() else []
    if ids != [it["qa_id"] for it in d["val"]]:
        missing.append(f"probes/{VAL_LIST} in the repo, equal to the list of the data command")
    if missing:
        raise Stop("A seed is trained only after Gate G, Gate D and Gate E (D176). Missing: "
                   + "; ".join(missing) + ". Nothing was loaded.")


def say_lora(lora):
    say(f"  LoRA on {lora['n_modules']} linear layers inside {lora['n_layers']} language layers "
        f"(layer stack '{lora['stack_prefix']}'; the model config names {lora['n_layers_config']})")
    say("  adapted, by place inside a layer: " + ", ".join(f"{k} x{v}" for k, v in lora["leaves"].items()))
    say("  linear layers per language layer (number: layers): "
        + ", ".join(f"{k}: {v}" for k, v in lora["linear_per_layer"].items()))
    say("  all linear layers of the base model by part (layers, weights): "
        + "; ".join(f"{k} {v['layers']} ({v['weights']:,})" for k, v in lora["components"].items()))
    if lora["other_linear_not_adapted"]:
        say("  linear layers outside every part, not adapted: " + ", ".join(lora["other_linear_not_adapted"][:8]))
    say(f"  trainable parameters: {lora['trainable_params']:,} of {lora['total_params']:,} "
        f"({100.0 * lora['trainable_params'] / max(1, lora['total_params']):.3f}%), stored as "
        f"{lora['param_dtype']}; PEFT {lora['peft']}")


RESUME_KEYS = (("ft_settings_sha256",), ("base", "model_revision"), ("base", "settings_sha256"),
               ("prompt", "template_sha256"), ("data", "dataset_sha256"),
               ("data", "train_ids_sha256_run"), ("data", "val_ids_sha256_run"))
ENV_KEYS = ("gpu", "torch", "transformers", "peft", "fla", "causal_conv1d")


def train_command(a):
    """Train one run of configs/finetune.yaml. Can be started again at any time."""
    root, repo = Path(a.root), Path(a.repo)
    fake = bool(getattr(a, "fake", False))
    ft = load_ft(repo / "configs" / "finetune.yaml")
    rc = run_cfg(ft, a.run)
    dry = rc["profile"] == "dry"
    prompts = promptlib.load_prompts(repo / "configs" / "prompts.yaml")
    cfg = promptlib.load_conditions(repo / "configs" / "conditions.yaml", prompts)
    cond = cfg["conditions"].get(ft["condition"])
    gcond = cfg["conditions"].get(ft["gate_g"]["generate_condition"])
    if not cond or not cond["image"] or cond["forced_choice"] \
            or (cond["question_lang"], cond["answer_lang"]) != ("bn", "bn"):
        raise Stop("configs/finetune.yaml: the training condition must be a Bangla question, "
                   "Bangla answer, with image, open answer (Design 7.3: the L-BB template).")
    if not gcond or gcond["template"] != cond["template"] or gcond["image"] != cond["image"]:
        raise Stop("configs/finetune.yaml: gate_g.generate_condition must use the template the "
                   "model is trained with.")
    models = H.load_models(repo / "configs" / "models.yaml")
    if ft["base_model"] not in models:
        raise Stop(f"configs/finetune.yaml: base_model '{ft['base_model']}' is not in configs/models.yaml.")
    base_key, m = ft["base_model"], models[ft["base_model"]]
    run_dir, smoke_dir = root / "models" / a.run, root / "outputs" / "smoke" / a.run
    git = H.git_info(repo)
    ext = "json" if fake else "pt"
    _header(f"train: run {a.run} ({'Gate G dry run' if dry else 'EXP-R seed ' + str(rc['seed'])})", git)
    say(f"base {base_key} = {m['model_id']}   prompts {prompts['prompt_version']} "
        f"(sha256 {prompts['sha256'][:12]})   training prompt: {ft['condition']}")
    say("Everything this run writes is model output or leads to it: Drive only, and not for the "
        "two HT-1 people before gate-A (D150).")

    if getattr(a, "restart", False):
        if not dry:
            raise Stop("--restart is for the dry run only. A seed that was started is finished, "
                       "or reported as failed (Design 7.3).")
        for p in (run_dir, smoke_dir):
            shutil.rmtree(p, ignore_errors=True)
        say("  --restart: the earlier dry run and its smoke outputs were deleted")
    man_path = run_dir / H.FT_MANIFEST
    old = H.read_json(man_path, None)
    if old and old.get("complete"):
        say(f"\n  {a.run} is already COMPLETE (adapter sha256 {old['adapter']['sha256'][:12]}). "
            "Nothing to do." + (" To repeat the dry run add --restart." if dry else ""))
        return 0

    say("\n[1] Data")
    d = define_data(Path(a.dataset) if getattr(a, "dataset", "") else root / "data" / "banglaindoorvqa.json",
                    root / "data" / "splits.csv", val_n=int(ft["val_n"]),
                    dry_n=int(ft["runs"]["ft-dry"]["n_train"]))
    check_data_manifest(d, root / "data" / "ft" / DATA_MANIFEST)
    if rc["n_train"] and rc["n_train"] != len(d["dry"]):
        raise Stop("configs/finetune.yaml: only the dry run may have n_train, and it is 100.")
    train_items = d["dry"] if rc["n_train"] else d["train"]
    val_items = d["val"][:rc["n_val"]]
    plan = make_plan(len(train_items), ft, rc)
    say(f"  {len(train_items)} training examples, {len(val_items)} validation items (the data "
        "command fixed both; hashes checked)")
    say(f"  plan: batches of {plan['per_device']}, {plan['accumulation']} batches per weight update; "
        f"{plan['batches_per_epoch']} batches and {plan['updates_per_epoch']} updates per epoch; "
        f"{plan['epochs']} epoch(s) = {plan['total_updates']} updates; warmup {plan['warmup_updates']}; "
        f"validation after updates {plan['eval_points']}")
    installed_peft = _installed("peft")
    if not fake and ft.get("peft_version") and str(ft["peft_version"]) != installed_peft:
        raise Stop(f"configs/finetune.yaml fixes peft_version {ft['peft_version']}; this session "
                   f"has '{installed_peft or 'none'}' (D171). Run: pip install -q "
                   f"peft=={ft['peft_version']}")
    if not dry and not fake:
        check_seed_requirements(ft, m, git, root, repo, d, installed_peft)
        say("  Gate E, the model commit, the PEFT version, the Gate G report and the validation "
            "list are in place (D176). Gate D is not checked by code: train only the model "
            "that passed it.")

    say("\n[2] Images")
    gate_items = []
    if dry:
        probe = H.load_probe(Path(a.rtest) if getattr(a, "rtest", "") else root / "probes" / "R-TEST.json")
        gate_items = H.select_items(probe["items"], sample=int(ft["gate_g"]["n_generate"]))
    img_drive = root / "data" / "images_1024"
    needed = images_of(train_items + val_items + gate_items)
    img_root = img_drive
    if fake:
        img_manifest = {f: {"sha256": "0" * 64, "bytes": "0"} for f in needed}
    else:
        img_manifest = imglib.read_manifest(img_drive)
        lost = [f for f in needed if f not in img_manifest]
        if lost:
            raise Stop(f"{len(lost)} of the {len(needed)} images this run needs are not prepared "
                       f"(first: {lost[0]}). Run the data command with --prepare-images first.")
        if getattr(a, "image_cache", ""):
            copied = cache_images(needed, img_drive, a.image_cache, img_manifest)
            img_root = Path(a.image_cache)
            say(f"  {len(needed)} prepared images; {copied} copied to {img_root} now. Every read "
                "is checked against images_manifest.csv.")
        else:
            say(f"  {len(needed)} prepared images, read from Drive and checked against the manifest")
    gate_works = H.build_work("from-memory", gcond, gate_items, prompts, None, None, img_manifest) \
        if dry else []

    say("\n[3] Model")
    run_dir.mkdir(parents=True, exist_ok=True)
    if fake:
        engine = FakeEngine(ft, rc, prompts, cond, **(getattr(a, "fake_opts", None) or {}))
    else:
        engine = TorchEngine(models, ft, rc, prompts, cond, img_root, img_manifest, run_dir,
                             allow_a100=bool(getattr(a, "allow_a100", False)),
                             grad_checkpointing=bool(getattr(a, "grad_checkpointing", False)))
    env = engine.env()
    say("  environment: " + json.dumps(env))
    e = engine.ending
    say(f"  an answer is followed by {e['tokens']} (token ids {e['ids']}): {e['rule']}")
    say(f"  generation stops on token ids {e['stop_token_ids']}; the template writes "
        f"{e['template_tail']!r} after an answer; generation prompt equals the head of a "
        f"finished turn: {e['generation_prompt_equals_finished_turn_head']}")
    settings, base_settings, tid = ft_settings(ft, rc), H.hf_settings(m), cond["template"]
    man = {
        "asset": "FT", "script_version": FT_VERSION, "run": a.run, "profile": rc["profile"],
        "seed": rc["seed"], "complete": False, "created_utc": H.utc_now(), "git": git,
        "base": {"model_key": base_key, "model_id": m["model_id"],
                 "model_revision": env.get("model_revision"), "settings": base_settings,
                 "settings_sha256": H.canonical_sha(base_settings)},
        "prompt": {"prompt_version": prompts["prompt_version"], "prompts_sha256": prompts["sha256"],
                   "condition": ft["condition"], "template_id": tid, "variant": "with_image",
                   "template_sha256": H.sha256_hex(prompts["templates"][tid]["with_image"])},
        "env": env, "ft_settings": settings, "ft_settings_sha256": H.canonical_sha(settings),
        "target": e,
        "data": dict(d["hashes"], n_train=len(train_items), n_val=len(val_items),
                     train_ids_sha256_run=ids_sha(sorted(it["qa_id"] for it in train_items)),
                     val_ids_sha256_run=ids_sha([it["qa_id"] for it in val_items])),
        "plan": plan, "images": {"root": str(img_drive), "prep": imglib.PREP_VERSION, "n": len(needed)},
        "base_probe": None, "lora": None, "check_backward": None, "result": None, "adapter": None}
    if old:
        diff = [".".join(p) for p in RESUME_KEYS if _dig(old, p) != _dig(man, p)]
        env_diff = [k for k in ENV_KEYS if (old.get("env") or {}).get(k) != env.get(k)]
        if diff or (env_diff and not getattr(a, "allow_env_change", False)):
            raise Stop(f"{run_dir} holds an unfinished run made differently (changed: "
                       f"{', '.join(diff + ['env.' + k for k in env_diff])}). One run is trained "
                       "under one setting and in one environment. If the environment change is "
                       "intended, write a decision row and add --allow-env-change. Nothing was "
                       "trained.")
        man["created_utc"], man["base_probe"] = old.get("created_utc") or man["created_utc"], old.get("base_probe")
    if not man["base_probe"]:
        man["base_probe"] = engine.probe_base(seeded(train_items, "ft-baseprobe", 3))
    bp = man["base_probe"]
    say(f"  base model on {bp['n']} training prompts: next token the same with and without the "
        f"target appended {bp['forward_prompt_equals_forward_full']}; the same as the first "
        f"generated token {bp['forward_equals_generate']}; answers that stop by themselves "
        f"{bp['stopped_by_itself']}, of these ending like the training target {bp['ends_like_target']}"
        f"; same next token in training mode {bp.get('train_mode_same_next_token')} (largest "
        f"logit difference {bp.get('train_mode_max_logit_diff')})"
        + (f"; other endings: {bp['tails']}" if bp.get("tails") else ""))
    lora = engine.attach_lora(rc["seed"])
    say_lora(lora)
    if old and _dig(old, ("lora", "targets_sha256")) not in (None, lora["targets_sha256"]):
        raise Stop(f"{run_dir}: the list of adapted layers differs from the earlier session of "
                   "this run. Send this line to the FT chat. Nothing was trained.")
    man["lora"] = lora
    first = next((ex for ex in (engine.build(it) for it in train_items[:8]) if not ex.get("skip")), None)
    if first is None:
        raise Stop("None of the first training examples could be built.")
    chk = engine.check_backward(first)
    man["check_backward"] = dict(chk, n_labels=first["n_labels"], n_tokens=first["n_tokens"],
                                 image_tokens=first["image_tokens"])
    say(f"  one training pass: loss {chk['loss']} on {first['n_labels']} loss positions of "
        f"{first['n_tokens']} tokens ({first['image_tokens']} image tokens); gradient reaches "
        f"{chk['lora_B_with_gradient']} of {chk['lora_B_tensors']} LoRA B tensors; {chk['seconds']} s; "
        f"peak GPU memory {chk['peak_gpu_gb']} GB")
    H.write_json(man_path, man)

    say("\n[4] Training")
    T = H.Ctx(plan=plan, rc=rc, ft=ft, train_items=train_items, val_items=val_items, run_dir=run_dir,
              prefetch=not getattr(a, "no_prefetch", False), say=say,
              stop_after_s=60.0 * float(getattr(a, "stop_after_minutes", 0) or 0))
    progress = train_run(engine, T)
    peak = max(engine.peak_gb(), float(_dig(old or {}, ("result", "peak_gpu_gb")) or 0))
    man["result"] = {"updates_done": progress["update"], "total_updates": plan["total_updates"],
                     "finished": progress["finished"], "stopped": progress["stopped"],
                     "would_stop_at": progress["would_stop_at"], "evals": progress["evals"],
                     "best": progress["best"], "skipped": progress["skipped"],
                     "val_skipped": progress["val_skipped"],
                     "train_seconds": round(progress["train_seconds"], 1),
                     "sessions": progress["sessions"], "peak_gpu_gb": peak}
    if progress.get("stop_now"):
        H.write_json(man_path, man)
        say(f"\n  NOT COMPLETE: {progress['stop_now']}. {progress['update']} of {plan['total_updates']} "
            "updates are done and saved. Run the same command again.")
        return 2
    adapter = finalize(engine, T, progress)
    stop = progress["stopped"]
    say(f"  training ended after update {progress['update']} of {plan['total_updates']}"
        + (f" (early stopping: {stop['reason']})" if stop else "")
        + (f"; early stopping would have ended it after update {progress['would_stop_at']} (off in the dry run)"
           if not stop and progress["would_stop_at"] else ""))
    say(f"  best checkpoint: after update {adapter['from_update']}, validation loss {adapter['val_loss']:.4f}; "
        f"{len(progress['skipped'])} training examples left out (too long or empty); "
        f"{progress['train_seconds'] / 60:.1f} min of training in {progress['sessions']} session(s)")

    if dry:
        say("\n[5] Gate G: answers from memory, then the overfit probe (D178)")
        source = engine.use_best(run_dir / "state" / f"best_lora.{ext}")
        mem = generate_from_memory(engine, gate_works, img_root, img_manifest,
                                   smoke_dir / "from_memory" / "answers.jsonl")
        say(f"  {mem['n']} R-TEST items answered under {gcond['condition_id']} by the trained model "
            f"(best checkpoint, weights from {source}): {mem['seconds_per_item']} s per item")
        pcfg = ft["gate_g"]["probe"]
        pr = overfit_probe(engine, pick_probe(train_items, pcfg), pcfg,
                           smoke_dir / "from_memory" / "overfit_probe.jsonl")
        if pr.get("ran"):
            say(f"  overfit probe on {pr['n']} training examples with different short answers: loss "
                f"{pr['loss_first']} to {pr['loss_last']} in {pr['updates']} updates; "
                f"{pr['reproduced']} gold answers reproduced, {pr['stopped']} stopped by themselves")
        else:
            say(f"  overfit probe not run: {pr.get('why')}")
        H.write_json(run_dir / "gate_g.json", {
            "utc": H.utc_now(), "weights_from": source, "from_memory": mem, "overfit_probe": pr,
            "base_probe": bp, "check_backward": man["check_backward"],
            "extended_per_token_inputs": sorted(engine.extended_keys)})

    man.update({"complete": True, "finished_utc": H.utc_now(), "adapter": adapter})
    H.write_json(man_path, man)
    if not dry and not getattr(a, "keep_state", False):
        shutil.rmtree(run_dir / "state", ignore_errors=True)
    say(f"\n  wrote: {run_dir / 'adapter' / H.ADAPTER_FILE}  (sha256 {adapter['sha256'][:12]}, "
        f"{adapter['bytes'] / 2 ** 20:.1f} MB; LoRA adapter only)")
    say(f"         {man_path}, train_log.jsonl, eval_log.jsonl" + (", gate_g.json" if dry else ""))
    say("\n  COMPLETE. Next: " + (f"python scripts/ft_finetune.py reload --run {a.run}" if dry else
                                 "the eval command, when all seeds are trained"))
    return 0


def reload_command(a):
    """Gate G, the harness part: the saved adapter and the base model on the same
    R-TEST items, then the base model on the items of the earlier smoke run."""
    root, repo = Path(a.root), Path(a.repo)
    ft = load_ft(repo / "configs" / "finetune.yaml")
    models = H.load_models(repo / "configs" / "models.yaml")
    key = next((k for k, m in models.items() if (m.get("adapter") or {}).get("run") == a.run), "")
    if not key:
        raise Stop(f"No model entry of configs/models.yaml has adapter.run = {a.run}.")
    base_key, g = ft["base_model"], ft["gate_g"]
    rtest = Path(a.rtest) if a.rtest else root / "probes" / "R-TEST.json"
    common = ["--root", root, "--repo", repo, "--stage", "smoke"]
    hdry = bool(getattr(a, "harness_dry_run", False))
    _header(f"reload: the adapter of {a.run} through the harness", H.git_info(repo))
    codes = []
    for model_key in (key, base_key):
        say(f"\n[{len(codes) + 1}] {model_key} on {g['n_generate']} R-TEST items under {g['generate_condition']}")
        codes.append(harness_run(repo, common + ["--run-name", a.run, "--model", model_key,
                                                 "--conditions", g["generate_condition"],
                                                 "--probe", rtest, "--sample", g["n_generate"]],
                                 dry=hdry, quiet=hdry))
    old = Path(a.base_old) if a.base_old else root / "outputs" / "smoke" / "t1" / base_key / "cache.jsonl"
    say(f"\n[3] {base_key} again on the items of the earlier smoke run ({old})")
    if not old.exists():
        say("  that file does not exist: the comparison of base answers is skipped. Give the "
            "cache.jsonl of the earlier Qwen smoke run with --base-old.")
    else:
        recs = list(H.Cache(old).records.values())
        runs = [r for r in (H.read_json(old.parent / "manifest.json", {}).get("runs") or [])
                if not r.get("dry_run")]
        probe_name = a.base_probe or ((runs[-1].get("probe") or {}).get("file") if runs else "")
        cfg = promptlib.load_conditions(repo / "configs" / "conditions.yaml",
                                        promptlib.load_prompts(repo / "configs" / "prompts.yaml"))
        conds = sorted({r["condition_id"] for r in recs
                        if cfg["conditions"].get(r["condition_id"], {}).get("question_lang") == "bn"
                        and not cfg["conditions"][r["condition_id"]]["forced_choice"]})
        if not probe_name or not conds:
            say("  the probe file or the conditions of that run could not be read from its "
                "manifest: give --base-probe FILE. Skipped.")
        else:
            ids_file = Path(os.environ.get("TMPDIR", "/tmp")) / "ft_base_ids.txt"
            ids_file.write_text("\n".join(sorted({r["qa_id"] for r in recs})) + "\n", encoding="utf-8")
            probe_path = Path(probe_name) if os.path.isabs(str(probe_name)) else root / "probes" / probe_name
            codes.append(harness_run(repo, common + ["--run-name", "ft-base", "--model", base_key,
                                                     "--conditions", ",".join(conds),
                                                     "--probe", probe_path, "--ids-file", ids_file],
                                     dry=hdry, quiet=hdry))
    bad = [c for c in codes if c != 0]
    say("\n  " + ("All harness runs are COMPLETE." if not bad else f"{len(bad)} harness run(s) did not complete: read the output above.")
        + f" Next: python scripts/ft_finetune.py gate-g --run {a.run}")
    return 0 if not bad else 2


def eval_command(a):
    """EXP-R evaluation (Design 7.3): R-TEST x the R conditions x base model and
    adapters, through the harness, one model version after the other."""
    root, repo = Path(a.root), Path(a.repo)
    ft = load_ft(repo / "configs" / "finetune.yaml")
    models = H.load_models(repo / "configs" / "models.yaml")
    seeds = [k for k, m in models.items() if (m.get("adapter") or {}).get("run") in DESIGN_SEEDS]
    keys = [k.strip() for k in a.models.split(",") if k.strip()] if a.models else [ft["base_model"]] + seeds
    for k in keys:
        if k not in models:
            raise Stop(f"--models: '{k}' is not in configs/models.yaml.")
    rtest = Path(a.probe) if a.probe else root / "probes" / "R-TEST.json"
    args = ["--root", root, "--repo", repo, "--stage", a.stage, "--conditions", a.conditions, "--probe", rtest]
    if a.run_name:
        args += ["--run-name", a.run_name]
    if a.english_file:
        args += ["--english-file", a.english_file, "--english-column", a.english_column]
    if a.sample:
        args += ["--sample", a.sample]
    if a.plan_only:
        args += ["--plan-only"]
    _header(f"eval: {a.conditions} on {rtest.name}, stage {a.stage}, models {', '.join(keys)}", H.git_info(repo))
    codes = {}
    for k in keys:
        say(f"\n[{len(codes) + 1}] {k}")
        hdry = bool(getattr(a, "harness_dry_run", False))
        codes[k] = harness_run(repo, args + ["--model", k], dry=hdry, quiet=hdry)
    say("\n[result]")
    _answers_table(root, a.stage, a.run_name, keys)
    bad = [k for k, c in codes.items() if c != 0]
    say("  " + ("COMPLETE for every model version." if not bad else
                f"NOT COMPLETE for: {', '.join(bad)} (read the output above; run the command again)"))
    return 0 if not bad else 2


def status_command(a):
    """Where every run of configs/finetune.yaml stands. Counts only."""
    root, repo = Path(a.root), Path(a.repo)
    ft = load_ft(repo / "configs" / "finetune.yaml")
    say(f"FT scripts {FT_VERSION}   status   {root / 'models'}")
    for name in ft["runs"]:
        run_dir = root / "models" / name
        man = H.read_json(run_dir / H.FT_MANIFEST, None)
        if not man:
            say(f"  {name:<8} not started")
            continue
        ups = read_log(run_dir / "train_log.jsonl", "update", "update")
        done = ups[-1]["update"] if ups else 0
        total = (man.get("plan") or {}).get("total_updates")
        evs = read_log(run_dir / "eval_log.jsonl", "eval", "update")
        best = min(evs, key=lambda r: r["val_loss"]) if evs else None
        if man.get("complete"):
            r = man["result"]
            say(f"  {name:<8} COMPLETE  {r['updates_done']} of {total} updates"
                + (f" (early stopping after update {r['stopped']['update']})" if r.get("stopped") else "")
                + f"; best after update {man['adapter']['from_update']}, validation loss "
                f"{man['adapter']['val_loss']:.4f}; adapter sha256 {man['adapter']['sha256'][:12]}; "
                f"{r['train_seconds'] / 3600:.2f} h on {man['env'].get('gpu')}")
        else:
            say(f"  {name:<8} in progress  {done} of {total} updates logged; {len(evs)} validations"
                + (f", best so far {best['val_loss']:.4f} after update {best['update']}" if best else "")
                + "; run the train command again to continue")
    return 0
