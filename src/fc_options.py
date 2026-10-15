"""
fc_options.py  (ASSET-11)
BanglaIndoorVQA main paper: THE rules of the forced-choice option sets and THE
one reader of the option file. Design v2.1, section 4.7.

Import this file. Never copy a function out of it. The command line is
scripts/asset11_options.py.

What it implements
    Design 4.7     four options per P-CLUST item: the gold, two cluster members,
                   one object of the same room outside the cluster; the order;
                   which items are valid for English forced choice (P-FCE)
    Addendum C4    an item is joined to its object through the scorer
                   (resolve_gold), never by name_id or by the text of the gold
    Addendum C5    English labels are canonical_en as frozen; four different
                   English labels; nothing from the gold's Tier-2 set
    D107           no option text may be read as an option label by the scorer
    D149           the record format the harness reads (qa_id, lang, options)

RULES BUILT IN HERE THAT DESIGN 4.7 DOES NOT STATE (decisions D195 to D202)
    A. Order of candidates: freq_obj_name, highest first. Equal counts are put
       in the order of a seeded rank of the name_id, SHA-256 of
       "42|ASSET-11|tie|<name_id>". The rank is the same for every item, so the
       draft set depends on the gold object only.                        D197
    B. Generic names: a candidate is skipped when one of its Tier-1 names,
       Bangla or English, is a Tier-2 name of the gold. Compared with the
       scorer's own term keys.                                           D197
    C. room_scope: "the same room_scope" means the same value. "any" is a
       value like the other four, so an "any" gold takes an "any" object.
       A value outside the five known ones stops the build.              D198
    D. Too few candidates: a slot that cannot be filled from the cluster
       takes the next object of the same room_scope outside the cluster. If
       that list ends too, the most frequent remaining object of any
       room_scope is used, and counted.                                  D198
    E. One set per item: the Bangla set and the English set hold the same
       four objects in the same four places. The gold is shown under the main
       name of its object (name_bn, canonical_en), also when the item's gold
       is a merged spelling. English sets exist for P-FCE items only.    D199
    F. P-FCE: the gold is en_unique = yes and no two of the four English
       labels share a form (the scorer's english_forms: lower case, articles
       removed, singular, spaces ignored).                               D199
    G. Order shown: the four slots (gold, distractor 1 to 3) are sorted by
       SHA-256 of "42|ASSET-11|order|<qa_id>|<slot number>". No call to
       random, no hash(): the order is the same in every session.        D200
    H. HT-6: a distractor the reviewer rejects is replaced, in its own place,
       by the next candidate of its slot. Nothing else in the set moves.  D201

THE OPTION FILE (Drive probes/fc_options.jsonl), one JSON record per line:
    {"qa_id": "...", "lang": "bn" or "en",
     "options": [four texts in the order shown],        <- read by the harness
     "correct_index": 0 to 3,                           <- position of the gold
     "option_ids": [four name_ids, same order],
     "roles": ["gold" | "sibling" | "outside", ...],
     "in_pfce": true or false, "ht6_changed": true or false}
The harness reads qa_id, lang and options and ignores the rest (D149). The
scorer calls score_forced_choice(answer, options, correct_index, lang).
"""

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import scoring
except ImportError:  # scoring.py sits next to this file in src/
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import scoring
from normalise import normalise

RULES_VERSION = "asset11-v1 2026-10-10"
SEED = 42
LANGS = ("bn", "en")
BN_LABELS = ("ক", "খ", "গ", "ঘ")
EN_LABELS = ("A", "B", "C", "D")
ROOM_SCOPES = ("kitchen", "bedroom", "living room", "bathroom", "any")
ROLE_GOLD, ROLE_SIBLING, ROLE_OUTSIDE = "gold", "sibling", "outside"
RECORD_KEYS = ("qa_id", "lang", "options", "correct_index", "option_ids", "roles",
               "in_pfce", "ht6_changed")
DRAFT_OPTIONS_KEY = "draft_options"   # a draft record has no "options": the harness refuses it
QA_ID_RE = re.compile(r"[A-Za-z]+\d+_\d+_q\d+")


class OptionError(Exception):
    """An option set cannot be built or read. The message says why."""


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(obj):
    return sha256_text(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")))


# --------------------------------------------------------------------------
# 1. The seeded ranks (rules A and G)
# --------------------------------------------------------------------------

def tie_rank(name_id):
    """Rule A: the place of an object among objects with the same frequency."""
    return sha256_text("%d|ASSET-11|tie|%s" % (SEED, name_id))


def shown_order(qa_id):
    """Rule G: the slot numbers in the order shown. Slot 0 is the gold, slots
    1 to 3 are the distractors. shown_order(q)[p] is the slot at place p."""
    return sorted(range(4), key=lambda k: sha256_text(
        "%d|ASSET-11|order|%s|%d" % (SEED, qa_id, k)))


# --------------------------------------------------------------------------
# 2. Reading one lexicon object
# --------------------------------------------------------------------------

def freq_of(obj):
    raw = str(obj.row.get("freq_obj_name", "")).strip()
    if not raw.isdigit():
        raise OptionError("lexicon: %s (%s) has no whole number in freq_obj_name: %r"
                          % (obj.name_id, obj.name_bn, raw))
    return int(raw)


def scope_of(obj):
    raw = str(obj.row.get("room_scope", "")).strip().lower()
    if raw not in ROOM_SCOPES:
        raise OptionError(
            "lexicon: %s (%s) has room_scope %r. Rule C knows only: %s. A cell with "
            "several rooms needs a new decision row before any option set is built."
            % (obj.name_id, obj.name_bn, raw, ", ".join(ROOM_SCOPES)))
    return raw


def _shown_text(obj, column):
    """The text put in front of a model: the frozen cell, unchanged."""
    text = str(obj.row.get(column, "")).strip()
    if not text:
        raise OptionError("lexicon: %s has an empty %s" % (obj.name_id, column))
    if normalise(text) != text or "\n" in text or "|" in text or ";" in text:
        raise OptionError(
            "lexicon: %s of %s is %r. An option text must be one name in normalise() "
            "form (no capital letter, no end mark, no hidden character)."
            % (column, obj.name_id, text))
    return text


def text_bn(obj):
    return _shown_text(obj, "name_bn")


def text_en(obj):
    return _shown_text(obj, "canonical_en")


def is_generic_of(gold, cand):
    """Rule B: is a name of `cand` a generic (Tier-2) name of `gold`?"""
    return bool(cand.keys_t1 & gold.keys_t2)


def is_kind_of(gold, cand):
    """The other direction: `gold` is a generic name of `cand`, so `cand` is a
    more specific kind of the gold. Design 4.7 keeps such a distractor (the
    scorer counts a more specific name as a sibling, Design 8.3 rule 4). It is
    only marked for the HT-6 reviewer."""
    return bool(gold.keys_t1 & cand.keys_t2)


def english_labels_differ(labels):
    """Rule F: no two labels share a form."""
    forms = [scoring.english_forms(x) for x in labels]
    if any(not f for f in forms):
        return False
    return all(not (forms[a] & forms[b])
               for a in range(len(forms)) for b in range(a + 1, len(forms)))


# --------------------------------------------------------------------------
# 3. One option set
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class OptionSet:
    qa_id: str
    gold_id: str
    draft: tuple          # the three distractors before HT-6, slots 1 to 3
    slots: tuple          # the three distractors after HT-6
    order: tuple          # shown_order(qa_id)
    rejected: tuple = ()  # HT-6 rejections that changed the set
    moot: tuple = ()      # HT-6 rejections that changed nothing

    def ids_shown(self):
        by_slot = (self.gold_id,) + self.slots
        return [by_slot[k] for k in self.order]

    @property
    def correct_index(self):
        return self.order.index(0)

    @property
    def changed(self):
        return self.slots != self.draft


class Builder:
    """Builds option sets from the frozen lexicon. Holds no item text."""

    def __init__(self, lexicon):
        self.lex = lexicon
        self.objects = lexicon.objects
        self._queues = {}
        recomputed = scoring.compute_en_unique(self.objects)
        differ = [i for i, o in self.objects.items()
                  if recomputed[i] is not None
                  and (str(o.row.get("en_unique", "")).strip().lower() == "yes") != recomputed[i]]
        if differ:
            raise OptionError(
                "lexicon: en_unique of %d object(s) is not what the scorer computes from "
                "the English names (first: %s). P-FCE cannot be drawn from this file."
                % (len(differ), differ[0]))

    # ---- the gold (C4) ----------------------------------------------------
    def gold_of(self, item):
        """The object of an item, through the scorer and nothing else."""
        if item.get("qa_type") not in scoring.NAMING_TYPES:
            raise OptionError("%s: type %r is not a naming item; forced choice is for "
                              "P-CLUST items only." % (item.get("qa_id"), item.get("qa_type")))
        gold = scoring.resolve_gold(item, self.lex, None)
        if gold.status != "ok" or not gold.name_id:
            raise OptionError(
                "%s: the scorer cannot join the gold answer to one lexicon object (%s%s). "
                "No option set is built from a guess."
                % (item.get("qa_id"), gold.status, ", " + gold.hint if gold.hint else ""))
        obj = self.objects[gold.name_id]
        if not obj.cluster_id:
            raise OptionError("%s: its object %s is in no cluster." % (item.get("qa_id"), obj.name_id))
        return obj

    # ---- candidates (rules A to D) -----------------------------------------
    def queues(self, gold_id):
        """The candidate lists of one gold object.
        sib      other members of its cluster, best first
        out      objects outside the cluster: same room_scope first (n_same of
                 them), then every other object (rule D, last resort)
        generic  objects skipped by rule B"""
        if gold_id in self._queues:
            return self._queues[gold_id]
        gold = self.objects[gold_id]
        scope = scope_of(gold)
        sib, same, other, generic = [], [], [], []
        for oid, obj in self.objects.items():
            if oid == gold_id:
                continue
            if is_generic_of(gold, obj):
                generic.append(oid)
            elif gold.cluster_id and obj.cluster_id == gold.cluster_id:
                sib.append(oid)
            elif scope_of(obj) == scope:
                same.append(oid)
            else:
                other.append(oid)

        def key(oid):
            return (-freq_of(self.objects[oid]), tie_rank(oid))

        q = {"sib": sorted(sib, key=key),
             "out": sorted(same, key=key) + sorted(other, key=key),
             "n_same": len(same), "generic": sorted(generic, key=key)}
        self._queues[gold_id] = q
        return q

    def fill(self, gold_id, rejected=()):
        """Return (draft, slots): the three distractors without and with the
        rejections. A rejected distractor is replaced in its own slot (rule H).
        The result depends on the set of rejections only, not on their order."""
        q = self.queues(gold_id)
        rejected = set(rejected)
        cluster_q = q["sib"] + q["out"]
        slot_q = (cluster_q, cluster_q, q["out"])

        def first(queue, used, skip):
            for cand in queue:
                if cand not in used and cand not in skip:
                    return cand
            raise OptionError("%s: no candidate is left for a slot." % gold_id)

        used = {gold_id}
        draft = []
        for queue in slot_q:
            cand = first(queue, used, ())
            draft.append(cand)
            used.add(cand)
        slots = [None if c in rejected else c for c in draft]
        used = {gold_id} | {c for c in slots if c}
        for k, queue in enumerate(slot_q):
            if slots[k] is None:
                slots[k] = first(queue, used, rejected)
                used.add(slots[k])
        return tuple(draft), tuple(slots)

    def prune(self, gold_id, rejected):
        """Split rejections into those that change the set and those that do
        not (a rejected replacement whose own cause was taken back)."""
        q = self.queues(gold_id)
        universe = set(q["sib"]) | set(q["out"])
        bad = [r for r in rejected if r not in universe]
        if bad:
            raise OptionError("%s: %s cannot be rejected: it is the gold, a generic name "
                              "of the gold, or not a lexicon object." % (gold_id, bad[0]))
        keep, moot = sorted(set(rejected)), []
        changed = True
        while changed:
            changed = False
            for r in keep:
                rest = [x for x in keep if x != r]
                if r not in self.fill(gold_id, rest)[1]:
                    keep, changed = rest, True
                    moot.append(r)
                    break
        return keep, moot

    def make(self, item, rejected=()):
        gold = self.gold_of(item)
        keep, moot = self.prune(gold.name_id, rejected)
        draft, slots = self.fill(gold.name_id, keep)
        return OptionSet(item["qa_id"], gold.name_id, draft, slots,
                         tuple(shown_order(item["qa_id"])), tuple(keep), tuple(moot))

    # ---- what is written -----------------------------------------------------
    def role_of(self, gold_id, oid):
        if oid == gold_id:
            return ROLE_GOLD
        gold = self.objects[gold_id]
        same = gold.cluster_id and self.objects[oid].cluster_id == gold.cluster_id
        return ROLE_SIBLING if same else ROLE_OUTSIDE

    def in_pfce(self, gold_id, ids):
        """Rule F, on the four objects of a set."""
        gold = self.objects[gold_id]
        if str(gold.row.get("en_unique", "")).strip().lower() != "yes":
            return False
        return english_labels_differ([text_en(self.objects[i]) for i in ids])

    def records(self, s):
        """(Bangla record, English record or None) of one set (rule E)."""
        ids = s.ids_shown()
        pfce = self.in_pfce(s.gold_id, ids)
        base = {"qa_id": s.qa_id, "lang": "bn",
                "options": [text_bn(self.objects[i]) for i in ids],
                "correct_index": s.correct_index, "option_ids": list(ids),
                "roles": [self.role_of(s.gold_id, i) for i in ids],
                "in_pfce": pfce, "ht6_changed": s.changed}
        en = None
        if pfce:
            en = dict(base, lang="en", options=[text_en(self.objects[i]) for i in ids])
        return base, en


# --------------------------------------------------------------------------
# 4. The option file: writing and THE reader
# --------------------------------------------------------------------------

def dump_record(rec, options_key="options"):
    out = {}
    for k in RECORD_KEYS:
        out[options_key if k == "options" else k] = rec[k]
    return json.dumps(out, ensure_ascii=False)


def _check_record(rec, where):
    if not isinstance(rec, dict):
        raise OptionError("%s: not a JSON object" % where)
    missing = [k for k in RECORD_KEYS if k not in rec]
    if missing:
        hint = ""
        if DRAFT_OPTIONS_KEY in rec:
            hint = (" This is a DRAFT file (before HT-6). Runs and scoring use "
                    "probes/fc_options.jsonl, written by the apply command.")
        raise OptionError("%s: missing %s.%s" % (where, ", ".join(missing), hint))
    if rec["lang"] not in LANGS:
        raise OptionError("%s: lang must be bn or en, found %r" % (where, rec["lang"]))
    opts = rec["options"]
    if not isinstance(opts, list) or len(opts) != 4:
        raise OptionError("%s: needs exactly 4 options (Design 4.7)" % where)
    for o in opts:
        if not isinstance(o, str) or not o or o != o.strip() or "\n" in o or "\r" in o:
            raise OptionError("%s: an option is empty, has outer spaces or a line break" % where)
    ci = rec["correct_index"]
    if isinstance(ci, bool) or ci not in (0, 1, 2, 3):
        raise OptionError("%s: correct_index must be 0, 1, 2 or 3" % where)
    ids, roles = rec["option_ids"], rec["roles"]
    if not isinstance(ids, list) or len(ids) != 4 or len(set(ids)) != 4:
        raise OptionError("%s: option_ids must be 4 different name_ids" % where)
    if not isinstance(roles, list) or len(roles) != 4 or roles.count(ROLE_GOLD) != 1 \
            or roles[ci] != ROLE_GOLD \
            or any(r not in (ROLE_GOLD, ROLE_SIBLING, ROLE_OUTSIDE) for r in roles):
        raise OptionError("%s: roles must hold one 'gold', at correct_index" % where)


def read_option_sets(path):
    """THE reader of probes/fc_options.jsonl. Returns {(qa_id, lang): record}.
    Use it in the scorer and the statistics; the harness has its own reader,
    which takes qa_id, lang and options from the same lines."""
    table = {}
    with open(str(path), "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            where = "%s line %d" % (Path(str(path)).name, n)
            try:
                rec = json.loads(line)
            except ValueError:
                raise OptionError("%s: not JSON" % where)
            _check_record(rec, where)
            key = (rec["qa_id"], rec["lang"])
            if key in table:
                raise OptionError("%s: a second set for %s in %s" % (where, key[0], key[1]))
            table[key] = rec
    if not table:
        raise OptionError("%s: no option set" % path)
    return table


def options_for(table, qa_id, lang):
    """(options, correct_index) for score_forced_choice. Fails loudly."""
    try:
        rec = table[(qa_id, lang)]
    except KeyError:
        raise OptionError("no %s option set for %s (an English set exists for P-FCE "
                          "items only)" % (lang, qa_id))
    return rec["options"], rec["correct_index"]


def read_ids(path):
    """qa_ids of a repo id list (probes/*.ids.txt), in file order."""
    with open(str(path), "r", encoding="utf-8-sig") as fh:
        return QA_ID_RE.findall(fh.read())


# --------------------------------------------------------------------------
# 5. Checks on a table of records (a draft in memory, or the final file)
#
# The checks do not call the functions that built the sets. They read the
# lexicon rows and ask the scorer, so a mistake in the builder cannot hide.
# --------------------------------------------------------------------------

def scorer_round_trip(table):
    """D107 and Design 8.5 on every set: an answer that is one option's text
    must be read as that option, by its text and not as a label; and each
    label must be read as its place. Returns the list of failures."""
    bad = []
    for (qa_id, lang), rec in table.items():
        opts, ci = rec["options"], rec["correct_index"]
        labels = BN_LABELS if lang == "bn" else EN_LABELS
        for k in range(4):
            res = scoring.score_forced_choice(opts[k], opts, ci, lang)
            if res["fc_choice"] != k or res["fc_method"] != "text":
                bad.append((qa_id, lang, "text of option %d read as %r by %s"
                            % (k + 1, res["fc_choice"], res["fc_method"])))
            res = scoring.score_forced_choice(labels[k], opts, ci, lang)
            if res["fc_choice"] != k or res["fc_method"] != "label" \
                    or (res["strict"] == 1) != (k == ci):
                bad.append((qa_id, lang, "label %s read as %r" % (labels[k], res["fc_choice"])))
    return bad


def open_label(lexicon, item, text, lang):
    """The label the scorer gives when `text` is an open answer to the item
    (Design 8.3): tier1, tier2, sibling, ambiguous, english_copy, other_wrong."""
    rec = scoring.score_item(item, "N-BB" if lang == "bn" else "N-BE", text, lexicon, None)
    return rec["label"], rec["strict"]


def _frequency_rule_broken(lex, gold, ids, rejected):
    """Design 4.7, stated on the result: cluster members come before outside
    objects, outside objects have the gold's room_scope, and nothing that was
    left out is more frequent than something that was taken. Read from the
    lexicon rows only. Returns a short reason, or ''."""
    objs = lex.objects
    chosen = [i for i in ids if i != gold.name_id]

    def usable(i):
        return i != gold.name_id and i not in rejected and not (objs[i].keys_t1 & gold.keys_t2)

    members = [i for i in lex.clusters.get(gold.cluster_id, ()) if usable(i)]
    sib = [i for i in chosen if objs[i].cluster_id == gold.cluster_id]
    out = [i for i in chosen if objs[i].cluster_id != gold.cluster_id]
    sib_left = [i for i in members if i not in chosen]
    if len(sib) > 2:
        return "more than two cluster members"
    if sib_left and len(sib) < 2:
        return "a cluster member was left out"
    if sib and sib_left and min(freq_of(objs[i]) for i in sib) < max(freq_of(objs[i]) for i in sib_left):
        return "a more frequent cluster member was left out"
    pool = [i for i in objs if usable(i) and objs[i].cluster_id != gold.cluster_id
            and scope_of(objs[i]) == scope_of(gold)]
    out_left = [i for i in pool if i not in chosen]
    if out_left and any(i not in pool for i in out):
        return "an outside object of another room_scope"
    if out and out_left and min(freq_of(objs[i]) for i in out) < max(freq_of(objs[i]) for i in out_left):
        return "a more frequent outside object was left out"
    return ""


def check_sets(builder, items, table, rejected_by_item=None):
    """Every check of ASSET-11 on a table {(qa_id, lang): record}.
    Returns {"checks": [(key, text, passed)], "numbers": {...}}.
    rejected_by_item: {qa_id: ids rejected in HT-6}; none of them may be shown."""
    lex = builder.lex
    objs = lex.objects
    n = len(items)
    names = ("bn", "four", "gold_once", "gold_tier1", "merged", "generic_keys", "second_right",
             "repeat", "text_wrong", "role_wrong", "order_wrong", "en", "en_missing", "en_extra",
             "en_differs", "en_same_label", "en_second_right", "flag_wrong", "other_scope",
             "freq_rule", "rejected_shown", "cluster_differs", "kinds")
    cnt = dict.fromkeys(names, 0)
    first_bad = {}
    pos_bn, pos_en = [0, 0, 0, 0], [0, 0, 0, 0]
    pfce_by_cluster, unique_by_cluster, items_by_cluster, shape = {}, {}, {}, {}
    pfce_ids, item_ids = [], set()

    def note(key, qa, why=""):
        cnt[key] += 1
        first_bad.setdefault(key, "%s%s" % (qa, " (" + why + ")" if why else ""))

    for it in items:
        qa = it["qa_id"]
        item_ids.add(qa)
        gold = builder.gold_of(it)                 # the scorer's resolve_gold (C4)
        cl = gold.cluster_id
        unique = str(gold.row.get("en_unique", "")).strip().lower() == "yes"
        items_by_cluster[cl] = items_by_cluster.get(cl, 0) + 1
        unique_by_cluster[cl] = unique_by_cluster.get(cl, 0) + (1 if unique else 0)
        if "cluster_id" in it and it["cluster_id"] != cl:
            note("cluster_differs", qa)
        if normalise(it["answer_bn"]) != gold.name_bn:
            cnt["merged"] += 1
        rec = table.get((qa, "bn"))
        if rec is None:
            continue
        cnt["bn"] += 1
        ids, opts, ci = rec["option_ids"], rec["options"], rec["correct_index"]
        rejected = set((rejected_by_item or {}).get(qa, ()))
        if len(opts) == 4 and len(ids) == 4:
            cnt["four"] += 1
        if ids.count(gold.name_id) == 1 and ids[ci] == gold.name_id:
            cnt["gold_once"] += 1
        if any(i not in objs for i in ids):
            note("text_wrong", qa, "an unknown name_id")
            continue
        if any(opts[k] != str(objs[ids[k]].row.get("name_bn", "")).strip() for k in range(4)):
            note("text_wrong", qa)
        if len(set(ids)) != 4 or len({lex.index.key_of(o) for o in opts}) != 4:
            note("repeat", qa)
        if opts[ci] == gold.name_bn and open_label(lex, it, opts[ci], "bn") == (scoring.TIER1, 1):
            cnt["gold_tier1"] += 1
        dis = [k for k in range(4) if k != ci]
        if any(objs[ids[k]].keys_t1 & gold.keys_t2 for k in dis):
            note("generic_keys", qa)
        for k in dis:
            label = open_label(lex, it, opts[k], "bn")[0]
            want = scoring.SIBLING if objs[ids[k]].cluster_id == cl else scoring.OTHER_WRONG
            if label not in (scoring.SIBLING, scoring.OTHER_WRONG):
                note("second_right", qa, "%s is scored %s" % (opts[k], label))
            elif label != want or rec["roles"][k] != \
                    (ROLE_SIBLING if want == scoring.SIBLING else ROLE_OUTSIDE):
                note("role_wrong", qa)
        if rec["roles"][ci] != ROLE_GOLD:
            note("role_wrong", qa)
        cnt["kinds"] += any(gold.keys_t1 & objs[ids[k]].keys_t2 for k in dis)
        if ci != shown_order(qa).index(0):
            note("order_wrong", qa)
        pos_bn[ci] += 1
        if any(objs[ids[k]].cluster_id != cl and scope_of(objs[ids[k]]) != scope_of(gold) for k in dis):
            note("other_scope", qa)
        why = _frequency_rule_broken(lex, gold, ids, rejected)
        if why:
            note("freq_rule", qa, why)
        key = "%d from the cluster + %d from outside" % (
            rec["roles"].count(ROLE_SIBLING), rec["roles"].count(ROLE_OUTSIDE))
        shape[key] = shape.get(key, 0) + 1
        if rejected & set(ids):
            note("rejected_shown", qa)
        labels = [str(objs[i].row.get("canonical_en", "")).strip() for i in ids]
        forms = [scoring.english_forms(x) for x in labels]
        want_en = unique and all(forms) and all(
            not (forms[x] & forms[y]) for x in range(4) for y in range(x + 1, 4))
        if rec["in_pfce"] != want_en:
            note("flag_wrong", qa)
        en = table.get((qa, "en"))
        if want_en and en is None:
            note("en_missing", qa)
        if not want_en and en is not None:
            note("en_extra", qa)
        if en is not None:
            cnt["en"] += 1
            if en["option_ids"] != ids or en["correct_index"] != ci or en["roles"] != rec["roles"] \
                    or en["options"] != labels:
                note("en_differs", qa)
            if len({lex.index.key_of(o) for o in en["options"]}) != 4:
                note("en_same_label", qa)
            for k in range(4):
                label, strict = open_label(lex, it, en["options"][k], "en")
                if (k == ci) != (label == scoring.TIER1) or (k == ci and strict != 1) \
                        or label == scoring.TIER2:
                    note("en_second_right", qa, "%s is scored %s" % (en["options"][k], label))
            if want_en:
                pfce_ids.append(qa)
                pfce_by_cluster[cl] = pfce_by_cluster.get(cl, 0) + 1
                pos_en[ci] += 1
    stray = sum(1 for (qa, _lang) in table if qa not in item_ids)
    trips = scorer_round_trip(table)
    n_unique = sum(unique_by_cluster.values())
    n_pfce = len(pfce_ids)
    lo = n / 4.0 - 4 * (n * 0.1875) ** 0.5       # 4 standard deviations of a fair draw
    hi = n / 4.0 + 4 * (n * 0.1875) ** 0.5
    over = [c for c in pfce_by_cluster if pfce_by_cluster[c] > unique_by_cluster.get(c, 0)]

    def first(*keys):
        hits = [first_bad[k] for k in keys if k in first_bad]
        return "" if not hits else "; first: %s" % hits[0]

    checks = [
        ("count", "%d Bangla option sets for %d items, %d with exactly 4 options"
         % (cnt["bn"], n, cnt["four"]), cnt["bn"] == n and cnt["four"] == n and stray == 0),
        ("gold_once", "the gold is in its set exactly once, at correct_index: %d of %d"
         % (cnt["gold_once"], n), cnt["gold_once"] == n),
        ("gold_name", "the gold is shown under the main name of the object the scorer "
         "resolves, and the scorer reads that text as Tier 1: %d of %d (%d golds are merged "
         "spellings)" % (cnt["gold_tier1"], n, cnt["merged"]), cnt["gold_tier1"] == n),
        ("generic", "no distractor is a generic (Tier-2) name of its gold: %d sets by the "
         "lexicon terms; and the scorer reads every distractor text as a wrong name (sibling "
         "or other), never as Tier 1 or Tier 2: %d sets break this%s"
         % (cnt["generic_keys"], cnt["second_right"], first("generic_keys", "second_right")),
         cnt["generic_keys"] == 0 and cnt["second_right"] == 0),
        ("repeat", "no option is repeated inside a set: %d sets break this%s"
         % (cnt["repeat"], first("repeat")), cnt["repeat"] == 0),
        ("texts", "every option text is the name_bn of its object, and every role fits what "
         "the scorer says (sibling or other): %d and %d sets break this%s"
         % (cnt["text_wrong"], cnt["role_wrong"], first("text_wrong", "role_wrong")),
         cnt["text_wrong"] == 0 and cnt["role_wrong"] == 0),
        ("frequency", "frequency rule of Design 4.7 (cluster members first, the gold's "
         "room_scope outside, nothing more frequent left out): %d sets break this%s"
         % (cnt["freq_rule"] + cnt["other_scope"], first("freq_rule", "other_scope")),
         cnt["freq_rule"] == 0 and cnt["other_scope"] == 0),
        ("position", "gold position: %s (a fair draw gives about %d each)"
         % (", ".join("%s %d" % (BN_LABELS[k], pos_bn[k]) for k in range(4)), round(n / 4.0)),
         cnt["order_wrong"] == 0 and all(lo <= c <= hi for c in pos_bn)),
        ("pfce", "P-FCE: %d items; at most %d (the en_unique items of P-CLUST)"
         % (n_pfce, n_unique), 0 < n_pfce <= n_unique and not over),
        ("english", "English sets: %d, for P-FCE items only (%d missing, %d too many), each "
         "with the same four objects in the same places as the Bangla set (%d differ), four "
         "different English labels (%d repeat one), and only the gold read as Tier 1 by the "
         "scorer (%d break this)%s"
         % (cnt["en"], cnt["en_missing"], cnt["en_extra"], cnt["en_differs"],
            cnt["en_same_label"], cnt["en_second_right"],
            first("en_missing", "en_extra", "en_differs", "en_same_label", "en_second_right")),
         cnt["en"] == n_pfce and cnt["en_missing"] == 0 and cnt["en_extra"] == 0
         and cnt["en_differs"] == 0 and cnt["en_same_label"] == 0
         and cnt["en_second_right"] == 0 and cnt["flag_wrong"] == 0),
        ("round_trip", "scorer round trip (Design 8.5, D107): each option text is read as "
         "that option and each label as its place, in %d sets: %d failures%s"
         % (len(table), len(trips), "" if not trips else " (first: %s %s, %s)" % trips[0]),
         not trips),
    ]
    if rejected_by_item is not None:
        checks.append(("rejected", "no option rejected in HT-6 is still shown: %d sets break "
                       "this%s" % (cnt["rejected_shown"], first("rejected_shown")),
                       cnt["rejected_shown"] == 0))
    if cnt["cluster_differs"]:
        checks.append(("cluster", "the probe file's cluster_id is the cluster of the resolved "
                       "object: %d items differ%s" % (cnt["cluster_differs"],
                                                      first("cluster_differs")), False))
    return {"checks": checks,
            "numbers": {"n_items": n, "n_bn": cnt["bn"], "n_en": cnt["en"], "n_pfce": n_pfce,
                        "n_en_unique": n_unique, "pfce_ids": pfce_ids,
                        "gold_position_bn": pos_bn, "gold_position_pfce": pos_en,
                        "pfce_by_cluster": pfce_by_cluster,
                        "en_unique_by_cluster": unique_by_cluster,
                        "items_by_cluster": items_by_cluster, "shape": shape,
                        "n_merged_gold": cnt["merged"], "n_kind_of_gold": cnt["kinds"],
                        "n_changed": sum(1 for (_q, lang), r in table.items()
                                         if lang == "bn" and r["ht6_changed"])}}


# --------------------------------------------------------------------------
# 6. configs/conditions.yaml: the one line of P-FCE
# --------------------------------------------------------------------------

_PFCE_LINE = re.compile(r"^(\s+)P-FCE:\s*\{[^}]*\}.*$")


def patch_conditions_text(text, file_name, n):
    """Return conditions.yaml with the P-FCE line set to file_name and n.
    Every other character of the file stays as it is."""
    eol = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(eol)
    hits = [i for i, line in enumerate(lines) if _PFCE_LINE.match(line)]
    if len(hits) != 1:
        raise OptionError("configs/conditions.yaml: expected exactly one line that starts "
                          "with 'P-FCE: {', found %d. Edit the line by hand." % len(hits))
    indent = _PFCE_LINE.match(lines[hits[0]]).group(1)
    lines[hits[0]] = ("%sP-FCE:         {file: %s,         expected_n: %d}    "
                      "# written by ASSET-11 after HT-6 (Design 4.7)" % (indent, file_name, n))
    return eol.join(lines)
