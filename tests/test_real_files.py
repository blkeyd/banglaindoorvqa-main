"""
test_real_files.py  (ASSET-9, stage 2)
Tests of src/scoring.py on the FROZEN files of the tag gate-A:
lexicon/lexicon.csv and lexicon/closed_vocab.csv, and probes/excluded_items.csv.
Standard library only.

Run from the repo root, together with the stage 1 tests:
    python -m unittest discover -s tests -v

Every word used here is read from the frozen files or from scoring.py, which
are in Git already; no question or gold text of the dataset is written here.
The A2 test needs the probe files with text, which live in Drive only. It runs
when the environment variable BIVQA_PROBES names the Drive probes/ folder, and
is skipped otherwise:
    BIVQA_PROBES=/content/drive/MyDrive/BanglaIndoorVQA/probes \
        python -m unittest discover -s tests -v

What is tested (addendum B5 and C5):
    D120   a refusal that ends in the word নয় stays a refusal; নয় alone is no
           number
    D117   the same short answer in both scripts gets the same label; the two
           known gaps of the frozen vocabulary are pinned
    8.3.4  the generic-name rule on stoves, tables and bags
    A2     the scorer's kind, count and room equal closed_kind, gold_count and
           gold_room on every item that is not excluded
    D110   an excluded item is never scored
"""

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import scoring as S                      # noqa: E402
from normalise import normalise          # noqa: E402

LEX = S.load_lexicon(ROOT / "lexicon" / "lexicon.csv")
VOC = S.load_closed_vocab(ROOT / "lexicon" / "closed_vocab.csv")
EXCLUDED = S.load_excluded(ROOT / "probes" / "excluded_items.csv")
NAY = normalise("নয়")


def item(qa_type, gold, qa_id="T0"):
    return {"qa_id": qa_id, "qa_type": qa_type, "answer_bn": gold}


def score(qa_type, gold, cond, answer, **kw):
    return S.score_item(item(qa_type, gold), cond, answer, LEX, VOC, **kw)


def first_bn_form(kind, class_id):
    return VOC.forms[kind][class_id]["bn"][0]


def ends_in_nay(phrase):
    toks = [t.text for t in S.tokenise(phrase)]
    return bool(toks) and toks[-1] == NAY


# --------------------------------------------------------------------------
# The frozen files load as at Gate A (addendum C2)
# --------------------------------------------------------------------------
class FrozenFileTests(unittest.TestCase):
    def test_lexicon_loads_without_warnings(self):
        self.assertEqual(LEX.warnings, [])

    def test_vocabulary_loads_without_warnings(self):
        self.assertEqual(VOC.warnings, [])

    def test_lexicon_counts_as_frozen(self):
        self.assertEqual(len(LEX.objects), 231)
        self.assertEqual(len(LEX.clusters), 16)
        self.assertEqual(sum(1 for o in LEX.objects.values() if o.cluster_id),
                         86)
        self.assertEqual(len(LEX.index.entries), 864)

    def test_vocabulary_counts_as_frozen(self):
        self.assertEqual({k: len(v) for k, v in VOC.classes.items()},
                         {"room": 4, "material": 16, "colour": 14,
                          "number": 29})

    def test_excluded_list_format(self):
        self.assertGreaterEqual(len(EXCLUDED), 28)
        self.assertTrue(set(EXCLUDED.values()) <= set(S.EXCLUDED_REASONS))

    def test_suffix_roles_cover_the_ending_list(self):
        self.assertEqual(set(S.BN_SUFFIX_ROLES), set(S.BN_SUFFIXES))
        self.assertEqual(len(S.BN_SUFFIXES), 16)


# --------------------------------------------------------------------------
# D120: a refusal that ends in নয় stays a refusal
# --------------------------------------------------------------------------
REFUSALS_NAY = [p for p in S.REFUSAL_PHRASES if ends_in_nay(p)]
GOLD_NINE = first_bn_form("number", "9")


class D120Tests(unittest.TestCase):
    def test_the_list_has_refusals_ending_in_nay(self):
        self.assertGreaterEqual(len(REFUSALS_NAY), 2)

    def test_bare_nay_is_not_a_number_form(self):
        self.assertNotIn(NAY, VOC.indexes["number"].entries)

    def test_no_number_form_is_a_word_of_a_refusal_phrase(self):
        words = {t.text for p in S.REFUSAL_PHRASES for t in S.tokenise(p)}
        numbers = set(VOC.indexes["number"].entries)
        self.assertEqual(words & numbers, set())

    def test_refusal_on_count_gold_nine(self):
        for phrase in REFUSALS_NAY:
            for answer in (phrase, phrase + "।"):
                r = score("count", GOLD_NINE, "L-BB", answer)
                self.assertEqual((r["label"], r["strict"], r["lenient"],
                                  r["found_value"], r["refusal"],
                                  r["content_found"]),
                                 (S.WRONG, 0, 0, None, True, False), answer)

    def test_refusal_without_image_condition(self):
        for phrase in REFUSALS_NAY:
            r = score("count", GOLD_NINE, "L-BB0", phrase)
            self.assertTrue(r["refusal"])
            self.assertEqual(r["label"], S.WRONG)

    def test_nay_alone_is_wrong_and_no_refusal(self):
        r = score("count", GOLD_NINE, "L-BB", NAY)
        self.assertEqual((r["label"], r["found_value"], r["refusal"]),
                         (S.WRONG, None, False))

    def test_the_gold_form_itself_is_still_nine(self):
        r = score("count", GOLD_NINE, "L-BB", GOLD_NINE)
        self.assertEqual((r["label"], r["strict"], r["found_value"]),
                         (S.CORRECT, 1, 9))

    def test_refusal_on_every_closed_kind(self):
        golds = [("scene_id", first_bn_form("room", "bedroom")),
                 ("obj_property", first_bn_form("material", "wood")),
                 ("obj_property", first_bn_form("colour", "white")),
                 ("obj_name", LEX.objects[sorted(LEX.objects)[0]].name_bn)]
        for qa_type, gold in golds:
            for phrase in REFUSALS_NAY:
                r = score(qa_type, gold, "L-BB0", phrase)
                self.assertTrue(r["scoreable"], (qa_type, r["gold_status"]))
                self.assertTrue(r["refusal"], (qa_type, phrase))
                self.assertEqual((r["strict"], r["lenient"]), (0, 0))


# --------------------------------------------------------------------------
# D117: the same short answer in both scripts gets the same label
# --------------------------------------------------------------------------
PAIRS = S.parallel_pairs(LEX, VOC)
LEX_PAIRS = [p for p in PAIRS if p[0] == "lexicon"]
VOC_PAIRS = [p for p in PAIRS if p[0] != "lexicon"]
QA_TYPE = {"room": "scene_id", "material": "obj_property",
           "colour": "obj_property", "number": "count"}


class D117Tests(unittest.TestCase):
    def test_pairs_found_in_the_frozen_files(self):
        self.assertEqual(len(LEX_PAIRS), 134)
        self.assertEqual(len(VOC_PAIRS), 22)
        acronyms = {p[1] for p in LEX_PAIRS if p[4] == "acronym"}
        self.assertEqual(len(acronyms), 4)

    def test_no_letter_spelling_without_its_english_acronym(self):
        for oid, obj in LEX.objects.items():
            english = {S._letters_only(e) for e in obj.t1_en}
            for bn in obj.t1_bn:
                spelled = S.spelled_letters(bn)
                if spelled and len(spelled) >= 2:
                    self.assertIn(spelled, english, oid)

    def test_lexicon_pairs_same_label_in_both_scripts(self):
        for _src, oid, bn, en, how in LEX_PAIRS:
            gold = LEX.objects[oid].name_bn
            own = [score("obj_name", gold, "N-BB", bn),
                   score("obj_name", gold, "N-BE", en)]
            crossed = [score("obj_name", gold, "N-BE", bn),
                       score("obj_name", gold, "N-BB", en)]
            for r in own:
                self.assertEqual((r["label"], r["strict"], r["lenient"]),
                                 (S.TIER1, 1, 1), (oid, how))
            for r in crossed:
                self.assertEqual((r["label"], r["strict"], r["lenient"]),
                                 (S.TIER1, 0, 1), (oid, how))

    def test_vocabulary_pairs_same_label_in_both_scripts(self):
        for kind, cid, bn, en, _how in VOC_PAIRS:
            gold = first_bn_form(kind, cid)
            qa_type = QA_TYPE[kind]
            own = [score(qa_type, gold, "L-BB", bn),
                   score(qa_type, gold, "L-BE", en)]
            crossed = [score(qa_type, gold, "L-BE", bn),
                       score(qa_type, gold, "L-BB", en)]
            for r in own:
                self.assertEqual((r["label"], r["strict"], r["lenient"]),
                                 (S.CORRECT, 1, 1), (kind, cid))
            for r in crossed:
                self.assertEqual((r["label"], r["strict"], r["lenient"]),
                                 (S.CORRECT, 0, 1), (kind, cid))

    def test_known_gap_genitive_only_forms(self):
        # The frozen vocabulary lists these two Bangla forms only with a
        # genitive ending. The ending rule (D104) never adds an ending, so the
        # bare Bangla form is scored wrong while the English form of the same
        # class is correct. Gate A froze the file; this goes to the lexicon
        # sensitivity analysis (Design 9.4 item 12), never to an edit.
        gaps = S.genitive_only_forms(VOC)
        self.assertEqual(sorted((k, c) for k, c, _f, _b in gaps),
                         [("material", "ceramic"), ("material", "plastic")])
        for kind, cid, form, bare in gaps:
            r = score("obj_property", form, "L-BB", form)
            self.assertEqual(r["label"], S.CORRECT)
            r = score("obj_property", form, "L-BB", bare)
            self.assertEqual((r["label"], r["found_value"]), (S.WRONG, []))
            for en in VOC.forms[kind][cid]["en"]:
                r = score("obj_property", form, "L-BE", en)
                self.assertEqual((r["label"], r["strict"]), (S.CORRECT, 1))


# --------------------------------------------------------------------------
# Design 8.3 rule 4 (D91): the generic-name rule on stoves, tables and bags
# --------------------------------------------------------------------------
GENERIC = [("চুলা", "stove", 4), ("টেবিল", "table", 6), ("ব্যাগ", "bag", 5)]


def holders_of(word):
    key = LEX.index.key_of(word)
    return [LEX.objects[i] for i in sorted(LEX.objects)
            if key in LEX.objects[i].keys_t2]


class GenericNameTests(unittest.TestCase):
    def test_each_generic_word_is_also_an_object_of_its_own(self):
        # This is what makes precedence matter: the generic word is another
        # object's Tier-1 name.
        for word, _en, _n in GENERIC:
            self.assertIn(LEX.index.key_of(word), LEX.bn_t1_key_to_id, word)

    def test_how_many_objects_list_the_generic_word(self):
        for word, _en, n in GENERIC:
            self.assertEqual(len(holders_of(word)), n, word)

    def test_generic_word_is_tier2_for_every_holder(self):
        for word, en, _n in GENERIC:
            for obj in holders_of(word):
                for cond, answer in (("N-BB", word), ("N-BE", en)):
                    r = score("obj_name", obj.name_bn, cond, answer)
                    self.assertEqual((r["label"], r["strict"], r["lenient"]),
                                     (S.TIER2, 0, 0), (obj.name_id, answer))

    def test_holder_name_is_tier1_for_itself(self):
        for word, _en, _n in GENERIC:
            for obj in holders_of(word):
                r = score("obj_name", obj.name_bn, "N-BB", obj.name_bn)
                self.assertEqual((r["label"], r["strict"]), (S.TIER1, 1))

    def test_generic_and_specific_name_together_is_tier1(self):
        for word, _en, _n in GENERIC:
            for obj in holders_of(word):
                r = score("obj_name", obj.name_bn, "N-BB",
                          word + ", " + obj.name_bn)
                self.assertEqual(r["label"], S.TIER1, obj.name_id)

    def test_generic_object_named_by_itself_is_tier1(self):
        for word, en, _n in GENERIC:
            r = score("obj_name", word, "N-BB", word)
            self.assertEqual((r["label"], r["strict"]), (S.TIER1, 1))
            r = score("obj_name", word, "N-BE", en)
            self.assertEqual((r["label"], r["strict"]), (S.TIER1, 1))

    def test_more_specific_name_for_the_generic_gold(self):
        # Design 8.3 rule 4: a name more specific than the gold is wrong; it
        # is labelled sibling when it is in the gold's cluster, else other
        # wrong (addendum B1).
        for word, _en, _n in GENERIC:
            owner = LEX.objects[LEX.bn_t1_key_to_id[LEX.index.key_of(word)]]
            for obj in holders_of(word):
                r = score("obj_name", word, "N-BB", obj.name_bn)
                same = bool(owner.cluster_id) and \
                    obj.cluster_id == owner.cluster_id
                self.assertEqual(r["label"],
                                 S.SIBLING if same else S.OTHER_WRONG,
                                 obj.name_id)
                self.assertEqual((r["strict"], r["lenient"]), (0, 0))


# --------------------------------------------------------------------------
# D110: excluded items, and the record keys of stage 2
# --------------------------------------------------------------------------
class ExcludedAndRecordTests(unittest.TestCase):
    def test_excluded_item_is_never_scored(self):
        qa_id = sorted(EXCLUDED)[0]
        gold = LEX.objects[sorted(LEX.objects)[0]].name_bn
        for cond, kw in (("N-BB", {}), ("L-BB0", {}), ("N-FCB", {})):
            r = S.score_item(item("obj_name", gold, qa_id), cond, gold, LEX,
                             VOC, excluded=EXCLUDED[qa_id], **kw)
            self.assertEqual((r["gold_status"], r["scoreable"]),
                             (S.EXCLUDED_STATUS, False))
            self.assertEqual(r["gold_hint"], EXCLUDED[qa_id])
            for k in ("label", "strict", "lenient", "fidelity", "refusal",
                      "content_lang_ok", "content_found", "fc_valid",
                      "fc_choice"):
                self.assertIsNone(r[k], k)

    def test_same_keys_for_every_record_with_real_files(self):
        gold = LEX.objects[sorted(LEX.objects)[0]].name_bn
        opts = [LEX.objects[i].name_bn for i in sorted(LEX.objects)[:4]]
        recs = [score("obj_name", gold, "N-BB", gold),
                score("count", GOLD_NINE, "L-BB", "৯"),
                score("scene_id", first_bn_form("room", "kitchen"), "L-BE",
                      "kitchen"),
                score("obj_property", first_bn_form("colour", "red"), "L-BB",
                      ""),
                score("spatial", gold, "L-BB", gold),
                score("obj_name", gold, "N-FCB", "ক", options=opts,
                      correct_index=0),
                score("obj_name", gold, "N-BB", gold, excluded="other")]
        self.assertEqual(len({tuple(r) for r in recs}), 1)
        self.assertEqual(recs[0]["scorer_version"], S.SCORER_VERSION)

    def test_content_found(self):
        gold = first_bn_form("material", "wood")
        self.assertTrue(score("obj_property", gold, "L-BB", gold)
                        ["content_found"])
        self.assertFalse(score("obj_property", gold, "L-BE",
                               "no idea at all")["content_found"])
        self.assertIsNone(score("spatial", gold, "L-BB", gold)
                          ["content_found"])

    def test_summarise_leaves_excluded_records_out(self):
        gold = first_bn_form("material", "wood")
        recs = [score("obj_property", gold, "L-BB", gold),
                score("obj_property", gold, "L-BB", "", excluded="other"),
                score("obj_property", gold, "L-BB", "", excluded="other")]
        row = S.summarise(recs)[0]
        self.assertEqual((row["n"], row["excluded"], row["scoreable"],
                          row["strict"], row["empty"]), (1, 2, 1, 100.0, 0.0))


# --------------------------------------------------------------------------
# Addendum A2 (B5 task 2) on the probe files in Drive
# --------------------------------------------------------------------------
PROBES = os.environ.get("BIVQA_PROBES", "")


class A2Tests(unittest.TestCase):
    def setUp(self):
        if not PROBES:
            self.skipTest("set BIVQA_PROBES to the Drive probes/ folder")

    def test_5c_fields_equal_the_scorer_on_every_item_not_excluded(self):
        seen = 0
        for name in ("P-CORE-CLOSED.json", "R-TEST.json", "PILOT.json"):
            path = Path(PROBES) / name
            self.assertTrue(path.exists(), path)
            for it in S.load_probe(path):
                if it["qa_id"] in EXCLUDED:
                    continue
                gold = S.resolve_gold(it, LEX, VOC)
                self.assertEqual(S.check_5c_fields(it, gold, VOC), [],
                                 "%s %s" % (name, it["qa_id"]))
                seen += 1
        self.assertGreater(seen, 0)


if __name__ == "__main__":
    unittest.main()
