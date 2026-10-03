"""
test_scoring.py  (ASSET-9)
Unit tests for src/scoring.py. Standard library only, no install needed.

Run from the repo root:
    python -m unittest discover -s tests -v

Every word in these tests comes from the made-up files in tests/fixtures.
They are not lexicon data and not dataset text, so this file may live in Git.
Each row of a table below becomes one test.
"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import scoring as S                      # noqa: E402
from normalise import normalise          # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"
LEX = S.load_lexicon(FIX / "lexicon_FIXTURE.csv")
VOC = S.load_closed_vocab(FIX / "closed_vocab_FIXTURE.csv")
PLAIN = S.MatchSettings(spacing_insensitive=False, suffix_tolerance=False)
LEX_PLAIN = S.load_lexicon(FIX / "lexicon_FIXTURE.csv", PLAIN)
VOC_PLAIN = S.load_closed_vocab(FIX / "closed_vocab_FIXTURE.csv", PLAIN)

T1, T2, SIB, AMB, COPY, OTHER = (S.TIER1, S.TIER2, S.SIBLING, S.AMBIGUOUS,
                                 S.ENGLISH_COPY, S.OTHER_WRONG)
OK, BAD, INV = S.CORRECT, S.WRONG, S.INVALID
OPT_BN = ["হাঁড়ি", "পাতিল", "দেকচি", "বালতি"]
OPT_EN = ["pot", "bucket", "door", "chair"]


def item(qa_type, gold):
    return {"qa_id": "T0", "qa_type": qa_type, "answer_bn": gold}


def score(qa_type, gold, cond, answer, lex=LEX, voc=VOC, **kw):
    return S.score_item(item(qa_type, gold), cond, answer, lex, voc, **kw)


def add_cases(cls, cases, make):
    """Turn each table row into one test method on cls."""
    for case in cases:
        name = "test_" + case[0]
        assert not hasattr(cls, name), "duplicate test name: " + name
        setattr(cls, name, make(*case[1:]))


# --------------------------------------------------------------------------
# Naming items (Design 8.3)
# name, gold, condition, answer, label, strict, lenient, fidelity
# --------------------------------------------------------------------------
NAMING = [
    # Tier 1 in Bangla
    ("bn_t1_plain", "বালতি", "N-BB", "বালতি", T1, 1, 1, True),
    ("bn_t1_answer_with_dari", "বালতি", "N-BB", "বালতি।", T1, 1, 1, True),
    ("bn_t1_gold_with_dari", "বালতি।", "N-BB", "বালতি", T1, 1, 1, True),
    ("bn_t1_lead_in_eti_ekti", "বালতি", "N-BB", "এটি একটি বালতি।",
     T1, 1, 1, True),
    ("bn_t1_lead_in_eta", "বালতি", "N-BB", "এটা বালতি", T1, 1, 1, True),
    ("bn_t1_lead_in_eti", "বালতি", "N-BB", "এটি বালতি", T1, 1, 1, True),
    ("bn_t1_classifier_ending", "বালতি", "N-BB", "বালতিটি", T1, 1, 1, True),
    ("bn_t1_genitive_ending", "বালতি", "N-BB", "বালতির", T1, 1, 1, True),
    ("bn_t1_two_endings", "বালতি", "N-BB", "বালতিটির", T1, 1, 1, True),
    ("bn_t1_with_adjective", "বালতি", "N-BB", "প্লাস্টিকের বালতি",
     T1, 1, 1, True),
    ("bn_t1_two_words", "পড়ার টেবিল", "N-BB", "পড়ার টেবিল", T1, 1, 1, True),
    ("bn_t1_two_words_ending", "পড়ার টেবিল", "N-BB", "পড়ার টেবিলটি",
     T1, 1, 1, True),
    ("bn_t1_joined_spelling", "সুইচ বোর্ড", "N-BB", "সুইচবোর্ড",
     T1, 1, 1, True),
    ("bn_t1_hyphen_spelling", "সুইচ বোর্ড", "N-BB", "সুইচ-বোর্ড",
     T1, 1, 1, True),
    ("bn_t1_split_spelling", "ল্যাপটপ", "N-BB", "ল্যাপ টপ", T1, 1, 1, True),
    ("bn_t1_listed_variant", "টেলিভিশন", "N-BB", "টিভি", T1, 1, 1, True),
    ("bn_t1_precomposed_letter", "চেয়ার", "N-BB", "চে\u09dfার",
     T1, 1, 1, True),
    ("bn_t1_markdown_bold", "বালতি", "N-BB", "**বালতি**", T1, 1, 1, True),
    ("bn_t1_curly_quotes", "বালতি", "N-BB", "\u201cবালতি\u201d",
     T1, 1, 1, True),
    ("bn_t1_with_its_own_generic", "বালতি", "N-BB", "বালতি, একটি পাত্র",
     T1, 1, 1, True),
    ("bn_t1_specificity_condition", "বালতি", "N-BBs", "বালতি",
     T1, 1, 1, True),
    ("bn_t1_english_question", "বালতি", "L-EB", "বালতি", T1, 1, 1, True),
    ("bn_t1_no_image", "বালতি", "L-BB0", "বালতি", T1, 1, 1, True),
    ("bn_t1_repair_condition", "বালতি", "R-BB", "বালতি", T1, 1, 1, True),
    # Tier 1 in English
    ("en_t1_plain", "বালতি", "N-BE", "bucket", T1, 1, 1, True),
    ("en_t1_capital_full_stop", "বালতি", "N-BE", "Bucket.", T1, 1, 1, True),
    ("en_t1_lead_in_it_is_a", "বালতি", "N-BE", "It is a bucket",
     T1, 1, 1, True),
    ("en_t1_lead_in_this_is_a", "বালতি", "N-BE", "This is a bucket.",
     T1, 1, 1, True),
    ("en_t1_lead_in_its_a", "বালতি", "N-BE", "It's a bucket",
     T1, 1, 1, True),
    ("en_t1_curly_apostrophe", "বালতি", "N-BE", "It\u2019s a bucket",
     T1, 1, 1, True),
    ("en_t1_article", "বালতি", "N-BE", "a bucket", T1, 1, 1, True),
    ("en_t1_plural", "বালতি", "N-BE", "buckets", T1, 1, 1, True),
    ("en_t1_listed_variant", "বালতি", "N-BE", "pail", T1, 1, 1, True),
    ("en_t1_with_adjectives", "বালতি", "N-BE", "a red plastic bucket",
     T1, 1, 1, True),
    ("en_t1_two_words", "পড়ার টেবিল", "N-BE", "study table",
     T1, 1, 1, True),
    ("en_t1_joined_spelling", "সুইচ বোর্ড", "N-BE", "switchboard",
     T1, 1, 1, True),
    ("en_t1_shared_english_name", "হাঁড়ি", "N-BE", "pot", T1, 1, 1, True),
    ("en_t1_acronym", "এসি", "N-BE", "AC", T1, 1, 1, True),
    ("en_t1_english_question", "বালতি", "L-EE", "bucket", T1, 1, 1, True),
    ("en_t1_repair_condition", "বালতি", "R-EE", "bucket", T1, 1, 1, True),
    # Right object, wrong language (8.3 rule 3)
    ("wrong_lang_english_for_bangla", "ফ্রিজ", "N-BB", "fridge",
     T1, 0, 1, False),
    ("wrong_lang_bangla_for_english", "ফ্রিজ", "N-BE", "ফ্রিজ",
     T1, 0, 1, False),
    ("wrong_lang_shared_english_name", "হাঁড়ি", "N-BB", "pot",
     T1, 0, 1, False),
    # Both languages in one answer: strict follows the name, not the letters
    ("both_langs_bangla_asked", "বালতি", "N-BB", "বালতি (bucket)",
     T1, 1, 1, False),
    ("both_langs_english_asked", "বালতি", "N-BE", "bucket (বালতি)",
     T1, 1, 1, True),
    # Tier 2: generic name
    ("bn_t2_plain", "বালতি", "N-BB", "পাত্র", T2, 0, 0, True),
    ("bn_t2_with_adjective", "বালতি", "N-BB", "প্লাস্টিকের পাত্র",
     T2, 0, 0, True),
    ("bn_t2_head_noun", "গ্যাসের চুলা", "N-BB", "চুলা", T2, 0, 0, True),
    ("en_t2_plain", "বালতি", "N-BE", "container", T2, 0, 0, True),
    ("en_t2_plural", "বালতি", "N-BE", "containers", T2, 0, 0, True),
    ("en_t2_for_bangla_request", "বালতি", "N-BB", "container",
     T2, 0, 0, False),
    # Generic-name precedence (8.3 rule 4, D91)
    ("precedence_bn_generic_member", "পড়ার টেবিল", "N-BB", "টেবিল",
     T2, 0, 0, True),
    ("precedence_en_generic_member", "পড়ার টেবিল", "N-BE", "table",
     T2, 0, 0, True),
    ("precedence_bn_showcase", "ওয়াল শোকেস", "N-BB", "শোকেস",
     T2, 0, 0, True),
    ("more_specific_than_gold_bn", "টেবিল", "N-BB", "পড়ার টেবিল",
     SIB, 0, 0, True),
    ("more_specific_than_gold_en", "টেবিল", "N-BE", "study table",
     SIB, 0, 0, True),
    ("more_specific_than_gold_showcase", "শোকেস", "N-BB", "ওয়াল শোকেস",
     SIB, 0, 0, True),
    # Sibling
    ("bn_sibling", "হাঁড়ি", "N-BB", "পাতিল", SIB, 0, 0, True),
    ("bn_sibling_with_ending", "হাঁড়ি", "N-BB", "পাতিলটি", SIB, 0, 0, True),
    ("bn_sibling_other_member", "ওয়াল শোকেস", "N-BB", "আলমারি",
     SIB, 0, 0, True),
    ("en_sibling", "শোকেস", "N-BE", "almirah", SIB, 0, 0, True),
    ("en_sibling_shared_name", "শোকেস", "N-BE", "wardrobe", SIB, 0, 0, True),
    ("sibling_beats_generic", "হাঁড়ি", "N-BB", "পাত্র, পাতিল",
     SIB, 0, 0, True),
    # Ambiguous
    ("ambiguous_gold_and_other", "বালতি", "N-BB", "বালতি ও দরজা",
     AMB, 0, 0, True),
    ("ambiguous_gold_and_sibling", "হাঁড়ি", "N-BB", "হাঁড়ি বা পাতিল",
     AMB, 0, 0, True),
    ("ambiguous_two_others", "বালতি", "N-BB", "দরজা ও জানালা",
     AMB, 0, 0, True),
    ("ambiguous_en", "বালতি", "N-BE", "a bucket and a chair",
     AMB, 0, 0, True),
    ("ambiguous_location_phrase", "বালিশ", "N-BB", "চেয়ারে বালিশ",
     AMB, 0, 0, True),
    # Other wrong
    ("bn_other_object", "বালতি", "N-BB", "দরজা", OTHER, 0, 0, True),
    ("en_other_object", "বালতি", "N-BE", "door", OTHER, 0, 0, True),
    ("bn_unlisted_word", "বালতি", "N-BB", "ঝুড়ি", OTHER, 0, 0, True),
    ("en_unlisted_word", "বালতি", "N-BE", "basket", OTHER, 0, 0, True),
    ("generic_of_other_objects_only", "দরজা", "N-BB", "পাত্র",
     OTHER, 0, 0, True),
    # English copy
    ("copy_plain", "বালতি", "N-BB", "বাকেট", COPY, 0, 1, True),
    ("copy_with_adjective", "বালতি", "N-BB", "প্লাস্টিকের বাকেট",
     COPY, 0, 1, True),
    ("copy_with_lead_in", "বালতি", "N-BB", "এটি একটি বাকেট",
     COPY, 0, 1, True),
    ("copy_after_determiner", "বালতি", "N-BB", "একটি বাকেট",
     COPY, 0, 1, True),
    ("copy_only_at_the_end_of_the_answer", "বালতি", "N-BB",
     "বাকেট রাখা আছে", OTHER, 0, 0, True),
    ("copy_with_ending_known_miss", "বালতি", "N-BB", "বাকেটটি",
     OTHER, 0, 0, True),
    ("copy_pillow", "বালিশ", "N-BB", "পিলো", COPY, 0, 1, True),
    ("copy_window", "জানালা", "N-BB", "উইন্ডো", COPY, 0, 1, True),
    ("copy_door", "দরজা", "N-BB", "ডোর", COPY, 0, 1, True),
    ("copy_when_english_asked", "বালতি", "N-BE", "বাকেট", COPY, 0, 1, False),
    ("copy_blocked_by_listed_term", "বালতি", "N-BB", "বাকেট ও দরজা",
     OTHER, 0, 0, True),
    ("copy_not_for_latin_letters", "বালতি", "N-BB", "baket",
     OTHER, 0, 0, False),
    ("copy_below_threshold_known_miss", "পর্দা", "N-BB", "কার্টেন",
     OTHER, 0, 0, True),
]


def make_naming(gold, cond, answer, label, strict, lenient, fidelity):
    def test(self):
        r = score("obj_name", gold, cond, answer)
        self.assertTrue(r["scoreable"], r["gold_status"])
        self.assertEqual(r["label"], label, r["matched"])
        self.assertEqual(r["strict"], strict)
        self.assertEqual(r["lenient"], lenient)
        self.assertEqual(r["fidelity"], fidelity)
    return test


class NamingTests(unittest.TestCase):
    def test_object_type_scored_like_obj_name(self):
        r = score("object", "বালতি", "L-BB", "বালতি")
        self.assertEqual((r["label"], r["strict"], r["kind"]),
                         (T1, 1, "naming"))

    def test_evidence_fields_tier1(self):
        r = score("obj_name", "বালতি", "N-BB", "বালতি")
        self.assertEqual(r["gold_name_id"], "FX01")
        self.assertEqual(r["label_lang"], "bn")
        self.assertTrue(r["content_lang_ok"])
        self.assertFalse(r["in_cluster"])
        self.assertEqual((r["en_route"], r["en_unique"]), ("no", "yes"))

    def test_evidence_fields_cluster_and_named_object(self):
        r = score("obj_name", "হাঁড়ি", "N-BB", "পাতিল")
        self.assertTrue(r["in_cluster"])
        self.assertEqual(r["cluster_id"], "FXC1")
        self.assertEqual(r["named_ids"], ["FX07"])

    def test_shared_english_name_lists_all_named_objects(self):
        r = score("obj_name", "শোকেস", "N-BE", "wardrobe")
        self.assertEqual(r["named_ids"], ["FX11", "FX12"])

    def test_copy_records_term_and_similarity(self):
        r = score("obj_name", "বালতি", "N-BB", "বাকেট")
        self.assertEqual(r["copy_term"], "bucket")
        self.assertAlmostEqual(r["copy_sim"], 0.667, places=3)

    def test_both_languages_sets_mixed_script(self):
        r = score("obj_name", "বালতি", "N-BB", "বালতি (bucket)")
        self.assertTrue(r["mixed_script"])
        self.assertEqual(r["label_lang"], "bn+en")


add_cases(NamingTests, NAMING, make_naming)


# --------------------------------------------------------------------------
# Empty answers and refusals
# name, condition, answer, is_empty, refusal, label
# --------------------------------------------------------------------------
REFUSAL = [
    ("empty_string", "N-BB", "", True, True, OTHER),
    ("none", "N-BB", None, True, True, OTHER),
    ("spaces_only", "N-BB", "   ", True, True, OTHER),
    ("dari_only", "N-BB", "।", True, True, OTHER),
    ("punctuation_only", "N-BB", "...", True, True, OTHER),
    ("bn_do_not_know", "N-BB", "দুঃখিত, আমি জানি না", False, True, OTHER),
    ("bn_cannot_see", "N-BB", "ছবিটি দেখতে পাচ্ছি না।", False, True, OTHER),
    ("bn_no_image", "L-BB0", "কোনো ছবি নেই", False, True, OTHER),
    ("en_cannot", "N-BE", "I cannot see the image.", False, True, OTHER),
    ("en_cant", "N-BE", "I can't tell", False, True, OTHER),
    ("en_not_sure", "N-BE", "Not sure", False, True, OTHER),
    ("plain_wrong_is_not_refusal", "N-BB", "দরজা", False, False, OTHER),
    ("hedged_right_answer_is_not_refusal", "N-BB", "নিশ্চিত নই, তবে বালতি",
     False, False, T1),
]


def make_refusal(cond, answer, is_empty, refusal, label):
    def test(self):
        r = score("obj_name", "বালতি", cond, answer)
        self.assertEqual(r["is_empty"], is_empty)
        self.assertEqual(r["refusal"], refusal)
        self.assertEqual(r["label"], label)
    return test


class RefusalTests(unittest.TestCase):
    def test_empty_answer_has_no_language(self):
        r = score("obj_name", "বালতি", "N-BB", "")
        self.assertEqual((r["lang"], r["fidelity"]), ("none", False))

    def test_refusal_on_count_item(self):
        r = score("count", "একটি", "L-BB", "দুঃখিত, বলতে পারছি না")
        self.assertEqual((r["label"], r["refusal"]), (BAD, True))

    def test_refusal_on_room_item(self):
        r = score("scene_id", "শোবার ঘর", "L-BE", "Sorry, no image")
        self.assertEqual((r["label"], r["refusal"]), (BAD, True))


add_cases(RefusalTests, REFUSAL, make_refusal)


# --------------------------------------------------------------------------
# Count (Design 8.4)
# name, gold, condition, answer, lenient, strict, fidelity
# --------------------------------------------------------------------------
COUNT = [
    ("bn_word_listed", "একটি।", "L-BB", "একটি", 1, 1, True),
    ("bn_word_bare", "একটি", "L-BB", "এক", 1, 1, True),
    ("bn_word_other_classifier", "একটি", "L-BB", "একটা", 1, 1, True),
    ("bn_word_irregular_form", "দুইটি", "L-BB", "দুটো", 1, 1, True),
    ("bn_word_classifier_ti", "তিনটি", "L-BB", "তিনটি", 1, 1, True),
    ("bn_word_classifier_te", "তিনটি", "L-BB", "তিনটে", 1, 1, True),
    ("bn_word_classifier_khana", "তিনটি", "L-BB", "তিনখানা", 1, 1, True),
    ("bn_word_with_noun", "তিনটি", "L-BB", "তিনটি চেয়ার", 1, 1, True),
    ("bn_digit", "তিনটি", "L-BB", "৩", 1, 1, True),
    ("bn_digit_classifier", "তিনটি", "L-BB", "৩টি", 1, 1, True),
    ("bn_digit_classifier_spaced", "তিনটি", "L-BB", "৩ টি", 1, 1, True),
    ("latin_digit_bangla_asked", "তিনটি", "L-BB", "3", 1, 1, True),
    ("latin_digit_bangla_classifier", "তিনটি", "L-BB", "3টি", 1, 1, True),
    ("bangla_digit_english_asked", "তিনটি", "L-BE", "৩", 1, 1, True),
    ("en_word", "তিনটি", "L-BE", "three", 1, 1, True),
    ("en_word_capital_noun", "তিনটি", "L-BE", "Three chairs.", 1, 1, True),
    ("en_digit", "তিনটি", "L-BE", "3", 1, 1, True),
    ("en_digit_with_noun", "তিনটি", "L-BE", "3 chairs", 1, 1, True),
    ("wrong_lang_english_word", "তিনটি", "L-BB", "three", 1, 0, False),
    ("wrong_lang_bangla_word", "তিনটি", "L-BE", "তিনটি", 1, 0, False),
    ("wrong_lang_digit_english_noun", "তিনটি", "L-BB", "3 chairs",
     1, 0, False),
    ("wrong_lang_digit_bangla_classifier", "তিনটি", "L-BE", "3টি",
     1, 0, False),
    ("first_number_wins_right", "দুইটি", "L-BB",
     "দুইটি চেয়ার ও একটি টেবিল", 1, 1, True),
    ("first_number_wins_wrong", "একটি", "L-BB",
     "দুইটি চেয়ার ও একটি টেবিল", 0, 0, True),
    ("wrong_number", "দুইটি", "L-BB", "তিনটি", 0, 0, True),
    ("no_number_bn", "দুইটি", "L-BB", "অনেকগুলো", 0, 0, True),
    ("no_number_object_name", "একটি", "L-BB", "চেয়ার", 0, 0, True),
    ("no_number_english_article", "একটি", "L-BE", "a chair", 0, 0, True),
    ("lead_in_removed_before_counting", "একটি", "L-BB", "এটি একটি",
     0, 0, False),
    ("no_image_condition", "একটি", "L-BB0", "একটি", 1, 1, True),
    ("english_question_bangla_answer", "একটি", "L-EB", "একটি", 1, 1, True),
    ("repair_no_image_condition", "একটি", "R-BB0", "একটি", 1, 1, True),
]


def make_closed(qa_type):
    def make(gold, cond, answer, lenient, strict, fidelity=None):
        def test(self):
            r = score(qa_type, gold, cond, answer)
            self.assertTrue(r["scoreable"], r["gold_status"])
            self.assertEqual(r["lenient"], lenient, r["found_value"])
            self.assertEqual(r["strict"], strict, r["found_value"])
            self.assertEqual(r["label"], OK if lenient else BAD)
            if fidelity is not None:
                self.assertEqual(r["fidelity"], fidelity)
        return test
    return make


class CountTests(unittest.TestCase):
    def test_gold_and_found_values(self):
        r = score("count", "তিনটি", "L-BB", "দুটো")
        self.assertEqual((r["gold_value"], r["found_value"], r["kind"]),
                         (3, 2, "count"))

    def test_digit_only_answer_language_is_digits(self):
        r = score("count", "তিনটি", "L-BB", "3")
        self.assertEqual(r["lang"], "digits")


add_cases(CountTests, COUNT, make_closed("count"))


# --------------------------------------------------------------------------
# Room (Design 8.4)   name, gold, condition, answer, lenient, strict
# --------------------------------------------------------------------------
ROOM = [
    ("bn_plain", "শোবার ঘর", "L-BB", "শোবার ঘর", 1, 1),
    ("bn_english_word_in_bangla_letters", "শোবার ঘর", "L-BB", "বেডরুম", 1, 1),
    ("bn_lead_in", "শোবার ঘর", "L-BB", "এটি একটি শোবার ঘর।", 1, 1),
    ("bn_locative_ending", "শোবার ঘর", "L-BB", "শোবার ঘরে", 1, 1),
    ("bn_genitive_phrase", "শোবার ঘর", "L-BB", "শোবার ঘরের ছবি", 1, 1),
    ("bn_split_spelling", "রান্নাঘর", "L-BB", "রান্না ঘর", 1, 1),
    ("bn_other_listed_form", "বসার ঘর", "L-BB", "ড্রয়িং রুম", 1, 1),
    ("bn_same_room_twice", "শোবার ঘর", "L-BB", "শোবার ঘর (বেডরুম)", 1, 1),
    ("en_plain", "শোবার ঘর", "L-BE", "bedroom", 1, 1),
    ("en_sentence", "শোবার ঘর", "L-BE", "It is a bedroom.", 1, 1),
    ("en_two_words", "বসার ঘর", "L-BE", "Living room", 1, 1),
    ("en_joined_spelling", "বসার ঘর", "L-BE", "livingroom", 1, 1),
    ("en_english_question", "শোবার ঘর", "L-EE", "Bedroom", 1, 1),
    ("wrong_lang_english", "শোবার ঘর", "L-BB", "bedroom", 1, 0),
    ("wrong_lang_bangla", "শোবার ঘর", "L-BE", "শোবার ঘর", 1, 0),
    ("two_rooms_is_wrong", "শোবার ঘর", "L-BB", "শোবার ঘর বা বসার ঘর", 0, 0),
    ("wrong_room", "শোবার ঘর", "L-BB", "রান্নাঘর", 0, 0),
    ("no_room_named", "শোবার ঘর", "L-BB", "ঘর", 0, 0),
]


class RoomTests(unittest.TestCase):
    def test_gold_value_is_room_class(self):
        r = score("scene_id", "শোবার ঘর", "L-BB", "বেডরুম")
        self.assertEqual((r["gold_value"], r["kind"]), (["bedroom"], "room"))


add_cases(RoomTests, ROOM, make_closed("scene_id"))


# --------------------------------------------------------------------------
# Material and colour (Design 8.4)
# --------------------------------------------------------------------------
PROPERTY = [
    ("mat_bare", "কাঠের।", "L-BB", "কাঠ", 1, 1),
    ("mat_genitive", "কাঠের", "L-BB", "কাঠের", 1, 1),
    ("mat_made_of", "কাঠের", "L-BB", "কাঠের তৈরি", 1, 1),
    ("mat_made_with", "কাঠের", "L-BB", "কাঠ দিয়ে তৈরি", 1, 1),
    ("mat_merged_class_bn", "স্টিলের", "L-BB", "লোহা", 1, 1),
    ("mat_merged_class_generic", "স্টিলের", "L-BB", "ধাতু", 1, 1),
    ("mat_merged_class_en", "স্টিলের", "L-BE", "Made of metal", 1, 1),
    ("mat_en_adjective", "কাঠের", "L-BE", "wooden", 1, 1),
    ("mat_spelling_forms", "কাঁচের", "L-BB", "কাচ", 1, 1),
    ("mat_wrong_lang", "কাঠের", "L-BB", "wood", 1, 0),
    ("mat_two_when_gold_has_one", "কাঠের", "L-BB", "কাঠ ও কাচ", 0, 0),
    ("mat_compound_gold_one_named", "কাঠ আর কাঁচের", "L-BB", "কাঠ", 1, 1),
    ("mat_compound_gold_both_named", "কাঠ আর কাঁচের", "L-BB",
     "কাঠ এবং গ্লাস", 1, 1),
    ("mat_compound_gold_extra_named", "কাঠ আর কাঁচের", "L-BB",
     "কাঠ ও প্লাস্টিক", 0, 0),
    ("mat_wrong", "কাঠের", "L-BB", "প্লাস্টিক", 0, 0),
    ("mat_unlisted_word", "কাঠের", "L-BB", "পাথর", 0, 0),
    ("col_plain", "সাদা", "L-BB", "সাদা", 1, 1),
    ("col_phrase", "সাদা", "L-BB", "সাদা রঙের", 1, 1),
    ("col_lightness_word_in_answer", "সবুজ", "L-BB", "গাঢ় সবুজ", 1, 1),
    ("col_lightness_word_in_gold", "গাঢ় লাল", "L-BB", "লাল", 1, 1),
    ("col_compound_gold_one", "সবুজ-হলুদ", "L-BB", "হলুদ", 1, 1),
    ("col_compound_gold_both", "সবুজ-হলুদ", "L-BB", "সবুজ ও হলুদ", 1, 1),
    ("col_compound_gold_extra", "সবুজ-হলুদ", "L-BB", "সবুজ ও লাল", 0, 0),
    ("col_en", "সবুজ", "L-BE", "Light green", 1, 1),
    ("col_wrong_lang", "সবুজ", "L-BB", "green", 1, 0),
    ("col_wrong", "সবুজ", "L-BB", "লাল", 0, 0),
    ("col_material_word_for_colour_gold", "সাদা", "L-BB", "কাঠ", 0, 0),
]


class PropertyTests(unittest.TestCase):
    def test_kind_comes_from_the_gold_material(self):
        r = score("obj_property", "কাঠের", "L-BB", "কাঠ")
        self.assertEqual((r["kind"], r["gold_value"]), ("material", ["wood"]))

    def test_kind_comes_from_the_gold_colour(self):
        r = score("obj_property", "সবুজ-হলুদ", "L-BB", "সবুজ")
        self.assertEqual((r["kind"], r["gold_value"]),
                         ("colour", ["green", "yellow"]))


add_cases(PropertyTests, PROPERTY, make_closed("obj_property"))


# --------------------------------------------------------------------------
# Forced choice (Design 8.5)
# name, condition, options, correct index, answer, label, choice, method
# --------------------------------------------------------------------------
FORCED = [
    ("bn_label", "N-FCB", OPT_BN, 1, "খ", OK, 1, "label"),
    ("bn_label_bracket", "N-FCB", OPT_BN, 1, "খ)", OK, 1, "label"),
    ("bn_label_two_brackets", "N-FCB", OPT_BN, 1, "(খ)", OK, 1, "label"),
    ("bn_label_dari", "N-FCB", OPT_BN, 1, "খ।", OK, 1, "label"),
    ("bn_label_after_cue", "N-FCB", OPT_BN, 1, "উত্তর: খ", OK, 1, "label"),
    ("bn_label_with_text", "N-FCB", OPT_BN, 1, "খ) পাতিল", OK, 1, "label"),
    ("bn_label_wrong", "N-FCB", OPT_BN, 1, "ক", BAD, 0, "label"),
    ("bn_first_label_wins", "N-FCB", OPT_BN, 1, "খ, গ নয়", OK, 1, "label"),
    ("bn_text_only", "N-FCB", OPT_BN, 1, "পাতিল", OK, 1, "text"),
    ("bn_text_only_wrong", "N-FCB", OPT_BN, 1, "হাঁড়ি", BAD, 0, "text"),
    ("bn_text_with_ending", "N-FCB", OPT_BN, 1, "পাতিলটি", OK, 1, "text"),
    ("bn_two_texts_invalid", "N-FCB", OPT_BN, 1, "হাঁড়ি বা পাতিল",
     INV, None, None),
    ("bn_empty_invalid", "N-FCB", OPT_BN, 1, "", INV, None, None),
    ("bn_latin_label_invalid", "N-FCB", OPT_BN, 1, "B", INV, None, None),
    ("bn_unrelated_word_invalid", "N-FCB", OPT_BN, 1, "দরজা",
     INV, None, None),
    ("bn_no_image_condition", "N-FCB0", OPT_BN, 1, "খ", OK, 1, "label"),
    ("en_label", "N-FCE", OPT_EN, 1, "B", OK, 1, "label"),
    ("en_label_lower_case", "N-FCE", OPT_EN, 1, "b", OK, 1, "label"),
    ("en_label_bracket", "N-FCE", OPT_EN, 1, "B)", OK, 1, "label"),
    ("en_label_two_brackets", "N-FCE", OPT_EN, 1, "(B)", OK, 1, "label"),
    ("en_label_dot_text", "N-FCE", OPT_EN, 1, "B. bucket", OK, 1, "label"),
    ("en_label_option_word", "N-FCE", OPT_EN, 1, "Option B", OK, 1, "label"),
    ("en_label_sentence", "N-FCE", OPT_EN, 1, "The answer is B.",
     OK, 1, "label"),
    ("en_label_markdown", "N-FCE", OPT_EN, 1, "**B**", OK, 1, "label"),
    ("en_label_a_alone", "N-FCE", OPT_EN, 1, "A", BAD, 0, "label"),
    ("en_label_a_ends_sentence", "N-FCE", OPT_EN, 1, "The answer is A.",
     BAD, 0, "label"),
    ("en_label_a_bracket_text", "N-FCE", OPT_EN, 1, "A) pot",
     BAD, 0, "label"),
    ("en_label_a_after_option", "N-FCE", OPT_EN, 0, "Option A is right",
     OK, 0, "label"),
    ("en_article_is_not_a_label", "N-FCE", OPT_EN, 1, "a bucket",
     OK, 1, "text"),
    ("en_text_in_sentence", "N-FCE", OPT_EN, 1, "It is a bucket",
     OK, 1, "text"),
    ("en_text_only", "N-FCE", OPT_EN, 1, "bucket", OK, 1, "text"),
    ("en_text_plural", "N-FCE", OPT_EN, 1, "buckets", OK, 1, "text"),
    ("en_label_beats_text", "N-FCE", OPT_EN, 1, "A) bucket",
     BAD, 0, "label"),
    ("en_two_texts_invalid", "N-FCE", OPT_EN, 1, "pot or bucket",
     INV, None, None),
    ("en_unknown_letter_invalid", "N-FCE", OPT_EN, 1, "E", INV, None, None),
    ("en_digit_invalid", "N-FCE", OPT_EN, 1, "2", INV, None, None),
    ("en_bangla_label_invalid", "N-FCE", OPT_EN, 1, "ক", INV, None, None),
    ("en_no_image_condition", "N-FCE0", OPT_EN, 1, "B", OK, 1, "label"),
]


def make_forced(cond, options, correct, answer, label, choice, method):
    def test(self):
        r = score("obj_name", "পাতিল", cond, answer, options=options,
                  correct_index=correct)
        self.assertEqual(r["label"], label)
        self.assertEqual(r["fc_choice"], choice)
        self.assertEqual(r["fc_method"], method)
        self.assertEqual(r["fc_valid"], label != INV)
        self.assertEqual(r["strict"], int(label == OK))
        self.assertEqual(r["lenient"], int(label == OK))
        self.assertIsNone(r["fidelity"])
    return test


class ForcedChoiceTests(unittest.TestCase):
    def fc(self, cond, options, answer):
        return score("obj_name", "পাতিল", cond, answer, options=options,
                     correct_index=1)

    def test_cross_flag_latin_label_in_bangla_list(self):
        self.assertTrue(self.fc("N-FCB", OPT_BN, "B")["fc_cross_label"])

    def test_cross_flag_bangla_label_in_english_list(self):
        self.assertTrue(self.fc("N-FCE", OPT_EN, "ক")["fc_cross_label"])

    def test_cross_flag_digit(self):
        self.assertTrue(self.fc("N-FCE", OPT_EN, "2")["fc_cross_label"])

    def test_cross_flag_off_for_empty(self):
        self.assertFalse(self.fc("N-FCB", OPT_BN, "")["fc_cross_label"])

    def test_empty_forced_choice_is_refusal(self):
        self.assertTrue(self.fc("N-FCB", OPT_BN, "")["refusal"])

    def test_needs_options(self):
        with self.assertRaises(ValueError):
            score("obj_name", "পাতিল", "N-FCB", "খ")

    def test_needs_four_options(self):
        with self.assertRaises(ValueError):
            score("obj_name", "পাতিল", "N-FCB", "খ", options=OPT_BN[:3],
                  correct_index=1)

    def test_needs_valid_correct_index(self):
        with self.assertRaises(ValueError):
            score("obj_name", "পাতিল", "N-FCB", "খ", options=OPT_BN,
                  correct_index=4)

    def test_longer_option_text_wins(self):
        opts = ["টেবিল", "পড়ার টেবিল", "চেয়ার", "দরজা"]
        r = score("obj_name", "পড়ার টেবিল", "N-FCB", "পড়ার টেবিল",
                  options=opts, correct_index=1)
        self.assertEqual((r["label"], r["fc_choice"]), (OK, 1))


add_cases(ForcedChoiceTests, FORCED, make_forced)


# --------------------------------------------------------------------------
# Answer language and fidelity (Design 8.2), lead-ins (Design 8.1)
# --------------------------------------------------------------------------
LANGUAGE = [
    ("bangla", "বালতি", "bn"),
    ("english", "bucket", "en"),
    ("bangla_digit_only", "৩", "digits"),
    ("latin_digit_only", "3", "digits"),
    ("empty", "", "none"),
    ("punctuation_only", "?!", "none"),
    ("other_script_devanagari", "बाल्टी", "mixed"),
    ("other_script_cjk", "水桶", "mixed"),
    ("bangla_majority", "টিভি TV", "bn"),
    ("latin_majority", "ফ্রিজ fridge", "en"),
    ("tie_counts_as_bangla", "সোফা sofa", "bn"),
    ("lead_in_removed_first", "It is a বালতি", "bn"),
    ("digit_with_bangla_classifier", "৩টি", "bn"),
    ("digit_with_english_noun", "3 chairs", "en"),
]
FIDELITY = [
    ("same_bn", "bn", "bn", True), ("same_en", "en", "en", True),
    ("english_for_bangla", "en", "bn", False),
    ("bangla_for_english", "bn", "en", False),
    ("digits_for_bangla", "digits", "bn", True),
    ("digits_for_english", "digits", "en", True),
    ("none_for_bangla", "none", "bn", False),
    ("mixed_for_english", "mixed", "en", False),
]
LEAD_IN = [
    ("eti_ekti", "এটি একটি বালতি", "বালতি"),
    ("eta", "এটা বালতি", "বালতি"),
    ("eti", "এটি বালতি", "বালতি"),
    ("it_is_a", "it is a bucket", "bucket"),
    ("this_is_a", "this is a bucket", "bucket"),
    ("its_a", "it's a bucket", "bucket"),
    ("with_colon", "এটি: বালতি", "বালতি"),
    ("it_is_an_is_not_listed", "it is an umbrella", "it is an umbrella"),
    ("not_inside_a_word", "এটিকে বালতি বলে", "এটিকে বালতি বলে"),
    ("only_at_the_start", "বালতি এটি", "বালতি এটি"),
    ("lead_in_alone", "এটি", ""),
]


class LanguageTests(unittest.TestCase):
    pass


add_cases(LanguageTests, [("lang_" + n, t, e) for n, t, e in LANGUAGE],
          lambda text, lang: lambda self: self.assertEqual(
              S.prepare_answer(text).lang["lang"], lang))
add_cases(LanguageTests, [("fidelity_" + n, a, b, c) for n, a, b, c in FIDELITY],
          lambda lang, req, want: lambda self: self.assertEqual(
              S.is_fidelity(lang, req), want))
add_cases(LanguageTests, [("lead_in_" + n, t, e) for n, t, e in LEAD_IN],
          lambda text, want: lambda self: self.assertEqual(
              S.strip_lead_in(normalise(text)), normalise(want)))


# --------------------------------------------------------------------------
# Tokens, endings, romanisation, edit similarity
# --------------------------------------------------------------------------
ROMANISE = [
    ("bucket", "বাকেট", "baket"), ("pillow", "পিলো", "pilo"),
    ("window", "উইন্ডো", "windo"), ("tap", "ট্যাপ", "tap"),
    ("blanket", "ব্ল্যাঙ্কেট", "blanket"), ("stove", "স্টোভ", "stov"),
    ("towel", "টাওয়েল", "tawel"), ("commode", "কমোড", "komod"),
    ("clock", "ক্লক", "klok"), ("shelf", "শেলফ", "shelf"),
    ("rack_with_joiner", "র\u200d্যাক", "rak"),
    ("aluminium", "অ্যালুমিনিয়াম", "aluminiyam"),
    ("stand", "স্ট্যান্ড", "stand"), ("sink", "সিঙ্ক", "sink"),
    ("door", "ডোর", "dor"), ("two_words", "গ্যাস স্টোভ", "gas stov"),
]


class TextTests(unittest.TestCase):
    def texts(self, s):
        return [t.text for t in S.tokenise(normalise(s))]

    def test_hyphen_separates_words(self):
        self.assertEqual(self.texts("শিল-পাটা"), ["শিল", "পাটা"])

    def test_digits_split_from_letters(self):
        toks = S.tokenise(normalise("৩টি"))
        self.assertEqual([(t.text, t.kind) for t in toks],
                         [("৩", "dig"), ("টি", "bn")])

    def test_apostrophe_stays_inside_word(self):
        self.assertEqual(self.texts("I'd say it's B"),
                         ["i'd", "say", "it's", "b"])

    def test_dari_separates_words(self):
        self.assertEqual(self.texts("বালতি। দরজা"), ["বালতি", "দরজা"])

    def test_joiner_placement_cannot_split_a_match(self):
        plain = self.texts("র্যাক")
        self.assertEqual(self.texts("র\u200d্যাক"), plain)
        self.assertEqual(self.texts("র্\u200dযাক"), plain)

    def test_precomposed_and_decomposed_letters_agree(self):
        self.assertEqual(self.texts("চে\u09dfার"),
                         self.texts("চে\u09af\u09bcার"))

    def test_n_words(self):
        r = score("obj_name", "বালতি", "N-BB", "এটি একটি বালতি।")
        self.assertEqual(r["n_words"], 3)

    def test_stems_bn_two_endings(self):
        self.assertIn(normalise("বালতি"), S.stems_bn(normalise("বালতিটির")))

    def test_stems_bn_genitive(self):
        self.assertIn(normalise("কাঠ"), S.stems_bn(normalise("কাঠের")))

    def test_stems_bn_never_leaves_one_letter(self):
        self.assertEqual(S.stems_bn(normalise("বর")), [])

    def test_stems_en_plural(self):
        self.assertEqual(S.stems_en("buckets"), ["bucket"])

    def test_stems_en_knives(self):
        self.assertIn("knife", S.stems_en("knives"))

    def test_stems_en_shelves(self):
        self.assertIn("shelf", S.stems_en("shelves"))

    def test_stems_en_boxes(self):
        self.assertIn("box", S.stems_en("boxes"))

    def test_stems_en_double_s_untouched(self):
        self.assertEqual(S.stems_en("glass"), [])

    def test_stems_en_possessive(self):
        self.assertIn("bucket", S.stems_en("bucket's"))

    def test_edit_distance(self):
        self.assertEqual(S.edit_distance("baket", "bucket"), 2)

    def test_edit_similarity_identical(self):
        self.assertEqual(S.edit_similarity("abc", "abc"), 1.0)

    def test_similar_above_threshold(self):
        self.assertTrue(S.is_similar("baket", "bucket"))

    def test_similar_exactly_at_threshold(self):
        self.assertTrue(S.is_similar("brum", "broom"))       # 3/5 = 0.6

    def test_similar_below_threshold(self):
        self.assertFalse(S.is_similar("karten", "curtain"))

    def test_similar_empty_strings(self):
        self.assertFalse(S.is_similar("", ""))

    def test_english_forms_article_and_plural(self):
        self.assertTrue(S.english_forms("the pots") & S.english_forms("pot"))


add_cases(TextTests, [("romanise_" + n, b, e) for n, b, e in ROMANISE],
          lambda bn, want: lambda self: self.assertEqual(
              S.romanise_bn(bn), want))


# --------------------------------------------------------------------------
# Gold answers
# --------------------------------------------------------------------------
class GoldTests(unittest.TestCase):
    def gold(self, qa_type, text):
        return S.resolve_gold(item(qa_type, text), LEX, VOC)

    def test_naming_exact(self):
        g = self.gold("obj_name", "বালতি।")
        self.assertEqual((g.status, g.kind, g.name_id),
                         ("ok", "naming", "FX01"))

    def test_naming_listed_variant(self):
        self.assertEqual(self.gold("object", "টিভি").name_id, "FX04")

    def test_naming_hyphen_spelling(self):
        self.assertEqual(self.gold("obj_name", "সুইচ-বোর্ড").name_id, "FX27")

    def test_naming_unlisted(self):
        g = self.gold("obj_name", "ঝুড়ি")
        self.assertEqual((g.status, g.hint), ("gold_unmapped", ""))

    def test_naming_adjective_plus_name_is_not_mapped(self):
        g = self.gold("obj_name", "পুরনো আলমারি")
        self.assertEqual(g.status, "gold_unmapped")
        self.assertEqual(g.hint, "contains: FX11")

    def test_naming_two_objects_is_not_mapped(self):
        g = self.gold("obj_name", "বালতি ও দরজা")
        self.assertEqual(g.status, "gold_unmapped")
        self.assertEqual(g.hint, "contains: FX01|FX17")

    def test_naming_generic_only_is_not_mapped(self):
        self.assertEqual(self.gold("obj_name", "পাত্র").status,
                         "gold_unmapped")

    def test_naming_inflected_gold_is_not_mapped(self):
        self.assertEqual(self.gold("obj_name", "বালতিটি").status,
                         "gold_unmapped")

    def test_not_a_closed_type(self):
        self.assertEqual(self.gold("spatial", "টেবিলের পাশে").status,
                         "not_closed_type")

    def test_count_exact(self):
        g = self.gold("count", "তিনটি।")
        self.assertEqual((g.status, g.number, g.residual), ("ok", 3, ()))

    def test_count_approximate_keeps_extra_word(self):
        g = self.gold("count", "প্রায় তিনটি")
        self.assertEqual(g.number, 3)
        self.assertEqual(g.residual, (normalise("প্রায়"),))

    def test_count_without_number(self):
        self.assertEqual(self.gold("count", "অনেকগুলো").status,
                         "gold_unmapped")

    def test_room_exact(self):
        g = self.gold("scene_id", "শোবার ঘর")
        self.assertEqual((g.status, set(g.classes)), ("ok", {"bedroom"}))

    def test_room_part_of_room_is_not_mapped(self):
        g = self.gold("scene_id", "শোবার ঘরের দরজা")
        self.assertEqual((g.status, g.hint),
                         ("gold_unmapped", "contains: bedroom"))

    def test_room_unlisted(self):
        self.assertEqual(self.gold("scene_id", "বারান্দা").status,
                         "gold_unmapped")

    def test_property_material(self):
        g = self.gold("obj_property", "কাঠের।")
        self.assertEqual((g.kind, set(g.classes)), ("material", {"wood"}))

    def test_property_colour(self):
        g = self.gold("obj_property", "সাদা")
        self.assertEqual((g.kind, set(g.classes)), ("colour", {"white"}))

    def test_property_extra_word_is_reported(self):
        g = self.gold("obj_property", "প্লাস্টিকের ঢাকনা")
        self.assertEqual((g.kind, g.residual),
                         ("material", (normalise("ঢাকনা"),)))

    def test_property_neither(self):
        self.assertEqual(self.gold("obj_property", "একটি").status,
                         "gold_unmapped")

    def test_property_both_kinds(self):
        self.assertEqual(self.gold("obj_property", "সাদা কাঠ").status,
                         "gold_ambiguous")

    def test_unscoreable_item_still_reports_language(self):
        r = score("obj_name", "ঝুড়ি", "N-BB", "bucket")
        self.assertFalse(r["scoreable"])
        self.assertEqual(r["gold_status"], "gold_unmapped")
        self.assertIsNone(r["strict"])
        self.assertIsNone(r["lenient"])
        self.assertIsNone(r["label"])
        self.assertEqual((r["lang"], r["fidelity"]), ("en", False))

    def test_free_form_type_is_unscoreable(self):
        r = score("spatial", "টেবিলের পাশে", "L-BB", "টেবিলের পাশে")
        self.assertEqual((r["scoreable"], r["gold_status"]),
                         (False, "not_closed_type"))


# --------------------------------------------------------------------------
# Conditions (Design section 7)
# --------------------------------------------------------------------------
CONDITION_LANG = [
    ("L-BB", "bn", "bn", True, "open"), ("L-BE", "bn", "en", True, "open"),
    ("L-EB", "en", "bn", True, "open"), ("L-EE", "en", "en", True, "open"),
    ("L-BB0", "bn", "bn", False, "open"),
    ("L-EE0", "en", "en", False, "open"),
    ("N-BB", "bn", "bn", True, "open"), ("N-BE", "bn", "en", True, "open"),
    ("N-BBs", "bn", "bn", True, "open"), ("N-FCB", "bn", "", True, "fc"),
    ("N-FCB0", "bn", "", False, "fc"), ("N-FCE", "en", "", True, "fc"),
    ("N-FCE0", "en", "", False, "fc"), ("R-BB", "bn", "bn", True, "open"),
    ("R-EE", "en", "en", True, "open"), ("R-BB0", "bn", "bn", False, "open"),
]


class ConditionTests(unittest.TestCase):
    def test_sixteen_conditions(self):
        self.assertEqual(len(S.CONDITIONS), 16)

    def test_unknown_condition_raises(self):
        with self.assertRaises(KeyError):
            S.condition_spec("L-XX")

    def test_specificity_flag(self):
        self.assertTrue(S.condition_spec("N-BBs").specificity)
        self.assertFalse(S.condition_spec("N-BB").specificity)


def make_condition(q, a, image, task):
    def make(cid):
        def test(self):
            c = S.condition_spec(cid)
            self.assertEqual((c.question_lang, c.answer_lang, c.image,
                              c.task, c.experiment),
                             (q, a, image, task, cid[0]))
        return test
    return make


for _cid, _q, _a, _img, _task in CONDITION_LANG:
    _name = "test_condition_" + _cid.replace("-", "_")
    setattr(ConditionTests, _name, make_condition(_q, _a, _img, _task)(_cid))


# --------------------------------------------------------------------------
# The two matching rules can be switched off (rules B and C)
# --------------------------------------------------------------------------
class PlainMatchingTests(unittest.TestCase):
    def s(self, qa_type, gold, cond, answer):
        return score(qa_type, gold, cond, answer, LEX_PLAIN, VOC_PLAIN)

    def test_plain_exact_still_matches(self):
        self.assertEqual(self.s("obj_name", "বালতি", "N-BB", "বালতি")["label"],
                         T1)

    def test_plain_two_words_still_match(self):
        r = self.s("obj_name", "সুইচ বোর্ড", "N-BB", "সুইচ বোর্ড")
        self.assertEqual(r["label"], T1)

    def test_plain_longest_match_still_wins(self):
        r = self.s("obj_name", "টেবিল", "N-BB", "পড়ার টেবিল")
        self.assertEqual(r["label"], SIB)

    def test_plain_no_bangla_ending(self):
        self.assertEqual(
            self.s("obj_name", "বালতি", "N-BB", "বালতিটি")["label"], OTHER)

    def test_plain_no_english_plural(self):
        self.assertEqual(
            self.s("obj_name", "বালতি", "N-BE", "buckets")["label"], OTHER)

    def test_plain_no_joined_spelling(self):
        joined = "পড়ারটেবিল"
        self.assertEqual(
            self.s("obj_name", "পড়ার টেবিল", "N-BB", joined)["label"], OTHER)
        self.assertEqual(
            score("obj_name", "পড়ার টেবিল", "N-BB", joined)["label"], T1)

    def test_unlisted_spelling_of_english_route_name_is_english_copy(self):
        # For an English-route object the Bangla name is the English name in
        # Bangla letters. A spelling of it that is not listed therefore lands
        # in English copy (lenient right, strict wrong), not in Tier 1.
        # HT-5 audits these as "acceptable but missing from the lexicon".
        r = self.s("obj_name", "সুইচ বোর্ড", "N-BB", "সুইচবোর্ড")
        self.assertEqual((r["label"], r["strict"], r["lenient"]),
                         (COPY, 0, 1))
        self.assertEqual(r["en_route"], "yes")

    def test_plain_classifier_needs_listing(self):
        self.assertEqual(self.s("count", "একটি", "L-BB", "একখানা")["lenient"],
                         0)
        self.assertEqual(score("count", "একটি", "L-BB", "একখানা")["lenient"],
                         1)

    def test_plain_genitive_gold_is_not_mapped(self):
        self.assertFalse(
            self.s("obj_property", "কাঠের", "L-BB", "কাঠ")["scoreable"])

    def test_settings_are_described(self):
        self.assertIn("suffix tolerance OFF", LEX_PLAIN.describe())
        self.assertIn("suffix tolerance ON", LEX.describe())


# --------------------------------------------------------------------------
# Lexicon and vocabulary files
# --------------------------------------------------------------------------
def lex_row(**kw):
    row = dict.fromkeys(S.LEXICON_COLUMNS, "")
    row.update(name_id="A1", name_bn="বালতি", canonical_en="bucket",
               en_route="no", en_unique="yes")
    row.update(kw)
    return row


def vocab_row(**kw):
    row = dict.fromkeys(S.VOCAB_COLUMNS, "")
    row.update(kind="colour", class_id="red", forms_bn="লাল", forms_en="red")
    row.update(kw)
    return row


def write_csv(rows, columns, encoding="utf-8"):
    fh = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                     encoding=encoding, newline="")
    with fh:
        w = csv.DictWriter(fh, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})
    return fh.name


class FileTests(unittest.TestCase):
    def lex(self, rows, columns=S.LEXICON_COLUMNS, **kw):
        return S.load_lexicon(write_csv(rows, columns, **kw))

    def voc(self, rows, columns=S.VOCAB_COLUMNS):
        return S.load_closed_vocab(write_csv(rows, columns))

    def test_fixture_lexicon_has_no_warnings(self):
        self.assertEqual(LEX.warnings, [])

    def test_fixture_vocab_has_no_warnings(self):
        self.assertEqual(VOC.warnings, [])

    def test_fixture_counts(self):
        self.assertEqual(len(LEX.objects), 28)
        self.assertEqual(len(LEX.clusters), 3)

    def test_en_unique_recomputed_equals_fixture_column(self):
        got = S.compute_en_unique(LEX.objects)
        for oid, obj in LEX.objects.items():
            self.assertEqual(got[oid], obj.en_unique, oid)

    def test_en_unique_expected_collisions(self):
        got = S.compute_en_unique(LEX.objects)
        for oid in ("FX06", "FX07", "FX08", "FX11", "FX12", "FX21", "FX22",
                    "FX23", "FX24"):
            self.assertFalse(got[oid], oid)
        self.assertTrue(got["FX01"])

    def test_siblings_come_from_cluster_id(self):
        self.assertEqual(LEX.siblings_of("FX06"), {"FX07", "FX08"})
        self.assertEqual(LEX.siblings_of("FX01"), set())

    def test_minimal_lexicon_loads(self):
        self.assertEqual(len(self.lex([lex_row()]).objects), 1)

    def test_semicolon_separator(self):
        lex = self.lex([lex_row(variants_en="pail; tub")])
        self.assertEqual(lex.objects["A1"].t1_en, ["bucket", "pail", "tub"])

    def test_bom_is_accepted(self):
        self.assertEqual(
            len(self.lex([lex_row()], encoding="utf-8-sig").objects), 1)

    def test_missing_column(self):
        cols = [c for c in S.LEXICON_COLUMNS if c != "super_bn"]
        with self.assertRaises(S.LexiconError):
            self.lex([lex_row()], cols)

    def test_duplicate_name_id(self):
        with self.assertRaises(S.LexiconError):
            self.lex([lex_row(), lex_row(name_bn="দরজা")])

    def test_empty_name_bn(self):
        with self.assertRaises(S.LexiconError):
            self.lex([lex_row(name_bn="")])

    def test_same_bangla_name_in_two_objects(self):
        with self.assertRaises(S.LexiconError):
            self.lex([lex_row(), lex_row(name_id="A2", name_bn="দরজা",
                                         variants_bn="বালতি")])

    def test_same_bangla_name_differing_only_by_space(self):
        with self.assertRaises(S.LexiconError):
            self.lex([lex_row(name_bn="সুইচ বোর্ড"),
                      lex_row(name_id="A2", name_bn="সুইচবোর্ড")])

    def test_term_in_both_tiers(self):
        with self.assertRaises(S.LexiconError):
            self.lex([lex_row(super_bn="বালতি")])

    def test_bad_yes_no(self):
        with self.assertRaises(S.LexiconError):
            self.lex([lex_row(en_route="maybe")])

    def test_warning_wrong_script_in_column(self):
        lex = self.lex([lex_row(variants_en="বালতি")])
        self.assertTrue(any("written in bn letters" in w
                            for w in lex.warnings))

    def test_warning_en_unique_disagrees(self):
        lex = self.lex([lex_row(), lex_row(name_id="A2", name_bn="দরজা",
                                           canonical_en="bucket")])
        self.assertEqual(sum("en_unique says yes" in w
                             for w in lex.warnings), 2)

    def test_warning_no_english_name(self):
        lex = self.lex([lex_row(canonical_en="")])
        self.assertTrue(any("no English name" in w for w in lex.warnings))

    def test_minimal_vocab_loads(self):
        self.assertEqual(self.voc([vocab_row()]).classes["colour"], {"red"})

    def test_vocab_unknown_kind(self):
        with self.assertRaises(S.LexiconError):
            self.voc([vocab_row(kind="shape")])

    def test_vocab_form_in_two_classes(self):
        with self.assertRaises(S.LexiconError):
            self.voc([vocab_row(), vocab_row(class_id="pink",
                                             forms_en="pink")])

    def test_vocab_number_class_must_be_digits(self):
        with self.assertRaises(S.LexiconError):
            self.voc([vocab_row(kind="number", class_id="three",
                                forms_bn="তিন", forms_en="three")])

    def test_vocab_class_without_forms(self):
        with self.assertRaises(S.LexiconError):
            self.voc([vocab_row(forms_bn="", forms_en="")])

    def test_vocab_duplicate_class(self):
        with self.assertRaises(S.LexiconError):
            self.voc([vocab_row(), vocab_row(forms_bn="রক্তিম",
                                             forms_en="crimson")])

    def test_vocab_warns_when_not_four_rooms(self):
        voc = self.voc([vocab_row()])
        self.assertTrue(any("4 room classes" in w for w in voc.warnings))


# --------------------------------------------------------------------------
# Output records, probe files, summary
# --------------------------------------------------------------------------
class OutputTests(unittest.TestCase):
    def test_same_keys_for_every_kind(self):
        recs = [score("obj_name", "বালতি", "N-BB", "বালতি"),
                score("count", "একটি", "L-BB", "একটি"),
                score("scene_id", "শোবার ঘর", "L-BE", "bedroom"),
                score("obj_property", "কাঠের", "L-BB", "কাঠ"),
                score("spatial", "ক", "L-BB", "ক"),
                score("obj_name", "পাতিল", "N-FCB", "খ", options=OPT_BN,
                      correct_index=1)]
        keys = [tuple(r.keys()) for r in recs]
        self.assertEqual(len(set(keys)), 1)

    def test_record_is_json(self):
        r = score("obj_name", "বালতি", "N-BB", "বালতি (bucket)")
        self.assertEqual(json.loads(json.dumps(r, ensure_ascii=False)), r)

    def test_deterministic(self):
        a = score("obj_name", "হাঁড়ি", "N-BB", "হাঁড়ি বা পাতিল")
        b = score("obj_name", "হাঁড়ি", "N-BB", "হাঁড়ি বা পাতিল")
        self.assertEqual(a, b)

    def test_version_is_recorded(self):
        r = score("obj_name", "বালতি", "N-BB", "বালতি")
        self.assertEqual(r["scorer_version"], S.SCORER_VERSION)

    def _probe(self, data):
        fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8")
        with fh:
            json.dump(data, fh, ensure_ascii=False)
        return fh.name

    def test_load_probe_wrapper(self):
        p = self._probe({"probe": "X", "items": [item("count", "একটি")]})
        self.assertEqual(len(S.load_probe(p)), 1)

    def test_load_probe_bare_list(self):
        p = self._probe([item("count", "একটি")])
        self.assertEqual(len(S.load_probe(p)), 1)

    def test_load_probe_missing_field(self):
        p = self._probe([{"qa_id": "T0", "qa_type": "count"}])
        with self.assertRaises(ValueError):
            S.load_probe(p)

    def test_summarise(self):
        recs = [score("obj_name", "বালতি", "N-BB", "বালতি"),
                score("obj_name", "বালতি", "N-BB", "fridge"),
                score("obj_name", "ঝুড়ি", "N-BB", "বালতি")]
        row = S.summarise(recs)[0]
        self.assertEqual((row["n"], row["scoreable"], row["strict"]),
                         (3, 2, 50.0))


if __name__ == "__main__":
    unittest.main()
