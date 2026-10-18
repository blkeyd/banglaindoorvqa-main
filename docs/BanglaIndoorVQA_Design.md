# BanglaIndoorVQA Main Paper: Locked Experimental Design

**Version 2.1 · 2 October 2026 · LOCKED** (v2.0 locked 1 October 2026; v2.1 changes are listed in §18)
**Supersedes:** `BanglaIndoorVQA_Experimental_Design_LOCKED.md` v1.0 (18 August 2026), in full.
**Companion handbook:** `BanglaIndoorVQA_guide.md` explains why each choice was made. This file says what to do.
**Target:** Frontiers in Artificial Intelligence, section *Computational Linguistics and Natural Language Processing*, article type *Original Research*.
**Backup:** Frontiers in Computer Science, section *Computer Vision*.

---

## 0. Read this first

### 0.1 Verdict

The design is locked. A full literature check (the October 2026 novelty report plus our own
2026 sweep) found no blocker to the Q2 target. Every part of this design either carries a
contribution or protects one. Anything a Q2 submission does not need was removed (§16).

### 0.2 Rules for using this file

1. This file is the plan. If something is not in this file, it is not run.
2. **No task ownership.** Where people are needed, this file says how many, and any rule such as
   "must be different people". The team lead assigns people.
3. **No dates.** Effort is given in person-days (pd) or GPU hours, for planning only.
4. **Never invent a number.** If a value is not known yet, write `[pending]` in any draft.
5. **Prices and model IDs move.** Verify both in the week of each run (§5, §12).
6. **Any change after lock** needs a dated entry in §18, and a row in the guide's decisions log
   if it changes a decision.

### 0.3 What changed from v1.0

| Area | v1.0 | v2.0 |
|---|---|---|
| Target | Q1 Elsevier, Q2 fallback | Frontiers in AI (Q2-level), CL and NLP section |
| Question types | all 8 types; LLM judge for free-form | 5 closed-answer types; deterministic scoring; no LLM judge |
| Models | 5 closed + 5 open | 3: Gemini 3.6 Flash, Qwen3.5-4B, Gemma 4 E4B |
| Language probe | 2×2 on 2,040 items | 2×2 on 1,118 closed items, plus a no-image arm and answer-language fidelity |
| Naming probe | tiers, forced choice, prompt ablation on 600 | tiers, 4-option forced choice with and without image, specificity prompt on all 1,000, English-collision rule |
| Bangla-specific test | cultural-specificity stratum (35 items) | English-route word-form test (lead result); cultural stratum exploratory |
| Fine-tuning | QLoRA FT-A and FT-B on Qwen3-VL-4B, 1 seed | LoRA on Qwen3.5-4B, one setting, 3 seeds, diagnostic readout |
| Human work | IAA study, judge rating, hazard gold, human ceiling | lexicon, translation, naming reference, scorer check, error audit |
| Pre-registration | OSF | analysis plan committed to GitHub before full runs |
| Compute | several accounts and a local GPU | Colab Pro (primary), one Kaggle account (secondary) |
| API spend | about $146 planned | about $13 expected, $40 hard cap |

### 0.4 Label map (v1 labels are retired)

| v1 | v2 |
|---|---|
| E1 and E2 (language factorial) and E9 (fertility) | **EXP-L** (Language and image) |
| E3 (naming probe) and E4 (prompt ablation) | **EXP-N** (Naming) |
| E6 (fine-tuning, FT-A only) | **EXP-R** (Repair) |
| E5, E7, E8, E10 | cut (§16) |
| CTRL-1 to CTRL-10 | folded into EXP-L, EXP-N and §9; labels retired |
| G1 | passed 10 Sep 2026; still valid |
| G2 to G8 | replaced by Gate A to Gate H (§11) |
| ASSET-1, 2, 4, 5, 8, 9 | kept, re-scoped (§4) |
| ASSET-3, 6, 7 | cut |
| (none) | new: ASSET-10, 11, 12 |

### 0.5 Claims this paper never makes

- That Western training data **causes** the failures.
- That the dataset or any model helps blind or low-vision users. Assistive use is one sentence of
  motivation, nothing more.
- "First" anything without "to our knowledge". Never "first Bangla VQA benchmark" or "first
  culturally grounded benchmark".
- That the gap between multiple-choice and open answers is a finding. It is well known.
- That studying naming specificity in VLMs, or crossing input and output language, is new.
- That the cultural-specificity stratum supports a claim. It is exploratory.
- That the fine-tuning is a method contribution. It is a diagnostic case study.
- Anything about the free-form question types. They are not studied in this paper.

---

## 1. The claim

### 1.1 Title

**Language or Knowledge? Decomposing Vision-Language Model Failure on Bangladeshi Household Interiors**

The question mark stays. The title fits every outcome in §1.4.

### 1.2 The research question in one paragraph

For each of three vision-language models, on natively photographed Bangladeshi homes with
natively written Bangla questions: how much of the error comes from (a) reading Bangla,
(b) writing Bangla, including replying in the wrong language, (c) not using the image, and
(d) not knowing the object or its specific everyday name? And when a model does know an object,
does it fail to name it in Bangla mainly when the Bangla name is **not** simply the English word
written in Bangla letters?

### 1.3 Research questions and hypotheses

- **RQ1 Language (EXP-L).** How large are the reading and writing costs of Bangla, how much does
  the image help in each language, and how often do models reply in the wrong language?
- **RQ2 Naming (EXP-N).** When models name household objects, how often do they give the specific
  everyday name, a generic name, or a confusable sibling? Where does the Bangla penalty sit?
- **RQ3 Repair (EXP-R).** What does a small amount of in-domain Bangla training change?

Each hypothesis is stated before any run. Both directions are publishable.

| # | Hypothesis | Tested by | Supported if | Not supported if |
|---|---|---|---|---|
| H1 | Writing Bangla costs more than reading it, and part of the writing cost is replying in English | EXP-L: P12 (from P1 and P2); fidelity | writing effect larger than reading effect, CI of the difference excludes 0 | reading effect equal or larger |
| H2 | The image helps less when the question is in Bangla, and adding the image lowers Bangla answer fidelity | EXP-L: P3, P4 | P3 > 0 and P4 > 0 | P3 or P4 not above 0 |
| H3 | Generic-name fallback is larger in Bangla answers than in English answers; the specificity prompt reduces it only where the model knows the word | EXP-N: P5, P6 | P5 > 0; P6 > 0 concentrated in language-bound items | P5 not above 0 |
| H4 (lead) | Among objects a model names correctly in English, it names them correctly in Bangla much more often when an English-route Bangla name exists; other loans behave like inherited words | EXP-N plus EXP-L naming items: P7 | P7 > 0 and the origin breakdown shows other loans close to inherited words | P7 not above 0 |
| H5 | Bangla fine-tuning raises Bangla accuracy and fidelity more than English accuracy, and part of the gain appears without the image | EXP-R: P8 | P8 > 0 | P8 not above 0 |

P1 to P8 and P12 are defined in §9.1.

### 1.4 What each outcome means

This replaces v1.0's decision table and its human ceiling. Per model, read the error budget
(§9.3) on Bangla naming errors:

| Largest share of Bangla naming errors | Reading |
|---|---|
| **Language-bound**: fixed when the model answers in English | The model knows the objects but cannot say them in Bangla: a language deficit |
| **Lexical-access-bound**: fixed only when the right Bangla name is offered as an option | The model recognises the Bangla name but cannot produce it: a retrieval deficit |
| **Not known**: wrong in every condition, after the audit removes gold and visibility problems | A knowledge deficit |

Then read H4: if the language-bound errors sit mainly on objects with no English-route name, the
language deficit is about vocabulary, not general Bangla weakness.

Report every model separately. Never average models into one number. All outcomes are
publishable; only the wording of the conclusion changes.

---

## 2. Contributions

In this order:

1. **C1 Word form (lead).** Evidence on where the Bangla naming penalty sits: whether it
   concentrates on objects whose Bangla name is not an English word written in Bangla letters,
   with word origin coded from the Bangla Academy dictionary (English loans, other loans,
   inherited words).
2. **C2 A three-tier Bangla-English household naming lexicon and scoring scheme**, released, with
   an explicit "same-cluster sibling = wrong" rule and an English-collision flag, and what it
   reveals: generic-name fallback, sibling confusion, and the effect of a specificity prompt.
3. **C3 Controlled evidence on natively written Bangla VQA** from a full question-language ×
   answer-language crossing on the same items, with a no-image arm and answer-language fidelity,
   separating reading, writing and image use.
4. **C4 A fine-tuning diagnostic case study** (one LoRA setting, 3 seeds): what small in-domain
   Bangla training changes in Bangla accuracy, English accuracy, no-image accuracy and fidelity.

**Resources:** the lexicon (with English names, tiers, clusters, origin and route flags), English
versions of all probe questions, probe definitions, prompts, scorer, and all model outputs.

**Safe novelty sentence** (use this, or a close variant, in the abstract and introduction):

> To our knowledge, no prior study combines natively written Bangla questions about natively
> photographed homes, a full question-by-answer language crossing with a no-image arm, three-tier
> open naming, and an English-transparency analysis of object names.

---
## 3. Data, splits and probe sets

### 3.1 Dataset facts this paper uses

| Item | Value |
|---|---|
| Images | 4,406 |
| Houses | 68 |
| QA pairs | 18,276 in 8 types |
| Bounding boxes | 3,620 |
| House-disjoint splits | train 43 houses, 3,296 images, 13,668 QA; val 10, 449, 1,863; test 15, 661, 2,745 |
| Capture types | overview 728, group 1,801, close-up 1,877 |
| Object names on box labels | 472 (135 with 5 or more instances) |
| Names tagged for cultural specificity (ASSET-4A) | 145: 6 culturally specific, 138 globally common, 1 ambiguous |
| Prior zero-shot baselines (DiB, relaxed match, short-answer QA) | Qwen3-VL-8B 24.8, Gemma 3 12B 29.6, InternVL3.5-8B 11.3 |

Two cautions before quoting any of these in the paper:

- `stats.json` counts raw surface forms. Per D44, every dataset figure in the paper is regenerated
  with `normalise()` first.
- The DiB run reports 13,869 short-answer and 4,407 free-form QA. `stats.json` implies 13,870 and
  4,406. Find the one item that changed type before citing the DiB numbers.

### 3.2 Gold fixes (do these first)

| qa_id | Type | Problem | Fix |
|---|---|---|---|
| H10_0023_q4 | count | the answer is দরজা, not a count | write the correct count from the image |
| H22_0065_q4 | count | প্রায়তিনটি is missing its space | প্রায় তিনটি (stays excluded: approximate) |
| H56_0011_q3 | spatial | the answer copies the question | fix in the DiB file; not used in this paper |

**Status: done (2 October 2026).** After fixing, re-run the 5A draw: its built-in check must print
"membership identical", meaning the probe items did not change. Apply the same fixes to the DiB
release file.

Corrected on 3 October 2026 in the merged dataset and in the DiB release file:
question of H07_0095_q1, H34_0042_q1, H43_0015_q1, H44_0028_q1, H36_0025_q4; question and
answer of H01_0005_q2, H01_0010_q2, H13_0101_q2, H23_0051_q4; answer of H01_0002_q1.
5A and 5B were re-run and printed "membership identical".

After 3 October 2026 the dataset is not edited for this paper. An item whose gold or question is
found faulty is listed in `probes/excluded_items.csv`; it is run and never scored. The list is
closed at Gate E.

### 3.3 The closed-answer rule

Every string is `normalise()`d first. An item is kept only if its question asks for the
answer kind of its type and its gold is a valid answer of that kind.

An item is **closed-answer** if one of these holds:

- `obj_name`, `object`: the question does not ask the room or a place, the function, the
  location, a count or a risk, and the gold is not a number.
- `count`: the question contains a how-many word from the fixed list in
  `asset5c_probes.py`, and the gold is one exact number. Approximate forms are out.
- `scene_id`: the question asks which room it is, and the gold is one of the four rooms.
  A gold within one character of a room name counts as that room. Part-of-room questions
  are out.
- `obj_property`: the question asks the colour or the material. Risk and other questions
  are out.

Each kept item carries `closed_kind` (name, count, room, colour or material), and count and
room items carry `gold_count` or `gold_room`. The scorer (ASSET-9) must derive the same
values from ASSET-10 and test that they agree.

`spatial`, `obj_function` and `assistive` are not closed-answer and are not studied in this paper.

Applied to P-CORE (2,040 items), the rule gives **P-CORE-CLOSED = 1,118 items**:

| Type | In P-CORE | Kept | Dropped |
|---|---|---|---|
| obj_name | 255 | 255 | 0 |
| object | 255 | 255 | 0 |
| count | 255 | 247 | 8: অনেকগুলো 3, প্রায় … 5 |
| scene_id | 255 | 148 | 107 part-of-room questions or non-room gold |
| obj_property | 255 | 213 | 42: 33 risk, 9 other |
| spatial, obj_function, assistive | 765 | 0 | 765 |
| **Total** | **2,040** | **1,118** | **922** |

P-CORE-CLOSED covers all 68 houses and 979 images. The capture numbers count **items**: close-up
468, group 427, overview 223. It has 343 unique question strings as written, 342 after
normalise(): object 206, count 113, obj_property 14 (13 normalised), obj_name 6, scene_id 6. Two
strings are used by both `obj_name` and `object`.
ASSET-5C stops unless the per-type counts equal the values above.

### 3.4 Probe sets

| Probe | Size | Built by | Status | Used in |
|---|---|---|---|---|
| P-CORE | 2,040 (30 per house, 255 per type), seed 42 | ASSET-5A | DONE | source of P-CORE-CLOSED |
| P-CORE-CLOSED | 1,118 run, 1,109 scored | ASSET-5C, rule in §3.3, no sampling | to run | EXP-L |
| P-NAME | 1,000 run, 991 scored: close-up `obj_name` items (D57, D58) | ASSET-5B (existing script, run as is) | ready to run | EXP-N |
| P-CLUST | 492 run, 483 scored: items in 7 clusters (subset of P-NAME); drawn by exact gold name; 8 P-NAME items whose gold is a merged spelling of a cluster object are scored with the sibling rule but are not in P-CLUST | ASSET-5B | ready to run | EXP-N forced choice, error budget |
| P-FCE | 454 run, 446 scored: P-CLUST items valid for English forced choice (§4.7) | ASSET-11, after Gate A | to run | EXP-N |
| NAME-REF | 200 run, 199 scored: items from P-NAME (§4.8) | ASSET-12, after Gate A | to run | HT-3 |
| R-TEST | 1,449 run (obj_name 280, object 378, count 363, scene_id 209, obj_property 219; 15 houses, 661 images; 246 items are also in P-CORE-CLOSED), 1,435 scored: every closed-answer item in the 15 test houses (rule in §3.3 applied to the full test split) | ASSET-5C | to run | EXP-R |
| PILOT | 100 run, 97 scored: 100 different items, 60 from P-CORE-CLOSED (12 per type), then 40 from P-NAME (20 in P-CLUST, 20 not); 49 houses | ASSET-5C, seeded rank (seed 42) | to run | pilot only |

The seven P-CLUST clusters (G1, counted on `obj_name` items): সংরক্ষণ-আসবাব 138; বৈদ্যুতিক সংযোগ 88;
বাথরুমের কল / পানি ব্যবস্থা 79; টেবিলের ধরন 61; টব / ফুলদানি 43; রান্নার পাত্র 42; ফ্যানের ধরন 41.
Total 492.

**Format rule:** every ASSET-5C output keeps the P-CORE.json record format (qa_id, image_id,
house_id, capture_type, qa_type, question_bn, answer_bn, image_file) and may only add fields. Code
built and tested on P-CORE.json then runs unchanged on every probe set.

The `in_E4_600` flag from D59 is no longer used (D78).

### 3.5 Houses per analysis

| Analysis | Houses resampled in the bootstrap |
|---|---|
| EXP-L | 68 |
| EXP-N | 61 (H39, H55, H64, H65, H66, H67 and H68 have no close-up images) |
| EXP-R | 15 test houses |

---

## 4. Instruments (assets)

### 4.1 Asset table

| ID | What | Status | People | Effort | Unblocks |
|---|---|---|---|---|---|
| ASSET-1 | cluster counts | DONE, G1 passed | none | none | none |
| ASSET-2 | English questions (§4.6) | stage 0 done; re-scoped | 2 post-editors + 2 different back-translators | about 2 pd of human time | every English-question cell (Gate B) |
| ASSET-4A | cultural-specificity tags | DONE, frozen | none | none | none |
| ASSET-4 v2 | `lexicon.csv` v2 (§4.2 to §4.4); one new script replaces the v1 `asset4b_lexicon.py` | not started | 2: writer and checker, never the same person per row | about 4.5 pd human, 0.5 pd code | Gate A |
| ASSET-5B | P-NAME, P-CLUST (existing script, run as is) | ready to run | 1 | minutes | EXP-N, ASSET-11, ASSET-12 |
| ASSET-5C | P-CORE-CLOSED, R-TEST, PILOT | new script | 1 coder | about 0.5 pd | EXP-L, EXP-R, pilot, ASSET-2 |
| ASSET-8 | inference harness (§4.9) | not started | 1 strong coder | about 4 pd | every run |
| ASSET-9 | scorer and at least 60 unit tests (§8) | `normalise.py` done | 1 coder | about 3 pd | every number (Gate F) |
| ASSET-10 | closed-answer vocabulary (§4.5) | new, built inside HT-1 | the same 2 as ASSET-4B | about 0.5 pd | scorer |
| ASSET-11 | forced-choice option sets and P-FCE (§4.7) | new, after Gate A | 1 coder, plus 1 reviewer (HT-6) | about 0.5 pd | EXP-N forced choice |
| ASSET-12 | NAME-REF draw, naming sheet, human names (§4.8) | new, after Gate A | 1 coder, plus 2 namers (HT-3) | about 0.25 pd code, 0.4 pd human | Table T5, Figure F5 |
| FT scripts | EXP-R training and evaluation (§7.3) | new | 1 coder | about 2 pd | EXP-R (Gate G) |
| Stats | analysis notebooks (§9) | new | 1 | about 3 pd | results |

Cut from v1.0: ASSET-3 (IAA study), ASSET-6 (hazard gold), ASSET-7 (human ceiling).

### 4.2 `lexicon.csv` v2 columns

The lexicon holds one object for every naming gold of P-NAME, P-CORE-CLOSED and R-TEST and for every
cluster member (231 objects).

| Column | Content | Rule |
|---|---|---|
| `name_id` | stable ID | "N_" plus the first 10 hex characters of the SHA-256 of the normalised main name |
| `name_bn` | canonical Bangla name | `BanglaIndoorVQA_Lexicon.txt` is the source of truth |
| `variants_bn` | other accepted Bangla names (Tier 1) | only forms attested in the corpus gold answers, or listed as colloquial in the lexicon file. A short name or an English word in Bangla letters accepted in one language is accepted in the other too; if the Bangla form is not attested, neither is accepted. |
| `canonical_en` | main English name | plain everyday English |
| `variants_en` | other accepted English names (Tier 1) | South Asian English terms allowed (almirah, lota). A short name or an English word in Bangla letters accepted in one language is accepted in the other too; if the Bangla form is not attested, neither is accepted. |
| `super_bn`, `super_en` | acceptable generic names (Tier 2) | curated per object, parallel in both languages. A generic cluster member's name (টেবিল, শোকেস) may be listed as Tier 2 for its more specific siblings; the HT-1 pair decides per object |
| `cluster_id`, `siblings` | confusable cluster | D43 applied; two different cluster members are never one object |
| `room_scope` | kitchen, bedroom, living room, bathroom, or any | from the room sections of the lexicon file |
| `cultural_specificity` | the 4A tag of the main name, else of a merged spelling, else `not_rated` | frozen |
| `origin3` | english, other_loan, inherited, or uncertain | §4.3 |
| `origin_note` | the dictionary label as written, or "rule: modern English" | §4.3 |
| `en_route` | yes or no | §4.4 |
| `en_unique` | yes or no | §4.4, computed by script |
| `freq_obj_name` | `normalise()`d count of `obj_name` answers | D44, D57 |

**Cells with several values.** Separator is `|` (`;` is read too). The scorer derives siblings from
`cluster_id`; the `siblings` column is for people. Objects outside every cluster have an empty
`cluster_id`. One Bangla Tier-1 name belongs to one object only; a term cannot be Tier 1 and Tier 2
of the same object (the loader stops on either).

**Hard rule:** no model output may be consulted while building the lexicon. That includes the
pilot, the DiB zero-shot outputs and any test run.

**Freeze (Gate A):** write `lexicon.csv`, run the validator, commit it to GitHub with the tag `gate-A`.
After Gate A nothing in the lexicon changes. Problems found later are handled as sensitivity
analyses (§9.4), never by editing the frozen file.

### 4.3 Origin coding (`origin3`)

1. **Source:** the Bangla Academy dictionary, which marks the origin of each word.
2. **Code the head noun**, the last noun of the name: গ্যাসের চুলা → চুলা; সুইচ বোর্ড → বোর্ড;
   জানালার গ্রিল → গ্রিল; কোলবালিশ → বালিশ.
3. **Map the dictionary's origin label:**
   - ইংরেজি → `english`
   - ফারসি, আরবি, পর্তুগিজ, তুর্কি, হিন্দি, ফরাসি, ওলন্দাজ, or any other foreign language → `other_loan`
   - তৎসম, অর্ধ-তৎসম, তদ্ভব, দেশি → `inherited`
   - The dictionary prints short labels: স. → `inherited`; মুন্ডা, কোল and other early languages of
     the region → `inherited`. If the label gives a chain of languages, code the language Bangla
     took the word from directly.
4. **Words missing from the dictionary:** A word found only under another spelling of the same word
   is coded from that entry, and the note records the spelling. A modern English word written in
   Bangla letters
   (রাউটার, ল্যাপটপ, আইপিএস) is coded `english` with the note "rule: modern English". Anything
   else is `uncertain`.
5. **People:** one person looks up every head noun; a second person checks every row against the
   dictionary. Disagreements get a second look-up and are logged. `uncertain` rows are left out
   of origin analyses only.
6. **Optional helper:** the PyPI package `bangla_dictionary` is built on the Bangla Academy
   dictionary data and returns origin labels. It may pre-fill the column; every pre-filled label
   is still checked in the dictionary itself. It was not used.
7. **Expected, to be confirmed by look-up, never coded from this list:** পর্দা, দরজা, আয়না, বালিশ
   (Persian); আলমারি, বালতি, জানালা (Portuguese); চেয়ার, টেবিল (English).

### 4.4 English route and English uniqueness

**`en_route` = yes** if at least one accepted Bangla Tier-1 name (`name_bn` or `variants_bn`) is
an accepted English name of the same object (`canonical_en` or `variants_en`) written in Bangla
letters. Examples: ফ্রিজ, সোফা, টেলিভিশন, টিভি, এসি.

- Test: read the Bangla name aloud. Would an English speaker recognise it as one of this object's
  accepted English names?
- It follows the accepted English names. কমোড has English origin, but it is `en_route` = yes only
  if "commode" is listed in `variants_en`.
- The whole Bangla name must be the English name. A name that is partly Bangla or carries a Bangla
  ending inside it (পড়ার টেবিল, সসের বোতল) is `en_route` = no.
- Coded by the lexicon writer, checked by the checker. A script cross-check flags rows where the
  romanised Bangla name is close to an English Tier-1 term (normalised edit similarity 0.6 or
  more, or an acronym match). Every disagreement between the script and the people is reviewed
  and logged.

**`en_unique` = yes** if the object's English Tier-1 set (`canonical_en` plus `variants_en`,
lower-cased, articles removed, singular) shares **no** term with any other object's English Tier-1
set. A script computes it over the whole lexicon.

- Shared in the frozen file: pot and cooking pot (হাঁড়ি, পাতিল, দেকচি), bowl (গামলা, বাটি), jar
  (কৌটা, বয়াম), strainer (চালুনি, ছাঁকনি, ঝাঁঝরি), blanket (কম্বল, লেপ), quilt (কাঁথা, লেপ), rug
  (মাদুর, শীতলপাটি), hanger (ওয়াল হ্যাঙ্গার, হ্যাঙ্গার). **17 objects are `en_unique` = no.**
  Wardrobe, stool and knife are **not** shared: the pair judged each to be the name of one object
  only (ওয়ারড্রব, টুল, ছুরি).
- **No stretching.** Do not invent rare English words to force uniqueness. If the natural English
  name is shared, the object is `en_unique` = no.
- **Use:** every Bangla-versus-English naming comparison uses `en_unique` = yes items only.
  `en_unique` = no items are analysed in Bangla only.

### 4.5 Closed-answer vocabulary (ASSET-10)

Written in full by the HT-1 pair and frozen with the lexicon at Gate A.

- **Rooms:** 4 classes, each with Bangla and English accepted forms, including common English
  forms in Bangla letters (বেডরুম, কিচেন, ড্রয়িং রুম).
  A short name or an English word in Bangla letters accepted in one language is accepted in the
  other too; if the Bangla form is not attested, neither is accepted.
- **Materials:** classes built from the 53 distinct gold strings in P-CORE-CLOSED. The 53 are 37
  material and 16 colour strings. R-TEST adds 15 more property strings. ASSET-10 must cover all of
  them. Merge what a
  careful person cannot tell apart in a photo: steel, iron, aluminium and "metal" → metal;
  পারটেক্স, board and wood → wood; কাচ, কাঁচ and গ্লাস → glass.
- **Colours:** basic colour classes. Ignore lightness words (হালকা, গাঢ়). A compound gold such as
  সবুজ-হলুদ becomes the set {green, yellow}.
- **Numbers:** Bangla number words with classifiers (এক, একটি, একটা, দুই, দুইটি, দুটি, দুটো …),
  Bangla digits, Latin digits, English number words.
  A number form that is also a word of a refusal phrase is not listed (the bare নয়).

`closed_vocab.csv` has columns `kind, class_id, forms_bn, forms_en, note`, one row per class. `kind`
is room, material, colour or number. For numbers `class_id` is the number in Latin digits.
`forms_bn` holds every accepted form in Bangla letters (English words in Bangla letters included);
`forms_en` every form in Latin letters. A form may sit in one class of a kind only.

### 4.6 English questions (ASSET-2)

**Scope:** 653 normalise()d strings: 649 (P-CORE-CLOSED 342, R-TEST adds 307) plus all 4 that
P-NAME adds (2 of them in P-CLUST). Two of the 4 serve no English-question condition; they were
translated so that every question string of every probe file has an English version. PILOT has 32
strings; they are translated first. **Answers are not translated:** naming answers use the
lexicon's English names; count, room, material and colour answers use ASSET-10.

| Stage | What | People | Effort |
|---|---|---|---|
| 1 | Machine-translation first pass with Gemini 3.6 Flash, **paid tier only**, prompt `mt-v2` (9 rules, text in `scripts/asset2_stage1_mt.py` and in the run manifest). Calls of 15 to 20 strings. Naming questions (`obj_name`, `object`) never share a call with count, room or property questions. A call holds question strings only: no answers, no images, no qa_ids | 1 | about 1 h |
| 2 | Post-edit 100% of strings; each post-editor re-checks a random 10% of the other's half and every string used by 10 or more items | 2 | about 12 h total |
| 3 | Blind back-translation of 150 strings drawn by seeded rank (seed 42) from the final English file into Bangla, without seeing the originals (75 each); the back-translators see item codes and the English only | 2, **different from stage 2** | about 3 h total |
| 3b | Post-editors compare back-translations with originals; mark each pair faithful or not (same meaning and difficulty); every not faithful pair counts, whatever its cause | the stage 2 pair; each pair is judged once, by the post-editor who did not post-edit it | about 1 h |
| 4 | Leakage scan (script, after Gate A): no naming question may contain a Tier-1 term of its gold object, found with the scorer's matching rules (English terms in the English question, Bangla terms in the Bangla question); excluded items are reported and do not count | script | minutes |

Output: `english_questions_final.csv`, one row per normalise()d string, English in `question_en`,
joined to items by `question_bn_norm` (addendum D4).

**Hard rules for post-editing:** keep the question's difficulty; do not disambiguate; never add
the object's name; keep the riddle style of `object` questions; keep plain household English;
use American spelling; give one Bangla wording or object name one English form in every question;
for an object of the frozen lexicon use its `canonical_en`; write a named object without an everyday
English word in Latin letters, without explanation.

**Pass (Gate B):** at least 95% faithful in stage 3b (report the Wilson 95% CI), and zero leaks in
naming questions (the rule of stage 4). If it fails, fix and repeat stage 3 on a fresh 150.
Pass at 143 of 150 or more. Faults found in a passing stage 3b are corrected in a new version of the
file; the reported percentage is the one measured before the corrections.

### 4.7 Forced-choice option sets (ASSET-11)

Built by the ASSET-11 script, after Gate A. Every P-CLUST item (492) gets **exactly 4 options**:

1. the gold name;
2. the 2 most frequent other members of its cluster (by `freq_obj_name`); equal counts in the order
   of the seeded rank `SHA-256("42|ASSET-11|tie|<name_id>")`. If the cluster has only
   one other member, use it and fill the free slot with the most frequent object of the same
   `room_scope` outside the cluster;
3. one out-of-cluster object of the same `room_scope` (most frequent not already used). It doubles
   as a guessing check.

- Same `room_scope` means the same value (`any` is a value). A cell with several rooms stops the
  build.
- Never use as a distractor an object whose name is in the gold's Tier-2 set (any Tier-1 name of the
  candidate, Bangla or English) (for example টেবিল for
  a পড়ার টেবিল); take the next candidate instead.
  A more specific kind of the gold is kept and marked for the reviewer.
- Order: the four slots sorted by `SHA-256("42|ASSET-11|order|<qa_id>|<slot>")`, identical on every
  run. Labels ক খ গ ঘ
  in Bangla, A B C D in English.
- Bangla options use `name_bn`, English options `canonical_en`; the gold is shown under the main
  name of its object. The English set holds the same four objects in the same places as the Bangla
  set.
- **English forced choice (P-FCE)** only where the gold is `en_unique` = yes **and** the four
  English labels are all different (as the scorer reads English forms), decided on the final sets
  after HT-6.
- **HT-6 review:** one person checks every option set for a second correct option (for example a
  distractor that is the main object too), swaps it, in the same place, for the next candidate of
  that slot, and logs the change.
  About 1 h for 492 sets.
  The review uses one sheet that shows the picture with the Bangla and English names; a replacement
  is checked at once; the final sets are rebuilt from the log. A doubt about a gold is flagged and
  never changed (D110).

### 4.8 Human naming reference (ASSET-12)

- **Draw NAME-REF (ASSET-12 script, after Gate A):** 200 P-NAME items, 50 from each cell of (in
  P-CLUST: yes or no) × (`en_route`: yes or no), by seeded rank (seed 42). If a cell has fewer than
  50, fill from the largest cell and log it (the cell with the most available items after the house
  exclusion, next items in its rank order; a filled item keeps its true cell). The rank is
  `SHA-256("42|NAME-REF|<qa_id>")`; cells come from resolve_gold on the frozen lexicon and from
  P-CLUST membership.
  Exclude items from houses annotated by either naming person.
- **Namers:** 2 native Bangla speakers. Each names all 200 independently, image only (the same
  resized images the models see), in random order, one or two words, with no dictionary,
  internet or help.
  Each namer gets a separate seeded order and codes that hide the qa_id; the images are
  byte-identical copies of the prepared images; a separate note asks for Bangla letters.
- **Instruction (shown in Bangla):**
  > ছবিতে মূল যে জিনিসটি দেখা যাচ্ছে, বাসায় আপনি সেটিকে যে নামে ডাকেন সেই নামটি লিখুন (এক বা দুই শব্দে)।
  > নিশ্চিত না হলে যা মনে হয় তাই লিখুন। কোনো অভিধান, ইন্টারনেট বা অন্য কারও সাহায্য নেবেন না।
- **Order rule (Gate H):** done before either namer sees any model output or scorer output.
- Scored by the same scorer as the models (Bangla, strict) (condition N-BB; reported numbers only
  with the gate-F scorer). About 1.5 h per person.

### 4.9 Inference harness (ASSET-8)

Must have:

- one config file that defines every condition;
- a per-call cache keyed by (model_id, condition_id, prompt_version, qa_id);
- retry with exponential backoff;
- the raw response logged for every call, including Gemini's thinking-token count;
- resumable runs;
- a running cost meter with a **hard stop at $40** total API spend;
  One shared meter: `src/cost_ledger.py` and the ledger file Drive `outputs/api_cost_ledger.jsonl`.
  Every billed script appends one line per call and checks the total before each call (D128).
- a Gemini Batch API path, and a Hugging Face transformers path for Qwen3.5 and Gemma 4 (greedy);
- one `build_prompt()` function, imported by both the harness and the fine-tuning scripts;
- an image loader that applies EXIF orientation first, then resizes (§5.2).
- three stages (smoke, pilot, main) with separate caches; stage main needs the tag `gate-E` and a
  clean Git tree.
- Option file: one record per item and language (`qa_id`, `lang`, `options`, `correct_index`,
  `option_ids`, `roles`, `in_pfce`, `ht6_changed`); the draft file cannot be run.

**Acceptance (Gate C):** the pilot re-runs from cache at zero cost, and at least 98% of answers
are non-empty and parseable (per model, D163 and D190: an open answer counts when it is not empty; a
forced-choice answer when it is not empty and holds one shown label, names one option, or is a
refusal).

---
## 5. Models and inference settings

### 5.1 Roster

| Model | ID (verify on the model card) | Access | Role |
|---|---|---|---|
| Gemini 3.6 Flash | `gemini-3.6-flash` | Gemini API, Batch mode, paid tier | frontier closed model; supports minimal thinking |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | open weights (Apache 2.0), Colab L4 | open model, family 1; fine-tuning target |
| Gemma 4 E4B | `google/gemma-4-E4B-it` | open weights (Apache 2.0), Colab L4 | open model, family 2, different tokenizer |

### 5.2 Fixed settings (all models, all conditions)

| Setting | Value |
|---|---|
| Prompt placement | instruction and question in the user turn; no system prompt |
| Decoding | Qwen3.5-4B and Gemma 4 E4B: greedy (`do_sample` false, one beam, repetition penalty 1.0, batch size 1). Gemini 3.6 Flash: no sampling parameter is sent, because the API ignores temperature, top_p and top_k; reported outputs are fixed by the response cache. |
| Output length | open models: `max_new_tokens` 32. Gemini: `max_output_tokens` 256, to leave room for minimal thinking; the visible answer is still short. For Gemini the limit includes thinking tokens; ASSET-2 stage 1 reported 0 thinking tokens at `minimal` in the 34 calls whose reports were read |
| Thinking | Gemini `thinking_level` = minimal (log `thoughts_token_count`); Qwen3.5 `enable_thinking` = False; Gemma 4 thinking off |
| Images | apply EXIF orientation; convert a non-sRGB colour profile to sRGB; resize so the longer side is 1,024 px (Lanczos, never enlarged); save as JPEG quality 90 without metadata; one prepared copy per image in Drive `data/images_1024/`, shared by all models |
| Qwen3.5 processor | `max_pixels` = 786,432 (1,024 × 768) |
| Gemma 4 processor | visual token budget 1,120 per image (`max_soft_tokens`); other values default |
| Gemini image setting | `mediaResolution` = MEDIA_RESOLUTION_HIGH on every image request; the image part comes before the text part |
| Precision | bf16 on L4 |
| Libraries | transformers 5.18.0 for every open-model run; peft 0.21.0 for training and adapter evaluation, with `torchao` removed from those sessions |

For the common 3:4 phone photos, a 1,024 px longer side equals the 768 px short side used in v1.0.

### 5.3 Platform rules

- Colab Pro with an **L4** is the default for everything. An A100 is allowed for fine-tuning only.
- **Never split one model's runs across platforms.** Kaggle's T4 has no bf16, so its outputs can
  differ slightly.
- Kaggle (one account, T4) may run **all** inference conditions of **one** open model when Colab is
  unavailable. Gemma 4 never runs on a T4 (known fp16 overflow issue). Qwen3.5 runs on a T4 only
  after a 20-item smoke test completes without errors.
- Fine-tuning never runs on a T4.
- Qwen3.5 runs with `flash-linear-attention` installed and with the PyTorch fallback for
  `causal_conv1d`. The harness records this state and stops if it changes.

### 5.4 Floor gate (Gate D)

From the pilot: if an open model scores **below 20% strict accuracy on the 60 closed pilot items (58
scored after D160 and D161)
in EN→EN**, replace it with the next size in its family (Qwen3.5-9B, or Gemma 4 12B) for EXP-L and
EXP-N.

A low **Bangla** score is never a reason to replace a model. That gap is what we measure.

Fine-tuning target: the lighter non-floored model, in this order: Qwen3.5-4B, then Gemma 4 E4B.

### 5.5 Gemini rules

- Run every Gemini condition inside one 7-day window. Record the model ID and run dates.
- If Google changes or retires the model during the window, re-run all Gemini conditions on the
  new one (about $13).
- No fallback is fixed in advance (D162). Each Gemini full run first checks that version
  3.6-flash-07-2026 is served. If Google sets a shutdown date or serves another version before the
  runs end, stop, choose the replacement in a decision row, and run the Gemini pilot conditions on
  it first. Never use a model ID that the API redirects to another model.
- **Paid tier only.** Free-tier data may be used to improve Google's products, which would leak
  our test items.
  `gemini-3.6-flash` also has a free tier, and free-tier content is used to improve Google's
  products. No API call reports the tier. Every billed script therefore needs the runner's flag
  `--paid-tier-confirmed`, given after a look at the AI Studio Projects page.

### 5.6 Reserve models

Not run unless a reviewer asks (§16): Qwen3.5-9B; one OpenAI model.

---

## 6. Prompts

### 6.1 Rules

- The **whole prompt is in the question language**, including the sentence that sets the answer
  language.
- One template per (question language, answer language). The same templates serve all three
  models and all five closed types.
- Zero-shot: no examples, no chain-of-thought.
- No-image versions remove **only** the clause that mentions the image.
- Prompts live in `prompts.yaml` with a version number. They may change between the pilot and
  Gate E only. After Gate E, any change re-runs every affected condition.

### 6.2 Templates

**L-BB and N-BB (question Bangla, answer Bangla):**
```
নিচের প্রশ্নটির উত্তর ছবিটি দেখে দিন। উত্তর শুধু বাংলায় লিখুন। উত্তর যতটা সম্ভব ছোট রাখুন: শুধু নাম, সংখ্যা বা এক-দুই শব্দ। কোনো ব্যাখ্যা দেবেন না।
প্রশ্ন: {question_bn}
উত্তর:
```

**L-BE and N-BE (question Bangla, answer English):**
```
নিচের প্রশ্নটির উত্তর ছবিটি দেখে দিন। উত্তর শুধু ইংরেজিতে লিখুন। উত্তর যতটা সম্ভব ছোট রাখুন: শুধু নাম, সংখ্যা বা এক-দুই শব্দ। কোনো ব্যাখ্যা দেবেন না।
প্রশ্ন: {question_bn}
উত্তর:
```

**L-EE (question English, answer English):**
```
Answer the question below by looking at the image. Write the answer in English only. Keep the answer as short as possible: only a name, a number, or one or two words. Do not explain.
Question: {question_en}
Answer:
```

**L-EB (question English, answer Bangla):**
```
Answer the question below by looking at the image. Write the answer in Bangla (Bengali script) only. Keep the answer as short as possible: only a name, a number, or one or two words. Do not explain.
Question: {question_en}
Answer:
```

**No-image versions (L-BB0, L-EE0, R-BB0):** the same text without the image clause, and no image
attached:
```
নিচের প্রশ্নটির উত্তর দিন। উত্তর শুধু বাংলায় লিখুন। … (rest unchanged)
Answer the question below. Write the answer in English only. … (rest unchanged)
```

**N-BBs (specificity prompt):** the N-BB template with one sentence added after the
answer-language sentence:
```
জিনিসটির সাধারণ শ্রেণির নাম নয়, বাংলাদেশের বাসাবাড়িতে একে যে নির্দিষ্ট নামে ডাকা হয় সেই নামটি লিখুন।
```

**N-FCB (forced choice, Bangla):**
```
ছবিটি দেখে নিচের প্রশ্নটির উত্তর দিন। চারটি বিকল্পের মধ্য থেকে সঠিক নামটি বেছে নিন। শুধু বিকল্পের অক্ষরটি লিখুন: ক, খ, গ অথবা ঘ।
প্রশ্ন: {question_bn}
ক) {option_1}
খ) {option_2}
গ) {option_3}
ঘ) {option_4}
উত্তর:
```

**N-FCE (forced choice, English):**
```
Answer the question below by looking at the image. Choose the correct name from the four options. Write only the option letter: A, B, C or D.
Question: {question_en}
A) {option_1}
B) {option_2}
C) {option_3}
D) {option_4}
Answer:
```

**N-FCB0 and N-FCE0:** the forced-choice templates without the image clause, no image attached.

The team proofreads the Bangla templates once before the pilot. Changes are allowed only before
Gate E.

---

## 7. Experiments

Condition IDs: the first letter is the experiment (L, N, R). Then the question language and the
answer language (B = Bangla, E = English). A trailing `0` means no image.

### 7.1 EXP-L: Language and image factorial

| Item | Value |
|---|---|
| Question | How much do reading Bangla, writing Bangla and using the image each matter, and how often do models reply in the wrong language? |
| Items | P-CORE-CLOSED, 1,118 items, 68 houses |
| Models | all three |
| Conditions | with image: L-BB, L-BE, L-EB, L-EE. Without image: L-BB0, L-EE0 |
| Baseline B0 | per item, the most common gold answer among items with the same question template from **other** houses (leave one house out); if the template appears nowhere else, the most common answer of the type. Computed from the full annotation file, no model needed |
| Outcomes | strict accuracy, lenient accuracy, fidelity, refusal rate; for `obj_name` and `object` items also the tiers, `en_route` and `en_unique` |
| Primary contrasts | P1, P2, P3, P4 (§9.1) |
| Calls per model | 4,472 with image, 2,236 text only |

### 7.2 EXP-N: Naming probe

| Item | Value |
|---|---|
| Question | Do models give the specific everyday name, a generic name or a sibling? Can they pick the right Bangla name when shown? Where does the Bangla penalty sit? |
| Items | P-NAME 1,000 (61 houses); P-CLUST 492; P-FCE 454 (446 scored) |
| Models | all three |
| Open naming | N-BB, N-BE, N-BBs (all 1,000 each) |
| Forced choice | N-FCB and N-FCB0 on P-CLUST; N-FCE and N-FCE0 on P-FCE |
| Outcomes | tiers, generic fallback, sibling confusion, English copies, fidelity, transfer rate, forced-choice accuracy against the no-image run and against chance (25%) |
| Analyses | error budget (§9.3); word form by `en_route` (primary) and `origin3` (secondary); cultural stratum (exploratory, descriptive only) |
| Primary contrasts | P5, P6, P7 (§9.1). P7 pools the `obj_name` and `object` items of EXP-L with P-NAME (each qa_id counted once) |
| Calls per model | 3,792 with image (planning), 792 text only |

### 7.3 EXP-R: Repair test (diagnostic case study)

**Question.** What does small in-domain Bangla training change: Bangla accuracy, English accuracy,
no-image accuracy, fidelity, or naming specificity?

**Model:** Qwen3.5-4B (or the substitute from §5.4). **Training data:** the train split (43
houses, 13,668 QA, all 8 types), Bangla questions with Bangla gold answers, formatted with the
L-BB template, with the image. The list of excluded items (D110) is not applied to training or
validation data. **Validation:** the fixed list FT-VAL400: 400 items of the val split by seeded rank
(seed 42), all types, used for
loss and early stopping only.

**Configuration (one setting):**

| Setting | Value |
|---|---|
| Method | LoRA; base weights in bf16 and frozen, LoRA weights in float32, forward pass under bf16 autocast (fits an L4) |
| Trainable parts | all linear layers of the language model (PEFT LoRA on every `nn.Linear` inside the language-model layers; the list of adapted layers is written to the run manifest). Vision encoder and projector frozen. Print and record the trainable-parameter count |
| LoRA | r = 16, alpha = 32, dropout 0.05 |
| Optimiser | learning rate 1e-4, cosine schedule, warmup ratio 0.03, weight decay 0.01; AdamW (betas 0.9 and 0.999), gradient clipping 1.0 |
| Batch | per device 2, gradient accumulation 8 (effective 16); run as single-example passes with the pooled token loss, so no padding |
| Length | max sequence 1,536 tokens; images as §5.2; a longer example is left out and counted |
| Epochs | up to 2; validation loss every 0.25 epoch; early stopping with patience 2; keep the best checkpoint |
| Loss | on the answer tokens and the end of turn only (prompt tokens masked) |
| Template | the same `build_prompt()` as inference; thinking disabled |
| Seeds | 42, 43, 44 (each sets the LoRA initialisation and data order) |
| Saved | LoRA adapters only |

**Dry run (Gate G), before any long run:** 100 training examples, 1 epoch. Check that:

- the loss does not fall near 0 within 50 steps (that would mean the prompt is not masked): mean
  loss of steps 41 to 50 at least 0.10 and no step below 0.01;
- generation on 20 R-TEST items returns short Bangla answers: at least 18 of 20 answers not empty,
  in Bangla script or digits, and of 8 words or fewer;
- the adapters save, reload, and give the same answers on those 20 items.
- every example has the answer and the end of turn as its only loss positions;
- an overfit probe on 8 training examples reproduces at least 7 gold answers and stops by itself
  (learning rate 5e-4, at most 60 updates, stop at loss 0.05);
- seconds per item are measured and projected against §12.2.

**Evaluation:** R-TEST × {R-BB, R-EE, R-BB0} × {base model, seed 42, seed 43, seed 44}, with the
same prompts and scorer as EXP-L. Base-model items already run in EXP-L are reused from the cache.

**Readouts:** ΔBN, ΔEN, Δno-image, Δfidelity, ΔTier-1, Δfallback and answer length, each as
fine-tuned minus base. P8 = ΔBN − ΔEN.

**Pre-declared readings:**

| Pattern | Reading |
|---|---|
| ΔBN > 0, ΔEN ≈ 0, Δno-image close to ΔBN | mostly answer habits and dataset priors, not visual knowledge |
| ΔBN > 0 and ΔEN > 0 | the gain carries across languages: some knowledge was added |
| ΔBN > 0, mostly through higher fidelity | the main fix is Bangla output habit |
| all Δ ≈ 0 | small in-domain data does not repair the deficit; report it plainly |

**If a seed fails twice,** report the seeds that ran and state it as a limitation. **If no seed
completes,** cut EXP-R. The paper stands on EXP-L and EXP-N (pre-declared, D72).

### 7.4 Experiment table

| Experiment | Conditions | Items per condition | Models | Gemini calls | GPU (L4 hours, planning) |
|---|---|---|---|---|---|
| Pilot | all EXP-L and EXP-N conditions | 60 closed, 40 naming | 3 | about 560 (standard API) | about 1 (both open models) |
| EXP-L | L-BB, L-BE, L-EB, L-EE, L-BB0, L-EE0 | 1,118 | 3 | 4,472 with image, 2,236 text | about 2 per open model |
| EXP-N | N-BB, N-BE, N-BBs; N-FCB, N-FCB0; N-FCE, N-FCE0 | 1,000; 492; 454 | 3 | 3,792 with image, 792 text | about 1.5 per open model |
| EXP-R training | 1 setting × 3 seeds | 13,668 training QA | Qwen3.5-4B | 0 | about 18 |
| EXP-R evaluation | R-BB, R-EE, R-BB0 × 4 model versions | 1,449 | Qwen3.5-4B | 0 | about 5.5 |
| Translation first pass | ASSET-2 stage 1 | 653 strings | Gemini | 34 with `mt-v2`, plus 2 with `mt-v1` | 0 |
| Smoke tests and dry runs | | | | 120 (20 standard, 100 in one Batch job) | about 2 |

---
## 8. Scoring (deterministic, no LLM judge)

All scoring is done by ASSET-9 with the frozen lexicon (§4.2) and vocabulary (§4.5). No human or
model judges any answer during scoring.

Items listed in `probes/excluded_items.csv` are scored as excluded: no label and no score, counted
apart in every summary (D110).

**Exact-gold rule:** a naming gold must equal a Bangla Tier-1 name of one object, and a room gold
must equal a listed room form, after normalise(), lead-in removal and the spacing rule. Otherwise
the item is unscoreable (strict and lenient empty) and reported, never mapped by guess. Count,
material and colour golds are read with the ending rule.

### 8.1 Normalise

Every gold answer and every model answer goes through `normalise.py` (Unicode NFC; remove
zero-width joiners not preceded by hasant; collapse whitespace; strip the final dari and
punctuation; lower-case Latin). Import it; never copy it. Then strip a fixed list of lead-in
phrases (এটি একটি, এটা, এটি, it is a, this is a, it's a).

### 8.2 Answer language and fidelity

- Count Bangla letters (Unicode block U+0980 to U+09FF) and Latin letters in the answer.
- **Bangla** if Bangla letters are at least half of all letters; **English** if Latin letters
  are; otherwise **mixed**.
- Bangla vowel signs count as letters, as Latin vowels do. Language is detected after the lead-in is
  removed. "Mixed" can only arise when a third script is present. Digits only: `digits`. Nothing:
  `none`.
- An answer of digits only matches any requested language.
- **Fidelity** = the detected language equals the requested answer language.

### 8.3 Naming items (`obj_name`, `object`)

1. Search the answer for every lexicon term (Tier 1 and Tier 2, every object, both languages) as
   whole words. Prefer the longest match, so পড়ার টেবিল wins over টেবিল.
   **Spacing rule:** a name matches whether or not its words are separated by a space or a hyphen
   (সুইচ বোর্ড, সুইচবোর্ড, সুইচ-বোর্ড). **Ending rule:** a closed list of 16 Bangla noun endings
   (classifiers, plural, genitive, locative, objective; up to two in a row) and the English plural
   and possessive are accepted on the last word of a name, and only when what is left is a listed
   term.
2. Assign one label:

| Label | Rule | Strict | Lenient |
|---|---|---|---|
| **Tier 1** | a Tier-1 term of the gold object is found, and no Tier-1 term of any other object | correct if in the requested language | correct |
| **Tier 2** | no gold Tier-1 term; a Tier-2 (generic) term of the gold object; no other object's Tier-1 term | wrong | wrong (counted separately as generic fallback) |
| **Sibling** | a Tier-1 term of a same-cluster sibling is found | wrong | wrong |
| **Ambiguous** | Tier-1 terms of two or more different objects are found | wrong | wrong |
| **English copy** | the answer is in Bangla letters, no lexicon term matches, and its romanised form is close to an English Tier-1 term of the gold (normalised edit similarity 0.6 or more). The last one, two or three Bangla words of the answer are romanised with a fixed table and compared. No word endings are removed. The 0.6 threshold is tested in exact arithmetic | wrong | correct |
| **Other wrong** | everything else, including refusals and empty answers | wrong | wrong |

**Sibling or ambiguous.** Gold Tier-1 term plus any other object's Tier-1 term: ambiguous. Two other
objects named: ambiguous. One other object named: sibling if it is in the gold's cluster, else other
wrong. A shared English name (pot) counts as the gold's own name.

3. A Tier-1 term in the **wrong language** (for example "fridge" when Bangla was requested) is
   wrong under strict scoring and correct under lenient scoring, and it counts against fidelity.
4. **Generic-name precedence.** If a found term is in the gold object's Tier-2 set, the label is
   Tier 2, even when the same term is another object's Tier-1 name (টেবিল for a পড়ার টেবিল). A name
   more specific than the gold counts as a sibling unless the lexicon lists it as a Tier-1 variant
   of the gold.

### 8.4 Count, room, material and colour

- **Count:** take the first number in the answer (Bangla words with classifiers such as টি, টা,
  খানা; Bangla digits; Latin digits; English words). Correct if it equals the gold number.
- **Room:** map the answer to one of the 4 room classes with ASSET-10. Correct if it matches the
  gold class. An answer naming two rooms is wrong.
- **Material and colour:** find every class term in the answer. Correct if at least one is found
  and every class found is in the gold set.
  The kind of an `obj_property` item is taken from its gold through ASSET-10, never from the
  question text.

Strict scoring additionally requires the requested language for these types, except for
digit-only counts.
**Strict rule:** the correct name, class or number must itself be written in the requested language
(at least one correct term in that language). Fidelity (§8.2) is a separate score on the whole
answer. A count given in digits is strict-correct if the answer is digits only or the whole answer
is in the requested language.

### 8.5 Forced choice

Find the option labels in the answer. Only the labels that were shown count (ক খ গ ঘ in Bangla
lists, A B C D in English lists); a label from the other script, or a digit, is invalid and flagged.
An English “a” is a label only when it stands alone, is marked like a label (“a)”, “(a)”, “a.”,
“a:”, “[a]”), follows option, choice or letter, or ends the answer. If one distinct label is named,
take it. If two or more are named, take the one label that an answer word (উত্তর, answer) marks
within the three words before it; if no label or several labels are marked, the answer is invalid.
If there is no label, match the option text. Otherwise the answer is invalid and scored wrong
(D169).
The options and the gold position are read from the ASSET-11 option file through
`src/fc_options.py`; the labels are those of the question language; the option file's sha256 is
stored with every scored answer.
`invalid` exists only in forced choice. An open answer always gets a label (an empty answer is
other_wrong or wrong).

### 8.6 Strict and lenient scores

- **Strict** (the headline score): the right answer in the requested language.
- **Lenient**: the right object or value in any language, including English copies. It measures
  knowledge of the object regardless of language.
- **Refusal rule:** an empty answer, or an answer that holds no scoreable content and contains a
  phrase from the fixed refusal list in `scoring.py`.

**What the scorer returns.** One flat record per answer, the same keys every time. The ones the
statistics need:
`label` (tier1, tier2, sibling, ambiguous, english_copy, other_wrong; or correct, wrong,
invalid), `strict`, `lenient`, `fidelity`, `lang`, `refusal`, `is_empty`, `n_words`,
`scoreable` and `gold_status`, `gold_name_id`, `in_cluster`, `en_route`, `en_unique`,
`label_lang`, `mixed_script`, `named_ids`, `copy_term`, `copy_sim`, `content_found`,
`refusal_phrase`, and the `fc_*` fields.
Unscoreable items have empty `strict` and `lenient` and must be counted, not dropped silently.

### 8.7 Unit tests and freeze (Gate F)

- At least 60 unit tests (445 at the Pilot reading, D169): every rule in §8.3 to §8.5 in both
  languages, English copies, the
  generic-name precedence, ambiguity, numbers in every form, compound colours, refusals and empty
  answers; and a test that the scorer derives the same `closed_kind`, `gold_count` and `gold_room`
  as ASSET-5C on every closed item that is not excluded.
- After the tests pass and HT-4 passes (§10), freeze the scorer: commit it with the tag `gate-F`.
- Any later change re-scores every answer, and is logged in §18.
- Before Gate E, one native Bangla speaker proofreads the ending list and the refusal phrases on the
  no-image pilot answers, blind to the scorer. A change is a scorer change: new version, all tests
  again, a decision row.

---

## 9. Analysis plan

This plan is fixed before any full run. It is committed to GitHub at Gate E with the tag `gate-E`;
the commit ID and date go into the paper's Methods.

### 9.1 Primary contrasts

For each zero-shot model (Gemini, Qwen3.5-4B, Gemma 4 E4B). Accuracy means strict accuracy unless
stated.

| # | Contrast | Definition | Hypothesis |
|---|---|---|---|
| P1 | Reading cost | ((L-EB − L-BB) + (L-EE − L-BE)) / 2 | H1 (estimate) |
| P2 | Writing cost | ((L-BE − L-BB) + (L-EE − L-EB)) / 2 | H1 (estimate) |
| P12 | Writing minus reading | P2 − P1 | H1 (> 0) |
| P3 | Image help, English minus Bangla | (L-EE − L-EE0) − (L-BB − L-BB0) | H2 (> 0) |
| P4 | Image-induced fidelity loss | fidelity(L-BB0) − fidelity(L-BB) | H2 (> 0) |
| P5 | Fallback, Bangla minus English | fallback(N-BB) − fallback(N-BE), `en_unique` items | H3 (> 0) |
| P6 | Specificity prompt effect | Tier-1 rate(N-BBs) − Tier-1 rate(N-BB) | H3 |
| P7 | Word form | transfer rate(`en_route` = yes) − transfer rate(`en_route` = no), `en_unique` items | H4 (> 0) |

For the fine-tuned model (EXP-R only):

| # | Contrast | Definition | Hypothesis |
|---|---|---|---|
| P8 | Repair, Bangla minus English | ΔBN − ΔEN, averaged over the 3 seeds | H5 (> 0) |

**Test:** a house-clustered paired bootstrap for every contrast (§9.2), with two-sided p-values.
**Multiple comparisons:** Holm correction within each model across the six hypothesis tests P12, P3,
P4, P5, P6 and P7. P1 and P2 are reported as estimates with CIs. P8 is a single test.
**Confirmatory models:** the GEE models in §9.2 must agree in direction with P1, P2 and P7.

### 9.2 Statistical methods

- **House-clustered bootstrap.** Resample houses with replacement, 10,000 times (68 houses for
  EXP-L, 61 for EXP-N, 15 for EXP-R). Recompute every quantity in each resample. Report the 2.5th
  and 97.5th percentiles as the 95% CI. Paired differences are computed inside each resample.
  Two-sided p = 2 × the smaller of (share of resamples at or below 0, share at or above 0).
  One seeded draw (seed 42) per house set; EXP-N quantities, P-CLUST and P-FCE included, resample
  the 61 EXP-N houses; a p of 0 is reported as below 2/B, and a Holm value derived from it is
  marked.
- **GEE.** A logistic regression that accounts for answers from the same house being related
  (statsmodels GEE, binomial family, exchangeable correlation, `groups = house_id`):
  - EXP-L, with-image cells: `correct ~ q_bn * a_bn + C(qa_type) + overview` (close-up equals
    obj_name or obj_property in P-CORE-CLOSED, so capture enters as one overview indicator)
  - Word form, naming items with a Bangla question (N-BB, N-BE and the EXP-L naming items in L-BB
    and L-BE), `en_unique` only:
    `correct ~ a_bn * en_route + log(freq_obj_name + 1) + C(item_type) + in_cluster`.
    The `a_bn:en_route` interaction is the formal test of H4.
- **Template sensitivity.** Re-fit the EXP-L GEE with `groups = template_id`, where `template_id`
  is the normalised question text with spelling variants of one template merged. Report it in the
  supplement; if it is more conservative, say so in the text.
  The template fold (D236): normalise(), then remove chandrabindu, ্যা to ্য, ী to ি, ঈ to ই, ূ to
  ু, ঊ to উ, ণ to ন, ষ and স to শ, ঙ to ং, remove ZWNJ, ZWJ and spaces. More conservative means a
  language term's SE grows by 10% or more or a CI that excluded 0 no longer does.
- **Seeds (EXP-R).** Report each seed and the mean; the bootstrap runs on the seed-averaged
  per-item scores.

### 9.3 Derived quantities

- **Strict accuracy, lenient accuracy, fidelity, refusal rate**: as defined in §8 (refusal: D106
  with the 57-phrase list of scorer 0.2.2-stage2; 21 phrases were added after reading pilot
  answers).
  All rates are computed over scoreable items that are not on the excluded list.
- **Identification** = (Tier 1 + Tier 2) / N. **Specificity** = Tier 1 / (Tier 1 + Tier 2).
  **Generic fallback** = Tier 2 / (Tier 1 + Tier 2). **Sibling confusion** = siblings / N (cluster
  items).
  Tier 1 and Tier 2 count only in the requested language (label tier1 or tier2 with
  `content_lang_ok`); P5 in any language is reported as information (D233).
- **Transfer rate:** among `en_unique` naming items answered at Tier 1 in English (N-BE, or L-BE
  for EXP-L items), the share also at Tier 1 in Bangla (N-BB or L-BB), strict.
- **Error budget** (per model, `en_unique` P-CLUST items; denominator = items not at Tier 1 in
  N-BB):
  - **A. Language-bound:** Tier 1 in N-BE. Split by the N-BB answer:
    A1 wrong language (English answer naming the object), A2 English copy, A3 generic Bangla name
    (Tier 2), A4 wrong Bangla name (sibling or other).
    A is strict Tier 1 in N-BE, split by the N-BB label: A1 tier1 not in Bangla, A2 english_copy, A3
    tier2, A4 the rest (count of empty or refusal answers inside A4 reported) (D234).
  - **B. Lexical-access-bound:** not A, and correct in N-FCB.
  - **C. Not known:** neither A nor B. HT-5 estimates how much of C is gold or visibility problems.
    The F3 mark is the share of audited C answers judged gold_wrong or not_visible, pooled over
    models (D234).
  - Report N-FCB0 accuracy on the items that are not A beside B, as the guessing level, with its
    invalid and refusal shares (D234).
  - Also report the two-way split (A versus not A) on all `en_unique` P-NAME items.
  - Report how many sibling labels fall on the item and sibling pairs that HT-6 judged also right
    for the picture.
  - The command `budget-items` writes the HT-5 sampling frame (model, qa_id, house, category; ids
    only) from the scored files alone (D234).

### 9.4 Secondary analyses (descriptive, CIs reported, no correction)

1. Accuracy per QA type and cell, against the B0 baseline.
2. Refusal rates per cell, especially without the image.
3. The error budget figure (§9.3).
4. Forced choice with the image against without it, and against chance (25%).
5. The word-form result split by `origin3` (English loan, other loan, inherited).
6. Cultural stratum: culturally specific against globally common (about 35 items; exploratory,
   descriptive only).
7. Human naming reference against the models: tier composition and generic fallback (Bangla).
8. Token cost: per item, the Bangla-to-English token ratio of the question for each model's
   tokenizer, used as a covariate in the EXP-L GEE. Gemini counts come from its token-counting
   endpoint (verify that it is free).
9. Visual covariates: box area fraction (for close-up items whose image holds one box) and capture
   type.
10. Per-annotator results, using `house_assignment.csv`.
11. Template sensitivity (§9.2).
12. Lexicon sensitivity: re-score with the variants that HT-5 marks as acceptable but missing. The
    frozen lexicon stays primary.
    It also re-scores the gaps found by ASSET-9 stage 2 (addendum B7): বেত, ছোবড়া and the bare পাট
    in material answers, the bare সিরামিক, রঙ as a form of the class liquid, a second colour named
    on a patterned object, and ব্যাগ for কম্বল রাখার ব্যাগ.
13. Metric check (supplement): chrF++ and BERTScore F1 by the method of Zhang et al., computed with
    transformers (layer 9 of bert-base-multilingual-cased, no idf, no rescaling) against the tiers
    on N-BB answers, to show
    why string metrics are not used. This replicates a known finding; it is not a contribution.

### 9.5 Agreement statistics (computed from work already planned, no extra people)

- **Scorer check (HT-4):** Cohen's κ between the scorer and each checker, between the two checkers,
  and the scorer's accuracy against the consensus label, with 95% CIs.
- **Naming reference (HT-3):** percent agreement and Cohen's κ on the six naming labels (primary)
  and on strict (secondary), with house-clustered bootstrap 95% CIs.
- **Translation (HT-2):** percent faithful with a Wilson 95% CI.
- **Cultural tags (4A, already computed):** Fleiss κ 0.4705 (0.34 without the 7 instruction
  examples) and Gwet AC1 0.9289; reported in the supplement with the prevalence explanation.

### 9.6 Minimum detectable differences (planning values; recompute from the pilot)

- Single-cell accuracy on 1,118 items: 95% CI half-width about 4 to 6 points after clustering.
- P7 transfer-rate difference: detectable with 80% power at roughly 13 to 17 points. The GEE
  interaction on all `en_unique` naming items is more sensitive.
- Report the recomputed values in Methods.

### 9.7 Pre-specification (Gate E)

Before any full run, commit to GitHub: this file, `prompts.yaml`, the scorer and its tests, the
analysis notebooks (run on placeholder data) (src/stats.py, scripts/analysis.py,
notebooks/analysis.ipynb), and the probe ID lists (qa_ids only, no question
text). Tag the commit `gate-E` and record the tag in PROJECT_STATE. This replaces OSF.

---
## 10. Human checks

No kappa study is run for its own sake. Every agreement number in §9.5 comes from work in this
table. People counts are minimums; the team lead assigns people.

| ID | Task | People | Effort | Must-follow rules | Output |
|---|---|---|---|---|---|
| HT-1 | Lexicon v2: English names, accepted variants, generic names, room scope, origin look-up, English route, closed-answer vocabulary (ASSET-4B, ASSET-10) | 2 | about 5 pd in total | the checker of a row is never its writer; no model output consulted | frozen `lexicon.csv`, vocabulary file (Gate A) |
| HT-2 | English questions (ASSET-2): post-edit, then blind back-translation | 4: 2 post-editors, 2 back-translators | about 16 h in total | back-translators never post-edited; leakage scan after Gate A | English question file (Gate B); stage 4 fixes agreed by the two post-editors |
| HT-3 | Human naming reference (ASSET-12) | 2 | about 1.5 h each | done before either person sees model or scorer output; no items from houses they annotated; model output means the text of any model answer to a dataset question and anything computed from it; translated question files are not model output; one namer may come from the HT-1 pair; that namer is reported separately and the other is the main human reference | 2 × 200 names; tier agreement and κ |
| HT-4 | Scorer check: label 200 scored answers blind to the scorer's label, using the written tier and vocabulary rules; settle disagreements to one consensus label | 2 | about 1.5 h each, plus 0.5 h consensus | stratified by model, answer language and scorer label, at least 10 per label; the two checkers do not open Drive `scored/` until both sheets are in (D168); the written rules include the refusal list; the count answers scored correct that hold a refusal phrase are shown to the checkers | scorer accuracy and κ (Gate F) |
| HT-5 | Error audit: 200 answers (140 from error-budget category C, 30 English copies, 30 sibling confusions), each marked as gold wrong, acceptable but missing from the lexicon, object not clearly visible, question ambiguous, or model wrong | 1 or 2 | about 3 h | after Gate F | gold-error rate with CI; missing-variant list for §9.4 item 12 |
| HT-6 | Forced-choice option review (ASSET-11) | 1 | about 1 h | before any forced-choice run; not an HT-3 namer; Drive `human/HT-6/` closed to the namers until Gate H | logged swaps |

**HT-4 pass:** the scorer matches the consensus label on at least 95% overall, and at least 90%
within every label. If not, fix the scorer logic (never the frozen lexicon), re-run the tests, and
check a fresh 100.

**HT-2 pass:** at least 95% faithful and zero naming-question leaks (§4.6).

---

## 11. Pilot and gates

**Pilot:** the PILOT set (§3.4) under every EXP-L and EXP-N condition, for all three models. Its
purposes are engineering checks, the floor check, timing and token measurement. Its outputs are
never used to build or change the lexicon.
Items per condition come from the PILOT.json fields `pilot_part` and `in_pclust`. One Gemini
condition pair (L-BB, N-BB) is repeated once to measure run-to-run change.
The forced-choice conditions run only on the final option files, after a free dry run whose prompts
are checked against the option file (D164).

| Gate | Condition | Blocks |
|---|---|---|
| G1 (v1) | passed 10 Sep 2026: seven clusters qualify | nothing |
| Gate A | `lexicon.csv` and ASSET-10 frozen and committed (tag `gate-A`); also freezes `probes/excluded_items.csv`, which may only grow until Gate E. Passed 2026-10-10 | scorer tests on the real lexicon, ASSET-11 (options and P-FCE), ASSET-12 (NAME-REF), leakage scan, English gold answers |
| Gate B | English questions pass (§4.6). Passed 2026-10-12 | every English-question condition (L-EB, L-EE, L-EE0, N-FCE, N-FCE0, R-EE) |
| Gate C | the pilot re-runs from cache at zero cost (shown by running every pilot command twice; the second run makes 0 calls and changes neither the cache nor the ledger); at least 98% of answers non-empty and parseable (per model, D163 and D190: an open answer counts when it is not empty; a forced-choice answer when it is not empty and holds one shown label, names one option, or is a refusal). Passed 2026-10-17 | full runs |
| Gate D | floor check (§5.4); at least 90% of pilot answers are 8 words or fewer (per model, over all its pilot answers, D163) (the floor applies to the open models, §5.4). Passed 2026-10-17 | full runs of that model |
| Gate E | prompts, scorer, analysis notebooks and probe ID lists committed to GitHub (tag `gate-E`) | full runs |
| Gate F | 60 or more unit tests green, and HT-4 passed; scorer frozen (tag `gate-F`) | reporting any number |
| Gate G | fine-tuning dry run passed (§7.3). Passed 2026-10-09 | EXP-R training, which also waits for Gates D and E |
| Gate H | HT-3 finished before the namers see any model or scorer output. Passed 2026-10-13 | HT-4 and HT-5 for those people |

---

## 12. Compute and cost

### 12.1 API (Gemini 3.6 Flash)

Planning prices: $1.50 per million input tokens and $7.50 per million output tokens (list price,
third-party listing at planning time). Batch mode assumed at half price. **Verify both on Google's
pricing page in the week of the run.**
Read on 2026-10-04: standard **0.75 and 3.75 USD**, Batch 0.375 and 1.875 USD, through 31 December
2026; 1.50 and 7.50 (Batch 0.75 and 3.75) from 1 January 2027. The planning costs and the 40 USD cap
are unchanged.

Planning token sizes: an image call about 1,000 input and up to 100 output tokens (answer plus
minimal thinking); a text call about 250 input and up to 100 output. The pilot measures the real
values and the cost meter tracks them.

| Item | Calls | Mode | Planning cost |
|---|---|---|---|
| EXP-L and EXP-N, with image | 8,264 | batch | about $9.30 |
| EXP-L and EXP-N, text only | 3,028 | batch | about $1.70 |
| Pilot | about 560 | standard | about $1.30 |
| D5 repeat | 100 | standard | 0.09 USD |
| Translation first pass | 34 calls with `mt-v2`, plus 2 calls with the retired `mt-v1` | standard | 0.098 USD in total |
| Harness smoke tests | 120 Gemini calls | 20 standard, 100 in one Batch job | 0.05 USD |
| **Expected total** | | | **about $13** |
| With one full re-run of any experiment | | | about $19 |
| **Hard cap (the harness stops)** | | | **$40** |

A Batch job is entered in the ledger as a worst-case reservation when it is submitted and replaced
by its real cost when it is collected.

The rest of the $100 to $120 budget is held as the reviewer reserve (§16.2).

### 12.2 GPU (Colab Pro, L4)

| Item | L4 hours (planning) |
|---|---|
| Pilot, both open models | about 1 |
| EXP-L, both open models | about 4 |
| EXP-N, both open models | about 3 |
| EXP-R training, 3 seeds | about 18 |
| EXP-R evaluation | about 5.5 |
| Smoke tests and dry run | about 2 |
| **Total** | **about 33.5** |

At about 1.71 compute units per L4 hour (measured by a third party in March 2026), this is about
57 units, a little over half of one Colab Pro month (100 units). Kaggle (one account) is the
overflow, under
the rules in §5.3.

---

## 13. Figures and tables

Frontiers allows up to 15 figures and tables in total. Plan: 6 figures and 5 tables.

| # | Content |
|---|---|
| F1 | Study design: data, probes, conditions, scoring (one schematic) |
| F2 | EXP-L: strict accuracy in the four language cells per model, with 95% CIs; no-image bars beside L-BB and L-EE; B0 baseline line |
| F3 | Error budget per model: stacked shares A1 to A4, B, C, with the guessing level from N-FCB0 and the audit's gold-error estimate marked |
| F4 | Word form: transfer rate by `en_route`, per model, with CIs; inset by `origin3` |
| F5 | Tier composition in N-BB, N-BE and N-BBs per model, with the human naming reference as a reference row |
| F6 | EXP-R: ΔBN, ΔEN, Δno-image and Δfidelity, mean of 3 seeds with the 95% CI on seed-averaged scores and the seed range |
| T1 | Dataset and probe sets (sizes, houses, types) |
| T2 | Models and fixed settings (IDs, run dates, thinking setting, access) |
| T3 | Positioning against the closest prior work (feature by feature) |
| T4 | Contrasts P1 to P8 and P12: estimate, 95% CI, Holm-adjusted p where tested |
| T5 | Instrument checks: translation faithfulness, scorer accuracy and κ, naming-reference agreement, audit categories |

Supplement: per-type results; forced-choice details; origin breakdown; cultural stratum; token
cost; visual covariates; per-annotator results; template sensitivity; lexicon sensitivity; metric
check; all prompts verbatim; a lexicon excerpt; the 4A agreement values.

---

## 14. Release and statements

### 14.1 What is released, and when

| When | What |
|---|---|
| At submission, public | GitHub repository: code, `prompts.yaml`, scorer and tests, analysis notebooks, configs |
| At submission, Supplementary Material | `lexicon.csv` v2 (our instrument) and all prompts |
| On request, until the dataset release (supervisor decision, D82) | images, probe item lists, English questions, model outputs |
| With the Mendeley dataset release | everything above, plus a Zenodo DOI for the repository |
| Optional, at acceptance | the EXP-R LoRA adapters on Hugging Face |

If the supervisor prefers, the lexicon also moves to "on request".

### 14.2 Draft statements (edit to fit; Frontiers requires each section)

**Data availability.** "The images and annotations analysed in this study belong to the
BanglaIndoorVQA dataset, described in a companion data article in preparation. They are available
from the corresponding author on reasonable request until their public release. The code, prompts
and naming lexicon are available at [GitHub URL] and in the Supplementary Material."

**Ethics.** "This study analyses an existing dataset of household photographs collected with the
written consent of the household owners (companion data article). No new data were collected from
human participants. Translation, naming and checking tasks were carried out by members of the
research team."

**Generative AI.** State every use, with names and versions: Gemini 3.6 Flash for the first-pass
machine translation of the English questions (all post-edited by native speakers), and any AI
assistance used in planning or writing. The authors take full responsibility for the content.

**Model reporting.** For every model: the exact ID, the access route, the run dates and every
setting in §5.2.

---
## 15. Risk register

| # | Risk | Likelihood | Mitigation already in the design | Fallback if it happens |
|---|---|---|---|---|
| R1 | H4 shows no word-form effect | medium | pre-declared; power planned (§9.6) | report a clean null with its CI. It still answers the title (the penalty is not about word form); C2 to C4 stand |
| R2 | An open model is near floor | low to medium | Gate D on the pilot | replace with the next size in its family (§5.4) |
| R3 | Gemini is near ceiling in Bangla | low | the decomposition still reads per model | report it as a finding: the frontier model solves most of it; the open models carry the decomposition |
| R4 | The English-copy detector is unreliable | medium | checked inside HT-4 | if fewer than 90% of English-copy labels are right, drop the label; H4 still runs on tier scores |
| R5 | Gemini 3.6 Flash changes or retires | medium | one 7-day window; IDs and dates recorded | the replacement rule in §5.5 (D162); re-run all Gemini conditions (about $13) |
| R6 | Fine-tuning fails | low to medium | dry run (Gate G); LoRA in bf16, no quantization | report the seeds that ran; if none, cut EXP-R (D72) |
| R7 | Platform differences change answers | low | one platform per model (§5.3) | re-run that model's conditions on one platform |
| R8 | The English questions fail Gate B | low | 100% post-edit, then back-translation | fix and repeat stage 3 on a fresh 150 |
| R9 | The lexicon takes longer than planned | medium | it is on the critical path; start first (§17) | English names for P-CORE-CLOSED naming items first, the rest after |
| R10 | Too few P-FCE items (fewer than 100) | low | English forced choice is secondary | report English forced choice as descriptive only |
| R11 | A new paper overlaps before submission | medium | wording already narrowed (§2, §0.5) | re-run the novelty sweep just before submission; adjust wording only |
| R12 | Text overlap with the DiB paper | low | the dataset section is written fresh; cite DiB | rewrite any flagged passage |
| R13 | Reviewers ask for more models or prompt checks | medium | §16.2 reserve, about $15 and a few GPU hours | run the reserve item during revision |
| R14 | Reviewers ask for data access | medium | supervisor decision (D82) | share on request, as decided |

---

## 16. What was cut, and the reserve list

### 16.1 Cut from v1.0 (reasons in the guide)

| Cut | One-line reason |
|---|---|
| E5 judge validation and P-JUDGE | no free-form types remain, so no judge is needed |
| E7 hazard diagnostic, P-HAZ, ASSET-6, Table 8 | supported no contribution |
| E8 grounding | cut on 29 Sep 2026 (D61) |
| E10 human ceiling and ASSET-7 | replaced by the error budget, the audit and the naming reference |
| ASSET-3 IAA study | replaced by checks that measure the instruments we actually use |
| OSF pre-registration | replaced by a GitHub commit before full runs |
| 10-model roster | 3 models answer the question; reserve if asked |
| FT-B (projector unfrozen) | one setting with 3 seeds is more useful than two settings with one seed |
| Out-of-domain benchmark check after fine-tuning | the English and no-image readouts already show what changed |
| Free-form question types | cannot be scored without a judge; not needed for the claim |
| BLEU-once demonstration, McNemar | not needed; the metric check and bootstrap cover them |
| Deliberate weak-Bangla model | the token-cost question is now answered item by item |
| One-shot prompt variant | the specificity prompt is the useful variant |
| Cross-model fertility regression | meaningless with 3 models; replaced by an item-level covariate |
| Capture-type and object-size table | kept as covariates instead |
| Decision table with a human ceiling | replaced by the error budget (§1.4, §9.3) |

### 16.2 Reviewer reserve (run only if a reviewer asks)

| Item | Cost |
|---|---|
| Qwen3.5-9B on EXP-L and EXP-N | about 6 L4 hours |
| One OpenAI model on EXP-L and EXP-N | about $10 to $20 |
| Prompt paraphrase check: 2 paraphrases on a 200-item subset | under $2 and about 1 L4 hour |
| A larger error audit | about 3 person-hours per 200 items |
| Image-resolution check (for the Computer Vision backup venue) | about 2 L4 hours |

---

## 17. Work order

No dates. Three tracks run in parallel. Arrows mean "must finish before".

**Track 1: data and lexicon (critical path)**
gold fixes (done) → ASSET-5B draw and ASSET-5C → ASSET-4 v2 script, then HT-1 lexicon and ASSET-10 →
**Gate A** → ASSET-11 (options and P-FCE) and HT-6 review; ASSET-12 (NAME-REF draw) → HT-3 naming
reference (**Gate H**).

**Track 2: English questions**
gold fixes → ASSET-2 stage 1 (PILOT items first) → stage 2 post-edit → stage 3 back-translation →
stage 4 leakage scan (needs Gate A) → **Gate B**.

**Track 3: code**
ASSET-8 harness → ASSET-9 scorer (written now and tested on a small made-up lexicon; re-tested on
the real lexicon at Gate A) → fine-tuning scripts → dry run (**Gate G**).

**Then, in order:**
1. Pilot (needs the harness, the prompts and the PILOT English questions) → **Gate C**, **Gate D**.
2. Fix prompts if the pilot shows a problem → commit everything → **Gate E**.
3. Full runs: EXP-L and EXP-N (Gemini in batch; open models on Colab), and EXP-R training, in
   parallel.
4. Scoring → HT-4 → **Gate F** → HT-5 audit.
5. Statistics, figures and tables.
6. Writing the result-dependent sections. The result-independent sections can be drafted from now
   (see the guide).

**First steps, in this order:** the GitHub repo and Drive folders; the ASSET-5B draw and the ASSET-5C
script; the ASSET-4 v2 script, then HT-1; ASSET-2 stage 1 on the PILOT strings; ASSET-8 and ASSET-9.
The gold fixes are done (confirm the 5A re-run check in PROJECT_STATE).

**Effort summary (planning):** human checks about 9 pd; code about 9.5 pd (harness 4, scorer 3,
fine-tuning scripts 2, ASSET-5C 0.5); statistics about 3 pd; GPU about 33.5 L4 hours; API about $13.

---

## 18. Change log

| Date | Version | Change |
|---|---|---|
| 18 Aug 2026 | 1.0 | Original locked design (Candidate 2) |
| 29 Sep 2026 | 1.0 | E8 cut, E6 kept (D61) |
| 1 Oct 2026 | 2.0 | Full redesign for the Frontiers Q2 target: closed-answer types only; 3 models; no-image arm and fidelity; English-route word-form test as the lead result; English-collision rule; 4-option forced choice with and without image; specificity prompt on all of P-NAME; LoRA with 3 seeds; human checks replace the IAA study, judge study, hazard gold and human ceiling; GitHub commit replaces OSF; Colab Pro primary, one Kaggle account secondary. Decisions D62 to D88 in the guide |
| 2 Oct 2026 | 2.1 | Freezes are now recorded only as Git tags `gate-A`, `gate-E` and `gate-F`, with no separate file-fingerprint tracking; seeded ranking stays as the sampling method. ASSET-5 reorganised: 5B runs as is; 5C draws P-CORE-CLOSED, R-TEST and PILOT; P-FCE moves to ASSET-11 and NAME-REF to ASSET-12; one ASSET-4 v2 script replaces the v1 4B script. Generic-name precedence in scoring (§8.3) and the lexicon (§4.2). Forced-choice distractor rule (§4.7). H1 tested as P12 = P2 − P1; Holm over six tests (§9.1). Gold fixes done. ASSET-5C outputs keep the P-CORE.json record format (§3.4). Decisions D89 to D93 in the guide, D95 in PROJECT_STATE |
| 17 Oct 2026 | 2.1 | Scorer 0.2.2-stage2 (D169): forced-choice label rules; all pilot answers re-scored. |
| 18 Oct 2026 | 2.1 | DESIGN_ADDENDUM.md sections A to J merged once, before the Gate E commit (D101, D241); the decision rows D96 to D239 are in the guide, Part 8. v1 decisions D47 to D56 signed off (D240). Six out-of-date places corrected (D243). The addendum stays the record of how each asset ran; after the tag the design is not edited (D244). Pre-specification commit with the tag `gate-E` (D242) |
