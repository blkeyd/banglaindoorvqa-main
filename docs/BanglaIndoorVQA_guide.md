# BanglaIndoorVQA Main Paper: Handbook

**Version 2.1 · 2 October 2026**
**Companion to:** `BanglaIndoorVQA_Design.md` v2.1 (the locked plan).
**Supersedes:** `BanglaIndoorVQA_MainPaper_Guide.md` v1.0 for all main-paper work.

> **What this file is.** A handbook that helps you understand the paper: why each experiment
> exists, why each design choice was made, what is new, and how to defend it. It does **not**
> repeat the plan. For any condition, item count, setting, cost or gate, open the design file.
> References like "Design §7.1" point there.

---

# PART 1: THE PAPER IN PLAIN WORDS

*Read this part first. About 1,200 words.*

## 1.1 What we are doing

We built BanglaIndoorVQA: photos of 68 real Bangladeshi homes with Bangla questions and answers
written by native speakers. A separate data paper (the DiB paper, still in progress) describes the
dataset. **This paper is not about the dataset.** It uses the dataset to answer one question about
vision-language models (VLMs):

> When a VLM gets a Bangla question about a Bangladeshi home wrong, is the problem the **language**
> (it cannot read or write Bangla well) or the **knowledge** (it does not know the object or its
> everyday name)?

The DiB paper's zero-shot baselines show the problem is real: strong open models answer only 11%
to 30% of the short-answer questions correctly. But a low score alone cannot tell you **why**. This
paper takes the "why" apart.

*Zero-shot means testing a model as it is, with no training on our data.*

## 1.2 Why the question matters

If the problem is language, the fix is better Bangla in models (more Bangla text, better
tokenizers). If it is knowledge, the fix is more images of local objects. If it is a mix, we want
to know the shares. Many recent papers report that models do worse in low-resource languages; very
few separate the causes with controlled tests on the same items. That separation is our job.

## 1.3 The story in five steps

1. **Ask the same question four ways.** Bangla or English question, Bangla or English answer.
   Comparing the four tells us how much it costs to *read* Bangla and how much to *write* it.
2. **Take the photo away.** If a model still answers correctly with no image, it is using
   language habits, not seeing. This also shows whether the image helps less in Bangla.
3. **Look inside the naming errors.** Did the model give the specific everyday name (বালতি), a
   generic one (পাত্র, "container"), or a look-alike sibling? Can it pick the right Bangla name
   when we show it four options?
4. **Ask which words are hard.** Some Bangla household names are simply English words in Bangla
   letters (ফ্রিজ, সোফা). Others are not (বালতি, বালিশ). If models fail mainly on the second kind,
   the Bangla problem is about **vocabulary**. This is our clearest new result (C1).
5. **Try a small repair.** Train one small model on our Bangla data and see exactly what improves:
   Bangla answers only, English too, or just answer habits that work even without the image.

## 1.4 What a good result looks like

Every outcome is publishable, because the paper is written as a measurement, not a bet.

| If we find | The paper says |
|---|---|
| Most errors are fixed by answering in English | VLMs know Bangladeshi household objects but cannot name them in Bangla: a language gap, and (if C1 holds) mainly a vocabulary gap |
| Most errors survive English and forced choice | VLMs do not know these objects: a knowledge gap |
| A mix, differing by model | The frontier model and the small open models fail for different reasons, and we show the shares |

The title ends with a question mark so it fits all three.

## 1.5 Why this can pass a Frontiers Q2 review

Frontiers reviewers check **validity**: a clear question, correct methods, honest statistics,
available code and data, and good grounding in the literature. They are told not to reject papers
for low perceived impact. That favours our design:

- **Controlled comparisons on the same items**, so differences are not caused by different data.
- **Deterministic scoring with a frozen lexicon**, so no judge model can be blamed for errors.
- **Statistics above the usual standard.** The novelty report read four Frontiers AI papers in
  full: none linked a code repository, only two ran any significance test, and only one reported
  confidence intervals. We report all three.
- **Narrow, honest claims.** We claim only what the data can show (Design §0.5).

The main risk is not novelty. It is **execution**: a sloppy lexicon, a leaky translation or a
buggy scorer would be an objective methods error, which is the main reason Frontiers papers are
rejected. That is why so much of the design protects the instruments.

## 1.6 What not to confuse

- **The DiB paper** describes how the dataset was built and gives baseline scores. This paper cites
  it and does not repeat it. Write this paper's dataset section fresh (no copied text), because
  Frontiers runs overlap checks.
- **Version 1.0 of the design** (18 August) is retired. Its experiment labels E1 to E10, CTRL-1 to
  CTRL-10 and gates G2 to G8 no longer exist. The new labels are EXP-L, EXP-N, EXP-R, Gate A to
  Gate H (Design §0.4).

---

# PART 2: WORDS YOU NEED

*One line each. Terms are grouped by where you meet them.*

**Models and setup**

| Term | Meaning |
|---|---|
| VLM | A model that takes an image and text and writes text. |
| Zero-shot | Testing a model as it is, without training it on our data. |
| Thinking level | How much hidden reasoning a model does before answering. We keep it minimal or off. |
| Batch API | Sending many requests at once for about half the price; answers come back later. |
| bf16, fp16 | Number formats used in GPU maths. The L4 supports bf16; the T4 only fp16. |
| Tokenizer, token | The tokenizer cuts text into pieces (tokens). Bangla often needs more tokens per word than English. |

**Items and conditions**

| Term | Meaning |
|---|---|
| Closed-answer question | A question with a short, checkable answer: a name, a number, a room, a material or a colour. |
| Free-form question | A question whose answer is a sentence. Not studied in this paper. |
| Probe set | A fixed list of test items used in an experiment. |
| Cell, condition | One way of asking, for example "Bangla question, English answer, with image". |
| Reading cost | How much accuracy drops when the question is in Bangla instead of English. |
| Writing cost | How much accuracy drops when the answer must be in Bangla instead of English. |
| No-image arm | The same question asked without the photo, to see what can be guessed from text alone. |
| Fidelity | Whether the model answers in the language we asked for. |
| Image-induced fidelity loss | Adding a photo makes some models reply in English even when asked in another language (Hinck et al., 2024). |
| B0 baseline | The most common answer to the same question in other houses. It shows what pure guessing gets. |

**Naming**

| Term | Meaning |
|---|---|
| Accept set | The answers we count as correct for an item. |
| Tier 1 | The specific everyday name (বালতি, "bucket"). |
| Tier 2 | An acceptable but generic name (পাত্র, "container"). Counted separately, not as correct. |
| Tier 3 | Wrong, including a confusable sibling. |
| Generic fallback | Answering with a broad category instead of the specific name. Older documents call it "superordinate retreat". |
| Cluster, sibling | A group of objects people confuse (হাঁড়ি, পাতিল, দেকচি). Each member is a sibling of the others. |
| Forced choice | Showing the model four names and asking it to pick one. |
| Lexical access | Being able to find the right word, either by producing it or by recognising it when shown. |
| English route | The object has an accepted Bangla name that is simply an English name of it in Bangla letters (ফ্রিজ). |
| English copy | A model writes an English word in Bangla letters that Bangla speakers do not use for this object (বাকেট for বালতি). |
| English-unique (`en_unique`) | The object's English name is not shared with any other object in the lexicon. হাঁড়ি is not English-unique: its English name "pot" is shared with পাতিল and দেকচি. |
| Word origin | Where a word came from: an English loan, another loan (Persian, Arabic, Portuguese and others), or an inherited word (তৎসম, তদ্ভব, দেশি). |
| Transfer rate | Of the objects a model names correctly in English, the share it also names correctly in Bangla. |
| Error budget | Sorting each Bangla naming error by what fixes it: answering in English, being shown options, or nothing. |
| Strict score | Correct and in the requested language. The headline score. |
| Lenient score | Correct object or value, in any language. It measures knowledge regardless of language. |

**Training**

| Term | Meaning |
|---|---|
| LoRA | Training a small add-on inside a model instead of changing all its weights. Cheap and fast. |
| QLoRA | LoRA on a model compressed to 4 bits. Not needed here: our 4B model fits on an L4 without compression. |
| Seed | The random starting point of a training run. Three seeds show how much results move by chance. |
| Early stopping | Stopping training when the loss on held-out data stops improving. |
| Validation loss | The model's error on data it does not train on, used to choose when to stop. |

**Statistics**

| Term | Meaning |
|---|---|
| Confidence interval (CI) | The range that very probably contains the true value. We use 95%. |
| Bootstrap | Re-drawing the data many times to see how much a number could move. |
| House-clustered bootstrap | Re-drawing whole houses instead of single questions, because questions from one house are related. |
| p-value | How surprising a result would be if there were no real effect. Small means surprising. |
| Holm correction | Adjusting p-values when several tests are run, so we do not "find" effects by chance. |
| GEE | A regression for yes/no outcomes that allows answers from the same house to be related. |
| Interaction | When the effect of one thing depends on another, for example the Bangla answer cost being larger for non-English-route names. |
| Cohen's κ (kappa) | Agreement between two raters after removing the agreement expected by chance. |
| Minimum detectable difference | The smallest real effect the study can reliably find. |
| Pre-specification | Fixing the analysis plan before seeing results. We do it with a dated GitHub commit. |

**Translation**

| Term | Meaning |
|---|---|
| Post-editing | A person correcting a machine translation. |
| Back-translation | Translating the English back into Bangla, without seeing the original, to check the meaning survived. |
| Leakage | An English question that accidentally contains its own answer. |

---
# PART 3: THE EXPERIMENTS EXPLAINED

*For each experiment: what it asks, how it answers it, how to read the results, and what it cannot
show. Settings and counts are in the design file.*

## 3.1 EXP-L: Language and image (Design §7.1)

**What it asks.** On the same closed-answer items, how much does Bangla cost when reading the
question, how much when writing the answer, and how much does the image help in each language?

**How it works.** Every item is asked four ways (Bangla or English question × Bangla or English
answer), plus twice without the photo (Bangla only and English only). Because the items never
change, any difference between cells comes from language or from the image, not from easier or
harder questions.

**Why both question and answer language.** Many studies only translate the whole item. That mixes
two different skills. A model may *understand* a Bangla question perfectly and still be unable to
*write* the Bangla answer. Crossing the two separates them.

**Why the no-image arm.** About three in four `object` questions are riddles that describe the
object ("What is the big furniture for sleeping called?"). Some can be answered without looking.
Also, 74% of the count answers are "one" in P-CORE (189 of 255). In P-CORE-CLOSED it is 189 of 247,
about 77%. Without the no-image arm and the B0 baseline, a reviewer
could say our accuracy is mostly guessing. With them, we can say exactly how much is seeing.

**Fidelity.** Hinck et al. (2024) showed that adding an image can push models to reply in English.
We record the reply language in every Bangla-answer cell, with and without the photo, so we can
test this on native Bangla at no extra cost. Fidelity also splits the writing cost into parts:
replying in English, copying an English word into Bangla letters, or giving a wrong Bangla word.

**How to read the results.**
- Writing cost larger than reading cost supports H1: producing Bangla is the harder part.
- If the image helps less with Bangla questions (P3 > 0), weak reading stops the model from using
  what it sees. A 2026 audit of Indian languages saw the same with translated science questions;
  we test it on natural household scenes.
- High no-image accuracy on a type means that type is partly answerable from text. Read its
  with-image score with that in mind.

**What it cannot show.** Whether a wrong answer means the model does not know the object or only
lacks the Bangla word for it. That needs EXP-N.

## 3.2 EXP-N: Naming (Design §7.2)

**What it asks.** When a model names a household object, does it give the specific everyday name,
a generic one, or a confusable sibling? If it fails in Bangla, is it because it does not know the
object, cannot find the Bangla word, or retreats to a generic word?

**Three tools, each answering one sub-question:**

1. **Open naming in Bangla and in English** (same Bangla question, different answer language).
   If the model names the object correctly in English but not in Bangla, it knows the object; the
   failure is in Bangla.
2. **Forced choice with four options.** If the model cannot produce the Bangla name but picks it
   from four options, the word is not missing; the model cannot retrieve it on its own. We also run
   forced choice **without the image**, to measure how much the options alone give away.
3. **The specificity prompt.** We add one sentence asking for the specific everyday name, not the
   category name. If generic answers turn into specific ones, the model knew the word and was just
   defaulting to the safe generic answer. If not, it did not know the word.

**The English-collision rule (why `en_unique` matters).** Some Bangla objects share one English
name. হাঁড়ি, পাতিল and দেকচি are all "pot" in English. If a model answers "pot", English scoring
must accept it for all three, while Bangla scoring demands the exact one. English would look
better for a reason that has nothing to do with knowledge. So every Bangla-versus-English naming
comparison uses only objects with a unique English name. The others are studied in Bangla only.
This closes a flaw a sharp reviewer would catch.

**The word-form test (C1).** For objects the model names correctly in English, we check how often
it also names them correctly in Bangla (the transfer rate), separately for:

- objects with an **English route**: an accepted Bangla name that is an English name in Bangla
  letters (ফ্রিজ, সোফা, টেলিভিশন);
- objects without one (বালতি, বালিশ, পর্দা).

If transfer is much higher with an English route, the model "gets to" Bangla through English and
fails where that road does not exist.

The word-origin coding (from the Bangla Academy dictionary) adds a second check. Many everyday
Bangla words are old loans from Persian or Portuguese (পর্দা, আলমারি), not native words. If those
behave like native words, the effect is about **English** specifically, not about borrowing in
general. The novelty report insisted on this, and it is right.

**The English copy (bonus signal).** When a model writes বাকেট for বালতি, it shows the mechanism
directly: it knows "bucket" and copies it into Bangla letters, but Bangla speakers do not say that.
If the detector proves unreliable in the scorer check, we drop it (Design R4).

**Example walk-through (ILLUSTRATION ONLY, NOT DATA).**
- A close-up of a বালতি (English-unique, no English route). In Bangla the model says পাত্র
  (Tier 2); in English it says "bucket" (Tier 1). The error is **language-bound**, subtype
  "generic Bangla name".
- A close-up of a ওয়াল শোকেস. In Bangla the model says আলমারি (a sibling); in English it gives a
  wrong name; in forced choice it picks ওয়াল শোকেস. The error is **lexical-access-bound**.
- A close-up the model gets wrong everywhere. The error is **not known**. The error audit checks
  whether some of these are really gold mistakes or objects that are hard to see.

**How to read the results.**
- A large language-bound share means "knows the object, cannot say it in Bangla".
- A large lexical-access share means "can recognise the Bangla word, cannot produce it".
- A large not-known share (after the audit) means a knowledge gap.
- Transfer much higher with an English route (P7 > 0) means the Bangla gap is about vocabulary.

**What it cannot show.** Anything about culture as a whole. Naming objects is not cultural
competence, and a 2025 position paper warns against treating it that way. The cultural-specificity
stratum (about 35 items) is reported as exploratory only.

## 3.3 EXP-R: Repair (Design §7.3)

**What it asks.** If we train one small model (Qwen3.5-4B) on our Bangla training houses, what
exactly gets better?

**Why it is a diagnostic and not a method.** We are not trying to build the best Bangla model. We
use training as a probe. Its readout is what makes it informative:

- **Bangla accuracy up, English flat:** the model learned Bangla answer habits for this data.
- **English up too:** some object knowledge carried across languages.
- **Gain also without the image:** the model learned answer patterns (priors), not seeing.
- **Fidelity up:** the main fix was simply answering in Bangla more reliably.

**Why three seeds.** One training run can be lucky or unlucky. Three runs with different random
starts show how stable the effect is. The novelty report asked for this.

**Why the test houses only.** Training used the 43 training houses. Testing on the same houses
would reward memorising those homes. The 15 test houses were never seen in training.

**What it cannot show.** Whether a larger model or more data would fix more. It is one model, one
setting: a case study.

## 3.4 The human checks: why each exists (Design §10)

| Check | The doubt it removes |
|---|---|
| HT-1 lexicon (two people, writer and checker) | "Your correct answers are just your own word choices." The lexicon is built only from names the annotators used, before any model output is seen, and checked by a second person |
| HT-2 translation, then back-translation by different people | "Your English questions are easier or leak the answer." Faithfulness is measured, leaks are scanned |
| HT-3 human naming reference | "Humans also say the generic word sometimes." Two people name the same 200 objects, giving a human baseline for generic fallback and a check on the gold names |
| HT-4 scorer check | "Your automatic scorer makes mistakes." Two people label 200 answers blind; we report the scorer's accuracy and κ |
| HT-5 error audit | "Some 'errors' are really gold mistakes or unclear photos." The audit estimates that share |
| HT-6 option review | "A distractor was also correct." Every option set is checked once |

The κ values in this paper come from HT-3 and HT-4, work we do anyway. No extra agreement study is
needed, which respects the supervisor's wish not to spend time on IAA.

## 3.5 How the three experiments fit together

- **EXP-L** gives the big picture on all closed types: reading, writing and image use.
- **EXP-N** zooms into naming, where the language-or-knowledge question is sharpest, and adds the
  word-form test.
- **EXP-R** asks what in-domain data changes, and reads the answer through the same lenses.

The error budget (Design §9.3) is the bridge. It turns the EXP-N conditions into one picture per
model: how many Bangla naming errors are language, lexical access, or knowledge.

---

# PART 4: WHY EACH DESIGN CHOICE

*Choice, reason, and what we rejected. Where the novelty report or a decision row drove a choice,
it is named.*

## 4.1 Scope

| Choice | Reason | Rejected alternative |
|---|---|---|
| Closed-answer types only | They can be scored by rules, so no judge model is needed. Primary claims rest only on rule-based scoring | Keeping free-form types with an LLM judge: needs judge validation by several raters, API money, and still invites doubt in Bangla |
| Drop the "part of the room" scene questions | Their answers are vague (ঘরের একটি কোণা) and cannot be scored by an accept set. v1.0 wrongly counted them as rule-scorable | Keeping them: a methods error a reviewer would catch |
| Three models | Two open families with different tokenizers plus one frontier model answer the question. Claims are written per model | 10 models: more cost and work with little gain for a diagnostic paper. A 4th model sits in the reserve list |
| Gemini as the closed model | Strong on vision and Indic languages, cheap in batch, and has a minimal-thinking setting | GPT or Claude as the only closed model: fine too, but Gemini was the strongest Indic performer in recent benchmarks |
| Gemini 3.6 Flash, not 3.7 Flash or 3.1 Pro | 3.6 Flash supports "minimal" thinking. 3.7 Flash only goes down to "low"; 3.1 Pro cannot turn thinking off | Letting the closed model think while the open models do not would make the comparison unfair |
| Small open models (about 4B) | Cheap to run and train on one L4; realistic for local deployment; Qwen3.5-4B is among the strongest small multimodal models | 8B to 12B models as defaults: more GPU time. They are the swap-in if a small model floors (Design §5.4) |
| No chain-of-thought | We measure direct answering. A 2026 audit found chain-of-thought hurt Bengali by about 14 points | Allowing reasoning: confounds the language question with reasoning in English |

## 4.2 Measurement

| Choice | Reason | Rejected alternative |
|---|---|---|
| Strict and lenient scores | Strict asks "right, in the right language"; lenient asks "does it know the object". The gap between them is part of the finding | Only strict: hides whether the model knew the object |
| Fidelity as an outcome | Image-induced English replies are a known problem; measuring it costs nothing | Ignoring reply language: would mix "replied in English" with "wrong" |
| Tier 2 counted separately | A generic name is not wrong in the ordinary sense, but it is not the specific name. Counting it separately is the whole point of the naming probe | Treating Tier 2 as correct (hides the fallback) or as plain wrong (hides the pattern) |
| Siblings count as wrong | Saying পাতিল for হাঁড়ি is a real error of specificity, not a near miss | Partial credit for siblings: blurs exactly the distinction we study |
| English-collision rule | Prevents English answers looking better because English is less specific (Part 3.2) | Comparing all items: biased comparison |
| English route as the main word-form variable | It is what a model can actually use: does an English road to the Bangla name exist? The novelty report asked us to frame the test as "English transparency" | Etymology alone: an English-origin word with a different English meaning (কমোড) gives a model no easy road |
| Origin coded from the Bangla Academy dictionary | Look-up, not opinion, so no agreement statistics are needed; it is a standard reference | Coding origin by intuition: unreliable, and many "native-feeling" words are Persian or Portuguese loans |
| Accepted variants only from the annotators' own usage | Keeps the gold answers human and independent of models | Adding variants after seeing model answers: inflates scores and is circular |
| Lexicon frozen before any model run | Removes any suspicion that answers shaped the scoring | Editing later: missing variants are handled as a sensitivity analysis instead |
| Four forced-choice options (2 siblings, 1 same-room object) | A fixed number gives a fixed chance level (25%). Siblings test real discrimination; the extra object catches pure guessing | All cluster members as options: varying chance levels and harder interpretation |
| Forced choice without the image | Options can give answers away (shown by "Answer Matching" research). This measures how much | Assuming options are neutral |
| Specificity prompt on all naming items | The novelty report singled it out as the genuinely new element of the forced-choice part, so it gets full power | Running it on 600 items (v1.0) |
| B0 baseline from other houses | Shows how much a type is solvable by guessing the common answer, without leaking the item's own answer | No baseline: the 74% "one" in counts would mislead readers |

## 4.3 Statistics

| Choice | Reason | Rejected alternative |
|---|---|---|
| House-clustered bootstrap | Questions from one house share furniture, light and annotator. Treating them as independent would make CIs too narrow | Plain per-item CIs: overconfident |
| GEE as a confirmation | A standard regression that respects house clusters; it gives the formal interaction test for H4 | McNemar: ignores clustering |
| Holm within each model | Six hypothesis tests per model (P12 and P3 to P7); Holm keeps false alarms in check and is easy to explain | No correction: Frontiers policy asks for multiple-comparison correction |
| Template sensitivity | Many questions reuse the same wording; we check that this does not inflate confidence | Ignoring template reuse |
| κ only from planned work | HT-3 and HT-4 already involve two people on the same items, so κ is free | A separate IAA study: time the supervisor does not want to spend, measuring something we do not use |
| GitHub commit instead of OSF | A dated, public commit shows the plan came before the results, at zero cost | OSF: the supervisor agreed it is not needed |
| Minimum detectable difference in Methods | Shows reviewers the study could find an effect of a stated size, so a null is meaningful | Silence on power |

## 4.4 Practical

| Choice | Reason | Rejected alternative |
|---|---|---|
| One platform per model | T4 (fp16) and L4 (bf16) can give slightly different answers; mixing them inside one model's runs would add noise that looks like an effect | Splitting runs to save time |
| Gemma 4 never on a T4 | A known fp16 overflow problem on the T4 | Risking silent errors |
| One resized copy of every image | Every model sees exactly the same pixels; EXIF rotation applied first so no photo is sideways | Each model loading originals in its own way |
| Gemini paid tier only, one 7-day window | Free-tier data can be used for training, which would leak our test items; one window avoids a mid-run model change | Free tier to save a few dollars |
| LoRA in bf16, not QLoRA | The 4B model fits on an L4 without compression, so we avoid compression error | QLoRA (v1.0): unnecessary here |
| One setting, three seeds | Shows stability; the novelty report asked for 2 to 3 seeds | Two settings with one seed each (v1.0's FT-A and FT-B) |
| Early stopping on validation loss | Stops before overfitting, with no hand-tuning | A fixed number of epochs |

## 4.5 What we cut, and why it does not hurt the Q2 target

| Cut | Why it does not hurt |
|---|---|
| Judge study (E5) | No free-form types remain, so there is nothing to judge |
| Hazard diagnostic (E7, Table 8, ASSET-6) | It supported no contribution |
| Grounding (E8) | Cut on 29 Sep (D61); grounding needs boxes for every item and was risky |
| Human ceiling (E10) | The error budget, the audit and the naming reference answer the same question more directly |
| IAA study (ASSET-3) | Dataset quality belongs to the DiB paper; this paper checks the instruments it actually uses |
| OSF | Replaced by a dated GitHub commit |
| Seven extra models | The claims are per model; a reserve list covers reviewer requests |
| Cross-model fertility regression | With 3 models a regression across models means nothing; token cost is now an item-level covariate |

---
# PART 5: CONTRIBUTIONS AND NOVELTY

## 5.1 The four contributions, explained

### C1 Word form (lead)

- **What it is.** Evidence on where the Bangla naming penalty sits: whether models fail mainly on
  objects whose Bangla name is not an English word in Bangla letters, with word origin coded from
  the Bangla Academy dictionary.
- **Why it matters.** It turns "models are worse in Bangla" into a specific, testable cause, and
  it points to a fix: Bangla object vocabulary, not general Bangla.
- **What is new.** The novelty report found no 2024 to 2026 VLM paper that compares English-origin
  names with other names in object naming. Close work exists only in other settings: cognates and
  shared word forms in text-only models (European language pairs), script consistency in VLMs
  (Punjabi), and a September 2026 study where Bengali fish names failed in CLIP-style recognition
  while English and Latin names worked.
- **Safe wording.** "English transparency of the Bangla object name", not "loanwords versus native
  words". Many everyday Bangla words are old Persian or Portuguese loans.
- **Never say.** "Borrowed words are harder." The hypothesis is about English specifically.

### C2 The three-tier lexicon and scoring scheme

- **What it is.** A released Bangla-English lexicon of household objects with specific names,
  generic names, confusable clusters, an explicit "sibling = wrong" rule, word origin, English
  route and English collision flags.
- **Why it matters.** It lets anyone score open naming in Bangla deterministically and separate
  "wrong" from "too generic".
- **What is new.** Naming specificity in VLMs is studied in English (basic-level naming,
  ManyNames, taxonomy-aware scoring). What is new is the **resource and scheme** for Bangla
  household objects, with the sibling rule and the collision flag.
- **Never say.** "We are the first to study naming specificity in VLMs."

### C3 Controlled evidence on native Bangla VQA

- **What it is.** Results from crossing question language and answer language on the same
  natively written items, with a no-image arm and fidelity measured.
- **Why it matters.** It separates reading, writing and image use for Bangla on real household
  scenes.
- **What is new.** The **combination on natively written Bangla items**. Each piece exists
  elsewhere: input and output language combinations (VLURes, ACL 2026 Findings; cross-lingual
  artwork explanation, NAACL 2025 Findings), no-image ablations (several 2026 papers, including an
  Indian-language audit that covers Bengali), and image-induced fidelity loss (Hinck et al.,
  EMNLP 2024 Findings).
- **Never say.** "We introduce a new method for crossing input and output language."

### C4 Fine-tuning diagnostic case study

- **What it is.** One LoRA setting, three seeds, read through Bangla, English, no-image and
  fidelity.
- **What is new.** The readout: whether Bangla-only training teaches vision, language, or answer
  habits. LoRA itself and its forgetting behaviour are well known.
- **Never say.** "Our fine-tuned model is a new state of the art."

## 5.2 Positioning against the closest work (draft of Table T3)

**Verify every cell against the paper itself before Table T3 goes into the manuscript.** The cells
below come from abstracts, the novelty report and full-text reads where we did them.

| Work | Native Bangla questions | Native home photos | Question × answer language crossing | No-image arm | Fidelity | Tiered open naming | Word-form test | Fine-tuning readout |
|---|---|---|---|---|---|---|---|---|
| BanglaVerse (EMNLP 2026 Findings) | yes | no (curated web images) | no (whole items translated) | not reported | check (error analysis) | no (multiple choice and captions) | no | no |
| BanglaProtha (WACV 2026) | yes | no (cultural web images) | no | not reported | no | no (multiple choice) | no | fine-tuning studied, not as a language readout |
| IndicVisionBench (arXiv 2025) | Bengali among 10 Indic languages (check: native or translated) | no | no (check the parallel subset) | yes | no | no | no | no |
| Indian-language audit (arXiv 2026) | no (machine-translated science items) | no | question language only | yes (one model) | partly (language confusion) | no | no | no |
| VLURes (ACL 2026 Findings) | no Bangla | no | yes | not reported | check | no | no | no |
| Hinck et al. (EMNLP 2024 Findings) | Bengali among 14 languages | no | no | yes (for fidelity) | yes | no | no | no |
| **This paper** | **yes** | **yes** | **yes** | **yes** | **yes** | **yes** | **yes** | **yes** |

## 5.3 The novelty sentence and how to use it

> To our knowledge, no prior study combines natively written Bangla questions about natively
> photographed homes, a full question-by-answer language crossing with a no-image arm, three-tier
> open naming, and an English-transparency analysis of object names.

- Keep "to our knowledge". Never drop it.
- Use it once in the abstract and once in the introduction. Do not repeat it in every section.
- Re-run a short novelty check right before submission (Part 9). Qwen3.5 and Gemma 4 are new; a
  Bangla evaluation of them could appear at any time.
- A modest extra point: no Bangla VQA evaluation of Qwen3.5-4B or Gemma 4 E4B was found. Mention
  it once, in Methods, not as a contribution.

## 5.4 Claims we never make, and why

| Claim | Why not |
|---|---|
| Western training data causes the failures | We do not test training data; we cannot show cause |
| The dataset or models help blind users | No user study; it would be an unsupported claim |
| "First" anything without "to our knowledge" | New papers appear every month; an absolute claim is easy to break |
| The multiple-choice versus open gap is a finding | It is well documented (SpatiaLab, ICLR 2026, reports about 23 points across 25 models, and a similar gap in humans) |
| Naming specificity or language crossing is a new topic | Both exist; our novelty is the combination, the resource and the word-form test |
| The cultural stratum shows anything | About 35 items; too few for a claim |
| The fine-tuning is a method contribution | It is one setting used as a probe |
| Anything about free-form answers | Not studied in this paper |

## 5.5 Literature map

**Rule:** cite only what is verified. "PR" means a peer-reviewed venue is confirmed. "arXiv" means
preprint only. "Check" means a detail (venue, authors or ID) must be confirmed on ACL Anthology,
OpenReview, CVF or the publisher page before citing.

| Topic | Paper | Status | Use it for |
|---|---|---|---|
| Bangla VLM evaluation | BanglaVerse (arXiv 2603.21165) | PR (EMNLP 2026 Findings) | closest competitor; its knowledge-versus-grounding conclusion rests on a domain heuristic; our controlled design answers that question directly |
| | BanglaProtha | PR (WACV 2026) | Bengali cultural multiple-choice VQA |
| | ChitroJera | check venue | regionally relevant Bangla VQA |
| | Bangla-Bayanno (arXiv 2508.19887) | arXiv | large translated Bangla VQA |
| | Bangla medical VQA | PR (ACL 2026 Findings) | Bangla and English QA compared in a medical domain |
| Indic and multilingual VLM evaluation | IndicVisionBench (arXiv 2511.04727) | arXiv | Bengali included; image-removal check |
| | Indian-language audit (arXiv 2603.26742) | arXiv | question-language swap; no-image ablation; chain-of-thought hurts Bengali |
| | CVQA | PR (NeurIPS 2024 Datasets and Benchmarks) | culturally diverse multilingual VQA |
| | MVL-SIB | PR (ACL 2025 Findings) | vision-language support falls faster than text support in low-resource languages |
| | Multilingual medical VQA degradation (arXiv 2608.22363) | arXiv | degradation depends on the capability tested |
| Language crossing and fidelity | VLURes | PR (ACL 2026 Findings) | input and output language combinations |
| | Hinck et al., DOI 10.18653/v1/2024.findings-emnlp.783 | PR (EMNLP 2024 Findings) | image-induced fidelity loss |
| | MultiQ | PR (ACL 2024 Findings), per the novelty report | language fidelity in text models |
| | Cross-lingual artwork explanation | PR (NAACL 2025 Findings), per the novelty report | input and output language in VLMs |
| | MTVQA | check venue | question language barely mattered for text-in-image VQA |
| No-image and text priors | "Do VLMs See or Guess?" (arXiv 2606.10400) | arXiv | text-prior reliance |
| | DISSECT (arXiv 2604.06250), Visual Ignorance (arXiv 2606.06890) | arXiv, per the novelty report | text priors in VLMs |
| | ViLP | ID not confirmed | do not cite until the arXiv ID is confirmed |
| Naming specificity | ManyNames, "Object Naming in Language and Vision" | PR (LREC 2020) | naming variation and entry-level names |
| | Basic Category Usage in VLMs (arXiv 2503.12530) | arXiv | VLMs prefer basic-level names, as humans do |
| | Taxonomy-aware evaluation of VLMs (arXiv 2504.05457) | check venue | partial credit for less specific answers; string metrics miss taxonomy |
| | Culture Affordance Atlas (arXiv 2512.03173) | PR (AAAI 2026) | local household objects in Dollar Street |
| | Nwatu et al. | PR (EMNLP 2023) | VLM performance gaps on lower-income household images |
| | Cultural competence position paper (arXiv 2505.22793) | arXiv | naming objects is not cultural competence |
| Multiple choice versus open | SpatiaLab | PR (ICLR 2026), per the novelty report | the gap is well known, in humans too |
| | AutoConverter | PR (CVPR 2025), per the novelty report | open answers are hard to score |
| | Answer Matching (arXiv 2507.02856) | arXiv | options can give answers away |
| Word form | Orthography and semantics in bilingual LLMs (arXiv 2501.09127) | arXiv | cognates in text models |
| | IchthyoNoma (arXiv 2609.03985) | arXiv | Bengali names fail where English names work (CLIP-style fish recognition) |
| | PuMVR, script consistency (arXiv 2606.17188) | arXiv, per the novelty report | VLM accuracy is not the same across scripts of one language |
| | Bangla word origin | Bangla Academy dictionary; a Bangla linguistics reference (for example Chatterji's *The Origin and Development of the Bengali Language*, check the edition) | coding source and background |
| Fine-tuning | Low-resource LMM survey | PR (Information Fusion 2026), per the novelty report | LoRA and forgetting in low-resource settings |
| | Multilingual textual regularization (arXiv 2503.22577) | arXiv | mixing multilingual text reduces English replies |
| | VLM QLoRA for excavators (doi 10.3389/frai.2025.1681277) | PR (Frontiers in AI 2025) | a Frontiers precedent for VLM fine-tuning |
| Models | Gemma 4 Technical Report (arXiv 2607.02770) | arXiv | model source |
| | Qwen3.5, Gemini 3.6 Flash | official model cards and blog posts | model sources; record the access date |
| Frontiers style references | doi 10.3389/frai.2026.1854873; doi 10.3389/frai.2026.1884321; doi 10.3389/frai.2025.1646176 | PR (Frontiers in AI) | section fit and writing style |

---

# PART 6: REVIEWER QUESTIONS AND OUR ANSWERS

*Prepare these before submission. Each answer points to where the evidence is.*

1. **"Crossing input and output language is not new (VLURes)."** Agreed, and we say so. Our
   contribution is the evidence on natively written Bangla items, combined with a no-image arm,
   fidelity, three-tier naming and the word-form test (Part 5.2).
2. **"Only three models."** The claims are per model, from two open families with different
   tokenizers and one frontier model. The design question is about decomposition, not a
   leaderboard. Reserve: Qwen3.5-9B or another closed model within days (Design §16.2).
3. **"Why no chain-of-thought or thinking?"** We measure direct answering. Reasoning in English
   would mix the language question with reasoning ability, and a 2026 audit found
   chain-of-thought hurt Bengali.
4. **"Your English questions are translations; maybe they are easier."** Every string was
   post-edited by native speakers under a no-disambiguation rule, back-translated blind by
   different people, and scanned for leaks (Design §4.6; Table T5).
5. **"Your gold names are your own choices."** Accepted names come only from the annotators' own
   usage, frozen before any model run. Two people named 200 objects; the one with no part in the
   lexicon gives the main human
   reference, and the audit estimates the gold-error rate (Tables T5, F5).
6. **"There is no inter-annotator agreement for the dataset."** The dataset paper covers
   construction. Here we measure the instruments the claims rest on: scorer agreement and κ,
   human naming agreement and κ, translation faithfulness, and an audited gold-error rate.
7. **"Why not BERTScore, chrF or an LLM judge?"** All scored items are closed-answer, so rules
   suffice. The supplement shows string metrics cannot tell a generic name from a wrong one,
   replicating earlier English findings.
8. **"English answers are easier because English is less specific (pot versus হাঁড়ি)."**
   Exactly, which is why every Bangla-versus-English naming comparison uses only English-unique
   objects (Design §4.4).
9. **"Forced choice is easier for everyone; what does it prove?"** We do not report the gap as a
   finding. We use forced choice only to sort errors, and we report a no-image forced-choice run
   showing how much the options alone give away.
10. **"The word-form effect is just that modern objects (fridge, sofa) are easier."** The
    transfer rate is computed only on objects the model already names correctly in English, so
    recognition is held fixed. The GEE adds frequency and cluster membership as covariates.
11. **"Why group Persian and Portuguese loans with native words?"** Our hypothesis is English
    transparency. The origin breakdown shows whether other loans behave like native words, which
    is the test of that idea.
12. **"Does adding the image hurt Bangla fidelity, as Hinck et al. found?"** We test it directly
    (P4) on native Bangla, for three current models.
13. **"Fine-tuning on in-domain data obviously helps."** We do not claim it helps. We report what
    changes: Bangla, English, no-image and fidelity, across three seeds, on unseen houses.
14. **"Only 15 test houses for fine-tuning."** Results are paired per item, with house-clustered
    CIs and three seeds. We state the limit.
15. **"Multiple comparisons."** Holm correction across the six hypothesis tests per model, all
    pre-specified in a dated, tagged GitHub commit before any full run.
16. **"Thinking set to minimal is not zero."** We log the thinking tokens for every call and
    report them.
17. **"Closed models change over time."** Exact model IDs and run dates are reported; outputs are
    available on request and released with the dataset.
18. **"Where is the data?"** On request until the dataset release (supervisor decision); code
    public; lexicon in the supplement.
19. **"Why not study culture directly?"** Only 6 culturally specific object names reach usable
    counts. We report that stratum as exploratory and make no claim from it.
20. **"Why these prompts?"** One fixed template per language pair, written fully in the question
    language, the same for every model, given verbatim in the supplement. A paraphrase check is in
    the reserve list.

---
# PART 7: WRITING MAP FOR FRONTIERS

## 7.1 Format facts (check the author guidelines again in the submission week)

- **Article type:** Original Research. Main text up to 12,000 words; abstract up to 350 words; up
  to 15 figures and tables in total.
- **Structure:** Abstract, Introduction, Materials and Methods, Results, Discussion. A Conclusion
  section is allowed. Related work can be a subsection of the Introduction.
- **Required end sections:** Data Availability Statement, Ethics Statement, Author Contributions,
  Funding, Conflict of Interest, Generative AI Statement, and Supplementary Material if any.
  Draft text for the first three is in Design §14.2.
- **Section:** Computational Linguistics and Natural Language Processing. The section's name has
  changed before; pick it from the submission system's dropdown.
- **If redirected to the backup** (Frontiers in Computer Science, Computer Vision): reviewers
  there want more vision-side analysis. Lead with the visual covariates, run the image-resolution
  check from the reserve list, and consider a title centred on "visual grounding versus lexical
  knowledge".

## 7.2 Section plan (about 8,800 words)

| Section | Words | Content | Fed by | Draft now? |
|---|---|---|---|---|
| Abstract | up to 350 | question, design, the key numbers, the novelty sentence | everything | last |
| 1 Introduction | about 1,600 | the problem (DiB baselines), the language-or-knowledge question, the five-step story, C1 to C4 | Part 1, Part 5 | yes; finalise after results |
| 1.1 Related work | about 1,000 of the 1,600 (25 to 30 papers; see `BanglaIndoorVQA_LitReview_Kit.md`) | Bangla VLM evaluation; language crossing and fidelity; text priors; naming specificity; word form; multiple choice versus open | Part 5.5 | yes, early |
| 2 Materials and Methods | about 3,200 | data (fresh text, cite DiB); probe sets; lexicon and word-form coding; English questions; models and prompts; experiments; scoring; statistics; fine-tuning | Design §3 to §9 | yes; check results as `[pending]` |
| 3 Results | about 2,300 | language and image (F2, T4); naming and the error budget (F3, F5); word form (F4); repair (F6); instrument checks (T5) | the runs | no |
| 4 Discussion | about 1,400 | what the decomposition says per model; links to BanglaVerse, Hinck et al. and the Indian-language audit; what it means for building Bangla VLMs; limitations (about 400 words) | results, Part 1.4 | limitations yes |
| 5 Conclusion | about 250 | the answer to the title, in plain words | results | no |

**Drafting order:** Methods → Related work → Limitations → Introduction → Results → Discussion →
Conclusion → Abstract. About 4,500 words can be drafted before any result exists.

## 7.3 Limitations already known (draft these now)

- The English questions are translations of Bangla originals, not native English. Post-editing and
  blind back-translation reduce but do not remove this.
- Gold names come from the team's lexicon. The human naming reference comes from two team members
  who are trained annotators; one of them
  helped write the lexicon, so the main human row is the other namer.
- Three models, two of them small. Results are per model and may not hold for every model.
- Closed-answer types only; free-form answers are not studied.
- The cultural-specificity stratum is small and exploratory.
- Fine-tuning: one model, one setting, 15 test houses.
- The 68 homes are not a statistical sample of Bangladeshi households; results describe these homes.
- Gemini's minimal thinking is not zero thinking; thinking tokens are reported.
- Word origin is coded on the head noun of each name, which simplifies compounds.

## 7.4 Style rules for the manuscript

- Simple academic English in the team's own tone. Define each technical term once, in one short
  sentence.
- **No em dashes or en dashes.** Use commas, colons, or "to" for ranges.
- Never write a result before it exists. Use `[pending: P7]` style placeholders.
- Report every model separately. Never average models into one number.
- Every novelty statement carries "to our knowledge".
- Use one term per idea throughout: "generic-name fallback" (define it once as superordinate
  naming), "English route", "English-unique", "fidelity", "strict" and "lenient".
- Every figure and table gets a caption that stands on its own, with CIs shown.
- The Generative AI Statement names every AI use truthfully.

---

# PART 8: DECISIONS LOG

v1.0 decisions D01 to D61 stay in the old guide for history. **Where they conflict with D62
onward, the newer decision wins.** Superseded: D37 (OSF) by D64; D59 (E4-600) by D78; the model
roster in D61 by D69; all v1 decisions about the judge, hazard, grounding, human ceiling and FT-B
by D66 to D72. Still in force: D41 to D46, D57 and D58. Signed off: D60 on 2026-10-03 and D47 to D56
on 2026-10-18 (D240), as built into the kept scripts; any about P-HAZ or other cut work lapse.

| # | Date | Decision | Reason |
|---|---|---|---|
| D62 | 2026-10-01 | Target: Frontiers in AI, Computational Linguistics and NLP section, Original Research. Backup: Frontiers in Computer Science, Computer Vision | The team's fixed Q2 target; best section fit per the novelty report; AIUB counts the backup's quartile; the university covers the fee |
| D63 | 2026-10-01 | No task ownership and no dates in plan files; people counts only | The team lead assigns people; an AI-estimated date could be wrong and waste novelty time |
| D64 | 2026-10-01 | OSF dropped; the analysis plan is committed to GitHub before full runs (Gate E). Supersedes D37 | Same protection at zero cost |
| D65 | 2026-10-01 | Closed-answer types only; rule in Design §3.3; P-CORE-CLOSED = 1,115 items | Deterministic scoring for every claim; part-of-room scene answers cannot be scored by rules |
| D66 | 2026-10-01 | Hazard diagnostic cut (E7, P-HAZ, ASSET-6, Table 8) | It supported no contribution |
| D67 | 2026-10-01 | LLM judge and judge validation cut (E5, P-JUDGE) | Nothing left to judge after D65 |
| D68 | 2026-10-01 | IAA study and human ceiling cut (ASSET-3, ASSET-7, E10); replaced by HT-3 to HT-5; κ only from that work | Measures the instruments the claims rest on; respects the supervisor's wish |
| D69 | 2026-10-01 | Three models: Gemini 3.6 Flash (minimal thinking), Qwen3.5-4B, Gemma 4 E4B; reserve list for reviewers | Enough for a per-model decomposition; overrides the roster in D61 |
| D70 | 2026-10-01 | Colab Pro (L4) primary, one Kaggle account secondary; one platform per model; no fine-tuning on a T4; Gemma 4 never on a T4 | bf16 and fp16 numerics differ; known Gemma 4 fp16 problem |
| D71 | 2026-10-01 | Gemini in batch, paid tier only, $40 hard cap; the rest of the budget held as reviewer reserve | Minimum cost; free-tier data may be used for training |
| D72 | 2026-10-01 | Fine-tuning: Qwen3.5-4B, LoRA in bf16, one setting, 3 seeds, framed as a diagnostic case study; FT-B and the out-of-domain check cut; EXP-R cut if no seed completes | The novelty report asked for 2 to 3 seeds and a diagnostic framing |
| D73 | 2026-10-01 | No-image arm (L-BB0, L-EE0) and B0 baseline added | Riddle-style object questions and "one"-dominated counts are partly answerable without seeing |
| D74 | 2026-10-01 | Answer-language fidelity measured; strict and lenient scores | Hinck et al. (2024); separates "replied in English" from "wrong" |
| D75 | 2026-10-01 | Word form: `en_route` primary, `origin3` from the Bangla Academy dictionary secondary; C1 is the lead contribution | The report's clearest novelty; English-transparency framing; many Bangla words are Persian or Portuguese loans |
| D76 | 2026-10-01 | Cultural-specificity stratum exploratory and descriptive only | 6 names, about 35 items |
| D77 | 2026-10-01 | `en_unique` rule: Bangla-versus-English naming comparisons only on English-unique objects | Shared English names (pot) would make English look better for the wrong reason |
| D78 | 2026-10-01 | Specificity prompt on all of P-NAME; the `in_E4_600` flag is unused. Supersedes D59 | The report singled out this prompt as the new element |
| D79 | 2026-10-01 | Forced choice on P-CLUST only, exactly 4 options, with and without the image; English forced choice only on P-FCE | Fixed chance level; options can give answers away; settles the open rule for non-cluster items |
| D80 | 2026-10-01 | The error budget replaces v1's decision table and human ceiling | A direct per-item reading of what fixes each error |
| D81 | 2026-10-01 | Token cost becomes an item-level covariate; the cross-model fertility regression is cut | A regression across 3 models means nothing |
| D82 | 2026-10-01 | No Mendeley link at submission; code public; lexicon in the supplement; the rest on request until the dataset release | Supervisor decision |
| D83 | 2026-10-01 | New labels EXP-L, EXP-N, EXP-R and Gate A to H; v1 labels E1 to E10, CTRL-n and G2 to G8 retired; ASSET-3, 6, 7 cut; ASSET-10, 11, 12 added | Avoid mixing the old and new plans |
| D84 | 2026-10-01 | Floor gate on EN→EN accuracy only, never on Bangla; swap to the next size; fine-tuning target order Qwen3.5-4B, then Gemma 4 E4B | A Bangla gap is the finding, not a reason to drop a model |
| D85 | 2026-10-01 | Gemini: one 7-day window; fallback 3.6 Flash, then 3.5 Flash, then 3.7 Flash; re-run all Gemini conditions if the model changes | Avoid a model change in the middle of the runs |
| D86 | 2026-10-01 | Whole prompt in the question language; same templates for every model and type; no-image versions remove only the image clause; frozen at Gate E | Keeps "question language" clean and comparisons fair |
| D87 | 2026-10-01 | Contribution order C1 to C4; the safe novelty sentence; the updated list of claims never made | The report's verdicts: narrow wording for its Parts 1, 2, 3 and 5 |
| D88 | 2026-10-01 | Human checks HT-1 to HT-6 with people counts and "different people" rules; HT-3 before the namers see any output (Gate H); κ reported where it is free | Each check must be independent |
| D89 | 2026-10-02 | No separate file-fingerprint tracking anywhere; freezes are recorded as Git tags `gate-A`, `gate-E`, `gate-F`; seeded ranking (seed 42) stays as the sampling method | OSF is gone and Git already records exact file versions; manual tracking added work without adding proof |
| D90 | 2026-10-02 | ASSET-5 reorganised: the existing 5B script runs as is; one new 5C script draws P-CORE-CLOSED, R-TEST and PILOT; P-FCE moves to ASSET-11 and NAME-REF to ASSET-12; one new ASSET-4 v2 script replaces the v1 4B script; finished work (ASSET-1, 4A, 5A) is kept | One asset per chat in one go, with stages only where Gate A forces them; re-running finished, frozen work adds risk and no value |
| D91 | 2026-10-02 | Scoring precedence: a term in the gold's Tier-2 set scores Tier 2 even if it is another object's Tier-1 name; a more specific name than the gold is a sibling unless listed as a variant. Forced-choice distractors never come from the gold's Tier-2 set | Clusters contain generic members (টেবিল, শোকেস); without the rule the scorer and the option sets are ambiguous |
| D92 | 2026-10-02 | H1 is tested as P12 = P2 − P1; Holm within each model over six tests (P12, P3 to P7); P1 and P2 are reported as estimates | The v2.0 family did not contain the actual H1 comparison |
| D93 | 2026-10-02 | Code lives in a private GitHub repo during the project and goes public at submission; data, question text, raw outputs, human sheets, keys and model weights never enter Git and stay in Google Drive | Supervisor decision D82; a file deleted from Git stays visible in its history once the repo is public |
| D96 | 2026-10-02 | The closed-answer rule checks the question and the gold of all five types (A2). ASSET-5C checks the verified per-type counts. Amends the number in D65: P-CORE-CLOSED = 1,118 | The design's table kept items whose question asks for something other than their type; they would be scored wrong by construction |
| D97 | 2026-10-02 | PILOT is 100 different items: the naming part skips ids already in the closed part and any P-NAME item whose question fails the rule | A repeated or broken item wastes a pilot slot |
| D98 | 2026-10-03 | Ten items corrected in the merged dataset and the DiB file (A3); 5A, 5B and 5C re-run; P-CORE, P-NAME and P-CLUST membership identical | The question or gold did not fit the item's type; four of the items sit in P-NAME |
| D99 | 2026-10-03 | File homes: probe files with text only in Drive `probes/`; ASSET-1 and 4A tables in Drive `results/assets/`; repo `probes/` holds six qa_id lists (P-CORE, P-CORE-CLOSED, R-TEST, PILOT, P-NAME, P-CLUST); 5A and 5B run with `--out` set to Drive `probes/` | One copy of each probe file, so no chat receives out-of-date text |
| D100 | 2026-10-03 | Three test-split items stay out of R-TEST: H25_0004_q4 and H35_0080_q4 (how-many word misspelt in a way not on the list fixed on 2 Oct) and H14_0063_q2 (a where-question whose gold is an object). The gold বাথরুমে (3 items in H14) counts as বাথরুম | The rule was fixed before the test split was seen; changing it afterwards would not be pre-specified |
| D101 | 2026-10-03 | Changes after the lock are recorded in `DESIGN_ADDENDUM.md`, which overrides the design and the guide. The design and guide are edited once, before the Gate E commit. Replaces rule 6 of Design §0.2 until then | One place to update, so files cannot disagree |
| D60 | signed off 2026-10-03 | The v1 decision D60, built into the 5B script, is signed off by the team lead now that 5B has run | It was the open sign-off noted in PROJECT_STATE |
| D102 | 2026-10-02 | File formats: `lexicon.csv` cells with several values use `\|`; siblings come from `cluster_id`; `closed_vocab.csv` has columns kind, class_id, forms_bn, forms_en, note (B1). The ASSET-4 v2 script writes these formats and imports the scorer's romanisation, similarity and English-uniqueness functions | No file defined them, and the script that writes the files must match the scorer that reads them |
| D103 | 2026-10-02 | Spacing rule: a name matches with or without a space or hyphen between its words | The gold answers themselves vary this way; it is spelling, not meaning |
| D104 | 2026-10-02 | Ending rule: 16 Bangla noun endings and the English plural are accepted on the last word of a name, only when the rest is a listed term | 125 of 147 material golds are genitive; without the rule Bangla answers are scored more harshly than English ones, which would inflate the writing cost |
| D105 | 2026-10-02 | Strict rule: the correct name, class or number must itself be in the requested language; fidelity stays a separate whole-answer score; digit-only counts fit any language | An answer giving both languages would otherwise flip on a letter count |
| D106 | 2026-10-02 | Refusal rule: an empty answer, or a refusal phrase in an answer with no scoreable content | §9.3 points to a definition that §8 does not contain |
| D107 | 2026-10-02 | Label rule for forced choice: only shown labels count; a lone "a" is a label only when alone, marked, cued, or last | "a bucket" must not be read as option A |
| D108 | 2026-10-02 | Exact-gold rule: a naming or room gold must equal a listed Bangla name exactly; otherwise the item is unscoreable and reported | Partial matching can map a gold to the wrong object (জানালার গ্রিল to জানালা) |
| D109 | 2026-10-03 | English-copy scope: the last one to three Bangla words of the answer are compared; no endings are removed | 40 false labels fell to 3 in 22,316 test scorings; Bangla puts the noun last |
| D110 | 2026-10-03 | Gold or question problems found from now on are not corrected in the dataset. The item goes into `probes/excluded_items.csv` (qa_id, probe sets, type, reason code, source; no text). Listed items are run and never scored. The reason must concern the gold or the question, never a model answer. The list closes at Gate E; later findings go to HT-5. Amends A3 and B5 tasks 1 and 2 | Each correction forced a re-run of 5A to 5C. An exclusion decided before any reported model run cannot bias a paired comparison, because the item leaves every cell at once |
| D111 | 2026-10-09 | A dictionary label of স. (Sanskrit, direct or through Prakrit) is coded `inherited`. A label that names two languages is copied whole into `origin_note` and coded by the head noun's own language | The dictionary prints the source language, not the words তৎসম or তদ্ভব; both are Sanskrit-derived |
| D112 | 2026-10-09 | If the head noun is in the dictionary only under another spelling of the same word, the origin is coded from that entry and `origin_note` records the spelling looked up. `uncertain` is only for a word with no entry under any spelling | A spelling difference does not change where a word comes from |
| D113 | 2026-10-09 | A label of মুন্ডা, কোল or another early language of the region is coded `inherited`. A chain of languages is coded by the language Bangla took the word from directly | Such words are দেশি, which Design §4.3 maps to `inherited`; the origin analysis asks how the word entered Bangla |
| D114 | 2026-10-10 | `en_route` is yes only when a whole accepted Bangla name is an English name of the object in Bangla letters, and that English name is listed. A name that is partly Bangla, or that has a Bangla ending inside it (পড়ার টেবিল, সসের বোতল), is no. The origin of the word plays no part | Design §4.4 says "an accepted English name written in Bangla letters"; part-English names would put non-English names into the English-route group of H4 |
| D115 | 2026-10-10 | Two different members of `clusters_parsed.csv` are never merged into one object; the script stops on it. A general member is handled as a generic name (Tier 2) of the specific ones | G1, P-CLUST and the sibling rule count the members separately |
| D116 | 2026-10-10 | P-CLUST stays at 492 items. The 8 P-NAME items whose gold is বুকশেলফ now belong to the cluster object বইয়ের তাক; they get the sibling rule in scoring and no forced choice, and they are not in the error budget | 5B drew P-CLUST by exact name before the lexicon existed; changing a probe set after it was drawn would not be pre-specified |
| D117 | 2026-10-10 | Parallel forms. A short name, or an English word in Bangla letters, that is accepted in one language must be accepted in the other, in the lexicon and in the vocabulary. If the Bangla form is not attested (Design §4.2), the English form is removed, or both become generic names | A short answer accepted in one language only would move the Bangla against English comparison through our own list |
| D118 | 2026-10-03 | The lexicon holds one object for every naming gold of P-NAME, P-CORE-CLOSED and R-TEST and for every cluster member, not for every name in the lexicon file. The freeze refuses to write while any such gold does not resolve to one object or is not excluded | The scorer needs every gold to equal a listed name (D108); an object that is never a gold and never a sibling cannot change a score |
| D119 | 2026-10-03 | `cultural_specificity` of an object is the 4A tag of its main name, else of a merged spelling, else `not_rated`. `not_rated` objects stay out of the cultural stratum. 4A stays frozen | Closes the open point of A5 (names without a 4A tag) without a new rating round |
| D120 | 2026-10-03 | A number form that is also a word of a refusal phrase is not allowed in the vocabulary. The bare word নয় is therefore not a form of 9; নয়টি, নয়টা and digits are | With নয় listed, the refusal "বলা সম্ভব নয়" is read as 9: scored correct when the gold is 9, and not counted as a refusal |
| D125 | 2026-10-04 | Stage 1 files are model output under the hard rule of Design §4.2. Until `gate-A` is tagged, nobody in the HT-1 pair opens a file in Drive `english/` other than `asset2_strings.csv`. Stages 2 and 3 before Gate A are done by other people. The script prints no question and no translation | 199 strings are count questions that name objects. If the lexicon writers saw Gemini's English names, the English accept sets would lean towards a model we evaluate |
| D126 | 2026-10-04 | No sampling parameter is sent to Gemini 3.6 Flash. "Greedy (temperature 0)" in Design §5.2 holds for the two open models only. Table T2, the Methods and the Limitations say so | Google's API ignores `temperature`, `top_p` and `top_k` for this model. A paper that claimed temperature 0 for Gemini would be wrong |
| D127 | 2026-10-04 | Stage 1 settings: the unit is the normalise()d question string (653); the most frequent surface form is sent; standard mode (Design §12.1); minimal thinking; `max_output_tokens` 4,096; question strings only; naming questions in their own calls; ছবি is "image" when it means the photograph, to match the English templates of Design §6.2 | Design §4.6 fixes the model and the call size only. The rest had to be set once and recorded |
| D128 | 2026-10-04 | One API cost ledger for the project: `src/cost_ledger.py` and Drive `outputs/api_cost_ledger.jsonl`. Every billed script imports it. Stage 1 has its own cap of 0.50 USD | The 40 USD hard stop must hold across scripts and chats |
| D129 | 2026-10-05 | The stage 1 prompt is `mt-v2`: rule 9 fixes the English for দেখা যাচ্ছে ("visible"), জিনিস ("thing") and বস্তু ("object"). All 653 strings use `mt-v2`; the `mt-v1` PILOT output stays in the cache as history | The `mt-v1` PILOT output gave one Bangla wording several English wordings. That variation would exist in English only |
| D130 | 2026-10-10 | HT-2 sheets are Excel files in Drive `english/stage2/` and `english/stage3/`, not `human/`. People appear only as role codes PE1, PE2, BT1, BT2. The script prints ids and counts only; files with text stay in Drive | Keeps the sheets beside the files they come from and keeps question text out of logs |
| D131 | 2026-10-10 | The post-edit split is a seeded rank (seed 42) within question type and high-use group; spelling variants and count questions with the same object name stay with one post-editor; every row needs an explicit keep or edit | One post-editor decides each group of related strings; an empty row would hide an unread string |
| D132 | 2026-10-10 | The cross-check is the random 10% of the other's half plus every string used by 10 or more items | 4 strings carry 53.8% and 25 strings 70.7% of the 3,069 items; a random 10% would see about 1 in 10 of them |
| D133 | 2026-10-10 | American spelling in all English questions | The first pass wrote it; D5 asked for one spelling |
| D134 | 2026-10-10 | No final file while review rows are unread; rows that are right are accepted with `--accept-review N`. A named object with no everyday English word is written in Latin letters with no explanation | Variation that exists in English only adds noise to EXP-L; an explanation would make the English question easier than the Bangla one |
| D135 | 2026-10-10 | Final files `english_questions_final.csv` and `english_questions_final_PILOT.csv`, English column `question_en`, `first_pass_en` kept, versions kept, late corrections only through `english/amendments/`, the PILOT file never replaced once the pilot has used it | D149 needs one named English column; every change must stay traceable |
| D136 | 2026-10-10 | The stage 3 draw is ranks 1 to 150 of a seeded ranking over all 653 final strings, 75 for each back-translator, item codes and English only, no translation or AI tool; a repeat uses ranks 151 to 300 | `asset2_strings.csv` links every string_id to its original, so sheets with string_id would not be blind |
| D137 | 2026-10-10 | Each stage 3b pair is judged once, by the post-editor who did not post-edit the string; every not faithful pair counts whatever its cause | Nobody grades their own English, and a fault cannot be excused by blaming the back-translator |
| D138 | 2026-10-10 | English faults found in stage 3b are corrected even after a pass, agreed by both post-editors, as a new version of the final file; the reported percentage is the one measured before | No known fault stays in the file, and the number is not improved after the fact |
| D139 | 2026-10-10 | A question that names an object of the frozen lexicon uses its `canonical_en` | Questions and accepted answers then use the same English word (C5 suggestion, adopted) |
| D140 | 2026-10-06 | The harness has two Gemini paths: standard mode for the pilot, Batch mode for EXP-L and EXP-N; smoke tests may use either. The stage decides and the harness enforces it | Design §7.4, §12.1 and D71 already say so; the project instructions box did not |
| D141 | 2026-10-06 | Billed harness tests in stage smoke, cap 0.50 USD: 20 standard-mode calls, then the 100 PILOT items as the first Batch job on the key | A wrong request body should fail on one cheap call, and the Batch path had never run |
| D142 | 2026-10-06 | Batch and the ledger. Before a submit: worst-case cost at Batch prices, the 40 USD check and the stage cap, an audit of every request body, then a reservation line in the ledger. At collection the reservation is released and the real cost from `usageMetadata` is written; a failed line is charged 0 and a reply without usage data is charged at the worst case. Never twice: answered items and items in an open job are not planned again; Google's job list is read first and an unknown live job (same stage, model and condition, or without a display name) stops the submit; an intent line is on disk before the create call; the create call is never retried, and an unknown outcome is settled by looking the job up by its display name | Batch creation is not idempotent, and the cap must hold while a job runs |
| D143 | 2026-10-06 | A well-formed reply is final, even if empty, cut or in the wrong language. Only transport errors, HTTP 408, 429 and 5xx, and failed Batch lines are tried again (3 tries per item in Batch mode) | Gemini cannot be made greedy (D126); asking again until a good answer appears would be selection |
| D144 | 2026-10-06 | Three stages: smoke, pilot, main. A cache never crosses stages. Main needs the tag `gate-E`, a clean Git tree, the full probe set of each condition, and Batch mode for Gemini. Caps: smoke 0.50, pilot 2.00, main 20.00 USD | Every reported output must be produced after the pre-specification commit. Cost: the 100 PILOT items are run again in main |
| D145 | 2026-10-06 | The cache key stays as in Design §4.9. Records also store hashes of the prompt text, the image bytes and the settings; a hit with another hash stops the run. `prompts.yaml` carries a version and a SHA-256 lock (version `exp-v1`) | English questions and templates can still change before Gate E under the same key |
| D146 | 2026-10-06 | Gemini image setting: `MEDIA_RESOLUTION_HIGH`; the image part comes before the text | Design §5.2 asks for one fixed value. HIGH is Google's recommended value for images and equals the default |
| D147 | 2026-10-06 | Open models: `do_sample` false, one beam, repetition penalty 1.0, shipped sampling values removed, batch size 1, bf16, the whole model on one GPU, the model commit fixed per stage. A change of GPU, torch, transformers, commit or Qwen kernel state stops the run | "Greedy" in Design §5.2 did not say what happens to the shipped generation settings |
| D148 | 2026-10-06 | Image loader: EXIF orientation first; a non-sRGB profile is converted to sRGB; RGB; longer side 1,024 px with Lanczos, never enlarged; JPEG 90; no metadata. Prepared once into Drive `data/images_1024/` with a SHA-256 manifest; a file that differs from the manifest is refused | Design §5.2 fixed the size and quality only; all three models must see the same bytes |
| D149 | 2026-10-06 | Interfaces. English questions: a CSV (UTF-8 with BOM) with `question_bn_norm` and one English column named on the command line, joined as in D4; the column `mt_en` is accepted in stage smoke only. Option sets: one record per item and language with `qa_id`, `lang` (bn or en) and `options` (four texts in the order shown). The answer field of an output record is `answer_text` | ASSET-2 stage 2 and ASSET-11 were not done; the harness fixes what it reads |
| D150 | 2026-10-06 | Everything under Drive `outputs/` is model output. The harness prints counts only; `peek` needs a flag. The HT-1 pair does not run the harness or open its files before `gate-A` | Naming answers of three models would bias the lexicon (Design §4.2, D125) |
| D151 | 2026-10-06 | Gemma 4 E4B runs with a visual token budget of 1,120 per image, not the processor default of 280 | At the default Gemma saw 266 tokens per image, against about 758 for Qwen and about 1,070 for Gemini; a "not known" error could then be a resolution effect |
| D152 | 2026-10-06 | Qwen3.5 kernels: `flash-linear-attention` installed in every session; `causal_conv1d` on its PyTorch fallback, because it does not install on Colab. This state is fixed for all Qwen runs | Design §5.3 asks to record the fast path, not to require it; transformers states that the fallback is correct, only slower |
| D153 | 2026-10-06 | Every open-model run uses transformers 5.18.0 | One library version per model, and v5.23 removes the way `max_pixels` is passed |
| D154 | 2026-10-10 | H49_0009_q4 is kept and scored. The post-editors judged its Bangla question clear without the missing word; the English is the question that was meant | D110 excludes an item only when the gold or the question is faulty; this question still asks one clear thing |
| D155 | 2026-10-14 | `models.yaml` pins Qwen3.5-4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a and Gemma 4 E4B ee0ef6023621cff504d758262d4e04895a5af4a2 before the first pilot run. A newer commit is adopted only with a decision row | Otherwise the first pilot run fixes whatever commit is newest; the pilot must measure the model that the full runs and Gate G use |
| D156 | 2026-10-14 | Items per condition come from PILOT.json fields through `--where`: EXP-L on `pilot_part = closed` (60); N-BB, N-BE, N-BBs on `naming` (40); N-FCB and N-FCB0 on `naming` with `in_pclust = true` (20); N-FCE and N-FCE0 on naming items that are in P-FCE. The English conditions run first on 3 items by seeded rank | Design 7.4 gives only sizes; qa_id lists cannot give the split |
| D157 | 2026-10-14 | `prompts.yaml` exp-v1 is final for the pilot. The team lead proofread the Bangla templates with no change. The N-FCB word order is kept as in Design 6.2. Cell 0 compares every template with Design 6.2 code point by code point and stops on any difference | Design 6.2 requires the proofreading; any change would need exp-v2 and a new FT dry run |
| D158 | 2026-10-14 | Gate C, first half: each pilot command runs twice. The second run must print "nothing to run: 0 calls", exit 0, and leave the cache and the ledger unchanged; the result is logged per model. The second half and Gate D use the ASSET-9 stage 2 scorer, with excluded items left out | Design 11 does not say how "re-runs from cache at zero cost" is shown |
| D159 | 2026-10-14 | One repeat of the 100 open_bn_bn requests of the pilot (L-BB on the 60 closed items, N-BB on the 40 naming items) in stage smoke, run name pilot-repeat, standard mode, right after the pilot Gemini run, about 0.09 USD. Agreement is counted against the pilot, against the Batch test of 2026-10-05 and, later, against main. Only these counts may be quoted, in the Methods or the Limitations | Gemini cannot be made greedy (D126); the pilot is the cheapest place to measure how stable a single run is |
| D160 | 2026-10-14 | H68_0023_q2 (P-CORE-CLOSED and PILOT; object, train split) goes on `probes/excluded_items.csv`: reason gold_wrong, source "Pilot image check", added_at Pilot. The gold শোপিস is a typing slip for শোকেস. The gold is not corrected | D110 excludes and never corrects; judged from the image, question and gold before any pilot answer existed |
| D161 | 2026-10-14 | Every item of P-CORE-CLOSED, R-TEST, P-NAME and P-CLUST whose Bangla question holds চিহ্নিত goes on the list: reason other, source "Pilot: question refers to the annotation mark", added_at Pilot. Five scored items hold it: 4 new rows (H02_0037_q2, H13_0005_q2, H13_0050_q2, H14_0044_q2) and H14_0046_q2, already listed under D217. No other form of the root appeared. Training data stay unchanged (D172) | One annotator used the word in 60 questions in 3 houses for the box of the annotation tool, which the models never see; one rule set before any answer treats all such items the same |
| D162 | 2026-10-15 | `gemini-3.6-flash` stays the Gemini model, and the fallback order of Design 5.5 is withdrawn. Each Gemini full run first checks that the harness's model check prints version 3.6-flash-07-2026, after the weekly model and price check (Design 0.2 rule 5). If Google sets a shutdown date or serves another version before the Gemini runs end, they stop; a decision row picks the replacement, which first runs the Gemini pilot conditions and passes Gates C and D. A model ID that the API redirects to another model is never used | On 2026-10-14 `gemini-3.5-flash` redirected to 3.6 Flash and `gemini-3.7-flash` to 3.8 Flash, which has no minimal thinking; the named fallbacks no longer exist |
| D163 | 2026-10-15 | How the pilot gates are read, fixed before any pilot answer is scored. Both gates are read per model, outside the excluded list. Gate C "parseable": the ASSET-9 stage 2 scorer gives the answer a label other than invalid; a refusal or a wrong-language answer is parseable (D143); an item whose gold is unscoreable is counted apart. Gate D 8-word rule: over all the model's pilot answers, forced choice included; shares per condition are information only. Gate D floor: strict accuracy on L-EE over the 58 scored closed items | Design 11 gives the thresholds but not the unit of reading; fixing it before scoring keeps the reading from bending to the results |
| D164 | 2026-10-16 | Forced choice runs only on the final ASSET-11 files, pinned by sha256: `fc_options.jsonl` (5288ba63...) and `P-FCE.json` (807e987b...). Items as in D156; the harness gets only the 20 PILOT naming ids in P-FCE. No billed or GPU call before a dry run and an `fc-verify` PASS made by the same script version | A wrong option set found after a billed run would leave answers in the cache that cannot be replaced (D145) |
| D165 | 2026-10-16 | One `asset9_scorer.py score` command per cache, into Drive `scored/pilot/`, with the excluded list and kept harness fields (prompt, image and settings hashes, mode, time, model version or revision); the scorer is the one pinned in `pilot.py` | One command for two caches would silently drop the D5 repeat (D188) |
| D166 | 2026-10-16 | Reading units, fixed before the answers existed. Gate C: one share per model over all 13 conditions. Gate D: the scorer's word count. The floor applies to the open models; Gemini's value is information (Design 5.4). Correct with a refusal phrase: lenient 1, strict shown beside it. The 10-item check: 4 forced choice, 2 no-image, 4 with image | These details were still open after D163 |
| D167 | 2026-10-16 | D5 is counted on 97 pairs. Text agreement is recounted beside strict and label agreement, for both comparisons. These replace the 98-pair figures of F2 | D110 and D185 |
| D168 | 2026-10-16 | The two future HT-4 checkers do not open Drive `scored/` until both HT-4 sheets are in | The full runs repeat the 97 pilot items; a pilot label seen early would break HT-4 blinding |
| D169 | 2026-10-17 | Scorer 0.2.2-stage2, forced choice only. Rule N: an English "a" right after "(" is not a label by that fact alone. Rule O (Design 8.5): one distinct label counts; with two or more, the one label marked by an answer word (উত্তর, উত্তরঃ, উত্তরটি, answer) within the three words before it counts; otherwise invalid. All pilot answers re-scored; the first scoring kept as a record. **This change was made after reading pilot answers**, before any full run | The 10-item check found "(a wall-mounted ...)" read as A, and Cell 12 found 9 Gemini N-FCB0 refusals listing "(ক, খ, গ অথবা ঘ)" read as ক, 4 of them scored correct: Gemini's no-image forced-choice accuracy read 21.1% instead of 0% |
| D170 | 2026-10-07 | EXP-R trains with PEFT LoRA and our own loop in `src/finetune.py`, on the model loaded by the harness, under transformers 5.18.0. Unsloth is not used | Unsloth's supported range ends at 5.17.0; one loader for training and inference; a fixed, recorded list of trained layers |
| D171 | 2026-10-07 | One PEFT version for training and adapter evaluation: 0.21.0, fixed in `configs/finetune.yaml` after the dry run passed. The harness refuses an adapter under another PEFT version | Same logic as D153 |
| D172 | 2026-10-07 | Training and validation data come from `splits.csv` and the dataset only. `probes/excluded_items.csv` is not applied | D110 is a scoring rule; the list is not final before Gate E; it covers probe items only |
| D173 | 2026-10-07 | Validation: the list FT-VAL400, 400 val QA by seeded rank (seed 42), all types, in the repo as qa_ids. Dry run: 100 train QA and 20 R-TEST items by seeded rank, generated under R-BB | Design 7.3 gives sizes only; R-EE waits for Gate B |
| D174 | 2026-10-07 | Harness edit (version `2026-10-07a`): an `adapter` entry in `configs/models.yaml` loads a LoRA adapter after the unchanged base load. Each adapter has its own `cache_id` and settings hash. The harness refuses an adapter trained on another base commit, with another prompt text, under other fixed settings or another PEFT version, or whose file changed. The dry-run adapter runs in stage smoke only. Entries `qwen_ft_dry`, `qwen_ft_s42`, `qwen_ft_s43`, `qwen_ft_s44` inherit every field of `qwen` | E5; base answers must not change, and an adapter must be evaluated on what it was trained on |
| D175 | 2026-10-07 | All 4,406 images are prepared once with the D148 loader; training reads the prepared copies, with the hash check on every read | Design 7.3: images as 5.2; the same bytes in training and inference |
| D176 | 2026-10-07 | Training a seed needs Gate G, Gate D for the target model, the tag `gate-E` with a clean tree, the model commit in `configs/models.yaml`, the fixed PEFT version and a passed Gate G report made with the current script version. The train command enforces all but Gate D | The prompts and the model can change until those gates |
| D177 | 2026-10-07 | Values Design 7.3 leaves open: base weights bf16 and frozen, LoRA weights float32, forward pass under bf16 autocast; LoRA on every `nn.Linear` inside the language-model layers and nowhere else; AdamW betas 0.9 and 0.999, eps 1e-8; gradient clipping 1.0; warmup over 3% of the planned updates, rounded up, then cosine to zero; a batch of 2 is two single-example passes with the pooled token loss, so no padding; the target is the gold answer as written plus the end of turn; an example over 1,536 tokens is left out and counted; validation loss is the token-weighted mean | Needed for exact Methods text |
| D178 | 2026-10-07 | Gate G also has: a count of loss positions per example; an overfit probe on 8 training examples with different short answers; seconds per item with projections against Design 12.2 | The three design checks cannot catch a wrong training target after only 7 weight updates |
| D179 | 2026-10-09 | Gate G thresholds: mean loss of steps 41 to 50 at least 0.10 and no step below 0.01; at least 18 of 20 answers not empty, in Bangla script or digits, and of 8 words or fewer; overfit probe at learning rate 5e-4, at most 60 updates, stop at loss 0.05, at least 7 of 8 answers reproduced and stopped | Design 7.3 names the checks without numbers |
| D180 | 2026-10-09 | Text rule: a line break inside a question or answer becomes one space; a QA pair with an empty question or answer is left out and counted. On the real dataset both counts are 0 | `build_prompt()` refuses a line break inside a question; the rule had to exist before the data were seen |
| D181 | 2026-10-09 | `torchao` is uninstalled in every session that trains or loads an adapter | PEFT 0.21.0 refuses the `torchao` 0.10.0 that Colab ships; we do not use it; upgrading it could move torch (D147). Base answers are unchanged without it (H2) |
| D182 | 2026-10-09 | EXP-R is planned at about 18 L4 hours of training for three seeds and about 5.5 of evaluation, replacing 10.5 and 5. The project total becomes about 33.5 L4 hours | Measured in the dry run; cutting seeds or epochs would change Design 7.3 |
| D185 | 2026-10-16 | Excluded items in the scorer: `gold_status` excluded, `scoreable` False, the reason in `gold_hint`, every score field empty. `load_excluded` reads `probes/excluded_items.csv`; `audit` and `score` read it at every run; `summarise` counts excluded records apart and keeps them out of every number | D110 and C5; D163 reads the gates outside the list |
| D186 | 2026-10-16 | A2 check: the scorer's kind, number and room class must equal `closed_kind`, `gold_count` and the class of `gold_room` on every closed item that is not excluded. A disagreement fails the audit; it is also a unit test when `BIVQA_PROBES` is set. Result: 2,641 of 2,641 agree | B5 task 2 |
| D187 | 2026-10-16 | Forced choice in `score` only through `fc_options` (`read_option_sets`, `options_for`). Label language is the condition's question language. A missing option set stops the run. The draft file is refused. `fc_options_sha256` is in every record and in the manifest | H4 and D196: the scorer scores the options the model saw |
| D188 | 2026-10-16 | `score` reads lines of `cache.jsonl`, answer field `answer_text` by default. The first record per cache key stands (D143). A record that disagrees with the condition table stops the run. A half-written last line is skipped and counted | E4; the Pilot reading may write to a cache while it is read |
| D189 | 2026-10-16 | Printouts give ids and counts only; text goes to CSV files in Drive. Folders: Drive `scored/asset9_stage2/` (`audit/`, `fc_dryrun/`, `pilot_test/`) and `human/ASSET-9/proofread/`. The 10-item check is read from `<out>_sample.csv` | Setup Part 7; no question or answer text in logs |
| D190 | 2026-10-16 | Gate C, second half, under D163 (D163 unchanged). Open conditions: the share of answers that are not empty; the share with `content_found` False is reported beside it as information. Forced choice: an answer is parseable when it is not empty and `fc_valid` is True or `refusal` is True. An empty answer, a label of the other script, a digit, or no single option named is not parseable. Read per model, outside the excluded list | `invalid` exists only in forced choice, so the open half could not fail otherwise; D163 calls a refusal parseable while the scorer labels a forced-choice refusal invalid |
| D191 | 2026-10-16 | Scorer 0.2.0-stage2: the key `content_found` in every record; `summarise` gains the columns `excluded`, `invalid` and `no_content`. No scoring rule changes (0 of 37,954 re-scorings differ) | Needed by D190 and by the proofreading |
| D192 | 2026-10-16 | The task 3 findings (B7 table: বেত, ছোবড়া, the bare পাট and সিরামিক, রঙ in the class liquid, a second colour on a patterned object, ব্যাগ for কম্বল রাখার ব্যাগ) are not edited. They go to HT-4, HT-5 and the lexicon sensitivity analysis (Design 9.4 item 12). The Scoring chat counts the answers each one touches | Gate A froze the files (Design 4.2) |
| D193 | 2026-10-16 | The ending list and the refusal phrases are proofread by one native Bangla speaker, alone, on a blind sheet of the distinct no-image pilot answers (L-BB0, L-EE0, three models). A change follows the scorer-change rules: new version, all tests again, a decision row saying it was made after reading pilot answers. Gates C and D do not wait for it | B5 and F5; a refusal flag changes no strict score |
| D194 | 2026-10-16 | Scorer 0.2.1-stage2. The refusal list gains 21 phrases (B6), taken from the proofreader's copied refusal words; **this change was made after reading pilot answers.** The endings stay as they are, because no wrong cut was named. The key `refusal_phrase` goes into every record. A count answer that holds a refusal phrase and a number (P118) is counted and sent to HT-4, with no rule change | The list missed 61 of the 120 refusals the proofreader marked, so reported refusal rates would have been about half their true size |
| D195 | 2026-10-10 | Two stages. `build` writes candidates, draft sets and the HT-6 sheet to Drive `human/HT-6/`. `apply` writes Drive `probes/fc_options.jsonl` and `probes/P-FCE.json`, repo `probes/P-FCE.ids.txt`, and the P-FCE line of `configs/conditions.yaml` | HT-6 changes the sets; files with gold positions stay in Drive |
| D196 | 2026-10-10 | One JSONL file, one record per item and language: `qa_id`, `lang`, `options` (D149) plus `correct_index`, `option_ids`, `roles`, `in_pfce`, `ht6_changed`. The draft file uses the key `draft_options`, so the harness refuses it | The scorer scores the options the model saw; no run before HT-6 |
| D197 | 2026-10-10 | Candidates by `freq_obj_name`, highest first; ties by a seeded rank of `name_id`. A candidate is skipped if any of its Tier-1 names, Bangla or English, is a Tier-2 name of the gold. A more specific kind of the gold is kept and marked | Design 4.7 has no tie rule; Design 8.3 rule 4 |
| D198 | 2026-10-10 | "Same room_scope" is the same value; `any` is a value like the others. A slot the cluster cannot fill takes the next same-room object outside it. A cell with several rooms stops the build | Five single values in the frozen file |
| D199 | 2026-10-10 | One set per item for both languages, same objects, same places. The gold is shown as `name_bn` or `canonical_en` of the resolved object. P-FCE is read from the final sets: gold `en_unique` yes and four English labels that share no form | Language is the only difference; C4 |
| D200 | 2026-10-10 | Order by SHA-256 rank per item; no `random`, no `hash()`. A replacement takes the place of the option it replaces | Same in every session; a swap never moves the gold |
| D201 | 2026-10-10 | HT-6 on one HTML sheet. The reviewer judges each distractor against the picture, Bangla and English name together. The next candidate appears at once and is checked too. Stage 2 rebuilds every set from the log and refuses a log that does not fit. A doubt about a gold is flagged, never changed (D110) | One pass; the log is the record |
| D202 | 2026-10-10 | One reviewer who is not an HT-3 namer. Drive `human/HT-6/` is closed to the namers until Gate H | The sheet shows 492 pictures beside their gold names; NAME-REF takes 100 from P-CLUST |
| D203 | 2026-10-16 | The 9 items in `human/HT-6/ht6_flags.csv` go into `probes/excluded_items.csv`, reason `gold_wrong`, source `HT-6 review`, added_at `ASSET-11`: H20_0058_q1, H28_0069_q1, H54_0006_q1 (ওয়ারড্রব shown, gold আলমারি); H05_0104_q1, H58_0020_q1 (ওয়াল শোকেস shown, gold শোকেস); H17_0010_q1 (socket shown, gold মাল্টিপ্লাগ); H25_0012_q1 (vase shown, gold টব); H21_0021_q1, H51_0047_q1 (gold does not fit the picture). Their option sets stay as they are; they are run and never scored | The reviewer judged from the picture and the gold only, with no model output. D110 excludes a faulty gold and never corrects it; the item leaves every cell at once |
| D204 | 2026-10-16 | `human/HT-6/ht6_swaps.csv` goes to HT-5 and the Statistics chat as information. Its 48 rows with reason `name` are item and sibling pairs where the reviewer judged the sibling also a right name for the picture. No score changes. The error budget text reports one count: how many sibling labels in the naming conditions fall on these pairs | A sibling answer on such an item may not be a knowledge error, and reviewers of the paper will ask; one count changes no pre-specified analysis |
| D205 | 2026-10-12 | Namers are two different native Bangla speakers from the research team. Not eligible: the team lead (read model answers through peek), the HT-1 pair (amended by D214), anyone who has seen model output (the text of any model answer to a dataset question: smoke, pilot, full runs, FT dry run, DiB zero-shot outputs, peek printouts, quotes) or scorer output (any label, score or rate computed from such answers). Not counted: the ASSET-2 English question files and count-only harness printouts. Until their own sheet is received a namer does not run or read the pilot, open Drive outputs/, scored/, models/, probes/, human/, the dataset or the lexicon, or do HT-6, ASSET-9 stage 2 work, HT-4 or HT-5; the namers do not discuss the task until both sheets are in. Methods state each namer's exposure | Gate H protects the human names from model naming and scorer knowledge; the English questions cannot shape the Bangla names people use at home |
| D206 | 2026-10-12 | `draw --namer1/--namer2` take an annotator_id or NA; only house_id and annotator_id are read; optional `--also-exclude1/2`; excluded houses are the union; role codes N1, N2; the link to annotator ids only in the Drive manifest | Design 4.8 excludes houses annotated by either namer |
| D207 | 2026-10-12 | Cell from resolve_gold and P-CLUST membership; excluded items, unresolved golds (stop), excluded houses and a second item on a drawn image are skipped; rank SHA-256("42\|NAME-REF\|<qa_id>"), 50 lowest per cell; fill from the cell with the most available items after the exclusion, filled items keep their true cell; fewer than 200 stops; the draw is redone only before any receipt, with a reason; no cap per object | Design 4.8 gives no fill procedure |
| D208 | 2026-10-12 | score_item, condition N-BB; final scores only with the gate-F scorer and the gate-A lexicon (checked by git); `--provisional` before, never reported; the excluded list is read again at scoring; versions and input sha256 recorded | Same scorer as the models; Gate F blocks reporting |
| D209 | 2026-10-12 | Separate seeded order per namer; codes N1-001 to N2-200; the key in Drive only; one item per image | Order effects do not line up between namers |
| D210 | 2026-10-12 | One zip per namer: Excel sheet plus byte-identical, metadata-free images named by code; the instruction as written, a separate Bangla typing note, a three-line declaration; leak scan of every name and cell (id patterns, question or gold text as whole words, lexicon terms); one image at a time in an ordinary viewer, no image search or AI | The design fixes images and instruction, not the medium; strict Bangla scoring needs Bangla letters |
| D211 | 2026-10-12 | `check` tests format only and never reads the lexicon or a scorer label; long names counted, never sent back; `receive` accepts only a passing sheet, records sha256 and UTC time, never overwrites; Gate H passes on both receipts, both declarations and the runner's confirmation | The names stay the namers' own; Gate H has evidence |
| D212 | 2026-10-12 | Primary: agreement and Cohen's kappa on the six labels; secondary: on strict; house-clustered bootstrap, 10,000, seed 42; per-namer and pooled shares; the same-other-object list goes to HT-5; HT-3 never adds an exclusion | "The tier" of Design 9.5 is the label of Design 8.3 |
| D213 | 2026-10-12 | Code in the repo; text in Drive human/HT-3/; NAME-REF.ids.txt committed after both receipts; ids and counts only in printouts; the runner is not a namer | Setup Part 7; Design 9.7 |
| D214 | 2026-10-13 | Namers A3 (N1) and A2 (N2), two team members; N2 is in the HT-1 pair, an exception to D205, because outside namers would need prior ethics approval and written consent and only one team member met D205. Each namer is reported separately; N1 is the main human row in F5 and T5, N2 a secondary row; Methods and Limitations say N2 helped write the lexicon; the draw records `--lexicon-writer N2`; declarations reworded to what Gate H needs. Outside namers not adopted | Kappa needs two namers; one human row stays independent of the lexicon |
| D215 | 2026-10-12 | H13_0089_q2 (P-CORE-CLOSED, object, gold কল) and H14_0085_q2 (R-TEST, object, gold তোয়ালে ঝোলানোর রিং) go into `probes/excluded_items.csv`: reason `other`, source `bn_question_leaks.csv`, added_at `ASSET-2 stage 4`. They are run and never scored; their English stays as it is | Each Bangla question holds its own gold name. D110 does not allow editing the question, and fixing only the English would make the English question harder than the Bangla one |
| D216 | 2026-10-12 | The stage 4 leak rule. Scope: every obj_name and object item of P-CORE-CLOSED, R-TEST and P-NAME, once per qa_id. English: the question the harness sends. Gold: `resolve_gold`. A leak is a match of the scorer's own search (whole words, longest match first, D103, D104) that holds a Tier-1 term of the gold: English terms in the English question, Bangla terms in the Bangla question; a shared English name counts. Gate B's leak half passes with 0 leaks, 0 items without English and 0 unresolved golds among items not excluded. Excluded items are scanned and reported, never counted. A gold Tier-2 term (list a) and a gold Tier-1 term inside another object's longer name (list b) are reported, never counted | Design 4.6 says "contain" without a matching rule; the scorer's rule decides scores. A5 ties the Bangla leaks to Gate B. A Tier-2 answer is wrong in both scores, and a longer name of another object names that other object |
| D217 | 2026-10-12 | Six more items go into `probes/excluded_items.csv` (reason `other`, source `stage 4 leak scan`, added_at `ASSET-2 stage 4`): H14_0046_q2, H14_0075_q2, H36_0081_q2, H14_0053_q2, H14_0076_q2, H14_0031_q2 | The same fault as D215: the Bangla question names its own gold in a form the lexicon does not list, or by an English name in Bangla letters; the freeze matched listed names only |
| D218 | 2026-10-12 | The five English-only leaks (q_a912a342c844, q_a19091421980, q_7cabec261774, q_3673fefbec82, q_ed5bed9f3727) are fixed by the two stage 2 post-editors, agreed by both, with the meaning and difficulty of the Bangla and no English Tier-1 term of the gold, tested by check-fixes, through `amendments_stage4_leaks.csv` and `merge --scope all --overwrite-final` (version 3). A string they cannot render faithfully is excluded (reason `other`). Result: 3 amended; H35_0016_q2 and H53_0021_q2 excluded. The PILOT file stays version 2 | The Bangla questions are sound, so the fault is fixed where it is (D9, D10, D138). None of the five is a PILOT string |
| D230 | 2026-10-18 | Layout and modes. One file of functions (`src/stats.py`), one script, one thin notebook. Placeholder mode is read from the records' stage; placeholder files go only under Drive `placeholder/`, each file name, table row and figure visibly marked. Real output goes to Drive `results/stats/`; never the repo, `scored/` or `outputs/`. Real mode needs stage main, 10,000 resamples, the lexicon and vocabulary equal to `gate-A`, scoring.py and normalise.py equal to `gate-F`, stats.py and analysis.py equal to `gate-E` (otherwise `--deviation "<reason>"` and a decision row), and the running code must be the repo's | Design 9.7; placeholder numbers must never be mistaken for results; the reported numbers must come from the pre-specified code |
| D231 | 2026-10-18 | Records and joins. Input is the scorer 0.2.2-stage2 record (B4, B6, B9 keys, `fc_options_sha256`, the kept harness fields). One scorer version and one prompt version. Excluded items come from the list and must agree with `gold_status`. An item joins its object through `gold_name_id` (forced choice: `resolve_gold` on the probe record). The scored sizes are pinned together with the excluded list's sha256: a new row fails Cell 0 until both pins are updated with a decision row | C4, C5, D110, D185 |
| D232 | 2026-10-18 | Bootstrap and tests as J1: one seeded draw per house set; ratio estimators; undefined resamples counted; p of 0 shown as below its floor and Holm bounds marked; P1, P2 estimates; Holm over P12, P3 to P7 within each model; P8 a single test; EXP-N on 61 houses, the P7 pool on its 68 | Design 9.1, 9.2 left these open |
| D233 | 2026-10-18 | EXP-N definitions as J1: requested-language tiers; P7 pool of en_unique naming items, P-NAME items from N-BE and N-BB and other P-CORE-CLOSED naming items from L-BE and L-BB, each qa_id once; seven F5 segments; human rows; log(freq + 1); `origin3` uncertain left out of the origin split | B5 asked for the language rule; frequency 0 exists |
| D234 | 2026-10-18 | Error budget as J1, the D204 count, the HT-5 mark, and the command `budget-items`, which writes the HT-5 sampling frame (model, qa_id, house, category; ids only) from the scored files alone | HT-5 samples its C answers from the budget, while the full run needs the HT-5 file |
| D235 | 2026-10-18 | GEE: statsmodels, binomial, exchangeable; the `overview` indicator; direction checks P1 against -(b_q + b_qa/2), P2 against -(b_a + b_qa/2), P7 against b(a_bn:en_route); the template rule of "more conservative"; a failed fit is reported, not hidden | Rank deficiency in the design formula; Design 9.1 asks for agreement in direction |
| D236 | 2026-10-18 | template_id: normalise(), then remove chandrabindu, ্যা to ্য, ী to ি, ঈ to ই, ূ to ু, ঊ to উ, ণ to ন, ষ and স to শ, ঙ to ং, remove ZWNJ, ZWJ and spaces. Merged groups checked once by one native speaker (all right). B0: most common normalise()d gold of the same qa_type and template_id from other houses, ties by SHA-256("42\|B0\|<gold>"), else the most common gold of the type from other houses; excluded items out; scored under L-BB, strict | Design 9.2 and 7.1 give no rule; the open issue on spelling variants is closed |
| D237 | 2026-10-18 | EXP-R readouts: dBN R-BB strict, dEN R-EE strict, d no-image R-BB0 strict, d fidelity R-BB, d Tier-1 rate and d generic fallback on R-TEST naming items under R-BB, d answer length in words; per seed and on seed-averaged item scores; fewer than 3 seeds is a warning, the seeds that ran are reported | Design 7.3 names the readouts only |
| D238 | 2026-10-18 | File contracts for later chats. `human/HT-4/ht4_labels.csv`: sample_id, model_key, condition_id, qa_id, house_id, scorer_label, checker1_label, checker2_label, consensus_label; the HT-4 chat reads its pass rule from `stats.ht4_stats`. `human/HT-5/ht5_audit.csv`: audit_id, model_key, condition_id, qa_id, house_id, stratum (C, english_copy, sibling; C rows are N-BB answers of budget category C), verdict (gold_wrong, missing_variant, not_visible, ambiguous, model_wrong). `human/HT-5/lexicon_sensitivity.csv`: row_id, source (HT-5, D192), op (add_term, remove_term, item_any_class), target (lexicon, room, material, colour, number), target_id, tier, lang, form, note; written by people in the HT-5 chat, D192 rows included. `results/stats_inputs/token_counts.csv`: qa_id, model_key, tokens_bn, tokens_en, counter, counted_utc | One implementation, and later chats write what this code reads |
| D239 | 2026-10-18 | Secondary details: the metric check of J1 with our own BERTScore code; box format from the data and box area only for single-box close-up items; the cultural stratum compares culturally_specific with globally_common (N-BE on en_unique items only); per-annotator results with their own house bootstrap; refusal and invalid shares beside every no-image and forced-choice accuracy; mixed-script counts per cell; token counts with the two pinned tokenizers and Gemini countTokens, no ledger line because it is not billed | Design 9.4 items 4, 6, 8, 9, 10, 13; F11; B5 |
| D240 | 2026-10-18 | The v1 decisions D47 to D56 are signed off by the team lead, as built into the frozen ASSET-4A and ASSET-5 scripts. Their text stays in the PROJECT_STATE of 2026-09-10 | They were the last open sign-off in PROJECT_STATE; the scripts ran and their outputs are frozen |
| D241 | 2026-10-18 | DESIGN_ADDENDUM.md is merged into the design and this guide once, before the gate-E commit (D101). Each "Exact design edit" is a find-and-replace that matched exactly once, applied in decision order; where two edits change the same text, the later decision wins. Sections A and B (stage 1) and the D1 rows, which have no such block, are applied from their rows (A1 to A3, B1, B4, D1); the B1 and B6 rows for §8.5 are merged into §8.5 as F9 asks. The other value tables, the as-run notes and the file notes stay in the addendum as the record. Design §6.2 is unchanged byte for byte. The version stays 2.1, with one §18 row. The addendum's decision rows are added to Part 8 | D101; one plan file for the pre-specification, and no text changed that no decision row names |
| D242 | 2026-10-18 | The gate-E commit adds the merged design and guide to repo `docs/`, with the same names and bytes as in Drive `docs/`, on top of the pushed tree that already holds `prompts.yaml`, the scorer and its tests, the analysis code (D230) and the probe id lists. A one-off Colab cell, not kept in the repo, checks every input (PASS or FAIL), runs dry first, then commits, tags `gate-E` (the same tag type as `gate-A`), pushes, checks the remote, and copies the two files to Drive `docs/`, keeping the old copies in `docs/old/` | Design §9.7 and addendum J5; the analysis files were pushed before the merge, so the tag carries them; one copy of the plan in Drive and in the repo |
| D243 | 2026-10-18 | Six places the merge left out of date are corrected with it: §12.2 (57 units is a little over half a Colab Pro month); §15 R5 (the replacement rule of §5.5, D162); §7.2 and §7.4 (P-FCE 454, 446 scored); §7.4 translation first pass (653 strings, 34 calls); §11 (Gates A, B, G and H marked passed with their dates); §4.2 `name_id` (how it is made, C4) | A merged decision or a finished run made each one wrong; the pre-specification should not carry known wrong text |
| D244 | 2026-10-18 | From the tag `gate-E`, the design and this guide in repo `docs/` are the pre-specification and are not edited. Later decisions go, as before, into new DESIGN_ADDENDUM sections (K onward) with decision rows; one that changes something the gate-E design fixes is a deviation and is reported in the Methods with its reason. Sections A to J stay in the addendum as the record of how each asset ran | The pre-specified plan stays fixed, and later chats keep one place to record changes |

---

# PART 9: HOW TO USE THIS FILE

*Addressed to a future AI session picking up this project.*

**Source of truth, in order:** `BanglaIndoorVQA_Design.md` v2.1, then this handbook, then
`PROJECT_STATE.md` for live status only. The v1.0 design, the v1.0 guide and any older plan are
retired.

**Assume, without re-asking:**
- The objective and the Frontiers target are locked (D62). Do not propose a new objective.
- The team exists and the team lead assigns people. Never assign tasks to named people. Give
  people counts only.
- No dates. Give effort in hours or person-days.
- Compute and budget are as in Design §12.

**Ask before acting when:**
- a task would spend API money or GPU hours beyond Design §12;
- a task would change a frozen asset (lexicon, probes, prompts, scorer) after its gate;
- a request conflicts with a row in Part 8: cite the row and let the team decide;
- a figure, table, model or condition would be added or removed.

**Never invent:**
- dataset numbers: use the design file or ask;
- results: write `[pending]`, never a placeholder number, not even as an example;
- citations: use only Part 5.5 or papers the team supplies, and confirm every "check" item first;
- prices and model IDs: verify them in the week of the run.

**Never do:**
- propose names or variants for the lexicon (an AI may format files or check spelling only);
- look at any model output while the lexicon is being built;
- use the Gemini free tier;
- mix platforms within one model's runs;
- commit any data-derived text file (questions, answers, model outputs, human sheets) to GitHub,
  even while the repo is private.

**Which Claude model to use, by task:**

| Task | Model and setting |
|---|---|
| Writing the harness, the scorer, ASSET-5C and the fine-tuning scripts | Claude Code, or Opus 5.5 at high effort |
| Small code fixes, file reformatting, proofreading | Sonnet 5.5 |
| Debugging a failing run or strange numbers | Opus 5.5 at high effort |
| Statistics and interpreting results | Opus 5.5 at max effort |
| Drafting manuscript sections | Opus 5.5 at high effort, with the style rules in Part 7.4 |
| Reviewer response | Opus 5.5 at max effort |
| Final novelty check before submission | Opus 5.5 with Research mode on |
| Quick fact checks (a price, a model ID) | any model with web search on |

**Where to look first:**

| Question | Look here |
|---|---|
| What exactly do we run? | Design §7 |
| How is an answer scored? | Design §8 |
| What does a result mean? | Design §1.4; this file, Part 3 |
| Why was something decided? | this file, Parts 4 and 8 |
| What can we claim? | Design §2; this file, Part 5 |
| What will reviewers ask? | this file, Part 6 |
| What do we write next? | this file, Part 7 |
| What is the live status? | `PROJECT_STATE.md` |

**When a decision is made,** append it to Part 8 in the same session, and add a line to Design §18
if it changes the plan. This file stays the source of truth only if it is kept up to date.
