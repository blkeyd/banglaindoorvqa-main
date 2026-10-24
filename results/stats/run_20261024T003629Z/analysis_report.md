# Analysis report

script analysis-2026-10-17, stats stats-2026-10-17, scorer 0.2.2-stage2, prompt exp-v1, UTC 2026-10-24T00:42:55Z, mode real
bootstrap: 10000 resamples, seed 42; houses EXP-L 68, EXP-N 61, P7 pooled 68, EXP-R 15

## EXP-L (P-CORE-CLOSED, 1109 scored items)
B0 strict: 38.2 [34.7, 41.8] (template 920, type fallback 189)
Gemini 3.6 Flash cells: L-BB 67.9 [64.8, 71.0]; L-BE 72.9 [69.9, 75.9]; L-EB 68.3 [64.8, 71.6]; L-EE 71.5 [68.6, 74.5]; L-BB0 12.6 [10.3, 15.0] (refusal 45.9); L-EE0 17.9 [15.1, 21.0] (refusal 30.6)
Qwen3.5-4B cells: L-BB 33.4 [31.0, 35.7]; L-BE 61.8 [58.4, 65.0]; L-EB 38.4 [35.8, 40.9]; L-EE 66.1 [63.1, 69.1]; L-BB0 7.9 [6.3, 9.8] (refusal 40.9); L-EE0 27.4 [23.8, 31.1] (refusal 0.0)
Gemma 4 E4B cells: L-BB 53.1 [50.0, 56.4]; L-BE 57.3 [54.1, 60.5]; L-EB 52.9 [49.6, 56.4]; L-EE 57.8 [54.4, 61.3]; L-BB0 8.5 [6.8, 10.2] (refusal 52.6); L-EE0 19.6 [16.7, 22.6] (refusal 24.3)

## Contrasts (T4)
Gemini 3.6 Flash P1: -0.5 [-2.1, 1.0] pp, p -, Holm -, GEE direction agrees
Gemini 3.6 Flash P2: 4.1 [2.0, 6.3] pp, p -, Holm -, GEE direction agrees
Gemini 3.6 Flash P12: 4.7 [2.3, 7.0] pp, p 0.0002, Holm 0.0012
Gemini 3.6 Flash P3: -1.7 [-5.0, 1.4] pp, p 0.3066, Holm 0.3066
Gemini 3.6 Flash P4: -1.7 [-2.5, -1.0] pp, p < 0.0002, Holm 0.0012
Gemini 3.6 Flash P5: 3.0 [1.3, 4.6] pp, p 0.0008, Holm 0.0024
Gemini 3.6 Flash P6: 4.0 [0.8, 7.3] pp, p 0.0170, Holm 0.0340
Gemini 3.6 Flash P7: -16.7 [-22.6, -10.8] pp, p < 0.0002, Holm 0.0012, GEE direction agrees
Gemini 3.6 Flash information: P5 any language 3.0 [1.3, 4.6]; P6 known in English 4.7 [0.0, 9.7] (n 511), not known 5.3 [-0.4, 10.3] (n 449)
Qwen3.5-4B P1: 4.7 [3.0, 6.4] pp, p -, Holm -, GEE direction agrees
Qwen3.5-4B P2: 28.0 [25.3, 30.8] pp, p -, Holm -, GEE direction agrees
Qwen3.5-4B P12: 23.4 [19.8, 27.0] pp, p < 0.0002, Holm < 0.0012
Qwen3.5-4B P3: 13.3 [9.9, 16.7] pp, p < 0.0002, Holm < 0.0012
Qwen3.5-4B P4: 0.0 [-0.5, 0.5] pp, p 1.0000, Holm 1.0000
Qwen3.5-4B P5: 12.0 [7.2, 17.4] pp, p < 0.0002, Holm < 0.0012
Qwen3.5-4B P6: -4.0 [-5.5, -2.7] pp, p < 0.0002, Holm < 0.0012
Qwen3.5-4B P7: -1.2 [-10.9, 8.1] pp, p 0.8158, Holm 1.0000, GEE direction disagrees
Qwen3.5-4B information: P5 any language 12.0 [7.2, 17.4]; P6 known in English -8.8 [-12.1, -5.9] (n 407), not known -0.7 [-1.8, 0.2] (n 553)
Gemma 4 E4B P1: 0.2 [-0.9, 1.3] pp, p -, Holm -, GEE direction agrees
Gemma 4 E4B P2: 4.5 [2.3, 6.7] pp, p -, Holm -, GEE direction agrees
Gemma 4 E4B P12: 4.3 [1.6, 7.1] pp, p 0.0026, Holm 0.0078
Gemma 4 E4B P3: -6.4 [-9.1, -3.6] pp, p < 0.0002, Holm < 0.0012
Gemma 4 E4B P4: 0.2 [0.0, 0.5] pp, p 0.2642, Holm 0.2642
Gemma 4 E4B P5: 5.4 [2.2, 8.7] pp, p 0.0006, Holm 0.0024
Gemma 4 E4B P6: 2.8 [1.0, 4.8] pp, p 0.0030, Holm 0.0078
Gemma 4 E4B P7: -16.6 [-24.9, -8.4] pp, p < 0.0002, Holm < 0.0012, GEE direction agrees
Gemma 4 E4B information: P5 any language 5.4 [2.2, 8.7]; P6 known in English 1.3 [-2.5, 5.4] (n 389), not known 3.3 [0.6, 6.2] (n 571)

## Word-form items (P7 pool, counts recomputed through resolve_gold; C5, F1, H1)
1311 items in 68 houses; by source and en_route: L_obj_name|no 52, L_obj_name|yes 61, L_object|no 79, L_object|yes 159, N_obj_name|no 352, N_obj_name|yes 608; origin3 uncertain 0 (left out of the origin split); P-NAME and P-CORE-CLOSED overlap counted once 131

## Mixed-script answers (B5: where strict and fidelity can disagree)
Gemini 3.6 Flash: L-BB 3; L-BE 0; L-EB 5; L-EE 0; L-BB0 4; L-EE0 0; N-BB 2; N-BE 0; N-BBs 1
Qwen3.5-4B: L-BB 4; L-BE 6; L-EB 2; L-EE 0; L-BB0 3; L-EE0 0; N-BB 3; N-BE 0; N-BBs 1
Gemma 4 E4B: L-BB 3; L-BE 0; L-EB 0; L-EE 0; L-BB0 0; L-EE0 0; N-BB 3; N-BE 0; N-BBs 0

## Error budget (F3) and D204
Gemini 3.6 Flash: denominator 258 of 463; A1 0.4 [0.0, 1.3], A2 4.3 [1.5, 7.3], A3 2.3 [0.4, 4.8], A4 22.1 [15.6, 28.5], B 60.1 [54.0, 66.5], C 10.9 [6.2, 16.7]; guessing level 7.7 [4.1, 11.3] (invalid 73.2, refusal 67.2); two-way A 36.1 [29.4, 42.6]; D204 sibling labels on HT-6 pairs: N-BB 6 of 58, N-BE 13 of 132, N-BBs 4 of 78
Qwen3.5-4B: denominator 460 of 463; A1 0.0 [0.0, 0.0], A2 2.8 [1.0, 5.2], A3 1.1 [0.2, 2.1], A4 19.6 [15.5, 23.5], B 35.9 [31.3, 41.0], C 40.7 [34.9, 46.6]; guessing level 23.6 [18.6, 28.5] (invalid 0.0, refusal 0.0); two-way A 32.3 [28.3, 36.1]; D204 sibling labels on HT-6 pairs: N-BB 0 of 30, N-BE 13 of 149, N-BBs 0 of 18
Gemma 4 E4B: denominator 337 of 463; A1 0.0 [0.0, 0.0], A2 2.7 [0.9, 4.6], A3 2.1 [0.3, 4.1], A4 16.0 [11.7, 20.6], B 59.1 [53.3, 64.9], C 20.2 [15.8, 25.0]; guessing level 3.7 [1.8, 6.2] (invalid 89.5, refusal 89.5); two-way A 20.8 [17.3, 24.5]; D204 sibling labels on HT-6 pairs: N-BB 5 of 85, N-BE 15 of 142, N-BBs 10 of 115
HT-5: C answers marked gold or visibility 0.7 [0.0, 2.3]

## Forced choice (F11 shares beside every accuracy)
Gemini 3.6 Flash P-CLUST: N-FCB 90.3 [86.8, 93.2] (invalid 0.0, refusal 0.0); N-FCB0 9.1 [6.5, 12.1] (invalid 70.8, refusal 66.7)
Gemini 3.6 Flash P-FCE: N-FCE 80.3 [75.4, 85.1] (invalid 0.4, refusal 0.0); N-FCE0 20.0 [16.7, 23.2] (invalid 30.3, refusal 28.9)
Qwen3.5-4B P-CLUST: N-FCB 50.1 [45.0, 55.2] (invalid 24.4, refusal 0.0); N-FCB0 26.5 [22.4, 30.3] (invalid 0.0, refusal 0.0)
Qwen3.5-4B P-FCE: N-FCE 72.2 [67.9, 75.6] (invalid 0.0, refusal 0.0); N-FCE0 33.9 [29.1, 38.3] (invalid 0.0, refusal 0.0)
Gemma 4 E4B P-CLUST: N-FCB 73.3 [69.3, 77.0] (invalid 0.0, refusal 0.0); N-FCB0 3.7 [2.0, 5.8] (invalid 90.5, refusal 90.5)
Gemma 4 E4B P-FCE: N-FCE 72.4 [68.1, 76.2] (invalid 0.0, refusal 0.0); N-FCE0 0.2 [0.0, 0.8] (invalid 99.8, refusal 99.8)

## EXP-R (R-TEST, 1435 items, 15 houses)
seeds qwen_ft_s42, qwen_ft_s43, qwen_ft_s44
dBN (R-BB strict): 50.1 [46.5, 53.6] points (per seed 50.8, 49.4, 50.0)
dEN (R-EE strict): 10.4 [8.6, 12.4] points (per seed 11.2, 9.6, 10.3)
d no-image (R-BB0 strict): 35.1 [29.0, 40.5] points (per seed 34.8, 35.1, 35.2)
d fidelity (R-BB): 0.3 [0.1, 0.5] points (per seed 0.3, 0.3, 0.3)
d answer length (R-BB): 0.00 [-0.05, 0.06] words (per seed 0.00, 0.00, 0.00)
d Tier-1 rate (R-BB naming): 61.6 [54.7, 67.5] points (per seed 62.5, 60.3, 61.9)
d generic fallback (R-BB naming): -23.6 [-37.9, -12.3] points (per seed -23.4, -23.6, -23.8)
P8: 39.7 [36.5, 43.1], p < 0.0002
R-BB0 refusal: qwen 39.2; qwen_ft_s42 0.0; qwen_ft_s43 0.0; qwen_ft_s44 0.0

## Instrument checks (T5)
HT-2 146 of 150 faithful, Wilson 93.34 to 98.96
HT-3 six-label agreement 70.4, kappa 0.274 (primary); strict N1 68.3 (main row), N2 82.9 (D214)
Human reference on 185 en_unique NAME-REF items (22 houses), Tier-1 rate and generic fallback: human N1 69.7 [61.7, 77.9], 1.5 [0.0, 3.5]; human N2 85.9 [79.9, 91.0], 1.9 [0.0, 4.1]; human pooled 77.8 [72.8, 83.1], 1.7 [0.3, 3.3]; gemini 56.2 [46.5, 64.4], 4.6 [1.0, 9.6]; qwen 15.7 [10.9, 19.6], 21.6 [10.8, 33.3]; gemma 41.1 [33.6, 47.9], 6.2 [1.8, 12.3]
HT-4 scorer accuracy 93.5 [89.8, 97.0]

## Lexicon sensitivity (S9)
answers changed: 352
gemini P1: primary -0.5 [-2.1, 1.0], sensitivity -0.8 [-2.4, 0.7]
gemini P2: primary 4.1 [2.0, 6.3], sensitivity 2.1 [0.0, 4.1]
gemini P12: primary 4.7 [2.3, 7.0], sensitivity 2.9 [0.5, 5.3]
gemini P3: primary -1.7 [-5.0, 1.4], sensitivity -3.7 [-6.7, -0.8]
gemini P4: primary -1.7 [-2.5, -1.0], sensitivity -1.7 [-2.5, -1.0]
gemini L-BB strict: primary 67.9 [64.8, 71.0], sensitivity 70.2 [67.1, 73.2]
gemini L-BE strict: primary 72.9 [69.9, 75.9], sensitivity 72.9 [69.8, 75.8]
gemini L-EB strict: primary 68.3 [64.8, 71.6], sensitivity 70.0 [66.5, 73.3]
gemini L-EE strict: primary 71.5 [68.6, 74.5], sensitivity 71.4 [68.6, 74.4]
gemini L-BB0 strict: primary 12.6 [10.3, 15.0], sensitivity 13.0 [10.7, 15.3]
gemini L-EE0 strict: primary 17.9 [15.1, 21.0], sensitivity 17.9 [15.1, 21.0]
gemini P5: primary 3.0 [1.3, 4.6], sensitivity 2.2 [0.6, 3.8]
gemini P6: primary 4.0 [0.8, 7.3], sensitivity -0.9 [-4.1, 2.4]
gemini P7: primary -16.7 [-22.6, -10.8], sensitivity -10.7 [-16.8, -5.0]
gemini error budget A share: primary 29.1 [22.3, 35.3], sensitivity 29.0 [22.3, 35.4]
qwen P1: primary 4.7 [3.0, 6.4], sensitivity 4.8 [3.1, 6.6]
qwen P2: primary 28.0 [25.3, 30.8], sensitivity 27.4 [24.6, 30.2]
qwen P12: primary 23.4 [19.8, 27.0], sensitivity 22.5 [18.9, 26.3]
qwen P3: primary 13.3 [9.9, 16.7], sensitivity 12.8 [9.5, 16.3]
qwen P4: primary 0.0 [-0.5, 0.5], sensitivity 0.0 [-0.5, 0.5]
qwen L-BB strict: primary 33.4 [31.0, 35.7], sensitivity 33.9 [31.5, 36.3]
qwen L-BE strict: primary 61.8 [58.4, 65.0], sensitivity 61.8 [58.4, 65.0]
qwen L-EB strict: primary 38.4 [35.8, 40.9], sensitivity 39.2 [36.6, 41.8]
qwen L-EE strict: primary 66.1 [63.1, 69.1], sensitivity 66.1 [63.1, 69.1]
qwen L-BB0 strict: primary 7.9 [6.3, 9.8], sensitivity 8.0 [6.4, 9.9]
qwen L-EE0 strict: primary 27.4 [23.8, 31.1], sensitivity 27.4 [23.8, 31.1]
qwen P5: primary 12.0 [7.2, 17.4], sensitivity 10.2 [5.8, 15.3]
qwen P6: primary -4.0 [-5.5, -2.7], sensitivity -4.2 [-5.9, -2.7]
qwen P7: primary -1.2 [-10.9, 8.1], sensitivity -2.6 [-11.5, 6.0]
qwen error budget A share: primary 23.5 [18.8, 27.9], sensitivity 22.5 [18.1, 26.8]
gemma P1: primary 0.2 [-0.9, 1.3], sensitivity 0.1 [-1.1, 1.2]
gemma P2: primary 4.5 [2.3, 6.7], sensitivity 3.8 [1.6, 6.0]
gemma P12: primary 4.3 [1.6, 7.1], sensitivity 3.7 [1.0, 6.5]
gemma P3: primary -6.4 [-9.1, -3.6], sensitivity -6.9 [-9.6, -4.3]
gemma P4: primary 0.2 [0.0, 0.5], sensitivity 0.2 [0.0, 0.5]
gemma L-BB strict: primary 53.1 [50.0, 56.4], sensitivity 53.8 [50.7, 57.1]
gemma L-BE strict: primary 57.3 [54.1, 60.5], sensitivity 57.2 [54.0, 60.5]
gemma L-EB strict: primary 52.9 [49.6, 56.4], sensitivity 53.5 [50.1, 56.9]
gemma L-EE strict: primary 57.8 [54.4, 61.3], sensitivity 57.7 [54.3, 61.2]
gemma L-BB0 strict: primary 8.5 [6.8, 10.2], sensitivity 8.7 [7.1, 10.5]
gemma L-EE0 strict: primary 19.6 [16.7, 22.6], sensitivity 19.6 [16.7, 22.6]
gemma P5: primary 5.4 [2.2, 8.7], sensitivity 4.8 [1.8, 7.9]
gemma P6: primary 2.8 [1.0, 4.8], sensitivity 2.9 [0.8, 5.2]
gemma P7: primary -16.6 [-24.9, -8.4], sensitivity -19.5 [-26.9, -12.0]
gemma error budget A share: primary 20.8 [15.7, 26.2], sensitivity 20.3 [15.2, 25.5]
P8: primary 39.7 [36.5, 43.1], sensitivity 39.2 [36.0, 42.6]

## Failed stages
none

