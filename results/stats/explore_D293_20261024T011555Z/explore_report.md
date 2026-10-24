# Exploratory word-form bound (D293)

script explore-p7-d293-2026-10-24, stats stats-2026-10-17, scorer 0.2.2-stage2, UTC 2026-10-24T01:15:55Z; primary run run_20261024T003629Z; 10,000 resamples, seed 42, the P7 pool (1311 items, 68 houses)

The pre-specified P7 is reproduced exactly. Bound: an answer labelled English copy in the Bangla condition counts as transferred. Exploratory; H4 is read from the primary P7 only.

- Gemini 3.6 Flash: primary P7 -16.7 [-22.6, -10.8]; bound -10.9 [-17.1, -5.0], p < 0.0002, Holm 0.0012; bound transfer yes / no 77.8 [72.9, 82.5] / 88.7 [84.6, 92.7]; English copies among the items named in English 72 / 31 (yes / no)
- Qwen3.5-4B: primary P7 -1.2 [-10.9, 8.1]; bound -1.8 [-10.6, 6.9], p 0.6930, Holm 1.0000; bound transfer yes / no 39.3 [33.2, 45.1] / 41.1 [34.1, 47.8]; English copies among the items named in English 21 / 18 (yes / no)
- Gemma 4 E4B: primary P7 -16.6 [-24.9, -8.4]; bound -18.8 [-26.0, -11.7], p < 0.0002, Holm < 0.0012; bound transfer yes / no 66.7 [60.1, 72.8] / 85.5 [80.6, 89.8]; English copies among the items named in English 17 / 17 (yes / no)

Reading: Gemini 3.6 Flash (bound CI excludes 0); Qwen3.5-4B (bound CI includes 0); Gemma 4 E4B (bound CI excludes 0)
