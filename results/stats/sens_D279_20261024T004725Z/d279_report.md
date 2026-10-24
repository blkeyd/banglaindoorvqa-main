# D279 scorer sensitivity (S11)

script analysis-2026-10-24 (pass analysis-d279-2026-10-24), stats stats-2026-10-17, scorer 0.2.2-stage2, prompt exp-v1, UTC 2026-10-24T00:47:34Z, mode real
deviation: D290: D279 sensitivity pass (sens-d279) added to scripts/analysis.py after the tag gate-E; the primary run is the unchanged gate-E code
bootstrap: 10000 resamples, seed 42, the same house resamples as the primary run
primary run: run_20261024T003629Z; 333 quantities recomputed on the primary records, differences 0 (last digits only 0); the primary columns show the values the primary run reported

## Change file
304 answers changed (rule a 294, rule b 10) in 37 model-condition cells; every row equals its primary record before the change; label, strict, lenient and content_lang_ok are taken from the file, every other field (fidelity, refusal, named objects) stays as scored

## Contrasts, percentage points [95% CI]; primary, then D279 rules a and b
Gemini 3.6 Flash P1: -0.5 [-2.1, 1.0]; D279 -0.4 [-2.0, 1.2]; change +0.14; Holm -, -; CI excludes 0: no / no
Gemini 3.6 Flash P2: 4.1 [2.0, 6.3]; D279 4.3 [2.1, 6.5]; change +0.14; Holm -, -; CI excludes 0: yes / yes
Gemini 3.6 Flash P12: 4.7 [2.3, 7.0]; D279 4.7 [2.3, 7.0]; change +0.00; Holm 0.0012, 0.0012; CI excludes 0: yes / yes
Gemini 3.6 Flash P3: -1.7 [-5.0, 1.4]; D279 -1.4 [-4.9, 1.9]; change +0.27; Holm 0.3066, 0.4020; CI excludes 0: no / no
Gemini 3.6 Flash P4: -1.7 [-2.5, -1.0]; D279 -1.7 [-2.5, -1.0]; change +0.00; Holm 0.0012, 0.0012; CI excludes 0: yes / yes
Gemini 3.6 Flash P5: 3.0 [1.3, 4.6]; D279 2.9 [1.3, 4.4]; change -0.06; Holm 0.0024, 0.0024; CI excludes 0: yes / yes
Gemini 3.6 Flash P6: 4.0 [0.8, 7.3]; D279 4.4 [1.4, 7.5]; change +0.40; Holm 0.0340, 0.0144; CI excludes 0: yes / yes
Gemini 3.6 Flash P7: -16.7 [-22.6, -10.8]; D279 -16.9 [-22.7, -11.1]; change -0.21; Holm 0.0012, 0.0012; CI excludes 0: yes / yes
Qwen3.5-4B P1: 4.7 [3.0, 6.4]; D279 4.5 [2.8, 6.3]; change -0.18; Holm -, -; CI excludes 0: yes / yes
Qwen3.5-4B P2: 28.0 [25.3, 30.8]; D279 28.0 [25.3, 30.6]; change -0.09; Holm -, -; CI excludes 0: yes / yes
Qwen3.5-4B P12: 23.4 [19.8, 27.0]; D279 23.4 [19.9, 27.0]; change +0.09; Holm < 0.0012, < 0.0012; CI excludes 0: yes / yes
Qwen3.5-4B P3: 13.3 [9.9, 16.7]; D279 13.2 [10.0, 16.5]; change -0.09; Holm < 0.0012, < 0.0012; CI excludes 0: yes / yes
Qwen3.5-4B P4: 0.0 [-0.5, 0.5]; D279 0.0 [-0.5, 0.5]; change +0.00; Holm 1.0000, 1.0000; CI excludes 0: no / no
Qwen3.5-4B P5: 12.0 [7.2, 17.4]; D279 12.3 [7.5, 17.6]; change +0.26; Holm < 0.0012, < 0.0012; CI excludes 0: yes / yes
Qwen3.5-4B P6: -4.0 [-5.5, -2.7]; D279 -3.8 [-5.3, -2.5]; change +0.20; Holm < 0.0012, < 0.0012; CI excludes 0: yes / yes
Qwen3.5-4B P7: -1.2 [-10.9, 8.1]; D279 -0.1 [-9.6, 9.1]; change +1.09; Holm 1.0000, 1.0000; CI excludes 0: no / no
Gemma 4 E4B P1: 0.2 [-0.9, 1.3]; D279 0.3 [-0.9, 1.4]; change +0.09; Holm -, -; CI excludes 0: no / no
Gemma 4 E4B P2: 4.5 [2.3, 6.7]; D279 4.5 [2.3, 6.7]; change +0.00; Holm -, -; CI excludes 0: yes / yes
Gemma 4 E4B P12: 4.3 [1.6, 7.1]; D279 4.2 [1.5, 7.0]; change -0.09; Holm 0.0078, 0.0048; CI excludes 0: yes / yes
Gemma 4 E4B P3: -6.4 [-9.1, -3.6]; D279 -6.2 [-9.1, -3.3]; change +0.18; Holm < 0.0012, < 0.0012; CI excludes 0: yes / yes
Gemma 4 E4B P4: 0.2 [0.0, 0.5]; D279 0.2 [0.0, 0.5]; change +0.00; Holm 0.2642, 0.2642; CI excludes 0: no / no
Gemma 4 E4B P5: 5.4 [2.2, 8.7]; D279 5.2 [2.0, 8.6]; change -0.20; Holm 0.0024, 0.0048; CI excludes 0: yes / yes
Gemma 4 E4B P6: 2.8 [1.0, 4.8]; D279 3.1 [1.3, 5.0]; change +0.30; Holm 0.0078, 0.0048; CI excludes 0: yes / yes
Gemma 4 E4B P7: -16.6 [-24.9, -8.4]; D279 -16.4 [-24.8, -8.0]; change +0.23; Holm < 0.0012, < 0.0012; CI excludes 0: yes / yes
Qwen3.5-4B fine-tuned (3 seeds) P8: 39.7 [36.5, 43.1]; D279 39.3 [36.0, 42.8]; change -0.35; Holm -, -; CI excludes 0: yes / yes
Gemini 3.6 Flash error budget A share (F3): 29.1 [22.3, 35.3]; D279 28.9 [22.0, 35.3]; change -0.16; Holm -, -; CI excludes 0: -
Qwen3.5-4B error budget A share (F3): 23.5 [18.8, 27.9]; D279 23.5 [18.8, 27.9]; change +0.00; Holm -, -; CI excludes 0: -
Gemma 4 E4B error budget A share (F3): 20.8 [15.7, 26.2]; D279 21.1 [15.9, 26.4]; change +0.30; Holm -, -; CI excludes 0: -
Gemini 3.6 Flash N-FCB accuracy (P-CLUST, n 483): 90.3 [86.8, 93.2]; D279 90.1 [86.5, 93.0]; change -0.21; Holm -, -; CI excludes 0: -
Gemini 3.6 Flash N-FCB0 accuracy (P-CLUST, n 483): 9.1 [6.5, 12.1]; D279 8.7 [6.1, 11.7]; change -0.41; Holm -, -; CI excludes 0: -
Gemini 3.6 Flash N-FCE0 accuracy (P-FCE, n 446): 20.0 [16.7, 23.2]; D279 19.5 [16.2, 22.9]; change -0.45; Holm -, -; CI excludes 0: -

## Readings that change between the two scorings
none: every CI keeps its side of 0 and every Holm p keeps its side of 0.05

