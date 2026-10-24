# S11. Scorer sensitivity (D279): every primary contrast, the error-budget A share and the forced-choice cells rule b changes, under the frozen scorer and under rules a and b

| model | quantity | primary pp [95% CI] | D279 rules a and b pp [95% CI] | change pp | p primary | p D279 | Holm p primary | Holm p D279 | CI excludes 0 (primary / D279) | note |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemini 3.6 Flash | P1 | -0.5 [-2.1, 1.0] | -0.4 [-2.0, 1.2] | +0.14 |  |  |  |  | no / no | estimate, no test |
| Gemini 3.6 Flash | P2 | 4.1 [2.0, 6.3] | 4.3 [2.1, 6.5] | +0.14 |  |  |  |  | yes / yes | estimate, no test |
| Gemini 3.6 Flash | P12 | 4.7 [2.3, 7.0] | 4.7 [2.3, 7.0] | +0.00 | 0.0002 | 0.0002 | 0.0012 | 0.0012 | yes / yes | Holm within the model over P12, P3 to P7 |
| Gemini 3.6 Flash | P3 | -1.7 [-5.0, 1.4] | -1.4 [-4.9, 1.9] | +0.27 | 0.3066 | 0.4020 | 0.3066 | 0.4020 | no / no | Holm within the model over P12, P3 to P7 |
| Gemini 3.6 Flash | P4 | -1.7 [-2.5, -1.0] | -1.7 [-2.5, -1.0] | +0.00 | < 0.0002 | < 0.0002 | 0.0012 | 0.0012 | yes / yes | Holm within the model over P12, P3 to P7; fidelity is not in changes.csv and stays as scored |
| Gemini 3.6 Flash | P5 | 3.0 [1.3, 4.6] | 2.9 [1.3, 4.4] | -0.06 | 0.0008 | 0.0008 | 0.0024 | 0.0024 | yes / yes | Holm within the model over P12, P3 to P7 |
| Gemini 3.6 Flash | P6 | 4.0 [0.8, 7.3] | 4.4 [1.4, 7.5] | +0.40 | 0.0170 | 0.0072 | 0.0340 | 0.0144 | yes / yes | Holm within the model over P12, P3 to P7 |
| Gemini 3.6 Flash | P7 | -16.7 [-22.6, -10.8] | -16.9 [-22.7, -11.1] | -0.21 | < 0.0002 | < 0.0002 | 0.0012 | 0.0012 | yes / yes | Holm within the model over P12, P3 to P7 |
| Qwen3.5-4B | P1 | 4.7 [3.0, 6.4] | 4.5 [2.8, 6.3] | -0.18 |  |  |  |  | yes / yes | estimate, no test |
| Qwen3.5-4B | P2 | 28.0 [25.3, 30.8] | 28.0 [25.3, 30.6] | -0.09 |  |  |  |  | yes / yes | estimate, no test |
| Qwen3.5-4B | P12 | 23.4 [19.8, 27.0] | 23.4 [19.9, 27.0] | +0.09 | < 0.0002 | < 0.0002 | < 0.0012 | < 0.0012 | yes / yes | Holm within the model over P12, P3 to P7 |
| Qwen3.5-4B | P3 | 13.3 [9.9, 16.7] | 13.2 [10.0, 16.5] | -0.09 | < 0.0002 | < 0.0002 | < 0.0012 | < 0.0012 | yes / yes | Holm within the model over P12, P3 to P7 |
| Qwen3.5-4B | P4 | 0.0 [-0.5, 0.5] | 0.0 [-0.5, 0.5] | +0.00 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | no / no | Holm within the model over P12, P3 to P7; fidelity is not in changes.csv and stays as scored |
| Qwen3.5-4B | P5 | 12.0 [7.2, 17.4] | 12.3 [7.5, 17.6] | +0.26 | < 0.0002 | < 0.0002 | < 0.0012 | < 0.0012 | yes / yes | Holm within the model over P12, P3 to P7 |
| Qwen3.5-4B | P6 | -4.0 [-5.5, -2.7] | -3.8 [-5.3, -2.5] | +0.20 | < 0.0002 | < 0.0002 | < 0.0012 | < 0.0012 | yes / yes | Holm within the model over P12, P3 to P7 |
| Qwen3.5-4B | P7 | -1.2 [-10.9, 8.1] | -0.1 [-9.6, 9.1] | +1.09 | 0.8158 | 0.9894 | 1.0000 | 1.0000 | no / no | Holm within the model over P12, P3 to P7 |
| Gemma 4 E4B | P1 | 0.2 [-0.9, 1.3] | 0.3 [-0.9, 1.4] | +0.09 |  |  |  |  | no / no | estimate, no test |
| Gemma 4 E4B | P2 | 4.5 [2.3, 6.7] | 4.5 [2.3, 6.7] | +0.00 |  |  |  |  | yes / yes | estimate, no test |
| Gemma 4 E4B | P12 | 4.3 [1.6, 7.1] | 4.2 [1.5, 7.0] | -0.09 | 0.0026 | 0.0022 | 0.0078 | 0.0048 | yes / yes | Holm within the model over P12, P3 to P7 |
| Gemma 4 E4B | P3 | -6.4 [-9.1, -3.6] | -6.2 [-9.1, -3.3] | +0.18 | < 0.0002 | < 0.0002 | < 0.0012 | < 0.0012 | yes / yes | Holm within the model over P12, P3 to P7 |
| Gemma 4 E4B | P4 | 0.2 [0.0, 0.5] | 0.2 [0.0, 0.5] | +0.00 | 0.2642 | 0.2642 | 0.2642 | 0.2642 | no / no | Holm within the model over P12, P3 to P7; fidelity is not in changes.csv and stays as scored |
| Gemma 4 E4B | P5 | 5.4 [2.2, 8.7] | 5.2 [2.0, 8.6] | -0.20 | 0.0006 | 0.0012 | 0.0024 | 0.0048 | yes / yes | Holm within the model over P12, P3 to P7 |
| Gemma 4 E4B | P6 | 2.8 [1.0, 4.8] | 3.1 [1.3, 5.0] | +0.30 | 0.0030 | 0.0012 | 0.0078 | 0.0048 | yes / yes | Holm within the model over P12, P3 to P7 |
| Gemma 4 E4B | P7 | -16.6 [-24.9, -8.4] | -16.4 [-24.8, -8.0] | +0.23 | < 0.0002 | < 0.0002 | < 0.0012 | < 0.0012 | yes / yes | Holm within the model over P12, P3 to P7 |
| Qwen3.5-4B fine-tuned (3 seeds) | P8 | 39.7 [36.5, 43.1] | 39.3 [36.0, 42.8] | -0.35 | < 0.0002 | < 0.0002 |  |  | yes / yes | single test (Design 9.1) |
| Gemini 3.6 Flash | error budget A share (F3) | 29.1 [22.3, 35.3] | 28.9 [22.0, 35.3] | -0.16 |  |  |  |  |  | denominator 258 / 256; primary A1 1, A2 11, A3 6, A4 57, B 155, C 28; D279 A1 1, A2 11, A3 6, A4 56, B 153, C 29 |
| Qwen3.5-4B | error budget A share (F3) | 23.5 [18.8, 27.9] | 23.5 [18.8, 27.9] | +0.00 |  |  |  |  |  | denominator 460 / 460; primary A1 0, A2 13, A3 5, A4 90, B 165, C 187; D279 A1 0, A2 13, A3 5, A4 90, B 165, C 187 |
| Gemma 4 E4B | error budget A share (F3) | 20.8 [15.7, 26.2] | 21.1 [15.9, 26.4] | +0.30 |  |  |  |  |  | denominator 337 / 337; primary A1 0, A2 9, A3 7, A4 54, B 199, C 68; D279 A1 0, A2 9, A3 7, A4 55, B 198, C 68 |
| Gemini 3.6 Flash | N-FCB accuracy (P-CLUST, n 483) | 90.3 [86.8, 93.2] | 90.1 [86.5, 93.0] | -0.21 |  |  |  |  |  | 1 answers made invalid by rule b; invalid share 0.0 [0.0, 0.0] to 0.2 [0.0, 0.7]; refusal share 0.0 [0.0, 0.0] (as scored) |
| Gemini 3.6 Flash | N-FCB0 accuracy (P-CLUST, n 483) | 9.1 [6.5, 12.1] | 8.7 [6.1, 11.7] | -0.41 |  |  |  |  |  | 6 answers made invalid by rule b; invalid share 70.8 [65.5, 75.9] to 72.0 [66.5, 77.0]; refusal share 66.7 [61.5, 71.3] (as scored) |
| Gemini 3.6 Flash | N-FCE0 accuracy (P-FCE, n 446) | 20.0 [16.7, 23.2] | 19.5 [16.2, 22.9] | -0.45 |  |  |  |  |  | 3 answers made invalid by rule b; invalid share 30.3 [26.1, 34.3] to 30.9 [26.6, 35.1]; refusal share 28.9 [24.8, 32.9] (as scored) |
