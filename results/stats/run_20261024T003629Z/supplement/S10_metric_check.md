# S10. Metric check on N-BB answers against the tiers (a known finding replicated)

| model | statistic | group | n | chrF++ | BERTScore F1 |
|---|---|---|---|---|---|
| Gemini 3.6 Flash | mean | tier1_req | 495 | 88.4 | 0.958 |
| Gemini 3.6 Flash | mean | tier1_other | 1 | 13.1 | 0.777 |
| Gemini 3.6 Flash | mean | tier2 | 31 | 42.3 | 0.880 |
| Gemini 3.6 Flash | mean | sibling | 58 | 6.6 | 0.759 |
| Gemini 3.6 Flash | mean | ambiguous | 31 | 43.5 | 0.805 |
| Gemini 3.6 Flash | mean | english_copy | 90 | 32.3 | 0.799 |
| Gemini 3.6 Flash | mean | other_wrong | 285 | 13.1 | 0.773 |
| Gemini 3.6 Flash | AUC | tier1_vs_tier2 | 495 / 31 | 0.930 | 0.809 |
| Gemini 3.6 Flash | AUC | tier1_vs_sibling | 495 / 58 | 0.984 | 0.967 |
| Qwen3.5-4B | mean | tier1_req | 157 | 81.5 | 0.944 |
| Qwen3.5-4B | mean | tier1_other | 0 | - | - |
| Qwen3.5-4B | mean | tier2 | 26 | 43.7 | 0.865 |
| Qwen3.5-4B | mean | sibling | 30 | 6.1 | 0.775 |
| Qwen3.5-4B | mean | ambiguous | 11 | 24.0 | 0.788 |
| Qwen3.5-4B | mean | english_copy | 39 | 21.3 | 0.819 |
| Qwen3.5-4B | mean | other_wrong | 728 | 8.2 | 0.765 |
| Qwen3.5-4B | AUC | tier1_vs_tier2 | 157 / 26 | 0.828 | 0.794 |
| Qwen3.5-4B | AUC | tier1_vs_sibling | 157 / 30 | 0.996 | 0.941 |
| Gemma 4 E4B | mean | tier1_req | 371 | 88.3 | 0.964 |
| Gemma 4 E4B | mean | tier1_other | 0 | - | - |
| Gemma 4 E4B | mean | tier2 | 43 | 35.4 | 0.855 |
| Gemma 4 E4B | mean | sibling | 85 | 7.4 | 0.765 |
| Gemma 4 E4B | mean | ambiguous | 15 | 34.7 | 0.788 |
| Gemma 4 E4B | mean | english_copy | 36 | 14.6 | 0.788 |
| Gemma 4 E4B | mean | other_wrong | 441 | 7.8 | 0.758 |
| Gemma 4 E4B | AUC | tier1_vs_tier2 | 371 / 43 | 0.925 | 0.861 |
| Gemma 4 E4B | AUC | tier1_vs_sibling | 371 / 85 | 0.989 | 0.968 |
