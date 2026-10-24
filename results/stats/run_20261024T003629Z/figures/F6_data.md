# F6 data (table view): fine-tuned minus base; one column per seed after the CI

| what | readout | unit | estimate [95% CI] | qwen_ft_s42 | qwen_ft_s43 | qwen_ft_s44 |
|---|---|---|---|---|---|---|
| mean of 3 seeds | dBN (R-BB strict) | points | 50.1 [46.5, 53.6] | 50.8 | 49.4 | 50.0 |
| mean of 3 seeds | dEN (R-EE strict) | points | 10.4 [8.6, 12.4] | 11.2 | 9.6 | 10.3 |
| mean of 3 seeds | d no-image (R-BB0 strict) | points | 35.1 [29.0, 40.5] | 34.8 | 35.1 | 35.2 |
| mean of 3 seeds | d fidelity (R-BB) | points | 0.3 [0.1, 0.5] | 0.3 | 0.3 | 0.3 |
| mean of 3 seeds | d answer length (R-BB) | words | 0.00 [-0.05, 0.06] | 0.00 | 0.00 | 0.00 |
| mean of 3 seeds | d Tier-1 rate (R-BB naming) | points | 61.6 [54.7, 67.5] | 62.5 | 60.3 | 61.9 |
| mean of 3 seeds | d generic fallback (R-BB naming) | points | -23.6 [-37.9, -12.3] | -23.4 | -23.6 | -23.8 |
| P8 | dBN minus dEN | points | 39.7 [36.5, 43.1] | 39.6 | 39.8 | 39.7 |
