fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/1/positions_on | quill | docs_per_second (docs/s) | 21965.813512 | [21924.211617, 22032.711351] | 22795.376214 | 22837.503961 | 2.258 | 80 | sampled
bulk/medium/1/positions_on | tantivy | docs_per_second (docs/s) | 194635.524775 | [190472.338944, 197055.496031] | 222472.939337 | 224506.962729 | 7.454 | 80 | sampled
bulk/medium/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.112106 | [0.110907, 0.114173] | 0.120722 | 0.121280 | 5.858 | 80 | sampled
bulk/medium/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.001282 | [0.992516, 1.011084] | 1.064598 | 1.310292 | 9.509 | 80 | sampled
