fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/1/positions_on | quill | docs_per_second (docs/s) | 60252.069806 | [59397.928592, 60553.808954] | 61260.247052 | 61260.247052 | 1.348 | 10 | sampled
bulk/medium/1/positions_on | tantivy | docs_per_second (docs/s) | 137593.034380 | [131349.773469, 141352.898487] | 144531.875405 | 144531.875405 | 3.597 | 10 | sampled
bulk/medium/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.437437 | [0.428744, 0.446829] | 0.461783 | 0.461783 | 2.747 | 10 | sampled
bulk/medium/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.980624 | [0.948096, 1.006134] | 1.037212 | 1.037212 | 3.882 | 10 | sampled
