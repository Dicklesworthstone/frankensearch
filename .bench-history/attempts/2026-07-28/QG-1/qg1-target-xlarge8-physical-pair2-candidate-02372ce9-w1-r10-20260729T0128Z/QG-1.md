fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/8/positions_on | quill | docs_per_second (docs/s) | 36190.843002 | [36003.581545, 36375.703045] | 36493.697514 | 36493.697514 | 0.587 | 10 | sampled
bulk/xlarge/8/positions_on | tantivy | docs_per_second (docs/s) | 182879.508899 | [166462.073819, 185995.119132] | 188971.865400 | 188971.865400 | 6.404 | 10 | sampled
bulk/xlarge/8/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.197288 | [0.194559, 0.217702] | 0.235568 | 0.235568 | 6.994 | 10 | sampled
bulk/xlarge/8/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.978179 | [0.928430, 1.061322] | 1.242510 | 1.242510 | 9.935 | 10 | sampled
