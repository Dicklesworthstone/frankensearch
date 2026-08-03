fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/8/positions_on | quill | docs_per_second (docs/s) | 20400.856782 | [20001.254682, 20594.814877] | 21593.633966 | 21791.546352 | 3.848 | 30 | sampled
bulk/medium/8/positions_on | tantivy | docs_per_second (docs/s) | 47555.318011 | [44518.043078, 50020.601063] | 62212.431047 | 65957.408971 | 21.266 | 30 | sampled
bulk/medium/8/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.431254 | [0.394635, 0.455288] | 0.815308 | 0.893793 | 28.265 | 30 | sampled
bulk/medium/8/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.984157 | [0.858930, 1.117099] | 1.820695 | 2.484896 | 35.646 | 30 | sampled
