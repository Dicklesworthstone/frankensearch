fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/1/positions_on | quill | docs_per_second (docs/s) | 88051.545002 | [86344.894940, 88934.876728] | 90967.889079 | 91916.319633 | 4.022 | 30 | sampled
bulk/medium/1/positions_on | tantivy | docs_per_second (docs/s) | 163602.209047 | [159398.258727, 210883.036408] | 250211.010451 | 251181.337227 | 20.078 | 30 | sampled
bulk/medium/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.528406 | [0.404097, 0.544508] | 0.603668 | 0.640601 | 18.492 | 30 | sampled
bulk/medium/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.997408 | [0.964552, 1.072001] | 1.535901 | 1.569680 | 26.412 | 30 | sampled
