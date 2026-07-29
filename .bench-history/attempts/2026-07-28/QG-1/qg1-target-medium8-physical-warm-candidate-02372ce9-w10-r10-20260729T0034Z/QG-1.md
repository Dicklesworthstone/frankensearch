fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/8/positions_on | quill | docs_per_second (docs/s) | 59853.788436 | [57499.184055, 60427.127424] | 60991.106481 | 60991.106481 | 2.332 | 10 | sampled
bulk/medium/8/positions_on | tantivy | docs_per_second (docs/s) | 147560.662148 | [141846.518430, 150603.460619] | 155194.466178 | 155194.466178 | 3.439 | 10 | sampled
bulk/medium/8/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.404421 | [0.383443, 0.427316] | 0.431599 | 0.431599 | 5.051 | 10 | sampled
bulk/medium/8/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.965987 | [0.876011, 1.017018] | 1.053838 | 1.053838 | 11.303 | 10 | sampled
