fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/1/positions_on | quill | docs_per_second (docs/s) | 59293.746849 | [54414.953167, 60126.045958] | 61247.379617 | 61247.379617 | 5.176 | 10 | sampled
bulk/medium/1/positions_on | tantivy | docs_per_second (docs/s) | 133400.610490 | [127784.301321, 144020.860125] | 152945.401434 | 152945.401434 | 6.914 | 10 | sampled
bulk/medium/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.425489 | [0.407935, 0.445484] | 0.456580 | 0.456580 | 4.952 | 10 | sampled
bulk/medium/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.981077 | [0.946503, 1.004529] | 1.032524 | 1.032524 | 3.650 | 10 | sampled
