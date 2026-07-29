fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/8/positions_on | quill | docs_per_second (docs/s) | 58242.222045 | [57647.880173, 60167.682301] | 60801.668491 | 60801.668491 | 2.782 | 10 | sampled
bulk/medium/8/positions_on | tantivy | docs_per_second (docs/s) | 144878.945878 | [141765.190402, 146146.110550] | 147141.534816 | 147141.534816 | 2.029 | 10 | sampled
bulk/medium/8/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.401924 | [0.396181, 0.413854] | 0.441984 | 0.441984 | 3.898 | 10 | sampled
bulk/medium/8/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.198710 | [1.012705, 1.315138] | 1.418015 | 1.418015 | 17.571 | 10 | sampled
