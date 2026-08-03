fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/1/positions_on | quill | docs_per_second (docs/s) | 15946.204885 | [15763.973879, 16139.849623] | 16732.175308 | 17366.606726 | 4.200 | 80 | sampled
bulk/medium/1/positions_on | tantivy | docs_per_second (docs/s) | 72411.697299 | [67817.666565, 76434.314407] | 96671.974100 | 99581.498397 | 20.673 | 80 | sampled
bulk/medium/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.221193 | [0.207113, 0.234114] | 0.332708 | 0.376177 | 22.086 | 80 | sampled
bulk/medium/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.940817 | [0.887635, 0.985714] | 1.251238 | 1.545138 | 24.046 | 80 | sampled
