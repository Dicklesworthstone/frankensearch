fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/8/positions_on | quill | docs_per_second (docs/s) | 59942.431956 | [59443.975346, 60364.972972] | 61547.800745 | 61692.995697 | 1.900 | 30 | sampled
bulk/medium/8/positions_on | tantivy | docs_per_second (docs/s) | 145243.272819 | [143551.121753, 146437.571852] | 149113.627740 | 150521.222481 | 2.712 | 30 | sampled
bulk/medium/8/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.414846 | [0.409986, 0.418354] | 0.430555 | 0.460520 | 3.145 | 30 | sampled
bulk/medium/8/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.009866 | [0.970288, 1.031023] | 1.130353 | 1.283518 | 6.835 | 30 | sampled
