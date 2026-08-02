fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/1/positions_on | quill | docs_per_second (docs/s) | 57377.472435 | [56151.162288, 58436.861909] | 60728.792318 | 61187.666883 | 3.952 | 60 | sampled
bulk/medium/1/positions_on | tantivy | docs_per_second (docs/s) | 167280.499341 | [164066.405550, 169939.440982] | 182596.857818 | 188805.026479 | 8.546 | 60 | sampled
bulk/medium/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.345034 | [0.337280, 0.351354] | 0.429665 | 0.456103 | 9.377 | 60 | sampled
bulk/medium/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.007193 | [0.979333, 1.040113] | 1.177893 | 1.352639 | 10.015 | 60 | sampled
