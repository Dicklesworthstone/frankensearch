fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/1/positions_on | quill | docs_per_second (docs/s) | 59745.012886 | [58686.910924, 60656.692392] | 61764.602158 | 61975.865074 | 2.548 | 30 | sampled
bulk/medium/1/positions_on | tantivy | docs_per_second (docs/s) | 176173.266833 | [169808.174915, 181555.134177] | 199511.069395 | 200086.764024 | 11.752 | 30 | sampled
bulk/medium/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.340396 | [0.327940, 0.346834] | 0.472453 | 0.486455 | 14.066 | 30 | sampled
bulk/medium/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.004514 | [0.976182, 1.025113] | 1.096417 | 1.393764 | 10.060 | 30 | sampled
