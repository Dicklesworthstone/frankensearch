fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/1/positions_on | quill | docs_per_second (docs/s) | 21055.427384 | [21010.077988, 21079.502753] | 21320.786213 | 21816.439854 | 1.062 | 80 | sampled
bulk/medium/1/positions_on | tantivy | docs_per_second (docs/s) | 184185.971541 | [182807.415168, 186251.020446] | 193745.174195 | 195639.782429 | 8.040 | 80 | sampled
bulk/medium/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.114468 | [0.113013, 0.115194] | 0.153741 | 0.159940 | 10.097 | 80 | sampled
bulk/medium/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.007384 | [0.997443, 1.016601] | 1.070589 | 1.374192 | 10.700 | 80 | sampled
