fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/8/positions_on | quill | docs_per_second (docs/s) | 59293.568236 | [58557.580158, 60291.082089] | 61317.634361 | 61825.112971 | 2.024 | 30 | sampled
bulk/medium/8/positions_on | tantivy | docs_per_second (docs/s) | 146762.526515 | [144056.524885, 147318.912746] | 155526.136163 | 160284.157396 | 3.800 | 30 | sampled
bulk/medium/8/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.405808 | [0.403288, 0.410703] | 0.448212 | 0.462295 | 4.719 | 30 | sampled
bulk/medium/8/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.988191 | [0.976003, 1.002628] | 1.046167 | 1.107362 | 5.880 | 30 | sampled
