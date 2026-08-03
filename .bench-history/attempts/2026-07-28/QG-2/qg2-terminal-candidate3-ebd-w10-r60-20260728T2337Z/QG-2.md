fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/1/positions_on | quill | docs_per_second (docs/s) | 58886.931972 | [58328.136915, 59473.992550] | 61553.453232 | 61832.291097 | 3.376 | 60 | sampled
bulk/medium/1/positions_on | tantivy | docs_per_second (docs/s) | 168449.241331 | [165638.780399, 170367.719676] | 184453.445387 | 195379.554874 | 8.084 | 60 | sampled
bulk/medium/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.348790 | [0.344552, 0.355943] | 0.395470 | 0.485883 | 9.589 | 60 | sampled
bulk/medium/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.008507 | [0.997181, 1.017172] | 1.076338 | 1.086130 | 8.333 | 60 | sampled
