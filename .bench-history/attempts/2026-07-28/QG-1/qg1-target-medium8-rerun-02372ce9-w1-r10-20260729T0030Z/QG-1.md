fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/8/positions_on | quill | docs_per_second (docs/s) | 58889.338526 | [58080.862744, 60532.434895] | 61128.467378 | 61128.467378 | 2.635 | 10 | sampled
bulk/medium/8/positions_on | tantivy | docs_per_second (docs/s) | 145136.142703 | [141155.541247, 147986.226661] | 149650.996218 | 149650.996218 | 6.758 | 10 | sampled
bulk/medium/8/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.408111 | [0.394410, 0.421554] | 0.526259 | 0.526259 | 9.330 | 10 | sampled
bulk/medium/8/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.956759 | [0.763874, 1.292196] | 1.335009 | 1.335009 | 23.305 | 10 | sampled
