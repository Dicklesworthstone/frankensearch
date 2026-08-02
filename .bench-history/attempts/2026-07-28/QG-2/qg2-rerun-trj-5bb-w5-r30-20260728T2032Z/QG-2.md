fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/1/positions_on | quill | docs_per_second (docs/s) | 60279.356921 | [59862.839723, 61086.196705] | 62093.741991 | 62491.351431 | 2.189 | 30 | sampled
bulk/medium/1/positions_on | tantivy | docs_per_second (docs/s) | 173886.648092 | [169733.287189, 179180.277599] | 196103.291887 | 200886.389498 | 5.365 | 30 | sampled
bulk/medium/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.345546 | [0.341425, 0.351114] | 0.367096 | 0.369385 | 4.580 | 30 | sampled
bulk/medium/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.019110 | [0.982464, 1.046989] | 1.450587 | 1.453007 | 16.376 | 30 | sampled
