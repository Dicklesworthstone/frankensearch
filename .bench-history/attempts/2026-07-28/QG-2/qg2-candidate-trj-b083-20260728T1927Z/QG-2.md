fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/1/positions_on | quill | docs_per_second (docs/s) | 56733.727174 | [54780.302429, 57403.603473] | 58075.351828 | 58075.351828 | 2.341 | 10 | sampled
bulk/medium/1/positions_on | tantivy | docs_per_second (docs/s) | 165187.462408 | [156419.072016, 168982.607093] | 178204.196819 | 178204.196819 | 4.979 | 10 | sampled
bulk/medium/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.342662 | [0.330467, 0.358951] | 0.368031 | 0.368031 | 4.409 | 10 | sampled
bulk/medium/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.970133 | [0.936478, 1.011197] | 1.034447 | 1.034447 | 3.992 | 10 | sampled
