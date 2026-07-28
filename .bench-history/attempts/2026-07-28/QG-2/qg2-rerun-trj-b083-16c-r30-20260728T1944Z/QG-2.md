fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/1/positions_on | quill | docs_per_second (docs/s) | 60951.045076 | [59789.371908, 61560.488593] | 62414.681393 | 62997.148534 | 2.015 | 30 | sampled
bulk/medium/1/positions_on | tantivy | docs_per_second (docs/s) | 171956.219605 | [170538.819872, 175403.409381] | 196532.405816 | 200642.887903 | 5.504 | 30 | sampled
bulk/medium/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.351193 | [0.346379, 0.357508] | 0.370063 | 0.372634 | 5.274 | 30 | sampled
bulk/medium/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.015390 | [0.977303, 1.028564] | 1.316321 | 1.392684 | 13.992 | 30 | sampled
