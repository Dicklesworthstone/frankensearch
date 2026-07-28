fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/1/positions_on | quill | docs_per_second (docs/s) | 60093.065658 | [59549.960994, 60545.429685] | 61703.529301 | 62183.695636 | 1.908 | 30 | sampled
bulk/medium/1/positions_on | tantivy | docs_per_second (docs/s) | 168328.238336 | [164855.510179, 172392.035194] | 195680.279557 | 196037.619415 | 10.283 | 30 | sampled
bulk/medium/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.355724 | [0.347941, 0.363887] | 0.492400 | 0.497981 | 12.754 | 30 | sampled
bulk/medium/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.011557 | [0.981395, 1.026351] | 1.270850 | 1.450152 | 11.514 | 30 | sampled
