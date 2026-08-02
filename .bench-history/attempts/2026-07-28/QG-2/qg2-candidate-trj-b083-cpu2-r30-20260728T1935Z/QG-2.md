fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/1/positions_on | quill | docs_per_second (docs/s) | 58834.388046 | [58363.916181, 59756.268783] | 60627.492314 | 60688.245879 | 1.927 | 30 | sampled
bulk/medium/1/positions_on | tantivy | docs_per_second (docs/s) | 143354.528419 | [140962.542159, 145558.905027] | 150076.083772 | 150444.956756 | 3.620 | 30 | sampled
bulk/medium/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.414361 | [0.407928, 0.422438] | 0.443826 | 0.448025 | 4.017 | 30 | sampled
bulk/medium/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.994545 | [0.978813, 1.025047] | 1.067555 | 1.075206 | 4.642 | 30 | sampled
