fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/1/positions_on | quill | docs_per_second (docs/s) | 59916.824489 | [59391.632877, 60551.440398] | 61468.350398 | 61888.402560 | 2.234 | 30 | sampled
bulk/medium/1/positions_on | tantivy | docs_per_second (docs/s) | 170265.798707 | [166428.628094, 173329.480854] | 191700.510433 | 198474.408990 | 8.582 | 30 | sampled
bulk/medium/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.351468 | [0.345776, 0.361292] | 0.460488 | 0.467496 | 9.623 | 30 | sampled
bulk/medium/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.012556 | [0.959109, 1.029503] | 1.322983 | 1.418143 | 13.708 | 30 | sampled
