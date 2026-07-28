fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/1/positions_on | quill | docs_per_second (docs/s) | 20600.811692 | [20459.864139, 20697.708674] | 21073.183043 | 21126.352551 | 11.723 | 80 | sampled
bulk/medium/1/positions_on | tantivy | docs_per_second (docs/s) | 176456.113524 | [173128.537123, 180321.439132] | 190600.964766 | 196271.471677 | 19.550 | 80 | sampled
bulk/medium/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.116139 | [0.113996, 0.118605] | 0.161943 | 0.167419 | 14.146 | 80 | sampled
bulk/medium/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.987888 | [0.977930, 0.999458] | 1.338819 | 1.356635 | 11.419 | 80 | sampled
