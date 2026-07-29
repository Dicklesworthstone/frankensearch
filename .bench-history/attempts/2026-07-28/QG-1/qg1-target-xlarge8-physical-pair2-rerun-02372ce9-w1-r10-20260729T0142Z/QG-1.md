fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/8/positions_on | quill | docs_per_second (docs/s) | 36236.107336 | [36103.153949, 36356.254561] | 36501.389688 | 36501.389688 | 0.381 | 10 | sampled
bulk/xlarge/8/positions_on | tantivy | docs_per_second (docs/s) | 175881.116554 | [169764.480425, 182851.149777] | 185025.293317 | 185025.293317 | 5.046 | 10 | sampled
bulk/xlarge/8/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.206296 | [0.198693, 0.213267] | 0.232888 | 0.232888 | 5.174 | 10 | sampled
bulk/xlarge/8/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.006406 | [0.974261, 1.055636] | 1.105650 | 1.105650 | 4.454 | 10 | sampled
