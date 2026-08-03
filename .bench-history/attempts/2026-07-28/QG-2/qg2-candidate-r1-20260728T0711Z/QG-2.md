fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/1/positions_on | quill | docs_per_second (docs/s) | 22186.527445 | [22024.223756, 22336.447882] | 22627.065896 | 22653.284444 | 2.874 | 80 | sampled
bulk/medium/1/positions_on | tantivy | docs_per_second (docs/s) | 200495.693102 | [195532.234706, 207930.983776] | 222854.678848 | 230461.341154 | 14.517 | 80 | sampled
bulk/medium/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.108698 | [0.106998, 0.111538] | 0.156849 | 0.162413 | 16.673 | 80 | sampled
bulk/medium/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.993438 | [0.979880, 1.019980] | 1.336857 | 1.399445 | 16.411 | 80 | sampled
