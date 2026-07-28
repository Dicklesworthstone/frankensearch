fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/1/positions_on | quill | docs_per_second (docs/s) | 58967.496036 | [58176.707654, 59557.466672] | 60820.092853 | 61255.587458 | 2.093 | 30 | sampled
bulk/medium/1/positions_on | tantivy | docs_per_second (docs/s) | 171186.778372 | [167044.591163, 175221.930517] | 187495.789548 | 188591.969901 | 8.260 | 30 | sampled
bulk/medium/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.342848 | [0.332873, 0.352546] | 0.467796 | 0.467974 | 10.359 | 30 | sampled
bulk/medium/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.996757 | [0.984431, 1.021417] | 1.357588 | 1.368422 | 11.132 | 30 | sampled
