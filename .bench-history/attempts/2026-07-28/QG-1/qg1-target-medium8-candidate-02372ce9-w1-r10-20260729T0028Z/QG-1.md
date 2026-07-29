fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/8/positions_on | quill | docs_per_second (docs/s) | 59734.153712 | [58989.711370, 60429.604038] | 60697.755639 | 60697.755639 | 1.896 | 10 | sampled
bulk/medium/8/positions_on | tantivy | docs_per_second (docs/s) | 142446.945526 | [139816.122012, 144738.300763] | 146375.911450 | 146375.911450 | 1.834 | 10 | sampled
bulk/medium/8/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.423926 | [0.409132, 0.428826] | 0.435832 | 0.435832 | 3.374 | 10 | sampled
bulk/medium/8/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.984610 | [0.872806, 1.014317] | 1.203820 | 1.203820 | 14.091 | 10 | sampled
