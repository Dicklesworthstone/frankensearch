fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
watch/medium/initial | quill | docs_per_second (docs/s) | 46170.160811 | [45792.116119, 48624.831418] | 49599.562889 | 49599.562889 | 3.094 | 10 | sampled
watch/medium/initial | tantivy | docs_per_second (docs/s) | 53032.027520 | [47507.461499, 56265.627778] | 57293.312656 | 57293.312656 | 10.049 | 10 | sampled
watch/medium/initial | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.898269 | [0.814406, 1.031285] | 1.101211 | 1.101211 | 11.667 | 10 | sampled
watch/medium/initial | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.946555 | [0.728639, 1.037331] | 10.462519 | 10.462519 | 162.485 | 10 | sampled
