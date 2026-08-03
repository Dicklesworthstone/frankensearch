fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
query/naturallanguage/k100/100k | quill | latency_ms (ms) | 98.952686 | [98.416782, 102.878045] | 103.592125 | 104.193921 | 2.321 | 12 | sampled
query/naturallanguage/k100/100k | tantivy | latency_ms (ms) | 0.903072 | [0.873009, 0.940727] | 0.956154 | 0.961029 | 4.262 | 12 | sampled
query/naturallanguage/k100/100k | paired_ab | latency_ms_quill_over_tantivy (ratio) | 110.703751 | [106.930043, 113.676243] | 119.424894 | 121.704849 | 5.188 | 12 | sampled
query/naturallanguage/k100/100k | paired_null | latency_ms_tantivy_over_tantivy (ratio) | 1.000704 | [0.998719, 1.002890] | 1.007912 | 1.096919 | 2.664 | 12 | sampled
