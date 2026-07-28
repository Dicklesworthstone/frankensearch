fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/1/positions_on | quill | docs_per_second (docs/s) | 59102.729413 | [58676.967727, 59514.833901] | 60596.593508 | 60678.532324 | 2.145 | 30 | sampled
bulk/medium/1/positions_on | tantivy | docs_per_second (docs/s) | 142965.044105 | [142141.309479, 146777.573466] | 152835.154472 | 154001.496814 | 4.200 | 30 | sampled
bulk/medium/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.412222 | [0.400083, 0.417475] | 0.442421 | 0.457762 | 5.154 | 30 | sampled
bulk/medium/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.003639 | [0.962635, 1.015330] | 1.052766 | 1.073382 | 4.781 | 30 | sampled
