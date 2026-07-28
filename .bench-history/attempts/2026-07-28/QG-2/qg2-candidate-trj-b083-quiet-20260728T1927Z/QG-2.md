fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/1/positions_on | quill | docs_per_second (docs/s) | 59984.370273 | [59086.597962, 60705.114429] | 61026.763729 | 61026.763729 | 1.826 | 10 | sampled
bulk/medium/1/positions_on | tantivy | docs_per_second (docs/s) | 176168.034313 | [169894.871975, 183332.557409] | 190501.322862 | 190501.322862 | 8.958 | 10 | sampled
bulk/medium/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.337959 | [0.328160, 0.359203] | 0.444467 | 0.444467 | 10.022 | 10 | sampled
bulk/medium/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.029174 | [0.949525, 1.090104] | 1.106012 | 1.106012 | 11.694 | 10 | sampled
