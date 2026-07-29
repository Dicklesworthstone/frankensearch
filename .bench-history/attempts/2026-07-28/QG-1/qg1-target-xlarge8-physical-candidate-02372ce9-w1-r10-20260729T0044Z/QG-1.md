fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/8/positions_on | quill | docs_per_second (docs/s) | 35359.117859 | [35234.809781, 35678.078777] | 35914.580895 | 35914.580895 | 0.797 | 10 | sampled
bulk/xlarge/8/positions_on | tantivy | docs_per_second (docs/s) | 169711.908498 | [162301.560177, 182682.592333] | 185520.804964 | 185520.804964 | 5.727 | 10 | sampled
bulk/xlarge/8/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.209635 | [0.194532, 0.219330] | 0.222677 | 0.222677 | 5.930 | 10 | sampled
bulk/xlarge/8/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.024376 | [0.966510, 1.088282] | 1.098196 | 1.098196 | 5.921 | 10 | sampled
