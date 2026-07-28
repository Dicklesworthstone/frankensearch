fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/medium/1/positions_on | quill | docs_per_second (docs/s) | 59817.999361 | [59203.634148, 60628.649988] | 63158.702397 | 63184.239745 | 2.548 | 30 | sampled
bulk/medium/1/positions_on | tantivy | docs_per_second (docs/s) | 171222.778564 | [168262.958664, 173820.356557] | 181787.712875 | 183666.086386 | 6.829 | 30 | sampled
bulk/medium/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.349777 | [0.344699, 0.356242] | 0.448249 | 0.476755 | 8.650 | 30 | sampled
bulk/medium/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.014546 | [0.981241, 1.025487] | 1.128707 | 1.359667 | 7.889 | 30 | sampled
