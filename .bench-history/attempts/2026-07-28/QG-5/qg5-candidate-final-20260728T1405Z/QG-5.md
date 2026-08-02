fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
compaction/xlarge/5pct | quill | wall_clock_ms (ms) | 472090.427558 | [454368.242804, 496948.375705] | 513933.646762 | 513933.646762 | 4.995 | 10 | sampled
compaction/xlarge/5pct | tantivy | wall_clock_ms (ms) | 5595.356647 | [5519.775866, 5750.195214] | 6209.259851 | 6209.259851 | 3.642 | 10 | sampled
compaction/xlarge/5pct | paired_ab | wall_clock_ms_quill_over_tantivy (ratio) | 81.232797 | [79.538048, 88.837896] | 94.151268 | 94.151268 | 6.307 | 10 | sampled
compaction/xlarge/5pct | paired_null | wall_clock_ms_tantivy_over_tantivy (ratio) | 0.993857 | [0.872782, 1.003735] | 1.136743 | 1.136743 | 18.575 | 10 | sampled
compaction/xlarge/20pct | quill | wall_clock_ms (ms) | 377231.605586 | [366001.962062, 398262.353689] | 445301.137102 | 445301.137102 | 6.475 | 10 | sampled
compaction/xlarge/20pct | tantivy | wall_clock_ms (ms) | 5173.235015 | [4883.248757, 6012.074957] | 7598.501491 | 7598.501491 | 15.382 | 10 | sampled
compaction/xlarge/20pct | paired_ab | wall_clock_ms_quill_over_tantivy (ratio) | 73.788885 | [64.477313, 76.181474] | 78.360924 | 78.360924 | 10.137 | 10 | sampled
compaction/xlarge/20pct | paired_null | wall_clock_ms_tantivy_over_tantivy (ratio) | 1.023110 | [0.962892, 1.073249] | 1.229016 | 1.229016 | 10.162 | 10 | sampled
compaction/xlarge/50pct | quill | wall_clock_ms (ms) | 249491.841810 | [242260.126905, 260605.803569] | 265873.976535 | 265873.976535 | 3.870 | 10 | sampled
compaction/xlarge/50pct | tantivy | wall_clock_ms (ms) | 3993.916614 | [3801.433994, 5572.566378] | 7955.429824 | 7955.429824 | 30.289 | 10 | sampled
compaction/xlarge/50pct | paired_ab | wall_clock_ms_quill_over_tantivy (ratio) | 61.455612 | [47.634649, 65.256259] | 69.247120 | 69.247120 | 20.708 | 10 | sampled
compaction/xlarge/50pct | paired_null | wall_clock_ms_tantivy_over_tantivy (ratio) | 0.995314 | [0.910846, 1.105079] | 1.975079 | 1.975079 | 28.946 | 10 | sampled
