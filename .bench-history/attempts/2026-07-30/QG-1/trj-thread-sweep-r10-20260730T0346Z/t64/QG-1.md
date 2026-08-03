host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | configured_engine_thread_widths=[64] | runtime_detected_isa=["aes", "avx2", "bmi2", "fma", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/64/positions_on | quill | docs_per_second (docs/s) | 25029.379410 | [24753.671556, 25328.515516] | 25367.790714 | 25367.790714 | 1.543 | 10 | sampled
bulk/xlarge/64/positions_on | tantivy | docs_per_second (docs/s) | 124060.934030 | [121210.082826, 124620.656979] | 128519.835930 | 128519.835930 | 2.097 | 10 | sampled
bulk/xlarge/64/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.201267 | [0.198519, 0.206829] | 0.210903 | 0.210903 | 2.276 | 10 | sampled
bulk/xlarge/64/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.021134 | [0.993127, 1.055239] | 1.083554 | 1.083554 | 4.170 | 10 | sampled
bulk/xlarge/64/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 1.006181 | [0.995941, 1.023127] | 1.041879 | 1.041879 | 1.802 | 10 | sampled
