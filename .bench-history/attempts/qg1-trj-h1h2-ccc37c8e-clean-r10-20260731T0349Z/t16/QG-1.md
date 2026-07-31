host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | configured_engine_thread_widths=[16] | runtime_detected_isa=["aes", "avx2", "bmi2", "fma", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/16/positions_on | quill | docs_per_second (docs/s) | 33473.596370 | [33317.612892, 33733.553811] | 33954.990524 | 33954.990524 | 0.721 | 10 | sampled
bulk/xlarge/16/positions_on | tantivy | docs_per_second (docs/s) | 117564.255649 | [111106.651068, 124097.197180] | 138407.743818 | 138407.743818 | 7.886 | 10 | sampled
bulk/xlarge/16/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.285757 | [0.270176, 0.300316] | 0.321877 | 0.321877 | 7.659 | 10 | sampled
bulk/xlarge/16/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.072212 | [0.938138, 1.157698] | 1.265561 | 1.265561 | 10.685 | 10 | sampled
bulk/xlarge/16/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 0.996756 | [0.991655, 1.005972] | 1.012702 | 1.012702 | 0.845 | 10 | sampled
