host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | configured_engine_thread_widths=[1] | runtime_detected_isa=["aes", "avx2", "bmi2", "fma", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/1/positions_on | quill | docs_per_second (docs/s) | 33381.135390 | [33177.078446, 33921.240686] | 34351.658770 | 34351.658770 | 1.412 | 10 | sampled
bulk/xlarge/1/positions_on | tantivy | docs_per_second (docs/s) | 133744.011899 | [131089.806514, 136869.397148] | 138689.706335 | 138689.706335 | 2.438 | 10 | sampled
bulk/xlarge/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.249112 | [0.245724, 0.255063] | 0.263239 | 0.263239 | 2.688 | 10 | sampled
bulk/xlarge/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.001444 | [0.971807, 1.020617] | 1.049974 | 1.049974 | 3.563 | 10 | sampled
bulk/xlarge/1/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 1.013172 | [0.979475, 1.023066] | 1.074946 | 1.074946 | 2.892 | 10 | sampled
