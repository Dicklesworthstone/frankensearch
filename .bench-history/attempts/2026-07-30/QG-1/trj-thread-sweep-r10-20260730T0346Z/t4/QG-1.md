host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | configured_engine_thread_widths=[4] | runtime_detected_isa=["aes", "avx2", "bmi2", "fma", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/4/positions_on | quill | docs_per_second (docs/s) | 33548.583811 | [33075.786696, 33860.812539] | 34123.122065 | 34123.122065 | 1.320 | 10 | sampled
bulk/xlarge/4/positions_on | tantivy | docs_per_second (docs/s) | 163632.891812 | [157999.584483, 167737.859170] | 170657.512914 | 170657.512914 | 3.872 | 10 | sampled
bulk/xlarge/4/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.203996 | [0.200164, 0.212998] | 0.219921 | 0.219921 | 3.881 | 10 | sampled
bulk/xlarge/4/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.061745 | [0.970209, 1.179354] | 1.245154 | 1.245154 | 10.043 | 10 | sampled
bulk/xlarge/4/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 1.000817 | [0.997241, 1.003395] | 1.007233 | 1.007233 | 0.542 | 10 | sampled
