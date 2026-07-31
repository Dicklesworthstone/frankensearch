host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | configured_engine_thread_widths=[1] | runtime_detected_isa=["aes", "avx2", "bmi2", "fma", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/1/positions_on | quill | docs_per_second (docs/s) | 35551.521461 | [35248.601917, 37181.465880] | 37202.277604 | 37202.277604 | 2.544 | 10 | sampled
bulk/xlarge/1/positions_on | tantivy | docs_per_second (docs/s) | 71930.064940 | [71127.783641, 72407.779572] | 73791.964638 | 73791.964638 | 1.361 | 10 | sampled
bulk/xlarge/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.499882 | [0.487438, 0.511549] | 0.529354 | 0.529354 | 3.178 | 10 | sampled
bulk/xlarge/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.009576 | [0.999715, 1.035060] | 1.047158 | 1.047158 | 2.016 | 10 | sampled
bulk/xlarge/1/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 1.001974 | [0.980393, 1.029851] | 1.032598 | 1.032598 | 2.750 | 10 | sampled
