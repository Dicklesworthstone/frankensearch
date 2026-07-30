host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | configured_engine_thread_widths=[2] | runtime_detected_isa=["aes", "avx2", "bmi2", "fma", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/2/positions_on | quill | docs_per_second (docs/s) | 35254.033055 | [35128.516034, 35815.710856] | 35841.864183 | 35841.864183 | 0.887 | 10 | sampled
bulk/xlarge/2/positions_on | tantivy | docs_per_second (docs/s) | 194518.424976 | [185201.392631, 201206.397529] | 208701.038214 | 208701.038214 | 4.750 | 10 | sampled
bulk/xlarge/2/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.181479 | [0.175613, 0.192343] | 0.194247 | 0.194247 | 4.626 | 10 | sampled
bulk/xlarge/2/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.035006 | [0.982923, 1.101492] | 1.127147 | 1.127147 | 5.423 | 10 | sampled
bulk/xlarge/2/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 0.984838 | [0.978340, 1.004762] | 1.024204 | 1.024204 | 1.565 | 10 | sampled
