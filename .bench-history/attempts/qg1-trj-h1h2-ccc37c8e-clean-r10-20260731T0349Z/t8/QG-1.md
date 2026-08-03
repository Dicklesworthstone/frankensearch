host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | configured_engine_thread_widths=[8] | runtime_detected_isa=["aes", "avx2", "bmi2", "fma", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/8/positions_on | quill | docs_per_second (docs/s) | 35409.346500 | [35244.042238, 35474.182239] | 35724.250752 | 35724.250752 | 0.414 | 10 | sampled
bulk/xlarge/8/positions_on | tantivy | docs_per_second (docs/s) | 126257.902574 | [122904.653939, 130558.700577] | 132034.126489 | 132034.126489 | 2.980 | 10 | sampled
bulk/xlarge/8/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.281265 | [0.269878, 0.286662] | 0.294322 | 0.294322 | 3.036 | 10 | sampled
bulk/xlarge/8/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.998773 | [0.903122, 1.045195] | 1.137412 | 1.137412 | 8.307 | 10 | sampled
bulk/xlarge/8/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 0.998907 | [0.994105, 1.005084] | 1.006480 | 1.006480 | 0.544 | 10 | sampled
