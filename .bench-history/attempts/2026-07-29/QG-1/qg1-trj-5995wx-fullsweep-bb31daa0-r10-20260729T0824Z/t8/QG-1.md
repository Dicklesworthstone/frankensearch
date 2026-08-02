host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | threads_actually_used=[8] | runtime_detected_isa=["avx2", "fma", "bmi2", "aes", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/8/positions_on | quill | docs_per_second (docs/s) | 33770.710314 | [33707.457295, 33947.344216] | 34168.777784 | 34168.777784 | 0.516 | 10 | sampled
bulk/xlarge/8/positions_on | tantivy | docs_per_second (docs/s) | 142833.746249 | [139030.868034, 145514.732973] | 145945.876717 | 145945.876717 | 3.798 | 10 | sampled
bulk/xlarge/8/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.237279 | [0.232398, 0.244989] | 0.263650 | 0.263650 | 4.124 | 10 | sampled
bulk/xlarge/8/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.030607 | [0.988788, 1.100675] | 1.141957 | 1.141957 | 5.874 | 10 | sampled
bulk/xlarge/8/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 1.004562 | [0.997820, 1.010276] | 1.019635 | 1.019635 | 1.064 | 10 | sampled
