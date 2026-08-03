host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | threads_actually_used=[2] | runtime_detected_isa=["avx2", "fma", "bmi2", "aes", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/2/positions_on | quill | docs_per_second (docs/s) | 34543.200060 | [34278.349517, 34804.123168] | 34862.363536 | 34862.363536 | 1.468 | 10 | sampled
bulk/xlarge/2/positions_on | tantivy | docs_per_second (docs/s) | 165817.833861 | [152966.943813, 176142.492909] | 178462.888781 | 178462.888781 | 7.530 | 10 | sampled
bulk/xlarge/2/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.208508 | [0.195378, 0.225094] | 0.246359 | 0.246359 | 8.671 | 10 | sampled
bulk/xlarge/2/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.997214 | [0.939986, 1.107782] | 1.184041 | 1.184041 | 8.920 | 10 | sampled
bulk/xlarge/2/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 0.996407 | [0.988306, 1.006881] | 1.014936 | 1.014936 | 1.046 | 10 | sampled
