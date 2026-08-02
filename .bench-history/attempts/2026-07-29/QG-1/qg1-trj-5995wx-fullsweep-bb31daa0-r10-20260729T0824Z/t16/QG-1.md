host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | threads_actually_used=[16] | runtime_detected_isa=["avx2", "fma", "bmi2", "aes", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/16/positions_on | quill | docs_per_second (docs/s) | 32791.135738 | [32711.110549, 32859.422917] | 32904.787542 | 32904.787542 | 0.260 | 10 | sampled
bulk/xlarge/16/positions_on | tantivy | docs_per_second (docs/s) | 136259.514106 | [131646.705997, 138886.217259] | 141088.915329 | 141088.915329 | 3.176 | 10 | sampled
bulk/xlarge/16/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.240611 | [0.236322, 0.248195] | 0.258702 | 0.258702 | 3.244 | 10 | sampled
bulk/xlarge/16/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.977351 | [0.891913, 1.086824] | 1.096515 | 1.096515 | 8.130 | 10 | sampled
bulk/xlarge/16/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 1.003461 | [0.993342, 1.006227] | 1.010379 | 1.010379 | 2.738 | 10 | sampled
