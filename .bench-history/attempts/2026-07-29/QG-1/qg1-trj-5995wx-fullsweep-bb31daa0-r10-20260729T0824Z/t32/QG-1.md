host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | threads_actually_used=[32] | runtime_detected_isa=["avx2", "fma", "bmi2", "aes", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/32/positions_on | quill | docs_per_second (docs/s) | 25142.024425 | [24706.769353, 25347.107343] | 25604.278410 | 25604.278410 | 2.630 | 10 | sampled
bulk/xlarge/32/positions_on | tantivy | docs_per_second (docs/s) | 117388.872405 | [115557.380207, 122162.204406] | 129159.975882 | 129159.975882 | 3.970 | 10 | sampled
bulk/xlarge/32/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.209423 | [0.199148, 0.216683] | 0.227475 | 0.227475 | 4.545 | 10 | sampled
bulk/xlarge/32/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.003475 | [0.947917, 1.059819] | 1.071682 | 1.071682 | 5.707 | 10 | sampled
bulk/xlarge/32/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 1.006079 | [0.989295, 1.013537] | 1.017768 | 1.017768 | 1.648 | 10 | sampled
