host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | threads_actually_used=[4] | runtime_detected_isa=["avx2", "fma", "bmi2", "aes", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/4/positions_on | quill | docs_per_second (docs/s) | 35589.080982 | [35388.876256, 35720.412677] | 35969.389234 | 35969.389234 | 0.639 | 10 | sampled
bulk/xlarge/4/positions_on | tantivy | docs_per_second (docs/s) | 173165.385115 | [168911.542458, 185089.239285] | 190974.287102 | 190974.287102 | 5.731 | 10 | sampled
bulk/xlarge/4/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.205067 | [0.195061, 0.210728] | 0.224948 | 0.224948 | 5.788 | 10 | sampled
bulk/xlarge/4/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.029780 | [0.906013, 1.076442] | 1.088303 | 1.088303 | 8.617 | 10 | sampled
bulk/xlarge/4/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 0.994271 | [0.988540, 1.005279] | 1.015810 | 1.015810 | 1.045 | 10 | sampled
