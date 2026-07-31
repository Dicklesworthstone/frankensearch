host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | configured_engine_thread_widths=[64] | runtime_detected_isa=["aes", "avx2", "bmi2", "fma", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/64/positions_on | quill | docs_per_second (docs/s) | 22417.171919 | [21719.418331, 23811.203941] | 24844.420294 | 24844.420294 | 5.117 | 10 | sampled
bulk/xlarge/64/positions_on | tantivy | docs_per_second (docs/s) | 89304.887651 | [87405.568348, 93631.022776] | 105363.536616 | 105363.536616 | 6.318 | 10 | sampled
bulk/xlarge/64/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.248270 | [0.233913, 0.269171] | 0.287874 | 0.287874 | 8.358 | 10 | sampled
bulk/xlarge/64/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.019307 | [0.975527, 1.058812] | 1.077215 | 1.077215 | 4.819 | 10 | sampled
bulk/xlarge/64/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 1.019208 | [0.997118, 1.041050] | 1.323521 | 1.323521 | 9.085 | 10 | sampled
