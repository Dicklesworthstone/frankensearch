host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | configured_engine_thread_widths=[8] | runtime_detected_isa=["aes", "avx2", "bmi2", "fma", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/8/positions_on | quill | docs_per_second (docs/s) | 33896.043733 | [33663.029941, 33948.886557] | 34337.309957 | 34337.309957 | 0.601 | 10 | sampled
bulk/xlarge/8/positions_on | tantivy | docs_per_second (docs/s) | 149192.426187 | [146292.697143, 152707.720549] | 159483.700304 | 159483.700304 | 3.772 | 10 | sampled
bulk/xlarge/8/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.226381 | [0.220979, 0.232432] | 0.246876 | 0.246876 | 4.201 | 10 | sampled
bulk/xlarge/8/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.999872 | [0.938410, 1.033428] | 1.057625 | 1.057625 | 5.194 | 10 | sampled
bulk/xlarge/8/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 1.000472 | [0.997858, 1.013281] | 1.023015 | 1.023015 | 1.334 | 10 | sampled
