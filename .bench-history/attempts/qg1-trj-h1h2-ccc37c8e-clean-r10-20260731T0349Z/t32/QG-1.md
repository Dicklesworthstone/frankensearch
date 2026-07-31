host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | configured_engine_thread_widths=[32] | runtime_detected_isa=["aes", "avx2", "bmi2", "fma", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/32/positions_on | quill | docs_per_second (docs/s) | 25712.146160 | [23997.679226, 26378.315829] | 27226.292113 | 27226.292113 | 7.686 | 10 | sampled
bulk/xlarge/32/positions_on | tantivy | docs_per_second (docs/s) | 109380.263950 | [102213.796614, 116259.721743] | 128914.350371 | 128914.350371 | 11.095 | 10 | sampled
bulk/xlarge/32/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.230809 | [0.210160, 0.254111] | 0.276618 | 0.276618 | 11.033 | 10 | sampled
bulk/xlarge/32/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.971548 | [0.942132, 1.035605] | 1.056566 | 1.056566 | 4.623 | 10 | sampled
bulk/xlarge/32/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 0.996132 | [0.983333, 1.004825] | 1.022423 | 1.022423 | 1.358 | 10 | sampled
