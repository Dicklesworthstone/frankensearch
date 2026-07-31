host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | configured_engine_thread_widths=[96] | runtime_detected_isa=["aes", "avx2", "bmi2", "fma", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/96/positions_on | quill | docs_per_second (docs/s) | 24465.701215 | [24087.332147, 25441.343973] | 25647.081371 | 25647.081371 | 3.179 | 10 | sampled
bulk/xlarge/96/positions_on | tantivy | docs_per_second (docs/s) | 64092.429866 | [59856.035405, 72651.536587] | 74665.189123 | 74665.189123 | 9.525 | 10 | sampled
bulk/xlarge/96/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.382646 | [0.344567, 0.413863] | 0.419301 | 0.419301 | 9.718 | 10 | sampled
bulk/xlarge/96/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.999471 | [0.985568, 1.021700] | 1.053792 | 1.053792 | 2.561 | 10 | sampled
bulk/xlarge/96/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 1.004594 | [0.938684, 1.040481] | 1.084311 | 1.084311 | 8.351 | 10 | sampled
