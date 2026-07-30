host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | configured_engine_thread_widths=[32] | runtime_detected_isa=["aes", "avx2", "bmi2", "fma", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/32/positions_on | quill | docs_per_second (docs/s) | 25293.485851 | [23861.953337, 25330.732873] | 25359.023374 | 25359.023374 | 4.726 | 10 | sampled
bulk/xlarge/32/positions_on | tantivy | docs_per_second (docs/s) | 127936.432809 | [120622.432630, 131845.362367] | 138132.173027 | 138132.173027 | 4.999 | 10 | sampled
bulk/xlarge/32/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.197665 | [0.183384, 0.210235] | 0.215504 | 0.215504 | 7.999 | 10 | sampled
bulk/xlarge/32/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.033556 | [0.959830, 1.074212] | 1.119584 | 1.119584 | 5.603 | 10 | sampled
bulk/xlarge/32/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 1.006201 | [1.001874, 1.030996] | 1.087774 | 1.087774 | 2.686 | 10 | sampled
