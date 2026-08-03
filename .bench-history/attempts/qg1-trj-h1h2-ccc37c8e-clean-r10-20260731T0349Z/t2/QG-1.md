host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | configured_engine_thread_widths=[2] | runtime_detected_isa=["aes", "avx2", "bmi2", "fma", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/2/positions_on | quill | docs_per_second (docs/s) | 36505.684718 | [36363.649804, 36909.708969] | 37045.059938 | 37045.059938 | 0.790 | 10 | sampled
bulk/xlarge/2/positions_on | tantivy | docs_per_second (docs/s) | 116424.790153 | [113887.736441, 119429.979540] | 123201.097455 | 123201.097455 | 2.808 | 10 | sampled
bulk/xlarge/2/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.312842 | [0.306422, 0.321050] | 0.323535 | 0.323535 | 2.558 | 10 | sampled
bulk/xlarge/2/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.038377 | [0.986496, 1.060963] | 1.129946 | 1.129946 | 4.866 | 10 | sampled
bulk/xlarge/2/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 1.012091 | [0.986909, 1.037052] | 1.040158 | 1.040158 | 2.599 | 10 | sampled
