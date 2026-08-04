host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | configured_engine_thread_widths=[128] | runtime_detected_isa=["aes", "avx2", "bmi2", "fma", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/128/positions_on | quill | docs_per_second (docs/s) | 22408.809882 | [21687.864515, 23686.777504] | 24575.185488 | 24575.185488 | 5.783 | 10 | sampled
bulk/xlarge/128/positions_on | tantivy | docs_per_second (docs/s) | 79804.619630 | [75278.415549, 85880.431386] | 89522.973113 | 89522.973113 | 9.278 | 10 | sampled
bulk/xlarge/128/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.275095 | [0.267489, 0.311525] | 0.345518 | 0.345518 | 11.075 | 10 | sampled
bulk/xlarge/128/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.975137 | [0.952548, 1.018570] | 1.073869 | 1.073869 | 4.069 | 10 | sampled
bulk/xlarge/128/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 1.001677 | [0.912244, 1.136503] | 1.257200 | 1.257200 | 14.330 | 10 | sampled
