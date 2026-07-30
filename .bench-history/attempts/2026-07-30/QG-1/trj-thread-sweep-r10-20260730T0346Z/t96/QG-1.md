host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | configured_engine_thread_widths=[96] | runtime_detected_isa=["aes", "avx2", "bmi2", "fma", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/96/positions_on | quill | docs_per_second (docs/s) | 26665.837383 | [26615.804315, 26737.071742] | 26776.670132 | 26776.670132 | 0.238 | 10 | sampled
bulk/xlarge/96/positions_on | tantivy | docs_per_second (docs/s) | 114905.833260 | [112767.507801, 118482.141649] | 121048.823005 | 121048.823005 | 4.421 | 10 | sampled
bulk/xlarge/96/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.232040 | [0.225840, 0.236904] | 0.261153 | 0.261153 | 4.671 | 10 | sampled
bulk/xlarge/96/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.975113 | [0.944523, 1.012650] | 1.061850 | 1.061850 | 4.812 | 10 | sampled
bulk/xlarge/96/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 1.000285 | [0.995590, 1.004703] | 1.009364 | 1.009364 | 0.863 | 10 | sampled
