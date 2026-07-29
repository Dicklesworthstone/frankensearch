host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | threads_actually_used=[1] | runtime_detected_isa=["avx2", "fma", "bmi2", "aes", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/1/positions_on | quill | docs_per_second (docs/s) | 34896.922712 | [34217.166497, 35107.326808] | 35505.122759 | 35505.122759 | 1.499 | 10 | sampled
bulk/xlarge/1/positions_on | tantivy | docs_per_second (docs/s) | 135684.166031 | [133546.146608, 139579.629460] | 150566.370240 | 150566.370240 | 3.932 | 10 | sampled
bulk/xlarge/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.256083 | [0.250320, 0.258549] | 0.267710 | 0.267710 | 4.248 | 10 | sampled
bulk/xlarge/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.009203 | [0.974809, 1.033599] | 1.069315 | 1.069315 | 4.328 | 10 | sampled
bulk/xlarge/1/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 0.995814 | [0.982199, 1.028535] | 1.047562 | 1.047562 | 3.383 | 10 | sampled
