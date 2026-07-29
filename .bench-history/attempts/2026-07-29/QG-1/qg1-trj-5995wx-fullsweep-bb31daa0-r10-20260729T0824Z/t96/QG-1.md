host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | threads_actually_used=[96] | runtime_detected_isa=["avx2", "fma", "bmi2", "aes", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/96/positions_on | quill | docs_per_second (docs/s) | 26570.019861 | [26157.841176, 27099.053790] | 28552.365667 | 28552.365667 | 2.870 | 10 | sampled
bulk/xlarge/96/positions_on | tantivy | docs_per_second (docs/s) | 112658.278770 | [109918.740449, 115464.053814] | 117992.367179 | 117992.367179 | 2.577 | 10 | sampled
bulk/xlarge/96/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.236490 | [0.234359, 0.239458] | 0.246480 | 0.246480 | 1.819 | 10 | sampled
bulk/xlarge/96/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.982556 | [0.956492, 1.026465] | 1.060546 | 1.060546 | 4.042 | 10 | sampled
bulk/xlarge/96/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 0.994897 | [0.985826, 1.004746] | 1.015027 | 1.015027 | 2.160 | 10 | sampled
