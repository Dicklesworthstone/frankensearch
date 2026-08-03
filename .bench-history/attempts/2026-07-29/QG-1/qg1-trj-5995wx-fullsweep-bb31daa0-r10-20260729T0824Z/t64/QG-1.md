host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | threads_actually_used=[64] | runtime_detected_isa=["avx2", "fma", "bmi2", "aes", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/64/positions_on | quill | docs_per_second (docs/s) | 28562.220550 | [28477.380275, 28642.742137] | 28724.583012 | 28724.583012 | 0.355 | 10 | sampled
bulk/xlarge/64/positions_on | tantivy | docs_per_second (docs/s) | 126486.750347 | [123062.630591, 128421.342939] | 133679.582935 | 133679.582935 | 2.814 | 10 | sampled
bulk/xlarge/64/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.226502 | [0.221930, 0.231632] | 0.235977 | 0.235977 | 2.633 | 10 | sampled
bulk/xlarge/64/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.995041 | [0.970685, 1.052641] | 1.118235 | 1.118235 | 5.312 | 10 | sampled
bulk/xlarge/64/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 1.005064 | [0.999291, 1.011164] | 1.048264 | 1.048264 | 1.411 | 10 | sampled
