host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | configured_engine_thread_widths=[128] | runtime_detected_isa=["aes", "avx2", "bmi2", "fma", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/128/positions_on | quill | docs_per_second (docs/s) | 26125.237459 | [25012.247626, 26905.389810] | 26927.707090 | 26927.707090 | 4.292 | 10 | sampled
bulk/xlarge/128/positions_on | tantivy | docs_per_second (docs/s) | 111044.316673 | [109660.527346, 113872.261503] | 117972.742816 | 117972.742816 | 3.096 | 10 | sampled
bulk/xlarge/128/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.230975 | [0.221999, 0.241524] | 0.254308 | 0.254308 | 5.258 | 10 | sampled
bulk/xlarge/128/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.008721 | [0.951635, 1.042144] | 1.074639 | 1.074639 | 4.937 | 10 | sampled
bulk/xlarge/128/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 0.984577 | [0.966841, 1.006051] | 1.054710 | 1.054710 | 2.978 | 10 | sampled
