host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | threads_actually_used=[128] | runtime_detected_isa=["avx2", "fma", "bmi2", "aes", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/128/positions_on | quill | docs_per_second (docs/s) | 28211.910882 | [27341.926858, 28488.135312] | 28506.857472 | 28506.857472 | 2.749 | 10 | sampled
bulk/xlarge/128/positions_on | tantivy | docs_per_second (docs/s) | 108901.703696 | [106310.272540, 111991.986255] | 112744.318151 | 112744.318151 | 2.689 | 10 | sampled
bulk/xlarge/128/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.254458 | [0.248676, 0.262211] | 0.265656 | 0.265656 | 2.836 | 10 | sampled
bulk/xlarge/128/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.013140 | [0.949719, 1.059121] | 1.076291 | 1.076291 | 6.577 | 10 | sampled
bulk/xlarge/128/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 0.999793 | [0.994249, 1.005438] | 1.007795 | 1.007795 | 0.872 | 10 | sampled
