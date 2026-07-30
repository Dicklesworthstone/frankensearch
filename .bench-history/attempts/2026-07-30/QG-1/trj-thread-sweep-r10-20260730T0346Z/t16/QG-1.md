host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | configured_engine_thread_widths=[16] | runtime_detected_isa=["aes", "avx2", "bmi2", "fma", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/16/positions_on | quill | docs_per_second (docs/s) | 29464.767863 | [29357.673889, 29632.492335] | 29869.992954 | 29869.992954 | 0.661 | 10 | sampled
bulk/xlarge/16/positions_on | tantivy | docs_per_second (docs/s) | 129896.194692 | [126428.818189, 138316.042927] | 140871.286708 | 140871.286708 | 4.452 | 10 | sampled
bulk/xlarge/16/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.226362 | [0.214720, 0.234257] | 0.237462 | 0.237462 | 4.463 | 10 | sampled
bulk/xlarge/16/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.977322 | [0.932177, 1.021675] | 1.055023 | 1.055023 | 5.710 | 10 | sampled
bulk/xlarge/16/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 1.000047 | [0.984881, 1.005116] | 1.043406 | 1.043406 | 2.757 | 10 | sampled
