host=threadripperje | physical_cores=64 | logical_threads=128 | process_available_threads=128 | configured_engine_thread_widths=[4] | runtime_detected_isa=["aes", "avx2", "bmi2", "fma", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/4/positions_on | quill | docs_per_second (docs/s) | 37270.192943 | [36767.329349, 37429.740324] | 37949.821100 | 37949.821100 | 1.784 | 10 | sampled
bulk/xlarge/4/positions_on | tantivy | docs_per_second (docs/s) | 132936.058526 | [128886.627999, 135727.653004] | 139465.932730 | 139465.932730 | 2.993 | 10 | sampled
bulk/xlarge/4/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.277612 | [0.273338, 0.284961] | 0.301394 | 0.301394 | 3.419 | 10 | sampled
bulk/xlarge/4/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.973066 | [0.940029, 1.034159] | 1.069499 | 1.069499 | 4.829 | 10 | sampled
bulk/xlarge/4/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 0.995543 | [0.983521, 1.007703] | 1.019129 | 1.019129 | 1.654 | 10 | sampled
