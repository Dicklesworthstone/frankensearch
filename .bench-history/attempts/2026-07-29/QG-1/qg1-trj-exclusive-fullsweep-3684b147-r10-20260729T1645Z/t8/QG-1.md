host=threadripperje | producer_os=linux | physical_cores=64 | logical_threads=128 | ram_bytes=536069869568 | numa_nodes=1 | process_available_threads=128 | thread_counts_requested=[8] | runtime_detected_isa=["avx2", "fma", "bmi2", "aes", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture=bulk/xlarge/8/positions_on | requested_threads=8 | affinity=0-127 | quill_threads_actually_used=1 | quill_peak_new_workers=8 | tantivy_threads_actually_used=595 | tantivy_peak_new_workers=25
engine=quill | role=subject | version=0.2.1 | executable_sha256=90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a | scheme=same-static-elf-v1
engine=tantivy | role=oracle | version=0.26.1 | executable_sha256=90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a | scheme=same-static-elf-v1
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/8/positions_on | quill | docs_per_second (docs/s) | 35854.674371 | [35575.160315, 36037.914001] | 36131.847806 | 36131.847806 | 2.408 | 10 | sampled
bulk/xlarge/8/positions_on | tantivy | docs_per_second (docs/s) | 144720.505968 | [141902.439573, 150128.107084] | 164136.480497 | 164136.480497 | 5.014 | 10 | sampled
bulk/xlarge/8/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.249382 | [0.231247, 0.252486] | 0.263134 | 0.263134 | 5.889 | 10 | sampled
bulk/xlarge/8/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.019092 | [0.967518, 1.069685] | 1.155691 | 1.155691 | 6.970 | 10 | sampled
bulk/xlarge/8/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 0.983728 | [0.980523, 0.993996] | 1.002163 | 1.002163 | 0.927 | 10 | sampled
